"""Paired OSNet/CLIP-ReID retrieval on identical frozen snapshot PNG crops."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from mtmc.reid.clipreid_model import load_clipreid_visual, preprocess_clipreid, require, sha256
from preview_osnet_embeddings import read_crops
from evaluate_reid_snapshot import (check_protocol, load_gt, label_observations,
    rank_queries, summarize, score_summary)

ROOT = Path(__file__).resolve().parents[1]


def features_valid(features, rows, dimension):
    require(features.shape == (rows, dimension) and features.dtype == np.float32
        and np.isfinite(features).all() and np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5, rtol=0),
        'Invalid embedding matrix')


def retrieval(features, labels, cameras):
    scores = np.clip(features @ features.T, -1, 1)
    rows, directions = [], []
    for a, b in permutations(cameras, 2):
        current = rank_queries(scores, labels, a, b)
        stats = summarize(current)
        ranks = [r['positive_rank'] for r in current if r['status'] == 'evaluated']
        stats['mAP_single_positive'] = float(np.mean([1 / r for r in ranks])) if ranks else None
        directions.append(dict(query_camera=a, gallery_camera=b, **stats))
        rows.extend(current)
    pooled = summarize(rows)
    ranks = [r['positive_rank'] for r in rows if r['status'] == 'evaluated']
    pooled['mAP_single_positive'] = float(np.mean([1 / r for r in ranks])) if ranks else None
    same, different = [], []
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            b = labels[j]
            if a['camera'] != b['camera'] and a['gt_id'] is not None and b['gt_id'] is not None:
                (same if a['gt_id'] == b['gt_id'] else different).append(float(scores[i, j]))
    return dict(directions=directions, pooled=pooled,
                pair_similarities=dict(same_gt=score_summary(same), different_gt=score_summary(different))), rows


def paired_transitions(old, new):
    require(len(old) == len(new), 'Paired query count differs')
    counts = Counter(both_correct=0, improved=0, worsened=0, both_wrong=0)
    changes = []
    for a, b in zip(old, new):
        fields = ('query_camera', 'gallery_camera', 'query_embedding_row', 'query_gt_id', 'status', 'gallery_size', 'positive_local_id')
        require(all(a[f] == b[f] for f in fields), 'Paired population or eligibility differs')
        if a['status'] != 'evaluated':
            continue
        first, second = a['positive_rank'] == 1, b['positive_rank'] == 1
        category = 'both_correct' if first and second else 'worsened' if first else 'improved' if second else 'both_wrong'
        counts[category] += 1
        if a['positive_rank'] != b['positive_rank']:
            changes.append(dict(query_camera=a['query_camera'], gallery_camera=a['gallery_camera'],
                query_local_id=a['query_local_id'], query_gt_id=a['query_gt_id'],
                osnet_rank=a['positive_rank'], clipreid_rank=b['positive_rank'], category=category))
    return dict(counts), changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='Original OSNet embedding manifest')
    parser.add_argument('--reference-evaluation', type=Path, required=True, help='Original OSNet snapshot report')
    parser.add_argument('--model-check-report', type=Path, required=True, help='Successful CUDA CLIP-ReID smoke report')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0', help='CPU only for technical fixtures')
    args = parser.parse_args()
    require(1 <= args.batch_size <= 64, 'Batch size must be in [1,64]')
    inputs = {}

    def checked(name, path, digest=None):
        path = Path(path).resolve(); actual = sha256(path)
        require(digest is None or actual == digest, 'Changed input: '+str(path))
        inputs[name] = dict(path=str(path), sha256=actual)
        return path

    rp = checked('osnet_manifest', args.reference)
    ref = json.loads(rp.read_text())
    prior = json.loads(checked('osnet_evaluation', args.reference_evaluation).read_text())
    require(prior['protocol']['name'] == 'scene_snapshot_cross_camera_retrieval_v1'
        and prior['inputs']['embedding_manifest']['sha256'] == inputs['osnet_manifest']['sha256'], 'Wrong reference evaluation')
    config_path = checked('clipreid_config', ROOT/'configs/models/clipreid_vit_b16_msmt17.json')
    config = json.loads(config_path.read_text())
    gate = json.loads(checked('model_check', args.model_check_report).read_text())
    require(gate.get('completed') is True and gate['protocol'] == 'clipreid_visual_smoke_v1'
        and gate['device'] == args.device and gate['checks'] and all(gate['checks'].values())
        and gate['model']['config_sha256'] == inputs['clipreid_config']['sha256'], 'Expected a successful same-device check of this model')
    for path, digest in gate['code_checksums'].items():
        checked('gate_code:'+path, ROOT/path, digest)
    for relative in ('scripts/compare_clipreid_snapshot.py','scripts/evaluate_reid_snapshot.py','scripts/preview_osnet_embeddings.py'):
        checked('code:'+relative, ROOT/relative)
    ep = checked('osnet_embeddings', rp.parent/ref['embeddings']['path'], ref['embeddings']['sha256'])
    require(prior['inputs']['embeddings']['sha256'] == inputs['osnet_embeddings']['sha256'], 'Reference features changed')
    old_features = np.load(ep, allow_pickle=False)
    cp = checked('crop_manifest', ref['source_manifest']['path'], ref['source_manifest']['sha256'])
    crops, source_records, paths = read_crops(cp)
    records = ref['records']
    require(len(records) == len(source_records), 'Crop count differs')
    features_valid(old_features, len(records), 512)
    frames = {r['frame_index'] for r in records}; cameras = sorted({r['camera'] for r in records})
    require(len(frames) == 1 and len(cameras) >= 2, 'Expected a multi-camera snapshot')
    frame = next(iter(frames))
    require(prior['protocol']['frame_index'] == frame and prior['protocol']['cameras'] == cameras, 'Reference scope differs')
    width, height = crops['configuration']['source_resolution_wh']
    require(prior['protocol']['source_resolution_wh'] == [width, height], 'Resolution differs')
    rgb = []
    for row, (record, original, path) in enumerate(zip(records, source_records, paths)):
        require(record['embedding_row'] == row and all(record[k] == v for k, v in original.items()), 'Source row mapping differs')
        require(abs(float(Fraction(frame, crops['configuration']['fps'])) - record['timestamp_seconds']) < 1e-9, 'Timestamp differs')
        checked('crop:'+str(row), path, record['crop_sha256'])
        with Image.open(path) as image:
            require(image.format == 'PNG' and image.mode == 'RGB'
                    and image.size == (record['width'], record['height']), 'Invalid crop image')
            rgb.append(np.array(image, dtype=np.uint8, copy=True))
    print('Identical source crops and frozen OSNet features: VERIFIED', flush=True)
    print('Frame:', frame, '; crops per camera:', dict(Counter(r['camera'] for r in records)), flush=True)
    if args.device == 'cuda:0':
        require(torch.cuda.is_available(), 'CUDA unavailable')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print(f'Phase 1: CLIP-ReID {args.device} FP32; no GT used for encoding...', flush=True)
    model, model_info = load_clipreid_visual(config_path, project_root=ROOT, device=args.device)
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(rgb), args.batch_size):
            x = preprocess_clipreid(rgb[start:start+args.batch_size]).to(args.device)
            raw = model.raw_features(x)
            require(torch.isfinite(raw).all().item() and (raw.norm(dim=1) > 1e-12).all().item(), 'Invalid raw model output')
            encoded.append(torch.nn.functional.normalize(raw, p=2, dim=1).cpu().numpy())
    new_features = np.concatenate(encoded, axis=0)
    features_valid(new_features, len(records), 1280)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT/'artifacts/reid_model_comparison'/run
    out.mkdir(parents=True, exist_ok=False)
    feature_path = out/'clipreid_embeddings.npy'
    np.save(feature_path, new_features, allow_pickle=False)
    feature_hash = sha256(feature_path)
    (out/'observations.json').write_text(json.dumps(records, indent=2)+'\n')
    print('Phase 2: features frozen; shared offline GT labeling and retrieval...', flush=True)
    check_protocol()
    gt_spec = prior['inputs']['gt']
    gt_path = checked('ground_truth', gt_spec['path'], gt_spec['sha256'])
    labels, matching = label_observations(records, load_gt(gt_path, frame, cameras), width, height)
    require(matching == prior['gt_matching'] and labels == prior['observations'], 'Frozen GT labels not reproduced')
    results, rankings = {}, {}
    for name, features in (('osnet', old_features), ('clipreid', new_features)):
        results[name], rankings[name] = retrieval(features, labels, cameras)
    require({k:v for k,v in results['osnet']['pooled'].items() if k != 'mAP_single_positive'} == prior['pooled'], 'Original pooled OSNet metrics differ')
    require([{k:v for k,v in r.items() if k != 'mAP_single_positive'} for r in results['osnet']['directions']] == prior['directions'], 'Original OSNet direction metrics differ')
    transitions, changes = paired_transitions(rankings['osnet'], rankings['clipreid'])
    artifacts = {'clipreid_embeddings.npy': dict(path=feature_path.name, sha256=feature_hash,
        shape=list(new_features.shape), dtype=str(new_features.dtype))}
    for name, rows in rankings.items():
        path = out/(name+'_rankings.csv')
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        artifacts[path.name] = dict(path=path.name, sha256=sha256(path))
    artifacts['observations.json'] = dict(path='observations.json', sha256=sha256(out/'observations.json'))
    for item in inputs.values():
        require(sha256(item['path']) == item['sha256'], 'Input changed during comparison')
    require(sha256(feature_path) == feature_hash, 'Frozen features changed')
    report = dict(completed=True, protocol='paired_reid_snapshot_v1', run_id=run, inputs=inputs,
        frame_index=frame, cameras=cameras, crop_count=len(records), batch_size=args.batch_size, device=args.device,
        feature_dimensions=dict(osnet=512, clipreid=1280), results=results, rank1_transitions=transitions,
        changed_rankings=changes, gt_matching=matching, model=model_info, artifacts=artifacts,
        versions={p:version(p) for p in ('torch','torchvision','numpy','scipy','pillow')},
        checks=dict(original_osnet_metrics=True, original_gt_labels=True, identical_crop_bytes=True,
                    keyed_row_mapping=True, common_query_population=True, inputs_unchanged=True),
        limits=['One synchronized integration frame; not independent validation or a global-ID evaluation.',
            'One positive per eligible query: AP equals reciprocal positive rank; mAP equals MRR here.',
            'Unknown gallery observations remain distractors; queries without gallery positives are excluded from rank metrics.',
            'Directed queries share observations and are not independent statistical samples.',
            'No threshold selection, runtime replacement, latency benchmark or model superiority claim.'])
    path = out/'report.json'; path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    def pct(value): return 'n/a' if value is None else f'{value:.2%}'
    print('Direction  Eligible  OSNet R1  CLIP-ReID R1  OSNet mAP  CLIP-ReID mAP')
    for a,b in zip(results['osnet']['directions'], results['clipreid']['directions']):
        print(f"{a['query_camera']}->{a['gallery_camera']}  {a['evaluated_queries']}/{a['queries']}  {pct(a['rank1'])}  {pct(b['rank1'])}  {pct(a['mAP_single_positive'])}  {pct(b['mAP_single_positive'])}")
    for name, result in results.items():
        p = result['pooled']; print(f"{name}: pooled Rank-1={p['rank1_hits']}/{p['evaluated_queries']} ({pct(p['rank1'])}); Rank-3={pct(p['rank3'])}; mAP={pct(p['mAP_single_positive'])}")
    print('Rank-1 transitions:', transitions)
    print('Original OSNet metrics and common GT labels: REPRODUCED')
    print('Report:', path)
    print('Paired Re-ID snapshot: COMPLETED; model selection and global evaluation pending')


if __name__ == '__main__':
    main()
