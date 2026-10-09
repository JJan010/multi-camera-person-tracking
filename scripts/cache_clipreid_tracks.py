"""Stream full-rate CLIP-ReID features for frozen local observations, without GT."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
from mtmc.data.scene import load_scene, require, sha256
from mtmc.reid.crops import crop_geometry
from compare_reid_temporal import selected_crops, valid_features

ROOT = Path(__file__).resolve().parents[1]
DIM = 1280


def trace_rows(scene, path, run_id):
    with gzip.open(path, 'rt') as stream:
        for frame in range(scene.rounds):
            line = stream.readline()
            require(bool(line), 'Truncated source trace')
            row = json.loads(line)
            require(row['run_id'] == run_id and type(row['frame_index']) is int
                    and row['frame_index'] == frame
                    and Fraction(row['timestamp']) == Fraction(frame, scene.fps), 'Source scope/time differs')
            yield row
        require(stream.readline() == '', 'Trailing source frames')


def metadata_records(scene, row, offset, source_vector_count):
    """Canonical metadata join; original tracker IDs, not segment/global IDs."""
    cameras = row['variants']['enabled']['cameras']
    require([c['camera'] for c in cameras] == list(scene.camera_ids), 'Camera coverage/order differs')
    records = []; encoded = 0
    for camera, spec in zip(cameras, scene.cameras):
        ids = camera['local_ids']; n = len(ids)
        require(all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == n,
                'Invalid/duplicate local IDs')
        require(all(len(camera[k]) == n for k in ('xyxy', 'confidence', 'detection_indices', 'embedding_rows')),
                'Camera array lengths differ')
        detections = camera['detection_indices']
        require(all(type(i) is int and i >= 0 for i in detections) and len(set(detections)) == n,
                'Invalid/duplicate candidate indices')
        for j in sorted(range(n), key=lambda j: ids[j]):
            box = camera['xyxy'][j]; score = camera['confidence'][j]
            require(type(score) in (int, float) and np.isfinite(score) and 0 <= score <= 1, 'Invalid score')
            bounds, fraction = crop_geometry(box, spec.width, spec.height)
            original = camera['embedding_rows'][j]
            require((original is None) == (bounds is None), 'Source crop availability differs')
            if bounds is not None:
                require(type(original) is int and 0 <= original < source_vector_count, 'Source row out of bounds')
            records.append(dict(camera=spec.camera_id, local_id=ids[j], frame_index=row['frame_index'],
                timestamp=str(Fraction(row['timestamp'])), source_xyxy=box, confidence=score,
                crop_xyxy_int=list(bounds) if bounds is not None else None, inside_image_fraction=fraction,
                detection_index=detections[j], source_embedding_row=original,
                embedding_row=offset + encoded if bounds is not None else None))
            encoded += bounds is not None
    return records


def inventory(scene, trace, run_id, source_vector_count):
    total = encoded = 0; cameras = Counter(); seen = set()
    for row in trace_rows(scene, trace, run_id):
        records = metadata_records(scene, row, encoded, source_vector_count)
        for r in records:
            cameras[str(r['camera'])] += 1
            if r['embedding_row'] is not None:
                require(r['source_embedding_row'] not in seen, 'Source candidate row reused by two observations')
                seen.add(r['source_embedding_row']); encoded += 1
        total += len(records)
    return dict(rounds=scene.rounds, observations=total, encoded=encoded, fully_outside=total-encoded,
                per_camera={str(c): cameras[str(c)] for c in scene.camera_ids})


def temporal_reference(records, vectors, selected):
    lookup = {}; used = []; selected = set(selected)
    for record in records:
        key = (record['frame_index'], record['camera'], record['local_id'])
        require(key not in lookup and key[0] in selected, 'Duplicate/out-of-scope temporal observation')
        lookup[key] = record
        index = record['embedding_row']
        if index is not None:
            require(type(index) is int and index >= 0, 'Invalid temporal vector row')
            used.append(index)
    require(sorted(used) == list(range(len(vectors))), 'Temporal vector mapping is not bijective')
    valid_features(vectors, len(vectors), DIM)
    return lookup


def verify_persisted(scene, trace, source_run, source_vector_count, output, expected):
    matrix = np.load(output/'clipreid_embeddings.npy', mmap_mode='r', allow_pickle=False)
    require(matrix.shape == (expected['encoded'], DIM) and matrix.dtype == np.float32, 'Persisted matrix shape/dtype differs')
    offset = total = 0
    with gzip.open(output/'observations.jsonl.gz', 'rt') as stream:
        for original in trace_rows(scene, trace, source_run):
            line = stream.readline(); require(bool(line), 'Truncated persisted observations')
            row = json.loads(line)
            require(row['frame_index'] == original['frame_index'] and row['timestamp'] == original['timestamp'],
                    'Persisted frame mapping differs')
            target = metadata_records(scene, original, offset, source_vector_count)
            require(len(target) == len(row['observations']), 'Persisted observation count differs')
            for want, got in zip(target, row['observations']):
                require({k:v for k,v in got.items() if k != 'rgb_sha256'} == want, 'Persisted source mapping differs')
                digest = got.get('rgb_sha256')
                require(digest is None if want['embedding_row'] is None else
                        isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                        'Invalid crop pixel checksum')
            count = sum(r['embedding_row'] is not None for r in target)
            valid_features(np.asarray(matrix[offset:offset+count]), count, DIM)
            offset += count; total += len(target)
        require(stream.readline() == '', 'Trailing persisted frames')
    require(offset == expected['encoded'] and total == expected['observations'], 'Persisted totals differ')


def collect(scene, batches, trace, source_run, source_vector_count, encode, reference, reference_vectors,
            selected, output, expected):
    """Bounded-RAM writer; injected encoder must return ordered keys, times and vectors."""
    count = expected['encoded']; selected = set(selected)
    path = output/'clipreid_embeddings.npy'
    if count:
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(count, DIM))
    else:
        np.save(path, np.empty((0, DIM), np.float32), allow_pickle=False)
        matrix = np.empty((0, DIM), np.float32)
    offset = total = compared = snapshots = 0; seen = set(); error_max = 0.; cosine_min = 1.
    with gzip.open(output/'observations.jsonl.gz', 'wt') as stream:
        for row in trace_rows(scene, trace, source_run):
            frame = row['frame_index']
            batch = next(batches)
            require(batch.frame_index == frame, 'Decoded frame differs')
            records, crops = selected_crops(scene, batch, row, offset)
            # Normalize tuple bounds to the persisted JSON schema.
            records = json.loads(json.dumps(records, allow_nan=False))
            want = metadata_records(scene, row, offset, source_vector_count)
            require([{k:v for k,v in r.items() if k != 'rgb_sha256'} for r in records] == want,
                    'Decoded crop/source metadata join differs')
            if crops:
                keys, times, features = encode(crops)
                require(keys == tuple(c.key for c in crops) and times == tuple(c.timestamp for c in crops),
                        'Encoder key/time mapping differs')
                valid_features(features, len(crops), DIM)
            else:
                features = np.empty((0, DIM), np.float32)  # Never execute a model on an empty batch.
            if frame in selected:
                snapshots += 1
                for record in records:
                    key = (frame, record['camera'], record['local_id'])
                    require(key in reference and key not in seen, 'Temporal observation coverage differs')
                    old = reference[key]; seen.add(key)
                    require({k:v for k,v in record.items() if k != 'embedding_row'} ==
                            {k:v for k,v in old.items() if k != 'embedding_row'}, 'Temporal crop pixels/provenance differ')
                    if record['embedding_row'] is None:
                        continue
                    vector = features[record['embedding_row']-offset]
                    previous = reference_vectors[old['embedding_row']]
                    error = float(np.max(np.abs(vector-previous)))
                    cosine = float(np.clip(np.dot(vector.astype(np.float64), previous), -1, 1))
                    require(error <= 1e-5 and cosine >= 1-1e-5, 'Temporal CLIP feature parity failed')
                    error_max = max(error_max, error); cosine_min = min(cosine_min, cosine); compared += 1
            require(offset + len(crops) <= count, 'More encoded crops than inventoried')
            matrix[offset:offset+len(crops)] = features
            offset += len(crops); total += len(records)
            stream.write(json.dumps(dict(frame_index=frame, timestamp=row['timestamp'], observations=records), allow_nan=False)+'\n')
            if (frame+1) % 60 == 0 or frame+1 == scene.rounds:
                if count: matrix.flush()
                print(f'Processed {frame+1}/{scene.rounds}; CLIP embeddings={offset}; reference checks={compared}', flush=True)
    if count: matrix.flush()
    del matrix
    require(offset == count and total == expected['observations'], 'Collection totals differ')
    require(seen == set(reference) and snapshots == len(selected), 'Temporal reference was not completely checked')
    verify_persisted(scene, trace, source_run, source_vector_count, output, expected)
    return dict(**expected, temporal_reference_parity=dict(frames=snapshots, compared=compared,
                max_absolute_error=error_max, minimum_cosine=cosine_min))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--temporal-report', type=Path, required=True)
    parser.add_argument('--model-check-report', type=Path, required=True)
    args = parser.parse_args(); inputs = {}
    def checked(name, path, digest=None):
        path = Path(path).resolve(); actual = sha256(path)
        require(digest is None or digest == actual, 'Changed input: '+str(path))
        inputs[name] = dict(path=str(path), sha256=actual)
        return path
    def passed(report, protocols):
        require(report.get('completed') is True and report['protocol'] in protocols
                and bool(report['checks']) and all(report['checks'].values()), 'Unverified/unsupported source report')
    source_path = checked('source_report', args.source_report)
    source = json.loads(source_path.read_text())
    passed(source, ('appearance_continuity_paired_v1', 'appearance_continuity_scene_transfer_v1'))
    temporal_path = checked('temporal_report', args.temporal_report)
    temporal = json.loads(temporal_path.read_text()); passed(temporal, ('paired_reid_temporal_grid_v1',))
    require(temporal['inputs']['source_report']['sha256'] == inputs['source_report']['sha256']
            and temporal['source_run_id'] == source['run_id'] and temporal['scene'] == source['scene'],
            'Temporal/source lineage differs')
    transfer = source['protocol'] == 'appearance_continuity_scene_transfer_v1'
    for name, key in (('scene_config', 'validation:scene_config' if transfer else 'scene_config'),
                      ('cache_report', 'validation:candidate_report' if transfer else 'cache_report'),
                      ('source_vectors', 'validation_cache:embeddings.npy' if transfer else 'embeddings.npy')):
        item = source['inputs'][key]; checked(name, item['path'], item['sha256'])
    loaded = load_scene(inputs['scene_config']['path'], project_root=ROOT); scene = loaded.runtime
    require(scene.scene == source['scene'] and scene.rounds == 3600 and scene.fps == 30, 'Scene/scope differs')
    for name, ref in (('source_manifest', loaded.source_manifest), ('video_manifest', loaded.video_manifest),
                      ('calibration', scene.calibration)):
        checked(name, ref.path, ref.sha256)
    for camera in scene.cameras: checked('video:'+str(camera.camera_id), camera.video.path, camera.video.sha256)
    item = source['artifacts']['global_tracks.jsonl.gz']
    trace = checked('source_tracks', source_path.parent/item['path'], item['sha256'])
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    require(cache.get('completed') is True and cache['artifacts']['embeddings.npy']['sha256'] == inputs['source_vectors']['sha256'],
            'Source candidate cache lineage differs')
    original = np.load(inputs['source_vectors']['path'], mmap_mode='r', allow_pickle=False)
    require(original.dtype == np.float32 and original.shape == (cache['summary']['encoded'], 512), 'Invalid source vector matrix')
    source_vector_count = len(original); del original
    for name in ('observations.json', 'clipreid_embeddings.npy'):
        item = temporal['artifacts'][name]
        checked('temporal:'+name, temporal_path.parent/item['path'], item['sha256'])
    vectors = np.load(inputs['temporal:clipreid_embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    selected = temporal['selected_frames']
    require(selected == list(range(2, 3600, 30)), 'Temporal sampling schedule differs')
    reference = temporal_reference(json.loads(Path(inputs['temporal:observations.json']['path']).read_text()), vectors, selected)
    config_path = checked('clipreid_config', ROOT/'configs/models/clipreid_vit_b16_msmt17.json')
    gate = json.loads(checked('model_check', args.model_check_report).read_text())
    passed(gate, ('clipreid_visual_smoke_v1',))
    require(gate['device'] == 'cuda:0' and gate['model']['config_sha256'] == inputs['clipreid_config']['sha256']
            and temporal['clipreid_model']['config_sha256'] == inputs['clipreid_config']['sha256'], 'Model gate/config differs')
    require(temporal['inputs']['clipreid_check']['sha256'] == inputs['model_check']['sha256'], 'Temporal model gate differs')
    for rel, digest in gate['code_checksums'].items(): checked('gate_code:'+rel, ROOT/rel, digest)
    # Check only runtime source code here; do not open GT or GT evaluation artifacts.
    for name, item in temporal['inputs'].items():
        if name.startswith('code:'): checked('temporal_'+name, item['path'], item['sha256'])
    for name, item in json.loads(config_path.read_text())['assets'].items():
        checked('model_asset:'+name, ROOT/item['path'], item['sha256'])
    for rel in ('scripts/cache_clipreid_tracks.py', 'scripts/check_clipreid_track_cache.py'):
        checked('code:'+rel, ROOT/rel)
    expected = inventory(scene, trace, source['run_id'], source_vector_count)
    require(expected['observations'] == source['summary']['observations'], 'Source observation count differs')
    import torch
    from mtmc.reid.clipreid_model import load_clipreid_visual, preprocess_clipreid
    from mtmc.video.replay import synchronized_replay
    require(torch.cuda.is_available(), 'CUDA unavailable'); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    print(f'Scene: {scene.scene}; {scene.rounds} frames; {expected["encoded"]} crops; dimension={DIM}', flush=True)
    model, model_info = load_clipreid_visual(config_path, project_root=ROOT, device='cuda:0')
    def encode(crops):
        chunks = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            for start in range(0, len(crops), 16):
                part = crops[start:start+16]
                raw = model.raw_features(preprocess_clipreid([c.rgb for c in part]).to('cuda:0'))
                require(torch.isfinite(raw).all().item() and (raw.norm(dim=1) > 1e-12).all().item(), 'Invalid raw CLIP features')
                chunks.append(torch.nn.functional.normalize(raw, p=2, dim=1).cpu().numpy())
        return tuple(c.key for c in crops), tuple(c.timestamp for c in crops), np.concatenate(chunks)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'artifacts/clipreid_track_cache'/run; output.mkdir(parents=True, exist_ok=False)
    status = output/'run_status.json'
    status.write_text(json.dumps(dict(completed=False, run_id=run))+'\n')
    try:
        print('CPU decoding + CLIP-ReID CUDA FP32; frozen local boxes; no GT, OSNet, detector or tracker execution...', flush=True)
        with synchronized_replay(scene.sources, fps=scene.fps, threads=1) as batches:
            counts = collect(scene, batches, trace, source['run_id'], source_vector_count, encode,
                             reference, vectors, selected, output, expected)
        for item in inputs.values(): require(sha256(item['path']) == item['sha256'], 'Input changed during collection')
        artifacts = {name:dict(path=name, sha256=sha256(output/name), bytes=(output/name).stat().st_size)
                     for name in ('clipreid_embeddings.npy', 'observations.jsonl.gz')}
        report = dict(completed=True, protocol='clipreid_frozen_track_cache_v1', run_id=run,
            scene=scene.scene, source_run_id=source['run_id'], summary=counts, inputs=inputs, artifacts=artifacts,
            model=model_info, feature_dim=DIM, dtype='float32', normalization='joint_l2',
            frame_range=[0, scene.rounds-1], cameras=list(scene.camera_ids), source_variant='enabled',
            inference=dict(device='cuda:0', batch_size=16, torch_threads=1, tf32=False, autocast=False),
            versions={p:version(p) for p in ('torch', 'torchvision', 'av', 'numpy', 'pillow')},
            checks=dict(full_source_mapping=True, persisted_rows_finite_normalized=True,
                        temporal_crop_pixel_parity=True, temporal_feature_parity=True,
                        no_GT_read=True, inputs_unchanged=True),
            limits=['Frozen local tracks originated from OSNet-aware tracking; this is not a detector-candidate cache.',
                    'All positive-area crops retained, including low-confidence and border crops; unavailable crops retain null rows.',
                    'Parity is verified on the fixed 120-frame temporal grid, not independently on every image.',
                    'No association, history, threshold calibration, global-ID evaluation or throughput benchmark.',
                    'No existing report, prediction, baseline or model is overwritten.'])
        (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        status.write_text(json.dumps(dict(completed=True, run_id=run))+'\n')
    except Exception as error:
        status.write_text(json.dumps(dict(completed=False, run_id=run, error=str(error)))+'\n'); raise
    print('Summary:', json.dumps(counts))
    print('Report:', output/'report.json')
    print('CLIP-ReID frozen track cache: COMPLETED; persisted mapping and temporal parity VERIFIED; global IDs unchanged')


if __name__ == '__main__':
    main()
