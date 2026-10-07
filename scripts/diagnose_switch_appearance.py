"""Inspect frozen detections and OSNet descriptors around a local-ID transition."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

import evaluate_mtmc_sequence as sequence
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
require = sequence.require


def cosine(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    require(a.shape == b.shape == (512,) and np.isfinite(a).all() and np.isfinite(b).all(),
            'Invalid descriptor')
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    require(abs(na - 1) < 1e-4 and abs(nb - 1) < 1e-4, 'Descriptor is not normalized')
    return float(np.clip(a @ b / (na * nb), -1, 1))


def detector_evidence(camera, gt_box):
    boxes = np.asarray(camera['detector_xyxy'], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(camera['detector_confidence'], dtype=np.float64)
    require(scores.shape == (len(boxes),) and np.isfinite(scores).all()
            and np.isfinite(boxes).all() and np.all(boxes[:, 2:] > boxes[:, :2])
            and np.all((scores >= 0) & (scores <= 1)), 'Invalid saved detections')
    if gt_box is None:
        return {'gt_present': False, 'best_iou': None, 'best_score': None, 'candidates': []}
    clip = sequence.metric.history.snapshot.clip_boxes
    gt = clip([gt_box], 1920, 1080)
    if not np.all(gt[0, 2:] > gt[0, :2]):
        return {'gt_present': True, 'best_iou': None, 'best_score': None, 'candidates': []}
    ious = sequence.metric.history.snapshot.pairwise_iou(gt, clip(boxes, 1920, 1080))[0]
    best = int(np.argmax(ious)) if len(ious) else None
    return {'gt_present': True, 'best_iou': float(ious[best]) if best is not None else None,
            'best_score': float(scores[best]) if best is not None else None,
            'candidates': [{'index': int(i), 'iou': float(ious[i]), 'score': float(scores[i]),
                            'xyxy': boxes[i].tolist()} for i in np.flatnonzero(ious >= .5)]}


def write_csv(path, rows):
    require(bool(rows), 'No rows to write')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-report', type=Path, required=True)
    parser.add_argument('--context-frames', type=int, default=30)
    args = parser.parse_args()
    require(args.context_frames > 0, 'Context must be positive')
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path

    print('Verifying frozen preview, traces and descriptor arrays...', flush=True)
    preview_path = checked('preview_report', args.preview_report)
    preview = json.loads(preview_path.read_text())
    require(preview.get('completed') is True
            and preview['protocol'] == 'scene_001_local_identity_transition_preview_v1',
            'Expected completed identity transition preview')
    for name in ('pipeline_report', 'tracks', 'global_tracks', 'ground_truth'):
        item = preview['inputs'][name]
        checked(name, item['path'], item['sha256'])
    source_path = Path(inputs['pipeline_report']['path'])
    source = json.loads(source_path.read_text())
    run, cfg, rounds = source['run_id'], source['configuration'], source['summary']['rounds']
    require(source.get('completed') and run == preview['source_run_id'] and cfg['fps'] == 30,
            'Pipeline scope differs')
    for name in ('tracks', 'global_tracks'):
        require(source['artifacts'][name]['sha256'] == inputs[name]['sha256'], 'Trace differs from pipeline')
    arrays = {}
    for name in ('embeddings', 'mean_embeddings'):
        item = source['artifacts'][name]
        path = checked(name, source_path.parent / item['path'], item['sha256'])
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        require(array.shape == (source['summary']['total_embeddings'], 512)
                and array.dtype == np.float32, 'Embedding array shape/dtype differs')
        arrays[name] = array
    latest, means = arrays['embeddings'], arrays['mean_embeddings']
    camera, local_id, event = preview['camera'], preview['local_id'], preview['transition']
    before, after = event['previous_evidence_frame'], event['frame']
    require((event['camera'], event['local_id']) == (camera, local_id)
            and 2 <= before < after < rounds and event['previous_gt'] != event['current_gt'],
            'Invalid selected transition')
    start, end = max(2, before - args.context_frames), min(rounds - 1, after + args.context_frames)
    print(f'Camera {camera}, local {local_id}; frames {start}..{end}; CPU only', flush=True)
    ground = sequence.load_ground_truth(inputs['ground_truth']['path'], rounds)
    timeline, nearby, detections = [], [], []
    previous_latest = previous_mean = anchor = None
    previous_frame = anchor_frame = None
    next_row = 0
    with Path(inputs['tracks']['path']).open() as local_file, Path(inputs['global_tracks']['path']).open() as global_file:
        for frame in range(rounds):
            a, b = local_file.readline(), global_file.readline()
            require(bool(a) and bool(b), 'Truncated trace')
            local, global_record = json.loads(a), json.loads(b)
            cameras, assigned, next_row = sequence.validate_round(
                local, global_record, frame=frame, run=run, config=cfg, next_row=next_row)
            if not start <= frame <= end:
                continue
            keys, boxes = cameras[camera]
            _, _, unique, _, _ = sequence.spatial_slot(ground.get((frame, camera), {}), keys, boxes)
            observations = {ObservationKey(o['camera'], o['local_id'], frame): o
                            for o in local['reid_observations'] if o['camera'] == camera}
            target_key = ObservationKey(camera, local_id, frame)
            target = observations.get(target_key)
            row_index = target['embedding_row'] if target else None
            current = latest[row_index] if row_index is not None else None
            mean = means[row_index] if row_index is not None else None
            if anchor is None and current is not None:
                anchor, anchor_frame = current.copy(), frame
            row = {'frame': frame, 'timestamp_s': frame / 30, 'local_id': local_id,
                   'global_id': assigned.get(target_key), 'diagnostic_gt': unique.get(target_key),
                   'embedding_row': row_index, 'previous_encoded_frame': previous_frame,
                   'encoded_gap_frames': frame - previous_frame if previous_frame is not None else None,
                   'latest_vs_previous_latest': cosine(current, previous_latest),
                   'latest_vs_previous_mean': cosine(current, previous_mean),
                   'latest_vs_anchor': cosine(current, anchor),
                   'mean_vs_anchor': cosine(mean, anchor),
                   'track_confidence': target['confidence'] if target else None}
            timeline.append(row)
            # Keep previous state strictly before the current observation for these comparisons.
            if current is not None:
                previous_latest, previous_mean, previous_frame = current, mean, frame
            for key in keys:
                item = observations[key]
                index = item['embedding_row']
                vector = latest[index] if index is not None else None
                nearby.append({'frame': frame, 'local_id': key.local_id, 'global_id': assigned[key],
                               'diagnostic_gt': unique.get(key), 'embedding_row': index,
                               'confidence': item['confidence'], 'latest_vs_anchor': cosine(vector, anchor)})
            saved_camera = next(c for c in local['cameras'] if c['camera'] == camera)
            detections.append({'frame': frame, 'targets': {str(g): detector_evidence(
                saved_camera, ground.get((frame, camera), {}).get(g))
                for g in (event['previous_gt'], event['current_gt'])}})
        require(local_file.readline() == global_file.readline() == '', 'Trailing trace rows')
    require(next_row == source['summary']['total_embeddings'] and anchor_frame is not None,
            'Embedding coverage differs or no target descriptor')
    by_frame = {r['frame']: r for r in timeline}
    for frame, gt, gid in ((before, event['previous_gt'], event['previous_global_id']),
                           (after, event['current_gt'], event['current_global_id'])):
        require(by_frame[frame]['diagnostic_gt'] == gt and by_frame[frame]['global_id'] == gid,
                'Preview transition evidence not reproduced')
    selected_frames = sorted(set(preview['frames']) | {before, after})
    selected_frames = [f for f in selected_frames if f in by_frame]
    print(f'Fixed appearance anchor: frame {anchor_frame}; selected by time, not GT')
    print('Frame   GT   Global    cos(prev)  cos(prior mean)  cos(anchor)  mean/anchor')
    def show(v):
        return 'n/a' if v is None else f'{v:.4f}'
    for f in selected_frames:
        r = by_frame[f]
        print(f"{f:5} {str(r['diagnostic_gt']):>4} {str(r['global_id']):>8} "
              f"{show(r['latest_vs_previous_latest']):>12} {show(r['latest_vs_previous_mean']):>16} "
              f"{show(r['latest_vs_anchor']):>12} {show(r['mean_vs_anchor']):>12}")
    selected_tracks = [r for r in nearby if r['frame'] in selected_frames
                       and (r['local_id'] == local_id
                            or r['diagnostic_gt'] in (event['previous_gt'], event['current_gt']))]
    print('Tracks of the two diagnostic identities in selected frames:')
    for r in selected_tracks:
        print(f"Frame {r['frame']}: L{r['local_id']}/G{r['global_id']}; GT={r['diagnostic_gt']}; "
              f"cos(anchor)={show(r['latest_vs_anchor'])}")
    detector_by_frame = {r['frame']: r for r in detections}
    print('Detector candidates around transition (IoU >= 0.5; not tracker assignment):')
    for f in range(max(start, before - 1), min(end, after + 1) + 1):
        print(json.dumps(detector_by_frame[f], allow_nan=False))
    for item in inputs.values():
        require(sha256(Path(item['path'])) == item['sha256'], 'Input changed during diagnostic')
    output = ROOT / 'artifacts/identity_switch_appearance' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'target_timeline.csv', timeline)
    write_csv(output / 'camera_tracks.csv', nearby)
    (output / 'detector_evidence.json').write_text(json.dumps(detections, indent=2, allow_nan=False) + '\n')
    report = {'completed': True, 'protocol': 'frozen_local_switch_appearance_diagnostic_v1',
              'source_run_id': run, 'camera': camera, 'local_id': local_id, 'transition': event,
              'frames': [start, end], 'anchor_frame': anchor_frame,
              'anchor_policy': 'First encoded target observation in time-selected context; no GT selection',
              'selected_timeline': [by_frame[f] for f in selected_frames], 'inputs': inputs,
              'selected_camera_tracks': selected_tracks,
              'artifacts': {p.name: {'path': p.name, 'sha256': sha256(p)} for p in sorted(output.iterdir())},
              'code_sha256': {str(Path(m.__file__).relative_to(ROOT)): sha256(Path(m.__file__))
                              for m in (sequence, sequence.baseline, sequence.metric.history.snapshot)},
              'numpy_version': np.__version__,
              'limits': ['GT only labels this offline diagnostic; predictions are unchanged.',
                         'Anchor can itself be contaminated; visual review is required.',
                         'Previous mean excludes current observation; current mean includes it.',
                         'Saved detector outputs have already passed the configured detector threshold.',
                         'One detector box can overlap both GT boxes; candidate counts are not independent assignments.',
                         'Tracked-box embeddings are not embeddings of all candidate detections.',
                         'No threshold selection, runtime safeguard or improvement claim.']}
    report['code_sha256'][str(Path(__file__).relative_to(ROOT))] = sha256(Path(__file__))
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {path}')
    print('Local switch appearance diagnostic: COMPLETED; runtime unchanged')


if __name__ == '__main__':
    main()
