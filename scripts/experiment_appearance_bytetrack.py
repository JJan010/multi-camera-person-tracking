"""Paired CPU replay: frozen ByteTrack, direct candidate output, and appearance gating."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from check_frozen_bytetrack_replay import candidate_arrays, compare_outputs, require
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'direct_iou', 'direct_appearance')
CAMERAS = (4, 5, 8)


def evaluate(trace_path, ground, rounds):
    """GT is accessed only after all runtime variants have been frozen."""
    import motmetrics as mm
    import evaluate_local_tracking as local_metric
    from evaluate_mtmc_pipeline import spatial_slot
    require(version('motmetrics') == '1.4.0', 'Expected motmetrics 1.4.0')
    mm.lap.default_solver = 'scipy'
    metrics, case = {}, []
    names = ['num_frames', 'num_objects', 'num_predictions', 'idtp', 'idfp', 'idfn',
             'idf1', 'idp', 'idr', 'precision', 'recall', 'num_switches', 'num_false_positives', 'num_misses']
    for variant in VARIANTS:
        accumulators = {c: mm.MOTAccumulator(auto_id=False) for c in CAMERAS}
        with gzip.open(trace_path, 'rt') as stream:
            for frame in range(rounds):
                line = stream.readline(); require(bool(line), 'Truncated experimental trace')
                record = json.loads(line)
                require(record['frame_index'] == frame, 'Experimental trace order differs')
                if frame < 2:
                    continue
                for item in record['variants'][variant]:
                    camera, ids = item['camera'], item['local_ids']
                    gt = ground[frame, camera]
                    gt_ids = sorted(gt)
                    boxes = local_metric.boxes_array([gt[g] for g in gt_ids])
                    visible = np.all(boxes[:, 2:] > boxes[:, :2], axis=1)
                    gt_ids = [g for g, keep in zip(gt_ids, visible) if keep]
                    predicted = local_metric.boxes_array(item['xyxy'])
                    require(len(ids) == len(predicted) == len(set(ids)), 'Duplicate experimental local ID')
                    accumulators[camera].update(gt_ids, ids,
                        local_metric.distances(boxes[visible], predicted), frameid=frame)
                    if camera == 5 and frame in (798, 808, 813, 814, 815, 820, 830):
                        keys = tuple(ObservationKey(camera, i, frame) for i in ids)
                        _, _, unique, _, _ = spatial_slot(gt, keys, np.asarray(item['xyxy']).reshape(-1, 4))
                        for key in keys:
                            label = unique.get(key)
                            if label in (0, 23) or key.local_id == 11:
                                case.append({'variant': variant, 'frame': frame, 'camera': camera,
                                             'local_id': key.local_id, 'unique_overlap_gt': label})
            require(stream.readline() == '', 'Trailing experimental trace')
        table = mm.metrics.create().compute_many(list(accumulators.values()), metrics=names,
                    names=[f'camera_{c:04d}' for c in CAMERAS], generate_overall=True)
        metrics[variant] = json.loads(table.to_json(orient='index'))
    return metrics, case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-report', type=Path, required=True)
    parser.add_argument('--appearance-threshold', type=float, default=.6)
    args = parser.parse_args()
    require(np.isfinite(args.appearance_threshold) and -1 <= args.appearance_threshold <= 1,
            'Invalid experimental threshold')
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve(); actual = sha256(path)
        require(expected is None or actual == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path

    print('Verifying exact replay gate and frozen candidate cache...', flush=True)
    gate_path = checked('replay_report', args.replay_report)
    gate = json.loads(gate_path.read_text())
    require(gate.get('completed') is True and gate.get('passed') is True
            and gate['protocol'] == 'frozen_bytetrack_exact_replay_v1', 'A passed exact-replay gate is required')
    for name, spec in gate['inputs'].items():
        checked(name, spec['path'], spec['sha256'])
    require(version('supervision') == '0.30.7', 'Expected pinned supervision 0.30.7')
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    require(cache['run_id'] == gate['cache_run_id'] and source['run_id'] == gate['source_run_id'], 'Mixed scopes')
    rounds, settings = source['summary']['rounds'], source['configuration']['tracker']
    features = np.load(inputs['embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(features.shape == (cache['summary']['encoded'], 512) and features.dtype == np.float32,
            'Invalid cached feature shape/dtype')
    from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack
    trackers = {name: {c: ExperimentalByteTrack(**settings, direct_output=name != 'baseline',
                 appearance_threshold=args.appearance_threshold if name == 'direct_appearance' else None)
                 for c in CAMERAS} for name in VARIANTS}
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/appearance_bytetrack' / run
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_status.json').write_text(json.dumps({'completed': False, 'run_id': run}) + '\n')
    trace_path = output / 'tracks.jsonl.gz'
    counts = {name: Counter() for name in VARIANTS}
    stats = Counter(); distinct = {name: {c: set() for c in CAMERAS} for name in VARIANTS}
    next_row = total_candidates = outside = 0
    print(f'Phase 1: {rounds} causal rounds, three variants; appearance threshold={args.appearance_threshold}; no GT', flush=True)
    with Path(inputs['tracks']['path']).open() as source_file, \
            Path(inputs['detections.jsonl']['path']).open() as cache_file, gzip.open(trace_path, 'wt') as saved:
        for frame in range(rounds):
            a, b = source_file.readline(), cache_file.readline()
            require(bool(a) and bool(b), 'Truncated source/cache')
            original, candidates = json.loads(a), json.loads(b)
            arrays, expected, next_row, missing = candidate_arrays(candidates, original, frame=frame,
                cache_run=cache['run_id'], source_run=source['run_id'], next_row=next_row)
            outside += missing
            record = {'run_id': run, 'source_run_id': source['run_id'], 'frame_index': frame,
                      'timestamp': str(Fraction(frame, 30)), 'variants': {name: [] for name in VARIANTS}}
            for camera in CAMERAS:
                boxes, scores, embedding_rows = arrays[camera]
                vectors = [features[i] if i is not None else None for i in embedding_rows]
                total_candidates += len(boxes)
                for name in VARIANTS:
                    tracker = trackers[name][camera]
                    result = tracker.update_candidates(boxes, scores, vectors)
                    if name == 'baseline':
                        comparison = compare_outputs(result, expected[camera])
                        require(comparison['passed'], f'Fork baseline differs from frozen output at camera {camera}, frame {frame}: {comparison}')
                    else:
                        indices = tracker.selected_indices
                        require(np.array_equal(result.xyxy, boxes[list(indices)])
                                and np.array_equal(result.confidence, scores[list(indices)]),
                                'Direct output lost accepted candidate provenance')
                    counts[name]['observations'] += len(result.tracker_id)
                    counts[name]['camera_updates'] += 1
                    distinct[name][camera].update(int(x) for x in result.tracker_id)
                    item = {'camera': camera, 'local_ids': result.tracker_id.tolist(),
                            'xyxy': result.xyxy.tolist(), 'confidence': result.confidence.tolist()}
                    if name != 'baseline':
                        item.update(detection_indices=list(indices), embedding_rows=[embedding_rows[i] for i in indices])
                    record['variants'][name].append(item)
                    if name == 'direct_appearance':
                        stats.update(tracker.statistics)
            saved.write(json.dumps(record, allow_nan=False) + '\n')
            if (frame + 1) % 300 == 0 or frame == rounds - 1:
                print(f"Replayed {frame + 1}/{rounds}; appearance-blocked high-stage pairs={stats['high_blocked']}", flush=True)
        require(source_file.readline() == cache_file.readline() == '', 'Trailing source/cache frames')
    require(next_row == len(features) and total_candidates == cache['summary']['detections']
            and outside == cache['summary']['fully_outside'], 'Candidate/feature accounting differs')
    require(counts['baseline']['observations'] == source['summary']['observations'], 'Baseline count differs')
    frozen_hash = sha256(trace_path)
    print('Fork with changes disabled reproduces all baseline outputs: EXACT', flush=True)
    print('Phase 2: offline local tracking evaluation of frozen outputs...', flush=True)
    spec = source['inputs']['scene_source_manifest']
    manifest_path = checked('scene_source_manifest', spec['path'], spec['sha256'])
    manifest = json.loads(manifest_path.read_text())
    require(manifest['dataset'] == 'nvidia/PhysicalAI-SmartSpaces'
            and manifest['revision'] == '2cbe9563cbe9f47f846e5c871ee994572bbbc60e'
            and manifest['scene'] == 'MTMC_Tracking_2024/train/scene_001', 'Unexpected GT source')
    entries = [item for item in manifest['files'] if Path(item['local_path']).name == 'ground_truth.txt']
    require(len(entries) == 1, 'Missing/duplicate GT')
    gt_path = checked('ground_truth', ROOT / entries[0]['local_path'], entries[0]['sha256'])
    from evaluate_mtmc_sequence import load_ground_truth
    ground = load_ground_truth(gt_path, rounds)
    metrics, case = evaluate(trace_path, ground, rounds)
    require(sha256(trace_path) == frozen_hash, 'Predictions changed during evaluation')
    for spec in inputs.values():
        require(sha256(Path(spec['path'])) == spec['sha256'], 'Input changed during experiment')
    (output / 'case_tracks.json').write_text(json.dumps(case, indent=2, allow_nan=False) + '\n')
    code = [ROOT / 'src/mtmc/tracking/appearance_track.py',
            ROOT / 'src/mtmc/tracking/_vendor/experimental_bytetrack.py',
            ROOT / 'scripts/check_frozen_bytetrack_replay.py', ROOT / 'scripts/evaluate_local_tracking.py',
            ROOT / 'scripts/evaluate_mtmc_sequence.py', ROOT / 'scripts/evaluate_mtmc_pipeline.py',
            ROOT / 'scripts/experiment_appearance_bytetrack.py']
    report = {'completed': True, 'protocol': 'local_appearance_bytetrack_paired_v1', 'run_id': run,
              'source_run_id': source['run_id'], 'cache_run_id': cache['run_id'], 'inputs': inputs,
              'configuration': {'runtime_frames': [0, rounds - 1], 'evaluation_frames': [2, rounds - 1],
                                'cameras': list(CAMERAS), 'tracker': settings,
                                'appearance_threshold': args.appearance_threshold,
                                'history_size': 8, 'history_max_age_frames': 30,
                                'appearance_stages': ['high', 'unconfirmed'],
                                'history_updates': 'Accepted high-score candidates only',
                                'unavailable_appearance': 'Existing motion/score fallback',
                                'cost_policy': 'Keep original costs; set incompatible eligible pairs to infinity before original solver',
                                'baseline': 'Fork, appearance disabled, original output remapping',
                                'direct_iou': 'Fork, appearance disabled, accepted candidate output',
                                'direct_appearance': 'Fork, appearance enabled, accepted candidate output'},
              'baseline_exact': True, 'metrics': metrics, 'appearance_counters': dict(stats),
              'counts': {name: {**dict(counts[name]), 'distinct_local_ids': {str(c): len(distinct[name][c]) for c in CAMERAS}}
                         for name in VARIANTS},
              'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code},
              'versions': {name: version(name) for name in ('supervision', 'numpy', 'scipy', 'motmetrics', 'pandas')},
              'artifacts': {'tracks': {'path': trace_path.name, 'sha256': frozen_hash},
                            'case_tracks': {'path': 'case_tracks.json', 'sha256': sha256(output / 'case_tracks.json')}},
              'limits': ['Development experiment on a reused training-scene minute, not independent validation.',
                         'Threshold 0.6 is an experimental setting; no deployment threshold is selected.',
                         'Metric is pooled independent-camera local IDF1, not global IDF1.',
                         'Same detector candidates; returned boxes/counts can differ by output policy.',
                         'Low-score association remains motion-only; missing/stale features permit fallback.',
                         'History can still become contaminated; this is not a complete Deep SORT or BoT-SORT implementation.',
                         'Runtime remains sequential CPU replay; no end-to-end throughput claim.']}
    path = output / 'report.json'; path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'run_status.json').write_text(json.dumps({'completed': True, 'run_id': run}) + '\n')
    print('Variant               Local IDF1  Precision  Recall   IDSW     FP     FN')
    for name in VARIANTS:
        m = metrics[name]['OVERALL']
        print(f"{name:22} {100*m['idf1']:8.2f}% {100*m['precision']:8.2f}% {100*m['recall']:7.2f}% "
              f"{int(m['num_switches']):6} {int(m['num_false_positives']):6} {int(m['num_misses']):6}")
    print('Appearance counters:', json.dumps(dict(stats)))
    print('Selected camera-5 transition evidence:')
    for item in case:
        if item['frame'] in (813, 815, 830):
            print(json.dumps(item))
    print(f'Report: {path}')
    print('Paired local tracking experiment: COMPLETED; global evaluation and validation pending')


if __name__ == '__main__':
    main()
