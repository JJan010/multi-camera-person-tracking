"""Paired CPU replay of high/low candidate competition; pinned appearance baseline."""
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
VARIANTS = ('staged', 'competitive_iou', 'competitive_appearance')
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
                    if ((camera == 5 and frame in (813, 815, 830))
                            or (camera == 8 and frame in (1438, 1439, 1440, 1441, 1446))):
                        keys = tuple(ObservationKey(camera, i, frame) for i in ids)
                        _, _, unique, _, _ = spatial_slot(gt, keys, np.asarray(item['xyxy']).reshape(-1, 4))
                        for key in keys:
                            label = unique.get(key)
                            if (camera == 5 and (label in (0, 23) or key.local_id == 11)) or (camera == 8 and (label in (1, 13) or key.local_id == 44)):
                                case.append({'variant': variant, 'frame': frame, 'camera': camera,
                                             'local_id': key.local_id, 'unique_overlap_gt': label})
            require(stream.readline() == '', 'Trailing experimental trace')
        table = mm.metrics.create().compute_many(list(accumulators.values()), metrics=names,
                    names=[f'camera_{c:04d}' for c in CAMERAS], generate_overall=True)
        metrics[variant] = json.loads(table.to_json(orient='index'))
    return metrics, case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-report', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        actual = sha256(path)
        require(expected is None or actual == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path

    print('Verifying frozen appearance experiment, source code and candidate cache...', flush=True)
    local_path = checked('local_report', args.local_report)
    local = json.loads(local_path.read_text())
    require(local.get('completed') is True and local.get('baseline_exact') is True
            and local['protocol'] == 'local_appearance_bytetrack_paired_v1', 'Expected completed appearance experiment')
    for name, spec in local['inputs'].items():
        checked(name, spec['path'], spec['sha256'])
    for name, digest in local['code_sha256'].items():
        checked('code:' + name, ROOT / name, digest)
    spec = local['artifacts']['tracks']
    prior_trace = checked('local_variants', local_path.parent / spec['path'], spec['sha256'])
    require(version('supervision') == '0.30.7', 'Expected supervision 0.30.7')
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    require(source['run_id'] == local['source_run_id'] and cache['run_id'] == local['cache_run_id'], 'Mixed source scope')
    rounds = source['summary']['rounds']
    require(local['configuration']['runtime_frames'] == [0, rounds - 1]
            and local['configuration']['evaluation_frames'] == [2, rounds - 1]
            and local['configuration']['history_size'] == 8
            and local['configuration']['history_max_age_frames'] == 30, 'Unsupported baseline configuration')
    settings = local['configuration']['tracker']
    threshold = local['configuration']['appearance_threshold']
    require(settings == source['configuration']['tracker'], 'Tracker configuration differs')
    features = np.load(inputs['embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(features.dtype == np.float32 and features.shape == (cache['summary']['encoded'], 512), 'Invalid feature cache')
    from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack
    from mtmc.tracking.competitive import CompetitiveByteTrack
    common = dict(**settings, direct_output=True, appearance_threshold=threshold)
    trackers = {
        'staged': {c: ExperimentalByteTrack(**common) for c in CAMERAS},
        'competitive_iou': {c: CompetitiveByteTrack(**common, refinement='iou') for c in CAMERAS},
        'competitive_appearance': {c: CompetitiveByteTrack(**common, refinement='appearance') for c in CAMERAS},
    }
    disabled = {c: CompetitiveByteTrack(**common, refinement='disabled') for c in CAMERAS}
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/competitive_bytetrack' / run
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run_status.json').write_text(json.dumps({'completed': False, 'run_id': run}) + '\n')
    trace_path = output / 'tracks.jsonl.gz'
    stats = {name: Counter() for name in VARIANTS}
    counts = {name: Counter() for name in VARIANTS}
    distinct = {name: {c: set() for c in CAMERAS} for name in VARIANTS}
    next_row = candidate_count = outside = 0

    def record_camera(tracker, result, boxes, scores, rows, camera):
        indices = list(tracker.selected_indices)
        require(np.array_equal(result.xyxy, boxes[indices])
                and np.array_equal(result.confidence, scores[indices]), 'Candidate output provenance differs')
        return {'camera': camera, 'local_ids': result.tracker_id.tolist(), 'xyxy': result.xyxy.tolist(),
                'confidence': result.confidence.tolist(), 'detection_indices': indices,
                'embedding_rows': [rows[i] for i in indices]}

    print(f'Phase 1: {rounds} causal rounds; three variants + disabled control; no GT input', flush=True)
    with Path(inputs['tracks']['path']).open() as original_file, Path(inputs['detections.jsonl']['path']).open() as cache_file, \
            gzip.open(prior_trace, 'rt') as prior_file, gzip.open(trace_path, 'wt') as saved:
        for frame in range(rounds):
            lines = [f.readline() for f in (original_file, cache_file, prior_file)]
            require(all(lines), 'Truncated replay input')
            original, candidates, prior = map(json.loads, lines)
            require(prior['frame_index'] == frame and prior['run_id'] == local['run_id']
                    and prior['source_run_id'] == source['run_id'], 'Mixed prior frame scope')
            expected = {c['camera']: c for c in prior['variants']['direct_appearance']}
            require(set(expected) == set(CAMERAS) and len(prior['variants']['direct_appearance']) == len(CAMERAS),
                    'Invalid prior cameras')
            arrays, _, next_row, missing = candidate_arrays(candidates, original, frame=frame,
                cache_run=cache['run_id'], source_run=source['run_id'], next_row=next_row)
            outside += missing
            record = {'run_id': run, 'source_run_id': source['run_id'], 'frame_index': frame,
                      'timestamp': str(Fraction(frame, 30)), 'variants': {n: [] for n in VARIANTS},
                      'refinements': {n: [] for n in VARIANTS if n != 'staged'}}
            for camera in CAMERAS:
                boxes, scores, rows = arrays[camera]
                candidate_count += len(boxes)
                vectors = [features[i] if i is not None else None for i in rows]
                control = disabled[camera]
                result = control.update_candidates(boxes, scores, vectors)
                require(record_camera(control, result, boxes, scores, rows, camera) == expected[camera],
                        f'Disabled refinement differs from frozen appearance baseline at {camera}/{frame}')
                for name in VARIANTS:
                    tracker = trackers[name][camera]
                    result = tracker.update_candidates(boxes, scores, vectors)
                    item = record_camera(tracker, result, boxes, scores, rows, camera)
                    if name == 'staged':
                        require(item == expected[camera], f'Staged baseline differs at {camera}/{frame}')
                    else:
                        stats[name].update(tracker.refinement_counts)
                        record['refinements'][name].extend({'camera': camera, **event} for event in tracker.refinement_events)
                    record['variants'][name].append(item)
                    counts[name]['observations'] += len(result.tracker_id)
                    counts[name]['camera_updates'] += 1
                    distinct[name][camera].update(item['local_ids'])
            saved.write(json.dumps(record, allow_nan=False) + '\n')
            if (frame + 1) % 300 == 0 or frame == rounds - 1:
                print(f"Replayed {frame+1}/{rounds}; replacements: IoU={stats['competitive_iou']['replacements']}, "
                      f"appearance={stats['competitive_appearance']['replacements']}", flush=True)
        require(all(f.readline() == '' for f in (original_file, cache_file, prior_file)), 'Trailing replay frames')
    require(next_row == len(features) and candidate_count == cache['summary']['detections']
            and outside == cache['summary']['fully_outside'], 'Candidate accounting differs')
    require(counts['staged']['observations'] == local['counts']['direct_appearance']['observations'], 'Baseline count differs')
    frozen_hash = sha256(trace_path)
    print('Staged baseline and disabled refinement reproduce all frozen camera outputs: EXACT', flush=True)
    print('Phase 2: offline local evaluation; GT is read after freezing all predictions...', flush=True)
    from evaluate_mtmc_sequence import load_ground_truth
    ground = load_ground_truth(Path(inputs['ground_truth']['path']), rounds)
    metrics, case = evaluate(trace_path, ground, rounds)
    for camera, values in local['metrics']['direct_appearance'].items():
        for key, value in values.items():
            actual = metrics['staged'][camera][key]
            require((actual is None and value is None) or
                    (actual is not None and value is not None and abs(actual - value) <= 1e-9),
                    f'Baseline metric differs: {camera}/{key}')
    require(sha256(trace_path) == frozen_hash, 'Predictions changed during evaluation')
    for spec in inputs.values():
        require(sha256(Path(spec['path'])) == spec['sha256'], 'Input changed during experiment')
    case_path = output / 'case_tracks.json'
    case_path.write_text(json.dumps(case, indent=2, allow_nan=False) + '\n')
    code = [ROOT / 'scripts/experiment_competitive_bytetrack.py', ROOT / 'src/mtmc/tracking/competitive.py',
            ROOT / 'src/mtmc/association/pairwise.py']
    report = {'completed': True, 'protocol': 'competitive_bytetrack_paired_v1', 'run_id': run,
              'source_run_id': source['run_id'], 'cache_run_id': cache['run_id'], 'inputs': inputs,
              'configuration': {'runtime_frames': [0, rounds-1], 'evaluation_frames': [2, rounds-1],
                                'tracker': settings, 'appearance_threshold': threshold,
                                'variants': list(VARIANTS), 'weak_motion_min_iou': .5,
                                'challenge_appearance_gate': 'Inclusive existing threshold; both features required',
                                'challenge_objective': 'Maximum summed strictly positive improvement over incumbent',
                                'incumbents': 'Original high-stage matches of activated currently tracked tracks only',
                                'weak_reservation': 'IoU >= 0.5 to any unmatched active predicted track reserves the candidate for stage two',
                                'history_updates': 'Original accepted high-score-only policy',
                                'missing_appearance': 'No challenge; original staged fallback retained',
                                'lost_unconfirmed_birth_policy': 'Unchanged; released high detections remain available'},
              'checks': {'staged_trace_exact': True, 'disabled_trace_exact': True, 'baseline_metrics_reproduced': True,
                         'all_candidates_accounted': True, 'frozen_predictions': True},
              'metrics': metrics, 'refinement_counters': {n: dict(v) for n,v in stats.items()},
              'counts': {n: {**dict(counts[n]), 'distinct_local_ids': {str(c): len(distinct[n][c]) for c in CAMERAS}}
                         for n in VARIANTS},
              'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in code},
              'versions': {n: version(n) for n in ('supervision','numpy','scipy','motmetrics','pandas')},
              'artifacts': {'tracks': {'path': trace_path.name, 'sha256': frozen_hash},
                            'case_tracks': {'path': case_path.name, 'sha256': sha256(case_path)}},
              'limits': ['Reused development sequence; no independent validation or deployment threshold selection.',
                         'Refinement is joint only among eligible incumbent tracks and weak candidates.',
                         'High-stage assignments for lost tracks and unmatched-track low recovery retain original policy.',
                         'Freed high detections may create duplicate tracks; FP, FN and fragmentation must be evaluated.',
                         'No end-to-end throughput or global identity improvement claim; global evaluation pending.']}
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'run_status.json').write_text(json.dumps({'completed': True, 'run_id': run}) + '\n')
    print('Variant                    Local IDF1 Precision Recall   IDSW     FP     FN')
    for name in VARIANTS:
        m = metrics[name]['OVERALL']
        print(f"{name:26} {100*m['idf1']:8.2f}% {100*m['precision']:8.2f}% {100*m['recall']:6.2f}% "
              f"{int(m['num_switches']):6} {int(m['num_false_positives']):6} {int(m['num_misses']):6}")
    print('Refinement counters:', json.dumps(report['refinement_counters']))
    print('Camera 8 transition evidence (offline GT diagnostic):')
    for row in case:
        if row['camera'] == 8:
            print(json.dumps(row))
    print(f'Report: {path}')
    print('Competitive tracking experiment: COMPLETED; global evaluation pending; runtime baseline unchanged')


if __name__ == '__main__':
    main()


