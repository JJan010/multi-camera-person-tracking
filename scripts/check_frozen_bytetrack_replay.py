"""Reproduce installed ByteTrack outputs from a verified detector-candidate cache."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import distribution, version
import json
from pathlib import Path

import numpy as np

from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import sha256

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4, 5, 8)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def candidate_arrays(cache_row, source_row, *, frame, cache_run, source_run, next_row):
    """Verify cache rows against the original detections before any dtype conversion."""
    require(cache_row['cache_run_id'] == cache_run and cache_row['source_run_id'] == source_run
            and source_row['run_id'] == source_run, 'Mixed run scopes')
    for row in (cache_row, source_row):
        require(type(row['frame_index']) is int and row['frame_index'] == frame
                and Fraction(row['timestamp']) == Fraction(frame, 30), 'Frame/time mismatch')
    cameras = {c['camera']: c for c in source_row['cameras']}
    require(len(cameras) == len(source_row['cameras']) == 3 and set(cameras) == set(CAMERAS),
            'Source camera coverage differs')
    records = {(r['camera_id'], r['detection_index']): r for r in cache_row['detections']}
    require(len(records) == len(cache_row['detections']), 'Duplicate candidate key')
    arrays, expected_keys, outside = {}, set(), 0
    for camera in CAMERAS:
        original = cameras[camera]
        boxes = np.asarray(original['detector_xyxy'], dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(original['detector_confidence'], dtype=np.float64)
        require(scores.shape == (len(boxes),) and np.isfinite(boxes).all()
                and np.all(boxes[:, 2:] > boxes[:, :2]) and np.isfinite(scores).all()
                and np.all((scores >= 0) & (scores <= 1)), 'Invalid source candidates')
        indices = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            key = camera, index
            require(key in records, 'Missing cached detection')
            item = records[key]
            require(type(item['camera_id']) is int and type(item['detection_index']) is int
                    and type(item['frame_index']) is int and item['frame_index'] == frame
                    and np.array_equal(np.asarray(item['xyxy'], dtype=np.float64), box)
                    and item['confidence'] == float(score), 'Frozen detection changed')
            bounds, fraction = crop_geometry(box, 1920, 1080)
            stored_bounds = tuple(item['crop_xyxy_int']) if item['crop_xyxy_int'] is not None else None
            require(stored_bounds == bounds and item['inside_image_fraction'] == fraction,
                    'Cached crop geometry differs')
            if bounds is None:
                require(item['embedding_row'] is None, 'Fully outside detection has a descriptor')
                outside += 1
            else:
                require(type(item['embedding_row']) is int and item['embedding_row'] == next_row,
                        'Embedding rows are not contiguous in source candidate order')
                next_row += 1
            indices.append(item['embedding_row'])
            expected_keys.add(key)
        # These dtypes reproduce RFDETRPersonDetector output and CameraTracker input.
        arrays[camera] = (boxes.astype(np.float32), scores.astype(np.float32), tuple(indices))
        require(np.array_equal(arrays[camera][0].astype(np.float64), boxes)
                and np.array_equal(arrays[camera][1].astype(np.float64), scores),
                'Source detection values are not losslessly representable as baseline float32')
    require(set(records) == expected_keys, 'Unexpected cached candidate')
    return arrays, cameras, next_row, outside


def compare_outputs(actual, expected):
    """Compare original detector-order outputs exactly, including score and local ID."""
    expected_ids = np.asarray(expected['local_ids'], dtype=np.int64)
    expected_boxes = np.asarray(expected['xyxy'], dtype=np.float64).reshape(-1, 4)
    expected_scores = np.asarray(expected['confidence'], dtype=np.float64)
    require(expected_boxes.shape == (len(expected_ids), 4)
            and expected_scores.shape == (len(expected_ids),)
            and len(np.unique(expected_ids)) == len(expected_ids), 'Invalid recorded tracks')
    actual_ids = np.asarray(actual.tracker_id)
    actual_boxes = np.asarray(actual.xyxy).reshape(-1, 4)
    actual_scores = np.asarray(actual.confidence)
    result = {'ids_equal': np.array_equal(actual_ids, expected_ids),
              'boxes_equal': np.array_equal(actual_boxes, expected_boxes),
              'scores_equal': np.array_equal(actual_scores, expected_scores),
              'expected_tracks': len(expected_ids), 'actual_tracks': len(actual_ids)}
    result['passed'] = all(result[name] for name in ('ids_equal', 'boxes_equal', 'scores_equal'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-report', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path

    print('Verifying cache, source detections and installed tracker code...', flush=True)
    cache_path = checked('cache_report', args.cache_report)
    cache = json.loads(cache_path.read_text())
    require(cache.get('completed') is True and cache['protocol'] == 'frozen_detector_candidate_embeddings_v1',
            'Expected completed detector candidate cache')
    for name in ('pipeline_report', 'tracks'):
        item = cache['inputs'][name]
        checked(name, item['path'], item['sha256'])
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    cfg, rounds = source['configuration'], source['summary']['rounds']
    require(source.get('completed') is True and source['protocol'] == 'scene_001_mtmc_sequential_fp32_v1'
            and source['run_id'] == cache['source_run_id'] and cfg['fps'] == 30
            and cfg['cameras'] == list(CAMERAS) and type(rounds) is int and 0 < rounds <= 23994
            and source['artifacts']['tracks']['sha256'] == inputs['tracks']['sha256'],
            'Pipeline/cache scope differs')
    require(cache['configuration']['frames'] == [0, rounds - 1]
            and cache['configuration']['rounds'] == rounds
            and cache['configuration']['cameras'] == list(CAMERAS)
            and cache['configuration']['fps'] == 30, 'Cache coverage differs')
    for name in ('detections.jsonl', 'embeddings.npy'):
        item = cache['artifacts'][name]
        checked(name, cache_path.parent / item['path'], item['sha256'])
    features = np.load(inputs['embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(features.dtype == np.float32 and features.shape == (cache['summary']['encoded'], 512),
            'Invalid cached feature shape/dtype')
    for first in range(0, len(features), 4096):
        block = features[first:first + 4096]
        require(np.isfinite(block).all() and np.allclose(np.linalg.norm(block, axis=1), 1, rtol=0, atol=1e-5),
                'Nonfinite or unnormalized cached features')
    audit_path = checked('tracker_source_audit', ROOT / 'docs/tracking/supervision_source_audit.json')
    audit = json.loads(audit_path.read_text())
    require(version('supervision') == audit['version'] == '0.30.7', 'Different supervision version')
    installed = distribution('supervision')
    for name, digest in audit['source_sha256'].items():
        path = installed.locate_file(name) if name.startswith('supervision/') else ROOT / name
        checked('code:' + name, path, digest)
    adapter_path = 'src/mtmc/tracking/bytetrack.py'
    require(source['code_sha256'][adapter_path] == audit['source_sha256'][adapter_path],
            'Baseline used a different local adapter')
    settings = cfg['tracker']
    require(settings == {'track_activation_threshold': .5, 'lost_track_buffer': 30,
                         'minimum_matching_threshold': .8, 'frame_rate': 30,
                         'minimum_consecutive_frames': 1}, 'Unexpected baseline settings')
    # Use the same public API as CameraTracker without importing the GPU detector adapter.
    import supervision as sv
    trackers = {camera: sv.ByteTrack(**settings) for camera in CAMERAS}
    counts, rows, examples = Counter(), [], []
    next_row = 0
    print(f'Replaying {rounds} rounds on CPU; no models, video decoding or GT...', flush=True)
    with Path(inputs['tracks']['path']).open() as source_file, \
            Path(inputs['detections.jsonl']['path']).open() as cache_file:
        for frame in range(rounds):
            a, b = source_file.readline(), cache_file.readline()
            require(bool(a) and bool(b), 'Truncated source/cache trace')
            original, candidates = json.loads(a), json.loads(b)
            arrays, expected, next_row, outside = candidate_arrays(
                candidates, original, frame=frame, cache_run=cache['run_id'],
                source_run=source['run_id'], next_row=next_row)
            counts['fully_outside'] += outside
            for camera in CAMERAS:
                boxes, scores, embedding_rows = arrays[camera]
                detections = sv.Detections(xyxy=boxes.copy(), confidence=scores.copy(),
                                           class_id=np.zeros(len(boxes), dtype=int))
                actual = trackers[camera].update_with_detections(detections)
                comparison = compare_outputs(actual, expected[camera])
                rows.append({'frame': frame, 'camera': camera, **comparison})
                counts['camera_updates'] += 1
                counts['detections'] += len(boxes)
                counts[f'camera_{camera}_detections'] += len(boxes)
                counts['expected_track_observations'] += comparison['expected_tracks']
                counts['replayed_track_observations'] += comparison['actual_tracks']
                counts['matching_camera_updates'] += int(comparison['passed'])
                counts['mismatching_camera_updates'] += int(not comparison['passed'])
                if not comparison['passed'] and len(examples) < 3:
                    examples.append({'frame': frame, 'camera': camera, 'comparison': comparison,
                                     'expected': {k: expected[camera][k] for k in ('local_ids', 'xyxy', 'confidence')},
                                     'actual': {'local_ids': actual.tracker_id.tolist(),
                                                'xyxy': actual.xyxy.tolist(), 'confidence': actual.confidence.tolist()}})
            if (frame + 1) % 300 == 0 or frame == rounds - 1:
                print(f"Processed {frame + 1}/{rounds}; mismatching camera updates={counts['mismatching_camera_updates']}",
                      flush=True)
        require(source_file.readline() == cache_file.readline() == '', 'Trailing source/cache frames')
    require(next_row == len(features) and counts['detections'] == cache['summary']['detections']
            and counts['fully_outside'] == cache['summary']['fully_outside'], 'Cache counters differ')
    for camera in CAMERAS:
        require(counts[f'camera_{camera}_detections'] == cache['summary'].get(f'camera_{camera}', 0),
                'Per-camera candidate count differs')
    require(counts['expected_track_observations'] == source['summary']['observations'],
            'Baseline observation count differs')
    for item in inputs.values():
        require(sha256(Path(item['path'])) == item['sha256'], 'Input changed during replay')
    output = ROOT / 'artifacts/frozen_bytetrack_replay' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'frame_parity.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    passed = counts['mismatching_camera_updates'] == 0
    report = {'completed': True, 'passed': passed, 'protocol': 'frozen_bytetrack_exact_replay_v1',
              'source_run_id': source['run_id'], 'cache_run_id': cache['run_id'],
              'configuration': {'frames': [0, rounds - 1], 'cameras': list(CAMERAS), 'tracker': settings,
                                'detections_dtype': 'float32', 'output_order': 'Original detector order',
                                'ground_truth_used': False, 'appearance_used_by_tracker': False},
              'summary': dict(counts), 'first_mismatches': examples, 'inputs': inputs,
              'versions': {name: version(name) for name in ('supervision', 'numpy', 'scipy')},
              'code_sha256': sha256(Path(__file__)),
              'artifacts': {'frame_parity.csv': {'path': 'frame_parity.csv', 'sha256': sha256(output / 'frame_parity.csv')}},
              'limits': ['Exact reproduction gate, not an appearance-assisted tracking experiment.',
                         'Cached descriptors are validated but are not passed to ByteTrack.',
                         'Equal outputs establish reference parity, not tracking correctness.',
                         'No new quality metrics or speed claims.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Summary:', json.dumps(dict(counts)))
    print(f'Report: {output / "report.json"}')
    if not passed:
        print('Frozen ByteTrack replay: FAILED; inspect first_mismatches before modifying association')
        raise SystemExit(1)
    print('All cached candidates and embedding rows: VERIFIED')
    print('All local IDs, boxes, scores and output order: EXACT')
    print('Frozen ByteTrack replay: PASSED')


if __name__ == '__main__':
    main()
