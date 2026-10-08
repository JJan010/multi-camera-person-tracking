"""Offline spatial evidence around frozen dormant returns; no policy changes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

import numpy as np
from mtmc.data.ground_truth import load_ground_truth, spatial_slot, clip_boxes, pairwise_iou
from mtmc.data.scene import load_scene, require, sha256

ROOT = Path(__file__).resolve().parents[1]


def key(value):
    return (value['camera_id'], value['local_id'], value['frame_index'])


def camera_evidence(camera, ground, width, height, gate):
    """Keep every camera prediction when deciding mutual spatial uniqueness."""
    ids = camera['local_ids']
    require(len(ids) == len(set(ids)), 'Duplicate local IDs')
    gt_ids, mask, unique, _, _ = spatial_slot(
        ground, ids, camera['xyxy'], width=width, height=height, min_iou=gate)
    gt_boxes = clip_boxes([ground[g] for g in gt_ids], width, height)
    boxes = clip_boxes(camera['xyxy'], width, height)
    ious = pairwise_iou(gt_boxes, boxes)
    result = {}
    for j, local_id in enumerate(ids):
        candidates = np.flatnonzero(mask[:, j])
        if local_id in unique:
            reason = 'mutually_unique'
        elif not len(candidates):
            reason = 'no_admissible_gt'
        elif len(candidates) > 1:
            reason = 'multiple_admissible_gt'
        else:
            reason = 'gt_has_competing_predictions'
        best = int(np.argmax(ious[:, j])) if len(gt_ids) else None
        result[local_id] = {
            'unique_gt': unique.get(local_id), 'reason': reason,
            'raw_xyxy': camera['xyxy'][j],
            'best_gt': gt_ids[best] if best is not None else None,
            'best_iou': float(ious[best, j]) if best is not None else None,
            'admissible_gt': [
                {'gt_id': gt_ids[i], 'iou': float(ious[i, j]),
                 'competing_local_ids': [ids[k] for k in np.flatnonzero(mask[i]) if k != j]}
                for i in candidates],
        }
    return result


def inspect(trace, scene, spec, ground, run_id, events, context):
    requests = {}
    for number, event in enumerate(events, 1):
        prior = event['previous_visible_evidence']
        require(prior is not None, 'Missing previous-visible record')
        require(prior['frame_index'] < event['frame_index'], 'Invalid event interval')
        for side, members, boundary, start, stop in (
            ('before', prior['members'], prior['frame_index'],
             max(spec.first_frame, prior['frame_index'] - context), prior['frame_index']),
            ('after', event['members'], event['frame_index'],
             event['frame_index'], min(spec.last_frame, event['frame_index'] + context)),
        ):
            for member in members:
                c, local, frame = key(member)
                require(frame == boundary, 'Member boundary frame differs')
                for f in range(start, stop + 1):
                    requests.setdefault((f, c), []).append((number, side, local, boundary))
    sizes = {c.camera_id: (c.width, c.height) for c in scene.cameras}
    rows = []; observed_events = []
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        count = 0
        for frame, line in enumerate(stream):
            value = json.loads(line); count += 1
            require(value['run_id'] == run_id and value['frame_index'] == frame
                    and value['timestamp'] == str(Fraction(frame, scene.fps)), 'Trace scope/time differs')
            data = value['variants']['enabled']
            observed_events.extend((frame, e) for e in data['reactivations'])
            if not any((frame, c) in requests for c in scene.camera_ids):
                continue
            assignments = {key(a['key']): a['global_id'] for a in data['identity']['assignments']}
            cameras = {c['camera']: c for c in data['cameras']}
            for c in scene.camera_ids:
                if (frame, c) not in requests:
                    continue
                camera = cameras[c]; width, height = sizes[c]
                labels = camera_evidence(camera, ground.slots[frame, c], width, height, spec.min_iou)
                for number, side, local, boundary in requests[frame, c]:
                    row = {'event': number, 'side': side, 'frame_index': frame,
                           'camera_id': c, 'local_id': local, 'boundary': frame == boundary,
                           'present': local in labels}
                    if local in labels:
                        row.update(labels[local])
                        row['global_id'] = assignments[c, local, frame]
                    rows.append(row)
    require(count == scene.rounds, 'Truncated/extra trace rounds')
    require(observed_events == [(e['frame_index'], {k: e[k] for k in
            ('global_id', 'superseded_new_id', 'members')}) for e in events], 'Return events differ')
    summaries = []
    for number, event in enumerate(events, 1):
        selected = [r for r in rows if r['event'] == number]
        sides = {}
        for side in ('before', 'after'):
            current = [r for r in selected if r['side'] == side]
            present = [r for r in current if r['present']]
            known = [r for r in present if r['unique_gt'] is not None]
            prior = event['previous_visible_evidence']
            members = prior['members'] if side == 'before' else event['members']
            expected_labels = prior['labels'] if side == 'before' else event['current_labels']
            boundary = [r for r in current if r['boundary']]
            require(len(boundary) == len(members) and all(r['present'] for r in boundary),
                    'Boundary member missing')
            lookup = {(r['camera_id'], r['local_id']): r for r in boundary}
            actual = [lookup[m['camera_id'], m['local_id']]['unique_gt'] for m in members]
            require(actual == expected_labels and all(r['global_id'] == event['global_id'] for r in boundary),
                    'Original endpoint labels/global IDs not reproduced')
            # Closest evidence is a context observation, never a replacement endpoint label.
            closest_frame = (max if side == 'before' else min)(
                (r['frame_index'] for r in known), default=None)
            sides[side] = {
                'boundary': boundary,
                'unique_gt_counts': dict(Counter(r['unique_gt'] for r in known)),
                'reason_counts': dict(Counter(r['reason'] for r in present)),
                'absent_slots': len(current) - len(present),
                'nearest_known': [r for r in known if r['frame_index'] == closest_frame],
            }
        summaries.append({'event': number, 'global_id': event['global_id'],
                          'original_category': event['category'],
                          'gap_frames': event['frame_index'] - prior['frame_index'],
                          'gap_seconds': str(Fraction(event['frame_index'] - prior['frame_index'], scene.fps)),
                          **sides})
    return summaries, rows


def self_check():
    box = [0, 0, 10, 10]
    def result(gt, ids, boxes):
        return camera_evidence({'local_ids': ids, 'xyxy': boxes}, gt, 100, 100, .5)
    require(result({0: box}, [9], [box])[9]['unique_gt'] == 0, 'ID zero lost')
    require(result({}, [9], [box])[9]['reason'] == 'no_admissible_gt', 'Empty GT differs')
    require(result({0: box, 1: box}, [9], [box])[9]['reason'] == 'multiple_admissible_gt', 'GT ambiguity differs')
    competing = result({0: box}, [9, 10], [box, box])
    require(competing[9]['reason'] == 'gt_has_competing_predictions'
            and competing[9]['admissible_gt'][0]['competing_local_ids'] == [10], 'Competing prediction lost')
    require(result({0: box}, [9], [[0, 0, 5, 10]])[9]['unique_gt'] == 0, 'Inclusive gate differs')
    print('ID zero, empty GT, ambiguous GT, competing predictions and inclusive IoU: PASSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-report', type=Path)
    parser.add_argument('--context-frames', type=int, default=15)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        if args.recovery_report is None:
            return
    require(args.recovery_report is not None, '--recovery-report is required')
    require(0 <= args.context_frames <= 120, 'Context must be 0..120 frames')
    pins = {}
    def pin(path, expected=None):
        path = Path(path).resolve(); digest = sha256(path)
        require(expected is None or digest == expected, 'Changed input: ' + str(path))
        pins[str(path)] = digest
        return path
    report_path = pin(args.recovery_report)
    report = json.loads(report_path.read_text())
    require(report['completed'] is True and report['protocol'] == 'dormant_recovery_paired_v1', 'Invalid source report')
    require(report['checks']['all_reactivations_diagnosed'] is True
            and report['checks']['frozen_inputs_outputs_unchanged'] is True, 'Source checks incomplete')
    artifacts = {}
    for name in ('global_tracks.jsonl.gz', 'reactivation_diagnostics.json'):
        item = report['artifacts'][name]
        path = (report_path.parent / item['path']).resolve()
        require(path.parent == report_path.parent, 'Artifact escaped run directory')
        artifacts[name] = pin(path, item['sha256'])
    ref = report['inputs']['scene_config']; config = pin(ref['path'], ref['sha256'])
    inputs = load_scene(config, project_root=ROOT)
    ref = report['inputs']['ground_truth']
    require(Path(ref['path']).resolve() == inputs.evaluation.ground_truth.path
            and ref['sha256'] == inputs.evaluation.ground_truth.sha256, 'Mixed GT source')
    pin(ref['path'], ref['sha256'])
    require(inputs.runtime.scene == report['scene'], 'Mixed scene')
    events = json.loads(artifacts['reactivation_diagnostics.json'].read_text())
    require(len(events) == report['summary']['lifecycle']['enabled']['reactivation_events'], 'Event count differs')
    print('Frozen return context; CPU, no models or decoding; all camera predictions retained...')
    ground = load_ground_truth(inputs.evaluation)
    summaries, observations = inspect(artifacts['global_tracks.jsonl.gz'], inputs.runtime,
        inputs.evaluation, ground, report['run_id'], events, args.context_frames)
    for path, digest in pins.items():
        require(sha256(path) == digest, 'Input changed during diagnostic: ' + path)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts' / 'dormant_recovery_context' / run
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / 'observations.json'
    rows_path.write_text(json.dumps(observations, indent=2, allow_nan=False) + '\n')
    result = {'completed': True, 'protocol': 'dormant_recovery_context_v1', 'source_run_id': report['run_id'],
              'context_frames_each_side': args.context_frames, 'inputs': pins, 'events': summaries,
              'checks': {'all_endpoint_labels_reproduced': True, 'return_events_reproduced': True,
                         'frozen_inputs_unchanged': True},
              'artifacts': {'observations': {'path': rows_path.name, 'sha256': sha256(rows_path)}},
              'code_sha256': sha256(Path(__file__)),
              'limits': ['GT context is offline, including future observations after return.',
                         'Context labels do not replace endpoint labels or prove archive descriptor purity.',
                         'All camera predictions participate in mutual spatial uniqueness.',
                         'No predictions, metrics or runtime settings were changed.']}
    path = output / 'report.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for event in summaries:
        print(f"\nGID {event['global_id']}; gap={event['gap_frames']} frames ({float(Fraction(event['gap_seconds'])):.3f}s); {event['original_category']}")
        for side in ('before', 'after'):
            value = event[side]
            print(f"  {side}: unique labels={value['unique_gt_counts']}; reasons={value['reason_counts']}; absent={value['absent_slots']}")
            for r in value['boundary']:
                print('    Endpoint:', json.dumps(r, allow_nan=False))
            print('    Nearest known:', json.dumps([
                {k: r[k] for k in ('frame_index', 'camera_id', 'local_id', 'global_id', 'unique_gt')}
                for r in value['nearest_known']]))
    print(f'\nReport: {path}')
    print('Dormant return context: COMPLETED; original endpoint labels reproduced; assignments unchanged')


if __name__ == '__main__':
    main()
