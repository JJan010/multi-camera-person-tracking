"""Observe the pinned tracker's decisions around one diagnostic label transition."""
import argparse
from datetime import datetime, timezone
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from check_frozen_bytetrack_replay import candidate_arrays, compare_outputs, require
from mtmc.reid.osnet import sha256

ROOT = Path(__file__).resolve().parents[1]


def cosine(a, b):
    if a is None or b is None:
        return None
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def observer_class():
    from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack

    class Observer(ExperimentalByteTrack):
        def __init__(self, *args, target_local_id, **kwargs):
            super().__init__(*args, **kwargs)
            self.target_local_id = target_local_id
            self.audit_enabled = False
            self.gate_audit = []

        def _appearance_gate(self, costs, tracks, detections, cost_limit, phase):
            output = super()._appearance_gate(costs, tracks, detections, cost_limit, phase)
            if self.audit_enabled:
                for i, track in enumerate(tracks):
                    if track.external_track_id != self.target_local_id:
                        continue
                    descriptor = track.descriptor(self.frame_id)
                    self.gate_audit.append({'phase': phase, 'candidates': [
                        {'detection_index': int(det.candidate_index), 'score': float(det.score),
                         'original_cost': float(costs[i, j]), 'motion_eligible': bool(costs[i, j] <= cost_limit),
                         'cosine_prior': cosine(descriptor, det.current_feature),
                         'appearance_blocked': bool(np.isinf(output[i, j])),
                         'appearance_unavailable': descriptor is None or det.current_feature is None}
                        for j, det in enumerate(detections)]})
            return output
    return Observer


def snapshot(tracker, frame, target):
    tracks = [t for t in [*tracker.tracked_tracks, *tracker.lost_tracks] if t.external_track_id == target]
    require(len(tracks) <= 1, 'Duplicate target track state')
    if not tracks:
        return None, []
    t = tracks[0]
    # ByteTrack's internal frame 1 corresponds to scene frame 0.
    return t.descriptor(frame + 1), [f - 1 for f, _ in t.appearance_history if 0 <= frame + 1 - f <= 30]


def observed_step(tracker, boxes, scores, vectors, target, frame, enabled):
    prior, frames = snapshot(tracker, frame, target)
    tracker.audit_enabled = enabled; tracker.gate_audit = []
    actual = tracker.update_candidates(boxes, scores, vectors)
    if not enabled:
        return actual, None
    row = {'frame': frame, 'local_id': target, 'prior_strong_sample_frames': frames,
           'visible': False, 'gate_audit': tracker.gate_audit}
    indices = [i for i, identity in enumerate(actual.tracker_id) if identity == target]
    require(len(indices) <= 1, 'Duplicate visible target')
    if indices:
        selected = tracker.selected_indices[indices[0]]
        score = float(scores[selected]); feature = vectors[selected]
        matches = [(p['phase'], d) for p in tracker.gate_audit for d in p['candidates'] if d['detection_index'] == selected]
        if score < tracker.track_activation_threshold:
            require(score > .1, 'Unexpected accepted low-score candidate')
            phase = 'low_score_iou_without_appearance_gate'
        elif matches:
            require(len(matches) == 1 and matches[0][1]['motion_eligible'] and not matches[0][1]['appearance_blocked'],
                    'Selected high candidate contradicts observed gate')
            phase = matches[0][0]
        else:
            phase = 'new_identity_or_preconfirmation_without_target_external_id'
        row.update(visible=True, detection_index=int(selected), score=score, selected_phase=phase,
                   cosine_prior=cosine(prior, feature))
    return actual, row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-report', type=Path, required=True)
    parser.add_argument('--camera', type=int, required=True)
    parser.add_argument('--local-id', type=int, required=True)
    args = parser.parse_args()
    require(args.camera in (4, 5, 8) and args.local_id > 0, 'Invalid target')
    inputs = {}
    def checked(name, path, digest=None):
        path = Path(path).resolve(); value = sha256(path)
        require(digest is None or value == digest, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': value}
        return path
    diagnostic_path = checked('diagnostic_report', args.diagnostic_report)
    diagnostic = json.loads(diagnostic_path.read_text())
    require(diagnostic.get('completed') is True and diagnostic['protocol'] == 'frozen_appearance_continuity_diagnostic_v1'
            and diagnostic['variant'] == 'direct_appearance', 'Expected appearance continuity diagnostic')
    spec = diagnostic['artifacts']['local_label_transitions.json']
    transitions = json.loads(checked('transitions', diagnostic_path.parent / spec['path'], spec['sha256']).read_text())
    transitions = sorted([t for t in transitions if (t['camera'], t['local_id']) == (args.camera, args.local_id)],
                         key=lambda t: (t['frame'], t['previous_evidence_frame']))
    require(bool(transitions), 'Target has no recorded evidence-label transition')
    selected = transitions[0]
    spec = diagnostic['inputs']['local_report']
    local_path = checked('local_report', spec['path'], spec['sha256']); local = json.loads(local_path.read_text())
    require(local.get('completed') is True and local.get('baseline_exact') is True
            and local['protocol'] == 'local_appearance_bytetrack_paired_v1', 'Invalid local experiment')
    for name, spec in local['inputs'].items():
        checked('local:' + name, spec['path'], spec['sha256'])
    spec = local['artifacts']['tracks']
    local_trace = checked('local_variants', local_path.parent / spec['path'], spec['sha256'])
    for name, digest in local['code_sha256'].items():
        checked('code:' + name, ROOT / name, digest)
    require(version('supervision') == '0.30.7', 'Expected pinned supervision')
    source = json.loads(Path(inputs['local:pipeline_report']['path']).read_text())
    cache = json.loads(Path(inputs['local:cache_report']['path']).read_text())
    require(source['run_id'] == local['source_run_id'] and cache['run_id'] == local['cache_run_id'], 'Mixed source scopes')
    features = np.load(inputs['local:embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(features.dtype == np.float32 and features.shape == (cache['summary']['encoded'], 512), 'Invalid cached vectors')
    start = max(0, selected['previous_evidence_frame'] - 10)
    end = min(source['summary']['rounds'] - 1, selected['frame'] + 15)
    # Bound report size if evidence observations are separated by a long gap.
    selected_frames = set(range(start, selected['previous_evidence_frame'] + 1)) | set(range(max(0, selected['frame'] - 10), end + 1))
    tracker = observer_class()(target_local_id=args.local_id, direct_output=True,
        appearance_threshold=local['configuration']['appearance_threshold'], **local['configuration']['tracker'])
    print('Selected earliest target evidence transition:', json.dumps(selected), flush=True)
    print(f'Replaying camera {args.camera}, scene frames 0..{end}; GT labels are not tracker inputs...', flush=True)
    timeline, next_row = [], 0
    with Path(inputs['local:tracks']['path']).open() as src, Path(inputs['local:detections.jsonl']['path']).open() as det, \
            gzip.open(local_trace, 'rt') as saved:
        for frame in range(end + 1):
            lines = [f.readline() for f in (src, det, saved)]; require(all(lines), 'Truncated replay input')
            original, candidates, expected = map(json.loads, lines)
            require(expected['frame_index'] == frame and expected['run_id'] == local['run_id']
                    and expected['source_run_id'] == source['run_id'], 'Local replay scope differs')
            arrays, _, next_row, _ = candidate_arrays(candidates, original, frame=frame,
                cache_run=cache['run_id'], source_run=source['run_id'], next_row=next_row)
            boxes, scores, rows = arrays[args.camera]
            vectors = [features[i] if i is not None else None for i in rows]
            actual, observation = observed_step(tracker, boxes, scores, vectors, args.local_id, frame, frame in selected_frames)
            expected_camera = [c for c in expected['variants']['direct_appearance'] if c['camera'] == args.camera]
            require(len(expected_camera) == 1 and compare_outputs(actual, expected_camera[0])['passed'],
                    f'Observer replay changed output at frame {frame}')
            require(list(tracker.selected_indices) == expected_camera[0]['detection_indices']
                    and [rows[i] for i in tracker.selected_indices] == expected_camera[0]['embedding_rows'],
                    'Accepted candidate provenance differs')
            if observation is not None:
                timeline.append(observation)
            if (frame + 1) % 600 == 0:
                print(f'Replayed {frame + 1}/{end + 1}; all target-camera outputs EXACT', flush=True)
    for spec in inputs.values():
        require(sha256(Path(spec['path'])) == spec['sha256'], 'Input changed during probe')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/local_gate_probe' / run; output.mkdir(parents=True, exist_ok=False)
    timeline_path = output / 'timeline.json'; timeline_path.write_text(json.dumps(timeline, indent=2, allow_nan=False) + '\n')
    report = {'completed': True, 'protocol': 'frozen_local_appearance_gate_probe_v1', 'run_id': run,
              'inputs': inputs, 'target': {'camera': args.camera, 'local_id': args.local_id}, 'selected_transition': selected,
              'configuration': local['configuration'], 'replayed_frames': [0, end], 'exact_camera_updates': end + 1,
              'artifacts': {'timeline': {'path': 'timeline.json', 'sha256': sha256(timeline_path)}},
              'script_sha256': sha256(Path(__file__)),
              'limits': ['Case selected offline from mutually unique GT evidence; no labels supplied to the tracker.',
                         'Evidence transitions, especially across gaps, are not proof of an instantaneous tracker switch.',
                         'Observer returns the original gate output and reproduces all target-camera outputs/provenance exactly.',
                         'The low-score stage is identified by the pinned score partition; its IoU costs are not instrumented.',
                         'Gate records explain this replay, not a counterfactual improvement from changing policy.']}
    report_path = output / 'report.json'; report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('Frame / score / accepted stage / cos(prior strong history) / strong samples')
    for row in timeline:
        if not row['visible']:
            print(row['frame'], 'not visible', 'strong samples:', len(row['prior_strong_sample_frames']))
            continue
        similarity = row['cosine_prior']; display = 'unavailable' if similarity is None else f'{similarity:.4f}'
        print(row['frame'], f"{row['score']:.4f}", row['selected_phase'], display, len(row['prior_strong_sample_frames']))
    print(f'Report: {report_path}')
    print('Local appearance gate probe: COMPLETED; all replayed camera outputs EXACT; policy unchanged')


if __name__ == '__main__':
    main()
