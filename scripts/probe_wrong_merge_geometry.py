"""Inspect the frame-149 wrong merge using frozen tracker boxes and calibration."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import audit_scene_geometry as geometry

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {5: (9, 24), 8: (17, 23)}  # local track and diagnostic GT; this probe only
FRAME = 149


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry-report', type=Path, required=True)
    parser.add_argument('--identity-report', type=Path, required=True)
    args = parser.parse_args()
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = geometry.sha256(path)
        geometry.require(expected is None or digest == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path

    gpath = checked('geometry_report', args.geometry_report)
    ipath = checked('identity_report', args.identity_report)
    audit = json.loads(gpath.read_text())
    identity = json.loads(ipath.read_text())
    geometry.require(audit.get('completed') and audit['protocol'] == 'scene_001_geometry_audit_v1', 'Unexpected geometry report')
    geometry.require(identity.get('completed') and identity['protocol']['name'] == 'scene_001_controlled_merge_paired_v1', 'Unexpected identity report')
    geometry.require(identity['protocol']['variant'] == 'mean' and identity['protocol']['threshold'] == .7, 'Unexpected setting')
    for name in ('calibration_2025_format.json', 'ground_truth.txt'):
        entry = audit['inputs'][name]
        checked(name, entry['path'], entry['sha256'])
    geometry.require(inputs['ground_truth.txt']['sha256'] == identity['inputs']['ground_truth']['sha256'], 'Different GT sources')
    entry = identity['inputs']['tracks']
    trace = checked('tracks', entry['path'], entry['sha256'])
    calibration = json.loads(Path(inputs['calibration_2025_format.json']['path']).read_text())
    gt = {}
    with Path(inputs['ground_truth.txt']['path']).open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            camera, person, frame = map(int, fields[:3])
            if frame == FRAME and camera in TARGETS and person == TARGETS[camera][1]:
                geometry.require(camera not in gt, 'Duplicate GT')
                gt[camera] = np.asarray(fields[3:], dtype=np.float64)
    geometry.require(set(gt) == set(TARGETS), 'Missing diagnostic GT observations')
    selected = None
    with trace.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record['frame_index'] == FRAME:
                geometry.require(selected is None, 'Duplicate frame')
                selected = record
    geometry.require(selected is not None and selected['run_id'] == identity['source_run_id'], 'Missing frame or wrong trace scope')
    geometry.require(Fraction(selected['timestamp']) == Fraction(FRAME, 30), 'Wrong timestamp')
    observations = {}
    for camera, (local, person) in TARGETS.items():
        sensors = [s for s in calibration['sensors'] if s['id'].lower() == f'camera_{camera:04d}']
        geometry.require(len(sensors) == 1, 'Invalid camera calibration')
        h = geometry.matrix(sensors[0], 'homography', (3,3))
        tracks = [c for c in selected['cameras'] if c['camera'] == camera]
        geometry.require(len(tracks) == 1 and tracks[0]['local_ids'].count(local) == 1, 'Missing/duplicate local track')
        c = tracks[0]
        box = np.asarray(c['xyxy'][c['local_ids'].index(local)], dtype=np.float64)
        geometry.require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]), 'Invalid tracked box')
        x,y,w,height,wx,wy = gt[camera]
        track_foot = np.array([(box[0]+box[2])/2, box[3]])
        gt_foot = np.array([x+w/2, y+height])
        inverse = np.linalg.inv(h)
        track_world, gt_foot_world = geometry.project(inverse, track_foot), geometry.project(inverse, gt_foot)
        geometry.require(track_world is not None and gt_foot_world is not None, 'Near-infinite foot projection')
        world = np.array([wx,wy])
        observations[camera] = {
            'local_id': local, 'diagnostic_gt_id': person, 'tracked_xyxy': box.tolist(),
            'tracked_box_fully_inside': bool(box[0]>=0 and box[1]>=0 and box[2]<=1920 and box[3]<=1080),
            'gt_world_xy': world.tolist(), 'gt_box_projected_xy': gt_foot_world.tolist(),
            'tracker_box_projected_xy': track_world.tolist(),
            'tracker_world_discrepancy': float(np.linalg.norm(track_world-world)),
            'gt_box_world_discrepancy': float(np.linalg.norm(gt_foot_world-world)),
        }
        print(f'Camera {camera}, local {local}, diagnostic GT {person}: ' + json.dumps(observations[camera]))
    distances = {field: float(np.linalg.norm(np.array(observations[5][field])-observations[8][field]))
                 for field in ('gt_world_xy','gt_box_projected_xy','tracker_box_projected_xy')}
    print('Cross-camera distances in native dataset units:')
    print(json.dumps(distances, indent=2))
    for item in inputs.values():
        geometry.require(geometry.sha256(item['path']) == item['sha256'], 'Input changed during probe')
    output = ROOT / 'artifacts/geometry_merge_probe' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    report = {'completed': True, 'protocol': 'scene_001_wrong_merge_geometry_probe_v1',
              'inputs': inputs, 'frame': FRAME, 'timestamp': str(Fraction(FRAME,30)),
              'source_run_id': identity['source_run_id'], 'observations': observations, 'pair_distances': distances,
              'code_sha256': geometry.sha256(Path(__file__)), 'helper_sha256': geometry.sha256(Path(geometry.__file__)),
              'limits': ['One previously selected wrong merge; no threshold selection or association replay',
                         'Diagnostic GT IDs explicitly select this case; this is not runtime logic',
                         'Raw un-clipped tracked and GT boxes; footpoint approximation may be unreliable',
                         'Native dataset units; no independent physical-scale verification']}
    path = output / 'report.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Report: {path}')
    print('Wrong-merge geometry probe: COMPLETED; association unchanged')


if __name__ == '__main__':
    main()
