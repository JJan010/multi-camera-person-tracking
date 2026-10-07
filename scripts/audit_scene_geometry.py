"""Audit pinned scene matrices and raw GT-box footpoint projection on CPU."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4, 5, 8)
RELATIVE_TOLERANCE = 1e-6


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def matrix(sensor, name, shape):
    value = np.asarray(sensor[name], dtype=np.float64)
    require(value.shape == shape and np.isfinite(value).all(), f'Invalid {name}')
    return value


def proportional_error(left, right):
    """Matrices represent the same projective map up to any nonzero scale."""
    require(np.linalg.norm(left) > 0 and np.linalg.norm(right) > 0, 'Zero matrix')
    a = left / np.linalg.norm(left)
    b = right / np.linalg.norm(right)
    return float(min(np.linalg.norm(a - b), np.linalg.norm(a + b)))


def project(h, xy):
    """Return None for a numerically near-infinite homogeneous point."""
    h = h / np.linalg.norm(h)
    q = h @ np.array([*xy, 1.0], dtype=np.float64)
    if not np.isfinite(q).all() or abs(q[2]) <= 1e-12 * np.linalg.norm(q):
        return None
    result = q[:2] / q[2]
    return result if np.isfinite(result).all() else None


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    require(np.isfinite(a).all(), 'Nonfinite diagnostic values')
    return {'n': len(a), 'median': float(np.median(a)) if len(a) else None,
            'p95': float(np.quantile(a, .95)) if len(a) else None,
            'max': float(np.max(a)) if len(a) else None}


def check_math():
    h = np.array([[2., 0., 10.], [0., 3., 20.], [.1, 0., 1.]])
    expected = np.array([14. / 1.2, 29. / 1.2])
    assert np.allclose(project(h, (2., 3.)), expected)
    assert np.allclose(project(np.linalg.inv(h), expected), (2., 3.))
    assert np.allclose(project(-7 * h, (2., 3.)), expected)
    assert proportional_error(h, -7 * h) < 1e-12
    assert proportional_error(h, np.linalg.inv(h)) > .1
    assert project(h, (-10., 0.)) is None
    print('Known projection, inverse, projective scale and horizon checks: PASSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path,
                        default=ROOT / 'configs/datasets/scene_001_source.json')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    check_math()
    if args.check_only:
        return
    manifest_path = args.source_manifest.resolve()
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    require(manifest['dataset'] == 'nvidia/PhysicalAI-SmartSpaces' and
            manifest['revision'] == '2cbe9563cbe9f47f846e5c871ee994572bbbc60e' and
            manifest['scene'] == 'MTMC_Tracking_2024/train/scene_001', 'Unexpected dataset revision/scene')
    inputs = {'source_manifest': {'path': str(manifest_path), 'sha256': manifest_hash}}
    for name in ('calibration_2025_format.json', 'ground_truth.txt'):
        entries = [x for x in manifest['files'] if Path(x['remote_path']).name == name]
        require(len(entries) == 1, f'Missing/duplicate source: {name}')
        entry = entries[0]
        path = (ROOT / entry['local_path']).resolve()
        require(sha256(path) == entry['sha256'], f'Source checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': entry['sha256']}
    print('Pinned calibration and GT checksums: VERIFIED', flush=True)
    calibration = json.loads(Path(inputs['calibration_2025_format.json']['path']).read_text())
    models, algebra = {}, {}
    for camera in CAMERAS:
        sensors = [s for s in calibration['sensors'] if s['id'].lower() == f'camera_{camera:04d}']
        require(len(sensors) == 1, f'Camera {camera}: missing/duplicate calibration')
        s = sensors[0]
        k, e = matrix(s, 'intrinsicMatrix', (3, 3)), matrix(s, 'extrinsicMatrix', (3, 4))
        p, h = matrix(s, 'cameraMatrix', (3, 4)), matrix(s, 'homography', (3, 3))
        require(np.linalg.matrix_rank(h) == 3, 'Singular homography')
        inverse = np.linalg.inv(h)
        errors = {'P_vs_K_E': proportional_error(p, k @ e),
                  'H_vs_P_columns_0_1_3': proportional_error(h, p[:, [0, 1, 3]])}
        attributes = {x['name']: x['value'] for x in s['attributes']}
        width, height = int(attributes['frameWidth']), int(attributes['frameHeight'])
        require((width, height, float(attributes['fps'])) == (1920, 1080, 30.), 'Unexpected video geometry')
        algebra[camera] = {'relative_errors': errors, 'homography_condition_number': float(np.linalg.cond(h)),
                          'rotation_orthogonality_error': float(np.linalg.norm(e[:, :3].T @ e[:, :3] - np.eye(3))),
                          'rotation_determinant': float(np.linalg.det(e[:, :3]))}
        require(max(errors.values()) <= RELATIVE_TOLERANCE,
                f'Camera {camera}: matrix convention differs; investigate before projection: {errors}')
        models[camera] = (h, inverse, width, height)
        print(f'Camera {camera}: P ~ K @ E and H ~ P[:, (0,1,3)]: VERIFIED; errors={errors}')
    print('Direction supported by matrix algebra: ground Z=0 -> image; use inverse for image -> ground.')
    print('Loading GT frames 2..299; no detector, tracker or identity changes...', flush=True)
    rows, seen, shared_world = [], set(), {}
    with Path(inputs['ground_truth.txt']['path']).open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            require(len(fields) == 9, 'Expected nine-column GT')
            camera, person, frame = map(int, fields[:3])
            if camera not in models or not 2 <= frame <= 299:
                continue
            key = camera, person, frame
            require(key not in seen, 'Duplicate GT key')
            seen.add(key)
            x, y, w, height_box, wx, wy = map(float, fields[3:])
            require(np.isfinite([x, y, w, height_box, wx, wy]).all() and w > 0 and height_box > 0, 'Invalid GT row')
            h, inv, width, height = models[camera]
            foot = np.array([x + w / 2, y + height_box])
            world = np.array([wx, wy])
            mapped = project(inv, foot)
            world_pixel = project(h, world)
            back = project(h, mapped) if mapped is not None else None
            inside = bool(x >= 0 and y >= 0 and x + w <= width and y + height_box <= height)
            shared_world.setdefault((frame, person), {})[camera] = world
            rows.append({'camera': camera, 'frame': frame, 'gt_id': person, 'box_fully_inside': inside,
                         'foot_u': float(foot[0]), 'foot_v': float(foot[1]), 'gt_world_x': wx, 'gt_world_y': wy,
                         'projected_x': None if mapped is None else float(mapped[0]),
                         'projected_y': None if mapped is None else float(mapped[1]),
                         'world_discrepancy': None if mapped is None else float(np.linalg.norm(mapped - world)),
                         'pixel_discrepancy': None if world_pixel is None else float(np.linalg.norm(world_pixel - foot)),
                         'roundtrip_px': None if back is None else float(np.linalg.norm(back - foot))})
    rows.sort(key=lambda r: (r['frame'], r['camera'], r['gt_id']))
    summaries = {}
    for camera in CAMERAS:
        selected = [r for r in rows if r['camera'] == camera]
        require(bool(selected), f'No GT for camera {camera}')
        summaries[camera] = {}
        for label, subset in [('all_raw_boxes', selected), ('fully_inside_boxes', [r for r in selected if r['box_fully_inside']])]:
            summary = {'rows': len(subset), 'invalid_inverse_projections': sum(r['projected_x'] is None for r in subset),
                       **{name: stats([r[name] for r in subset if r[name] is not None])
                          for name in ('world_discrepancy', 'pixel_discrepancy', 'roundtrip_px')}}
            summaries[camera][label] = summary
            print(f'Camera {camera}, {label}: ' + json.dumps(summary))
    gt_pair_stats = {}
    for a, b in combinations(CAMERAS, 2):
        gt_pair_stats[f'{a}-{b}'] = stats([np.linalg.norm(v[a] - v[b]) for v in shared_world.values() if a in v and b in v])
    print('Same-frame/same-person GT world-coordinate disagreement:', json.dumps(gt_pair_stats))
    for item in inputs.values():
        require(sha256(item['path']) == item['sha256'], 'Input changed during audit')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/geometry_audit' / run_id
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / 'gt_projection.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {'completed': True, 'run_id': run_id, 'protocol': 'scene_001_geometry_audit_v1',
              'inputs': inputs, 'calibration_type': calibration.get('calibrationType'),
              'cameras': list(CAMERAS), 'frame_range': [2, 299], 'algebra_tolerance': RELATIVE_TOLERANCE,
              'matrix_checks': algebra, 'projection_diagnostics': summaries, 'gt_world_consistency': gt_pair_stats,
              'direction': 'H maps world Z=0 to image; inverse H maps image to world Z=0',
              'coordinate_policy': 'Raw GT xywh; foot=(x+w/2,y+h); no clipping or pixel offset',
              'world_units': 'native dataset coordinates; metric unit not independently verified by this audit',
              'limits': ['Algebra/roundtrip does not establish physical calibration accuracy',
                         'GT-box bottom center is a footpoint proxy, not a measured foot contact',
                         'Fully inside the image does not imply unoccluded feet',
                         'GT world-position semantics and physical units need confirmation',
                         'No geometry association gate or distance threshold is selected'],
              'numpy_version': np.__version__, 'code_sha256': sha256(Path(__file__)),
              'artifacts': {'gt_projection.csv': {'sha256': sha256(csv_path)}}}
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {path}')
    print('Geometry audit: COMPLETED; physical accuracy and distance gates: NOT YET VALIDATED')


if __name__ == '__main__':
    main()
