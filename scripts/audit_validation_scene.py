"""Audit decoded timelines and calibration for the frozen validation subset."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from audit_scene_geometry import (
    check_math, matrix, project, proportional_error, require, sha256, stats,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = 'MTMC_Tracking_2024/val/scene_041'
REVISION = '2cbe9563cbe9f47f846e5c871ee994572bbbc60e'


def verify(spec):
    path = Path(spec['path'])
    if not path.is_absolute():
        path = ROOT / path
    require(sha256(path) == spec['sha256'], f'Checksum mismatch: {path}')
    return path.resolve()


def inspect_frames(frames, fps, expected_count, size):
    count = missing = backwards = mismatches = wrong_size = 0
    previous = first_time = last_time = None
    examples, bases = [], set()
    for index, frame in enumerate(frames):
        count += 1
        if (frame.width, frame.height) != size:
            wrong_size += 1
        tb = frame.time_base
        if frame.pts is None or tb is None or Fraction(tb) <= 0:
            missing += 1
            continue
        tb = Fraction(tb); bases.add(str(tb))
        time = frame.pts * tb
        if first_time is None:
            first_time = time
        last_time = time
        if previous is not None and time <= previous:
            backwards += 1
        previous = time
        expected = Fraction(index, 1) / fps
        if time != expected:
            mismatches += 1
            if len(examples) < 10:
                examples.append({'index': index, 'actual': str(time), 'expected': str(expected)})
    return {'decoded_frames': count, 'expected_frames': expected_count,
            'frame_time_bases': sorted(bases), 'missing_timestamps': missing,
            'nonincreasing_steps': backwards, 'timeline_mismatches': mismatches,
            'dimension_mismatches': wrong_size, 'first_mismatches': examples,
            'first_time': None if first_time is None else str(first_time),
            'last_time': None if last_time is None else str(last_time),
            'passed': count == expected_count and not (missing or backwards or mismatches or wrong_size)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.video_report.resolve()
    inputs = {'video_report': {'path': str(report_path), 'sha256': sha256(report_path)}}
    prepared = json.loads(report_path.read_text())
    require(prepared['completed'] and prepared['protocol'] == 'validation_video_preparation_v1'
            and prepared['scene'] == SCENE, 'Unexpected video preparation report')
    for name, spec in prepared['inputs'].items():
        path = verify(spec)
        inputs[name] = {'path': str(path), 'sha256': spec['sha256']}
    selection = json.loads(Path(inputs['selection']['path']).read_text())
    manifest = json.loads(Path(inputs['video_manifest']['path']).read_text())
    annotation_audit = json.loads(Path(inputs['annotation_report']['path']).read_text())
    source_spec = annotation_audit['inputs']['source_manifest']
    source_path = verify(source_spec)
    inputs['source_manifest'] = {'path': str(source_path), 'sha256': source_spec['sha256']}
    source = json.loads(source_path.read_text())
    for value in (selection, manifest, source):
        require((value['dataset'], value['revision'], value['scene']) ==
                ('nvidia/PhysicalAI-SmartSpaces', REVISION, SCENE), 'Unexpected pinned dataset')
    require(selection['source_manifest_sha256'] == source_spec['sha256'], 'Selection source differs')
    cameras = tuple(selection['camera_ids'])
    require(cameras == (361, 362, 364) and manifest['camera_ids'] == list(cameras)
            and prepared['camera_ids'] == list(cameras), 'Camera selection differs')
    start, stop = selection['evaluation_frame_range']
    require((start, stop) == (2, 3599) and selection['frame_range'] == [0, 3599], 'Unexpected interval')
    for name in ('ground_truth.txt', 'calibration_2025_format.json'):
        items = [x for x in source['files'] if Path(x['remote_path']).name == name]
        require(len(items) == 1, 'Missing/duplicate annotation file')
        item = items[0]; spec = {'path': str(ROOT / item['local_path']), 'sha256': item['sha256']}
        verify(spec); inputs[name] = spec
    require(len(manifest['files']) == len(cameras) and
            {x['camera'] for x in manifest['files']} == set(cameras), 'Video entries differ')
    videos = {}
    for item in manifest['files']:
        spec = {'path': str(ROOT / item['local_path']), 'sha256': item['sha256']}
        videos[item['camera']] = verify(spec); inputs[f'video_{item["camera"]}'] = spec
    metadata = {x['camera']: x['stream'] for x in prepared['video_metadata']}
    require(set(metadata) == set(cameras), 'Missing video metadata')
    calibration = json.loads(Path(inputs['calibration_2025_format.json']['path']).read_text())
    models, checks = {}, {}
    check_math()
    for camera in cameras:
        sensors = [s for s in calibration['sensors'] if s.get('id', '').lower() == f'camera_{camera:04d}']
        require(len(sensors) == 1, 'Missing/duplicate calibrated camera')
        sensor = sensors[0]
        attrs = {a['name']: a['value'] for a in sensor['attributes']}
        width, height, fps = int(attrs['frameWidth']), int(attrs['frameHeight']), Fraction(attrs['fps'])
        meta = metadata[camera]
        require((width, height) == (meta['width'], meta['height']) and
                fps == Fraction(meta['avg_frame_rate']) == Fraction(meta['r_frame_rate']) == 30,
                'Calibration and video dimensions/FPS disagree')
        k, e = matrix(sensor, 'intrinsicMatrix', (3, 3)), matrix(sensor, 'extrinsicMatrix', (3, 4))
        p, h = matrix(sensor, 'cameraMatrix', (3, 4)), matrix(sensor, 'homography', (3, 3))
        require(np.linalg.matrix_rank(h) == 3, 'Singular homography')
        errors = {'P_vs_K_E': proportional_error(p, k @ e),
                  'H_vs_P_columns_0_1_3': proportional_error(h, p[:, [0, 1, 3]])}
        require(max(errors.values()) <= 1e-6, 'Unknown projection convention')
        checks[camera] = {'relative_errors': errors, 'condition_number': float(np.linalg.cond(h))}
        models[camera] = (h, np.linalg.inv(h), width, height, fps)
        print(f'Camera {camera}: calibration/video metadata and matrix conventions VERIFIED')
    import av
    timelines = []
    for camera in cameras:
        print(f'Camera {camera}: decoding the full video on CPU...', flush=True)
        _, _, width, height, fps = models[camera]
        expected_count = int(metadata[camera]['nb_frames'])
        require(expected_count > stop, 'Video too short for selected window')
        with av.open(str(videos[camera])) as container:
            stream = container.streams.video[0]
            summary = inspect_frames(container.decode(stream), fps, expected_count, (width, height))
        summary['camera'] = camera; timelines.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts/validation_scene_audit' / run
    out.mkdir(parents=True, exist_ok=False)
    result = {'completed': False, 'protocol': 'validation_scene_audit_v1', 'run_id': run,
              'scene': SCENE, 'cameras': list(cameras), 'frame_range': [start, stop],
              'inputs': inputs, 'matrix_checks': checks, 'timelines': timelines,
              'timestamp_audit_passed': all(s['passed'] for s in timelines),
              'visual_alignment_reviewed': False, 'artifacts': {},
              'code_sha256': sha256(Path(__file__)),
              'geometry_helper_sha256': sha256(Path(__file__).with_name('audit_scene_geometry.py')),
              'av_version': av.__version__, 'numpy_version': np.__version__}
    output = out / 'report.json'
    if not result['timestamp_audit_passed']:
        output.write_text(json.dumps(result, indent=2) + '\n')
        raise SystemExit(f'Timestamp audit FAILED; inspect {output}')
    print('GT geometry diagnostics in frames 2..3599; no thresholds will be fitted...', flush=True)
    rows, seen, worlds, frames_by_camera = [], set(), {}, {c: set() for c in cameras}
    with Path(inputs['ground_truth.txt']['path']).open() as handle:
        for line in handle:
            fields = line.split()
            if not fields: continue
            require(len(fields) == 9, 'Invalid GT schema')
            camera, person, frame = map(int, fields[:3])
            if camera not in models: continue
            require(0 <= frame < int(metadata[camera]['nb_frames']), 'GT frame outside decoded video')
            if not start <= frame <= stop: continue
            key = camera, person, frame
            require(key not in seen, 'Duplicate GT key'); seen.add(key)
            frames_by_camera[camera].add(frame)
            x, y, w, bh, wx, wy = map(float, fields[3:])
            require(w > 0 and bh > 0 and np.isfinite([x, y, w, bh, wx, wy]).all(), 'Invalid GT box')
            h, inv, width, height, _ = models[camera]
            foot, world = np.array([x+w/2, y+bh]), np.array([wx, wy])
            mapped, pixel = project(inv, foot), project(h, world)
            back = project(h, mapped) if mapped is not None else None
            worlds.setdefault((frame, person), {})[camera] = world
            rows.append({'camera': camera, 'frame': frame, 'gt_id': person,
                         'box_fully_inside': x >= 0 and y >= 0 and x+w <= width and y+bh <= height,
                         'foot_u': float(foot[0]), 'foot_v': float(foot[1]), 'gt_world_x': wx, 'gt_world_y': wy,
                         'projected_x': None if mapped is None else float(mapped[0]),
                         'projected_y': None if mapped is None else float(mapped[1]),
                         'world_discrepancy': None if mapped is None else float(np.linalg.norm(mapped-world)),
                         'pixel_discrepancy': None if pixel is None else float(np.linalg.norm(pixel-foot)),
                         'roundtrip_px': None if back is None else float(np.linalg.norm(back-foot))})
    summaries, absent = {}, {}
    for camera in cameras:
        selected = [r for r in rows if r['camera'] == camera]
        require(selected, 'No GT in selected camera/window')
        missing = sorted(set(range(start, stop+1)) - frames_by_camera[camera])
        absent[camera] = missing
        print(f'Camera {camera}: frames without GT rows={len(missing)}; first={missing[:10]}')
        summaries[camera] = {}
        for name, subset in [('all_raw_boxes', selected), ('fully_inside_boxes', [r for r in selected if r['box_fully_inside']])]:
            summary = {'rows': len(subset), 'invalid_inverse_projections': sum(r['projected_x'] is None for r in subset),
                       **{metric: stats([r[metric] for r in subset if r[metric] is not None])
                          for metric in ('world_discrepancy', 'pixel_discrepancy', 'roundtrip_px')}}
            summaries[camera][name] = summary
            print(f'Camera {camera}, {name}: {json.dumps(summary)}')
    pair_stats = {f'{a}-{b}': stats([np.linalg.norm(v[a]-v[b]) for v in worlds.values() if a in v and b in v])
                  for a, b in combinations(cameras, 2)}
    print('Shared GT world-coordinate disagreement:', json.dumps(pair_stats))
    for spec in inputs.values(): verify(spec)
    rows.sort(key=lambda r: (r['frame'], r['camera'], r['gt_id']))
    csv_path = out / 'gt_projection.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result.update(completed=True, projection_diagnostics=summaries, gt_world_consistency=pair_stats,
                  frames_without_gt_rows=absent,
                  direction='H: world Z=0 -> image; inverse H: image -> world Z=0',
                  coordinate_policy='Raw xywh box bottom center; no clipping or pixel offset',
                  world_units='native dataset units; not independently established as meters',
                  limits=['Equal decoded PTS grids do not by themselves prove synchronized scene content.',
                          'No automatic determination of the GT-to-video frame offset.',
                          'Missing GT rows are not independently certified empty frames.',
                          'Matrix algebra and roundtrip do not establish physical calibration accuracy.',
                          'Bottom center is a ground-contact proxy; fully inside does not imply unoccluded.',
                          'No association thresholds selected or modified.'])
    result['artifacts']['gt_projection.csv'] = {'sha256': sha256(csv_path)}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'Report: {output}')
    print('Validation timeline/geometry audit: COMPLETED; visual alignment review pending')


if __name__ == '__main__':
    main()
