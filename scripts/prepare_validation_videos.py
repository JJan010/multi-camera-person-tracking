"""Freeze the coverage-based camera choice, download pinned videos, inspect metadata."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from prepare_validation_annotations import DATASET, REVISION, hashes, require

ROOT = Path(__file__).resolve().parents[1]
SCENE = 'MTMC_Tracking_2024/val/scene_041'
CAMERAS = (361, 362, 364)


def save_fixed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(json.loads(path.read_text()) == value,
                f'Existing configuration differs; not overwritten: {path}')
    else:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked_input(spec):
    path = Path(spec['path'])
    if not path.is_absolute():
        path = ROOT / path
    require(hashes(path)[0] == spec['sha256'], f'Checksum mismatch: {path}')
    return path


def select_cameras(gt_path, calibration_path):
    calibration = json.loads(calibration_path.read_text())
    sizes = {}
    for sensor in calibration['sensors']:
        if sensor.get('type') == 'camera':
            camera = int(sensor['id'].rsplit('_', 1)[1])
            attrs = {a['name']: a['value'] for a in sensor['attributes']}
            sizes[camera] = (int(attrs['frameWidth']), int(attrs['frameHeight']))
    gt = np.loadtxt(gt_path, dtype=np.float64, ndmin=2)
    require(gt.shape[1] == 9 and np.isfinite(gt).all(), 'Unexpected GT schema')
    window = gt[(gt[:, 2] >= 0) & (gt[:, 2] < 3600)]
    observations = {}
    for camera, (width, height) in sorted(sizes.items()):
        rows = window[window[:, 0] == camera]
        x, y, w, h = rows[:, 3:7].T
        rows = rows[(x+w > 0) & (y+h > 0) & (x < width) & (y < height)]
        observations[camera] = set(map(tuple, rows[:, 1:3].astype(np.int64).tolist()))
    require({361, 362} <= observations.keys(), 'Anchor cameras missing')
    require(observations[361] & observations[362], 'Anchor pair has no shared observations')
    candidates = []
    for camera in sorted(observations):
        if camera in (361, 362):
            continue
        candidates.append({'camera': camera,
                           'with_361': len(observations[camera] & observations[361]),
                           'with_362': len(observations[camera] & observations[362])})
    eligible = [x['camera'] for x in candidates if x['with_361'] or x['with_362']]
    require(eligible and min(eligible) == CAMERAS[2], 'Coverage rule no longer reproduces camera 364')
    return candidates


def source_checksum(entry):
    lfs = getattr(entry, 'lfs', None)
    if lfs is not None:
        value = lfs.get('sha256', lfs.get('oid')) if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        require(isinstance(value, str), 'Missing upstream LFS hash')
        value = value.removeprefix('sha256:')
        require(len(value) == 64, 'Invalid upstream LFS hash')
        return 'sha256', value
    return 'git_blob_sha1', entry.blob_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotation-report', type=Path, required=True)
    args = parser.parse_args()
    require(shutil.which('ffprobe') is not None, 'ffprobe is required')
    audit_path = args.annotation_report.resolve()
    audit_hash = hashes(audit_path)[0]
    audit = json.loads(audit_path.read_text())
    require(audit['completed'] and audit['protocol'] == 'validation_annotation_audit_v1'
            and audit['scene'] == SCENE and audit['frame_stop_exclusive'] == 3600,
            'Unexpected annotation audit')
    source_path = checked_input(audit['inputs']['source_manifest'])
    source = json.loads(source_path.read_text())
    require((source['dataset'], source['revision'], source['scene']) == (DATASET, REVISION, SCENE),
            'Unexpected source manifest')
    local = {}
    for item in source['files']:
        path = ROOT / item['local_path']
        require(hashes(path)[0] == item['sha256'], f'Annotation checksum mismatch: {path}')
        local[Path(item['remote_path']).name] = path
    candidates = select_cameras(local['ground_truth.txt'], local['calibration_2025_format.json'])
    config_dir = ROOT / 'configs/datasets'
    selection = {'dataset': DATASET, 'revision': REVISION, 'scene': SCENE,
                 'camera_ids': list(CAMERAS), 'frame_range': [0, 3599],
                 'rule': 'Keep anchors 361/362; choose lowest other camera ID with positive same-frame/person overlap with either anchor.',
                 'coverage_candidates': candidates,
                 'source_manifest_sha256': hashes(source_path)[0],
                 'selection_uses': 'annotation coverage, not tracker quality',
                 'runtime_starts_at_frame': 0, 'evaluation_frame_range': [2, 3599],
                 'frame_alignment_validated': False}
    selection_path = config_dir / 'scene_041_selection.json'
    save_fixed(selection_path, selection)
    print('Camera selection reproduced and saved: 361, 362, 364', flush=True)
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=False)
    files = []
    for camera in CAMERAS:
        folder = f'{SCENE}/camera_{camera:04d}'
        entries = {e.path: e for e in api.list_repo_tree(repo_id=DATASET, repo_type='dataset',
                   revision=REVISION, path_in_repo=folder, recursive=False)}
        entry = entries[f'{folder}/video.mp4']
        algorithm, digest = source_checksum(entry)
        files.append({'camera': camera, 'remote_path': entry.path, 'size_bytes': entry.size,
                      'source_hash_algorithm': algorithm, 'source_hash': digest})
        print(f'Camera {camera}: {entry.size / 1024**2:.2f} MiB')
    plan = {'dataset': DATASET, 'revision': REVISION, 'scene': SCENE,
            'camera_ids': list(CAMERAS), 'files': files}
    save_fixed(config_dir / 'scene_041_video_plan.json', plan)
    print(f'Total video size: {sum(f["size_bytes"] for f in files)/1024**3:.3f} GiB', flush=True)
    manifest = {k: plan[k] for k in ('dataset', 'revision', 'scene', 'camera_ids')}
    manifest['files'] = []
    metadata = []
    for item in files:
        print(f'Downloading/verifying camera {item["camera"]}...', flush=True)
        data_root = ROOT / 'data/physicalai_smartspaces'
        path = Path(hf_hub_download(repo_id=DATASET, repo_type='dataset', revision=REVISION,
                    filename=item['remote_path'], local_dir=data_root, token=False)).resolve()
        require(path == (data_root / item['remote_path']).resolve(), 'Unexpected video location')
        require(path.stat().st_size == item['size_bytes'], 'Video size mismatch')
        sha, blob = hashes(path)
        measured = sha if item['source_hash_algorithm'] == 'sha256' else blob
        require(measured == item['source_hash'], 'Upstream video checksum mismatch')
        manifest['files'].append({**item, 'local_path': path.relative_to(ROOT.resolve()).as_posix(),
                                  'sha256': sha, 'source_checksum_verified': True})
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                   'stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,avg_frame_rate,time_base,start_pts,start_time,duration,nb_frames',
                   '-of', 'json', str(path)]
        probe = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        require(len(probe.get('streams', [])) == 1, 'Expected a primary video stream')
        stream = probe['streams'][0]
        require(stream['width'] > 0 and stream['height'] > 0, 'Invalid video dimensions')
        metadata.append({'camera': item['camera'], 'stream': stream})
        print(f'CAMERA {item["camera"]}:\n{json.dumps(stream, indent=2)}', flush=True)
    manifest_path = config_dir / 'scene_041_video_manifest.json'
    save_fixed(manifest_path, manifest)
    require(hashes(audit_path)[0] == audit_hash, 'Annotation audit changed during download')
    checked_input(audit['inputs']['source_manifest'])
    for item in source['files']:
        require(hashes(ROOT / item['local_path'])[0] == item['sha256'], 'Annotations changed during download')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'artifacts/validation_videos' / run
    out.mkdir(parents=True, exist_ok=False)
    report = {'completed': True, 'protocol': 'validation_video_preparation_v1', 'run_id': run,
              'scene': SCENE, 'camera_ids': list(CAMERAS),
              'inputs': {'annotation_report': {'path': str(audit_path), 'sha256': audit_hash},
                         'selection': {'path': str(selection_path.resolve()), 'sha256': hashes(selection_path)[0]},
                         'video_manifest': {'path': str(manifest_path.resolve()), 'sha256': hashes(manifest_path)[0]}},
              'video_metadata': metadata, 'script_sha256': hashes(Path(__file__))[0],
              'helper_sha256': hashes(Path(__file__).with_name('prepare_validation_annotations.py'))[0],
              'ffprobe_version': subprocess.run(['ffprobe', '-version'], check=True, capture_output=True, text=True).stdout.splitlines()[0],
              'timestamp_audit_passed': False, 'visual_alignment_reviewed': False,
              'limits': ['Container metadata is not a decoded timestamp audit.',
                         'No inference, tracking quality evaluation or parameter tuning.']}
    report_path = out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Video manifest: {manifest_path}\nReport: {report_path}')
    print('Pinned videos and metadata: COMPLETED; timestamp and geometry checks pending')


if __name__ == '__main__':
    main()
