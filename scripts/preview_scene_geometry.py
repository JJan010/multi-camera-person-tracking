"""Visual audit of raw GT footpoint proxies versus projected GT world positions."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import audit_scene_geometry as audit

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4, 5, 8)


def choose_cases(rows):
    cases = {}
    for camera in CAMERAS:
        eligible = [r for r in rows if int(r['camera']) == camera and
                    r['box_fully_inside'] == 'True' and r['world_discrepancy'] != '']
        audit.require(bool(eligible), f'No fully inside projections for camera {camera}')
        worst = min(eligible, key=lambda r: (-float(r['world_discrepancy']), int(r['frame']), int(r['gt_id'])))
        cases[camera] = {'frame': int(worst['frame']), 'gt_id': int(worst['gt_id']),
                         'world_discrepancy': float(worst['world_discrepancy'])}
    return cases


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def annotate(image, records, h):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    details = []
    for gt_id, values in sorted(records.items()):
        x, y, w, height, wx, wy = values
        foot = np.array([x + w / 2, y + height])
        projected = audit.project(h, (wx, wy))
        audit.require(projected is not None, 'GT world projection is near infinity')
        mapped = audit.project(np.linalg.inv(h), foot)
        audit.require(mapped is not None, 'Footpoint inverse projection is near infinity')
        left, top = max(0, x), max(0, y)
        right, bottom = min(image.width - 1, x + w), min(image.height - 1, y + height)
        if right > left and bottom > top:
            draw.rectangle((left, top, right, bottom), outline='white', width=3)
            label = f'GT {gt_id}'
            pos = (min(left + 3, image.width - 130), max(0, top - 30))
            draw.rectangle(draw.textbbox(pos, label, font=font(24)), fill='black')
            draw.text(pos, label, fill='white', font=font(24))
        draw.line([tuple(foot), tuple(projected)], fill='yellow', width=4)
        u, v = foot
        draw.ellipse((u-9, v-9, u+9, v+9), outline='cyan', width=4)
        u, v = projected
        draw.line((u-12, v, u+12, v), fill='magenta', width=4)
        draw.line((u, v-12, u, v+12), fill='magenta', width=4)
        details.append({'gt_id': gt_id, 'raw_xywh': [x,y,w,height], 'foot_uv': foot.tolist(),
                        'gt_world_xy': [wx,wy], 'projected_gt_uv': projected.tolist(),
                        'estimated_world_xy': mapped.tolist(),
                        'world_discrepancy': float(np.linalg.norm(mapped - [wx,wy])),
                        'pixel_discrepancy': float(np.linalg.norm(projected - foot)),
                        'projected_gt_in_image': bool(0 <= projected[0] < image.width and 0 <= projected[1] < image.height)})
    return image, details


def tile(image, lines):
    result = Image.new('RGB', (960, 640), 'black')
    fitted = ImageOps.contain(image, (960, 540))
    result.paste(fitted, ((960-fitted.width)//2, 100+(540-fitted.height)//2))
    draw = ImageDraw.Draw(result)
    for index, line in enumerate(lines):
        draw.text((10, 5 + 30 * index), line, font=font(22), fill='white')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.audit_report.resolve()
    inputs = {'audit_report': {'path': str(report_path), 'sha256': audit.sha256(report_path)}}
    report = json.loads(report_path.read_text())
    audit.require(report.get('completed') and report['protocol'] == 'scene_001_geometry_audit_v1', 'Unexpected audit')
    for name, item in report['inputs'].items():
        audit.require(audit.sha256(item['path']) == item['sha256'], f'Changed audit input: {name}')
        inputs[name] = item
    projection_path = report_path.parent / 'gt_projection.csv'
    projection_hash = report['artifacts']['gt_projection.csv']['sha256']
    audit.require(audit.sha256(projection_path) == projection_hash, 'Changed projection CSV')
    inputs['projection_csv'] = {'path': str(projection_path), 'sha256': projection_hash}
    with projection_path.open() as handle:
        cases = choose_cases(list(csv.DictReader(handle)))
    print('Worst fully-inside cases selected deterministically:', json.dumps(cases), flush=True)
    video_manifest_path = ROOT / 'configs/datasets/scene_001_video_manifest.json'
    video_manifest = json.loads(video_manifest_path.read_text())
    inputs['video_manifest'] = {'path': str(video_manifest_path), 'sha256': audit.sha256(video_manifest_path)}
    source = json.loads(Path(inputs['source_manifest']['path']).read_text())
    audit.require(all(video_manifest[k] == source[k] for k in ('dataset','revision','scene')), 'Video/source context differs')
    videos = {}
    for camera in CAMERAS:
        entries = [r for r in video_manifest['files'] if r['camera'] == camera]
        audit.require(len(entries) == 1, 'Missing/duplicate video entry')
        path = (ROOT / entries[0]['local_path']).resolve()
        audit.require(audit.sha256(path) == entries[0]['sha256'], f'Video checksum mismatch: {camera}')
        videos[camera] = path
        inputs[f'video_{camera}'] = {'path': str(path), 'sha256': entries[0]['sha256']}
    calibration = json.loads(Path(inputs['calibration_2025_format.json']['path']).read_text())
    matrices = {camera: np.asarray(next(s for s in calibration['sensors'] if s['id'].lower() == f'camera_{camera:04d}')['homography'], dtype=np.float64) for camera in CAMERAS}
    wanted = {(camera, 149) for camera in CAMERAS} | {(camera, cases[camera]['frame']) for camera in CAMERAS}
    gt = {key: {} for key in wanted}
    with Path(inputs['ground_truth.txt']['path']).open() as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            camera, person, frame = map(int, fields[:3])
            if (camera, frame) in wanted:
                audit.require(person not in gt[camera,frame], 'Duplicate GT row')
                gt[camera,frame][person] = tuple(map(float, fields[3:]))
    import av
    raw = {}
    print('Decoding selected frames sequentially with exact PTS checks...', flush=True)
    for camera in CAMERAS:
        requested = {frame for c, frame in wanted if c == camera}
        with av.open(str(videos[camera])) as container:
            for index, frame in enumerate(container.decode(video=0)):
                audit.require(frame.pts is not None and Fraction(frame.pts) * frame.time_base == Fraction(index,30), 'Video timestamp mismatch')
                if index in requested:
                    image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                    audit.require(image.size == (1920,1080), 'Unexpected image size')
                    raw[camera,index] = image
                if index == max(requested):
                    break
    audit.require(set(raw) == wanted, 'Video ended before requested frames')
    output = ROOT / 'artifacts/geometry_preview' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    contacts = {name: Image.new('RGB', (2880,640), 'black') for name in ('frame149', 'worst_inside')}
    records = []
    for column, camera in enumerate(CAMERAS):
        for kind, frame_id, ids in [('frame149',149,{23,24}), ('worst_inside',cases[camera]['frame'],{cases[camera]['gt_id']})]:
            selected = {p: v for p,v in gt[camera,frame_id].items() if p in ids}
            if kind == 'worst_inside':
                audit.require(len(selected) == 1, 'Missing worst-case GT')
            image, details = annotate(raw[camera,frame_id], selected, matrices[camera])
            name = f'{kind}_camera_{camera:04d}_frame_{frame_id:06d}.png'
            image.save(output / name)
            view = image
            if kind == 'worst_inside':
                d = details[0]
                audit.require(abs(d['world_discrepancy'] - cases[camera]['world_discrepancy']) < 1e-9, 'CSV/preview projection mismatch')
                x,y,w,h = d['raw_xywh']
                u,v = d['projected_gt_uv']
                bounds = (max(0,int(np.floor(min(x,u)-80))), max(0,int(np.floor(min(y,v)-80))),
                          min(1920,int(np.ceil(max(x+w,u)+80))), min(1080,int(np.ceil(max(y+h,v)+80))))
                view = image.crop(bounds)
            labels = [f'Camera {camera} | frame {frame_id} | GT {sorted(selected)}',
                      'Cyan circle: box foot | magenta cross: projected GT',
                      ('World discrepancy: ' + ', '.join(f"{d['gt_id']}={d['world_discrepancy']:.3f}" for d in details)) or 'No selected GT visible']
            contacts[kind].paste(tile(view, labels), (column*960,0))
            records.append({'kind':kind,'camera':camera,'frame':frame_id,'image':name,'observations':details,
                            'missing_requested_gt_ids':sorted(ids-set(selected))})
            print(f'{name}: ' + labels[2])
    for kind, image in contacts.items():
        image.save(output / f'{kind}_contact.jpg', quality=95)
    for item in inputs.values():
        audit.require(audit.sha256(item['path']) == item['sha256'], 'Input changed during preview')
    summary = {'completed':True,'protocol':'scene_001_geometry_visual_audit_v1','inputs':inputs,
               'code_sha256':audit.sha256(Path(__file__)), 'audit_helper_sha256':audit.sha256(Path(audit.__file__)),
               'cases':records,'versions':{'numpy':np.__version__,'av':av.__version__},
               'interpretation':'GT boxes and GT world positions only; not tracked boxes, keypoints or a geometry gate',
               'world_units':report['world_units'],
               'selection':'frame 149 for GT 23/24; highest discrepancy fully-inside GT box per camera, frames 2..299',
               'artifacts':{p.name:{'sha256':audit.sha256(p)} for p in sorted(output.iterdir())}}
    path = output / 'report.json'
    path.write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'Report: {path}')
    print(f'Contact sheets: {output}')
    print('Geometry projection preview: COMPLETED; visual review pending')


if __name__ == '__main__':
    main()
