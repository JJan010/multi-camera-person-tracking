"""Explain frozen CLEAR misses using the exact cached detector candidates."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import zipfile

import numpy as np
from diagnose_scene_false_positives import METRICS, geometry
from mtmc.data.ground_truth import load_ground_truth, clip_boxes
from mtmc.data.scene import load_scene, require, sha256

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ('no_admissible_cached_detection', 'admissible_detection_not_emitted', 'admissible_prediction_unmatched')


def select_examples(rows):
    examples = []
    for category in CATEGORIES:
        subset = [r for r in rows if r['category'] == category]
        counts = Counter(r['gt_id'] for r in subset)
        for identity, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]:
            observations = sorted((r for r in subset if r['gt_id'] == identity), key=lambda r: r['frame'])
            examples.append(dict(observations[len(observations)//2], category_identity_misses=count))
    return examples


def diagnose(trace, candidates, scene, spec, ground, source, camera, variant):
    import motmetrics as mm
    mm.lap.default_solver = 'scipy'
    acc = mm.MOTAccumulator(auto_id=False)
    metadata = {}
    size = next(c for c in scene.cameras if c.camera_id == camera)
    with gzip.open(trace, 'rt') as tracked, Path(candidates).open() as cached:
        for frame in range(scene.rounds):
            a, b = tracked.readline(), cached.readline()
            require(bool(a) and bool(b), 'Truncated tracking/candidate stream')
            record, detections = json.loads(a), json.loads(b)
            time = str(Fraction(frame, scene.fps))
            require(record['run_id'] == source['run_id'] and record['source_run_id'] == source['cache_run_id']
                    and record['scene'] == scene.scene and record['frame_index'] == frame and record['timestamp'] == time,
                    'Prediction scope differs')
            require(detections['cache_run_id'] == source['cache_run_id'] and detections['scene'] == scene.scene
                    and detections['frame_index'] == frame and detections['timestamp'] == time
                    and detections['camera_ids'] == list(scene.camera_ids), 'Candidate scope differs')
            items = record['variants'][variant]['cameras']
            require([x['camera'] for x in items] == list(scene.camera_ids), 'Camera coverage differs')
            item = next(x for x in items if x['camera'] == camera)
            dets = [d for d in detections['detections'] if d['camera_id'] == camera]
            require([d['detection_index'] for d in dets] == list(range(len(dets)))
                    and all(d['frame_index'] == frame for d in dets), 'Candidate indexing differs')
            local = item['local_ids']
            require(len(local) == len(set(local)) and all(type(i) is int and i >= 0 for i in local)
                    and all(len(item[k]) == len(local) for k in ('xyxy', 'confidence', 'detection_indices')),
                    'Malformed local predictions')
            indices = item['detection_indices']
            require(len(set(indices)) == len(indices), 'Repeated selected candidate')
            for j, index in enumerate(indices):
                require(type(index) is int and 0 <= index < len(dets)
                        and item['xyxy'][j] == dets[index]['xyxy']
                        and item['confidence'][j] == dets[index]['confidence'], 'Selected candidate provenance differs')
            if not spec.first_frame <= frame <= spec.last_frame:
                continue
            truth = ground.slots[frame, camera]
            ids, prediction_iou = geometry(truth, item['xyxy'], size.width, size.height)
            det_ids, candidate_iou = geometry(truth, [d['xyxy'] for d in dets], size.width, size.height)
            require(ids == det_ids, 'GT geometry differs')
            acc.update(ids, local, np.where(prediction_iou >= spec.min_iou, 1.-prediction_iou, np.nan), frameid=frame)
            for i, identity in enumerate(ids):
                eligible = np.flatnonzero(candidate_iou[i] >= spec.min_iou).tolist()
                prediction_count = int((prediction_iou[i] >= spec.min_iou).sum())
                category = (CATEGORIES[0] if not eligible else CATEGORIES[1] if prediction_count == 0 else CATEGORIES[2])
                best = int(np.argmax(candidate_iou[i])) if dets else None
                clipped = clip_boxes([truth[identity]], size.width, size.height)[0]
                metadata[frame, identity] = {
                    'frame': frame, 'camera': camera, 'gt_id': identity, 'category': category,
                    'gt_xyxy': truth[identity], 'clipped_width': float(clipped[2]-clipped[0]),
                    'clipped_height': float(clipped[3]-clipped[1]),
                    'partially_outside': bool(np.any(clipped != truth[identity])),
                    'admissible_candidate_count': len(eligible), 'admissible_prediction_count': prediction_count,
                    'max_admissible_score': max((dets[j]['confidence'] for j in eligible), default=None),
                    'best_candidate_index': best, 'best_candidate_xyxy': dets[best]['xyxy'] if best is not None else None,
                    'best_candidate_score': dets[best]['confidence'] if best is not None else None,
                    'best_candidate_iou': float(candidate_iou[i, best]) if best is not None else 0.}
            if (frame+1) % 600 == 0:
                print(f'Inspected {frame+1}/{scene.rounds}', flush=True)
        require(tracked.readline() == '' and cached.readline() == '', 'Trailing input frames')
    measured = json.loads(mm.metrics.create().compute(acc, metrics=METRICS).to_json(orient='records'))[0]
    expected = source['results'][variant]['local'][f'camera_{camera:04d}']
    for name in METRICS:
        x, y = measured[name], expected[name]
        require((x is None and y is None) or (x is not None and y is not None and abs(x-y) <= 1e-8),
                f'Local metric not reproduced: {name}: {x} vs {y}')
    rows = [dict(metadata[int(frame), int(event['OId'])])
            for (frame, _), event in acc.mot_events.iterrows() if event['Type'] == 'MISS']
    require(len(rows) == int(measured['num_misses']), 'MISS count differs')
    return measured, rows


def render(scene, ground, examples, camera, trace, variant, output):
    if not examples:
        return []
    import av
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    wanted = {e['frame'] for e in examples}
    predictions = {}
    with gzip.open(trace, 'rt') as stream:
        for line in stream:
            record = json.loads(line)
            if record['frame_index'] in wanted:
                predictions[record['frame_index']] = next(x for x in record['variants'][variant]['cameras'] if x['camera'] == camera)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    cam = next(c for c in scene.cameras if c.camera_id == camera)
    def box(draw, raw, color, label):
        b = clip_boxes([raw], cam.width, cam.height)[0]
        if np.any(b[2:] <= b[:2]):
            return
        draw.rectangle(tuple(b), outline=color, width=3)
        draw.text((b[0], max(0, b[1]-22)), label, font=font, fill=color, stroke_width=1, stroke_fill='black')
    found = set()
    names = []
    with av.open(str(cam.video.path)) as container:
        for frame_index, frame in enumerate(container.decode(video=0)):
            require(frame.pts is not None and frame.time_base is not None
                    and frame.pts*Fraction(frame.time_base) == Fraction(frame_index, scene.fps), 'PTS differs')
            require((frame.width, frame.height) == (cam.width, cam.height), 'Video size differs')
            if frame_index in wanted:
                raw = Image.fromarray(frame.to_ndarray(format='rgb24'))
                for number, example in enumerate(examples):
                    if example['frame'] != frame_index:
                        continue
                    image = raw.copy()
                    draw = ImageDraw.Draw(image)
                    for identity, rectangle in sorted(ground.slots[frame_index, camera].items()):
                        box(draw, rectangle, 'lime', f'GT {identity}')
                    item = predictions[frame_index]
                    for identity, rectangle in zip(item['local_ids'], item['xyxy']):
                        box(draw, rectangle, 'cyan', f'L{identity}')
                    if example['best_candidate_xyxy'] is not None:
                        box(draw, example['best_candidate_xyxy'], 'yellow', f"D{example['best_candidate_index']}")
                    box(draw, example['gt_xyxy'], 'red', f"MISS GT {example['gt_id']}")
                    b = clip_boxes([example['gt_xyxy']], cam.width, cam.height)[0]
                    bounds = (max(0, int(b[0])-100), max(0, int(b[1])-100),
                              min(cam.width, int(np.ceil(b[2]))+100), min(cam.height, int(np.ceil(b[3]))+100))
                    sheet = Image.new('RGB', (1200, 1070), 'black')
                    sd = ImageDraw.Draw(sheet)
                    lines = [f"camera {camera}, frame {frame_index}, missed GT {example['gt_id']}",
                             example['category'], 'red=missed GT; green=GT; cyan=tracks; yellow=highest-IoU cached detection',
                             f"best candidate IoU={example['best_candidate_iou']:.3f}; score={example['best_candidate_score']}"]
                    for j, text in enumerate(lines):
                        sd.text((10, 3+j*24), text, fill='white', font=font)
                    full = ImageOps.contain(image, (1200, 675))
                    sheet.paste(full, ((1200-full.width)//2, 105))
                    for offset, view in ((0, raw), (600, image)):
                        crop = ImageOps.contain(view.crop(bounds), (595, 280))
                        sheet.paste(crop, (offset+(595-crop.width)//2, 785))
                    name = f"example_{number+1:02d}_frame_{frame_index:06d}_gt_{example['gt_id']:04d}.jpg"
                    sheet.save(output/name, quality=92)
                    example['image'] = name
                    names.append(name)
                found.add(frame_index)
            if frame_index >= max(wanted):
                break
    require(found == wanted, 'Missing selected video frames')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-report', type=Path, required=True)
    parser.add_argument('--camera', type=int, default=364)
    parser.add_argument('--variant', choices=['staged', 'competitive_iou'], default='staged')
    args = parser.parse_args()
    inputs = {}
    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f'Checksum changed: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path
    source_path = checked('evaluation_report', args.evaluation_report)
    source = json.loads(source_path.read_text())
    require(source.get('completed') is True and source['protocol'] == 'scene_paired_tracking_evaluation_v1', 'Unexpected evaluation')
    require(version('motmetrics') == '1.4.0', 'Expected motmetrics 1.4.0')
    for package in ('motmetrics', 'numpy', 'scipy', 'pandas'):
        require(version(package) == source['versions'][package], f'Changed evaluation dependency: {package}')
    for name in ('scene_config', 'source_manifest', 'video_manifest', 'calibration', 'ground_truth', 'candidate_report'):
        asset = source['inputs'][name]
        checked(name, asset['path'], asset['sha256'])
    loaded = load_scene(inputs['scene_config']['path'], project_root=ROOT)
    scene, spec = loaded.runtime, loaded.evaluation
    require(scene.scene == source['scene'] and args.camera in scene.camera_ids
            and source['evaluation']['frames'] == [spec.first_frame, spec.last_frame]
            and source['evaluation']['min_iou'] == spec.min_iou
            and spec.ground_truth.sha256 == inputs['ground_truth']['sha256'], 'Scene/GT scope differs')
    for cam in scene.cameras:
        require(source['inputs'][f'video_{cam.camera_id}']['sha256'] == cam.video.sha256, 'Video lineage differs')
        checked(f'video_{cam.camera_id}', cam.video.path, cam.video.sha256)
    for rel in ('src/mtmc/data/ground_truth.py', 'src/mtmc/data/scene.py'):
        checked('code:'+rel, ROOT/rel, source['inputs']['code:'+rel]['sha256'])
    checked('diagnostic_code', __file__)
    checked('diagnostic_helper', ROOT/'scripts/diagnose_scene_false_positives.py')
    cache_path = Path(inputs['candidate_report']['path'])
    cache = json.loads(cache_path.read_text())
    require(cache.get('completed') is True and cache['protocol'] == 'scene_candidate_collection_fp32_v1'
            and cache['run_id'] == source['cache_run_id'] and cache['scene'] == scene.scene, 'Cache lineage differs')
    asset = cache['artifacts']['detections.jsonl']
    require(asset['sha256'] == source['inputs']['detections.jsonl']['sha256'], 'Evaluation candidate hash differs')
    candidates = checked('candidates', cache_path.parent/asset['path'], asset['sha256'])
    asset = source['artifacts']['paired_tracks.jsonl.gz']
    trace = checked('paired_tracks', source_path.parent/asset['path'], asset['sha256'])
    ground = load_ground_truth(spec)
    print(f'Frozen camera {args.camera}, {args.variant}; matching misses to cached candidates...', flush=True)
    metrics, rows = diagnose(trace, candidates, scene, spec, ground, source, args.camera, args.variant)
    examples = select_examples(rows)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'artifacts/scene_misses'/run
    output.mkdir(parents=True, exist_ok=False)
    counts = dict(Counter(r['category'] for r in rows))
    identities = sorted(Counter(r['gt_id'] for r in rows).items(), key=lambda x: (-x[1], x[0]))
    print('Local metrics and exact detection-to-track provenance: VERIFIED')
    print('MISS categories:', json.dumps(counts))
    print('Top GT identities by missed observations:', identities[:10])
    print('Missed boxes partially outside:', sum(r['partially_outside'] for r in rows))
    with (output/'misses.csv').open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('Decoding selected frames on CPU; no model or tracker execution...', flush=True)
    images = render(scene, ground, examples, args.camera, trace, args.variant, output)
    for asset in inputs.values():
        require(sha256(asset['path']) == asset['sha256'], 'Input changed during diagnostic')
    report = {'completed': True, 'protocol': 'scene_miss_diagnostic_v1', 'run_id': run, 'scene': scene.scene,
              'camera': args.camera, 'variant': args.variant, 'inputs': inputs, 'reproduced_local_metrics': metrics,
              'categories': counts, 'top_gt_misses': [{'gt_id': i, 'misses': n} for i, n in identities[:10]],
              'cached_detector_threshold': cache['configuration']['detector_threshold'],
              'partially_outside_misses': sum(r['partially_outside'] for r in rows),
              'examples': examples, 'visual_review_completed': False,
              'artifacts': {name: {'path': name, 'sha256': sha256(output/name)} for name in ['misses.csv']+images},
              'checks': {'local_metrics_reproduced': True, 'candidate_provenance_exact': True, 'inputs_unchanged': True},
              'limits': ['No suitable cached detection does not exclude detections below the frozen collection threshold.',
                         'A candidate box can overlap multiple people: availability alone does not prove a tracker error.',
                         'Visual samples are diagnostic, not a random prevalence sample. Occlusion is not inferred from box dimensions.',
                         'No thresholds, predictions, annotations or runtime states changed.']}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with zipfile.ZipFile(output/'visual_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['report.json', 'misses.csv']+images:
            archive.write(output/name, name)
    print(f'Report: {output / "report.json"}')
    print(f'Visual review: {output / "visual_review.zip"}')
    print('Frozen miss diagnostic: COMPLETED; visual review pending')


if __name__ == '__main__':
    main()
