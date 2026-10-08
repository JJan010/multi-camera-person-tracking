"""Reproduce camera-local CLEAR false positives and inspect deterministic examples.

Offline diagnostic only: no detector, tracker, embedding or identity updates.
"""
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
from mtmc.data.ground_truth import load_ground_truth, clip_boxes, pairwise_iou
from mtmc.data.scene import load_scene, require, sha256

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['num_frames', 'num_objects', 'num_predictions', 'num_false_positives',
           'num_misses', 'num_switches', 'idf1', 'precision', 'recall']


def geometry(truth, boxes, width, height):
    ids = sorted(truth)
    gt = clip_boxes([truth[i] for i in ids], width, height)
    keep = np.all(gt[:, 2:] > gt[:, :2], axis=1)
    return [i for i, k in zip(ids, keep) if k], pairwise_iou(gt[keep], clip_boxes(boxes, width, height))


def category(raw_gt_count, best_iou, gate):
    if raw_gt_count == 0:
        return 'empty_gt_slot'
    return 'no_admissible_gt' if best_iou < gate else 'admissible_but_unmatched'


def select_examples(rows):
    """Two most FP-heavy tracks per category; median FP frame, fixed tie breaks."""
    selected = []
    for name in ('empty_gt_slot', 'no_admissible_gt', 'admissible_but_unmatched'):
        subset = [r for r in rows if r['category'] == name]
        counts = Counter(r['local_id'] for r in subset)
        for identity, count in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:2]:
            observations = sorted((r for r in subset if r['local_id'] == identity), key=lambda r: r['frame'])
            selected.append(dict(observations[len(observations)//2], category_track_fp_count=count))
    return selected


def diagnose(trace, scene, spec, ground, source, camera, variant):
    import motmetrics as mm
    mm.lap.default_solver = 'scipy'
    acc = mm.MOTAccumulator(auto_id=False)
    metadata = {}
    camera_spec = next(c for c in scene.cameras if c.camera_id == camera)
    with gzip.open(trace, 'rt') as stream:
        for frame in range(scene.rounds):
            line = stream.readline()
            require(bool(line), 'Truncated trace')
            record = json.loads(line)
            require(record['run_id'] == source['run_id'] and record['source_run_id'] == source['cache_run_id']
                    and record['scene'] == scene.scene and record['frame_index'] == frame
                    and record['timestamp'] == str(Fraction(frame, scene.fps)), 'Trace scope differs')
            items = record['variants'][variant]['cameras']
            require([x['camera'] for x in items] == list(scene.camera_ids), 'Camera coverage differs')
            item = next(x for x in items if x['camera'] == camera)
            if not spec.first_frame <= frame <= spec.last_frame:
                continue
            ids = item['local_ids']
            require(len(set(ids)) == len(ids) and all(type(i) is int and i >= 0 for i in ids)
                    and all(len(item[k]) == len(ids) for k in ('xyxy', 'confidence', 'detection_indices')),
                    'Malformed local trace')
            truth = ground.slots[frame, camera]
            gt_ids, iou = geometry(truth, item['xyxy'], camera_spec.width, camera_spec.height)
            acc.update(gt_ids, ids, np.where(iou >= spec.min_iou, 1.-iou, np.nan), frameid=frame)
            for j, identity in enumerate(ids):
                best = int(np.argmax(iou[:, j])) if gt_ids else None
                metadata[frame, identity] = {
                    'frame': frame, 'camera': camera, 'local_id': identity,
                    'detection_index': item['detection_indices'][j], 'score': item['confidence'][j],
                    'xyxy': item['xyxy'][j], 'raw_gt_count': len(truth),
                    'best_gt_id': gt_ids[best] if best is not None else None,
                    'best_iou': float(iou[best, j]) if best is not None else 0.,
                    'admissible_gt_count': int((iou[:, j] >= spec.min_iou).sum())}
            if (frame+1) % 600 == 0:
                print(f'Inspected {frame+1}/{scene.rounds}', flush=True)
        require(stream.readline() == '', 'Trailing trace frames')
    measured = json.loads(mm.metrics.create().compute(acc, metrics=METRICS).to_json(orient='records'))[0]
    expected = source['results'][variant]['local'][f'camera_{camera:04d}']
    for name in METRICS:
        a, b = measured[name], expected[name]
        require((a is None and b is None) or (a is not None and b is not None and abs(a-b) <= 1e-8),
                f'Camera metric not reproduced: {name}: {a} vs {b}')
    rows = []
    for (frame, _), event in acc.mot_events.iterrows():
        if event['Type'] != 'FP':
            continue
        entry = dict(metadata[int(frame), int(event['HId'])])
        entry['category'] = category(entry['raw_gt_count'], entry['best_iou'], spec.min_iou)
        rows.append(entry)
    require(len(rows) == int(measured['num_false_positives']), 'FP event count differs')
    expected_empty = source['results'][variant]['predictions_in_empty_gt_slots'][str(camera)]
    require(sum(r['category'] == 'empty_gt_slot' for r in rows) == expected_empty, 'Empty-GT FP count differs')
    return measured, rows


def render_examples(scene, ground, camera, examples, output):
    if not examples:
        return []
    import av
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    selected_camera = next(c for c in scene.cameras if c.camera_id == camera)
    wanted = {r['frame'] for r in examples}
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    found = set()
    names = []
    with av.open(str(selected_camera.video.path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            require(frame.pts is not None and frame.time_base is not None
                    and frame.pts * Fraction(frame.time_base) == Fraction(index, scene.fps), 'Video PTS differs')
            require((frame.width, frame.height) == (selected_camera.width, selected_camera.height), 'Video size differs')
            if index in wanted:
                raw = Image.fromarray(frame.to_ndarray(format='rgb24'))
                for number, sample in enumerate(examples):
                    if sample['frame'] != index:
                        continue
                    annotated = raw.copy()
                    draw = ImageDraw.Draw(annotated)
                    for identity, box in sorted(ground.slots[index, camera].items()):
                        b = clip_boxes([box], raw.width, raw.height)[0]
                        if np.any(b[2:] <= b[:2]):
                            continue
                        draw.rectangle(tuple(b), outline='lime', width=3)
                        draw.text((b[0], max(0, b[1]-20)), f'GT {identity}', font=font, fill='lime', stroke_width=1, stroke_fill='black')
                    b = clip_boxes([sample['xyxy']], raw.width, raw.height)[0]
                    if np.all(b[2:] > b[:2]):
                        draw.rectangle(tuple(b), outline='red', width=4)
                        draw.text((b[0], min(raw.height-25, b[3])), f"FP local {sample['local_id']}", font=font,
                                  fill='red', stroke_width=1, stroke_fill='black')
                    # A broad context crop, without changing the original predictions.
                    if np.any(b[2:] <= b[:2]):
                        bounds = (0, 0, raw.width, raw.height)
                    else:
                        bounds = (max(0, int(b[0])-100), max(0, int(b[1])-100),
                                  min(raw.width, int(np.ceil(b[2]))+100), min(raw.height, int(np.ceil(b[3]))+100))
                    sheet = Image.new('RGB', (1200, 1040), 'black')
                    sd = ImageDraw.Draw(sheet)
                    lines = [f"camera {camera}, frame {index}, local {sample['local_id']}, score {sample['score']:.3f}",
                             f"{sample['category']}; best IoU {sample['best_iou']:.3f}; green=GT, red=selected FP",
                             'Top: annotated scene. Bottom: raw context (left), annotated context (right).']
                    for j, text in enumerate(lines):
                        sd.text((10, 5+23*j), text, font=font, fill='white')
                    full = ImageOps.contain(annotated, (1200, 675))
                    sheet.paste(full, ((1200-full.width)//2, 80))
                    for offset, image in ((0, raw), (600, annotated)):
                        crop = ImageOps.contain(image.crop(bounds), (595, 275))
                        sheet.paste(crop, (offset+(595-crop.width)//2, 760))
                    name = f"example_{number+1:02d}_frame_{index:06d}_local_{sample['local_id']:04d}.jpg"
                    sheet.save(output/name, quality=92)
                    sample['image'] = name
                    sample['context_bounds'] = list(bounds)
                    names.append(name)
                found.add(index)
            if index >= max(wanted):
                break
    require(found == wanted, 'Selected video frames missing')
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-report', type=Path, required=True)
    parser.add_argument('--camera', type=int, default=362)
    parser.add_argument('--variant', choices=['staged', 'competitive_iou'], default='staged')
    args = parser.parse_args()
    require(version('motmetrics') == '1.4.0', 'Expected pinned motmetrics 1.4.0')
    inputs = {}
    def checked(name, path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        require(expected is None or digest == expected, f'Input checksum changed: {path}')
        inputs[name] = {'path': str(path), 'sha256': digest}
        return path
    source_path = checked('evaluation_report', args.evaluation_report)
    source = json.loads(source_path.read_text())
    require(source.get('completed') is True and source['protocol'] == 'scene_paired_tracking_evaluation_v1', 'Unexpected evaluation')
    for package in ('motmetrics', 'numpy', 'scipy', 'pandas'):
        require(version(package) == source['versions'][package], f'Evaluation dependency changed: {package}')
    for name in ('scene_config', 'source_manifest', 'video_manifest', 'calibration', 'ground_truth'):
        asset = source['inputs'][name]
        checked(name, asset['path'], asset['sha256'])
    for rel in ('src/mtmc/data/ground_truth.py', 'src/mtmc/data/scene.py'):
        checked('code:'+rel, ROOT/rel, source['inputs']['code:'+rel]['sha256'])
    loaded = load_scene(inputs['scene_config']['path'], project_root=ROOT)
    scene, spec = loaded.runtime, loaded.evaluation
    require(spec.ground_truth.sha256 == inputs['ground_truth']['sha256'], 'GT lineage differs')
    require(args.camera in scene.camera_ids and source['scene'] == scene.scene
            and source['evaluation']['frames'] == [spec.first_frame, spec.last_frame]
            and source['evaluation']['min_iou'] == spec.min_iou, 'Scene/evaluation scope differs')
    for c in scene.cameras:
        asset = source['inputs'][f'video_{c.camera_id}']
        require(asset['sha256'] == c.video.sha256, 'Video lineage differs')
        checked(f'video_{c.camera_id}', c.video.path, c.video.sha256)
    checked('diagnostic_code', __file__)
    asset = source['artifacts']['paired_tracks.jsonl.gz']
    trace = checked('paired_tracks', source_path.parent/asset['path'], asset['sha256'])
    ground = load_ground_truth(spec)
    print(f'Frozen camera {args.camera}, {args.variant}; reproducing local matching events...', flush=True)
    metrics, rows = diagnose(trace, scene, spec, ground, source, args.camera, args.variant)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'artifacts/scene_false_positives'/run
    output.mkdir(parents=True, exist_ok=False)
    categories = dict(Counter(r['category'] for r in rows))
    tracks = Counter(r['local_id'] for r in rows)
    bins = Counter((r['frame']//(10*scene.fps))*10 for r in rows)
    examples = select_examples(rows)
    print('Local metrics reproduced: VERIFIED')
    print('FP categories:', json.dumps(categories))
    print('Top local IDs by FP observations:', sorted(tracks.items(), key=lambda x: (-x[1], x[0]))[:10])
    print('FP observations per 10-second scene interval:', sorted(bins.items()))
    csv_path = output/'false_positives.csv'
    fields = ['frame', 'camera', 'local_id', 'detection_index', 'score', 'raw_gt_count',
              'best_gt_id', 'best_iou', 'admissible_gt_count', 'category', 'xyxy']
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print('Decoding selected context frames on CPU; no models or tracking...', flush=True)
    images = render_examples(scene, ground, args.camera, examples, output)
    for asset in inputs.values():
        require(sha256(asset['path']) == asset['sha256'], 'Input changed during diagnostic')
    report = {'completed': True, 'protocol': 'scene_false_positive_diagnostic_v1', 'run_id': run,
              'scene': scene.scene, 'camera': args.camera, 'variant': args.variant, 'inputs': inputs,
              'reproduced_local_metrics': metrics, 'categories': categories,
              'top_fp_tracks': [{'local_id': i, 'fp_observations': n} for i, n in sorted(tracks.items(), key=lambda x: (-x[1], x[0]))[:10]],
              'ten_second_bins': [{'start_seconds': t, 'fp_observations': n} for t, n in sorted(bins.items())],
              'examples': examples, 'visual_review_completed': False,
              'checks': {'local_metrics_reproduced': True, 'FP_events_reproduced': True, 'frozen_inputs_unchanged': True},
              'artifacts': {name: {'path': name, 'sha256': sha256(output/name)} for name in ['false_positives.csv']+images},
              'limits': ['FP means unmatched under the pinned annotation/matching protocol, not necessarily background.',
                         'An admissible but unmatched prediction suggests competition, not proof of a duplicate.',
                         'Examples are deterministic diagnostic selections, not a representative prevalence sample.',
                         'No thresholds, predictions, tracker states, GT or camera regions changed.']}
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    with zipfile.ZipFile(output/'visual_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['report.json', 'false_positives.csv']+images:
            archive.write(output/name, name)
    print(f'Report: {output / "report.json"}')
    print(f'Visual review: {output / "visual_review.zip"}')
    print('Frozen false-positive diagnostic: COMPLETED; visual review pending')


if __name__ == '__main__':
    main()
