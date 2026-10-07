"""Render an audited local transition and its actual prior strong-sample crops."""
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from math import ceil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import evaluate_mtmc_sequence as sequence
from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import ObservationKey, sha256
from preview_identity_switch import annotate, common_roi, font, panel

ROOT = Path(__file__).resolve().parents[1]
require = sequence.require


def collect_views(path, run, camera, local_id, frames, ground, event):
    views = {}
    with gzip.open(path, 'rt') as stream:
        for frame in range(max(frames) + 1):
            line = stream.readline(); require(bool(line), 'Truncated global trace')
            row = json.loads(line)
            require(row['frame_index'] == frame and row['run_id'] == run, 'Global trace scope differs')
            if frame not in frames:
                continue
            data = row['variants']['direct_appearance']
            cameras = [c for c in data['cameras'] if c['camera'] == camera]
            require(len(cameras) == 1, 'Missing/duplicate selected camera')
            current = cameras[0]
            keys = tuple(ObservationKey(camera, i, frame) for i in current['local_ids'])
            assigned = {ObservationKey(**a['key']): a['global_id'] for a in data['identity']['assignments']}
            boxes = np.asarray(current['xyxy'], np.float64).reshape(-1, 4)
            gt, mask, unique, _, _ = sequence.spatial_slot(ground.get((frame, camera), {}), keys, boxes)
            key = ObservationKey(camera, local_id, frame)
            candidates = [gt[i] for i in np.flatnonzero(mask[:, keys.index(key)])] if key in keys else []
            views[frame] = {'frame': frame, 'target_unique_gt_id': unique.get(key), 'target_spatial_candidates': candidates,
                'tracks': [{'local_id': k.local_id, 'global_id': assigned[k], 'xyxy': box.tolist(),
                            'score': current['confidence'][j], 'embedding_row': current['embedding_rows'][j],
                            'detection_index': current['detection_indices'][j]}
                           for j, (k, box) in enumerate(zip(keys, boxes))],
                'selected_gt_boxes': {g: ground.get((frame, camera), {}).get(g) for g in (event['previous_gt'], event['current_gt'])}}
    require(set(views) == set(frames), 'Selected frame coverage differs')
    return views


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-report', type=Path, required=True)
    args = parser.parse_args(); inputs = {}
    def checked(name, path, digest=None):
        path = Path(path).resolve(); actual = sha256(path)
        require(digest is None or actual == digest, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path
    probe_path = checked('probe_report', args.probe_report); probe = json.loads(probe_path.read_text())
    require(probe.get('completed') is True and probe['protocol'] == 'frozen_local_appearance_gate_probe_v1', 'Expected gate probe')
    event = probe['selected_transition']; camera = probe['target']['camera']; local_id = probe['target']['local_id']
    require((event['camera'], event['local_id']) == (camera, local_id), 'Target/transition mismatch')
    spec = probe['artifacts']['timeline']; timeline = json.loads(checked('timeline', probe_path.parent / spec['path'], spec['sha256']).read_text())
    audited = [r for r in timeline if r['frame'] == event['frame']]
    require(len(audited) == 1 and audited[0]['visible'], 'Missing audited transition observation')
    history_frames = audited[0]['prior_strong_sample_frames']
    require(len(set(history_frames)) == len(history_frames) and all(f < event['frame'] for f in history_frames), 'Noncausal gallery')
    spec = probe['inputs']['diagnostic_report']; diagnostic_path = checked('diagnostic_report', spec['path'], spec['sha256'])
    diagnostic = json.loads(diagnostic_path.read_text())
    spec = diagnostic['inputs']['evaluation_report']; evaluation_path = checked('evaluation_report', spec['path'], spec['sha256'])
    evaluation = json.loads(evaluation_path.read_text())
    require(evaluation.get('completed') is True and evaluation['protocol'] == 'appearance_local_to_global_paired_v1', 'Expected appearance evaluation')
    require(evaluation['inputs']['local_report']['sha256'] == probe['inputs']['local_report']['sha256'], 'Different local experiment')
    spec = evaluation['artifacts']['global_tracks']; trace = checked('global_tracks', evaluation_path.parent / spec['path'], spec['sha256'])
    spec = evaluation['inputs']['ground_truth']; gt_path = checked('ground_truth', spec['path'], spec['sha256'])
    spec = probe['inputs']['local:pipeline_report']; source_path = checked('pipeline_report', spec['path'], spec['sha256'])
    source = json.loads(source_path.read_text()); rounds = source['summary']['rounds']
    require(source['run_id'] == evaluation['source_run_id'], 'Different pipeline source')
    before, after = event['previous_evidence_frame'], event['frame']
    require(2 <= before < after < rounds, 'Invalid transition bounds')
    context = sorted({max(0, before - 10), before, after - 1, after, min(rounds - 1, after + 5),
                      *(range(before + 1, after) if after - before <= 5 else ())})
    frames = sorted(set(context) | set(history_frames))
    ground = sequence.load_ground_truth(gt_path, rounds)
    views = collect_views(trace, evaluation['run_id'], camera, local_id, frames, ground, event)
    targets = {f: next((t for t in v['tracks'] if t['local_id'] == local_id), None) for f, v in views.items()}
    for f, g in ((before, event['previous_gt']), (after, event['current_gt'])):
        require(views[f]['target_unique_gt_id'] == g, 'Diagnostic GT evidence differs')
    require(targets[after]['detection_index'] == audited[0]['detection_index']
            and targets[after]['score'] == audited[0]['score'], 'Audited accepted detection differs')
    for f in history_frames:
        require(targets[f] is not None and targets[f]['score'] >= probe['configuration']['tracker']['track_activation_threshold'],
                'Prior strong sample is unavailable in emitted local records; explicit internal-candidate lookup required')
    roi_boxes = [targets[f]['xyxy'] for f in context if targets[f] is not None]
    roi_boxes += [b for f in context for b in views[f]['selected_gt_boxes'].values() if b is not None]
    roi = common_roi(roi_boxes)
    spec = source['inputs'][f'video_{camera}']; video = checked('video', spec['path'], spec['sha256'])
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/gate_transition_preview' / run; output.mkdir(parents=True, exist_ok=False)
    annotated_sheet = Image.new('RGB', (2400, 600 * ceil(len(context) / 3)), '#151515')
    raw_sheet = annotated_sheet.copy()
    gallery_frames = [*history_frames, after]
    gallery = Image.new('RGB', (960, 350 * ceil(len(gallery_frames) / 4)), '#151515')
    covered = set()
    print('Transition:', json.dumps(event)); print('Context:', context); print('Prior strong gallery:', history_frames)
    print('Decoding one camera with exact PTS; no inference or tracker changes...', flush=True)
    import av
    with av.open(str(video)) as container:
        stream = container.streams.video[0]; stream.codec_context.thread_count = 1; stream.thread_type = 'SLICE'
        for index, frame in enumerate(container.decode(stream)):
            require(frame.pts is not None and frame.time_base is not None
                    and Fraction(frame.pts) * frame.time_base == Fraction(index, 30), 'Video timestamp mismatch')
            if index in views:
                image = Image.fromarray(frame.to_ndarray(format='rgb24')); require(image.size == (1920, 1080), 'Unexpected video size')
                if index in context:
                    target = targets[index]
                    text = [f'Camera {camera} | frame {index} | {index/30:.3f}s',
                            f"L{local_id}/G{target['global_id']}" if target else f'L{local_id}: absent',
                            f"Unique GT: {views[index]['target_unique_gt_id']} | candidates: {views[index]['target_spatial_candidates']}",
                            f"Cyan: target | green: GT {event['previous_gt']} | pink: GT {event['current_gt']}"]
                    n = context.index(index); pos = ((n % 3) * 800, (n // 3) * 600)
                    annotated = annotate(image, views[index], event, camera, local_id)
                    annotated_sheet.paste(panel(annotated.crop(roi), text), pos)
                    raw_sheet.paste(panel(image.crop(roi), text[:3]), pos)
                    annotated.save(output / f'frame_{index:06d}_annotated.png')
                    image.crop(roi).save(output / f'frame_{index:06d}_raw_context.png')
                if index in gallery_frames:
                    bounds, _ = crop_geometry(targets[index]['xyxy'], 1920, 1080)
                    require(bounds is not None, 'Gallery crop outside image')
                    crop = image.crop(bounds); crop.save(output / f'frame_{index:06d}_raw_crop.png')
                    tile = Image.new('RGB', (240, 350), '#151515'); draw = ImageDraw.Draw(tile)
                    role = 'CURRENT (not prior)' if index == after else 'PRIOR strong sample'
                    for j, line in enumerate((role, f'Frame {index}', f"Unique GT: {views[index]['target_unique_gt_id']}")):
                        draw.text((6, 5 + 23 * j), line, fill='white', font=font(16))
                    fitted = ImageOps.contain(crop, (228, 270)); tile.paste(fitted, ((240-fitted.width)//2, 76))
                    n = gallery_frames.index(index); gallery.paste(tile, ((n % 4) * 240, (n // 4) * 350))
                covered.add(index)
            if index == max(frames):
                break
    require(covered == set(frames), 'Missing decoded frames')
    annotated_sheet.save(output / 'transition_annotated.jpg', quality=95)
    raw_sheet.save(output / 'transition_raw.jpg', quality=95)
    gallery.save(output / 'prior_gallery_and_current.jpg', quality=95)
    for spec in inputs.values():
        require(sha256(Path(spec['path'])) == spec['sha256'], 'Input changed during preview')
    report = {'completed': True, 'protocol': 'audited_appearance_transition_preview_v1', 'run_id': run, 'inputs': inputs,
              'transition': event, 'context_frames': context, 'prior_gallery_frames': history_frames, 'roi': roi,
              'views': [views[f] for f in frames], 'script_sha256': sha256(Path(__file__)),
              'artifacts': {p.name: {'path': p.name, 'sha256': sha256(p)} for p in sorted(output.iterdir())},
              'limits': ['Unique IoU GT evidence is diagnostic, not manual visual confirmation.',
                         'Raw gallery crops use accepted boxes with the original integer crop bounds; display resizing is presentation only.',
                         'No new embedding computation, tracking, threshold adjustment or retrospective repair.']}
    path = output / 'report.json'; path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f'Report: {path}'); print(f'Contact sheets: {output}')
    print('Audited transition preview: COMPLETED; visual review pending')


if __name__ == '__main__':
    main()
