"""Compare cached candidates to the exact causal gallery of a frozen local track."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))


def descriptor(values):
    values = np.asarray(values, np.float64)
    require(values.ndim == 2 and values.shape[1] == 512 and len(values) > 0,
            'Expected a nonempty 512-dimensional gallery')
    require(np.isfinite(values).all() and np.all(np.abs(np.linalg.norm(values, axis=1) - 1) < 1e-4),
            'Invalid normalized gallery vectors')
    mean = values.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-12 else values[-1].copy()


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-report', required=True, type=Path)
    args = parser.parse_args()
    inputs = {}

    def checked(name, path, digest=None):
        path = Path(path).resolve()
        actual = sha256(path)
        require(digest is None or digest == actual, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path

    def linked(name, spec, parent=Path('.')):
        return checked(name, parent / spec['path'], spec['sha256'])

    preview_path = checked('preview_report', args.preview_report)
    preview = json.loads(preview_path.read_text())
    require(preview.get('completed') and preview['protocol'] == 'audited_appearance_transition_preview_v1',
            'Expected completed transition preview')
    probe_path = linked('probe_report', preview['inputs']['probe_report'])
    probe = json.loads(probe_path.read_text())
    require(probe.get('completed') and probe['protocol'] == 'frozen_local_appearance_gate_probe_v1',
            'Expected completed gate probe')
    require(preview['transition'] == probe['selected_transition'], 'Mixed transitions')
    timeline_path = linked('timeline', probe['artifacts']['timeline'], probe_path.parent)
    require(sha256(timeline_path) == preview['inputs']['timeline']['sha256'], 'Mixed timelines')
    timeline = {r['frame']: r for r in json.loads(timeline_path.read_text())}
    local_path = linked('local_report', probe['inputs']['local_report'])
    local = json.loads(local_path.read_text())
    require(local.get('completed') and local['protocol'] == 'local_appearance_bytetrack_paired_v1'
            and local['configuration'] == probe['configuration'], 'Mixed local experiment')
    local_trace = linked('local_variants', local['artifacts']['tracks'], local_path.parent)
    require(sha256(local_trace) == probe['inputs']['local_variants']['sha256'], 'Mixed local trace')
    paths = {}
    for name in ('cache_report', 'detections.jsonl', 'embeddings.npy'):
        paths[name] = linked(name, local['inputs'][name])
        require(inputs[name]['sha256'] == probe['inputs']['local:' + name]['sha256'], 'Mixed probe/cache inputs')
    cache = json.loads(paths['cache_report'].read_text())
    require(cache['run_id'] == local['cache_run_id'], 'Mixed cache scope')
    features = np.load(paths['embeddings.npy'], mmap_mode='r', allow_pickle=False)
    require(features.dtype == np.float32 and features.shape == (cache['summary']['encoded'], 512),
            'Unexpected cached embedding array')
    camera, target = probe['target']['camera'], probe['target']['local_id']
    views = {v['frame']: v for v in preview['views']}
    frames = sorted(preview['context_frames'])
    needed = set(frames)
    for f in frames:
        row = timeline[f]
        require(row['visible'] and row['local_id'] == target, 'Target not observed at requested frame')
        history = row['prior_strong_sample_frames']
        require(0 < len(history) <= 8 and history == sorted(set(history))
                and all(0 < f - past <= 30 for past in history), 'Invalid causal gallery')
        needed.update(history)

    # Preserve the accepted candidate mapping; never recover it using GT or nearest IoU.
    accepted = {}
    with gzip.open(local_trace, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            f = row['frame_index']
            if f > max(needed):
                break
            if f not in needed:
                continue
            require(row['run_id'] == local['run_id'] and row['source_run_id'] == local['source_run_id'], 'Mixed trace scope')
            cams = [c for c in row['variants']['direct_appearance'] if c['camera'] == camera]
            require(len(cams) == 1, 'Missing or duplicate camera')
            c = cams[0]
            indices = [i for i, identity in enumerate(c['local_ids']) if identity == target]
            require(len(indices) == 1, 'A requested gallery observation was not emitted; cannot reconstruct it')
            i = indices[0]
            accepted[f] = {'index': c['detection_indices'][i], 'row': c['embedding_rows'][i],
                           'box': c['xyxy'][i], 'score': c['confidence'][i]}
    require(set(accepted) == needed, 'Missing local history observations')

    candidates = {}
    with paths['detections.jsonl'].open() as handle:
        for line in handle:
            row = json.loads(line)
            f = row['frame_index']
            if f > max(needed):
                break
            if f not in needed:
                continue
            require(row['cache_run_id'] == cache['run_id'] and row['source_run_id'] == local['source_run_id'], 'Mixed candidate scope')
            values = [d for d in row['detections'] if d['camera_id'] == camera]
            require(len({d['detection_index'] for d in values}) == len(values), 'Duplicate candidate indices')
            by_index = {d['detection_index']: d for d in values}
            source = accepted[f]
            d = by_index[source['index']]
            require(d['embedding_row'] == source['row'] and d['xyxy'] == source['box']
                    and d['confidence'] == source['score'], 'Accepted candidate provenance differs')
            require(source['row'] is not None and 0 <= source['row'] < len(features), 'Missing accepted feature')
            candidates[f] = values
    require(set(candidates) == needed, 'Missing candidate frames')

    activation = local['configuration']['tracker']['track_activation_threshold']
    output_rows = []
    for f in frames:
        prior_frames = timeline[f]['prior_strong_sample_frames']
        require(all(accepted[p]['score'] >= activation for p in prior_frames), 'Weak sample entered strong gallery')
        prior = descriptor([features[accepted[p]['row']] for p in prior_frames])
        parity = abs(cosine(prior, features[accepted[f]['row']]) - timeline[f]['cosine_prior'])
        require(parity <= 1e-10, f'Prior descriptor does not reproduce observed gate cosine at {f}')
        require(accepted[f]['index'] == timeline[f]['detection_index'], 'Selected index differs from probe')
        print(f'\nFrame {f}; exact prior gallery: {prior_frames}')
        print('  Index Stage   Score   Cos_prior  Delta/accepted   GT IoUs (diagnostic)')
        accepted_cos = cosine(prior, features[accepted[f]['row']])
        values = []
        for d in candidates[f]:
            vector_row = d['embedding_row']
            sim = None
            if vector_row is not None:
                require(0 <= vector_row < len(features), 'Invalid candidate feature row')
                vector = features[vector_row]
                require(np.isfinite(vector).all() and abs(np.linalg.norm(vector) - 1) < 1e-4, 'Invalid candidate feature')
                sim = cosine(prior, vector)
            overlaps = {identity: iou(d['xyxy'], box) for identity, box in views[f]['selected_gt_boxes'].items()}
            selected = d['detection_index'] == accepted[f]['index']
            stage = 'high' if d['confidence'] >= activation else ('low' if d['confidence'] > .1 else 'unused')
            item = {'detection_index': d['detection_index'], 'score': d['confidence'], 'stage': stage,
                    'accepted': selected, 'cosine_prior': sim, 'delta_vs_accepted': None if sim is None else sim - accepted_cos,
                    'diagnostic_gt_ious': overlaps}
            values.append(item)
            # GT affects only which rows are printed, never descriptors or rankings saved in the report.
            if selected or max(overlaps.values(), default=0) >= .5:
                similarity = 'n/a' if sim is None else f'{sim:.4f}'
                delta = 'n/a' if sim is None else f'{sim - accepted_cos:+.4f}'
                label = ', '.join(f'{k}={v:.4f}' for k, v in overlaps.items())
                print(f"{'*' if selected else ' '} {d['detection_index']:5d} {stage:6s} {d['confidence']:.4f} {similarity:>10s} {delta:>15s}   {label}")
        output_rows.append({'frame': f, 'prior_strong_sample_frames': prior_frames, 'cosine_parity_error': parity,
                            'candidates': values})
    for spec in inputs.values():
        require(sha256(spec['path']) == spec['sha256'], 'Input changed during comparison')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/candidate_appearance_probe' / run
    output.mkdir(parents=True, exist_ok=False)
    report = {'completed': True, 'protocol': 'frozen_candidate_appearance_comparison_v1', 'run_id': run,
              'inputs': inputs, 'target': probe['target'], 'frames': output_rows,
              'script_sha256': sha256(__file__),
              'limits': ['GT-selected diagnostic context; no tracker, inference or runtime changes.',
                         'Every candidate compared to the same actual prior descriptor; accepted cosine reproduced.',
                         'All candidate similarities saved; GT IoU filters console display only.',
                         'No Kalman-motion eligibility or competition with other tracks evaluated.',
                         'Higher appearance similarity alone does not establish a feasible or correct reassignment.']}
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print('\nAccepted candidate provenance and prior-cosine parity: VERIFIED')
    print(f'Report: {path}')
    print('Frozen candidate comparison: COMPLETED; assignments unchanged')


if __name__ == '__main__':
    main()
