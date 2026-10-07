"""Replay unchanged global identity logic over frozen local tracking variants."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import evaluate_mtmc_sequence as sequence
from check_frozen_bytetrack_replay import candidate_arrays, compare_outputs
from mtmc.pipeline.core import IdentityStage
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch, sha256
from run_mtmc import load_matrices, dump

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('baseline', 'direct_iou', 'direct_appearance')
require = sequence.require


def direct_observations(cameras, candidates, vectors, frame):
    """Join accepted candidates to new local IDs using recorded indices, never IoU."""
    require(len(cameras) == 3 and sorted(c['camera'] for c in cameras) == [4, 5, 8], 'Local camera coverage differs')
    lookup = {(r['camera_id'], r['detection_index']): r for r in candidates['detections']}
    require(len(lookup) == len(candidates['detections']), 'Duplicate candidate key')
    records, keys, embeddings = [], [], []
    for camera in sorted(cameras, key=lambda c: c['camera']):
        c = camera['camera']; ids = camera['local_ids']
        n = len(ids)
        require(all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == n
                and all(len(camera[k]) == n for k in ('xyxy', 'confidence', 'detection_indices', 'embedding_rows')),
                'Local output columns differ')
        require(len(set(camera['detection_indices'])) == n, 'One candidate assigned to multiple local tracks')
        for i in sorted(range(n), key=lambda j: ids[j]):
            index = camera['detection_indices'][i]
            require(type(index) is int and (c, index) in lookup, 'Unknown accepted candidate')
            item = lookup[c, index]
            require(item['frame_index'] == frame
                    and camera['xyxy'][i] == item['xyxy'] and camera['confidence'][i] == item['confidence']
                    and camera['embedding_rows'][i] == item['embedding_row'], 'Candidate provenance differs')
            key = ObservationKey(c, ids[i], frame)
            bounds, fraction = crop_geometry(item['xyxy'], 1920, 1080)
            row = item['embedding_row']
            require((row is None) == (bounds is None), 'Descriptor availability differs')
            records.append(CropRecord(key, float(item['confidence']), tuple(item['xyxy']), bounds, fraction))
            if row is not None:
                require(type(row) is int and 0 <= row < len(vectors), 'Invalid embedding row')
                keys.append(key); embeddings.append(vectors[row])
    array = np.stack(embeddings).astype(np.float32, copy=False) if embeddings else np.empty((0, 512), np.float32)
    return tuple(records), ReIDBatch(tuple(keys), (Fraction(frame, 30),) * len(keys), array)


def lifecycle(record, counters, emitted):
    emitted.update(a['global_id'] for a in record['assignments'])
    counters['observations'] += len(record['assignments'])
    counters['allocated_ids'] += len({a['global_id'] for a in record['base_assignments'] if a['reason'] == 'new_identity'})
    counters['absorbed_ids'] += sum(len(e['absorbed_global_ids']) for e in record['merge_events'])
    counters['expired_ids'] += len(record['expired_global_ids'])
    counters['merge_events'] += len(record['merge_events'])
    counters['retained_ids_at_end'] = len(record['identities'])
    require(counters['allocated_ids'] == counters['absorbed_ids'] + counters['expired_ids'] + counters['retained_ids_at_end'],
            'Global lifecycle accounting differs')


def quality(path, ground, rounds):
    metrics, mapping, merge_counts = {}, {}, {}
    for variant in VARIANTS:
        counter = sequence.metric.IdentityCounts(); slots = []; categories = Counter()
        with gzip.open(path, 'rt') as stream:
            for frame in range(rounds):
                line = stream.readline(); require(bool(line), 'Truncated global experiment')
                record = json.loads(line); require(record['frame_index'] == frame, 'Global frame order differs')
                data = record['variants'][variant]
                assigned = {ObservationKey(**a['key']): a['global_id'] for a in data['identity']['assignments']}
                require(len(assigned) == len(data['identity']['assignments']), 'Duplicate global assignment')
                seen, unique = set(), {}
                for camera in data['cameras']:
                    c = camera['camera']; keys = tuple(ObservationKey(c, i, frame) for i in camera['local_ids'])
                    require(len(keys) == len(set(keys)), 'Duplicate local key')
                    seen.update(keys)
                    if frame < 2:
                        continue
                    boxes = np.asarray(camera['xyxy'], np.float64).reshape(-1, 4)
                    gt, mask, evidence, _, _ = sequence.spatial_slot(ground[frame, c], keys, boxes)
                    unique.update(evidence)
                    ids = [assigned[k] for k in keys]
                    counter.update(gt, ids, mask); slots.append((gt, ids, mask))
                require(seen == set(assigned), 'Global/local observation coverage differs')
                for event in data['identity']['merge_events']:
                    labels = [unique.get(ObservationKey(**m)) for m in event['members']]
                    category = ('unannotated_frame' if frame < 2 else 'unresolved' if not labels or any(x is None for x in labels)
                                else 'all_visible_members_same_gt' if len(set(labels)) == 1 else 'different_known_gt')
                    categories[category] += 1
            require(stream.readline() == '', 'Trailing global frames')
        metrics[variant], mapping[variant] = counter.result()
        sequence.metric.check_reference(metrics[variant], sequence.metric.reference_metrics(slots))
        merge_counts[variant] = dict(categories)
    return metrics, mapping, merge_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-report', type=Path, required=True)
    parser.add_argument('--baseline-evaluation', type=Path, required=True)
    parser.add_argument('--split-frame', type=int)
    parser.add_argument('--reference-global-report', type=Path)
    args = parser.parse_args()
    if (args.split_frame is None) != (args.reference_global_report is None):
        parser.error('--split-frame and --reference-global-report must be supplied together')
    inputs = {}

    def checked(name, path, expected=None):
        path = Path(path).resolve(); actual = sha256(path)
        require(expected is None or actual == expected, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}; return path

    print('Verifying frozen local variants, candidate cache and unchanged global modules...', flush=True)
    local_path = checked('local_report', args.local_report)
    local = json.loads(local_path.read_text())
    require(local.get('completed') is True and local.get('baseline_exact') is True
            and local['protocol'] == 'local_appearance_bytetrack_paired_v1', 'Expected completed paired local experiment')
    for name in ('cache_report', 'pipeline_report', 'tracks', 'detections.jsonl', 'embeddings.npy'):
        spec = local['inputs'][name]; checked(name, spec['path'], spec['sha256'])
    spec = local['artifacts']['tracks']; local_trace = checked('local_variants', local_path.parent / spec['path'], spec['sha256'])
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    require(source['run_id'] == local['source_run_id'] and cache['run_id'] == local['cache_run_id'], 'Mixed source scopes')
    rounds, cfg = source['summary']['rounds'], source['configuration']
    require(cfg['cameras'] == [4, 5, 8] and cfg['fps'] == 30
            and local['configuration']['runtime_frames'] == [0, rounds - 1], 'Unexpected replay scope')
    short_global = short_local = short_trace = None
    if args.split_frame is not None:
        require(2 < args.split_frame < rounds, 'Split must be inside the evaluated run')
        reference_path = checked('short_global_report', args.reference_global_report)
        short_global = json.loads(reference_path.read_text())
        require(short_global.get('completed') is True and short_global['protocol'] == 'appearance_local_to_global_paired_v1',
                'Expected completed shorter global experiment')
        spec = short_global['inputs']['local_report']
        short_local_path = checked('short_local_report', spec['path'], spec['sha256'])
        short_local = json.loads(short_local_path.read_text())
        require(short_local.get('completed') is True and short_local.get('baseline_exact') is True
                and short_local['protocol'] == local['protocol']
                and short_global['local_experiment_run_id'] == short_local['run_id'], 'Invalid shorter local experiment')
        require(short_local['configuration']['runtime_frames'] == [0, args.split_frame - 1]
                and short_global['configuration']['frames'] == [0, args.split_frame - 1], 'Reference/split duration differs')
        omit = {'runtime_frames', 'evaluation_frames'}
        require({k: v for k, v in local['configuration'].items() if k not in omit} ==
                {k: v for k, v in short_local['configuration'].items() if k not in omit}, 'Local settings changed across intervals')
        for key in ('appearance_variant', 'appearance_threshold', 'geometry', 'identity', 'history'):
            require(short_global['configuration'][key] == cfg[key], f'Global setting changed: {key}')
        require(short_global['configuration']['local_appearance_threshold'] == local['configuration']['appearance_threshold'],
                'Local appearance threshold changed')
        spec = short_local['artifacts']['tracks']
        short_trace = checked('short_local_variants', short_local_path.parent / spec['path'], spec['sha256'])
    required_code = {'src/mtmc/pipeline/core.py', 'src/mtmc/reid/history.py',
                     *('src/mtmc/association/' + name + '.py' for name in
                       ('geometry', 'pairwise', 'grouping', 'global_identity', 'controlled_merge'))}
    require(required_code <= set(source['code_sha256']), 'Missing original global code provenance')
    for relative, digest in source['code_sha256'].items():
        if relative.startswith('src/mtmc/association/') or relative in ('src/mtmc/pipeline/core.py', 'src/mtmc/reid/history.py'):
            checked('code:' + relative, ROOT / relative, digest)
    pipeline_path = Path(inputs['pipeline_report']['path'])
    spec = source['artifacts']['global_tracks']; globals_path = checked('original_globals', pipeline_path.parent / spec['path'], spec['sha256'])
    spec = source['inputs']['calibration']; calibration_path = checked('calibration', spec['path'], spec['sha256'])
    matrices = load_matrices(calibration_path, cfg['cameras'])
    baseline_path = checked('baseline_evaluation', args.baseline_evaluation)
    baseline = json.loads(baseline_path.read_text())
    require(baseline.get('completed') is True and baseline['protocol']['name'] == 'scene_001_sequence_global_2d_identity_v1'
            and baseline['inputs']['pipeline_report']['sha256'] == inputs['pipeline_report']['sha256'],
            'Expected evaluation of the same original pipeline')
    vectors = np.load(inputs['embeddings.npy']['path'], mmap_mode='r', allow_pickle=False)
    require(vectors.dtype == np.float32 and vectors.shape == (cache['summary']['encoded'], 512), 'Invalid cached features')
    stages, histories = {}, {}
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for name in VARIANTS[1:]:
        scope = run + '/' + name
        stages[name] = IdentityStage(scope, matrices, cfg['coordinate_space'], variant=cfg['appearance_variant'],
            threshold=cfg['appearance_threshold'], **cfg['geometry'], identity_configuration=cfg['identity'])
        histories[name] = AppearanceHistory(scope, max_observations=cfg['history']['max_observations'],
                                            max_age=Fraction(cfg['history']['max_age_seconds']))
    output = ROOT / 'artifacts/appearance_global' / run; output.mkdir(parents=True, exist_ok=False)
    (output / 'run_status.json').write_text(json.dumps({'completed': False, 'run_id': run}) + '\n')
    trace_path = output / 'global_tracks.jsonl.gz'
    counters = {name: Counter() for name in VARIANTS}; emitted = {name: set() for name in VARIANTS}
    cached_next = original_next = total_candidates = 0
    print('Phase 1: replay unchanged history, geometry, grouping and global identity; no GT input...', flush=True)
    with Path(inputs['tracks']['path']).open() as src, globals_path.open() as glob, \
            Path(inputs['detections.jsonl']['path']).open() as det, gzip.open(local_trace, 'rt') as loc, \
            gzip.open(trace_path, 'wt') as saved:
        for frame in range(rounds):
            lines = [f.readline() for f in (src, glob, det, loc)]; require(all(lines), 'Truncated frozen input')
            original, original_global, candidates, variants = map(json.loads, lines)
            require(variants['frame_index'] == frame and variants['source_run_id'] == source['run_id']
                    and variants['run_id'] == local['run_id'], 'Local variant scope differs')
            _, _, original_next = sequence.validate_round(original, original_global, frame=frame,
                run=source['run_id'], config=cfg, next_row=original_next)
            arrays, expected, cached_next, _ = candidate_arrays(candidates, original, frame=frame,
                cache_run=cache['run_id'], source_run=source['run_id'], next_row=cached_next)
            total_candidates += sum(len(a[0]) for a in arrays.values())
            for camera in variants['variants']['baseline']:
                actual = SimpleNamespace(tracker_id=np.asarray(camera['local_ids']), xyxy=np.asarray(camera['xyxy']).reshape(-1, 4),
                                         confidence=np.asarray(camera['confidence']))
                require(compare_outputs(actual, expected[camera['camera']])['passed'], 'Baseline local trace differs')
            current = {'baseline': {'identity': original_global, 'cameras': variants['variants']['baseline']}}
            for name in VARIANTS[1:]:
                cameras = variants['variants'][name]
                records, features = direct_observations(cameras, candidates, vectors, frame)
                history = histories[name].update(frame, Fraction(frame, 30), features)
                descriptor = history.mean if cfg['appearance_variant'] == 'mean' else history.latest
                identity, _, _, _, _, _ = stages[name].update(frame, Fraction(frame, 30), descriptor, records)
                current[name] = {'identity': json.loads(dump(asdict(identity))), 'cameras': cameras}
            for name in VARIANTS:
                lifecycle(current[name]['identity'], counters[name], emitted[name])
            saved.write(dump({'run_id': run, 'frame_index': frame, 'timestamp': str(Fraction(frame, 30)), 'variants': current}) + '\n')
            if (frame + 1) % 300 == 0 or frame == rounds - 1:
                print(f'Replayed {frame + 1}/{rounds}', flush=True)
        require(all(f.readline() == '' for f in (src, glob, det, loc)), 'Trailing frozen input')
    require(original_next == source['summary']['total_embeddings'] and cached_next == len(vectors)
            and total_candidates == cache['summary']['detections'], 'Feature/candidate coverage differs')
    for name in VARIANTS:
        require(counters[name]['observations'] == local['counts'][name]['observations'], 'Local/global observation count differs')
        counters[name]['ever_emitted_ids'] = len(emitted[name])
    frozen = sha256(trace_path)
    print('Phase 2: offline shared global-ID evaluation; one identity matching across all cameras...', flush=True)
    spec = baseline['inputs']['ground_truth']; gt_path = checked('ground_truth', spec['path'], spec['sha256'])
    ground = sequence.load_ground_truth(gt_path, rounds)
    metrics, mappings, categories = quality(trace_path, ground, rounds)
    require(metrics['baseline'] == baseline['pipeline_metrics'], 'Original global quality did not reproduce exactly')
    windows = None
    if args.split_frame is not None:
        from evaluate_tracking_windows import check_local_prefix, evaluate_windows, validate_windows
        print('Phase 3: disjoint interval metrics; frozen predictions and continuous runtime state...', flush=True)
        check_local_prefix(short_trace, trace_path, args.split_frame, short_local['run_id'], run)
        windows = evaluate_windows(trace_path, ground, rounds, args.split_frame)
        validate_windows(windows, metrics, local['metrics'], short_global['metrics'], short_local['metrics'])
        print('All three local prefixes, first-minute metrics and window denominators: VERIFIED', flush=True)
    require(sha256(trace_path) == frozen, 'Global predictions changed during evaluation')
    for spec in inputs.values():
        require(sha256(Path(spec['path'])) == spec['sha256'], 'Input changed during replay')
    (output / 'identity_matching.json').write_text(json.dumps(mappings, indent=2) + '\n')
    report = {'completed': True, 'protocol': 'appearance_local_to_global_paired_v1', 'run_id': run,
              'source_run_id': source['run_id'], 'local_experiment_run_id': local['run_id'], 'inputs': inputs,
              'configuration': {'frames': [0, rounds - 1], 'evaluation_frames': [2, rounds - 1],
                                'appearance_variant': cfg['appearance_variant'], 'appearance_threshold': cfg['appearance_threshold'],
                                'geometry': cfg['geometry'], 'identity': cfg['identity'], 'history': cfg['history'],
                                'local_appearance_threshold': local['configuration']['appearance_threshold']},
              'metrics': metrics, 'lifecycle': {k: dict(v) for k, v in counters.items()}, 'accepted_merge_diagnostics': categories,
              'checks': {'original_global_metrics_exact': True, 'all_variants_motmetrics_agreement': True,
                         'candidate_provenance': True, 'global_algorithm_source_hashes': True},
              'artifacts': {'global_tracks': {'path': trace_path.name, 'sha256': frozen},
                            'identity_matching': {'path': 'identity_matching.json', 'sha256': sha256(output / 'identity_matching.json')}},
              'script_sha256': sha256(Path(__file__)),
              'limits': ['Same reused development minute; no independent validation or deployment threshold selection.',
                         'Original global outputs are preserved as the baseline, not recomputed with cached features.',
                         'Direct variants share cached candidate features and identical downstream policy.',
                         'Original-vs-direct comparison includes output mapping and small cache numerical differences.',
                         'Global history still uses all encoded tracked observations, independently of the local strong-sample gallery.',
                         'Returned box counts may differ; each variant uses its own prediction denominator.',
                         'Same-GT visible merge members do not establish the purity of the whole identity.']}
    if windows is not None:
        report['limits'][0] = 'Combined development prefix and unseen temporal suffix; see temporal_evaluation for separate scores.'
        report['temporal_evaluation'] = {
            'split_frame': args.split_frame, 'runtime_reset_at_split': False,
            'evaluation_accumulators_reset_at_split': True, 'windows': windows,
            'checks': {'local_variant_prefix_exact': True, 'first_window_local_and_global_metrics_reproduced': True,
                       'all_global_window_metrics_motmetrics_agreement': True, 'count_partition_verified': True},
            'limits': ['Second window shares scene and people with development; not independent-scene validation.',
                       'Each window has its own identity assignment, so ID scores/counts are not additive.',
                       'Window IDSW starts with empty evaluator history; boundary-crossing changes are not counted there.']}
        report['window_evaluator_sha256'] = sha256(ROOT / 'scripts/evaluate_tracking_windows.py')
    path = output / 'report.json'; path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (output / 'run_status.json').write_text(json.dumps({'completed': True, 'run_id': run}) + '\n')
    print('Variant                Global IDF1   IDP     IDR     IDTP    IDFP    IDFN')
    for name in VARIANTS:
        m = metrics[name]
        print(f"{name:22} {100*m['idf1']:9.2f}% {100*m['idp']:6.2f}% {100*m['idr']:6.2f}% "
              f"{m['idtp']:7} {m['idfp']:7} {m['idfn']:7}")
        print('  Lifecycle:', json.dumps(dict(counters[name])))
        print('  Accepted merges:', json.dumps(categories[name]))
    if windows is not None:
        for name, window in windows.items():
            print(f"Window {name}: frames {window['frames'][0]}..{window['frames'][1]}")
            print('Variant                 Local IDF1  Global IDF1  Local IDSW    Global IDTP/IDFP/IDFN')
            for variant in VARIANTS:
                g = window['variants'][variant]['global']
                m = window['variants'][variant]['local']['OVERALL']
                print(f"{variant:22} {100*m['idf1']:9.2f}% {100*g['idf1']:10.2f}% {m['num_switches']:11} "
                      f"   {g['idtp']}/{g['idfp']}/{g['idfn']}")
    print(f'Report: {path}')
    print('Local-to-global paired evaluation: COMPLETED; runtime baseline unchanged')


if __name__ == '__main__':
    main()
