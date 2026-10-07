"""Paired causal replay followed by offline identity evaluation of a merge policy."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import replay_global_identity as replay_io
import evaluate_global_identity as evaluation
from mtmc.association import controlled_merge, global_identity
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
require = evaluation.require


def to_plain(value):
    return json.loads(json.dumps(value, default=replay_io.json_default, allow_nan=False))


def read_controlled_record(record, *, source_run, scope, frame, variant, threshold, expected_rows):
    """Validate the new policy explicitly; never relabel it as the old policy."""
    require((record['run_id'], record['identity_scope'], record['frame_index'], record['descriptor_variant'],
             record['min_similarity'], record['policy']) ==
            (source_run, scope, frame, variant, threshold, controlled_merge.POLICY), 'Controlled trace context mismatch')
    require(Fraction(record['timestamp']) == Fraction(frame, 30), 'Controlled timestamp mismatch')
    assigned, slots = {}, set()
    for item in record['assignments']:
        key, gid = ObservationKey(**item['key']), item['global_id']
        require(key in expected_rows and key not in assigned and item['embedding_row'] == expected_rows[key],
                'Controlled observation/row mismatch')
        require(type(gid) is int and gid > 0 and (gid, key.camera_id) not in slots, 'Duplicate/invalid identity slot')
        assigned[key] = gid
        slots.add((gid, key.camera_id))
    require(set(assigned) == set(expected_rows), 'Controlled prediction coverage differs')
    return assigned


def merge_label_category(members, labels):
    values = [labels.get(ObservationKey(**key)) for key in members]
    known = {x for x in values if x is not None}
    if len(known) > 1:
        return 'different_known_gt'
    return 'unresolved' if not values or None in values else 'all_visible_members_same_gt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-evaluation-report', type=Path, required=True)
    parser.add_argument('--variant', choices=('latest', 'mean'), required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--min-support-rounds', type=int, required=True)
    parser.add_argument('--min-support-seconds', required=True)
    parser.add_argument('--max-evidence-gap', required=True)
    args = parser.parse_args()
    target = args.variant, args.threshold
    inputs = {}
    def remember(name, path, expected_hash=None):
        path = Path(path).resolve()
        actual = sha256(path)
        require(expected_hash is None or actual == expected_hash, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path
    print('Verifying paired baseline, frozen groups and assignment checksums...', flush=True)
    baseline_path = remember('baseline_evaluation_report', args.baseline_evaluation_report)
    baseline_eval = json.loads(baseline_path.read_text(encoding='utf-8'))
    require(baseline_eval.get('completed') is True and baseline_eval['protocol']['name'] == 'scene_001_global_2d_identity_v1',
            'Expected completed shared identity evaluation')
    baseline_rows = [r for r in baseline_eval['results'] if (r['variant'], r['threshold']) == target]
    require(len(baseline_rows) == 1, 'Baseline setting is missing or duplicated')
    baseline_metrics = baseline_rows[0]
    entry = baseline_eval['inputs']['identity_report']
    base_report_path = remember('baseline_identity_report', entry['path'], entry['sha256'])
    base_report = json.loads(base_report_path.read_text(encoding='utf-8'))
    require(base_report['replay_id'] == baseline_eval['source_replay_id'] and base_report.get('completed') is True,
            'Baseline replay/evaluation mismatch')
    bp = base_report['protocol']
    require(bp['policy'] == global_identity.POLICY and
            (bp['cameras'], bp['first_frame'], bp['last_frame'], bp['fps']) == ([4, 5, 8], 2, 299, 30), 'Unsupported baseline scope')
    entry = base_report['inputs']['grouping_report']
    grouping_path = remember('grouping_report', entry['path'], entry['sha256'])
    grouping_report = json.loads(grouping_path.read_text(encoding='utf-8'))
    require(grouping_report.get('completed') is True and
            grouping_report['protocol']['name'] == 'scene_001_multicamera_grouping_diagnostic_v1' and
            grouping_report['source_run_id'] == base_report['source_run_id'], 'Unexpected grouping source')
    entry = base_report['inputs']['groups']
    groups_path = remember('groups', entry['path'], entry['sha256'])
    require(entry['sha256'] == grouping_report['artifacts']['groups.jsonl.gz']['sha256'], 'Grouping hashes differ')
    entry = baseline_eval['inputs']['assignments']
    assignment_path = remember('baseline_assignments', entry['path'], entry['sha256'])
    require(entry['sha256'] == base_report['artifacts']['assignments.jsonl.gz']['sha256'], 'Assignment hashes differ')
    for name in ('tracks', 'ground_truth'):
        item = baseline_eval['inputs'][name]
        remember(name, item['path'], item['sha256'])
    settings = [(v, t) for v in bp['variants'] for t in bp['thresholds']]
    expected_settings = [(v, t) for v in grouping_report['protocol']['variants'] for t in grouping_report['protocol']['thresholds']]
    require(settings == expected_settings and target in settings and len(settings) == len(set(settings)), 'Setting grids differ')
    base_selected = [r for r in base_report['results'] if (r['variant'], r['threshold']) == target]
    group_selected = [r for r in grouping_report['results'] if (r['variant'], r['threshold']) == target]
    require(len(base_selected) == len(group_selected) == 1, 'Invalid source setting summary')
    idle = Fraction(bp['max_idle_seconds'])
    configuration = {'max_idle_seconds': str(idle), 'min_support_rounds': args.min_support_rounds,
                     'min_support_seconds': str(Fraction(args.min_support_seconds)),
                     'max_evidence_gap': str(Fraction(args.max_evidence_gap))}
    common = dict(run_id=base_report['source_run_id'], max_idle=idle, descriptor_variant=args.variant, min_similarity=args.threshold)
    base_manager = global_identity.GlobalIdentityManager(**common)
    candidate = controlled_merge.ControlledMergeIdentityManager(**common, min_support_rounds=args.min_support_rounds,
                min_support_seconds=Fraction(args.min_support_seconds), max_evidence_gap=Fraction(args.max_evidence_gap))
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%S%fZ')
    scope = f'{run_id}/{controlled_merge.POLICY}/{args.variant}/tau={args.threshold!r}'
    output = ROOT / 'artifacts/controlled_merge' / run_id
    output.mkdir(parents=True, exist_ok=False)
    traces_path = output / 'assignments.jsonl.gz'
    decision_counts, reset_counts, original_counts, original_edges = Counter(), Counter(), Counter(), Counter()
    allocated, expired, absorbed, emitted, source_rows = set(), set(), set(), set(), set()
    base_assignments, rows_by_frame, frame_stats = {}, {}, []
    merge_count = 0
    print('Phase 1: causal replay; GT is not passed to either manager...', flush=True)
    with gzip.open(groups_path, 'rt', encoding='utf-8') as groups, \
            gzip.open(assignment_path, 'rt', encoding='utf-8') as old, \
            gzip.open(traces_path, 'wt', encoding='utf-8') as destination:
        for number in range(2, 300):
            for setting in settings:
                gline, aline = groups.readline(), old.readline()
                require(bool(gline) and bool(aline), 'Truncated paired source traces')
                g, saved = json.loads(gline), json.loads(aline)
                require((g['frame_index'], g['variant'], g['threshold']) == (number, *setting) and
                        (saved['frame_index'], saved['descriptor_variant'], saved['min_similarity']) == (number, *setting),
                        'Paired source ordering differs')
                if setting != target:
                    continue
                runtime, labels = replay_io.decode_round(g, run_id=common['run_id'], frame=number,
                    variant=args.variant, threshold=args.threshold, fps=30, cameras=(4, 5, 8))
                counts, edges = replay_io.source_counts(g, runtime, labels)
                original_counts.update(counts)
                original_edges.update(edges)
                rows = {r['embedding_row'] for r in labels.values()}
                require(not rows & source_rows, 'Source embedding row reused across frames')
                source_rows.update(rows)
                rows_by_frame[number] = {k: r['embedding_row'] for k, r in labels.items()}
                baseline = base_manager.update(runtime)
                rebuilt = to_plain(replay_io.encode_output(baseline, labels, base_selected[0]['identity_scope']))
                require(rebuilt == saved, 'Baseline replay does not reproduce the full frozen record')
                base_assignments[number] = {a.key: a.global_id for a in baseline.assignments}
                result = candidate.update(runtime)
                require({a.key for a in result.assignments} == set(labels), 'Candidate changed observation coverage')
                fresh = {a.global_id for a in result.base_assignments if a.reason == 'new_identity'}
                require(not fresh & allocated, 'Allocated global ID reused')
                allocated.update(fresh)
                require(not expired.intersection(result.expired_global_ids), 'Identity expired twice')
                expired.update(result.expired_global_ids)
                current_absorbed = {gid for e in result.merge_events for gid in e.absorbed_global_ids}
                require(not current_absorbed & (absorbed | expired), 'Identity absorbed twice or after expiry')
                absorbed.update(current_absorbed)
                retained = {s.global_id for s in result.identities}
                require(len(allocated) == len(expired) + len(absorbed) + len(retained) and
                        allocated == expired | absorbed | retained, 'Identity lifetime accounting failed')
                emitted.update(a.global_id for a in result.assignments)
                require(not {a.global_id for a in result.assignments} & (absorbed | expired), 'Retired identity emitted')
                require(len({(a.global_id, a.key.camera_id) for a in result.assignments}) == len(result.assignments),
                        'Repeated current camera in global identity')
                decision_counts.update(d.outcome for d in result.merge_decisions)
                reset_counts.update(r.reason for r in result.candidate_resets)
                merge_count += len(result.merge_events)
                frame_stats.append({'frame_index': number, 'observations': len(result.assignments),
                    'allocated_ids': len(fresh), 'merge_events': len(result.merge_events),
                    'absorbed_ids': len(current_absorbed), 'expired_ids': len(result.expired_global_ids),
                    'retained_ids': len(retained), 'pending_candidates': len(result.pending_candidates)})
                destination.write(json.dumps(replay_io.encode_output(result, labels, scope),
                                  default=replay_io.json_default, allow_nan=False) + '\n')
            if (number + 1) % 60 == 0:
                print(f'Replayed through frame {number}/299', flush=True)
        require(groups.readline() == old.readline() == '', 'Unexpected trailing source records')
    require(dict(original_counts) == group_selected[0]['groups'], 'Grouping counters differ')
    require(all(original_edges[k] == group_selected[0]['after'][k] for k in replay_io.grouping_eval.EDGE_FIELDS),
            'Source retained-edge counts differ')
    require(len(source_rows) == base_selected[0]['observations'], 'Source observation count differs')
    frozen_hash = sha256(traces_path)
    print('Full baseline records reproduced: VERIFIED', flush=True)
    print('Phase 2: offline box/GT evaluation of already frozen outputs...', flush=True)
    ground = evaluation.history.load_ground_truth(Path(inputs['ground_truth']['path']))
    slots, frame_maps, excluded, outside = evaluation.build_slots(Path(inputs['tracks']['path']), ground, common['run_id'])
    require(frame_maps == rows_by_frame, 'Replay and evaluated box row mappings differ')
    before, after = evaluation.IdentityCounts(), evaluation.IdentityCounts()
    merge_diagnostics = []
    category_counts = Counter({k: 0 for k in ('all_visible_members_same_gt', 'different_known_gt', 'unresolved')})
    with gzip.open(traces_path, 'rt', encoding='utf-8') as handle:
        for number in range(2, 300):
            line = handle.readline()
            require(bool(line), 'Truncated candidate trace')
            row = json.loads(line)
            assigned = read_controlled_record(row, source_run=common['run_id'], scope=scope, frame=number,
                variant=args.variant, threshold=args.threshold, expected_rows=frame_maps[number])
            unique_gt = {}
            for camera in (4, 5, 8):
                gt, keys, mask = slots[number, camera]
                before.update(gt, [base_assignments[number][k] for k in keys], mask)
                after.update(gt, [assigned[k] for k in keys], mask)
                gt_degree, pred_degree = mask.sum(axis=1), mask.sum(axis=0)
                for i, j in zip(*np.nonzero(mask)):
                    if gt_degree[i] == pred_degree[j] == 1:
                        unique_gt[keys[j]] = gt[i]
            for event in row['merge_events']:
                category = merge_label_category(event['members'], unique_gt)
                category_counts[category] += 1
                merge_diagnostics.append({**event, 'diagnostic_category': category,
                    'diagnostic_members': [{**key, 'unique_overlap_gt_id': unique_gt.get(ObservationKey(**key))}
                                           for key in event['members']]})
        require(handle.readline() == '', 'Unexpected trailing candidate records')
    baseline_result, baseline_mapping = before.result()
    result_metrics, result_mapping = after.result()
    require(all(baseline_result[k] == baseline_metrics[k] for k in baseline_result), 'Baseline metrics not reproduced')
    require(all(baseline_result[k] == result_metrics[k] for k in ('gt_observations', 'predicted_observations', 'camera_time_slots')),
            'Paired evaluation denominators differ')
    require(result_metrics['predicted_identities'] == len(emitted), 'Emitted identity count differs from metric count')
    require(sum(category_counts.values()) == merge_count, 'Merge diagnostic coverage differs')
    require(sha256(traces_path) == frozen_hash, 'Offline evaluation changed frozen output')
    for item in inputs.values():
        evaluation.checked(item['path'], item['sha256'])
    comparison = [dict(policy=global_identity.POLICY, **baseline_result),
                  dict(policy=controlled_merge.POLICY, **result_metrics)]
    evaluation.history.write_csv(output / 'comparison.csv', comparison)
    evaluation.history.write_csv(output / 'by_frame.csv', frame_stats)
    (output / 'merge_diagnostics.json').write_text(json.dumps(merge_diagnostics, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'identity_matching.json').write_text(json.dumps({'baseline': baseline_mapping, 'controlled': result_mapping}, indent=2) + '\n', encoding='utf-8')
    delta = 100 * (result_metrics['idf1'] - baseline_result['idf1'])
    report = {
        'completed': True, 'experiment_id': run_id, 'created_utc': now.isoformat(), 'source_run_id': common['run_id'],
        'identity_scope': scope,
        'protocol': {'name': 'scene_001_controlled_merge_paired_v1', 'policy': controlled_merge.POLICY,
            'variant': args.variant, 'threshold': args.threshold, 'configuration': configuration,
            'cameras': [4, 5, 8], 'first_frame': 2, 'last_frame': 299, 'fps': 30, 'min_iou': 0.5,
            'role': 'Paired integration experiment; no deployment calibration', 'selected_threshold': None,
            'runtime_gt_used': False, 'past_assignments_rewritten': False,
            'metric_protocol': 'One shared GT/predicted identity assignment across all camera/time slots',
            'merge_diagnostic': 'Mutually unique spatial overlaps of visible event members at acceptance time only'},
        'inputs': inputs, 'baseline_metrics': baseline_result, 'controlled_metrics': result_metrics,
        'delta_idf1_pp': delta,
        'lifecycle': {'allocated_ids': len(allocated), 'ever_emitted_ids': len(emitted), 'absorbed_ids': len(absorbed),
            'expired_ids': len(expired), 'retained_ids_at_end': len(retained), 'merge_events': merge_count},
        'merge_decisions': dict(decision_counts), 'candidate_resets': dict(reset_counts),
        'merge_event_diagnostics': dict(category_counts), 'merge_examples': merge_diagnostics[:10],
        'excluded_zero_area_gt': excluded, 'fully_outside_predictions': outside,
        'checks': {'full_baseline_trace_parity': True, 'baseline_metric_parity': True, 'same_observations_and_boxes': True,
            'same_metric_denominators': True, 'identity_lifecycle_accounting': True, 'evaluation_did_not_change_output': True},
        'versions': {name: version(name) for name in ('numpy', 'scipy')},
        'code_sha256': {q.name: sha256(q) for q in (Path(__file__), Path(controlled_merge.__file__),
            Path(global_identity.__file__), Path(replay_io.__file__), Path(evaluation.__file__),
            Path(evaluation.history.__file__), Path(evaluation.history.snapshot.__file__))},
        'artifacts': {q.name: {'path': q.name, 'sha256': sha256(q)} for q in sorted(output.iterdir()) if q.is_file()},
        'limits': ['Single reused training fragment; explicit experimental confirmation parameters, not final calibration',
            'Correct visible members at merge time do not certify retained absent members or later identity continuity',
            'Same-GT and wrong-GT merge categories use diagnostic IoU evidence, not visual proof',
            'Full-group heuristic can still merge stable incorrect groups and propagate local track identity changes',
            'No direct numeric-ID comparison across runs; compare identity metrics and evidence',
            'Absorbed IDs remain part of the scored historical prediction namespace; no retrospective ID replacement',
            'No performance benchmark, parameter sweep, end-of-video expiry flush or rollback of accepted merges'],
    }
    path = output / 'report.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Policy                 IDF1      IDP      IDR    IDTP   IDFP   IDFN')
    for name, metrics in (('No-merge baseline', baseline_result), ('Controlled merge', result_metrics)):
        print(f"{name:<21} {metrics['idf1']:8.2%} {metrics['idp']:8.2%} {metrics['idr']:8.2%} "
              f"{metrics['idtp']:7} {metrics['idfp']:6} {metrics['idfn']:6}")
    print(f'Delta IDF1: {delta:+.2f} pp')
    print('Lifecycle:', json.dumps(report['lifecycle']))
    print('Merge decisions:', json.dumps(dict(decision_counts)))
    print('Accepted merge diagnostics:', json.dumps(dict(category_counts)))
    print('Baseline trace, baseline metrics, common denominators and lifecycle: VERIFIED')
    print(f'Report: {path}')
    print('Controlled merge paired experiment: COMPLETED')


if __name__ == '__main__':
    main()
