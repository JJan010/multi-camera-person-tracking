"""Diagnose identity continuity in frozen paired local/global tracking outputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

import numpy as np

import diagnose_global_identity as spatial
from diagnose_mtmc_sequence import Evidence, pooled_local
import evaluate_mtmc_sequence as sequence
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
require = sequence.require
CAMERAS = (4, 5, 8)


def mapping_gap(counters):
    results = {name: counter.result() for name, counter in counters.items()}
    maps = {name: {r['gt_id']: r for r in rows} for name, (_, rows) in results.items()}
    rows = []
    for gt in sorted(counters['full'].ground):
        selected = {name: maps[name].get(gt, {'global_id': None, 'idtp': 0}) for name in maps}
        rows.append({'gt_id': gt, 'selected_assignments': selected,
                     'separate_minus_shared_idtp': selected['first']['idtp'] + selected['second']['idtp'] - selected['full']['idtp']})
    rows.sort(key=lambda r: (-r['separate_minus_shared_idtp'], r['gt_id']))
    total = results['first'][0]['idtp'] + results['second'][0]['idtp'] - results['full'][0]['idtp']
    require(total >= 0 and sum(r['separate_minus_shared_idtp'] for r in rows) == total, 'Invalid identity assignment gap')
    return total, rows, results


def analyze(path, ground, rounds, split, variant, run_id):
    require(2 < split < rounds, 'Invalid diagnostic interval')
    counters = {w: sequence.metric.IdentityCounts() for w in ('full', 'first', 'second')}
    local = {w: {c: sequence.metric.IdentityCounts() for c in CAMERAS} for w in counters}
    evidence = {w: Evidence() for w in counters}
    spatial_tp = Counter(); stats = {w: Counter() for w in counters}
    with gzip.open(path, 'rt') as stream:
        for frame in range(rounds):
            line = stream.readline(); require(bool(line), 'Truncated diagnostic trace')
            row = json.loads(line)
            require(row['run_id'] == run_id and row['frame_index'] == frame
                    and Fraction(row['timestamp']) == Fraction(frame, 30), 'Mixed diagnostic scope/time')
            if frame < 2:
                continue
            data = row['variants'][variant]
            assigned = {ObservationKey(**a['key']): a['global_id'] for a in data['identity']['assignments']}
            require(len(assigned) == len(data['identity']['assignments']), 'Duplicate assignment key')
            require(sorted(c['camera'] for c in data['cameras']) == list(CAMERAS), 'Invalid camera coverage')
            scopes = ('full', 'first' if frame < split else 'second')
            all_evidence, seen = [], set()
            for camera in data['cameras']:
                c = camera['camera']; ids = camera['local_ids']
                keys = tuple(ObservationKey(c, i, frame) for i in ids)
                require(len(keys) == len(set(keys)), 'Duplicate local ID')
                boxes = np.asarray(camera['xyxy'], np.float64).reshape(-1, 4)
                gt, mask, _, _, _ = sequence.spatial_slot(ground[frame, c], keys, boxes)
                gids = [assigned[k] for k in keys]; seen.update(keys)
                require(len(gids) == len(set(gids)), 'Same-camera global ID collision')
                ceiling, items, counts = spatial.inspect_slot(gt, keys, gids, mask)
                all_evidence.extend(items)
                for scope in scopes:
                    counters[scope].update(gt, gids, mask)
                    local[scope][c].update(gt, ids, mask)
                    spatial_tp[scope] += ceiling; stats[scope].update(counts)
            require(seen == set(assigned), 'Local/global coverage differs')
            for scope in scopes:
                evidence[scope].update(frame, all_evidence)
            if (frame + 1) % 600 == 0 or frame == rounds - 1:
                print(f'Inspected {frame + 1}/{rounds}', flush=True)
        require(stream.readline() == '', 'Trailing diagnostic records')
    gap, mapping_rows, metrics = mapping_gap(counters)
    scopes = {}
    for scope, ev in evidence.items():
        gt_rows, global_rows, local_rows = ev.tables()
        local_metrics, _ = pooled_local(local[scope])
        scopes[scope] = {'global_metrics': metrics[scope][0], 'local_metrics': local_metrics,
                        'decomposition': spatial.decompose(metrics[scope][0], spatial_tp[scope]),
                        'spatial_evidence_counts': dict(stats[scope]),
                        'evidence_summary': {
                            'gt_identities_with_multiple_global_ids': sum(len(h) > 1 for h in ev.gt_global.values()),
                            'global_ids_with_multiple_gt_ids': sum(len(h) > 1 for h in ev.global_gt.values()),
                            'local_tracks_with_multiple_gt_ids': sum(len(h) > 1 for h in ev.local_gt.values()),
                            'local_evidence_label_transitions': len(ev.transitions)},
                        'gt_fragmentation': gt_rows, 'global_mixing': global_rows, 'local_mixing': local_rows}
    return {'scopes': scopes, 'separate_minus_shared_idtp': gap, 'mapping_contributions': mapping_rows}, evidence['full']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-report', type=Path, required=True)
    parser.add_argument('--variant', choices=('baseline', 'direct_iou', 'direct_appearance'), default='direct_appearance')
    args = parser.parse_args()
    inputs = {}
    def checked(name, path, digest=None):
        path = Path(path).resolve(); actual = sha256(path)
        require(digest is None or digest == actual, f'Checksum mismatch: {path}')
        inputs[name] = {'path': str(path), 'sha256': actual}
        return path
    report_path = checked('evaluation_report', args.evaluation_report)
    report = json.loads(report_path.read_text())
    require(report.get('completed') is True and report['protocol'] == 'appearance_local_to_global_paired_v1'
            and 'temporal_evaluation' in report, 'Expected completed temporal appearance evaluation')
    temporal = report['temporal_evaluation']; split = temporal['split_frame']
    require(temporal['runtime_reset_at_split'] is False, 'Expected continuous runtime')
    first, last = report['configuration']['frames']; rounds = last + 1
    require(first == 0 and temporal['windows']['first']['frames'] == [2, split - 1]
            and temporal['windows']['second']['frames'] == [split, last], 'Invalid recorded windows')
    spec = report['artifacts']['global_tracks']
    trace = checked('global_tracks', report_path.parent / spec['path'], spec['sha256'])
    spec = report['inputs']['ground_truth']; gt_path = checked('ground_truth', spec['path'], spec['sha256'])
    spec = report['inputs']['local_report']; local_path = checked('local_report', spec['path'], spec['sha256'])
    local_report = json.loads(local_path.read_text())
    require(local_report['run_id'] == report['local_experiment_run_id'], 'Local experiment scope differs')
    print(f'Frozen continuity diagnostic: {args.variant}; no models or state changes...', flush=True)
    ground = sequence.load_ground_truth(gt_path, rounds)
    result, evidence = analyze(trace, ground, rounds, split, args.variant, report['run_id'])
    for scope, data in result['scopes'].items():
        expected = report['metrics'][args.variant] if scope == 'full' else temporal['windows'][scope]['variants'][args.variant]['global']
        require(data['global_metrics'] == expected, f'Global metrics differ: {scope}')
        expected_local = local_report['metrics'][args.variant]['OVERALL'] if scope == 'full' else temporal['windows'][scope]['variants'][args.variant]['local']['OVERALL']
        for key in ('idtp', 'idfp', 'idfn', 'idf1', 'idp', 'idr'):
            require(abs(data['local_metrics'][key] - expected_local[key]) <= 1e-9, f'Local {key} differs: {scope}')
    for spec in inputs.values():
        require(sha256(Path(spec['path'])) == spec['sha256'], 'Input changed during diagnostic')
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT / 'artifacts/appearance_continuity' / run; output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name, value in (('evidence_segments.json', evidence.all_segments()),
                        ('local_label_transitions.json', evidence.transitions)):
        path = output / name; path.write_text(json.dumps(value, indent=2) + '\n')
        artifacts[name] = {'path': name, 'sha256': sha256(path)}
    result.update(completed=True, protocol='frozen_appearance_continuity_diagnostic_v1', run_id=run,
                  source_evaluation_run_id=report['run_id'], variant=args.variant, inputs=inputs,
                  configuration={'frames': [2, last], 'split_frame': split}, artifacts=artifacts,
                  script_sha256=sha256(Path(__file__)),
                  limits=['GT is used only for offline diagnosis, never to repair assignments.',
                          'Evidence uses mutually unique IoU matches; ambiguous and missing evidence is excluded.',
                          'Evidence label transitions are not CLEAR IDSW and may span gaps.',
                          'Per-GT mapping contributions depend on the selected optimal assignments; ties can change attribution.',
                          'An individual contribution can be negative due to one-to-one assignment competition.',
                          'Separate-minus-shared IDTP is not an event count or a recoverable runtime guarantee.'])
    path = output / 'report.json'; path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for scope, data in result['scopes'].items():
        print(f"{scope}: global IDF1={100*data['global_metrics']['idf1']:.2f}%; "
              f"spatial ceiling={100*data['decomposition']['framewise_spatial_f1_ceiling']:.2f}%")
        print('  Evidence:', json.dumps(data['evidence_summary']))
    print('Separate-minus-shared IDTP:', result['separate_minus_shared_idtp'])
    print('Top mapping contributions: GT / first GID / second GID / full GID / IDTP difference')
    for row in result['mapping_contributions'][:5]:
        m = row['selected_assignments']
        print(row['gt_id'], m['first']['global_id'], m['second']['global_id'], m['full']['global_id'], row['separate_minus_shared_idtp'])
    for name, fields in (('gt_fragmentation', ('gt_id', 'distinct_global_ids', 'evidence_outside_largest_id', 'simultaneous_split_frames')),
                         ('global_mixing', ('global_id', 'distinct_gt_ids', 'evidence_outside_largest_gt', 'simultaneous_mix_frames')),
                         ('local_mixing', ('camera', 'local_id', 'distinct_gt_ids', 'evidence_outside_largest_gt'))):
        print('Full-run top ' + name + ': ' + ' / '.join(fields))
        for row in result['scopes']['full'][name][:5]:
            print(*(row[k] for k in fields))
    print(f'Report: {path}')
    print('Appearance continuity diagnostic: COMPLETED; all full/window metrics reproduced; predictions unchanged')


if __name__ == '__main__':
    main()
