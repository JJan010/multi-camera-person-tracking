"""Known-answer tests for interval boundaries and preserved frozen identities."""
import copy
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile

from evaluate_appearance_global import quality
from evaluate_tracking_windows import VARIANTS, CAMERAS, check_local_prefix, evaluate_windows, validate_windows
from experiment_appearance_bytetrack import evaluate as local_quality


def save(path, records):
    with gzip.open(path, 'wt') as stream:
        for record in records:
            stream.write(json.dumps(record) + '\n')


def local_records(records, run):
    return [{'run_id': run, 'frame_index': r['frame_index'], 'timestamp': r['timestamp'],
             'variants': {v: r['variants'][v]['cameras'] for v in VARIANTS}} for r in records]


def main():
    rounds, split = 6, 4
    ground = {(f, c): ({7: [10, 10, 30, 60]} if c == 4 else {})
              for f in range(2, rounds) for c in CAMERAS}
    records = []
    for frame in range(rounds):
        variants = {}
        for variant in VARIANTS:
            identity = 10 if frame < split or variant == 'direct_appearance' else 20
            cameras = []; assignments = []
            for camera in CAMERAS:
                visible = camera == 4 and frame >= 2
                cameras.append({'camera': camera, 'local_ids': [identity] if visible else [],
                                'xyxy': [[10, 10, 30, 60]] if visible else [],
                                'confidence': [0.9] if visible else []})
                if visible:
                    assignments.append({'key': {'camera_id': camera, 'local_id': identity, 'frame_index': frame},
                                        'global_id': identity})
            variants[variant] = {'cameras': cameras, 'identity': {'assignments': assignments, 'merge_events': []}}
        records.append({'run_id': 'long', 'frame_index': frame, 'timestamp': str(Fraction(frame, 30)), 'variants': variants})
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory); full = root / 'full.gz'; prefix = root / 'prefix.gz'
        lp = root / 'local_prefix.gz'; lf = root / 'local_full.gz'
        save(full, records); save(prefix, records[:split])
        save(lp, local_records(records[:split], 'short')); save(lf, local_records(records, 'long-local'))
        windows = evaluate_windows(full, ground, rounds, split)
        full_metrics, _, _ = quality(full, ground, rounds)
        prefix_metrics, _, _ = quality(prefix, ground, split)
        full_local, _ = local_quality(lf, ground, rounds)
        prefix_local, _ = local_quality(lp, ground, split)
        validate_windows(windows, full_metrics, full_local, prefix_metrics, prefix_local)
        check_local_prefix(lp, full, split, 'short', 'long')
        for window in windows.values():
            for variant in VARIANTS:
                data = window['variants'][variant]
                assert data['global']['idf1'] == 1
                assert data['local']['OVERALL']['idf1'] == 1
                assert data['local']['OVERALL']['num_switches'] == 0
        assert full_metrics['baseline']['idf1'] == .5
        assert full_metrics['direct_appearance']['idf1'] == 1
        assert full_local['baseline']['OVERALL']['num_switches'] == 1
        print('Boundary switch: each window IDF1=100%, full baseline IDF1=50%; runtime IDs preserved: OK')
        print('Window IDSW excludes the boundary-crossing switch; full-run IDSW includes it: OK')
        changed = copy.deepcopy(records)
        changed[5]['variants']['direct_iou']['cameras'][0]['local_ids'] = [21]
        assignment = changed[5]['variants']['direct_iou']['identity']['assignments'][0]
        assignment['key']['local_id'] = assignment['global_id'] = 21
        save(full, changed)
        altered = evaluate_windows(full, ground, rounds, split)
        m = altered['second']['variants']['direct_iou']
        assert m['global']['idf1'] == .5 and m['local']['OVERALL']['num_switches'] == 1
        assert altered['first'] == windows['first']
        print('Within-window switch is counted; split frame included once; first window unchanged: OK')
        changed[3]['variants']['baseline']['cameras'][0]['confidence'] = [.8]
        save(full, changed)
        try:
            check_local_prefix(lp, full, split, 'short', 'long')
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError('Changed prefix accepted')
        print('Changed prior local outputs rejected, including confidence/provenance fields: OK')
    print('Tracking window evaluation checks: PASSED')


if __name__ == '__main__':
    main()
