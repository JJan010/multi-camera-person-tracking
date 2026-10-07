"""Known-answer continuity diagnostics: fragmentation, mixing and stable IDs."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile

from diagnose_appearance_continuity import analyze, CAMERAS


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'trace.gz'
        for case in ('stable', 'fragmented', 'takeover'):
            records, ground = [], {}
            for frame in range(6):
                gt_id = 8 if case == 'takeover' and frame >= 4 else 7
                track_id = 20 if case == 'fragmented' and frame >= 4 else 10
                cameras, assigned = [], []
                for camera in CAMERAS:
                    visible = camera == 4 and frame >= 2
                    ground[frame, camera] = {gt_id: [10, 10, 30, 60]} if visible else {}
                    cameras.append({'camera': camera, 'local_ids': [track_id] if visible else [],
                                    'xyxy': [[10, 10, 30, 60]] if visible else []})
                    if visible:
                        assigned.append({'key': {'camera_id': camera, 'local_id': track_id, 'frame_index': frame},
                                         'global_id': track_id})
                records.append({'run_id': 'fixture', 'frame_index': frame, 'timestamp': str(Fraction(frame, 30)),
                                'variants': {'direct_appearance': {'cameras': cameras, 'identity': {'assignments': assigned}}}})
            with gzip.open(path, 'wt') as stream:
                for row in records:
                    stream.write(json.dumps(row) + '\n')
            result, evidence = analyze(path, ground, 6, 4, 'direct_appearance', 'fixture')
            full = result['scopes']['full']
            assert full['decomposition']['framewise_spatial_f1_ceiling'] == 1
            assert result['scopes']['first']['global_metrics']['idf1'] == 1
            assert result['scopes']['second']['global_metrics']['idf1'] == 1
            assert result['separate_minus_shared_idtp'] == (0 if case == 'stable' else 2)
            assert full['global_metrics']['idf1'] == (1 if case == 'stable' else .5)
            summary = full['evidence_summary']
            assert summary['gt_identities_with_multiple_global_ids'] == int(case == 'fragmented')
            assert summary['global_ids_with_multiple_gt_ids'] == int(case == 'takeover')
            assert summary['local_tracks_with_multiple_gt_ids'] == int(case == 'takeover')
            assert len(evidence.transitions) == int(case == 'takeover')
            print(case + ': spatial ceiling, mapping gap and distinct fragmentation/mixing evidence: OK')
    print('Appearance continuity checks: PASSED')


if __name__ == '__main__':
    main()
