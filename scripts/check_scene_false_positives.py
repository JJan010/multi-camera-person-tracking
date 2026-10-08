"""Known-answer FP event categories, including a competing prediction and IoU boundary."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from diagnose_scene_false_positives import diagnose, geometry, select_examples
from mtmc.data.ground_truth import GroundTruth


def main():
    scene = SimpleNamespace(scene='fixture', rounds=3, fps=30, camera_ids=(362,),
                            cameras=(SimpleNamespace(camera_id=362, width=100, height=100),))
    spec = SimpleNamespace(first_frame=0, last_frame=2, min_iou=.5)
    expected = {'num_frames': 3, 'num_objects': 2, 'num_predictions': 5,
                'num_false_positives': 3, 'num_misses': 0, 'num_switches': 0,
                'idf1': 4/7, 'precision': 2/5, 'recall': 1.}
    source = {'run_id': 'fixture', 'cache_run_id': 'cache', 'results': {'staged': {
        'local': {'camera_0362': expected}, 'predictions_in_empty_gt_slots': {'362': 1}}}}
    ground = GroundTruth({(0, 362): {0: [0, 0, 10, 10]},
                          (1, 362): {0: [0, 0, 10, 10]}, (2, 362): {}}, ((2, 362),))
    observations = [([1, 2], [[0, 0, 10, 10], [0, 0, 10, 10]]),
                    ([1, 3], [[0, 0, 20, 10], [30, 30, 40, 40]]),
                    ([1], [[0, 0, 10, 10]])]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'trace.jsonl.gz'
        with gzip.open(path, 'wt') as stream:
            for frame, (ids, boxes) in enumerate(observations):
                record = {'run_id': 'fixture', 'source_run_id': 'cache', 'scene': 'fixture',
                          'frame_index': frame, 'timestamp': str(Fraction(frame, 30)),
                          'variants': {'staged': {'cameras': [{'camera': 362, 'local_ids': ids,
                              'xyxy': boxes, 'confidence': [.8]*len(ids), 'detection_indices': list(range(len(ids)))}]}}}
                stream.write(json.dumps(record)+'\n')
        metrics, rows = diagnose(path, scene, spec, ground, source, 362, 'staged')
        assert metrics['num_false_positives'] == 3
        assert {r['category'] for r in rows} == {'empty_gt_slot', 'no_admissible_gt', 'admissible_but_unmatched'}
        assert next(r for r in rows if r['frame'] == 1)['local_id'] == 3
        assert select_examples(rows) == select_examples(list(reversed(rows)))
        source['results']['staged']['local']['camera_0362']['num_false_positives'] = 4
        try:
            diagnose(path, scene, spec, ground, source, 362, 'staged')
        except ValueError as error:
            assert 'not reproduced' in str(error)
        else:
            raise AssertionError('A mismatching source metric was accepted')
    ids, iou = geometry({0: [0, 0, 10, 10], 1: [-20, 0, -10, 10]}, [[0, 0, 20, 10]], 100, 100)
    assert ids == [0] and iou.shape == (1, 1) and iou[0, 0] == .5
    print('CLEAR FP events: empty GT, absent geometric candidate and competing prediction: OK')
    print('Inclusive IoU boundary, GT ID zero, clipped-out GT and exact source counts: OK')
    print('Deterministic sample selection and mismatching metric rejection: OK')
    print('False-positive diagnostic checks: PASSED')


if __name__ == '__main__':
    main()
