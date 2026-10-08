"""Known-answer tests for frozen miss attribution and detection provenance."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from diagnose_scene_misses import CATEGORIES, diagnose, select_examples
from mtmc.data.ground_truth import GroundTruth


def main():
    scene = SimpleNamespace(scene='fixture', rounds=4, fps=30, camera_ids=(364,),
                            cameras=(SimpleNamespace(camera_id=364, width=100, height=100),))
    spec = SimpleNamespace(first_frame=0, last_frame=3, min_iou=.5)
    expected = {'num_frames': 4, 'num_objects': 4, 'num_predictions': 1,
                'num_false_positives': 0, 'num_misses': 3, 'num_switches': 0,
                'idf1': .4, 'precision': 1., 'recall': .25}
    source = {'run_id': 'fixture', 'cache_run_id': 'cache', 'results': {'staged': {'local': {'camera_0364': expected}}}}
    ground = GroundTruth({(0, 364): {0: [0, 0, 10, 10]}, (1, 364): {0: [0, 0, 10, 10]},
                          (2, 364): {0: [0, 0, 10, 10], 1: [0, 0, 10, 10]}, (3, 364): {}}, ((3, 364),))
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        trace, cache = folder/'trace.jsonl.gz', folder/'cache.jsonl'
        with gzip.open(trace, 'wt') as output, cache.open('w') as saved:
            for frame in range(4):
                boxes = [[0, 0, 20, 10]] if frame == 1 else [[0, 0, 10, 10]] if frame == 2 else []
                scores = [.3] if frame == 1 else [.8] if frame == 2 else []
                dets = [{'camera_id': 364, 'frame_index': frame, 'detection_index': i, 'xyxy': box, 'confidence': scores[i]}
                        for i, box in enumerate(boxes)]
                saved.write(json.dumps({'cache_run_id': 'cache', 'scene': 'fixture', 'frame_index': frame,
                    'timestamp': str(Fraction(frame, 30)), 'camera_ids': [364], 'detections': dets})+'\n')
                emitted = frame == 2
                item = {'camera': 364, 'local_ids': [0] if emitted else [], 'xyxy': boxes if emitted else [],
                        'confidence': scores if emitted else [], 'detection_indices': [0] if emitted else []}
                output.write(json.dumps({'run_id': 'fixture', 'source_run_id': 'cache', 'scene': 'fixture',
                    'frame_index': frame, 'timestamp': str(Fraction(frame, 30)), 'variants': {'staged': {'cameras': [item]}}})+'\n')
        metrics, rows = diagnose(trace, cache, scene, spec, ground, source, 364, 'staged')
        assert metrics['num_misses'] == 3 and {r['category'] for r in rows} == set(CATEGORIES)
        at_boundary = next(r for r in rows if r['frame'] == 1)
        assert at_boundary['best_candidate_iou'] == .5 and at_boundary['category'] == CATEGORIES[1]
        assert select_examples(rows) == select_examples(list(reversed(rows)))
        data = cache.read_text().splitlines()
        altered = json.loads(data[2]); altered['detections'][0]['confidence'] = .9
        data[2] = json.dumps(altered); cache.write_text('\n'.join(data)+'\n')
        try:
            diagnose(trace, cache, scene, spec, ground, source, 364, 'staged')
        except ValueError as error:
            assert 'provenance differs' in str(error)
        else:
            raise AssertionError('Changed candidate was accepted')
    print('Known MISS counts and all three attribution categories: OK')
    print('IoU=0.5, GT/local ID zero, empty frames and competing GT: OK')
    print('Deterministic examples and changed candidate provenance rejection: OK')
    print('Frozen miss diagnostic checks: PASSED')


if __name__ == '__main__':
    main()
