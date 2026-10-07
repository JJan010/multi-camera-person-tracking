"""Known-answer checks for cache validation and exact local tracking comparison."""
import copy
from types import SimpleNamespace as NS
import numpy as np

from check_detection_embedding_cache import fixture
from cache_detection_embeddings import prepare_candidates
from check_frozen_bytetrack_replay import candidate_arrays, compare_outputs


def rejected(callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid input accepted')


def main():
    local, batch = fixture()
    for camera in local['cameras']:
        camera['detector_xyxy'] = np.asarray(camera['detector_xyxy'], dtype=np.float32).reshape(-1, 4).tolist()
        camera['detector_confidence'] = np.asarray(camera['detector_confidence'], dtype=np.float32).tolist()
    _, records = prepare_candidates(local, batch, source_run='fixture', row_offset=0)
    cached = {'cache_run_id': 'cache', 'source_run_id': 'fixture', 'frame_index': 0,
              'timestamp': '0', 'detections': records}
    def validate(value):
        return candidate_arrays(value, local, frame=0, cache_run='cache', source_run='fixture', next_row=0)
    arrays, _, count, outside = validate(cached)
    assert count == 3 and outside == 1 and arrays[4][2] == (0, None, 1) and arrays[8][2] == (2,)
    assert arrays[5][0].shape == (0, 4) and arrays[4][0].dtype == arrays[4][1].dtype == np.float32
    reordered = copy.deepcopy(cached); reordered['detections'].reverse()
    reordered_arrays, _, _, _ = validate(reordered)
    assert np.array_equal(arrays[4][0], reordered_arrays[4][0])
    print('Candidate key mapping, original float32 values, empty camera and unavailable features: OK')
    for field, value in [('confidence', .123), ('embedding_row', 2), ('frame_index', 1),
                         ('crop_xyxy_int', [0, 0, 1, 1]), ('inside_image_fraction', -.5)]:
        damaged = copy.deepcopy(cached);damaged['detections'][0][field] = value
        rejected(lambda: validate(damaged))
    damaged = copy.deepcopy(cached);damaged['detections'].append(copy.deepcopy(damaged['detections'][0]))
    rejected(lambda: validate(damaged))
    damaged = copy.deepcopy(cached);damaged['source_run_id'] = 'wrong'
    rejected(lambda: validate(damaged))
    print('Altered score, row, frame, crop, duplicate key and mixed run rejected: OK')
    expected = {'local_ids': [1, 2], 'xyxy': [[1, 2, 3, 4], [4, 5, 6, 7]], 'confidence': [.5, .75]}
    actual = NS(tracker_id=np.array([1, 2]), xyxy=np.array(expected['xyxy'], np.float32),
                confidence=np.array(expected['confidence'], np.float32))
    assert compare_outputs(actual, expected)['passed']
    actual.tracker_id = actual.tracker_id[::-1]
    assert not compare_outputs(actual, expected)['ids_equal']
    actual.tracker_id = actual.tracker_id[::-1]
    actual.xyxy[0, 0] = np.nextafter(np.float32(1), np.float32(2))
    assert not compare_outputs(actual, expected)['boxes_equal']
    actual.xyxy = np.array(expected['xyxy'], np.float32)
    actual.confidence[0] = .25
    assert not compare_outputs(actual, expected)['scores_equal']
    assert compare_outputs(NS(tracker_id=np.array([], int), xyxy=np.empty((0, 4)), confidence=np.array([])),
                           {'local_ids': [], 'xyxy': [], 'confidence': []})['passed']
    print('Exact outputs pass; changed IDs, one-ULP boxes and scores fail; empty outputs pass: OK')
    print('Frozen tracking contract checks: PASSED')


if __name__ == '__main__':
    main()
