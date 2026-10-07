"""Check exact candidate joins, missing descriptors and global lifecycle accounting."""
from collections import Counter
import copy
import numpy as np
from cache_detection_embeddings import prepare_candidates
from check_detection_embedding_cache import fixture
from evaluate_appearance_global import direct_observations, lifecycle


def rejected(callback):
    try:
        callback()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('Invalid provenance accepted')


def main():
    local, batch = fixture()
    for c in local['cameras']:
        c['detector_xyxy'] = np.asarray(c['detector_xyxy'], np.float32).reshape(-1, 4).tolist()
        c['detector_confidence'] = np.asarray(c['detector_confidence'], np.float32).tolist()
    _, records = prepare_candidates(local, batch, source_run='fixture', row_offset=0)
    by_key = {(r['camera_id'], r['detection_index']): r for r in records}
    cameras = []
    for camera, indices, ids in ((8, [0], [3]), (5, [], []), (4, [2, 0, 1], [10, 3, 12])):
        selected = [by_key[camera, i] for i in indices]
        cameras.append({'camera': camera, 'local_ids': ids, 'detection_indices': indices,
                        'xyxy': [r['xyxy'] for r in selected], 'confidence': [r['confidence'] for r in selected],
                        'embedding_rows': [r['embedding_row'] for r in selected]})
    vectors = np.eye(3, 512, dtype=np.float32)
    saved = {'detections': records}
    items, features = direct_observations(cameras, saved, vectors, 0)
    assert [(r.key.camera_id, r.key.local_id) for r in items] == [(4, 3), (4, 10), (4, 12), (8, 3)]
    assert items[2].crop_xyxy_int is None
    assert [(k.camera_id, k.local_id) for k in features.keys] == [(4, 3), (4, 10), (8, 3)]
    assert np.array_equal(features.embeddings, vectors)
    reordered, vectors2 = direct_observations(list(reversed(cameras)), saved, vectors, 0)
    assert reordered == items and np.array_equal(vectors2.embeddings, features.embeddings)
    print('Candidate-index join, duplicate boxes, camera-scoped IDs, order and unavailable descriptors: OK')
    for name, value in [('embedding_rows', [0, 0, None]), ('detection_indices', [0, 0, 1])]:
        damaged = copy.deepcopy(cameras); damaged[2][name] = value
        rejected(lambda: direct_observations(damaged, saved, vectors, 0))
    damaged = copy.deepcopy(cameras); damaged[2]['xyxy'][0][0] += 1
    rejected(lambda: direct_observations(damaged, saved, vectors, 0))
    rejected(lambda: direct_observations(cameras, saved, vectors, 1))
    print('Wrong feature row, repeated candidate, changed box and wrong frame rejected: OK')
    counts, emitted = Counter(), set()
    record = {'assignments': [{'global_id': 1}, {'global_id': 1}],
              'base_assignments': [{'global_id': 1, 'reason': 'new_identity'},
                                   {'global_id': 1, 'reason': 'new_identity'},
                                   {'global_id': 2, 'reason': 'new_identity'}],
              'merge_events': [{'absorbed_global_ids': [2]}], 'expired_global_ids': [], 'identities': [{}]}
    lifecycle(record, counts, emitted)
    assert counts['allocated_ids'] == 2 and counts['absorbed_ids'] == 1 and counts['retained_ids_at_end'] == 1
    assert emitted == {1}
    print('Allocated, absorbed, expired and retained identity accounting: OK')
    print('Local-to-global input checks: PASSED')


if __name__ == '__main__':
    main()
