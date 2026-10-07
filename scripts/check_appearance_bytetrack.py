"""Synthetic checks for the isolated appearance-assisted ByteTrack experiment."""
import numpy as np
import supervision as sv
from mtmc.tracking.appearance_track import gate_costs
from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack

SETTINGS = dict(track_activation_threshold=.5, lost_track_buffer=30,
                minimum_matching_threshold=.8, frame_rate=30, minimum_consecutive_frames=1)


def main():
    a = np.zeros(512, np.float32); a[0] = 1
    b = np.zeros(512, np.float32); b[1] = 1
    cost = np.array([[.1, .2]], np.float32)
    masked, counters = gate_costs(cost, [a], [b, a], .8, .6)
    assert np.isinf(masked[0, 0]) and masked[0, 1] == cost[0, 1] and np.isfinite(cost).all()
    assert counters == dict(compared=2, blocked=1, unavailable=0)
    assert gate_costs(cost, [a], [b, a], .8, 0)[0][0, 0] == cost[0, 0]
    assert np.array_equal(gate_costs(cost, [None], [b, a], .8, .6)[0], cost)
    assert gate_costs(np.empty((0, 1), np.float32), [], [a], .8, .6)[0].shape == (0, 1)
    print('Pre-assignment mask, inclusive cosine boundary, missing-feature fallback and empty sets: OK')
    direct = ExperimentalByteTrack(**SETTINGS, direct_output=True)
    guarded = ExperimentalByteTrack(**SETTINGS, direct_output=True, appearance_threshold=.6)
    box = np.array([[0, 0, 10, 20]], np.float32)
    score = np.array([.9], np.float32)
    for tracker in (direct, guarded):
        assert tracker.update_candidates(box, score, [a]).tracker_id.tolist() == [1]
    boxes = np.array([[0, 0, 10, 20], [1, 0, 11, 20]], np.float32)
    scores = np.array([.9, .2], np.float32)
    wrong = direct.update_candidates(boxes, scores, [b, a])
    recovered = guarded.update_candidates(boxes, scores, [b, a])
    assert wrong.tracker_id.tolist() == recovered.tracker_id.tolist() == [1]
    assert direct.selected_indices == (0,) and guarded.selected_indices == (1,)
    assert np.array_equal(recovered.xyxy, boxes[[1]])
    original = next(t for t in guarded.tracked_tracks if t.external_track_id == 1)
    assert len(original.appearance_history) == 1 and np.array_equal(original.descriptor(2), a)
    assert original.descriptor(31) is not None and original.descriptor(32) is None
    assert guarded.statistics['high_blocked'] == 1
    print('Wrong strong candidate blocked; correct weak candidate recovers ID; direct output keeps its index: OK')
    print('Weak observations do not refresh appearance; inclusive 30-frame age limit: OK')
    resolved = guarded.update_candidates(boxes, np.array([.9, .9], np.float32), [b, a])
    assert guarded.selected_indices == (0, 1) and resolved.tracker_id.tolist() == [2, 1]
    original = next(t for t in guarded.tracked_tracks if t.external_track_id == 1)
    assert np.array_equal(original.descriptor(3), a)
    print('Next round preserves the recovered ID and confirms a separate ID for the other person: OK')
    frame_before = guarded.frame_id
    try:
        guarded.update_candidates(boxes, scores, [np.zeros(512, np.float32), a])
    except ValueError:
        pass
    else:
        raise AssertionError('Zero feature accepted')
    assert guarded.frame_id == frame_before and not guarded.failed
    print('Malformed features rejected before tracker state mutation: OK')
    # Exact fork control over a changing sequence: births, gaps, low scores, empties.
    installed, control = sv.ByteTrack(**SETTINGS), ExperimentalByteTrack(**SETTINGS)
    rng = np.random.default_rng(8107)
    for f in range(120):
        if f % 17 == 0:
            boxes, scores = np.empty((0, 4), np.float32), np.empty(0, np.float32)
        else:
            centers = np.array([[30 + f / 2, 40], [100 - f / 4, 80], [150, 100]], np.float32)
            centers += rng.normal(0, .3, centers.shape).astype(np.float32)
            boxes = np.concatenate([centers, centers + [20, 40]], axis=1).astype(np.float32)
            scores = np.array([.9, .2 if f % 3 == 0 else .8, .9], np.float32)
        original = installed.update_with_detections(sv.Detections(xyxy=boxes.copy(), confidence=scores.copy(),
                                                                class_id=np.zeros(len(scores), int)))
        fork = control.update_candidates(boxes, scores, [a] * len(scores))
        assert np.array_equal(original.tracker_id, fork.tracker_id)
        assert np.array_equal(original.xyxy, fork.xyxy) and np.array_equal(original.confidence, fork.confidence)
    print('Experimental fork with changes disabled matches installed tracker on synthetic sequence: EXACT')
    print('Appearance-assisted ByteTrack checks: PASSED; synthetic threshold is not calibrated')


if __name__ == '__main__':
    main()
