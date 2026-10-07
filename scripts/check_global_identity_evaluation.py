"""Known-answer and motmetrics parity checks for a shared MTMC ID assignment."""

import numpy as np
from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference, history


def evaluate(slots):
    counts = IdentityCounts()
    normalized = []
    for ground, predicted, mask in slots:
        mask = np.asarray(mask, dtype=bool).reshape(len(ground), len(predicted))
        counts.update(ground, predicted, mask)
        normalized.append((ground, predicted, mask))
    metrics, _ = counts.result()
    check_reference(metrics, reference_metrics(normalized))
    return metrics


def main():
    cases = [
        ("same_ID_across_cameras", [([1], [10], [[1]]), ([1], [10], [[1]])], (2, 0, 0), 1.0),
        ("different_ID_per_camera", [([1], [10], [[1]]), ([1], [20], [[1]])], (1, 1, 1), 0.5),
        ("two_people_merged_across_cameras", [([1], [10], [[1]]), ([2], [10], [[1]])], (1, 1, 1), 0.5),
        ("identity_changes_over_time", [([1], [10], [[1]]), ([1], [20], [[1]])], (1, 1, 1), 0.5),
        ("missed_person", [([1], [10], [[1]]), ([1], [], [[]])], (1, 0, 1), 2 / 3),
        ("false_observation", [([1], [10, 20], [[1, 0]])], (1, 1, 0), 2 / 3),
        ("no_spatial_overlap", [([1], [10], [[0]])], (0, 1, 1), 0.0),
        ("all_spatial_candidates_retained", [([1], [10, 20], [[1, 1]]), ([1], [20], [[1]])], (2, 1, 0), 0.8),
        ("no_GT", [([], [10], [])], (0, 1, 0), 0.0),
        ("no_predictions", [([1], [], [[]])], (0, 0, 1), 0.0),
        ("both_empty", [([], [], [])], (0, 0, 0), None),
    ]
    for name, slots, expected, idf1 in cases:
        result = evaluate(slots)
        assert tuple(result[k] for k in ("idtp", "idfp", "idfn")) == expected, name
        assert result["idf1"] is None if idf1 is None else np.isclose(result["idf1"], idf1)
        print(f"{name}: known answer and motmetrics agreement: OK")
    # Perfect individual-camera evaluations are not a correct global average.
    local = [evaluate([([1], [gid], [[1]])])["idf1"] for gid in (10, 20)]
    assert local == [1, 1] and evaluate(cases[1][1])["idf1"] == 0.5
    print("Per-camera 100% + 100% can mean global IDF1=50%: DEMONSTRATED")
    gt = history.snapshot.clip_boxes([[0, 0, 10, 10]], 1920, 1080)
    pred = history.snapshot.clip_boxes([[0, 0, 5, 10], [0, 0, 4, 10], [2000, 0, 2010, 10]], 1920, 1080)
    np.testing.assert_array_equal(history.snapshot.pairwise_iou(gt, pred) >= 0.5, [[True, False, False]])
    print("Inclusive IoU=0.5 gate and fully outside prediction: OK")
    counter = IdentityCounts()
    try:
        counter.update([1], [10, 10], np.ones((1, 2), dtype=bool))
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate identity accepted within a camera/time slot")
    assert counter.slots == 0 and not counter.ground and not counter.predicted
    rng = np.random.default_rng(61007)
    for _ in range(30):
        slots = []
        for _ in range(6):
            g = [int(x) for x in rng.permutation(3)[:int(rng.integers(4))]]
            p = [int(x + 10) for x in rng.permutation(4)[:int(rng.integers(5))]]
            slots.append((g, p, rng.random((len(g), len(p))) < 0.4))
        evaluate(slots)
    print("Random ambiguous and empty-slot cases match motmetrics: OK")
    print("Global identity evaluation checks: PASSED")


if __name__ == "__main__":
    main()
