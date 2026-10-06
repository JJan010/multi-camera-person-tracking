"""Known-answer tests of partial association; all scores/thresholds are synthetic."""

from dataclasses import replace
from fractions import Fraction
from itertools import product

import numpy as np

from mtmc.association.pairwise import CameraAppearance, associate_camera_pair, solve_partial_assignment
from mtmc.reid.osnet import ObservationKey, ReIDBatch


def reject(call):
    try:
        call()
    except (ValueError, TypeError):
        return
    raise AssertionError("Invalid contract was accepted")


def camera(camera_id, ids, axes):
    features = np.zeros((len(ids), 512), dtype=np.float32)
    for i, axis in enumerate(axes):
        features[i, axis] = 1
    batch = ReIDBatch(tuple(ObservationKey(camera_id, i, 30) for i in ids),
                      (Fraction(1),) * len(ids), features)
    return CameraAppearance("synthetic-session", camera_id, 30, Fraction(1), "mean", batch)


def associations(result):
    return {(p.left.local_id, p.right.local_id) for p in result.matches}


def brute_force_gain(scores, threshold):
    n, m = scores.shape
    best = 0.0
    for choices in product(range(-1, m), repeat=n):
        columns = [j for j in choices if j >= 0]
        if len(columns) != len(set(columns)):
            continue
        if any(j >= 0 and scores[i, j] <= threshold for i, j in enumerate(choices)):
            continue
        best = max(best, sum(scores[i, j] - threshold for i, j in enumerate(choices) if j >= 0))
    return best


def main():
    threshold = .8  # Illustrative only. NOT calibrated on OSNet or the integration clip.
    match = lambda scores: solve_partial_assignment(np.asarray(scores), min_similarity=threshold)
    assert match([[.95, .94], [.93, .10]]) == ((0, 1), (1, 0))
    print("Joint assignment resolves competing first choices: OK")
    assert match([[.81, .79], [.79, .0]]) == ((0, 0),)
    print("Threshold applied before assignment preserves the valid pair: OK")
    assert match([[.95, .81], [.81, .0]]) == ((0, 0),)
    assert match([[.8, .1], [.2, .79]]) == ()
    assert match(np.zeros((0, 2))) == match(np.zeros((2, 0))) == ()
    print("Unmatched option, strict threshold boundary and empty sets: OK")

    rng = np.random.default_rng(123)
    for n in range(1, 4):
        for m in range(1, 4):
            for _ in range(10):
                scores = rng.uniform(-1, 1, (n, m))
                pairs = solve_partial_assignment(scores, min_similarity=.1)
                actual = sum(scores[i, j] - .1 for i, j in pairs)
                assert np.isclose(actual, brute_force_gain(scores, .1), rtol=0, atol=1e-12)
    print("Partial-assignment objective matches exhaustive small-case solutions: OK")

    a, b = camera(4, [10, 20, 30], [0, 1, 2]), camera(5, [7, 8], [1, 0])
    result = associate_camera_pair(a, b, min_similarity=threshold)
    assert associations(result) == {(10, 8), (20, 7)}
    assert result.unmatched_left[0].key.local_id == 30
    assert result.unmatched_left[0].reason == "no_candidate_above_threshold"
    assert not result.unmatched_right
    changed = camera(4, [30, 20, 10], [2, 1, 0])
    assert associate_camera_pair(changed, b, min_similarity=threshold) == result
    reverse = associate_camera_pair(b, changed, min_similarity=threshold)
    assert associations(reverse) == {(y, x) for x, y in associations(result)}
    tied_a, tied_b = camera(4, [20, 10], [0, 0]), camera(5, [8, 7], [0, 0])
    tied = associate_camera_pair(tied_a, tied_b, min_similarity=threshold)
    assert tied == associate_camera_pair(camera(4, [10, 20], [0, 0]), camera(5, [7, 8], [0, 0]),
                                         min_similarity=threshold)
    assert {(x, y) for y, x in associations(associate_camera_pair(tied_b, tied_a, min_similarity=threshold))} == associations(tied)
    competition = associate_camera_pair(tied_a, camera(5, [1], [0]), min_similarity=threshold)
    assert len(competition.matches) == 1 and competition.unmatched_left[0].reason == "assignment_competition"
    empty = associate_camera_pair(a, camera(5, [], []), min_similarity=threshold)
    assert len(empty.unmatched_left) == 3
    assert all(x.reason == "empty_opposite_camera" for x in empty.unmatched_left)
    assert not associate_camera_pair(camera(4, [], []), camera(5, [], []), min_similarity=threshold).matches
    print("Camera/local-ID mapping, order invariance and unmatched reasons: OK")

    reject(lambda: associate_camera_pair(a, a, min_similarity=threshold))
    for field, value in (("run_id", "other-session"), ("frame_index", 31),
                         ("timestamp", Fraction(2)), ("descriptor_variant", "latest")):
        reject(lambda field=field, value=value: associate_camera_pair(a, replace(b, **{field: value}),
                                                                      min_similarity=threshold))
    reject(lambda: associate_camera_pair(camera(4, [1, 1], [0, 1]), b, min_similarity=threshold))
    bad = camera(4, [1], [0])
    bad.observations.embeddings[0] *= 2
    reject(lambda: associate_camera_pair(bad, b, min_similarity=threshold))
    for value in (float("nan"), 1.01, True):
        reject(lambda value=value: solve_partial_assignment([[.9]], min_similarity=value))
    reject(lambda: solve_partial_assignment([[float("nan")]], min_similarity=threshold))
    print("Mixed sessions/times/variants, duplicate IDs and invalid descriptors rejected: OK")
    print("Synthetic example: camera 4 ID 10 <-> camera 5 ID 8; camera 4 ID 20 <-> camera 5 ID 7")
    print("Thresholds in this test are synthetic examples, not deployment settings.")
    print("Pairwise association smoke test: PASSED")


if __name__ == "__main__":
    main()
