"""Known-answer checks for frozen decision reconstruction and grouping metrics."""

from copy import deepcopy
from fractions import Fraction
from io import StringIO
from itertools import combinations

import evaluate_multicamera_grouping as evaluation
from mtmc.association import pairwise
from mtmc.reid.osnet import ObservationKey

CAMERAS = (4, 5, 8)
A, B, C, D = (ObservationKey(c, i, 2) for c, i in ((4, 1), (5, 1), (8, 1), (4, 2)))


def fixture(gt, edges):
    labels = {key: {"camera": key.camera_id, "local_id": key.local_id, "frame_index": 2,
                    "embedding_row": row, "gt_id": identity} for row, (key, identity) in enumerate(gt.items())}
    records = []
    for a, b in combinations(CAMERAS, 2):
        matches = tuple(pairwise.PairMatch(x, y, score) for x, y, score in edges
                        if (x.camera_id, y.camera_id) == (a, b))
        used = {key for match in matches for key in (match.left, match.right)}
        unmatched = {c: tuple(pairwise.UnmatchedObservation(k, "no_candidate_above_threshold", 0.1)
                              for k in labels if k.camera_id == c and k not in used) for c in (a, b)}
        result = pairwise.PairAssociation("test", 2, Fraction(2, 30), "mean", a, b, 0.5,
                                         matches, unmatched[a], unmatched[b])
        counts, saved_matches, saved_unmatched = evaluation.baseline.score_pair(result, labels)
        records.append({"run_id": "test", "frame_index": 2, "timestamp": "1/15", "variant": "mean",
                        "threshold": 0.5, "camera_a": a, "camera_b": b, "counts": dict(counts),
                        "matches": saved_matches, "unmatched": saved_unmatched})
    return records


def run(records):
    pairs, labels, groups, before, _ = evaluation.reconstruct(
        records, run_id="test", frame=2, variant="mean", threshold=0.5, cameras=CAMERAS, fps=30)
    after, removed, counts, _ = evaluation.evaluate_groups(pairs, labels, groups)
    return before, after, removed, counts, groups


def rejects(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def main():
    triangle = fixture({A: 7, B: 7, C: 7}, [(A, B, 0.9), (A, C, 0.8), (B, C, 0.7)])
    before, after, removed, groups, _ = run(triangle)
    assert before["available_positive_pairs"] == after["correct_links"] == 3
    assert removed["accepted_links"] == 0 and groups["three_member_groups"] == 1
    assert groups["merged"] == 2 and groups["already_in_same_group"] == 1
    print("Closed true triangle: three retained pairs, one group: OK")

    chain = fixture({A: 7, B: 7, C: 7}, [(A, B, 0.9), (B, C, 0.8)])
    before, after, removed, groups, _ = run(chain)
    assert before["correct_links"] == 2 and after["correct_links"] == removed["correct_links"] == 1
    assert before["available_positive_pairs"] == 3
    assert evaluation.rates(after, 3)["recall_available_pairs"] == 1 / 3
    assert groups["singletons"] == groups["two_member_groups"] == groups["rejected_missing_pair_support"] == 1
    print("True open chain loses one correct link; recall denominator remains three: OK")

    conflict = fixture({A: 1, B: 1, C: 2, D: 2}, [(A, B, 0.9), (B, C, 0.8), (D, C, 0.95)])
    before, after, removed, groups, _ = run(conflict)
    assert before["correct_links"] == after["correct_links"] == 2
    assert removed["wrong_known_links"] == groups["rejected_camera_conflict"] == 1
    assert groups["observations"] == 4 and groups["two_member_groups"] == 2
    print("Camera conflict removes a wrong link and preserves all four observations: OK")

    unknown = fixture({A: 1, B: None, C: 2}, [(A, B, 0.9), (A, C, 0.8), (B, C, 0.7)])
    _, after, _, groups, _ = run(unknown)
    assert after["unresolved_gt_links"] == 2 and after["wrong_known_links"] == 1
    assert groups["linked_groups_different_known_gt"] == 1
    assert groups["linked_groups_unresolved_gt"] == 0
    _, after, _, groups, _ = run(fixture({A: 1, B: None}, [(A, B, 0.9)]))
    assert after["unresolved_gt_links"] == 1 and evaluation.rates(after, 0)["precision_labeled"] is None
    assert groups["linked_groups_unresolved_gt"] == 1
    print("Unknown GT stays unresolved; known identity conflict is never hidden by it: OK")

    before, after, _, groups, _ = run(fixture({A: 1, B: 1, C: 1}, []))
    assert groups["singletons"] == 3 and after["accepted_links"] == 0
    rates = evaluation.rates(after, before["available_positive_pairs"])
    assert rates["precision_labeled"] is None and rates["recall_available_pairs"] == 0
    assert groups["linked_groups_all_known_same_gt"] == 0
    _, _, _, groups, _ = run(fixture({}, []))
    assert groups["observations"] == groups["groups"] == 0
    print("Reject-all and empty rounds: singletons are not counted as identity successes: OK")

    damaged = deepcopy(triangle)
    damaged[1]["matches"][0]["left_embedding_row"] = 99
    rejects(lambda: run(damaged))
    damaged = deepcopy(triangle)
    damaged[0]["matches"][0]["outcome"] = "wrong_known"
    rejects(lambda: run(damaged))
    damaged = deepcopy(triangle)
    damaged[0]["counts"]["correct_links"] += 1
    rejects(lambda: run(damaged))
    damaged = deepcopy(triangle)
    damaged[0]["timestamp"] = "1/30"
    rejects(lambda: run(damaged))
    rejects(lambda: run(triangle[::-1]))
    rejects(lambda: run(triangle[:2]))
    rejects(lambda: evaluation.read_round(StringIO('{}\n'), 3))
    print("Inconsistent source rows, labels/outcomes, counts, time, pair order and truncation rejected: OK")
    print("Grouping evaluation checks: PASSED")


if __name__ == "__main__":
    main()
