"""Known-answer checks of pairwise metrics and three-camera connectivity audit."""

from fractions import Fraction

import numpy as np

import evaluate_pairwise_association as evaluation
from mtmc.association.pairwise import associate_camera_pair
from mtmc.reid.osnet import ObservationKey


def main():
    labels = [
        {"camera": 4, "local_id": 10, "frame_index": 2, "embedding_row": 0, "gt_id": 1},
        {"camera": 4, "local_id": 20, "frame_index": 2, "embedding_row": 1, "gt_id": 2},
        {"camera": 4, "local_id": 30, "frame_index": 2, "embedding_row": 2, "gt_id": 3},
        {"camera": 4, "local_id": 40, "frame_index": 2, "embedding_row": 3, "gt_id": None},
        {"camera": 5, "local_id": 7, "frame_index": 2, "embedding_row": 4, "gt_id": 1},
        {"camera": 5, "local_id": 8, "frame_index": 2, "embedding_row": 5, "gt_id": 4},
        {"camera": 5, "local_id": 9, "frame_index": 2, "embedding_row": 6, "gt_id": None},
    ]
    features = np.zeros((7, 512), dtype=np.float32)
    for row, axis in enumerate([0, 1, 2, 3, 0, 1, 2]):
        features[row, axis] = 1
    lookup = {evaluation.key_of(r): r for r in labels}
    a = evaluation.make_camera("test", 4, 2, "mean", labels, features)
    b = evaluation.make_camera("test", 5, 2, "mean", labels, features)
    assert a.timestamp == Fraction(1, 15)
    result = associate_camera_pair(a, b, min_similarity=.8)
    counts, decisions, unmatched = evaluation.score_pair(result, lookup)
    assert counts["accepted_links"] == 3
    assert counts["correct_links"] == counts["wrong_known_links"] == counts["unresolved_gt_links"] == 1
    assert counts["available_positive_pairs"] == 1 and counts["missed_positive_pairs"] == 0
    assert counts["endpoints"] == 7 and counts["unmatched_endpoints"] == 1
    assert counts["no_positive_endpoints"] == counts["no_positive_linked"] == 3
    assert counts["unlabeled_endpoints"] == 2 and counts["unlabeled_unmatched"] == 1
    summary = evaluation.summarize(counts)
    assert summary["precision_labeled"] == .5
    assert summary["verified_correct_fraction_all_links"] == 1 / 3
    assert summary["accepted_link_label_coverage"] == 2 / 3
    assert summary["recall_available_pairs"] == 1
    assert summary["no_positive_abstention_rate"] == 0
    assert {d["outcome"] for d in decisions} == {"correct", "wrong_known", "unresolved_gt"}
    assert unmatched[0]["embedding_row"] == 3
    print("Correct/wrong/unresolved links, endpoint counts and metric denominators: OK")

    rejected = associate_camera_pair(a, b, min_similarity=1)
    none, _, _ = evaluation.score_pair(rejected, lookup)
    summary = evaluation.summarize(none)
    assert none["accepted_links"] == 0 and none["missed_positive_pairs"] == 1
    assert summary["precision_labeled"] is None and summary["recall_available_pairs"] == 0
    assert summary["no_positive_abstention_rate"] == 1
    empty = evaluation.make_camera("test", 8, 2, "mean", [], features)
    empty_counts, _, _ = evaluation.score_pair(associate_camera_pair(a, empty, min_similarity=.8), lookup)
    assert empty_counts["available_positive_pairs"] == 0
    assert evaluation.summarize(empty_counts)["recall_available_pairs"] is None
    assert empty_counts["unmatched_empty_opposite_camera"] == 4
    print("Reject-all and empty-gallery controls use zero/undefined metrics correctly: OK")

    a1, a2, b1, c1 = [ObservationKey(camera, identity, 2) for camera, identity in ((4, 1), (4, 2), (5, 1), (8, 1))]
    graph_labels = {k: {"gt_id": 1 if k != a2 else 2, "embedding_row": i}
                    for i, k in enumerate((a1, a2, b1, c1))}
    open_counts, issues = evaluation.audit_components([(a1, b1), (b1, c1)], graph_labels)
    assert open_counts["open_three_camera_components"] == 1 and open_counts["camera_conflict_components"] == 0
    assert issues[0]["open_three_camera"] and not issues[0]["camera_conflict"]
    closed, issues = evaluation.audit_components([(a1, b1), (b1, c1), (a1, c1)], graph_labels)
    assert closed["closed_three_camera_components"] == 1 and not issues
    conflict, issues = evaluation.audit_components([(a1, b1), (b1, c1), (c1, a2)], graph_labels)
    assert conflict["camera_conflict_components"] == conflict["frames_with_camera_conflict"] == 1
    assert conflict["components_with_different_known_gt"] == 1
    assert len(issues[0]["members"]) == 4
    nothing, issues = evaluation.audit_components([], graph_labels)
    assert nothing["linked_components"] == 0 and nothing["frames"] == 1 and not issues
    print("Open chains, closed triangles and duplicate-camera components distinguished: OK")
    print("Pairwise evaluation checks: PASSED")


if __name__ == "__main__":
    main()
