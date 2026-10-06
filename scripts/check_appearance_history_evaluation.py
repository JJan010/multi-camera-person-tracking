"""Known-answer CPU checks for the paired appearance-history evaluation."""

from copy import deepcopy

import numpy as np

import evaluate_appearance_history as evaluation


def must_reject(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Expected rejection of invalid history provenance")


def main():
    evaluation.snapshot.check_protocol()
    labels = [
        {"camera": 4, "local_id": 1, "gt_id": 10, "embedding_row": 100},
        {"camera": 4, "local_id": 2, "gt_id": 20, "embedding_row": 101},
        {"camera": 4, "local_id": 3, "gt_id": None, "embedding_row": 102},
        {"camera": 4, "local_id": 4, "gt_id": 30, "embedding_row": 103},
        {"camera": 5, "local_id": 1, "gt_id": None, "embedding_row": 200},
        {"camera": 5, "local_id": 2, "gt_id": 10, "embedding_row": 201},
        {"camera": 5, "local_id": 3, "gt_id": 20, "embedding_row": 202},
    ]
    for label in labels:
        label["history_gt_status"] = "consistent_with_current"
    left, right = np.zeros((7, 7)), np.zeros((7, 7))
    left[0, 4:], right[0, 4:] = [.9, .8, .1], [.8, .9, .1]  # Improved.
    left[1, 4:], right[1, 4:] = [.1, .2, .9], [.9, .1, .8]  # Worsened.
    rows = evaluation.compare_direction(labels, left, right, 4, 5, 2)
    assert rows[0]["query_embedding_row"] == 100 and rows[0]["positive_embedding_row"] == 201
    assert rows[0]["latest_positive_rank"] == 2 and rows[0]["mean_positive_rank"] == 1
    assert rows[2]["status"] == "unmatched_query" and rows[3]["status"] == "no_positive_in_gallery"
    summary = evaluation.paired_summary(rows)
    assert summary["eligible_queries"] == 2 and summary["queries"] == 4
    assert summary["latest"]["rank1"] == summary["mean"]["rank1"] == .5
    assert summary["latest"]["rank3"] == summary["mean"]["rank3"] == 1
    assert summary["rank1_transitions"] == dict(both_correct=0, improved=1, worsened=1, both_wrong=0)
    assert summary["rank1_delta_percentage_points"] == 0
    assert evaluation.paired_summary(rows[:1])["rank1_delta_percentage_points"] == 100
    assert evaluation.paired_summary(rows[1:2])["rank1_delta_percentage_points"] == -100
    ties = evaluation.compare_direction(labels, np.zeros((7, 7)), np.zeros((7, 7)), 4, 5, 2)
    assert ties[0]["latest_top1_local_id"] == 1 and ties[0]["latest_positive_rank"] == 2
    empty = evaluation.compare_direction(labels, left, right, 4, 8, 2)
    assert evaluation.paired_summary(empty)["eligible_queries"] == 0
    assert evaluation.paired_summary([])["latest"]["rank1"] is None
    print("Paired eligibility, distractors, source-row mapping, ties and transitions: OK")

    status = evaluation.history_label_status
    assert status(10, [10, 10])["history_gt_status"] == "consistent_with_current"
    assert status(10, [None, 10])["history_gt_status"] == "unknown_members"
    assert status(10, [20, None, 10])["history_gt_status"] == "conflict_with_current"
    assert status(None, [10, None])["history_gt_status"] == "current_unmatched"
    assert status(10, [20, None, 10])["known_different_members"] == 1
    assert status(10, [20, None, 10])["distinct_known_gt_ids"] == 2
    assert evaluation.history_pair_group(dict(query_history_status="consistent_with_current",
                                             positive_history_status="conflict_with_current")) == "query_or_positive_conflict"
    assert evaluation.history_pair_group(dict(query_history_status="unknown_members",
                                             positive_history_status="consistent_with_current")) == "unknown_members"
    print("Mixed histories and unknown GT members remain explicit: OK")

    observations = [{"camera": 4, "local_id": 1, "frame_index": i, "embedding_row": i} for i in range(3)]
    features = np.zeros((3, 512), dtype=np.float32)
    features[0, 0], features[1, 1], features[2, 2] = 1, 1, 1
    histories = []
    means = []
    for i in range(3):
        selected = list(range(max(0, i - 1), i + 1))
        histories.append({**observations[i], "run_id": "test", "timestamp": str(evaluation.Fraction(i, 30)),
                          "source_embedding_rows": selected, "source_frames": selected,
                          "history_size": len(selected), "used_latest_fallback": False})
        mean = features[selected].mean(axis=0)
        means.append(mean / np.linalg.norm(mean))
    means = np.asarray(means)
    config = {"max_observations": 2, "max_age_seconds": "1"}

    def verify(items, matrix=means):
        evaluation.validate_history_rows(items, observations, features, matrix, "test", config)

    verify(histories)
    changed = deepcopy(histories)
    changed[1]["source_embedding_rows"] = [0, 2]
    must_reject(lambda: verify(changed))
    changed = deepcopy(histories)
    changed[2]["camera"] = 5
    must_reject(lambda: verify(changed))
    changed = deepcopy(histories)
    changed[2].update(source_embedding_rows=[2], source_frames=[2], history_size=1)
    must_reject(lambda: verify(changed))
    must_reject(lambda: verify(histories, features))
    print("Causal window, cross-camera mapping and reconstructed means: OK")
    print("Appearance history evaluation checks: PASSED")


if __name__ == "__main__":
    main()
