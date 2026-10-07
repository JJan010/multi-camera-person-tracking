"""Checks for the saved-group boundary and identity trace serialization."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import json

from check_global_identity import frame, manager, A, B, C, rejects
import replay_global_identity as replay


def saved(number, groups):
    source = frame(number, groups)
    row = asdict(source)
    row["timestamp"] = str(source.timestamp)
    row["variant"] = row.pop("descriptor_variant")
    row["threshold"] = row.pop("min_similarity")
    index = 0
    for group in row["groups"]:
        for key in group:
            key.update(embedding_row=index, diagnostic_gt_id=7)
            index += 1
    return row


def decode(row):
    return replay.decode_round(row, run_id="synthetic-global-check", frame=0,
                               variant="mean", threshold=0.5, fps=30, cameras=(4, 5, 8))


def main():
    row = saved(0, [(A, B, C)])
    runtime, labels = decode(row)
    counts, edges = replay.source_counts(row, runtime, labels)
    assert counts["three_member_groups"] == 1 and edges["correct_links"] == 3
    changed = deepcopy(row)
    for index, group in enumerate(changed["groups"]):
        for member_index, member in enumerate(group):
            member["diagnostic_gt_id"] = None if member_index == 0 else 100 + member_index
    other, other_labels = decode(changed)
    assert runtime == other
    output = manager().update(runtime)
    assert output == manager().update(other)
    print("Changing diagnostic GT labels leaves runtime input and global assignments unchanged: OK")
    result = replay.encode_output(output, labels, "test/mean/tau=0.5/idle=1")
    restored = json.loads(json.dumps(result, default=replay.json_default, allow_nan=False))
    assert restored["timestamp"] == "0" and restored["identities"][0]["last_seen"] == "0"
    assert [a["embedding_row"] for a in restored["assignments"]] == [0, 1, 2]
    assert {a["global_id"] for a in restored["assignments"]} == {1}
    assert all(a["diagnostic_gt_id"] == 7 for a in restored["assignments"])
    print("JSON trace preserves exact times, observation keys, source rows and explicit identity scope: OK")
    metrics = Counter({k: 0 for k in replay.FIELDS})
    created = replay.add_counts(metrics, output)
    assert created == {1} and metrics["new_global_ids"] == 1 and metrics["new_identity_observations"] == 3
    assert metrics["groups_new_identity"] == 1
    print("One three-camera identity counts once; three assigned observations remain explicit: OK")
    damaged = deepcopy(row)
    damaged["groups"][0][1]["embedding_row"] = 0
    rejects(lambda: decode(damaged))
    damaged = deepcopy(row)
    damaged["groups"] = (damaged["groups"][0], damaged["groups"][0])
    rejects(lambda: decode(damaged))
    damaged = deepcopy(row)
    damaged["timestamp"] = "1/30"
    rejects(lambda: decode(damaged))
    damaged = deepcopy(row)
    damaged["threshold"] = 0.6
    rejects(lambda: decode(damaged))
    damaged = deepcopy(row)
    damaged["decisions"] = damaged["decisions"][:-1]
    rejects(lambda: replay.source_counts(damaged, runtime, labels))
    print("Duplicate rows/observations, wrong scope/time and incomplete clique evidence rejected: OK")
    print("Global identity replay checks: PASSED")


if __name__ == "__main__":
    main()
