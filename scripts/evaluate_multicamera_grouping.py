"""Evaluate stateless grouping on frozen pair decisions; no inference or GT relabeling."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
from itertools import combinations
import json
import math
from pathlib import Path

import evaluate_pairwise_association as baseline
from mtmc.association import grouping, pairwise
from mtmc.reid.osnet import ObservationKey

ROOT = Path(__file__).resolve().parents[1]
require = baseline.require
sha256 = baseline.sha256
EDGE_FIELDS = ("accepted_links", "correct_links", "wrong_known_links", "unresolved_gt_links")
GROUP_FIELDS = (
    "frames", "observations", "groups", "singletons", "two_member_groups", "three_member_groups",
    "linked_groups_all_known_same_gt", "linked_groups_different_known_gt", "linked_groups_unresolved_gt",
    "merged", "already_in_same_group", "rejected_camera_conflict", "rejected_missing_pair_support",
)
REASONS = {"empty_opposite_camera", "no_candidate_above_threshold", "assignment_competition"}


def integer(value, name):
    require(type(value) is int and value >= 0, f"Invalid {name}")
    return value


def reconstruct(records, *, run_id, frame, variant, threshold, cameras, fps):
    """Separate saved diagnostic labels from GT-free runtime PairAssociation inputs."""
    expected_pairs = list(combinations(cameras, 2))
    require(len(records) == len(expected_pairs), "Incomplete camera-pair round")
    labels, pairs = {}, []

    def add_label(camera, local_id, row, gt_id):
        key = ObservationKey(camera, integer(local_id, "local ID"), frame)
        integer(row, "embedding row")
        require(gt_id is None or type(gt_id) is int, "Invalid GT label")
        label = {"camera": camera, "local_id": local_id, "frame_index": frame,
                 "embedding_row": row, "gt_id": gt_id}
        require(key not in labels or labels[key] == label, "Inconsistent labels or source rows across camera pairs")
        labels[key] = label
        return key

    for record, (a, b) in zip(records, expected_pairs):
        require((record["run_id"], record["frame_index"], record["variant"], record["threshold"],
                 record["camera_a"], record["camera_b"]) == (run_id, frame, variant, threshold, a, b),
                "Unexpected record context/order; duplicate, missing or mixed records")
        require(Fraction(record["timestamp"]) == Fraction(frame, fps), "Incorrect scene timestamp")
        matches, unmatched = [], {a: [], b: []}
        for item in record["matches"]:
            left = add_label(a, item["left_local_id"], item["left_embedding_row"], item["left_gt_id"])
            right = add_label(b, item["right_local_id"], item["right_embedding_row"], item["right_gt_id"])
            matches.append(pairwise.PairMatch(left, right, item["similarity"]))
        for item in record["unmatched"]:
            camera = item["camera"]
            require(camera in (a, b), "Unmatched observation from another camera")
            key = add_label(camera, item["local_id"], item["embedding_row"], item["gt_id"])
            score = item["best_similarity"]
            require(item["reason"] in REASONS, "Unknown unmatched reason")
            require(score is None or (type(score) in (float, int) and math.isfinite(score) and -1 <= score <= 1),
                    "Invalid unmatched best similarity")
            unmatched[camera].append(pairwise.UnmatchedObservation(key, item["reason"], score))
        pairs.append(pairwise.PairAssociation(run_id, frame, Fraction(frame, fps), variant, a, b, threshold,
                                             tuple(matches), tuple(unmatched[a]), tuple(unmatched[b])))
    require(len({r["embedding_row"] for r in labels.values()}) == len(labels), "Source row used by multiple observations")
    for camera in cameras:
        known = [r["gt_id"] for k, r in labels.items() if k.camera_id == camera and r["gt_id"] is not None]
        require(len(known) == len(set(known)), "GT assignment is not one-to-one within a camera")
    # Run the actual grouping boundary validation before recomputing diagnostic metrics.
    result = grouping.group_pair_associations(pairs)
    before, per_pair = Counter(), {}
    for pair, record in zip(pairs, records):
        counts, matches, unmatched = baseline.score_pair(pair, labels)
        require(dict(counts) == record["counts"], "Saved per-pair counts disagree with reconstructed decisions")
        require(matches == record["matches"] and unmatched == record["unmatched"],
                "Saved outcomes or unmatched metadata disagree with diagnostic labels")
        before.update(counts)
        per_pair[pair.left_camera, pair.right_camera] = counts
    return pairs, labels, result, before, per_pair


def count_edges(edges, labels):
    counts = Counter({field: 0 for field in EDGE_FIELDS})
    for a, b in edges:
        left, right = labels[a]["gt_id"], labels[b]["gt_id"]
        field = ("unresolved_gt_links" if left is None or right is None else
                 "correct_links" if left == right else "wrong_known_links")
        counts[field] += 1
        counts["accepted_links"] += 1
    return counts


def evaluate_groups(pairs, labels, result):
    """Count retained edges and nontrivial groups; singleton is not a quality success."""
    original = {grouping.edge_key(m.left, m.right) for p in pairs for m in p.matches}
    retained, members = set(), []
    counts = Counter({field: 0 for field in GROUP_FIELDS})
    counts["frames"] = 1
    for group in result.groups:
        require(1 <= len(group) <= 3, "This diagnostic expects three cameras")
        require(len({k.camera_id for k in group}) == len(group), "Repeated camera inside a group")
        members.extend(group)
        retained.update(grouping.edge_key(a, b) for a, b in combinations(group, 2))
        counts["groups"] += 1
        counts[{1: "singletons", 2: "two_member_groups", 3: "three_member_groups"}[len(group)]] += 1
        if len(group) > 1:
            gt = [labels[k]["gt_id"] for k in group]
            known = {x for x in gt if x is not None}
            category = ("different_known_gt" if len(known) > 1 else
                        "unresolved_gt" if None in gt else "all_known_same_gt")
            counts["linked_groups_" + category] += 1
    require(len(members) == len(set(members)) and set(members) == set(labels), "Observations lost or duplicated")
    counts["observations"] = len(members)
    require(retained <= original, "Grouping introduced an unsupported edge")
    decision_edges = set()
    for decision in result.decisions:
        edge = grouping.edge_key(decision.left, decision.right)
        require(edge not in decision_edges, "Repeated grouping decision")
        decision_edges.add(edge)
        require(decision.outcome in GROUP_FIELDS[9:], "Unknown grouping decision outcome")
        counts[decision.outcome] += 1
        require((edge in retained) == (decision.outcome in ("merged", "already_in_same_group")),
                "Grouping decisions disagree with final membership")
    require(decision_edges == original, "A source edge has no grouping decision")
    require(len(retained) == counts["two_member_groups"] + 3 * counts["three_member_groups"], "Group edge accounting")
    require(len(members) == counts["singletons"] + 2 * counts["two_member_groups"] + 3 * counts["three_member_groups"],
            "Group observation accounting")
    return count_edges(retained, labels), count_edges(original - retained, labels), counts, retained


def rates(counts, available):
    accepted, correct, wrong = (counts[x] for x in ("accepted_links", "correct_links", "wrong_known_links"))
    return {**{field: counts[field] for field in EDGE_FIELDS}, "available_positive_pairs": available,
            "missed_positive_pairs": available - correct,
            "precision_labeled": correct / (correct + wrong) if correct + wrong else None,
            "recall_available_pairs": correct / available if available else None,
            "accepted_link_label_coverage": (correct + wrong) / accepted if accepted else None,
            "verified_correct_fraction_all_links": correct / accepted if accepted else None}


def summary(variant, threshold, before, after, removed, groups, components):
    available = before["available_positive_pairs"]
    require(all(before[f] == after[f] + removed[f] for f in EDGE_FIELDS), "Before/after accounting mismatch")
    return {"variant": variant, "threshold": threshold, "before": rates(before, available),
            "after": rates(after, available), "removed": {f: removed[f] for f in EDGE_FIELDS},
            "groups": dict(groups), "source_components": dict(components)}


def flatten(row):
    return {"variant": row["variant"], "threshold": row["threshold"],
            **{f"{section}_{key}": value for section in ("before", "after", "removed", "groups", "source_components")
               for key, value in row[section].items()}}


def read_round(handle, count):
    rows = []
    for _ in range(count):
        line = handle.readline()
        require(bool(line), "Decision trace ended before the expected round was complete")
        rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairwise-report", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.pairwise_report.resolve()
    source = json.loads(report_path.read_text(encoding="utf-8"))
    protocol = source["protocol"]
    require(source.get("completed") is True, "Source evaluation did not complete")
    require(protocol["name"] == "scene_001_partial_pairwise_threshold_diagnostic_v1", "Unsupported source protocol")
    cameras = tuple(protocol["cameras"])
    require(cameras == (4, 5, 8) and protocol["pairs"] == [[4, 5], [4, 8], [5, 8]], "Unexpected camera set")
    first, last, fps = (protocol[k] for k in ("first_frame", "last_frame", "fps"))
    require((first, last, fps) == (2, 299, 30), "Unexpected diagnostic timeline")
    config = protocol["configuration"]
    require(config["role"] == "integration_diagnostic_only" and config["variants"] == ["latest", "mean"],
            "Unexpected diagnostic configuration")
    thresholds = [pairwise.validate_threshold(t) for t in config["thresholds"]]
    require(thresholds and thresholds == sorted(set(thresholds)), "Thresholds must be unique and sorted")
    settings = [(v, t) for v in config["variants"] for t in thresholds]
    expected = {(r["variant"], r["threshold"]): r for r in source["results"]}
    require(len(expected) == len(source["results"]) and set(expected) == set(settings), "Source summary settings mismatch")
    expected_pairs = {(r["variant"], r["threshold"], r["camera_a"], r["camera_b"]): r for r in source["by_pair"]}
    require(len(expected_pairs) == len(source["by_pair"]) == len(settings) * 3
            and set(expected_pairs) == {(v, t, a, b) for v, t in settings for a, b in combinations(cameras, 2)},
            "Source per-camera-pair summary mismatch")
    entry = source["artifacts"]["decisions.jsonl.gz"]
    decisions_path = (report_path.parent / entry["path"]).resolve()
    print("Verifying frozen pairwise decisions and source report...", flush=True)
    require(sha256(decisions_path) == entry["sha256"], "Decision trace checksum mismatch")
    report_hash = sha256(report_path)
    now = datetime.now(timezone.utc)
    output = ROOT / "artifacts/multicamera_grouping" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    before_totals, after_totals, removed_totals = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    group_totals, component_totals, pair_totals = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    by_frame, used_rows = [], set()
    print("Replaying saved decisions on CPU; no detector, OSNet or GT relabeling...", flush=True)
    with gzip.open(decisions_path, "rt", encoding="utf-8") as stream, \
            gzip.open(output / "groups.jsonl.gz", "wt", encoding="utf-8") as destination:
        for frame in range(first, last + 1):
            reference_labels = None
            for variant, threshold in settings:
                records = read_round(stream, 3)
                pairs, labels, result, before, per_pair = reconstruct(
                    records, run_id=source["source_run_id"], frame=frame, variant=variant, threshold=threshold,
                    cameras=cameras, fps=fps)
                if reference_labels is None:
                    reference_labels = labels
                    rows = {r["embedding_row"] for r in labels.values()}
                    require(not rows & used_rows, "Embedding row reused across video frames")
                    used_rows.update(rows)
                else:
                    require(labels == reference_labels, "Variants/thresholds changed observations, labels or source rows")
                after, removed, group_counts, _ = evaluate_groups(pairs, labels, result)
                components, _ = baseline.audit_components([(m.left, m.right) for p in pairs for m in p.matches], labels)
                setting = (variant, threshold)
                before_totals[setting].update(before)
                after_totals[setting].update(after)
                removed_totals[setting].update(removed)
                group_totals[setting].update(group_counts)
                component_totals[setting].update(components)
                for (a, b), counts in per_pair.items():
                    pair_totals[variant, threshold, a, b].update(counts)
                row = summary(variant, threshold, before, after, removed, group_counts, components)
                by_frame.append({"frame_index": frame, **flatten(row)})
                detail = {"run_id": result.run_id, "frame_index": frame, "timestamp": str(result.timestamp),
                          "variant": variant, "threshold": threshold, "policy": result.policy,
                          "groups": [[{**asdict(k), "embedding_row": labels[k]["embedding_row"],
                                       "diagnostic_gt_id": labels[k]["gt_id"]} for k in group] for group in result.groups],
                          "decisions": [asdict(d) for d in result.decisions]}
                destination.write(json.dumps(detail, allow_nan=False) + "\n")
            if (frame + 1) % 60 == 0:
                print(f"Processed through frame {frame}/{last}", flush=True)
        require(stream.readline() == "", "Unexpected trailing decision records")
    summaries = []
    for setting in settings:
        require(all(before_totals[setting][f] == expected[setting][f] for f in baseline.PAIR_FIELDS),
                "Reconstructed baseline totals differ from source report")
        require(dict(component_totals[setting]) == expected[setting]["components"], "Source component counts differ")
        require(group_totals[setting]["observations"] == len(used_rows), "Observation coverage differs by setting")
        summaries.append(summary(*setting, before_totals[setting], after_totals[setting], removed_totals[setting],
                                 group_totals[setting], component_totals[setting]))
    for key, counts in pair_totals.items():
        require(all(counts[f] == expected_pairs[key][f] for f in baseline.PAIR_FIELDS), "Source pair totals differ")
    require(len({r["before"]["available_positive_pairs"] for r in summaries}) == 1, "Positive denominator changed")
    require(sha256(decisions_path) == entry["sha256"] and sha256(report_path) == report_hash,
            "Input files changed during evaluation")
    baseline.history_eval.write_csv(output / "summary.csv", [flatten(r) for r in summaries])
    baseline.history_eval.write_csv(output / "by_frame.csv", by_frame)
    report = {
        "completed": True, "created_utc": now.isoformat(), "source_run_id": source["source_run_id"],
        "protocol": {"name": "scene_001_multicamera_grouping_diagnostic_v1", "policy": grouping.POLICY,
                     "cameras": cameras, "first_frame": first, "last_frame": last, "fps": fps,
                     "variants": config["variants"], "thresholds": thresholds,
                     "ground_truth_used_by_grouping": False, "persistent_global_ids_created": False,
                     "selected_threshold": None,
                     "pair_precision": "correct / (correct + wrong_known); unresolved counted separately",
                     "pair_recall": "correct / original available positive pairs, unchanged after grouping",
                     "groups": "Frame-local partition including singletons; array indices are not global IDs",
                     "linked_group_labels": "Different known GT IDs take precedence over unknown labels; only size >= 2"},
        "inputs": {"pairwise_report": {"path": str(report_path), "sha256": report_hash},
                   "decisions": {"path": str(decisions_path), "sha256": entry["sha256"]}},
        "inherited_source_inputs_not_reverified": source["inputs"],
        "inherited_source_code_sha256": source["code_sha256"],
        "code_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(grouping.__file__), Path(pairwise.__file__),
                                                   Path(baseline.__file__), Path(baseline.history_eval.__file__))},
        "versions": {p: version(p) for p in ("numpy", "scipy")},
        "checks": {"decision_checksums_verified": True, "baseline_counts_reproduced": True,
                   "same_observations_labels_and_rows_for_all_settings": True, "all_observations_preserved": True,
                   "retained_edges_subset_of_original": True, "unique_cameras_per_group": True,
                   "source_row_count": len(used_rows), "groups_records": len(by_frame)},
        "results": summaries,
        "artifacts": {p.name: {"path": p.name, "sha256": sha256(p)} for p in sorted(output.iterdir()) if p.is_file()},
        "limits": ["Integration diagnostic, not threshold calibration or independent validation",
                   "Inherits frozen IoU-derived GT labels; raw GT, boxes, embeddings and earlier input hashes are not reverified",
                   "No independent observations across time; no confidence intervals or significance claims",
                   "Greedy complete-support grouping is not a globally optimal partition",
                   "Grouping can remove correct edges; no guaranteed precision improvement",
                   "No structural camera conflict is a policy invariant, not proof of identity correctness",
                   "Pair recall is conditional on current GT-assigned observations, not all people in the scene",
                   "No persistent global IDs, temporal recovery, global IDF1 or performance measurement"],
    }
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Variant Tau    P_before  P_after   R_before  R_after  DropOK DropWrong DropUnk Singletons")
    def pct(x):
        return f"{x:8.2%}" if x is not None else "     n/a"
    for row in summaries:
        b, a, d, g = (row[k] for k in ("before", "after", "removed", "groups"))
        print(f"{row['variant']:<7} {row['threshold']:4.2f}  {pct(b['precision_labeled'])} {pct(a['precision_labeled'])} "
              f" {pct(b['recall_available_pairs'])} {pct(a['recall_available_pairs'])} "
              f"{d['correct_links']:7} {d['wrong_known_links']:9} {d['unresolved_gt_links']:7} {g['singletons']:10}")
    print("Baseline counts reproduced: VERIFIED")
    print("All observations preserved; retained links have complete pair support: VERIFIED")
    print("Threshold selection: NONE; persistent global IDs: NONE")
    print(f"Report: {target}")
    print("Multi-camera grouping diagnostic: COMPLETED")


if __name__ == "__main__":
    main()
