"""Replay frozen pairwise association over a diagnostic threshold grid, on CPU."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path

import numpy as np

import evaluate_appearance_history as history_eval
from mtmc.association import pairwise
from mtmc.reid.osnet import ObservationKey, ReIDBatch

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = history_eval.CAMERAS
PAIR_FIELDS = (
    "camera_pair_frames", "accepted_links", "correct_links", "wrong_known_links", "unresolved_gt_links",
    "available_positive_pairs", "missed_positive_pairs", "endpoints", "unmatched_endpoints",
    "no_positive_endpoints", "no_positive_linked", "no_positive_abstained",
    "unlabeled_endpoints", "unlabeled_unmatched",
    "unmatched_empty_opposite_camera", "unmatched_no_candidate_above_threshold", "unmatched_assignment_competition",
)
COMPONENT_FIELDS = (
    "frames", "linked_components", "camera_conflict_components", "open_three_camera_components",
    "closed_three_camera_components", "frames_with_camera_conflict", "frames_with_open_three_camera_component",
    "components_with_different_known_gt", "components_with_unknown_gt",
)
require = history_eval.require
sha256 = history_eval.snapshot.sha256


def key_of(label):
    return ObservationKey(label["camera"], label["local_id"], label["frame_index"])


def score_pair(result, labels):
    """GT is read only after association; unmatched GT endpoints remain explicit."""
    left = {k: r for k, r in labels.items() if k.camera_id == result.left_camera}
    right = {k: r for k, r in labels.items() if k.camera_id == result.right_camera}
    require(all(k.frame_index == result.frame_index for k in (*left, *right)), "Mixed evaluation frames")
    known = {result.left_camera: {r["gt_id"] for r in left.values() if r["gt_id"] is not None},
             result.right_camera: {r["gt_id"] for r in right.values() if r["gt_id"] is not None}}
    counts = Counter({field: 0 for field in PAIR_FIELDS})
    counts.update(camera_pair_frames=1, endpoints=len(left) + len(right),
                  available_positive_pairs=len(known[result.left_camera] & known[result.right_camera]))
    seen_left, seen_right, decisions = set(), set(), []
    for match in result.matches:
        require(match.left in left and match.right in right, "Association key missing from frozen labels")
        require(match.left not in seen_left and match.right not in seen_right, "Association is not one-to-one")
        seen_left.add(match.left)
        seen_right.add(match.right)
        a, b = left[match.left], right[match.right]
        outcome = ("unresolved_gt" if a["gt_id"] is None or b["gt_id"] is None else
                   "correct" if a["gt_id"] == b["gt_id"] else "wrong_known")
        counts[{"unresolved_gt": "unresolved_gt_links", "correct": "correct_links",
                "wrong_known": "wrong_known_links"}[outcome]] += 1
        counts["accepted_links"] += 1
        decisions.append({"left_local_id": match.left.local_id, "right_local_id": match.right.local_id,
                          "left_embedding_row": a["embedding_row"], "right_embedding_row": b["embedding_row"],
                          "left_gt_id": a["gt_id"], "right_gt_id": b["gt_id"],
                          "similarity": match.cosine_similarity, "outcome": outcome})
    unmatched = (*result.unmatched_left, *result.unmatched_right)
    unmatched_keys = {item.key for item in unmatched}
    require(len(unmatched_keys) == len(unmatched), "Duplicate unmatched observation")
    require(unmatched_keys.isdisjoint(seen_left | seen_right)
            and unmatched_keys | seen_left | seen_right == set(left) | set(right),
            "Matched/unmatched observations do not partition the pair inputs")
    for item in unmatched:
        field = "unmatched_" + item.reason
        require(field in PAIR_FIELDS, "Unknown unmatched reason")
        counts[field] += 1
    counts["unmatched_endpoints"] = len(unmatched)
    for current, other_camera in ((left, result.right_camera), (right, result.left_camera)):
        for key, label in current.items():
            if label["gt_id"] is None:
                counts["unlabeled_endpoints"] += 1
                counts["unlabeled_unmatched"] += int(key in unmatched_keys)
            elif label["gt_id"] not in known[other_camera]:
                counts["no_positive_endpoints"] += 1
                counts["no_positive_abstained" if key in unmatched_keys else "no_positive_linked"] += 1
    counts["missed_positive_pairs"] = counts["available_positive_pairs"] - counts["correct_links"]
    require(counts["missed_positive_pairs"] >= 0, "Correct links exceed available positive pairs")
    require(counts["endpoints"] == 2 * counts["accepted_links"] + counts["unmatched_endpoints"],
            "Pair endpoint accounting mismatch")
    unmatched_details = [{"camera": item.key.camera_id, "local_id": item.key.local_id,
                          "embedding_row": labels[item.key]["embedding_row"], "gt_id": labels[item.key]["gt_id"],
                          "reason": item.reason, "best_similarity": item.best_similarity} for item in unmatched]
    return counts, decisions, unmatched_details


def summarize(counts):
    def ratio(a, b):
        return counts[a] / b if b else None
    return {**{field: counts[field] for field in PAIR_FIELDS},
            "precision_labeled": ratio("correct_links", counts["correct_links"] + counts["wrong_known_links"]),
            "verified_correct_fraction_all_links": ratio("correct_links", counts["accepted_links"]),
            "accepted_link_label_coverage": (counts["correct_links"] + counts["wrong_known_links"]) / counts["accepted_links"]
            if counts["accepted_links"] else None,
            "recall_available_pairs": ratio("correct_links", counts["available_positive_pairs"]),
            "no_positive_abstention_rate": ratio("no_positive_abstained", counts["no_positive_endpoints"])}


def audit_components(links, labels):
    """Inspect pairwise-link connectivity; never assign a global identity to it."""
    adjacency = defaultdict(set)
    for a, b in links:
        require(a.camera_id != b.camera_id and a.frame_index == b.frame_index, "Invalid diagnostic edge")
        adjacency[a].add(b)
        adjacency[b].add(a)
    counts = Counter({field: 0 for field in COMPONENT_FIELDS})
    counts["frames"] = 1
    visited, issues = set(), []
    ordering = lambda k: (k.camera_id, k.local_id, k.frame_index)
    for start in sorted(adjacency, key=ordering):
        if start in visited:
            continue
        members, stack = set(), [start]
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(adjacency[current] - members)
        visited.update(members)
        cameras = [k.camera_id for k in members]
        edge_count = sum(len(adjacency[k]) for k in members) // 2
        conflict = len(cameras) != len(set(cameras))
        open_three = len(members) == len(set(cameras)) == 3 and edge_count == 2
        closed_three = len(members) == len(set(cameras)) == 3 and edge_count == 3
        gt_ids = [labels[k]["gt_id"] for k in members]
        counts["linked_components"] += 1
        counts["camera_conflict_components"] += int(conflict)
        counts["open_three_camera_components"] += int(open_three)
        counts["closed_three_camera_components"] += int(closed_three)
        counts["components_with_different_known_gt"] += int(len({x for x in gt_ids if x is not None}) > 1)
        counts["components_with_unknown_gt"] += int(None in gt_ids)
        if conflict or open_three:
            issues.append({"camera_conflict": conflict, "open_three_camera": open_three, "edges": edge_count,
                           "members": [{**asdict(k), "embedding_row": labels[k]["embedding_row"],
                                        "gt_id": labels[k]["gt_id"]} for k in sorted(members, key=ordering)]})
    counts["frames_with_camera_conflict"] = int(counts["camera_conflict_components"] > 0)
    counts["frames_with_open_three_camera_component"] = int(counts["open_three_camera_components"] > 0)
    return counts, issues


def make_camera(run_id, camera_id, frame, variant, records, features):
    selected = sorted((r for r in records if r["camera"] == camera_id), key=lambda r: r["local_id"])
    timestamp = Fraction(frame, history_eval.FPS)
    indices = [r["embedding_row"] for r in selected]
    batch = ReIDBatch(tuple(key_of(r) for r in selected), (timestamp,) * len(selected),
                      np.asarray(features[indices]))
    return pairwise.CameraAppearance(run_id, camera_id, frame, timestamp, variant, batch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/association/pairwise_diagnostic.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "configs/datasets/scene_001_source.json")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = history_eval.read_json(config_path)
    require(config["role"] == "integration_diagnostic_only", "This script is only an integration diagnostic")
    require(config["variants"] == ["latest", "mean"], "Expected paired latest/mean experiment")
    thresholds = [pairwise.validate_threshold(x) for x in config["thresholds"]]
    require(bool(thresholds) and thresholds == sorted(set(thresholds)), "Thresholds must be unique and sorted")
    print("Verifying frozen embeddings, histories and checksums...", flush=True)
    history, source, trace, histories, latest, mean, paths = history_eval.load_inputs(args.history_report.resolve())
    manifest_path = args.source_manifest.resolve()
    manifest = history_eval.read_json(manifest_path)
    entries = [x for x in manifest["files"] if Path(x["local_path"]).name == "ground_truth.txt"]
    require(len(entries) == 1, "Expected one GT entry in source manifest")
    gt_path = history_eval.checked_path(ROOT, {"path": entries[0]["local_path"], "sha256": entries[0]["sha256"]})
    paths.update(ground_truth=gt_path, source_manifest=manifest_path, experiment_config=config_path)
    print("Assigning fixed GT labels once; replay uses appearance only...", flush=True)
    labels, gt_matching = history_eval.label_trace(trace, history_eval.load_ground_truth(gt_path))
    frames = defaultdict(list)
    for label in labels:
        frames[label["frame_index"]].append(label)
    now = datetime.now(timezone.utc)
    output = ROOT / "artifacts/pairwise_association" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    totals, pair_totals, component_totals = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    examples, by_frame = defaultdict(list), []
    print(f"Thresholds: {thresholds}; variants: latest, mean; frames: 2..299", flush=True)
    with gzip.open(output / "decisions.jsonl.gz", "wt", encoding="utf-8") as decisions_file:
        for frame in range(history_eval.FIRST, history_eval.LAST + 1):
            records = frames[frame]
            lookup = {key_of(r): r for r in records}
            require(len(lookup) == len(records), "Duplicate frozen observation key")
            for variant, features in (("latest", latest), ("mean", mean)):
                cameras = {c: make_camera(source["run_id"], c, frame, variant, records, features) for c in CAMERAS}
                for threshold in thresholds:
                    group = (variant, threshold)
                    links, round_counts = [], Counter()
                    for a, b in combinations(CAMERAS, 2):
                        result = pairwise.associate_camera_pair(cameras[a], cameras[b], min_similarity=threshold)
                        counts, decisions, unmatched = score_pair(result, lookup)
                        totals[group].update(counts)
                        round_counts.update(counts)
                        pair_totals[variant, threshold, a, b].update(counts)
                        links.extend((m.left, m.right) for m in result.matches)
                        record = {"run_id": source["run_id"], "frame_index": frame, "timestamp": str(result.timestamp),
                                  "variant": variant, "threshold": threshold, "camera_a": a, "camera_b": b,
                                  "counts": dict(counts), "matches": decisions, "unmatched": unmatched}
                        decisions_file.write(json.dumps(record, allow_nan=False) + "\n")
                    component_counts, issues = audit_components(links, lookup)
                    component_totals[group].update(component_counts)
                    if len(examples[group]) < 5 and issues:
                        examples[group].append({"frame_index": frame, "issues": issues})
                    by_frame.append({"frame_index": frame, "variant": variant, "threshold": threshold,
                                     **summarize(round_counts), **dict(component_counts)})
            if (frame + 1) % 60 == 0:
                print(f"Processed through frame {frame}/299", flush=True)
    summaries = [{"variant": v, "threshold": t, **summarize(totals[v, t]),
                  "components": dict(component_totals[v, t])} for v in config["variants"] for t in thresholds]
    per_pair = [{"variant": v, "threshold": t, "camera_a": a, "camera_b": b,
                 **summarize(pair_totals[v, t, a, b])} for v in config["variants"] for t in thresholds
                for a, b in combinations(CAMERAS, 2)]
    denominators = {s["available_positive_pairs"] for s in summaries}
    require(len(denominators) == 1, "Variant or threshold changed positive-pair denominator")
    for row in summaries:
        if row["threshold"] == 1:
            require(row["accepted_links"] == 0, "Reject-all control unexpectedly accepted a link")
        require(row["components"]["frames"] == history_eval.LAST - history_eval.FIRST + 1, "Missing component rounds")
    history_eval.write_csv(output / "summary.csv", [
        {**{k: v for k, v in s.items() if k != "components"}, **s["components"]} for s in summaries])
    history_eval.write_csv(output / "by_pair.csv", per_pair)
    history_eval.write_csv(output / "by_frame.csv", by_frame)
    history_eval.write_csv(output / "gt_matching.csv", gt_matching)
    report = {
        "completed": True, "created_utc": now.isoformat(), "source_run_id": source["run_id"],
        "protocol": {
            "name": "scene_001_partial_pairwise_threshold_diagnostic_v1", "cameras": CAMERAS,
            "first_frame": history_eval.FIRST, "last_frame": history_eval.LAST,
            "fps": history_eval.FPS, "configuration": config, "history_policy": history["configuration"],
            "pairs": list(combinations(CAMERAS, 2)), "counting": "Each unordered camera pair once per frame",
            "gate": "strict similarity > threshold", "objective": "maximize sum(similarity - threshold); unmatched gain zero",
            "labels": "Fixed framewise clipped-box IoU >= 0.5; max valid cardinality then max IoU sum",
            "ground_truth_used_by_association": False, "global_ids_created": False, "selected_threshold": None,
            "unknown_labels": "Accepted links with either GT label missing are unresolved, not silently correct/incorrect",
            "precision_labeled": "correct / (correct + wrong_known)",
            "verified_correct_fraction_all_links": "correct / all accepted, including unresolved in denominator",
            "recall_available_pairs": "correct / shared assigned GT identities in the two current observation sets",
            "missed_positive_pairs": "available positive pairs minus correct accepted links",
            "no_positive_endpoints": "Assigned endpoint GT has no assigned counterpart in the opposite current camera",
            "components": "Connectivity audit only; a missing third link is not proof of different identities",
            "examples": "First five frames with camera-conflict or open-three-camera components per variant/threshold",
        },
        "inputs": {k: {"path": str(p), "sha256": sha256(p)} for k, p in paths.items()},
        "code_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(pairwise.__file__), Path(history_eval.__file__),
                                                   Path(history_eval.history_io.__file__), Path(history_eval.snapshot.__file__))},
        "versions": {p: version(p) for p in ("numpy", "scipy")},
        "results": summaries, "by_pair": per_pair,
        "component_examples": [{"variant": v, "threshold": t, "frames": examples[v, t]}
                               for v in config["variants"] for t in thresholds],
        "artifacts": {p.name: {"path": p.name, "sha256": sha256(p)} for p in sorted(output.iterdir()) if p.is_file()},
        "limits": ["Exploratory integration sweep; no validation calibration or automatic threshold selection",
                   "Counts reuse people across frames/camera pairs; not independent trials",
                   "IoU-derived labels are diagnostic and may be ambiguous",
                   "Recall is conditional on both people observations being present and GT-assigned",
                   "No persistent global IDs or global IDF1; framewise connected components are only audited",
                   "No temporal ID-switch recovery, geometry, delayed handover or crop-quality filtering",
                   "This run is not a performance benchmark"],
    }
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Variant Tau   Links Correct Wrong Unknown   P_known  R_pairs  CamConflictFrames")
    def percent(value):
        return f"{value:7.2%}" if value is not None else "    n/a"
    for row in summaries:
        print(f"{row['variant']:<7} {row['threshold']:4.2f} {row['accepted_links']:6} {row['correct_links']:7} "
              f"{row['wrong_known_links']:5} {row['unresolved_gt_links']:7} "
              f"{percent(row['precision_labeled'])} {percent(row['recall_available_pairs'])} "
              f"{row['components']['frames_with_camera_conflict']:8}")
    print("Available positive pairs per variant/threshold:", next(iter(denominators)))
    print("Threshold selection: NONE (integration diagnostic only)")
    print(f"Report: {target}")
    print("Pairwise association diagnostic: COMPLETED")


if __name__ == "__main__":
    main()
