"""Explain frozen global ID errors without changing predictions or calibrating settings."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
import evaluate_global_identity as evaluation
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
require = evaluation.require


def inspect_slot(gt_ids, keys, global_ids, mask):
    """Return spatial cardinality ceiling and mutually unique overlap evidence."""
    require(mask.shape == (len(gt_ids), len(keys)) and len(keys) == len(global_ids), "Invalid slot shape")
    require(mask.dtype == np.bool_, "Expected Boolean IoU gate")
    ceiling = 0
    if mask.shape[0] and mask.shape[1]:
        rows, cols = linear_sum_assignment(-mask.astype(np.int64))
        ceiling = int(mask[rows, cols].sum())
    gt_degree, pred_degree = mask.sum(axis=1), mask.sum(axis=0)
    evidence = []
    for i, j in zip(*np.nonzero(mask)):
        if gt_degree[i] == pred_degree[j] == 1:
            evidence.append((keys[j], gt_ids[i], global_ids[j]))
    return ceiling, evidence, {
        "gt_without_spatial_candidate": int((gt_degree == 0).sum()),
        "predictions_without_spatial_candidate": int((pred_degree == 0).sum()),
        "gt_with_multiple_candidates": int((gt_degree > 1).sum()),
        "predictions_with_multiple_candidates": int((pred_degree > 1).sum()),
        "mutually_unique_pairs": len(evidence),
    }


def decompose(metrics, spatial_tp):
    gt, pred, idtp = (metrics[k] for k in ("gt_observations", "predicted_observations", "idtp"))
    require(idtp <= spatial_tp <= min(gt, pred), "Invalid spatial ceiling")
    gap = spatial_tp - idtp
    result = {"framewise_maximum_spatial_matches": spatial_tp, "global_identity_matches": idtp,
              "shared_identity_assignment_gap": gap,
              "minimum_fn_with_fixed_boxes": gt - spatial_tp, "minimum_fp_with_fixed_boxes": pred - spatial_tp,
              "framewise_spatial_f1_ceiling": 2 * spatial_tp / (gt + pred) if gt + pred else None}
    require(metrics["idfn"] == result["minimum_fn_with_fixed_boxes"] + gap, "IDFN decomposition failed")
    require(metrics["idfp"] == result["minimum_fp_with_fixed_boxes"] + gap, "IDFP decomposition failed")
    return result


def classify_conflict(members, unique_gt):
    labels = [unique_gt.get(ObservationKey(**member)) for member in members]
    known = {x for x in labels if x is not None}
    if len(known) > 1:
        return "different_known_gt"
    if None in labels or not labels:
        return "unresolved"
    return "all_members_same_gt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--variant", choices=("latest", "mean"), required=True)
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    evaluation_path = args.evaluation_report.resolve()
    source = json.loads(evaluation_path.read_text(encoding="utf-8"))
    require(source.get("completed") is True and source["protocol"]["name"] == "scene_001_global_2d_identity_v1",
            "Expected a completed shared global identity evaluation")
    selected = [r for r in source["results"] if (r["variant"], r["threshold"]) == (args.variant, args.threshold)]
    require(len(selected) == 1, "Requested diagnostic setting missing from evaluation")
    expected = selected[0]
    inputs = {"evaluation_report": {"path": str(evaluation_path), "sha256": sha256(evaluation_path)}}
    print("Verifying evaluated assignments, boxes and GT...", flush=True)
    for name in ("identity_report", "assignments", "tracks", "ground_truth"):
        entry = source["inputs"][name]
        path = evaluation.checked(entry["path"], entry["sha256"])
        inputs[name] = {"path": str(path), "sha256": entry["sha256"]}
    replay = json.loads(Path(inputs["identity_report"]["path"]).read_text(encoding="utf-8"))
    settings = [(v, t) for v in replay["protocol"]["variants"] for t in replay["protocol"]["thresholds"]]
    replay_rows = {(r["variant"], r["threshold"]): r for r in replay["results"]}
    target = args.variant, args.threshold
    require(len(set(settings)) == len(settings) and set(settings) == set(replay_rows) and target in settings, "Invalid replay settings")
    require(replay["replay_id"] == source["source_replay_id"], "Evaluation/replay mismatch")
    ground = evaluation.history.load_ground_truth(Path(inputs["ground_truth"]["path"]))
    slots, frame_maps, _, _ = evaluation.build_slots(Path(inputs["tracks"]["path"]), ground, replay["source_run_id"])
    counts, spatial_tp, spatial = evaluation.IdentityCounts(), 0, Counter()
    gt_global, global_gt, local_gt = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    support = defaultdict(list)
    gt_simultaneous, global_simultaneous = Counter(), Counter()
    conflict_counts = Counter({k: 0 for k in ("total", "all_members_same_gt", "different_known_gt", "unresolved")})
    conflict_examples = {k: [] for k in ("all_members_same_gt", "different_known_gt", "unresolved")}
    print(f"Diagnosing {args.variant}, threshold={args.threshold}; predictions stay frozen...", flush=True)
    with gzip.open(inputs["assignments"]["path"], "rt", encoding="utf-8") as stream:
        for number in range(2, 300):
            for setting in settings:
                line = stream.readline()
                require(bool(line), "Truncated assignment trace")
                record = json.loads(line)
                require((record["frame_index"], record["descriptor_variant"], record["min_similarity"])
                        == (number, *setting), "Unexpected assignment trace order")
                if setting != target:
                    continue
                assigned = evaluation.read_assignments(record, run_id=replay["source_run_id"], number=number,
                    variant=setting[0], threshold=setting[1], identity_scope=replay_rows[setting]["identity_scope"],
                    expected_rows=frame_maps[number])
                unique_gt, frame_gt, frame_global = {}, defaultdict(set), defaultdict(set)
                for camera in evaluation.CAMERAS:
                    gt, keys, mask = slots[number, camera]
                    gids = [assigned[k] for k in keys]
                    counts.update(gt, gids, mask)
                    ceiling, evidence, stats = inspect_slot(gt, keys, gids, mask)
                    spatial_tp += ceiling
                    spatial.update(stats)
                    for key, identity, gid in evidence:
                        unique_gt[key] = identity
                        gt_global[identity][gid] += 1
                        global_gt[gid][identity] += 1
                        local_gt[key.camera_id, key.local_id][identity] += 1
                        support[identity, gid, key.camera_id, key.local_id].append(number)
                        frame_gt[identity].add(gid)
                        frame_global[gid].add(identity)
                gt_simultaneous.update(identity for identity, gids in frame_gt.items() if len(gids) > 1)
                global_simultaneous.update(gid for gid, identities in frame_global.items() if len(identities) > 1)
                for decision in record["decisions"]:
                    if decision["outcome"] != "rejected_multiple_existing_ids":
                        continue
                    require(all(ObservationKey(**k) in assigned for k in decision["members"]), "Unknown conflict member")
                    category = classify_conflict(decision["members"], unique_gt)
                    conflict_counts["total"] += 1
                    conflict_counts[category] += 1
                    if len(conflict_examples[category]) < 5:
                        conflict_examples[category].append({"frame_index": number, "timestamp": str(Fraction(number, 30)),
                            "anchor_global_ids": decision["anchor_global_ids"],
                            "members": [{**member, "global_id": assigned[ObservationKey(**member)],
                                "unique_overlap_gt_id": unique_gt.get(ObservationKey(**member))} for member in decision["members"]]})
            if (number + 1) % 60 == 0:
                print(f"Inspected through frame {number}/299", flush=True)
        require(stream.readline() == "", "Unexpected trailing assignments")
    metrics, _ = counts.result()
    require(all(metrics[k] == expected[k] for k in metrics), "Frozen global metrics were not reproduced exactly")
    require(conflict_counts["total"] == replay_rows[target]["groups_rejected_multiple_existing_ids"], "Conflict count differs from replay")
    decomposition = decompose(metrics, spatial_tp)
    gt_rows = []
    for identity, histogram in gt_global.items():
        major, n = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))[0]
        total = sum(histogram.values())
        gt_rows.append({"gt_id": identity, "unique_overlap_observations": total,
            "distinct_global_ids": len(histogram), "largest_global_id": major, "largest_global_support": n,
            "support_outside_largest_global": total - n,
            "frames_with_multiple_global_ids": gt_simultaneous[identity],
            "support_by_global_id": json.dumps(dict(sorted(histogram.items())))})
    gt_rows.sort(key=lambda r: (-r["support_outside_largest_global"], r["gt_id"]))
    global_rows = []
    for gid, histogram in global_gt.items():
        major, n = sorted(histogram.items(), key=lambda item: (-item[1], item[0]))[0]
        total = sum(histogram.values())
        global_rows.append({"global_id": gid, "unique_overlap_observations": total,
            "distinct_gt_ids": len(histogram), "largest_gt_id": major, "largest_gt_support": n,
            "support_outside_largest_gt": total - n,
            "frames_with_multiple_gt_ids": global_simultaneous[gid],
            "support_by_gt_id": json.dumps(dict(sorted(histogram.items())))})
    global_rows.sort(key=lambda r: (-r["support_outside_largest_gt"], r["global_id"]))
    local_rows = [{"camera": c, "local_id": local, "distinct_gt_ids": len(histogram),
                   "unique_overlap_observations": sum(histogram.values()),
                   "support_by_gt_id": json.dumps(dict(sorted(histogram.items())))}
                  for (c, local), histogram in sorted(local_gt.items())]
    support_rows = [{"gt_id": gt, "global_id": gid, "camera": c, "local_id": local,
                     "observations": len(frames), "first_frame": min(frames), "last_frame": max(frames)}
                    for (gt, gid, c, local), frames in sorted(support.items())]
    now = datetime.now(timezone.utc)
    output = ROOT / "artifacts/global_identity_diagnostic" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for filename, rows in (("gt_fragmentation.csv", gt_rows), ("global_identity_mixing.csv", global_rows),
                           ("local_track_labels.csv", local_rows), ("support_by_track.csv", support_rows)):
        if rows:
            evaluation.history.write_csv(output / filename, rows)
    (output / "conflict_examples.json").write_text(json.dumps(conflict_examples, indent=2) + "\n", encoding="utf-8")
    for entry in inputs.values():
        evaluation.checked(entry["path"], entry["sha256"])
    report = {
        "completed": True, "created_utc": now.isoformat(), "source_replay_id": replay["replay_id"],
        "protocol": {"name": "scene_001_global_identity_failure_diagnostic_v1", "variant": args.variant,
            "threshold": args.threshold, "selection_role": "Explicit diagnostic case, not deployment calibration",
            "cameras": evaluation.CAMERAS, "first_frame": 2, "last_frame": 299, "min_iou": 0.5,
            "evidence": "Only mutually unique IoU>=0.5 overlaps: one candidate on both GT and prediction sides",
            "spatial_ceiling": "Independent maximum-cardinality box matching per camera/frame; GT-assisted upper bound",
            "predictions_modified": False, "ground_truth_used_for_runtime_changes": False},
        "inputs": inputs, "global_metrics_reproduced": metrics, "decomposition": decomposition,
        "spatial_diagnostics": dict(spatial), "existing_id_conflicts": dict(conflict_counts),
        "evidence_summary": {
            "gt_identities_with_evidence": len(gt_rows), "global_ids_with_evidence": len(global_rows),
            "gt_identities_with_multiple_global_ids": sum(r["distinct_global_ids"] > 1 for r in gt_rows),
            "global_ids_with_multiple_gt_ids": sum(r["distinct_gt_ids"] > 1 for r in global_rows),
            "local_tracks_with_multiple_gt_ids": sum(r["distinct_gt_ids"] > 1 for r in local_rows)},
        "top_gt_fragmentation": gt_rows[:5], "top_global_mixing": [r for r in global_rows if r["distinct_gt_ids"] > 1][:5],
        "conflict_examples": conflict_examples,
        "code_sha256": {q.name: sha256(q) for q in (Path(__file__), Path(evaluation.__file__),
                          Path(evaluation.history.__file__), Path(evaluation.history.snapshot.__file__))},
        "artifacts": {q.name: {"path": q.name, "sha256": sha256(q)} for q in sorted(output.iterdir()) if q.is_file()},
        "limits": ["Spatial ceiling is not a promised achievable causal identity score",
            "Assignment gap includes competition and duplicate compatible boxes, not only incorrect association logic",
            "Unique IoU overlap is diagnostic evidence, not visual proof; excludes ambiguous overlaps",
            "Evidence histograms are not IDTP/IDFP/IDFN and are not CLEAR fragmentation or ID-switch metrics",
            "Repeated same-person conflicts suggest investigation, not permission to use GT to merge runtime IDs",
            "First/last frame ranges in support CSV may contain gaps; support counts are observation counts",
            "Short reused training clip; no final threshold or merge rule is selected"],
    }
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Reproduced global IDF1: {metrics['idf1']:.2%}")
    print(f"Framewise spatial F1 ceiling: {decomposition['framewise_spatial_f1_ceiling']:.2%}")
    print(f"IDFN {metrics['idfn']} = fixed-box minimum {decomposition['minimum_fn_with_fixed_boxes']} + shared-ID gap {decomposition['shared_identity_assignment_gap']}")
    print(f"IDFP {metrics['idfp']} = fixed-box minimum {decomposition['minimum_fp_with_fixed_boxes']} + shared-ID gap {decomposition['shared_identity_assignment_gap']}")
    print("Existing-ID conflict diagnostics:", json.dumps(dict(conflict_counts)))
    print("Mutually unique overlap evidence:", json.dumps(report["evidence_summary"]))
    print("Top GT fragmentation: GT / global IDs / evidence outside largest ID / simultaneous-split frames")
    for r in gt_rows[:5]:
        print(r["gt_id"], r["distinct_global_ids"], r["support_outside_largest_global"], r["frames_with_multiple_global_ids"])
    print("Top global mixing: global ID / GT IDs / evidence outside largest GT / simultaneous-mix frames")
    for r in report["top_global_mixing"]:
        print(r["global_id"], r["distinct_gt_ids"], r["support_outside_largest_gt"], r["frames_with_multiple_gt_ids"])
    print(f"Report: {path}")
    print("Global identity failure diagnostic: COMPLETED")


if __name__ == "__main__":
    main()
