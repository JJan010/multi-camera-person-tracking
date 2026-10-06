"""Compare latest and mean descriptors on identical frozen cross-camera queries.

CPU-only diagnostic for scene_001, cameras 4/5/8, video frames 2..299.
Uses existing snapshot IoU/retrieval helpers without changing their protocol.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path

import numpy as np

import evaluate_reid_snapshot as snapshot
import preview_appearance_history as history_io

ROOT = Path(__file__).resolve().parents[1]
CAMERAS, FIRST, LAST, FPS = (4, 5, 8), 2, 299, 30
WIDTH, HEIGHT = 1920, 1080


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def checked_path(base, item):
    path = (base / item["path"]).resolve()
    require(snapshot.sha256(path) == item["sha256"], f"Checksum mismatch: {path}")
    return path


def validate_history_rows(records, observations, latest, means, run_id, config):
    """Independently check row mapping, exact window membership and mean values."""
    require(len(records) == len(observations) == len(latest) == len(means),
            "History/source observation count mismatch")
    cap, age = config["max_observations"], Fraction(config["max_age_seconds"])
    require(type(cap) is int and cap > 0 and age > 0, "Invalid history policy")
    retained = defaultdict(list)
    for i, (record, source) in enumerate(zip(records, observations)):
        require(record["run_id"] == run_id and record["embedding_row"] == i
                and source["embedding_row"] == i, "History run/row mismatch")
        require(all(record[k] == source[k] for k in ("camera", "local_id", "frame_index")),
                "History observation key mismatch")
        frame = source["frame_index"]
        now = Fraction(frame, FPS)
        require(Fraction(record["timestamp"]) == now, "History timestamp mismatch")
        key = (source["camera"], source["local_id"])
        previous = retained[key]
        require(not previous or observations[previous[-1]]["frame_index"] < frame,
                "Duplicate or out-of-order local observation")
        expected = [j for j in previous
                    if now - Fraction(observations[j]["frame_index"], FPS) <= age]
        expected = (expected + [i])[-cap:]
        retained[key] = expected
        frames = [observations[j]["frame_index"] for j in expected]
        require(record["source_embedding_rows"] == expected
                and record["source_frames"] == frames
                and record["history_size"] == len(expected),
                "History membership differs from causal count/age policy")
        vector = np.mean(latest[expected], axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        fallback = norm <= 1e-12
        target = latest[i] if fallback else (vector / norm).astype(np.float32)
        require(record["used_latest_fallback"] == fallback, "Fallback flag mismatch")
        require(np.allclose(means[i], target, rtol=0, atol=1e-6),
                f"Mean does not match its source vectors: row {i}")


def load_inputs(path):
    report = read_json(path)
    require(report["completed"] is True, "History replay did not complete")
    source_path = checked_path(path.parent, report["source_report"])
    source, trace, latest, tracks_path, latest_path = history_io.read_frozen_run(source_path)
    require(source["run_id"] == report["source_run_id"], "Source run mismatch")
    require(snapshot.sha256(tracks_path) == report["source_tracks"]["sha256"]
            and snapshot.sha256(latest_path) == report["source_embeddings"]["sha256"],
            "History refers to different source artifacts")
    require(source["configuration"]["fps"] == FPS
            and sorted(source["configuration"]["cameras"]) == list(CAMERAS)
            and len(trace) == LAST + 1, "Expected scene_001 baseline: 300 rounds, cameras 4/5/8, 30 FPS")
    mean_path = checked_path(path.parent, report["artifacts"]["mean_embeddings"])
    rows_path = checked_path(path.parent, report["artifacts"]["observations"])
    means = np.load(mean_path, mmap_mode="r", allow_pickle=False)
    require(means.dtype == np.float32 and means.shape == latest.shape
            and np.isfinite(means).all()
            and np.allclose(np.linalg.norm(means, axis=1), 1, rtol=0, atol=1e-5),
            "Invalid normalized mean descriptor matrix")
    with rows_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    observations = [item for row in trace for item in row["reid_observations"]
                    if item["status"] == "encoded"]
    validate_history_rows(records, observations, latest, means, source["run_id"], report["configuration"])
    paths = {"history_report": path, "source_report": source_path, "tracks": tracks_path,
             "latest_embeddings": latest_path, "mean_embeddings": mean_path, "history_rows": rows_path}
    return report, source, trace, records, latest, means, paths


def load_ground_truth(path):
    ground = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            require(len(fields) == 9, "Expected nine GT columns")
            camera, identity, frame = map(int, fields[:3])
            if camera not in CAMERAS or not FIRST <= frame <= LAST:
                continue
            x, y, width, height = map(float, fields[3:7])
            require(np.isfinite([x, y, width, height]).all() and width > 0 and height > 0,
                    "Invalid GT box")
            require(identity not in ground[frame, camera], "Duplicate GT identity in one camera/frame")
            ground[frame, camera][identity] = [x, y, x + width, y + height]
    # The audited selected cameras have GT on EVERY frame 2..299. Fail on a truncated input.
    require(all(ground[f, c] for f in range(FIRST, LAST + 1) for c in CAMERAS),
            "Incomplete GT coverage for the fixed integration subset")
    return ground


def label_trace(trace, ground):
    all_labels, matching = [], []
    for row in trace:
        frame = row["frame_index"]
        records = [item for item in row["reid_observations"] if item["status"] == "encoded"]
        if frame < FIRST:
            all_labels.extend({**item, "gt_id": None, "gt_iou": None,
                               "gt_valid_candidates": 0, "gt_status": "unannotated_frame"}
                              for item in records)
            continue
        labels, stats = snapshot.label_observations(
            records, {c: ground[frame, c] for c in CAMERAS}, WIDTH, HEIGHT)
        for item in labels:
            item["gt_status"] = "matched" if item["gt_id"] is not None else "unmatched"
        all_labels.extend(labels)
        for c in CAMERAS:
            if not any(s["camera"] == c for s in stats):
                boxes = snapshot.clip_boxes(list(ground[frame, c].values()), WIDTH, HEIGHT)
                visible = int(np.all(boxes[:, 2:] > boxes[:, :2], axis=1).sum())
                stats.append({"camera": c, "crops": 0, "gt_visible": visible,
                              "matched_crops": 0, "unmatched_crops": 0, "unmatched_gt": visible,
                              "excluded_zero_area_gt": len(boxes) - visible})
        matching.extend({"frame_index": frame, **s} for s in stats)
    require([r["embedding_row"] for r in all_labels] == list(range(len(all_labels))),
            "GT labeling changed source row order")
    return all_labels, matching


def history_label_status(current_gt, source_gt):
    """Diagnostic label agreement; not proof that a local track has one identity."""
    known = [x for x in source_gt if x is not None]
    different = sum(x != current_gt for x in known) if current_gt is not None else None
    unknown = len(source_gt) - len(known)
    if current_gt is None:
        status = "current_unmatched"
    elif different:
        status = "conflict_with_current"
    elif unknown:
        status = "unknown_members"
    else:
        status = "consistent_with_current"
    return {"history_gt_status": status, "known_different_members": different,
            "unknown_members": unknown, "distinct_known_gt_ids": len(set(known))}


def annotate_histories(labels, histories):
    diagnostics = []
    for label, history in zip(labels, histories):
        gt_ids = [labels[j]["gt_id"] for j in history["source_embedding_rows"]]
        diagnostic = history_label_status(label["gt_id"], gt_ids)
        label.update(diagnostic)
        diagnostics.append({k: label[k] for k in (
            "embedding_row", "frame_index", "camera", "local_id", "gt_id", "gt_iou",
            "gt_status", "gt_valid_candidates", "history_gt_status", "known_different_members",
            "unknown_members", "distinct_known_gt_ids")})
        diagnostics[-1].update(history_size=history["history_size"],
                               source_embedding_rows=json.dumps(history["source_embedding_rows"]),
                               source_gt_ids=json.dumps(gt_ids))
    return diagnostics


def compare_direction(labels, latest_scores, mean_scores, a, b, frame):
    baseline = snapshot.rank_queries(latest_scores, labels, a, b)
    averaged = snapshot.rank_queries(mean_scores, labels, a, b)
    require(len(baseline) == len(averaged), "Different query counts")
    positive_by_id = {r["gt_id"]: r for r in labels if r["camera"] == b and r["gt_id"] is not None}
    rows = []
    for left, right in zip(baseline, averaged):
        require(all(left[k] == right[k] for k in (
            "query_embedding_row", "query_gt_id", "status", "gallery_size", "positive_local_id")),
            "Descriptor variant changed query eligibility")
        query = labels[left["query_embedding_row"]]
        positive = positive_by_id.get(query["gt_id"])
        row = {"frame_index": frame, "query_camera": a, "gallery_camera": b,
               "query_local_id": query["local_id"], "query_embedding_row": query["embedding_row"],
               "query_gt_id": query["gt_id"], "status": left["status"], "gallery_size": left["gallery_size"],
               "query_history_status": query["history_gt_status"],
               "positive_embedding_row": positive["embedding_row"] if positive else None,
               "positive_history_status": positive["history_gt_status"] if positive else None}
        for variant, result in (("latest", left), ("mean", right)):
            for key, value in result.items():
                if key.startswith("positive_") or key.startswith("top"):
                    row[f"{variant}_{key}"] = value
        rows.append(row)
    return rows


def paired_summary(rows):
    eligible = [r for r in rows if r["status"] == "evaluated"]
    n = len(eligible)
    variants = {}
    for variant in ("latest", "mean"):
        ranks = [r[f"{variant}_positive_rank"] for r in eligible]
        hit1, hit3 = sum(r == 1 for r in ranks), sum(r <= 3 for r in ranks)
        variants[variant] = {"rank1_hits": hit1, "rank3_hits": hit3,
                             "rank1": hit1 / n if n else None, "rank3": hit3 / n if n else None}
    improved = sum(r["latest_positive_rank"] != 1 and r["mean_positive_rank"] == 1 for r in eligible)
    worsened = sum(r["latest_positive_rank"] == 1 and r["mean_positive_rank"] != 1 for r in eligible)
    both_correct = sum(r["latest_positive_rank"] == r["mean_positive_rank"] == 1 for r in eligible)
    return {"queries": len(rows), "eligible_queries": n,
            "unmatched_queries": sum(r["status"] == "unmatched_query" for r in rows),
            "no_positive_in_gallery": sum(r["status"] == "no_positive_in_gallery" for r in rows),
            **variants, "rank1_delta_percentage_points": 100 * (improved - worsened) / n if n else None,
            "rank1_transitions": {"both_correct": both_correct, "improved": improved, "worsened": worsened,
                                  "both_wrong": n - both_correct - improved - worsened}}


def history_pair_group(row):
    statuses = (row["query_history_status"], row["positive_history_status"])
    if "conflict_with_current" in statuses:
        return "query_or_positive_conflict"
    if statuses == ("consistent_with_current", "consistent_with_current"):
        return "query_and_positive_consistent"
    return "unknown_members"


def write_csv(path, rows):
    require(bool(rows), f"Cannot write empty diagnostic table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-report", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "configs/datasets/scene_001_source.json")
    args = parser.parse_args()
    print("Verifying frozen source, history checksums and row mapping...", flush=True)
    history, source, trace, histories, latest, means, paths = load_inputs(args.history_report.resolve())
    manifest_path = args.source_manifest.resolve()
    manifest = read_json(manifest_path)
    gt_entries = [item for item in manifest["files"] if Path(item["local_path"]).name == "ground_truth.txt"]
    require(len(gt_entries) == 1, "Source manifest must identify exactly one ground_truth.txt")
    gt_path = checked_path(ROOT, {"path": gt_entries[0]["local_path"], "sha256": gt_entries[0]["sha256"]})
    paths.update(source_manifest=manifest_path, ground_truth=gt_path)
    print("Loading GT once; labeling fixed observations by IoU...", flush=True)
    labels, matching = label_trace(trace, load_ground_truth(gt_path))
    diagnostics = annotate_histories(labels, histories)
    rankings, frame_results = [], []
    by_frame = defaultdict(list)
    for label in labels:
        by_frame[label["frame_index"]].append(label)
    print("Comparing latest/latest against mean/mean for all six camera directions...", flush=True)
    for frame in range(FIRST, LAST + 1):
        current = by_frame[frame]
        indices = [r["embedding_row"] for r in current]
        x, y = np.asarray(latest[indices]), np.asarray(means[indices])
        # Float64 accumulation avoids unnecessary CPU BLAS rounding near tied scores.
        x, y = x.astype(np.float64), y.astype(np.float64)
        left, right = np.clip(x @ x.T, -1, 1), np.clip(y @ y.T, -1, 1)
        current_rows = []
        for a, b in permutations(CAMERAS, 2):
            current_rows.extend(compare_direction(current, left, right, a, b, frame))
        rankings.extend(current_rows)
        summary = paired_summary(current_rows)
        frame_results.append({"frame_index": frame, "eligible_queries": summary["eligible_queries"],
                              "latest_rank1": summary["latest"]["rank1"], "mean_rank1": summary["mean"]["rank1"],
                              "delta_pp": summary["rank1_delta_percentage_points"],
                              **summary["rank1_transitions"]})
    directions = [{"query_camera": a, "gallery_camera": b,
                   **paired_summary([r for r in rankings if r["query_camera"] == a and r["gallery_camera"] == b])}
                  for a, b in permutations(CAMERAS, 2)]
    pooled = paired_summary(rankings)
    require(pooled["eligible_queries"] > 0, "No eligible cross-camera queries")
    groups = defaultdict(list)
    for row in rankings:
        if row["status"] == "evaluated":
            groups[history_pair_group(row)].append(row)
    evaluated_labels = [r for r in diagnostics if FIRST <= r["frame_index"] <= LAST]
    output = ROOT / "artifacts/appearance_history_evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("rankings.csv", rankings), ("by_frame.csv", frame_results),
                       ("history_labels.csv", diagnostics), ("gt_matching.csv", matching)):
        write_csv(output / name, rows)
    report = {
        "completed": True, "source_run_id": source["run_id"],
        "protocol": {
            "name": "scene_001_latest_vs_mean_retrieval_v1", "cameras": CAMERAS,
            "first_frame": FIRST, "last_frame": LAST, "fps": FPS, "source_resolution_wh": [WIDTH, HEIGHT],
            "variants": ["latest query / latest gallery", "mean query / mean gallery"],
            "gt_assignment": "Once per camera/frame: clipped boxes; IoU >= 0.5; max valid count then max IoU sum",
            "gallery": "All encoded current observations in the target camera, including unmatched distractors",
            "query_eligibility": "Assigned query GT has an assigned observation in the current target camera",
            "ranking": "Normalized dot product accumulated in float64; descending; stable source row ties",
            "similarity_threshold": None, "same_queries_for_both_variants": True,
            "aggregation": "Pool eligible directed queries; no frame/camera averaging of percentages",
            "frames_0_1": "Excluded from retrieval; retained as unknown GT members of early histories",
            "history_policy": history["configuration"],
            "gt_usage": "Evaluation labels only; no history resets, crop filters or descriptor changes",
        },
        "inputs": {k: {"path": str(p), "sha256": snapshot.sha256(p)} for k, p in paths.items()},
        "code_sha256": {p.name: snapshot.sha256(p) for p in (
            Path(__file__), Path(snapshot.__file__), Path(history_io.__file__))},
        "versions": {p: version(p) for p in ("numpy", "scipy")},
        "pooled": pooled, "directions": directions,
        "history_diagnostic": {
            "evaluated_observations": len(evaluated_labels),
            "status_counts": dict(Counter(r["history_gt_status"] for r in evaluated_labels)),
            "histories_with_multiple_known_gt_ids": sum(r["distinct_known_gt_ids"] > 1 for r in evaluated_labels),
            "unannotated_observations_before_first_frame": len(labels) - len(evaluated_labels),
            "conditional_retrieval_by_query_and_positive_history": {k: paired_summary(v) for k, v in groups.items()},
            "fixed_examples": [r for r in diagnostics if r["camera"] == 8 and r["local_id"] in (10, 12)
                               and r["frame_index"] in (210, 213, 214, 217, 221)],
        },
        "gt_matching_totals": {str(c): {k: sum(r[k] for r in matching if r["camera"] == c)
                               for k in ("crops", "gt_visible", "matched_crops", "unmatched_crops",
                                         "unmatched_gt", "excluded_zero_area_gt")} for c in CAMERAS},
        "artifacts": {p.name: {"path": p.name, "sha256": snapshot.sha256(p)} for p in sorted(output.glob("*.csv"))},
        "limits": [
            "One integration clip, not an independent validation/test or official benchmark",
            "Nearby frames and directed queries are correlated; counts are not independent people",
            "Retrieval requires a positive in the gallery; it does not evaluate rejection of absent identities",
            "GT IoU labels can be ambiguous; label conflicts are diagnostics, not motmetrics IDSW events",
            "History strata use query/positive labels; distractor histories can also change the ranking",
            "Rank-1 is not global IDF1; this script creates no global IDs and does not modify tracking",
            "No causal claims about end-to-end speed from this CPU evaluation",
        ],
    }
    target = output / "report.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Direction  Eligible / queries    Latest R1    Mean R1    Delta pp")
    for item in directions + [{"query_camera": "ALL", "gallery_camera": "", **pooled}]:
        a, b = item["query_camera"], item["gallery_camera"]
        def percent(value):
            return f"{value:.2%}" if value is not None else "n/a"
        delta = item["rank1_delta_percentage_points"]
        change = f"{delta:+.2f}" if delta is not None else "n/a"
        print(f"{str(a):>3}->{str(b):<3} {item['eligible_queries']:7}/{item['queries']:<7} "
              f"{percent(item['latest']['rank1']):>10} {percent(item['mean']['rank1']):>10} {change:>10}")
    print("Rank-1 transitions:", json.dumps(pooled["rank1_transitions"]))
    print("History GT diagnostics:", json.dumps(report["history_diagnostic"]["status_counts"]))
    print(f"Report: {target}")
    print("Appearance history retrieval evaluation: COMPLETED")


if __name__ == "__main__":
    main()
