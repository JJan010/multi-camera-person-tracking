"""Diagnose cross-camera OSNet retrieval on one synchronized frame."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
MIN_IOU = 0.5


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_boxes(values, width, height):
    boxes = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Invalid original box coordinates")
    return np.clip(boxes, [0, 0, 0, 0], [width, height, width, height])


def pairwise_iou(a, b):
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                              - np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    union = ((a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
             + (b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection)
    return np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)


def assign_iou(ious):
    """First maximize valid match count, then total IoU; embeddings are not used."""
    if min(ious.shape) == 0:
        return []
    valid = ious >= MIN_IOU
    penalty = min(ious.shape) + 1.0
    a, b = linear_sum_assignment(np.where(valid, 1.0 - ious, penalty))
    return [(int(i), int(j)) for i, j in zip(a, b) if valid[i, j]]


def load_gt(path, frame, cameras):
    ground = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            camera, identity, gt_frame = map(int, fields[:3])
            if gt_frame != frame or camera not in cameras:
                continue
            if len(fields) != 9 or identity in ground[camera]:
                raise ValueError(f"Invalid/duplicate GT row: camera={camera}, frame={frame}")
            x, y, width, height = map(float, fields[3:7])
            if width <= 0 or height <= 0:
                raise ValueError("Nonpositive GT box")
            ground[camera][identity] = [x, y, x + width, y + height]
    if any(not ground[camera] for camera in cameras):
        raise ValueError("No GT for an evaluated camera/frame; check the GT file")
    return ground


def label_observations(records, ground, width, height):
    labels = [{**r, "gt_id": None, "gt_iou": None, "gt_valid_candidates": 0}
              for r in records]
    stats = []
    for camera in sorted({r["camera"] for r in records}):
        indices = [i for i, r in enumerate(records) if r["camera"] == camera]
        gt_ids = sorted(ground[camera])
        gt_boxes = clip_boxes([ground[camera][i] for i in gt_ids], width, height)
        visible = np.all(gt_boxes[:, 2:] > gt_boxes[:, :2], axis=1)
        gt_ids = np.asarray(gt_ids)[visible]
        gt_boxes = gt_boxes[visible]
        boxes = clip_boxes([records[i]["source_xyxy"] for i in indices], width, height)
        ious = pairwise_iou(boxes, gt_boxes)
        for i, row_index in enumerate(indices):
            labels[row_index]["gt_valid_candidates"] = int((ious[i] >= MIN_IOU).sum())
        matches = assign_iou(ious)
        for i, j in matches:
            labels[indices[i]].update(gt_id=int(gt_ids[j]), gt_iou=float(ious[i, j]))
        stats.append({"camera": camera, "crops": len(indices), "gt_visible": len(gt_ids),
                      "matched_crops": len(matches), "unmatched_crops": len(indices) - len(matches),
                      "unmatched_gt": len(gt_ids) - len(matches),
                      "excluded_zero_area_gt": int((~visible).sum())})
    return labels, stats


def rank_queries(similarities, labels, query_camera, gallery_camera):
    queries = [i for i, r in enumerate(labels) if r["camera"] == query_camera]
    gallery = [i for i, r in enumerate(labels) if r["camera"] == gallery_camera]
    rows = []
    for i in queries:
        # Stable ties are broken by original embedding row order.
        order = [gallery[k] for k in np.argsort(-similarities[i, gallery], kind="stable")]
        query = labels[i]
        gt_id = query["gt_id"]
        positives = [j for j in order if gt_id is not None and labels[j]["gt_id"] == gt_id]
        if len(positives) > 1:
            raise ValueError("Expected at most one assigned GT identity per camera")
        status = ("unmatched_query" if gt_id is None else
                  "no_positive_in_gallery" if not positives else "evaluated")
        row = {"query_camera": query_camera, "query_local_id": query["local_id"],
               "query_embedding_row": i, "query_gt_id": gt_id, "gallery_camera": gallery_camera,
               "status": status, "gallery_size": len(gallery),
               "positive_rank": order.index(positives[0]) + 1 if positives else None,
               "positive_local_id": labels[positives[0]]["local_id"] if positives else None,
               "positive_similarity": float(similarities[i, positives[0]]) if positives else None}
        for rank in range(1, 4):
            j = order[rank - 1] if rank <= len(order) else None
            row[f"top{rank}_local_id"] = labels[j]["local_id"] if j is not None else None
            row[f"top{rank}_gt_id"] = labels[j]["gt_id"] if j is not None else None
            row[f"top{rank}_similarity"] = float(similarities[i, j]) if j is not None else None
        rows.append(row)
    return rows


def summarize(rows):
    evaluated = [r for r in rows if r["status"] == "evaluated"]
    n = len(evaluated)
    rank1 = sum(r["positive_rank"] == 1 for r in evaluated)
    rank3 = sum(r["positive_rank"] <= 3 for r in evaluated)
    return {"queries": len(rows), "evaluated_queries": n,
            "unmatched_queries": sum(r["status"] == "unmatched_query" for r in rows),
            "no_positive_in_gallery": sum(r["status"] == "no_positive_in_gallery" for r in rows),
            "rank1_hits": rank1, "rank1": rank1 / n if n else None,
            "rank3_hits": rank3, "rank3": rank3 / n if n else None}


def score_summary(values):
    if not values:
        return {"count": 0}
    return {"count": len(values), **dict(zip(
        ("min", "p05", "median", "p95", "max"),
        map(float, np.quantile(values, [0, 0.05, 0.5, 0.95, 1]))))}


def check_protocol():
    a = clip_boxes([[-5, 0, 10, 10]], 20, 20)
    b = clip_boxes([[0, 0, 10, 10], [0, 0, 5, 10], [0, 0, 4, 10]], 20, 20)
    np.testing.assert_allclose(pairwise_iou(a, b), [[1, 0.5, 0.4]])
    assert assign_iou(np.array([[0.5, 0.49]])) == [(0, 0)]
    assert set(assign_iou(np.array([[0.99, 0.5], [0.5, 0.49]]))) == {(0, 1), (1, 0)}
    assert assign_iou(np.zeros((2, 3))) == []
    assert assign_iou(np.zeros((0, 3))) == []
    labels = [
        {"camera": 4, "local_id": 1, "gt_id": 10},
        {"camera": 4, "local_id": 2, "gt_id": 20},
        {"camera": 4, "local_id": 3, "gt_id": None},
        {"camera": 5, "local_id": 1, "gt_id": None},
        {"camera": 5, "local_id": 2, "gt_id": 10},
    ]
    scores = np.zeros((5, 5))
    scores[0, 3:5] = [0.9, 0.8]  # Unknown gallery observation remains a distractor.
    rows = rank_queries(scores, labels, 4, 5)
    assert rows[0]["positive_rank"] == 2
    assert rows[1]["status"] == "no_positive_in_gallery"
    assert rows[2]["status"] == "unmatched_query"
    summary = summarize(rows)
    assert summary["evaluated_queries"] == 1 and summary["rank1"] == 0 and summary["rank3"] == 1
    print("GT matching and retrieval checks: PASSED")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Embedding manifest.json")
    parser.add_argument("--gt", type=Path, default=ROOT / (
        "data/physicalai_smartspaces/MTMC_Tracking_2024/train/scene_001/ground_truth.txt"))
    args = parser.parse_args()
    check_protocol()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    embedding_path = manifest_path.parent / manifest["embeddings"]["path"]
    if sha256(embedding_path) != manifest["embeddings"]["sha256"]:
        raise ValueError("Embedding checksum mismatch")
    crop_manifest_path = Path(manifest["source_manifest"]["path"])
    if sha256(crop_manifest_path) != manifest["source_manifest"]["sha256"]:
        raise ValueError("Source crop manifest checksum mismatch")
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    width, height = crop_manifest["configuration"]["source_resolution_wh"]
    records = manifest["records"]
    features = np.load(embedding_path, allow_pickle=False)
    if (features.shape != (len(records), 512) or not records or features.dtype != np.float32
            or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError("Expected finite L2-normalized float32 embeddings, shape (N, 512)")
    if [r["embedding_row"] for r in records] != list(range(len(records))):
        raise ValueError("Embedding row mapping is inconsistent")
    keys = [(r["camera"], r["local_id"], r["frame_index"]) for r in records]
    frames = {r["frame_index"] for r in records}
    if len(set(keys)) != len(keys) or len(frames) != 1:
        raise ValueError("Expected unique observations from one synchronized frame")
    frame = frames.pop()
    cameras = sorted({r["camera"] for r in records})
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required")
    for record in records:
        original = crop_manifest["crops"][record["source_manifest_index"]]
        if any(record[key] != value for key, value in original.items()):
            raise ValueError("Embedding observation differs from its source crop record")
    print(f"Loading GT for frame {frame}, cameras {cameras}...", flush=True)
    ground = load_gt(args.gt, frame, cameras)
    labels, matching = label_observations(records, ground, width, height)
    for item in matching:
        print(f"Camera {item['camera']}: matched {item['matched_crops']}/{item['crops']} crops; "
              f"unmatched GT={item['unmatched_gt']}")

    # Ranking uses only appearance; GT determines labels and evaluation eligibility.
    similarities = np.clip(features @ features.T, -1.0, 1.0)
    rows, directions = [], []
    print("Direction   Eligible / queries   Rank-1   Rank-3")
    for a, b in permutations(cameras, 2):
        current = rank_queries(similarities, labels, a, b)
        stats = summarize(current)
        directions.append({"query_camera": a, "gallery_camera": b, **stats})
        rows.extend(current)
        r1 = f"{stats['rank1']:.1%}" if stats["rank1"] is not None else "n/a"
        r3 = f"{stats['rank3']:.1%}" if stats["rank3"] is not None else "n/a"
        print(f"{a:2d} -> {b:<2d}           {stats['evaluated_queries']:2d} / {stats['queries']:<2d}       {r1:>6}   {r3:>6}")
    overall = summarize(rows)
    print(f"Pooled Rank-1: {overall['rank1_hits']}/{overall['evaluated_queries']}")
    same, different = [], []
    for i, a in enumerate(labels):
        for j in range(i + 1, len(labels)):
            b = labels[j]
            if a["camera"] == b["camera"] or a["gt_id"] is None or b["gt_id"] is None:
                continue
            target = same if a["gt_id"] == b["gt_id"] else different
            target.append(float(similarities[i, j]))

    output = ROOT / "artifacts/reid_evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "rankings.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "protocol": {
            "name": "scene_snapshot_cross_camera_retrieval_v1", "frame_index": frame,
            "cameras": cameras, "source_resolution_wh": [width, height],
            "gt_assignment": "Clip boxes; IoU >= 0.5; max valid cardinality then max total IoU",
            "gt_box_policy": "Exclude zero-area GT after clipping; keep all crop observations",
            "similarity": "Dot product of L2-normalized embeddings; descending order",
            "gallery": "All observations in the target camera, including unmatched observations",
            "query_eligibility": "Query has assigned GT and that GT has an assigned gallery observation",
            "ties": "Original embedding row order", "similarity_threshold": None,
            "aggregation": "Pool eligible directed queries; observations are reused across directions",
            "scope": "One integration frame; diagnostic labels from IoU; not an official benchmark",
        },
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in {
            "embedding_manifest": manifest_path, "embeddings": embedding_path, "gt": args.gt,
        }.items()},
        "versions": {p: version(p) for p in ("numpy", "scipy")},
        "script_sha256": sha256(Path(__file__)),
        "gt_matching": matching, "directions": directions, "pooled": overall,
        "pair_similarities": {"same_gt": score_summary(same), "different_gt": score_summary(different),
                              "counting": "Each unordered cross-camera pair with assigned GT, once"},
        "observations": labels,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Rankings: {output / 'rankings.csv'}")
    print("Re-ID retrieval diagnostic: COMPLETED")


if __name__ == "__main__":
    main()
