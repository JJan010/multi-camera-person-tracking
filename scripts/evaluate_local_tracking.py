"""Evaluate local tracking on scene_001, cameras 4/5/8, frames 2–299."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import motmetrics as mm
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4, 5, 8)
FIRST, LAST, FPS = 2, 299, 30
WIDTH, HEIGHT, MIN_IOU = 1920, 1080, 0.5


def boxes_array(values):
    boxes = np.asarray(values, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError("Non-finite or nonpositive original box")
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, WIDTH)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, HEIGHT)
    return boxes


def distances(gt_boxes, pred_boxes):
    intersection = np.maximum(
        np.minimum(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])
        - np.maximum(gt_boxes[:, None, :2], pred_boxes[None, :, :2]), 0
    ).prod(axis=2)
    union = (
        (gt_boxes[:, 2:] - gt_boxes[:, :2]).prod(axis=1)[:, None]
        + (pred_boxes[:, 2:] - pred_boxes[:, :2]).prod(axis=1)[None, :]
        - intersection
    )
    iou = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    return np.where(iou >= MIN_IOU, 1.0 - iou, np.nan)


def check_geometry():
    gt = boxes_array([[0, 0, 10, 10]])
    pred = boxes_array([[0, 0, 10, 10], [0, 0, 5, 10], [0, 0, 4, 10]])
    np.testing.assert_allclose(distances(gt, pred), [[0, 0.5, np.nan]])
    np.testing.assert_allclose(boxes_array([[-5, -5, 10, 10]]), gt)
    if distances(gt, boxes_array([])).shape != (1, 0):
        raise AssertionError("Invalid distance shape for empty predictions")
    if distances(boxes_array([]), pred).shape != (0, 3):
        raise AssertionError("Invalid distance shape for empty ground truth")
    if not np.isnan(distances(gt, boxes_array([[2000, 0, 2010, 10]]))).all():
        raise AssertionError("Fully outside prediction must remain unmatched")
    print("IoU adapter checks: PASSED")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", type=Path,
                        default=ROOT / "artifacts/tracking_preview/tracks.jsonl")
    parser.add_argument("--gt", type=Path, default=ROOT / (
        "data/physicalai_smartspaces/MTMC_Tracking_2024/train/scene_001/ground_truth.txt"
    ))
    args = parser.parse_args()
    if version("motmetrics") != "1.4.0":
        raise RuntimeError("This protocol requires motmetrics==1.4.0")
    check_geometry()
    mm.lap.default_solver = "scipy"

    with args.tracks.open(encoding="utf-8") as handle:
        trace = [json.loads(line) for line in handle]
    if [row["frame_index"] for row in trace] != list(range(LAST + 1)):
        raise ValueError("Expected every video frame from 0 through 299 exactly once")

    gt = defaultdict(dict)
    print("Loading ground truth...")
    with args.gt.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            camera, identity, frame = map(int, fields[:3])
            if camera not in CAMERAS or not FIRST <= frame <= LAST:
                continue
            if len(fields) != 9 or identity in gt[camera, frame]:
                raise ValueError(f"Invalid or duplicate GT row: {camera}, {frame}")
            x, y, width, height = map(float, fields[3:7])
            gt[camera, frame][identity] = [x, y, x + width, y + height]
    if any(not any(gt[camera, f] for f in range(FIRST, LAST + 1)) for camera in CAMERAS):
        raise ValueError("Ground truth is missing for an evaluated camera")

    accumulators = {c: mm.MOTAccumulator(auto_id=False) for c in CAMERAS}
    excluded_gt = {c: 0 for c in CAMERAS}
    outside_predictions = {c: 0 for c in CAMERAS}
    for row in trace:
        frame = row["frame_index"]
        if not np.isclose(row["timestamp_seconds"], frame / FPS, rtol=0, atol=1e-9):
            raise ValueError(f"Unexpected timestamp on frame {frame}")
        if sorted(item["camera"] for item in row["cameras"]) != list(CAMERAS):
            raise ValueError(f"Unexpected camera set on frame {frame}")
        if frame < FIRST:
            continue
        for item in row["cameras"]:
            camera = item["camera"]
            pred_ids = item["local_ids"]
            pred_boxes = boxes_array(item["xyxy"])
            if (any(type(i) is not int for i in pred_ids)
                    or len(set(pred_ids)) != len(pred_ids)
                    or len(pred_ids) != len(pred_boxes)):
                raise ValueError(f"Invalid predictions: camera {camera}, frame {frame}")
            outside_predictions[camera] += int(
                np.any(pred_boxes[:, 2:] <= pred_boxes[:, :2], axis=1).sum()
            )
            ground = gt[camera, frame]
            gt_ids = np.asarray(list(ground), dtype=np.int64)
            gt_boxes = boxes_array(list(ground.values()))
            visible = np.all(gt_boxes[:, 2:] > gt_boxes[:, :2], axis=1)
            excluded_gt[camera] += int((~visible).sum())
            accumulators[camera].update(
                gt_ids[visible], pred_ids, distances(gt_boxes[visible], pred_boxes),
                frameid=frame,
            )

    metrics = ["num_frames", "num_objects", "num_predictions", "idtp", "idfp", "idfn",
               "idf1", "precision", "recall", "num_switches", "num_false_positives", "num_misses"]
    host = mm.metrics.create()
    summary = host.compute_many(
        list(accumulators.values()), metrics=metrics,
        names=[f"camera_{c:04d}" for c in CAMERAS], generate_overall=True,
    ).rename(index={"OVERALL": "ALL_CAMERAS_LOCAL"})
    display = ["idf1", "precision", "recall", "num_switches", "num_false_positives", "num_misses"]
    print(mm.io.render_summary(summary[display], formatters=host.formatters,
                              namemap=mm.io.motchallenge_metric_names))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts/tracking_evaluation" / stamp
    output.mkdir(parents=True, exist_ok=False)
    summary.to_csv(output / "metrics.csv", index_label="sequence")
    for camera, accumulator in accumulators.items():
        accumulator.mot_events.to_csv(output / f"camera_{camera:04d}_events.csv")
    report = {
        "protocol": {
            "name": "scene_001_local_2d_iou_v1", "cameras": CAMERAS,
            "first_frame": FIRST, "last_frame": LAST, "fps": FPS,
            "width": WIDTH, "height": HEIGHT, "min_iou": MIN_IOU,
            "box_policy": "Clip GT and predictions; exclude zero-area GT; keep all predictions",
            "max_switch_time": "infinity", "solver": "scipy",
            "aggregation": "Pool local identity counts; not cross-camera identity evaluation",
            "scope": "Integration subset, not official AI City evaluation",
        },
        "versions": {p: version(p) for p in ("motmetrics", "numpy", "scipy", "pandas")},
        "inputs": {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                   for name, path in {"tracks": args.tracks, "gt": args.gt}.items()},
        "script_sha256": sha256(Path(__file__)),
        "excluded_gt": excluded_gt, "fully_outside_predictions": outside_predictions,
        "metrics": json.loads(summary.to_json(orient="index")),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
