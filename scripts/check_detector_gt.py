"""Diagnostic one-frame detection matching, not official benchmark evaluation."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
preview = ROOT / "artifacts/detector_preview"
prediction = json.loads((preview / "predictions.json").read_text())
frame_index = prediction["frame_index"]
cameras = {item["camera"] for item in prediction["cameras"]}
gt = defaultdict(list)

scene = ROOT / "data/physicalai_smartspaces/MTMC_Tracking_2024/train/scene_001"
with (scene / "ground_truth.txt").open() as handle:
    for line in handle:
        if not line.strip():
            continue
        fields = line.split()
        camera, person, frame = map(int, fields[:3])
        if camera in cameras and frame == frame_index:
            x, y, w, h = map(float, fields[3:7])
            gt[camera].append((person, [x, y, x + w, y + h]))


def clip(boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, 1920)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, 1080)
    return boxes


def pairwise_iou(a, b):
    lower = np.maximum(a[:, None, :2], b[None, :, :2])
    upper = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(upper - lower, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection, union,
        out=np.zeros_like(intersection), where=union > 0,
    )


report = {
    "frame_index": frame_index,
    "confidence_threshold": prediction["threshold"],
    "iou_threshold": 0.5,
    "policy": "Clip both GT and predictions to image; exclude zero-area GT.",
    "evaluation": "Diagnostic maximum-cardinality matching, then maximum IoU.",
    "cameras": [],
}

for item in prediction["cameras"]:
    camera = item["camera"]
    ids = np.array([row[0] for row in gt[camera]], dtype=int)
    boxes_gt = clip([row[1] for row in gt[camera]])
    valid = np.all(boxes_gt[:, 2:] > boxes_gt[:, :2], axis=1)
    excluded_ids = ids[~valid].tolist()
    ids, boxes_gt = ids[valid], boxes_gt[valid]
    boxes_pred = clip([row["xyxy"] for row in item["detections"]])

    iou = pairwise_iou(boxes_gt, boxes_pred)
    eligible = iou >= report["iou_threshold"]
    # Prioritize the number of valid matches over their total IoU.
    bonus = min(len(boxes_gt), len(boxes_pred)) + 1
    rewards = np.where(eligible, bonus + iou, 0.0)
    rows, cols = linear_sum_assignment(rewards, maximize=True)
    accepted = eligible[rows, cols]
    rows, cols = rows[accepted], cols[accepted]

    tp = len(rows)
    fp = len(boxes_pred) - tp
    fn = len(boxes_gt) - tp
    result = {
        "camera": camera,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "unmatched_gt_ids": ids[
            ~np.isin(np.arange(len(ids)), rows)
        ].tolist(),
        "excluded_gt_ids": excluded_ids,
        "matched_ious": iou[rows, cols].tolist(),
    }
    report["cameras"].append(result)
    print(json.dumps(
        {key: value for key, value in result.items() if key != "matched_ious"},
        indent=2,
    ))

output = preview / "gt_matching.json"
output.write_text(json.dumps(report, indent=2) + "\n")
print("Report:", output)
