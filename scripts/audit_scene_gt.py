"""Audit scene_001 annotations and shared camera observations."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
source = json.loads((root / "configs/datasets/scene_001_source.json").read_text())
scene = root / "data/physicalai_smartspaces" / source["scene"]
calibration = json.loads((scene / "calibration_2025_format.json").read_text())

sizes = {}
for sensor in calibration["sensors"]:
    camera = int(sensor["id"].rsplit("_", 1)[1])
    if camera in sizes:
        raise ValueError(f"Duplicate camera in calibration: {camera}")
    attrs = {item["name"]: item["value"] for item in sensor["attributes"]}
    sizes[camera] = (int(attrs["frameWidth"]), int(attrs["frameHeight"]))

print("Loading ground truth...", flush=True)
gt = np.loadtxt(scene / "ground_truth.txt", dtype=np.float64, ndmin=2)
if gt.shape[1] != 9 or len(gt) == 0 or not np.isfinite(gt).all():
    raise ValueError("Expected a non-empty, finite table with nine columns.")
keys = gt[:, :3]
if np.any(keys < 0) or np.any(keys != np.floor(keys)) or np.any(keys >= 2**63):
    raise ValueError("Camera, person and frame IDs must be non-negative int64 values.")
keys = keys.astype(np.int64)
unknown = set(keys[:, 0].tolist()) - set(sizes)
if unknown:
    raise ValueError(f"Cameras missing from calibration: {sorted(unknown)}")

observations, identities, per_camera = {}, {}, []
for camera in sorted(sizes):
    mask = keys[:, 0] == camera
    rows, local_keys = gt[mask], keys[mask]
    observations[camera] = set(map(tuple, local_keys[:, 1:3].tolist()))
    identities[camera] = set(local_keys[:, 1].tolist())
    width, height = sizes[camera]
    x, y, w, h = rows[:, 3:7].T
    record = {
        "camera": camera,
        "rows": len(rows),
        "persons": len(identities[camera]),
        "annotated_frames": len(set(local_keys[:, 2].tolist())),
        "frame_min": int(local_keys[:, 2].min()) if len(rows) else None,
        "frame_max": int(local_keys[:, 2].max()) if len(rows) else None,
        "duplicate_keys": len(rows) - len(observations[camera]),
        "nonpositive_boxes": int(np.sum((w <= 0) | (h <= 0))),
        "outside_image": int(np.sum((x < 0) | (y < 0) | (x + w > width) | (y + h > height))),
    }
    per_camera.append(record)
    print(record)

pairs = []
for a, b in combinations(sorted(sizes), 2):
    common = observations[a] & observations[b]
    pairs.append({
        "camera_a": a,
        "camera_b": b,
        "shared_ids": len(identities[a] & identities[b]),
        "same_frame_ids": len({person for person, frame in common}),
        "same_frame_observations": len(common),
        "same_frame_count": len({frame for person, frame in common}),
    })
pairs.sort(key=lambda item: (-item["same_frame_observations"], -item["shared_ids"]))

report = {
    "scene": source["scene"],
    "revision": source["revision"],
    "total_rows": len(gt),
    "pair_basis": "unique GT (person_id, frame_id) keys; video sync not yet verified",
    "per_camera": per_camera,
    "pairs": pairs,
}
output = root / "artifacts/dataset_audit/scene_001_gt_summary.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("\nTOP 10 PAIRS BY SAME-FRAME OBSERVATIONS:")
for pair in pairs[:10]:
    print(pair)
print("\nReport:", output)
