"""Check the reusable detector against the recorded preview."""

import json
import os
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "artifacts/models/huggingface")

import torch

from mtmc.detection.rfdetr import RFDETRPersonDetector
from mtmc.video.replay import synchronized_replay

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

reference = json.loads(
    (ROOT / "docs/detector/first_preview.json").read_text()
)
manifest = json.loads(
    (ROOT / "configs/datasets/scene_001_video_manifest.json").read_text()
)
sources = {
    item["camera"]: ROOT / item["local_path"]
    for item in manifest["files"]
}

detector = RFDETRPersonDetector(
    weights=ROOT / reference["weights_path"],
    threshold=reference["configuration"]["threshold"],
)

with synchronized_replay(sources) as batches:
    batch = next(islice(batches, 300, 301))

results = detector.detect(batch)
expected_counts = {
    item["camera"]: item["TP"] + item["FP"]
    for item in reference["matching"]["cameras"]
}
if tuple(result.camera_id for result in results) != tuple(sorted(sources)):
    raise RuntimeError("Unexpected camera order")

for result in results:
    count = len(result.xyxy)
    print(
        f"Camera {result.camera_id}: frame={result.frame_index}, "
        f"time={float(result.timestamp):.3f}s, persons={count}, "
        f"boxes={result.xyxy.shape}, scores={result.confidence.shape}"
    )
    if count != expected_counts[result.camera_id]:
        raise RuntimeError("Detection count differs from the recorded preview")

print("Detector adapter smoke test: PASSED")
