"""Run pretrained RF-DETR Small on one synchronized three-camera round."""

import hashlib
import json
import os
from importlib.metadata import version
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Set cache locations before importing model libraries.
os.environ["RF_HOME"] = str(ROOT / "artifacts/models/rfdetr")
os.environ["HF_HOME"] = str(ROOT / "artifacts/models/huggingface")

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRSmall

from mtmc.video.replay import synchronized_replay

FRAME_INDEX = 300
THRESHOLD = 0.5
OUTPUT = ROOT / "artifacts/detector_preview"
OUTPUT.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this preview")

# Explicit FP32 baseline, without TF32 acceleration.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

manifest = json.loads(
    (ROOT / "configs/datasets/scene_001_video_manifest.json").read_text()
)
sources = {
    item["camera"]: ROOT / item["local_path"]
    for item in manifest["files"]
}

print(f"Reading synchronized frame {FRAME_INDEX}...", flush=True)
with synchronized_replay(sources) as batches:
    batch = next(islice(batches, FRAME_INDEX, FRAME_INDEX + 1))

print("Loading RF-DETR Small; first run may download weights...", flush=True)
model = RFDETRSmall(device="cuda")

network = model.model.model
parameter = next(network.parameters())
print("Model device:", parameter.device)
print("Model dtype:", parameter.dtype)
print("Input resolution:", model.model_config.resolution)

with torch.inference_mode():
    results = model.predict(
        [frame.rgb for frame in batch.frames],
        threshold=THRESHOLD,
        include_source_image=False,
    )
torch.cuda.synchronize()

# RF-DETR moves weights to the configured device on first predict().
network = model.model.model
parameter = next(network.parameters())
print("Model device after predict:", parameter.device)
print("Model dtype after predict:", parameter.dtype)

if any(p.device.type != "cuda" for p in network.parameters()):
    raise RuntimeError("Some model parameters are not on CUDA")
if any(p.dtype != torch.float32 for p in network.parameters() if p.is_floating_point()):
    raise RuntimeError("Expected FP32 model parameters")

if not isinstance(results, list) or len(results) != len(batch.frames):
    raise RuntimeError("Unexpected number of prediction results")

weights = Path(model.model_config.pretrain_weights)
digest = hashlib.sha256()
with weights.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

report = {
    "model": "RFDETRSmall",
    "rfdetr": version("rfdetr"),
    "torch": torch.__version__,
    "device": str(parameter.device),
    "dtype": str(parameter.dtype),
    "tf32": False,
    "resolution": model.model_config.resolution,
    "threshold": THRESHOLD,
    "weights_path": str(weights),
    "weights_sha256": digest.hexdigest(),
    "dataset_revision": manifest["revision"],
    "frame_index": batch.frame_index,
    "timestamp_seconds": float(batch.timestamp),
    "cameras": [],
}

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = (
    ImageFont.truetype(str(font_path), 20)
    if font_path.exists()
    else ImageFont.load_default()
)

for frame, detections in zip(batch.frames, results):
    names = detections.data.get("class_name")
    if names is None:
        raise RuntimeError("Predictions have no class-name mapping")

    # Use the model's class-name mapping, not an assumed numeric person ID.
    persons = detections[np.asarray(names) == "person"]
    if not np.isfinite(persons.xyxy).all():
        raise RuntimeError("Non-finite detection coordinates")

    image = Image.fromarray(frame.rgb)
    draw = ImageDraw.Draw(image)
    records = []

    for box, score, class_id in zip(
        persons.xyxy, persons.confidence, persons.class_id
    ):
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            raise RuntimeError("Nonpositive detection box")

        draw.rectangle((x1, y1, x2, y2), outline="cyan", width=3)
        label = f"person {float(score):.2f}"
        position = (
            max(0, min(x1, image.width - 150)),
            max(30, min(y1, image.height - 26)),
        )
        draw.rectangle(draw.textbbox(position, label, font=font), fill="black")
        draw.text(position, label, font=font, fill="cyan")
        records.append({
            "xyxy": [x1, y1, x2, y2],
            "confidence": float(score),
            "class_id": int(class_id),
        })

    title = (
        f"Camera {frame.camera_id} | frame={frame.frame_index} | "
        f"RF-DETR Small | persons={len(persons)} | threshold={THRESHOLD}"
    )
    draw.rectangle((0, 0, image.width, 29), fill="black")
    draw.text((5, 2), title, font=font, fill="white")

    target = OUTPUT / f"camera_{frame.camera_id:04d}_frame_{FRAME_INDEX:06d}.jpg"
    image.save(target, quality=95)
    report["cameras"].append({
        "camera": frame.camera_id,
        "person_count": len(persons),
        "detections": records,
    })
    print(f"Camera {frame.camera_id}: {len(persons)} persons; {target.name}")

report_path = OUTPUT / "predictions.json"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("Report:", report_path)
print("Detector preview: PASSED")
