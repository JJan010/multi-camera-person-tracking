"""Render ten seconds of per-camera tracking and save frame-level results."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / "artifacts/models/huggingface")

import torch
from PIL import Image, ImageDraw, ImageFont

from mtmc.detection.rfdetr import RFDETRPersonDetector
from mtmc.tracking.bytetrack import CameraTracker, TRACKER_SETTINGS
from mtmc.video.replay import synchronized_replay

ROUNDS = 300
THRESHOLD = 0.1
OUTPUT = ROOT / "artifacts/tracking_preview"
OUTPUT.mkdir(parents=True, exist_ok=True)

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
if sorted(sources) != [4, 5, 8]:
    raise RuntimeError("Expected cameras 4, 5 and 8")

detector = RFDETRPersonDetector(
    ROOT / reference["weights_path"],
    threshold=THRESHOLD,
)
trackers = {camera: CameraTracker(camera) for camera in sorted(sources)}
seen_ids = {camera: set() for camera in sources}

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = (
    ImageFont.truetype(str(font_path), 16)
    if font_path.exists()
    else ImageFont.load_default()
)

video_path = OUTPUT / "local_tracking.mp4"
command = [
    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
    "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
    "-r", "30", "-i", "pipe:0",
    "-an", "-c:v", "libx264", "-preset", "veryfast",
    "-crf", "23", "-pix_fmt", "yuv420p",
    "-threads", "1", "-movflags", "+faststart",
    str(video_path),
]

print("Processing 300 rounds; decoding, detection, tracking and rendering...", flush=True)

with (
    synchronized_replay(sources) as batches,
    (OUTPUT / "tracks.jsonl").open("w", encoding="utf-8") as trace,
    subprocess.Popen(command, stdin=subprocess.PIPE) as encoder,
):
    try:
        for index in range(ROUNDS):
            batch = next(batches)
            detections = detector.detect(batch)
            tracks = [
                trackers[item.camera_id].update(item)
                for item in detections
            ]

            canvas = Image.new("RGB", (1280, 720), color=(20, 20, 20))
            record = {
                "frame_index": index,
                "timestamp_seconds": float(batch.timestamp),
                "cameras": [],
            }

            for position, (frame, raw, result) in enumerate(zip(batch.frames, detections, tracks)):
                tile = Image.fromarray(frame.rgb).resize(
                    (640, 360), Image.Resampling.BILINEAR
                )
                draw = ImageDraw.Draw(tile)

                for box, local_id in zip(result.xyxy, result.local_ids):
                    x1, y1, x2, y2 = (float(value) / 3 for value in box)
                    color = "cyan"
                    draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
                    label = f"{result.camera_id}:{int(local_id)}"
                    location = (
                        max(0, min(x1, 570)),
                        max(25, min(y1, 338)),
                    )
                    draw.rectangle(
                        draw.textbbox(location, label, font=font), fill="black"
                    )
                    draw.text(location, label, font=font, fill=color)

                draw.rectangle((0, 0, 640, 24), fill="black")
                draw.text(
                    (5, 3),
                    f"Camera {result.camera_id} | visible tracks: {len(result.local_ids)}",
                    font=font, fill="white",
                )
                canvas.paste(tile, ((position % 2) * 640, (position // 2) * 360))

                seen_ids[result.camera_id].update(result.local_ids.tolist())
                record["cameras"].append({
                    "camera": result.camera_id,
                    "detector_xyxy": raw.xyxy.tolist(),
                    "detector_confidence": raw.confidence.tolist(),
                    "local_ids": result.local_ids.tolist(),
                    "xyxy": result.xyxy.tolist(),
                    "confidence": result.confidence.tolist(),
                })

            panel = ImageDraw.Draw(canvas)
            panel.multiline_text(
                (660, 390),
                f"Frame: {index}\n"
                f"Video time: {float(batch.timestamp):.3f} s\n"
                "Labels: camera_id:local_id\n"
                "RF-DETR Small FP32 + ByteTrack\n"
                "IDs are local to each camera.",
                font=font, fill="white", spacing=12,
            )

            trace.write(json.dumps(record) + "\n")
            encoder.stdin.write(canvas.tobytes())

            if (index + 1) % 60 == 0:
                print(f"Processed {index + 1}/{ROUNDS} rounds", flush=True)

        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("FFmpeg encoding failed")
    finally:
        if not encoder.stdin.closed:
            encoder.stdin.close()

summary = {
    "rounds": ROUNDS,
    "fps": 30,
    "detector_threshold": THRESHOLD,
    "tracker_settings": TRACKER_SETTINGS,
    "distinct_local_ids": {
        camera: len(ids) for camera, ids in seen_ids.items()
    },
    "note": "Distinct local IDs are not a count of unique people.",
}
(OUTPUT / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
print("Video:", video_path)
print("Tracking preview: PASSED")
