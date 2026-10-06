"""Render diagnostic GT overlays using zero-based video frame indices."""

import csv
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "data/physicalai_smartspaces/MTMC_Tracking_2024/train/scene_001"
OUTPUT = ROOT / "artifacts/dataset_audit/gt_preview"
CAMERAS = (4, 5, 8)
FRAMES = (2, 300, 1800)

OUTPUT.mkdir(parents=True, exist_ok=True)

# Read the GT once, retaining only the requested cameras and frames.
annotations = defaultdict(list)
with (SCENE / "ground_truth.txt").open(encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        fields = line.split()
        camera, person, frame = map(int, fields[:3])
        if camera in CAMERAS and frame in FRAMES:
            x, y, width, height = map(float, fields[3:7])
            annotations[camera, frame].append((person, x, y, width, height))

font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = (
    ImageFont.truetype(str(font_path), 20)
    if font_path.exists()
    else ImageFont.load_default()
)

summary = []
selection = "+".join(f"eq(n\\,{frame})" for frame in FRAMES)

for camera in CAMERAS:
    camera_dir = OUTPUT / f"camera_{camera:04d}"
    camera_dir.mkdir(parents=True, exist_ok=True)
    video = SCENE / f"camera_{camera:04d}/video.mp4"

    print(f"Decoding camera {camera}...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(video),
            "-map", "0:v:0",
            "-vf", f"select={selection}",
            "-vsync", "0",
            "-frames:v", str(len(FRAMES)),
            "-start_number", "1",
            str(camera_dir / "raw_%02d.png"),
        ],
        check=True,
    )

    for position, frame in enumerate(FRAMES, start=1):
        with Image.open(camera_dir / f"raw_{position:02d}.png") as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        image_width, image_height = image.size
        rows = annotations[camera, frame]
        partial = outside = 0

        for person, x, y, width, height in rows:
            # Clip only for rendering; source GT remains unchanged.
            left = max(0.0, x)
            top = max(0.0, y)
            right = min(float(image_width), x + width)
            bottom = min(float(image_height), y + height)

            if right <= left or bottom <= top:
                outside += 1
                continue

            clipped = (
                x < 0 or y < 0
                or x + width > image_width
                or y + height > image_height
            )
            partial += int(clipped)
            color = "orange" if clipped else "lime"
            draw.rectangle(
                (left, top, min(right, image_width - 1),
                 min(bottom, image_height - 1)),
                outline=color,
                width=3,
            )
            label = f"ID {person}"
            label_x = min(left + 3, max(0, image_width - 110))
            label_y = min(max(30, top), max(0, image_height - 26))
            bounds = draw.textbbox((label_x, label_y), label, font=font)
            draw.rectangle(bounds, fill="black")
            draw.text((label_x, label_y), label, font=font, fill=color)

        title = (
            f"Camera {camera} | video n={frame} | GT frame={frame} | "
            f"GT={len(rows)} partial={partial} outside={outside}"
        )
        draw.rectangle((0, 0, image_width, 29), fill="black")
        draw.text((5, 2), title, font=font, fill="white")

        target = camera_dir / f"frame_{frame:06d}_gt.jpg"
        image.save(target, quality=95)
        summary.append(
            {
                "camera": camera,
                "video_frame_zero_based": frame,
                "gt_frame": frame,
                "gt_boxes": len(rows),
                "partially_outside": partial,
                "fully_outside": outside,
            }
        )
        print(f"  {target.name}: {summary[-1]}", flush=True)

with (OUTPUT / "summary.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print(f"\nPreview directory: {OUTPUT}")
