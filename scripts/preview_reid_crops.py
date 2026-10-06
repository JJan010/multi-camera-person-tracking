"""Inspect original-resolution RGB crops from recorded local tracks."""

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CAMERAS = (4, 5, 8)
FPS, WIDTH, HEIGHT = 30, 1920, 1080


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def crop_geometry(xyxy, width, height):
    box = np.asarray(xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Expected a finite xyxy box")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Original box must have positive width and height")
    clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    area = (box[2] - box[0]) * (box[3] - box[1])
    clipped_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    inside_fraction = float(clipped_area / area)
    if clipped_area <= 0:
        return None, inside_fraction
    left, top = np.floor(clipped[:2]).astype(int)
    right, bottom = np.ceil(clipped[2:]).astype(int)
    return (int(left), int(top), int(right), int(bottom)), inside_fraction


def check_geometry():
    bounds, fraction = crop_geometry([1.2, 2.8, 4.1, 7.0], 10, 10)
    if bounds != (1, 2, 5, 7) or fraction != 1:
        raise AssertionError("Incorrect fractional crop coordinates")
    bounds, fraction = crop_geometry([-2, -2, 2, 2], 10, 10)
    if bounds != (0, 0, 2, 2) or fraction != 0.25:
        raise AssertionError("Incorrect crop clipping")
    if crop_geometry([12, 1, 15, 5], 10, 10) != (None, 0):
        raise AssertionError("Fully outside box must be skipped")
    pixels = np.arange(10 * 10 * 3, dtype=np.uint16).reshape(10, 10, 3)
    left, top, right, bottom = crop_geometry([1, 2, 4, 7], 10, 10)[0]
    crop = pixels[top:bottom, left:right]
    if crop.shape != (5, 3, 3) or not np.array_equal(crop[0, 0], pixels[2, 1]):
        raise AssertionError("Incorrect crop axis order")
    print("Crop geometry checks: PASSED")


def read_rgb(path, frame_index):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        stream.codec_context.thread_type = "SLICE"
        for index, frame in enumerate(container.decode(stream)):
            if index != frame_index:
                continue
            if frame.pts is None or frame.time_base is None:
                raise RuntimeError(f"Missing timestamp: {path}")
            timestamp = frame.pts * Fraction(frame.time_base)
            if timestamp != Fraction(frame_index, FPS):
                raise RuntimeError(f"Unexpected timestamp: {path}, {timestamp}")
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (HEIGHT, WIDTH, 3) or rgb.dtype != np.uint8:
                raise RuntimeError(f"Unexpected image format: {path}")
            return rgb, {"pts": frame.pts, "time_base": str(frame.time_base),
                         "timestamp_seconds": float(timestamp)}
    raise RuntimeError(f"Video ended before frame {frame_index}: {path}")


def contact_sheet(entries, output, camera, frame_index):
    columns, tile_width, tile_height, header = 6, 180, 320, 44
    rows = max(1, math.ceil(len(entries) / columns))
    sheet = Image.new("RGB", (columns * tile_width, header + rows * tile_height), "#20242b")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 14) if font_path.exists() else ImageFont.load_default()
    draw.text((12, 12), f"Camera {camera} | frame {frame_index} | local IDs | preview only",
              fill="white", font=font)
    for index, item in enumerate(entries):
        x = (index % columns) * tile_width
        y = header + (index // columns) * tile_height
        with Image.open(output / item["crop_path"]) as crop:
            thumbnail = ImageOps.contain(crop, (128, 256), Image.Resampling.LANCZOS)
        sheet.paste(thumbnail, (x + (tile_width - thumbnail.width) // 2, y))
        draw.text((x + 8, y + 260), f"cam{camera}:{item['local_id']}  score={item['confidence']:.2f}",
                  fill="white", font=font)
        draw.text((x + 8, y + 278), f"crop {item['width']}x{item['height']} px", fill="white", font=font)
        draw.text((x + 8, y + 296), f"inside={item['inside_image_fraction']:.1%}",
                  fill="#ffd580" if item["inside_image_fraction"] < 1 else "#b8d9c4", font=font)
    path = output / f"camera_{camera:04d}_contact_sheet.jpg"
    sheet.save(path, quality=95)
    return path.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=int, default=150)
    parser.add_argument("--tracks", type=Path, default=ROOT / "artifacts/tracking_preview/tracks.jsonl")
    parser.add_argument("--video-root", type=Path, default=ROOT / (
        "data/physicalai_smartspaces/MTMC_Tracking_2024/train/scene_001"
    ))
    args = parser.parse_args()
    check_geometry()
    with args.tracks.open(encoding="utf-8") as handle:
        trace = [json.loads(line) for line in handle]
    matches = [row for row in trace if row["frame_index"] == args.frame]
    if args.frame < 0 or len(matches) != 1:
        raise ValueError("Requested frame must occur exactly once in the trace")
    row = matches[0]
    if not math.isclose(row["timestamp_seconds"], args.frame / FPS, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Unexpected trace timestamp")
    if sorted(item["camera"] for item in row["cameras"]) != list(CAMERAS):
        raise ValueError("Expected exactly cameras 4, 5, 8")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts/reid_crops" / stamp
    output.mkdir(parents=True, exist_ok=False)
    records, cameras = [], []
    for item in sorted(row["cameras"], key=lambda value: value["camera"]):
        camera = item["camera"]
        ids, boxes, scores = item["local_ids"], item["xyxy"], item["confidence"]
        if (any(type(identity) is not int or identity < 0 for identity in ids)
                or len(set(ids)) != len(ids) or not len(ids) == len(boxes) == len(scores)):
            raise ValueError(f"Invalid observation IDs or array lengths: camera {camera}")
        if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
            raise ValueError(f"Invalid confidence: camera {camera}")
        video = args.video_root / f"camera_{camera:04d}/video.mp4"
        rgb, timing = read_rgb(video, args.frame)
        folder = output / f"camera_{camera:04d}"
        folder.mkdir()
        entries, partial, skipped = [], 0, 0
        for identity, box, score in sorted(zip(ids, boxes, scores), key=lambda value: value[0]):
            bounds, inside = crop_geometry(box, WIDTH, HEIGHT)
            record = {"camera": camera, "local_id": identity, "frame_index": args.frame,
                      "timestamp_seconds": args.frame / FPS, "confidence": score,
                      "source_xyxy": box, "crop_xyxy_int": bounds, "inside_image_fraction": inside}
            if bounds is None:
                record.update(status="skipped_fully_outside", crop_path=None)
                records.append(record)
                skipped += 1
                continue
            left, top, right, bottom = bounds
            crop = Image.fromarray(rgb[top:bottom, left:right])
            path = folder / f"frame_{args.frame:06d}_local_{identity:06d}.png"
            crop.save(path)
            record.update(status="saved", crop_path=path.relative_to(output).as_posix(),
                          width=crop.width, height=crop.height)
            records.append(record)
            entries.append(record)
            partial += int(inside < 1)
        sheet = contact_sheet(entries, output, camera, args.frame)
        camera_report = {"camera": camera, "video_path": str(video.resolve()), **timing,
                         "observations": len(ids), "saved_crops": len(entries),
                         "partially_outside": partial, "fully_outside": skipped,
                         "contact_sheet": sheet}
        cameras.append(camera_report)
        print(f"Camera {camera}: observations={len(ids)}, saved={len(entries)}, "
              f"partially_outside={partial}, fully_outside={skipped}")

    report = {
        "configuration": {"frame_index": args.frame, "cameras": CAMERAS, "fps": FPS,
                          "source_resolution_wh": [WIDTH, HEIGHT], "color": "RGB", "dtype": "uint8",
                          "crop_policy": "Clip; floor left/top; ceil right/bottom; no padding; no resize",
                          "selection": "All tracker output boxes except fully outside boxes",
                          "contact_sheet": "Aspect-preserving thumbnails, for visual inspection only",
                          "inside_image_fraction": "Clipped float box area / original float box area; not visibility"},
        "versions": {package: version(package) for package in ("av", "numpy", "pillow")},
        "tracks": {"path": str(args.tracks.resolve()), "sha256": sha256(args.tracks)},
        "script_sha256": sha256(Path(__file__)), "cameras": cameras, "crops": records,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {output / 'manifest.json'}")
    print(f"Contact sheets: {output}")
    print("Re-ID crop preview: PASSED")


if __name__ == "__main__":
    main()
