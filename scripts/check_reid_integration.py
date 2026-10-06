"""Check replay -> recorded tracks -> in-memory crops -> OSNet against a snapshot."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from mtmc.reid.crops import build_person_crops
from mtmc.reid.osnet import ObservationKey, OSNetEncoder, sha256

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-5


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_file(path, expected_hash):
    require(sha256(path) == expected_hash, f"Checksum mismatch: {path}")


def record_key(record):
    return ObservationKey(record["camera"], record["local_id"], record["frame_index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, help="Override trace path; hash must match reference")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")

    import torch
    torch.set_num_threads(args.torch_threads)
    from mtmc.tracking.bytetrack import CameraTracks
    from mtmc.video.replay import synchronized_replay

    reference_path = args.reference.resolve()
    reference = load_json(reference_path)
    crop_manifest_path = Path(reference["source_manifest"]["path"])
    verify_file(crop_manifest_path, reference["source_manifest"]["sha256"])
    crop_manifest = load_json(crop_manifest_path)
    require(reference["source_tracks"] == crop_manifest["tracks"], "Source trace provenance differs")
    tracks_path = (args.tracks or Path(crop_manifest["tracks"]["path"])).resolve()
    verify_file(tracks_path, crop_manifest["tracks"]["sha256"])
    feature_path = reference_path.parent / reference["embeddings"]["path"]
    verify_file(feature_path, reference["embeddings"]["sha256"])
    expected = np.load(feature_path, allow_pickle=False)
    records = reference["records"]
    require(bool(records) and [r["embedding_row"] for r in records] == list(range(len(records))),
            "Invalid reference row mapping")
    require(expected.shape == (len(records), 512) and expected.dtype == np.float32
            and np.isfinite(expected).all()
            and np.allclose(np.linalg.norm(expected, axis=1), 1, rtol=0, atol=ATOL),
            "Invalid reference embeddings")
    config_path = ROOT / "configs/models/osnet_x1_0_msmt17.json"
    require(load_json(config_path) == reference["model_configuration"], "Model configuration changed")

    configuration = crop_manifest["configuration"]
    frame_index, fps = configuration["frame_index"], configuration["fps"]
    require(type(frame_index) is int and frame_index >= 0 and type(fps) is int and fps > 0,
            "Invalid reference frame/fps")
    timestamp = Fraction(frame_index, fps)
    cameras = sorted(configuration["cameras"])
    require(cameras == [4, 5, 8], "This check expects scene_001 cameras 4, 5, 8")
    with tracks_path.open(encoding="utf-8") as handle:
        rows = [row for line in handle if (row := json.loads(line))["frame_index"] == frame_index]
    require(len(rows) == 1, "Reference frame must occur exactly once in trace")
    row = rows[0]
    require(math.isclose(row["timestamp_seconds"], float(timestamp), rel_tol=0, abs_tol=1e-9),
            "Trace timestamp mismatch")
    require(sorted(r["camera"] for r in row["cameras"]) == cameras, "Trace cameras differ")
    tracks = []
    for item in row["cameras"]:
        require(all(type(v) is int and v >= 0 for v in item["local_ids"]), "Invalid trace local IDs")
        tracks.append(CameraTracks(
            camera_id=item["camera"], frame_index=frame_index, timestamp=timestamp,
            local_ids=np.asarray(item["local_ids"], dtype=np.int64),
            # Preserve the numbers from the JSON trace without additional float32 rounding.
            xyxy=np.asarray(item["xyxy"], dtype=np.float64).reshape(-1, 4),
            confidence=np.asarray(item["confidence"], dtype=np.float64)))

    video_manifest_path = ROOT / "configs/datasets/scene_001_video_manifest.json"
    video_manifest = load_json(video_manifest_path)
    sources = {}
    print("Verifying source trace, embeddings and video checksums...", flush=True)
    require(sorted(item["camera"] for item in video_manifest["files"]) == cameras,
            "Video manifest cameras differ")
    for item in video_manifest["files"]:
        path = ROOT / item["local_path"]
        require(path.stat().st_size == item["size_bytes"], f"Video size mismatch: {path}")
        verify_file(path, item["sha256"])
        sources[item["camera"]] = path
    print(f"Replaying through frame {frame_index} ({float(timestamp):.3f} s)...", flush=True)
    with synchronized_replay(sources, fps=fps, threads=1) as batches:
        for _ in range(frame_index + 1):
            batch = next(batches)
    width, height = configuration["source_resolution_wh"]
    require(all(frame.rgb.shape == (height, width, 3) for frame in batch.frames),
            "Decoded resolution differs from the reference")

    prepared = build_person_crops(batch, tracks)
    by_key = {crop.key: crop for crop in prepared.crops}
    metadata = {item.key: item for item in prepared.records}
    source_records = crop_manifest["crops"]
    require(len(metadata) == len(source_records) and set(metadata) == {record_key(r) for r in source_records},
            "Observation coverage differs from crop manifest")
    for original in source_records:
        item = metadata[record_key(original)]
        bounds = list(item.crop_xyxy_int) if item.crop_xyxy_int is not None else None
        require(bounds == original["crop_xyxy_int"]
                and list(item.source_xyxy) == original["source_xyxy"]
                and item.confidence == original["confidence"]
                and math.isclose(item.inside_image_fraction, original["inside_image_fraction"],
                                 rel_tol=0, abs_tol=1e-12), "Crop geometry/metadata differs")
        require(original["status"] == ("saved" if bounds is not None else "skipped_fully_outside"),
                "Crop selection differs")
    require(len(by_key) == len(records) and set(by_key) == {record_key(r) for r in records},
            "Reference crop keys differ")

    # PNGs are used ONLY as verification fixtures; encoder inputs come from replay.
    crops = []
    for record in records:
        original = source_records[record["source_manifest_index"]]
        require(all(record[k] == v for k, v in original.items()), "Embedding crop provenance differs")
        crop = by_key[record_key(record)]
        require(crop.timestamp == timestamp and math.isclose(
            float(timestamp), record["timestamp_seconds"], rel_tol=0, abs_tol=1e-9),
            "Crop timestamp differs")
        path = crop_manifest_path.parent / record["crop_path"]
        verify_file(path, record["crop_sha256"])
        with Image.open(path) as image:
            require(image.mode == "RGB" and image.size == (record["width"], record["height"]),
                    "Reference PNG format differs")
            require(np.array_equal(crop.rgb, np.asarray(image)), f"RGB pixel mismatch: {crop.key}")
        crops.append(crop)
    counts = dict(sorted(Counter(c.key.camera_id for c in crops).items()))
    print(f"In-memory crops: {len(crops)}; cameras: {counts}")
    print(f"RGB pixel parity against saved PNGs: {len(crops)}/{len(crops)} EXACT")

    reversed_batch = type(batch)(batch.frame_index, batch.timestamp, tuple(reversed(batch.frames)))
    reordered = build_person_crops(reversed_batch, list(reversed(tracks)))
    require([c.key for c in reordered.crops] == [c.key for c in prepared.crops], "Order changed crop keys")
    require(all(np.array_equal(a.rgb, b.rgb) for a, b in zip(reordered.crops, prepared.crops)),
            "Order changed crop pixels")
    print("Changed camera/track input order preserves crop mapping: OK")

    encoder = OSNetEncoder(config_path, project_root=ROOT)
    calls = []
    hook = encoder.model.register_forward_pre_hook(lambda _model, _inputs: calls.append(1))
    try:
        result = encoder.encode(crops)
    finally:
        hook.remove()
    require(len(calls) == 1, "Expected one OSNet call for all cameras")
    require(result.keys == tuple(record_key(r) for r in records)
            and result.timestamps == (timestamp,) * len(records), "Embedding mapping differs")
    require(result.embeddings.shape == expected.shape and result.embeddings.dtype == np.float32,
            "Embedding output shape/dtype differs")
    error = float(np.max(np.abs(result.embeddings - expected)))
    cosine = float(np.min(np.clip(np.sum(result.embeddings * expected, axis=1), -1, 1)))
    require(np.isfinite(error) and np.isfinite(cosine) and error <= ATOL and cosine >= 1 - ATOL,
            f"Embedding parity failed: max_abs={error}, min_cosine={cosine}")
    print(f"Embedding parity: max_abs={error:.9g}, min_cosine={cosine:.9f}")
    print("One OSNet batch; observation keys and timestamps preserved: OK")

    now = datetime.now(timezone.utc)
    output = ROOT / "artifacts/reid_integration_checks" / now.strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    features_path = output / "embeddings.npy"
    np.save(features_path, result.embeddings, allow_pickle=False)
    paths = ["src/mtmc/video/reader.py", "src/mtmc/video/replay.py", "src/mtmc/tracking/bytetrack.py",
             "src/mtmc/reid/crops.py", "src/mtmc/reid/osnet.py", "scripts/check_reid_integration.py"]
    report = {
        "schema_version": 1, "created_utc": now.isoformat(), "passed": True,
        "scope": "Replay + recorded local tracks + in-memory crops + OSNet; no detector rerun, no GT",
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "tracks": {"path": str(tracks_path), "sha256": sha256(tracks_path)},
        "video_manifest": {"path": str(video_manifest_path), "sha256": sha256(video_manifest_path)},
        "model_config_sha256": sha256(config_path), "source_sha256": {p: sha256(ROOT / p) for p in paths},
        "frame_index": frame_index, "timestamp": str(timestamp), "fps": fps,
        "crop_count": len(crops), "crops_per_camera": counts,
        "skipped_fully_outside": sum(r.crop_xyxy_int is None for r in prepared.records),
        "checks": {"exact_rgb_pixel_matches": len(crops), "input_reordering": True,
                   "model_calls": len(calls), "observation_mapping": True,
                   "max_absolute_error": error, "minimum_cosine": cosine},
        "tolerances": {"max_absolute_error": ATOL, "minimum_cosine": 1 - ATOL},
        "runtime": {"torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                    "device": str(encoder.device), "dtype": str(next(encoder.model.parameters()).dtype),
                    "gpu": torch.cuda.get_device_name(encoder.device), "cuda_build": torch.version.cuda},
        "versions": {p: version(p) for p in ("torch", "torchvision", "av", "numpy", "pillow", "supervision")},
        "embeddings": {"path": features_path.name, "sha256": sha256(features_path),
                       "shape": list(result.embeddings.shape), "dtype": str(result.embeddings.dtype)},
        "records": [{"embedding_row": i, "camera": c.key.camera_id, "local_id": c.key.local_id,
                     "frame_index": c.key.frame_index, "timestamp": str(c.timestamp),
                     "crop_xyxy_int": metadata[c.key].crop_xyxy_int} for i, c in enumerate(crops)],
        "note": "Correctness check for one frame, not an integrated performance benchmark.",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print("Re-ID in-memory integration: PASSED")


if __name__ == "__main__":
    main()
