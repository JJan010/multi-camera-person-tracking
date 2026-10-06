"""Run a sequential RF-DETR + ByteTrack + OSNet baseline on scene_001."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

import numpy as np

from mtmc.reid.crops import build_person_crops
from mtmc.reid.osnet import sha256

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("replay_ms", "detect_ms", "track_ms", "crop_ms", "encode_ms", "core_ms", "record_ms")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
            and np.all(values >= 0), "Invalid timing samples")
    return {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95))}


def git_output(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def process_round(batches, detector, trackers, encoder, synchronize):
    """Process one causal round; synchronize the detector timing boundary.

    The encoder returns CPU-ready features, so its host call already waits for
    the required CUDA work. No file I/O is inside the measured core interval.
    """
    t0 = perf_counter_ns()
    batch = next(batches)
    t1 = perf_counter_ns()
    detections = detector.detect(batch)
    synchronize()
    t2 = perf_counter_ns()
    tracks = tuple(trackers[item.camera_id].update(item) for item in detections)
    t3 = perf_counter_ns()
    prepared = build_person_crops(batch, tracks)
    t4 = perf_counter_ns()
    features = encoder.encode(prepared.crops)
    t5 = perf_counter_ns()
    timing = dict(zip(PHASES[:6], (
        (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6,
        (t4 - t3) / 1e6, (t5 - t4) / 1e6, (t5 - t0) / 1e6)))
    require(features.keys == tuple(c.key for c in prepared.crops)
            and features.timestamps == tuple(c.timestamp for c in prepared.crops),
            "Encoder changed observation mapping")
    require(features.embeddings.shape == (len(prepared.crops), 512)
            and features.embeddings.dtype == np.float32, "Unexpected embedding matrix")
    return batch, detections, tracks, prepared, features, timing


def trace_record(run_id, batch, detections, tracks, prepared, features, row_offset):
    """Store only small metadata; each embedding row is scoped to this run."""
    row_by_key = {key: row_offset + i for i, key in enumerate(features.keys)}
    detection_by_camera = {item.camera_id: item for item in detections}
    cameras = []
    for item in sorted(tracks, key=lambda track: track.camera_id):
        raw = detection_by_camera[item.camera_id]
        cameras.append({
            "camera": item.camera_id, "detector_xyxy": raw.xyxy.tolist(),
            "detector_confidence": raw.confidence.tolist(), "local_ids": item.local_ids.tolist(),
            "xyxy": item.xyxy.tolist(), "confidence": item.confidence.tolist(),
        })
    observations = []
    for item in prepared.records:
        encoded = item.crop_xyxy_int is not None
        require((item.key in row_by_key) == encoded, "Crop/embedding coverage differs")
        observations.append({
            "camera": item.key.camera_id, "local_id": item.key.local_id,
            "frame_index": item.key.frame_index, "confidence": item.confidence,
            "source_xyxy": item.source_xyxy, "crop_xyxy_int": item.crop_xyxy_int,
            "inside_image_fraction": item.inside_image_fraction,
            "status": "encoded" if encoded else "skipped_fully_outside",
            "embedding_row": row_by_key.get(item.key),
        })
    return {"run_id": run_id, "frame_index": batch.frame_index,
            "timestamp_seconds": float(batch.timestamp), "timestamp": str(batch.timestamp),
            "cameras": cameras, "reid_observations": observations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=300, help="Total rounds starting at frame 0")
    parser.add_argument("--warmup", type=int, default=30, help="Initial rounds excluded from timing summary")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.warmup < args.rounds or args.torch_threads < 1:
        parser.error("Require 0 <= warmup < rounds and positive torch-threads")

    os.environ["RF_HOME"] = str(ROOT / "artifacts/models/rfdetr")
    os.environ["HF_HOME"] = str(ROOT / "artifacts/models/huggingface")
    import torch
    torch.set_num_threads(args.torch_threads)
    require(torch.cuda.is_available(), "CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    import av
    from importlib.metadata import version
    from mtmc.detection.rfdetr import RFDETRPersonDetector
    from mtmc.reid.osnet import OSNetEncoder
    from mtmc.tracking.bytetrack import CameraTracker, TRACKER_SETTINGS
    from mtmc.video.replay import synchronized_replay

    video_manifest_path = ROOT / "configs/datasets/scene_001_video_manifest.json"
    video_manifest = json.loads(video_manifest_path.read_text(encoding="utf-8"))
    reference_path = ROOT / "docs/detector/first_preview.json"
    detector_reference = json.loads(reference_path.read_text(encoding="utf-8"))
    weights = ROOT / detector_reference["weights_path"]
    # Previous scripts use either a top-level hash or configuration.weights_sha256.
    declared_hashes = [value for value in (
        detector_reference.get("weights_sha256"),
        detector_reference.get("configuration", {}).get("weights_sha256")) if value is not None]
    require(bool(declared_hashes) and len(set(declared_hashes)) == 1,
            "Missing or conflicting detector weight hashes in first_preview.json")
    weight_hash = sha256(weights)
    require(weight_hash == declared_hashes[0], "Detector weights differ from the reference")
    sources = {}
    require(sorted(item["camera"] for item in video_manifest["files"]) == [4, 5, 8],
            "Expected exactly cameras 4, 5, 8")
    print("Verifying video checksums...", flush=True)
    for item in video_manifest["files"]:
        path = ROOT / item["local_path"]
        require(path.stat().st_size == item["size_bytes"] and sha256(path) == item["sha256"],
                f"Video differs from manifest: {path}")
        sources[item["camera"]] = path

    print("Loading persistent RF-DETR and OSNet models...", flush=True)
    started = perf_counter_ns()
    detector = RFDETRPersonDetector(weights, threshold=0.1)
    detector_initialized = perf_counter_ns()
    osnet_config_path = ROOT / "configs/models/osnet_x1_0_msmt17.json"
    encoder = OSNetEncoder(osnet_config_path, project_root=ROOT)
    torch.cuda.synchronize()
    encoder_initialized = perf_counter_ns()
    trackers = {camera: CameraTracker(camera) for camera in sources}

    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts/local_reid" / run_id
    output.mkdir(parents=True, exist_ok=False)
    trace_path = output / "tracks.jsonl"
    feature_chunks, samples = [], []
    seen_ids = {camera: set() for camera in sources}
    total_embeddings, total_skipped = 0, 0
    measured_started, measured_finished = None, None
    peak_allocated, peak_reserved = 0, 0
    print(f"Processing {args.rounds} rounds; initial {args.warmup} excluded from timing summary...", flush=True)
    with synchronized_replay(sources, fps=30, threads=1) as batches, trace_path.open("w", encoding="utf-8") as trace:
        for index in range(args.rounds):
            if index == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                measured_started = perf_counter_ns()
            batch, detections, tracks, prepared, features, timing = process_round(
                batches, detector, trackers, encoder, torch.cuda.synchronize)
            require(batch.frame_index == index, "Replay skipped or repeated a frame")
            record_started = perf_counter_ns()
            record = trace_record(run_id, batch, detections, tracks, prepared, features, total_embeddings)
            trace.write(json.dumps(record, allow_nan=False) + "\n")
            feature_chunks.append(features.embeddings)  # CPU arrays only; no RGB views retained.
            count = len(features.keys)
            skipped = sum(item.crop_xyxy_int is None for item in prepared.records)
            total_embeddings += count
            total_skipped += skipped
            for item in tracks:
                seen_ids[item.camera_id].update(int(v) for v in item.local_ids)
            timing["record_ms"] = (perf_counter_ns() - record_started) / 1e6
            samples.append({"frame_index": index, "measured": index >= args.warmup,
                            "crops": count, "skipped_fully_outside": skipped, **timing})
            # Drop caller references to full images before requesting the next batch.
            del batch, detections, tracks, prepared, features, record
            if index + 1 == args.rounds:
                measured_finished = perf_counter_ns()
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**2
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**2
            if (index + 1) % 60 == 0 or index + 1 == args.rounds:
                print(f"Processed {index + 1}/{args.rounds}; embeddings: {total_embeddings}", flush=True)

    measured = [sample for sample in samples if sample["measured"]]
    elapsed_s = (measured_finished - measured_started) / 1e9
    measured_count = len(measured)
    summary = {
        "rounds": args.rounds, "images": args.rounds * len(sources), "warmup_rounds": args.warmup,
        "measured_frame_range": [args.warmup, args.rounds - 1], "measured_rounds": measured_count,
        "total_embeddings": total_embeddings, "skipped_fully_outside": total_skipped,
        "distinct_local_ids": {camera: len(ids) for camera, ids in seen_ids.items()},
        "measured_elapsed_s": elapsed_s, "rounds_per_second": measured_count / elapsed_s,
        "images_per_second": measured_count * len(sources) / elapsed_s,
        "crops_per_second": sum(s["crops"] for s in measured) / elapsed_s,
        "crops_per_measured_round": {"min": min(s["crops"] for s in measured),
                                     "mean": float(np.mean([s["crops"] for s in measured])),
                                     "max": max(s["crops"] for s in measured)},
        "timings": {phase: stats([sample[phase] for sample in measured]) for phase in PHASES},
        "peak_torch_allocated_mib": peak_allocated, "peak_torch_reserved_mib": peak_reserved,
    }
    features_path = output / "embeddings.npy"
    embeddings = np.concatenate(feature_chunks, axis=0)
    require(embeddings.shape == (total_embeddings, 512), "Final embedding matrix shape mismatch")
    require(np.isfinite(embeddings).all() and np.allclose(np.linalg.norm(embeddings, axis=1), 1,
                                                       rtol=0, atol=1e-5), "Invalid final embedding matrix")
    np.save(features_path, embeddings, allow_pickle=False)
    paths = ["src/mtmc/video/reader.py", "src/mtmc/video/replay.py", "src/mtmc/detection/rfdetr.py",
             "src/mtmc/tracking/bytetrack.py", "src/mtmc/reid/crops.py", "src/mtmc/reid/osnet.py",
             "scripts/run_local_reid.py"]
    report = {
        "schema_version": 1, "run_id": run_id, "created_utc": now.isoformat(), "completed": True,
        "summary": summary, "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--porcelain"),
        "source_sha256": {path: sha256(ROOT / path) for path in paths},
        "video_manifest": {"path": str(video_manifest_path), "sha256": sha256(video_manifest_path),
                           "content": video_manifest},
        "detector_reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "configuration": {"fps": 30, "cameras": sorted(sources), "detector_threshold": 0.1,
                          "detector_weights_sha256": weight_hash, "tracker": TRACKER_SETTINGS,
                          "osnet": encoder.configuration, "osnet_config_sha256": sha256(osnet_config_path),
                          "precision": "FP32", "tf32": False, "cudnn_benchmark": False,
                          "reid_policy": "All visible track crops every round; only fully outside boxes skipped",
                          "detector_batch_size": len(sources), "reid_batch": "All camera crops in one call",
                          "decoder_threads_per_camera": 1, "schedule": "Sequential, no dropped frames",
                          "repeats": 1, "rendering": False, "ground_truth_used": False,
                          "torch_cpu_threads": torch.get_num_threads(),
                          "torch_interop_threads": torch.get_num_interop_threads()},
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "gpu": torch.cuda.get_device_name(0), "cuda_build": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version(), "ffmpeg_libraries": av.library_versions,
                    "versions": {p: version(p) for p in ("torch", "torchvision", "rfdetr", "supervision",
                                                        "av", "numpy", "pillow")}},
        "initialization_ms": {"detector_constructor": (detector_initialized - started) / 1e6,
                              "osnet_constructor": (encoder_initialized - detector_initialized) / 1e6},
        "timing_protocol": {
            "clock": "perf_counter_ns; milliseconds",
            "detect_ms": "Adapter wall time plus CUDA synchronization; includes pre/postprocessing",
            "encode_ms": "Complete OSNet adapter including CPU preprocessing, transfers and CPU-ready features",
            "core_ms": "Replay + detect + track + crop + encode, excluding record serialization",
            "record_ms": "Metadata construction, buffered JSONL write, CPU feature retention and counters",
            "throughput": "Measured loop wall time including record work and intermediate progress prints",
            "excluded": "Input verification, constructors, replay open, trace close/final flush, final NPY/report writes",
            "warmup": "First rounds processed and saved normally; tracker state continues into measured rounds",
            "filesystem_cache": "Uncontrolled; video checksum verification reads all source files before replay",
            "limits": "Single integration run with variable crop counts, not a definitive backend benchmark",
        },
        "samples": samples,
        "artifacts": {"tracks": {"path": trace_path.name, "sha256": sha256(trace_path)},
                      "embeddings": {"path": features_path.name, "sha256": sha256(features_path),
                                     "shape": list(embeddings.shape), "dtype": str(embeddings.dtype)}},
        "note": "Local IDs are camera/run scoped, not global identities or unique-person counts.",
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    print(f"Report: {report_path}")
    print(f"Tracks: {trace_path}")
    print(f"Embeddings: {features_path}")
    print("Local tracking + Re-ID pipeline: COMPLETED")


if __name__ == "__main__":
    main()
