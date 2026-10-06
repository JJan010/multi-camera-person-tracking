"""Benchmark CPU preprocessing, CUDA OSNet forward and the complete FP32 adapter."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def statistics(samples):
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Timing samples must be finite positive milliseconds")
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95))}


def wall_benchmark(function, warmup, iterations, before_measure=None):
    for _ in range(warmup):
        value = function()
        del value
    if before_measure is not None:
        before_measure()
    samples = []
    loop_start = perf_counter_ns()
    for _ in range(iterations):
        start = perf_counter_ns()
        value = function()
        samples.append((perf_counter_ns() - start) / 1e6)
        del value
    elapsed = (perf_counter_ns() - loop_start) / 1e9
    return {"timings": statistics(samples), "elapsed_s": elapsed, "samples_ms": samples}


def cuda_forward_benchmark(encoder, crops, warmup, iterations):
    import torch

    inputs = encoder.prepare_inputs(crops).to(encoder.device, dtype=torch.float32)
    torch.cuda.synchronize(encoder.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Initialize the events outside measured iterations.
    start.record()
    end.record()
    end.synchronize()
    samples = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for _ in range(warmup):
            output = encoder.model(inputs)
            del output
        torch.cuda.synchronize(encoder.device)
        torch.cuda.reset_peak_memory_stats(encoder.device)
        loop_start = perf_counter_ns()
        for _ in range(iterations):
            start.record()
            output = encoder.model(inputs)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
            del output
        elapsed = (perf_counter_ns() - loop_start) / 1e9
    return {"timings": statistics(samples), "host_loop_elapsed_s": elapsed,
            "samples_ms": samples, **memory_stats(encoder.device)}


def memory_stats(device):
    import torch
    return {"peak_torch_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_torch_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}


def load_reference(path):
    from mtmc.reid.osnet import ObservationKey, PersonCrop

    reference = json.loads(path.read_text(encoding="utf-8"))
    feature_path = path.parent / reference["embeddings"]["path"]
    if sha256(feature_path) != reference["embeddings"]["sha256"]:
        raise ValueError("Reference feature checksum mismatch")
    expected = np.load(feature_path, allow_pickle=False)
    crop_manifest_path = Path(reference["source_manifest"]["path"])
    if sha256(crop_manifest_path) != reference["source_manifest"]["sha256"]:
        raise ValueError("Crop manifest checksum mismatch")
    source = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    records = reference["records"]
    if (not records or [r["embedding_row"] for r in records] != list(range(len(records)))
            or expected.shape != (len(records), 512) or expected.dtype != np.float32
            or not np.isfinite(expected).all()
            or not np.allclose(np.linalg.norm(expected, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError("Invalid reference features or row mapping")
    crops = []
    for record in records:
        original = source["crops"][record["source_manifest_index"]]
        if any(record[key] != value for key, value in original.items()):
            raise ValueError("Reference differs from source crop observation")
        crop_path = crop_manifest_path.parent / record["crop_path"]
        if sha256(crop_path) != record["crop_sha256"]:
            raise ValueError(f"Crop checksum mismatch: {crop_path}")
        with Image.open(crop_path) as image:
            if image.mode != "RGB" or image.size != (record["width"], record["height"]):
                raise ValueError(f"Invalid RGB crop: {crop_path}")
            rgb = np.array(image, dtype=np.uint8, copy=True)
        timestamp = Fraction(record["frame_index"], source["configuration"]["fps"])
        if abs(float(timestamp) - record["timestamp_seconds"]) > 1e-9:
            raise ValueError("Unexpected source timestamp")
        crops.append(PersonCrop(ObservationKey(record["camera"], record["local_id"],
                                              record["frame_index"]), timestamp, rgb))
    return reference, crops, expected


def command_metadata(arguments):
    try:
        result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, timeout=10)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Saved embedding manifest.json")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=0,
                        help="0 preserves PyTorch default; a positive value sets intra-op CPU threads")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.repeats) < 1 or args.torch_threads < 0:
        parser.error("warmup/iterations/repeats must be positive; torch-threads must be >= 0")

    import torch
    from mtmc.reid.osnet import OSNetEncoder

    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    reference_path = args.reference.resolve()
    print("Loading and verifying crops into RAM...", flush=True)
    reference, crops, expected = load_reference(reference_path)
    config_path = ROOT / "configs/models/osnet_x1_0_msmt17.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != reference["model_configuration"]:
        raise ValueError("Model configuration differs from reference")
    print(f"Batch: {len(crops)} crops; cameras: {dict(sorted(Counter(c.key.camera_id for c in crops).items()))}")
    print(f"CPU torch threads: {torch.get_num_threads()}; interop: {torch.get_num_interop_threads()}")
    print(f"Per phase: warmup={args.warmup}, iterations={args.iterations}, repeats={args.repeats}", flush=True)

    before = perf_counter_ns()
    encoder = OSNetEncoder(config_path, project_root=ROOT)
    torch.cuda.synchronize(encoder.device)
    initialization_ms = (perf_counter_ns() - before) / 1e6
    before = perf_counter_ns()
    first = encoder.encode(crops)
    first_encode_ms = (perf_counter_ns() - before) / 1e6
    error = float(np.max(np.abs(first.embeddings - expected)))
    cosine = float(np.min(np.clip(np.sum(first.embeddings * expected, axis=1), -1, 1)))
    if (first.keys != tuple(c.key for c in crops) or first.timestamps != tuple(c.timestamp for c in crops)
            or not np.isfinite(error) or error > 1e-5 or not np.isfinite(cosine) or cosine < 1 - 1e-5):
        raise RuntimeError(f"Reference parity failed: max_abs={error}, min_cosine={cosine}")
    del first
    print(f"Initialization: {initialization_ms:.3f} ms; first encode: {first_encode_ms:.3f} ms")
    print(f"Reference parity: max_abs={error:.9g}, min_cosine={cosine:.9g}", flush=True)

    props = torch.cuda.get_device_properties(encoder.device)
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                          if line.startswith("model name")), "unknown")
    provenance = {
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "script_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(ROOT / "src/mtmc/reid/osnet.py"),
        "model_configuration": config,
        "configuration_sha256": sha256(config_path),
        "versions": {"python": platform.python_version(), **{
            p: version(p) for p in ("torch", "torchvision", "numpy", "pillow")}},
        "hardware": {"gpu": props.name, "gpu_memory_mib": props.total_memory / 2**20,
                     "compute_capability": [props.major, props.minor], "cuda_build": torch.version.cuda,
                     "cudnn_version": torch.backends.cudnn.version(), "cpu": cpu_model,
                     "logical_cpu_count": os.cpu_count(), "platform": platform.platform()},
        "threads": {"torch_intraop": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                    "environment": {name: os.environ.get(name) for name in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "git_head": command_metadata(["git", "rev-parse", "HEAD"]),
        "git_status": command_metadata(["git", "status", "--short", "--branch"]),
        "nvidia_smi": command_metadata(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                        "--format=csv,noheader"]),
    }
    results = []
    for repeat in range(1, args.repeats + 1):
        print(f"Run {repeat}/{args.repeats}: CPU preprocessing...", flush=True)
        cpu = wall_benchmark(lambda: encoder.prepare_inputs(crops), args.warmup, args.iterations)
        print(f"Run {repeat}/{args.repeats}: CUDA forward...", flush=True)
        forward = cuda_forward_benchmark(encoder, crops, args.warmup, args.iterations)
        print(f"Run {repeat}/{args.repeats}: complete adapter...", flush=True)

        def before_encode_measurement():
            torch.cuda.synchronize(encoder.device)
            torch.cuda.reset_peak_memory_stats(encoder.device)

        encode = wall_benchmark(lambda: encoder.encode(crops), args.warmup, args.iterations,
                                before_measure=before_encode_measurement)
        encode.update(memory_stats(encoder.device))
        encode["batches_per_second"] = args.iterations / encode["elapsed_s"]
        encode["crops_per_second"] = args.iterations * len(crops) / encode["elapsed_s"]
        result = {"repeat": repeat, "preprocess_cpu": cpu, "forward_cuda": forward, "encode_total": encode}
        results.append(result)
        print(json.dumps({"repeat": repeat,
                          "preprocess_cpu": cpu["timings"], "forward_cuda": forward["timings"],
                          "encode_total": {**encode["timings"], "batches_per_second": encode["batches_per_second"],
                                           "crops_per_second": encode["crops_per_second"],
                                           "peak_torch_allocated_mib": encode["peak_torch_allocated_mib"],
                                           "peak_torch_reserved_mib": encode["peak_torch_reserved_mib"]}}, indent=2), flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "artifacts/benchmarks" / f"osnet_fp32_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": {
            "name": "osnet_fp32_in_memory_snapshot_v1", "batch_size": len(crops),
            "warmup_per_phase_per_repeat": args.warmup, "iterations_per_phase_per_repeat": args.iterations,
            "repeats": args.repeats, "phase_order": ["preprocess_cpu", "forward_cuda", "encode_total"],
            "sampling": "Repeat the same verified crops already in RAM; no file I/O in timed phases",
            "preprocess_cpu": "Wall time: input validation, PIL resize, tensor/channel normalization, stacking",
            "forward_cuda": "CUDA-event interval: raw model forward on resident input; end-event sync each iteration",
            "encode_total": "Wall time: actual encode including validation, preprocessing, H2D, forward, L2, D2H, output checks",
            "gpu_interval_note": "Includes device-stream gaps between kernel launches; not a sum of kernel-only durations",
            "loop_elapsed_note": "Host loop includes timer/list/output-release overhead; throughput uses full encode loop elapsed",
            "memory": "PyTorch allocated/reserved peaks include the model; cache retained across phases/repeats",
            "excluded": ["video decoding", "detection", "tracking", "crop extraction", "global association", "rendering"],
            "precision": "float32", "tf32": False, "cudnn_benchmark": False,
            "note": "Isolated serialized component measurements; do not sum phase medians or infer whole-pipeline FPS",
        },
        "provenance": provenance,
        "startup": {"encoder_initialization_ms": initialization_ms, "first_encode_ms": first_encode_ms,
                    "scope": "Initialization includes asset checks/load/H2D after imports and crop loading; one sample"},
        "parity": {"max_absolute_error": error, "minimum_cosine": cosine},
        "runs": results,
    }
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {output}")
    print("OSNet FP32 benchmark: COMPLETED")


if __name__ == "__main__":
    main()
