"""Benchmark sequential replay plus batched RF-DETR FP32 inference."""

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter_ns

ROOT = Path(__file__).resolve().parents[1]
os.environ["RF_HOME"] = str(ROOT / "artifacts/models/rfdetr")
os.environ["HF_HOME"] = str(ROOT / "artifacts/models/huggingface")

import av
import numpy as np
import torch
from rfdetr import RFDETRSmall

from mtmc.video.replay import synchronized_replay

WARMUP = 30
ROUNDS = 300
REPEATS = 3
THRESHOLD = 0.5

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

manifest = json.loads(
    (ROOT / "configs/datasets/scene_001_video_manifest.json").read_text()
)
sources = {
    item["camera"]: ROOT / item["local_path"]
    for item in manifest["files"]
}
if sorted(sources) != [4, 5, 8]:
    raise RuntimeError("Expected cameras 4, 5 and 8")


def git_output(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True
    ).strip()


def stats(values):
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


@torch.inference_mode()
def predict_persons(batch):
    predictions = model.predict(
        [frame.rgb for frame in batch.frames],
        threshold=THRESHOLD,
        include_source_image=False,
    )
    if not isinstance(predictions, list) or len(predictions) != 3:
        raise RuntimeError("Unexpected prediction batch")
    return predictions


print("Loading RF-DETR Small...", flush=True)
model = RFDETRSmall(device="cuda")

weights = Path(model.model_config.pretrain_weights)
digest = hashlib.sha256()
with weights.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)

reference = json.loads(
    (ROOT / "docs/detector/first_preview.json").read_text()
)
if digest.hexdigest() != reference["configuration"]["weights_sha256"]:
    raise RuntimeError("Weights differ from the recorded preview baseline")

report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_status": git_output("status", "--porcelain"),
    "benchmark_script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "rfdetr": version("rfdetr"),
    "supervision": version("supervision"),
    "transformers": version("transformers"),
    "pyav": av.__version__,
    "ffmpeg_libraries": av.library_versions,
    "torch_cpu_threads": torch.get_num_threads(),
    "weights_sha256": digest.hexdigest(),
    "source_manifest": manifest,
    "configuration": {
        "model": "RFDETRSmall",
        "precision": "FP32",
        "tf32": False,
        "optimized": False,
        "resolution": model.model_config.resolution,
        "confidence_threshold": THRESHOLD,
        "batch_size": 3,
        "decoder_threads_per_camera": 1,
        "scheduling": "sequential replay, then batched prediction",
        "warmup_rounds": WARMUP,
        "measured_rounds": ROUNDS,
        "measured_frame_range": [WARMUP, WARMUP + ROUNDS - 1],
        "repeats": REPEATS,
        "model_reused_across_repeats": True,
        "filesystem_cache": "uncontrolled",
        "visualization": False,
        "playback_rate_limit": False,
    },
    "runs": [],
}

for repeat in range(REPEATS):
    print(f"Run {repeat + 1}/{REPEATS}: warmup...", flush=True)
    with synchronized_replay(sources) as batches:
        for _ in range(WARMUP):
            predictions = predict_persons(next(batches))
            del predictions
        torch.cuda.synchronize()

        parameters = list(model.model.model.parameters())
        if any(p.device.type != "cuda" for p in parameters):
            raise RuntimeError("Model parameters are not all on CUDA")
        if any(
            p.dtype != torch.float32
            for p in parameters if p.is_floating_point()
        ):
            raise RuntimeError("Expected FP32 parameters")
        del parameters

        torch.cuda.reset_peak_memory_stats()
        samples = {"replay_ms": [], "predict_ms": [], "round_ms": []}
        person_counts = []
        print(f"Run {repeat + 1}/{REPEATS}: measuring...", flush=True)

        measured_started = perf_counter_ns()
        for _ in range(ROUNDS):
            started = perf_counter_ns()
            batch = next(batches)
            replay_finished = perf_counter_ns()

            predictions = predict_persons(batch)
            torch.cuda.synchronize()
            predict_finished = perf_counter_ns()

            persons = []
            for detections in predictions:
                names = detections.data.get("class_name")
                if names is None:
                    raise RuntimeError("Missing class-name mapping")
                persons.append(detections[np.asarray(names) == "person"])
            finished = perf_counter_ns()

            samples["replay_ms"].append(
                (replay_finished - started) / 1e6
            )
            samples["predict_ms"].append(
                (predict_finished - replay_finished) / 1e6
            )
            samples["round_ms"].append((finished - started) / 1e6)
            person_counts.append([len(item) for item in persons])
            del batch, predictions, persons, detections, names

        elapsed_s = (perf_counter_ns() - measured_started) / 1e9

    summary = {
        "repeat": repeat + 1,
        "elapsed_s": elapsed_s,
        "rounds_per_second": ROUNDS / elapsed_s,
        "images_per_second": 3 * ROUNDS / elapsed_s,
        "timings": {name: stats(values) for name, values in samples.items()},
        "peak_torch_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2
        ),
        "peak_torch_reserved_mib": (
            torch.cuda.max_memory_reserved() / 1024**2
        ),
    }
    report["runs"].append({
        **summary,
        "raw_samples": samples,
        "person_counts_by_round": person_counts,
    })
    print(json.dumps(summary, indent=2), flush=True)

output_dir = ROOT / "artifacts/benchmarks"
output_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = output_dir / f"detector_fp32_{stamp}.json"
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("Report:", output)
