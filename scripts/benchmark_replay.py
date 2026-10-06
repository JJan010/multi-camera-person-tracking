"""Benchmark sequential three-camera replay to RGB arrays in CPU memory."""

import json
import platform
import subprocess
from contextlib import ExitStack
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns

import av
import numpy as np

from mtmc.video.reader import CPUVideoReader

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 60
ROUNDS = 900
REPEATS = 3
THREADS = 1

manifest = json.loads(
    (ROOT / "configs/datasets/scene_001_video_manifest.json").read_text()
)
items = sorted(manifest["files"], key=lambda item: item["camera"])
if [item["camera"] for item in items] != [4, 5, 8]:
    raise RuntimeError("This benchmark expects cameras 4, 5 and 8")


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


def read_round(readers, index):
    packets = [reader.read() for reader in readers]
    expected_time = Fraction(index, 30)
    for packet in packets:
        if packet.frame_index != index or packet.timestamp != expected_time:
            raise RuntimeError(
                f"Camera {packet.camera_id}: unexpected frame index/time"
            )
        if packet.rgb.shape != (1080, 1920, 3):
            raise RuntimeError("Unexpected image dimensions")
        if packet.rgb.dtype != np.uint8:
            raise RuntimeError("Unexpected image dtype")
    return packets


report = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_output("rev-parse", "HEAD"),
    "git_status": git_output("status", "--porcelain"),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "pyav": av.__version__,
    "numpy": np.__version__,
    "ffmpeg_libraries": av.library_versions,
    "source_manifest": manifest,
    "configuration": {
        "backend": "pyav_cpu",
        "scheduling": "sequential_cameras",
        "decoder_threads_per_camera": THREADS,
        "decoder_thread_type": "SLICE",
        "output": "RGB uint8 arrays in CPU RAM",
        "warmup_rounds": WARMUP,
        "measured_rounds": ROUNDS,
        "repeats": REPEATS,
        "playback_rate_limit": False,
        "filesystem_cache": "uncontrolled; no cold-cache claim",
    },
    "runs": [],
}

for repeat in range(REPEATS):
    print(f"Run {repeat + 1}/{REPEATS}...", flush=True)
    with ExitStack() as stack:
        readers = [
            stack.enter_context(
                CPUVideoReader(
                    ROOT / item["local_path"], item["camera"], THREADS
                )
            )
            for item in items
        ]
        startup = []
        for reader in readers:
            started = perf_counter_ns()
            packet = reader.read()
            if packet.frame_index != 0 or packet.timestamp != 0:
                raise RuntimeError("Unexpected first frame")
            startup.append({
                "camera": reader.camera_id,
                "open_ms": reader.open_ms,
                "first_read_rgb_ms": (perf_counter_ns() - started) / 1e6,
            })
            del packet

        for index in range(1, WARMUP):
            read_round(readers, index)

        round_times = []
        per_camera = {
            item["camera"]: {"read_decode_ms": [], "rgb_conversion_ms": []}
            for item in items
        }

        measured_started = perf_counter_ns()
        for index in range(WARMUP, WARMUP + ROUNDS):
            started = perf_counter_ns()
            packets = read_round(readers, index)
            elapsed_ms = (perf_counter_ns() - started) / 1e6
            round_times.append(elapsed_ms)

            for packet in packets:
                samples = per_camera[packet.camera_id]
                samples["read_decode_ms"].append(packet.read_decode_ms)
                samples["rgb_conversion_ms"].append(packet.rgb_conversion_ms)
            del packets, packet

        elapsed_s = (perf_counter_ns() - measured_started) / 1e9

    run = {
        "repeat": repeat + 1,
        "elapsed_s": elapsed_s,
        "rounds_per_second": ROUNDS / elapsed_s,
        "images_per_second": ROUNDS * len(items) / elapsed_s,
        "round_latency": stats(round_times),
        "startup": startup,
        "per_camera": {
            camera: {name: stats(values) for name, values in samples.items()}
            for camera, samples in per_camera.items()
        },
        "raw_samples": {
            "round_ms": round_times,
            "per_camera": per_camera,
        },
    }
    report["runs"].append(run)
    print(json.dumps(
        {key: run[key] for key in (
            "repeat", "elapsed_s", "rounds_per_second",
            "images_per_second", "round_latency"
        )},
        indent=2,
    ), flush=True)

output_dir = ROOT / "artifacts/benchmarks"
output_dir.mkdir(parents=True, exist_ok=True)
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = output_dir / f"replay_cpu_{stamp}.json"
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"\nReport: {output}")
