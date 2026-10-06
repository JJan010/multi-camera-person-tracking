"""Check decoded frame counts and presentation timelines for scene 001."""

import json
import subprocess
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads(
    (ROOT / "configs/datasets/scene_001_video_manifest.json").read_text()
)
output = ROOT / "artifacts/dataset_audit/scene_001_timestamps.json"
output.parent.mkdir(parents=True, exist_ok=True)

report = {
    "revision": manifest["revision"],
    "timestamp_field": "best_effort_timestamp",
    "expected_fps": 30,
    "expected_frames": 23994,
    "videos": [],
}

for item in manifest["files"]:
    camera = item["camera"]
    print(f"Camera {camera}: decoding the full video...", flush=True)

    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_frames", "-show_streams",
            "-show_entries",
            "frame=best_effort_timestamp:stream=time_base",
            "-of", "json",
            str(ROOT / item["local_path"]),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    if result.stderr.strip():
        raise RuntimeError(f"Camera {camera}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    time_base = Fraction(data["streams"][0]["time_base"])
    frames = data.get("frames", [])
    timestamps = [
        int(frame["best_effort_timestamp"])
        if "best_effort_timestamp" in frame else None
        for frame in frames
    ]

    missing = sum(value is None for value in timestamps)
    mismatches = [
        index
        for index, value in enumerate(timestamps)
        if value is not None
        and value * time_base != Fraction(index, 30)
    ]
    nonincreasing = sum(
        current <= previous
        for previous, current in zip(timestamps, timestamps[1:])
        if previous is not None and current is not None
    )
    passed = (
        len(frames) == report["expected_frames"]
        and missing == 0
        and not mismatches
        and nonincreasing == 0
    )
    summary = {
        "camera": camera,
        "decoded_frames": len(frames),
        "time_base": str(time_base),
        "missing_timestamps": missing,
        "nonincreasing_steps": nonincreasing,
        "timeline_mismatches": len(mismatches),
        "first_mismatch_indices": mismatches[:10],
        "passed": passed,
    }
    report["videos"].append(summary)
    print(json.dumps(summary, indent=2), flush=True)

report["passed"] = bool(report["videos"]) and all(
    video["passed"] for video in report["videos"]
)
output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"\nReport: {output}")
print("Timestamp audit:", "PASSED" if report["passed"] else "FAILED")
raise SystemExit(0 if report["passed"] else 1)
