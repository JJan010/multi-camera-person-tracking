"""Download the videos listed in the pinned scene plan."""

import hashlib
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
plan = json.loads((root / "configs/datasets/scene_001_video_plan.json").read_text())
manifest = {key: plan[key] for key in ("dataset", "revision", "scene", "camera_ids")}
manifest["files"] = []

for item in plan["files"]:
    remote = PurePosixPath(item["path"])
    if remote.is_absolute() or ".." in remote.parts:
        raise ValueError(f"Invalid remote path: {remote}")
    target = root / "data/physicalai_smartspaces" / remote
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = target
    expected_size = item["size"]

    if not target.exists():
        candidate = target.with_suffix(".mp4.part")
        url = (
            f"https://huggingface.co/datasets/{plan['dataset']}/resolve/"
            f"{plan['revision']}/{quote(remote.as_posix(), safe='/')}"
        )
        print(f"Downloading camera {item['camera']}...", flush=True)
        received, next_log = 0, 64 * 1024**2
        with urlopen(url, timeout=60) as response, candidate.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                received += len(chunk)
                if received >= next_log:
                    print(f"  {received / 1024**2:.0f} / {expected_size / 1024**2:.0f} MiB", flush=True)
                    next_log += 64 * 1024**2

    if candidate.stat().st_size != expected_size:
        raise RuntimeError(f"Size mismatch: {candidate}")

    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    expected_sha = (item.get("lfs") or {}).get("oid")
    if expected_sha:
        expected_sha = expected_sha.removeprefix("sha256:")
        if sha256 != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch: {candidate}")

    if candidate != target:
        candidate.replace(target)

    manifest["files"].append({
        "camera": item["camera"],
        "remote_path": remote.as_posix(),
        "local_path": target.relative_to(root).as_posix(),
        "size_bytes": expected_size,
        "sha256": sha256,
        "source_sha256_verified": bool(expected_sha),
    })
    verification = "size + source SHA-256" if expected_sha else "size; local SHA-256 recorded"
    print(f"Camera {item['camera']}: OK ({verification})", flush=True)

output = root / "configs/datasets/scene_001_video_manifest.json"
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Video manifest saved:", output)
