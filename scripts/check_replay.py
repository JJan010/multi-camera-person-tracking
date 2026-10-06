"""Check synchronized replay and reopening after an early stop."""

import json
from fractions import Fraction
from itertools import islice
from pathlib import Path

import numpy as np

from mtmc.video.replay import synchronized_replay

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads(
    (ROOT / "configs/datasets/scene_001_video_manifest.json").read_text()
)
sources = {
    item["camera"]: ROOT / item["local_path"]
    for item in manifest["files"]
}
expected_cameras = tuple(sorted(sources))
if expected_cameras != (4, 5, 8):
    raise RuntimeError(f"Unexpected cameras: {expected_cameras}")

count = 0
with synchronized_replay(sources) as batches:
    for batch in islice(batches, 300):
        if batch.frame_index != count:
            raise RuntimeError("Unexpected batch index")
        if batch.timestamp != Fraction(count, 30):
            raise RuntimeError("Unexpected batch timestamp")
        if tuple(frame.camera_id for frame in batch.frames) != expected_cameras:
            raise RuntimeError("Missing camera or unexpected camera order")

        for frame in batch.frames:
            if frame.rgb.shape != (1080, 1920, 3):
                raise RuntimeError("Unexpected image shape")
            if frame.rgb.dtype != np.uint8:
                raise RuntimeError("Unexpected image dtype")

        if count in (0, 1, 299):
            print(
                f"Round {count}: "
                f"time={float(batch.timestamp):.6f}s, "
                f"cameras={expected_cameras}"
            )
        count += 1

if count != 300:
    raise RuntimeError(f"Expected 300 rounds, received {count}")

with synchronized_replay(sources) as batches:
    first = next(batches)
    if first.frame_index != 0 or first.timestamp != 0:
        raise RuntimeError("Replay did not restart at zero")

print("Checked rounds:", count)
print("Checked images:", count * len(sources))
print("Reopen from frame zero: OK")
print("Replay smoke test: PASSED")
