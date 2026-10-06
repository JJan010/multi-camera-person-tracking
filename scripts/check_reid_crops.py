"""Check crop geometry and observation matching without CUDA or video files."""

from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from mtmc.reid.crops import build_person_crops, crop_geometry
from mtmc.reid.osnet import ObservationKey, validate_crops


def fixtures():
    # Structural fixtures have the same fields as FrameBatch and CameraTracks.
    frames, tracks = [], []
    for camera in (8, 4, 5):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb[:, :, 0] = camera
        rgb[:, :, 1] = np.arange(10)[None, :]
        rgb[:, :, 2] = np.arange(10)[:, None]
        common = dict(camera_id=camera, frame_index=150, timestamp=Fraction(5))
        frames.append(SimpleNamespace(**common, rgb=rgb))
        tracks.append(SimpleNamespace(
            **common, local_ids=np.array([7], dtype=np.int64),
            xyxy=np.array([[1.2, 2.8, 4.1, 7.0]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32)))
    batch = SimpleNamespace(frame_index=150, timestamp=Fraction(5), frames=tuple(frames))
    return batch, tracks


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    require(crop_geometry([1.2, 2.8, 4.1, 7], 10, 10) == ((1, 2, 5, 7), 1.0),
            "Incorrect rounding")
    require(crop_geometry([-2, -2, 2, 2], 10, 10) == ((0, 0, 2, 2), 0.25),
            "Incorrect clipping/area fraction")
    require(crop_geometry([10, 1, 12, 3], 10, 10) == (None, 0.0),
            "Outside box was not skipped")
    batch, tracks = fixtures()
    result = build_person_crops(batch, list(reversed(tracks)))
    require([c.key for c in result.crops] == [ObservationKey(c, 7, 150) for c in (4, 5, 8)],
            "Camera-local key/order mismatch")
    frames = {f.camera_id: f for f in batch.frames}
    for crop in result.crops:
        source = frames[crop.key.camera_id].rgb
        require(np.array_equal(crop.rgb, source[2:7, 1:5]), "Wrong camera or slice axes")
        require(np.shares_memory(crop.rgb, source) and not crop.rgb.flags.writeable,
                "Crop must be a read-only borrowed view")
        require(source.flags.writeable, "Source array flags were changed")
    validate_crops(result.crops)
    print("Camera lookup, local ID scope, rounding and borrowed RGB pixels: OK")

    batch, tracks = fixtures()
    tracks[0].xyxy[0] = [-2, -2, 2, 2]
    tracks[0].confidence[0] = 0.01
    tracks[1].xyxy[0] = [10, 1, 12, 3]
    result = build_person_crops(batch, tracks)
    require([c.key.camera_id for c in result.crops] == [5, 8], "Unexpected crop selection")
    require(result.records[0].crop_xyxy_int is None, "Skipped observation must be recorded")
    require(result.records[-1].inside_image_fraction == 0.25, "Partial box metadata lost")
    print("Partial crops retained; fully outside observations explicitly recorded: OK")

    batch, tracks = fixtures()
    for item in tracks:
        item.local_ids = np.empty(0, dtype=np.int64)
        item.xyxy = np.empty((0, 4), dtype=np.float32)
        item.confidence = np.empty(0, dtype=np.float32)
    result = build_person_crops(batch, tracks)
    require(not result.crops and not result.records, "Empty track sets must return empty crops")
    print("All cameras without visible tracks: OK")

    batch, tracks = fixtures()
    cases = []
    changed = deepcopy(tracks)
    changed[0].frame_index += 1
    cases.append(("wrong track frame", batch, changed))
    changed = deepcopy(tracks)
    changed[0].timestamp += Fraction(1, 30)
    cases.append(("wrong track time", batch, changed))
    changed_batch = deepcopy(batch)
    changed_batch.frames[0].timestamp += Fraction(1, 30)
    cases.append(("wrong image time", changed_batch, tracks))
    cases.append(("missing camera", batch, tracks[:-1]))
    cases.append(("duplicate camera", batch, tracks + tracks[:1]))
    changed = deepcopy(tracks)
    changed[0].local_ids = np.array([7, 7])
    changed[0].xyxy = np.tile(changed[0].xyxy, (2, 1))
    changed[0].confidence = np.tile(changed[0].confidence, 2)
    cases.append(("duplicate local ID", batch, changed))
    changed_batch = deepcopy(batch)
    changed_batch.frames[0].rgb = changed_batch.frames[0].rgb.astype(np.float32)
    cases.append(("float image", changed_batch, tracks))
    for label, changed_batch, changed_tracks in cases:
        try:
            build_person_crops(changed_batch, changed_tracks)
        except ValueError:
            continue
        raise AssertionError(f"Accepted invalid input: {label}")
    print("Wrong time, camera coverage, duplicate IDs and image dtype rejected: OK")
    print("In-memory crop checks: PASSED")


if __name__ == "__main__":
    main()
