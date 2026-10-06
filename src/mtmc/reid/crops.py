"""Match synchronized frames to local tracks and borrow original RGB crops."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .osnet import ObservationKey, PersonCrop

if TYPE_CHECKING:
    from mtmc.tracking.bytetrack import CameraTracks
    from mtmc.video.replay import FrameBatch


@dataclass(frozen=True)
class CropRecord:
    key: ObservationKey
    confidence: float
    source_xyxy: tuple[float, float, float, float]
    crop_xyxy_int: tuple[int, int, int, int] | None
    inside_image_fraction: float  # Geometric area fraction, NOT visibility.


@dataclass(frozen=True)
class CropBatch:
    crops: tuple[PersonCrop, ...]
    records: tuple[CropRecord, ...]  # Includes fully outside boxes (bounds=None).


def crop_geometry(xyxy, width: int, height: int):
    """Clip a positive float box; floor left/top, ceil right/bottom.

    Right/bottom are exclusive NumPy slice limits. Fully outside boxes return
    (None, 0.0). No padding, resizing or quality selection is performed here.
    """
    if any(type(v) is not int or v <= 0 for v in (width, height)):
        raise ValueError("Image width and height must be positive integers")
    box = np.asarray(xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Expected a finite xyxy box")
    if np.any(box[2:] <= box[:2]):
        raise ValueError("Original box must have positive width and height")
    clipped = np.clip(box, [0, 0, 0, 0], [width, height, width, height])
    area = (box[2] - box[0]) * (box[3] - box[1])
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid original box area")
    clipped_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
    if clipped_area <= 0:
        return None, 0.0
    bounds = (*np.floor(clipped[:2]), *np.ceil(clipped[2:]))
    return tuple(int(v) for v in bounds), float(clipped_area / area)


def _check_time(item, frame_index: int, timestamp: Fraction):
    if type(item.camera_id) is not int or item.camera_id < 0:
        raise ValueError("Camera ID must be a nonnegative Python integer")
    if (type(item.frame_index) is not int or item.frame_index != frame_index
            or not isinstance(item.timestamp, Fraction) or item.timestamp != timestamp):
        raise ValueError("Frame/track does not match the batch index and timestamp")


def build_person_crops(batch: FrameBatch, tracks: Sequence[CameraTracks]) -> CropBatch:
    """Return crops ordered by camera ID, then local ID, within one replay run.

    Exactly one CameraTracks entry (possibly empty) is required per frame.
    Camera matching uses IDs rather than positions in the input sequences.
    Crops are read-only CPU views: do not mutate/reuse the source RGB buffers
    until encode() has completed. Retaining a view also retains its full frame.
    No file I/O, model execution, GT access or cross-frame state is used.
    """
    if (type(batch.frame_index) is not int or batch.frame_index < 0
            or not isinstance(batch.timestamp, Fraction) or batch.timestamp < 0):
        raise ValueError("Invalid batch frame index or timestamp")
    if not batch.frames:
        raise ValueError("At least one frame is required")
    frames_by_camera = {}
    for frame in batch.frames:
        _check_time(frame, batch.frame_index, batch.timestamp)
        if frame.camera_id in frames_by_camera:
            raise ValueError("Duplicate frame camera ID")
        rgb = frame.rgb
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0):
            raise ValueError("Expected a nonempty HWC uint8 RGB frame")
        frames_by_camera[frame.camera_id] = frame

    tracks_by_camera = {}
    for item in tracks:
        _check_time(item, batch.frame_index, batch.timestamp)
        if item.camera_id in tracks_by_camera:
            raise ValueError("Duplicate track camera ID")
        ids, boxes, scores = item.local_ids, item.xyxy, item.confidence
        if (not isinstance(ids, np.ndarray) or ids.ndim != 1
                or ids.dtype.kind not in "iu" or np.any(ids < 0)
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("Local IDs must be unique nonnegative integers per camera")
        if (not isinstance(boxes, np.ndarray) or boxes.shape != (len(ids), 4)
                or boxes.dtype.kind not in "fiu" or not np.isfinite(boxes).all()
                or np.any(boxes[:, 2:] <= boxes[:, :2])):
            raise ValueError("Expected finite, positive xyxy boxes, shape (N, 4)")
        if (not isinstance(scores, np.ndarray) or scores.shape != (len(ids),)
                or scores.dtype.kind not in "fiu" or not np.isfinite(scores).all()
                or np.any((scores < 0) | (scores > 1))):
            raise ValueError("Expected confidence values in [0, 1], shape (N,)")
        tracks_by_camera[item.camera_id] = item
    if frames_by_camera.keys() != tracks_by_camera.keys():
        raise ValueError("Frame cameras and track cameras must match exactly")

    crops, records = [], []
    for camera in sorted(frames_by_camera):
        frame, item = frames_by_camera[camera], tracks_by_camera[camera]
        height, width = frame.rgb.shape[:2]
        for index in np.argsort(item.local_ids):
            key = ObservationKey(camera, int(item.local_ids[index]), batch.frame_index)
            box = tuple(float(v) for v in item.xyxy[index])
            bounds, fraction = crop_geometry(box, width, height)
            records.append(CropRecord(key, float(item.confidence[index]), box, bounds, fraction))
            if bounds is not None:
                left, top, right, bottom = bounds
                rgb = frame.rgb[top:bottom, left:right].view()
                rgb.setflags(write=False)
                crops.append(PersonCrop(key, batch.timestamp, rgb))
    return CropBatch(tuple(crops), tuple(records))
