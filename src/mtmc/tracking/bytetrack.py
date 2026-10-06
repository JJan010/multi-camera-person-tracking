"""One independent ByteTrack instance per camera, updated at every frame."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import supervision as sv

from mtmc.detection.rfdetr import CameraDetections

TRACKER_SETTINGS = {
    "track_activation_threshold": 0.5,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "frame_rate": 30,
    "minimum_consecutive_frames": 1,
}


@dataclass
class CameraTracks:
    camera_id: int
    frame_index: int
    timestamp: Fraction
    local_ids: np.ndarray
    xyxy: np.ndarray
    confidence: np.ndarray


class CameraTracker:
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.next_frame_index = 0
        self.tracker = sv.ByteTrack(**TRACKER_SETTINGS)

    def update(self, detections: CameraDetections) -> CameraTracks:
        if detections.camera_id != self.camera_id:
            raise ValueError("Detections belong to another camera")
        if detections.frame_index != self.next_frame_index:
            raise ValueError(
                f"Camera {self.camera_id}: expected frame "
                f"{self.next_frame_index}, got {detections.frame_index}"
            )
        if detections.timestamp != Fraction(self.next_frame_index, 30):
            raise ValueError("Unexpected timestamp for 30 FPS replay")

        inputs = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy(),
            class_id=np.zeros(len(detections.xyxy), dtype=int),
        )
        tracked = self.tracker.update_with_detections(inputs)
        self.next_frame_index += 1

        return CameraTracks(
            camera_id=self.camera_id,
            frame_index=detections.frame_index,
            timestamp=detections.timestamp,
            local_ids=tracked.tracker_id.copy(),
            xyxy=tracked.xyxy.copy(),
            confidence=tracked.confidence.copy(),
        )
