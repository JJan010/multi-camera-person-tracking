"""RF-DETR Small adapter: synchronized RGB frames to person detections."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from mtmc.video.replay import FrameBatch


@dataclass
class CameraDetections:
    camera_id: int
    frame_index: int
    timestamp: Fraction
    xyxy: np.ndarray
    confidence: np.ndarray


class RFDETRPersonDetector:
    def __init__(self, weights: Path, threshold: float = 0.5):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if not torch.cuda.is_available():
            raise RuntimeError("This detector requires CUDA")

        from rfdetr import RFDETRSmall

        self.threshold = threshold
        self.model = RFDETRSmall(
            device="cuda",
            pretrain_weights=str(weights.resolve()),
        )
        self._device_checked = False

    @torch.inference_mode()
    def detect(self, batch: FrameBatch) -> tuple[CameraDetections, ...]:
        if not batch.frames:
            raise ValueError("Empty frame batch")

        camera_ids = [frame.camera_id for frame in batch.frames]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Duplicate camera IDs in batch")

        for frame in batch.frames:
            if (
                frame.frame_index != batch.frame_index
                or frame.timestamp != batch.timestamp
            ):
                raise ValueError("Frame does not match batch index/time")
            if (
                frame.rgb.ndim != 3
                or frame.rgb.shape[2] != 3
                or frame.rgb.dtype != np.uint8
            ):
                raise ValueError("Expected an HWC RGB uint8 image")

        predictions = self.model.predict(
            [frame.rgb for frame in batch.frames],
            threshold=self.threshold,
            include_source_image=False,
        )
        if not isinstance(predictions, list) or len(predictions) != len(batch.frames):
            raise RuntimeError("Prediction count does not match input batch")

        # RF-DETR moves the model to CUDA on first predict().
        if not self._device_checked:
            for parameter in self.model.model.model.parameters():
                if parameter.device.type != "cuda":
                    raise RuntimeError("Model parameter is not on CUDA")
                if parameter.is_floating_point() and parameter.dtype != torch.float32:
                    raise RuntimeError("Expected FP32 model parameters")
            self._device_checked = True

        output = []
        for frame, prediction in zip(batch.frames, predictions):
            names = prediction.data.get("class_name")
            if names is None:
                raise RuntimeError("Missing class-name mapping")
            people = prediction[np.asarray(names) == "person"]

            boxes = np.asarray(people.xyxy, dtype=np.float32).reshape(-1, 4).copy()
            scores = np.asarray(people.confidence, dtype=np.float32).reshape(-1).copy()
            if len(boxes) != len(scores):
                raise RuntimeError("Box and confidence counts differ")
            if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
                raise RuntimeError("Non-finite detection")
            if np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise RuntimeError("Nonpositive detection box")

            output.append(CameraDetections(
                camera_id=frame.camera_id,
                frame_index=batch.frame_index,
                timestamp=batch.timestamp,
                xyxy=boxes,
                confidence=scores,
            ))

        return tuple(output)
