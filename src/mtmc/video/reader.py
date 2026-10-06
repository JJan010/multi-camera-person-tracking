"""Sequential CPU video reader with explicit timestamps and timings."""

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from time import perf_counter_ns

import av
import numpy as np


@dataclass
class FramePacket:
    camera_id: int
    frame_index: int
    timestamp: Fraction
    rgb: np.ndarray
    read_decode_ms: float
    rgb_conversion_ms: float


class CPUVideoReader:
    def __init__(self, path: Path, camera_id: int, threads: int = 1):
        if threads < 1:
            raise ValueError("threads must be positive")

        self.camera_id = camera_id
        self.frame_index = 0
        started = perf_counter_ns()
        self.container = av.open(str(path))
        try:
            self.stream = self.container.streams.video[0]
            self.stream.codec_context.thread_count = threads
            self.stream.thread_type = "SLICE"
            self.frames = iter(self.container.decode(self.stream))
        except Exception:
            self.container.close()
            raise
        self.open_ms = (perf_counter_ns() - started) / 1e6

    def read(self) -> FramePacket:
        started = perf_counter_ns()
        frame = next(self.frames)
        decoded = perf_counter_ns()

        if frame.pts is None or frame.time_base is None:
            raise RuntimeError("Decoded frame has no presentation timestamp")
        timestamp = Fraction(frame.pts) * frame.time_base

        conversion_started = perf_counter_ns()
        rgb = frame.to_ndarray(format="rgb24")
        converted = perf_counter_ns()

        packet = FramePacket(
            camera_id=self.camera_id,
            frame_index=self.frame_index,
            timestamp=timestamp,
            rgb=rgb,
            read_decode_ms=(decoded - started) / 1e6,
            rgb_conversion_ms=(converted - conversion_started) / 1e6,
        )
        self.frame_index += 1
        return packet

    def close(self):
        self.container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
