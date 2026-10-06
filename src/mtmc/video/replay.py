"""Strict replay of synchronized, constant-frame-rate video files."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Mapping

from mtmc.video.reader import CPUVideoReader, FramePacket


@dataclass
class FrameBatch:
    frame_index: int
    timestamp: Fraction
    frames: tuple[FramePacket, ...]


def _iterate_batches(
    readers: list[CPUVideoReader], fps: int
) -> Iterator[FrameBatch]:
    index = 0

    while True:
        packets = []
        ended = []

        for reader in readers:
            try:
                packets.append(reader.read())
            except StopIteration:
                ended.append(reader.camera_id)

        if len(ended) == len(readers):
            return
        if ended:
            raise RuntimeError(
                f"Videos ended at different lengths; "
                f"frame={index}, ended cameras={ended}"
            )

        expected_time = Fraction(index, fps)
        for packet in packets:
            if packet.frame_index != index:
                raise RuntimeError(
                    f"Camera {packet.camera_id}: expected frame {index}, "
                    f"got {packet.frame_index}"
                )
            if packet.timestamp != expected_time:
                raise RuntimeError(
                    f"Camera {packet.camera_id}, frame {index}: "
                    f"expected time {expected_time}, got {packet.timestamp}"
                )

        yield FrameBatch(
            frame_index=index,
            timestamp=expected_time,
            frames=tuple(packets),
        )
        index += 1


@contextmanager
def synchronized_replay(
    sources: Mapping[int, Path],
    fps: int = 30,
    threads: int = 1,
) -> Iterator[Iterator[FrameBatch]]:
    if not sources:
        raise ValueError("At least one camera is required")
    if fps <= 0:
        raise ValueError("fps must be positive")

    with ExitStack() as stack:
        readers = [
            stack.enter_context(CPUVideoReader(path, camera, threads))
            for camera, path in sorted(sources.items())
        ]
        batches = _iterate_batches(readers, fps)
        try:
            yield batches
        finally:
            batches.close()
