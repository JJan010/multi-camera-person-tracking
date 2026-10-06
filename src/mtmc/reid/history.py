"""Causal, run-scoped appearance history using scene timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .osnet import ObservationKey, ReIDBatch


@dataclass(frozen=True)
class HistoryBatch:
    run_id: str
    latest: ReIDBatch
    mean: ReIDBatch
    source_frames: tuple[tuple[int, ...], ...]
    source_times: tuple[tuple[Fraction, ...], ...]
    mean_norms_before_normalization: tuple[float, ...]
    used_latest_fallback: tuple[bool, ...]


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp: Fraction
    vector: np.ndarray


class AppearanceHistory:
    """Store recent vectors per (camera, local ID) within one fixed run.

    This does not detect identity switches, merge IDs or select crop quality.
    update() must be called for every round, including empty observation sets,
    so age eviction follows scene time. Output contains only current observations.
    A restart/new ID generation requires a new instance and a distinct run_id.
    """

    def __init__(self, run_id: str, *, max_observations: int = 8,
                 max_age: Fraction = Fraction(1)):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("A nonempty run ID is required")
        if type(max_observations) is not int or max_observations <= 0:
            raise ValueError("max_observations must be a positive integer")
        if not isinstance(max_age, Fraction) or max_age <= 0:
            raise ValueError("max_age must be a positive Fraction in scene seconds")
        self.run_id = run_id
        self.max_observations = max_observations
        self.max_age = max_age
        self._samples: dict[tuple[int, int], tuple[_Sample, ...]] = {}
        self._last_frame = -1
        self._last_time: Fraction | None = None

    @property
    def active_tracks(self) -> int:
        return len(self._samples)

    @property
    def stored_vectors(self) -> int:
        return sum(len(items) for items in self._samples.values())

    def update(self, frame_index: int, timestamp: Fraction, observations: ReIDBatch) -> HistoryBatch:
        """Include current observations, then return latest and normalized mean.

        Keep samples whose age is <= max_age; from those, keep at most the most
        recent max_observations. Unobserved tracks can remain in memory but are
        not emitted. Inputs are validated before state changes; stored vectors
        are owned copies, and output arrays do not expose internal state.
        """
        if type(frame_index) is not int or frame_index <= self._last_frame:
            raise ValueError("Frame indices must be nonnegative and strictly increasing")
        if (not isinstance(timestamp, Fraction) or timestamp < 0
                or (self._last_time is not None and timestamp <= self._last_time)):
            raise ValueError("Scene timestamps must be nonnegative and strictly increasing")
        if not isinstance(observations, ReIDBatch):
            raise TypeError("Expected a ReIDBatch")
        keys, times, vectors = observations.keys, observations.timestamps, observations.embeddings
        if (not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32
                or vectors.shape != (len(keys), 512) or len(times) != len(keys)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError("Expected finite, L2-normalized float32 embeddings, shape (N, 512)")
        seen = set()
        for key, time in zip(keys, times):
            if (not isinstance(key, ObservationKey)
                    or any(type(v) is not int or v < 0 for v in
                           (key.camera_id, key.local_id, key.frame_index))
                    or key.frame_index != frame_index
                    or not isinstance(time, Fraction) or time != timestamp):
                raise ValueError("Observation camera/ID/frame/time does not satisfy the round contract")
            local_key = (key.camera_id, key.local_id)
            if local_key in seen:
                raise ValueError("Duplicate camera/local-ID observation in one round")
            seen.add(local_key)

        state = {}
        for key, items in self._samples.items():
            retained = tuple(item for item in items if timestamp - item.timestamp <= self.max_age)
            if retained:
                state[key] = retained
        latest = vectors.copy()
        averaged = np.empty_like(latest)
        source_frames, source_times, mean_norms, fallbacks = [], [], [], []
        for index, key in enumerate(keys):
            local_key = (key.camera_id, key.local_id)
            sample = _Sample(frame_index, timestamp, latest[index].copy())
            items = (*state.get(local_key, ()), sample)[-self.max_observations:]
            state[local_key] = items
            # Double-precision accumulation; returned descriptors remain float32.
            mean = np.mean(np.stack([item.vector for item in items]), axis=0, dtype=np.float64)
            length = float(np.linalg.norm(mean))
            # Signed unit vectors could cancel. Make this rare numerical policy explicit.
            fallback = length <= 1e-12
            averaged[index] = latest[index] if fallback else (mean / length).astype(np.float32)
            source_frames.append(tuple(item.frame_index for item in items))
            source_times.append(tuple(item.timestamp for item in items))
            mean_norms.append(length)
            fallbacks.append(fallback)
        self._samples = state
        self._last_frame, self._last_time = frame_index, timestamp
        return HistoryBatch(
            self.run_id, ReIDBatch(tuple(keys), tuple(times), latest),
            ReIDBatch(tuple(keys), tuple(times), averaged), tuple(source_frames),
            tuple(source_times), tuple(mean_norms), tuple(fallbacks))
