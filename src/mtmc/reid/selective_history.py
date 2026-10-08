"""Experimental selective global appearance memory; no tracker or GT access.

This module is not installed into the baseline pipeline by importing it.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import numpy as np

from .osnet import ObservationKey, ReIDBatch


@dataclass(frozen=True)
class HistoryDecision:
    key: ObservationKey
    accepted_update: bool
    status: str  # updated, reused, unavailable
    source_frames: tuple[int, ...]
    source_times: tuple[Fraction, ...]
    mean_norm_before_normalization: float | None
    used_latest_accepted_fallback: bool


@dataclass(frozen=True)
class SelectiveHistoryBatch:
    run_id: str
    mean: ReIDBatch  # Only current observations with a retained descriptor.
    decisions: tuple[HistoryDecision, ...]  # Every current input observation.


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp: Fraction
    vector: np.ndarray


def confidence_update_mask(observations: ReIDBatch, scores: Mapping, *, minimum_score: float):
    """Confidence is a proxy, not a visibility/occlusion classifier.

    Scores are joined by observation key, never by accidental row order.
    The boundary is inclusive. This selects memory updates, not detections.
    """
    if (type(minimum_score) not in (int, float) or not np.isfinite(minimum_score)
            or not 0 <= minimum_score <= 1):
        raise ValueError('Invalid minimum score')
    if not isinstance(observations, ReIDBatch) or not isinstance(scores, Mapping):
        raise TypeError('Expected ReIDBatch and key-to-score mapping')
    if len(set(observations.keys)) != len(observations.keys) or set(scores) != set(observations.keys):
        raise ValueError('Confidence keys must cover every observation exactly')
    result = {}
    for key in observations.keys:
        score = scores[key]
        if (isinstance(score, (bool, np.bool_)) or not isinstance(score, (int, float, np.integer, np.floating))
                or not np.isfinite(score) or not 0 <= score <= 1):
            raise ValueError('Invalid detection confidence')
        result[key] = bool(score >= minimum_score)
    return result


class SelectiveAppearanceHistory:
    """Bounded causal memory with explicit update acceptance and availability.

    Every round, including empty rounds, advances expiry by scene time.
    Rejected samples never enter memory. A weak current observation may reuse
    the mean of unexpired accepted samples for its own camera/local ID.
    No history means no descriptor, not a zero vector or weak-vector fallback.
    Reuse does not refresh the timestamps of the source samples.

    The caller must preserve all tracked boxes when a descriptor is unavailable.
    This class does not remove tracks, associate identities or detect ID switches.
    """
    def __init__(self, run_id: str, *, max_observations: int = 8, max_age: Fraction = Fraction(1)):
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError('A nonempty run ID is required')
        if type(max_observations) is not int or max_observations <= 0:
            raise ValueError('max_observations must be positive')
        if not isinstance(max_age, Fraction) or max_age <= 0:
            raise ValueError('max_age must be a positive Fraction')
        self.run_id = run_id
        self.max_observations = max_observations
        self.max_age = max_age
        self._samples: dict[tuple[int, int], tuple[_Sample, ...]] = {}
        self._last_frame = -1
        self._last_time: Fraction | None = None

    @property
    def active_tracks(self):
        return len(self._samples)

    @property
    def stored_vectors(self):
        return sum(len(items) for items in self._samples.values())

    def update(self, frame_index: int, timestamp: Fraction, observations: ReIDBatch,
               *, accept_update: Mapping[ObservationKey, bool]) -> SelectiveHistoryBatch:
        if type(frame_index) is not int or frame_index < 0 or frame_index <= self._last_frame:
            raise ValueError('Frame indices must be nonnegative and strictly increasing')
        if (not isinstance(timestamp, Fraction) or timestamp < 0
                or (self._last_time is not None and timestamp <= self._last_time)):
            raise ValueError('Scene time must strictly increase')
        if not isinstance(observations, ReIDBatch):
            raise TypeError('Expected ReIDBatch')
        keys, times, vectors = observations.keys, observations.timestamps, observations.embeddings
        if (not isinstance(vectors, np.ndarray) or vectors.dtype != np.float32
                or vectors.shape != (len(keys), 512) or len(times) != len(keys)
                or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-5)):
            raise ValueError('Expected finite normalized float32 (N,512) vectors')
        seen = set()
        for key, time in zip(keys, times):
            if (not isinstance(key, ObservationKey)
                    or any(type(v) is not int or v < 0 for v in (key.camera_id, key.local_id, key.frame_index))
                    or key.frame_index != frame_index or not isinstance(time, Fraction) or time != timestamp
                    or (key.camera_id, key.local_id) in seen):
                raise ValueError('Invalid or duplicate observation scope')
            seen.add((key.camera_id, key.local_id))
        if (not isinstance(accept_update, Mapping) or set(accept_update) != set(keys)
                or any(type(v) is not bool for v in accept_update.values())):
            raise ValueError('Explicit boolean update decision required for every input key')

        # Compute a complete next state before committing; no old array is modified.
        state = {}
        for local_key, items in self._samples.items():
            retained = tuple(item for item in items if timestamp-item.timestamp <= self.max_age)
            if retained:
                state[local_key] = retained
        output_keys, output_vectors, decisions = [], [], []
        for index, key in enumerate(keys):
            local_key = key.camera_id, key.local_id
            accepted = accept_update[key]
            items = state.get(local_key, ())
            if accepted:
                items = (*items, _Sample(frame_index, timestamp, vectors[index].copy()))[-self.max_observations:]
                state[local_key] = items
            if not items:
                decisions.append(HistoryDecision(key, accepted, 'unavailable', (), (), None, False))
                continue
            mean = np.mean(np.stack([item.vector for item in items]), axis=0, dtype=np.float64)
            length = float(np.linalg.norm(mean))
            fallback = length <= 1e-12
            descriptor = items[-1].vector.copy() if fallback else (mean/length).astype(np.float32)
            output_keys.append(key)
            output_vectors.append(descriptor)
            decisions.append(HistoryDecision(key, accepted, 'updated' if accepted else 'reused',
                tuple(item.frame_index for item in items), tuple(item.timestamp for item in items), length, fallback))
        features = np.stack(output_vectors).astype(np.float32) if output_vectors else np.empty((0,512), np.float32)
        result = SelectiveHistoryBatch(self.run_id,
            ReIDBatch(tuple(output_keys), (timestamp,)*len(output_keys), features), tuple(decisions))
        self._samples = state
        self._last_frame, self._last_time = frame_index, timestamp
        return result
