"""Model-scoped appearance batches and dimension-explicit causal history.

This separate module leaves the frozen OSNet API and experiments unchanged.
"""
from dataclasses import dataclass
from fractions import Fraction
import re

import numpy as np
from .osnet import ObservationKey


def require(ok, message):
    if not ok:
        raise ValueError(message)


@dataclass(frozen=True)
class FeatureSpace:
    model_config_sha256: str
    dimension: int
    normalization: str = 'l2'

    def __post_init__(self):
        require(isinstance(self.model_config_sha256, str)
                and re.fullmatch('[0-9a-f]{64}', self.model_config_sha256) is not None,
                'Feature space requires a SHA-256 of the pinned model/preprocessing configuration')
        require(type(self.dimension) is int and self.dimension > 0, 'Invalid feature dimension')
        require(self.normalization == 'l2', 'Expected L2-normalized features')


@dataclass(frozen=True)
class FeatureBatch:
    run_id: str
    space: FeatureSpace
    keys: tuple[ObservationKey, ...]
    timestamps: tuple[Fraction, ...]
    embeddings: np.ndarray


def validate_batch(batch, *, run_id, space, frame_index, timestamp):
    require(isinstance(batch, FeatureBatch) and batch.run_id == run_id and batch.space == space,
            'Feature batch run/model space differs')
    require(isinstance(batch.keys, tuple) and isinstance(batch.timestamps, tuple), 'Expected tuple keys/times')
    vectors = batch.embeddings
    require(isinstance(vectors, np.ndarray) and vectors.dtype == np.float32
            and vectors.shape == (len(batch.keys), space.dimension)
            and len(batch.timestamps) == len(batch.keys) and np.isfinite(vectors).all()
            and np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-5),
            'Expected finite normalized float32 features with the declared dimension')
    seen = set()
    for key, time in zip(batch.keys, batch.timestamps):
        require(isinstance(key, ObservationKey)
                and all(type(v) is int and v >= 0 for v in (key.camera_id, key.local_id, key.frame_index))
                and key.frame_index == frame_index and isinstance(time, Fraction) and time == timestamp,
                'Observation key/time differs from the round')
        identity = (key.camera_id, key.local_id)
        require(identity not in seen, 'Duplicate camera/local ID'); seen.add(identity)


@dataclass(frozen=True)
class FeatureHistoryBatch:
    run_id: str
    latest: FeatureBatch
    mean: FeatureBatch
    source_frames: tuple[tuple[int, ...], ...]
    source_times: tuple[tuple[Fraction, ...], ...]
    mean_norms_before_normalization: tuple[float, ...]
    used_latest_fallback: tuple[bool, ...]


@dataclass(frozen=True)
class _Sample:
    frame_index: int
    timestamp: Fraction
    vector: np.ndarray


class FeatureHistory:
    """Same numerical/age policy as AppearanceHistory, with explicit model space.

    Current samples participate in their own returned mean. Only current
    observations are emitted. Missing tracks retain unexpired evidence but do
    not fabricate observations. Call on empty rounds to advance scene time.
    Inputs and results do not share vector storage with retained state.
    """
    def __init__(self, run_id, space, *, max_observations=8, max_age=Fraction(1)):
        require(isinstance(run_id, str) and bool(run_id.strip()), 'Missing run ID')
        require(isinstance(space, FeatureSpace), 'Expected FeatureSpace')
        require(type(max_observations) is int and max_observations > 0, 'Invalid history size')
        require(isinstance(max_age, Fraction) and max_age > 0, 'Invalid history age')
        self.run_id = run_id; self.space = space
        self.max_observations = max_observations; self.max_age = max_age
        self._samples = {}; self._last_frame = -1; self._last_time = None

    @property
    def active_tracks(self):
        return len(self._samples)

    @property
    def stored_vectors(self):
        return sum(len(items) for items in self._samples.values())

    def update(self, frame_index, timestamp, observations):
        require(type(frame_index) is int and frame_index > self._last_frame, 'Frames must strictly increase')
        require(isinstance(timestamp, Fraction) and timestamp >= 0
                and (self._last_time is None or timestamp > self._last_time), 'Scene time must strictly increase')
        validate_batch(observations, run_id=self.run_id, space=self.space, frame_index=frame_index, timestamp=timestamp)
        state = {}
        for key, items in self._samples.items():
            retained = tuple(item for item in items if timestamp-item.timestamp <= self.max_age)
            if retained:
                state[key] = retained
        latest = observations.embeddings.copy(); averaged = np.empty_like(latest)
        frames = []; times = []; norms = []; fallbacks = []
        for index, key in enumerate(observations.keys):
            local = (key.camera_id, key.local_id)
            sample = _Sample(frame_index, timestamp, latest[index].copy())
            items = (*state.get(local, ()), sample)[-self.max_observations:]
            state[local] = items
            mean = np.mean(np.stack([item.vector for item in items]), axis=0, dtype=np.float64)
            length = float(np.linalg.norm(mean)); fallback = length <= 1e-12
            averaged[index] = latest[index] if fallback else (mean/length).astype(np.float32)
            frames.append(tuple(item.frame_index for item in items))
            times.append(tuple(item.timestamp for item in items)); norms.append(length); fallbacks.append(fallback)
        output = FeatureHistoryBatch(self.run_id,
            FeatureBatch(self.run_id, self.space, observations.keys, observations.timestamps, latest),
            FeatureBatch(self.run_id, self.space, observations.keys, observations.timestamps, averaged),
            tuple(frames), tuple(times), tuple(norms), tuple(fallbacks))
        self._samples = state; self._last_frame = frame_index; self._last_time = timestamp
        return output
