"""Explicit causal local-track segmentation, before appearance history/global ID.

This module applies break decisions supplied by a separate policy. It does not
detect switches, access GT, or change ByteTrack, boxes, scores or feature values.
"""
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from mtmc.reid.crops import CropRecord
from mtmc.reid.osnet import ObservationKey, ReIDBatch

# Disjoint namespaces: original tracker IDs must satisfy this explicit contract.
# Internal IDs remain exactly representable as JSON/JavaScript integer values.
ORIGINAL_ID_LIMIT = 1 << 31
MAX_EFFECTIVE_ID = (1 << 53) - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class SegmentBreak:
    key: ObservationKey                    # ORIGINAL tracker key at this round
    expected_generation: int
    reason: str                            # supplied by the decision policy


@dataclass(frozen=True)
class SegmentRound:
    run_id: str
    frame_index: int
    timestamp: Fraction
    records: tuple[CropRecord, ...]         # all visible tracks, including outside
    features: ReIDBatch                     # encoded subset; original keys
    breaks: tuple[SegmentBreak, ...] = ()


@dataclass(frozen=True)
class SegmentBinding:
    source_key: ObservationKey
    identity_key: ObservationKey
    generation: int
    started_frame: int
    started_time: Fraction


@dataclass(frozen=True)
class SegmentEvent:
    source_key: ObservationKey
    previous_identity_key: ObservationKey  # at its LAST observed frame
    identity_key: ObservationKey           # at the current break frame
    previous_last_seen: Fraction
    generation: int
    reason: str


@dataclass(frozen=True)
class SegmentedRound:
    run_id: str
    frame_index: int
    timestamp: Fraction
    bindings: tuple[SegmentBinding, ...]    # same order as source records
    records: tuple[CropRecord, ...]         # identity keys, same order/values
    features: ReIDBatch                     # identity keys, same row order/values
    events: tuple[SegmentEvent, ...]        # canonical (camera, original ID) order

    def source_key(self, identity_key):
        for binding in self.bindings:
            if binding.identity_key == identity_key:
                return binding.source_key
        raise ValueError('Identity key is not in this round')


@dataclass(frozen=True)
class _State:
    identity_local_id: int
    generation: int
    started_frame: int
    started_time: Fraction
    last_frame: int
    last_time: Fraction


class LocalTrackSegments:
    """Own one run's current segment per original camera/local ID.

    No break => original keys are unchanged, even when enabled. At an explicit
    break, allocate a fresh effective ID, from the current observation onward.
    Ordinary upstream history/identity modules therefore see a new local key.
    Their old history/binding expires by their own existing rules; it is not
    erased here. The new segment may join another identity through association.

    Break requests require an existing track and the current generation; stale
    requests cannot silently repeat a cut. Validation includes every record and
    feature before state commits. Outputs own their feature arrays. Runtime
    failures in downstream modules require abort/replay by the pipeline owner.

    Scalar state grows with the number of original tracks seen in this run.
    There is no vector cache, GT lookup, wall-clock timing or automatic expiry.
    Original local IDs must not be reused for unrelated tracks within a run.
    """
    def __init__(self, run_id, camera_ids, *, enabled):
        require(isinstance(run_id, str) and bool(run_id.strip()), 'Missing run scope')
        require(isinstance(camera_ids, tuple) and bool(camera_ids)
                and all(type(c) is int and c >= 0 for c in camera_ids)
                and len(set(camera_ids)) == len(camera_ids), 'Invalid camera IDs')
        require(type(enabled) is bool, 'enabled must be bool')
        self.run_id, self.camera_ids, self.enabled = run_id, tuple(sorted(camera_ids)), enabled
        self._state = {}
        self._next_id = ORIGINAL_ID_LIMIT
        self._last_frame = -1
        self._last_time = None

    @property
    def known_tracks(self):
        return len(self._state)

    @property
    def allocated_segments(self):
        return self._next_id - ORIGINAL_ID_LIMIT

    def _key(self, key, frame):
        require(isinstance(key, ObservationKey)
                and all(type(v) is int for v in (key.camera_id, key.local_id, key.frame_index))
                and key.camera_id in self.camera_ids and key.frame_index == frame
                and 0 <= key.local_id < ORIGINAL_ID_LIMIT, 'Invalid original observation key/namespace')

    def _validate(self, batch):
        require(isinstance(batch, SegmentRound) and batch.run_id == self.run_id, 'Mixed segment scope')
        frame, time = batch.frame_index, batch.timestamp
        require(type(frame) is int and frame > self._last_frame, 'Frames must strictly increase')
        require(isinstance(time, Fraction) and time >= 0
                and (self._last_time is None or time > self._last_time), 'Time must strictly increase')
        require(isinstance(batch.records, tuple) and isinstance(batch.breaks, tuple), 'Expected immutable inputs')
        seen = set(); encoded = set()
        for record in batch.records:
            require(isinstance(record, CropRecord), 'Expected CropRecord')
            self._key(record.key, frame)
            require(record.key not in seen, 'Duplicate observation')
            seen.add(record.key)
            require(type(record.confidence) in (int, float) and np.isfinite(record.confidence)
                    and 0 <= record.confidence <= 1, 'Invalid confidence')
            box = np.asarray(record.source_xyxy, dtype=np.float64)
            require(isinstance(record.source_xyxy, tuple) and box.shape == (4,)
                    and np.isfinite(box).all() and np.all(box[2:] > box[:2]), 'Invalid raw box')
            fraction = record.inside_image_fraction
            require(type(fraction) in (int, float) and np.isfinite(fraction)
                    and 0 <= fraction <= 1, 'Invalid crop fraction')
            if record.crop_xyxy_int is None:
                require(fraction == 0, 'Outside crop has positive area fraction')
            else:
                bounds = record.crop_xyxy_int
                require(isinstance(bounds, tuple) and len(bounds) == 4
                        and all(type(x) is int and x >= 0 for x in bounds)
                        and bounds[2] > bounds[0] and bounds[3] > bounds[1]
                        and fraction > 0, 'Invalid integer crop bounds')
                encoded.add(record.key)
        features = batch.features
        require(isinstance(features, ReIDBatch) and isinstance(features.keys, tuple)
                and isinstance(features.timestamps, tuple), 'Expected immutable feature mapping')
        for k in features.keys:
            self._key(k, frame)
        require(len(features.keys) == len(set(features.keys)) and set(features.keys) == encoded,
                'Feature keys must match encoded records exactly')
        require(len(features.timestamps) == len(features.keys)
                and all(isinstance(t, Fraction) and t == time for t in features.timestamps), 'Feature time differs')
        vectors = features.embeddings
        require(isinstance(vectors, np.ndarray) and vectors.dtype == np.float32
                and vectors.shape == (len(features.keys), 512) and np.isfinite(vectors).all()
                and np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-5), 'Invalid unit features')
        requests = {}
        require(self.enabled or not batch.breaks, 'Break requests supplied to disabled segmenter')
        for request in batch.breaks:
            require(isinstance(request, SegmentBreak), 'Expected SegmentBreak')
            self._key(request.key, frame)
            require(request.key in seen and request.key not in requests, 'Missing/duplicate break observation')
            old = self._state.get((request.key.camera_id, request.key.local_id))
            require(old is not None, 'Cannot break a newly observed track')
            require(type(request.expected_generation) is int
                    and request.expected_generation == old.generation, 'Stale segment generation')
            require(isinstance(request.reason, str) and bool(request.reason.strip()), 'Missing break reason')
            requests[request.key] = request
        return requests

    def update(self, batch):
        requests = self._validate(batch)
        frame, time = batch.frame_index, batch.timestamp
        state = dict(self._state); next_id = self._next_id; events = []
        # Allocation must not depend on camera/record/feature/request order.
        for key in sorted(requests, key=lambda k: (k.camera_id, k.local_id)):
            require(next_id <= MAX_EFFECTIVE_ID, 'Effective ID namespace exhausted')
            old = state[key.camera_id, key.local_id]
            new = _State(next_id, old.generation + 1, frame, time, frame, time)
            state[key.camera_id, key.local_id] = new
            events.append(SegmentEvent(key, ObservationKey(key.camera_id, old.identity_local_id, old.last_frame),
                ObservationKey(key.camera_id, next_id, frame), old.last_time, new.generation, requests[key].reason))
            next_id += 1
        bindings = []; by_key = {}
        for record in batch.records:
            key = record.key; local = key.camera_id, key.local_id
            old = state.get(local, _State(key.local_id, 0, frame, time, frame, time))
            current = replace(old, last_frame=frame, last_time=time)
            state[local] = current
            effective = ObservationKey(key.camera_id, current.identity_local_id, frame)
            by_key[key] = effective
            bindings.append(SegmentBinding(key, effective, current.generation, current.started_frame, current.started_time))
        records = tuple(replace(record, key=by_key[record.key]) for record in batch.records)
        features = ReIDBatch(tuple(by_key[k] for k in batch.features.keys), batch.features.timestamps,
                             batch.features.embeddings.copy())
        result = SegmentedRound(self.run_id, frame, time, tuple(bindings), records, features, tuple(events))
        self._state, self._next_id = state, next_id
        self._last_frame, self._last_time = frame, time
        return result
