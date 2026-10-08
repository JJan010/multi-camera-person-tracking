"""Scene-aware paired tracking/identity runtime over already encoded candidates.

No files, models, ground truth or evaluation code are accessed here. One object
owns both independent variants; every call advances all cameras by one frame.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
from fractions import Fraction
import json

import numpy as np

from mtmc.data.scene import RuntimeScene, require
from mtmc.pipeline.core import IdentityStage
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack
from mtmc.tracking.competitive import CompetitiveByteTrack

VARIANTS = ('staged', 'competitive_iou')


@dataclass(frozen=True)
class CameraCandidates:
    camera_id: int
    boxes: np.ndarray                    # float32 (N,4), raw original-image xyxy
    scores: np.ndarray                   # float32 (N,)
    embedding_rows: tuple[int | None, ...]
    features: tuple[np.ndarray | None, ...]  # float32 (512,), L2 normalized


@dataclass(frozen=True)
class CandidateRound:
    source_run_id: str
    frame_index: int
    timestamp: Fraction
    cameras: tuple[CameraCandidates, ...]


def json_record(value):
    """Detach dataclass output, with exact rational scene times serialized as text."""
    def default(item):
        if isinstance(item, Fraction):
            return str(item)
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f'Unsupported output type: {type(item).__name__}')
    return json.loads(json.dumps(value, default=default, allow_nan=False))


def validate_policy(value):
    """Copy the frozen experiment settings; do not adapt them to a new scene."""
    policy = deepcopy(value)
    require(set(policy) == {'schema_version', 'variants', 'local', 'global'}
            and policy['schema_version'] == 1 and policy['variants'] == list(VARIANTS),
            'Unsupported paired policy')
    local = policy['local']
    require(set(local) == {'tracker', 'appearance_threshold', 'strong_history_size',
                          'strong_history_max_age_frames', 'weak_motion_min_iou'},
            'Unexpected local policy keys')
    # These constants belong to the frozen CandidateTrack/refinement code.
    require(local['strong_history_size'] == 8 and local['strong_history_max_age_frames'] == 30
            and local['weak_motion_min_iou'] == .5, 'Policy differs from pinned local implementation')
    tracker = local['tracker']
    require(set(tracker) == {'track_activation_threshold', 'lost_track_buffer',
                            'minimum_matching_threshold', 'frame_rate', 'minimum_consecutive_frames'},
            'Unexpected tracker settings')
    for name in ('track_activation_threshold', 'minimum_matching_threshold'):
        require(type(tracker[name]) in (int, float) and np.isfinite(tracker[name])
                and 0 < tracker[name] <= 1, f'Invalid {name}')
    for name in ('lost_track_buffer', 'minimum_consecutive_frames'):
        require(type(tracker[name]) is int and tracker[name] > 0, f'Invalid {name}')
    require(type(tracker['frame_rate']) is int and tracker['frame_rate'] == 30,
            'Frozen local history is defined for 30 FPS')
    tau = local['appearance_threshold']
    require(type(tau) in (int, float) and np.isfinite(tau) and -1 <= tau <= 1,
            'Invalid local appearance threshold')
    glob = policy['global']
    require(set(glob) == {'history', 'appearance_variant', 'appearance_threshold', 'geometry', 'identity'}
            and set(glob['history']) == {'max_observations', 'max_age_seconds'}
            and set(glob['geometry']) == {'max_distance', 'unavailable_policy'},
            'Unexpected global policy keys')
    # Remaining values are validated by the history and identity constructors.
    return policy


class PairedAssociation:
    """Sequential owner of staged + competitive_iou local/global state.

    Preflight validates/copies *all* cameras before any update. A malformed
    input is retryable. An exception during stateful work fails the whole run:
    partial updates are not rolled back or silently retried. Not thread-safe.
    Candidate order inside a camera is significant; camera input order is not.
    Embedding rows count valid crops in sorted-camera/candidate order from zero.
    """
    def __init__(self, scene, policy, *, run_id, source_run_id):
        require(isinstance(scene, RuntimeScene), 'Pass RuntimeScene, not evaluation inputs')
        require(all(isinstance(x, str) and x.strip() for x in (run_id, source_run_id)), 'Missing run scope')
        self._policy = validate_policy(policy)
        require(type(scene.fps) is int and scene.fps == self._policy['local']['tracker']['frame_rate']
                and type(scene.rounds) is int and scene.rounds > 0, 'Scene timing differs from frozen policy')
        ids = scene.camera_ids
        require(len(ids) >= 2 and all(type(c) is int and c >= 0 for c in ids)
                and ids == tuple(sorted(set(ids))), 'Invalid scene cameras')
        require(all(type(v) is int and v > 0 for c in scene.cameras for v in (c.width,c.height)),
                'Invalid camera dimensions')
        self.run_id, self.source_run_id = run_id, source_run_id
        self.fps, self.rounds, self.camera_ids = scene.fps, scene.rounds, ids
        self.sizes = {c.camera_id: (c.width, c.height) for c in scene.cameras}
        self.trackers, self.histories, self.stages = {}, {}, {}
        local, glob = self._policy['local'], self._policy['global']
        common = dict(**local['tracker'], direct_output=True, appearance_threshold=local['appearance_threshold'])
        for name in VARIANTS:
            scope = run_id + '/' + name
            self.trackers[name] = {
                c: (ExperimentalByteTrack(**common) if name == 'staged'
                    else CompetitiveByteTrack(**common, refinement='iou')) for c in ids}
            self.histories[name] = AppearanceHistory(scope,
                max_observations=glob['history']['max_observations'],
                max_age=Fraction(glob['history']['max_age_seconds']))
            self.stages[name] = IdentityStage(scope, scene.matrices, scene.coordinate_space,
                variant=glob['appearance_variant'], threshold=glob['appearance_threshold'],
                **glob['geometry'], identity_configuration=glob['identity'])
        self.next_frame = self.next_embedding_row = 0
        self.failed = False

    @property
    def policy(self):
        return deepcopy(self._policy)

    def _preflight(self, batch):
        require(isinstance(batch, CandidateRound) and batch.source_run_id == self.source_run_id,
                'Mixed candidate scope')
        require(type(batch.frame_index) is int and batch.frame_index == self.next_frame < self.rounds
                and isinstance(batch.timestamp, Fraction)
                and batch.timestamp == Fraction(self.next_frame, self.fps), 'Nonconsecutive scene frame/time')
        require(isinstance(batch.cameras, tuple) and all(isinstance(c, CameraCandidates) for c in batch.cameras),
                'Invalid camera candidates')
        ids = [c.camera_id for c in batch.cameras]
        require(all(type(c) is int for c in ids) and sorted(ids) == list(self.camera_ids),
                'Missing, unexpected or duplicate candidate camera')
        prepared, next_row = [], self.next_embedding_row
        for camera in sorted(batch.cameras, key=lambda c: c.camera_id):
            boxes, scores = camera.boxes, camera.scores
            require(isinstance(boxes, np.ndarray) and boxes.dtype == np.float32 and boxes.ndim == 2
                    and boxes.shape[1] == 4 and np.isfinite(boxes).all()
                    and np.all(boxes[:, 2:] > boxes[:, :2]), 'Invalid candidate boxes')
            require(isinstance(scores, np.ndarray) and scores.dtype == np.float32
                    and scores.shape == (len(boxes),) and np.isfinite(scores).all()
                    and np.all((scores >= 0) & (scores <= 1)), 'Invalid candidate scores')
            require(isinstance(camera.embedding_rows, tuple) and isinstance(camera.features, tuple)
                    and len(camera.embedding_rows) == len(camera.features) == len(boxes), 'Candidate mapping differs')
            features, geometry = [], []
            for box, row, vector in zip(boxes, camera.embedding_rows, camera.features):
                bounds, fraction = crop_geometry(box, *self.sizes[camera.camera_id])
                geometry.append((bounds, fraction))
                if bounds is None:
                    require(row is None and vector is None, 'Fully outside candidate has an embedding')
                    features.append(None)
                    continue
                require(type(row) is int and row == next_row, 'Noncontiguous or reused candidate embedding row')
                require(isinstance(vector, np.ndarray) and vector.dtype == np.float32
                        and vector.shape == (512,) and np.isfinite(vector).all()
                        and abs(float(np.linalg.norm(vector)) - 1) <= 1e-5, 'Invalid normalized candidate embedding')
                next_row += 1
                features.append(vector.copy())
            prepared.append((CameraCandidates(camera.camera_id, boxes.copy(), scores.copy(),
                                             camera.embedding_rows, tuple(features)), tuple(geometry)))
        return prepared, next_row

    def step(self, batch):
        if self.failed:
            raise RuntimeError('Paired run failed; construct fresh state and replay from frame zero')
        prepared, next_row = self._preflight(batch)
        try:
            output = self._advance(batch, prepared)
        except Exception:
            self.failed = True
            raise
        self.next_frame += 1
        self.next_embedding_row = next_row
        return output

    def _advance(self, batch, prepared):
        output = {'run_id': self.run_id, 'source_run_id': self.source_run_id,
                  'frame_index': batch.frame_index, 'timestamp': str(batch.timestamp),
                  'variants': {}, 'refinements': {}, 'refinement_counters': {}}
        for name in VARIANTS:
            cameras, records, keys, vectors, events = [], [], [], [], []
            counters = {}
            for candidates, geometry in prepared:
                c = candidates.camera_id
                tracker = self.trackers[name][c]
                result = tracker.update_candidates(candidates.boxes, candidates.scores, candidates.features)
                indices = list(tracker.selected_indices)
                ids = result.tracker_id.tolist()
                require(len(ids) == len(indices) == len(set(ids)) == len(set(indices))
                        and all(type(i) is int and i >= 0 for i in ids)
                        and all(type(i) is int and 0 <= i < len(candidates.boxes) for i in indices),
                        'Invalid selected candidate/ID mapping')
                require(np.array_equal(result.xyxy, candidates.boxes[indices])
                        and np.array_equal(result.confidence, candidates.scores[indices]), 'Selected candidate values differ')
                cameras.append({'camera': c, 'local_ids': ids, 'xyxy': result.xyxy.tolist(),
                                'confidence': result.confidence.tolist(), 'detection_indices': indices,
                                'embedding_rows': [candidates.embedding_rows[i] for i in indices]})
                for p in sorted(range(len(ids)), key=lambda i: ids[i]):
                    i = indices[p]
                    key = ObservationKey(c, ids[p], batch.frame_index)
                    bounds, fraction = geometry[i]
                    records.append(CropRecord(key, float(candidates.scores[i]),
                        tuple(float(x) for x in candidates.boxes[i]), bounds, fraction))
                    if bounds is not None:
                        keys.append(key); vectors.append(candidates.features[i])
                if name != 'staged':
                    events.extend({'camera': c, **event} for event in tracker.refinement_events)
                    for key, value in tracker.refinement_counts.items():
                        counters[key] = counters.get(key, 0) + value
            features = np.stack(vectors).astype(np.float32) if vectors else np.empty((0,512), np.float32)
            observations = ReIDBatch(tuple(keys), (batch.timestamp,) * len(keys), features)
            history = self.histories[name].update(batch.frame_index, batch.timestamp, observations)
            descriptors = history.mean if self._policy['global']['appearance_variant'] == 'mean' else history.latest
            identity, *_ = self.stages[name].update(batch.frame_index, batch.timestamp, descriptors, tuple(records))
            output['variants'][name] = {'cameras': cameras, 'identity': json_record(asdict(identity))}
            output['refinements'][name] = deepcopy(events)
            output['refinement_counters'][name] = counters
        return output
