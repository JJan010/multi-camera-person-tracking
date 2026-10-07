"""Compose existing adapters without ground truth or experiment-script imports."""
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import combinations
from time import perf_counter_ns

import numpy as np

from mtmc.association.controlled_merge import ControlledMergeIdentityManager
from mtmc.association.geometry import (
    CameraGroundPositions, GroundObservation, associate_camera_pair_with_geometry,
    group_geometry_associations, project_box_foot,
)
from mtmc.association.grouping import key_order
from mtmc.association.pairwise import CameraAppearance, validate_threshold
from mtmc.reid.crops import build_person_crops
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch


def require(condition, message):
    if not condition:
        raise ValueError(message)


class IdentityStage:
    """Geometry, complete-support grouping, then one persistent ID manager.

    All track records are retained. Tracks without a current embedding enter
    as singletons: no fabricated descriptor, new appearance link or geometry
    match. Their existing local-to-global binding can still provide continuity.
    Geometry never overrides an already retained local identity binding.
    """
    def __init__(self, run_id, matrices, coordinate_space, *, variant, threshold,
                 max_distance, unavailable_policy, identity_configuration):
        require(len(matrices) >= 2 and all(type(c) is int and c >= 0 for c in matrices),
                "At least two distinct camera IDs required")
        require(isinstance(coordinate_space, str) and bool(coordinate_space.strip()),
                "Coordinate space required")
        require(variant in ("latest", "mean"), "Invalid descriptor variant")
        require(type(max_distance) in (int, float) and np.isfinite(max_distance) and max_distance > 0,
                "Positive finite distance required")
        require(unavailable_policy in ("appearance_only", "reject"), "Unknown geometry fallback")
        self.matrices = {}
        for camera, value in matrices.items():
            matrix = np.array(value, dtype=np.float64, copy=True)
            require(matrix.shape == (3, 3) and np.isfinite(matrix).all()
                    and np.linalg.matrix_rank(matrix) == 3, "Invalid homography")
            self.matrices[camera] = matrix
        self.cameras = tuple(sorted(matrices))
        self.run_id, self.coordinate_space, self.variant = run_id, coordinate_space, variant
        self.threshold = validate_threshold(threshold)
        self.geometry_configuration = dict(max_distance=float(max_distance), unavailable_policy=unavailable_policy)
        cfg = identity_configuration
        require(set(cfg) == {"max_idle_seconds", "min_support_rounds", "min_support_seconds", "max_evidence_gap"},
                "Unexpected identity configuration keys")
        self.manager = ControlledMergeIdentityManager(
            run_id=run_id, max_idle=Fraction(cfg["max_idle_seconds"]), descriptor_variant=variant,
            min_similarity=self.threshold, min_support_rounds=cfg["min_support_rounds"],
            min_support_seconds=Fraction(cfg["min_support_seconds"]),
            max_evidence_gap=Fraction(cfg["max_evidence_gap"]))

    def update(self, frame_index, timestamp, descriptors, records):
        require(type(frame_index) is int and frame_index >= 0
                and isinstance(timestamp, Fraction) and timestamp >= 0, "Invalid round context")
        require(isinstance(descriptors, ReIDBatch), "Expected ReIDBatch")
        keys, vectors = descriptors.keys, descriptors.embeddings
        require(len(set(keys)) == len(keys) and len(descriptors.timestamps) == len(keys), "Invalid descriptor keys/times")
        require(isinstance(vectors, np.ndarray) and vectors.dtype == np.float32
                and vectors.shape == (len(keys), 512) and np.isfinite(vectors).all()
                and np.allclose(np.linalg.norm(vectors, axis=1), 1, rtol=0, atol=1e-5), "Invalid descriptors")
        rows = {}
        for record in records:
            key = record.key
            require(isinstance(key, ObservationKey) and all(type(v) is int and v >= 0 for v in key_order(key))
                    and key.frame_index == frame_index and key.camera_id in self.cameras
                    and key not in rows, "Invalid or duplicate track key")
            box = np.asarray(record.source_xyxy, dtype=np.float64)
            require(box.shape == (4,) and np.isfinite(box).all() and np.all(box[2:] > box[:2]), "Invalid tracked box")
            rows[key] = record
        expected = {key for key, record in rows.items() if record.crop_xyxy_int is not None}
        require(set(keys) == expected and all(t == timestamp and isinstance(t, Fraction) for t in descriptors.timestamps),
                "Current embedding/track coverage or timestamp mismatch")
        appearances, grounds = {}, {}
        t0 = perf_counter_ns()
        for camera in self.cameras:
            indices = sorted((i for i, key in enumerate(keys) if key.camera_id == camera), key=lambda i: key_order(keys[i]))
            selected = tuple(keys[i] for i in indices)
            features = ReIDBatch(selected, (timestamp,) * len(selected), vectors[indices].copy())
            appearances[camera] = CameraAppearance(self.run_id, camera, frame_index, timestamp, self.variant, features)
            positions = tuple(GroundObservation(key, project_box_foot(self.matrices[camera], rows[key].source_xyxy))
                              for key in selected)
            grounds[camera] = CameraGroundPositions(self.run_id, camera, frame_index, timestamp,
                                                    self.coordinate_space, positions)
        t1 = perf_counter_ns()
        pairs = tuple(associate_camera_pair_with_geometry(appearances[a], appearances[b], grounds[a], grounds[b],
                      min_similarity=self.threshold, **self.geometry_configuration) for a, b in combinations(self.cameras, 2))
        t2 = perf_counter_ns()
        grouped = group_geometry_associations(pairs)
        missing = tuple(sorted(set(rows) - expected, key=key_order))
        # Preserve every local track, including fully outside, unencoded boxes.
        groups = (*grouped.groups, *((key,) for key in missing))
        grouped = replace(grouped, groups=tuple(sorted(groups, key=lambda g: tuple(key_order(k) for k in g))))
        t3 = perf_counter_ns()
        result = self.manager.update(grouped)
        t4 = perf_counter_ns()
        require({a.key for a in result.assignments} == set(rows), "Global assignment coverage differs")
        timing = {name: (end-start)/1e6 for name, start, end in (
            ("project_ms", t0, t1), ("associate_ms", t1, t2), ("group_ms", t2, t3), ("identity_ms", t3, t4))}
        return result, grouped, pairs, grounds, missing, timing


@dataclass
class PipelineRound:
    batch: object
    detections: tuple
    tracks: tuple
    prepared: object
    features: ReIDBatch
    history: object
    identities: object
    groups: object
    pairs: tuple
    grounds: dict
    unencoded: tuple
    timings: dict


class MTMCPipeline:
    """One owner, fresh state per run; fail-stop on errors, never retry a round.

    Models and camera trackers are persistent injected adapters. No disk I/O or
    GT access occurs in step(). The caller must release the returned round after
    recording it, because it temporarily owns RGB buffers and crop views.
    """
    def __init__(self, detector, trackers, encoder, identity_stage, *, synchronize,
                 history_max_observations=8, history_max_age=Fraction(1)):
        require(set(trackers) == set(identity_stage.cameras), "Tracker/calibration cameras differ")
        self.detector, self.trackers, self.encoder = detector, dict(trackers), encoder
        self.identity_stage, self.synchronize = identity_stage, synchronize
        self.history = AppearanceHistory(identity_stage.run_id, max_observations=history_max_observations,
                                         max_age=history_max_age)
        self.next_frame = 0
        self.failed = False

    def step(self, batches):
        if self.failed:
            raise RuntimeError("Failed pipeline cannot be retried; create a new run")
        try:
            return self._step(batches)
        except Exception:
            self.failed = True
            raise

    def _step(self, batches):
        timing = {}
        started = perf_counter_ns()
        batch = next(batches)
        require(batch.frame_index == self.next_frame and batch.timestamp == Fraction(self.next_frame, 30),
                "Pipeline requires consecutive 30 FPS rounds from frame zero")
        require(tuple(sorted(f.camera_id for f in batch.frames)) == self.identity_stage.cameras,
                "Replay camera set differs")
        end = perf_counter_ns(); timing["replay_ms"] = (end-started)/1e6
        t = end
        detections = tuple(self.detector.detect(batch))
        self.synchronize()
        require(tuple(sorted(d.camera_id for d in detections)) == self.identity_stage.cameras, "Detector camera set differs")
        end = perf_counter_ns(); timing["detect_ms"] = (end-t)/1e6
        t = end
        tracks = tuple(self.trackers[d.camera_id].update(d) for d in detections)
        end = perf_counter_ns(); timing["track_ms"] = (end-t)/1e6
        t = end
        prepared = build_person_crops(batch, tracks)
        end = perf_counter_ns(); timing["crop_ms"] = (end-t)/1e6
        t = end
        features = self.encoder.encode(prepared.crops)
        require(features.keys == tuple(c.key for c in prepared.crops)
                and features.timestamps == tuple(c.timestamp for c in prepared.crops), "Encoder changed crop mapping")
        end = perf_counter_ns(); timing["encode_ms"] = (end-t)/1e6
        t = end
        history = self.history.update(batch.frame_index, batch.timestamp, features)
        end = perf_counter_ns(); timing["history_ms"] = (end-t)/1e6
        descriptors = history.mean if self.identity_stage.variant == "mean" else history.latest
        t = end
        result, groups, pairs, grounds, unencoded, identity_timing = self.identity_stage.update(
            batch.frame_index, batch.timestamp, descriptors, prepared.records)
        end = perf_counter_ns()
        timing.update(identity_timing)
        timing["association_block_ms"] = (end-t)/1e6
        timing["core_ms"] = (end-started)/1e6
        self.next_frame += 1
        return PipelineRound(batch, detections, tracks, prepared, features, history, result,
                             groups, pairs, grounds, unencoded, timing)
