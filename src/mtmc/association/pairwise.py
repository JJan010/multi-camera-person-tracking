"""Stateless, same-time two-camera appearance matching with an unmatched option."""

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from mtmc.reid.osnet import ObservationKey, ReIDBatch


@dataclass(frozen=True)
class CameraAppearance:
    run_id: str
    camera_id: int
    frame_index: int
    timestamp: Fraction
    descriptor_variant: str  # "latest" or "mean"
    observations: ReIDBatch


@dataclass(frozen=True)
class PairMatch:
    left: ObservationKey
    right: ObservationKey
    cosine_similarity: float


@dataclass(frozen=True)
class UnmatchedObservation:
    key: ObservationKey
    reason: str  # empty_opposite_camera / no_candidate_above_threshold / assignment_competition
    best_similarity: float | None


@dataclass(frozen=True)
class PairAssociation:
    run_id: str
    frame_index: int
    timestamp: Fraction
    descriptor_variant: str
    left_camera: int
    right_camera: int
    min_similarity: float
    matches: tuple[PairMatch, ...]
    unmatched_left: tuple[UnmatchedObservation, ...]
    unmatched_right: tuple[UnmatchedObservation, ...]


def validate_threshold(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("min_similarity must be an explicit finite number in [-1, 1]")
    value = float(value)
    if not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("min_similarity must be an explicit finite number in [-1, 1]")
    return value


def solve_partial_assignment(similarities, *, min_similarity):
    """Maximize sum(similarity - threshold), allowing unmatched rows/columns.

    Only strictly positive gains are admissible. A dummy column for each row
    makes rejection possible even for a square matrix. Invalid edges are removed
    before optimization; this is NOT full assignment followed by thresholding.
    The objective does not maximize the number of matches first.
    """
    threshold = validate_threshold(min_similarity)
    scores = np.asarray(similarities, dtype=np.float64)
    if (scores.ndim != 2 or not np.isfinite(scores).all()
            or np.any(scores < -1) or np.any(scores > 1)):
        raise ValueError("Expected a finite two-dimensional cosine matrix in [-1, 1]")
    n, m = scores.shape
    if n == 0 or m == 0:
        return ()
    costs = np.zeros((n, m + n), dtype=np.float64)
    costs[:, :m] = np.where(scores > threshold, threshold - scores, 1.0)
    rows, cols = linear_sum_assignment(costs)
    return tuple((int(i), int(j)) for i, j in zip(rows, cols) if j < m)


def _validate_camera(camera):
    if not isinstance(camera, CameraAppearance):
        raise TypeError("Expected CameraAppearance")
    if not isinstance(camera.run_id, str) or not camera.run_id.strip():
        raise ValueError("A nonempty run/session-generation ID is required")
    if any(type(v) is not int or v < 0 for v in (camera.camera_id, camera.frame_index)):
        raise ValueError("Camera/frame must be nonnegative Python integers")
    if not isinstance(camera.timestamp, Fraction) or camera.timestamp < 0:
        raise ValueError("Timestamp must be a nonnegative Fraction")
    if camera.descriptor_variant not in ("latest", "mean"):
        raise ValueError("Expected descriptor variant latest or mean")
    batch = camera.observations
    if not isinstance(batch, ReIDBatch):
        raise TypeError("Expected ReIDBatch observations")
    features = batch.embeddings
    if (not isinstance(features, np.ndarray) or features.dtype != np.float32
            or features.shape != (len(batch.keys), 512) or len(batch.timestamps) != len(batch.keys)
            or not np.isfinite(features).all()
            or not np.allclose(np.linalg.norm(features, axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError("Expected finite normalized float32 embeddings, shape (N, 512)")
    seen = set()
    for key, timestamp in zip(batch.keys, batch.timestamps):
        if (not isinstance(key, ObservationKey)
                or any(type(v) is not int or v < 0 for v in (key.camera_id, key.local_id, key.frame_index))
                or key.camera_id != camera.camera_id or key.frame_index != camera.frame_index
                or not isinstance(timestamp, Fraction) or timestamp != camera.timestamp):
            raise ValueError("Observation key/time differs from its camera envelope")
        if key.local_id in seen:
            raise ValueError("Duplicate local ID within one camera/frame")
        seen.add(key.local_id)


def associate_camera_pair(left, right, *, min_similarity):
    """Return candidate pairs, not persistent global IDs. No GT or state updates.

    Both camera observations must come from one run, frame, exact scene time and
    descriptor variant. Caller is responsible for synchronized overlapping views.
    Sort camera orientation and local IDs before solving for input-order stability.
    """
    threshold = validate_threshold(min_similarity)
    _validate_camera(left)
    _validate_camera(right)
    if left.camera_id == right.camera_id:
        raise ValueError("Two distinct cameras are required")
    if any(getattr(left, name) != getattr(right, name) for name in
           ("run_id", "frame_index", "timestamp", "descriptor_variant")):
        raise ValueError("Cannot mix runs, frames, timestamps or descriptor variants")
    a, b = sorted((left, right), key=lambda c: c.camera_id)
    ai = sorted(range(len(a.observations.keys)), key=lambda i: a.observations.keys[i].local_id)
    bi = sorted(range(len(b.observations.keys)), key=lambda i: b.observations.keys[i].local_id)
    ak = [a.observations.keys[i] for i in ai]
    bk = [b.observations.keys[i] for i in bi]
    x = a.observations.embeddings[ai].astype(np.float64)
    y = b.observations.embeddings[bi].astype(np.float64)
    scores = np.clip(x @ y.T, -1, 1)
    pairs = solve_partial_assignment(scores, min_similarity=threshold)
    used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}

    def unmatched(keys, matrix, used):
        items = []
        for i, key in enumerate(keys):
            if i in used:
                continue
            best = float(matrix[i].max()) if matrix.shape[1] else None
            reason = ("empty_opposite_camera" if best is None else
                      "no_candidate_above_threshold" if best <= threshold else "assignment_competition")
            items.append(UnmatchedObservation(key, reason, best))
        return tuple(items)

    matches = tuple(PairMatch(ak[i], bk[j], float(scores[i, j])) for i, j in pairs)
    ua, ub = unmatched(ak, scores, used_a), unmatched(bk, scores.T, used_b)
    if left.camera_id != a.camera_id:
        matches = tuple(sorted((PairMatch(p.right, p.left, p.cosine_similarity) for p in matches),
                               key=lambda p: p.left.local_id))
        ua, ub = ub, ua
    return PairAssociation(left.run_id, left.frame_index, left.timestamp, left.descriptor_variant,
                           left.camera_id, right.camera_id, threshold, matches, ua, ub)
