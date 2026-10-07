"""Stateless ground-distance admissibility before partial appearance assignment."""
from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from mtmc.reid.osnet import ObservationKey
from . import pairwise

POLICY = 'ground_distance_before_assignment_v1'


@dataclass(frozen=True)
class GroundObservation:
    key: ObservationKey
    xy: tuple[float, float] | None  # None explicitly denotes unavailable geometry


@dataclass(frozen=True)
class CameraGroundPositions:
    run_id: str
    camera_id: int
    frame_index: int
    timestamp: Fraction
    coordinate_space: str  # Shared scene/plane/unit/calibration-generation identifier
    observations: tuple[GroundObservation, ...]


@dataclass(frozen=True)
class GeometryCandidate:
    left: ObservationKey
    right: ObservationKey
    distance: float | None
    geometry_decision: str
    appearance_eligible: bool
    selected: bool


@dataclass(frozen=True)
class GeometryPairResult:
    policy: str
    coordinate_space: str
    max_distance: float
    unavailable_policy: str
    association: pairwise.PairAssociation
    candidates: tuple[GeometryCandidate, ...]


def _positive_distance(value):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, float, np.integer, np.floating)) or
            not math.isfinite(float(value)) or value <= 0):
        raise ValueError('max_distance must be an explicit finite positive number')
    return float(value)


def project_box_foot(h_world_to_image, xyxy):
    """Raw box-bottom center -> world Z=0; None for near-infinite projection.

    Does not infer visibility, undistort an image or clip the box. A finite
    result is not a guarantee of visible ground contact. Invalid inputs and
    singular matrices raise, rather than silently invoking a fallback.
    """
    h = np.asarray(h_world_to_image, dtype=np.float64)
    box = np.asarray(xyxy, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all() or np.linalg.matrix_rank(h) != 3:
        raise ValueError('Expected a finite invertible 3x3 ground-to-image homography')
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError('Expected a finite positive-area xyxy box')
    inverse = np.linalg.inv(h)
    inverse /= np.linalg.norm(inverse)
    q = inverse @ np.array([box[0]/2 + box[2]/2, box[3], 1.0])
    if not np.isfinite(q).all() or abs(q[2]) <= 1e-12 * np.linalg.norm(q):
        return None
    point = q[:2] / q[2]
    return tuple(map(float, point)) if np.isfinite(point).all() else None


def solve_geometry_assignment(similarities, *, min_similarity, admissible):
    """Same gain objective as baseline, restricted before optimization."""
    threshold = pairwise.validate_threshold(min_similarity)
    scores = np.asarray(similarities, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or np.any(scores < -1) or np.any(scores > 1):
        raise ValueError('Expected a finite two-dimensional cosine matrix in [-1, 1]')
    if not isinstance(admissible, np.ndarray) or admissible.dtype != np.bool_ or admissible.shape != scores.shape:
        raise ValueError('Expected a boolean admissibility array with the cosine-matrix shape')
    # -1 can never satisfy the strict appearance gate, even when threshold=-1.
    # The original solver supplies unmatched dummy columns with zero gain.
    masked = np.where(admissible, scores, -1.0)
    result = pairwise.solve_partial_assignment(masked, min_similarity=threshold)
    if any(not admissible[i, j] or scores[i, j] <= threshold for i, j in result):
        raise RuntimeError('Assignment violated admissibility')
    return result


def _positions(ground, appearance):
    if not isinstance(ground, CameraGroundPositions):
        raise TypeError('Expected CameraGroundPositions')
    if (not isinstance(ground.run_id, str) or not ground.run_id.strip() or
            any(type(x) is not int or x < 0 for x in (ground.camera_id, ground.frame_index)) or
            not isinstance(ground.timestamp, Fraction) or ground.timestamp < 0):
        raise ValueError('Invalid geometry session/camera/frame/exact time')
    if any(getattr(ground, k) != getattr(appearance, k) for k in ('run_id', 'camera_id', 'frame_index', 'timestamp')):
        raise ValueError('Geometry and appearance scopes differ')
    if not isinstance(ground.coordinate_space, str) or not ground.coordinate_space.strip():
        raise ValueError('A shared coordinate-space identifier is required')
    if not isinstance(ground.observations, tuple):
        raise ValueError('Expected an immutable tuple of ground observations')
    expected, result = set(appearance.observations.keys), {}
    for item in ground.observations:
        if not isinstance(item, GroundObservation):
            raise TypeError('Expected GroundObservation')
        if (not isinstance(item.key, ObservationKey) or
                any(type(x) is not int or x < 0 for x in (item.key.camera_id, item.key.local_id, item.key.frame_index))):
            raise ValueError('Invalid ground observation key')
        if item.key not in expected or item.key in result:
            raise ValueError('Unknown or duplicate ground observation key')
        if item.xy is not None:
            if (not isinstance(item.xy, tuple) or len(item.xy) != 2 or any(
                isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating))
                or not math.isfinite(float(x)) for x in item.xy)):
                raise ValueError('Ground position must be a finite numeric pair or explicit None')
            result[item.key] = tuple(map(float, item.xy))
        else:
            result[item.key] = None
    if set(result) != expected:
        raise ValueError('Geometry must cover every appearance key; use explicit None for unavailable points')
    return result


def associate_camera_pair_with_geometry(left, right, left_ground, right_ground, *,
                                        min_similarity, max_distance, unavailable_policy):
    """Gate every candidate before assignment; no GT, IDs or temporal state.

    unavailable_policy is required: 'appearance_only' permits the appearance
    fallback, 'reject' blocks a pair with either unavailable point. This choice
    applies only to explicit missing geometry, never to malformed input.
    """
    threshold = pairwise.validate_threshold(min_similarity)
    limit = _positive_distance(max_distance)
    if unavailable_policy not in ('appearance_only', 'reject'):
        raise ValueError('Explicit unavailable_policy must be appearance_only or reject')
    pairwise._validate_camera(left)
    pairwise._validate_camera(right)
    if left.camera_id == right.camera_id or any(getattr(left, k) != getattr(right, k)
                for k in ('run_id', 'frame_index', 'timestamp', 'descriptor_variant')):
        raise ValueError('Expected distinct synchronized cameras with the same session and descriptor variant')
    lp, rp = _positions(left_ground, left), _positions(right_ground, right)
    if left_ground.coordinate_space != right_ground.coordinate_space:
        raise ValueError('Cannot mix coordinate spaces, units or calibration generations')
    (a, ap), (b, bp) = sorted(((left, lp), (right, rp)), key=lambda item: item[0].camera_id)
    ai = sorted(range(len(a.observations.keys)), key=lambda i: a.observations.keys[i].local_id)
    bi = sorted(range(len(b.observations.keys)), key=lambda i: b.observations.keys[i].local_id)
    ak, bk = [a.observations.keys[i] for i in ai], [b.observations.keys[i] for i in bi]
    scores = np.clip(a.observations.embeddings[ai].astype(np.float64) @
                     b.observations.embeddings[bi].astype(np.float64).T, -1, 1)
    admissible = np.zeros(scores.shape, dtype=bool)
    decisions = {}
    for i, key_a in enumerate(ak):
        for j, key_b in enumerate(bk):
            pa, pb = ap[key_a], bp[key_b]
            if pa is None or pb is None:
                permitted = unavailable_policy == 'appearance_only'
                distance = None
                decision = 'unavailable_appearance_fallback' if permitted else 'unavailable_rejected'
            else:
                distance = math.dist(pa, pb)
                if not math.isfinite(distance):
                    raise ValueError('Ground distance overflow')
                permitted = distance <= limit
                decision = 'within_distance' if permitted else 'outside_distance'
            admissible[i, j] = permitted
            decisions[i, j] = distance, decision
    pairs = solve_geometry_assignment(scores, min_similarity=threshold, admissible=admissible)
    selected = set(pairs)
    used_a, used_b = {i for i, _ in pairs}, {j for _, j in pairs}

    def unmatched(keys, matrix, mask, used):
        items = []
        for i, key in enumerate(keys):
            if i in used:
                continue
            best = float(matrix[i].max()) if matrix.shape[1] else None
            reason = ('empty_opposite_camera' if best is None else
                      'no_candidate_above_threshold' if best <= threshold else
                      'geometry_blocked' if not np.any((matrix[i] > threshold) & mask[i]) else
                      'assignment_competition')
            items.append(pairwise.UnmatchedObservation(key, reason, best))
        return tuple(items)

    matches = tuple(pairwise.PairMatch(ak[i], bk[j], float(scores[i, j])) for i, j in pairs)
    ua, ub = unmatched(ak, scores, admissible, used_a), unmatched(bk, scores.T, admissible.T, used_b)
    candidates = tuple(GeometryCandidate(ak[i], bk[j], distance, decision,
                       bool(scores[i, j] > threshold), (i, j) in selected)
                       for (i, j), (distance, decision) in decisions.items())
    if left.camera_id != a.camera_id:
        matches = tuple(sorted((pairwise.PairMatch(m.right, m.left, m.cosine_similarity) for m in matches),
                               key=lambda m: m.left.local_id))
        ua, ub = ub, ua
        candidates = tuple(sorted((GeometryCandidate(c.right, c.left, c.distance, c.geometry_decision,
                                  c.appearance_eligible, c.selected) for c in candidates),
                                  key=lambda c: (c.left.local_id, c.right.local_id)))
    association = pairwise.PairAssociation(left.run_id, left.frame_index, left.timestamp, left.descriptor_variant,
                                           left.camera_id, right.camera_id, threshold, matches, ua, ub)
    return GeometryPairResult(POLICY, left_ground.coordinate_space, limit, unavailable_policy, association, candidates)


def group_geometry_associations(results):
    """Validate a common geometry policy before invoking the existing grouper.

    Returned FrameGroups retains the original grouping contract. The caller
    must separately preserve the GeometryPairResult policy metadata in its
    run configuration and audit output.
    """
    from .grouping import group_pair_associations
    results = tuple(results)
    if not results or not all(isinstance(r, GeometryPairResult) for r in results):
        raise ValueError('Expected nonempty GeometryPairResult inputs')
    first = results[0]
    _positive_distance(first.max_distance)
    if (first.policy != POLICY or first.unavailable_policy not in ('appearance_only', 'reject') or
            not isinstance(first.coordinate_space, str) or not first.coordinate_space.strip()):
        raise ValueError('Invalid geometry grouping policy')
    if any(any(getattr(r, k) != getattr(first, k) for k in
               ('policy', 'coordinate_space', 'max_distance', 'unavailable_policy')) for r in results):
        raise ValueError('Cannot group mixed geometry settings or coordinate spaces')
    return group_pair_associations(r.association for r in results)
