"""Model-scoped, dimension-explicit geometry association.

The gain objective, geometry rules and result records retain the frozen baseline
semantics. FeatureSpace stays attached until the checked grouping boundary.
"""
from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch, validate_batch, require
from . import pairwise
from .geometry import (POLICY, GeometryCandidate, GeometryPairResult,
    _positions, _positive_distance, solve_geometry_assignment, group_geometry_associations)


@dataclass(frozen=True)
class CameraFeatures:
    run_id: str
    camera_id: int
    frame_index: int
    timestamp: Fraction
    descriptor_variant: str
    observations: FeatureBatch


@dataclass(frozen=True)
class FeaturePairResult:
    space: FeatureSpace
    result: GeometryPairResult


def validate_camera(camera):
    require(isinstance(camera, CameraFeatures), 'Expected CameraFeatures')
    require(isinstance(camera.run_id,str) and bool(camera.run_id.strip()), 'Missing run ID')
    require(type(camera.camera_id) is int and camera.camera_id >= 0
            and type(camera.frame_index) is int and camera.frame_index >= 0
            and isinstance(camera.timestamp,Fraction) and camera.timestamp >= 0,
            'Invalid camera/frame/time')
    require(camera.descriptor_variant in ('latest','mean'), 'Invalid descriptor variant')
    require(isinstance(camera.observations, FeatureBatch), 'Expected FeatureBatch')
    require(isinstance(camera.observations.space,FeatureSpace),'Invalid feature space')
    validate_batch(camera.observations,run_id=camera.run_id,space=camera.observations.space,
                   frame_index=camera.frame_index,timestamp=camera.timestamp)
    require(all(k.camera_id==camera.camera_id for k in camera.observations.keys), 'Mixed camera keys')


def associate_feature_pair(left,right,left_ground,right_ground,*,min_similarity,max_distance,unavailable_policy):
    validate_camera(left);validate_camera(right)
    require(left.observations.space==right.observations.space,'Cannot compare different model feature spaces')
    result=_associate(left,right,left_ground,right_ground,min_similarity=min_similarity,
                      max_distance=max_distance,unavailable_policy=unavailable_policy)
    return FeaturePairResult(left.observations.space,result)


def group_feature_pairs(results):
    results=tuple(results)
    require(bool(results) and all(isinstance(r,FeaturePairResult) for r in results),'Expected model-scoped pair results')
    require(isinstance(results[0].space,FeatureSpace) and all(r.space==results[0].space for r in results),
            'Cannot group links from different feature spaces')
    return group_geometry_associations(r.result for r in results)


def _associate(left, right, left_ground, right_ground, *,
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
    validate_camera(left)
    validate_camera(right)
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

