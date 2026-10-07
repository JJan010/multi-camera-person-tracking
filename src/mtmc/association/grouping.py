"""Conservative, stateless grouping of a complete set of camera-pair decisions."""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math

from mtmc.reid.osnet import ObservationKey
from .pairwise import PairAssociation, PairMatch, UnmatchedObservation, validate_threshold

POLICY = "complete_support_greedy_v1"


@dataclass(frozen=True)
class GroupingDecision:
    left: ObservationKey
    right: ObservationKey
    cosine_similarity: float
    outcome: str
    conflicting_cameras: tuple[int, ...] = ()
    missing_support: tuple[tuple[ObservationKey, ObservationKey], ...] = ()


@dataclass(frozen=True)
class FrameGroups:
    run_id: str
    frame_index: int
    timestamp: Fraction
    descriptor_variant: str
    min_similarity: float
    policy: str
    groups: tuple[tuple[ObservationKey, ...], ...]
    decisions: tuple[GroupingDecision, ...]


def key_order(key):
    return key.camera_id, key.local_id, key.frame_index


def edge_key(a, b):
    return tuple(sorted((a, b), key=key_order))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _read_pairs(results):
    results = tuple(results)
    _require(bool(results), "At least one camera-pair result is required")
    _require(all(isinstance(x, PairAssociation) for x in results), "Expected PairAssociation results")
    first = results[0]
    _require(isinstance(first.run_id, str) and bool(first.run_id.strip()), "Missing session/generation scope")
    _require(type(first.frame_index) is int and first.frame_index >= 0, "Invalid frame index")
    _require(isinstance(first.timestamp, Fraction) and first.timestamp >= 0, "Invalid scene timestamp")
    _require(first.descriptor_variant in ("latest", "mean"), "Invalid descriptor variant")
    threshold = validate_threshold(first.min_similarity)
    camera_nodes, seen_pairs, edges = {}, set(), {}
    for result in results:
        _require(all(getattr(result, k) == getattr(first, k) for k in
                     ("run_id", "frame_index", "timestamp", "descriptor_variant")),
                 "Cannot mix sessions, frames, times or descriptor variants")
        _require(isinstance(result.timestamp, Fraction), "Expected exact scene timestamps")
        _require(validate_threshold(result.min_similarity) == threshold, "Cannot mix pair thresholds")
        a, b = result.left_camera, result.right_camera
        _require(all(type(c) is int and c >= 0 for c in (a, b)) and a != b, "Invalid camera pair")
        pair = tuple(sorted((a, b)))
        _require(pair not in seen_pairs, "Duplicate camera-pair result")
        seen_pairs.add(pair)
        local_nodes = {a: set(), b: set()}

        def add(key, camera):
            _require(isinstance(key, ObservationKey), "Expected ObservationKey")
            _require(all(type(v) is int and v >= 0 for v in key_order(key)), "Invalid observation key")
            _require(key.camera_id == camera and key.frame_index == first.frame_index,
                     "Observation camera/frame differs from pair context")
            _require(key not in local_nodes[camera], "Observation repeated in matched/unmatched pair inputs")
            local_nodes[camera].add(key)

        for match in result.matches:
            _require(isinstance(match, PairMatch), "Expected PairMatch")
            add(match.left, a)
            add(match.right, b)
            score = float(match.cosine_similarity)
            _require(math.isfinite(score) and threshold < score <= 1, "Accepted score violates pair gate")
            edge = edge_key(match.left, match.right)
            _require(edge not in edges, "Duplicate accepted edge")
            edges[edge] = score
        for unmatched, camera in ((result.unmatched_left, a), (result.unmatched_right, b)):
            for item in unmatched:
                _require(isinstance(item, UnmatchedObservation), "Expected UnmatchedObservation")
                add(item.key, camera)
        for camera, nodes in local_nodes.items():
            if camera in camera_nodes:
                _require(camera_nodes[camera] == nodes, "A camera has different observations across its pair results")
            camera_nodes[camera] = nodes
    _require(seen_pairs == set(combinations(sorted(camera_nodes), 2)),
             "All camera pairs must be present, including pairs with no matches")
    nodes = tuple(sorted(set().union(*camera_nodes.values()), key=key_order))
    return first, threshold, nodes, edges


def group_pair_associations(results):
    """Partition current observations; never create persistent global IDs.

    Inspect accepted edges by descending score, then canonical observation keys.
    Merge groups only if their cameras are disjoint and every cross-group member
    pair has an accepted input edge. Unmatched observations remain singletons.
    This deterministic greedy heuristic is not a global optimum or proof of identity.
    """
    context, threshold, nodes, edges = _read_pairs(results)
    groups = {k: frozenset((k,)) for k in nodes}
    owners = {k: k for k in nodes}
    decisions = []
    ordered_edges = sorted(edges, key=lambda e: (-edges[e], key_order(e[0]), key_order(e[1])))
    for a, b in ordered_edges:
        ga, gb = owners[a], owners[b]
        if ga == gb:
            decisions.append(GroupingDecision(a, b, edges[a, b], "already_in_same_group"))
            continue
        left, right = groups[ga], groups[gb]
        conflict = tuple(sorted({k.camera_id for k in left} & {k.camera_id for k in right}))
        if conflict:
            decisions.append(GroupingDecision(a, b, edges[a, b], "rejected_camera_conflict", conflict))
            continue
        missing = tuple(sorted((edge_key(x, y) for x in left for y in right if edge_key(x, y) not in edges),
                               key=lambda e: (key_order(e[0]), key_order(e[1]))))
        if missing:
            decisions.append(GroupingDecision(a, b, edges[a, b], "rejected_missing_pair_support",
                                               missing_support=missing))
            continue
        merged = left | right
        representative = min(merged, key=key_order)
        del groups[ga]
        del groups[gb]
        groups[representative] = merged
        for node in merged:
            owners[node] = representative
        decisions.append(GroupingDecision(a, b, edges[a, b], "merged"))
    partition = tuple(sorted((tuple(sorted(group, key=key_order)) for group in groups.values()),
                             key=lambda group: tuple(key_order(k) for k in group)))
    flattened = [k for group in partition for k in group]
    _require(len(flattened) == len(nodes) and set(flattened) == set(nodes), "Grouping lost or duplicated observations")
    for group in partition:
        _require(len({k.camera_id for k in group}) == len(group), "Group contains a repeated camera")
        _require(all(edge_key(a, b) in edges for a, b in combinations(group, 2)), "Group lacks pair support")
    return FrameGroups(context.run_id, context.frame_index, context.timestamp, context.descriptor_variant,
                       threshold, POLICY, partition, tuple(decisions))
