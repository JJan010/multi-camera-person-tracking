"""Causal local-anchor identity registry; never merges established global IDs."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

from mtmc.reid.osnet import ObservationKey
from .grouping import FrameGroups, POLICY as GROUPING_POLICY, key_order
from .pairwise import validate_threshold

POLICY = "local_anchor_no_merge_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, order=True)
class LocalTrackKey:
    camera_id: int
    local_id: int


@dataclass(frozen=True)
class GlobalObservation:
    key: ObservationKey
    global_id: int
    reason: str


@dataclass(frozen=True)
class IdentityDecision:
    members: tuple[ObservationKey, ...]
    anchor_global_ids: tuple[int, ...]
    outcome: str
    conflicting_cameras: tuple[int, ...] = ()


@dataclass(frozen=True)
class IdentityState:
    global_id: int
    status: str  # visible / lost, never a claim of confirmed identity correctness
    last_seen: Fraction
    current_members: tuple[ObservationKey, ...]
    retained_local_tracks: tuple[LocalTrackKey, ...]


@dataclass(frozen=True)
class GlobalFrame:
    run_id: str
    frame_index: int
    timestamp: Fraction
    descriptor_variant: str
    min_similarity: float
    policy: str
    assignments: tuple[GlobalObservation, ...]
    decisions: tuple[IdentityDecision, ...]
    identities: tuple[IdentityState, ...]
    expired_local_tracks: tuple[LocalTrackKey, ...]
    expired_global_ids: tuple[int, ...]


@dataclass(frozen=True)
class _Binding:
    global_id: int
    last_seen: Fraction


class GlobalIdentityManager:
    """One instance per run, descriptor variant and threshold.

    A still-retained (camera, local_id) is an identity anchor. New tracks can
    attach to one existing global ID through a current FrameGroups group.
    Multiple existing IDs are never merged. An absent local binding still
    reserves its identity's camera until its own age exceeds max_idle.
    Simultaneous competing attachments are all rejected, not first-come wins.

    Global IDs start at 1 and are never reused within this manager. They must
    always be interpreted with run_id. Input frames and scene times strictly
    increase. Empty rounds advance expiry. No GT, images, features or wall clock
    are used. All validation and planning precede the state commit.
    """

    def __init__(self, run_id, *, max_idle, descriptor_variant, min_similarity):
        require(isinstance(run_id, str) and bool(run_id.strip()), "Nonempty run_id required")
        require(isinstance(max_idle, Fraction) and max_idle > 0, "max_idle must be a positive Fraction")
        require(descriptor_variant in ("latest", "mean"), "Invalid descriptor variant")
        self.run_id = run_id
        self.max_idle = max_idle
        self.descriptor_variant = descriptor_variant
        self.min_similarity = validate_threshold(min_similarity)
        self._bindings = {}
        self._next_id = 1
        self._last_frame = None
        self._last_time = None

    def _validate(self, frame):
        require(isinstance(frame, FrameGroups), "Expected FrameGroups")
        require(frame.run_id == self.run_id, "Mixed run/session scopes")
        require(frame.descriptor_variant == self.descriptor_variant
                and validate_threshold(frame.min_similarity) == self.min_similarity,
                "Mixed experiment variants or thresholds")
        require(frame.policy == GROUPING_POLICY, "Unsupported upstream grouping policy")
        require(type(frame.frame_index) is int and frame.frame_index >= 0, "Invalid frame index")
        require(isinstance(frame.timestamp, Fraction) and frame.timestamp >= 0, "Invalid scene timestamp")
        require(self._last_frame is None or frame.frame_index > self._last_frame, "Frames must strictly increase")
        require(self._last_time is None or frame.timestamp > self._last_time, "Scene time must strictly increase")
        require(isinstance(frame.groups, tuple), "Expected immutable group partition")
        seen, groups = set(), []
        for group in frame.groups:
            require(isinstance(group, tuple) and bool(group), "Empty or mutable group")
            cameras = set()
            for key in group:
                require(isinstance(key, ObservationKey), "Expected ObservationKey")
                require(all(type(v) is int and v >= 0 for v in key_order(key)), "Invalid observation key")
                require(key.frame_index == frame.frame_index, "Mixed observation frames")
                require(key not in seen and key.camera_id not in cameras, "Duplicate observation or camera in group")
                seen.add(key)
                cameras.add(key.camera_id)
            groups.append(tuple(sorted(group, key=key_order)))
        return tuple(sorted(groups, key=lambda group: tuple(key_order(k) for k in group)))

    @staticmethod
    def _local(key):
        return LocalTrackKey(key.camera_id, key.local_id)

    def update(self, frame):
        groups = self._validate(frame)
        now = frame.timestamp
        old_ids = {b.global_id for b in self._bindings.values()}
        expired = tuple(sorted(k for k, b in self._bindings.items() if now - b.last_seen > self.max_idle))
        live = {k: b for k, b in self._bindings.items() if now - b.last_seen <= self.max_idle}
        live_ids = {b.global_id for b in live.values()}
        expired_ids = tuple(sorted(old_ids - live_ids))
        reserved = {(b.global_id, k.camera_id): k for k, b in live.items()}
        require(len(reserved) == len(live), "Corrupt state: repeated identity/camera binding")

        # Snapshot all prior anchors before considering any attachment in this round.
        plans, requests = [], defaultdict(list)
        for index, group in enumerate(groups):
            anchors = tuple(sorted({live[self._local(k)].global_id for k in group if self._local(k) in live}))
            new = tuple(k for k in group if self._local(k) not in live)
            conflict = ()
            if len(anchors) > 1:
                outcome = "rejected_multiple_existing_ids"
            elif not anchors:
                outcome = "new_identity"
            elif not new:
                outcome = "continued"
            else:
                gid = anchors[0]
                conflict = tuple(sorted(k.camera_id for k in new if (gid, k.camera_id) in reserved))
                outcome = "rejected_reserved_camera" if conflict else "attached"
                if not conflict:
                    for key in new:
                        requests[gid, key.camera_id].append(index)
            plans.append({"members": group, "anchors": anchors, "new": new,
                          "outcome": outcome, "conflict": conflict})
        # Reject all candidate groups that compete for an identity/camera slot.
        competition = defaultdict(set)
        for (_, camera), indices in requests.items():
            if len(indices) > 1:
                for index in indices:
                    competition[index].add(camera)
        for index, cameras in competition.items():
            plans[index]["outcome"] = "rejected_competing_attachments"
            plans[index]["conflict"] = tuple(sorted(cameras))

        assignments, decisions = [], []
        staged = dict(live)
        next_id = self._next_id
        for plan in plans:
            group, anchors, new = plan["members"], plan["anchors"], plan["new"]
            outcome = plan["outcome"]
            for key in group:
                local = self._local(key)
                if local in live:
                    gid = live[local].global_id
                    staged[local] = _Binding(gid, now)
                    assignments.append(GlobalObservation(key, gid, "local_continuity"))
            if new:
                if outcome == "attached":
                    gid, reason = anchors[0], "group_attachment"
                else:
                    # New members of a conflicted group keep their mutual support,
                    # but cannot borrow either existing ID. They form one fresh ID.
                    gid, reason = next_id, "new_identity"
                    next_id += 1
                for key in new:
                    staged[self._local(key)] = _Binding(gid, now)
                    assignments.append(GlobalObservation(key, gid, reason))
            decisions.append(IdentityDecision(group, anchors, outcome, plan["conflict"]))
        assignments = tuple(sorted(assignments, key=lambda a: key_order(a.key)))
        expected = {k for group in groups for k in group}
        require(len(assignments) == len(expected) and {a.key for a in assignments} == expected,
                "Global assignment lost or duplicated observations")
        slots = [(a.global_id, a.key.camera_id) for a in assignments]
        require(len(slots) == len(set(slots)), "Two current tracks from one camera share a global ID")
        retained_slots = [(b.global_id, k.camera_id) for k, b in staged.items()]
        require(len(retained_slots) == len(set(retained_slots)), "Two retained tracks reserve the same identity/camera")
        current = defaultdict(list)
        retained = defaultdict(list)
        for item in assignments:
            current[item.global_id].append(item.key)
        for key, binding in staged.items():
            retained[binding.global_id].append(key)
        states = tuple(IdentityState(
            gid, "visible" if current[gid] else "lost", max(staged[k].last_seen for k in retained[gid]),
            tuple(current[gid]), tuple(sorted(retained[gid]))) for gid in sorted(retained))
        result = GlobalFrame(self.run_id, frame.frame_index, now, self.descriptor_variant, self.min_similarity,
                             POLICY, assignments, tuple(decisions), states, expired, expired_ids)
        self._bindings = staged
        self._next_id = next_id
        self._last_frame = frame.frame_index
        self._last_time = now
        return result
