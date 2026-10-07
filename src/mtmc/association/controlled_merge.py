"""Causal, whole-visible-group confirmation for merging retained global IDs."""

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction

from mtmc.reid.osnet import ObservationKey
from .global_identity import GlobalIdentityManager, GlobalFrame, GlobalObservation, IdentityState, LocalTrackKey, require
from .grouping import key_order

POLICY = "whole_visible_group_confirmed_merge_v1"


@dataclass(frozen=True)
class PendingMerge:
    global_ids: tuple[int, ...]
    retained_membership: tuple[tuple[int, tuple[LocalTrackKey, ...]], ...]
    first_time: Fraction
    last_time: Fraction
    support_rounds: int


@dataclass(frozen=True)
class MergeDecision:
    global_ids: tuple[int, ...]
    members: tuple[ObservationKey, ...]
    outcome: str
    support_rounds: int = 0
    support_span: Fraction = Fraction(0)
    conflicting_cameras: tuple[int, ...] = ()
    missing_visible_members: tuple[ObservationKey, ...] = ()


@dataclass(frozen=True)
class MergeEvent:
    canonical_global_id: int
    absorbed_global_ids: tuple[int, ...]
    frame_index: int
    timestamp: Fraction
    support_rounds: int
    first_support_time: Fraction
    members: tuple[ObservationKey, ...]
    retained_membership_before: tuple[tuple[int, tuple[LocalTrackKey, ...]], ...]


@dataclass(frozen=True)
class CandidateReset:
    global_ids: tuple[int, ...]
    reason: str


@dataclass(frozen=True)
class ControlledGlobalFrame(GlobalFrame):
    base_assignments: tuple[GlobalObservation, ...]
    merge_decisions: tuple[MergeDecision, ...]
    merge_events: tuple[MergeEvent, ...]
    pending_candidates: tuple[PendingMerge, ...]
    candidate_resets: tuple[CandidateReset, ...]


class ControlledMergeIdentityManager(GlobalIdentityManager):
    """Experimental extension of the frozen no-merge registry.

    First compute the base round, then inspect groups spanning >=2 resulting IDs.
    An eligible group covers ALL currently visible members of these identities,
    and their ALL retained bindings have disjoint camera sets. Eligible evidence
    must recur in every delivered round, within max_evidence_gap, with identical
    retained membership. Both count and scene-time span must reach explicit limits.

    Confirmation remains a heuristic, not proof of identity correctness. No GT,
    images, features or future observations are input. State is staged in a copy;
    failures commit neither base state nor confirmation evidence. Existing baseline
    internals are used by this subclass, without changing their implementation.
    """

    def __init__(self, run_id, *, max_idle, descriptor_variant, min_similarity,
                 min_support_rounds, min_support_seconds, max_evidence_gap):
        super().__init__(run_id, max_idle=max_idle, descriptor_variant=descriptor_variant,
                         min_similarity=min_similarity)
        require(type(min_support_rounds) is int and min_support_rounds >= 2, "At least two support rounds required")
        require(isinstance(min_support_seconds, Fraction) and min_support_seconds > 0, "Positive exact support span required")
        require(isinstance(max_evidence_gap, Fraction) and max_evidence_gap > 0, "Positive exact evidence gap required")
        self.min_support_rounds = min_support_rounds
        self.min_support_seconds = min_support_seconds
        self.max_evidence_gap = max_evidence_gap
        self._pending = {}

    @staticmethod
    def _gate(ids, group, bindings, visible):
        membership = tuple((gid, tuple(sorted(k for k, b in bindings.items() if b.global_id == gid))) for gid in ids)
        require(all(keys for _, keys in membership), "Candidate references an unretained identity")
        camera_counts = Counter(k.camera_id for _, keys in membership for k in keys)
        conflicts = tuple(sorted(c for c, count in camera_counts.items() if count > 1))
        missing = tuple(sorted(set().union(*(visible[gid] for gid in ids)) - set(group), key=key_order))
        return membership, conflicts, missing

    def update(self, frame):
        # A transaction includes base assignment, expiry, evidence and accepted merges.
        working = deepcopy(self)
        base = GlobalIdentityManager.update(working, frame)
        by_key = {a.key: a.global_id for a in base.assignments}
        visible = defaultdict(set)
        for assignment in base.assignments:
            visible[assignment.global_id].add(assignment.key)
        groups = tuple(sorted((tuple(sorted(g, key=key_order)) for g in frame.groups),
                              key=lambda g: tuple(key_order(k) for k in g)))
        previous, pending, decisions, ready, resets = working._pending, {}, [], [], []
        for group in groups:
            ids = tuple(sorted({by_key[k] for k in group}))
            if len(ids) < 2:
                continue
            membership, conflicts, missing = working._gate(ids, group, working._bindings, visible)
            if conflicts:
                decisions.append(MergeDecision(ids, group, "blocked_retained_camera_conflict", conflicting_cameras=conflicts,
                                               missing_visible_members=missing))
                continue
            if missing:
                decisions.append(MergeDecision(ids, group, "blocked_incomplete_visible_support", missing_visible_members=missing))
                continue
            old = previous.get(ids)
            if old is not None and old.retained_membership != membership:
                resets.append(CandidateReset(ids, "retained_membership_changed"))
                old = None
            elif old is not None and frame.timestamp - old.last_time > self.max_evidence_gap:
                resets.append(CandidateReset(ids, "evidence_gap_exceeded"))
                old = None
            candidate = PendingMerge(ids, membership, old.first_time if old else frame.timestamp,
                                     frame.timestamp, old.support_rounds + 1 if old else 1)
            span = frame.timestamp - candidate.first_time
            confirmed = candidate.support_rounds >= self.min_support_rounds and span >= self.min_support_seconds
            decisions.append(MergeDecision(ids, group, "confirmed" if confirmed else "pending", candidate.support_rounds, span))
            if confirmed:
                ready.append((candidate, group))
            else:
                pending[ids] = candidate
        # Full visible coverage means eligible candidates cannot share an identity:
        # current groups partition observations, each candidate ID has a visible member.
        eligible_ids = [gid for candidate in pending.values() for gid in candidate.global_ids]
        eligible_ids.extend(gid for candidate, _ in ready for gid in candidate.global_ids)
        require(len(eligible_ids) == len(set(eligible_ids)), "Competing eligible merge candidates")

        remap, events = {}, []
        for candidate, group in ready:
            ids = candidate.global_ids
            membership, conflicts, missing = working._gate(ids, group, working._bindings, visible)
            require(not conflicts and not missing and membership == candidate.retained_membership,
                    "Candidate became invalid before merge commit")
            canonical = min(ids)
            absorbed = tuple(g for g in ids if g != canonical)
            for local, binding in tuple(working._bindings.items()):
                if binding.global_id in absorbed:
                    working._bindings[local] = replace(binding, global_id=canonical)
            remap.update((gid, canonical) for gid in absorbed)
            events.append(MergeEvent(canonical, absorbed, frame.frame_index, frame.timestamp,
                                    candidate.support_rounds, candidate.first_time, group, membership))
        accepted = {candidate.global_ids for candidate, _ in ready}
        already_reset = {r.global_ids for r in resets}
        retained_ids = {b.global_id for b in working._bindings.values()}
        for ids in sorted(previous):
            if ids in already_reset:
                continue
            if ids in accepted:
                resets.append(CandidateReset(ids, "merge_accepted"))
            elif ids not in pending:
                reason = "identity_not_retained" if any(gid not in retained_ids for gid in ids) else "no_eligible_support_this_round"
                resets.append(CandidateReset(ids, reason))
        assignments = tuple(replace(a, global_id=remap[a.global_id], reason="confirmed_merge")
                            if a.global_id in remap else a for a in base.assignments)
        slots = [(a.global_id, a.key.camera_id) for a in assignments]
        require(len(slots) == len(set(slots)), "Merge created duplicate current camera membership")
        retained_slots = [(b.global_id, k.camera_id) for k, b in working._bindings.items()]
        require(len(retained_slots) == len(set(retained_slots)), "Merge created duplicate retained camera membership")
        require({a.key for a in assignments} == set(by_key), "Merge changed observation coverage")
        current, retained = defaultdict(list), defaultdict(list)
        for a in assignments:
            current[a.global_id].append(a.key)
        for k, b in working._bindings.items():
            retained[b.global_id].append(k)
        identities = tuple(IdentityState(gid, "visible" if current[gid] else "lost",
            max(working._bindings[k].last_seen for k in retained[gid]), tuple(current[gid]), tuple(sorted(retained[gid])))
            for gid in sorted(retained))
        result = ControlledGlobalFrame(base.run_id, base.frame_index, base.timestamp, base.descriptor_variant,
            base.min_similarity, POLICY, assignments, base.decisions, identities, base.expired_local_tracks,
            base.expired_global_ids, base.assignments, tuple(decisions), tuple(events),
            tuple(pending[k] for k in sorted(pending)), tuple(sorted(resets, key=lambda r: (r.global_ids, r.reason))))
        working._pending = pending
        for name in ("_bindings", "_next_id", "_last_frame", "_last_time", "_pending"):
            setattr(self, name, getattr(working, name))
        return result
