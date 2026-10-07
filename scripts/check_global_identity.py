"""Synthetic checks for causal global-ID continuity; no calibrated thresholds."""

from dataclasses import replace
from fractions import Fraction
from itertools import combinations

from mtmc.association.global_identity import GlobalIdentityManager
from mtmc.association.grouping import group_pair_associations
from mtmc.association.pairwise import PairAssociation, PairMatch, UnmatchedObservation
from mtmc.reid.osnet import ObservationKey

CAMERAS = (4, 5, 8)
A, B, C, D, E = (4, 10), (5, 20), (8, 30), (8, 31), (5, 21)


def frame(number, groups, *, timestamp=None, run_id="synthetic-global-check"):
    """Build actual pairwise inputs and use the production grouping API."""
    timestamp = Fraction(number, 30) if timestamp is None else timestamp
    nodes = {ObservationKey(*key, number) for group in groups for key in group}
    edges = {frozenset((ObservationKey(*a, number), ObservationKey(*b, number)))
             for group in groups for a, b in combinations(group, 2)}
    pairs = []
    for a, b in combinations(CAMERAS, 2):
        left = sorted((k for k in nodes if k.camera_id == a), key=lambda k: k.local_id)
        right = sorted((k for k in nodes if k.camera_id == b), key=lambda k: k.local_id)
        matches = tuple(PairMatch(x, y, 0.9) for x in left for y in right if frozenset((x, y)) in edges)
        matched = {k for m in matches for k in (m.left, m.right)}
        def unmatched(keys, opposite):
            return tuple(UnmatchedObservation(k, "no_candidate_above_threshold" if opposite else "empty_opposite_camera",
                                              0.1 if opposite else None) for k in keys if k not in matched)
        pairs.append(PairAssociation(run_id, number, timestamp, "mean", a, b, 0.5, matches,
                                     unmatched(left, right), unmatched(right, left)))
    return group_pair_associations(pairs)


def manager(run_id="synthetic-global-check"):
    return GlobalIdentityManager(run_id, max_idle=Fraction(1), descriptor_variant="mean", min_similarity=0.5)


def mapping(result):
    return {(a.key.camera_id, a.key.local_id): a.global_id for a in result.assignments}


def rejects(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Invalid input accepted")


def main():
    registry = manager()
    start = registry.update(frame(0, [(A, B)]))
    assert mapping(start) == {A: 1, B: 1}
    split = registry.update(frame(1, [(A,), (B,)]))
    assert mapping(split) == {A: 1, B: 1}
    added = registry.update(frame(2, [(A, B, C)]))
    assert mapping(added) == {A: 1, B: 1, C: 1}
    assert added.assignments[-1].reason == "group_attachment"
    assert mapping(start) == {A: 1, B: 1}  # frozen earlier output is unchanged
    print("Local continuity preserves global ID when frame groups split: OK")
    print("New camera track joins one existing identity through a current group: OK")

    registry = manager()
    registry.update(frame(0, [(A,), (B,)]))
    conflicted = registry.update(frame(1, [(A, B, C)]))
    assert mapping(conflicted) == {A: 1, B: 2, C: 3}
    assert conflicted.decisions[0].outcome == "rejected_multiple_existing_ids"
    still_split = registry.update(frame(2, [(A, B, C)]))
    assert mapping(still_split) == {A: 1, B: 2, C: 3}
    print("Existing identities are not silently merged; fragmentation limitation demonstrated: OK")

    registry = manager()
    registry.update(frame(0, [(A, B)]))
    lost = registry.update(frame(1, [], timestamp=Fraction(1, 2)))
    assert not lost.assignments and len(lost.identities) == 1 and lost.identities[0].status == "lost"
    boundary = registry.update(frame(2, [(A,)], timestamp=Fraction(1)))
    assert mapping(boundary) == {A: 1} and not boundary.expired_local_tracks
    expired = registry.update(frame(3, [], timestamp=Fraction(61, 30)))
    assert expired.expired_global_ids == (1,) and not expired.identities
    returned = registry.update(frame(4, [(A,)], timestamp=Fraction(62, 30)))
    assert mapping(returned) == {A: 2}
    print("Empty rounds, inclusive idle boundary, expiry and non-reused global IDs: OK")

    registry = manager()
    registry.update(frame(0, [(A, B)]))
    reserved = registry.update(frame(1, [(A, E)]))
    assert mapping(reserved) == {A: 1, E: 2}
    assert reserved.decisions[0].outcome == "rejected_reserved_camera"
    assert reserved.decisions[0].conflicting_cameras == (5,)
    print("An absent retained track reserves its identity/camera slot until expiry: OK")

    registry = manager()
    registry.update(frame(0, [(A, B)]))
    registry.update(frame(1, [(A,)], timestamp=Fraction(1, 2)))
    renewal = registry.update(frame(2, [(A, E)], timestamp=Fraction(11, 10)))
    assert mapping(renewal) == {A: 1, E: 1}
    assert [(k.camera_id, k.local_id) for k in renewal.expired_local_tracks] == [B]
    assert renewal.expired_global_ids == ()
    print("Bindings expire individually; a visible identity can accept a replacement after expiry: OK")

    registry = manager()
    registry.update(frame(0, [(A, B)]))
    competition = registry.update(frame(1, [(A, C), (B, D)]))
    assert mapping(competition) == {A: 1, B: 1, C: 2, D: 3}
    assert all(d.outcome == "rejected_competing_attachments" for d in competition.decisions)
    assert len({(a.global_id, a.key.camera_id) for a in competition.assignments}) == len(competition.assignments)
    print("Competing same-camera attachments are both rejected; no first-come winner: OK")

    one, two = manager(), manager()
    for number, groups in enumerate(([(A, B)], [(A, C), (B, D)], [(A,), (B,), (C,), (D,)])):
        original = frame(number, groups)
        reordered = replace(original, groups=tuple(tuple(reversed(g)) for g in reversed(original.groups)))
        assert one.update(original) == two.update(reordered)
    print("Group/member order produces identical assignments and decisions: OK")

    tested, control = manager(), manager()
    tested.update(frame(0, [(A,)]))
    control.update(frame(0, [(A,)]))
    good = frame(1, [(A, B)])
    invalid = [replace(good, run_id="other"), replace(good, descriptor_variant="latest"),
               replace(good, min_similarity=0.6), replace(good, timestamp=Fraction(0)),
               replace(good, frame_index=0), replace(good, policy="other"),
               replace(good, groups=good.groups + good.groups),
               replace(good, groups=((ObservationKey(4, 10, 1), ObservationKey(4, 11, 1)),)),
               replace(good, groups=((ObservationKey(4, 10, 99),),))]
    for bad in invalid:
        rejects(lambda: tested.update(bad))
    assert tested.update(good) == control.update(good)
    rejects(lambda: tested.update(good))
    print("Mixed scopes/settings, invalid partitions and nonincreasing time rejected without state mutation: OK")

    other = manager("another-run").update(frame(0, [(A,)], run_id="another-run"))
    assert mapping(other) == {A: 1} and other.run_id != start.run_id
    print("Global IDs are scoped by run/session, not universal person identifiers: OK")
    print("Synthetic threshold and idle time are test settings, not deployment calibration.")
    print("Global identity smoke test: PASSED")


if __name__ == "__main__":
    main()
