"""Synthetic contract checks; thresholds and durations are not calibrated settings."""

from dataclasses import replace
from fractions import Fraction

from check_global_identity import frame, manager as baseline, mapping, rejects, A, B, C, D, E
from mtmc.association.controlled_merge import ControlledMergeIdentityManager


def manager(**overrides):
    args = dict(max_idle=Fraction(1), descriptor_variant='mean', min_similarity=0.5,
                min_support_rounds=3, min_support_seconds=Fraction(2, 30), max_evidence_gap=Fraction(2, 30))
    args.update(overrides)
    return ControlledMergeIdentityManager('synthetic-global-check', **args)


def main():
    registry, control = manager(), baseline()
    outputs = []
    for n, groups in enumerate(([(A,), (B,)], [(A, B)], [(A, B)], [(A, B)])):
        current = frame(n, groups)
        result = registry.update(current)
        reference = control.update(current)
        assert result.base_assignments == reference.assignments
        if n < 3:
            assert result.assignments == reference.assignments and not result.merge_events
        outputs.append(result)
    assert mapping(outputs[3]) == {A: 1, B: 1}
    assert outputs[3].merge_events[0].absorbed_global_ids == (2,)
    assert outputs[3].merge_events[0].support_rounds == 3
    assert mapping(outputs[0]) == mapping(outputs[2]) == {A: 1, B: 2}
    assert mapping(registry.update(frame(4, [(A,), (B,)]))) == {A: 1, B: 1}
    print('Confirmed merge applies now and later; baseline and past outputs remain unchanged: OK')

    registry = manager()
    registry.update(frame(0, [(A, B), (C,)]))
    for n in range(1, 5):
        result = registry.update(frame(n, [(A, C), (B,)]))
        assert not result.merge_events and not result.pending_candidates
        assert result.merge_decisions[0].outcome == 'blocked_incomplete_visible_support'
        assert result.merge_decisions[0].missing_visible_members[0].local_id == B[1]
    for n in range(5, 8):
        result = registry.update(frame(n, [(A, B, C)]))
    assert len(result.merge_events) == 1
    print('Repeated pair support cannot bypass another visible identity member: OK')

    registry = manager()
    registry.update(frame(0, [(A, B), (E,)]))
    for n in range(1, 5):
        result = registry.update(frame(n, [(A, E)]))
        assert result.merge_decisions[0].outcome == 'blocked_retained_camera_conflict'
        assert result.merge_decisions[0].conflicting_cameras == (5,)
        assert not result.merge_events
    print('Absent but retained same-camera bindings block merging: OK')

    for missing_groups in ([(A,), (B,)], []):
        registry = manager()
        registry.update(frame(0, [(A,), (B,)]))
        registry.update(frame(1, [(A, B)]))
        result = registry.update(frame(2, missing_groups))
        assert not result.pending_candidates and result.candidate_resets
        result = registry.update(frame(3, [(A, B)]))
        assert result.pending_candidates[0].support_rounds == 1
    print('Unsupported and empty rounds clear confirmation evidence: OK')

    registry = manager()
    registry.update(frame(0, [(A,), (B,)]))
    registry.update(frame(1, [(A, B)]))
    result = registry.update(frame(10, [(A, B)]))
    assert result.pending_candidates[0].support_rounds == 1
    assert result.candidate_resets[0].reason == 'evidence_gap_exceeded'
    print('Excess scene-time gap restarts evidence even without intermediate calls: OK')

    registry = manager(max_evidence_gap=Fraction(2))
    registry.update(frame(0, [(A, B), (C,)]))
    registry.update(frame(1, [(A, B, C)]))
    registry.update(frame(2, [(A, C)], timestamp=Fraction(1, 2)))
    result = registry.update(frame(3, [(A, C)], timestamp=Fraction(26, 25)))
    assert result.pending_candidates[0].support_rounds == 1
    assert result.candidate_resets[0].reason == 'retained_membership_changed'
    assert not result.merge_events
    print('Changed retained membership resets the candidate fingerprint: OK')

    registry = manager()
    registry.update(frame(0, [(A,), (B,)]))
    for n in range(1, 4):
        result = registry.update(frame(n, [(A, B)], timestamp=Fraction(n, 1000)))
    assert result.pending_candidates[0].support_rounds == 3 and not result.merge_events
    result = registry.update(frame(4, [(A, B)], timestamp=Fraction(1, 1000) + Fraction(2, 30)))
    assert result.merge_events
    registry = manager(max_evidence_gap=Fraction(1))
    registry.update(frame(0, [(A,), (B,)]))
    registry.update(frame(1, [(A, B)]))
    result = registry.update(frame(2, [(A, B)], timestamp=Fraction(1, 2)))
    assert not result.merge_events and result.pending_candidates[0].support_rounds == 2
    print('Both support-round count and exact elapsed scene time are required: OK')

    first, second = manager(), manager()
    for n in range(4):
        original = frame(n, [(A,), (B,), (C,)] if n == 0 else [(A, B, C)])
        reordered = replace(original, groups=tuple(tuple(reversed(g)) for g in reversed(original.groups)))
        a, b = first.update(original), second.update(reordered)
        assert a == b
    assert len(a.merge_events) == 1 and a.merge_events[0].absorbed_global_ids == (2, 3)
    assert a.merge_events[0].support_rounds == 3
    assert mapping(first.update(frame(4, [(A, B, C), (D,)])))[D] == 4
    print('Three-view evidence counts once per round; canonical ID and ordering are deterministic: OK')
    print('Absorbed global IDs are not reused for new tracks: OK')

    registry = manager()
    registry.update(frame(0, [(A, C), (B,)]))
    for n in range(1, 4):
        result = registry.update(frame(n, [(A, B)]))
    assert result.merge_events
    result = registry.update(frame(4, [(A, B)], timestamp=Fraction(31, 30)))
    assert [(k.camera_id, k.local_id) for k in result.expired_local_tracks] == [C]
    assert not result.expired_global_ids
    result = registry.update(frame(5, [], timestamp=Fraction(62, 30)))
    assert result.expired_global_ids == (1,)
    print('Merging preserves individual last-seen times; only the canonical identity later expires: OK')

    tested, control = manager(), manager()
    for n in (0, 1):
        f = frame(n, [(A,), (B,)] if n == 0 else [(A, B)])
        assert tested.update(f) == control.update(f)
    rejects(lambda: tested.update(frame(1, [(A, B)])))
    rejects(lambda: tested.update(replace(frame(2, [(A, B)]), run_id='different')))
    assert tested.update(frame(2, [(A, B)])) == control.update(frame(2, [(A, B)]))
    assert tested.update(frame(3, [(A, B)])) == control.update(frame(3, [(A, B)]))
    print('Invalid input commits neither identity state nor evidence: OK')
    print('Synthetic timing and thresholds are test fixtures, not deployment settings.')
    print('Controlled merge smoke test: PASSED')


if __name__ == '__main__':
    main()
