"""Known-answer checks for fixed-box ceilings and cautious identity evidence."""

from dataclasses import asdict
import numpy as np
from mtmc.reid.osnet import ObservationKey
from diagnose_global_identity import inspect_slot, decompose, classify_conflict
from evaluate_global_identity import IdentityCounts


def main():
    key = ObservationKey(4, 1, 2)
    counts = IdentityCounts()
    counts.update([7], [1], np.array([[True]]))
    counts.update([7], [2], np.array([[True]]))
    metrics, _ = counts.result()
    parts = decompose(metrics, 2)
    assert parts['shared_identity_assignment_gap'] == 1
    assert parts['minimum_fn_with_fixed_boxes'] == parts['minimum_fp_with_fixed_boxes'] == 0
    assert parts['framewise_spatial_f1_ceiling'] == 1 and metrics['idf1'] == 0.5
    print('Perfect spatial boxes can still have a shared identity gap: OK')

    mask = np.array([[True, True], [True, False]])
    keys = [key, ObservationKey(4, 2, 2)]
    ceiling, evidence, stats = inspect_slot([7, 8], keys, [1, 2], mask)
    assert ceiling == 2 and not evidence
    assert stats['gt_with_multiple_candidates'] == stats['predictions_with_multiple_candidates'] == 1
    print('Spatial ceiling uses joint maximum cardinality; ambiguous overlaps are excluded from evidence: OK')

    ceiling, evidence, stats = inspect_slot([7, 8], keys, [1, 2], np.array([[True, False], [False, False]]))
    assert ceiling == 1 and evidence == [(key, 7, 1)]
    assert stats['gt_without_spatial_candidate'] == stats['predictions_without_spatial_candidate'] == 1
    counts = IdentityCounts()
    counts.update([7, 8], [1, 2], np.array([[True, False], [False, False]]))
    metrics, _ = counts.result()
    parts = decompose(metrics, ceiling)
    assert parts['shared_identity_assignment_gap'] == 0
    assert parts['minimum_fn_with_fixed_boxes'] == parts['minimum_fp_with_fixed_boxes'] == 1
    print('Spatial failures remain in the fixed-box minimum instead of being called identity errors: OK')

    for ng, npred in ((0, 0), (1, 0), (0, 1)):
        ceiling, evidence, _ = inspect_slot(list(range(ng)), keys[:npred], list(range(npred)), np.zeros((ng, npred), bool))
        assert ceiling == 0 and evidence == []
    print('Empty GT, empty predictions and empty slots: OK')

    other = ObservationKey(5, 3, 2)
    third = ObservationKey(8, 4, 2)
    members = [asdict(key), asdict(other)]
    assert classify_conflict(members, {key: 7, other: 7}) == 'all_members_same_gt'
    assert classify_conflict(members, {key: 7, other: 8}) == 'different_known_gt'
    assert classify_conflict(members, {key: 7}) == 'unresolved'
    assert classify_conflict(members + [asdict(third)], {key: 7, other: 8}) == 'different_known_gt'
    print('Same-person, different-person and unresolved conflict evidence stay distinct: OK')
    print('Global identity diagnostic checks: PASSED')


if __name__ == '__main__':
    main()
