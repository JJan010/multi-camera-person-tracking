"""Check the controlled trace boundary and causal paired scoring example."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import numpy as np

from check_global_identity import frame, manager as baseline, rejects, A, B
from check_controlled_merge import manager
from mtmc.association import global_identity
from mtmc.reid.osnet import ObservationKey
from run_controlled_merge_experiment import to_plain, read_controlled_record, merge_label_category, replay_io, evaluation


def main():
    before, after = evaluation.IdentityCounts(), evaluation.IdentityCounts()
    old, new = baseline(), manager()
    snapshots = []
    for number in range(4):
        f = frame(number, [(A,), (B,)] if number == 0 else [(A, B)])
        a, b = old.update(f), new.update(f)
        labels = {item.key: {'embedding_row': 2 * number + i, 'diagnostic_gt_id': 7}
                  for i, item in enumerate(b.assignments)}
        raw = to_plain(replay_io.encode_output(b, labels, 'example-scope'))
        parsed = read_controlled_record(raw, source_run=f.run_id, scope='example-scope', frame=number,
            variant='mean', threshold=0.5, expected_rows={k: r['embedding_row'] for k, r in labels.items()})
        assert parsed == {item.key: item.global_id for item in b.assignments}
        for previous, current in zip(a.assignments, b.assignments):
            before.update([7], [previous.global_id], np.array([[True]]))
            after.update([7], [current.global_id], np.array([[True]]))
        snapshots.append(raw)
    assert before.result()[0]['idf1'] == 0.5
    assert after.result()[0]['idf1'] == 0.625
    assert [r['assignments'][1]['global_id'] for r in snapshots] == [2, 2, 2, 1]
    assert snapshots[-1]['merge_events'][0]['timestamp'] == '1/10'
    assert snapshots[-1]['merge_events'][0]['first_support_time'] == '1/30'
    print('Causal example: baseline IDF1=50%; controlled=62.5%; early errors are not rewritten: OK')
    print('New-policy JSON preserves exact merge times, prior assignments and observation rows: OK')
    bad = deepcopy(raw)
    bad['policy'] = global_identity.POLICY
    rejects(lambda: read_controlled_record(bad, source_run=f.run_id, scope='example-scope', frame=3,
        variant='mean', threshold=0.5, expected_rows={k: r['embedding_row'] for k, r in labels.items()}))
    bad = deepcopy(raw)
    bad['assignments'][0]['embedding_row'] = 100
    rejects(lambda: read_controlled_record(bad, source_run=f.run_id, scope='example-scope', frame=3,
        variant='mean', threshold=0.5, expected_rows={k: r['embedding_row'] for k, r in labels.items()}))
    print('Wrong policy and source-row mappings are rejected: OK')
    x, y = ObservationKey(*A, 3), ObservationKey(*B, 3)
    members = [asdict(x), asdict(y)]
    assert merge_label_category(members, {x: 7, y: 7}) == 'all_visible_members_same_gt'
    assert merge_label_category(members, {x: 7, y: 8}) == 'different_known_gt'
    assert merge_label_category(members, {x: 7}) == 'unresolved'
    print('Post-replay merge labels distinguish same, different and unresolved evidence: OK')
    print('Controlled merge experiment checks: PASSED')


if __name__ == '__main__':
    main()
