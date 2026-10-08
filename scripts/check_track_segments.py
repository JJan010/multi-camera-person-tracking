"""Synthetic contract checks, not a switch detector or quality experiment."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np

from mtmc.association.global_identity import LocalTrackKey
from mtmc.pipeline.core import IdentityStage
from mtmc.reid.crops import CropRecord
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.tracking.segments import (
    LocalTrackSegments, SegmentRound, SegmentBreak, ORIGINAL_ID_LIMIT,
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def vector(i):
    value = np.zeros(512, np.float32); value[i] = 1
    return value


def batch(frame, rows=None, *, breaks=(), run='fixture'):
    # Row: camera, ORIGINAL local ID, appearance identity (None = outside).
    rows = [(4, 0, 0), (5, 0, 0), (8, 0, 0), (4, 1, 1), (8, 1, 1)] if rows is None else rows
    records = []; keys = []; vectors = []; time = Fraction(frame, 30)
    for c, local, person in rows:
        key = ObservationKey(c, local, frame)
        if person is None:
            records.append(CropRecord(key, .8, (110., 0., 120., 10.), None, 0.))
        else:
            offset = 20 * person
            box = (float(offset), 0., float(offset + 10), 10.)
            records.append(CropRecord(key, .8, box, tuple(map(int, box)), 1.))
            keys.append(key); vectors.append(vector(person))
    matrix = np.stack(vectors) if vectors else np.empty((0, 512), np.float32)
    return SegmentRound(run, frame, time, tuple(records), ReIDBatch(tuple(keys), (time,) * len(keys), matrix), breaks)


def state(segmenter):
    return deepcopy((segmenter._state, segmenter._next_id, segmenter._last_frame, segmenter._last_time))


def reject(segmenter, value):
    before = state(segmenter)
    try:
        segmenter.update(value)
    except (ValueError, TypeError):
        require(state(segmenter) == before, 'Rejected batch mutated state')
    else:
        raise AssertionError('Malformed batch accepted')


def stage():
    return IdentityStage('fixture', {c: np.eye(3) for c in (4, 5, 8)}, 'synthetic_plane',
        variant='mean', threshold=.7, max_distance=2., unavailable_policy='reject',
        identity_configuration={'max_idle_seconds': '1', 'min_support_rounds': 3,
                                'min_support_seconds': '1/5', 'max_evidence_gap': '1/10'})


def main():
    # No-action controls exercise the actual existing history and global stage.
    histories = [AppearanceHistory('fixture') for _ in range(3)]
    stages = [stage() for _ in range(3)]
    disabled = LocalTrackSegments('fixture', (4, 5, 8), enabled=False)
    enabled = LocalTrackSegments('fixture', (4, 5, 8), enabled=True)
    for frame in range(120):
        source = batch(frame, [] if frame in (40, 41, 42) else None)
        transformed = [disabled.update(source), enabled.update(source)]
        for current in transformed:
            require(current.records == source.records and current.features.keys == source.features.keys
                    and current.features.timestamps == source.features.timestamps
                    and np.array_equal(current.features.embeddings, source.features.embeddings)
                    and not current.events, 'No-action input parity differs')
        results = []
        for h, s, current in zip(histories, stages, [source, *transformed]):
            history = h.update(frame, source.timestamp, current.features)
            identity, *_ = s.update(frame, source.timestamp, history.mean, current.records)
            results.append(asdict(identity))
        require(results[0] == results[1] == results[2], 'No-action global records differ')
    print('Disabled and enabled-without-breaks: exact original history/global output over 120 rounds: OK')

    segmenter = LocalTrackSegments('fixture', (4, 5, 8), enabled=True)
    h = AppearanceHistory('fixture'); identity_stage = stage(); earlier = []
    for frame in range(5):
        source = batch(frame); current = segmenter.update(source)
        history = h.update(frame, source.timestamp, current.features)
        identity, *_ = identity_stage.update(frame, source.timestamp, history.mean, current.records)
        earlier.append(identity)
    saved = deepcopy(earlier)
    # External synthetic break: no quality decision is made inside segmenter.
    split_key = ObservationKey(5, 0, 5)
    source = batch(5, [(4,0,0),(5,0,1),(8,0,0),(4,1,1),(8,1,1)],
                   breaks=(SegmentBreak(split_key, 0, 'synthetic_confirmed_takeover'),))
    current = segmenter.update(source)
    binding = next(b for b in current.bindings if b.source_key == split_key)
    require(binding.generation == 1 and binding.identity_key.local_id == ORIGINAL_ID_LIMIT,
            'Segment namespace differs')
    require(current.source_key(binding.identity_key) == split_key, 'Source provenance lost')
    require(np.array_equal(current.features.embeddings, source.features.embeddings), 'Feature values changed')
    require(all(replace(r, key=current.source_key(r.key)) == original
                for r, original in zip(current.records, source.records)), 'Box/score provenance differs')
    history = h.update(5, source.timestamp, current.features)
    position = history.mean.keys.index(binding.identity_key)
    require(history.source_frames[position] == (5,)
            and np.array_equal(history.mean.embeddings[position], vector(1)), 'New history contains old person')
    identity, *_ = identity_stage.update(5, source.timestamp, history.mean, current.records)
    by_source = {current.source_key(a.key): a.global_id for a in identity.assignments}
    require(by_source[split_key] == by_source[ObservationKey(4,1,5)]
            != by_source[ObservationKey(4,0,5)], 'New segment cannot join existing other identity')
    require(LocalTrackKey(5,0) in identity_stage.manager._bindings, 'Old retained binding was silently deleted')
    require(earlier == saved, 'Past outputs changed')
    later = segmenter.update(batch(6, [(5,0,1)]))
    require(later.bindings[0].identity_key.local_id == binding.identity_key.local_id
            and later.bindings[0].generation == 1, 'Cut did not persist')
    print('Explicit cut starts fresh appearance history; current group can attach it to another global ID: OK')
    print('Original tracker ID/boxes/scores/features and past outputs preserved; old binding retains normal expiry: OK')

    # Shuffled records, independent feature order, reversed break order.
    a = LocalTrackSegments('fixture', (8,4,5), enabled=True)
    b = LocalTrackSegments('fixture', (4,5,8), enabled=True)
    for manager in (a,b):
        manager.update(batch(0))
    requests = tuple(SegmentBreak(ObservationKey(c,0,1),0,'fixture') for c in (8,4))
    original = batch(1, breaks=requests)
    shuffled = replace(original, records=tuple(reversed(original.records)), breaks=tuple(reversed(requests)),
        features=ReIDBatch(original.features.keys[::-1], original.features.timestamps[::-1],
                          original.features.embeddings[::-1]))
    x,y = a.update(original),b.update(shuffled)
    require(x.events == y.events and state(a) == state(b), 'Input order changed allocation')
    require({r.source_key:r for r in x.bindings} == {r.source_key:r for r in y.bindings}, 'Binding map differs')
    require(all(np.array_equal(v, dict(zip(y.features.keys,y.features.embeddings))[k])
                for k,v in zip(x.features.keys,x.features.embeddings)), 'Feature mapping changed')
    original.features.embeddings[:] = 0
    require(np.allclose(np.linalg.norm(x.features.embeddings,axis=1),1), 'Output aliases input features')
    print('Camera/record/feature/request order preserves keyed results; returned feature arrays are owned: OK')

    z = LocalTrackSegments('fixture', (4,5,8), enabled=True)
    z.update(batch(0, [(4,0,None)]))
    outside = z.update(batch(1, [(4,0,None)], breaks=(SegmentBreak(ObservationKey(4,0,1),0,'fixture'),)))
    require(len(outside.records) == 1 and outside.features.embeddings.shape == (0,512), 'Outside track fabricated features')
    old_id = outside.bindings[0].identity_key.local_id
    z.update(batch(2, []));z.update(batch(100, []))
    returned = z.update(batch(101, [(4,0,0),(5,ORIGINAL_ID_LIMIT-1,1)]))
    require(returned.bindings[0].identity_key.local_id == old_id and not returned.events,
            'Absence implicitly cut/reverted segment')
    twice = z.update(batch(102, [(4,0,1)], breaks=(SegmentBreak(ObservationKey(4,0,102),1,'fixture'),)))
    require(twice.bindings[0].generation == 2 and twice.bindings[0].identity_key.local_id > old_id,
            'Repeated cut reused identity key')
    print('ID zero, outside crops, empty rounds, gaps and repeated cuts retain provenance without ID reuse: OK')

    c = LocalTrackSegments('fixture', (4,5,8), enabled=True);c.update(batch(0))
    good = batch(1, breaks=(SegmentBreak(ObservationKey(5,0,1),0,'fixture'),))
    invalid = [replace(good, run_id='other'), replace(good, timestamp=Fraction(0)),
               replace(good, records=good.records + good.records[:1]),
               replace(good, breaks=good.breaks + good.breaks),
               replace(good, breaks=(replace(good.breaks[0],expected_generation=1),)),
               replace(good, breaks=(replace(good.breaks[0],reason=''),)),
               batch(1, [(4, ORIGINAL_ID_LIMIT,0)]),
               batch(1, [(4,2,0)], breaks=(SegmentBreak(ObservationKey(4,2,1),0,'new'),)),
               replace(good, records=(*good.records[:-1],replace(good.records[-1],confidence=float('nan')))),
               replace(good, features=replace(good.features,embeddings=good.features.embeddings.astype(np.float64))),
               replace(good, features=replace(good.features,keys=good.features.keys[::-1][:-1]))]
    for value in invalid:
        reject(c,value)
    c.update(good)
    reject(c,batch(2,breaks=(SegmentBreak(ObservationKey(5,0,2),0,'stale'),)))
    c.update(batch(2))
    d = LocalTrackSegments('fixture',(4,5,8),enabled=False);d.update(batch(0));reject(d,good)
    fresh = LocalTrackSegments('new_run',(4,5,8),enabled=True).update(batch(0,run='new_run'))
    require(all(b.generation == 0 for b in fresh.bindings), 'Run state leaked')
    print('Malformed inputs and stale cuts reject before mutation; corrected rounds and run isolation: OK')
    print('Track segmentation contract: PASSED; automatic switch detection and quality evaluation NOT YET implemented')
    print('All break events and appearance values above are synthetic fixtures, not calibrated decision rules.')


if __name__ == '__main__':
    main()
