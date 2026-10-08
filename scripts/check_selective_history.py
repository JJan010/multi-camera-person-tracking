"""Selective global-memory contract tests; no models, videos, GT or tracker updates."""
from fractions import Fraction
import numpy as np

from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.selective_history import SelectiveAppearanceHistory, confidence_update_mask


def unit(i):
    vector = np.zeros(512, np.float32)
    vector[i] = 1
    return vector


def batch(frame, entries, time=None):
    time = Fraction(frame, 30) if time is None else time
    return ReIDBatch(tuple(ObservationKey(c, i, frame) for c, i, _ in entries), (time,)*len(entries),
                     np.asarray([v for _, _, v in entries], np.float32).reshape(-1,512))


def update(memory, frame, entries, accept, time=None):
    observations = batch(frame, entries, time)
    timestamp = Fraction(frame,30) if time is None else time
    return memory.update(frame, timestamp, observations,
                         accept_update={k: value for k, value in zip(observations.keys, accept)})


def rejected(call):
    try:
        call()
    except (TypeError, ValueError):
        return
    raise AssertionError('Malformed input accepted')


def main():
    rng = np.random.default_rng(17)
    original, selective = AppearanceHistory('parity'), SelectiveAppearanceHistory('parity')
    for frame in range(120):
        entries = []
        if frame % 17:
            for camera, identity in ((0,0), (4,7), (8,7)):
                if (frame+camera) % 11 == 0:
                    continue
                vector = rng.normal(size=512).astype(np.float32)
                vector /= np.linalg.norm(vector)
                entries.append((camera,identity,vector))
        if frame % 2:
            entries.reverse()
        observations = batch(frame, entries)
        a = original.update(frame, Fraction(frame,30), observations)
        b = selective.update(frame, Fraction(frame,30), observations, accept_update={k: True for k in observations.keys})
        assert a.mean.keys == b.mean.keys and a.mean.timestamps == b.mean.timestamps
        np.testing.assert_array_equal(a.mean.embeddings, b.mean.embeddings)
        assert a.source_frames == tuple(d.source_frames for d in b.decisions)
        assert a.source_times == tuple(d.source_times for d in b.decisions)
        assert original.active_tracks == selective.active_tracks and original.stored_vectors == selective.stored_vectors
    print('All updates accepted: exact original-history parity over 120 synthetic rounds: OK')

    x,y,z = unit(0),unit(1),unit(2)
    memory = SelectiveAppearanceHistory('selective')
    first = batch(0, [(4,7,x)])
    result = memory.update(0,Fraction(0),first,accept_update={first.keys[0]:True})
    first.embeddings[:] = 0
    result.mean.embeddings[:] = 0
    r = update(memory,1,[(8,7,y),(4,7,z)],[False,False])
    assert [d.status for d in r.decisions] == ['unavailable','reused']
    assert r.mean.keys == (ObservationKey(4,7,1),) and r.decisions[1].source_frames == (0,)
    np.testing.assert_array_equal(r.mean.embeddings[0],x)
    assert memory.stored_vectors == 1
    kept = r.mean.embeddings.copy()
    update(memory,2,[(4,7,y)],[True])
    np.testing.assert_array_equal(r.mean.embeddings,kept)
    print('Weak update rejected; same-camera/local history reused; new weak track unavailable; owned arrays: OK')

    expiry = SelectiveAppearanceHistory('expiry')
    update(expiry,0,[(4,7,x)],[True])
    r = update(expiry,30,[(4,7,y)],[False])
    assert r.decisions[0].status == 'reused' and r.decisions[0].source_times == (Fraction(0),)
    r = update(expiry,31,[(4,7,y)],[False])
    assert r.decisions[0].status == 'unavailable' and r.mean.embeddings.shape == (0,512)
    assert expiry.stored_vectors == 0
    partial = SelectiveAppearanceHistory('partial', max_observations=2)
    update(partial,0,[(4,7,x)],[True])
    update(partial,15,[(4,7,y)],[True])
    r = update(partial,31,[(4,7,z)],[False])
    assert r.decisions[0].source_frames == (15,)
    np.testing.assert_array_equal(r.mean.embeddings[0],y)
    update(partial,46,[],[])
    assert partial.stored_vectors == 0
    print('Inclusive scene-age boundary; reuse never refreshes samples; partial expiry recomputes mean; empty rounds evict: OK')

    limited = SelectiveAppearanceHistory('limited',max_observations=2)
    for f,v in enumerate((x,y,z)):
        r = update(limited,f,[(4,7,v)],[True])
    assert r.decisions[0].source_frames == (1,2) and limited.stored_vectors == 2
    cancel = SelectiveAppearanceHistory('cancel')
    update(cancel,0,[(4,7,x)],[True]);update(cancel,1,[(4,7,-x)],[True])
    r = update(cancel,2,[(4,7,z)],[False])
    assert r.decisions[0].used_latest_accepted_fallback and r.decisions[0].source_frames == (0,1)
    np.testing.assert_array_equal(r.mean.embeddings[0],-x)
    print('Observation limit and cancellation fallback use only accepted samples: OK')

    observations = batch(3,[(4,7,x),(8,7,y)])
    mask = confidence_update_mask(observations,{observations.keys[1]:.4999,observations.keys[0]:.5},minimum_score=.5)
    assert mask == {observations.keys[0]:True, observations.keys[1]:False}
    rejected(lambda:confidence_update_mask(observations,{observations.keys[0]:.5},minimum_score=.5))
    rejected(lambda:confidence_update_mask(observations,{observations.keys[0]:.5,observations.keys[1]:float('nan')},minimum_score=.5))
    print('Key-based confidence mapping, inclusive 0.5 boundary and malformed scores: OK')

    guarded = SelectiveAppearanceHistory('guarded')
    update(guarded,0,[(4,7,x)],[True])
    valid = batch(1,[(4,7,y)])
    rejected(lambda:guarded.update(1,Fraction(1,30),valid,accept_update={}))
    rejected(lambda:guarded.update(1,Fraction(1,30),valid,accept_update={valid.keys[0]:1}))
    rejected(lambda:update(guarded,1,[(4,7,np.zeros(512,np.float32))],[False]))
    rejected(lambda:update(guarded,1,[(4,7,y),(4,7,z)],[True,False]))
    rejected(lambda:guarded.update(2,Fraction(1,30),valid,accept_update={valid.keys[0]:True}))
    r = update(guarded,1,[(4,7,y)],[True])
    assert r.decisions[0].source_frames == (0,1)
    rejected(lambda:update(guarded,1,[(4,7,z)],[True]))
    fresh = SelectiveAppearanceHistory('fresh')
    assert update(fresh,0,[(4,7,x)],[False]).decisions[0].status == 'unavailable'
    print('Invalid vectors/keys/masks/time rejected before state mutation; run isolation: OK')

    forward,reverse = SelectiveAppearanceHistory('order'),SelectiveAppearanceHistory('order')
    for f,entries,flags in [(0,[(4,7,x),(8,7,y)],[True,True]),(1,[(4,7,z),(8,7,x)],[False,True])]:
        a = update(forward,f,entries,flags);b = update(reverse,f,list(reversed(entries)),list(reversed(flags)))
        assert {d.key:d for d in a.decisions} == {d.key:d for d in b.decisions}
        lookup = dict(zip(b.mean.keys,b.mean.embeddings))
        for key,vector in zip(a.mean.keys,a.mean.embeddings):
            np.testing.assert_array_equal(vector,lookup[key])
    print('Input order preserves key-to-descriptor mapping; absent tracks are not emitted: OK')
    print('Selective history contract: PASSED; global integration and quality evaluation pending')


if __name__ == '__main__':
    main()
