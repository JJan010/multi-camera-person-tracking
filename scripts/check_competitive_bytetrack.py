"""Known-answer checks for constrained, causal high/low candidate competition."""
from itertools import product
import numpy as np
from mtmc.tracking.competitive import CompetitiveByteTrack, challenge_assignment
from mtmc.tracking._vendor.experimental_bytetrack import ExperimentalByteTrack

SETTINGS = dict(track_activation_threshold=.5, lost_track_buffer=30,
                minimum_matching_threshold=.8, frame_rate=30, minimum_consecutive_frames=1,
                direct_output=True, appearance_threshold=.6)


def new(mode):
    return CompetitiveByteTrack(**SETTINGS, refinement=mode)


def main():
    a = np.zeros(512, np.float32); a[0] = 1
    mixed = np.zeros(512, np.float32); mixed[:2] = [.8, .6]
    box = np.array([[0, 0, 10, 20]], np.float32)
    boxes = np.array([[0, 0, 10, 20], [1, 0, 11, 20]], np.float32)
    scores = np.array([.9, .3], np.float32)
    trackers = {'staged': ExperimentalByteTrack(**SETTINGS), 'iou': new('iou'), 'appearance': new('appearance')}
    for t in trackers.values():
        assert t.update_candidates(box, np.array([.9], np.float32), [a]).tracker_id.tolist() == [1]
        result = t.update_candidates(boxes, scores, [mixed, a])
        assert result.tracker_id.tolist() == [1]
    assert trackers['staged'].selected_indices == trackers['iou'].selected_indices == (0,)
    assert trackers['appearance'].selected_indices == (1,)
    track = next(t for t in trackers['appearance'].tracked_tracks if t.external_track_id == 1)
    assert len(track.appearance_history) == 1 and np.array_equal(track.descriptor(2), a)
    assert trackers['appearance'].refinement_events[0]['prior_sample_frames'] == [0]
    print('Appearance can select a better weak candidate before an admissible high match; IoU control keeps incumbent: OK')
    print('Accepted weak observation keeps ID and candidate provenance without refreshing strong history: OK')

    choices, _ = challenge_assignment([1,1], [[.8,.8],[.8,.8]], [.6,.6], [[1,.9],[.95,.61]], threshold=.6, mode='appearance')
    assert set(choices) == {(0,1),(1,0)}
    print('Competing tracks receive unique candidates by joint total improvement, not independent argmax: OK')
    choices, mask = challenge_assignment([.5], [[.5,.499,.9]], [.6], [[.6,1,np.nan]], threshold=.6, mode='appearance')
    assert mask.tolist() == [[True,False,False]] and choices == ()
    assert challenge_assignment([.5], [[.8]], [np.nan], [[1]], threshold=.6, mode='appearance')[0] == ()
    assert challenge_assignment([], np.empty((0,2)), [], np.empty((0,2)), threshold=.6, mode='appearance')[0] == ()
    assert challenge_assignment([.5], np.empty((1,0)), [.7], np.empty((1,0)), threshold=.6, mode='appearance')[0] == ()
    print('Inclusive motion/cosine gates, strict improvement, missing features and empty inputs: OK')

    rng = np.random.default_rng(711)
    for mode in ('iou','appearance'):
        for _ in range(25):
            hi, lo = rng.random(3), rng.random((3,3))
            hc, lc = rng.uniform(.6,1,3), rng.uniform(.4,1,(3,3))
            chosen, allowed = challenge_assignment(hi,lo,hc,lc,threshold=.6,mode=mode)
            gain = lc-hc[:,None] if mode=='appearance' else lo-hi[:,None]
            best = 0.
            for selection in product(range(-1,3),repeat=3):
                indices = [j for j in selection if j>=0]
                if len(indices)!=len(set(indices)):
                    continue
                if any(not allowed[i,j] or gain[i,j]<=0 for i,j in enumerate(selection) if j>=0):
                    continue
                best = max(best, sum(gain[i,j] for i,j in enumerate(selection) if j>=0))
            assert abs(sum(gain[i,j] for i,j in chosen)-best)<1e-12
    print('Refinement objective agrees with exhaustive partial assignments: OK')

    # Unmatched track 2 has a feasible low-stage recovery; track 1 cannot steal it.
    t=new('appearance')
    initial=np.array([[0,0,10,20],[3,0,13,20]],np.float32)
    t.update_candidates(initial,np.array([.9,.9],np.float32),[a,a])
    result=t.update_candidates(boxes,scores,[mixed,a])
    assert result.tracker_id.tolist()==[1,2] and t.selected_indices==(0,1)
    assert t.refinement_counts['weak_reserved_for_unmatched']==1 and not t.refinement_events
    print('Weak recovery options for unmatched active tracks are reserved before refinement: OK')

    for weak_box, weak_feature in (([100,0,110,20],a),([1,0,11,20],None)):
        t=new('appearance');t.update_candidates(box,np.array([.9],np.float32),[a])
        t.update_candidates(np.array([box[0],weak_box],np.float32),scores,[mixed,weak_feature])
        assert t.selected_indices==(0,) and not t.refinement_events
    t=new('appearance')
    assert len(t.update_candidates(box,np.array([.3],np.float32),[a]).tracker_id)==0
    t=new('appearance');t.update_candidates(box,np.array([.9],np.float32),[a])
    t.update_candidates(np.empty((0,4),np.float32),np.empty(0,np.float32),[])
    t.update_candidates(boxes,scores,[mixed,a])
    assert t.selected_indices==(0,) and not t.refinement_events
    print('Far/missing-feature weak candidates cannot challenge; weak-only births and lost-track challenges are disabled: OK')

    baseline, disabled = ExperimentalByteTrack(**SETTINGS), new('disabled')
    for f in range(150):
        if f%19==0:
            b,s,v=np.empty((0,4),np.float32),np.empty(0,np.float32),[]
        else:
            centers=np.array([[f*.3,20],[80-f*.2,50],[140,80]],np.float32)
            centers+=rng.normal(0,.5,centers.shape).astype(np.float32)
            b=np.concatenate([centers,centers+[20,40]],axis=1).astype(np.float32)
            s=np.array([.9,.3 if f%4==0 else .8,.7],np.float32)
            v=[a,mixed,a if f%5 else None]
        x,y=baseline.update_candidates(b,s,v),disabled.update_candidates(b,s,v)
        assert np.array_equal(x.xyxy,y.xyxy) and np.array_equal(x.confidence,y.confidence)
        assert np.array_equal(x.tracker_id,y.tracker_id) and baseline.selected_indices==disabled.selected_indices
    before=disabled.frame_id
    try:
        disabled.update_candidates(box,np.array([.9],np.float32),[np.zeros(512,np.float32)])
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid vector accepted')
    assert before==disabled.frame_id and not disabled.failed
    print('Disabled refinement reproduces the pinned appearance tracker; malformed features do not advance state: EXACT')
    print('Competitive ByteTrack checks: PASSED; synthetic examples are not deployment calibration')


if __name__ == '__main__':
    main()
