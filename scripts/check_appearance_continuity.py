"""Synthetic checks for one declared appearance-discontinuity hypothesis."""
from dataclasses import asdict, replace
from fractions import Fraction
import json
from pathlib import Path
import pickle
from unittest.mock import patch

import numpy as np

from check_track_segments import batch, stage, vector
from mtmc.reid.history import AppearanceHistory
from mtmc.tracking.continuity import AppearanceContinuity, ContinuitySettings, reference, _Sample
from mtmc.tracking.segments import LocalTrackSegments, SegmentBreak
from mtmc.reid.osnet import ObservationKey

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_settings():
    config = json.loads((ROOT / 'configs/tracking/appearance_continuity_experiment.json').read_text())
    require(config['schema_version'] == 1 and config['experiment'] == 'sparse_reference_confirmed_discontinuity_v1'
            and config['threshold_search'] is False, 'Unexpected experiment configuration')
    raw = dict(config['continuity'])
    for field in ('max_reference_age', 'reference_sample_interval', 'min_support_seconds', 'max_support_gap'):
        raw[field] = Fraction(raw[field])
    return ContinuitySettings(**raw)


def source(frame, person=0, *, confidence=.8, outside=False):
    value = batch(frame, [(4,0,None if outside else person)])
    return replace(value, records=tuple(replace(r, confidence=confidence) for r in value.records))


def warm(settings):
    owner = AppearanceContinuity('fixture', (4,5,8), enabled=True, settings=settings)
    for frame in range(18):
        owner.update(source(frame))
    return owner


def main():
    settings = load_settings()
    require(settings == ContinuitySettings(8,Fraction(3),Fraction(1,5),3,.5,.6,.8,3,Fraction(1,5),Fraction(1,10)),
            'Declared single hypothesis changed')
    controllers = [AppearanceContinuity('fixture',(4,5,8),enabled=v,settings=settings) for v in (False,True)]
    histories = [AppearanceHistory('fixture') for _ in range(3)]; stages = [stage() for _ in range(3)]
    for frame in range(120):
        value = batch(frame, [] if frame in (60,61) else None)
        outputs = [c.update(value) for c in controllers]
        identities = []
        for h,s,current in zip(histories,stages,[value,*(o.segmented for o in outputs)]):
            history = h.update(frame,value.timestamp,current.features)
            result,*_ = s.update(frame,value.timestamp,history.mean,current.records)
            identities.append(asdict(result))
        require(identities[0] == identities[1] == identities[2], 'Clean no-cut global records differ')
    require(not controllers[0]._samples and controllers[0].stored_vectors == 0, 'Disabled policy stored appearance')
    print('Disabled and clean enabled policy: original history/global records EXACT over 120 rounds: OK')

    owner = warm(settings)
    require(tuple(s.frame for s in owner._samples[4,0]) == (0,6,12), 'Reference is not temporally sparse')
    decisions = []
    for frame in range(18,25):
        result = owner.update(source(frame,1)); decision = result.decisions[0]; decisions.append(decision)
        require(decision.reference_frames == (0,6,12), 'Pending reference assimilated contradiction')
        require(all(t < Fraction(frame,30) for t in decision.reference_times), 'Self/future reference')
        require(bool(result.segmented.events) == (frame == 24), 'Cut ignored scene-time confirmation')
    require(decisions[-1].support_rounds == 7 and decisions[-1].support_span == Fraction(1,5), 'Support accounting differs')
    require(tuple(s.frame for s in owner._samples[4,0]) == (24,), 'Cut seeded pre-cut reference samples')
    print('Sparse prior reference freezes before contamination; both support count and elapsed scene time required: OK')

    # Agreement, weak/missing/absent observations each interrupt confirmation.
    for kind in ('agreement','weak','outside','absent'):
        owner = warm(settings); owner.update(source(18,1))
        value = (source(19) if kind == 'agreement' else source(19,1,confidence=.49) if kind == 'weak'
                 else source(19,outside=True) if kind == 'outside' else batch(19,[]))
        result = owner.update(value)
        require(owner.pending_tracks == 0 and not result.segmented.events and result.resets, 'Evidence not interrupted')
        again = owner.update(source(20,1))
        require(again.decisions[0].support_rounds == 1, 'Interrupted evidence was reused')
    owner = warm(settings);owner.update(source(18,1))
    gap = owner.update(source(22,1))
    require(gap.decisions[0].support_rounds == 1
            and gap.resets[0].reason == 'support_time_gap', 'Undelivered large gap did not restart support')
    owner = warm(settings)
    for frame in (18,21,24):
        result = owner.update(source(frame,1))
    require(len(result.segmented.events) == 1, 'Inclusive .1s support gap or .2s duration differs')
    print('Transient disagreement, weak/missing/absent observations and excessive scene gaps cannot accumulate a cut: OK')

    owner = warm(settings)
    for frame in range(18,35):
        result = owner.update(source(frame,1 + frame % 2))
        require(not result.segmented.events and result.decisions[0].support_rounds == 1, 'Incoherent candidates accumulated')
    expired = owner.update(batch(200,[]))
    require(owner.stored_vectors == 0 and owner.pending_tracks == 0, 'Old vector evidence did not expire')
    owner = AppearanceContinuity('fixture',(4,5,8),enabled=True,
                                settings=replace(settings,max_reference_age=Fraction(1,2)))
    for frame in range(13):owner.update(source(frame))
    for frame in (13,14,15):pending=owner.update(source(frame,1))
    require(owner.pending_tracks==1 and pending.decisions[0].support_rounds==3,
            'Reference age equality incorrectly expired pending evidence')
    expired = owner.update(source(16,1))
    require(not expired.segmented.events and owner.pending_tracks==0
            and any(r.reason=='frozen_reference_expired' for r in expired.resets),
            'Expired frozen reference did not clear confirmation')
    print('Candidate appearance must remain coherent; stale references and empty rounds release vector evidence: OK')

    # Explicit boundary values avoid floating-point ambiguity in synthetic fixtures.
    owner = warm(replace(settings,break_similarity=0.))
    boundary = owner.update(source(18,1,confidence=.5))
    require(boundary.decisions[0].reference_similarity == 0.
            and boundary.decisions[0].outcome == 'reference_updated' and owner.pending_tracks == 0,
            'Strict contradiction/inclusive confidence boundary differs')
    owner = warm(replace(settings,candidate_similarity=1.))
    for frame in range(18,25):result=owner.update(source(frame,1))
    require(len(result.segmented.events) == 1, 'Inclusive candidate coherence boundary differs')
    samples=(_Sample(0,Fraction(0),vector(0)),_Sample(1,Fraction(1,30),-vector(0)))
    ref,frames,times,fallback=reference(samples)
    require(fallback and frames==(1,) and times==(Fraction(1,30),) and np.array_equal(ref,-vector(0)),
            'Cancellation fallback provenance differs')
    print('Confidence, strict contradiction, coherent-candidate and time boundaries; cancellation provenance: OK')

    # Illustrate why this guard differs from an eight-consecutive-frame mean.
    # The same numerical pattern could be pose change: no GT person claim here.
    owner = AppearanceContinuity('fixture',(4,5,8),enabled=True,settings=settings)
    past=[]; short_similarities=[]; cuts=[]
    for frame in range(61):
        angle=np.deg2rad(2*frame);v=np.zeros(512,np.float32);v[:2]=[np.cos(angle),np.sin(angle)]
        value=source(frame);value=replace(value,features=replace(value.features,embeddings=v[None,:].copy()))
        if past:
            mean=np.mean(past[-8:],axis=0,dtype=np.float64);mean/=np.linalg.norm(mean)
            short_similarities.append(float(np.dot(v.astype(np.float64),mean)))
        result=owner.update(value);cuts.extend(result.segmented.events);past.append(v)
    require(cuts and min(short_similarities) >= settings.break_similarity, 'Sparse/short history distinction not demonstrated')
    print('Slow drift can trigger a cut while short rolling history agrees; possible pose-change false cuts: DEMONSTRATED')

    # Clone-based transaction: malformed later input and downstream failure cannot commit.
    owner=warm(settings);good=source(18,1)
    for value in (replace(good,run_id='other'),replace(good,timestamp=Fraction(0)),
                  replace(good,features=replace(good.features,embeddings=np.full((1,512),np.nan,np.float32))),
                  replace(good,breaks=(SegmentBreak(ObservationKey(4,0,18),0,'external'),))):
        before=pickle.dumps(owner)
        try:owner.update(value)
        except (ValueError,TypeError):pass
        else:raise AssertionError('Malformed input accepted')
        require(pickle.dumps(owner)==before,'Rejected input mutated state')
    before=pickle.dumps(owner)
    with patch.object(LocalTrackSegments,'update',side_effect=RuntimeError('injected after policy planning')):
        try:owner.update(good)
        except RuntimeError:pass
        else:raise AssertionError('Injected failure absent')
    require(pickle.dumps(owner)==before,'Partial policy work committed after failed segment update')
    result=owner.update(good);good.features.embeddings[:]=0
    require(np.isclose(np.linalg.norm(owner._pending[4,0].seed),1), 'Stored candidate aliases input')
    result.segmented.features.embeddings[:]=0
    require(np.isclose(np.linalg.norm(owner._pending[4,0].seed),1), 'Stored candidate aliases output')
    print('Malformed input and internal failure are atomic; corrected rounds and owned vector storage: OK')

    a=AppearanceContinuity('fixture',(4,5,8),enabled=True,settings=settings)
    b=AppearanceContinuity('fixture',(8,5,4),enabled=True,settings=settings)
    for frame in range(25):
        p=0 if frame<18 else 1;value=batch(frame,[(4,0,p),(5,0,p)])
        other=replace(value,records=value.records[::-1],features=replace(value.features,
            keys=value.features.keys[::-1],timestamps=value.features.timestamps[::-1],embeddings=value.features.embeddings[::-1]))
        x,y=a.update(value),b.update(other)
        require(x.decisions==y.decisions and x.resets==y.resets and x.segmented.events==y.segmented.events,
                'Order changed policy decisions')
    print('Independent cameras and input order preserve break decisions, generations and allocated segment IDs: OK')
    print('Appearance continuity contract: PASSED; frozen replay integration and quality evaluation pending')
    print('Declared settings are one development hypothesis, not calibrated deployment thresholds.')


if __name__ == '__main__':
    main()
