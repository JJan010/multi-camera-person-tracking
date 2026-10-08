"""Known-answer source-time and raw-box projection adapter checks."""
from dataclasses import replace
from fractions import Fraction as F
import numpy as np

from mtmc.association.dormant import ArchiveSettings
from mtmc.pipeline.core import IdentityStage
from mtmc.pipeline.recovery import RecoveryIdentityStage
from mtmc.reid.crops import CropRecord
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey as K, ReIDBatch


def args():
    return dict(run_id='evidence-check',matrices={4:np.eye(3),8:np.eye(3)},coordinate_space='native-test',
        variant='mean',threshold=.7,max_distance=2.,unavailable_policy='appearance_only',
        identity_configuration={'max_idle_seconds':'1','min_support_rounds':2,
            'min_support_seconds':'1/10','max_evidence_gap':'1/2'})


def stage(enabled=False):
    return RecoveryIdentityStage(**args(),recovery_enabled=enabled,
        archive_settings=ArchiveSettings(F(5),.8,.05,0.,2.,128) if enabled else None,
        history_max_observations=8,history_max_age=F(1))


def batch(frame,time,members,sign=1):
    keys=tuple(K(c,i,frame) for c,i in members)
    vectors=np.zeros((len(keys),512),np.float32);vectors[:,0]=sign
    observations=ReIDBatch(keys,(time,)*len(keys),vectors)
    records=tuple(CropRecord(k,.9,(0.,0.,1.,1.),(0,0,1,1),1.) for k in keys)
    return observations,records


def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Malformed input accepted')


def main():
    original=IdentityStage(**args());control=stage();history=AppearanceHistory('evidence-check')
    for frame in range(90):
        time=F(frame,30);members=[] if 15<=frame<55 else [(4,0),(8,3)]
        obs,records=batch(frame,time,members);h=history.update(frame,time,obs)
        expected=original.update(frame,time,h.mean,records)
        actual=control.update_with_history(frame,time,h,records)
        assert actual[0]==expected[0] and actual[1]==expected[1]
        for row in control.last_evidence.observations:
            index=h.mean.keys.index(row.key)
            assert np.array_equal(row.descriptor,h.mean.embeddings[index])
            assert row.descriptor_time==h.source_times[index][0] and row.ground_time==time
            assert row.ground_xy==(.5,1.)
    print('Disabled adapter: full original identity/grouping parity and exact key/vector/projection mapping over 90 rounds: OK')

    hstore=AppearanceHistory('evidence-check');s=stage()
    obs,records=batch(0,F(0),[(4,0)])
    s.update_with_history(0,F(0),hstore.update(0,F(0),obs),records)
    obs,records=batch(1,F(1,30),[(4,0)],sign=-1)
    h=hstore.update(1,F(1,30),obs);assert h.used_latest_fallback==(True,)
    s.update_with_history(1,F(1,30),h,records)
    assert s.last_evidence.observations[0].descriptor_time==F(1,30)
    assert h.source_times[0][0]==0
    print('Latest-vector cancellation fallback keeps the actual latest source time, not the cancelled mean age: OK')

    s=stage();hstore=AppearanceHistory('evidence-check')
    obs,_=batch(0,F(0),[]);h=hstore.update(0,F(0),obs)
    record=CropRecord(K(4,0,0),.8,(-4.,0.,-3.,1.),None,0.)
    result=s.update_with_history(0,F(0),h,(record,))[0]
    assert len(result.assignments)==1
    assert s.last_evidence.observations[0].descriptor is None and s.last_evidence.observations[0].ground_xy is None
    print('Unencoded outside observation is preserved with explicit unavailable archive evidence: OK')

    s=stage(True);store=AppearanceHistory('evidence-check')
    for n,t,members in ((0,F(0),[(4,0)]),(1,F(11,10),[]),(2,F(6,5),[(8,9)])):
        obs,records=batch(n,t,members)
        result=s.update_with_history(n,t,store.update(n,t,obs),records)[0]
    assert result.assignments[0].global_id==1 and s.registry.last_audit.reactivations
    print('History -> raw-box geometry -> grouping -> archive -> recovered registry ID: OK')

    obs,records=batch(0,F(0),[(4,0),(8,0)])
    h=AppearanceHistory('evidence-check').update(0,F(0),obs)
    bad=[replace(h,run_id='other'),replace(h,source_times=((F(1),),(F(0),))),
         replace(h,source_frames=((1,),(0,))),replace(h,used_latest_fallback=(True,False)),
         replace(h,mean=replace(h.mean,keys=tuple(reversed(h.mean.keys))))]
    for malformed in bad:
        s=stage();rejected(lambda:s.update_with_history(0,F(0),malformed,records))
        assert not s.failed and s.registry.last_audit is None
        assert s.update_with_history(0,F(0),h,records)[0]==stage().update_with_history(0,F(0),h,records)[0]
    # Reversing complete keyed provenance rows is valid and preserves decisions.
    reordered=replace(h,latest=replace(h.latest,keys=tuple(reversed(h.latest.keys)),embeddings=h.latest.embeddings[::-1].copy()),
        mean=replace(h.mean,keys=tuple(reversed(h.mean.keys)),embeddings=h.mean.embeddings[::-1].copy()))
    assert stage().update_with_history(0,F(0),reordered,records[::-1])[0]==stage().update_with_history(0,F(0),h,records)[0]
    print('Malformed provenance rejected before registry mutation; correctly keyed input permutations preserve output: OK')
    print('Recovery evidence adapter: PASSED; paired frozen-video quality evaluation pending')


if __name__=='__main__':main()
