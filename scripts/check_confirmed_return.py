"""Synthetic confirmation/archive contracts; no recording-quality claims."""
from dataclasses import replace
from fractions import Fraction as F
from copy import deepcopy
import pickle
import numpy as np
from mtmc.reid.feature_history import FeatureSpace
from mtmc.reid.osnet import ObservationKey
from mtmc.association.dormant import ArchiveSettings, InactiveIdentity
from mtmc.association.confirmed_return import (ConfirmationSettings, ConfirmedReturnArchive,
    ConfirmationRound, YoungIdentity, GallerySample, gallery_reference)

SPACE=FeatureSpace('a'*64,1280)


def vec(index=0):
    v=np.zeros(1280,np.float32);v[index]=1;return v


def angle(x):
    v=vec();v[0]=x;v[1]=np.sqrt(1-x*x);return v


def settings(**kw):
    return replace(ConfirmationSettings(ArchiveSettings(F(30),.8,.05,0.,2.,128),F(2),F(1,10),3,F(1,5),F(1,10),.8),**kw)


def owner(cfg=None):return ConfirmedReturnArchive('run','floor',(4,5,8),space=SPACE,settings=cfg or settings())


def snapshot(gid=0,v=None):return InactiveIdentity(gid,F(0),vec() if v is None else v,F(0),(0.,0.),F(0))


def query(frame,pid=100,v=None,*,quality=True,members=None):
    members=members or (ObservationKey(4,pid,frame),)
    return YoungIdentity(pid,F(1,3),members,tuple((k.camera_id,k.local_id) for k in members),
        vec() if v is None else v,F(frame,30),(0.,0.),quality)


def event(frame,queries=(),retirements=(),blocked=None):
    return ConfirmationRound('run','floor',SPACE,frame,F(frame,30),tuple(retirements),tuple(queries),
        tuple(sorted(q.global_id for q in queries)) if blocked is None else tuple(blocked))


def rejects(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid input accepted')


def main():
    a=owner();past=[]
    for f in range(10,17):
        r=a.step(event(f,(query(f),),(snapshot(),) if f==10 else ()));past.append(r)
        assert r.decisions[0].outcome==('confirmed' if f==16 else 'pending')
    assert past[-1].decisions[0].archived_id==0 and past[-1].decisions[0].support_rounds==7
    assert past[-1].decisions[0].support_span==F(1,5) and a.retained_ids==()
    assert all(r.decisions[0].archived_id is None for r in past[:-1])
    rejects(lambda:a.step(event(17,(query(17),))))
    print('ID zero, support count AND elapsed time, one claim, and unchanged earlier decisions: OK')

    a=owner();a.step(event(10,(query(10,quality=False),),(snapshot(),)))
    for f in range(11,18):r=a.step(event(f,(query(f),)))
    assert r.decisions[0].outcome=='confirmed'
    for interruption in ('weak','absent','missing','gap','retained'):
        a=owner();a.step(event(10,(query(10),),(snapshot(),)))
        if interruption=='weak':r=a.step(event(11,(query(11,quality=False),)))
        elif interruption=='absent':r=a.step(event(11))
        elif interruption=='missing':r=a.step(event(11,(replace(query(11),descriptor=None,descriptor_time=None),)))
        elif interruption=='gap':r=a.step(event(14,(query(14),)))
        else:r=a.step(event(11,(replace(query(11),retained_members=((4,100),(5,101))),)))
        assert a.pending_count==(1 if interruption in ('gap','retained') else 0)
        if a.pending_count:assert r.decisions[0].support_rounds==1
        assert r.resets
    print('Rejected first crop can retry; weak/missing/absent input, time gaps and membership changes reset support: OK')

    a=owner();a.step(event(10,(query(10,v=angle(.9)),),(snapshot(),)))
    other=angle(.9);other[1]*=-1
    r=a.step(event(11,(query(11,v=other),)))
    assert r.resets==((100,'candidate_incoherent'),) and r.decisions[0].support_rounds==1
    a=owner();a.step(event(10,(query(10),),(snapshot(),)))
    r=a.step(event(11,(query(11),),blocked=(0,100)))
    assert r.blocked_removed==(0,) and a.retained_ids==() and a.pending_count==0
    print('Candidate coherence and live/absorbed-ID blocking cannot be bypassed by repeated support: OK')

    cfg=settings(archive=ArchiveSettings(F(30),.9,.05,0.,2.,128))
    a=owner(cfg);r=a.step(event(10,(query(10),),(snapshot(0,angle(.92)),snapshot(1,angle(.89)))))
    assert r.decisions[0].outcome=='ambiguous_query'
    a=owner();r=a.step(event(10,(query(10,100),query(10,101)),(snapshot(),)))
    assert all(d.outcome=='ambiguous_identity' for d in r.decisions)
    cfg=settings(archive=ArchiveSettings(F(30),.5,.125,0.,2.,128))
    r=owner(cfg).step(event(10,(query(10),),(snapshot(0,angle(.875)),snapshot(1,angle(.75)))))
    assert r.decisions[0].outcome=='pending'
    cfg=settings(archive=ArchiveSettings(F(30),.875,.05,0.,2.,128))
    r=owner(cfg).step(event(10,(query(10),),(snapshot(0,angle(.875)),)))
    assert r.decisions[0].outcome=='appearance_rejected'
    print('Below-threshold rivals count toward ambiguity; competing queries, inclusive margin and strict appearance boundary: OK')

    a=owner();q=replace(query(10),ground_xy=(2.,0.))
    assert a.step(event(10,(q,),(snapshot(),))).decisions[0].outcome=='pending'
    a=owner();q=replace(query(10),ground_xy=None)
    assert a.step(event(10,(q,),(snapshot(),))).decisions[0].outcome=='missing_geometry'
    cfg=settings(archive=ArchiveSettings(F(1),.8,.05,0.,2.,1))
    a=owner(cfg);e=replace(snapshot(),last_seen=F(1,2))
    a.step(event(30,retirements=(e,)));assert a.retained_ids==(0,)
    r=a.step(event(31));assert r.expired==(0,)
    a=owner(cfg);r=a.step(event(10,retirements=(snapshot(2),snapshot(0))))
    assert r.capacity_evicted==(0,) and a.retained_ids==(2,)
    a=owner();r=a.step(event(71,(query(71),),(snapshot(),)))
    assert r.decisions[0].outcome=='query_too_old'
    print('Inclusive geometry/age boundaries, explicit missing geometry, independent old evidence age and deterministic capacity: OK')

    a,b=owner(),owner()
    for f in range(10,17):
        qs=(query(f,100),query(f,101,vec(1)))
        es=(snapshot(0),snapshot(1,vec(1))) if f==10 else ()
        x=a.step(event(f,qs,es));y=b.step(event(f,qs[::-1],es[::-1]))
        assert x==y
    assert {d.archived_id for d in x.decisions}=={0,1}
    a=owner();members=tuple(ObservationKey(c,100,10) for c in (4,5,8))
    assert a.step(event(10,(query(10,members=members),),(snapshot(),))).decisions[0].support_rounds==1
    print('Two independent returns stay one-to-one; camera/query/retirement order invariant; three views count once: OK')

    a=owner();v=vec();q=query(10,v=v);s=snapshot(v=v)
    a.step(event(10,(q,),(s,)));v[:]=0
    assert np.isclose(np.linalg.norm(a._archive._entries[0].descriptor),1)
    assert np.isclose(np.linalg.norm(a._pending[100].seed),1)
    good=event(11,(query(11),))
    for bad in (replace(good,run_id='other'),replace(good,space=FeatureSpace('b'*64,1280)),
                replace(good,queries=(replace(query(11),descriptor=np.zeros(512,np.float32)),)),
                replace(good,frame_index=10),replace(good,queries=(query(11),query(11)))):
        before=pickle.dumps(a);rejects(lambda:a.step(bad));assert pickle.dumps(a)==before
    a.step(good)
    assert a.vector_payload_bytes<=1280*4*(a.settings.archive.max_identities+a.settings.max_queries_per_round)
    print('Owned vectors, model/run isolation and invalid rounds reject atomically; corrected round remains usable: OK')

    samples=tuple(GallerySample(ObservationKey(c,0,f),F(f,30),vec(),.5,.01,(0.,0.)) for c in (4,5) for f in range(10))
    ref,selected=gallery_reference(0,F(9,30),samples,now=F(1),space=SPACE,max_age=F(30))
    assert len(selected)==16 and ref.descriptor_time==F(2,30) and ref.ground_time==F(9,30)
    assert ref.last_seen==F(9,30)
    original=ref.descriptor.copy();selected[0].vector[:]=0;assert np.array_equal(ref.descriptor,original)
    ref,selected=gallery_reference(0,F(9,30),samples,now=F(1),space=SPACE,max_age=F(4,5))
    assert len(selected)==8 and ref.descriptor_time==F(1,5)
    weak=replace(samples[-1],confidence=.49)
    ref,selected=gallery_reference(0,F(9,30),(weak,),now=F(1),space=SPACE,max_age=F(30))
    assert ref.descriptor is None and not selected
    pair=(samples[0],replace(samples[1],vector=-vec()))
    ref,_=gallery_reference(0,F(1,30),pair,now=F(1),space=SPACE,max_age=F(30))
    assert ref.descriptor is None and ref.descriptor_time is None
    print('Per-camera gallery cap, inclusive quality gates, source ages, partial expiry and cancellation: OK')
    print('Confirmed return archive contract: PASSED; registry integration and recording quality NOT YET evaluated')
    print('Synthetic thresholds are fixtures; configuration is one retrospective development hypothesis.')


if __name__=='__main__':main()
