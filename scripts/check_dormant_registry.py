"""Synthetic registry/archive bridge checks, not a tracking quality benchmark."""
from dataclasses import replace
from fractions import Fraction as F
from unittest.mock import patch
import numpy as np

from mtmc.association.controlled_merge import ControlledMergeIdentityManager
from mtmc.association.dormant import ArchiveSettings, DormantIdentityArchive
from mtmc.association.grouping import FrameGroups, POLICY
from mtmc.association.recovery import RecoveryIdentityManager, RecoveryEvidence, ObservationEvidence
from mtmc.reid.osnet import ObservationKey as K

SETTINGS=ArchiveSettings(F(5),.8,.05,2.,.5,8)
A=(4,0);B=(8,0);C=(4,1);D=(8,1)


def base():
    return ControlledMergeIdentityManager('recovery-test',max_idle=F(1),descriptor_variant='mean',
        min_similarity=.5,min_support_rounds=2,min_support_seconds=F(1,10),max_evidence_gap=F(1,2))


def manager(enabled=True,settings=SETTINGS):
    return RecoveryIdentityManager(base(),coordinate_space='test-ground',camera_ids=(4,8),
        enabled=enabled,archive_settings=settings if enabled else None)


def frame(n,time,groups):
    return FrameGroups('recovery-test',n,time,'mean',.5,POLICY,
        tuple(tuple(K(c,i,n) for c,i in g) for g in groups),())


def evidence(f,*,missing=False,axis=None):
    rows=[]
    for group in f.groups:
        for key in group:
            vector=np.zeros(512,np.float32);vector[(axis or {}).get((key.camera_id,key.local_id),0)]=1
            rows.append(ObservationEvidence(key,None if missing else vector,None if missing else f.timestamp,
                None if missing else (0.,0.),None if missing else f.timestamp))
    return RecoveryEvidence(f.run_id,'test-ground',f.frame_index,f.timestamp,tuple(rows))


def step(m,n,time,groups,**kwargs):
    f=frame(n,time,groups);return m.update(f,evidence(f,**kwargs))


def mapping(result):
    return {(a.key.camera_id,a.key.local_id):a.global_id for a in result.assignments}


def rejected(call):
    try: call()
    except ValueError: return
    raise AssertionError('Invalid input accepted')


def retired_manager():
    m=manager();step(m,0,F(0),[(A,)])
    step(m,1,F(11,10),[])
    assert m.archived_ids==(1,)
    return m


def main():
    old,new=base(),manager(False)
    for n in range(120):
        groups=[] if 15<=n<55 or 70<=n<110 else [(A,),(B,)] if n in (0,55,110) else [(A,B)]
        f=frame(n,F(n,30),groups)
        a,b=old.update(f),new.update(f)
        assert a==b and new.last_audit.baseline==a
        assert old._bindings==new._base._bindings and old._pending==new._base._pending
        assert old._next_id==new._base._next_id and not new.archived_ids
    print('Recovery disabled: full original records, bindings, pending merges and allocator EXACT over 120 rounds: OK')

    m=manager();first=step(m,0,F(0),[(A,)])
    step(m,1,F(1),[]);assert not m.archived_ids
    step(m,2,F(11,10),[]);assert m.archived_ids==(1,)
    recovered=step(m,3,F(6,5),[(C,)])
    assert mapping(first)=={A:1} and mapping(recovered)=={C:1}
    assert m.last_audit.reactivations[0].superseded_new_id==2
    assert recovered.assignments[0].reason=='dormant_reactivation' and not m.archived_ids
    continued=step(m,4,F(13,10),[(C,),(D,)],axis={D:1})
    assert mapping(continued)=={C:1,D:3} and mapping(first)=={A:1}
    counts=dict(m.last_audit.lifecycle)
    assert counts['allocated_id_slots']==3 and counts['ever_emitted_ids']==2
    assert counts['reactivation_events']==counts['superseded_new_ids']==1
    print('Retirement after idle expiry, same-camera recovery, future local continuity and non-reused provisional IDs: OK')

    m=manager();step(m,0,F(0),[(A,)])
    result=step(m,1,F(1,2),[(C,)])
    assert mapping(result)=={C:2} and not m.last_audit.reactivations and not m.archived_ids
    assert len(result.identities)==2
    print('An absent retained identity is not offered for reactivation; camera reservations stay intact: OK')

    m=retired_manager();result=step(m,2,F(6,5),[(C,),(D,)])
    assert mapping(result)=={C:2,D:3} and m.archived_ids==(1,)
    assert {d.outcome for d in m.last_audit.archive.decisions}=={'ambiguous_identity'}
    print('Competing return groups receive fresh IDs when archive evidence is ambiguous: OK')

    m=retired_manager();r=step(m,2,F(6,5),[(C,D)])
    assert mapping(r)=={C:1,D:1} and len(m.last_audit.reactivations)==1
    step(m,3,F(12,5),[]);assert m.archived_ids==(1,)
    r=step(m,4,F(5,2),[(A,)])
    assert mapping(r)=={A:1}
    counts=dict(m.last_audit.lifecycle)
    assert counts['expiration_events']==counts['reactivation_events']==2 and counts['currently_inactive_ids']==0
    print('A multi-camera group claims one identity; repeated retire/return cycles conserve lifecycle counts: OK')

    m=manager();step(m,0,F(0),[(A,),(B,)])
    step(m,1,F(1,10),[(A,B)])
    merged=step(m,2,F(1,5),[(A,B)])
    assert merged.merge_events[0].absorbed_global_ids==(2,)
    step(m,3,F(13,10),[]);assert m.archived_ids==(1,)
    r=step(m,4,F(7,5),[(C,)])
    assert mapping(r)=={C:1} and m.last_audit.reactivations[0].superseded_new_id==3
    assert dict(m.last_audit.lifecycle)['absorbed_ids']==1
    print('Merge canonicalization is respected; absorbed aliases are never archived or reactivated: OK')

    m=manager(settings=replace(SETTINGS,max_age=F(3)))
    step(m,0,F(0),[(A,)])
    for n in range(1,5): step(m,n,F(n,2),[(A,)],missing=True)
    step(m,5,F(31,10),[])
    assert not m.archived_ids and m.last_audit.archive.expired==(1,)
    m=manager();step(m,0,F(0),[(A,B)])
    step(m,1,F(1,10),[(A,),(B,)])
    assert m.last_audit.snapshot_decisions==((1,'unsupported_visible_partition'),)
    assert m._snapshots[1].descriptor_time==0 and m._snapshots[1].ground_time==0
    print('Missing evidence and split support cannot refresh snapshot source times; stale evidence expires: OK')

    m=manager();f=frame(0,F(0),[(A,B)]);e=evidence(f)
    opposite=-e.observations[1].descriptor
    e=replace(e,observations=(e.observations[0],replace(e.observations[1],descriptor=opposite)))
    m.update(f,e);assert m._snapshots[1].descriptor is None
    step(m,1,F(11,10),[]);assert not m.archived_ids
    print('Cancelling multi-view appearance has no fabricated archival descriptor: OK')

    one,two=manager(),manager()
    sequence=[(0,F(0),[(A,),(B,)]),(1,F(1,10),[(A,B)]),(2,F(1,5),[(A,B)]),
              (3,F(13,10),[]),(4,F(7,5),[(C,D)])]
    for n,t,groups in sequence:
        f=frame(n,t,groups);e=evidence(f)
        a=one.update(f,e)
        f=replace(f,groups=tuple(tuple(reversed(g)) for g in reversed(f.groups)))
        b=two.update(f,replace(e,observations=tuple(reversed(e.observations))))
        assert a==b and one.last_audit==two.last_audit
    print('Group, member and evidence ordering preserves full output and recovery audit: OK')

    f=frame(0,F(0),[(A,)])
    valid=evidence(f)
    malformed=[replace(valid,run_id='other'),replace(valid,coordinate_space='other'),
        replace(valid,observations=()),replace(valid,observations=valid.observations*2),
        replace(valid,observations=(replace(valid.observations[0],descriptor_time=F(1)),)),
        replace(valid,observations=(replace(valid.observations[0],descriptor=np.zeros(512,np.float32)),))]
    for e in malformed:
        m=manager();rejected(lambda:m.update(f,e))
        assert m.update(f,valid)==manager().update(f,valid)
    m=retired_manager();f=frame(2,F(6,5),[(C,)])
    original=DormantIdentityArchive.step
    def fail_after_claim(archive,event):
        original(archive,event)
        raise RuntimeError('Injected failure after archive claim')
    with patch.object(DormantIdentityArchive,'step',fail_after_claim):
        try: m.update(f,evidence(f))
        except RuntimeError: pass
        else: raise AssertionError('Expected injected failure')
    assert m.archived_ids==(1,) and m._base._last_frame==1
    assert m.update(f,evidence(f))==retired_manager().update(f,evidence(f))
    print('Invalid input and failure after archive consumption commit neither registry, archive nor counters: OK')
    print('Dormant registry bridge: PASSED; frozen-trace parity and quality evaluation pending')
    print('All enabled recovery settings are synthetic fixtures, not deployment parameters.')


if __name__=='__main__': main()
