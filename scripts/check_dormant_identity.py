"""Known-answer inactive-identity tests; all thresholds are synthetic fixtures."""
from dataclasses import replace
from fractions import Fraction as F
import numpy as np

from mtmc.association.dormant import (ArchiveSettings, InactiveIdentity, ReturnQuery,
    ArchiveRound, DormantIdentityArchive)
from mtmc.reid.osnet import ObservationKey as K


def vector(i=0):
    value=np.zeros(512,np.float32);value[i]=1.;return value


SETTINGS=ArchiveSettings(F(5),.8,.05,2.,.5,8)


def archive(settings=SETTINGS):
    return DormantIdentityArchive('fixture','native-test-space',(0,4,8),settings)


def memory(gid=7,*,seen=F(0),feature=None,point=(0.,0.),evidence_time=None):
    evidence_time=seen if evidence_time is None else evidence_time
    return InactiveIdentity(gid,seen,vector() if feature is None else feature,
                            evidence_time,point,evidence_time)


def query(frame=1,camera=4,local=10,*,feature=None,point=(0.,0.)):
    return ReturnQuery((K(camera,local,frame),),vector() if feature is None else feature,point)


def event(frame=1,time=F(1),*,retired=(),queries=(),blocked=()):
    return ArchiveRound('fixture','native-test-space',frame,time,retired,queries,blocked)


def rejected(call):
    try: call()
    except ValueError: return
    raise AssertionError('Expected ValueError')


def main():
    a=archive()
    first=a.step(event(retired=(memory(),)))
    assert first.retained_ids==(7,)
    matched=a.step(event(2,F(2),queries=(query(2),)))
    assert matched.decisions[0].global_id==7 and not matched.retained_ids
    again=a.step(event(3,F(3),queries=(query(3,local=11),),blocked=(7,)))
    assert again.decisions[0].global_id is None
    assert first.retained_ids==(7,) and matched.decisions[0].global_id==7
    print('A returning new local track claims an archived global ID once; past outputs stay unchanged: OK')

    a=archive()
    matched=a.step(event(retired=(memory(0),),queries=(query(camera=0,local=0),)))
    assert matched.decisions[0].global_id==0
    a.step(event(2,F(2),blocked=(0,)))
    result=a.step(event(3,F(3),retired=(memory(0,seen=F(2)),),queries=(query(3,local=2),)))
    assert result.decisions[0].global_id==0
    print('ID zero, same-camera return and explicit later retirement of a reactivated identity: OK')

    a=archive();a.step(event(retired=(memory(),)))
    result=a.step(event(2,F(2),queries=(query(2),),blocked=(7,)))
    assert result.blocked_removed==(7,) and result.decisions[0].global_id is None
    rejected(lambda:archive().step(event(retired=(memory(),),blocked=(7,))))
    print('Live/retained or absorbed IDs supplied as blocked cannot be recovered from the archive: OK')

    a=archive()
    result=a.step(event(retired=(memory(7),memory(8)),queries=(query(),)))
    assert result.decisions[0].outcome=='ambiguous_query' and result.retained_ids==(7,8)
    a=archive()
    result=a.step(event(retired=(memory(),),queries=(query(),query(camera=8,local=11))))
    assert {d.outcome for d in result.decisions}=={'ambiguous_identity'} and result.retained_ids==(7,)
    print('Tied identity alternatives and competing return groups are rejected without first-come allocation: OK')

    near=vector();near[0]=.99;near[1]=np.sqrt(1-.99**2)
    result=archive().step(event(retired=(memory(),),queries=(query(),query(camera=8,feature=near))))
    assert all(d.global_id is None for d in result.decisions)
    result=archive().step(event(retired=(memory(7),memory(8,feature=vector(1))),
        queries=(query(),query(camera=8,feature=vector(1)))))
    assert [d.global_id for d in result.decisions]==[7,8]
    print('A small similarity advantage remains ambiguous; two distinct mutual matches remain one-to-one: OK')

    alternative=vector();alternative[0]=.75;alternative[1]=np.sqrt(1-.75**2)
    settings=replace(SETTINGS,min_similarity=.5,min_margin=.25)
    result=archive(settings).step(event(retired=(memory(7),memory(8,feature=alternative)),queries=(query(),)))
    assert result.decisions[0].global_id==7
    alternative[0]=np.nextafter(np.float32(.75),np.float32(1.))
    alternative[1]=np.sqrt(1-float(alternative[0])**2)
    result=archive(settings).step(event(retired=(memory(7),memory(8,feature=alternative)),queries=(query(),)))
    assert result.decisions[0].outcome=='ambiguous_query'
    print('Ambiguity margin is inclusive at the configured boundary: OK')

    other=vector();other[0]=.8;other[1]=-.6
    competitor=vector();competitor[0]=.95;competitor[1]=np.sqrt(1-.95**2)
    result=archive(replace(SETTINGS,min_similarity=.5,min_margin=.01)).step(event(
        retired=(memory(7),memory(8,feature=other)),queries=(query(),query(camera=8,feature=competitor))))
    assert [d.global_id for d in result.decisions]==[7,None]
    assert result.decisions[1].outcome=='not_mutual_best'
    assert sum(e.outcome=='eligible' for e in result.evidence)==4
    print('Known tradeoff: mutual-best rejection can leave a valid alternative unused; no optimal-assignment claim: DEMONSTRATED')

    a=archive();a.step(event(retired=(memory(),)))
    result=a.step(event(2,F(5)))
    assert result.retained_ids==(7,)
    result=a.step(event(3,F(501,100)))
    assert result.expired==(7,) and not result.retained_ids
    stale=memory(seen=F(4),evidence_time=F(0))
    result=archive().step(event(time=F(6),retired=(stale,)))
    assert result.expired==(7,)
    print('Inclusive scene-age boundary; empty rounds expire memory; retirement never refreshes old evidence: OK')

    settings=replace(SETTINGS,max_speed=1.,position_slack=0.)
    result=archive(settings).step(event(time=F(5),retired=(memory(),),queries=(query(point=(3.,4.)),)))
    assert result.decisions[0].global_id==7 and result.evidence[0].distance_limit==5
    result=archive(settings).step(event(time=F(4),retired=(memory(),),queries=(query(point=(3.,4.)),)))
    assert result.decisions[0].global_id is None and result.evidence[0].outcome=='geometry_rejected'
    result=archive(replace(SETTINGS,min_similarity=0.)).step(event(retired=(memory(),),queries=(query(feature=vector(1)),)))
    assert result.evidence[0].outcome=='appearance_rejected'
    print('Inclusive motion-distance boundary and strict appearance boundary: OK')

    missing_query=replace(query(),descriptor=None)
    result=archive().step(event(retired=(memory(),),queries=(missing_query,)))
    assert result.decisions[0].outcome=='missing_descriptor' and result.retained_ids==(7,)
    result=archive().step(event(retired=(memory(),),queries=(replace(query(),ground_xy=None),)))
    assert result.decisions[0].outcome=='missing_geometry'
    incomplete=replace(memory(),descriptor=None,descriptor_time=None)
    result=archive().step(event(retired=(incomplete,),queries=(query(),)))
    assert result.skipped_retirements==((7,'missing_evidence'),) and not result.retained_ids
    print('Missing appearance or geometry has explicit unmatched behavior; no permissive return fallback: OK')

    records=(memory(7),memory(8,seen=F(1,2)),memory(9,seen=F(1,2)))
    a=archive(replace(SETTINGS,max_identities=2))
    result=a.step(event(retired=records))
    assert result.capacity_evicted==(7,) and result.retained_ids==(8,9)
    assert a.vector_payload_bytes==2*512*4
    b=archive(replace(SETTINGS,max_identities=2))
    assert b.step(event(retired=tuple(reversed(records))))==result
    print('Capacity bound and oldest-last-seen eviction are deterministic; vector payload is bounded: OK')

    a=archive();source=vector();a.step(event(retired=(memory(feature=source),)))
    source[:]=vector(1)
    assert a.step(event(2,F(2),queries=(query(2),))).decisions[0].global_id==7
    qs=(ReturnQuery((K(8,2,1),K(4,1,1)),vector(),(0.,0.)),query(camera=0,feature=vector(1)))
    inputs=event(retired=(memory(7),memory(8,feature=vector(1))),queries=qs)
    a=archive().step(inputs)
    reordered=replace(inputs,retirements=tuple(reversed(inputs.retirements)),
        queries=tuple(replace(q,members=tuple(reversed(q.members))) for q in reversed(qs)))
    assert archive().step(reordered)==a
    print('Owned descriptor storage; retirement/query/member order preserves canonical decisions: OK')

    invalids=[replace(event(),run_id='other'),replace(event(),coordinate_space='other'),
        event(retired=(memory(),memory())),event(blocked=(7,7)),
        event(queries=(query(),query())),event(retired=(replace(memory(),last_seen=F(1)),)),
        event(retired=(replace(memory(),descriptor_time=F(1)),)),
        event(queries=(replace(query(),descriptor=np.zeros(512,np.float32)),)),
        event(queries=(replace(query(),ground_xy=(float('nan'),0.)),)),
        event(queries=(replace(query(),members=(K(4,10,2),)),)),
        event(queries=(replace(query(),members=(K(4,10,1),K(4,11,1))),))]
    for bad in invalids:
        a=archive();rejected(lambda:a.step(bad))
        good=event(retired=(memory(),),queries=(query(),))
        assert a.step(good)==archive().step(good)
    a=archive();a.step(event(retired=(memory(),)))
    rejected(lambda:a.step(event(2,F(1))))
    rejected(lambda:a.step(event(1,F(2))))
    assert a.step(event(2,F(2),queries=(query(2),))).decisions[0].global_id==7
    print('Malformed scopes, keys, vectors, geometry and times rejected before mutation; rejected rounds can be corrected: OK')
    print('Dormant identity archive contract: PASSED; registry integration and quality evaluation pending')
    print('All thresholds, speeds, ages and margins above are synthetic examples, not calibrated settings.')


if __name__=='__main__':
    main()
