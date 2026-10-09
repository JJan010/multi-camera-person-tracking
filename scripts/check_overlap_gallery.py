"""Known-answer gallery admission, causal isolation, age and return fixtures."""
from copy import deepcopy
from dataclasses import asdict, replace
from fractions import Fraction as F
from pathlib import Path
import json
import pickle
import numpy as np
from mtmc.reid.osnet import ObservationKey
from mtmc.reid.feature_history import FeatureBatch, FeatureSpace
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.pipeline.confirmed_return import ConfirmedReturnStage
from mtmc.pipeline.overlap_gallery import OverlapGalleryStage, overlap_scores

ROOT=Path(__file__).resolve().parents[1]
SPACE=FeatureSpace('a'*64,1280)
SIZES={4:(100,100),5:(100,100)}


def record(frame,local=0,box=(20.,20.,40.,60.),confidence=.8,camera=4):
    bounds,fraction=crop_geometry(box,*SIZES[camera])
    return CropRecord(ObservationKey(camera,local,frame),confidence,box,bounds,fraction)


def stage(config,overlap=None):
    kw=dict(space=SPACE,variant='mean',threshold=.7,max_distance=2.,unavailable_policy='appearance_only',
        identity_configuration=dict(max_idle_seconds='1',min_support_rounds=3,min_support_seconds='1/5',max_evidence_gap='1/10'),
        recovery_enabled=True,configuration=config,image_sizes=SIZES)
    cls=ConfirmedReturnStage if overlap is None else OverlapGalleryStage
    if overlap is not None:kw.update(overlap_enabled=overlap,max_overlap=.3)
    return cls('fixture',{4:np.eye(3),5:np.eye(3)},'native',**kw)


def advance(owner,frame,records):
    keys=tuple(r.key for r in records if r.crop_xyxy_int is not None)
    x=np.zeros((len(keys),1280),np.float32)
    for i,k in enumerate(keys):x[i,k.local_id%2]=1
    batch=FeatureBatch('fixture',SPACE,keys,(F(frame,30),)*len(keys),x)
    return owner.update_with_raw(frame,F(frame,30),batch,batch,records)


def main():
    config=json.loads((ROOT/'configs/association/clipreid_confirmed_return.json').read_text())
    a=record(0,box=(10.,10.,20.,20.))
    b=record(0,1,box=(17.,10.,27.,20.),confidence=.1)
    scores=overlap_scores((a,b),SIZES)
    assert scores[a.key]==scores[b.key]==.3
    assert overlap_scores((b,a),SIZES)==scores
    big=record(0,1,box=(5.,5.,25.,25.))
    scores=overlap_scores((a,big),SIZES)
    assert scores[a.key]==1 and scores[big.key]==.25
    other=record(0,1,box=a.source_xyxy,camera=5)
    outside=record(0,2,box=(110.,10.,120.,20.))
    assert all(v==0 for v in overlap_scores((a,other,outside),SIZES).values())
    clipped=record(0,0,box=(-10.,10.,10.,20.))
    contained=record(0,1,box=(0.,10.,5.,20.))
    assert overlap_scores((clipped,contained),SIZES)[clipped.key]==.5
    boundary=stage(config,True);advance(boundary,0,(a,b))
    assert boundary.registry.last_overlap['counts']['accepted']==1
    assert boundary.registry.last_overlap['counts'].get('overlap_rejected',0)==0
    print('Own-area overlap, clipping, asymmetry, weak competitors, camera isolation and inclusive boundary: OK')

    original,disabled,enabled=stage(config),stage(config,False),stage(config,True)
    returns=[]
    for frame in range(150):
        rs=()
        if frame<=20:rs=(record(frame),)
        elif 91<=frame<=99:
            rs=(record(frame,local=8,camera=5,box=(0.,20.,20.,60.) if frame==91 else (20.,20.,40.,60.)),)
        result=[advance(s,frame,rs) for s in (original,disabled,enabled)]
        assert result[0][:5]==result[1][:5]==result[2][:5]
        assert asdict(original.registry.last_audit)==asdict(disabled.registry.last_audit)==asdict(enabled.registry.last_audit)
        assert pickle.dumps(original.registry.base)==pickle.dumps(disabled.registry.base)
        for gid in original.registry.galleries:
            assert pickle.dumps(original.registry.galleries[gid])==pickle.dumps(disabled.registry.galleries[gid])
        returns.extend(frame for _ in enabled.registry.last_audit.reactivations)
    assert returns==[98]
    print('150 rounds: disabled and clean enabled gallery reproduce original decisions, return, lifecycle and past outputs: OK')

    short=deepcopy(config);short['gallery']['max_sample_age_seconds']='1/3'
    protected=stage(short,True);control=stage(short,False)
    for frame in range(19):
        rs=(record(frame),) if frame<8 else (record(frame),record(frame,1,confidence=.1))
        result=advance(protected,frame,rs);expected=advance(control,frame,rs)
        assert result[:5]==expected[:5]  # Gallery admission does not mask queries or observations.
        assert asdict(protected.registry.last_audit.archive)==asdict(control.registry.last_audit.archive)
        if frame==8:
            assert [s.key.frame_index for s in protected.registry.galleries[1]]==list(range(8))
            assert protected.registry.last_overlap['counts']['overlap_rejected']==1
            assert len(result[0].assignments)==2
        if frame==10:assert protected.registry.galleries[1][0].key.frame_index==0
        if frame==11:assert protected.registry.galleries[1][0].key.frame_index==1
        if frame==17:assert [s.key.frame_index for s in protected.registry.galleries[1]]==[7]
        if frame==18:assert not protected.registry.galleries[1]
    print('Rejected updates preserve older gallery slots, never refresh age, and leave raw queries/observations unchanged: OK')
    print('Inclusive age boundary and partial expiry remove actual samples without fabricating replacement evidence: OK')

    protected,control=stage(config,True),stage(config,False)
    return_frames={True:[],False:[]}
    for frame in range(150):
        rs=()
        if frame<=20:rs=(record(frame),record(frame,1,confidence=.1))
        elif 91<=frame<=99:rs=(record(frame,local=8,camera=5),)
        for flag,owner in ((True,protected),(False,control)):
            advance(owner,frame,rs)
            return_frames[flag].extend(frame for _ in owner.registry.last_audit.reactivations)
    assert return_frames[False]==[97] and return_frames[True]==[]
    assert protected.registry.last_audit.lifecycle['reactivation_events']==0
    print('All-overlapped history can remove a recoverable reference: loss of coverage is explicitly demonstrated: OK')

    ordered,reversed_order=stage(config,True),stage(config,True)
    for frame in range(12):
        rs=(record(frame),record(frame,1,box=(30.,20.,50.,60.)),record(frame,2,camera=5))
        r1=advance(ordered,frame,rs);r2=advance(reversed_order,frame,rs[::-1])
        assert r1[:5]==r2[:5]
        assert ordered.registry.last_overlap==reversed_order.registry.last_overlap
    before=pickle.dumps(ordered.registry)
    for rs in ((record(12),record(12)),(replace(record(12),confidence=float('nan')),)):
        try:advance(ordered,12,rs)
        except ValueError:pass
        else:raise AssertionError('Malformed admission accepted')
        assert pickle.dumps(ordered.registry)==before
    advance(ordered,12,(record(12),))
    broken=stage(config,True);broken.registry.configuration['max_live_gallery_identities']=0
    before=pickle.dumps(broken.registry)
    try:advance(broken,0,(record(0),))
    except ValueError:pass
    else:raise AssertionError('Internal failure absent')
    assert broken.failed and pickle.dumps(broken.registry)==before
    print('Input order and ID zero; preflight correction and atomic registry failure including transient mask: OK')
    print('Gallery overlap contract: PASSED; tracking quality and full frozen replay still require the paired experiment')
    print('The 0.3 overlap threshold is one development hypothesis, not a calibrated setting.')


if __name__=='__main__':main()
