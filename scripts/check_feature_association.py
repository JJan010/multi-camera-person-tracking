"""Geometry/identity compatibility and feature-space isolation fixtures."""
from dataclasses import asdict, replace
from fractions import Fraction
from itertools import combinations
import pickle
import numpy as np
from mtmc.association.pairwise import CameraAppearance
from mtmc.association.geometry import (CameraGroundPositions, GroundObservation,
    associate_camera_pair_with_geometry)
from mtmc.association.feature_geometry import CameraFeatures, associate_feature_pair, group_feature_pairs
from mtmc.pipeline.core import IdentityStage
from mtmc.pipeline.feature_identity import FeatureIdentityStage
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch
from mtmc.reid.crops import CropRecord


def same_stages(old,new):
    a=list(old[:5]);b=list(new[:5]);b[2]=tuple(p.result for p in b[2])
    def plain(v):
        if hasattr(v,'__dataclass_fields__'):return {k:plain(x) for k,x in asdict(v).items()}
        if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return [plain(x) for x in v]
        return v
    assert plain(a)==plain(b),'Pair/group/ground/identity records differ'


def rejected(call):
    try:call()
    except (ValueError,TypeError):return
    raise AssertionError('Invalid input accepted')


def main():
    space=FeatureSpace('a'*64,512);other=FeatureSpace('b'*64,512)
    wide=FeatureSpace('c'*64,1280);rng=np.random.default_rng(14)
    # Nontrivial random normalized features, empty camera slots, missing geometry.
    for frame in range(24):
        old={};new={};grounds={};t=Fraction(frame,30)
        for camera in (0,4,8):
            n=(frame+camera)%4;keys=tuple(ObservationKey(camera,i,frame) for i in range(n))
            x=rng.normal(size=(n,512)).astype(np.float32);x/=np.linalg.norm(x,axis=1,keepdims=True)
            old[camera]=CameraAppearance('fixture',camera,frame,t,'mean',ReIDBatch(keys,(t,)*n,x))
            new[camera]=CameraFeatures('fixture',camera,frame,t,'mean',FeatureBatch('fixture',space,keys,(t,)*n,x))
            grounds[camera]=CameraGroundPositions('fixture',camera,frame,t,'floor',
                tuple(GroundObservation(k,None if k.local_id%2 else (float(k.local_id),0.)) for k in keys))
        results=[]
        for a,b in combinations((0,4,8),2):
            cfg=dict(min_similarity=-.1 if frame%2 else .7,max_distance=2.,unavailable_policy='reject' if frame%2 else 'appearance_only')
            baseline=associate_camera_pair_with_geometry(old[a],old[b],grounds[a],grounds[b],**cfg)
            got=associate_feature_pair(new[a],new[b],grounds[a],grounds[b],**cfg)
            assert baseline==got.result;results.append(got)
            reverse=associate_feature_pair(new[b],new[a],grounds[b],grounds[a],**cfg)
            assert reverse.result==associate_camera_pair_with_geometry(old[b],old[a],grounds[b],grounds[a],**cfg)
        group_feature_pairs(results)
        rejected(lambda:group_feature_pairs([replace(results[0],space=other),*results[1:]]))
        mismatched=replace(new[4],observations=replace(new[4].observations,space=other))
        rejected(lambda:associate_feature_pair(new[0],mismatched,grounds[0],grounds[4],**cfg))
    print('512-D pair results EXACT: orientation, empty cameras, geometry fallback/rejection; mixed spaces rejected: OK')
    cfg=dict(variant='mean',threshold=.7,max_distance=2.,unavailable_policy='appearance_only',
             identity_configuration=dict(max_idle_seconds='1',min_support_rounds=3,min_support_seconds='1/5',max_evidence_gap='1/10'))
    matrices={c:np.eye(3) for c in (0,4,8)}
    old=IdentityStage('fixture',matrices,'floor',**cfg)
    new=FeatureIdentityStage('fixture',matrices,'floor',space=space,**cfg)
    large=FeatureIdentityStage('wide',matrices,'floor',space=wide,**cfg)
    merges=0
    for frame in range(120):
        t=Fraction(frame,30);keys=[];records=[];vectors=[]
        # First allocate separate identities, then sustained common evidence permits a merge.
        for c in (0,4,8):
            if 50<=frame<=85 or (c==4 and frame in (22,23)):continue
            k=ObservationKey(c,0,frame);keys.append(k)
            vector=np.zeros(512,np.float32);vector[(c//4) if frame<3 else 0]=1;vectors.append(vector)
            records.append(CropRecord(k,.8,(0.,0.,2.,4.),(0,0,2,4),1.))
        # Unencoded positive-box observation outside the image survives as a singleton.
        if frame==10:
            records.append(CropRecord(ObservationKey(0,9,frame),.2,(20.,0.,22.,4.),None,0.))
        x=np.stack(vectors) if vectors else np.empty((0,512),np.float32)
        legacy=old.update(frame,t,ReIDBatch(tuple(keys),(t,)*len(keys),x),tuple(records))
        current=new.update(frame,t,FeatureBatch('fixture',space,tuple(keys),(t,)*len(keys),x),tuple(records))
        same_stages(legacy,current)
        assert pickle.dumps(old.manager)==pickle.dumps(new.manager)
        # Padding exists ONLY in this synthetic fixture to establish dimension handling.
        expanded=np.pad(x,((0,0),(0,768)))
        large_result=large.update(frame,t,FeatureBatch('wide',wide,tuple(keys),(t,)*len(keys),expanded),tuple(records))
        assert [(a.key,a.global_id) for a in current[0].assignments]==[(a.key,a.global_id) for a in large_result[0].assignments]
        merges+=len(current[0].merge_events)
    assert merges>0
    print('120 rounds: legacy pairs/groups/global decisions and manager state EXACT; 1280-D accepts synthetic features: OK')
    frame=120;t=Fraction(4);k=ObservationKey(0,0,frame)
    x=np.zeros((1,512),np.float32);x[0,0]=1
    batch=FeatureBatch('fixture',space,(k,),(t,),x)
    record=CropRecord(k,.8,(0.,0.,2.,4.),(0,0,2,4),1.)
    for bad in (replace(batch,space=other),replace(batch,run_id='other'),replace(batch,embeddings=np.zeros((1,1280),np.float32))):
        before=pickle.dumps(new.manager);rejected(lambda:new.update(frame,t,bad,(record,)))
        assert pickle.dumps(new.manager)==before
    new.update(frame,t,batch,(record,))
    print('Wrong model/run/dimension rejected before identity mutation; corrected round accepted: OK')
    print('Feature association contract: PASSED; synthetic thresholds do not calibrate CLIP-ReID')


if __name__=='__main__':main()
