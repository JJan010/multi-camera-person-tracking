"""Synthetic integration checks; no models, videos, GT or quality claims."""
from dataclasses import replace
from fractions import Fraction
import numpy as np

from mtmc.pipeline.core import IdentityStage
from mtmc.pipeline.available_identity import AvailableIdentityStage
from mtmc.reid.crops import CropRecord
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.selective_history import SelectiveAppearanceHistory, confidence_update_mask


def stage(cls=AvailableIdentityStage):
    return cls('synthetic', {c:np.eye(3) for c in (4,5,8)}, 'synthetic-ground',
        variant='mean', threshold=.7, max_distance=2., unavailable_policy='appearance_only',
        identity_configuration={'max_idle_seconds':'1','min_support_rounds':3,
            'min_support_seconds':'1/5','max_evidence_gap':'1/10'})


def observations(frame, members, scores=None):
    keys=tuple(ObservationKey(c,i,frame) for c,i in members)
    vectors=np.zeros((len(keys),512),np.float32);vectors[:,0]=1
    scores=[.9]*len(keys) if scores is None else scores
    records=tuple(CropRecord(k,s,(0.,0.,1.,1.),(0,0,1,1),1.) for k,s in zip(keys,scores))
    return ReIDBatch(keys,(Fraction(frame,30),)*len(keys),vectors),records


def subset(batch, indices):
    return ReIDBatch(tuple(batch.keys[i] for i in indices),tuple(batch.timestamps[i] for i in indices),
                     batch.embeddings[list(indices)].copy())


def assigned(result):
    return {(a.key.camera_id,a.key.local_id):a.global_id for a in result.assignments}


def main():
    old,new=stage(IdentityStage),stage()
    for frame in range(60):
        members=[] if frame in (4,5) else [(4,0),(5,7),(8,3)]
        batch,records=observations(frame,members)
        expected=old.update(frame,Fraction(frame,30),batch,records)
        actual=new.update(frame,Fraction(frame,30),batch,records,unavailable_keys=())
        assert actual[0]==expected[0] and actual[1]==expected[1] and actual[4]==expected[4]
    print('All descriptors available: original full identity records and grouping EXACT over 60 rounds: OK')

    engine=stage();batch,records=observations(0,[(4,0),(5,7)])
    initial=engine.update(0,Fraction(0),batch,records,unavailable_keys=())[0]
    mapping=assigned(initial);assert len(set(mapping.values()))==1
    batch,records=observations(1,[(4,0),(5,7),(8,9)],[.9,.2,.2])
    original_records=records
    result,grouped,*_=engine.update(1,Fraction(1,30),subset(batch,[0]),records,unavailable_keys=batch.keys[1:])
    current=assigned(result)
    assert current[4,0]==current[5,7]==mapping[4,0] and current[8,9]!=mapping[4,0]
    assert all(len(g)==1 for g in grouped.groups) and len(result.assignments)==3
    assert records==original_records and all(r.crop_xyxy_int is not None for r in records)
    assert assigned(initial)==mapping
    print('Unavailable appearance preserves local continuity, all observations and original crop metadata: OK')

    batch,records=observations(0,[(4,0),(5,7)])
    result=stage().update(0,Fraction(0),subset(batch,[]),records,unavailable_keys=batch.keys)[0]
    assert len(set(assigned(result).values()))==2
    print('Two new tracks without descriptors cannot create a cross-camera appearance link: OK')

    history=SelectiveAppearanceHistory('synthetic',max_observations=8,max_age=Fraction(1))
    engine=stage()
    for frame,scores in ((0,[.2,.9]),(1,[.9,.9]),(2,[.2,.2]),(33,[.2,.2])):
        batch,records=observations(frame,[(4,0),(5,7)],scores)
        selected=history.update(frame,Fraction(frame,30),batch,
            accept_update=confidence_update_mask(batch,dict(zip(batch.keys,scores)),minimum_score=.5))
        missing=tuple(k for k in batch.keys if k not in selected.mean.keys)
        output=engine.update(frame,Fraction(frame,30),selected.mean,records,unavailable_keys=missing)[0]
        assert len(output.assignments)==2
        assert len(selected.mean.keys)==(1 if frame==0 else 0 if frame==33 else 2)
        if frame==2: assert all(d.status=='reused' and d.source_frames[-1]==1 for d in selected.decisions)
    batch,records=observations(34,[])
    empty=history.update(34,Fraction(34,30),batch,accept_update={})
    assert not engine.update(34,Fraction(34,30),empty.mean,records,unavailable_keys=())[0].assignments
    print('Selective history integrates weak reuse, expiry and empty rounds without dropping tracks: OK')

    left,right=stage(),stage()
    batch,records=observations(0,[(4,0),(5,7),(8,9)])
    a=left.update(0,Fraction(0),subset(batch,[0,2]),records,unavailable_keys=(batch.keys[1],))[0]
    b=right.update(0,Fraction(0),subset(batch,[2,0]),tuple(reversed(records)),unavailable_keys=(batch.keys[1],))[0]
    assert a==b
    print('Key-based mapping and observation/descriptor order preserve the complete identity result: OK')

    for kind in ('missing','unknown','overlap','outside','bad_time','bad_vector'):
        engine=stage();batch,records=observations(0,[(4,0),(5,7)])
        features=subset(batch,[0]);missing=(batch.keys[1],)
        if kind=='missing': missing=()
        elif kind=='unknown': missing=(ObservationKey(8,99,0),)
        elif kind=='overlap': missing=batch.keys
        elif kind=='outside': records=(replace(records[0],crop_xyxy_int=None,inside_image_fraction=0.),records[1])
        elif kind=='bad_time': features=replace(features,timestamps=(Fraction(1),))
        elif kind=='bad_vector': features=replace(features,embeddings=np.zeros((1,512),np.float32))
        try: engine.update(0,Fraction(0),features,records,unavailable_keys=missing)
        except ValueError: pass
        else: raise AssertionError('Malformed input accepted: '+kind)
        batch,records=observations(0,[(4,0),(5,7)])
        actual=engine.update(0,Fraction(0),batch,records,unavailable_keys=())[0]
        expected=stage().update(0,Fraction(0),batch,records,unavailable_keys=())[0]
        assert actual==expected
    print('Invalid availability partition, outside descriptors, timestamps and vectors rejected before identity mutation: OK')
    print('Selective history global integration contract: PASSED; quality evaluation pending')


if __name__=='__main__':
    main()
