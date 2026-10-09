"""Disabled parity, delayed whole-registry return and transactional failure fixtures."""
from dataclasses import asdict
from fractions import Fraction as F
from pathlib import Path
import json
import pickle
import numpy as np
from mtmc.reid.osnet import ObservationKey
from mtmc.reid.feature_history import FeatureBatch, FeatureSpace
from mtmc.reid.crops import CropRecord
from mtmc.pipeline.feature_identity import FeatureIdentityStage
from mtmc.pipeline.confirmed_return import ConfirmedReturnStage

ROOT=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((ROOT/'configs/association/clipreid_confirmed_return.json').read_text())
    space=FeatureSpace('a'*64,1280);matrices={4:np.eye(3),5:np.eye(3)}
    kwargs=dict(space=space,variant='mean',threshold=.7,max_distance=2.,unavailable_policy='appearance_only',
        identity_configuration=dict(max_idle_seconds='1',min_support_rounds=3,min_support_seconds='1/5',max_evidence_gap='1/10'))
    old=FeatureIdentityStage('run',matrices,'native',**kwargs)
    disabled=ConfirmedReturnStage('run',matrices,'native',**kwargs,recovery_enabled=False,configuration=config,image_sizes={4:(100,100),5:(100,100)})
    enabled=ConfirmedReturnStage('run',matrices,'native',**kwargs,recovery_enabled=True,configuration=config,image_sizes={4:(100,100),5:(100,100)})
    stored=[];returns=[];good=None
    for frame in range(150):
        t=F(frame,30);keys=();records=();x=np.empty((0,1280),np.float32)
        if frame<=20 or 91<=frame<=99:
            camera,local=(4,0) if frame<=20 else (5,9)
            key=ObservationKey(camera,local,frame);keys=(key,)
            box=(0.,20.,20.,60.) if frame==91 else (20.,20.,40.,60.)
            records=(CropRecord(key,.8,box,tuple(map(int,box)),1.),)
            x=np.zeros((1,1280),np.float32);x[0,0]=1
        batch=FeatureBatch('run',space,keys,(t,)*len(keys),x)
        reference=old.update(frame,t,batch,records)
        unchanged=disabled.update_with_raw(frame,t,batch,batch,records)
        actual=enabled.update_with_raw(frame,t,batch,batch,records)
        assert reference[:5]==unchanged[:5]
        assert pickle.dumps(old.manager)==pickle.dumps(disabled.registry.base)
        assert disabled.registry.archive is None and not disabled.registry.galleries
        stored.append(asdict(actual[0]))
        for event in enabled.registry.last_audit.reactivations:returns.append((frame,event))
        if frame==91:
            assert enabled.registry.last_audit.archive.decisions[0].outcome=='query_quality_rejected'
        if frame==98:
            good=enabled.registry.last_audit.lifecycle
            assert actual[0].assignments[0].global_id==1
            assert all(b.global_id==1 for b in enabled.registry.base._bindings.values())
    assert [f for f,_ in returns]==[98]
    assert returns[0][1]['global_id']==1 and returns[0][1]['provisional_id']==2
    assert all(stored[f]['assignments'][0]['global_id']==2 for f in range(91,98))
    assert good['allocated_id_slots']==2 and good['superseded_ids']==1 and good['reactivation_events']==1
    assert good['retained_ids']==1 and good['currently_inactive_ids']==0
    assert enabled.registry.archive.retained_ids==(1,)
    assert enabled.registry.last_audit.lifecycle['expiration_events']==2
    print('150 rounds: recovery disabled reproduces all baseline outputs/state EXACT; no recovery vectors stored: OK')
    print('Weak first crop retries; confirmed return remaps all bindings now and later; past emitted provisional IDs stay unchanged: OK')
    print('Allocation/absorption/expiry/return accounting and later retirement of recovered identity: OK')
    # Force a failure after base advancement inside the transaction. No partial base/archive state may commit.
    broken=ConfirmedReturnStage('run',matrices,'native',**kwargs,recovery_enabled=True,configuration=config,image_sizes={4:(100,100),5:(100,100)})
    broken.registry.configuration['max_live_gallery_identities']=0
    k=ObservationKey(4,0,0);x=np.zeros((1,1280),np.float32);x[0,0]=1
    batch=FeatureBatch('run',space,(k,),(F(0),),x);record=CropRecord(k,.8,(20.,20.,40.,60.),(20,20,40,60),1.)
    before=pickle.dumps(broken.registry)
    try:broken.update_with_raw(0,F(0),batch,batch,(record,))
    except ValueError:pass
    else:raise AssertionError('Injected failure absent')
    assert pickle.dumps(broken.registry)==before and broken.failed
    try:broken.update_with_raw(0,F(0),batch,batch,(record,))
    except ValueError:pass
    else:raise AssertionError('Failed stage accepted retry')
    print('Failure after base advancement commits neither registry nor archive; failed pipeline cannot retry: OK')
    print('Confirmed return registry contract: PASSED; full frozen replay and quality evaluation pending')


if __name__=='__main__':main()
