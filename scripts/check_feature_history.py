"""Model-space isolation and exact compatibility with the frozen OSNet history."""
from dataclasses import replace
from fractions import Fraction
import pickle
import numpy as np
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch, FeatureHistory


def require(ok, message):
    if not ok: raise AssertionError(message)


def compare_history(old, new):
    for name in ('latest', 'mean'):
        a, b = getattr(old, name), getattr(new, name)
        require(a.keys == b.keys and a.timestamps == b.timestamps
                and np.array_equal(a.embeddings, b.embeddings), 'History vectors/key mapping differ')
    for name in ('run_id', 'source_frames', 'source_times', 'mean_norms_before_normalization', 'used_latest_fallback'):
        require(getattr(old, name) == getattr(new, name), 'History provenance differs: '+name)


def main():
    rng = np.random.default_rng(17)
    osnet = FeatureSpace('a'*64,512); clip = FeatureSpace('b'*64,1280)
    legacy = AppearanceHistory('fixture'); modern = FeatureHistory('fixture',osnet)
    for frame in range(180):
        ids = [] if frame in (25,26,99) else [(0,0),(17,4)] if frame%3 else [(17,4)]
        keys = tuple(ObservationKey(c,i,frame) for c,i in ids); time = Fraction(frame,30)
        vectors = rng.normal(size=(len(keys),512)).astype(np.float32)
        vectors /= np.linalg.norm(vectors,axis=1,keepdims=True)
        old = legacy.update(frame,time,ReIDBatch(keys,(time,)*len(keys),vectors))
        new = modern.update(frame,time,FeatureBatch('fixture',osnet,keys,(time,)*len(keys),vectors))
        compare_history(old,new)
        require((legacy.active_tracks,legacy.stored_vectors)==(modern.active_tracks,modern.stored_vectors),'State counts differ')
    print('512-D history: vectors, keys, provenance and state counts match frozen implementation EXACTLY: OK')
    def batch(frame, space=clip, person=0, sign=1):
        v=np.zeros((1,space.dimension),np.float32);v[0,0]=sign
        return FeatureBatch('fixture',space,(ObservationKey(0,person,frame),),(Fraction(frame,30),),v)
    owner=FeatureHistory('fixture',clip)
    owner.update(0,Fraction(0),batch(0))
    result=owner.update(30,Fraction(1),batch(30,sign=-1))
    require(result.source_frames==((0,30),) and result.used_latest_fallback==(True,), 'Inclusive age/cancellation differs')
    result=owner.update(31,Fraction(31,30),batch(31,person=1))
    require(result.source_frames==((31,),),'New segment inherited old history')
    empty=FeatureBatch('fixture',clip,(),(),np.empty((0,1280),np.float32))
    result=owner.update(62,Fraction(62,30),empty)
    require(owner.stored_vectors==0 and result.mean.embeddings.shape==(0,1280),'Empty round did not expire evidence')
    print('1280-D history: inclusive age, cancellation, new segment, gaps and empty-round expiry: OK')
    owner=FeatureHistory('fixture',clip); good=batch(0)
    for bad in (replace(good,run_id='other'),replace(good,space=FeatureSpace('c'*64,1280)),
                replace(good,space=osnet),replace(good,embeddings=np.ones((1,512),np.float32)),
                replace(good,embeddings=np.full((1,1280),np.nan,np.float32)),
                replace(good,keys=good.keys*2,timestamps=good.timestamps*2,embeddings=np.repeat(good.embeddings,2,axis=0))):
        before=pickle.dumps(owner)
        try:owner.update(0,Fraction(0),bad)
        except ValueError:pass
        else:raise AssertionError('Malformed/mixed-space input accepted')
        require(pickle.dumps(owner)==before,'Rejected round mutated history')
    result=owner.update(0,Fraction(0),good);good.embeddings[:]=0;result.mean.embeddings[:]=0;result.latest.embeddings[:]=0
    result=owner.update(1,Fraction(1,30),batch(1))
    require(result.mean.embeddings[0,0]==1,'History aliases caller arrays')
    print('Mixed model/run/dimension and invalid vectors rejected before mutation; corrected round and owned storage: OK')
    a=FeatureHistory('fixture',clip);b=FeatureHistory('fixture',clip)
    for frame in range(12):
        first=batch(frame);second=batch(frame,person=7,sign=-1)
        x=FeatureBatch('fixture',clip,first.keys+second.keys,first.timestamps+second.timestamps,
                       np.concatenate((first.embeddings,second.embeddings)))
        y=replace(x,keys=x.keys[::-1],timestamps=x.timestamps[::-1],embeddings=x.embeddings[::-1])
        left=a.update(frame,Fraction(frame,30),x);right=b.update(frame,Fraction(frame,30),y)
        require(np.array_equal(left.mean.embeddings,right.mean.embeddings[::-1])
                and left.source_frames==right.source_frames[::-1],'Input order changed keyed output')
    print('Camera/local-ID isolation and order-invariant keyed histories: OK')
    check_frozen_replay()
    print('Feature history contract: PASSED; global association and quality evaluation pending')


def check_frozen_replay():
    import gzip
    import json
    from pathlib import Path
    import tempfile
    from types import SimpleNamespace
    from replay_feature_history import replay, segment_keys
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);trace=root/'trace.gz';mapping=root/'observations.gz'
        rows=[];observations=[]
        for frame in range(4):
            available=frame<3
            key=dict(camera_id=0,local_id=0,frame_index=frame)
            effective=dict(key,local_id=0 if frame<2 else (1<<31))
            bindings=[dict(source_key=key,identity_key=effective)] if available else []
            rows.append(dict(run_id='original',frame_index=frame,timestamp=str(Fraction(frame,30)),
                             variants=dict(enabled=dict(segment_bindings=bindings))))
            records=[dict(camera=0,local_id=0,frame_index=frame,embedding_row=frame,source_embedding_row=frame)] if available else []
            observations.append(dict(frame_index=frame,timestamp=str(Fraction(frame,30)),observations=records))
        for path,values in ((trace,rows),(mapping,observations)):
            with gzip.open(path,'wt') as stream:
                for row in values:stream.write(json.dumps(row)+'\n')
        old=np.zeros((3,512),np.float32);old[0,0]=1;old[1:,1]=1
        clip=np.zeros((3,1280),np.float32);clip[0,4]=1;clip[1:,5]=1
        summary=replay(SimpleNamespace(rounds=4,fps=30),trace,'original',mapping,old,clip,
                       FeatureSpace('a'*64,512),FeatureSpace('b'*64,1280),root,'fixture',
                       dict(max_observations=8,max_age=Fraction(1)))
        means=np.load(root/'mean_embeddings.npy')
        require(summary['osnet_exact_rows']==3 and summary['observations']==3,'Replay population differs')
        require(np.array_equal(means[0],clip[0]) and np.array_equal(means[2],clip[2]),'Frozen cut did not start fresh history')
        require(np.isclose(means[1,4],2**-.5) and np.isclose(means[1,5],2**-.5),'Wrong causal mean')
        with gzip.open(root/'history_rows.jsonl.gz','rt') as stream:result=[json.loads(line) for line in stream]
        require(result[1]['observations'][0]['source_frames']==[0,1]
                and result[2]['observations'][0]['source_frames']==[2]
                and result[3]['observations']==[],'Persisted segment history provenance differs')
        bad=dict(rows[0]);bad['variants']=dict(enabled=dict(segment_bindings=[]))
        try:segment_keys(bad,observations[0]['observations'])
        except ValueError:pass
        else:raise AssertionError('Missing segment binding accepted')
    print('Frozen source-to-segment join, CPU replay, fresh cut history and persisted CLIP means: OK')


if __name__=='__main__':main()
