"""Known-answer paired global replay and offline evaluation checks."""
from dataclasses import asdict
from fractions import Fraction
from types import SimpleNamespace
from pathlib import Path
import tempfile
import gzip
import json
import copy
import numpy as np
from check_feature_identity_replay import jsonable, inputs_for_round
from experiment_clipreid_global import replay, evaluate
from mtmc.data.ground_truth import GroundTruth
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.reid.feature_history import FeatureSpace
from mtmc.pipeline.core import IdentityStage

def main():
    cfg=dict(appearance_variant='mean',appearance_threshold=.7,
     geometry=dict(max_distance=2.,unavailable_policy='appearance_only'),
     identity=dict(max_idle_seconds='1',min_support_rounds=3,min_support_seconds='1/5',max_evidence_gap='1/10'))
    scene=SimpleNamespace(rounds=50,fps=30,camera_ids=(4,5,8),
     cameras=tuple(SimpleNamespace(camera_id=c,width=100,height=100) for c in (4,5,8)),
     matrices={c:np.eye(3) for c in (4,5,8)},coordinate_space='floor')
    source={'run_id':'fixture'}; scope='fixture/enabled'
    history=AppearanceHistory(scope)
    stage=IdentityStage(scope,scene.matrices,scene.coordinate_space,variant='mean',threshold=.7,
     **cfg['geometry'],identity_configuration=cfg['identity'])
    traces=[];caches=[];histories=[];vectors=[];means=[]
    for frame in range(scene.rounds):
     t=Fraction(frame,30); cameras=[];bindings=[];records=[];keys=[];xs=[];obs=[];hs=[]
     for c in scene.camera_ids:
      row=dict(camera=c,local_ids=[],xyxy=[],confidence=[],detection_indices=[],embedding_rows=[])
      cameras.append(row)
      if 15<=frame<49 or (c==5 and frame==4):continue
      raw=[110.,0.,112.,4.] if c==8 and frame==8 else [0.,0.,2.,4.]
      bounds,fraction=crop_geometry(raw,100,100); key=ObservationKey(c,0,frame)
      segment=ObservationKey(c,(1<<31) if c==4 and frame>=10 else 0,frame)
      index=len(vectors) if bounds is not None else None
      row['local_ids'].append(0);row['xyxy'].append(raw);row['confidence'].append(.8)
      row['detection_indices'].append(2);row['embedding_rows'].append(index)
      bindings.append(dict(source_key=asdict(key),identity_key=asdict(segment)))
      records.append(CropRecord(segment,.8,tuple(raw),bounds,fraction))
      obs.append(dict(camera=c,local_id=0,frame_index=frame,timestamp=str(t),source_xyxy=raw,
       confidence=.8,detection_index=2,source_embedding_row=index,crop_xyxy_int=list(bounds) if bounds else None,
       inside_image_fraction=fraction,embedding_row=index))
      hs.append(dict(source_key=asdict(key),identity_key=asdict(segment),embedding_row=index,source_frames=[],source_times=[]))
      if bounds is not None:
       x=np.zeros(512,np.float32);x[0]=1;vectors.append(x);keys.append(segment);xs.append(x)
     batch=ReIDBatch(tuple(keys),(t,)*len(keys),np.stack(xs) if xs else np.empty((0,512),np.float32))
     h=history.update(frame,t,batch); p=0
     for item in hs:
      if item['embedding_row'] is not None:
       item['source_frames']=list(h.source_frames[p]);item['source_times']=[str(v) for v in h.source_times[p]]
       means.append(np.pad(h.mean.embeddings[p],(0,768)));p+=1
     identity=stage.update(frame,t,h.mean,tuple(records))[0]
     traces.append(dict(run_id='fixture',frame_index=frame,timestamp=str(t),variants=dict(enabled=dict(
      cameras=cameras,segment_bindings=bindings,identity_runtime=jsonable(asdict(identity))))))
     caches.append(dict(frame_index=frame,timestamp=str(t),observations=obs))
     histories.append(dict(frame_index=frame,timestamp=str(t),observations=hs))
    vectors=np.stack(vectors);means=np.stack(means)
    with tempfile.TemporaryDirectory() as d:
     paths=[Path(d)/n for n in ('trace.gz','cache.gz','history.gz')]
     for path,rows in zip(paths,(traces,caches,histories)):
      with gzip.open(path,'wt') as f:
       for row in rows:f.write(json.dumps(row)+'\n')
     inputs=dict(zip(('trace','observations','history'),paths))
     spaces=dict(osnet=FeatureSpace('a'*64,512),clipreid=FeatureSpace('b'*64,1280))
     output=Path(d)/'equal.gz'
     counts=replay(scene,source,inputs,vectors,means,spaces,cfg,output,'test')
     assert counts['lifecycle']['osnet']==counts['lifecycle']['clipreid']
     with gzip.open(output,'rt') as f:
      for line in f:
       row=json.loads(line)
       a=row['variants']['osnet']['identity_runtime'];b=row['variants']['clipreid']['identity_runtime']
       assert {**a,'run_id':b['run_id']}==b
     print('Equivalent synthetic features: full 512/1280-D global parity, cuts, gaps and outside observations: OK')
     ground=GroundTruth({},())
     for row in traces:
      for c in row['variants']['enabled']['cameras']:
       f=row['frame_index'];camera=c['camera']
       ground.slots[f,camera]={0:c['xyxy'][0]} if c['local_ids'] and not (f==8 and camera==8) else {}
     spec=SimpleNamespace(first_frame=0,last_frame=49,min_iou=.5)
     quality,_=evaluate(output,scene,spec,ground,run='test',split=25)
     assert quality['global_metrics']['osnet']==quality['global_metrics']['clipreid']
     assert quality['global_metrics']['osnet']['full']['predicted_observations']==len(vectors)+1
     print('Offline full/window metrics, common denominators and empty GT slots: motmetrics agreement OK')
     changed=means.copy()
     for row in caches:
      for obs in row['observations']:
       if obs['camera']==5 and obs['embedding_row'] is not None:
        changed[obs['embedding_row']]=0;changed[obs['embedding_row'],1]=1
     altered=Path(d)/'changed.gz'
     replay(scene,source,inputs,vectors,changed,spaces,cfg,altered,'test')
     different,_=evaluate(altered,scene,spec,ground,run='test',split=25)
     assert different['global_metrics']['osnet']==quality['global_metrics']['osnet']
     assert different['global_metrics']['clipreid']['full']['idf1']<quality['global_metrics']['clipreid']['full']['idf1']
     print('Only CLIP features changed: OSNet control fixed; known cross-camera mismatch lowers CLIP IDF1: OK')
     bad=copy.deepcopy(caches[0]);bad['observations'][0]['source_embedding_row']=1
     try:inputs_for_round(scene,traces[0],bad,histories[0],0,vectors)
     except ValueError:pass
     else:raise AssertionError('Changed row mapping accepted')
     broken=means.copy();broken[0,0]=float('nan')
     try:replay(scene,source,inputs,vectors,broken,spaces,cfg,Path(d)/'invalid.gz','test')
     except ValueError:pass
     else:raise AssertionError('Invalid model features accepted')
     print('Changed provenance and invalid CLIP vectors rejected: OK')
    print('CLIP global experiment contract: PASSED; synthetic features do not measure real model quality')


if __name__=='__main__':main()
