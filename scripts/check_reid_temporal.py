"""Known-answer collection, cache parity and temporal retrieval fixtures."""
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
from compare_reid_temporal import collect,label_samples,evaluate_samples,rank_direction
from mtmc.data.scene import RuntimeScene,CameraInput,FileRef,EvaluationSpec
from mtmc.data.ground_truth import GroundTruth


def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);dummy=FileRef(root/'unused','0'*64)
        scene=RuntimeScene('fixture','fixture','0'*40,1,21,
            tuple(CameraInput(c,dummy,w,80,np.eye(3)) for c,w in ((0,100),(12,110),(31,120))),dummy,'fixture')
        selected=[0,10,20];rows=[];batches=[];cache=[];ground={};calls=[]
        for frame in range(21):
            cameras=[];packets=[]
            for c,w in ((0,100),(12,110),(31,120)):
                image=np.zeros((80,w,3),np.uint8)
                image[10:30,0:10]=[1,0,0];image[10:30,30:40]=[2,0,0]
                packets.append(SimpleNamespace(camera_id=c,frame_index=frame,timestamp=Fraction(frame),rgb=image))
                cam=dict(camera=c,local_ids=[],xyxy=[],confidence=[],embedding_rows=[],detection_indices=[])
                known={}
                if c!=31:
                    for person,box in ((0,[0,10,10,30]),(1,[30,10,40,30]),(7,[w+1,10,w+11,30])):
                        row=None
                        if person!=7:
                            row=len(cache);v=np.zeros(512,np.float32);v[person if frame<10 else 1-person]=1;cache.append(v)
                            if not(c==12 and frame==10):known[person]=box
                        for key,value in (('local_ids',person),('xyxy',box),('confidence',.8),('embedding_rows',row),('detection_indices',person)):
                            cam[key].append(value)
                cameras.append(cam);ground[frame,c]=known
            rows.append(dict(run_id='fixture',frame_index=frame,timestamp=str(Fraction(frame)),variants=dict(enabled=dict(cameras=cameras))))
            batches.append(SimpleNamespace(frame_index=frame,timestamp=Fraction(frame),frames=tuple(packets)))
        path=root/'trace.gz'
        with gzip.open(path,'wt') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        def encode(crops):
            calls.append(crops[0].key.frame_index if crops else None)
            keys=tuple(c.key for c in crops);values={name:np.zeros((len(crops),dim),np.float32) for name,dim in (('osnet',512),('clipreid',1280))}
            for i,c in enumerate(crops):
                person=int(c.rgb[0,0,0])-1
                values['osnet'][i,person if c.key.frame_index<10 else 1-person]=1
                values['clipreid'][i,person]=1
            return {name:(keys,value) for name,value in values.items()}
        counts=collect(scene,iter(batches),path,'fixture',np.stack(cache),encode,selected,root)
        assert calls==selected and counts['paired_crops']==12 and counts['fully_outside']==6
        assert counts['source_runtime_observations']==126 and counts['osnet_reference_parity']['max_absolute_error']==0
        records=json.loads((root/'observations.json').read_text())
        assert any(r['local_id']==0 and r['crop_xyxy_int'][0]==0 for r in records)
        assert [r['embedding_row'] for r in records if r['embedding_row'] is not None]==list(range(12))
        spec=EvaluationSpec(dummy,scene.camera_ids,0,20,0,.5,'retain_slot_and_count_predictions')
        labels,matching=label_samples(scene,spec,GroundTruth(ground,()),records,selected)
        features={n:np.load(root/(n+'_embeddings.npy')) for n in ('osnet','clipreid')}
        metrics,transitions=evaluate_samples(labels,features,selected,scene.camera_ids,10,root)
        same=metrics['same_time_cross_camera']
        assert same['osnet']['full']['rank1']==same['clipreid']['full']['rank1']==1
        assert same['osnet']['full']['unmatched_queries']>0 and same['osnet']['full']['no_positive_in_gallery']>0
        assert transitions['plus_10s_cross_camera']['improved']>0 and transitions['plus_10s_same_camera']['improved']>0
        assert metrics['plus_10s_cross_camera']['clipreid']['full']['rank1']==1
        # Temporal row keys and camera IDs can differ; only GT establishes positives.
        q=[dict(camera=0,local_id=0,gt_id=0,embedding_row=0)]
        gallery=[dict(camera=12,local_id=99,gt_id=None,embedding_row=1),dict(camera=12,local_id=7,gt_id=0,embedding_row=2)]
        vector=np.array([[1,0],[1,0],[0,1]],np.float32)
        ranked,_,_=rank_direction(vector,q,gallery,'fixture',0,10,0,12)
        assert ranked[0]['positive_rank']==2 and ranked[0]['status']=='evaluated'
        empty,_,_=rank_direction(vector,q,[],'fixture',0,10,0,12)
        assert empty[0]['status']=='no_positive_in_gallery'
        # A key permutation or corrupted reference must stop collection.
        def bad_encode(crops):
            result=encode(crops);k,v=result['clipreid'];result['clipreid']=(k[::-1],v);return result
        for supplied,encoder in ((np.stack(cache),bad_encode),(np.roll(np.stack(cache),1,axis=1),encode)):
            try:collect(scene,iter(batches),path,'fixture',supplied,encoder,selected,root)
            except ValueError:pass
            else:raise AssertionError('Broken feature provenance accepted')
    print('Fixed sampling, exact crop/key/row mapping, reference parity, GT zero, empty cameras/GT, outside crops, temporal positives and changed mapping rejection: PASSED')
    print('Synthetic encoder fixtures establish protocol behavior, not model quality.')


if __name__=='__main__':main()
