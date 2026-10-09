"""Known-answer causal reference selection and frozen return-ranking checks."""
import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from probe_clipreid_return import probe, add_offline_labels, normalized_average, new_memory, observe, reference
from mtmc.data.ground_truth import GroundTruth


def main():
    scene=SimpleNamespace(rounds=6,fps=1,camera_ids=(4,5),
        cameras=tuple(SimpleNamespace(camera_id=c,width=100,height=100,homography=np.eye(3)) for c in (4,5)))
    rows=[];caches=[];histories=[];features=[];ground=GroundTruth({},())
    for frame in range(6):
        cameras=[];bindings=[];assignments=[];base=[];samples=[];hrows=[]
        for camera in (4,5):
            present=(camera==4 and frame!=2) or (camera==5 and frame==0)
            gid=0 if camera==4 and frame<2 else 5 if camera==4 else 2
            local=0 if frame<2 else 9;key=dict(camera_id=camera,local_id=local,frame_index=frame)
            box=[0.,20.,20.,60.] if frame==1 else [20.,20.,40.,60.]
            score=.2 if frame==1 else .8
            cameras.append(dict(camera=camera,local_ids=[local] if present else [],xyxy=[box] if present else [],confidence=[score] if present else []))
            ground.slots[frame,camera]={0 if camera==4 else 1:box} if present else {}
            if not present:continue
            index=len(features);vector=np.zeros(1280,np.float32)
            if camera==5:vector[:2]=[.8,.6]
            else:vector[1 if frame in (1,4,5) else 0]=1.
            features.append(vector)
            bindings.append(dict(source_key=key,identity_key=key))
            assignments.append(dict(key=key,global_id=gid,reason='fixture'))
            base.append(dict(key=key,global_id=gid,reason='new_identity' if frame in (0,3) else 'fixture'))
            samples.append(dict(camera=camera,local_id=local,frame_index=frame,timestamp=str(frame),
                source_xyxy=box,confidence=score,crop_xyxy_int=[int(x) for x in box],inside_image_fraction=1.,embedding_row=index))
            hrows.append(dict(source_key=key,identity_key=key,embedding_row=index,source_frames=[frame]))
        live=[0,2] if frame==0 else [0,2] if frame==1 else [0] if frame==2 else [5]
        runtime=dict(assignments=assignments,base_assignments=base,merge_events=[],
            expired_global_ids=[2] if frame==2 else [0] if frame==3 else [],identities=[dict(global_id=g) for g in live])
        rows.append(dict(run_id='fixture',frame_index=frame,timestamp=str(frame),cameras=cameras,segment_bindings=bindings,
            variants=dict(clipreid=dict(identity_runtime=runtime,identity=dict(assignments=assignments)))))
        caches.append(dict(frame_index=frame,timestamp=str(frame),observations=samples))
        histories.append(dict(frame_index=frame,timestamp=str(frame),observations=hrows))
    vectors=np.stack(features)
    with TemporaryDirectory() as directory:
        paths=[Path(directory)/n for n in ('trace.gz','cache.gz','history.gz')]
        for path,values in zip(paths,(rows,caches,histories)):
            with gzip.open(path,'wt') as f:
                for row in values:f.write(json.dumps(row)+'\n')
        with patch('probe_clipreid_return.load_ground_truth',side_effect=AssertionError('GT forbidden in selection')):
            r=probe(scene,*paths,vectors,vectors,run='fixture',old_gid=0,new_gid=5,
                admission=dict(min_confidence=.5,border_fraction=.01))
        assert r['frame_index']==3 and r['old_last_seen_frame']==1 and r['old_expired_frame']==3
        assert r['old_last_observation_by_camera']=={4:1}
        a=r['comparisons']['last_mean']['target'];b=r['comparisons']['confidence_border_gallery']['target']
        assert a['cosine']==0. and a['rank']==2 and a['last_seen_age_seconds']==2.
        assert b['cosine']==1. and b['rank']==1 and b['newest_sample_age_seconds']==3.
        assert r['query_observations'][0]['frame_index']==3
        assert all(s['frame_index']<3 for s in b['source_samples'])
        print('ID zero, true last-seen/expiry, degraded last mean versus earlier gallery and competitor rank: OK')
        print('Future observations do not change birth query; reference selection performs no GT access: OK')
        r=json.loads(json.dumps(r));before=copy.deepcopy(r)
        add_offline_labels(r,paths[0],scene,SimpleNamespace(first_frame=0,last_frame=5,min_iou=.5),ground)
        assert r['query_observations'][0]['diagnostic_unique_gt']==0
        for mode in r['comparisons']:
            assert r['comparisons'][mode]['target']['cosine']==before['comparisons'][mode]['target']['cosine']
            assert r['comparisons'][mode]['target']['rank']==before['comparisons'][mode]['target']['rank']
        print('Offline labels preserve frozen ranks and treat GT zero as known: OK')
    # Quality gates affect storage only; rejection cannot refresh a previous accepted sample.
    memory=new_memory()
    for frame in range(20):
        observe(memory,dict(frame_index=frame,camera_id=4,local_id=0,embedding_row=0,
            confidence_passed=True,border_passed=True))
    assert [s['frame_index'] for s in memory['gallery']['confidence_border_gallery'][4]]==list(range(12,20))
    observe(memory,dict(frame_index=20,camera_id=4,local_id=0,embedding_row=0,confidence_passed=False,border_passed=False))
    assert memory['gallery']['confidence_border_gallery'][4][-1]['frame_index']==19
    assert normalized_average([vectors[0],-vectors[0]]) is None
    try:normalized_average([np.zeros(1280,np.float32)])
    except ValueError:pass
    else:raise AssertionError('Unnormalized features accepted')
    assert reference(new_memory(),'last_mean',vectors,vectors)[2]=='no_samples'
    print('Bounded per-camera sample count, no rejected-sample refresh, cancellation and missing reference: OK')
    print('CLIP return probe checks: PASSED; synthetic cases do not calibrate recovery')


if __name__=='__main__':main()
