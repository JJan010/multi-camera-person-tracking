"""Known lifecycle and merge-context answers for the offline model diagnostic."""
import copy
import gzip
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from diagnose_clipreid_global import inspect, merge_category
from mtmc.data.ground_truth import GroundTruth


def main():
    scene=SimpleNamespace(rounds=5,fps=1,camera_ids=(4,5),
        cameras=tuple(SimpleNamespace(camera_id=c,width=100,height=100) for c in (4,5)))
    spec=SimpleNamespace(first_frame=0,last_frame=4,min_iou=.5)
    ground=GroundTruth({},());rows=[]
    for frame in range(5):
        cameras=[];bindings=[];assign=[];internal=[];base=[]
        for camera in (4,5):
            present=camera==5 or frame<3
            local=9 if camera==5 and frame>=3 else 0
            segment=(1<<31) if camera==5 and frame>=3 else local
            gid=(0 if camera==4 or frame==2 else 2 if frame<2 else 3)
            key=dict(camera_id=camera,local_id=local,frame_index=frame)
            effective=dict(camera_id=camera,local_id=segment,frame_index=frame)
            cameras.append(dict(camera=camera,local_ids=[local] if present else [],
                xyxy=[[0.,0.,2.,4.]] if present else [],confidence=[.8] if present else []))
            ground.slots[frame,camera]={camera-4:[0.,0.,2.,4.]} if present else {}
            if not present:continue
            bindings.append(dict(source_key=key,identity_key=effective))
            assign.append(dict(key=key,global_id=gid,reason='fixture'))
            internal.append(dict(key=effective,global_id=gid,reason='fixture'))
            base.append(dict(key=effective,global_id=gid,reason='new_identity' if frame==0 or frame==3 else 'fixture'))
        merge=[];full=[]
        if frame==2:
            merge=[dict(canonical_global_id=0,absorbed_global_ids=[2],frame_index=2,timestamp='2',members=[a['key'] for a in assign])]
            full=[dict(**merge[0],support_rounds=2,first_support_time='1',retained_membership_before=[])]
        ids=([dict(global_id=0,status='visible',last_seen=str(frame)),dict(global_id=2,status='visible',last_seen=str(frame))] if frame<2
             else [dict(global_id=0,status='visible',last_seen='2')] if frame==2
             else [dict(global_id=0,status='lost',last_seen='2'),dict(global_id=3,status='visible',last_seen='3')] if frame==3
             else [dict(global_id=3,status='visible',last_seen='4')])
        variants={}
        for name in ('osnet','clipreid'):
            variants[name]=dict(identity=dict(run_id='fixture/'+name,frame_index=frame,timestamp=str(frame),assignments=assign,merge_events=merge),
                identity_runtime=dict(frame_index=frame,timestamp=str(frame),assignments=internal,
                    base_assignments=base,identities=ids,expired_global_ids=[0] if frame==4 else [],merge_events=full))
        rows.append(dict(run_id='fixture',frame_index=frame,timestamp=str(frame),cameras=cameras,segment_bindings=bindings,variants=variants))
    with TemporaryDirectory() as directory:
        path=Path(directory)/'trace.gz'
        def write(values):
            with gzip.open(path,'wt') as f:
                for row in values:f.write(json.dumps(row)+'\n')
        write(rows)
        summaries,timelines,transitions,merges=inspect(path,scene,spec,ground,run='fixture',split=3,target_gt=1)
        for name in summaries:
            s=summaries[name];t=transitions[name];m=merges[name][0]
            assert s['metrics']['full']['predicted_observations']==8
            assert s['accepted_merge_diagnostics']=={'different_known_gt':1}
            assert [e['old_id_after_current_round']['status'] for e in t]==['absorbed','lost']
            assert t[0]['same_original_local_id'] and t[0]['same_segment_id']
            assert not t[1]['same_original_local_id'] and not t[1]['same_segment_id']
            assert m['prior_gid_label_counts']=={0:{0:1},2:{1:1}}
            assert m['prior_window_frames']==[1,1]
            assert s['lifecycle_event_counts']=={'allocated':3,'merge':1,'expired':1}
        assert summaries['osnet']==summaries['clipreid']
        print('ID zero, known wrong merge, prior-only evidence, absorbed/lost state and changed local/segment: OK')
        changed=copy.deepcopy(ground);changed.slots[2,5]={}
        s,_,t,m=inspect(path,scene,spec,changed,run='fixture',split=3,target_gt=1)
        assert s['osnet']['accepted_merge_diagnostics']=={'unresolved':1}
        assert t['osnet'][0]['gap_frames']==2
        assert m['osnet'][0]['members'][1]['unique_gt'] is None
        assert merge_category([0,0],True)=='all_visible_members_same_gt'
        assert merge_category([0,None],True)=='unresolved'
        print('Empty GT gives unknown evidence; gaps stay explicit; GT zero remains a valid label: OK')
        bad=copy.deepcopy(rows);bad[0]['variants']['osnet']['identity_runtime']['assignments'][0]['global_id']=99
        write(bad)
        try:inspect(path,scene,spec,ground,run='fixture',split=3,target_gt=1)
        except ValueError:pass
        else:raise AssertionError('Incorrect segment/source join accepted')
        print('Mismatched original/internal global-ID mapping rejected: OK')
    print('CLIP global diagnostic checks: PASSED; no model-quality claim')


if __name__=='__main__':main()
