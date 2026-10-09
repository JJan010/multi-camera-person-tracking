"""Offline identity transitions and merge evidence in frozen paired model outputs."""
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

from diagnose_continuity_transfer import mapping_summary, identity_status
from evaluate_global_identity import IdentityCounts
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.reid.osnet import ObservationKey

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('osnet','clipreid')


def merge_category(labels, within):
    if not within:return 'unannotated_frame'
    if not labels or any(v is None for v in labels):return 'unresolved'
    return 'all_visible_members_same_gt' if len(set(labels))==1 else 'different_known_gt'


def inspect(trace,scene,spec,ground,*,run,split,target_gt):
    counters={n:{w:IdentityCounts() for w in ('full','first','second')} for n in VARIANTS}
    previous={n:{} for n in VARIANTS};states={n:{} for n in VARIANTS};ended={n:{} for n in VARIANTS}
    recent={n:deque() for n in VARIANTS};transitions={n:[] for n in VARIANTS}
    timelines={n:[] for n in VARIANTS};merges={n:[] for n in VARIANTS};lifecycle={n:Counter() for n in VARIANTS}
    evidence={n:defaultdict(Counter) for n in VARIANTS}
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for frame in range(scene.rounds):
            line=stream.readline();require(bool(line),'Truncated trace');row=json.loads(line)
            require(row['run_id']==run and row['frame_index']==frame and row['timestamp']==str(Fraction(frame,scene.fps))
                and set(row['variants'])==set(VARIANTS),'Trace scope differs')
            require([c['camera'] for c in row['cameras']]==list(scene.camera_ids),'Camera coverage differs')
            bindings={ObservationKey(**b['source_key']):b for b in row['segment_bindings']}
            require(len(bindings)==len(row['segment_bindings']),'Duplicate segment bindings')
            within=spec.first_frame<=frame<=spec.last_frame
            unique={};spatial=[];raw={}
            for c in row['cameras']:
                camera=c['camera'];keys=tuple(ObservationKey(camera,i,frame) for i in c['local_ids'])
                require(len(keys)==len(set(keys)) and len(keys)==len(c['xyxy'])==len(c['confidence']),'Invalid observations')
                raw.update({k:dict(xyxy=box,confidence=score) for k,box,score in zip(keys,c['xyxy'],c['confidence'])})
                if within:
                    gt,mask,labels,*_=spatial_slot(ground.slots[frame,camera],keys,c['xyxy'],
                        width=sizes[camera][0],height=sizes[camera][1],min_iou=spec.min_iou)
                    unique.update(labels);spatial.append((keys,gt,mask))
            require(set(bindings)==set(raw),'Binding coverage differs')
            for name in VARIANTS:
                data=row['variants'][name];runtime=data['identity_runtime'];view=data['identity']
                require(view['run_id']==run+'/'+name and view['frame_index']==runtime['frame_index']==frame
                    and view['timestamp']==runtime['timestamp']==row['timestamp'],'Identity context differs')
                assigned={ObservationKey(**a['key']):a for a in view['assignments']}
                require(len(assigned)==len(view['assignments']) and set(assigned)==set(raw),'Identity coverage differs')
                internal={ObservationKey(**a['key']):a['global_id'] for a in runtime['assignments']}
                require(len(internal)==len(runtime['assignments'])==len(assigned),'Internal coverage differs')
                for key,a in assigned.items():
                    require(internal.get(ObservationKey(**bindings[key]['identity_key']))==a['global_id'],'Source/segment global ID differs')
                for keys,gt,mask in spatial:
                    gids=[assigned[k]['global_id'] for k in keys]
                    for w in ('full','first' if frame<split else 'second'):counters[name][w].update(gt,gids,mask)
                current={r['global_id']:r for r in runtime['identities']}
                before=states[name];prior_ended=dict(ended[name]);events=[]
                for gid in sorted({a['global_id'] for a in runtime['base_assignments'] if a['reason']=='new_identity'}):
                    events.append(dict(kind='allocated',global_ids=[gid]));lifecycle[name]['allocated']+=1
                for gid in runtime['expired_global_ids']:
                    ended[name][gid]=dict(status='expired',frame_index=frame)
                    events.append(dict(kind='expired',global_ids=[gid]));lifecycle[name]['expired']+=1
                for event in runtime['merge_events']:
                    for gid in event['absorbed_global_ids']:
                        ended[name][gid]=dict(status='absorbed',frame_index=frame,canonical_global_id=event['canonical_global_id'])
                    events.append(dict(kind='merge',global_ids=[event['canonical_global_id'],*event['absorbed_global_ids']]))
                    lifecycle[name]['merge']+=1
                # Previous one second only. Current observations are not inserted until after merge context collection.
                while recent[name] and recent[name][0][0]<frame-scene.fps:recent[name].popleft()
                frame_evidence=[];target_rows=[]
                for key in sorted(assigned,key=lambda k:(k.camera_id,k.local_id)):
                    label=unique.get(key);a=assigned[key];gid=a['global_id'];binding=bindings[key]
                    item=dict(frame_index=frame,camera_id=key.camera_id,local_id=key.local_id,
                        segment_id=binding['identity_key']['local_id'],global_id=gid,unique_gt=label,
                        assignment_reason=a['reason'],**raw[key])
                    frame_evidence.append(item)
                    if label is not None:evidence[name][gid][label]+=1
                    if label!=target_gt:continue
                    target_rows.append(item);prior=previous[name].get(key.camera_id)
                    if prior and prior['global_id']!=gid:
                        transitions[name].append(dict(previous=prior,current=item,gap_frames=frame-prior['frame_index'],
                            same_original_local_id=prior['local_id']==key.local_id,
                            same_segment_id=prior['segment_id']==item['segment_id'],
                            old_id_previous_round=identity_status(prior['global_id'],before,prior_ended),
                            old_id_after_current_round=identity_status(prior['global_id'],current,ended[name]),
                            same_round_events=[e for e in events if prior['global_id'] in e['global_ids'] or gid in e['global_ids']]))
                    previous[name][key.camera_id]=item
                if target_rows:timelines[name].append(dict(frame_index=frame,observations=target_rows))
                require(len(view['merge_events'])==len(runtime['merge_events']),'Merge view coverage differs')
                for event,full in zip(view['merge_events'],runtime['merge_events']):
                    ids=[event['canonical_global_id'],*event['absorbed_global_ids']]
                    require(ids==[full['canonical_global_id'],*full['absorbed_global_ids']],'Merge view IDs differ')
                    labels=[unique.get(ObservationKey(**k)) for k in event['members']]
                    category=merge_category(labels,within)
                    members=[dict(key=k,unique_gt=unique.get(ObservationKey(**k)),**raw[ObservationKey(**k)]) for k in event['members']]
                    prior_counts={gid:Counter() for gid in ids};unknown=Counter()
                    for _,observations in recent[name]:
                        for item in observations:
                            gid=item['global_id']
                            if gid not in prior_counts:continue
                            if item['unique_gt'] is None:unknown[gid]+=1
                            else:prior_counts[gid][item['unique_gt']]+=1
                    merges[name].append(dict(frame_index=frame,category=category,event=event,
                        support_rounds=full['support_rounds'],first_support_time=full['first_support_time'],
                        retained_membership_before=full['retained_membership_before'],members=members,
                        prior_window_frames=[max(0,frame-scene.fps),frame-1],
                        prior_gid_label_counts={g:dict(v) for g,v in prior_counts.items()},
                        prior_gid_unknown_counts={g:unknown[g] for g in ids}))
                recent[name].append((frame,frame_evidence));states[name]=current
            if (frame+1)%600==0 or frame+1==scene.rounds:print(f'Inspected {frame+1}/{scene.rounds}',flush=True)
        require(stream.readline()=='','Trailing trace')
    summaries={}
    for name in VARIANTS:
        target_ids=Counter(o['global_id'] for row in timelines[name] for o in row['observations'])
        summaries[name]=dict(**mapping_summary(counters[name]),lifecycle_event_counts=dict(lifecycle[name]),
            accepted_merge_diagnostics=dict(Counter(e['category'] for e in merges[name])),
            target_gt=target_gt,target_evidence_observations=sum(target_ids.values()),
            target_evidence_by_global_id=dict(target_ids),target_evidence_transitions=len(transitions[name]),
            previous_id_status_counts=dict(Counter(t['old_id_after_current_round']['status'] for t in transitions[name])),
            target_global_id_purity_evidence={g:dict(evidence[name][g]) for g in sorted(target_ids)})
    return summaries,timelines,transitions,merges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-report',type=Path,required=True)
    parser.add_argument('--gt-id',type=int,default=15)
    args=parser.parse_args();require(args.gt_id>=0,'Negative GT ID');inputs={}
    def checked(name,path,digest=None):
        p=Path(path).resolve();actual=sha256(p);require(digest is None or digest==actual,'Changed input: '+str(p))
        inputs[name]=dict(path=str(p),sha256=actual);return p
    rp=checked('experiment_report',args.experiment_report);report=json.loads(rp.read_text())
    require(report.get('completed') is True and report['protocol']=='clipreid_fixed_tracks_global_v1'
        and report['checks'] and all(report['checks'].values()),'Unverified experiment')
    item=report['inputs']['history:scene_config'];loaded=load_scene(checked('scene_config',item['path'],item['sha256']),project_root=ROOT)
    scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==report['scene'] and report['windows']['full']==[spec.first_frame,spec.last_frame],'Mixed scene/window')
    require(report['inputs']['evaluation_ground_truth']['sha256']==spec.ground_truth.sha256,'Mixed GT')
    split=report['windows']['second'][0]
    require(report['windows']['first']==[spec.first_frame,split-1] and report['windows']['second']==[split,spec.last_frame],
        'Wrong window split')
    item=report['artifacts']['global_tracks.jsonl.gz'];trace=checked('trace',rp.parent/item['path'],item['sha256'])
    checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
    for rel in ('scripts/diagnose_clipreid_global.py','scripts/check_clipreid_global_diagnostic.py',
        'scripts/diagnose_continuity_transfer.py','scripts/evaluate_global_identity.py','src/mtmc/data/ground_truth.py','src/mtmc/data/scene.py'):
        checked('code:'+rel,ROOT/rel)
    print(f'Frozen model comparison: GT {args.gt_id} transitions and all merge context; offline GT only...',flush=True)
    summaries,timelines,transitions,merges=inspect(trace,scene,spec,load_ground_truth(spec),run=report['run_id'],split=split,target_gt=args.gt_id)
    for name in VARIANTS:
        result=summaries[name]
        require(result['metrics']==report['global_metrics'][name],'Full/window metrics differ')
        require(result['accepted_merge_diagnostics']==report['accepted_merge_diagnostics'][name],'Merge labels differ')
        counts=result['lifecycle_event_counts'];original=report['summary']['lifecycle'][name]
        for key,field in [('allocated','allocated_ids'),('expired','expired_ids'),('merge','merge_events')]:
            require(counts.get(key,0)==original[field],'Lifecycle differs: '+field)
    for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during diagnostic')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/clipreid_global_diagnostic'/run
    out.mkdir(parents=True,exist_ok=False);artifacts={}
    for filename,data in [('target_timeline.json',timelines),('target_transitions.json',transitions),('merge_context.json',merges)]:
        p=out/filename;p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');artifacts[filename]=dict(path=filename,sha256=sha256(p))
    result=dict(completed=True,protocol='clipreid_global_context_v1',run_id=run,experiment_run_id=report['run_id'],
        scene=scene.scene,target_gt=args.gt_id,inputs=inputs,summary=summaries,artifacts=artifacts,
        checks=dict(metrics_reproduced=True,merge_labels_reproduced=True,lifecycle_reproduced=True,predictions_unchanged=True),
        limits=['Mutually unique IoU evidence is not a visual identity verification; unknown intervals are not inferred.',
            'Evidence transitions may reflect a merge, fragmentation, mixing or gaps; they are not CLEAR IDSW.',
            'Numeric GIDs are model/run-specific. Prior merge context covers only one second and does not prove lifetime purity.',
            'Offline GT selects the diagnostic target only; no runtime policy or thresholds change.'])
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name in VARIANTS:
        s=summaries[name];print('\nMODEL:',name)
        print('Separate-minus-shared IDTP:',s['separate_minus_shared_idtp'])
        print('Target GT evidence by GID:',s['target_evidence_by_global_id'])
        print('Previous ID states at evidence transitions:',s['previous_id_status_counts'])
        for t in transitions[name][:16]:
            p,c=t['previous'],t['current']
            print(f"  camera {c['camera_id']}: frame {p['frame_index']} G{p['global_id']} -> frame {c['frame_index']} G{c['global_id']}; "
                f"gap={t['gap_frames']}; old={t['old_id_after_current_round']['status']}; same_local={t['same_original_local_id']}; same_segment={t['same_segment_id']}")
        for event in merges[name]:
            if event['category']=='different_known_gt':print('WRONG VISIBLE-MEMBER MERGE:',json.dumps(event))
    print('Report:',out/'report.json')
    print('Model global diagnostic: COMPLETED; metrics/lifecycle/merge labels reproduced; assignments unchanged')


if __name__=='__main__':main()
