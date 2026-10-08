"""Offline mapping-gap and lifecycle context; never modify identity predictions."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
import numpy as np
from evaluate_global_identity import IdentityCounts
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.reid.osnet import ObservationKey

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('disabled','enabled')


def mapping_summary(counters):
    results={w:c.result() for w,c in counters.items()}
    maps={w:{r['gt_id']:r for r in rows} for w,(_,rows) in results.items()}
    rows=[]
    for gt in sorted(counters['full'].ground):
        selected={w:maps[w].get(gt,{'gt_id':gt,'global_id':None,'idtp':0}) for w in maps}
        gap=selected['first']['idtp']+selected['second']['idtp']-selected['full']['idtp']
        gids=sorted({r['global_id'] for r in selected.values() if r['global_id'] is not None})
        rows.append({'gt_id':gt,'selected_assignments':selected,'separate_minus_shared_idtp':gap,
            'selected_id_overlap_counts':[{'global_id':g,**{w:c.potential[gt,g] for w,c in counters.items()}} for g in gids]})
    gap=results['first'][0]['idtp']+results['second'][0]['idtp']-results['full'][0]['idtp']
    require(gap>=0 and sum(r['separate_minus_shared_idtp'] for r in rows)==gap,'Mapping gap accounting differs')
    return {'metrics':{w:r[0] for w,r in results.items()},'separate_minus_shared_idtp':gap,
            'mapping_contributions':sorted(rows,key=lambda r:(-r['separate_minus_shared_idtp'],r['gt_id']))}


def identity_status(gid,states,ended):
    if gid in states:return {'status':states[gid]['status'],'last_seen':states[gid]['last_seen']}
    if gid in ended:return ended[gid]
    return {'status':'not_in_recorded_state'}


def analyze(trace,audit,scene,spec,ground,*,run_id,split):
    require(spec.first_frame<split<=spec.last_frame,'Invalid split')
    counters={n:{w:IdentityCounts() for w in ('full','first','second')} for n in VARIANTS}
    previous={n:{} for n in VARIANTS};states={n:{} for n in VARIANTS};ended={n:{} for n in VARIANTS}
    transitions={n:[] for n in VARIANTS};ledger={n:[] for n in VARIANTS};evidence_counts={n:Counter() for n in VARIANTS}
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as source,gzip.open(audit,'rt') as audit_stream:
        for frame in range(scene.rounds):
            a,b=source.readline(),audit_stream.readline();require(a and b,'Truncated trace/audit')
            row,actions=json.loads(a),json.loads(b)
            for r in (row,actions):
                require(r['run_id']==run_id and r['frame_index']==frame
                        and Fraction(r['timestamp'])==Fraction(frame,scene.fps) and set(r['variants'])==set(VARIANTS),'Mixed scope/time')
            require(row['variants']['disabled']['cameras']==row['variants']['enabled']['cameras'],'Frozen local observations differ')
            for name in VARIANTS:
                data=row['variants'][name];runtime=data['identity_runtime'];identity=data['identity']
                require(runtime['run_id']==identity['run_id']==run_id+'/'+name and runtime['frame_index']==identity['frame_index']==frame,'Identity scope differs')
                assignments={ObservationKey(**r['key']):r for r in identity['assignments']}
                bindings={ObservationKey(**r['source_key']):r for r in data['segment_bindings']}
                require(len(assignments)==len(identity['assignments']) and len(bindings)==len(data['segment_bindings']),'Duplicate keys')
                current={r['global_id']:r for r in runtime['identities']}
                before=states[name];prior_ended=dict(ended[name]);events=[]
                allocated=sorted({r['global_id'] for r in runtime['base_assignments'] if r['reason']=='new_identity'})
                for gid in allocated:events.append({'type':'allocated','global_ids':[gid]})
                for gid in runtime['expired_global_ids']:
                    ended[name][gid]={'status':'expired','frame_index':frame}
                    events.append({'type':'expired','global_ids':[gid]})
                for event in runtime['merge_events']:
                    target=event['canonical_global_id']
                    for gid in event['absorbed_global_ids']:ended[name][gid]={'status':'absorbed','frame_index':frame,'canonical_global_id':target}
                    events.append({'type':'merge','global_ids':[target,*event['absorbed_global_ids']],'detail':event})
                splits=actions['variants'][name]['segment_events']
                for event in splits:
                    key=ObservationKey(**event['source_key']);require(key in assignments,'Split source absent')
                    events.append({'type':'split','global_ids':[assignments[key]['global_id']], 'detail':event})
                ledger[name].extend({'frame_index':frame,**event} for event in events)
                require([c['camera'] for c in data['cameras']]==list(scene.camera_ids),'Camera coverage differs')
                seen=set()
                for camera in data['cameras']:
                    c=camera['camera'];keys=tuple(ObservationKey(c,i,frame) for i in camera['local_ids'])
                    require(len(keys)==len(set(keys)) and len(keys)==len(camera['xyxy']),'Invalid local rows')
                    seen.update(keys);gids=[assignments[k]['global_id'] for k in keys]
                    require(len(set(gids))==len(gids),'Same-camera global collision')
                    if not spec.first_frame<=frame<=spec.last_frame:continue
                    gt,mask,unique,_,_=spatial_slot(ground.slots[frame,c],keys,camera['xyxy'],width=sizes[c][0],height=sizes[c][1],min_iou=spec.min_iou)
                    for window in ('full','first' if frame<split else 'second'):counters[name][window].update(gt,gids,mask)
                    for key in keys:
                        label=unique.get(key)
                        evidence_counts[name]['unknown' if label is None else 'mutually_unique']+=1
                        if label is None:continue
                        gid=assignments[key]['global_id'];binding=bindings[key]
                        observation={'frame_index':frame,'camera_id':c,'gt_id':label,'local_id':key.local_id,
                            'global_id':gid,'generation':binding['generation'],'identity_key':binding['identity_key'],
                            'assignment_reason':assignments[key]['reason']}
                        old=previous[name].get((c,label))
                        if old and old['global_id']!=gid:
                            transitions[name].append({'previous':old,'current':observation,'gap_frames':frame-old['frame_index'],
                                'same_original_local_id':old['local_id']==key.local_id,
                                'same_effective_local_id':old['identity_key']['local_id']==binding['identity_key']['local_id'],
                                'old_id_previous_round':identity_status(old['global_id'],before,prior_ended),
                                'old_id_after_current_round':identity_status(old['global_id'],current,ended[name]),
                                'same_round_events':[e for e in events if old['global_id'] in e['global_ids'] or gid in e['global_ids']]})
                        previous[name][c,label]=observation
                require(seen==set(assignments)==set(bindings),'Original-key coverage differs')
                states[name]=current
            if (frame+1)%600==0 or frame+1==scene.rounds:print(f'Inspected {frame+1}/{scene.rounds}',flush=True)
        require(source.readline()==audit_stream.readline()=='','Trailing records')
    summaries={n:{**mapping_summary(counters[n]),'unique_evidence_counts':dict(evidence_counts[n]),
        'evidence_global_id_transitions':len(transitions[n]),'lifecycle_event_counts':dict(Counter(e['type'] for e in ledger[n]))} for n in VARIANTS}
    return summaries,transitions,ledger


def self_check():
    for case in ('stable','fragmented','mixed','empty'):
        counts={w:IdentityCounts() for w in ('full','first','second')}
        for frame in range(6):
            gt=[] if case=='empty' else [0 if case!='mixed' or frame<3 else 1]
            ids=[] if case=='empty' else [5 if case!='fragmented' or frame<3 else 8]
            for w in ('full','first' if frame<3 else 'second'):counts[w].update(gt,ids,np.ones((len(gt),len(ids)),bool))
        summary=mapping_summary(counts)
        require(summary['separate_minus_shared_idtp']==(3 if case in ('fragmented','mixed') else 0),'Known mapping gap differs')
    require(identity_status(0,{0:{'status':'lost','last_seen':'1'}},{})['status']=='lost','Retained ID zero differs')
    require(identity_status(0,{}, {0:{'status':'expired','frame_index':3}})['status']=='expired','Expiry differs')
    require(identity_status(0,{}, {0:{'status':'absorbed','canonical_global_id':2}})['status']=='absorbed','Absorption differs')
    print('Known mapping gaps: stable, fragmentation, mixing, empty; ID zero and lifecycle states: PASSED')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--validation-report',type=Path);parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:self_check();return
    require(args.validation_report is not None,'Provide --validation-report');inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path);require(digest is None or digest==actual,'Changed input: '+str(path));inputs[name]={'path':str(path),'sha256':actual};return path
    report_path=checked('validation_report',args.validation_report);report=json.loads(report_path.read_text())
    require(report.get('completed') is True and report['protocol']=='appearance_continuity_scene_transfer_v1'
        and report['checks'] and all(v is True for v in report['checks'].values()),'Unverified transfer report')
    item=report['inputs']['validation:scene_config'];loaded=load_scene(checked('scene_config',item['path'],item['sha256']),project_root=ROOT)
    scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==report['scene'] and report['windows']['full']==[spec.first_frame,spec.last_frame],'Scene/evaluation differs')
    split=report['windows']['second'][0]
    require(report['windows']['first']==[spec.first_frame,split-1] and report['windows']['second']==[split,spec.last_frame],'Windows differ')
    paths={}
    for name in ('global_tracks.jsonl.gz','continuity_audit.jsonl.gz'):
        item=report['artifacts'][name];paths[name]=checked(name,report_path.parent/item['path'],item['sha256'])
    checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
    require(report['inputs']['validation_ground_truth']['sha256']==spec.ground_truth.sha256,'GT lineage differs')
    for rel in ('scripts/diagnose_continuity_transfer.py','scripts/evaluate_global_identity.py','src/mtmc/data/ground_truth.py','src/mtmc/data/scene.py'):checked('code:'+rel,ROOT/rel)
    print('Frozen mapping/lifecycle diagnostic; GT used offline only; no models or state changes...',flush=True)
    results,transitions,ledger=analyze(paths['global_tracks.jsonl.gz'],paths['continuity_audit.jsonl.gz'],scene,spec,load_ground_truth(spec),run_id=report['run_id'],split=split)
    for name in VARIANTS:
        require(results[name]['metrics']==report['global'][name],'Full/window metrics differ: '+name)
        count=results[name]['lifecycle_event_counts'];life=report['summary']['lifecycle'][name]
        require(count.get('allocated',0)==life['allocated_ids'] and count.get('expired',0)==life['expired_ids'] and count.get('merge',0)==life['merge_events'],'Lifecycle differs')
        require(count.get('split',0)==(report['summary']['split_events'] if name=='enabled' else 0),'Split counts differ')
    for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during diagnosis')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/continuity_transfer_diagnostic'/run;out.mkdir(parents=True,exist_ok=False)
    context={}
    for name in VARIANTS:
        context[name]=[]
        for contribution in results[name]['mapping_contributions'][:8]:
            gt=contribution['gt_id']
            ids={r['global_id'] for r in contribution['selected_assignments'].values() if r['global_id'] is not None}
            context[name].append({'mapping':contribution,
                'evidence_transitions':[t for t in transitions[name] if t['current']['gt_id']==gt],
                'events_for_selected_ids':[e for e in ledger[name] if ids.intersection(e['global_ids'])]})
    artifacts={}
    for filename,data in (('evidence_transitions.json',transitions),('lifecycle_events.json',ledger),('top_mapping_context.json',context)):
        path=out/filename;path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');artifacts[filename]={'path':filename,'sha256':sha256(path)}
    result={'completed':True,'protocol':'continuity_transfer_mapping_context_v1','run_id':run,'source_run_id':report['run_id'],'scene':scene.scene,'inputs':inputs,'variants':results,'artifacts':artifacts,
        'limits':['Optimal matching IDs are not dominant labels; tie choices affect per-person attribution.',
        'All spatially admissible candidates contribute to metrics; only mutually unique matches label evidence transitions.',
        'Evidence transitions may span unknown or absent observations and are not CLEAR IDSW.',
        'Same-round lifecycle events are context, not proof of causation; use the full ledger for intervening events.',
        'Separate-minus-shared IDTP is neither an event count nor a guaranteed recoverable accuracy gain.',
        'No predictions, settings or runtime state were changed.']}
    path=out/'report.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name in VARIANTS:
        r=results[name];print('\nVARIANT:',name);print('Separate-minus-shared IDTP:',r['separate_minus_shared_idtp'])
        print('GT / first matched GID / second matched GID / full matched GID / IDTP difference')
        for item in r['mapping_contributions'][:8]:
            m=item['selected_assignments'];print(item['gt_id'],m['first']['global_id'],m['second']['global_id'],m['full']['global_id'],item['separate_minus_shared_idtp'])
        print('Evidence ID changes by previous ID state:',dict(Counter(t['old_id_after_current_round']['status'] for t in transitions[name])))
        print('Lifecycle events:',r['lifecycle_event_counts'])
    print(f'Report: {path}');print('Continuity transfer diagnostic: COMPLETED; full/window metrics and lifecycle reproduced; predictions unchanged')


if __name__=='__main__':main()
