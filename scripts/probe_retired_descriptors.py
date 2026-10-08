"""Reconstruct frozen identity snapshots; rank retired descriptors without recovery."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
import gzip,json
from pathlib import Path
import numpy as np
from mtmc.data.scene import load_scene,require,sha256
from mtmc.data.candidates import validate_cache_round
from mtmc.pipeline.core import IdentityStage
from mtmc.pipeline.recovery import RecoveryIdentityStage
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey
from validate_appearance_continuity import selected_observations
from run_mtmc import dump

ROOT=Path(__file__).resolve().parents[1]
# Offline case selection from the prior diagnostic; never used by the runtime.
TARGETS={81:44,89:44,96:44,108:4,127:4,165:96}


def comparison(snapshot,query,now,settings):
    desc,time,point,ground_time=query
    ages={'last_seen':float(now-snapshot.last_seen),
          'descriptor':float(now-snapshot.descriptor_time) if snapshot.descriptor_time is not None else None,
          'ground':float(now-snapshot.ground_time) if snapshot.ground_time is not None else None}
    cosine=float(np.dot(desc.astype(np.float64),snapshot.descriptor.astype(np.float64))) if desc is not None and snapshot.descriptor is not None else None
    distance=float(np.linalg.norm(np.array(point)-snapshot.ground_xy)) if point is not None and snapshot.ground_xy is not None else None
    limit=settings['position_slack']+settings['max_speed']*ages['ground'] if ages['ground'] is not None else None
    return {'global_id':snapshot.global_id,'cosine':cosine,'ground_distance':distance,'distance_limit':limit,
        'source_age_seconds':ages,'archive_age_gate':all(a is not None and a<=float(Fraction(settings['max_age_seconds'])) for a in ages.values()),
        'appearance_gate':cosine is not None and cosine>settings['min_similarity'],
        'geometry_gate':distance is not None and distance<=limit,
        'query_descriptor_age_seconds':float(now-time) if time is not None else None,
        'query_age_gate':time is not None and now-time<=Fraction(settings['max_age_seconds'])}


def replay(scene,policy,trace,candidates,vectors,cache_run,run_id,settings,targets):
    cfg=policy['global'];scope=run_id+'/enabled'
    kwargs=dict(variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],**cfg['geometry'],identity_configuration=cfg['identity'])
    stage=IdentityStage(scope,scene.matrices,scene.coordinate_space,**kwargs)
    helper=RecoveryIdentityStage(scope,scene.matrices,scene.coordinate_space,**kwargs,recovery_enabled=False,
        history_max_observations=cfg['history']['max_observations'],history_max_age=Fraction(cfg['history']['max_age_seconds']))
    history=AppearanceHistory(scope,max_observations=cfg['history']['max_observations'],max_age=Fraction(cfg['history']['max_age_seconds']))
    registry=helper.registry;retired={};provenance={};retired_meta={};queries=[];next_row=0;births=set();snapshots=Counter()
    with gzip.open(trace,'rt') as source,Path(candidates).open() as candidate_stream:
        for frame in range(scene.rounds):
            a,b=source.readline(),candidate_stream.readline();require(a and b,'Truncated frozen input')
            saved,cached=json.loads(a),json.loads(b);now=Fraction(frame,scene.fps)
            require(saved['run_id']==run_id and saved['frame_index']==frame and Fraction(saved['timestamp'])==now,'Mixed frozen scope')
            next_row,_,_=validate_cache_round(cached,scene,vectors,run_id=cache_run,frame=frame,row_offset=next_row)
            data=saved['variants']['enabled'];records,features=selected_observations(scene,data['cameras'],cached,vectors,frame)
            bindings={ObservationKey(**x['source_key']):ObservationKey(**x['identity_key']) for x in data['segment_bindings']}
            require(len(bindings)==len(records) and set(bindings)=={r.key for r in records},'Segment source mapping differs')
            records=tuple(replace(r,key=bindings[r.key]) for r in records)
            features=replace(features,keys=tuple(bindings[k] for k in features.keys))
            h=history.update(frame,now,features)
            identity,groups,*_=stage.update(frame,now,h.mean,records)
            require(json.loads(dump(asdict(identity)))==data['identity_runtime'],'Frozen enabled runtime mismatch at '+str(frame))
            ev=helper.evidence_from_history(frame,now,h,records);rows={e.key:e for e in ev.observations}
            for gid in identity.expired_global_ids:
                require(gid in registry._snapshots,'Expired identity missing snapshot')
                retired[gid]=registry._snapshots[gid]
                retired_meta[gid]={'retired_frame':frame,'last_seen':str(retired[gid].last_seen),'reference':provenance.get(gid)}
            base={a.key:a for a in identity.base_assignments}
            for group in groups.groups:
                gids={base[k].global_id for k in group}
                eligible=len(gids)==1 and all(base[k].reason=='new_identity' for k in group)
                if not eligible:continue
                gid=next(iter(gids));births.add(gid)
                if gid not in targets:continue
                query=registry._aggregate(group,rows,now)
                pairs=[comparison(s,query,now,settings) for s in retired.values()]
                pairs.sort(key=lambda x:(x['cosine'] is None,-(x['cosine'] if x['cosine'] is not None else -2),x['global_id']))
                for rank,pair in enumerate(pairs,1):pair['appearance_rank_ignoring_gates']=rank
                wanted=targets[gid];selected=next((p for p in pairs if p['global_id']==wanted),None)
                item={'frame_index':frame,'new_global_id':gid,'target_old_global_id':wanted,
                    'source_members':[asdict(next(k for k,v in bindings.items() if v==key)) for key in group],
                    'target_comparison':selected,'target_retirement':retired_meta.get(wanted),
                    'top5_retired_appearance_ignoring_age_geometry':pairs[:5],
                    'all_pair_gates_passed_count':sum(p['archive_age_gate'] and p['query_age_gate'] and p['appearance_gate'] and p['geometry_gate'] for p in pairs)}
                queries.append(item)
            decisions=[];registry._update_snapshots(identity,groups.groups,rows,decisions)
            inverse={v:k for k,v in bindings.items()};indexes={k:i for i,k in enumerate(h.mean.keys)}
            for gid,status in decisions:
                snapshots[status]+=1
                if status=='snapshot_updated':
                    state=next(s for s in identity.identities if s.global_id==gid)
                    provenance[gid]={'updated_frame':frame,'members':[{'source_key':asdict(inverse[k]),
                        'identity_key':asdict(k),'history_frames':list(h.source_frames[indexes[k]]),
                        'used_latest_fallback':h.used_latest_fallback[indexes[k]]} for k in state.current_members]}
                elif gid not in registry._snapshots or registry._snapshots[gid].descriptor is None:provenance.pop(gid,None)
            provenance={g:v for g,v in provenance.items() if g in registry._snapshots}
            if (frame+1)%600==0 or frame+1==scene.rounds:print(f'Replayed {frame+1}/{scene.rounds}; enabled state EXACT; probed births={len(queries)}',flush=True)
        require(source.readline()==candidate_stream.readline()=='','Trailing frozen input')
    require(next_row==len(vectors),'Feature row coverage differs')
    return {'queries':queries,'targets_without_unanchored_birth':sorted(set(targets)-{q['new_global_id'] for q in queries}),
            'retirements':len(retired),'snapshot_decisions':dict(snapshots)}


def self_check():
    from mtmc.association.dormant import InactiveIdentity
    vector=np.zeros(512,np.float32);vector[0]=1
    snapshot=InactiveIdentity(0,Fraction(0),vector,Fraction(0),(0.,0.),Fraction(0))
    settings={'max_age_seconds':'5','min_similarity':.8,'position_slack':2.,'max_speed':0.}
    query=(vector,Fraction(5),(2.,0.),Fraction(5))
    pair=comparison(snapshot,query,Fraction(5),settings)
    require(pair['archive_age_gate'] and pair['appearance_gate'] and pair['geometry_gate'],'Inclusive age/distance differs')
    require(not comparison(snapshot,query,Fraction(151,30),settings)['archive_age_gate'],'Stale reference accepted')
    require(not comparison(snapshot,query,Fraction(5),{**settings,'min_similarity':1.})['appearance_gate'],'Similarity boundary not strict')
    missing=replace(snapshot,descriptor=None,descriptor_time=None,ground_xy=None,ground_time=None)
    pair=comparison(missing,query,Fraction(5),settings)
    require(pair['cosine'] is None and not any(pair[k] for k in ('archive_age_gate','appearance_gate','geometry_gate')),'Missing evidence accepted')
    print('ID zero, inclusive age/distance, strict appearance, expired and missing references: PASSED')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--validation-report',type=Path);p.add_argument('--self-check',action='store_true');args=p.parse_args();inputs={}
    if args.self_check:self_check();return
    require(args.validation_report is not None,'Provide --validation-report')
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path);require(digest is None or digest==actual,'Changed input: '+str(path));inputs[name]={'path':str(path),'sha256':actual};return path
    rp=checked('validation_report',args.validation_report);report=json.loads(rp.read_text())
    require(report.get('completed') is True and report['protocol']=='appearance_continuity_scene_transfer_v1' and all(report['checks'].values()),'Unverified transfer')
    for name in ('validation:scene_config','validation_cache:detections.jsonl','validation_cache:embeddings.npy','validation:candidate_report'):
        item=report['inputs'][name];checked(name,item['path'],item['sha256'])
    for name in ('global_tracks.jsonl.gz',):
        item=report['artifacts'][name];checked(name,rp.parent/item['path'],item['sha256'])
    scene=load_scene(inputs['validation:scene_config']['path'],project_root=ROOT).runtime
    require(scene.scene==report['scene']=='MTMC_Tracking_2024/val/scene_041' and scene.rounds==3600,'Unexpected probe scene')
    cache=json.loads(Path(inputs['validation:candidate_report']['path']).read_text());policy=report['configuration']['paired_policy']
    config=json.loads(checked('archive_configuration',ROOT/'configs/association/dormant_recovery_experiment.json').read_text());settings=config['archive']
    require(settings=={'max_age_seconds':'5','min_similarity':.8,'min_margin':.05,'max_speed':0.,'position_slack':2.,'max_identities':128},'Archived hypothesis changed')
    for rel in ('scripts/probe_retired_descriptors.py','scripts/validate_appearance_continuity.py','src/mtmc/association/recovery.py','src/mtmc/pipeline/recovery.py','src/mtmc/reid/history.py'):
        checked('code:'+rel,ROOT/rel)
    vectors=np.load(inputs['validation_cache:embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    print('Reconstructing frozen segmented histories and snapshots; CPU; no GT, models or recovery...',flush=True)
    result=replay(scene,policy,inputs['global_tracks.jsonl.gz']['path'],inputs['validation_cache:detections.jsonl']['path'],vectors,cache['run_id'],report['run_id'],settings,TARGETS)
    for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during probe')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/retired_descriptor_probe'/run;out.mkdir(parents=True,exist_ok=False)
    result.update(completed=True,protocol='frozen_retired_descriptor_probe_v1',source_run_id=report['run_id'],inputs=inputs,archive_settings=settings,
        limits=['Offline selected cases, not a retrieval benchmark or threshold selection.',
        'Retired descriptors deliberately retained without age/capacity eviction for diagnosis; rankings are not live archive decisions.',
        'No mutual margin/group competition or actual reactivation is simulated; pair gates alone do not establish eligibility.',
        'Reference identity purity is not established by an earlier GT association; source history provenance is included.',
        'No models, GT inputs, prediction changes or deployment settings changes.'])
    path=out/'report.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for item in result['queries']:
        print(f"Frame {item['frame_index']}: new GID={item['new_global_id']}, target old GID={item['target_old_global_id']}")
        print('Target:',json.dumps(item['target_comparison']))
        print('Top retired appearance: ID / cosine / age gate / geometry gate')
        for candidate in item['top5_retired_appearance_ignoring_age_geometry']:
            print(candidate['global_id'],candidate['cosine'],candidate['archive_age_gate'],candidate['geometry_gate'])
    print('Targets without an unanchored birth:',result['targets_without_unanchored_birth']);print('Report:',path)
    print('Retired descriptor probe: COMPLETED; frozen enabled runtime EXACT; recovery remains disabled')


if __name__=='__main__':main()
