"""Paired confirmed recovery on frozen CLIP features; exact disabled control first."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

import numpy as np
from check_feature_identity_replay import inputs_for_round, jsonable
from experiment_clipreid_global import evaluation_view
from evaluate_appearance_global import lifecycle
from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch
from mtmc.reid.osnet import ObservationKey
from mtmc.pipeline.confirmed_return import ConfirmedReturnStage

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('disabled','enabled')


def replay(scene,source,paths,old_vectors,raw,means,space,configuration,out,run):
    cfg=source['configuration'];scope=source['run_id']+'/clipreid'
    stages={n:ConfirmedReturnStage(scope if n=='disabled' else run+'/'+n,scene.matrices,scene.coordinate_space,
        space=space,variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],**cfg['geometry'],
        identity_configuration=cfg['identity'],recovery_enabled=n=='enabled',configuration=configuration,
        image_sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}) for n in VARIANTS}
    offset=observations=0;control_counts=Counter();emitted=set();decision_counts=Counter();reset_counts=Counter();peak=Counter()
    with gzip.open(paths['trace'],'rt') as a,gzip.open(paths['observations'],'rt') as b,gzip.open(paths['history'],'rt') as c, \
            gzip.open(out,'xt') as saved:
        for frame in range(scene.rounds):
            lines=[f.readline() for f in (a,b,c)];require(all(lines),'Truncated source');row,cached,hrow=map(json.loads,lines)
            require(row['run_id']==source['run_id'] and row['frame_index']==frame,'Frozen experiment scope differs')
            # Existing checked join expects the continuity-source envelope; observations and keys are unchanged.
            envelope=dict(frame_index=frame,timestamp=row['timestamp'],variants=dict(enabled=dict(
                cameras=row['cameras'],segment_bindings=row['segment_bindings'])))
            start=offset;records,batch,_,offset=inputs_for_round(scene,envelope,cached,hrow,offset,old_vectors)
            t=Fraction(frame,scene.fps);variants={}
            for n in VARIANTS:
                stage=stages[n]
                latest=FeatureBatch(stage.run_id,space,batch.keys,batch.timestamps,np.asarray(raw[start:offset]))
                averaged=FeatureBatch(stage.run_id,space,batch.keys,batch.timestamps,np.asarray(means[start:offset]))
                result,*_=stage.update_with_raw(frame,t,averaged,latest,records)
                runtime=jsonable(asdict(result));audit=stage.registry.last_audit
                if n=='disabled':
                    require(runtime==row['variants']['clipreid']['identity_runtime'],f'Frozen CLIP control differs at {frame}')
                    require(stage.registry.archive is None and not stage.registry.galleries,'Disabled recovery allocated vectors')
                    lifecycle(runtime,control_counts,emitted)
                else:
                    decision_counts.update(d.outcome for d in audit.archive.decisions)
                    reset_counts.update(reason for _,reason in audit.archive.resets)
                    peak['gallery_vectors']=max(peak['gallery_vectors'],audit.lifecycle['gallery_vectors'])
                    peak['archive_ids']=max(peak['archive_ids'],audit.lifecycle['archived_ids'])
                    peak['pending_returns']=max(peak['pending_returns'],audit.lifecycle['pending_returns'])
                inverse={ObservationKey(**x['identity_key']):x['source_key'] for x in row['segment_bindings']}
                returns=[dict(provisional_id=e['provisional_id'],global_id=e['global_id'],
                    members=[inverse[k] for k in e['members']]) for e in audit.reactivations]
                variants[n]=dict(identity_runtime=runtime,identity=evaluation_view(result,row['segment_bindings'],run+'/'+n),
                    recovery_decisions=jsonable(asdict(audit.archive)) if audit.archive is not None else None,
                    reactivations=returns,lifecycle=audit.lifecycle)
            saved.write(json.dumps(dict(run_id=run,frame_index=frame,timestamp=str(t),cameras=row['cameras'],
                segment_bindings=row['segment_bindings'],variants=variants),allow_nan=False)+'\n')
            observations+=len(records)
            if (frame+1)%600==0 or frame+1==scene.rounds:
                print(f"Replayed {frame+1}/{scene.rounds}; disabled CLIP EXACT; confirmed returns={stages['enabled'].registry.returns}",flush=True)
        require(all(f.readline()=='' for f in (a,b,c)) and offset==len(raw)==len(means),'Incomplete source coverage')
    control_counts['ever_emitted_ids']=len(emitted)
    require(dict(control_counts)==source['summary']['lifecycle']['clipreid'],'Frozen lifecycle differs')
    return dict(rounds=scene.rounds,observations=observations,encoded=offset,disabled_lifecycle=dict(control_counts),
        enabled_lifecycle=stages['enabled'].registry.last_audit.lifecycle,return_decisions=dict(decision_counts),
        evidence_resets=dict(reset_counts),peaks=dict(peak))


def return_diagnostics(trace,scene,spec,ground):
    previous={};events=[];sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for line in stream:
            row=json.loads(line);frame=row['frame_index'];data=row['variants']['enabled'];labels={}
            if spec.first_frame<=frame<=spec.last_frame:
                for c in row['cameras']:
                    keys=tuple(ObservationKey(c['camera'],i,frame) for i in c['local_ids'])
                    _,_,unique,*_=spatial_slot(ground.slots[frame,c['camera']],keys,c['xyxy'],
                        width=sizes[c['camera']][0],height=sizes[c['camera']][1],min_iou=spec.min_iou)
                    labels.update(unique)
            for event in data['reactivations']:
                prior=previous.get(event['global_id']);current=[labels.get(ObservationKey(**k)) for k in event['members']]
                old=prior['labels'] if prior else []
                category=('unresolved' if not old or not current or any(v is None for v in (*old,*current)) else
                    'mixed_previous_or_current' if len(set(old))!=1 or len(set(current))!=1 else
                    'same_last_visible_gt' if old[0]==current[0] else 'different_last_visible_gt')
                events.append(dict(frame_index=frame,category=category,**event,previous_visible_evidence=prior,current_labels=current))
            groups={}
            for a in data['identity']['assignments']:groups.setdefault(a['global_id'],[]).append(a['key'])
            for gid,keys in groups.items():previous[gid]=dict(frame_index=frame,members=keys,labels=[labels.get(ObservationKey(**k)) for k in keys])
    return dict(Counter(e['category'] for e in events)),events


def evaluate(trace, scene, spec, ground, *, run, split):
    """One shared identity assignment per full sequence/window, across all cameras."""
    require(spec.first_frame<split<=spec.last_frame,'Invalid split')
    windows=dict(full=[spec.first_frame,spec.last_frame],first=[spec.first_frame,split-1],second=[split,spec.last_frame])
    counts={n:{w:IdentityCounts() for w in windows} for n in VARIANTS}
    slots={n:{w:[] for w in windows} for n in VARIANTS}
    merges={n:Counter() for n in VARIANTS};sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for frame in range(scene.rounds):
            line=stream.readline();require(bool(line),'Truncated predictions');row=json.loads(line)
            require(row['run_id']==run and row['frame_index']==frame and row['timestamp']==str(Fraction(frame,scene.fps))
                and set(row['variants'])==set(VARIANTS),'Prediction scope differs')
            require([c['camera'] for c in row['cameras']]==list(scene.camera_ids),'Wrong camera coverage')
            within=spec.first_frame<=frame<=spec.last_frame
            spatial=[];unique={};expected=set()
            for camera in row['cameras']:
                c=camera['camera'];keys=tuple(ObservationKey(c,i,frame) for i in camera['local_ids'])
                require(len(keys)==len(set(keys)),'Duplicate local observations');expected.update(keys)
                if within:
                    gt,mask,labels,*_=spatial_slot(ground.slots[frame,c],keys,camera['xyxy'],
                        width=sizes[c][0],height=sizes[c][1],min_iou=spec.min_iou)
                    spatial.append((keys,gt,mask));unique.update(labels)
            for name in VARIANTS:
                identity=row['variants'][name]['identity']
                require(identity['run_id']==run+'/'+name and identity['frame_index']==frame
                    and identity['timestamp']==row['timestamp'],'Wrong identity scope')
                assigned={ObservationKey(**a['key']):a['global_id'] for a in identity['assignments']}
                require(len(assigned)==len(identity['assignments']) and set(assigned)==expected
                    and all(type(g) is int and g>=0 for g in assigned.values())
                    and len({(k.camera_id,g) for k,g in assigned.items()})==len(assigned),'Invalid global coverage/uniqueness')
                for keys,gt,mask in spatial:
                    pred=[assigned[k] for k in keys]
                    for w in ('full','first' if frame<split else 'second'):
                        counts[name][w].update(gt,pred,mask);slots[name][w].append((gt,pred,mask))
                for event in identity['merge_events']:
                    labels=[unique.get(ObservationKey(**k)) for k in event['members']]
                    category=('unannotated_frame' if not within else 'unresolved' if not labels or any(v is None for v in labels)
                        else 'all_visible_members_same_gt' if len(set(labels))==1 else 'different_known_gt')
                    merges[name][category]+=1
        require(stream.readline()=='','Trailing predictions')
    metrics={n:{} for n in VARIANTS};mappings={n:{} for n in VARIANTS}
    for name in VARIANTS:
        for w,bounds in windows.items():
            value,mapping=counts[name][w].result()
            check_reference(value,reference_metrics(slots[name][w]))
            require(value['camera_time_slots']==(bounds[1]-bounds[0]+1)*len(scene.camera_ids),'Dropped camera/time slots')
            metrics[name][w]=value;mappings[name][w]=mapping
        for field in ('camera_time_slots','gt_observations','predicted_observations'):
            require(metrics[name]['first'][field]+metrics[name]['second'][field]==metrics[name]['full'][field],
                'Window denominators differ')
        print(f'{name}: full/window shared metrics and motmetrics VERIFIED',flush=True)
    for w in windows:
        for field in ('camera_time_slots','gt_observations','predicted_observations'):
            require(metrics['disabled'][w][field]==metrics['enabled'][w][field],'Recovery populations differ')
    return dict(global_metrics=metrics,windows=windows,accepted_merge_diagnostics={n:dict(v) for n,v in merges.items()}),mappings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-report',type=Path,required=True)
    args=p.parse_args();inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path);require(digest is None or digest==actual,'Changed input: '+str(path))
        inputs[name]=dict(path=str(path),sha256=actual);return path
    rp=checked('experiment_report',args.experiment_report);source=json.loads(rp.read_text())
    require(source.get('completed') is True and source['protocol']=='clipreid_fixed_tracks_global_v1'
        and source['checks'] and all(source['checks'].values()),'Unverified frozen CLIP experiment')
    # Verify the frozen causal dependency chain, excluding raw GT until phase 2.
    for name,item in source['inputs'].items():
        if name!='evaluation_ground_truth':checked('source:'+name,item['path'],item['sha256'])
    loaded=load_scene(inputs['source:history:scene_config']['path'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==source['scene'] and scene.rounds==3600 and scene.fps==30
        and (spec.first_frame,spec.last_frame)==(2,3599),'Unexpected scene/interval')
    require(source['inputs']['evaluation_ground_truth']['sha256']==spec.ground_truth.sha256,'Different evaluation GT')
    config_path=checked('configuration',ROOT/'configs/association/clipreid_confirmed_return.json')
    config=json.loads(config_path.read_text());space=FeatureSpace(**source['spaces']['clipreid'])
    item=source['artifacts']['global_tracks.jsonl.gz'];trace=checked('frozen_trace',rp.parent/item['path'],item['sha256'])
    paths=dict(trace=trace,observations=Path(inputs['source:history:observations.jsonl.gz']['path']),
        history=Path(inputs['source:artifact:history_rows.jsonl.gz']['path']))
    old=np.load(inputs['source:history:source_vectors']['path'],mmap_mode='r',allow_pickle=False)
    raw=np.load(inputs['source:history:clipreid_embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    means=np.load(inputs['source:artifact:mean_embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    for array in (raw,means):require(array.dtype==np.float32 and array.shape==(source['summary']['encoded'],1280),'Invalid CLIP matrix')
    for rel in ('scripts/experiment_confirmed_clip_return.py','scripts/check_confirmed_return.py','scripts/check_confirmed_return_registry.py',
        'src/mtmc/association/confirmed_return.py','src/mtmc/pipeline/confirmed_return.py','src/mtmc/association/dormant.py','src/mtmc/reid/sample_quality.py'):
        checked('code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/confirmed_clip_return'/run;out.mkdir(parents=True,exist_ok=False)
    status=out/'run_status.json';status.write_text('{"completed":false}\n');output=out/'global_tracks.jsonl.gz'
    try:
        print('Phase 1: frozen tracks/CLIP features; two causal registries; no GT, models or decoding...',flush=True)
        summary=replay(scene,source,paths,old,raw,means,space,config,output,run)
        require(summary['observations']==source['summary']['observations'] and summary['encoded']==source['summary']['encoded'],'Population differs')
        digest=sha256(output);freeze=out/'prediction_freeze.json'
        freeze.write_text(json.dumps(dict(run_id=run,sha256=digest,configuration=config),indent=2)+'\n')
        print('Phase 2: outputs frozen; offline full/window quality and return diagnostics...',flush=True)
        ground=load_ground_truth(spec);checked('evaluation_ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mappings=evaluate(output,scene,spec,ground,run=run,split=1800)
        require(quality['global_metrics']['disabled']==source['global_metrics']['clipreid'],'Frozen CLIP metrics differ')
        require(quality['accepted_merge_diagnostics']['disabled']==source['accepted_merge_diagnostics']['clipreid'],'Control merge diagnostics differ')
        categories,events=return_diagnostics(output,scene,spec,ground)
        require(len(events)==summary['enabled_lifecycle']['reactivation_events'],'Return event coverage differs')
        require(sha256(output)==digest,'Predictions changed during evaluation')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during experiment')
        matching=out/'identity_matching.json';matching.write_text(json.dumps(mappings,indent=2)+'\n')
        event_path=out/'reactivation_diagnostics.json';event_path.write_text(json.dumps(events,indent=2)+'\n')
        report=dict(completed=True,protocol='confirmed_clip_return_paired_v1',run_id=run,scene=scene.scene,source_run_id=source['run_id'],
            inputs=inputs,configuration=config,space=asdict(space),summary=summary,**quality,return_gt_diagnostics=categories,
            checks=dict(disabled_full_records_exact=True,disabled_metrics_lifecycle_exact=True,
                common_observations=True,full_window_motmetrics_verified=True,predictions_frozen_before_gt=True,inputs_unchanged=True),
            artifacts={f.name:dict(path=f.name,sha256=sha256(f)) for f in (output,freeze,matching,event_path)},
            limits=['One retrospective development hypothesis; no threshold sweep or production setting selected.',
                'Return diagnostics compare last visible GT labels, not whole-gallery purity.',
                'Local tracking, segment cuts and cross-camera association thresholds are frozen; later global decisions can change after a return.',
                'Previously emitted provisional IDs remain in historical output; superseded IDs are never reused.',
                'Native motion units are not assumed to be meters; scene_041 is previously inspected development evidence.'])
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');status.write_text('{"completed":true}\n')
    except Exception as e:
        status.write_text(json.dumps(dict(completed=False,error=str(e)))+'\n');raise
    for w in ('full','first','second'):
        print('Window:',w)
        for n in VARIANTS:
            m=quality['global_metrics'][n][w]
            print(f"  {n:8} IDF1={100*m['idf1']:.2f}% IDTP={m['idtp']} IDFP={m['idfp']} IDFN={m['idfn']}")
    print('Enabled lifecycle:',json.dumps(summary['enabled_lifecycle']))
    print('Return decisions:',json.dumps(summary['return_decisions']))
    print('Evidence resets:',json.dumps(summary['evidence_resets']))
    print('Return GT diagnostics:',json.dumps(categories))
    print('Report:',out/'report.json')
    print('Confirmed CLIP return experiment: COMPLETED; runtime baseline unchanged; no deployment setting selected')


if __name__=='__main__':main()
