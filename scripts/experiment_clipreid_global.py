"""Fixed-track OSNet/CLIP cross-camera comparison; one declared threshold transfer."""
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
from evaluate_appearance_global import lifecycle
from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.reid.osnet import ObservationKey
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch, FeatureHistory
from mtmc.pipeline.feature_identity import FeatureIdentityStage

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('osnet','clipreid')
PROTOCOL='clipreid_fixed_tracks_global_v1'


def evaluation_view(identity, bindings, scope):
    inverse={ObservationKey(**b['identity_key']):b['source_key'] for b in bindings}
    require(len(inverse)==len(bindings),'Duplicate segment binding')
    return dict(run_id=scope,frame_index=identity.frame_index,timestamp=str(identity.timestamp),
        assignments=[dict(key=inverse[a.key],global_id=a.global_id,reason=a.reason) for a in identity.assignments],
        merge_events=[dict(canonical_global_id=e.canonical_global_id,absorbed_global_ids=list(e.absorbed_global_ids),
            frame_index=e.frame_index,timestamp=str(e.timestamp),members=[inverse[k] for k in e.members])
            for e in identity.merge_events])


def replay(scene, source, paths, vectors, clip_means, spaces, cfg, output, run):
    """No GT argument or loading; source segment cuts remain frozen for both models."""
    scopes={'osnet':source['run_id']+'/enabled','clipreid':run+'/clipreid'}
    stages={name:FeatureIdentityStage(scopes[name],scene.matrices,scene.coordinate_space,space=spaces[name],
        variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],**cfg['geometry'],
        identity_configuration=cfg['identity']) for name in VARIANTS}
    history=FeatureHistory(scopes['osnet'],spaces['osnet'],max_observations=8,max_age=Fraction(1))
    counts={n:Counter() for n in VARIANTS};emitted={n:set() for n in VARIANTS}
    offset=observations=0
    with gzip.open(paths['trace'],'rt') as a,gzip.open(paths['observations'],'rt') as b, \
            gzip.open(paths['history'],'rt') as c,gzip.open(output,'xt') as saved:
        for frame in range(scene.rounds):
            lines=[f.readline() for f in (a,b,c)];require(all(lines),'Truncated source')
            row,cached,hrow=map(json.loads,lines)
            require(row['run_id']==source['run_id'] and row['frame_index']==frame,'Wrong source scope')
            previous=offset
            records,batch,provenance,offset=inputs_for_round(scene,row,cached,hrow,offset,vectors)
            t=Fraction(frame,scene.fps)
            h=history.update(frame,t,FeatureBatch(scopes['osnet'],spaces['osnet'],batch.keys,batch.timestamps,batch.embeddings))
            for i,item in enumerate(provenance):
                require(item['source_frames']==list(h.source_frames[i])
                    and item['source_times']==[str(x) for x in h.source_times[i]],'Stored history participation differs')
            descriptors={'osnet':h.mean,'clipreid':FeatureBatch(scopes['clipreid'],spaces['clipreid'],batch.keys,
                batch.timestamps,np.asarray(clip_means[previous:offset]))}
            original=row['variants']['enabled'];variants={}
            for name in VARIANTS:
                identity,*_=stages[name].update(frame,t,descriptors[name],records)
                runtime=jsonable(asdict(identity))
                if name=='osnet':
                    require(runtime==original['identity_runtime'],f'OSNet full frozen control differs at {frame}')
                lifecycle(runtime,counts[name],emitted[name])
                # The full internal record retains segment keys; evaluation uses original observation keys.
                variants[name]=dict(identity_runtime=runtime,
                    identity=evaluation_view(identity,original['segment_bindings'],run+'/'+name))
            saved.write(json.dumps(dict(run_id=run,frame_index=frame,timestamp=str(t),
                cameras=original['cameras'],segment_bindings=original['segment_bindings'],variants=variants),allow_nan=False)+'\n')
            observations+=len(records)
            if (frame+1)%600==0 or frame+1==scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; OSNet full control EXACT; CLIP global assignments recorded',flush=True)
        require(all(f.readline()=='' for f in (a,b,c)),'Trailing source frames')
    require(offset==len(clip_means),'Unconsumed CLIP rows')
    for name in VARIANTS:counts[name]['ever_emitted_ids']=len(emitted[name])
    return dict(rounds=scene.rounds,observations=observations,encoded=offset,
                lifecycle={n:dict(v) for n,v in counts.items()})


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
            require(metrics['osnet'][w][field]==metrics['clipreid'][w][field],'Model populations differ')
    return dict(global_metrics=metrics,windows=windows,accepted_merge_diagnostics={n:dict(v) for n,v in merges.items()}),mappings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compatibility-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,digest=None):
        p=Path(path).resolve();actual=sha256(p);require(digest is None or digest==actual,'Changed input: '+str(p))
        inputs[name]=dict(path=str(p),sha256=actual);return p
    gate_path=checked('compatibility_report',args.compatibility_report);gate=json.loads(gate_path.read_text())
    require(gate.get('completed') is True and gate['protocol']=='feature_identity_osnet_parity_v1'
        and gate['checks'] and all(gate['checks'].values()),'Unverified compatibility gate')
    for name,item in gate['inputs'].items():checked(name,item['path'],item['sha256'])
    source=json.loads(Path(inputs['history:source_report']['path']).read_text())
    loaded=load_scene(inputs['history:scene_config']['path'],project_root=ROOT);scene=loaded.runtime;spec=loaded.evaluation
    require(scene.scene==gate['scene']==source['scene'] and source['run_id']==gate['source_run_id'],'Mixed lineage')
    require(scene.rounds==3600 and scene.fps==30 and (spec.first_frame,spec.last_frame)==(2,3599),'Unexpected experiment interval')
    cfg=gate['configuration']
    require(cfg['appearance_variant']=='mean' and cfg['appearance_threshold']==.7
        and cfg['history']['max_observations']==8 and Fraction(cfg['history']['max_age_seconds'])==1,
        'Expected one fixed mean/0.7 transfer experiment')
    spaces={n:FeatureSpace(**gate['spaces'][n]) for n in VARIANTS}
    require(spaces['osnet'].dimension==512 and spaces['clipreid'].dimension==1280,'Unexpected dimensions')
    vectors=np.load(inputs['history:source_vectors']['path'],mmap_mode='r',allow_pickle=False)
    clip=np.load(inputs['artifact:mean_embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(clip.dtype==np.float32 and clip.shape==(gate['summary']['encoded'],1280),'Invalid CLIP means')
    paths=dict(trace=Path(inputs['history:source_tracks']['path']),observations=Path(inputs['history:observations.jsonl.gz']['path']),
        history=Path(inputs['artifact:history_rows.jsonl.gz']['path']))
    # Fingerprint new code and reused metric helpers; prior gate already pins runtime dependencies.
    for rel in ('scripts/experiment_clipreid_global.py','scripts/check_clipreid_global_experiment.py',
        'scripts/evaluate_appearance_global.py','scripts/evaluate_global_identity.py','src/mtmc/data/ground_truth.py'):
        checked('experiment_code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/clipreid_global_experiment'/run
    out.mkdir(parents=True,exist_ok=False);status=out/'run_status.json';status.write_text('{"completed": false}\n')
    trace=out/'global_tracks.jsonl.gz'
    try:
        print('Phase 1: CPU replay; frozen local tracks and segment cuts; OSNet/CLIP means; no GT or models...',flush=True)
        summary=replay(scene,source,paths,vectors,clip,spaces,cfg,trace,run)
        require(summary['observations']==gate['summary']['observations'] and summary['encoded']==gate['summary']['encoded'],
            'Source populations differ')
        frozen=sha256(trace)
        (out/'prediction_freeze.json').write_text(json.dumps(dict(run_id=run,sha256=frozen,configuration=cfg,
            threshold_selection=False,local_tracking_and_segment_cuts='frozen_osnet',spaces=gate['spaces']),indent=2)+'\n')
        print('Phase 2: predictions frozen; offline GT evaluation...',flush=True)
        ground=load_ground_truth(spec);checked('evaluation_ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mappings=evaluate(trace,scene,spec,ground,run=run,split=1800)
        require(quality['global_metrics']['osnet']==source['global']['enabled'],'Frozen OSNet full/window metrics differ')
        require(summary['lifecycle']['osnet']==source['summary']['lifecycle']['enabled'],'Frozen OSNet lifecycle differs')
        for name in VARIANTS:
            require(sum(quality['accepted_merge_diagnostics'][name].values())==summary['lifecycle'][name]['merge_events'],
                'Merge diagnostic coverage differs')
        require(sha256(trace)==frozen,'Predictions changed during evaluation')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during experiment')
        matching=out/'identity_matching.json';matching.write_text(json.dumps(mappings,indent=2)+'\n')
        result=dict(completed=True,protocol=PROTOCOL,run_id=run,scene=scene.scene,source_run_id=source['run_id'],
            inputs=inputs,configuration=cfg,spaces=gate['spaces'],summary=summary,**quality,
            checks=dict(frozen_osnet_records_exact=True,osnet_metrics_lifecycle_reproduced=True,
                common_observations_and_segments=True,full_and_window_motmetrics_verified=True,
                predictions_frozen_before_gt=True,inputs_unchanged=True),
            artifacts={p.name:dict(path=p.name,sha256=sha256(p)) for p in (trace,out/'prediction_freeze.json',matching)},
            limits=['Only cross-camera descriptors change; local tracker and continuity cuts still depend on frozen OSNet.',
                'Mean/0.7 is an unchanged numerical threshold transfer, not calibrated CLIP similarity.',
                'No threshold sweep, archive recovery, model selection, runtime replacement or speed claim.',
                'Scene 001 is development data; scene 041 has already been inspected and is not an untouched final test.'])
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps(dict(completed=True,run_id=run))+'\n')
    except Exception as e:
        status.write_text(json.dumps(dict(completed=False,run_id=run,error=str(e)))+'\n');raise
    for window in ('full','first','second'):
        print('Window:',window)
        for name in VARIANTS:
            m=quality['global_metrics'][name][window]
            print(f"  {name:8} IDF1={100*m['idf1']:.2f}% IDTP={m['idtp']} IDFP={m['idfp']} IDFN={m['idfn']}")
        delta=100*(quality['global_metrics']['clipreid'][window]['idf1']-quality['global_metrics']['osnet'][window]['idf1'])
        print(f'  Delta CLIP/OSNet: {delta:+.2f} pp')
    print('Lifecycle:',json.dumps(summary['lifecycle']))
    print('Accepted merges:',json.dumps(quality['accepted_merge_diagnostics']))
    print('Report:',out/'report.json')
    print('CLIP fixed-track global experiment: COMPLETED; runtime unchanged; no threshold selected')


if __name__=='__main__':main()
