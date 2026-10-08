"""Unchanged-policy transfer to the already inspected scene_041 frozen tracks."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from evaluate_scene_pair import read_variant
from evaluate_appearance_global import direct_observations, lifecycle
from experiment_appearance_continuity import settings_from_configuration, evaluation_view, diagnose_splits
from experiment_dormant_recovery import evaluate
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.candidates import validate_cache_round
from mtmc.data.ground_truth import load_ground_truth
from mtmc.pipeline.core import IdentityStage
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.tracking.continuity import AppearanceContinuity
from mtmc.tracking.segments import SegmentRound
from run_mtmc import dump

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('disabled','enabled')


def selected_observations(scene,cameras,cached,vectors,frame):
    """Join selected source tracks by camera/detection index/embedding row, not IoU."""
    require([c['camera'] for c in cameras]==list(scene.camera_ids),'Camera coverage/order differs')
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    candidates={(d['camera_id'],d['detection_index']):d for d in cached['detections']}
    require(len(candidates)==len(cached['detections']),'Duplicate candidate key')
    records=[];keys=[];values=[]
    for camera in cameras:
        c=camera['camera'];ids=camera['local_ids'];n=len(ids)
        require(all(type(i) is int and i>=0 for i in ids) and len(set(ids))==n
                and all(len(camera[k])==n for k in ('xyxy','confidence','detection_indices','embedding_rows')),
                'Malformed source track arrays')
        indices=camera['detection_indices']
        require(all(type(i) is int and i>=0 for i in indices) and len(set(indices))==n,'Invalid selected candidate indices')
        for p in sorted(range(n),key=lambda p:ids[p]):
            item=candidates[c,indices[p]];row=item['embedding_row']
            require(item['frame_index']==frame and item['xyxy']==camera['xyxy'][p]
                    and item['confidence']==camera['confidence'][p] and row==camera['embedding_rows'][p],
                    'Selected candidate provenance differs')
            bounds,fraction=crop_geometry(item['xyxy'],*sizes[c])
            require((bounds is None)==(row is None),'Crop/embedding availability differs')
            key=ObservationKey(c,ids[p],frame)
            records.append(CropRecord(key,float(item['confidence']),tuple(item['xyxy']),bounds,fraction))
            if row is not None:
                require(type(row) is int and 0<=row<len(vectors),'Invalid selected feature row')
                keys.append(key);values.append(vectors[row])
    matrix=np.stack(values).astype(np.float32,copy=False) if values else np.empty((0,512),np.float32)
    return tuple(records),ReIDBatch(tuple(keys),(Fraction(frame,scene.fps),)*len(keys),matrix)


def check_development_adapter(scene,cache,reference,candidate_path,reference_path,vectors):
    """Check the new generic join against the old scene_001 join on every row."""
    row=observations=0
    with Path(candidate_path).open() as candidates,gzip.open(reference_path,'rt') as frozen:
        for frame in range(scene.rounds):
            a,b=candidates.readline(),frozen.readline();require(a and b,'Truncated development source')
            cached,old=json.loads(a),json.loads(b)
            require(old['run_id']==reference['run_id'] and old['frame_index']==frame
                    and cached['cache_run_id']==cache['run_id'],'Mixed development trace')
            normalized={'cache_run_id':cached['cache_run_id'],'scene':scene.scene,'camera_ids':list(scene.camera_ids),
                        'frame_index':cached['frame_index'],'timestamp':cached['timestamp'],'detections':cached['detections']}
            row,_,_=validate_cache_round(normalized,scene,vectors,run_id=cache['run_id'],frame=frame,row_offset=row)
            cameras=old['variants']['staged']['cameras']
            original,features=direct_observations(cameras,cached,vectors,frame)
            generic,new_features=selected_observations(scene,cameras,normalized,vectors,frame)
            require(original==generic and features.keys==new_features.keys and features.timestamps==new_features.timestamps
                    and np.array_equal(features.embeddings,new_features.embeddings),'Generic join changed development inputs')
            observations+=len(original)
        require(candidates.readline()==frozen.readline()=='','Trailing development source')
    require(row==len(vectors)==cache['summary']['encoded'],'Development row coverage differs')
    print(f'Development adapter parity: {scene.rounds} rounds, {observations} selected observations EXACT',flush=True)
    return {'rounds':scene.rounds,'observations':observations,'candidate_rows':row,'exact':True}


def replay(scene,policy,*,run_id,cache,reference,candidate_path,reference_path,vectors,output,settings):
    cfg=policy['global']
    owners={n:AppearanceContinuity(run_id+'/'+n,scene.camera_ids,enabled=n=='enabled',settings=settings) for n in VARIANTS}
    histories={n:AppearanceHistory(run_id+'/'+n,max_observations=cfg['history']['max_observations'],
                  max_age=Fraction(cfg['history']['max_age_seconds'])) for n in VARIANTS}
    stages={n:IdentityStage(run_id+'/'+n,scene.matrices,scene.coordinate_space,variant=cfg['appearance_variant'],
                threshold=cfg['appearance_threshold'],**cfg['geometry'],identity_configuration=cfg['identity']) for n in VARIANTS}
    counts={n:Counter() for n in VARIANTS};emitted={n:set() for n in VARIANTS}
    decisions=Counter();resets=Counter();peaks=Counter();cuts=observations=row=0
    candidates_count=Counter(detections=0,encoded=0,fully_outside=0)
    candidates_count.update({f'camera_{c}':0 for c in scene.camera_ids})
    with Path(candidate_path).open() as candidate_stream,gzip.open(reference_path,'rt') as old_stream, \
            gzip.open(output/'global_tracks.jsonl.gz','xt') as saved, \
            gzip.open(output/'continuity_audit.jsonl.gz','xt') as audit:
        for frame in range(scene.rounds):
            a,b=candidate_stream.readline(),old_stream.readline();require(a and b,'Truncated validation trace')
            cached,old=json.loads(a),json.loads(b);time=Fraction(frame,scene.fps)
            row,missing,per_camera=validate_cache_round(cached,scene,vectors,run_id=cache['run_id'],frame=frame,row_offset=row)
            previous,_=read_variant(old,scene,frame=frame,run_id=reference['run_id'],cache_run=cache['run_id'],variant='staged')
            cameras=previous['cameras'];records,features=selected_observations(scene,cameras,cached,vectors,frame)
            observations+=len(records);variants={};audits={}
            candidates_count['fully_outside']+=missing
            for c,n in per_camera.items():candidates_count[f'camera_{c}']+=n;candidates_count['detections']+=n
            for name in VARIANTS:
                processed=owners[name].update(SegmentRound(run_id+'/'+name,frame,time,records,features))
                segmented=processed.segmented
                require(segmented.features.timestamps==features.timestamps
                        and np.array_equal(segmented.features.embeddings,features.embeddings)
                        and tuple(segmented.source_key(k) for k in segmented.features.keys)==features.keys,'Feature values/mapping changed')
                require(len(segmented.records)==len(records),'Observation count changed')
                for current,source in zip(segmented.records,records):
                    require(segmented.source_key(current.key)==source.key
                            and {k:v for k,v in asdict(current).items() if k!='key'}
                            =={k:v for k,v in asdict(source).items() if k!='key'},'Source box/score/crop changed')
                if name=='disabled':
                    require(segmented.records==records and segmented.features.keys==features.keys
                            and not segmented.events and not processed.decisions and owners[name].stored_vectors==0,
                            'Disabled continuity owner changed inputs/state')
                history=histories[name].update(frame,time,segmented.features)
                for event in segmented.events:
                    require(history.source_frames[history.mean.keys.index(event.identity_key)]==(frame,),
                            'New segment inherited old appearance history')
                identity,*_=stages[name].update(frame,time,history.mean,segmented.records)
                runtime=json.loads(dump(asdict(identity)))
                if name=='disabled':
                    require(runtime=={**previous['identity'],'run_id':run_id+'/'+name},
                            f'Disabled validation full record differs: frame {frame}')
                lifecycle(runtime,counts[name],emitted[name]);view=evaluation_view(identity,segmented)
                require(len(view['assignments'])==len(records)
                        and {ObservationKey(**a['key']) for a in view['assignments']}=={r.key for r in records},
                        'Evaluation projection lost observations')
                variants[name]={'cameras':cameras,'identity':view,'identity_runtime':runtime,
                                'segment_bindings':[asdict(v) for v in segmented.bindings]}
                audits[name]={'decisions':[asdict(v) for v in processed.decisions],
                              'resets':[asdict(v) for v in processed.resets],
                              'segment_events':[asdict(v) for v in segmented.events]}
                if name=='enabled':
                    decisions.update(d.outcome for d in processed.decisions);resets.update(r.reason for r in processed.resets)
                    cuts+=len(segmented.events)
                    peaks['vectors']=max(peaks['vectors'],owners[name].stored_vectors)
                    peaks['pending_tracks']=max(peaks['pending_tracks'],owners[name].pending_tracks)
                    peaks['original_tracks']=max(peaks['original_tracks'],owners[name].segmenter.known_tracks)
            header={'run_id':run_id,'frame_index':frame,'timestamp':str(time)}
            saved.write(dump({**header,'variants':variants})+'\n');audit.write(dump({**header,'variants':audits})+'\n')
            if (frame+1)%300==0 or frame+1==scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; validation control EXACT; splits={cuts}',flush=True)
        require(candidate_stream.readline()==old_stream.readline()=='','Trailing validation trace')
    require(row==len(vectors)==cache['summary']['encoded'],'Candidate embedding coverage differs')
    candidates_count['encoded']=row
    for k,v in candidates_count.items():require(v==cache['summary'][k],'Candidate counts differ: '+k)
    life={n:{**dict(counts[n]),'ever_emitted_ids':len(emitted[n])} for n in VARIANTS}
    require(life['disabled']==reference['summary']['lifecycle']['staged'],'Validation control lifecycle differs')
    require(all(v['observations']==observations for v in life.values()),'Runtime observations changed')
    require(cuts==decisions['split_confirmed']==owners['enabled'].segmenter.allocated_segments,'Split accounting differs')
    return {'lifecycle':life,'split_events':cuts,'decision_counts':dict(decisions),'reset_counts':dict(resets),
            'policy_peaks':dict(peaks),'observations':observations,'candidates':dict(candidates_count)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development-report',type=Path,required=True)
    parser.add_argument('--validation-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve();digest=sha256(path)
        require(expected is None or digest==expected,'Changed input: '+str(path))
        inputs[name]={'path':str(path),'sha256':digest};return path
    print('Verifying development policy/code and frozen validation inputs; no GT...',flush=True)
    dev=json.loads(checked('development_report',args.development_report).read_text())
    required=('disabled_full_runtime_records_exact','disabled_metrics_lifecycle_merges_exact','original_local_metrics_unchanged',
              'boxes_scores_features_rows_unchanged','new_segments_start_fresh_history','source_key_projection_complete',
              'full_window_metrics_motmetrics_agree','all_split_events_diagnosed','predictions_frozen_before_GT','frozen_inputs_outputs_unchanged')
    require(dev.get('completed') is True and dev['protocol']=='appearance_continuity_paired_v1'
            and all(dev['checks'].get(k) is True for k in required),'Unverified development experiment')
    for name,item in dev['inputs'].items():
        if name!='ground_truth':checked('development:'+name,item['path'],item['sha256'])
    config=dev['configuration'];settings=settings_from_configuration(config)
    require(config==json.loads(Path(dev['inputs']['experiment_configuration']['path']).read_text())
            and config['threshold_search'] is False and config['dormant_recovery_enabled'] is False,
            'Development configuration changed')
    require(config['development_scene']==dev['scene']=='MTMC_Tracking_2024/train/scene_001'
            and config['local_variant']=='staged' and config['global_history_update_policy']=='all_updates',
            'Unexpected development scope')
    # No command-line policy override: validation uses the already evaluated settings.
    val_path=checked('validation_reference',args.validation_report);val=json.loads(val_path.read_text())
    val_checks=('frozen_policy_used','all_candidate_rows_consumed','predictions_frozen_before_GT',
                'local_global_control_denominators_agree','global_and_control_motmetrics_agree',
                'all_empty_GT_slots_retained','predictions_unchanged_during_evaluation')
    require(val.get('completed') is True and val['protocol']=='scene_paired_tracking_evaluation_v1'
            and all(val['checks'].get(k) is True for k in val_checks),'Unverified validation baseline')
    for name,item in val['inputs'].items():
        if name!='ground_truth':checked('validation:'+name,item['path'],item['sha256'])
    loaded=load_scene(val['inputs']['scene_config']['path'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    policy=val['configuration'];dev_policy=json.loads(Path(dev['inputs']['frozen_policy']['path']).read_text())
    require(policy==dev_policy and policy['global']==dev['global_policy'],'Global/local settings differ between scenes')
    require(scene.scene==val['scene']=='MTMC_Tracking_2024/val/scene_041' and scene.camera_ids==(361,362,364)
            and scene.fps==30 and scene.rounds==3600 and val['runtime_frames']==[0,3599]
            and val['evaluation']['frames']==[spec.first_frame,spec.last_frame]==[2,3599]
            and val['evaluation']['cameras']==list(scene.camera_ids) and spec.min_iou==.5 and spec.gt_to_video_offset==0,
            'Unexpected validation scene/evaluation')
    require(val['inputs']['ground_truth']['sha256']==spec.ground_truth.sha256,'Mixed validation GT reference')
    cache_path=Path(val['inputs']['candidate_report']['path']);cache=json.loads(cache_path.read_text())
    require(cache.get('completed') is True and cache['protocol']=='scene_candidate_collection_fp32_v1'
            and cache['run_id']==val['cache_run_id'] and cache['scene']==scene.scene
            and cache['dataset']==scene.dataset and cache['revision']==scene.revision,'Mixed validation cache')
    for name in ('scene_config','frozen_policy'):
        require(cache['inputs'][name]['sha256']==val['inputs'][name]['sha256'],'Validation cache lineage differs')
    for name in ('detections.jsonl','embeddings.npy'):
        item=cache['artifacts'][name]
        checked('validation_cache:'+name,cache_path.parent/item['path'],item['sha256'])
        require(item['sha256']==val['inputs'][name]['sha256'],'Validation candidate artifact differs')
    trace_ref=val['artifacts']['paired_tracks.jsonl.gz']
    reference_path=checked('validation_frozen_trace',val_path.parent/trace_ref['path'],trace_ref['sha256'])
    for package in ('supervision','numpy','scipy','motmetrics','pandas'):
        require(version(package)==dev['versions'][package]==val['versions'][package],'Dependency changed: '+package)
    for rel in ('scripts/validate_appearance_continuity.py','scripts/evaluate_scene_pair.py','src/mtmc/data/candidates.py'):
        checked('transfer_code:'+rel,ROOT/rel)
    # Regression gate for the generic feature join, using the already verified development cache.
    dev_scene=load_scene(dev['inputs']['scene_config']['path'],project_root=ROOT).runtime
    dev_cache=json.loads(Path(dev['inputs']['cache_report']['path']).read_text())
    dev_ref=json.loads(Path(dev['inputs']['reference_global_report']['path']).read_text())
    dev_vectors=np.load(dev['inputs']['embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    adapter=check_development_adapter(dev_scene,dev_cache,dev_ref,dev['inputs']['detections.jsonl']['path'],
                                      dev['inputs']['global_variants']['path'],dev_vectors)
    require(adapter['observations']==dev['summary']['observations'],'Development observation coverage differs')
    vectors=np.load(inputs['validation_cache:embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(vectors.dtype==np.float32 and vectors.shape==(cache['summary']['encoded'],512),'Validation feature shape differs')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/appearance_continuity_validation'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    try:
        print('Phase 1: unchanged continuity policy on scene_041; fixed local observations; no GT...',flush=True)
        summary=replay(scene,policy,run_id=run,cache=cache,reference=val,
            candidate_path=inputs['validation_cache:detections.jsonl']['path'],reference_path=reference_path,
            vectors=vectors,output=output,settings=settings)
        outputs=[output/n for n in ('global_tracks.jsonl.gz','continuity_audit.jsonl.gz')]
        frozen={p.name:sha256(p) for p in outputs};freeze=output/'predictions_frozen.json'
        freeze.write_text(json.dumps({'run_id':run,'development_run_id':dev['run_id'],'scene':scene.scene,
            'continuity_configuration':config,'global_policy':policy['global'],'sha256':frozen},indent=2,allow_nan=False)+'\n')
        print('Phase 2: predictions frozen; offline validation full/window evaluation...',flush=True)
        ground=load_ground_truth(spec);checked('validation_ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mappings=evaluate(outputs[0],scene,spec,ground,run_id=run,split=1800)
        baseline=val['results']['staged']
        require(quality['global']['disabled']['full']==baseline['global'],'Validation baseline quality differs')
        require(quality['accepted_merge_diagnostics']['disabled']==baseline['accepted_merge_diagnostics'],
                'Validation baseline merge diagnostics differ')
        for camera,values in baseline['local'].items():
            for field,expected in values.items():
                actual=quality['local_shared'][camera][field]
                require(actual==expected or (actual is not None and expected is not None and abs(actual-expected)<=1e-9),
                        'Original local metrics changed: '+camera+'/'+field)
        for name in VARIANTS:
            require(sum(quality['accepted_merge_diagnostics'][name].values())==summary['lifecycle'][name]['merge_events'],
                    'Merge diagnostic coverage differs')
        categories,events=diagnose_splits(outputs[0],outputs[1],scene,spec,ground,run_id=run,expected_events=summary['split_events'])
        for name,digest in frozen.items():require(sha256(output/name)==digest,'Frozen validation output changed')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during transfer experiment')
        matching=output/'identity_matching.json';matching.write_text(json.dumps(mappings,indent=2,allow_nan=False)+'\n')
        event_path=output/'split_diagnostics.json';event_path.write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
        report={'completed':True,'protocol':'appearance_continuity_scene_transfer_v1','run_id':run,
            'scene':scene.scene,'development_run_id':dev['run_id'],'validation_reference_run_id':val['run_id'],
            'inputs':inputs,'configuration':{'scene':scene.scene,'runtime_frames':[0,3599],'evaluation_frames':[2,3599],
                'frozen_development_policy':config,'paired_policy':policy},'development_adapter_parity':adapter,
            'summary':summary,**quality,'split_reference_gt_diagnostics':categories,
            'development_global_metrics':dev['global'],
            'checks':{'development_join_parity_exact':True,'policy_code_settings_unchanged':True,
                'validation_control_full_records_lifecycle_exact':True,'validation_control_quality_merges_exact':True,
                'original_local_metrics_unchanged':True,'same_boxes_scores_features_rows':True,
                'fresh_segment_history_and_original_key_projection':True,'full_window_metrics_motmetrics_agree':True,
                'all_split_events_diagnosed':True,'predictions_frozen_before_GT':True,'frozen_inputs_outputs_unchanged':True},
            'versions':{p:version(p) for p in ('supervision','numpy','scipy','motmetrics','pandas')},
            'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in (*outputs,freeze,matching,event_path)},
            'limits':['Scene_041 has already been inspected; this is unchanged-policy transfer, not an untouched final test.',
                'One two-minute subset and selected overlapping cameras per scene do not establish universal generalization.',
                'Local metrics use original frozen tracker IDs; split GT categories do not establish lifetime purity or causal benefit.',
                'No ROI/GT exclusions added; original empty slots remain in evaluation.',
                'Dormant recovery off; no models, decoding, threshold tuning or end-to-end speed claim.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,'error':str(error)})+'\n');raise
    for window,bounds in quality['windows'].items():
        print(f'Window {window}: frames {bounds}')
        print('Variant       Global IDF1     IDTP     IDFP     IDFN')
        for name in VARIANTS:
            m=quality['global'][name][window]
            print(f'{name:12} {100*m["idf1"]:11.2f}% {m["idtp"]:8} {m["idfp"]:8} {m["idfn"]:8}')
    print('Lifecycle:',json.dumps(summary['lifecycle']))
    print('Split events:',summary['split_events'])
    print('Prior-reference/current GT diagnostics:',json.dumps(categories))
    print('Policy decisions:',json.dumps(summary['decision_counts']))
    print('Evidence resets:',json.dumps(summary['reset_counts']))
    print(f'Report: {path}')
    print('Appearance continuity scene transfer: COMPLETED; policy unchanged; runtime baseline unchanged')


if __name__=='__main__':main()
