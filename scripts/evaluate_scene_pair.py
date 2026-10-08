"""Replay the two frozen policies, freeze outputs, then evaluate a configured scene."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from evaluate_appearance_global import lifecycle
from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference
from mtmc.data.candidates import validate_cache_round
from mtmc.data.ground_truth import load_ground_truth, spatial_slot, clip_boxes, pairwise_iou
from mtmc.data.scene import load_scene, require, sha256
from mtmc.pipeline.paired import PairedAssociation, CandidateRound, CameraCandidates, VARIANTS
from mtmc.reid.osnet import ObservationKey

ROOT=Path(__file__).resolve().parents[1]


def candidate_batch(record, scene, vectors):
    """Call after validate_cache_round; join by recorded rows, never by IoU."""
    by_camera={c:[] for c in scene.camera_ids}
    for item in record['detections']:by_camera[item['camera_id']].append(item)
    cameras=[]
    for c,items in by_camera.items():
        rows=tuple(x['embedding_row'] for x in items)
        cameras.append(CameraCandidates(c,np.asarray([x['xyxy'] for x in items],np.float32).reshape(-1,4),
            np.asarray([x['confidence'] for x in items],np.float32),rows,
            tuple(vectors[i] if i is not None else None for i in rows)))
    return CandidateRound(record['cache_run_id'],record['frame_index'],Fraction(record['timestamp']),tuple(cameras))


def replay(scene, policy, *, cache_run, run_id, candidate_path, vectors, output_path):
    runtime=PairedAssociation(scene,policy,run_id=run_id,source_run_id=cache_run)
    counters={n:Counter() for n in VARIANTS};emitted={n:set() for n in VARIANTS}
    refinement={n:Counter() for n in VARIANTS}
    candidate_counts=Counter(detections=0,encoded=0,fully_outside=0)
    candidate_counts.update({f'camera_{c}':0 for c in scene.camera_ids})
    row=0
    with Path(candidate_path).open() as stream,gzip.open(output_path,'xt') as saved:
        for frame in range(scene.rounds):
            line=stream.readline();require(bool(line),'Truncated candidate stream')
            record=json.loads(line)
            row,missing,per_camera=validate_cache_round(record,scene,vectors,run_id=cache_run,frame=frame,row_offset=row)
            result=runtime.step(candidate_batch(record,scene,vectors))
            result['scene']=scene.scene
            for name in VARIANTS:
                lifecycle(result['variants'][name]['identity'],counters[name],emitted[name])
                refinement[name].update(result['refinement_counters'][name])
            saved.write(json.dumps(result,allow_nan=False)+'\n')
            candidate_counts['fully_outside']+=missing
            for c,n in per_camera.items():candidate_counts[f'camera_{c}']+=n;candidate_counts['detections']+=n
            if (frame+1)%300==0 or frame+1==scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; both policies, no GT input',flush=True)
        require(stream.readline()=='','Trailing candidate stream')
    require(row==runtime.next_embedding_row==len(vectors),'Candidate rows not consumed exactly')
    candidate_counts['encoded']=row
    for name in VARIANTS:counters[name]['ever_emitted_ids']=len(emitted[name])
    return {'candidates':dict(candidate_counts),'lifecycle':{n:dict(c) for n,c in counters.items()},
            'refinement_counters':{n:dict(c) for n,c in refinement.items()}}


def read_variant(record, scene, *, frame, run_id, cache_run, variant):
    require(record['run_id']==run_id and record['source_run_id']==cache_run and record['scene']==scene.scene
            and type(record['frame_index']) is int and record['frame_index']==frame
            and record['timestamp']==str(Fraction(frame,scene.fps))
            and set(record['variants'])==set(VARIANTS),'Frozen result scope differs')
    data=record['variants'][variant];identity=data['identity']
    require(identity['run_id']==run_id+'/'+variant and identity['frame_index']==frame
            and identity['timestamp']==record['timestamp'],'Frozen identity scope differs')
    cameras=data['cameras']
    require([c['camera'] for c in cameras]==list(scene.camera_ids),'Frozen camera coverage/order differs')
    expected=set()
    for item in cameras:
        ids=item['local_ids'];n=len(ids)
        require(all(type(i) is int and i>=0 for i in ids) and len(set(ids))==n
                and all(len(item[k])==n for k in ('xyxy','confidence','detection_indices','embedding_rows')),
                'Malformed frozen local observations')
        indices=item['detection_indices']
        require(all(type(i) is int and i>=0 for i in indices) and len(set(indices))==n,'Invalid selected candidates')
        expected.update(ObservationKey(item['camera'],i,frame) for i in ids)
    assigned={};occupied=set()
    for a in identity['assignments']:
        key=ObservationKey(**a['key']);gid=a['global_id']
        require(all(type(x) is int and x>=0 for x in (key.camera_id,key.local_id,key.frame_index))
                and key in expected and key not in assigned and type(gid) is int and gid>=0
                and (key.camera_id,gid) not in occupied,'Invalid global observation/camera uniqueness')
        assigned[key]=gid;occupied.add((key.camera_id,gid))
    require(set(assigned)==expected,'Global/local observation coverage differs')
    return data,assigned


def evaluate(trace_path, scene, spec, ground, *, run_id, cache_run):
    """Evaluate frozen boxes/IDs. Never call a tracker or modify its predictions."""
    import motmetrics as mm
    require(version('motmetrics')=='1.4.0','Expected pinned motmetrics 1.4.0')
    require(scene.camera_ids==spec.camera_ids and 0<=spec.first_frame<=spec.last_frame<scene.rounds,
            'Evaluation scope differs from runtime')
    mm.lap.default_solver='scipy'
    names=['num_frames','num_objects','num_predictions','idtp','idfp','idfn','idf1','idp','idr',
           'precision','recall','num_switches','num_false_positives','num_misses']
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    expected_slots=(spec.last_frame-spec.first_frame+1)*len(scene.cameras)
    results={};mappings={}
    for variant in VARIANTS:
        global_count=IdentityCounts();control_count=IdentityCounts()
        global_slots=[];control_slots=[];local_to_control={}
        local={c:mm.MOTAccumulator(auto_id=False) for c in scene.camera_ids}
        merges=Counter();counts=Counter();empty_predictions=Counter({c:0 for c in scene.camera_ids})
        raw_empty_slots=Counter({c:0 for c in scene.camera_ids})
        excluded=Counter({c:0 for c in scene.camera_ids});outside=Counter({c:0 for c in scene.camera_ids})
        with gzip.open(trace_path,'rt') as stream:
            for frame in range(scene.rounds):
                line=stream.readline();require(bool(line),'Truncated frozen prediction stream')
                data,assigned=read_variant(json.loads(line),scene,frame=frame,run_id=run_id,cache_run=cache_run,variant=variant)
                counts['runtime_observations']+=len(assigned)
                within=spec.first_frame<=frame<=spec.last_frame
                unique={}
                if within:
                    for item in data['cameras']:
                        c=item['camera'];ids=item['local_ids'];w,h=sizes[c]
                        keys=tuple(ObservationKey(c,i,frame) for i in ids)
                        truth=ground.slots[frame,c]
                        gt,mask,evidence,dropped,fully_outside=spatial_slot(truth,keys,item['xyxy'],width=w,height=h,min_iou=spec.min_iou)
                        unique.update(evidence);excluded[c]+=dropped;outside[c]+=fully_outside
                        if not truth:
                            raw_empty_slots[c]+=1;empty_predictions[c]+=len(ids)
                        predicted=[assigned[k] for k in keys]
                        global_count.update(gt,predicted,mask);global_slots.append((gt,predicted,mask))
                        control=[]
                        for i in ids:
                            key=c,i
                            if key not in local_to_control:local_to_control[key]=len(local_to_control)+1
                            control.append(local_to_control[key])
                        control_count.update(gt,control,mask);control_slots.append((gt,control,mask))
                        # Actual IoU costs preserve CLEAR matching/IDSW behavior;
                        # a binary admissibility matrix alone is insufficient here.
                        gt_boxes=clip_boxes([truth[g] for g in gt],w,h)
                        pred_boxes=clip_boxes(item['xyxy'],w,h)
                        iou=pairwise_iou(gt_boxes,pred_boxes)
                        costs=np.where(iou>=spec.min_iou,1.-iou,np.nan)
                        require(np.array_equal(np.isfinite(costs),mask),'Local/global spatial gates differ')
                        local[c].update(gt,ids,costs,frameid=frame)
                for event in data['identity']['merge_events']:
                    labels=[unique.get(ObservationKey(**m)) for m in event['members']]
                    category=('outside_evaluation' if not within else 'unresolved' if not labels or any(x is None for x in labels)
                              else 'all_visible_members_same_gt' if len(set(labels))==1 else 'different_known_gt')
                    merges[category]+=1
            require(stream.readline()=='','Trailing frozen prediction frames')
        global_metrics,global_map=global_count.result();control_metrics,control_map=control_count.result()
        check_reference(global_metrics,reference_metrics(global_slots))
        check_reference(control_metrics,reference_metrics(control_slots))
        table=mm.metrics.create().compute_many([local[c] for c in scene.camera_ids],metrics=names,
            names=[f'camera_{c:04d}' for c in scene.camera_ids],generate_overall=True)
        local_metrics=json.loads(table.to_json(orient='index'));pooled=local_metrics['OVERALL']
        require(pooled['num_objects']==global_metrics['gt_observations']==control_metrics['gt_observations']
                and pooled['num_predictions']==global_metrics['predicted_observations']==control_metrics['predicted_observations']
                and pooled['num_frames']==global_metrics['camera_time_slots']==control_metrics['camera_time_slots']==expected_slots,
                'Evaluation denominators differ')
        for c in scene.camera_ids:
            require(raw_empty_slots[c]==sum(1 for f,cam in ground.empty_slots if cam==c),'Empty GT slots dropped')
        results[variant]={'global':global_metrics,'no_cross_camera_control':control_metrics,'local':local_metrics,
            'accepted_merge_diagnostics':dict(merges),'runtime_observations':counts['runtime_observations'],
            'empty_gt_slots':dict(raw_empty_slots),'predictions_in_empty_gt_slots':dict(empty_predictions),
            'excluded_zero_area_gt':dict(excluded),'fully_outside_predictions':dict(outside)}
        mappings[variant]={'global':global_map,'control':control_map,
            'control_local_id_map':[{'camera':c,'local_id':i,'control_id':value} for (c,i),value in local_to_control.items()]}
        print(f'{variant}: local/global/control denominators and global metrics/motmetrics VERIFIED',flush=True)
    require(results['staged']['global']['gt_observations']==results['competitive_iou']['global']['gt_observations'],
            'Variant GT denominators differ')
    return results,mappings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve();digest=sha256(path)
        require(expected is None or digest==expected,f'Checksum mismatch: {path}')
        inputs[name]={'path':str(path),'sha256':digest};return path
    print('Verifying frozen candidate cache, scene configuration and paired policy...',flush=True)
    cache_path=checked('candidate_report',args.candidate_report);cache=json.loads(cache_path.read_text())
    require(cache.get('completed') is True and cache['protocol']=='scene_candidate_collection_fp32_v1'
            and all(cache['checks'].get(k) is True for k in ('persisted_candidates_and_embeddings_verified',
                         'all_frames_recorded','frozen_models_settings_and_code_verified')),'Expected complete scene candidate cache')
    for name in ('scene_config','frozen_policy','paired_runtime_parity'):
        spec=cache['inputs'][name];checked(name,spec['path'],spec['sha256'])
    loaded=load_scene(inputs['scene_config']['path'],project_root=ROOT);scene=loaded.runtime
    for name,ref in (('source_manifest',loaded.source_manifest),('video_manifest',loaded.video_manifest),('calibration',scene.calibration)):
        require(cache['inputs'][name]['sha256']==ref.sha256,'Cached scene asset differs')
        checked(name,ref.path,ref.sha256)
    for c in scene.cameras:
        require(cache['inputs'][f'video_{c.camera_id}']['sha256']==c.video.sha256,'Cached video differs')
        checked(f'video_{c.camera_id}',c.video.path,c.video.sha256)
    require((cache['scene'],cache['dataset'],cache['revision'])==(scene.scene,scene.dataset,scene.revision),'Mixed dataset/scene')
    cfg=cache['configuration']
    require(cfg['frames']==[0,scene.rounds-1] and cfg['rounds']==scene.rounds and cfg['cameras']==list(scene.camera_ids)
            and cfg['fps']==scene.fps and cfg['dtype']=='float32' and cfg['feature_dim']==512
            and cfg['image_sizes']=={str(c.camera_id):[c.width,c.height] for c in scene.cameras},'Cache configuration differs')
    policy=json.loads(Path(inputs['frozen_policy']['path']).read_text())
    gate=json.loads(Path(inputs['paired_runtime_parity']['path']).read_text())
    require(gate.get('passed') is True and gate.get('completed') is True
            and gate['protocol']=='paired_scene_runtime_parity_v1' and policy==gate['configuration']
            and gate['inputs']['frozen_policy']['sha256']==inputs['frozen_policy']['sha256'],'Frozen policy/parity lineage differs')
    # Code fingerprints from the real-data parity gate include the old local and
    # global implementations in its verified input set.
    for key,spec in gate['inputs'].items():
        if key.startswith('code:'):checked(key,ROOT/key[5:],spec['sha256'])
    for rel,digest in gate['code_sha256'].items():checked('code:'+rel,ROOT/rel,digest)
    require(version('supervision')==gate['versions']['supervision']=='0.30.7','Tracker package version differs')
    for name in ('numpy','scipy'):require(version(name)==gate['versions'][name],f'Runtime package differs: {name}')
    for rel in ('src/mtmc/data/candidates.py','src/mtmc/data/scene.py'):
        spec=cache['inputs']['code:'+rel];checked('code:'+rel,ROOT/rel,spec['sha256'])
    for name in ('detections.jsonl','embeddings.npy'):
        spec=cache['artifacts'][name];checked(name,cache_path.parent/spec['path'],spec['sha256'])
    vectors=np.load(inputs['embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(vectors.dtype==np.float32 and vectors.shape==(cache['summary']['encoded'],512),'Feature archive shape differs')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/scene_pair_evaluation'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    trace=output/'paired_tracks.jsonl.gz'
    code=[Path(__file__),ROOT/'scripts/evaluate_global_identity.py',ROOT/'scripts/evaluate_appearance_global.py',
          ROOT/'src/mtmc/data/ground_truth.py',ROOT/'src/mtmc/pipeline/paired.py']
    for p in code:checked('code:'+p.relative_to(ROOT).as_posix(),p)
    try:
        print('Phase 1: causal paired replay on CPU; no models, decoding or GT...',flush=True)
        summary=replay(scene,policy,cache_run=cache['run_id'],run_id=run,candidate_path=inputs['detections.jsonl']['path'],
                       vectors=vectors,output_path=trace)
        for k,v in summary['candidates'].items():require(cache['summary'][k]==v,'Source candidate count differs')
        frozen=sha256(trace)
        # Commit the prediction fingerprint before opening ground truth.
        (output/'predictions_frozen.json').write_text(json.dumps({'run_id':run,'sha256':frozen,
            'path':trace.name,'configuration':policy,'summary':summary},indent=2)+'\n')
        print(f'Phase 2: predictions frozen; loading GT for frames {loaded.evaluation.first_frame}..{loaded.evaluation.last_frame}...',flush=True)
        ground=load_ground_truth(loaded.evaluation)
        checked('ground_truth',loaded.evaluation.ground_truth.path,loaded.evaluation.ground_truth.sha256)
        results,mapping=evaluate(trace,scene,loaded.evaluation,ground,run_id=run,cache_run=cache['run_id'])
        for n in VARIANTS:
            require(results[n]['runtime_observations']==summary['lifecycle'][n]['observations']
                    and sum(results[n]['accepted_merge_diagnostics'].values())==summary['lifecycle'][n]['merge_events'],
                    'Runtime/evaluation accounting differs')
        require(sha256(trace)==frozen,'Predictions changed during evaluation')
        for spec in inputs.values():require(sha256(spec['path'])==spec['sha256'],'Input changed during experiment')
        matching=output/'identity_matching.json';matching.write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
        report={'completed':True,'protocol':'scene_paired_tracking_evaluation_v1','run_id':run,'cache_run_id':cache['run_id'],
            'scene':scene.scene,'dataset':scene.dataset,'revision':scene.revision,'inputs':inputs,'configuration':policy,
            'runtime_frames':[0,scene.rounds-1],'evaluation':{
                'frames':[loaded.evaluation.first_frame,loaded.evaluation.last_frame],'cameras':list(scene.camera_ids),
                'min_iou':loaded.evaluation.min_iou,'gt_to_video_offset':loaded.evaluation.gt_to_video_offset,
                'empty_slot_policy':loaded.evaluation.empty_slot_policy,
                'box_policy':'Clip both; exclude zero-area GT; keep all predictions',
                'global_matching':'One shared GT/predicted ID assignment across all cameras and evaluated frames',
                'local_matching':'Independent per-camera identities; true IoU costs; pooled counts',
                'control':'Camera-scoped local IDs evaluated globally, independently for each variant'},
            'summary':summary,'results':results,
            'checks':{'frozen_policy_used':True,'all_candidate_rows_consumed':True,'predictions_frozen_before_GT':True,
                'local_global_control_denominators_agree':True,'global_and_control_motmetrics_agree':True,
                'all_empty_GT_slots_retained':True,'predictions_unchanged_during_evaluation':True},
            'versions':{n:version(n) for n in ('supervision','numpy','scipy','motmetrics','pandas')},
            'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in (trace,matching,output/'predictions_frozen.json')},
            'limits':['One selected validation scene, three overlapping cameras and two-minute subset; not universal generalization.',
                'No threshold selection, tuning, state reset or past-ID rewriting during this run.',
                'Variant predictions/counts can differ; candidates, GT and evaluation protocol are shared.',
                'Empty GT slots count predictions as false observations under the supplied annotation protocol.',
                'Visible-member merge diagnostics do not prove purity of the full retained identity.',
                'Global 2D identity protocol is a project metric, not the official AI City world-coordinate HOTA.',
                'CPU replay excludes decoding and inference; no end-to-end speed claim.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,'error':str(error)})+'\n')
        raise
    def percent(x):return 'n/a' if x is None else f'{100*x:.2f}%'
    print('Variant              Local IDF1  Global IDF1  Control IDF1  Local IDSW   FP      FN')
    for n in VARIANTS:
        r=results[n];local=r['local']['OVERALL']
        print(f"{n:20} {percent(local['idf1']):>10} {percent(r['global']['idf1']):>12} {percent(r['no_cross_camera_control']['idf1']):>13} "
              f"{int(local['num_switches']):10} {int(local['num_false_positives']):7} {int(local['num_misses']):7}")
        print('  Global identity counts:',json.dumps({k:r['global'][k] for k in ('gt_observations','predicted_observations','idtp','idfp','idfn')}))
        print('  Accepted merges:',json.dumps(r['accepted_merge_diagnostics']))
        print('  Predictions in empty GT slots:',json.dumps(r['predictions_in_empty_gt_slots']))
    print('Shared frozen candidates and unchanged policy: VERIFIED; threshold selection: NONE')
    print(f'Report: {path}')
    print('Scene paired tracking evaluation: COMPLETED')


if __name__=='__main__':main()
