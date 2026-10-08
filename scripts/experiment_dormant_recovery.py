"""One fixed dormant-recovery hypothesis on frozen staged local observations."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
from fractions import Fraction
import gzip
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np

from check_frozen_bytetrack_replay import candidate_arrays
from evaluate_appearance_global import direct_observations
from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference
from mtmc.association.dormant import ArchiveSettings
from mtmc.data.ground_truth import load_ground_truth, spatial_slot, clip_boxes, pairwise_iou
from mtmc.data.scene import load_scene, require, sha256
from mtmc.pipeline.recovery import RecoveryIdentityStage
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey
from run_mtmc import dump

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('disabled','enabled')


def archive_settings(config):
    return ArchiveSettings(Fraction(config['max_age_seconds']),config['min_similarity'],config['min_margin'],
        config['max_speed'],config['position_slack'],config['max_identities'])


def replay(scene,policy,*,run_id,source_run,cache,reference,paths,vectors,output,settings):
    cfg=policy['global']
    history=AppearanceHistory(run_id,max_observations=cfg['history']['max_observations'],
        max_age=Fraction(cfg['history']['max_age_seconds']))
    stages={n:RecoveryIdentityStage(run_id+'/'+n,scene.matrices,scene.coordinate_space,
        variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],**cfg['geometry'],
        identity_configuration=cfg['identity'],recovery_enabled=n=='enabled',archive_settings=settings if n=='enabled' else None,
        history_max_observations=history.max_observations,history_max_age=history.max_age) for n in VARIANTS}
    decisions=Counter();snapshots={n:Counter() for n in VARIANTS};peak_archive=0
    archive_counts=Counter({k:0 for k in ('retired','skipped_retirements','blocked_removed','expired','capacity_evicted')})
    next_row=candidate_count=outside_count=observations=0
    with Path(paths['tracks']).open() as src,Path(paths['detections.jsonl']).open() as det, \
            gzip.open(paths['global_variants'],'rt') as ref,gzip.open(output/'global_tracks.jsonl.gz','xt') as saved, \
            gzip.open(output/'registry_audit.jsonl.gz','xt') as audit_saved, \
            gzip.open(output/'history_provenance.jsonl.gz','xt') as provenance:
        for frame in range(scene.rounds):
            lines=[stream.readline() for stream in (src,det,ref)];require(all(lines),'Truncated source')
            original,cached,previous=map(json.loads,lines);time=Fraction(frame,scene.fps)
            require(previous['run_id']==reference['run_id'] and previous['frame_index']==frame
                    and previous['timestamp']==str(time),'Reference scope differs')
            arrays,_,next_row,outside=candidate_arrays(cached,original,frame=frame,cache_run=cache['run_id'],
                source_run=source_run,next_row=next_row)
            candidate_count+=sum(len(a[0]) for a in arrays.values());outside_count+=outside
            cameras=previous['variants']['staged']['cameras']
            records,features=direct_observations(cameras,cached,vectors,frame)
            current_history=history.update(frame,time,features);observations+=len(records)
            variants={};audits={};groups={};evidence_maps={}
            for name in VARIANTS:
                stage=stages[name]
                identity,grouped,_,grounds,_,_=stage.update_with_history(frame,time,
                    replace(current_history,run_id=run_id+'/'+name),records)
                audit=stage.registry.last_audit
                groups[name]=replace(grouped,run_id='comparison')
                variants[name]={'cameras':cameras,'identity':json.loads(dump(asdict(identity))),
                    'reactivations':[asdict(e) for e in audit.reactivations]}
                audits[name]=asdict(audit)
                for _,status in audit.snapshot_decisions:snapshots[name][status]+=1
                actual={row.key:row for row in stage.last_evidence.observations}
                require(set(actual)=={r.key for r in records},'Evidence observation coverage differs')
                projected={o.key:o.xy for camera in grounds.values() for o in camera.observations}
                index={key:i for i,key in enumerate(current_history.mean.keys)}
                metadata=[]
                for key,row in actual.items():
                    require(row.ground_xy==projected.get(key),'Evidence projection differs from association geometry')
                    if key in index:
                        i=index[key]
                        expected_time=time if current_history.used_latest_fallback[i] else current_history.source_times[i][0]
                        require(np.array_equal(row.descriptor,current_history.mean.embeddings[i])
                                and row.descriptor_time==expected_time,'Descriptor row/source provenance differs')
                        source_frames=current_history.source_frames[i];source_times=current_history.source_times[i]
                        fallback=current_history.used_latest_fallback[i]
                    else:
                        require(row.descriptor is None and row.descriptor_time is None,'Outside crop acquired appearance')
                        source_frames=source_times=();fallback=False
                    metadata.append({'key':asdict(key),'source_frames':source_frames,'source_times':source_times,
                        'latest_fallback':fallback,'effective_descriptor_time':row.descriptor_time,
                        'ground_xy':row.ground_xy,'ground_time':row.ground_time})
                evidence_maps[name]=metadata
                if name=='enabled':
                    for decision in audit.archive.decisions:decisions[decision.outcome]+=1
                    for field in archive_counts:archive_counts[field]+=len(getattr(audit.archive,field))
                    peak_archive=max(peak_archive,len(audit.archive.retained_ids))
            require(groups['disabled']==groups['enabled'] and evidence_maps['disabled']==evidence_maps['enabled'],
                    'Upstream groups or evidence changed between variants')
            expected=previous['variants']['staged']['identity']
            require(expected['run_id']==reference['run_id']+'/staged'
                    and variants['disabled']['identity']=={**expected,'run_id':run_id+'/disabled'},
                    f'Disabled reference record differs at frame {frame}')
            header={'run_id':run_id,'frame_index':frame,'timestamp':str(time)}
            saved.write(dump({**header,'variants':variants})+'\n')
            audit_saved.write(dump({**header,'variants':audits})+'\n')
            provenance.write(dump({**header,'observations':evidence_maps['disabled']})+'\n')
            if (frame+1)%300==0 or frame+1==scene.rounds:
                count=dict(stages['enabled'].registry.last_audit.lifecycle)['reactivation_events']
                print(f'Replayed {frame+1}/{scene.rounds}; disabled control EXACT; reactivations={count}',flush=True)
        require(all(stream.readline()=='' for stream in (src,det,ref)),'Trailing source frames')
    require(next_row==len(vectors)==cache['summary']['encoded'] and candidate_count==cache['summary']['detections']
            and outside_count==cache['summary']['fully_outside'],'Candidate coverage differs')
    lifecycle={n:dict(stage.registry.last_audit.lifecycle) for n,stage in stages.items()}
    rename={'allocated_id_slots':'allocated_ids','expiration_events':'expired_ids','retained_ids':'retained_ids_at_end'}
    for field in ('observations','allocated_id_slots','ever_emitted_ids','absorbed_ids','expiration_events','retained_ids','merge_events'):
        require(lifecycle['disabled'][field]==reference['lifecycle']['staged'][rename.get(field,field)],'Control lifecycle differs')
    require(all(c['observations']==observations for c in lifecycle.values()),'Tracked observations were lost')
    require(lifecycle['enabled']['reactivation_events']==decisions['matched'],'Claim/event counts differ')
    return {'lifecycle':lifecycle,'return_decisions':dict(decisions),'archive_events':dict(archive_counts),
        'snapshot_decisions':{n:dict(c) for n,c in snapshots.items()},'peak_archived_identities':peak_archive,
        'peak_archive_vector_payload_bytes':peak_archive*512*4,
        'observations':observations,'candidate_observations':candidate_count,'candidate_embedding_rows':next_row}


def evaluate(trace, scene, spec, ground, *, run_id, split):
    import motmetrics as mm
    mm.lap.default_solver = 'scipy'
    require(spec.first_frame < split <= spec.last_frame, 'Invalid evaluation split')
    intervals = {'full':[spec.first_frame,spec.last_frame], 'first':[spec.first_frame,split-1], 'second':[split,spec.last_frame]}
    counts = {n:{w:IdentityCounts() for w in intervals} for n in VARIANTS}
    slots = {n:{w:[] for w in intervals} for n in VARIANTS}
    local = {c:mm.MOTAccumulator(auto_id=False) for c in scene.camera_ids}
    merges = {n:Counter() for n in VARIANTS}
    sizes = {c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for frame in range(scene.rounds):
            line = stream.readline(); require(bool(line),'Truncated experiment trace')
            row = json.loads(line)
            require(row['run_id'] == run_id and row['frame_index'] == frame
                    and row['timestamp'] == str(Fraction(frame,scene.fps)) and set(row['variants']) == set(VARIANTS),
                    'Experiment output scope differs')
            require(row['variants']['disabled']['cameras'] == row['variants']['enabled']['cameras'],
                    'Local records changed between variants')
            within = spec.first_frame <= frame <= spec.last_frame
            for name in VARIANTS:
                data = row['variants'][name]; identity = data['identity']
                require(identity['run_id'] == run_id+'/'+name and identity['frame_index'] == frame
                        and identity['timestamp'] == row['timestamp'], 'Identity output scope differs')
                assigned = {ObservationKey(**a['key']):a['global_id'] for a in identity['assignments']}
                require(len(assigned) == len(identity['assignments'])
                        and all(type(g) is int and g>=0 for g in assigned.values())
                        and len({(k.camera_id,g) for k,g in assigned.items()}) == len(assigned),
                        'Duplicate observation or identity/camera collision')
                require([c['camera'] for c in data['cameras']] == list(scene.camera_ids), 'Camera coverage differs')
                seen,unique = set(),{}
                for camera in data['cameras']:
                    c = camera['camera']; ids = camera['local_ids']
                    keys = tuple(ObservationKey(c,i,frame) for i in ids)
                    require(len(keys)==len(set(keys)) and set(keys)<=set(assigned),'Local/global coverage differs')
                    seen.update(keys)
                    if not within: continue
                    truth = ground.slots[frame,c]; width,height = sizes[c]
                    gt,mask,evidence,_,_ = spatial_slot(truth,keys,camera['xyxy'],width=width,height=height,min_iou=spec.min_iou)
                    unique.update(evidence)
                    predictions = [assigned[k] for k in keys]
                    for window in ('full','first' if frame<split else 'second'):
                        counts[name][window].update(gt,predictions,mask)
                        slots[name][window].append((gt,predictions,mask))
                    if name == 'disabled':
                        iou = pairwise_iou(clip_boxes([truth[g] for g in gt],width,height),clip_boxes(camera['xyxy'],width,height))
                        require(np.array_equal(iou>=spec.min_iou,mask),'Local/global gates differ')
                        local[c].update(gt,ids,np.where(iou>=spec.min_iou,1.-iou,np.nan),frameid=frame)
                require(seen == set(assigned), 'Observation lost in evaluation')
                for event in identity['merge_events']:
                    labels = [unique.get(ObservationKey(**key)) for key in event['members']]
                    kind = ('unannotated_frame' if not within else 'unresolved' if not labels or any(x is None for x in labels)
                            else 'all_visible_members_same_gt' if len(set(labels))==1 else 'different_known_gt')
                    merges[name][kind] += 1
        require(stream.readline() == '', 'Trailing experiment frames')
    metrics,mappings = {},{}
    for name in VARIANTS:
        metrics[name],mappings[name] = {},{}
        for window,bounds in intervals.items():
            value,mapping = counts[name][window].result()
            check_reference(value,reference_metrics(slots[name][window]))
            require(value['camera_time_slots'] == (bounds[1]-bounds[0]+1)*len(scene.camera_ids), 'Dropped evaluation slots')
            metrics[name][window],mappings[name][window] = value,mapping
        for field in ('camera_time_slots','gt_observations','predicted_observations'):
            require(metrics[name]['first'][field]+metrics[name]['second'][field] == metrics[name]['full'][field],
                    'Window denominators do not partition the full run')
        print(f'{name}: full and disjoint-window global metrics/motmetrics VERIFIED',flush=True)
    names = ['num_frames','num_objects','num_predictions','idtp','idfp','idfn','idf1','idp','idr',
             'precision','recall','num_switches','num_false_positives','num_misses']
    table = mm.metrics.create().compute_many([local[c] for c in scene.camera_ids],metrics=names,
        names=[f'camera_{c:04d}' for c in scene.camera_ids],generate_overall=True)
    local_metrics = json.loads(table.to_json(orient='index'))
    for window in intervals:
        for field in ('camera_time_slots','gt_observations','predicted_observations'):
            require(metrics['disabled'][window][field] == metrics['enabled'][window][field], 'Variant denominators differ')
    pooled = local_metrics['OVERALL']; full = metrics['disabled']['full']
    require(pooled['num_objects']==full['gt_observations'] and pooled['num_predictions']==full['predicted_observations'],
            'Local/global denominators differ')
    return {'global':metrics,'local_shared':local_metrics,'windows':intervals,
            'accepted_merge_diagnostics':{n:dict(v) for n,v in merges.items()}},mappings


def diagnose_returns(trace,scene,spec,ground):
    """Offline last-visible-label continuity, not archive-feature purity or truth input."""
    previous={};counts=Counter();details=[]
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for line in stream:
            row=json.loads(line);frame=row['frame_index'];data=row['variants']['enabled']
            unique={}
            if spec.first_frame<=frame<=spec.last_frame:
                for camera in data['cameras']:
                    c=camera['camera'];width,height=sizes[c]
                    keys=tuple(ObservationKey(c,i,frame) for i in camera['local_ids'])
                    _,_,labels,_,_=spatial_slot(ground.slots[frame,c],keys,camera['xyxy'],
                        width=width,height=height,min_iou=spec.min_iou)
                    unique.update(labels)
            for event in data['reactivations']:
                gid=event['global_id'];prior=previous.get(gid)
                current_labels=[unique.get(ObservationKey(**k)) for k in event['members']]
                old_labels=prior['labels'] if prior else []
                if not old_labels or not current_labels or any(x is None for x in (*old_labels,*current_labels)):
                    category='unresolved'
                elif len(set(old_labels))!=1 or len(set(current_labels))!=1:
                    category='mixed_previous_or_current'
                elif old_labels[0]==current_labels[0]:category='same_last_visible_gt'
                else:category='different_last_visible_gt'
                counts[category]+=1
                details.append({'frame_index':frame,'timestamp':row['timestamp'],**event,'category':category,
                    'previous_visible_evidence':prior,'current_labels':current_labels})
            visible=defaultdict(list)
            for assignment in data['identity']['assignments']:
                visible[assignment['global_id']].append(ObservationKey(**assignment['key']))
            for gid,keys in visible.items():
                previous[gid]={'frame_index':frame,'members':[asdict(k) for k in keys],
                    'labels':[unique.get(k) for k in keys]}
    return dict(counts),details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-report',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,default=ROOT/'configs/association/dormant_recovery_experiment.json')
    args=parser.parse_args();inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve();digest=sha256(path)
        require(expected is None or digest==expected,'Changed input: '+str(path))
        inputs[name]={'path':str(path),'sha256':digest};return path
    print('Verifying disabled bridge gate, frozen staged trace, candidate embeddings and code...',flush=True)
    gate=json.loads(checked('bridge_report',args.bridge_report).read_text())
    gate_checks=('both_local_traces_exact','both_global_records_exact_except_run_scope',
                 'refinement_decisions_and_counts_exact','original_parity_summary_exact',
                 'bridge_lifecycle_matches_baseline','no_archive_or_reactivation_when_disabled','frozen_inputs_unchanged')
    require(gate.get('completed') is True and gate.get('passed') is True
            and gate['protocol']=='dormant_registry_disabled_parity_v1'
            and gate['configuration']['recovery_enabled'] is False
            and all(gate['checks'].get(name) is True for name in gate_checks),'Invalid disabled bridge parity gate')
    for name,spec in gate['inputs'].items():
        require(name!='ground_truth','Unexpected raw GT in bridge parity inputs')
        checked(name,spec['path'],spec['sha256'])
    for package in ('supervision','numpy','scipy'):
        require(version(package)==gate['versions'][package],'Changed dependency: '+package)
    loaded=load_scene(inputs['scene_config']['path'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    policy=json.loads(Path(inputs['frozen_policy']['path']).read_text())
    require(policy==gate['configuration']['frozen_policy'],'Frozen policy differs')
    config=json.loads(checked('experiment_configuration',args.configuration).read_text())
    expected_archive={'max_age_seconds':'5','min_similarity':.8,'min_margin':.05,'max_speed':0.,
                      'position_slack':2.,'max_identities':128}
    require(config['schema_version']==1 and config['experiment']=='dormant_birth_recovery_fixed_radius_v1'
            and config['development_scene']==scene.scene=='MTMC_Tracking_2024/train/scene_001'
            and scene.camera_ids==(4,5,8) and scene.fps==30 and scene.rounds==3600
            and config['local_variant']=='staged' and config['history_update_policy']=='all_updates'
            and config['runtime_frames']==[0,3599] and config['evaluation_frames']==[spec.first_frame,spec.last_frame]==[2,3599]
            and config['variants']==list(VARIANTS) and config['archive']==expected_archive
            and config['spatial_policy']=='fixed_radius_in_native_units_no_motion_prediction'
            and config['recovery_scope']=='wholly_new_unanchored_groups_at_birth_only'
            and config['gt_used_during_replay'] is False and config['threshold_search'] is False,
            'This runner requires the declared single development hypothesis')
    require(policy['global']['appearance_variant']=='mean' and policy['global']['history']['max_observations']==8
            and Fraction(policy['global']['history']['max_age_seconds'])==1
            and config['archive']['position_slack']==policy['global']['geometry']['max_distance']
            and spec.min_iou==.5 and spec.gt_to_video_offset==0,'History/geometry/evaluation policy differs')
    source,cache,local,reference=[json.loads(Path(inputs[n]['path']).read_text()) for n in
        ('pipeline_report','cache_report','local_report','reference_global_report')]
    require(reference['source_run_id']==local['source_run_id']==cache['source_run_id']==source['run_id']
            and reference['local_experiment_run_id']==local['run_id'] and local['cache_run_id']==cache['run_id']
            and reference['configuration']['global']==policy['global']
            and reference['inputs']['ground_truth']['sha256']==spec.ground_truth.sha256,'Mixed source/evaluation lineage')
    for package in ('numpy','scipy','motmetrics','pandas'):
        require(version(package)==local['versions'][package],'Changed evaluation dependency: '+package)
    vectors=np.load(inputs['embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(vectors.dtype==np.float32 and vectors.shape==(cache['summary']['encoded'],512),'Invalid embedding archive')
    for rel in ('src/mtmc/pipeline/recovery.py','scripts/check_recovery_evidence.py',
                'scripts/experiment_dormant_recovery.py','scripts/evaluate_global_identity.py','src/mtmc/data/ground_truth.py'):
        checked('experiment_code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/dormant_recovery'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    try:
        print('Fixed experimental archive settings:',json.dumps(config['archive']),flush=True)
        print('Phase 1: two causal registry variants; fixed local tracks/history/geometry; no GT...',flush=True)
        summary=replay(scene,policy,run_id=run,source_run=source['run_id'],cache=cache,reference=reference,
            paths={k:v['path'] for k,v in inputs.items()},vectors=vectors,output=output,settings=archive_settings(config['archive']))
        outputs=[output/name for name in ('global_tracks.jsonl.gz','registry_audit.jsonl.gz','history_provenance.jsonl.gz')]
        frozen={p.name:sha256(p) for p in outputs}
        frozen_path=output/'predictions_frozen.json'
        frozen_path.write_text(json.dumps({'run_id':run,'configuration':config,'global_policy':policy['global'],
            'sha256':frozen},indent=2,allow_nan=False)+'\n')
        print('Phase 2: outputs frozen; offline full/window shared identity evaluation...',flush=True)
        ground=load_ground_truth(spec);checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mapping=evaluate(outputs[0],scene,spec,ground,run_id=run,split=1800)
        require(quality['global']['disabled']['full']==reference['metrics']['staged'],'Control full metrics differ')
        require(quality['accepted_merge_diagnostics']['disabled']==reference['accepted_merge_diagnostics']['staged'],
                'Control merge diagnostics differ')
        for camera,values in local['metrics']['staged'].items():
            for key,expected in values.items():
                actual=quality['local_shared'][camera][key]
                require(actual==expected or (actual is not None and expected is not None and abs(actual-expected)<=1e-9),
                        f'Local metric changed: {camera}/{key}')
        diagnostics,events=diagnose_returns(outputs[0],scene,spec,ground)
        require(sum(diagnostics.values())==summary['lifecycle']['enabled']['reactivation_events'],'Return diagnostic counts differ')
        for name in VARIANTS:
            require(sum(quality['accepted_merge_diagnostics'][name].values())==summary['lifecycle'][name]['merge_events'],
                    'Merge diagnostic counts differ')
        for name,digest in frozen.items():require(sha256(output/name)==digest,'Frozen output changed during evaluation')
        for asset in inputs.values():require(sha256(asset['path'])==asset['sha256'],'Input changed during experiment')
        mapping_path=output/'identity_matching.json';mapping_path.write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
        events_path=output/'reactivation_diagnostics.json';events_path.write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
        report={'completed':True,'protocol':'dormant_recovery_paired_v1','run_id':run,'scene':scene.scene,'inputs':inputs,
            'configuration':config,'global_policy':policy['global'],'summary':summary,**quality,
            'reactivation_last_visible_diagnostics':diagnostics,
            'checks':{'disabled_full_records_exact_except_run_scope':True,'disabled_metrics_lifecycle_merges_exact':True,
                'same_local_observations_history_and_groups':True,'key_vector_time_projection_provenance':True,
                'full_window_metrics_motmetrics_agree':True,'local_metrics_unchanged':True,
                'predictions_frozen_before_GT':True,'all_reactivations_diagnosed':True,'frozen_inputs_outputs_unchanged':True},
            'versions':{n:version(n) for n in ('supervision','numpy','scipy','motmetrics','pandas')},
            'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in (*outputs,frozen_path,mapping_path,events_path)},
            'limits':['One fixed hypothesis on reused training-scene observations; no independent-validation claim.',
                'Fixed native-coordinate return radius, no calibrated human speed or meter assumption.',
                'Birth-time recovery only; ambiguous births are not retried as already anchored tracks.',
                'Single-round return decision, no active-ID switch repair or automatic wrong-return correction.',
                'Last-visible GT continuity is an offline diagnostic, not a certificate of archive-feature purity.',
                'Cumulative inactive counts differ from the bounded archive size; superseded IDs consume allocator slots.',
                'No models or decoding; no end-to-end speed measurement or runtime policy promotion.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,
            'error':str(error)})+'\n');raise
    for window,bounds in quality['windows'].items():
        print(f'Window {window}: frames {bounds}')
        print('Variant       Global IDF1     IDTP     IDFP     IDFN')
        for name in VARIANTS:
            m=quality['global'][name][window]
            print(f'{name:12} {100*m["idf1"]:11.2f}% {m["idtp"]:8} {m["idfp"]:8} {m["idfn"]:8}')
    print('Lifecycle:',json.dumps(summary['lifecycle']))
    print('Return decisions:',json.dumps(summary['return_decisions']))
    print('Archive events:',json.dumps(summary['archive_events']))
    print('Last-visible GT diagnostics:',json.dumps(diagnostics))
    print(f'Report: {path}')
    print('Dormant recovery paired experiment: COMPLETED; runtime baseline unchanged')


if __name__=='__main__':main()

