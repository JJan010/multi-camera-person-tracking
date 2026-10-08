"""Paired causal segment-policy replay on frozen staged tracks, then offline GT."""
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

from check_frozen_bytetrack_replay import candidate_arrays
from evaluate_appearance_global import direct_observations, lifecycle
from experiment_dormant_recovery import evaluate
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.pipeline.core import IdentityStage
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.osnet import ObservationKey
from mtmc.tracking.continuity import AppearanceContinuity, ContinuitySettings
from mtmc.tracking.segments import SegmentRound
from run_mtmc import dump

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('disabled', 'enabled')


def settings_from_configuration(config):
    raw = dict(config['continuity'])
    for name in ('max_reference_age','reference_sample_interval','min_support_seconds','max_support_gap'):
        raw[name] = Fraction(raw[name])
    return ContinuitySettings(**raw)


def evaluation_view(identity, segmented):
    """A separate projection for GT joins. Full internal runtime record is saved too."""
    def source(key):
        return asdict(segmented.source_key(key))
    return {'run_id': identity.run_id, 'frame_index': identity.frame_index,
            'timestamp': str(identity.timestamp), 'representation': 'original_observation_keys_v1',
            'assignments': [{'key':source(a.key),'global_id':a.global_id,'reason':a.reason}
                            for a in identity.assignments],
            'merge_events': [{'canonical_global_id':e.canonical_global_id,
                             'absorbed_global_ids':list(e.absorbed_global_ids),
                             'frame_index':e.frame_index,'timestamp':str(e.timestamp),
                             'members':[source(k) for k in e.members]} for e in identity.merge_events]}


def replay(scene, policy, *, run_id, source_run, cache, reference, paths, vectors, output, settings):
    cfg = policy['global']
    owners = {n: AppearanceContinuity(run_id+'/'+n, scene.camera_ids, enabled=n=='enabled', settings=settings)
              for n in VARIANTS}
    histories = {n: AppearanceHistory(run_id+'/'+n, max_observations=cfg['history']['max_observations'],
                                    max_age=Fraction(cfg['history']['max_age_seconds'])) for n in VARIANTS}
    stages = {n: IdentityStage(run_id+'/'+n,scene.matrices,scene.coordinate_space,
                  variant=cfg['appearance_variant'], threshold=cfg['appearance_threshold'],
                  **cfg['geometry'],identity_configuration=cfg['identity']) for n in VARIANTS}
    counts={n:Counter() for n in VARIANTS};emitted={n:set() for n in VARIANTS}
    decisions=Counter();resets=Counter();peaks=Counter();split_count=0
    next_row=candidate_count=outside_count=observations=0
    with Path(paths['tracks']).open() as original_stream, Path(paths['detections.jsonl']).open() as cache_stream, \
            gzip.open(paths['global_variants'],'rt') as reference_stream, \
            gzip.open(output/'global_tracks.jsonl.gz','xt') as saved, \
            gzip.open(output/'continuity_audit.jsonl.gz','xt') as audit:
        for frame in range(scene.rounds):
            lines=[stream.readline() for stream in (original_stream,cache_stream,reference_stream)]
            require(all(lines),'Truncated frozen source')
            original,cached,prior=map(json.loads,lines);time=Fraction(frame,scene.fps)
            require(prior['run_id']==reference['run_id'] and prior['frame_index']==frame
                    and prior['timestamp']==str(time),'Frozen reference scope differs')
            arrays,_,next_row,outside=candidate_arrays(cached,original,frame=frame,cache_run=cache['run_id'],
                                                      source_run=source_run,next_row=next_row)
            candidate_count+=sum(len(a[0]) for a in arrays.values());outside_count+=outside
            cameras=prior['variants']['staged']['cameras']
            records,features=direct_observations(cameras,cached,vectors,frame)
            observations+=len(records);variants={};audits={}
            for name in VARIANTS:
                processed=owners[name].update(SegmentRound(run_id+'/'+name,frame,time,records,features))
                segmented=processed.segmented
                require(segmented.features.timestamps==features.timestamps
                        and np.array_equal(segmented.features.embeddings,features.embeddings)
                        and tuple(segmented.source_key(k) for k in segmented.features.keys)==features.keys,
                        'Source feature rows changed during segmentation')
                require(len(segmented.records)==len(records),'Observation count changed')
                for current,source in zip(segmented.records,records):
                    require(segmented.source_key(current.key)==source.key
                            and {k:v for k,v in asdict(current).items() if k!='key'}
                            == {k:v for k,v in asdict(source).items() if k!='key'}, 'Source box/score/crop changed')
                if name=='disabled':
                    require(segmented.records==records and segmented.features.keys==features.keys
                            and not segmented.events and not processed.decisions
                            and owners[name].stored_vectors==0,'Disabled policy is not pass-through')
                history=histories[name].update(frame,time,segmented.features)
                for event in segmented.events:
                    i=history.mean.keys.index(event.identity_key)
                    require(history.source_frames[i]==(frame,),'New segment inherited prior history samples')
                identity,*_=stages[name].update(frame,time,history.mean,segmented.records)
                runtime=json.loads(dump(asdict(identity)))
                if name=='disabled':
                    require(runtime=={**prior['variants']['staged']['identity'],'run_id':run_id+'/'+name},
                            f'Disabled complete identity record differs: frame {frame}')
                lifecycle(runtime,counts[name],emitted[name])
                view=evaluation_view(identity,segmented)
                require({ObservationKey(**a['key']) for a in view['assignments']}=={r.key for r in records}
                        and len(view['assignments'])==len(records), 'Evaluation projection lost source observations')
                variants[name]={'cameras':cameras,'identity':view,'identity_runtime':runtime,
                                'segment_bindings':[asdict(b) for b in segmented.bindings]}
                audits[name]={'decisions':[asdict(d) for d in processed.decisions],
                              'resets':[asdict(r) for r in processed.resets],
                              'segment_events':[asdict(e) for e in segmented.events]}
                if name=='enabled':
                    decisions.update(d.outcome for d in processed.decisions)
                    resets.update(r.reason for r in processed.resets)
                    split_count+=len(segmented.events)
                    peaks['vectors']=max(peaks['vectors'],owners[name].stored_vectors)
                    peaks['pending_tracks']=max(peaks['pending_tracks'],owners[name].pending_tracks)
                    peaks['original_tracks']=max(peaks['original_tracks'],owners[name].segmenter.known_tracks)
            header={'run_id':run_id,'frame_index':frame,'timestamp':str(time)}
            saved.write(dump({**header,'variants':variants})+'\n')
            audit.write(dump({**header,'variants':audits})+'\n')
            if (frame+1)%300==0 or frame+1==scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; disabled full control EXACT; splits={split_count}',flush=True)
        require(all(stream.readline()=='' for stream in (original_stream,cache_stream,reference_stream)), 'Trailing source frames')
    require(next_row==len(vectors)==cache['summary']['encoded'] and candidate_count==cache['summary']['detections']
            and outside_count==cache['summary']['fully_outside'],'Frozen candidate coverage differs')
    lifecycle_counts={n:{**dict(counts[n]),'ever_emitted_ids':len(emitted[n])} for n in VARIANTS}
    require(lifecycle_counts['disabled']==reference['lifecycle']['staged'],'Disabled lifecycle differs')
    require(all(c['observations']==observations for c in lifecycle_counts.values()),'Runtime denominators differ')
    require(split_count==decisions['split_confirmed']==owners['enabled'].segmenter.allocated_segments,
            'Split/event/generation accounting differs')
    return {'lifecycle':lifecycle_counts,'split_events':split_count,'decision_counts':dict(decisions),
            'reset_counts':dict(resets),'policy_peaks':dict(peaks),'observations':observations,
            'candidate_observations':candidate_count,'candidate_embedding_rows':next_row}


def diagnose_splits(trace, audit, scene, spec, ground, *, run_id, expected_events):
    """Compare actual prior-reference GT evidence to confirmation/current evidence offline."""
    events=[]; wanted=set()
    with gzip.open(audit,'rt') as stream:
        for frame,line in enumerate(stream):
            row=json.loads(line)
            require(row['run_id']==run_id and row['frame_index']==frame
                    and row['timestamp']==str(Fraction(frame,scene.fps)), 'Audit scope differs')
            data=row['variants']['enabled'];by_key={ObservationKey(**d['key']):d for d in data['decisions']}
            for event in data['segment_events']:
                k=ObservationKey(**event['source_key']);d=by_key[k]
                require(d['outcome']=='split_confirmed','Segment event lacks confirmed decision')
                first=(Fraction(row['timestamp'])-Fraction(d['support_span']))*scene.fps
                require(first.denominator==1,'Nonintegral support start frame');first=int(first)
                require(len(d['reference_frames'])==len(d['reference_times'])>0
                        and all(f<first for f in d['reference_frames']), 'Noncausal reference provenance')
                for f,t in zip(d['reference_frames'],d['reference_times']):
                    require(Fraction(t)==Fraction(f,scene.fps),'Reference source time differs')
                previous=event['previous_identity_key']['frame_index']
                frames=set(d['reference_frames'])|set(range(first,frame+1))|{previous}
                wanted.update((f,k.camera_id,k.local_id) for f in frames)
                events.append({'frame_index':frame,'timestamp':row['timestamp'],**event,
                               'decision':d,'first_support_frame':first})
    require(len(events)==expected_events,'Audit event count differs')
    needed_slots={(f,c) for f,c,_ in wanted};evidence={}
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for frame,line in enumerate(stream):
            if not any((frame,c) in needed_slots for c in scene.camera_ids):continue
            row=json.loads(line);require(row['run_id']==run_id and row['frame_index']==frame,'Trace scope differs')
            data=row['variants']['enabled']
            maps={n:{ObservationKey(**a['key']):a['global_id'] for a in row['variants'][n]['identity']['assignments']}
                  for n in VARIANTS}
            for camera in data['cameras']:
                c=camera['camera']
                if (frame,c) not in needed_slots:continue
                keys=tuple(ObservationKey(c,i,frame) for i in camera['local_ids'])
                labels={}
                if spec.first_frame<=frame<=spec.last_frame:
                    width,height=sizes[c]
                    _,_,labels,_,_=spatial_slot(ground.slots[frame,c],keys,camera['xyxy'],
                                                width=width,height=height,min_iou=spec.min_iou)
                for k in keys:
                    if (frame,c,k.local_id) in wanted:
                        evidence[frame,c,k.local_id]={'frame_index':frame,'unique_gt':labels.get(k),
                            'enabled_global_id':maps['enabled'][k],'disabled_global_id':maps['disabled'][k]}
    categories=Counter();details=[]
    for event in events:
        k=ObservationKey(**event['source_key']);d=event['decision']
        def at(frame):
            require((frame,k.camera_id,k.local_id) in evidence,'Missing source observation in split evidence')
            return evidence[frame,k.camera_id,k.local_id]
        refs=[at(f) for f in d['reference_frames']]
        current=at(k.frame_index);previous=at(event['previous_identity_key']['frame_index'])
        known=[r['unique_gt'] for r in refs if r['unique_gt'] is not None]
        if current['unique_gt'] is None or len(known)!=len(refs):category='unresolved_reference_or_current'
        elif len(set(known))>1:category='mixed_reference_gt'
        elif known[0]==current['unique_gt']:category='same_reference_and_current_gt'
        else:category='different_reference_and_current_gt'
        # Omitted intermediate calls are not supported by this consecutive-frame runner.
        support=[at(f) for f in range(event['first_support_frame'],k.frame_index+1)]
        categories[category]+=1
        details.append({**event,'diagnostic_category':category,'reference_evidence':refs,
                        'confirmation_evidence':support,'previous_observation':previous,'current_observation':current})
    return dict(categories),details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-report',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,default=ROOT/'configs/tracking/appearance_continuity_experiment.json')
    args=parser.parse_args();inputs={}
    def checked(name,path,expected=None):
        path=Path(path).resolve();digest=sha256(path)
        require(expected is None or digest==expected,'Changed input: '+str(path))
        inputs[name]={'path':str(path),'sha256':digest};return path
    print('Verifying frozen runtime gate, staged reference and one declared continuity policy; no GT...',flush=True)
    gate=json.loads(checked('bridge_report',args.bridge_report).read_text())
    names=('both_local_traces_exact','both_global_records_exact_except_run_scope','refinement_decisions_and_counts_exact',
           'original_parity_summary_exact','bridge_lifecycle_matches_baseline','no_archive_or_reactivation_when_disabled','frozen_inputs_unchanged')
    require(gate.get('completed') is True and gate.get('passed') is True
            and gate['protocol']=='dormant_registry_disabled_parity_v1'
            and gate['configuration']['recovery_enabled'] is False
            and all(gate['checks'].get(n) is True for n in names),'Invalid runtime parity gate')
    for name,item in gate['inputs'].items():
        require(name!='ground_truth','Raw GT should not be in runtime parity inputs')
        checked(name,item['path'],item['sha256'])
    for package in ('supervision','numpy','scipy'):
        require(version(package)==gate['versions'][package],'Dependency changed: '+package)
    loaded=load_scene(inputs['scene_config']['path'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    policy=json.loads(Path(inputs['frozen_policy']['path']).read_text())
    require(policy==gate['configuration']['frozen_policy'],'Frozen global/local policy differs')
    config=json.loads(checked('experiment_configuration',args.configuration).read_text())
    settings=settings_from_configuration(config)
    require(settings==ContinuitySettings(8,Fraction(3),Fraction(1,5),3,.5,.6,.8,3,Fraction(1,5),Fraction(1,10)),
            'Runner requires the declared single continuity hypothesis')
    require(config['schema_version']==1 and config['experiment']=='sparse_reference_confirmed_discontinuity_v1'
            and config['development_scene']==scene.scene=='MTMC_Tracking_2024/train/scene_001'
            and scene.camera_ids==(4,5,8) and scene.fps==30 and scene.rounds==3600
            and config['local_variant']=='staged' and config['global_history_update_policy']=='all_updates'
            and config['runtime_frames']==[0,3599] and config['evaluation_frames']==[spec.first_frame,spec.last_frame]==[2,3599]
            and config['variants']==list(VARIANTS) and config['gt_used_during_replay'] is False
            and config['threshold_search'] is False and config['dormant_recovery_enabled'] is False,
            'Unexpected experiment scope')
    require(policy['global']['appearance_variant']=='mean' and policy['global']['history']['max_observations']==8
            and Fraction(policy['global']['history']['max_age_seconds'])==1
            and spec.min_iou==.5 and spec.gt_to_video_offset==0,'History/evaluation differs')
    source,cache,local,reference=[json.loads(Path(inputs[n]['path']).read_text()) for n in
        ('pipeline_report','cache_report','local_report','reference_global_report')]
    require(reference['source_run_id']==local['source_run_id']==cache['source_run_id']==source['run_id']
            and reference['local_experiment_run_id']==local['run_id'] and local['cache_run_id']==cache['run_id']
            and reference['configuration']['global']==policy['global']
            and reference['inputs']['ground_truth']['sha256']==spec.ground_truth.sha256,'Mixed experiment lineage')
    for package in ('numpy','scipy','motmetrics','pandas'):
        require(version(package)==local['versions'][package],'Evaluation dependency changed: '+package)
    vectors=np.load(inputs['embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(vectors.dtype==np.float32 and vectors.shape==(cache['summary']['encoded'],512),'Invalid embedding archive')
    for rel in ('src/mtmc/tracking/segments.py','src/mtmc/tracking/continuity.py',
                'scripts/experiment_appearance_continuity.py','scripts/check_track_segments.py',
                'scripts/check_appearance_continuity.py','scripts/experiment_dormant_recovery.py',
                'scripts/evaluate_global_identity.py','src/mtmc/data/ground_truth.py'):
        checked('experiment_code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/appearance_continuity_experiment'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    try:
        print('Phase 1: two causal CPU replays; unchanged local observations; no models/decoding/GT...',flush=True)
        summary=replay(scene,policy,run_id=run,source_run=source['run_id'],cache=cache,reference=reference,
            paths={k:v['path'] for k,v in inputs.items()},vectors=vectors,output=output,settings=settings)
        outputs=[output/n for n in ('global_tracks.jsonl.gz','continuity_audit.jsonl.gz')]
        frozen={p.name:sha256(p) for p in outputs}
        frozen_path=output/'predictions_frozen.json'
        frozen_path.write_text(json.dumps({'run_id':run,'configuration':config,'global_policy':policy['global'],
                                          'sha256':frozen},indent=2,allow_nan=False)+'\n')
        print('Phase 2: predictions frozen; offline full/window identity evaluation...',flush=True)
        ground=load_ground_truth(spec);checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mappings=evaluate(outputs[0],scene,spec,ground,run_id=run,split=1800)
        require(quality['global']['disabled']['full']==reference['metrics']['staged'],'Disabled quality differs')
        require(quality['accepted_merge_diagnostics']['disabled']==reference['accepted_merge_diagnostics']['staged'],
                'Disabled merge diagnostics differ')
        for camera,values in local['metrics']['staged'].items():
            for field,expected in values.items():
                actual=quality['local_shared'][camera][field]
                require(actual==expected or (actual is not None and expected is not None and abs(actual-expected)<=1e-9),
                        'Original local metrics changed: '+camera+'/'+field)
        for n in VARIANTS:
            require(sum(quality['accepted_merge_diagnostics'][n].values())==summary['lifecycle'][n]['merge_events'],
                    'Merge diagnostic coverage differs')
        categories,events=diagnose_splits(outputs[0],outputs[1],scene,spec,ground,run_id=run,expected_events=summary['split_events'])
        for name,digest in frozen.items():require(sha256(output/name)==digest,'Frozen output changed during evaluation')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during experiment')
        matching_path=output/'identity_matching.json';matching_path.write_text(json.dumps(mappings,indent=2,allow_nan=False)+'\n')
        events_path=output/'split_diagnostics.json';events_path.write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
        report={'completed':True,'protocol':'appearance_continuity_paired_v1','run_id':run,'scene':scene.scene,
            'inputs':inputs,'configuration':config,'global_policy':policy['global'],'summary':summary,**quality,
            'split_reference_gt_diagnostics':categories,
            'checks':{'disabled_full_runtime_records_exact':True,'disabled_metrics_lifecycle_merges_exact':True,
                'original_local_metrics_unchanged':True,'boxes_scores_features_rows_unchanged':True,
                'new_segments_start_fresh_history':True,'source_key_projection_complete':True,
                'full_window_metrics_motmetrics_agree':True,'all_split_events_diagnosed':True,
                'predictions_frozen_before_GT':True,'frozen_inputs_outputs_unchanged':True},
            'versions':{p:version(p) for p in ('supervision','numpy','scipy','motmetrics','pandas')},
            'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in (*outputs,frozen_path,matching_path,events_path)},
            'limits':['One fixed uncalibrated hypothesis on reused training-scene data; no deployment promotion.',
                'Local quality uses unchanged ORIGINAL tracker IDs; internal segment-ID local quality is not reported.',
                'identity_runtime contains internal segment keys; identity is a separate source-key projection for evaluation.',
                'Split GT categories describe prior reference/current evidence, not full lifetime purity or causal benefit.',
                'Same-GT evidence suggests possible unnecessary fragmentation; unresolved events are not successes.',
                'Confirmation latency and pose/occlusion changes can cause missed switches or false cuts.',
                'Dormant recovery remains off; no models or end-to-end speed measurements.']}
        path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,'error':str(error)})+'\n')
        raise
    for window,bounds in quality['windows'].items():
        print(f'Window {window}: frames {bounds}')
        print('Variant       Global IDF1     IDTP     IDFP     IDFN')
        for n in VARIANTS:
            m=quality['global'][n][window]
            print(f'{n:12} {100*m["idf1"]:11.2f}% {m["idtp"]:8} {m["idfp"]:8} {m["idfn"]:8}')
    print('Lifecycle:',json.dumps(summary['lifecycle']))
    print('Split events:',summary['split_events'])
    print('Policy decisions:',json.dumps(summary['decision_counts']))
    print('Evidence resets:',json.dumps(summary['reset_counts']))
    print('Prior-reference/current GT diagnostics:',json.dumps(categories))
    print(f'Report: {path}')
    print('Appearance continuity paired experiment: COMPLETED; runtime baseline unchanged')


if __name__=='__main__':main()
