"""Paired global-history ablation on frozen scene_001 staged local tracks."""
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
from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference
from mtmc.data.ground_truth import load_ground_truth, spatial_slot, clip_boxes, pairwise_iou
from mtmc.data.scene import load_scene, require, sha256
from mtmc.pipeline.available_identity import AvailableIdentityStage
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.selective_history import SelectiveAppearanceHistory, confidence_update_mask
from mtmc.reid.osnet import ObservationKey
from run_mtmc import dump

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('all_updates', 'strong_updates')


def replay(scene, policy, *, run_id, source_run, cache, reference, paths, vectors, output, minimum_score):
    cfg = policy['global']
    histories = {n:SelectiveAppearanceHistory(run_id+'/'+n,max_observations=cfg['history']['max_observations'],
                 max_age=Fraction(cfg['history']['max_age_seconds'])) for n in VARIANTS}
    original = AppearanceHistory('control',max_observations=cfg['history']['max_observations'],
                                 max_age=Fraction(cfg['history']['max_age_seconds']))
    stages = {n:AvailableIdentityStage(run_id+'/'+n,scene.matrices,scene.coordinate_space,
        variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],**cfg['geometry'],
        identity_configuration=cfg['identity']) for n in VARIANTS}
    counters = {n:Counter() for n in VARIANTS}; emitted = {n:set() for n in VARIANTS}
    memory_fields = ('encoded_observations','unencoded_observations','available_descriptors',
                     'observations_without_descriptor','updated','reused','unavailable',
                     'accepted_updates','rejected_updates')
    memory_counts = {n:Counter({k:0 for k in memory_fields}) for n in VARIANTS}
    peaks = {n:Counter() for n in VARIANTS}
    next_row = candidate_count = outside_count = 0
    with Path(paths['tracks']).open() as src, Path(paths['detections.jsonl']).open() as det, \
            gzip.open(paths['global_variants'],'rt') as ref, \
            gzip.open(output/'global_tracks.jsonl.gz','xt') as saved, \
            gzip.open(output/'history_decisions.jsonl.gz','xt') as history_saved:
        for frame in range(scene.rounds):
            lines = [f.readline() for f in (src,det,ref)]
            require(all(lines), 'Truncated frozen source')
            original_row,candidates,previous = map(json.loads,lines)
            timestamp = Fraction(frame,scene.fps)
            require(previous['run_id'] == reference['run_id'] and previous['frame_index'] == frame
                    and previous['timestamp'] == str(timestamp), 'Global reference scope differs')
            arrays,_,next_row,outside = candidate_arrays(candidates,original_row,frame=frame,
                cache_run=cache['run_id'],source_run=source_run,next_row=next_row)
            candidate_count += sum(len(a[0]) for a in arrays.values()); outside_count += outside
            cameras = previous['variants']['staged']['cameras']
            records,features = direct_observations(cameras,candidates,vectors,frame)
            feature_keys = set(features.keys)
            scores = {r.key:r.confidence for r in records if r.key in feature_keys}
            masks = {'all_updates':{k:True for k in features.keys},
                     'strong_updates':confidence_update_mask(features,scores,minimum_score=minimum_score)}
            old_history = original.update(frame,timestamp,features)
            current,history_audit = {}, {}
            for name in VARIANTS:
                history = histories[name].update(frame,timestamp,features,accept_update=masks[name])
                if name == 'all_updates':
                    require(history.mean.keys == old_history.mean.keys
                            and np.array_equal(history.mean.embeddings,old_history.mean.embeddings)
                            and tuple(d.source_frames for d in history.decisions) == old_history.source_frames,
                            'All-accepted descriptor/source parity differs')
                available = set(history.mean.keys)
                missing = tuple(r.key for r in records if r.key not in available)
                identity,*_ = stages[name].update(frame,timestamp,history.mean,records,unavailable_keys=missing)
                identity = json.loads(dump(asdict(identity)))
                current[name] = {'cameras':cameras,'identity':identity}
                lifecycle(identity,counters[name],emitted[name])
                counts = memory_counts[name]
                counts['encoded_observations'] += len(features.keys)
                counts['unencoded_observations'] += len(records)-len(features.keys)
                counts['available_descriptors'] += len(available)
                counts['observations_without_descriptor'] += len(missing)
                for decision in history.decisions:
                    counts[decision.status] += 1
                    counts['accepted_updates'] += int(decision.accepted_update)
                    counts['rejected_updates'] += int(not decision.accepted_update)
                    require(all(t <= timestamp and timestamp-t <= histories[name].max_age for t in decision.source_times),
                            'Noncausal or expired source sample')
                peaks[name]['tracks'] = max(peaks[name]['tracks'],histories[name].active_tracks)
                peaks[name]['vectors'] = max(peaks[name]['vectors'],histories[name].stored_vectors)
                history_audit[name] = [asdict(d) for d in history.decisions]
            expected = previous['variants']['staged']['identity']
            require(expected['run_id'] == reference['run_id']+'/staged', 'Reference identity scope differs')
            require(current['all_updates']['identity'] == {**expected,'run_id':run_id+'/all_updates'},
                    f'Control global records differ at frame {frame}')
            saved.write(dump({'run_id':run_id,'source_reference_run':reference['run_id'],'frame_index':frame,
                              'timestamp':str(timestamp),'variants':current})+'\n')
            history_saved.write(dump({'run_id':run_id,'frame_index':frame,'timestamp':str(timestamp),
                                      'variants':history_audit})+'\n')
            if (frame+1)%300 == 0 or frame+1 == scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; all-updates control EXACT',flush=True)
        require(all(f.readline()=='' for f in (src,det,ref)), 'Trailing source frames')
    require(next_row == len(vectors) == cache['summary']['encoded']
            and candidate_count == cache['summary']['detections']
            and outside_count == cache['summary']['fully_outside'], 'Candidate coverage differs')
    for name in VARIANTS:
        counters[name]['ever_emitted_ids'] = len(emitted[name])
        counts = memory_counts[name]
        require(counts['encoded_observations'] == counts['accepted_updates']+counts['rejected_updates']
                == counts['updated']+counts['reused']+counts['unavailable'], 'History accounting differs')
        require(counts['available_descriptors'] == counts['updated']+counts['reused']
                and counts['observations_without_descriptor'] == counts['unavailable']+counts['unencoded_observations']
                and counts['encoded_observations']+counts['unencoded_observations'] == counters[name]['observations'],
                'Descriptor/track accounting differs')
    require(dict(counters['all_updates']) == reference['lifecycle']['staged'], 'Control lifecycle differs')
    require(counters['all_updates']['observations'] == counters['strong_updates']['observations'], 'Observations were dropped')
    return {'lifecycle':{n:dict(c) for n,c in counters.items()},
            'history':{n:dict(c) for n,c in memory_counts.items()},'history_peaks':{n:dict(c) for n,c in peaks.items()},
            'candidate_observations':candidate_count,'candidate_embedding_rows':next_row}


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
            require(row['variants']['all_updates']['cameras'] == row['variants']['strong_updates']['cameras'],
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
                    if name == 'all_updates':
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
            require(metrics['all_updates'][window][field] == metrics['strong_updates'][window][field], 'Variant denominators differ')
    pooled = local_metrics['OVERALL']; full = metrics['all_updates']['full']
    require(pooled['num_objects']==full['gt_observations'] and pooled['num_predictions']==full['predicted_observations'],
            'Local/global denominators differ')
    return {'global':metrics,'local_shared':local_metrics,'windows':intervals,
            'accepted_merge_diagnostics':{n:dict(v) for n,v in merges.items()}},mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-report',type=Path,required=True)
    parser.add_argument('--configuration',type=Path,default=ROOT/'configs/reid/selective_history_experiment.json')
    args = parser.parse_args(); inputs = {}
    def checked(name,path,expected=None):
        path = Path(path).resolve(); digest = sha256(path)
        require(expected is None or digest == expected, f'Changed input: {path}')
        inputs[name] = {'path':str(path),'sha256':digest}; return path
    print('Verifying frozen scene, candidate cache, staged reference and code checksums...',flush=True)
    gate_path = checked('parity_report',args.parity_report)
    gate = json.loads(gate_path.read_text())
    required = ('both_local_traces_exact','both_global_records_exact_except_run_scope',
                'refinement_decisions_and_counts_exact','candidate_provenance_and_coverage','lifecycle_exact')
    require(gate.get('completed') is True and gate.get('passed') is True and gate['protocol']=='paired_scene_runtime_parity_v1'
            and all(gate['checks'].get(k) is True for k in required), 'Expected verified scene runtime parity report')
    # The gate contains runtime assets, reports and code, not a raw GT input.
    for name,asset in gate['inputs'].items():
        require(name != 'ground_truth', 'Unexpected raw GT in runtime parity inputs')
        checked(name,ROOT/name[5:] if name.startswith('code:') else asset['path'],asset['sha256'])
    for name,digest in gate['code_sha256'].items(): checked('code:'+name,ROOT/name,digest)
    for package in ('numpy','scipy'): require(version(package)==gate['versions'][package],f'Changed dependency: {package}')
    require(version('motmetrics')=='1.4.0','Expected motmetrics 1.4.0')
    config_path = checked('experiment_configuration',args.configuration)
    config = json.loads(config_path.read_text())
    loaded = load_scene(inputs['scene_config']['path'],project_root=ROOT); scene,spec = loaded.runtime,loaded.evaluation
    policy = json.loads(Path(inputs['frozen_policy']['path']).read_text())
    require(policy==gate['configuration'],'Policy differs from verified gate')
    require(scene.scene==config['development_scene']=='MTMC_Tracking_2024/train/scene_001'
            and scene.camera_ids==(4,5,8) and scene.fps==30 and scene.rounds==3600
            and all((c.width,c.height)==(1920,1080) for c in scene.cameras),'Expected two-minute development scene')
    require(config['experiment']=='confidence_only_global_history_v1' and config['local_variant']=='staged'
            and config['runtime_frames']==[0,scene.rounds-1] and config['evaluation_frames']==[spec.first_frame,spec.last_frame]
            and config['descriptor_variant']==policy['global']['appearance_variant']=='mean'
            and config['max_observations']==policy['global']['history']['max_observations']==8
            and Fraction(config['max_age_seconds'])==Fraction(policy['global']['history']['max_age_seconds'])==1
            and config['variants']=={'all_updates':{'update_policy':'accept_all'},
                'strong_updates':{'update_policy':'confidence_at_least','minimum_score':.5}}
            and policy['local']['tracker']['track_activation_threshold']==.5, 'Experiment policy differs')
    reference = json.loads(Path(inputs['reference_global_report']['path']).read_text())
    cache = json.loads(Path(inputs['cache_report']['path']).read_text())
    local = json.loads(Path(inputs['local_report']['path']).read_text())
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    require(reference.get('completed') is True and reference['protocol']=='competitive_local_to_global_paired_v1'
            and local.get('completed') is True and cache.get('completed') is True
            and reference['configuration']['global']==policy['global']
            and reference['source_run_id']==cache['source_run_id']==local['source_run_id']==source['run_id']
            and reference['local_experiment_run_id']==local['run_id'] and local['cache_run_id']==cache['run_id'], 'Mixed reference lineage')
    require(reference['inputs']['ground_truth']['sha256']==spec.ground_truth.sha256
            and spec.gt_to_video_offset==0 and spec.min_iou==.5, 'GT/evaluation lineage differs')
    for package in ('numpy','scipy','motmetrics','pandas'):
        require(version(package)==local['versions'][package],f'Changed evaluation dependency: {package}')
    vectors = np.load(inputs['embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(vectors.dtype==np.float32 and vectors.shape==(cache['summary']['encoded'],512),'Invalid candidate embedding archive')
    code = ['src/mtmc/pipeline/available_identity.py','src/mtmc/reid/selective_history.py',
            'scripts/experiment_selective_history.py','scripts/evaluate_global_identity.py','src/mtmc/data/ground_truth.py']
    for rel in code: checked('experiment_code:'+rel,ROOT/rel)
    run = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = ROOT/'artifacts/selective_history'/run; output.mkdir(parents=True,exist_ok=False)
    status = output/'run_status.json'; status.write_text(json.dumps({'completed':False,'run_id':run})+'\n')
    try:
        print('Phase 1: frozen staged local tracks; two causal global histories; no models or GT...',flush=True)
        summary = replay(scene,policy,run_id=run,source_run=source['run_id'],cache=cache,reference=reference,
            paths={k:v['path'] for k,v in inputs.items()},vectors=vectors,output=output,minimum_score=.5)
        trace = output/'global_tracks.jsonl.gz'; audit = output/'history_decisions.jsonl.gz'
        frozen = {p.name:sha256(p) for p in (trace,audit)}
        (output/'predictions_frozen.json').write_text(json.dumps({'run_id':run,'sha256':frozen,
            'configuration':config,'global_policy':policy['global']},indent=2)+'\n')
        print('Phase 2: predictions frozen; offline full-sequence and window evaluation...',flush=True)
        ground = load_ground_truth(spec); checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mapping = evaluate(trace,scene,spec,ground,run_id=run,split=1800)
        require(quality['global']['all_updates']['full']==reference['metrics']['staged'],'Control global metrics differ')
        require(quality['accepted_merge_diagnostics']['all_updates']==reference['accepted_merge_diagnostics']['staged'],
                'Control merge diagnostics differ')
        for camera,values in local['metrics']['staged'].items():
            for key,expected in values.items():
                actual = quality['local_shared'][camera][key]
                require(actual==expected or (actual is not None and expected is not None and abs(actual-expected)<=1e-9),
                        f'Frozen local metric differs: {camera}/{key}')
        for name in VARIANTS:
            require(sum(quality['accepted_merge_diagnostics'][name].values())==summary['lifecycle'][name]['merge_events'],
                    'Merge accounting differs')
        for name,digest in frozen.items(): require(sha256(output/name)==digest,'Frozen outputs changed during evaluation')
        for asset in inputs.values(): require(sha256(asset['path'])==asset['sha256'],'Input changed during experiment')
        match_path = output/'identity_matching.json'; match_path.write_text(json.dumps(mapping,indent=2,allow_nan=False)+'\n')
        report = {'completed':True,'protocol':'selective_global_history_paired_v1','run_id':run,'scene':scene.scene,
            'inputs':inputs,'configuration':config,'global_policy':policy['global'],'summary':summary,**quality,
            'checks':{'all_updates_descriptors_exact':True,'all_updates_full_identity_records_exact_except_run_scope':True,
                      'all_updates_metrics_lifecycle_merges_exact':True,'both_local_traces_unchanged':True,
                      'all_observations_preserved':True,'full_and_window_metrics_motmetrics_agree':True,
                      'predictions_frozen_before_GT':True,'frozen_inputs_and_outputs_unchanged':True},
            'versions':{n:version(n) for n in ('numpy','scipy','motmetrics','pandas')},
            'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in
                         (trace,audit,output/'predictions_frozen.json',match_path)},
            'limits':['Reused development scene; not independent validation or a threshold search.',
                      'Confidence-only update selection is not a validated visibility or crop-quality classifier.',
                      'Frozen local IDs may already contain identity switches; selective memory does not repair them.',
                      'Window matching resets evaluator assignment only; runtime identity/history state remains continuous.',
                      'No model execution, skipped-inference optimization, or end-to-end performance measurement.']}
        path = output/'report.json'; path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps({'completed':True,'run_id':run})+'\n')
    except Exception as error:
        status.write_text(json.dumps({'completed':False,'run_id':run,'error_type':type(error).__name__,'error':str(error)})+'\n')
        raise
    for window in ('full','first','second'):
        print(f'Window {window}: frames {quality["windows"][window]}')
        print('Variant          Global IDF1   IDTP    IDFP    IDFN')
        for name in VARIANTS:
            m = quality['global'][name][window]
            print(f'{name:16} {100*m["idf1"]:10.2f}% {m["idtp"]:7} {m["idfp"]:7} {m["idfn"]:7}')
    print('History counters:',json.dumps(summary['history']))
    print('Lifecycle:',json.dumps(summary['lifecycle']))
    print('Accepted merges:',json.dumps(quality['accepted_merge_diagnostics']))
    print(f'Report: {path}')
    print('Selective history paired experiment: COMPLETED; runtime baseline unchanged')


if __name__=='__main__':
    main()
