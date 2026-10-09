"""Paired overlap admission: recovery ON in both variants; frozen boxes and CLIP features."""
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
from experiment_confirmed_clip_return import evaluate
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch
from mtmc.reid.osnet import ObservationKey
from mtmc.pipeline.overlap_gallery import OverlapGalleryStage

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('disabled', 'enabled')  # Only the overlap gate differs; recovery is always ON.


def replay(scene, source, global_configuration, paths, old_vectors, raw, means, space, gate, out, run):
    cfg = global_configuration
    stages = {n: OverlapGalleryStage(source['run_id']+'/enabled' if n == 'disabled' else run+'/'+n,
        scene.matrices, scene.coordinate_space, space=space, variant=cfg['appearance_variant'],
        threshold=cfg['appearance_threshold'], **cfg['geometry'], identity_configuration=cfg['identity'],
        recovery_enabled=True, configuration=source['configuration'], overlap_enabled=n == 'enabled',
        max_overlap=gate['max_overlap'], image_sizes={c.camera_id:(c.width,c.height) for c in scene.cameras})
        for n in VARIANTS}
    counters = {n: {k:Counter() for k in ('admission','return_decisions','resets','archive_events','peaks')} for n in VARIANTS}
    offset = observations = 0
    with gzip.open(paths['trace'],'rt') as a, gzip.open(paths['observations'],'rt') as b, \
            gzip.open(paths['history'],'rt') as c, gzip.open(out,'xt') as saved:
        for frame in range(scene.rounds):
            lines = [f.readline() for f in (a,b,c)]
            require(all(lines), 'Truncated source')
            row,cached,hrow = map(json.loads,lines)
            require(row['run_id'] == source['run_id'] and row['frame_index'] == frame, 'Wrong source scope')
            envelope = dict(frame_index=frame,timestamp=row['timestamp'],variants=dict(enabled=dict(
                cameras=row['cameras'],segment_bindings=row['segment_bindings'])))
            start = offset
            records,batch,_,offset = inputs_for_round(scene,envelope,cached,hrow,offset,old_vectors)
            t = Fraction(frame,scene.fps)
            variants = {}
            for name,stage in stages.items():
                latest = FeatureBatch(stage.run_id,space,batch.keys,batch.timestamps,np.asarray(raw[start:offset]))
                averaged = FeatureBatch(stage.run_id,space,batch.keys,batch.timestamps,np.asarray(means[start:offset]))
                result,*_ = stage.update_with_raw(frame,t,averaged,latest,records)
                audit = stage.registry.last_audit
                runtime = jsonable(asdict(result))
                inverse = {ObservationKey(**v['identity_key']):v['source_key'] for v in row['segment_bindings']}
                returns = [dict(provisional_id=e['provisional_id'],global_id=e['global_id'],
                    members=[inverse[k] for k in e['members']]) for e in audit.reactivations]
                archive = jsonable(asdict(audit.archive))
                if name == 'disabled':
                    expected = row['variants']['enabled']
                    require(runtime == expected['identity_runtime'] and archive == expected['recovery_decisions']
                        and returns == expected['reactivations'] and audit.lifecycle == expected['lifecycle'],
                        f'Original recovered control differs at frame {frame}')
                stats = counters[name]
                stats['admission'].update(stage.registry.last_overlap['counts'])
                stats['return_decisions'].update(d.outcome for d in audit.archive.decisions)
                stats['resets'].update(reason for _,reason in audit.archive.resets)
                for field in ('retired','skipped_retirements','expired','capacity_evicted','blocked_removed'):
                    stats['archive_events'][field] += len(getattr(audit.archive,field))
                stats['archive_events'].update('skipped:'+reason for _,reason in audit.archive.skipped_retirements)
                for key in ('gallery_vectors','archived_ids','pending_returns'):
                    stats['peaks'][key] = max(stats['peaks'][key], audit.lifecycle[key])
                admissions = [dict(**{k:v for k,v in d.items() if k!='key'}, key=inverse[d['key']],
                                   identity_key=asdict(d['key'])) for d in stage.registry.last_overlap['decisions']]
                variants[name] = dict(identity_runtime=runtime,
                    identity=evaluation_view(result,row['segment_bindings'],run+'/'+name),
                    recovery_decisions=archive,reactivations=returns,lifecycle=audit.lifecycle,
                    gallery_admission=admissions)
            saved.write(json.dumps(dict(run_id=run,frame_index=frame,timestamp=str(t),cameras=row['cameras'],
                segment_bindings=row['segment_bindings'],variants=variants),allow_nan=False)+'\n')
            observations += len(records)
            if (frame+1)%600 == 0 or frame+1 == scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; recovered control EXACT; '
                      f"overlap-blocked={counters['enabled']['admission']['overlap_rejected']}; "
                      f"returns={stages['enabled'].registry.returns}",flush=True)
        require(all(f.readline()=='' for f in (a,b,c)) and offset==len(raw)==len(means), 'Incomplete source coverage')
    require(stages['disabled'].registry.last_audit.lifecycle == source['summary']['enabled_lifecycle'],
            'Control lifecycle differs')
    return dict(rounds=scene.rounds,observations=observations,encoded=offset,
        variants={n:dict(lifecycle=stages[n].registry.last_audit.lifecycle,
                         **{k:dict(v) for k,v in counters[n].items()}) for n in VARIANTS})


def return_diagnostics(trace,scene,spec,ground):
    previous={n:{} for n in VARIANTS}; events={n:[] for n in VARIANTS}
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream:
        for line in stream:
            row=json.loads(line); frame=row['frame_index']; labels={}
            if spec.first_frame <= frame <= spec.last_frame:
                for c in row['cameras']:
                    keys=tuple(ObservationKey(c['camera'],i,frame) for i in c['local_ids'])
                    _,_,unique,*_=spatial_slot(ground.slots[frame,c['camera']],keys,c['xyxy'],
                        width=sizes[c['camera']][0],height=sizes[c['camera']][1],min_iou=spec.min_iou)
                    labels.update(unique)
            for name in VARIANTS:
                data=row['variants'][name]
                for event in data['reactivations']:
                    prior=previous[name].get(event['global_id'])
                    current=[labels.get(ObservationKey(**k)) for k in event['members']]
                    old=prior['labels'] if prior else []
                    category=('unresolved' if not old or not current or any(v is None for v in (*old,*current)) else
                        'mixed_previous_or_current' if len(set(old))!=1 or len(set(current))!=1 else
                        'same_last_visible_gt' if old[0]==current[0] else 'different_last_visible_gt')
                    events[name].append(dict(frame_index=frame,category=category,**event,
                        previous_visible_evidence=prior,current_labels=current))
                groups={}
                for a in data['identity']['assignments']: groups.setdefault(a['global_id'],[]).append(a['key'])
                for gid,keys in groups.items():
                    previous[name][gid]=dict(frame_index=frame,members=keys,
                        labels=[labels.get(ObservationKey(**k)) for k in keys])
    return {n:dict(Counter(e['category'] for e in v)) for n,v in events.items()},events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recovery-report',type=Path,required=True)
    args=p.parse_args(); inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve(); actual=sha256(path)
        require(digest is None or digest==actual,'Changed input: '+str(path))
        inputs[name]=dict(path=str(path),sha256=actual); return path
    rp=checked('recovery_report',args.recovery_report); source=json.loads(rp.read_text())
    require(source.get('completed') is True and source['protocol']=='confirmed_clip_return_paired_v1'
        and source['checks'] and all(source['checks'].values()),'Unverified recovery experiment')
    for name,item in source['inputs'].items():
        if name!='evaluation_ground_truth': checked('source:'+name,item['path'],item['sha256'])
    def dependency(name): return Path(inputs['source:'+name]['path'])
    loaded=load_scene(dependency('source:history:scene_config'),project_root=ROOT)
    scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==source['scene'] and scene.rounds==3600 and scene.fps==30
        and (spec.first_frame,spec.last_frame)==(2,3599),'Unexpected scene/interval')
    require(source['inputs']['evaluation_ground_truth']['sha256']==spec.ground_truth.sha256,'Wrong GT')
    original=json.loads(dependency('experiment_report').read_text())
    require(original['run_id']==source['source_run_id'],'Wrong original identity policy')
    configpath=checked('gate_configuration',ROOT/'configs/reid/overlap_gallery_experiment.json')
    gate=json.loads(configpath.read_text())
    require(gate==dict(schema_version=1,experiment='gallery_only_tracked_overlap_v1',threshold_search=False,
        max_overlap=.3,metric='max_same_camera_intersection_over_own_clipped_continuous_area',
        boundary='accept_at_equality',competitors='all_current_local_track_boxes_including_weak',
        scope='archive_gallery_updates_only',recovery_enabled_in_both_variants=True),'Declared hypothesis changed')
    item=source['artifacts']['global_tracks.jsonl.gz']
    paths=dict(trace=checked('frozen_trace',rp.parent/item['path'],item['sha256']),
        observations=dependency('source:history:observations.jsonl.gz'),history=dependency('source:artifact:history_rows.jsonl.gz'))
    old=np.load(dependency('source:history:source_vectors'),mmap_mode='r',allow_pickle=False)
    raw=np.load(dependency('source:history:clipreid_embeddings.npy'),mmap_mode='r',allow_pickle=False)
    means=np.load(dependency('source:artifact:mean_embeddings.npy'),mmap_mode='r',allow_pickle=False)
    space=FeatureSpace(**source['space'])
    for a in (raw,means): require(a.dtype==np.float32 and a.shape==(source['summary']['encoded'],space.dimension),'Bad matrix')
    for rel in ('scripts/experiment_overlap_gallery.py','scripts/check_overlap_gallery.py','src/mtmc/pipeline/overlap_gallery.py',
                'scripts/experiment_confirmed_clip_return.py','scripts/check_feature_identity_replay.py',
                'scripts/experiment_clipreid_global.py','src/mtmc/pipeline/confirmed_return.py'):
        checked('code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'artifacts/overlap_gallery'/run; out.mkdir(parents=True,exist_ok=False)
    status=out/'run_status.json'; status.write_text('{"completed":false}\n')
    output=out/'global_tracks.jsonl.gz'
    try:
        print('Phase 1: gallery-only overlap admission; recovery ON in both variants; no GT/models/decoding...',flush=True)
        summary=replay(scene,source,original['configuration'],paths,old,raw,means,space,gate,output,run)
        require(summary['observations']==source['summary']['observations'] and summary['encoded']==source['summary']['encoded'],
                'Changed population')
        digest=sha256(output); freeze=out/'prediction_freeze.json'
        freeze.write_text(json.dumps(dict(run_id=run,sha256=digest,gate_configuration=gate),indent=2)+'\n')
        print('Phase 2: predictions frozen; offline full/window quality evaluation...',flush=True)
        ground=load_ground_truth(spec); checked('evaluation_ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        quality,mappings=evaluate(output,scene,spec,ground,run=run,split=1800)
        require(quality['global_metrics']['disabled']==source['global_metrics']['enabled'],'Control quality differs')
        require(quality['accepted_merge_diagnostics']['disabled']==source['accepted_merge_diagnostics']['enabled'],
                'Control merge labels differ')
        categories,events=return_diagnostics(output,scene,spec,ground)
        require(categories['disabled']==source['return_gt_diagnostics'],'Control return labels differ')
        for n in VARIANTS:
            require(len(events[n])==summary['variants'][n]['lifecycle']['reactivation_events'],'Missing return events')
        require(sha256(output)==digest,'Predictions changed during evaluation')
        for item in inputs.values(): require(sha256(item['path'])==item['sha256'],'Input changed during experiment')
        matching=out/'identity_matching.json'; matching.write_text(json.dumps(mappings,indent=2)+'\n')
        ep=out/'reactivation_diagnostics.json'; ep.write_text(json.dumps(events,indent=2)+'\n')
        report=dict(completed=True,protocol='gallery_only_overlap_paired_v1',run_id=run,scene=scene.scene,
            source_run_id=source['run_id'],inputs=inputs,configuration=gate,recovery_configuration=source['configuration'],
            space=asdict(space),variant_meaning=dict(disabled='recovery_on_overlap_off',enabled='recovery_on_overlap_on'),
            summary=summary,**quality,return_gt_diagnostics=categories,
            checks=dict(recovered_control_records_exact=True,control_metrics_lifecycle_exact=True,
                fixed_local_observations=True,full_window_motmetrics_verified=True,predictions_frozen_before_gt=True,inputs_unchanged=True),
            artifacts={f.name:dict(path=f.name,sha256=sha256(f)) for f in (output,freeze,matching,ep)},
            limits=['One development hypothesis; both scenes have already been inspected.',
                'Overlap is not an occlusion detector: static occluders and missing person boxes are not covered.',
                'Duplicates and close but distinct people can reject useful samples.',
                'Only gallery updates are gated; current return queries and association descriptors are unchanged.',
                'Older evidence keeps its true age; past predictions are not relabeled.',
                'Endpoint GT labels do not establish whole-gallery identity purity.'])
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text('{"completed":true}\n')
    except Exception as e:
        status.write_text(json.dumps(dict(completed=False,error=str(e)))+'\n'); raise
    for w in ('full','first','second'):
        print('Window:',w)
        for n in VARIANTS:
            m=quality['global_metrics'][n][w]
            print(f"  overlap {n:8} IDF1={100*m['idf1']:.2f}% IDTP={m['idtp']} IDFP={m['idfp']} IDFN={m['idfn']}")
    for n in VARIANTS:
        print('Variant:',n,'; summary:',json.dumps(summary['variants'][n]))
        print('Return GT diagnostics:',json.dumps(categories[n]))
    print('Report:',out/'report.json')
    print('Gallery overlap experiment: COMPLETED; runtime baseline unchanged; deployment setting NOT selected')


if __name__=='__main__': main()
