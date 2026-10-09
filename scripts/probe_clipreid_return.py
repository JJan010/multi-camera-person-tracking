"""Read-only CLIP return probe: retired references, sample provenance and competitors."""
import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path

import numpy as np
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot
from mtmc.reid.osnet import ObservationKey
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.reid.sample_quality import assess_sample
from mtmc.association.geometry import project_box_foot

ROOT=Path(__file__).resolve().parents[1]
MODES=('last_mean','confidence_gallery','confidence_border_gallery')


def normalized_average(vectors):
    x=np.asarray(vectors,dtype=np.float64)
    require(x.ndim==2 and x.shape[1]==1280 and len(x)>0 and np.isfinite(x).all(),'Invalid feature vectors')
    require(np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-5,rtol=0),'Non-unit features')
    mean=x.mean(axis=0);norm=float(np.linalg.norm(mean))
    return None if norm<=1e-12 else mean/norm


def new_memory():
    return dict(latest=None,gallery={p:defaultdict(lambda:deque(maxlen=8)) for p in MODES[1:]},
                last_seen=None,last_by_camera={})


def observe(memory,item):
    frame=item['frame_index'];camera=item['camera_id']
    memory['last_seen']=frame;memory['last_by_camera'][camera]=frame
    if item['embedding_row'] is None:return
    old=memory['latest']
    # A deterministic single latest history mean. Same-frame ties use camera/local ID.
    if old is None or (-frame,camera,item['local_id'])<(-old['frame_index'],old['camera_id'],old['local_id']):
        memory['latest']=item
    if item['confidence_passed']:memory['gallery']['confidence_gallery'][camera].append(item)
    if item['confidence_passed'] and item['border_passed']:
        memory['gallery']['confidence_border_gallery'][camera].append(item)


def reference(memory,mode,raw,means):
    if mode=='last_mean':
        samples=[] if memory['latest'] is None else [memory['latest']]
        array=means
    else:
        samples=sorted((r for queue in memory['gallery'][mode].values() for r in queue),
                       key=lambda r:(r['frame_index'],r['camera_id'],r['local_id']))
        array=raw
    if not samples:return None,[], 'no_samples'
    vector=normalized_average([array[r['embedding_row']] for r in samples])
    return vector,samples, 'cancellation' if vector is None else 'available'


def rank_references(memories,expired,absorbed,live,query,frame,fps,raw,means,old_gid):
    encoded=[r for r in query if r['embedding_row'] is not None]
    require(encoded,'Birth has no appearance query')
    q=normalized_average([means[r['embedding_row']] for r in encoded]);require(q is not None,'Query cancellation')
    all_results={}
    for mode in MODES:
        candidates=[];unavailable=Counter()
        for gid in sorted(expired):
            if gid in absorbed or gid in live:continue
            vector,samples,status=reference(memories[gid],mode,raw,means)
            if vector is None:unavailable[status]+=1;continue
            latest=memories[gid]['latest']
            # Distances are contextual only: each available query point against the last encoded reference point.
            distances=[]
            if latest is not None and latest['ground_xy'] is not None:
                for r in encoded:
                    if r['ground_xy'] is not None:
                        distances.append(dict(camera_id=r['camera_id'],distance=math.dist(latest['ground_xy'],r['ground_xy'])))
            candidates.append(dict(global_id=gid,cosine=float(np.clip(np.dot(q,vector),-1,1)),
                expired_frame=expired[gid],last_seen_frame=memories[gid]['last_seen'],
                last_seen_age_seconds=(frame-memories[gid]['last_seen'])/fps,
                sample_count=len(samples),oldest_sample_age_seconds=(frame-min(r['frame_index'] for r in samples))/fps,
                newest_sample_age_seconds=(frame-max(r['frame_index'] for r in samples))/fps,
                source_samples=samples,last_encoded_ground_observation=latest,ground_distances=distances))
        candidates.sort(key=lambda r:(-r['cosine'],r['global_id']))
        for i,c in enumerate(candidates):c['rank']=i+1
        target=next((r for r in candidates if r['global_id']==old_gid),None)
        others=[r for r in candidates if r['global_id']!=old_gid]
        target_status=('live' if old_gid in live else 'absorbed' if old_gid in absorbed else
                       'not_expired' if old_gid not in expired else
                       reference(memories[old_gid],mode,raw,means)[2])
        all_results[mode]=dict(eligible_retired_id_count=len(candidates),unavailable_counts=dict(unavailable),
            target_status=target_status,target=target,top5=candidates[:5],
            target_minus_best_other_cosine=(target['cosine']-others[0]['cosine']) if target is not None and others else None)
    return all_results


def probe(scene,trace,cache,history,raw,means,*,run,old_gid,new_gid,admission):
    memories=defaultdict(new_memory);expired={};absorbed={};births={};target=None;offset=0
    cameras={c.camera_id:c for c in scene.cameras}
    with gzip.open(trace,'rt') as a,gzip.open(cache,'rt') as b,gzip.open(history,'rt') as c:
        for frame in range(scene.rounds):
            lines=[f.readline() for f in (a,b,c)];require(all(lines),'Truncated inputs')
            row,cached,saved=map(json.loads,lines);t=Fraction(frame,scene.fps)
            require(row['run_id']==run and all(r['frame_index']==frame and Fraction(r['timestamp'])==t for r in (row,cached,saved)),
                'Mixed frame/time/scope')
            data=row['variants']['clipreid'];runtime=data['identity_runtime']
            require([v['camera'] for v in row['cameras']]==list(scene.camera_ids),'Camera coverage differs')
            observed={}
            for camera in row['cameras']:
                require(len(camera['local_ids'])==len(camera['xyxy'])==len(camera['confidence']),'Malformed camera arrays')
                for local,box,score in zip(camera['local_ids'],camera['xyxy'],camera['confidence']):
                    k=ObservationKey(camera['camera'],local,frame);require(k not in observed,'Duplicate local observation')
                    observed[k]=(box,score)
            bindings={ObservationKey(**x['source_key']):ObservationKey(**x['identity_key']) for x in row['segment_bindings']}
            assigned={ObservationKey(**x['key']):x['global_id'] for x in data['identity']['assignments']}
            internal={ObservationKey(**x['key']):x['global_id'] for x in runtime['assignments']}
            require(set(bindings)==set(assigned)==set(observed) and len(bindings)==len(row['segment_bindings'])==len(assigned)
                ==len(data['identity']['assignments'])==len(internal)==len(runtime['assignments']),'Invalid mapping')
            current={r['global_id']:r for r in runtime['identities']}
            for a0 in runtime['base_assignments']:
                if a0['reason']=='new_identity':births.setdefault(a0['global_id'],frame)
            for g in runtime['expired_global_ids']:expired[g]=frame
            for event in runtime['merge_events']:
                for g in event['absorbed_global_ids']:absorbed[g]=dict(frame_index=frame,canonical_global_id=event['canonical_global_id'])
            require(len(cached['observations'])==len(saved['observations'])==len(assigned),'Observation coverage differs')
            seen=set();query=[]
            for sample,h in zip(cached['observations'],saved['observations']):
                key=ObservationKey(sample['camera'],sample['local_id'],frame)
                require(key in assigned and key not in seen and sample['frame_index']==frame and Fraction(sample['timestamp'])==t,'Wrong sample key')
                seen.add(key);gid=assigned[key]
                require((sample['source_xyxy'],sample['confidence'])==observed[key],'Frozen box/confidence differs')
                require(ObservationKey(**h['source_key'])==key and ObservationKey(**h['identity_key'])==bindings[key]
                    and internal.get(bindings[key])==gid and h['embedding_row']==sample['embedding_row'],'Segment/row mapping differs')
                camera=cameras[key.camera_id];bounds,fraction=crop_geometry(sample['source_xyxy'],camera.width,camera.height)
                require(sample['crop_xyxy_int']==(list(bounds) if bounds is not None else None)
                    and sample['inside_image_fraction']==fraction,'Crop provenance differs')
                index=sample['embedding_row']
                if bounds is None:require(index is None,'Outside crop encoded')
                else:require(type(index) is int and index==offset,'Noncontiguous rows');offset+=1
                record=CropRecord(key,sample['confidence'],tuple(sample['source_xyxy']),bounds,fraction)
                quality=assess_sample(record,camera.width,camera.height,**admission)
                item=dict(camera_id=key.camera_id,local_id=key.local_id,segment_id=bindings[key].local_id,frame_index=frame,
                    global_id=gid,embedding_row=index,confidence=sample['confidence'],xyxy=sample['source_xyxy'],
                    confidence_passed=quality.confidence_passed,border_passed=quality.border_passed,
                    normalized_clearance=quality.normalized_clearance,history_source_frames=h['source_frames'],
                    ground_xy=project_box_foot(camera.homography,tuple(sample['source_xyxy'])) if bounds is not None else None)
                observe(memories[gid],item)
                if gid==new_gid:query.append(item)
            require(seen==set(assigned),'Missing source samples')
            if births.get(new_gid)==frame:
                require(target is None and query,'Target birth absent from emitted output')
                results=rank_references(memories,expired,absorbed,current,query,frame,scene.fps,raw,means,old_gid)
                target=dict(frame_index=frame,timestamp=str(t),query_global_id=new_gid,target_old_global_id=old_gid,
                    old_birth_frame=births.get(old_gid),old_expired_frame=expired.get(old_gid),old_absorbed=absorbed.get(old_gid),
                    old_last_seen_frame=memories[old_gid]['last_seen'] if old_gid in memories else None,
                    old_last_observation_by_camera=dict(memories[old_gid]['last_by_camera']) if old_gid in memories else {},
                    old_final_encoded_observation=memories[old_gid]['latest'] if old_gid in memories else None,
                    query_observations=query,comparisons=results)
            if (frame+1)%600==0 or frame+1==scene.rounds:print(f'Inspected {frame+1}/{scene.rounds}; frozen CLIP observations only',flush=True)
        require(all(f.readline()=='' for f in (a,b,c)) and offset==len(raw)==len(means),'Incomplete input coverage')
    require(target is not None,'Requested new GID was not allocated/emitted')
    return target


def add_offline_labels(result,trace,scene,spec,ground):
    """Rankings are already frozen. Labels annotate selected samples, never select them."""
    selected={}
    def collect(value):
        if isinstance(value,dict):
            if all(k in value for k in ('camera_id','local_id','frame_index','embedding_row')):
                selected.setdefault(value['frame_index'],set()).add(ObservationKey(value['camera_id'],value['local_id'],value['frame_index']))
            for v in value.values():collect(v)
        elif isinstance(value,list):
            for v in value:collect(v)
    collect(result);labels={};sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as f:
        for line in f:
            row=json.loads(line);frame=row['frame_index']
            if frame not in selected:continue
            for camera in row['cameras']:
                c=camera['camera'];keys=tuple(ObservationKey(c,i,frame) for i in camera['local_ids'])
                known={}
                if spec.first_frame<=frame<=spec.last_frame:
                    _,_,known,*_=spatial_slot(ground.slots[frame,c],keys,camera['xyxy'],width=sizes[c][0],height=sizes[c][1],min_iou=spec.min_iou)
                for k in keys:
                    if k in selected[frame]:labels[k]=known.get(k)
    require(set(labels)==set().union(*selected.values()),'Diagnostic label coverage differs')
    def annotate(value):
        if isinstance(value,dict):
            if all(k in value for k in ('camera_id','local_id','frame_index','embedding_row')):
                value['diagnostic_unique_gt']=labels[ObservationKey(value['camera_id'],value['local_id'],value['frame_index'])]
            for v in list(value.values()):annotate(v)
        elif isinstance(value,list):
            for v in value:annotate(v)
    annotate(result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-report',type=Path,required=True)
    p.add_argument('--old-gid',type=int,default=11);p.add_argument('--new-gid',type=int,default=76);args=p.parse_args()
    require(min(args.old_gid,args.new_gid)>=0 and args.old_gid!=args.new_gid,'Invalid IDs');inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path);require(digest is None or digest==actual,'Changed input: '+str(path))
        inputs[name]=dict(path=str(path),sha256=actual);return path
    rp=checked('experiment_report',args.experiment_report);report=json.loads(rp.read_text())
    require(report.get('completed') is True and report['protocol']=='clipreid_fixed_tracks_global_v1'
        and report['checks'] and all(report['checks'].values()),'Unverified model experiment')
    keys=('history:scene_config','history:observations.jsonl.gz','artifact:history_rows.jsonl.gz',
          'history:clipreid_embeddings.npy','artifact:mean_embeddings.npy')
    paths={k:checked(k,report['inputs'][k]['path'],report['inputs'][k]['sha256']) for k in keys}
    item=report['artifacts']['global_tracks.jsonl.gz'];trace=checked('trace',rp.parent/item['path'],item['sha256'])
    loaded=load_scene(paths['history:scene_config'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==report['scene'] and spec.ground_truth.sha256==report['inputs']['evaluation_ground_truth']['sha256'],'Mixed scene/GT')
    quality_path=checked('admission_configuration',ROOT/'configs/reid/sample_admission_experiment.json')
    quality=json.loads(quality_path.read_text());admission=dict(min_confidence=quality['min_confidence'],border_fraction=quality['border_fraction'])
    require(admission==dict(min_confidence=.5,border_fraction=.01),'Expected previously declared admission hypothesis')
    raw=np.load(paths['history:clipreid_embeddings.npy'],mmap_mode='r',allow_pickle=False)
    means=np.load(paths['artifact:mean_embeddings.npy'],mmap_mode='r',allow_pickle=False)
    for array in (raw,means):
        require(array.dtype==np.float32 and array.shape==(report['summary']['encoded'],1280),'Unexpected CLIP matrix')
        for i in range(0,len(array),4096):
            part=np.asarray(array[i:i+4096]);require(np.isfinite(part).all() and np.allclose(np.linalg.norm(part,axis=1),1,atol=1e-5,rtol=0),'Invalid CLIP features')
    for rel in ('scripts/probe_clipreid_return.py','scripts/check_clipreid_return_probe.py','src/mtmc/reid/sample_quality.py',
                'src/mtmc/association/geometry.py','src/mtmc/data/ground_truth.py','src/mtmc/data/scene.py'):
        checked('code:'+rel,ROOT/rel)
    print('Phase 1: frozen identity/feature join; reference selection and rankings without GT...',flush=True)
    result=probe(scene,trace,paths['history:observations.jsonl.gz'],paths['artifact:history_rows.jsonl.gz'],raw,means,
        run=report['run_id'],old_gid=args.old_gid,new_gid=args.new_gid,admission=admission)
    # Exact JSON snapshot before any offline GT annotation, including all appearance ranks.
    result=json.loads(json.dumps(result,allow_nan=False));unlabeled=json.dumps(result,indent=2,allow_nan=False)+'\n'
    print('Phase 2: offline labels for frozen selected references and query; no reranking...',flush=True)
    ground=load_ground_truth(spec);checked('evaluation_ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
    add_offline_labels(result,trace,scene,spec,ground)
    def remove_labels(value):
        if isinstance(value,dict):return {k:remove_labels(v) for k,v in value.items() if k!='diagnostic_unique_gt'}
        if isinstance(value,list):return [remove_labels(v) for v in value]
        return value
    require(remove_labels(result)==json.loads(unlabeled),'Offline labels changed reference selection or rankings')
    for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during probe')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/clipreid_return_probe'/run;out.mkdir(parents=True,exist_ok=False)
    frozen=out/'rankings_before_gt.json';frozen.write_text(unlabeled)
    output=dict(completed=True,protocol='clipreid_retired_reference_probe_v1',run_id=run,experiment_run_id=report['run_id'],
        inputs=inputs,configuration=dict(admission=admission,raw_gallery_max_per_camera=8,age_geometry_gates=False),result=result,
        checks=dict(frozen_observation_feature_mapping=True,all_feature_rows_consumed=True,reference_selection_before_gt=True,
            offline_annotation_preserves_rankings=True,inputs_unchanged=True),
        artifacts={frozen.name:dict(path=frozen.name,sha256=sha256(frozen))},
        limits=['Case selection is retrospective; references/rankings use no GT and do not reactivate identities.',
            'Gallery modes average raw samples; last_mean uses one stored history mean. These are diagnostic alternatives, not an implemented recovery policy.',
            'Reference samples belong to the exact emitted GID, without inherited samples from absorbed IDs.',
            'Ranks include all previously expired, nonabsorbed, nonlive IDs. No age, motion or ambiguity gate is calibrated.',
            'Sample admission is a previous experimental hypothesis, not proof of crop quality; a mean row label does not label every contributing history sample.',
            'Ground distances are in native calibration units and do not establish an admissible motion threshold.'])
    (out/'report.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Lifecycle:',json.dumps({k:result[k] for k in ('frame_index','query_global_id','target_old_global_id','old_birth_frame','old_last_seen_frame','old_expired_frame','old_last_observation_by_camera')}))
    print('Query:',json.dumps(result['query_observations']))
    for mode,values in result['comparisons'].items():
        t=values['target'];print('\nMODE:',mode,'target status:',values['target_status'])
        if t:
            print('Target:',json.dumps({k:t[k] for k in ('global_id','cosine','rank','last_seen_age_seconds','sample_count','oldest_sample_age_seconds','newest_sample_age_seconds','ground_distances')}))
            print('Target sample GT:',dict(Counter(str(r['diagnostic_unique_gt']) for r in t['source_samples'])))
            print('Target minus best other cosine:',values['target_minus_best_other_cosine'])
        print('Top retired references: GID / cosine / last-seen age / selected sample GT')
        for r in values['top5']:
            print(r['global_id'],round(r['cosine'],6),round(r['last_seen_age_seconds'],3),dict(Counter(str(s['diagnostic_unique_gt']) for s in r['source_samples'])))
    print('Report:',out/'report.json')
    print('CLIP return probe: COMPLETED; recovery disabled; predictions unchanged')


if __name__=='__main__':main()
