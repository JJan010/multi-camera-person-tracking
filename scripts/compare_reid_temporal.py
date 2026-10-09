"""Fixed-grid paired appearance retrieval; frozen local boxes, no tracker changes."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import numpy as np
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth
from mtmc.data.candidates import validate_frame_batch
from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import ObservationKey, PersonCrop
from evaluate_reid_snapshot import label_observations, summarize, score_summary

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('osnet','clipreid')


def valid_features(array, n, dim):
    require(isinstance(array,np.ndarray) and array.dtype==np.float32 and array.shape==(n,dim)
            and np.isfinite(array).all() and np.allclose(np.linalg.norm(array,axis=1),1,atol=1e-5,rtol=0),
            'Invalid model feature matrix')


def selected_crops(scene,batch,row,offset):
    """Join original camera/local keys to frozen candidate boxes, never GT or global IDs."""
    images=validate_frame_batch(scene,batch,expected_frame=batch.frame_index)
    cameras=row['variants']['enabled']['cameras']
    require([c['camera'] for c in cameras]==list(scene.camera_ids),'Source cameras differ')
    records=[];crops=[]
    for cam in cameras:
        c=cam['camera'];ids=cam['local_ids'];n=len(ids)
        require(len(set(ids))==n and all(type(i) is int and i>=0 for i in ids)
                and all(len(cam[k])==n for k in ('xyxy','confidence','embedding_rows','detection_indices')),
                'Invalid source rows')
        require(len(set(cam['detection_indices']))==n,'Repeated accepted candidate')
        rgb=images[c].rgb;h,w,_=rgb.shape
        for j in sorted(range(n),key=lambda j:ids[j]):
            key=ObservationKey(c,ids[j],batch.frame_index)
            box=cam['xyxy'][j];score=cam['confidence'][j]
            require(np.isfinite(score) and 0<=score<=1,'Invalid confidence')
            bounds,fraction=crop_geometry(box,w,h);source_row=cam['embedding_rows'][j]
            require((bounds is None)==(source_row is None),'Source crop availability differs')
            target_row=offset+len(crops) if bounds is not None else None
            digest=None
            if bounds is not None:
                require(type(source_row) is int and source_row>=0,'Invalid reference row')
                x1,y1,x2,y2=bounds;pixels=rgb[y1:y2,x1:x2].view();pixels.setflags(write=False)
                digest=hashlib.sha256(pixels.tobytes(order='C')).hexdigest()
                crops.append(PersonCrop(key,batch.timestamp,pixels))
            records.append(dict(camera=c,local_id=ids[j],frame_index=batch.frame_index,
                timestamp=str(batch.timestamp),source_xyxy=box,confidence=score,
                crop_xyxy_int=bounds,inside_image_fraction=fraction,rgb_sha256=digest,
                detection_index=cam['detection_indices'][j],source_embedding_row=source_row,embedding_row=target_row))
    return records,tuple(crops)


def collect(scene,batches,trace,source_run,cache,encode,selected,output):
    """Injected encoder returns two keyed matrices for exactly the same crop objects."""
    selected=set(selected);records=[];arrays={m:[] for m in MODELS};offset=0;runtime_count=0
    max_error=0.;min_cosine=1.;compared=0;snapshots=0
    with gzip.open(trace,'rt') as stream:
        for frame in range(scene.rounds):
            line=stream.readline();require(line,'Truncated source trace');row=json.loads(line)
            require(row['run_id']==source_run and row['frame_index']==frame
                    and Fraction(row['timestamp'])==Fraction(frame,scene.fps),'Source scope/time differs')
            batch=next(batches)
            validate_frame_batch(scene,batch,expected_frame=frame)
            runtime_count+=sum(len(c['local_ids']) for c in row['variants']['enabled']['cameras'])
            if frame not in selected:continue
            current,crops=selected_crops(scene,batch,row,offset)
            encoded_records=[r for r in current if r['embedding_row'] is not None]
            result=encode(crops)
            require(set(result)==set(MODELS),'Missing model output')
            expected_keys=tuple(c.key for c in crops)
            for name,dimension in (('osnet',512),('clipreid',1280)):
                keys,features=result[name]
                require(keys==expected_keys,'Encoder observation mapping differs')
                valid_features(features,len(crops),dimension)
            if crops:
                indices=[r['source_embedding_row'] for r in encoded_records]
                require(all(i<len(cache) for i in indices),'Reference row out of bounds')
                old=np.asarray(cache[indices]);valid_features(old,len(crops),512)
                actual=result['osnet'][1]
                error=float(np.max(np.abs(actual-old)))
                cosine=float(np.min(np.clip(np.sum(actual.astype(np.float64)*old,axis=1),-1,1)))
                require(error<=1e-5 and cosine>=1-1e-5,'Decoded crop/OSNet reference parity failed')
                max_error=max(max_error,error);min_cosine=min(min_cosine,cosine);compared+=len(crops)
                # Keep the original frozen baseline values after proving recomputation parity.
                arrays['osnet'].append(old.copy());arrays['clipreid'].append(result['clipreid'][1].copy())
            records.extend(current);offset+=len(crops);snapshots+=1
            if snapshots%20==0 or snapshots==len(selected):
                print(f'Sampled {snapshots}/{len(selected)} frames; paired crops={offset}; OSNet reference parity OK',flush=True)
        require(stream.readline()=='','Trailing source frames')
    require(snapshots==len(selected),'Sample schedule not fully covered')
    for name,dim in (('osnet',512),('clipreid',1280)):
        matrix=np.concatenate(arrays[name]) if arrays[name] else np.empty((0,dim),np.float32)
        valid_features(matrix,offset,dim);np.save(output/(name+'_embeddings.npy'),matrix,allow_pickle=False)
    (output/'observations.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    return dict(sampled_frames=snapshots,selected_observations=len(records),paired_crops=offset,
        fully_outside=len(records)-offset,source_runtime_observations=runtime_count,
        osnet_reference_parity=dict(compared=compared,max_absolute_error=max_error,minimum_cosine=min_cosine))


def label_samples(scene,spec,ground,records,selected):
    labels=[];matching=[];sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    grouped=defaultdict(list)
    for record in records:grouped[record['frame_index'],record['camera']].append(record)
    for frame in selected:
        require(spec.first_frame<=frame<=spec.last_frame,'Sample outside evaluation interval')
        for camera in scene.camera_ids:
            current=grouped[frame,camera]
            # All original predictions participate; no quality/confidence/GT sample selection.
            labeled,stats=label_observations(current,{camera:ground.slots[frame,camera]},*sizes[camera])
            labels.extend(labeled)
            matching.extend(dict(frame_index=frame,**s) for s in stats)
    return labels,matching


def rank_direction(features,query,gallery,scope,qframe,gframe,qcam,gcam):
    query=[r for r in query if r['embedding_row'] is not None]
    gallery=[r for r in gallery if r['embedding_row'] is not None]
    gi=[r['embedding_row'] for r in gallery]
    rows=[];same=[];different=[]
    for q in query:
        scores=np.clip(features[q['embedding_row']]@features[gi].T,-1,1)
        order=np.argsort(-scores,kind='stable').tolist()
        positives=[j for j in range(len(gallery)) if q['gt_id'] is not None and gallery[j]['gt_id']==q['gt_id']]
        require(len(positives)<=1,'Multiple assigned positives in one gallery slot')
        positive=positives[0] if positives else None
        status='unmatched_query' if q['gt_id'] is None else 'no_positive_in_gallery' if positive is None else 'evaluated'
        top=gallery[order[0]] if order else None
        rows.append(dict(scope=scope,query_frame=qframe,gallery_frame=gframe,query_camera=qcam,gallery_camera=gcam,
            query_local_id=q['local_id'],query_embedding_row=q['embedding_row'],query_gt_id=q['gt_id'],status=status,
            gallery_size=len(gallery),positive_rank=order.index(positive)+1 if positive is not None else None,
            top1_local_id=top['local_id'] if top else None,top1_gt_id=top['gt_id'] if top else None))
        if q['gt_id'] is not None:
            for j,g in enumerate(gallery):
                if g['gt_id'] is not None:
                    (same if g['gt_id']==q['gt_id'] else different).append(float(scores[j]))
    return rows,same,different


def summarize_with_ap(rows):
    values=summarize(rows)
    ranks=[r['positive_rank'] for r in rows if r['status']=='evaluated']
    values['mAP_single_positive']=float(np.mean([1/r for r in ranks])) if ranks else None
    return values


def evaluate_samples(labels,features,selected,cameras,gap,output):
    grouped=defaultdict(list)
    for record in labels:grouped[record['frame_index'],record['camera']].append(record)
    for group in grouped.values():group.sort(key=lambda r:r['local_id'])
    scopes=('same_time_cross_camera','plus_10s_cross_camera','plus_10s_same_camera')
    summaries={};transitions={};frames=set(selected);all_rows={m:[] for m in MODELS}
    for scope in scopes:
        rows={m:[] for m in MODELS};pairs={m:dict(same=[],different=[]) for m in MODELS};slots=0
        for qf in selected:
            gf=qf if scope==scopes[0] else qf+gap
            if gf not in frames:continue
            for qc in cameras:
                for gc in cameras:
                    if (qc==gc)!=(scope==scopes[2]):continue
                    slots+=1
                    for name in MODELS:
                        current,same,different=rank_direction(features[name],grouped[qf,qc],grouped[gf,gc],scope,qf,gf,qc,gc)
                        rows[name].extend(current);pairs[name]['same'].extend(same);pairs[name]['different'].extend(different)
        summaries[scope]={}
        for name in MODELS:
            summaries[scope][name]=dict(full=summarize_with_ap(rows[name]),
                first_query_minute=summarize_with_ap([r for r in rows[name] if r['query_frame']<1800]),
                second_query_minute=summarize_with_ap([r for r in rows[name] if r['query_frame']>=1800]),
                directions={f'{a}->{b}':summarize_with_ap([r for r in rows[name] if r['query_camera']==a and r['gallery_camera']==b])
                    for a in cameras for b in cameras if (a==b)==(scope==scopes[2])},
                evaluated_slot_pairs=slots,pair_similarities={k:score_summary(v) for k,v in pairs[name].items()})
            all_rows[name].extend(rows[name])
        counts=Counter(both_correct=0,improved=0,worsened=0,both_wrong=0)
        require(len(rows['osnet'])==len(rows['clipreid']),'Query count differs')
        for a,b in zip(rows['osnet'],rows['clipreid']):
            require(all(a[k]==b[k] for k in ('query_frame','gallery_frame','query_camera','gallery_camera','query_embedding_row','query_gt_id','status','gallery_size')),'Model populations differ')
            if a['status']!='evaluated':continue
            first,second=a['positive_rank']==1,b['positive_rank']==1
            counts['both_correct' if first and second else 'worsened' if first else 'improved' if second else 'both_wrong']+=1
        transitions[scope]=dict(counts)
    with gzip.open(output/'rankings.jsonl.gz','wt') as stream:
        for name in MODELS:
            for row in all_rows[name]:stream.write(json.dumps(dict(model=name,**row),allow_nan=False)+'\n')
    return summaries,transitions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-report',type=Path,required=True)
    parser.add_argument('--model-check-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path)
        require(digest is None or actual==digest,'Changed input: '+str(path))
        inputs[name]=dict(path=str(path),sha256=actual);return path
    rp=checked('source_report',args.source_report);source=json.loads(rp.read_text())
    require(source.get('completed') is True and source['checks'] and all(source['checks'].values()),'Unverified source report')
    kind=source['protocol'];require(kind in ('appearance_continuity_paired_v1','appearance_continuity_scene_transfer_v1'),'Unsupported trace')
    transfer=kind=='appearance_continuity_scene_transfer_v1'
    sk='validation:scene_config' if transfer else 'scene_config'
    ek='validation_cache:embeddings.npy' if transfer else 'embeddings.npy'
    ck='validation:candidate_report' if transfer else 'cache_report'
    gk='validation_ground_truth' if transfer else 'ground_truth'
    for target,key in (('scene_config',sk),('reference_vectors',ek),('cache_report',ck)):
        item=source['inputs'][key];checked(target,item['path'],item['sha256'])
    loaded=load_scene(inputs['scene_config']['path'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==source['scene'] and source['inputs'][gk]['sha256']==spec.ground_truth.sha256,'Mixed scene/GT lineage')
    cache_report=json.loads(Path(inputs['cache_report']['path']).read_text())
    require(cache_report.get('completed') is True and cache_report['artifacts']['embeddings.npy']['sha256']==inputs['reference_vectors']['sha256'],'Cache lineage differs')
    ospec=cache_report['inputs']['osnet_configuration'];osnet_path=checked('osnet_config',ospec['path'],ospec['sha256'])
    cache=np.load(inputs['reference_vectors']['path'],mmap_mode='r',allow_pickle=False)
    require(cache.dtype==np.float32 and cache.shape==(cache_report['summary']['encoded'],512),'Invalid reference matrix')
    item=source['artifacts']['global_tracks.jsonl.gz'];trace=checked('source_tracks',rp.parent/item['path'],item['sha256'])
    config=json.loads(checked('experiment_config',ROOT/'configs/reid/model_comparison_temporal.json').read_text())
    require(config['experiment']=='fixed_grid_reid_model_comparison_v1'
            and (config['frame_start'],config['frame_stop'],config['frame_stride'],config['fps'],config['time_gap_frames'],config['batch_size'])==(2,3600,30,30,300,16)
            and config['threshold_selection'] is False and config['source_variant']=='enabled'
            and config['crop_admission']=='all_positive_area_crops_from_frozen_local_observations'
            and config['protocols']==['same_time_cross_camera','plus_10s_cross_camera','plus_10s_same_camera'],'Declared sampling changed')
    require(scene.fps==30 and scene.rounds==3600 and spec.first_frame<=2 and spec.last_frame>=3572 and spec.min_iou==.5,'Unsupported scene scope')
    selected=list(range(2,3600,30))
    for name,ref in (('source_manifest',loaded.source_manifest),('video_manifest',loaded.video_manifest),('calibration',scene.calibration)):
        checked(name,ref.path,ref.sha256)
    for camera in scene.cameras:checked('video:'+str(camera.camera_id),camera.video.path,camera.video.sha256)
    clip_path=checked('clipreid_config',ROOT/'configs/models/clipreid_vit_b16_msmt17.json')
    gate=json.loads(checked('clipreid_check',args.model_check_report).read_text())
    require(gate.get('completed') is True and gate['protocol']=='clipreid_visual_smoke_v1'
            and gate['device']=='cuda:0' and gate['checks'] and all(gate['checks'].values())
            and gate['model']['config_sha256']==inputs['clipreid_config']['sha256'],'Missing matching CUDA gate')
    for rel,digest in gate['code_checksums'].items():checked('gate_code:'+rel,ROOT/rel,digest)
    for model,p in (('osnet',osnet_path),('clipreid',clip_path)):
        for name,item in json.loads(p.read_text())['assets'].items():checked(model+'_asset:'+name,ROOT/item['path'],item['sha256'])
    for rel in ('scripts/compare_reid_temporal.py','scripts/evaluate_reid_snapshot.py','src/mtmc/reid/osnet.py','src/mtmc/reid/crops.py','src/mtmc/data/scene.py','src/mtmc/data/ground_truth.py','src/mtmc/data/candidates.py','src/mtmc/video/reader.py','src/mtmc/video/replay.py'):
        checked('code:'+rel,ROOT/rel)
    import torch
    from mtmc.reid.osnet import OSNetEncoder
    from mtmc.reid.clipreid_model import load_clipreid_visual,preprocess_clipreid
    from mtmc.video.replay import synchronized_replay
    require(torch.cuda.is_available(),'CUDA unavailable');torch.set_num_threads(1)
    osnet=OSNetEncoder(osnet_path,project_root=ROOT)
    clip,clip_info=load_clipreid_visual(clip_path,project_root=ROOT,device='cuda:0')
    def encode(crops):
        keys=tuple(c.key for c in crops);arrays={m:[] for m in MODELS}
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            for start in range(0,len(crops),16):
                part=crops[start:start+16];old=osnet.encode(part)
                require(old.keys==tuple(c.key for c in part) and old.timestamps==tuple(c.timestamp for c in part),'OSNet key/time mapping differs')
                raw=clip.raw_features(preprocess_clipreid([c.rgb for c in part]).to('cuda:0'))
                require(torch.isfinite(raw).all().item() and (raw.norm(dim=1)>1e-12).all().item(),'Invalid CLIP raw features')
                arrays['osnet'].append(old.embeddings)
                arrays['clipreid'].append(torch.nn.functional.normalize(raw,p=2,dim=1).cpu().numpy())
        return {name:(keys,np.concatenate(arrays[name]) if arrays[name] else np.empty((0,dim),np.float32)) for name,dim in (('osnet',512),('clipreid',1280))}
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/reid_temporal_comparison'/run
    out.mkdir(parents=True,exist_ok=False);status=out/'run_status.json';status.write_text(json.dumps(dict(completed=False,run_id=run))+'\n')
    try:
        print(f'Scene: {scene.scene}; fixed samples: {len(selected)}; frames 2..3572, every 30 frames',flush=True)
        print('Phase 1: CPU decode + two CUDA encoders; frozen local boxes; no GT, detector or tracker...',flush=True)
        with synchronized_replay(scene.sources,fps=scene.fps,threads=1) as batches:
            counts=collect(scene,batches,trace,source['run_id'],cache,encode,selected,out)
        require(counts['source_runtime_observations']==source['summary']['observations'],'Source observation total differs')
        frozen={n:sha256(out/n) for n in ('osnet_embeddings.npy','clipreid_embeddings.npy','observations.json')}
        (out/'features_frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print('Phase 2: features frozen; offline same-time and +10-second retrieval...',flush=True)
        checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
        records=json.loads((out/'observations.json').read_text())
        labels,matching=label_samples(scene,spec,load_ground_truth(spec),records,selected)
        features={name:np.load(out/(name+'_embeddings.npy'),allow_pickle=False) for name in MODELS}
        metrics,transitions=evaluate_samples(labels,features,selected,scene.camera_ids,300,out)
        for name,digest in frozen.items():require(sha256(out/name)==digest,'Frozen features changed')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during comparison')
        (out/'labeled_observations.json').write_text(json.dumps(labels,indent=2,allow_nan=False)+'\n')
        artifacts={n:dict(path=n,sha256=sha256(out/n)) for n in (*frozen,'features_frozen.json','rankings.jsonl.gz','labeled_observations.json')}
        report=dict(completed=True,protocol='paired_reid_temporal_grid_v1',run_id=run,scene=scene.scene,
            source_run_id=source['run_id'],configuration=config,selected_frames=selected,summary=counts,
            metrics=metrics,rank1_transitions=transitions,gt_matching=matching,inputs=inputs,artifacts=artifacts,
            clipreid_model=clip_info,versions={p:version(p) for p in ('torch','torchvision','av','numpy','scipy','pillow')},
            checks=dict(frozen_local_records=True,osnet_recomputation_parity=True,same_crop_pixels=True,
                paired_query_population=True,features_frozen_before_GT=True,inputs_outputs_unchanged=True),
            limits=['Fixed local tracks were produced using OSNet; this is a conditional encoder comparison, not end-to-end tracker replacement.',
                'Both scenes informed development; no held-out generalization or threshold selection.',
                'Future-frame galleries evaluate appearance over time, not causal recovery or absence/reappearance.',
                'Single-positive gallery slots imply mAP=MRR; unknown gallery observations remain distractors.',
                'Queries without an assigned gallery positive are excluded from rank metrics and counted explicitly.',
                'Query-window summaries may have gallery frames in the next window; they are not global-ID metrics.',
                'Samples/directions share people and frames; no independent-sample significance claim.',
                'No crop filter, temporal feature averaging, identity association or latency benchmark.'])
        path=out/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps(dict(completed=True,run_id=run))+'\n')
    except Exception as error:
        status.write_text(json.dumps(dict(completed=False,run_id=run,error=str(error)))+'\n');raise
    def pct(x):return 'n/a' if x is None else f'{x:.2%}'
    print('Protocol / eligible queries / OSNet R1 / CLIP R1 / OSNet mAP / CLIP mAP')
    for scope,values in metrics.items():
        a,b=[values[m]['full'] for m in MODELS]
        print(scope,f"{a['evaluated_queries']}/{a['queries']}",pct(a['rank1']),pct(b['rank1']),pct(a['mAP_single_positive']),pct(b['mAP_single_positive']))
        print('  Rank-1 transitions:',transitions[scope])
    print('OSNet reference parity:',counts['osnet_reference_parity'])
    print('Report:',path)
    print('Temporal model comparison: COMPLETED; global IDs and runtime baseline unchanged')


if __name__=='__main__':main()
