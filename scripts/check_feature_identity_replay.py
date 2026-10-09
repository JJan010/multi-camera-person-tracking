"""Verify generic feature association against legacy and frozen global outputs."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
import pickle

import numpy as np
from mtmc.data.scene import load_scene, require, sha256
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch, FeatureHistory
from mtmc.reid.crops import CropRecord, crop_geometry
from mtmc.pipeline.core import IdentityStage
from mtmc.pipeline.feature_identity import FeatureIdentityStage
from check_feature_association import same_stages

ROOT=Path(__file__).resolve().parents[1]


def jsonable(value):
    def default(v):
        if isinstance(v,Fraction):return str(v)
        raise TypeError(type(v).__name__)
    return json.loads(json.dumps(value,default=default,allow_nan=False))


def inputs_for_round(scene,source,cached,history,offset,old_vectors):
    frame=source['frame_index'];time=Fraction(frame,scene.fps)
    require(cached['frame_index']==history['frame_index']==frame
            and Fraction(cached['timestamp'])==Fraction(history['timestamp'])==Fraction(source['timestamp'])==time,
            'Trace/cache/history time mismatch')
    cameras=source['variants']['enabled']['cameras']
    require([c['camera'] for c in cameras]==list(scene.camera_ids),'Camera coverage/order differs')
    records=cached['observations']; by_source={};expected=[]
    for c,spec in zip(cameras,scene.cameras):
        ids=c['local_ids'];n=len(ids)
        require(all(len(c[k])==n for k in ('xyxy','confidence','detection_indices','embedding_rows'))
                and len(set(ids))==n,'Malformed source arrays')
        for j in sorted(range(n),key=lambda j:ids[j]):
            key=ObservationKey(c['camera'],ids[j],frame);bounds,fraction=crop_geometry(c['xyxy'][j],spec.width,spec.height)
            by_source[key]=(c,j,bounds,fraction);expected.append(key)
    require(len(records)==len(expected),'Observation count differs')
    binding={}
    for b in source['variants']['enabled']['segment_bindings']:
        a=ObservationKey(**b['source_key']);k=ObservationKey(**b['identity_key'])
        require(a not in binding and a.camera_id==k.camera_id and a.frame_index==k.frame_index==frame,'Invalid frozen segment binding')
        binding[a]=k
    require(set(binding)==set(expected) and len(set(binding.values()))==len(binding),'Segment mapping is not bijective')
    hs=history['observations'];require(len(hs)==len(records),'History mapping count differs')
    out=[];keys=[];values=[];provenance=[]
    for key,r,h in zip(expected,records,hs):
        require((r['camera'],r['local_id'],r['frame_index'])==(key.camera_id,key.local_id,frame)
                and Fraction(r['timestamp'])==time,'Original observation key differs')
        c,j,bounds,fraction=by_source[key]
        require(r['source_xyxy']==c['xyxy'][j] and r['confidence']==c['confidence'][j]
                and r['detection_index']==c['detection_indices'][j] and r['source_embedding_row']==c['embedding_rows'][j]
                and r['crop_xyxy_int']==(list(bounds) if bounds is not None else None)
                and r['inside_image_fraction']==fraction,'Frozen candidate/crop provenance differs')
        segment=binding[key]
        require(h['source_key']==asdict(key) and h['identity_key']==asdict(segment)
                and h['embedding_row']==r['embedding_row'],'History segment/row mapping differs')
        out.append(CropRecord(segment,r['confidence'],tuple(r['source_xyxy']),bounds,fraction))
        if bounds is None:
            require(r['embedding_row'] is None and r['source_embedding_row'] is None
                    and h['source_frames']==h['source_times']==[],'Unavailable feature has history')
        else:
            require(type(r['embedding_row']) is int and r['embedding_row']==offset,'Cache row sequence differs')
            index=r['source_embedding_row'];require(type(index) is int and 0<=index<len(old_vectors),'Invalid OSNet row')
            keys.append(segment);values.append(old_vectors[index]);provenance.append(h);offset+=1
    matrix=np.stack(values) if values else np.empty((0,512),np.float32)
    return tuple(out),ReIDBatch(tuple(keys),(time,)*len(keys),matrix),provenance,offset


def replay(scene,source,trace,observations,history_rows,old_vectors,clip_means,space,cfg):
    scope=source['run_id']+'/enabled'
    history=AppearanceHistory(scope,max_observations=8,max_age=Fraction(1))
    modern_history=FeatureHistory(scope,space,max_observations=8,max_age=Fraction(1))
    kwargs=dict(variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],
                **cfg['geometry'],identity_configuration=cfg['identity'])
    old=IdentityStage(scope,scene.matrices,scene.coordinate_space,**kwargs)
    new=FeatureIdentityStage(scope,scene.matrices,scene.coordinate_space,space=space,**kwargs)
    offset=observed=merges=0
    with gzip.open(trace,'rt') as a,gzip.open(observations,'rt') as b,gzip.open(history_rows,'rt') as c:
        for frame in range(scene.rounds):
            lines=[s.readline() for s in (a,b,c)];require(all(lines),'Truncated input')
            row,cached,saved=map(json.loads,lines)
            require(row['run_id']==source['run_id'] and type(row['frame_index']) is int and row['frame_index']==frame,'Source scope mismatch')
            previous=offset
            records,batch,provenance,offset=inputs_for_round(scene,row,cached,saved,offset,old_vectors)
            time=Fraction(frame,scene.fps)
            first=history.update(frame,time,batch)
            second=modern_history.update(frame,time,FeatureBatch(scope,space,batch.keys,batch.timestamps,batch.embeddings))
            require(np.array_equal(first.mean.embeddings,second.mean.embeddings)
                    and first.source_frames==second.source_frames and first.source_times==second.source_times,'History compatibility differs')
            for i,item in enumerate(provenance):
                require(item['source_frames']==list(first.source_frames[i])
                        and item['source_times']==[str(t) for t in first.source_times[i]],'Stored CLIP sample participation differs')
            means=np.asarray(clip_means[previous:offset])
            require(means.dtype==np.float32 and means.shape==(len(batch.keys),1280) and np.isfinite(means).all()
                    and np.allclose(np.linalg.norm(means,axis=1),1,atol=1e-5,rtol=0),'Invalid stored CLIP means')
            baseline=old.update(frame,time,first.mean,records)
            actual=new.update(frame,time,second.mean,records)
            same_stages(baseline,actual)
            require(pickle.dumps(old.manager)==pickle.dumps(new.manager),'Manager state differs')
            require(jsonable(asdict(actual[0]))==row['variants']['enabled']['identity_runtime'],
                    f'Frozen full global output differs at frame {frame}')
            observed+=len(records);merges+=len(actual[0].merge_events)
            if (frame+1)%600==0 or frame+1==scene.rounds:
                print(f'Replayed {frame+1}/{scene.rounds}; OSNet pairs/groups/global state EXACT; frozen global records EXACT',flush=True)
        require(all(s.readline()=='' for s in (a,b,c)),'Trailing input frames')
    require(offset==len(clip_means),'CLIP mean rows not fully covered')
    return dict(rounds=scene.rounds,observations=observed,encoded=offset,merge_events=merges,
                final_retained_global_ids=len(actual[0].identities))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path)
        require(digest is None or digest==actual,'Changed input: '+str(path))
        inputs[name]=dict(path=str(path),sha256=actual);return path
    hp=checked('history_report',args.history_report);report=json.loads(hp.read_text())
    require(report.get('completed') is True and report['protocol']=='dimension_explicit_history_replay_v1'
            and report['checks'] and all(report['checks'].values()),'Unverified history report')
    for name,item in report['inputs'].items():checked('history:'+name,item['path'],item['sha256'])
    for name,item in report['artifacts'].items():checked('artifact:'+name,hp.parent/item['path'],item['sha256'])
    source=json.loads(Path(inputs['history:source_report']['path']).read_text())
    require(source.get('completed') is True and source['checks'] and all(source['checks'].values())
            and source['run_id']==report['source_run_id'],'Unverified source')
    loaded=load_scene(inputs['history:scene_config']['path'],project_root=ROOT);scene=loaded.runtime
    require(scene.scene==report['scene']==source['scene'],'Mixed scenes')
    cfg=source['global_policy'] if source['protocol']=='appearance_continuity_paired_v1' else source['configuration']['paired_policy']['global']
    require(cfg['history']['max_observations']==8 and Fraction(cfg['history']['max_age_seconds'])==1,'History settings differ')
    space=FeatureSpace(**report['spaces']['osnet'])
    require(space.dimension==512 and space.model_config_sha256==inputs['history:osnet_config']['sha256'],'Wrong OSNet feature space')
    clip_space=FeatureSpace(**report['spaces']['clipreid'])
    require(clip_space.dimension==1280 and clip_space.model_config_sha256==inputs['history:clipreid_config']['sha256'],'Wrong CLIP feature space')
    old=np.load(inputs['history:source_vectors']['path'],mmap_mode='r',allow_pickle=False)
    means=np.load(inputs['artifact:mean_embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(old.dtype==np.float32 and old.ndim==2 and old.shape[1]==512,'Invalid OSNet source')
    require(means.dtype==np.float32 and means.shape==(report['summary']['encoded'],1280),'Invalid CLIP means')
    for rel in ('scripts/check_feature_association.py','scripts/check_feature_identity_replay.py',
                'src/mtmc/association/feature_geometry.py','src/mtmc/pipeline/feature_identity.py',
                'src/mtmc/association/geometry.py','src/mtmc/association/pairwise.py','src/mtmc/association/grouping.py',
                'src/mtmc/association/global_identity.py','src/mtmc/association/controlled_merge.py','src/mtmc/pipeline/core.py'):
        checked('code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/feature_identity_checks'/run
    out.mkdir(parents=True,exist_ok=False);status=out/'run_status.json'
    status.write_text(json.dumps(dict(completed=False,run_id=run))+'\n')
    try:
        print('CPU replay: fixed OSNet policy; legacy and dimension-explicit association; no GT/models/decoding...',flush=True)
        summary=replay(scene,source,Path(inputs['history:source_tracks']['path']),Path(inputs['history:observations.jsonl.gz']['path']),
            Path(inputs['artifact:history_rows.jsonl.gz']['path']),old,means,space,cfg)
        require(summary['observations']==report['summary']['observations'] and summary['encoded']==report['summary']['encoded'],'Population differs')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during parity replay')
        result=dict(completed=True,protocol='feature_identity_osnet_parity_v1',run_id=run,scene=scene.scene,
            source_run_id=source['run_id'],history_run_id=report['run_id'],summary=summary,inputs=inputs,
            configuration=cfg,spaces=report['spaces'],checks=dict(legacy_pair_group_geometry_exact=True,
                legacy_manager_state_exact=True,frozen_full_identity_records_exact=True,clip_history_mapping_verified=True,inputs_unchanged=True),
            limits=['OSNet is the only descriptor used for identity decisions in this compatibility gate.',
                    'CLIP means and their frozen segment mapping are verified, but no CLIP quality is measured.',
                    'No threshold selection, GT access, local tracking change or runtime baseline modification.'])
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps(dict(completed=True,run_id=run))+'\n')
    except Exception as error:
        status.write_text(json.dumps(dict(completed=False,run_id=run,error=str(error)))+'\n');raise
    print('Summary:',json.dumps(summary))
    print('Report:',out/'report.json')
    print('Feature identity compatibility: PASSED; CLIP global quality not yet evaluated')


if __name__=='__main__':main()
