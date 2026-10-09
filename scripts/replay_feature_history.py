"""CPU-only history replay on frozen segment keys: OSNet parity and CLIP means."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path

import numpy as np
from mtmc.data.scene import load_scene, require, sha256
from mtmc.reid.osnet import ObservationKey, ReIDBatch
from mtmc.reid.history import AppearanceHistory
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch, FeatureHistory
from cache_clipreid_tracks import trace_rows, verify_persisted
from check_feature_history import compare_history
from compare_reid_temporal import valid_features

ROOT = Path(__file__).resolve().parents[1]


def segment_keys(row, records):
    """Frozen cuts are controls, not recomputed CLIP decisions."""
    source_keys = tuple(ObservationKey(r['camera'],r['local_id'],r['frame_index']) for r in records)
    bindings = row['variants']['enabled']['segment_bindings']; mapping = {}; targets = set()
    for binding in bindings:
        a = ObservationKey(**binding['source_key']); b = ObservationKey(**binding['identity_key'])
        require(all(type(v) is int and v >= 0 for k in (a,b) for v in (k.camera_id,k.local_id,k.frame_index))
                and a.camera_id == b.camera_id and a.frame_index == b.frame_index == row['frame_index']
                and a not in mapping and b not in targets, 'Invalid/duplicate frozen segment binding')
        mapping[a] = b; targets.add(b)
    require(set(mapping) == set(source_keys) and len(set(source_keys)) == len(source_keys), 'Segment/source coverage differs')
    return tuple(mapping[k] for k in source_keys)


def replay(scene, trace, source_run, observations_path, old_vectors, clip_vectors, osnet_space, clip_space,
           output, run_id, history_settings):
    owners = {name:FeatureHistory(run_id, space, **history_settings)
              for name,space in (('osnet',osnet_space),('clipreid',clip_space))}
    legacy = AppearanceHistory(run_id, **history_settings)
    n = len(clip_vectors)
    if n:
        means = np.lib.format.open_memmap(output/'mean_embeddings.npy',mode='w+',dtype=np.float32,shape=(n,1280))
    else:
        np.save(output/'mean_embeddings.npy',np.empty((0,1280),np.float32));means=np.empty((0,1280),np.float32)
    total = offset = fallbacks = peak_tracks = peak_vectors = 0
    with gzip.open(observations_path,'rt') as stream, gzip.open(output/'history_rows.jsonl.gz','wt') as sink:
        for source in trace_rows(scene,trace,source_run):
            line = stream.readline(); require(bool(line),'Truncated cache mapping'); cached=json.loads(line)
            frame=source['frame_index'];time=Fraction(source['timestamp'])
            require(cached['frame_index']==frame and Fraction(cached['timestamp'])==time,'Cache time differs')
            records=cached['observations']; effective=segment_keys(source,records)
            selected=[i for i,r in enumerate(records) if r['embedding_row'] is not None]
            target_rows=[records[i]['embedding_row'] for i in selected]
            require(target_rows==list(range(offset,offset+len(selected))),'Noncontiguous feature mapping')
            keys=tuple(effective[i] for i in selected)
            source_rows=[records[i]['source_embedding_row'] for i in selected]
            require(all(type(i) is int and 0<=i<len(old_vectors) for i in source_rows),'Invalid OSNet reference row')
            old=np.asarray(old_vectors[source_rows]);new=np.asarray(clip_vectors[target_rows])
            old_batch=ReIDBatch(keys,(time,)*len(keys),old)
            baseline=legacy.update(frame,time,old_batch)
            modern=owners['osnet'].update(frame,time,FeatureBatch(run_id,osnet_space,keys,old_batch.timestamps,old))
            compare_history(baseline,modern)
            require((legacy.active_tracks,legacy.stored_vectors)==(owners['osnet'].active_tracks,owners['osnet'].stored_vectors),
                    'OSNet state accounting differs')
            current=owners['clipreid'].update(frame,time,FeatureBatch(run_id,clip_space,keys,old_batch.timestamps,new))
            # Sample participation and age rules must be model-independent.
            require(current.source_frames==modern.source_frames and current.source_times==modern.source_times,
                    'Model changed history sample admission')
            valid_features(current.mean.embeddings,len(keys),1280)
            means[offset:offset+len(keys)]=current.mean.embeddings
            details=[];p=0
            for r,k in zip(records,effective):
                available=r['embedding_row'] is not None
                details.append(dict(source_key=dict(camera_id=r['camera'],local_id=r['local_id'],frame_index=frame),
                    identity_key=asdict(k),embedding_row=r['embedding_row'],
                    source_frames=list(current.source_frames[p]) if available else [],
                    source_times=[str(t) for t in current.source_times[p]] if available else [],
                    used_latest_fallback=current.used_latest_fallback[p] if available else None))
                p+=available
            sink.write(json.dumps(dict(frame_index=frame,timestamp=str(time),observations=details),allow_nan=False)+'\n')
            offset+=len(keys);total+=len(records);fallbacks+=sum(current.used_latest_fallback)
            peak_tracks=max(peak_tracks,owners['clipreid'].active_tracks)
            peak_vectors=max(peak_vectors,owners['clipreid'].stored_vectors)
            if (frame+1)%600==0 or frame+1==scene.rounds:
                if n:means.flush()
                print(f'Replayed {frame+1}/{scene.rounds}; OSNet history EXACT; CLIP means={offset}',flush=True)
        require(stream.readline()=='','Trailing mapping frames')
    if n:means.flush()
    del means
    require(offset==n,'Incomplete mean vector mapping')
    persisted=np.load(output/'mean_embeddings.npy',mmap_mode='r',allow_pickle=False)
    for start in range(0,n,4096):valid_features(np.asarray(persisted[start:start+4096]),len(persisted[start:start+4096]),1280)
    return dict(rounds=scene.rounds,observations=total,encoded=offset,osnet_exact_rows=offset,
                clip_cancellation_fallbacks=fallbacks,peak_cached_tracks=peak_tracks,peak_cached_vectors=peak_vectors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-report',type=Path,required=True)
    args=parser.parse_args();inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path)
        require(digest is None or actual==digest,'Changed input: '+str(path))
        inputs[name]=dict(path=str(path),sha256=actual);return path
    cache_path=checked('clip_cache_report',args.cache_report);cache=json.loads(cache_path.read_text())
    require(cache.get('completed') is True and cache['protocol']=='clipreid_frozen_track_cache_v1'
            and cache['feature_dim']==1280 and cache['checks'] and all(cache['checks'].values()),'Unverified CLIP cache')
    for name in ('source_report','source_tracks','scene_config','source_vectors','cache_report','clipreid_config'):
        item=cache['inputs'][name];checked(name,item['path'],item['sha256'])
    for name in ('clipreid_embeddings.npy','observations.jsonl.gz'):
        item=cache['artifacts'][name];checked(name,cache_path.parent/item['path'],item['sha256'])
    source=json.loads(Path(inputs['source_report']['path']).read_text())
    require(source.get('completed') is True and source['checks'] and all(source['checks'].values())
            and source['run_id']==cache['source_run_id'],'Unverified/mixed source')
    loaded=load_scene(inputs['scene_config']['path'],project_root=ROOT);scene=loaded.runtime
    require(scene.scene==cache['scene']==source['scene'],'Mixed scene')
    settings=source['global_policy']['history'] if source['protocol']=='appearance_continuity_paired_v1' else source['configuration']['paired_policy']['global']['history']
    require(settings['max_observations']==8 and Fraction(settings['max_age_seconds'])==1,'Frozen history policy differs')
    history_settings=dict(max_observations=8,max_age=Fraction(1))
    original_cache=json.loads(Path(inputs['cache_report']['path']).read_text())
    model=original_cache['inputs']['osnet_configuration']
    osnet_config=checked('osnet_config',model['path'],model['sha256'])
    require(original_cache['artifacts']['embeddings.npy']['sha256']==inputs['source_vectors']['sha256'],'OSNet cache lineage differs')
    old=np.load(inputs['source_vectors']['path'],mmap_mode='r',allow_pickle=False)
    clip=np.load(inputs['clipreid_embeddings.npy']['path'],mmap_mode='r',allow_pickle=False)
    require(old.dtype==np.float32 and old.shape==(original_cache['summary']['encoded'],512),'Invalid OSNet matrix')
    spaces=dict(osnet=FeatureSpace(sha256(osnet_config),512),
                clipreid=FeatureSpace(inputs['clipreid_config']['sha256'],1280))
    print('Verifying persisted full-rate mapping and vectors; CPU only, no GT/model execution...',flush=True)
    verify_persisted(scene,Path(inputs['source_tracks']['path']),source['run_id'],len(old),cache_path.parent,cache['summary'])
    # Pin the actual reference implementation used in the numerical parity check.
    for rel in ('src/mtmc/reid/history.py','src/mtmc/reid/osnet.py','src/mtmc/reid/feature_history.py',
                'src/mtmc/data/scene.py','scripts/cache_clipreid_tracks.py','scripts/compare_reid_temporal.py',
                'scripts/check_feature_history.py','scripts/replay_feature_history.py'):
        checked('code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=ROOT/'artifacts/feature_history'/run;output.mkdir(parents=True,exist_ok=False)
    status=output/'run_status.json';status.write_text(json.dumps(dict(completed=False,run_id=run))+'\n')
    try:
        counts=replay(scene,Path(inputs['source_tracks']['path']),source['run_id'],Path(inputs['observations.jsonl.gz']['path']),
            old,clip,spaces['osnet'],spaces['clipreid'],output,run,history_settings)
        require(counts['observations']==cache['summary']['observations'] and counts['encoded']==cache['summary']['encoded'],
                'Source denominators differ')
        for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during replay')
        artifacts={name:dict(path=name,sha256=sha256(output/name)) for name in ('mean_embeddings.npy','history_rows.jsonl.gz')}
        report=dict(completed=True,protocol='dimension_explicit_history_replay_v1',run_id=run,scene=scene.scene,
            source_run_id=source['run_id'],cache_run_id=cache['run_id'],summary=counts,
            spaces={k:asdict(v) for k,v in spaces.items()},history=dict(max_observations=8,max_age_seconds='1',update_policy='all_available'),
            inputs=inputs,artifacts=artifacts,checks=dict(osnet_history_exact=True,model_spaces_separate=True,
                frozen_segment_mapping=True,sample_participation_equal=True,clip_means_finite_normalized=True,inputs_unchanged=True),
            limits=['OSNet means/provenance are compared against the existing AppearanceHistory on identical frozen inputs.',
                'Local tracks AND continuity cuts remain frozen from the OSNet-based source; CLIP does not choose new cuts.',
                'No models, GT, pairwise association, global identity manager, threshold selection or quality evaluation run.',
                'This proves history compatibility, not full global-runtime parity or improved IDF1.',
                'All available crops update history; no selective quality filter or dormant recovery.'])
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        status.write_text(json.dumps(dict(completed=True,run_id=run))+'\n')
    except Exception as error:
        status.write_text(json.dumps(dict(completed=False,run_id=run,error=str(error)))+'\n');raise
    print('Summary:',json.dumps(counts))
    print('Report:',output/'report.json')
    print('Dimension-explicit history replay: PASSED; global IDs unchanged')


if __name__=='__main__':main()
