"""Read-only replay audit of actual archived gallery samples for every confirmed return."""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import mtmc.pipeline.confirmed_return as registry_module
from mtmc.association.confirmed_return import ConfirmedReturnArchive, _FeatureArchive
from mtmc.pipeline.confirmed_return import ConfirmedReturnStage
from mtmc.reid.feature_history import FeatureSpace, FeatureBatch
from mtmc.reid.osnet import ObservationKey
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth
from check_feature_identity_replay import inputs_for_round, jsonable
from experiment_clipreid_global import evaluation_view
from diagnose_recovery_context import camera_evidence

ROOT = Path(__file__).resolve().parents[1]


def cosine(a,b):
    return float(np.clip(np.dot(a.astype(np.float64),b.astype(np.float64)),-1,1))


def digest(vector):
    return hashlib.sha256(np.asarray(vector,dtype='<f4').tobytes()).hexdigest()


def key(value):
    return value['camera_id'],value['local_id'],value['frame_index']


class GalleryObserver:
    """Calls original functions once and records owned copies; never changes inputs/results."""
    def __init__(self):
        self.references={};self.events=[];self.pair_evidence=();self.current_raw={}

    @contextmanager
    def installed(self):
        build=registry_module.gallery_reference
        step=_FeatureArchive.step
        advance=ConfirmedReturnArchive._advance
        def gallery(*args,**kwargs):
            result=build(*args,**kwargs)
            reference,selected=result
            # Owned vectors already returned by the pure builder; copy for the diagnostic.
            from copy import deepcopy
            self.references[reference.global_id]=(deepcopy(reference),deepcopy(selected),kwargs['now'])
            return result
        def archive_step(owner,event):
            result=step(owner,event)
            self.pair_evidence=result.evidence
            return result
        def archive_advance(owner,event,raw,queries,filtered):
            references={**owner._archive._entries,**{r.global_id:r for r in event.retirements}}
            result=advance(owner,event,raw,queries,filtered)
            query_by_id={q.global_id:q for q in queries}
            for decision in result.decisions:
                if decision.outcome!='confirmed':continue
                gid=decision.archived_id;q=query_by_id[decision.provisional_id]
                ref=references[gid]
                require(gid in self.references,'Missing retired-gallery provenance')
                built,samples,retired_at=self.references[gid]
                require(built.last_seen==ref.last_seen and built.descriptor_time==ref.descriptor_time
                        and built.ground_time==ref.ground_time and built.ground_xy==ref.ground_xy
                        and np.array_equal(built.descriptor,ref.descriptor),'Archived reference differs from captured builder')
                reconstructed=np.mean([s.vector for s in samples],axis=0,dtype=np.float64)
                reconstructed=(reconstructed/np.linalg.norm(reconstructed)).astype(np.float32)
                require(np.array_equal(reconstructed,ref.descriptor),'Raw gallery does not reproduce reference')
                actual=cosine(q.descriptor,ref.descriptor)
                require(abs(actual-decision.cosine)<1e-12,'Accepted cosine differs')
                pairs=sorted((e for e in self.pair_evidence if e.members==q.members),
                             key=lambda e:(-e.similarity,e.global_id))
                feasible=[e for e in pairs if e.ground_distance<=e.distance_limit]
                require(feasible and feasible[0].global_id==gid,'Confirmed target not top feasible reference')
                match=feasible[0]
                rivals=[e for e in feasible if e.global_id!=gid]
                current=[self.current_raw[k] for k in q.members]
                rebuilt=np.mean(current,axis=0,dtype=np.float64)
                rebuilt=(rebuilt/np.linalg.norm(rebuilt)).astype(np.float32)
                require(np.array_equal(rebuilt,q.descriptor),'Current raw rows do not reproduce query')
                self.events.append(dict(frame_index=event.frame_index,provisional_id=q.global_id,global_id=gid,
                    timestamp=str(event.timestamp),support_rounds=decision.support_rounds,support_span=str(decision.support_span),
                    cosine=actual,feasible_reference_margin=actual-rivals[0].similarity if rivals else None,
                    geometry=dict(distance=match.ground_distance,limit=match.distance_limit,query_xy=q.ground_xy,
                                  reference_xy=ref.ground_xy,reference_ground_time=str(ref.ground_time)),
                    source_ages_seconds=dict(last_seen=float(event.timestamp-ref.last_seen),
                        oldest_descriptor_sample=float(event.timestamp-ref.descriptor_time),
                        ground=float(event.timestamp-ref.ground_time)),
                    reference=dict(retired_at=str(retired_at),last_seen=str(ref.last_seen),
                        descriptor_time=str(ref.descriptor_time),descriptor_sha256=digest(ref.descriptor)),
                    gallery=[dict(identity_key=asdict(s.key),timestamp=str(s.timestamp),confidence=s.confidence,
                        normalized_clearance=s.normalized_clearance,ground_xy=s.ground_xy,vector_sha256=digest(s.vector),
                        cosine_to_query=cosine(s.vector,q.descriptor)) for s in samples],
                    query=[dict(identity_key=asdict(k),vector_sha256=digest(v),cosine_to_reference=cosine(v,ref.descriptor))
                           for k,v in zip(q.members,current)],
                    top_references=[jsonable(asdict(e)) for e in pairs[:5]]))
            return result
        with ExitStack() as stack:
            stack.enter_context(patch.object(registry_module,'gallery_reference',gallery))
            stack.enter_context(patch.object(_FeatureArchive,'step',archive_step))
            stack.enter_context(patch.object(ConfirmedReturnArchive,'_advance',archive_advance))
            yield self


def replay(scene,source,report,paths,old,raw,means,space):
    cfg=source['configuration'];run=report['run_id'];scope=run+'/enabled'
    stage=ConfirmedReturnStage(scope,scene.matrices,scene.coordinate_space,space=space,
        variant=cfg['appearance_variant'],threshold=cfg['appearance_threshold'],**cfg['geometry'],
        identity_configuration=cfg['identity'],recovery_enabled=True,configuration=report['configuration'],
        image_sizes={c.camera_id:(c.width,c.height) for c in scene.cameras})
    observer=GalleryObserver();offset=0
    with observer.installed(),ExitStack() as stack:
        streams=[stack.enter_context(gzip.open(paths[n],'rt')) for n in ('trace','observations','history')]
        for frame in range(scene.rounds):
            lines=[f.readline() for f in streams];require(all(lines),'Truncated frozen input')
            row,cached,hrow=map(json.loads,lines)
            require(row['run_id']==run and row['frame_index']==frame,'Frozen trace scope differs')
            envelope=dict(frame_index=frame,timestamp=row['timestamp'],variants=dict(enabled=dict(
                cameras=row['cameras'],segment_bindings=row['segment_bindings'])))
            start=offset;records,batch,_,offset=inputs_for_round(scene,envelope,cached,hrow,offset,old)
            t=Fraction(frame,scene.fps)
            latest=FeatureBatch(scope,space,batch.keys,batch.timestamps,np.asarray(raw[start:offset]))
            averaged=FeatureBatch(scope,space,batch.keys,batch.timestamps,np.asarray(means[start:offset]))
            observer.current_raw={k:v for k,v in zip(latest.keys,latest.embeddings)}
            result,*_=stage.update_with_raw(frame,t,averaged,latest,records)
            audit=stage.registry.last_audit
            inverse={ObservationKey(**b['identity_key']):b['source_key'] for b in row['segment_bindings']}
            returns=[dict(provisional_id=e['provisional_id'],global_id=e['global_id'],
                          members=[inverse[k] for k in e['members']]) for e in audit.reactivations]
            actual=dict(identity_runtime=jsonable(asdict(result)),identity=evaluation_view(result,row['segment_bindings'],scope),
                recovery_decisions=jsonable(asdict(audit.archive)),reactivations=returns,lifecycle=audit.lifecycle)
            require(actual==row['variants']['enabled'],f'Enabled full record differs at frame {frame}')
            if (frame+1)%600==0:print(f'Replayed {frame+1}/{scene.rounds}; enabled state EXACT; captured returns={len(observer.events)}',flush=True)
        require(all(f.readline()=='' for f in streams) and offset==len(raw)==len(means),'Unused/extra feature rows')
    require(stage.registry.last_audit.lifecycle==report['summary']['enabled_lifecycle'],'Final lifecycle differs')
    require(len(observer.events)==report['summary']['enabled_lifecycle']['reactivation_events'],'Return coverage differs')
    return observer.events


def label_samples(trace,history,raw,events,scene,spec,ground):
    requests={}
    for event in events:
        for kind in ('gallery','query'):
            for sample in event[kind]:requests.setdefault(key(sample['identity_key']),[]).append(sample)
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras};seen=set()
    with gzip.open(trace,'rt') as a,gzip.open(history,'rt') as b:
        for frame in range(scene.rounds):
            row=json.loads(a.readline());hrow=json.loads(b.readline())
            require(row['frame_index']==hrow['frame_index']==frame,'Offline row mapping differs')
            wanted={k for k in requests if k[2]==frame}
            if not wanted:continue
            bindings={key(v['identity_key']):v['source_key'] for v in row['segment_bindings']}
            feature_rows={key(v['identity_key']):v['embedding_row'] for v in hrow['observations']}
            assignments={key(v['key']):v['global_id'] for v in row['variants']['enabled']['identity']['assignments']}
            for camera in row['cameras']:
                c=camera['camera'];local_wanted=[k for k in wanted if k[0]==c]
                if not local_wanted:continue
                width,height=sizes[c]
                if spec.first_frame<=frame<=spec.last_frame:
                    evidence=camera_evidence(camera,ground.slots[frame,c],width,height,spec.min_iou)
                else:
                    evidence={local:dict(unique_gt=None,reason='outside_evaluation_range',raw_xyxy=box,
                        best_gt=None,best_iou=None,admissible_gt=[]) for local,box in zip(camera['local_ids'],camera['xyxy'])}
                positions={i:j for j,i in enumerate(camera['local_ids'])}
                for k in local_wanted:
                    require(k in bindings and k in feature_rows,'Selected sample not in frozen mapping')
                    original=bindings[k];local=original['local_id'];j=positions[local];v=raw[feature_rows[k]]
                    for sample in requests[k]:
                        require(sample['vector_sha256']==digest(v),'Selected raw sample differs from persisted embedding')
                        if 'confidence' in sample:require(sample['confidence']==camera['confidence'][j],'Gallery confidence differs')
                        sample.update(source_key=original,embedding_row=feature_rows[k],
                            observed_global_id=assignments[key(original)],box_confidence=camera['confidence'][j],
                            spatial_evidence=evidence[local])
                    seen.add(k)
        require(a.readline()==b.readline()=='','Extra offline rows')
    require(seen==set(requests),'Unlabeled selected samples')
    for event in events:
        for kind in ('gallery','query'):
            samples=event[kind]
            event[kind+'_gt_counts']=dict(Counter(str(s['spatial_evidence']['unique_gt']) for s in samples))
            event[kind+'_spatial_reasons']=dict(Counter(s['spatial_evidence']['reason'] for s in samples))
    return events


def self_check():
    from types import SimpleNamespace
    from mtmc.association.confirmed_return import GallerySample, gallery_reference
    space=FeatureSpace('a'*64,1280);v=np.zeros(1280,np.float32);v[0]=1
    sample=GallerySample(ObservationKey(4,0,0),Fraction(0),v,.8,.1,(1.,1.))
    kwargs=dict(now=Fraction(1),space=space,max_age=Fraction(30))
    expected=gallery_reference(1,Fraction(0),(sample,),**kwargs)
    with GalleryObserver().installed() as observer:
        actual=registry_module.gallery_reference(1,Fraction(0),(sample,),**kwargs)
        require(np.array_equal(actual[0].descriptor,expected[0].descriptor),'Observer changed reference')
        require(observer.references[1][1][0].key==sample.key,'Sample provenance lost')
        actual[1][0].vector[:]=0
        require(np.array_equal(observer.references[1][1][0].vector,v),'Observer aliases returned samples')
    require(cosine(v,v)==1 and digest(v)==digest(v.copy()),'Vector audit differs')
    print('Original reference parity, ID zero provenance and owned observer samples: PASSED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-report',type=Path)
    parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:
        self_check()
        if args.recovery_report is None:return
    require(args.recovery_report is not None,'--recovery-report required')
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();value=sha256(path)
        require(expected is None or value==expected,'Changed input: '+str(path))
        pins[str(path)]=value;return path
    rp=pin(args.recovery_report);report=json.loads(rp.read_text())
    require(report['completed'] is True and report['protocol']=='confirmed_clip_return_paired_v1'
            and report['checks'] and all(report['checks'].values()),'Unverified source')
    # Raw GT is deliberately read only after replay and snapshot freezing.
    refs={}
    for name,ref in report['inputs'].items():
        if name!='evaluation_ground_truth':refs[name]=pin(ref['path'],ref['sha256'])
    source=json.loads(refs['experiment_report'].read_text())
    loaded=load_scene(refs['source:history:scene_config'],project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==report['scene']==source['scene'],'Mixed scenes')
    art=report['artifacts']['global_tracks.jsonl.gz'];trace=pin(rp.parent/art['path'],art['sha256'])
    for name in ('scripts/audit_confirmed_return_gallery.py','scripts/diagnose_recovery_context.py'):pin(ROOT/name)
    old=np.load(refs['source:history:source_vectors'],mmap_mode='r',allow_pickle=False)
    raw=np.load(refs['source:history:clipreid_embeddings.npy'],mmap_mode='r',allow_pickle=False)
    means=np.load(refs['source:artifact:mean_embeddings.npy'],mmap_mode='r',allow_pickle=False)
    space=FeatureSpace(**report['space'])
    for array in (raw,means):require(array.dtype==np.float32 and array.shape==(report['summary']['encoded'],space.dimension),'Invalid features')
    paths=dict(trace=trace,observations=refs['source:history:observations.jsonl.gz'],history=refs['source:artifact:history_rows.jsonl.gz'])
    print('Phase 1: exact enabled replay; capture actual gallery/query samples; no GT, models or decoding...',flush=True)
    events=replay(scene,source,report,paths,old,raw,means,space)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/confirmed_return_gallery'/run
    out.mkdir(parents=True,exist_ok=False)
    frozen=out/'unlabeled_samples.json';frozen.write_text(json.dumps(events,indent=2,allow_nan=False)+'\n');frozen_hash=sha256(frozen)
    print('Phase 2: selected samples frozen; offline GT labels for exact source boxes...',flush=True)
    ref=report['inputs']['evaluation_ground_truth']
    require(Path(ref['path']).resolve()==spec.ground_truth.path and ref['sha256']==spec.ground_truth.sha256,'Mixed GT source')
    pin(ref['path'],ref['sha256'])
    label_samples(trace,paths['history'],raw,events,scene,spec,load_ground_truth(spec))
    for path,value in pins.items():require(sha256(Path(path))==value,'Input changed during audit')
    require(sha256(frozen)==frozen_hash,'Selection changed during offline labeling')
    output=out/'report.json'
    output.write_text(json.dumps(dict(completed=True,protocol='confirmed_return_gallery_audit_v1',scene=scene.scene,
        source_run_id=report['run_id'],inputs=pins,events=events,
        checks=dict(enabled_full_records_exact=True,raw_gallery_reference_exact=True,query_raw_mean_exact=True,
            accepted_cosine_reproduced=True,selected_vectors_match_persisted_rows=True,selection_frozen_before_gt=True,inputs_unchanged=True),
        artifacts=dict(unlabeled_samples=dict(path=frozen.name,sha256=frozen_hash)),
        limits=['GT labels describe selected boxes only; unknown and mixed references remain explicit.',
                'All camera predictions participate in spatial uniqueness.',
                'No thresholds, sample selection, ranking, identity assignments or metrics are changed.',
                'Archive provenance is audited for every confirmed return, not just GT-selected failures.']),indent=2,allow_nan=False)+'\n')
    for event in events:
        print(f"Frame {event['frame_index']}: G{event['provisional_id']} -> G{event['global_id']}; cosine={event['cosine']:.6f}; margin={event['feasible_reference_margin']}")
        print('  Gallery GT:',json.dumps(event['gallery_gt_counts']),'Query GT:',json.dumps(event['query_gt_counts']))
        print('  Geometry:',json.dumps(event['geometry']),'Ages:',json.dumps(event['source_ages_seconds']))
    print('Report:',output)
    print('Confirmed return gallery audit: COMPLETED; exact replay and sample provenance VERIFIED; predictions unchanged')


if __name__=='__main__':main()
