"""Freeze appearance-admission masks, then audit their coverage using offline GT."""
import argparse,gzip,json
from collections import Counter,defaultdict
from datetime import datetime,timezone
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from mtmc.data.scene import load_scene,require,sha256
from mtmc.data.ground_truth import load_ground_truth,spatial_slot
from mtmc.reid.osnet import ObservationKey
from mtmc.reid.crops import CropRecord,crop_geometry
from mtmc.reid.sample_quality import assess_sample

ROOT=Path(__file__).resolve().parents[1]
NAMES=('all_available','confidence_only','confidence_and_border')


def tracked_records(row,scene):
    frame=row['frame_index'];data=row['variants']['enabled'];cams=data['cameras']
    require([c['camera'] for c in cams]==list(scene.camera_ids),'Camera coverage differs')
    sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    bindings={ObservationKey(**b['source_key']):ObservationKey(**b['identity_key']) for b in data['segment_bindings']}
    records=[]
    for cam in cams:
        c=cam['camera'];ids=cam['local_ids'];require(len(set(ids))==len(ids) and all(len(cam[k])==len(ids) for k in ('xyxy','confidence','embedding_rows')),'Invalid track rows')
        for i,local in enumerate(ids):
            key=ObservationKey(c,local,frame);bounds,fraction=crop_geometry(cam['xyxy'][i],*sizes[c])
            require((bounds is None)==(cam['embedding_rows'][i] is None),'Source crop availability differs')
            records.append(CropRecord(key,cam['confidence'][i],tuple(cam['xyxy'][i]),bounds,fraction))
    require(len(bindings)==len(data['segment_bindings'])==len(records) and set(bindings)=={r.key for r in records},'Segment mapping differs')
    return records,bindings


def freeze(trace,path,scene,run_id,settings):
    count=0;sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    with gzip.open(trace,'rt') as stream,gzip.open(path,'xt') as output:
        for frame in range(scene.rounds):
            line=stream.readline();require(line,'Truncated source');row=json.loads(line)
            require(row['run_id']==run_id and row['frame_index']==frame and Fraction(row['timestamp'])==Fraction(frame,scene.fps),'Mixed source scope')
            records,bindings=tracked_records(row,scene);items=[]
            for record in records:
                quality=assess_sample(record,*sizes[record.key.camera_id],**settings)
                accepted=quality.policies
                require(not accepted['confidence_and_border'] or accepted['confidence_only'],'Invalid subset')
                require(not accepted['confidence_only'] or accepted['all_available'],'Invalid availability subset')
                items.append({'key':asdict(record.key),'identity_key':asdict(bindings[record.key]),'accepted':accepted,
                    'normalized_clearance':quality.normalized_clearance,'confidence':record.confidence})
            count+=len(items);output.write(json.dumps({'frame_index':frame,'items':items},allow_nan=False)+'\n')
            if (frame+1)%600==0:print(f'Admission masks frozen through {frame+1}/{scene.rounds}',flush=True)
        require(stream.readline()=='','Trailing source frames')
    return count


def audit(trace,masks,scene,spec,ground):
    totals={n:Counter() for n in NAMES};cameras={n:{c:Counter() for c in scene.camera_ids} for n in NAMES}
    observed=set();accepted_segments={n:set() for n in NAMES};person_counts=defaultdict(Counter)
    observed_people=set();accepted_people={n:set() for n in NAMES};sizes={c.camera_id:(c.width,c.height) for c in scene.cameras}
    cases=[]
    with gzip.open(trace,'rt') as stream,gzip.open(masks,'rt') as decisions:
        for frame in range(scene.rounds):
            a,b=stream.readline(),decisions.readline();require(a and b,'Truncated evaluation source');row,maskrow=json.loads(a),json.loads(b)
            require(row['frame_index']==maskrow['frame_index']==frame,'Mask frame differs')
            entries={ObservationKey(**r['key']):r for r in maskrow['items']};require(len(entries)==len(maskrow['items']),'Duplicate mask key')
            if not spec.first_frame<=frame<=spec.last_frame:continue
            data=row['variants']['enabled'];seen=set()
            for camera in data['cameras']:
                c=camera['camera'];keys=tuple(ObservationKey(c,i,frame) for i in camera['local_ids']);seen.update(keys)
                gt,spatial,unique,_,_=spatial_slot(ground.slots[frame,c],keys,camera['xyxy'],width=sizes[c][0],height=sizes[c][1],min_iou=spec.min_iou)
                observed_people.update(gt)
                for j,key in enumerate(keys):
                    item=entries[key];effective=item['identity_key'];segment=(c,effective['local_id']);observed.add(segment)
                    label=unique.get(key)
                    category='mutually_unique_gt' if label is not None else 'no_admissible_gt' if not spatial[:,j].any() else 'ambiguous_gt'
                    for name in NAMES:
                        accepted=item['accepted'][name]
                        for counter in (totals[name],cameras[name][c]):
                            counter['observations']+=1;counter['accepted']+=accepted;counter['rejected']+=not accepted
                            counter[category]+=1;counter[('accepted_' if accepted else 'rejected_')+category]+=1
                        if accepted:accepted_segments[name].add(segment)
                        if label is not None:
                            person_counts[label][name+'_observations']+=1;person_counts[label][name+'_accepted']+=accepted
                            if accepted:accepted_people[name].add(label)
                    if scene.camera_ids==(361,362,364) and c==361 and ((1673<=frame<=1680 and key.local_id==22) or (frame in (1648,1649,1651) and key.local_id==22) or (frame in (2747,2779,2780,2781) and key.local_id==33)):
                        cases.append({**item,'diagnostic_gt':label,'spatial_category':category})
            require(seen==set(entries),'Mask observation coverage differs')
        require(stream.readline()==decisions.readline()=='','Trailing audit frames')
    for name in NAMES:
        counter=totals[name]
        require(counter['accepted']+counter['rejected']==counter['observations'],'Admission accounting differs')
        require(sum(counter[c] for c in ('mutually_unique_gt','no_admissible_gt','ambiguous_gt'))==counter['observations'],'GT category accounting differs')
    return {'totals':{n:dict(c) for n,c in totals.items()},'per_camera':{n:{str(k):dict(v) for k,v in c.items()} for n,c in cameras.items()},
        'segments_observed':len(observed),'segments_without_accepted_sample':{n:len(observed-accepted_segments[n]) for n in NAMES},
        'GT_people_without_uniquely_matched_accepted_sample':{n:sorted(observed_people-accepted_people[n]) for n in NAMES},
        'per_gt_unique_evidence':{str(k):dict(v) for k,v in sorted(person_counts.items())},'inspected_visual_cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-report',type=Path,required=True);args=p.parse_args();inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path);require(digest is None or actual==digest,'Changed input: '+str(path));inputs[name]={'path':str(path),'sha256':actual};return path
    rp=checked('source_report',args.source_report);source=json.loads(rp.read_text())
    require(source.get('completed') is True and source['protocol'] in ('appearance_continuity_paired_v1','appearance_continuity_scene_transfer_v1') and source['checks'] and all(source['checks'].values()),'Expected completed continuity experiment')
    key='scene_config' if source['protocol']=='appearance_continuity_paired_v1' else 'validation:scene_config'
    item=source['inputs'][key];loaded=load_scene(checked('scene_config',item['path'],item['sha256']),project_root=ROOT)
    scene,spec=loaded.runtime,loaded.evaluation;require(scene.scene==source['scene'],'Mixed scene')
    gt_key='ground_truth' if source['protocol']=='appearance_continuity_paired_v1' else 'validation_ground_truth'
    require(source['inputs'][gt_key]['sha256']==spec.ground_truth.sha256,'Mixed ground-truth lineage')
    item=source['artifacts']['global_tracks.jsonl.gz'];trace=checked('global_tracks',rp.parent/item['path'],item['sha256'])
    configuration=json.loads(checked('configuration',ROOT/'configs/reid/sample_admission_experiment.json').read_text())
    require(configuration['experiment']=='confidence_and_normalized_border_admission_v1' and configuration['min_confidence']==.5 and configuration['border_fraction']==.01 and configuration['threshold_search'] is False and configuration['variants']==list(NAMES),'Declared hypothesis changed')
    settings={k:configuration[k] for k in ('min_confidence','border_fraction')}
    for rel in ('scripts/audit_sample_admission.py','src/mtmc/reid/sample_quality.py','src/mtmc/reid/crops.py','src/mtmc/data/ground_truth.py'):checked('code:'+rel,ROOT/rel)
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/sample_admission'/run;out.mkdir(parents=True,exist_ok=False);mask=out/'admission.jsonl.gz'
    print('Phase 1: sample decisions from frozen boxes/confidence; no GT, models or runtime changes...',flush=True)
    count=freeze(trace,mask,scene,source['run_id'],settings);require(count==source['summary']['observations'],'Runtime observation count differs');digest=sha256(mask)
    print('Phase 2: masks frozen; offline GT coverage audit...',flush=True)
    checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
    result=audit(trace,mask,scene,spec,load_ground_truth(spec));expected=source['global']['enabled']['full']['predicted_observations']
    require(all(v['observations']==expected for v in result['totals'].values()),'Evaluation denominator differs')
    for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during audit')
    require(sha256(mask)==digest,'Frozen masks changed')
    result.update(run_id=run,completed=True,protocol='frozen_sample_admission_audit_v1',scene=scene.scene,source_run_id=source['run_id'],inputs=inputs,configuration=configuration,
        runtime_observations=count,evaluation_frames=[spec.first_frame,spec.last_frame],artifacts={'admission.jsonl.gz':{'path':mask.name,'sha256':digest}},
        limits=['Masks are not integrated into histories, archives, tracking or global identity; quality metrics are unchanged by construction.',
        'Mutually unique GT is spatial evidence, not proof that a crop is suitable for Re-ID.',
        'No-admissible-GT and ambiguous-GT categories are not necessarily background/false detections.',
        'Segment sample coverage is measured only within the evaluation interval, not descriptor availability over time.',
        'Both scenes have informed development; this is not independent validation.'])
    path=out/'report.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Scene:',scene.scene)
    print('Policy / accepted / rejected / accepted unique GT / rejected unique GT / accepted no GT / accepted ambiguous')
    for name,v in result['totals'].items():print(name,v.get('accepted',0),v.get('rejected',0),v.get('accepted_mutually_unique_gt',0),v.get('rejected_mutually_unique_gt',0),v.get('accepted_no_admissible_gt',0),v.get('accepted_ambiguous_gt',0))
    print('Segments without accepted sample:',result['segments_without_accepted_sample'])
    print('GT people without unique accepted sample:',result['GT_people_without_uniquely_matched_accepted_sample'])
    print('Visual cases: frame / local / score / normalized clearance / accepted policies')
    for c in result['inspected_visual_cases']:print(c['key']['frame_index'],c['key']['local_id'],round(c['confidence'],4),round(c['normalized_clearance'],6),c['accepted'])
    print('Report:',path);print('Sample admission audit: COMPLETED; predictions and runtime unchanged')


if __name__=='__main__':main()
