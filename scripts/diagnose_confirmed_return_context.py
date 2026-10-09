"""Offline context for confirmed returns; all predictions participate in GT matching."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from diagnose_recovery_context import camera_evidence, self_check as spatial_checks
from mtmc.data.ground_truth import load_ground_truth
from mtmc.data.scene import load_scene, require, sha256

ROOT = Path(__file__).resolve().parents[1]


def key(value):
    return value['camera_id'], value['local_id'], value['frame_index']


def context_category(before, after):
    a, b = set(before), set(after)
    if not a or not b:
        return 'insufficient_context'
    if len(a) != 1 or len(b) != 1:
        return 'mixed_context'
    return 'same_context_gt' if a == b else 'different_context_gt'


def inspect(trace, scene, spec, ground, run_id, events, context):
    requests = {}
    for number, event in enumerate(events, 1):
        prior = event['previous_visible_evidence']
        require(prior is not None and prior['frame_index'] < event['frame_index'], 'Invalid previous evidence')
        for side, source, boundary in (('before', prior['members'], prior['frame_index']),
                                       ('after', event['members'], event['frame_index'])):
            start = max(spec.first_frame, boundary-context) if side == 'before' else boundary
            stop = boundary if side == 'before' else min(spec.last_frame, boundary+context)
            for member in source:
                camera, local, frame = key(member)
                require(frame == boundary, 'Endpoint frame differs')
                for f in range(start, stop+1):
                    requests.setdefault((f, camera), []).append((number, side, local, boundary))
    sizes = {c.camera_id:(c.width,c.height) for c in scene.cameras}
    rows, observed, decisions = [], [], []
    count = 0
    with gzip.open(trace, 'rt', encoding='utf-8') as stream:
        for frame, line in enumerate(stream):
            value = json.loads(line); count += 1
            require(value['run_id'] == run_id and value['frame_index'] == frame
                    and value['timestamp'] == str(Fraction(frame, scene.fps)), 'Trace scope/time differs')
            data = value['variants']['enabled']
            observed.extend((frame, e) for e in data['reactivations'])
            audit = data.get('recovery_decisions')
            if audit:
                targets = {e['provisional_id'] for e in events}
                for decision in audit['decisions']:
                    if decision['provisional_id'] in targets:
                        decisions.append(dict(frame_index=frame, **decision))
            assignments = {key(a['key']):a['global_id'] for a in data['identity']['assignments']}
            segments = {key(b['source_key']):key(b['identity_key'])[:2] for b in value['segment_bindings']}
            for camera in value['cameras']:
                c = camera['camera']
                if (frame,c) not in requests:
                    continue
                width,height = sizes[c]
                # Do not restrict predictions to selected tracks before computing uniqueness.
                labels = camera_evidence(camera, ground.slots[frame,c], width,height,spec.min_iou)
                for number,side,local,boundary in requests[frame,c]:
                    row = dict(event=number,side=side,frame_index=frame,camera_id=c,local_id=local,
                               boundary=frame==boundary,present=local in labels)
                    if row['present']:
                        k=(c,local,frame)
                        require(k in assignments and k in segments, 'Missing assignment/segment')
                        row.update(labels[local],global_id=assignments[k],segment=list(segments[k]))
                    rows.append(row)
    require(count==scene.rounds, 'Incomplete trace')
    require(observed==[(e['frame_index'],{k:e[k] for k in ('provisional_id','global_id','members')})
                       for e in events], 'Saved return events differ')
    summaries=[]
    for number,event in enumerate(events,1):
        sides={}
        for side in ('before','after'):
            current=[r for r in rows if r['event']==number and r['side']==side]
            endpoint=[r for r in current if r['boundary']]
            prior=event['previous_visible_evidence']
            members=prior['members'] if side=='before' else event['members']
            expected=prior['labels'] if side=='before' else event['current_labels']
            require(len(endpoint)==len(members) and all(r['present'] for r in endpoint), 'Missing endpoint')
            lookup={(r['camera_id'],r['local_id']):r for r in endpoint}
            require([lookup[m['camera_id'],m['local_id']]['unique_gt'] for m in members]==expected
                    and all(r['global_id']==event['global_id'] for r in endpoint), 'Endpoint labels/IDs differ')
            eligible=[]
            for row in current:
                ref=lookup[row['camera_id'],row['local_id']]
                row['same_endpoint_scope']=(row['present'] and row['segment']==ref['segment']
                                             and row['global_id']==ref['global_id'])
                if row['same_endpoint_scope']: eligible.append(row)
            known=[r for r in eligible if r['unique_gt'] is not None]
            nearest=(max if side=='before' else min)((r['frame_index'] for r in known),default=None)
            sides[side]=dict(boundary=endpoint,unique_gt_counts=dict(Counter(r['unique_gt'] for r in known)),
                reason_counts=dict(Counter(r['reason'] for r in eligible)),
                absent_slots=sum(not r['present'] for r in current),
                excluded_other_segment_or_gid=sum(r['present'] and not r['same_endpoint_scope'] for r in current),
                nearest_known=[{k:r[k] for k in ('frame_index','camera_id','local_id','segment','global_id','unique_gt')}
                               for r in known if r['frame_index']==nearest])
        summaries.append(dict(event=number,global_id=event['global_id'],provisional_id=event['provisional_id'],
            frame_index=event['frame_index'],original_category=event['category'],
            gap_seconds=str(Fraction(event['frame_index']-event['previous_visible_evidence']['frame_index'],scene.fps)),
            context_category=context_category(sides['before']['unique_gt_counts'],sides['after']['unique_gt_counts']),
            **sides))
    return summaries,rows,decisions


def self_check():
    spatial_checks()
    require(context_category({0:2},{0:3})=='same_context_gt', 'ID zero lost')
    require(context_category({0:2},{1:3})=='different_context_gt', 'Mismatch lost')
    require(context_category({0:2,1:1},{1:3})=='mixed_context', 'Mixing lost')
    require(context_category({},{1:3})=='insufficient_context', 'Unknown treated as known')
    scene=SimpleNamespace(rounds=8,fps=30,cameras=[SimpleNamespace(camera_id=4,width=100,height=100)])
    spec=SimpleNamespace(first_frame=0,last_frame=7,min_iou=.5)
    ground=SimpleNamespace(slots={}); trace=[]
    event=dict(frame_index=5,category='unresolved',provisional_id=2,global_id=1,
        members=[dict(camera_id=4,local_id=9,frame_index=5)],current_labels=[0],
        previous_visible_evidence=dict(frame_index=2,members=[dict(camera_id=4,local_id=8,frame_index=2)],labels=[None]))
    for f in range(8):
        present=f<=2 or f>=5;local=8 if f<=2 else 9;box=[10,10,30,60]
        # Frame 0 and frame 7 have another segment; neither may contaminate context.
        segment=100 if f in (0,7) else local
        label=1 if f in (0,7) else 0
        ground.slots[f,4]={label:box} if present and f!=2 else {}
        k=dict(camera_id=4,local_id=local,frame_index=f)
        trace.append(dict(run_id='fixture',frame_index=f,timestamp=str(Fraction(f,30)),
            cameras=[dict(camera=4,local_ids=[local] if present else [],xyxy=[box] if present else [])],
            segment_bindings=[dict(source_key=k,identity_key=dict(k,local_id=segment))] if present else [],
            variants=dict(enabled=dict(identity=dict(assignments=[dict(key=k,global_id=1)] if present else []),
                reactivations=[{n:event[n] for n in ('provisional_id','global_id','members')}] if f==5 else [],
                recovery_decisions=None))))
    with tempfile.TemporaryDirectory() as d:
        path=Path(d)/'trace.gz'
        with gzip.open(path,'wt') as stream:
            for row in trace:stream.write(json.dumps(row)+'\n')
        summaries,rows,_=inspect(path,scene,spec,ground,'fixture',[event],2)
    result=summaries[0]
    require(result['original_category']=='unresolved' and result['context_category']=='same_context_gt', 'Context changed endpoint')
    require(result['before']['unique_gt_counts']=={0:1} and result['after']['unique_gt_counts']=={0:2}, 'Scope filtering differs')
    require(result['before']['excluded_other_segment_or_gid']==1 and result['after']['excluded_other_segment_or_gid']==1,'Segments crossed')
    print('Serialized context, endpoint reproduction, ID zero and segment boundaries: PASSED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-report',type=Path)
    parser.add_argument('--context-frames',type=int,default=30)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        self_check()
        if args.recovery_report is None:return
    require(args.recovery_report is not None and 0<=args.context_frames<=120,'Report required; context must be 0..120')
    pins={}
    def pin(path,digest=None):
        path=Path(path).resolve();actual=sha256(path)
        require(digest is None or actual==digest,'Changed input: '+str(path))
        pins[str(path)]=actual;return path
    rp=pin(args.recovery_report);report=json.loads(rp.read_text())
    require(report['completed'] is True and report['protocol']=='confirmed_clip_return_paired_v1'
            and report['checks'] and all(report['checks'].values()),'Unverified source')
    paths={}
    for name in ('global_tracks.jsonl.gz','reactivation_diagnostics.json'):
        ref=report['artifacts'][name];path=(rp.parent/ref['path']).resolve()
        require(path.parent==rp.parent,'Artifact escaped run')
        paths[name]=pin(path,ref['sha256'])
    ref=report['inputs']['source:history:scene_config'];config=pin(ref['path'],ref['sha256'])
    loaded=load_scene(config,project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==report['scene'],'Scene mismatch')
    ref=report['inputs']['evaluation_ground_truth']
    require(Path(ref['path']).resolve()==spec.ground_truth.path and ref['sha256']==spec.ground_truth.sha256,'Mixed GT')
    pin(ref['path'],ref['sha256'])
    for file in (Path(__file__),ROOT/'scripts/diagnose_recovery_context.py',ROOT/'src/mtmc/data/ground_truth.py',ROOT/'src/mtmc/data/scene.py'):pin(file)
    events=json.loads(paths['reactivation_diagnostics.json'].read_text())
    require(len(events)==report['summary']['enabled_lifecycle']['reactivation_events'],'Event population differs')
    require(dict(Counter(e['category'] for e in events))==report['return_gt_diagnostics'],'Endpoint category counts differ')
    print('Frozen confirmed-return context; CPU; no models, decoding or state changes...',flush=True)
    summaries,rows,decisions=inspect(paths['global_tracks.jsonl.gz'],scene,spec,load_ground_truth(spec),report['run_id'],events,args.context_frames)
    for path,digest in pins.items():require(sha256(Path(path))==digest,'Input changed during diagnostic')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'artifacts/confirmed_return_context'/run;out.mkdir(parents=True,exist_ok=False)
    for name,value in (('observations.json',rows),('confirmation_decisions.json',decisions)):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    result=dict(completed=True,protocol='confirmed_return_context_v1',source_run_id=report['run_id'],scene=scene.scene,
        inputs=pins,context_frames=args.context_frames,events=summaries,
        context_counts=dict(Counter(e['context_category'] for e in summaries)),
        checks=dict(endpoint_labels_exact=True,events_exact=True,source_files_unchanged=True),
        artifacts={n:dict(path=n,sha256=sha256(out/n)) for n in ('observations.json','confirmation_decisions.json')},
        limits=['Offline context includes future frames; never used by runtime.',
                'Context is restricted to the endpoint local track, segment and global ID.',
                'Context evidence does not prove gallery purity or whole-identity correctness.',
                'Original endpoint categories and predictions remain unchanged.'])
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for event in summaries:
        print(f"\nFrame {event['frame_index']}: G{event['provisional_id']} -> G{event['global_id']}; {event['original_category']} / {event['context_category']}")
        for side in ('before','after'):
            value=event[side]
            print(f"  {side}: labels={value['unique_gt_counts']}; reasons={value['reason_counts']}; excluded_scope={value['excluded_other_segment_or_gid']}; absent={value['absent_slots']}")
            print('  nearest:',json.dumps(value['nearest_known']))
    print('\nContext counts:',json.dumps(result['context_counts']))
    print('Report:',out/'report.json')
    print('Confirmed return context: COMPLETED; endpoints reproduced; predictions unchanged')


if __name__=='__main__':main()
