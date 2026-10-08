"""Prepare scene configs and check old-input parity plus empty-GT contracts."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np
from mtmc.data.scene import FileRef, EvaluationSpec, load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth, spatial_slot

ROOT=Path(__file__).resolve().parents[1]


def reject(call):
    try:call()
    except (ValueError,KeyError,FileNotFoundError):return
    raise AssertionError('Invalid input was accepted')


def fixture(root):
    scene='MTMC_Tracking_2024/val/scene_041'; ids=[11,27]
    base={'dataset':'nvidia/PhysicalAI-SmartSpaces','revision':'a'*40,'scene':scene}
    def write(name,value):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value));return path
    sensors=[]
    for c,w,h in [(11,200,100),(27,320,240)]:
        sensors.append({'id':f'Camera_{c:04d}','type':'camera',
                        'attributes':[{'name':k,'value':str(v)} for k,v in [('fps',25),('frameWidth',w),('frameHeight',h)]],
                        'intrinsicMatrix':np.eye(3).tolist(),
                        'extrinsicMatrix':[[1,0,0,0],[0,1,0,0],[0,0,1,1]],
                        'cameraMatrix':[[1,0,0,0],[0,1,0,0],[0,0,1,1]],'homography':np.eye(3).tolist()})
    cal=write('cal.json',{'sensors':sensors})
    gt=root/'gt.txt';gt.write_text('11 0 0 1 2 3 4 5 6\n27 8 2 5 6 7 8 9 10\n')
    source=write('source.json',{**base,'files':[{'remote_path':scene+'/'+name,'local_path':p.name,'sha256':sha256(p)}
                 for name,p in [('ground_truth.txt',gt),('calibration_2025_format.json',cal)]]})
    files=[]
    for c in ids:
        p=root/f'video{c}.mp4';p.write_bytes(b'synthetic-not-decoded')
        files.append({'camera':c,'remote_path':f'{scene}/camera_{c:04d}/video.mp4','local_path':p.name,
                      'size_bytes':p.stat().st_size,'sha256':sha256(p)})
    videos=write('videos.json',{**base,'camera_ids':ids,'files':files})
    config={**base,'schema_version':1,'camera_ids':ids,'fps':25,'runtime':{'first_frame':0,'rounds':3},
            'evaluation':{'first_frame':0,'last_frame':2,'gt_to_video_offset':0,'min_iou':.5,
                          'empty_slot_policy':'retain_slot_and_count_predictions'},
            'source_manifest':{'path':source.name,'sha256':sha256(source)},
            'video_manifest':{'path':videos.name,'sha256':sha256(videos)}}
    cp=write('config.json',config)
    return cp,config,gt


def synthetic_checks():
    from evaluate_mtmc_pipeline import spatial_slot as old_spatial
    from evaluate_global_identity import IdentityCounts, reference_metrics, check_reference
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);cp,cfg,gt=fixture(root)
        inputs=load_scene(cp,project_root=root)
        require(inputs.runtime.camera_ids==(11,27) and inputs.runtime.fps==25
                and [(c.width,c.height) for c in inputs.runtime.cameras]==[(200,100),(320,240)],'Runtime fixture differs')
        backup=gt.with_suffix('.saved');gt.rename(backup)
        try:
            load_scene(cp,project_root=root)
            reject(lambda:load_ground_truth(inputs.evaluation))
        finally:backup.rename(gt)
        print('Runtime scene loads without reading GT; offline evaluation requires GT: OK')
        ground=load_ground_truth(inputs.evaluation)
        require(len(ground.slots)==6 and len(ground.empty_slots)==4 and
                ground.slots[0,11][0]==[1.,2.,4.,6.] and ground.slots[1,27]=={},'GT fixture differs')
        reject(lambda:ground.slots[1,99])
        shifted=load_ground_truth(replace(inputs.evaluation,gt_to_video_offset=-1))
        require(shifted.slots[1,27][8]==[5.,6.,12.,14.] and not shifted.slots[2,27], 'Explicit GT offset differs')
        print('Nonstandard cameras/FPS/sizes, ID zero, empty slots and explicit frame mapping: OK')
        original=gt.read_text()
        for text in [original+original.splitlines()[0]+'\n','11 0 0 1 2 0 4 5 6\n','11 0 0 nan 2 3 4 5 6\n','11 0 0 1 2 3 4\n']:
            gt.write_text(text)
            reject(lambda:load_ground_truth(replace(inputs.evaluation,ground_truth=FileRef(gt,sha256(gt)))))
        gt.write_text(original)
        gt.write_text(original+'\n');reject(lambda:load_ground_truth(inputs.evaluation));gt.write_text(original)
        for changes in [{'camera_ids':[11,11]},{'fps':30},{'runtime':{'first_frame':1,'rounds':3}},
                        {'evaluation':{**cfg['evaluation'],'last_frame':3}},
                        {'video_manifest':{**cfg['video_manifest'],'sha256':'0'*64}},
                        {'source_manifest':{**cfg['source_manifest'],'path':'../source.json'}}]:
            cp.write_text(json.dumps({**cfg,**changes}));reject(lambda:load_scene(cp,project_root=root))
        cp.write_text(json.dumps(cfg))
        print('Duplicate/malformed GT, changed checksums and incompatible scene configs rejected: OK')
    cases=[({},[],[]),({},[1],[[0,0,10,10]]),({0:[0,0,10,10]},[],[]),
           ({0:[0,0,10,10]},[1,2],[[0,0,5,10],[2000,0,2010,10]]),
           ({0:[2000,0,2010,10]},[1],[[2000,0,2010,10]])]
    for gt,keys,boxes in cases:
        a=spatial_slot(gt,keys,boxes,width=1920,height=1080);b=old_spatial(gt,keys,boxes)
        require(a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:],'Spatial known-case parity differs')
    exact=spatial_slot({0:[0,0,10,10]},[1],[[0,0,5,10]],width=20,height=20)
    require(exact[1].tolist()==[[True]],'Inclusive IoU=.5 boundary changed')
    empty=spatial_slot({},[7],[[0,0,10,10]],width=20,height=20)
    counts=IdentityCounts();counts.update(empty[0],[7],empty[1]);metrics,_=counts.result()
    require(metrics['idfp']==1 and metrics['idfn']==0 and metrics['camera_time_slots']==1,'Empty GT prediction was dropped')
    check_reference(metrics,reference_metrics([(empty[0],[7],empty[1])]))
    rng=np.random.default_rng(21)
    for _ in range(100):
        n,m=map(int,rng.integers(0,9,size=2));xy=rng.uniform(-100,2000,(n,2));wh=rng.uniform(1,300,(n,2))
        gt={i:list(row) for i,row in enumerate(np.concatenate((xy,xy+wh),axis=1))}
        xy=rng.uniform(-100,2000,(m,2));wh=rng.uniform(1,300,(m,2));boxes=np.concatenate((xy,xy+wh),axis=1)
        a=spatial_slot(gt,list(range(m)),boxes,width=1920,height=1080);b=old_spatial(gt,list(range(m)),boxes)
        require(a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:],'Spatial random parity differs')
    print('Clipping, all IoU candidates, empty masks, outside boxes and legacy spatial parity: OK')
    print('Prediction in empty GT slot counts as IDFP; motmetrics agreement: OK')


def checked_report(path,inputs,name):
    path=Path(path).resolve();inputs[name]={'path':str(path),'sha256':sha256(path)}
    return json.loads(path.read_text())


def pinned_ref(spec):
    path=Path(spec['path']).resolve()
    require(path.is_relative_to(ROOT.resolve()),'Evidence path is outside project')
    require(sha256(path)==spec['sha256'],'Input manifest changed')
    return {'path':path.relative_to(ROOT.resolve()).as_posix(),'sha256':spec['sha256']}


def prepare_config(source_spec,video_spec,cameras,scene,provenance):
    source_ref,video_ref=pinned_ref(source_spec),pinned_ref(video_spec)
    source=json.loads((ROOT/source_ref['path']).read_text())
    require(source['scene']==scene,'Unexpected scene')
    config={k:source[k] for k in ('dataset','revision','scene')}
    config.update(schema_version=1,camera_ids=list(cameras),fps=30,runtime={'first_frame':0,'rounds':3600},
                  evaluation={'first_frame':2,'last_frame':3599,'gt_to_video_offset':0,'min_iou':.5,
                              'empty_slot_policy':'retain_slot_and_count_predictions'},
                  source_manifest=source_ref,video_manifest=video_ref,preparation_evidence=provenance)
    path=ROOT/'configs/scenes'/f'{Path(scene).name}_two_minutes.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():require(json.loads(path.read_text())==config,'Existing scene config differs; not overwritten')
    else:path.write_text(json.dumps(config,indent=2)+'\n')
    return path,load_scene(path,project_root=ROOT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run-report',type=Path)
    parser.add_argument('--validation-audit-report',type=Path)
    parser.add_argument('--synthetic-only',action='store_true')
    args=parser.parse_args()
    synthetic_checks()
    if args.synthetic_only:
        print('Scene input synthetic checks: PASSED');return
    require(args.training_run_report is not None and args.validation_audit_report is not None,'Both real-data reports are required')
    inputs={}
    train=checked_report(args.training_run_report,inputs,'training_run_report')
    validation=checked_report(args.validation_audit_report,inputs,'validation_audit_report')
    require(train['completed'] and train['protocol']=='scene_001_mtmc_sequential_fp32_v1'
            and train['summary']['rounds']==3600 and train['configuration']['cameras']==[4,5,8]
            and train['configuration']['fps']==30 and train['configuration']['identity_start_frame']==0,
            'Expected the frozen two-minute scene_001 run')
    require(validation['completed'] and validation['protocol']=='validation_scene_audit_v1'
            and validation['timestamp_audit_passed'] and validation['cameras']==[361,362,364]
            and validation['frame_range']==[2,3599], 'Expected validation scene audit')
    train_config,a=prepare_config(train['inputs']['scene_source_manifest'],train['inputs']['video_manifest'],
                    (4,5,8),'MTMC_Tracking_2024/train/scene_001',pinned_ref(inputs['training_run_report']))
    val_config,b=prepare_config(validation['inputs']['source_manifest'],validation['inputs']['video_manifest'],
                    (361,362,364),'MTMC_Tracking_2024/val/scene_041',pinned_ref(inputs['validation_audit_report']))
    from evaluate_mtmc_sequence import load_ground_truth as old_gt
    from evaluate_mtmc_pipeline import spatial_slot as old_spatial
    from run_mtmc import load_matrices as old_matrices
    print('Loading both scenes on CPU; checking legacy scene_001 GT and calibration...',flush=True)
    ground=load_ground_truth(a.evaluation);legacy=old_gt(a.evaluation.ground_truth.path,3600)
    require(ground.slots==dict(legacy) and not ground.empty_slots,'scene_001 GT changed')
    matrices=old_matrices(a.runtime.calibration.path,a.runtime.camera_ids)
    require(all(np.array_equal(matrices[c],a.runtime.matrices[c]) for c in matrices),'scene_001 calibration changed')
    require(a.runtime.coordinate_space==train['configuration']['coordinate_space'],'scene_001 coordinate namespace changed')
    print('scene_001: every GT slot/ID/box and every homography EXACT',flush=True)
    trace_item=train['artifacts']['tracks'];trace=args.training_run_report.resolve().parent/trace_item['path']
    require(sha256(trace)==trace_item['sha256'],'Frozen tracks changed')
    inputs['training_tracks']={'path':str(trace),'sha256':trace_item['sha256']}
    compared=predictions=0
    with trace.open() as handle:
        for f in range(3600):
            text=handle.readline();require(bool(text),'Truncated trace');record=json.loads(text)
            require(record['frame_index']==f and record['run_id']==train['run_id']
                    and Fraction(record['timestamp'])==Fraction(f,30),'Trace frame/time/scope differs')
            require(sorted(c['camera'] for c in record['cameras'])==[4,5,8],'Trace cameras differ')
            if f<2:continue
            for c in record['cameras']:
                keys=tuple(c['local_ids']);boxes=c['xyxy'];g=ground.slots[f,c['camera']]
                new=spatial_slot(g,keys,boxes,width=1920,height=1080,min_iou=a.evaluation.min_iou)
                old=old_spatial(legacy[f,c['camera']],keys,boxes)
                require(new[0]==old[0] and np.array_equal(new[1],old[1]) and new[2:]==old[2:],'Frozen-trace spatial parity differs')
                compared+=1;predictions+=len(keys)
        require(handle.readline()=='','Extra trace frames')
    print(f'scene_001: {compared} camera/frame IoU masks and diagnostics EXACT; predictions={predictions}')
    vg=load_ground_truth(b.evaluation);val_counts={}
    for camera in b.runtime.camera_ids:
        missing=[f for f,c in vg.empty_slots if c==camera]
        require(missing==validation['frames_without_gt_rows'][str(camera)],'Validation empty-slot coverage differs')
        rows=sum(len(vg.slots[f,camera]) for f in range(2,3600))
        require(rows==validation['projection_diagnostics'][str(camera)]['all_raw_boxes']['rows'],'Validation GT count differs')
        val_counts[camera]={'slots':3598,'observations':rows,'empty_slots':len(missing)}
        print(f'scene_041 camera {camera}: {json.dumps(val_counts[camera])}')
    for name,scene in [('training',a),('validation',b)]:
        for role,ref in [('configuration',scene.configuration),('source_manifest',scene.source_manifest),
                         ('video_manifest',scene.video_manifest),('calibration',scene.runtime.calibration),
                         ('ground_truth',scene.evaluation.ground_truth)]:
            inputs[f'{name}_{role}']={'path':str(ref.path),'sha256':ref.sha256}
        for camera in scene.runtime.cameras:
            inputs[f'{name}_video_{camera.camera_id}']={'path':str(camera.video.path),'sha256':camera.video.sha256}
    for spec in inputs.values():require(sha256(spec['path'])==spec['sha256'],'Input changed during checks')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'artifacts/scene_input_checks'/run;out.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),ROOT/'src/mtmc/data/scene.py',ROOT/'src/mtmc/data/ground_truth.py',
          ROOT/'scripts/evaluate_mtmc_sequence.py',ROOT/'scripts/evaluate_mtmc_pipeline.py',ROOT/'scripts/run_mtmc.py',
          ROOT/'scripts/evaluate_reid_snapshot.py',ROOT/'scripts/evaluate_global_identity.py']
    report={'completed':True,'protocol':'scene_input_contract_checks_v1','run_id':run,'inputs':inputs,
            'training_parity':{'gt_slots':len(ground.slots),'spatial_slots':compared,'prediction_observations':predictions,
                              'all_gt_boxes_exact':True,'all_spatial_outputs_exact':True,'homographies_exact':True},
            'validation_coverage':val_counts,'synthetic_checks_passed':True,'runtime_modified':False,
            'versions':{name:version(name) for name in ('numpy','scipy','motmetrics','pandas')},
            'code_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in code},
            'limits':['Input/GT/spatial-adapter parity only; no new pipeline inference or identity-quality evaluation.',
                      'Empty GT slots are retained under the supplied annotation protocol; annotation completeness is not independently certified.',
                      'Existing old runners are not switched to this loader by this check.']}
    path=out/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Configs: {train_config}\n         {val_config}\nReport: {path}')
    print('Scene input contract and legacy parity: PASSED; runtime integration pending')


if __name__=='__main__':main()
