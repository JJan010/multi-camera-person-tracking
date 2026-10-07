"""Inspect a frozen local-label transition using exact video frames and GT overlays."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

import evaluate_mtmc_sequence as sequence
from mtmc.reid.osnet import ObservationKey, sha256

ROOT = Path(__file__).resolve().parents[1]
require = sequence.require
CYAN, GREEN, MAGENTA = '#00e5ff', '#78ff65', '#ff62db'


def select_transition(rows,camera,local_id,rounds):
    selected = [r for r in rows if (r['camera'],r['local_id']) == (camera,local_id)]
    require(bool(selected),'No diagnostic GT-label transition for this local track')
    event = min(selected,key=lambda r:(r['frame'],r['previous_evidence_frame'],r['previous_gt'],r['current_gt']))
    before,after = event['previous_evidence_frame'],event['frame']
    require(2 <= before < after < rounds and event['previous_gt'] != event['current_gt']
            and event['gap_frames'] == after-before,'Invalid diagnostic transition')
    frames = sorted({max(0,before-15),max(0,before-5),before,after,min(rounds-1,after+5),min(rounds-1,after+15)})
    return event,frames


def common_roi(boxes,width=1920,height=1080,padding=80):
    require(bool(boxes),'No boxes available for context crop')
    values = np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    require(np.isfinite(values).all() and np.all(values[:,2:] > values[:,:2]),'Invalid ROI source boxes')
    bounds = (max(0,int(np.floor(values[:,0].min()-padding))),
              max(0,int(np.floor(values[:,1].min()-padding))),
              min(width,int(np.ceil(values[:,2].max()+padding))),
              min(height,int(np.ceil(values[:,3].max()+padding))))
    require(bounds[2] > bounds[0] and bounds[3] > bounds[1],'Context lies fully outside image')
    return bounds


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default()


def label(draw,position,text,color,size=20):
    f = font(size)
    draw.rectangle(draw.textbbox(position,text,font=f),fill='black')
    draw.text(position,text,font=f,fill=color)


def box(draw,xyxy,color,dashed=False,width=3):
    x1,y1,x2,y2 = map(float,xyxy)
    if not dashed:
        draw.rectangle((x1,y1,x2,y2),outline=color,width=width)
        return
    for x in np.arange(x1,x2,16):
        for y in (y1,y2): draw.line((x,y,min(x+9,x2),y),fill=color,width=width)
    for y in np.arange(y1,y2,16):
        for x in (x1,x2): draw.line((x,y,x,min(y+9,y2)),fill=color,width=width)


def annotate(image,view,event,camera,local_id):
    image = image.copy();draw = ImageDraw.Draw(image)
    # Other local tracks provide context without introducing more GT colors.
    for item in view['tracks']:
        if item['local_id'] == local_id: continue
        box(draw,item['xyxy'],'#adadad',width=1)
        x,y,_,_ = item['xyxy']
        label(draw,(max(0,x),max(0,y-16)),f"L{item['local_id']}/G{item['global_id']}",'#d0d0d0',size=16)
    for gt,color in ((event['previous_gt'],GREEN),(event['current_gt'],MAGENTA)):
        coordinates = view['selected_gt_boxes'].get(gt)
        if coordinates is None: continue
        box(draw,coordinates,color,dashed=True,width=4)
        x,y,_,bottom = coordinates
        label(draw,(max(0,x+3),max(0,y+4 if gt == event['previous_gt'] else bottom-26)),f'GT {gt}',color)
    target = next((t for t in view['tracks'] if t['local_id']==local_id),None)
    if target:
        box(draw,target['xyxy'],CYAN,width=2)
        x,y,_,_ = target['xyxy']
        label(draw,(max(0,x),max(0,y-28)),f"TARGET L{local_id}/G{target['global_id']}",CYAN,size=24)
    return image


def panel(image,lines):
    out = Image.new('RGB',(800,600),'#151515')
    fitted = ImageOps.contain(image,(800,470))
    out.paste(fitted,((800-fitted.width)//2,130+(470-fitted.height)//2))
    draw = ImageDraw.Draw(out)
    for i,line in enumerate(lines):
        draw.text((10,8+28*i),line,font=font(21),fill='white')
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic-report',type=Path,required=True)
    parser.add_argument('--camera',type=int)
    parser.add_argument('--local-id',type=int)
    args = parser.parse_args()
    if (args.camera is None) != (args.local_id is None):
        parser.error('Supply both --camera and --local-id, or neither')
    inputs = {}
    def checked(name,path,digest=None):
        path = Path(path).resolve();actual = sha256(path)
        require(digest is None or actual == digest,f'Checksum mismatch: {path}')
        inputs[name] = {'path':str(path),'sha256':actual}
        return path
    diagnostic_path = checked('diagnostic_report',args.diagnostic_report)
    diagnostic = json.loads(diagnostic_path.read_text())
    require(diagnostic.get('completed') and diagnostic['protocol']['name']=='scene_001_sequence_failure_diagnostic_v1',
            'Expected sequence failure diagnostic')
    for name in ('pipeline_report','tracks','global_tracks','ground_truth'):
        item = diagnostic['inputs'][name];checked(name,item['path'],item['sha256'])
    source = json.loads(Path(inputs['pipeline_report']['path']).read_text())
    run,rounds,cfg = source['run_id'],source['summary']['rounds'],source['configuration']
    require(run == diagnostic['source_run_id'] and diagnostic['protocol']['frames']==[2,rounds-1], 'Different diagnostic run')
    if args.camera is None:
        require(bool(diagnostic['top_local_mixing']),'No local mixing case to inspect')
        target = diagnostic['top_local_mixing'][0]
        camera,local_id = target['camera'],target['local_id']
    else:
        camera,local_id = args.camera,args.local_id
    require(camera in sequence.CAMERAS and local_id >= 0,'Invalid selected camera/local ID')
    item = diagnostic['artifacts']['local_label_transitions.csv']
    transitions_path = checked('local_label_transitions',diagnostic_path.parent/item['path'],item['sha256'])
    with transitions_path.open() as stream:
        transitions = [{k:int(v) for k,v in r.items()} for r in csv.DictReader(stream)]
    event,frames = select_transition(transitions,camera,local_id,rounds)
    print('Selected earliest transition for target track:',json.dumps(event),flush=True)
    print('Context frames:',frames,flush=True)
    ground = sequence.load_ground_truth(inputs['ground_truth']['path'],rounds)
    views = {};next_row=0
    with Path(inputs['tracks']['path']).open() as tracks,Path(inputs['global_tracks']['path']).open() as globals_file:
        for frame in range(rounds):
            a,b=tracks.readline(),globals_file.readline();require(bool(a) and bool(b),'Truncated frozen trace')
            local,global_record=json.loads(a),json.loads(b)
            cameras,assigned,next_row=sequence.validate_round(local,global_record,frame=frame,run=run,config=cfg,next_row=next_row)
            if frame not in frames: continue
            keys,boxes=cameras[camera]
            gt,mask,unique,_,_=sequence.spatial_slot(ground.get((frame,camera),{}),keys,boxes)
            target_key=ObservationKey(camera,local_id,frame)
            candidates=[]
            if target_key in keys:
                column=keys.index(target_key)
                candidates=[g for i,g in enumerate(gt) if mask[i,column]]
            views[frame]={'frame':frame,'timestamp':str(Fraction(frame,30)),
                'tracks':[{'local_id':key.local_id,'global_id':assigned[key],'xyxy':coordinates.tolist()}
                          for key,coordinates in zip(keys,boxes)],
                'target_unique_gt_id':unique.get(target_key),'target_spatial_candidates':candidates,
                'selected_gt_boxes':{gt:ground.get((frame,camera),{}).get(gt) for gt in (event['previous_gt'],event['current_gt'])}}
        require(tracks.readline()==globals_file.readline()=='','Trailing frozen trace rows')
    require(next_row==source['summary']['total_embeddings'] and set(views)==set(frames),'Source coverage differs')
    for frame,gt,gid in ((event['previous_evidence_frame'],event['previous_gt'],event['previous_global_id']),
                         (event['frame'],event['current_gt'],event['current_global_id'])):
        require(views[frame]['target_unique_gt_id']==gt,'Selected transition GT evidence not reproduced')
        target=next((t for t in views[frame]['tracks'] if t['local_id']==local_id),None)
        require(target is not None and target['global_id']==gid,'Selected transition global ID differs')
    context=[]
    for view in views.values():
        context.extend(t['xyxy'] for t in view['tracks'] if t['local_id']==local_id)
        context.extend(b for b in view['selected_gt_boxes'].values() if b is not None)
    roi=common_roi(context)
    item=source['inputs']['video_manifest']
    manifest_path=checked('video_manifest',item['path'],item['sha256'])
    manifest=json.loads(manifest_path.read_text())
    require(manifest['dataset']=='nvidia/PhysicalAI-SmartSpaces'
            and manifest['revision']=='2cbe9563cbe9f47f846e5c871ee994572bbbc60e'
            and manifest['scene']=='MTMC_Tracking_2024/train/scene_001','Different video scene/revision')
    entries=[x for x in manifest['files'] if x['camera']==camera]
    require(len(entries)==1,'Missing/duplicate camera video')
    video=checked('video',ROOT/entries[0]['local_path'],entries[0]['sha256'])
    require(inputs['video']['sha256']==source['inputs'][f'video_{camera}']['sha256'],'Video differs from pipeline input')
    import av
    output=ROOT/'artifacts/identity_switch_preview'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True,exist_ok=False)
    contact=Image.new('RGB',(2400,1200),'#151515')
    covered=set()
    print('Decoding the selected camera sequentially with exact PTS checks...',flush=True)
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.codec_context.thread_count=1;stream.thread_type='SLICE'
        for index,frame in enumerate(container.decode(stream)):
            require(frame.pts is not None and frame.time_base is not None
                    and Fraction(frame.pts)*frame.time_base==Fraction(index,30),'Video timestamp mismatch')
            if index in views:
                rgb=Image.fromarray(frame.to_ndarray(format='rgb24'))
                require(rgb.size==(1920,1080),'Unexpected frame dimensions')
                view=views[index]
                target=next((t for t in view['tracks'] if t['local_id']==local_id),None)
                annotated=annotate(rgb,view,event,camera,local_id)
                name=f'camera_{camera:04d}_local_{local_id:04d}_frame_{index:06d}'
                rgb.crop(roi).save(output/(name+'_raw_context.png'))
                annotated.save(output/(name+'_annotated.png'))
                view.update(raw_context_image=name+'_raw_context.png',annotated_image=name+'_annotated.png')
                text=[f'Camera {camera} | frame {index} | t={index/30:.3f}s',
                      f"Target L{local_id}: G{target['global_id']}" if target else f'Target L{local_id}: absent',
                      f"Unique GT: {view['target_unique_gt_id']} | candidates: {view['target_spatial_candidates']}",
                      f"Cyan: target | green: GT {event['previous_gt']} | pink: GT {event['current_gt']}"]
                position=frames.index(index)
                contact.paste(panel(annotated.crop(roi),text),((position%3)*800,(position//3)*600))
                covered.add(index)
                print(f"Frame {index}: local={local_id}; global={target['global_id'] if target else None}; unique GT={view['target_unique_gt_id']}")
            if index==max(frames):break
    require(covered==set(frames),'Video ended before selected context frames')
    contact_name=f'camera_{camera:04d}_local_{local_id:04d}_contact.jpg'
    contact.save(output/contact_name,quality=95)
    for item in inputs.values():require(sha256(Path(item['path']))==item['sha256'],'Input changed during preview')
    report={'completed':True,'protocol':'scene_001_local_identity_transition_preview_v1','source_run_id':run,
        'camera':camera,'local_id':local_id,'selection':'Earliest diagnostic GT-label transition of chosen local track',
        'target_selection':'Explicit camera/local ID' if args.camera is not None else 'Highest evidence outside dominant GT in diagnostic ranking',
        'transition':event,'frames':frames,'shared_context_roi_xyxy':roi,'views':[views[f] for f in frames],
        'inputs':inputs,'code_sha256':{str(Path(m.__file__).relative_to(ROOT)):sha256(Path(m.__file__)) for m in
                                    (sequence,sequence.baseline,sequence.metric.history.snapshot)},
        'versions':{'numpy':np.__version__,'av':av.__version__},
        'artifacts':{p.name:{'path':p.name,'sha256':sha256(p)} for p in sorted(output.iterdir())},
        'limits':['Diagnostic IoU labels, not manual visual identity confirmation',
                  'One selected local track and earliest transition; not every failure or a complete causal attribution',
                  'Frames can straddle an evidence gap; previous/current evidence frame indices are explicit',
                  'Shared image crop only changes presentation; raw tracked/GT boxes and runtime IDs are unchanged',
                  'No model execution, threshold changes or retrospective identity repair']}
    report['code_sha256'][str(Path(__file__).relative_to(ROOT))]=sha256(Path(__file__))
    path=output/'report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Report: {path}')
    print(f'Contact sheet: {output/contact_name}')
    print('Identity transition preview: COMPLETED; visual review pending')


if __name__=='__main__':
    main()
