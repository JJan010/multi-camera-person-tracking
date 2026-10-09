"""Visualize exact archived samples and current queries from a verified gallery audit."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps
from mtmc.data.scene import load_scene, require, sha256
from mtmc.data.ground_truth import load_ground_truth
from mtmc.reid.crops import crop_geometry
from diagnose_recovery_context import camera_evidence

ROOT=Path(__file__).resolve().parents[1]


def font():
    try:return ImageFont.truetype('DejaVuSans.ttf',16)
    except OSError:return ImageFont.load_default()


def render_sample(raw,sample,ground,kind,event_frame,out,stem):
    box=sample['spatial_evidence']['raw_xyxy'];width,height=raw.size
    bounds,_=crop_geometry(box,width,height)
    require(bounds is not None,'Selected gallery/query crop is fully outside')
    crop=raw.crop(bounds);crop.save(out/(stem+'_crop.png'))
    x1,y1,x2,y2=box;pad=max(60,math.ceil(max(x2-x1,y2-y1)*.7))
    roi=(max(0,math.floor(x1)-pad),max(0,math.floor(y1)-pad),
         min(width,math.ceil(x2)+pad),min(height,math.ceil(y2)+pad))
    plain=raw.crop(roi);plain.save(out/(stem+'_context_raw.png'))
    annotated=plain.copy();draw=ImageDraw.Draw(annotated);fnt=font()
    for gt,gt_box in ground.items():
        a,b,c,d=gt_box
        if c<=roi[0] or a>=roi[2] or d<=roi[1] or b>=roi[3]:continue
        coords=(a-roi[0],b-roi[1],c-roi[0],d-roi[1])
        draw.rectangle(coords,outline='#00d97e',width=2)
        draw.text((max(0,coords[0]),max(0,coords[1]-19)),f'GT {gt}',font=fnt,fill='#00ff99',stroke_width=1,stroke_fill='black')
    draw.rectangle((x1-roi[0],y1-roi[1],x2-roi[0],y2-roi[1]),outline='#ff4545',width=3)
    annotated.save(out/(stem+'_context_annotated.png'))
    tile=Image.new('RGB',(640,430),'#18212a' if kind=='gallery' else '#163a4d')
    draw=ImageDraw.Draw(tile);source=sample['source_key'];identity=sample['identity_key']
    label=sample['spatial_evidence']['unique_gt'];reason=sample['spatial_evidence']['reason']
    cos=sample['cosine_to_query'] if kind=='gallery' else sample['cosine_to_reference']
    lines=[f"{kind.upper()} | event {event_frame} | camera {source['camera_id']} | frame {source['frame_index']}",
           f"L{source['local_id']} / segment {identity['local_id']} / observed G{sample['observed_global_id']}",
           f"GT={label} | score={sample['box_confidence']:.3f} | cosine={cos:.4f}",reason,
           'Raw model crop                       Context: red prediction / green GT']
    draw.multiline_text((10,8),'\n'.join(lines),font=fnt,fill='white',spacing=3)
    for image,left,budget in ((crop,8,(170,300)),(annotated,188,(444,300))):
        resized=ImageOps.contain(image,budget)
        tile.paste(resized,(left+(budget[0]-resized.width)//2,120+(budget[1]-resized.height)//2))
    return tile,dict(crop_bounds=list(bounds),context_bounds=list(roi),stem=stem)


def write_sheets(tiles,out,prefix):
    names=[]
    for start in range(0,len(tiles),9):
        page=tiles[start:start+9];columns=min(3,len(page));rows=(len(page)+columns-1)//columns
        canvas=Image.new('RGB',(columns*640,rows*430),'#101820')
        for i,tile in enumerate(page):canvas.paste(tile,((i%columns)*640,(i//columns)*430))
        name=f'{prefix}_page_{start//9+1:02d}.jpg';canvas.save(out/name,quality=94);names.append(name)
    return names


def self_check():
    raw=Image.new('RGB',(180,140),'#557799')
    sample=dict(source_key=dict(camera_id=0,local_id=0,frame_index=0),
        identity_key=dict(camera_id=0,local_id=2147483648,frame_index=0),
        spatial_evidence=dict(raw_xyxy=[-2.,10.,40.,100.],unique_gt=0,reason='mutually_unique'),
        cosine_to_query=.97,box_confidence=.8,observed_global_id=0)
    with tempfile.TemporaryDirectory() as d:
        out=Path(d);tile,meta=render_sample(raw,sample,{0:[0,10,40,100]},'gallery',20,out,'sample')
        require(Image.open(out/'sample_crop.png').size==(40,90),'Clipped crop differs')
        require(tile.size==(640,430) and meta['crop_bounds']==[0,10,40,100],'Tile/provenance differs')
        names=write_sheets([tile]*10,out,'event');require(len(names)==2,'Page bound differs')
    print('Exact crop bounds, ID zero, segment labels and contact-sheet pagination: PASSED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gallery-report',type=Path)
    parser.add_argument('--event-frame',type=int,action='append',default=[])
    parser.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:
        self_check()
        if args.gallery_report is None:return
    require(args.gallery_report is not None and args.event_frame,'Gallery report and event frame required')
    require(len(set(args.event_frame))==len(args.event_frame),'Duplicate requested event frame')
    pins={}
    def pin(path,expected=None):
        path=Path(path).resolve();value=sha256(path)
        require(expected is None or value==expected,'Checksum differs: '+str(path))
        pins[str(path)]=value;return path
    ap=pin(args.gallery_report);audit=json.loads(ap.read_text())
    require(audit['completed'] is True and audit['protocol']=='confirmed_return_gallery_audit_v1'
            and audit['checks'] and all(audit['checks'].values()),'Unverified gallery audit')
    candidates=[Path(p) for p in audit['inputs'] if Path(p).name=='report.json'
                and Path(p).parent.name==audit['source_run_id']]
    require(len(candidates)==1,'Recovery report provenance ambiguous')
    rp=pin(candidates[0],audit['inputs'][str(candidates[0])]);recovery=json.loads(rp.read_text())
    require(recovery['run_id']==audit['source_run_id'] and recovery['protocol']=='confirmed_clip_return_paired_v1','Mixed recovery run')
    ref=recovery['inputs']['source:history:scene_config'];config=pin(ref['path'],ref['sha256'])
    loaded=load_scene(config,project_root=ROOT);scene,spec=loaded.runtime,loaded.evaluation
    require(scene.scene==audit['scene']==recovery['scene'],'Mixed scenes')
    pin(spec.ground_truth.path,spec.ground_truth.sha256);ground=load_ground_truth(spec)
    ref=recovery['artifacts']['global_tracks.jsonl.gz'];trace=pin(rp.parent/ref['path'],ref['sha256'])
    pin(Path(__file__));pin(ROOT/'scripts/diagnose_recovery_context.py')
    events=[e for e in audit['events'] if e['frame_index'] in args.event_frame]
    require(len(events)==len(args.event_frame),'Requested event missing or ambiguous')
    selections=defaultdict(list);panels={};details=[]
    for event in events:
        panels[event['frame_index']]={}
        index=0
        # Queries first, then all archived samples; no GT-driven sample pruning.
        for kind in ('query','gallery'):
            for sample in event[kind]:
                source=sample['source_key'];c,f=source['camera_id'],source['frame_index']
                selections[c,f].append((event['frame_index'],index,kind,sample));index+=1
    # Reproduce selected boxes/labels using all predictions, preserving ambiguity.
    checked=set()
    with gzip.open(trace,'rt') as stream:
        for line in stream:
            row=json.loads(line);f=row['frame_index']
            require(row['run_id']==recovery['run_id'],'Mixed trace scope')
            for camera in row['cameras']:
                c=camera['camera']
                if (c,f) not in selections:continue
                setting=next(v for v in scene.cameras if v.camera_id==c)
                if spec.first_frame<=f<=spec.last_frame:
                    evidence=camera_evidence(camera,ground.slots[f,c],setting.width,setting.height,spec.min_iou)
                else:
                    evidence={i:dict(unique_gt=None,reason='outside_evaluation_range',raw_xyxy=box)
                              for i,box in zip(camera['local_ids'],camera['xyxy'])}
                bindings={tuple(b['source_key'][k] for k in ('camera_id','local_id','frame_index')):b['identity_key']
                          for b in row['segment_bindings']}
                for _,_,_,sample in selections[c,f]:
                    local=sample['source_key']['local_id'];ev=evidence[local];saved=sample['spatial_evidence']
                    require(all(ev[k]==saved[k] for k in ('unique_gt','reason','raw_xyxy')),'Selected spatial evidence differs')
                    require(bindings[c,local,f]==sample['identity_key'],'Selected segment mapping differs')
                checked.add((c,f))
    require(checked==set(selections),'Incomplete selected-frame provenance')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'artifacts/confirmed_return_gallery_preview'/run;out.mkdir(parents=True,exist_ok=False)
    import av
    decoded=set()
    for camera in scene.cameras:
        frames={f for c,f in selections if c==camera.camera_id}
        if not frames:continue
        video=pin(camera.video.path,camera.video.sha256)
        print(f'Decoding camera {camera.camera_id} through frame {max(frames)}; exact PTS; no inference...',flush=True)
        with av.open(str(video)) as container:
            stream=container.streams.video[0];stream.thread_count=1
            for f,frame in enumerate(container.decode(stream)):
                require(frame.pts is not None and Fraction(frame.pts)*Fraction(frame.time_base)==Fraction(f,scene.fps),'PTS mismatch')
                if f in frames:
                    raw=Image.fromarray(frame.to_ndarray(format='rgb24'))
                    require(raw.size==(camera.width,camera.height),'Frame dimensions differ')
                    for event_frame,index,kind,sample in selections[camera.camera_id,f]:
                        stem=f'event{event_frame:06d}_{index:02d}_{kind}_c{camera.camera_id}_f{f:06d}'
                        tile,meta=render_sample(raw,sample,ground.slots.get((f,camera.camera_id),{}),kind,event_frame,out,stem)
                        panels[event_frame][index]=tile
                        details.append(dict(event_frame=event_frame,index=index,kind=kind,sample=sample,**meta))
                    decoded.add((camera.camera_id,f))
                if f>=max(frames):break
    require(decoded==set(selections),'Selected frames missing in video')
    sheets=[]
    for frame,tiles in sorted(panels.items()):sheets.extend(write_sheets([tiles[i] for i in sorted(tiles)],out,f'event_{frame:06d}'))
    for path,value in pins.items():require(sha256(Path(path))==value,'Input changed during preview')
    result=dict(completed=True,protocol='confirmed_return_gallery_preview_v1',source_run_id=audit['source_run_id'],
        scene=scene.scene,event_frames=sorted(args.event_frame),inputs=pins,samples=details,contact_sheets=sheets,
        checks=dict(selected_boxes_labels_segments_reproduced=True,selected_video_pts_exact=True,inputs_unchanged=True),
        limits=['Offline visual review only; no model or tracker execution and no threshold changes.',
                'GT None means no unique spatial match, not a proven false detection.',
                'Raw crop PNGs preserve RGB pixels and integer bounds; contact sheets are resized.',
                'Red is the target prediction; green boxes and numbers are GT annotations.'])
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with zipfile.ZipFile(out/'visual_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.iterdir()):
            if path.name!='visual_review.zip':archive.write(path,path.name)
    print('Report:',out/'report.json');print('Visual review:',out/'visual_review.zip')
    print('Confirmed return gallery preview: COMPLETED; visual review pending; predictions unchanged')


if __name__=='__main__':main()
