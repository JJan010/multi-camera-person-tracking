"""Visual review of actual archived samples and the birth of enabled GID 127."""
import argparse,gzip,json,zipfile
from datetime import datetime,timezone
from fractions import Fraction
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
from mtmc.data.scene import load_scene,require,sha256
from mtmc.data.ground_truth import load_ground_truth,spatial_slot
from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import ObservationKey

ROOT=Path(__file__).resolve().parents[1]


def tile(image,label):
    result=Image.new('RGB',(320,400),'#202020');draw=ImageDraw.Draw(result)
    draw.multiline_text((8,8),label,fill='white',spacing=4)
    if image is not None:
        preview=ImageOps.contain(image,(304,320));result.paste(preview,((320-preview.width)//2,72+(320-preview.height)//2))
    return result


def sheet(tiles,path):
    columns=4;result=Image.new('RGB',(320*columns,400*((len(tiles)+columns-1)//columns)),'#303030')
    for i,item in enumerate(tiles):result.paste(item,((i%columns)*320,(i//columns)*400))
    result.save(path,quality=92)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--probe-report',type=Path,required=True);args=parser.parse_args();inputs={}
    def checked(name,path,digest=None):
        path=Path(path).resolve();actual=sha256(path);require(digest is None or actual==digest,'Checksum differs: '+str(path));inputs[name]={'path':str(path),'sha256':actual};return path
    probe_path=checked('probe_report',args.probe_report);probe=json.loads(probe_path.read_text())
    require(probe.get('completed') is True and probe['protocol']=='frozen_retired_descriptor_probe_v1','Unverified probe')
    item=probe['inputs']['validation_report'];vp=checked('validation_report',item['path'],item['sha256']);validation=json.loads(vp.read_text())
    require(validation['run_id']==probe['source_run_id'],'Mixed validation run')
    item=validation['inputs']['validation:scene_config'];loaded=load_scene(checked('scene_config',item['path'],item['sha256']),project_root=ROOT)
    scene,spec=loaded.runtime,loaded.evaluation;ground=load_ground_truth(spec);checked('ground_truth',spec.ground_truth.path,spec.ground_truth.sha256)
    item=validation['artifacts']['global_tracks.jsonl.gz'];trace=checked('trace',vp.parent/item['path'],item['sha256'])
    ref=next(q['target_retirement']['reference'] for q in probe['queries'] if q['target_old_global_id']==44 and q['target_retirement'] and q['target_retirement']['reference'])
    birth=next(q for q in probe['queries'] if q['new_global_id']==127)
    require(len(ref['members'])==1 and len(birth['source_members'])==1,'Expected single-camera inspected cases')
    member=ref['members'][0];source=member['source_key'];start=birth['source_members'][0]
    require(source['camera_id']==start['camera_id']==361 and source['local_id']==22 and start['local_id']==33 and birth['frame_index']==2747,'Unexpected selected cases')
    reference_frames=member['history_frames'][-1:] if member['used_latest_fallback'] else member['history_frames']
    selections={f:{'local':22,'category':'actual_reference'} for f in reference_frames}
    for f in (2747,2748,2760,2774,2775,2778,2779,2780,2781):selections[f]={'local':33,'category':'birth_context'}
    camera=next(c for c in scene.cameras if c.camera_id==361);views={};strong=[]
    with gzip.open(trace,'rt') as stream:
        for line in stream:
            row=json.loads(line);f=row['frame_index']
            require(row['run_id']==validation['run_id'],'Mixed trace scope')
            if f>max(selections):break
            data=row['variants']['enabled'];cam=next(c for c in data['cameras'] if c['camera']==361)
            if f<min(reference_frames) and 22 in cam['local_ids']:
                i=cam['local_ids'].index(22)
                if cam['confidence'][i]>=.5:strong.append((f,data,cam))
            if f in selections:views[f]=(data,cam)
    for f,data,cam in strong[-3:]:selections[f]={'local':22,'category':'earlier_strong_context'};views[f]=(data,cam)
    require(set(views)==set(selections),'Missing requested frame')
    details={}
    for f,(data,cam) in views.items():
        keys=tuple(ObservationKey(361,i,f) for i in cam['local_ids'])
        _,_,unique,_,_=spatial_slot(ground.slots[f,361],keys,cam['xyxy'],width=camera.width,height=camera.height,min_iou=spec.min_iou)
        target=ObservationKey(361,selections[f]['local'],f)
        assignments={ObservationKey(**a['key']):a['global_id'] for a in data['identity']['assignments']}
        i=keys.index(target) if target in keys else None
        details[f]={**selections[f],'frame':f,'global_id':assignments.get(target),'unique_gt':unique.get(target),
            'box':cam['xyxy'][i] if i is not None else None,'confidence':cam['confidence'][i] if i is not None else None}
        if selections[f]['category']=='actual_reference':require(assignments.get(target)==44,'Reference source differs')
    video=checked('video',camera.video.path,camera.video.sha256)
    checked('script',Path(__file__))
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');output=ROOT/'artifacts/retired_reference_preview'/run;output.mkdir(parents=True,exist_ok=False)
    import av
    panels={category:[] for category in ('actual_reference','birth_context','earlier_strong_context')};contexts=[];decoded=set()
    print('Decoding camera 361 sequentially; exact PTS checks; no model or tracking...',flush=True)
    with av.open(str(video)) as container:
        stream=container.streams.video[0];stream.thread_count=1
        for f,frame in enumerate(container.decode(stream)):
            require(frame.pts is not None and Fraction(frame.pts)*Fraction(frame.time_base)==Fraction(f,scene.fps),'Video frame/PTS mismatch')
            if f in details:
                d=details[f];raw=Image.fromarray(frame.to_ndarray(format='rgb24'));require(raw.size==(camera.width,camera.height),'Video dimensions differ')
                annotated=raw.copy();draw=ImageDraw.Draw(annotated)
                for gt,box in ground.slots[f,361].items():
                    draw.rectangle(tuple(box),outline='#00cc66',width=2);draw.text((box[0],max(0,box[1]-12)),f'GT {gt}',fill='#00ff77')
                crop=None
                if d['box'] is not None:
                    bounds,_=crop_geometry(d['box'],camera.width,camera.height)
                    if bounds is not None:
                        crop=raw.crop(bounds);crop.save(output/f'crop_f{f:06d}_L{d["local"]}.png')
                    box=d['box'];draw.rectangle(tuple(box),outline='#ff3030',width=4)
                    draw.text((box[0],max(0,box[1]-26)),f'L{d["local"]} G{d["global_id"]}',fill='#ff3030')
                    x1,y1,x2,y2=box;pad=max(80,int(max(x2-x1,y2-y1)))
                    roi=(max(0,int(x1)-pad),max(0,int(y1)-pad),min(camera.width,int(x2)+pad),min(camera.height,int(y2)+pad))
                    context=annotated.crop(roi)
                else:context=annotated
                score='absent' if d['confidence'] is None else f'{d["confidence"]:.3f}'
                label=f'f={f} L={d["local"]} G={d["global_id"]}\nscore={score} unique_GT={d["unique_gt"]}\n{d["category"]}'
                panels[d['category']].append(tile(crop,label));contexts.append(tile(context,label))
                raw.save(output/f'raw_f{f:06d}.jpg',quality=94);annotated.save(output/f'annotated_f{f:06d}.jpg',quality=94)
                decoded.add(f)
            if f>=max(details):break
    require(decoded==set(details),'Selected video coverage differs')
    for category,items in panels.items():
        if items:sheet(items,output/(category+'.jpg'))
    sheet(contexts,output/'context_contact.jpg')
    for item in inputs.values():require(sha256(item['path'])==item['sha256'],'Input changed during preview')
    report={'completed':True,'source_run_id':validation['run_id'],'inputs':inputs,'frames':[details[f] for f in sorted(details)],
        'limits':['GT null means no mutually unique match, not confirmed false detection.',
        'Earlier high-confidence samples are visual context only; no replacement descriptor or quality rule is evaluated.',
        'Crop PNGs use original RGB integer crop bounds; contact sheets are resized for viewing.',
        'Red boxes are target predictions, green boxes are GT; absent tracks are not interpolated.']}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with zipfile.ZipFile(output/'visual_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.name!='visual_review.zip':archive.write(path,path.name)
    print('Frames:',sorted(details));print('Visual review:',output/'visual_review.zip')
    print('Retired reference preview: COMPLETED; visual review pending; predictions unchanged')


if __name__=='__main__':main()
