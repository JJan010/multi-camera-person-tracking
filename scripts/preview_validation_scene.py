"""Deterministic visual review of validation GT alignment, projections and GT gaps."""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import audit_scene_geometry as geometry

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def longest_gap(values):
    values = sorted(set(values))
    if not values:
        return None
    runs = []; start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous)); start = value
        previous = value
    runs.append((start, previous))
    return min(runs, key=lambda ab: (-(ab[1]-ab[0]+1), ab[0]))


def select_cases(rows, gaps, cameras, start, stop):
    worst, missing = {}, {}
    for camera in cameras:
        eligible = [r for r in rows if int(r['camera']) == camera and
                    r['box_fully_inside'] == 'True' and r['world_discrepancy'] != '']
        geometry.require(eligible, f'No eligible projection for camera {camera}')
        row = min(eligible, key=lambda r: (-float(r['world_discrepancy']), int(r['frame']), int(r['gt_id'])))
        worst[camera] = {'frame': int(row['frame']), 'gt_id': int(row['gt_id']),
                         'world_discrepancy': float(row['world_discrepancy'])}
        gap = longest_gap(gaps[str(camera)])
        if gap is not None:
            a, b = gap
            missing[camera] = {'start': a, 'end': b, 'length': b-a+1,
                               'context': [max(start, a-1), (a+b)//2, min(stop, b+1)]}
    return worst, missing


def draw_view(raw, records, h, target=None):
    image = raw.copy(); draw = ImageDraw.Draw(image)
    for person, values in sorted(records.items()):
        x, y, w, height, _, _ = values
        rect = (max(0,x), max(0,y), min(image.width-1,x+w), min(image.height-1,y+height))
        if rect[2] <= rect[0] or rect[3] <= rect[1]: continue
        color = 'yellow' if person == target else 'white'
        draw.rectangle(rect, outline=color, width=4 if person == target else 2)
        label = f'GT {person}'
        pos = (max(0,min(rect[0]+2,image.width-160)), max(0,rect[1]-26))
        draw.rectangle(draw.textbbox(pos,label,font=font(22)),fill='black')
        draw.text(pos,label,font=font(22),fill=color)
    detail = None
    if target is not None:
        geometry.require(target in records, 'Target GT missing')
        x,y,w,bh,wx,wy = records[target]
        foot = np.array([x+w/2,y+bh])
        pixel = geometry.project(h,(wx,wy)); world = geometry.project(np.linalg.inv(h),foot)
        geometry.require(pixel is not None and world is not None, 'Unavailable selected projection')
        draw.line([tuple(foot),tuple(pixel)],fill='yellow',width=4)
        u,v = foot;draw.ellipse((u-9,v-9,u+9,v+9),outline='cyan',width=4)
        u,v = pixel;draw.line((u-12,v,u+12,v),fill='magenta',width=4);draw.line((u,v-12,u,v+12),fill='magenta',width=4)
        bounds = (max(0,int(np.floor(min(x,u)-100))),max(0,int(np.floor(min(y,v)-100))),
                  min(image.width,int(np.ceil(max(x+w,u)+100))),min(image.height,int(np.ceil(max(y+bh,v)+100))))
        detail = {'gt_id':target,'xywh':[x,y,w,bh],'box_foot_uv':foot.tolist(),
                  'gt_world_xy':[wx,wy],'projected_gt_uv':pixel.tolist(),
                  'world_discrepancy':float(np.linalg.norm(world-[wx,wy])), 'zoom_bounds':list(bounds)}
    return image, detail


def tile(image, lines):
    result = Image.new('RGB',(960,630),'black')
    fitted = ImageOps.contain(image,(960,540))
    result.paste(fitted,((960-fitted.width)//2,90+(540-fitted.height)//2))
    draw = ImageDraw.Draw(result)
    for i,line in enumerate(lines):draw.text((10,5+27*i),line,font=font(21),fill='white')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-report',type=Path,required=True)
    args = parser.parse_args()
    audit_path = args.audit_report.resolve(); audit = json.loads(audit_path.read_text())
    geometry.require(audit['completed'] and audit['protocol']=='validation_scene_audit_v1'
                     and audit['timestamp_audit_passed'] and audit['scene']=='MTMC_Tracking_2024/val/scene_041', 'Unexpected audit')
    inputs = {'audit_report':{'path':str(audit_path),'sha256':geometry.sha256(audit_path)}}
    for name,spec in audit['inputs'].items():
        geometry.require(geometry.sha256(spec['path'])==spec['sha256'],f'Input changed: {name}')
        inputs[name]=spec
    geometry.require(geometry.sha256(Path(geometry.__file__))==audit['geometry_helper_sha256'],'Projection helper differs from audit')
    projection_path=audit_path.parent/'gt_projection.csv'
    expected=audit['artifacts']['gt_projection.csv']['sha256']
    geometry.require(geometry.sha256(projection_path)==expected,'Projection CSV changed')
    inputs['projection_csv']={'path':str(projection_path),'sha256':expected}
    cameras=tuple(audit['cameras']);start,stop=audit['frame_range']
    geometry.require(cameras==(361,362,364) and (start,stop)==(2,3599),'Unexpected selection')
    with projection_path.open() as handle: rows=list(csv.DictReader(handle))
    worst,gaps=select_cases(rows,audit['frames_without_gt_rows'],cameras,start,stop)
    print('Worst fully-inside projections:',json.dumps(worst),flush=True)
    print('Longest intervals without GT rows:',json.dumps(gaps),flush=True)
    fixed_frame=300
    wanted={(c,fixed_frame) for c in cameras}|{(c,worst[c]['frame']) for c in cameras}
    wanted.update((c,f) for c,g in gaps.items() for f in g['context'])
    gt={key:{} for key in wanted};coverage={c:set() for c in cameras}
    with Path(inputs['ground_truth.txt']['path']).open() as handle:
        for line in handle:
            fields=line.split()
            if not fields:continue
            camera,person,frame=map(int,fields[:3])
            if camera in coverage and start<=frame<=stop:coverage[camera].add(frame)
            if (camera,frame) in wanted:
                geometry.require(person not in gt[camera,frame],'Duplicate GT')
                gt[camera,frame][person]=tuple(map(float,fields[3:]))
    for c in cameras:
        geometry.require(sorted(set(range(start,stop+1))-coverage[c])==audit['frames_without_gt_rows'][str(c)],'GT gap list differs')
    calibration=json.loads(Path(inputs['calibration_2025_format.json']['path']).read_text())
    matrices={c:np.asarray(next(s for s in calibration['sensors'] if s['id'].lower()==f'camera_{c:04d}')['homography'],dtype=np.float64) for c in cameras}
    import av
    raw={}
    print('Sequential CPU decoding to selected frames; exact PTS checks; no models...',flush=True)
    for c in cameras:
        frames={f for camera,f in wanted if camera==c}
        with av.open(inputs[f'video_{c}']['path']) as container:
            for i,frame in enumerate(container.decode(video=0)):
                geometry.require(frame.pts is not None and frame.time_base is not None and
                                 frame.pts*Fraction(frame.time_base)==Fraction(i,30),'PTS mismatch')
                if i in frames:
                    image=Image.fromarray(frame.to_ndarray(format='rgb24'))
                    geometry.require(image.size==(1920,1080),'Unexpected frame dimensions')
                    raw[c,i]=image
                if i==max(frames):break
    geometry.require(set(raw)==wanted,'Missing requested video frame')
    run=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'artifacts/validation_scene_preview'/run;out.mkdir(parents=True,exist_ok=False)
    sheets={kind:Image.new('RGB',(2880,630),'black') for kind in
            ('alignment_raw','alignment_annotated','worst_inside_raw','worst_inside_annotated')}
    if gaps:
        for kind in ('gt_gaps_raw','gt_gaps_annotated'):
            sheets[kind]=Image.new('RGB',(2880,630*len(gaps)),'black')
    cases=[]
    def render(kind,c,f,column,row=0,target=None):
        annotated,detail=draw_view(raw[c,f],gt[c,f],matrices[c],target)
        bounds=detail['zoom_bounds'] if detail else None
        names={}
        for variant,image in [('raw',raw[c,f]),('annotated',annotated)]:
            name=f'{kind}_camera_{c:04d}_frame_{f:06d}_{variant}.png'
            image.save(out/name);names[variant]=name
            view=image.crop(bounds) if bounds else image
            lines=[f'Camera {c} | frame {f} | t={f/30:.3f}s | GT rows={len(gt[c,f])}',
                   f'GT {target}; world discrepancy={detail["world_discrepancy"]:.3f}' if detail else 'All GT boxes; no detection/tracking predictions',
                   'Cyan circle: box bottom | magenta cross: projected GT' if detail else ('No GT rows: inspect visible scene' if not gt[c,f] else 'White boxes: GT annotations')]
            sheets[f'{kind}_{variant}'].paste(tile(view,lines),(column*960,row*630))
        cases.append({'kind':kind,'camera':c,'frame':f,'gt_rows':len(gt[c,f]),'detail':detail,'images':names})
        print(f'{kind}: camera={c}, frame={f}, GT rows={len(gt[c,f])}')
        return detail
    for column,c in enumerate(cameras):
        render('alignment',c,fixed_frame,column)
        detail=render('worst_inside',c,worst[c]['frame'],column,target=worst[c]['gt_id'])
        geometry.require(abs(detail['world_discrepancy']-worst[c]['world_discrepancy'])<1e-9,'Projection preview/CSV mismatch')
    for row,(c,gap) in enumerate(sorted(gaps.items())):
        for column,f in enumerate(gap['context']):render('gt_gaps',c,f,column,row)
    for kind,image in sheets.items():image.save(out/f'{kind}.jpg',quality=95)
    for spec in inputs.values():geometry.require(geometry.sha256(spec['path'])==spec['sha256'],'Input changed during preview')
    report={'completed':True,'protocol':'validation_scene_visual_preview_v1','run_id':run,'inputs':inputs,
            'scene':audit['scene'],'cameras':list(cameras),'fixed_alignment_frame':fixed_frame,
            'worst_inside_cases':worst,'longest_gt_gaps':gaps,'cases':cases,
            'selection':'Fixed frame 300; highest finite fully-inside discrepancy per camera, earliest tie; longest GT gap, earliest tie, with neighbor frames.',
            'world_units':audit['world_units'],'visual_review_completed':False,
            'code_sha256':geometry.sha256(Path(__file__)),'geometry_helper_sha256':geometry.sha256(Path(geometry.__file__)),
            'av_version':av.__version__,'numpy_version':np.__version__,
            'limits':['GT frame equals zero-based video index is the candidate convention shown, not proved automatically.',
                      'One inspected gap per camera does not certify every unannotated frame.',
                      'GT-only review: no model metrics, threshold tuning or tracking changes.'],
            'artifacts':{p.name:{'sha256':geometry.sha256(p)} for p in sorted(out.iterdir()) if p.is_file()}}
    rp=out/'report.json';rp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    bundle=out/'visual_review.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(rp,rp.name)
        for p in sorted(out.glob('*.jpg')):z.write(p,p.name)
    print(f'Report: {rp}\nUpload bundle: {bundle}')
    print('Validation visual preview: COMPLETED; human visual review pending')


if __name__=='__main__':main()
