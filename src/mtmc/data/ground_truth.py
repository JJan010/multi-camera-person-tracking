"""Offline nine-column GT loading and camera-sized spatial admissibility."""
from dataclasses import dataclass
import numpy as np
from .scene import EvaluationSpec, require


@dataclass
class GroundTruth:
    slots: dict
    empty_slots: tuple


def load_ground_truth(spec: EvaluationSpec):
    require(isinstance(spec,EvaluationSpec),'Expected EvaluationSpec')
    require(spec.camera_ids and all(type(c) is int and c>=0 for c in spec.camera_ids)
            and tuple(sorted(set(spec.camera_ids)))==spec.camera_ids, 'Invalid GT cameras')
    require(type(spec.first_frame) is int and type(spec.last_frame) is int
            and 0<=spec.first_frame<=spec.last_frame and type(spec.gt_to_video_offset) is int,
            'Invalid GT interval/offset')
    require(spec.empty_slot_policy=='retain_slot_and_count_predictions','Unknown empty-slot policy')
    slots={(f,c):{} for f in range(spec.first_frame,spec.last_frame+1) for c in spec.camera_ids}
    with spec.ground_truth.verify().open(encoding='utf-8') as handle:
        for number,line in enumerate(handle,1):
            fields=line.split()
            if not fields:continue
            require(len(fields)==9,f'Expected nine GT columns, line {number}')
            camera,identity,annotation_frame=map(int,fields[:3])
            require(min(camera,identity,annotation_frame)>=0, f'Negative GT key, line {number}')
            frame=annotation_frame+spec.gt_to_video_offset
            if (frame,camera) not in slots:continue
            x,y,w,h,wx,wy=map(float,fields[3:])
            require(np.isfinite([x,y,w,h,wx,wy]).all() and w>0 and h>0,'Invalid GT box/coordinates')
            require(identity not in slots[frame,camera],'Duplicate GT identity in camera/frame')
            slots[frame,camera][identity]=[x,y,x+w,y+h]
    spec.ground_truth.verify()
    return GroundTruth(slots,tuple(key for key,values in slots.items() if not values))


def clip_boxes(values,width,height):
    require(type(width) is int and type(height) is int and width>0 and height>0,'Invalid image dimensions')
    boxes=np.asarray(values,dtype=np.float64)
    if boxes.shape==(0,):boxes=boxes.reshape(0,4)
    require(boxes.ndim==2 and boxes.shape[1]==4 and np.isfinite(boxes).all()
            and np.all(boxes[:,2:]>boxes[:,:2]),'Invalid raw xyxy boxes')
    return np.clip(boxes,[0,0,0,0],[width,height,width,height])


def pairwise_iou(a,b):
    intersection=np.maximum(0,np.minimum(a[:,None,2:],b[None,:,2:])-
                            np.maximum(a[:,None,:2],b[None,:,:2])).prod(axis=2)
    union=((a[:,2:]-a[:,:2]).prod(axis=1)[:,None]+
           (b[:,2:]-b[:,:2]).prod(axis=1)[None,:]-intersection)
    return np.divide(intersection,union,out=np.zeros_like(union),where=union>0)


def spatial_slot(ground,keys,predictions,*,width,height,min_iou=.5):
    require(type(min_iou) in (int,float) and np.isfinite(min_iou) and 0<min_iou<=1,'Invalid IoU gate')
    require(all(type(g) is int and g>=0 for g in ground),'Invalid GT identity')
    keys=tuple(keys)
    require(len(keys)==len(set(keys)),'Duplicate prediction key')
    gt_ids=sorted(ground)
    boxes=clip_boxes([ground[g] for g in gt_ids],width,height)
    visible=np.all(boxes[:,2:]>boxes[:,:2],axis=1)
    gt_ids=[g for g,keep in zip(gt_ids,visible) if keep]
    predicted=clip_boxes(predictions,width,height)
    require(len(keys)==len(predicted),'Prediction keys/boxes differ')
    mask=pairwise_iou(boxes[visible],predicted)>=min_iou
    unique={};gd,pd=mask.sum(axis=1),mask.sum(axis=0)
    for i,j in zip(*np.nonzero(mask)):
        if gd[i]==pd[j]==1:unique[keys[j]]=gt_ids[i]
    return gt_ids,mask,unique,int((~visible).sum()),int(np.any(predicted[:,2:]<=predicted[:,:2],axis=1).sum())
