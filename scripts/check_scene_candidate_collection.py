"""CPU tests of scene-aware crops, recording and read-back; no model quality claim."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS

import numpy as np

from collect_scene_candidates import collect
from mtmc.data.candidates import cache_record, prepare_candidate_crops, validate_cache_round
from mtmc.data.scene import CameraInput, FileRef, RuntimeScene
from mtmc.reid.osnet import ReIDBatch


def scene():
    missing=FileRef(Path('/nonexistent/scene-candidate-test'),'0'*64)
    return RuntimeScene('synthetic','synthetic','0'*40,30,4,
        (CameraInput(0,missing,8,6,np.eye(3)),CameraInput(17,missing,10,8,np.eye(3))),missing,'synthetic/Z0')


def frame_batch(frame):
    return NS(frame_index=frame,timestamp=Fraction(frame,30),frames=tuple(
        NS(camera_id=c,frame_index=frame,timestamp=Fraction(frame,30),
           rgb=(np.arange(w*h*3).reshape(h,w,3)%256).astype(np.uint8)) for c,w,h in ((0,8,6),(17,10,8))))


class Detector:
    def __init__(self):self.calls=0
    def detect(self,batch):
        self.calls+=1
        empty=batch.frame_index==1
        return tuple(NS(camera_id=c,frame_index=batch.frame_index,timestamp=batch.timestamp,
            xyxy=np.empty((0,4),np.float32) if empty else np.array(boxes,np.float32),
            confidence=np.empty(0,np.float32) if empty else np.array(scores,np.float32))
            for c,boxes,scores in ((17,[[0,0,4,6],[0,0,4,6]],[.8,.2]),
                                  (0,[[1.2,-1,5.1,4.3],[12,1,14,3]],[.9,.11])))


class Encoder:
    def __init__(self):self.batches=[]
    def encode(self,crops):
        self.batches.append(tuple(c.key for c in crops))
        vectors=np.zeros((len(crops),512),np.float32)
        for i,c in enumerate(crops):vectors[i,int(c.rgb.sum())%512]=1
        return ReIDBatch(tuple(c.key for c in crops),tuple(c.timestamp for c in crops),vectors)


def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError,StopIteration):return
    raise AssertionError('Expected input rejection')


def main():
    sc=scene();batch=frame_batch(0);detections=Detector().detect(batch)
    crops,records=prepare_candidate_crops(sc,batch,detections,row_offset=0)
    assert [(x['camera_id'],x['detection_index'],x['embedding_row']) for x in records]==[(0,0,0),(0,1,None),(17,0,1),(17,1,2)]
    assert [c.key.local_id for c in crops]==[0,0,1]
    assert np.array_equal(crops[0].rgb,batch.frames[0].rgb[0:5,1:6])
    assert np.shares_memory(crops[0].rgb,batch.frames[0].rgb) and not crops[0].rgb.flags.writeable
    assert np.array_equal(crops[1].rgb,crops[2].rgb)
    assert records[3]['confidence']==float(np.float32(.2))
    reordered=NS(frame_index=0,timestamp=Fraction(0),frames=tuple(reversed(batch.frames)))
    c2,r2=prepare_candidate_crops(sc,reordered,tuple(reversed(detections)),row_offset=0)
    assert r2==records and all(a.key==b.key and np.array_equal(a.rgb,b.rgb) for a,b in zip(crops,c2))
    print('Nonstandard cameras/sizes, raw RGB crop parity, weak/duplicate/outside candidates and camera order: OK')
    vectors=Encoder().encode(crops).embeddings
    record=json.loads(json.dumps(cache_record('test',sc,batch,records)))
    assert validate_cache_round(record,sc,vectors,run_id='test',frame=0,row_offset=0)==(3,1,{0:2,17:2})
    for field,value in [('embedding_row',1),('detection_index',1),('crop_xyxy_int',[0,0,1,1]),('inside_image_fraction',0.)]:
        bad=deepcopy(record);bad['detections'][0][field]=value
        reject(lambda:validate_cache_round(bad,sc,vectors,run_id='test',frame=0,row_offset=0))
    bad=deepcopy(record);bad['camera_ids']=[0]
    reject(lambda:validate_cache_round(bad,sc,vectors,run_id='test',frame=0,row_offset=0))
    reject(lambda:validate_cache_round(record,sc,vectors,run_id='other',frame=0,row_offset=0))
    broken=vectors.copy();broken[0]=0
    reject(lambda:validate_cache_round(record,sc,broken,run_id='test',frame=0,row_offset=0))
    wrong=deepcopy(detections);wrong[1].confidence=wrong[1].confidence.astype(np.float64)
    reject(lambda:prepare_candidate_crops(sc,batch,wrong,row_offset=0))
    wrong=deepcopy(batch);wrong.frames[0].timestamp=Fraction(1,30)
    reject(lambda:prepare_candidate_crops(sc,wrong,detections,row_offset=0))
    reject(lambda:prepare_candidate_crops(sc,batch,detections[:1],row_offset=0))
    print('Mixed scopes/times, incomplete cameras and altered crop/embedding mappings rejected: OK')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp);detector=Detector();encoder=Encoder()
        summary=collect(sc,iter(frame_batch(i) for i in range(4)),detector,encoder,run_id='test',output=out,
                        batch_size=2,warmup=1,synchronize=lambda:None)
        assert detector.calls==4 and [len(c) for c in encoder.batches]==[2,1]*3
        assert (summary['rounds'],summary['images'],summary['detections'],summary['encoded'],summary['fully_outside'])==(4,8,12,9,3)
        assert summary['measured_frame_range']==[1,3]
        saved=[json.loads(x) for x in (out/'detections.jsonl').read_text().splitlines()]
        assert len(saved)==4 and saved[1]['detections']==[] and saved[1]['camera_ids']==[0,17]
        assert np.load(out/'embeddings.npy').shape==(9,512)
        assert len((out/'timings.jsonl').read_text().splitlines())==4
        assert not (out/'embeddings.f32.partial').exists()
    print('Complete collector: chunked encoding, empty round, warmup retained, file read-back and bounded archive: OK')
    with tempfile.TemporaryDirectory() as tmp:
        out=Path(tmp)
        reject(lambda:collect(sc,iter([frame_batch(0)]),Detector(),Encoder(),run_id='test',output=out,
                             batch_size=2,warmup=1,synchronize=lambda:None))
        assert not (out/'embeddings.npy').exists() and (out/'embeddings.f32.partial').exists()
    print('Truncated replay cannot produce a finalized feature archive: OK')
    print('Scene candidate collection contract: PASSED; CUDA/model execution not tested here')


if __name__=='__main__':main()
