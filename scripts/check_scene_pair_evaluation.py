"""Known-answer scene-aware local/global evaluation and cached replay integration."""
from copy import deepcopy
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np

import evaluate_scene_pair as experiment
from check_paired_runtime_contract import policy
from mtmc.data.ground_truth import GroundTruth
from mtmc.data.scene import CameraInput, EvaluationSpec, FileRef, RuntimeScene


def scene(rounds=4):
    missing=FileRef(Path('/nonexistent/scene-pair-fixture'),'0'*64)
    return RuntimeScene('synthetic','synthetic','0'*40,30,rounds,
        (CameraInput(11,missing,50,50,np.eye(3)),CameraInput(27,missing,80,60,np.eye(3))),missing,'synthetic/Z0')


def specification(sc):
    return EvaluationSpec(sc.calibration,sc.camera_ids,0,sc.rounds-1,0,.5,'retain_slot_and_count_predictions')


def record(frame, *, empty=False):
    cameras=[{'camera':c,'local_ids':[] if empty else [0], 'xyxy':[] if empty else [[0.,0.,10.,10.]],
        'confidence':[] if empty else [float(np.float32(.9))], 'detection_indices':[] if empty else [0],
        'embedding_rows':[] if empty else [frame*2+i]} for i,c in enumerate((11,27))]
    return {'run_id':'test','source_run_id':'cache','scene':'synthetic','frame_index':frame,
        'timestamp':str(Fraction(frame,30)),'variants':{n:{'cameras':deepcopy(cameras),
        'identity':{'run_id':'test/'+n,'frame_index':frame,'timestamp':str(Fraction(frame,30)),
            'assignments':[] if empty else [{'key':{'camera_id':c,'local_id':0,'frame_index':frame},'global_id':7} for c in (11,27)],
            'merge_events':[]}} for n in experiment.VARIANTS}}


def write(path, records):
    with gzip.open(path,'wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')


def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,TypeError):return
    raise AssertionError('Expected rejection')


def main():
    sc=scene();spec=specification(sc)
    truth={(f,c):{0:[0.,0.,10.,10.]} for f in range(4) for c in sc.camera_ids}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'trace.gz';records=[record(f) for f in range(4)];write(path,records)
        result,_=experiment.evaluate(path,sc,spec,GroundTruth(truth,()),run_id='test',cache_run='cache')
        for r in result.values():
            assert r['global']['idf1']==r['local']['OVERALL']['idf1']==1.
            assert r['no_cross_camera_control']['idf1']==.5
            assert r['global']['gt_observations']==r['global']['predicted_observations']==8
        print('Shared person across nonstandard cameras: local/global 100%, camera-scoped control 50%: OK')

        # One unannotated slot, one unmatched outside prediction and one ignored
        # zero-area-after-clipping GT. Frame 0 exercises inclusive IoU == 0.5.
        truth=deepcopy(truth);truth[1,11]={};truth[3,27][1]=[-20.,0.,-10.,10.]
        records[0]['variants']['staged']['cameras'][0]['xyxy']=[[0.,0.,5.,10.]]
        records[0]['variants']['competitive_iou']['cameras'][0]['xyxy']=[[0.,0.,5.,10.]]
        for n in experiment.VARIANTS:records[2]['variants'][n]['cameras'][0]['xyxy']=[[55.,0.,65.,10.]]
        write(path,records)
        result,_=experiment.evaluate(path,sc,spec,GroundTruth(truth,((1,11),)),run_id='test',cache_run='cache')
        for r in result.values():
            assert (r['global']['idtp'],r['global']['idfp'],r['global']['idfn'])==(6,2,1)
            assert r['global']['idf1']==.8
            assert r['empty_gt_slots']=={11:1,27:0} and r['predictions_in_empty_gt_slots']=={11:1,27:0}
            assert r['fully_outside_predictions']=={11:1,27:0} and r['excluded_zero_area_gt']=={11:0,27:1}
            assert r['local']['OVERALL']['num_false_positives']==2 and r['local']['OVERALL']['num_misses']==1
        print('GT ID zero, inclusive IoU, empty-GT false observation, outside prediction and GT clipping: OK')

        empty={(f,c):{} for f in range(4) for c in sc.camera_ids}
        write(path,[record(f,empty=True) for f in range(4)])
        result,_=experiment.evaluate(path,sc,spec,GroundTruth(empty,tuple(empty)),run_id='test',cache_run='cache')
        for r in result.values():
            assert r['global']['idf1'] is None and r['global']['camera_time_slots']==8
            assert r['global']['idtp']==r['global']['idfp']==r['global']['idfn']==0
        print('All-empty sequence retains all camera/time slots and undefined IDF1: OK')
        bad=record(0);bad['variants']['staged']['identity']['assignments'].pop()
        reject(lambda:experiment.read_variant(bad,sc,frame=0,run_id='test',cache_run='cache',variant='staged'))
        bad=record(0);bad['variants']['staged']['identity']['run_id']='another/staged'
        reject(lambda:experiment.read_variant(bad,sc,frame=0,run_id='test',cache_run='cache',variant='staged'))
        print('Incomplete global assignment and mixed run scopes rejected: OK')

    # Actual tracker/history/global runtime from candidate cache; no GT may load
    # until all outputs have been written. A middle empty round is not skipped.
    sc=scene(rounds=3);rows=[];offset=0
    for f in range(3):
        ds=[]
        if f!=1:
            for c in sc.camera_ids:
                ds.append({'camera_id':c,'detection_index':0,'frame_index':f,'xyxy':[0.,0.,10.,10.],
                    'confidence':float(np.float32(.9)),'crop_xyxy_int':[0,0,10,10],
                    'inside_image_fraction':1.,'embedding_row':offset});offset+=1
        rows.append({'cache_run_id':'cache','scene':sc.scene,'frame_index':f,'timestamp':str(Fraction(f,30)),
                     'camera_ids':list(sc.camera_ids),'detections':ds})
    vectors=np.zeros((offset,512),np.float32);vectors[:,0]=1
    with tempfile.TemporaryDirectory() as tmp:
        src=Path(tmp)/'candidates.jsonl';out=Path(tmp)/'paired.gz'
        src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with patch.object(experiment,'load_ground_truth',side_effect=AssertionError('GT accessed during replay')):
            summary=experiment.replay(sc,policy(),cache_run='cache',run_id='test',candidate_path=src,vectors=vectors,output_path=out)
        assert summary['candidates']=={'detections':4,'encoded':4,'fully_outside':0,'camera_11':2,'camera_27':2}
        truth={(f,c):{0:[0.,0.,10.,10.]} for f in range(3) for c in sc.camera_ids}
        result,_=experiment.evaluate(out,sc,specification(sc),GroundTruth(truth,()),run_id='test',cache_run='cache')
        for n,r in result.items():
            assert (r['global']['idtp'],r['global']['idfp'],r['global']['idfn'])==(4,0,2)
            assert r['global']['idf1']==.8 and r['no_cross_camera_control']['idf1']==.4
            assert summary['lifecycle'][n]['observations']==r['runtime_observations']==4
        bad=deepcopy(rows);bad[2]['detections'][0]['embedding_row']=0
        src.write_text(''.join(json.dumps(r)+'\n' for r in bad))
        reject(lambda:experiment.replay(sc,policy(),cache_run='cache',run_id='test',candidate_path=src,vectors=vectors,output_path=Path(tmp)/'bad.gz'))
    print('Cache -> two causal runtimes -> frozen outputs -> known-answer evaluation; reused row rejected: OK')
    print('Scene paired evaluation checks: PASSED; synthetic fixtures do not measure validation quality')


if __name__=='__main__':main()
