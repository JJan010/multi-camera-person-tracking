"""Known-answer admission accounting without changing spatial competitors."""
from dataclasses import asdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import tempfile
import numpy as np
from audit_sample_admission import freeze, audit, NAMES
from mtmc.data.scene import RuntimeScene, CameraInput, FileRef, EvaluationSpec, sha256
from mtmc.data.ground_truth import load_ground_truth
from mtmc.reid.crops import crop_geometry
from mtmc.reid.osnet import ObservationKey


def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp); unused=FileRef(root/'unused','0'*64)
        scene=RuntimeScene('fixture','fixture','0'*40,30,2,
            (CameraInput(361,unused,100,80,np.eye(3)),),unused,'fixture')
        # ID 0 is a genuine, uniquely matched border observation. Two other
        # predictions compete for GT 1; rejecting one must not relabel the other.
        cases=[(0,[0.,10.,10.,30.],.8),(1,[30.,10.,40.,30.],.8),
               (2,[30.,10.,40.,30.],.2),(3,[60.,10.,70.,30.],.8),
               (4,[110.,10.,120.,30.],.8)]
        trace=root/'trace.jsonl.gz'
        with gzip.open(trace,'wt') as stream:
            for frame in range(2):
                selected=cases if frame==0 else [(3,[60.,10.,70.,30.],.8)]
                camera=dict(camera=361,local_ids=[],xyxy=[],confidence=[],embedding_rows=[])
                bindings=[]
                for local,box,score in selected:
                    key=asdict(ObservationKey(361,local,frame))
                    bindings.append(dict(source_key=key,identity_key=key))
                    camera['local_ids'].append(local);camera['xyxy'].append(box)
                    camera['confidence'].append(score)
                    bounds,_=crop_geometry(box,100,80)
                    camera['embedding_rows'].append(local if bounds else None)
                stream.write(json.dumps(dict(run_id='fixture',frame_index=frame,
                    timestamp=str(Fraction(frame,30)),variants=dict(enabled=dict(
                        cameras=[camera],segment_bindings=bindings))))+'\n')
        masks=root/'masks.jsonl.gz'
        assert freeze(trace,masks,scene,'fixture',dict(min_confidence=.5,border_fraction=.01))==6
        gt=root/'gt.txt'
        gt.write_text('361 0 0 0 10 10 20 0 0\n361 1 0 30 10 10 20 0 0\n')
        spec=EvaluationSpec(FileRef(gt,sha256(gt)),scene.camera_ids,0,1,0,.5,'retain_slot_and_count_predictions')
        result=audit(trace,masks,scene,spec,load_ground_truth(spec))
        for name,accepted in zip(NAMES,(5,4,3)):
            c=result['totals'][name]
            assert c['observations']==6 and c['accepted']==accepted
            assert c['mutually_unique_gt']==1 and c['ambiguous_gt']==2 and c['no_admissible_gt']==3
            assert c['accepted_no_admissible_gt']==2
        assert result['totals']['confidence_only']['accepted_ambiguous_gt']==1
        assert result['totals']['confidence_and_border']['rejected_mutually_unique_gt']==1
        assert result['segments_without_accepted_sample']==dict(zip(NAMES,(1,2,3)))
        assert result['GT_people_without_uniquely_matched_accepted_sample']['confidence_and_border']==[0,1]
        assert result['GT_people_without_uniquely_matched_accepted_sample']['all_available']==[1]
    print('Known admission counts, genuine border-sample loss, unchanged ambiguous competitors, empty GT, outside crops and ID zero: PASSED')


if __name__=='__main__':main()
