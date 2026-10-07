"""Check deterministic transition selection and display-only context bounds."""
from pathlib import Path
import tempfile

from PIL import Image
import preview_identity_switch as preview


def main():
    rows=[{'camera':5,'local_id':11,'previous_evidence_frame':99,'frame':100,'gap_frames':1,
           'previous_gt':7,'current_gt':8,'previous_global_id':22,'current_global_id':22},
          {'camera':5,'local_id':11,'previous_evidence_frame':199,'frame':205,'gap_frames':6,
           'previous_gt':8,'current_gt':7,'previous_global_id':22,'current_global_id':22}]
    event,frames=preview.select_transition(list(reversed(rows)),5,11,300)
    assert event==rows[0] and frames==[84,94,99,100,105,115]
    early={**rows[0],'previous_evidence_frame':2,'frame':3}
    _,frames=preview.select_transition([early],5,11,10)
    assert frames==[0,2,3,8,9]
    try:
        preview.select_transition(rows,4,99,300)
    except ValueError:
        pass
    else:
        raise AssertionError('Missing target transition accepted')
    print('Earliest transition selection, input order, boundary frames and missing target: OK')
    assert preview.common_roi([[10,20,50,80],[100,200,150,250]],width=300,height=300,padding=10)==(0,10,160,260)
    view={'tracks':[{'local_id':11,'global_id':22,'xyxy':[10,20,50,80]},
                    {'local_id':12,'global_id':23,'xyxy':[100,20,150,80]}],
          'selected_gt_boxes':{7:[9,19,51,81],8:[100,20,150,80]}}
    source=Image.new('RGB',(300,300),'#353535')
    before=source.tobytes()
    annotated=preview.annotate(source,view,event,5,11)
    assert source.tobytes()==before and annotated.tobytes()!=before
    panel=preview.panel(annotated,['Synthetic layout check','No real video used'])
    assert panel.size==(800,600)
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory)/'preview.png';panel.save(p)
        with Image.open(p) as loaded:assert loaded.size==(800,600)
    print('Shared context bounds, immutable source pixels and annotation output: OK')
    print('Identity transition preview checks: PASSED; real video decode is checked during preview')


if __name__=='__main__':
    main()
