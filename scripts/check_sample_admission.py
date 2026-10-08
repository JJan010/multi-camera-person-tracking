"""Known answers for appearance-sample admission; not accuracy calibration."""
from dataclasses import replace
from mtmc.reid.crops import CropRecord,crop_geometry
from mtmc.reid.osnet import ObservationKey
from mtmc.reid.sample_quality import assess_sample


def record(box,score=.5,w=100,h=200):
    bounds,frac=crop_geometry(box,w,h)
    return CropRecord(ObservationKey(0,0,0),score,tuple(box),bounds,frac)


def check():
    settings=dict(min_confidence=.5,border_fraction=.01)
    def assess(box,score=.5,w=100,h=200):return assess_sample(record(box,score,w,h),w,h,**settings)
    assert all(assess((1,2,99,198)).policies.values())
    for box in ((.5,2,99,198),(1,1,99,198),(1,2,99.5,198),(1,2,99,199)):
        r=assess(box);assert r.available and not r.border_passed
    assert not assess((1,2,99,198),.49).policies['confidence_only']
    assert assess((0,0,10,20),.9).policies=={'all_available':True,'confidence_only':True,'confidence_and_border':False}
    assert not any(assess((-20,1,-1,20)).policies.values())
    assert assess((-1,2,10,20)).normalized_clearance<0
    assert assess((2,4,198,396),w=200,h=400).policies==assess((1,2,99,198)).policies
    r=record((1,2,99,198));before=repr(r);assess_sample(r,100,200,**settings);assert repr(r)==before
    for bad in (replace(r,confidence=float('nan')),replace(r,crop_xyxy_int=None),replace(r,inside_image_fraction=.1),replace(r,key=ObservationKey(0,-1,0))):
        try:assess_sample(bad,100,200,**settings)
        except ValueError:pass
        else:raise AssertionError('Malformed input accepted')
    for setting in ({**settings,'border_fraction':-.1},{**settings,'min_confidence':float('nan')}):
        try:assess_sample(r,100,200,**setting)
        except ValueError:pass
        else:raise AssertionError('Malformed settings accepted')
    print('Inclusive confidence/border; four edges; raw outside boxes; scale invariance; ID zero; invalid inputs and immutable records: PASSED')
    print('Admission is a hypothesis about sample suitability, not a person/occlusion classifier.')


if __name__=='__main__':check()
