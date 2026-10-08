"""Stateless experimental admission of appearance samples, not person detection."""
from dataclasses import dataclass
import math
from .crops import CropRecord, crop_geometry
from .osnet import ObservationKey


@dataclass(frozen=True)
class SampleQuality:
    available: bool
    confidence_passed: bool
    border_passed: bool
    normalized_clearance: float

    @property
    def policies(self):
        return {'all_available':self.available,
                'confidence_only':self.available and self.confidence_passed,
                'confidence_and_border':self.available and self.confidence_passed and self.border_passed}


def assess_sample(record,width,height,*,min_confidence,border_fraction):
    """Use raw xyxy and per-axis image dimensions; boundaries are inclusive.

    This function consumes no GT, image pixels, embeddings or identity history.
    It neither deletes a tracked observation nor refreshes an old descriptor.
    """
    if not isinstance(record,CropRecord) or not isinstance(record.key,ObservationKey):raise ValueError('Expected crop record/key')
    if any(type(x) is not int or x<0 for x in (record.key.camera_id,record.key.local_id,record.key.frame_index)):raise ValueError('Invalid key')
    for x,limit in ((min_confidence,1.),(border_fraction,.5)):
        if type(x) not in (float,int) or not math.isfinite(x) or not 0<=x<=limit:raise ValueError('Invalid admission settings')
    score=record.confidence
    if type(score) not in (float,int) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid score')
    bounds,fraction=crop_geometry(record.source_xyxy,width,height)
    if bounds!=record.crop_xyxy_int or fraction!=record.inside_image_fraction:raise ValueError('Crop geometry differs from raw box')
    x1,y1,x2,y2=record.source_xyxy
    clearance=min(x1/width,y1/height,(width-x2)/width,(height-y2)/height)
    return SampleQuality(bounds is not None,score>=min_confidence,clearance>=border_fraction,float(clearance))
