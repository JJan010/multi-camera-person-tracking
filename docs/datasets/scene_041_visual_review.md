# Scene 041 visual review

Input: validation_scene_audit_v1 report for cameras 361/362/364, frames
2..3599, with all decoded timelines passing. No detector or tracker runs.

The preview verifies all upstream hashes, the projection CSV and the exact
projection helper version. It uses three deterministic case selections:

- Frame 300 for every camera: raw image and all GT boxes.
- Highest finite world discrepancy among fully-inside GT boxes in each
  camera, with earliest-frame/person tie breaking: raw and annotated zooms.
- Longest consecutive interval without GT rows per camera, earliest tie:
  last preceding frame, midpoint and first following frame, clipped to the
  audited interval. Cameras without such an interval have no gap row.

All requested video frames are decoded sequentially with exact rational
PTS checks. The assumed display convention is GT frame = zero-based decoded
index. Visual inspection can expose gross misalignment; these snapshots do
not prove a zero-versus-one-frame offset or scene synchronization precisely.

White boxes and labels are GT, not model predictions. In worst-case zooms,
the selected GT box is yellow, its bottom center has a cyan circle, and its
GT world position projected into the image has a magenta cross. A yellow
line connects these two points. Coordinates use the existing raw-box
convention without clipping before projection. Discrepancy is expressed in
native dataset units, not independently established meters.

Review questions:

1. Do GT boxes align with people in the displayed images?
2. At large projection discrepancies, are feet occluded, the box affected by
   occlusion, or the planar ground/contact assumptions visibly questionable?
   Such explanations remain hypotheses unless the images support them.
3. During the sampled GT gaps, is the visible area empty, or are people
   visible without annotation? One selected gap does not establish the
   annotation completeness of all other gaps.

The outputs include full-resolution raw/annotated PNGs, paired contact
sheets and report.json. visual_review.zip contains the six contact-sheet
JPGs when gaps exist, plus the report. Full-resolution PNGs remain local;
the report records their hashes but they are not included in that compact
upload bundle. No thresholds, camera choices or tracking outputs change.

This is a data interpretation check before the frozen validation comparison,
not a step for fitting association thresholds on the validation scene.
