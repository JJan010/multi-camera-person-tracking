# Scene 041: visual review of camera 362 false positives

Source diagnostic: `20261007T215537181420Z`, staged, camera 362.
All six supplied JPEGs were inspected, and the report artifact hashes checked.
This review supplements the source report; its original visual-review-pending
flag is preserved rather than rewriting the frozen artifact.

## Findings

- Frame 753, local 9; frame 1016, local 9; frame 2773, local 29:
  the selected box covers the same pallet/shelving area at the bottom right.
  No person is visible inside the selected boxes. These samples establish
  actual background false detections.
- Frame 574, local 3: a partially occluded person is clearly visible inside
  the selected box despite no GT rows for the camera/frame. Under the pinned
  protocol this remains an FP. The image establishes a visible-person/GT
  discrepancy, not why the annotation is absent. Annotation policy, mapping
  or omission must be assessed before claiming a dataset error.
- Frames 2432/local 35 and 124/local 2: selected predictions overlap annotated
  people and have admissible IoU, but remain unmatched in the evaluator.
  The other predictions are omitted in these views; the images alone do not
  identify the competing prediction or establish its correctness.

## Quantitative context

Camera 362 has 4047 local FP observations: 3714 without an admissible GT box,
325 in empty GT slots, and 8 admissible but unmatched.
Local 9 contributes 1486 consecutive FP frames (226..1711).
Local 29 contributes 1653 consecutive FP frames (1947..3599).
Together these tracks contribute 3139/4047 = 77.56% of camera-local FP.
This is a track-level count, not a visual classification of every frame.
The CSV box coordinates remain concentrated in the reviewed pallet region.

Local 9 score median/max: 0.31173/0.72364; 191 FP observations have score
at least 0.5 and 34 at least 0.6. Local 29 median/max: 0.51476/0.75490;
951 have score at least 0.5 and 137 at least 0.6.
The problem is therefore not limited to weak detection maintenance.

The inspected tracker code uses high-stage threshold 0.5, low-score
association above 0.1, and new-track threshold 0.6. ByteTrack's use of weak
observations can preserve real occluded people, but repeated background
boxes can also maintain an already active false track. The latter is a
mechanistic interpretation; the images alone do not identify every match
stage or the birth event.

OSNet describes the appearance of the supplied crop; similarity between
repeated crops of a pallet does not establish that the crop contains a
person. Likewise, homography consistency is not a person classifier.

## Decision and next scope

Keep staged and the original evaluation protocol. Do not delete local IDs
9/29, mask this validation image region to improve the published baseline,
or fit a new confidence threshold to these examples. A stationary-track
filter could remove real stationary people.

Record this as a detector/background failure plus a separate annotation
coverage caveat. Camera 364 missed observations are the next bounded review
before selecting a general improvement. If detector fine-tuning or hard
negative mining is pursued, construct the training data from the training
split, preserve this result, and assess the new policy on an additional
untouched scene. Confidence filtering alone has not been validated.

Reference: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating
Every Detection Box*, ECCV 2022, https://arxiv.org/abs/2110.06864.
