# Configurable scene inputs and empty ground-truth slots

This change adds a scene-input boundary without switching existing runners
or modifying detector, tracker, Re-ID, association or identity policies.

## Files and responsibility

- src/mtmc/data/scene.py: scene configuration, pinned file references,
  calibration and video verification. RuntimeScene contains video/camera/
  calibration/timing data. EvaluationSpec separately contains the GT file
  reference and evaluation frame/IoU conventions.
- src/mtmc/data/ground_truth.py: offline GT parser and image-sized spatial
  admissibility. The parser explicitly creates every evaluated camera/frame
  slot, including empty dictionaries for frames without GT rows.
- scripts/check_scene_inputs.py: creates frozen configurations from existing
  evidence, runs contract checks and compares the new input handling with
  the unchanged scene_001 implementation.

load_scene verifies manifest, calibration and video contents. It constructs
an evaluation GT reference without opening GT contents. Tracking callers
must receive SceneInputs.runtime. GT is read explicitly by the offline
load_ground_truth(EvaluationSpec) function. This separation prevents the
normal runtime preparation path from depending on GT labels.

Supported replay scope: synchronized constant integer FPS, runtime starting
at zero, a configured number of rounds, per-camera image dimensions and
invertible Z=0 homographies in the audited convention. Camera IDs need not
be contiguous. This is not variable-FPS, camera-offset or live-stream support.
The input loader checks files and calibration; it does not decode video or
replace the preceding timeline and visual audits.

## Evaluation convention

The two prepared configurations use runtime 0..3599 and evaluation 2..3599,
with GT-to-video index offset 0 and inclusive IoU >= 0.5. The parser preserves
ID zero and raw xywh-to-xyxy arithmetic. Different explicit offsets are
supported by the parser and tested; none are inferred or tuned on the data.

Spatial matching preserves the existing policy: clip GT and prediction boxes
at the camera image bounds, exclude zero-area clipped GT, retain every
prediction including fully outside boxes, and retain every admissible IoU
pair. Mutually unique overlaps are diagnostics, not a replacement for the
full spatial candidate matrix or global identity matching.

A frame with empty GT has a 0-by-N admissibility matrix. N predictions remain
in the metric denominator and count as false observations under the supplied
annotation protocol. Empty slots are not skipped. A missing camera/frame key
outside the configured interval raises KeyError instead of silently creating
an empty slot. Duplicate GT IDs in a slot and invalid boxes are rejected.
Annotation completeness remains a dataset limitation, not inferred from the
absence of rows.

## Checks

Synthetic checks cover arbitrary cameras/FPS/dimensions, GT ID zero, explicit
frame offsets, runtime loading while the GT file is absent, empty-frame
metric accounting against motmetrics, malformed/duplicate GT, changed
checksums, scene-scope mismatch and spatial parity with the old implementation.

The real-data command:

1. Verifies the frozen 3600-round scene_001 pipeline report and scene_041
   timeline/geometry audit and builds configs/scenes/scene_001_two_minutes.json
   and scene_041_two_minutes.json. Different existing configs are not overwritten.
2. Compares every scene_001 GT slot/ID/box and homography with the old loaders.
3. Compares every camera/frame spatial output on the frozen original tracking
   trace over frames 2..3599: 10794 slots, including all admissible IoU pairs,
   diagnostic matches and clipped/outside counts.
4. Reproduces scene_041 GT counts and its exact lists of empty camera/frame slots.

Report: artifacts/scene_input_checks/<UTC run>/report.json. Configuration
paths are relative to the project root; evidence hashes and code versions
are recorded. Existing audit reports retain their own absolute paths.

A passing result establishes input/GT/spatial-adapter parity. It does not
claim new GPU inference parity, complete end-to-end identity parity or a
validation IDF1 result. The next step is to connect these inputs to a runner
for the already frozen staged and competitive_iou policies and an evaluator,
with scene_001 regression checks before scene_041 quality evaluation.
