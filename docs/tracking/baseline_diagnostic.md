# Local tracking baseline diagnostic

## Scope

- Dataset: PhysicalAI-SmartSpaces, MTMC_Tracking_2024/train/scene_001.
- Cameras: 4, 5, 8.
- Video frames: 0–299 at 30 FPS.
- Ground-truth comparison: frames 2–299.
- Detector: RF-DETR Small, FP32, confidence threshold 0.1.
- Tracker: supervision 0.30.7 ByteTrack, independent state per camera.
- Settings: activation threshold 0.5, lost buffer 30,
  matching cost threshold 0.8, minimum consecutive frames 1.
- GT is used only for diagnostics.

## Verification

Replaying the recorded raw detections through the same ByteTrack version
and settings reproduced every recorded output box and local ID.

## Observed failures

1. Camera 5, GT person 21, local ID 17 -> 22:
   frames 88–99 contain no detection with GT IoU >= 0.5.
   On frame 104, the returning detection is accurate, but its association
   cost with lost track 17 is approximately 0.865, exceeding 0.8.
   The old track is still retained; buffer expiry is not the cause.

2. Camera 5, GT person 21, local ID 22 -> 26:
   on frames 220–224, low-confidence inaccurate detections update the
   track and move it away from the person. ID 26 appears on frame 231.

3. Camera 8, GT persons 10 and 19:
   local IDs 7 and 14 are swapped from frame 99 through frame 299.

## Interpretation

These are diagnostic examples, not official IDSW, IDF1 or HOTA results.
Per-frame GT matching uses one-to-one assignment at IoU >= 0.5,
with boxes clipped to the image.

Keep this configuration as a baseline. Evaluate later changes on a
separate validation sequence, not only these ten seconds.

Reference: https://arxiv.org/abs/2110.06864
