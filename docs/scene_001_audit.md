# Scene 001: initial integration data audit

## Source

- Dataset: nvidia/PhysicalAI-SmartSpaces
- Revision: 2cbe9563cbe9f47f846e5c871ee994572bbbc60e
- License: CC-BY-4.0
- Source: https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
- Scene: MTMC_Tracking_2024/train/scene_001
- Selected cameras: 4, 5, 8
- Role: initial integration and debugging, not held-out evaluation.

Source and video manifests are stored in configs/datasets/.
Downloaded data and generated artifacts are excluded from Git.

## Ground truth

Columns:
camera_id, person_id, frame_id, x, y, width, height, world_x, world_y.

The full scene audit found no duplicate camera/person/frame keys
and no nonpositive bounding-box dimensions.
Annotated frame IDs range from 2 to 23993.
Some original boxes extend beyond image boundaries.

## Video metadata

All three selected videos report:
- H.264 High, yuv420p
- 1920 x 1080
- 30/1 nominal and average frame rate
- Time base: 1/15360
- Start PTS: 0
- Duration: 799.8 seconds
- Frame count: 23994

These are reported metadata, not a full decoded-frame or timestamp audit.

## Diagnostic previews

Generated samples: zero-based video frames 2, 300, 1800.
Working mapping: GT frame ID equals zero-based video frame index.
No frame offset has been applied.

Across nine samples:
- 157 GT boxes
- 15 partially outside the image
- 0 fully outside the image

Visual review covered frames 300 and 1800 from all three cameras.
Boxes appeared aligned with people, with consistent IDs in inspected
cross-camera examples. No large systematic offset was observed.
This does not establish frame-exact synchronization for the full videos.

Boxes are clipped only for preview rendering; source GT is unchanged.
In-bounds boxes may still contain occluded people.

## Next checks

- Verify decoded frame timestamps and counts.
- Implement deterministic multi-camera replay.
- Keep GT restricted to diagnostics and evaluation.
- Define separate validation and test data before tuning and final evaluation.
