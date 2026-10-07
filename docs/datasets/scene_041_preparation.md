# Scene 041 validation preparation

Dataset: nvidia/PhysicalAI-SmartSpaces, revision
2cbe9563cbe9f47f846e5c871ee994572bbbc60e, CC-BY-4.0.
Scene: MTMC_Tracking_2024/val/scene_041.

The original metadata-based candidate was cameras 361/362/363. In frames
0..3599, camera 363 had no same-frame/person GT overlap with either anchor.
The existing cross-camera association mechanism needs simultaneous evidence.
We retained anchors 361/362 and selected the lowest other camera ID with
positive same-frame/person overlap with at least one anchor: 364.
This rule uses GT coverage, not model scores or tracking quality. It selects
an overlapping-view validation subset, not a representative random sample
of all possible camera configurations. It does not validate transitions
between cameras without simultaneous overlap.

Observed pair coverage (positive intersection with image bounds):

| Pair | Person-frame observations | Persons | Frames |
| --- | ---: | ---: | ---: |
| 361-362 | 2422 | 17 | 1962 |
| 361-364 | 14874 | 24 | 3575 |
| 362-364 | 8898 | 23 | 3192 |

Planned runtime frames: 0..3599; planned quality evaluation: 2..3599.
The GT/video frame convention and timestamp alignment still require checks.
No runtime state reset at the warmup boundary is planned. Frames without GT
rows must be represented in evaluation; missing rows alone do not establish
whether the scene is empty or annotations are unavailable.

## Preparation script

Run scripts/prepare_validation_videos.py with --annotation-report pointing
to the completed validation_annotation_audit_v1 report. It requires the
previous scripts/prepare_validation_annotations.py helper, NumPy,
huggingface_hub and the system ffprobe already used in this project.

The script verifies the annotation manifest/checksums, reproduces the
selection rule, records all candidate coverage counts, and freezes
configs/datasets/scene_041_selection.json. It then lists the three exact
video files at the pinned revision, writes scene_041_video_plan.json,
downloads only those videos and verifies upstream checksums. Local SHA-256
checksums are recorded in scene_041_video_manifest.json.

Container metadata is saved in a timestamped artifacts/validation_videos
report. A matching existing configuration is accepted on rerun; a differing
configuration is not overwritten. Download metadata and partial-transfer
handling are delegated to huggingface_hub. No models or tracking run.

Next: decoded timestamp checks, calibration/projection and visual alignment
checks, then scene-independent input/evaluation handling with preserved
scene_001 behavior. Compare frozen staged and competitive_iou configurations
before considering any runtime promotion or new parameter fitting.

Reference for pinned downloads:
https://huggingface.co/docs/huggingface_hub/package_reference/file_download
