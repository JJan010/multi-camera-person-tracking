# Validation annotation audit

Candidate: MTMC_Tracking_2024/val/scene_041, the first scene in lexical
catalog order. Candidate cameras: 361, 362, 363, selected before tracking
quality evaluation. Audit window: frame indices [0, 3600).

Dataset: nvidia/PhysicalAI-SmartSpaces.
Revision: 2cbe9563cbe9f47f846e5c871ee994572bbbc60e.
License: CC-BY-4.0.

The script downloads only ground_truth.txt and calibration_2025_format.json.
It verifies sizes and upstream LFS SHA-256 or Git blob SHA-1, then records
local SHA-256 checksums in configs/datasets/scene_041_source.json.
A differing existing manifest is not overwritten.

The audit checks the nine-column GT schema, integral camera/person/frame
keys, positive box dimensions, duplicate person/frame keys per camera,
calibration camera coverage, and image-boundary intersections. Person ID 0
is valid. Per-camera counts cover the entire annotation file. Pair coverage
uses only the requested frame window and boxes with positive intersection
with the image. It does not rank alternative camera subsets.

GT coverage is not tracking accuracy or a guarantee of visual visibility.
Missing rows are not proof of an empty frame. Video duration, PTS alignment,
frame-index convention and projection accuracy still require separate
verification. The window is not yet a verified two-minute recording.

No detector, tracker, embedding model or global identity manager runs.
No association threshold is selected. Reports are saved in a new timestamped
artifacts/validation_annotations directory. This prepares an unseen-scene
comparison of frozen staged and competitive_iou policies; it does not
establish generalization by itself.
