# Scene 041 timeline and geometry audit

Input: completed validation_video_preparation_v1 report for the frozen
361/362/364 selection. All linked manifests, annotations and videos are
checked against their recorded SHA-256 hashes. The dataset revision remains
2cbe9563cbe9f47f846e5c871ee994572bbbc60e.

The script uses the existing numerical projection helpers from
scripts/audit_scene_geometry.py; that file and the scene_001 workflow are
unchanged. Its hash is recorded in the new report.

## Decoded timeline

PyAV decodes all three complete files sequentially on CPU, without RGB
conversion, inference or video output. For zero-based decoded frame i,
frame.pts * frame.time_base must equal i / 30 exactly using rational
arithmetic. Missing timestamps, repeated/decreasing times, unexpected image
dimensions, timeline mismatches and decoded count mismatches fail the audit.
The expected count comes from the previously captured container metadata.
Failures in the timeline contract are saved with passed=false and prevent
the GT projection phase. Decoder exceptions stop processing.

A common PTS grid demonstrates timestamp compatibility for replay. It does
not independently prove alignment of scene content, GT frame numbering or
physical synchronization. No timing benchmark is claimed for this audit.

## Geometry and annotation coverage

Calibration image dimensions and FPS must match each video's metadata.
For every camera, P must be proportional to K @ E and H proportional to
P[:, (0,1,3)], within normalized matrix error 1e-6. H must be invertible.
This establishes the stored projection direction under the supplied model:
world Z=0 to image. Image bottom-center points are projected with inverse H.

GT diagnostics use frames 2..3599 and raw boxes, without clipping or pixel
offsets. Reports distinguish all boxes and fully inside boxes, recording
world discrepancy, pixel discrepancy, roundtrip errors and unavailable
projections. World coordinates remain in native dataset units. Small
roundtrip error alone is not a physical accuracy measurement.

Every camera's frame indices without GT rows are saved explicitly. The
audit neither labels these frames empty nor removes them from evaluation.
All GT indices for selected cameras must lie within decoded video bounds.
Same-frame/person world-coordinate disagreement is summarized per pair.

Outputs: timestamped report.json and gt_projection.csv under
artifacts/validation_scene_audit/. No tracking parameters or identity
assignments are changed. The next gate is visual GT/video/projection review.

Time representation reference:
https://pyav.org/docs/stable/api/time.html
