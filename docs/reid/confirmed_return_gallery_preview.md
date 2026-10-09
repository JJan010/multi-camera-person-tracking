# Actual archive sample visual review

Use `scripts/preview_confirmed_return_gallery.py --gallery-report <report.json>
--event-frame <frame>`; repeat `--event-frame` for additional events in the same
scene. The input must be a verified `confirmed_return_gallery_audit_v1` report.

Every selected query and archived sample from the requested events is shown.
No sample is filtered by its GT label. Original tracked boxes, effective segment
keys and spatial labels are reproduced against the frozen trace before decoding.
All predictions in the camera participate in spatial ambiguity checks.

The script verifies pinned video files, decodes sequentially on CPU and checks
exact frame PTS. Each sample produces its original RGB model crop, an unannotated
neighborhood, and an annotated neighborhood with the target box in red and GT
boxes in green. Contact sheets put the raw crop beside annotated context, with
source frame, camera, local ID, segment, observed global ID, score, cosine, GT and
spatial matching reason. Queries appear first; pages contain at most nine panels.

Output: `artifacts/confirmed_return_gallery_preview/<run>/visual_review.zip`,
containing a report, contact sheets and PNGs. This is offline visual review:
no inference, new tracking decisions, metric changes or threshold tuning.
An unknown GT label is not treated as a false positive or a different person.

Local self-check covers clipping and exact integer crop bounds, ID zero,
segment provenance labels and pagination. Real source videos are available only
in the user's environment; their decode and exact PTS checks run there.
