# GT projection visual audit

The numeric audit run `20261007T110015275411Z` established projective-matrix
consistency and an invertible ground-to-image homography for cameras 4/5/8.
For matching GT identities at identical frames, the supplied world-coordinate
values agree exactly across the three camera pairs in frames 2..299. This is
consistency of annotations, not independent localization accuracy.

## Observed discrepancy: fully inside GT boxes

Distances below are native dataset coordinate units, not yet independently
verified physical units. They compare the inverse-projected raw box bottom
center with the provided GT world position.

| Camera | Rows | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 4 | 4393 | 0.1701 | 0.3548 | 0.5804 |
| 5 | 5087 | 0.1210 | 0.3293 | 1.5768 |
| 8 | 5540 | 0.1674 | 0.3349 | 2.0622 |

With all raw boxes included, p95 is 0.3858, 0.6440 and 0.3520 respectively.
There were no invalid inverse projections. Maximum numerical roundtrip error
is about 2.15e-12 pixels; that only establishes numerical consistency.

Fully-inside boxes have smaller p95 values in this fragment, but inside-image
status does not guarantee reliable foot localization. The outlier in camera
8 survives the inside-image restriction. These GT-box statistics must not be
assumed to describe detector/tracker boxes without measuring those boxes too.

## Reproducible visual selection

The preview verifies the audit report's input hashes, projection CSV checksum
and pinned video manifest/checksums. It selects:

1. Frame 149 in cameras 4/5/8, showing GT 23 and 24 wherever present. This
   provides context for the previously diagnosed wrong identity merge.
2. The largest world discrepancy among fully-inside boxes for each camera,
   frames 2..299. Exact ties use smallest frame, then GT ID.

Videos are decoded sequentially from frame zero with PyAV. Decoded timestamps
must equal index/30 exactly. The preview reuses NumPy, Pillow and PyAV; it
requires the earlier `scripts/audit_scene_geometry.py` helper.

All annotations in this preview are GT annotations, not runtime tracks:

- White rectangle: raw GT box, clipped for display only.
- Cyan circle: raw box bottom-center approximation of a footpoint.
- Magenta cross: GT world position projected through H at Z=0.
- Yellow line: pixel discrepancy between those two image points.

A magenta cross outside the image is not clamped to the image boundary. Its
original position and in-image status are recorded in the JSON report.
GT IDs printed on images are not local or global tracker IDs.

## Run in the WSL project

```bash
python scripts/preview_scene_geometry.py \
  --audit-report artifacts/geometry_audit/20261007T110015275411Z/report.json
```

A new `artifacts/geometry_preview/<UTC-run>/` directory contains full-resolution
annotated PNGs, two contact sheets and a provenance report. `frame149_contact.jpg`
shows the complete views; `worst_inside_contact.jpg` shows enlarged regions
around the selected boxes and projected points. Resizing is for display only;
all stored diagnostic coordinates remain in original 1920x1080 image space.

Inspect where the projected GT point and box bottom center lie relative to
visible feet, occlusions, body extent and the ground. A few selected images
cannot establish an error distribution or approve a geometry gate. The
script neither changes association nor chooses a distance threshold.
