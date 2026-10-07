# Scene calibration and footpoint audit

This CPU-only diagnostic starts geometric validation without changing any
association or global identity policy. It uses the existing pinned source
manifest and checks the calibration and GT SHA-256 hashes before and after
reading them. There are no new dependencies beyond NumPy.

## Model and conventions

A projective camera maps a world point to an image point as
`[u,v,1] ~ P [X,Y,Z,1]`. The tilde means equality up to a nonzero scale.
For the plane `Z=0`, the relevant matrix is `H = P[:, [0,1,3]]`.
Consequently, `H` maps ground coordinates to image coordinates, and its
inverse maps image points to the ground plane. Divide by the third
homogeneous component after multiplication. Near-zero denominators are
reported as invalid projections rather than coerced to finite points.

The audit checks whether the actual file supports this interpretation:

- Finite matrix shapes: K 3x3, E 3x4, P 3x4 and H 3x3.
- Invertible H.
- P proportional to K @ E.
- H proportional to the X, Y and translation columns of P.
- Expected image dimensions and frame rate for cameras 4/5/8.

The relative matrix comparison uses Frobenius normalization and accepts
positive or negative projective scales. Its tolerance of 1e-6 is a numeric
consistency tolerance, not a localization or association distance threshold.
Failure stops projection instead of silently changing the convention.
Rotation orthogonality, rotation determinant and the H condition number are
reported for inspection. A condition number depends on coordinate units;
it is not by itself a calibration-quality score.

The NVIDIA configuration reference describes the homography in this schema
as ground-to-image. Its prose about extrinsics is ambiguous (including a
3x3 shape description); this audit therefore checks the actual 3x4 matrices
and their numerical relationship instead of relying on that prose alone.

## Empirical probe

Read GT frames 2..299 for cameras 4/5/8. For raw GT box `(x,y,w,h)`, use
`(x+w/2, y+h)` as an approximate ground-contact image point. No clipping,
integer rounding, pixel offset or geometric correction is applied.

Report separately for all raw boxes and fully inside boxes:

- Distance between inverse-projected footpoint and GT world `(X,Y)`.
- Pixel distance between forward-projected GT world position and footpoint.
- Forward/inverse roundtrip pixel error (numerical self-consistency only).
- Number of invalid inverse projections.

Also report disagreement of GT world coordinates for the same GT identity
at the same frame across each camera pair. This checks whether the GT
coordinates themselves appear to describe shared positions across views.

Distances are labelled native dataset coordinate units. This audit does not
independently establish their physical scale. Verify units and the semantic
meaning of the annotated world position before selecting a distance gate.
A GT box is not a direct measurement of foot contact. Occlusion, pose, box
extent and camera perspective can produce discrepancies even with accurate
calibration; fully inside does not mean unoccluded. No discrepancy value is
used as an automatic physical-calibration pass/fail threshold.

## Running

From the WSL project root with `.venv` active:

```bash
python scripts/audit_scene_geometry.py
```

For synthetic mathematical checks only:

```bash
python scripts/audit_scene_geometry.py --check-only
```

Output: `artifacts/geometry_audit/<UTC-run>/report.json` and
`gt_projection.csv`. The report contains source hashes, code hash, matrix
checks and empirical summaries. CSV rows retain camera/frame/GT identity,
raw footpoint, GT and projected world positions and discrepancies.

The completion message deliberately does not certify physical accuracy.
Small roundtrip error also occurs for a physically wrong but invertible
homography. Visual reprojection checks and an uncertainty analysis are later
steps; this diagnostic does not apply a gate or change the 85.03% baseline.

## References

- Hartley and Zisserman (2004), *Multiple View Geometry in Computer Vision*,
  second edition. Camera matrices and plane-induced projective mappings:
  https://robots.ox.ac.uk/~vgg/publications/2004/Hartley04c/
- NVIDIA, Multi-Camera Tracking configuration, calibration schema and
  localization/foot-position discussion:
  https://docs.nvidia.com/mms/text/MDX_Multi_Camera_Tracking_MS_Configuration.html
- Dataset: NVIDIA PhysicalAI-SmartSpaces; the local source manifest pins
  revision `2cbe9563cbe9f47f846e5c871ee994572bbbc60e`:
  https://huggingface.co/datasets/nvidia/PhysicalAI-SmartSpaces
