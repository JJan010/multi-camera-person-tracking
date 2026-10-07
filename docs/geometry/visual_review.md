# Geometry visual review and targeted tracked-box probe

Reviewed images come from preview run `20261007T111525044390Z`, based on
geometry audit `20261007T110015275411Z`. This is a qualitative review of a
small, deliberately selected set, not an accuracy validation.

## Observations

| Camera/frame/GT | Discrepancy | Visible evidence |
| --- | ---: | --- |
| 4 / 136 / 5 | 0.580 | The person is walking, with separated feet and an oblique view. The projected GT point differs from the box bottom center. This alone does not establish the exact cause or GT ground-anchor semantics. |
| 5 / 293 / 20 | 1.577 | The selected person's lower body is obscured by a foreground person. The GT box covers the visible upper body and ends above the feet. |
| 8 / 199 / 8 | 2.062 | Foreground people obscure the selected person's lower body. The selected GT box ends around the upper-body region rather than ground contact. |

All distances use native dataset coordinate units. These examples show why
being fully inside the image is insufficient to make the box bottom a
reliable ground-plane point. An algebraically correct homography maps any
image ray to the chosen ground plane; it does not detect that the supplied
pixel lies on a torso rather than at ground contact.

A projected ground point can appear over a foreground person's body because
that person occludes the background ground location. That alone is not a
reason to relabel the GT identity or reject the calibration.

In frame 149, camera 4 has no selected GT 23/24 annotation. Camera 5 contains
GT 24, whose GT-box discrepancy is approximately 0.083; camera 8 contains
GT 23, whose GT-box discrepancy is approximately 0.176. Small individual
projection discrepancies do not themselves give the distance between these
two people. The prior identity diagnostics establish that the erroneously
merged local tracks are camera 5/local 9 and camera 8/local 17.

## Next measurement

`probe_wrong_merge_geometry.py` computes at frame 149:

1. Distance between the two annotated GT world positions.
2. Distance between their inverse-projected raw GT-box bottom centers.
3. Distance between inverse-projected bottom centers of their frozen
   ByteTrack boxes, together with each tracked-point discrepancy versus GT.

The probe verifies source checksums and scene/run/timestamp relationships.
It uses the existing calibration helper; no new dependency is required.
GT IDs are explicitly used for this offline diagnostic, never as a proposed
runtime association input. No distance gate is selected and no historical
identity outputs are changed.

Run from the WSL project root with `.venv` active:

```bash
python scripts/probe_wrong_merge_geometry.py \
  --geometry-report artifacts/geometry_audit/20261007T110015275411Z/report.json \
  --identity-report artifacts/controlled_merge/20261007T104647560154Z/report.json
```

A timestamped report is saved under `artifacts/geometry_merge_probe/`.
A large separation in this one wrong case would motivate a broader paired
analysis on tracked boxes. It would not justify a threshold by itself;
correct associations and occluded cases must also be measured.
