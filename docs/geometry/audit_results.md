# Geometry audit: results and wrong-merge case

Scene: `MTMC_Tracking_2024/train/scene_001`; cameras 4/5/8.
Dataset revision: `2cbe9563cbe9f47f846e5c871ee994572bbbc60e`.
Audit range: frames 2..299 at 30 FPS.

Source runs:

- Matrix and GT-box audit: `20261007T110015275411Z`.
- Projection preview: `20261007T111525044390Z`.
- Tracked-box wrong-merge probe: `20261007T112120168558Z`.
- Unchanged controlled identity baseline: `20261007T104647560154Z`.

## Matrix and coordinate checks

Pinned calibration and GT checksums were verified. For all three cameras,
`P ~ K @ E` and `H ~ P[:, [0,1,3]]`; the largest reported proportionality
error is approximately 1.3e-16. The homography maps world-plane Z=0 coordinates
to image coordinates; inverse H maps an image point to that plane.

There were no invalid inverse projections among the inspected GT boxes.
Maximum numerical forward/inverse roundtrip discrepancy was approximately
2.15e-12 pixels. These checks establish numerical consistency, not physical
calibration accuracy.

Same-frame/same-person annotated world positions agree exactly across all
inspected camera pairs: 4072 observations for 4/5, 4286 for 4/8 and 4412 for
5/8. These counts refer to repeated pair observations, not independent people.

## Raw GT-box footpoint approximation

The footpoint proxy is `(x+w/2, y+h)` from the raw GT box. Distances below
compare its inverse projection with the annotated world position. Coordinates
were not clipped or offset. Units are native dataset coordinates; their
physical scale has not been independently established in this audit.

| Camera | All-box rows | All-box p95 | Fully-inside rows | Inside median | Inside p95 | Inside maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4649 | 0.385805 | 4393 | 0.170083 | 0.354766 | 0.580411 |
| 5 | 6013 | 0.644047 | 5087 | 0.120968 | 0.329333 | 1.576823 |
| 8 | 5655 | 0.352028 | 5540 | 0.167443 | 0.334887 | 2.062187 |

Visual inspection of the selected inside-image outliers shows lower-body
occlusion for camera 5/frame 293/GT 20 and camera 8/frame 199/GT 8. The boxes
end at the visible upper body rather than ground contact. Camera 4/frame
136/GT 5 shows an oblique view of a walking person with separated feet; this
image alone does not isolate the cause or establish the precise GT anchor.

Thus, inside-image status does not establish valid foot contact. A homography
cannot determine whether its input pixel belongs to the ground or a torso.
The numerical discrepancy includes footpoint-proxy and annotation-semantic
effects and should not be described solely as calibration error.

## Wrong merge at frame 149

The controlled identity manager merged camera 8/local 17 (GT 23) into the
identity containing camera 5/local 9 (GT 24). The following probe uses the
original frozen ByteTrack boxes and the verified calibration.

| Quantity | Distance in native dataset units |
| --- | ---: |
| Annotated GT world-position separation | 17.464799 |
| Separation from raw GT-box footpoint projections | 17.273844 |
| Separation from frozen tracker-box footpoint projections | 17.404844 |
| Tracker-point discrepancy from GT, camera 5/local 9 | 0.158985 |
| Tracker-point discrepancy from GT, camera 8/local 17 | 0.116014 |

The tracked-box projections are `[5.5361300565, -2.8919144674]` and
`[-0.0619438749, 13.5880785063]` respectively. Both tracked boxes are fully
inside their images. The measured separation is large relative to the two
individual discrepancies in this case. Geometry therefore supplies evidence
against this particular appearance-based association.

This does not measure a geometry gate's effect on global IDF1. No association
was changed, no distance threshold was chosen, and the controlled baseline
remains IDF1=85.03%. A counterfactual runtime replay is required to measure
the effect of rejecting links or merges.

## Next experiment

Measure cross-camera distances from frozen tracked boxes over the complete
diagnostic fragment, distinguishing same-person, different-person and
unresolved pairs using offline GT evidence. Include correct pairs and
occluded/outlier cases, not only the selected wrong merge. Keep GT out of
runtime geometry and association decisions.

Single-observation GT-box error quantiles are not thresholds for pairwise
tracked-point distances. Quality flags and missing/unreliable geometry need
explicit handling. Selecting or evaluating a gate on this reused training
fragment remains an integration experiment, not independent validation.

Full provenance reports are stored alongside this document as
`calibration_audit_metrics.json`, `projection_preview_metrics.json` and
`wrong_merge_probe_metrics.json`. Large images and per-observation artifacts
remain under the corresponding ignored artifact run directories.
