# Scene 041 input review results

Scene: MTMC_Tracking_2024/val/scene_041; cameras 361/362/364.
Planned continuous runtime: frames 0..3599; quality evaluation: 2..3599.
Dataset revision: 2cbe9563cbe9f47f846e5c871ee994572bbbc60e.

Evidence:
- Annotation audit: 20261007T203319407142Z.
- Pinned video preparation: 20261007T203915111446Z.
- Timeline/geometry audit: 20261007T204521274581Z.
- Visual preview: 20261007T204922645465Z.

The six uploaded contact sheets were opened, compared in raw and annotated
form, and checked against their SHA-256 hashes in the preview report.
The full-resolution individual PNGs were not in the upload bundle and were
not inspected. This review records interpretation of sampled images; it
preserves the original automated report, including its pre-review flag.

## Timestamp and matrix evidence

All three files decode to 23994 frames, with exact i/30 presentation times,
no missing/nonincreasing timestamps, and no dimension mismatches. Calibration
and video dimensions/FPS agree. P is proportional to K @ E and H to the
Z=0 columns of P. These checks establish numerical and timeline consistency,
not independently measured physical calibration accuracy or content sync.
Shared same-frame/person GT world positions agree exactly across the cameras.

## Visual findings

At frame 300, inspected GT boxes generally align with visible people or
visible body portions. No obvious gross GT/video displacement appears in
these contact sheets. This does not resolve a possible one-frame offset or
certify annotation completeness for every visible fragment.

| Camera | Frame | GT ID | Box width x height | World discrepancy | Visual evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| 361 | 629 | 14 | 49 x 103 px | 1.201355 | Another person obscures the lower body; the selected box bottom is above the hidden feet. |
| 362 | 3422 | 19 | 22 x 30 px | 3.451801 | A small upper-body/head fragment remains visible above a large foreground occlusion. |
| 364 | 174 | 11 | 23 x 37 px | 4.856122 | The selected person is mostly hidden; the box surrounds a small visible fragment above the occluding region. |

These images support an occlusion-related failure of the bottom-center
proxy. A box entirely inside image bounds can still omit the hidden lower
body. Inverting a ground-plane homography for an elevated body point does
not recover the person's ground contact. The magenta projected GT points
in occluded areas are consistent with hidden ground locations; they are not
visible measurements of the feet. These cases do not prove that every
projection discrepancy has this cause or that calibration is physically
perfect. Native dataset coordinate units are retained.

This failure mechanism is also discussed by Kim et al., "Addressing the
Occlusion Problem in Multi-Camera People Tracking With Human Pose
Estimation," CVPR Workshops, AI City, 2023 (discussion of homography and
occluded lower bodies around Fig. 7):
https://openaccess.thecvf.com/content/CVPR2023W/AICity/papers/Kim_Addressing_the_Occlusion_Problem_in_Multi-Camera_People_Tracking_With_Human_CVPRW_2023_paper.pdf

## Sampled intervals without GT rows

- Camera 361: longest interval 1737..1757, 21 frames. Preview frames
  1736/1747/1758 show a partial person leaving, a middle frame with no clearly
  visible person in the inspected view, and a partial person entering.
- Camera 362: longest interval 690..799, 110 frames. Preview frames
  689/744/800 show small annotated fragments at the interval boundaries and
  no clearly visible person in the middle frame at contact-sheet resolution.

The sampled gaps are consistent with temporary absence of visible annotated
people. This is not proof that all 320 camera-362 frames without GT rows are
complete empty-scene annotations. The total 21/320/0 frame counts apply to
cameras 361/362/364 over evaluation frames 2..3599.

## Decision and next implementation step

Proceed with the selected input subset and document these limitations.
Keep all frozen staged and competitive_iou settings, including geometry,
unchanged for the first validation comparison. Do not discard hard examples
or unannotated frames to improve metrics. Evaluate against the supplied GT
and state its interpretation limits. An empty GT slot must be represented:
predictions there remain eligible to count as false positives under that
protocol. Confirmed annotation issues would require explicit treatment,
not silently dropping frames.

Next: make scene/camera/video/calibration/GT inputs configurable and support
empty per-camera GT slots. Preserve the existing scene_001 reference behavior
before running scene_041 inference and frozen-policy comparison. Candidate
embeddings and local association remain causal; GT is used only offline for
metrics. Any later changes motivated by these validation observations count
as development and require a new untouched evaluation set.
