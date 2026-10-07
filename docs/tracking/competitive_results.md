# Competitive association: mixed local/global development result

## Provenance

- Dataset: `nvidia/PhysicalAI-SmartSpaces`.
- Revision: `2cbe9563cbe9f47f846e5c871ee994572bbbc60e`.
- Scene: `MTMC_Tracking_2024/train/scene_001`, cameras 4/5/8.
- Continuous runtime: frames 0..3599; evaluation: frames 2..3599.
- Frozen local experiment: `artifacts/competitive_bytetrack/20261007T184813424022Z/report.json`.
- Frozen global experiment: `artifacts/competitive_global/20261007T190139714061Z/report.json`.
- Staged global reference: `artifacts/appearance_global/20261007T174247370251Z/report.json`, variant `direct_appearance`.
- Candidate appearance probe: `artifacts/candidate_appearance_probe/20261007T183937481045Z/report.json`.

Companion files `competitive_local_metrics.json` and `competitive_global_metrics.json`
must be copied from the corresponding full reports, not reconstructed from the
rounded tables below. The large traces, embeddings and recordings remain local
artifacts; do not add them to Git.

## Paired results

| Variant | Local IDF1 | Global IDF1 | Local IDSW | Local FP | Local FN |
|---|---:|---:|---:|---:|---:|
| staged | 44.86% | 53.03% | 324 | 3105 | 6851 |
| competitive_iou | 43.30% | 55.17% | 441 | 3359 | 6650 |
| competitive_appearance | 43.05% | 54.65% | 446 | 3642 | 6709 |

| Variant | Global IDTP | Global IDFP | Global IDFN | Global IDP | Global IDR |
|---|---:|---:|---:|---:|---:|
| staged | 96696 | 83765 | 87511 | 53.58% | 52.49% |
| competitive_iou | 100725 | 80191 | 83482 | 55.68% | 54.68% |
| competitive_appearance | 99835 | 81305 | 84372 | 55.11% | 54.20% |

The IoU-ranked refinement gains about 2.14 global IDF1 percentage points and
4029 identity-correct observations relative to staged, while local IDF1 falls
about 1.56 points. The appearance-ranked refinement gains about 1.62 global
points and 3139 identity-correct observations, while local IDF1 falls about
1.81 points. Percentage values are rounded; full reports are authoritative.

Evaluation GT observations: 184207 for every variant. Prediction observations
are 180461 / 180916 / 181140 respectively. These differ from runtime totals
180547 / 181002 / 181226 because frames 0 and 1 are not evaluated. Each metric
uses its own verified prediction denominator. Local FP/FN are not global IDFP/IDFN.

## Identity lifecycle and diagnostic limits

| Variant | Allocated IDs | Absorbed IDs | Expired IDs | Retained at end | Merge events |
|---|---:|---:|---:|---:|---:|
| staged | 267 | 155 | 84 | 28 | 151 |
| competitive_iou | 354 | 190 | 136 | 28 | 181 |
| competitive_appearance | 366 | 191 | 148 | 27 | 187 |

Visible-member merge evidence was same-GT / unresolved: 148/3, 173/8, 179/8.
No merge was labeled different-known-GT under this diagnostic. This does not
prove that every merge or the whole retained identity is correct. The lifecycle
counts indicate increased identity turnover; allocated IDs are not person counts.

Both the disabled local refinement and the staged local replay reproduced the
frozen appearance baseline exactly. The global staged replay reproduced every
reference decision and retained state, allowing only the explicit run-scope
change. Metrics, lifecycle, merge diagnostics and local/global denominators were
verified, and all three global evaluations agreed with motmetrics.

## Interpretation

The earlier local-only assessment remains correct: local tracking regressed.
The complete MTMC assessment is different: both refinements improved the final
shared identity score on this development sequence. Global IDF1 is not an average
of per-camera IDF1, and a downstream identity manager can combine distinct local
fragments or propagate errors differently after changes to local associations.

One possible mechanism is that shorter/reallocated local tracks limit the time
for which an incorrect local anchor contaminates a global identity. Another is
that changed boxes/features alter grouping and merges. These are hypotheses,
not causal conclusions established by aggregate scores. More ID switches do not
in themselves cause improved global identity quality.

The diagnostic case initially motivating the change is not the acceptance
criterion. Local/global numeric IDs are scoped independently in each variant;
the same number cannot be treated as a cross-variant person correspondence.

## Decision and next gate

1. Keep the existing full runtime unchanged and preserve `staged` as the
   experimental reference. This result is neither an unconditional rejection
   nor a deployment promotion of competitive association.
2. Prioritize `competitive_iou` as the candidate for a frozen-policy validation
   comparison against `staged`. On this sequence it exceeds the appearance-ranked
   variant in both local and global IDF1, with fewer local FP, FN and ID switches.
3. The name `competitive_iou` refers only to refinement ranking. The variant still
   uses OSNet, the existing appearance threshold and appearance eligibility gates.
4. Preserve `competitive_appearance` as an ablation result, not the preferred
   candidate. The selected few frames did not predict which full-run ranking
   policy would work best.
5. Freeze current policy and thresholds before evaluating another scene. Select
   data based on declared format/calibration/coverage requirements, not favorable
   tracking scores. Adapt scene paths, camera IDs and evaluation metadata explicitly.
6. Report global IDF1 as the main MTMC objective alongside local IDF1, local IDSW,
   FP/FN, global lifecycle, runtime and memory. Do not accept an apparent global
   improvement while hiding a local regression. Any tolerance for this tradeoff
   must be defined before inspecting validation results.

The entire inspected two-minute scene is now development data. An uninspected
scene provides a stronger generalization check than a nearby interval with the
same people/cameras. A final untouched test should remain separate. No statistical
significance, calibrated deployment threshold, real-world transfer, or FPS gain
is claimed here.

After validation and policy selection, integrate the selected variant through
the detector-candidate embedding path and verify end-to-end parity. Then optimize
OSNet/detector execution (ONNX/TensorRT), preprocessing and decoding while checking
quality against the frozen reference. Both models already run on CUDA FP32 in
the reference pipeline; the remaining work concerns backend/transfer/decoder
optimization, not initial GPU enablement.

## Reference

Ristani et al., *Performance Measures and a Data Set for Multi-Target,
Multi-Camera Tracking*, ECCV Workshops 2016: https://arxiv.org/abs/1609.01775.
Identity metrics assess identity correspondence over observations, which is why
the evaluation scope and shared cross-camera matching must be explicit.
