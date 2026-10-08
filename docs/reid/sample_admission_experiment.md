# Frozen appearance-sample admission audit

Status: one development hypothesis; no runtime integration or quality claim.

The retired-reference visual review exposed background-dominated crops near
an image edge. High detection confidence alone did not guarantee a useful
person descriptor. This audit measures the coverage cost of two simple
sample-admission rules before introducing them into appearance memory.

## Fixed comparison

| Policy | Admission |
| --- | --- |
| all_available | A nonempty image crop exists. |
| confidence_only | Available crop and confidence >= 0.5. |
| confidence_and_border | Previous rule and normalized raw-box clearance >= 0.01. |

For raw box (x1,y1,x2,y2), image width W and height H:

`clearance = min(x1/W, y1/H, (W-x2)/W, (H-y2)/H)`.

A 1% margin means 19.2 horizontal pixels and 10.8 vertical pixels at
1920 x 1080. Boundaries are inclusive. Raw coordinates are used before
clipping: a box can be entirely inside the image while still lying very
close to its edge. The 1% margin is an explicit experimental choice, not a
published recommendation or calibrated operating threshold.

This is not a visibility, occlusion or background classifier. Useful partial
people may be rejected, and poor crops away from borders may still pass.
Detection confidence is not a calibrated probability of Re-ID suitability.

## Inputs and separation from evaluation

Run the same audit on the completed enabled continuity outputs for scene_001
and scene_041. The script verifies the source report, scene configuration,
trace and GT checksums. It freezes admission masks for all runtime rounds
before loading GT. It then evaluates the original evaluation interval with
all original predicted boxes still participating in the IoU graph.

GT is diagnostic only. Rejecting a sample never removes its tracked box from
this graph or makes a formerly ambiguous competitor become uniquely matched.
Original local keys and segmented identity keys are both recorded.

The spatial categories are mutually exclusive:

- mutually_unique_gt: exactly one admissible GT/prediction partner each;
- no_admissible_gt: no IoU candidate satisfies the fixed evaluation gate;
- ambiguous_gt: one or more candidates exist without mutual uniqueness.

These categories are not crop-quality labels. In particular, lack of an
admissible GT match does not establish that a crop contains only background.
The report includes accepted and rejected counts per category and camera,
segment coverage, per-person unique evidence, and the inspected camera-361
examples when present. Segment coverage is over the evaluation interval; it
is not a simulation of descriptor availability, aging or recovery success.

Both scenes have informed development of this hypothesis. Scene_041 is not
an independent validation set for this new rule. A later acceptance decision
requires new held-out evidence.

## Running in WSL

From `/home/jakjan/projects/multi-camera-person-tracking`, with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_sample_admission.py
PYTHONPATH="$PWD/src" python scripts/check_sample_admission_audit.py

PYTHONPATH="$PWD/src" python scripts/audit_sample_admission.py \
  --source-report artifacts/appearance_continuity_experiment/20261008T105229140571Z/report.json

PYTHONPATH="$PWD/src" python scripts/audit_sample_admission.py \
  --source-report artifacts/appearance_continuity_validation/20261008T111045386342Z/report.json
```

Outputs: `artifacts/sample_admission/<run_id>/report.json` and
`admission.jsonl.gz`. No decoder, detector, encoder or tracker runs.

## Interpretation and next decision

Compare suppression of the known background examples with loss of uniquely
matched samples and segments left without any accepted sample. Do not infer
an IDF1 improvement from admission counts. If the trade-off justifies a
follow-up, integrate admission before appearance aggregation in an isolated
experiment, retain original sample timestamps when reusing old evidence,
and compare global identity quality and availability against the frozen
control. Do not refresh old evidence merely because a new sample was rejected.

ByteTrack deliberately uses low-score detections to recover tracks. Sample
admission to Re-ID memory is a separate decision from retaining a detection
for tracking. Reference: Zhang et al., *ByteTrack: Multi-Object Tracking by
Associating Every Detection Box*, https://arxiv.org/abs/2110.06864.

## Technical verification

Synthetic checks cover inclusive boundaries, all four edges, resolution
scaling, outside crops, ID zero, immutable records and malformed inputs.
Known-answer audit checks cover valid border samples being rejected, unchanged
ambiguous competitors, empty GT slots and exact category accounting.
A local 120-round integration fixture was also exercised with stable and
changing descriptors, empty cameras/rounds and segmented keys. These checks
establish technical behavior; actual scene results must come from WSL.
