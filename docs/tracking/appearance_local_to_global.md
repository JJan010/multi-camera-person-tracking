# Frozen local-to-global appearance comparison

The local appearance experiment repaired the inspected camera-5/local-11
takeover, but only modestly improved pooled local IDF1. The next question is
whether this improves persistent cross-camera identities under the unchanged
downstream policy.

## Observed local results

Source experiment: `20261007T162104712584Z`, scene_001, cameras 4/5/8,
runtime frames 0..1799, evaluation frames 2..1799.

| Variant | Local IDF1 | Precision | Recall | IDSW | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 64.45% | 98.49% | 96.76% | 120 | 1456 | 3175 |
| direct_iou | 64.44% | 98.48% | 96.77% | 120 | 1461 | 3167 |
| direct_appearance | 65.56% | 98.47% | 96.76% | 123 | 1469 | 3169 |

Differences based on these rounded values are +1.11 percentage points versus
baseline and +1.12 points versus direct_iou. The number of rejected high-stage
candidate pairs (1614) is not a count of prevented identity switches.

## Protocol

1. Verify frozen reports, local outputs, candidate embeddings, calibration and
   the original global association implementation hashes.
2. Preserve the original full-pipeline global trace as the baseline.
3. For each direct variant, join accepted detection indices to cached vectors
   and the new local IDs. No spatial rematching is used for this join.
4. Causally replay an isolated appearance history and IdentityStage from frame
   zero, with the original geometry, grouping and controlled-merge settings.
5. Freeze the resulting global assignments before loading GT for evaluation.
6. Evaluate each variant using one shared identity assignment across all cameras
   and frames, same-camera/same-frame IoU >= 0.5, and the existing clipping policy.
   Check all three results against motmetrics and reproduce the original global
   metrics exactly. The original minute baseline is 71.75% global IDF1.

No detector, encoder, local tracker or video decoder is run in this experiment.
No runtime thresholds are selected or modified. The local appearance threshold
is 0.6; the downstream appearance threshold remains the original configuration.
The strong-observation gallery inside the experimental local tracker is distinct
from the downstream global history, which still uses all encoded tracked
observations according to its unchanged policy.

## Run in the project WSL terminal

```bash
cd /home/jakjan/projects/multi-camera-person-tracking
source .venv/bin/activate

PYTHONPATH="$PWD/src" python scripts/check_local_to_global_inputs.py

PYTHONPATH="$PWD/src" python scripts/evaluate_appearance_global.py \
  --local-report artifacts/appearance_bytetrack/20261007T162104712584Z/report.json \
  --baseline-evaluation artifacts/mtmc_sequence_evaluation/20261007T134209413468Z/report.json
```

This extension uses the scripts and modules already present in this project;
it is not a standalone package. Existing source artifacts must remain available
at their recorded paths and with their recorded checksums.

Outputs are placed in a new `artifacts/appearance_global/<run_id>/` directory:
`report.json`, `global_tracks.jsonl.gz`, `identity_matching.json`, and
`run_status.json`. Original experiments are not overwritten.

## Interpretation and limitations

- The principal controlled contrast is direct_appearance versus direct_iou:
  both share candidate features and output mapping, with identical global logic.
- Baseline versus either direct variant also includes the change from the old
  final output rematching to direct candidate mapping and small numerical
  differences in cached versus original embeddings.
- Every variant uses the same GT population, but its own prediction count.
  Local tracking changes can change the number of returned boxes.
- Report both global IDF1 and the lifecycle/merge diagnostics. More or fewer
  allocated identities alone does not establish a quality improvement.
- A merge whose current visible members share a GT identity does not prove that
  every past or retained member of that global identity is correct.
- This is the same reused development minute. Improvement here requires
  independent validation before adopting thresholds or claiming generalization.
- CPU replay duration is not end-to-end pipeline throughput. The cost of
  encoding all detector candidates must be measured when integrating a chosen
  variant into the full pipeline.

The input smoke test covers exact index joins (including duplicate boxes),
camera-scoped local IDs, missing descriptors, invalid provenance and lifecycle
accounting. A 300-round technical fixture with synthetic features/calibration
also exercised the complete replay/evaluation flow during development. Its
scores are not dataset performance results.

Reference: Ristani et al., *Performance Measures and a Data Set for Multi-Target,
Multi-Camera Tracking* (2016), https://arxiv.org/abs/1609.01775.
