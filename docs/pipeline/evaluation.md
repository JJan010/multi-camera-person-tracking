# Evaluate the full video-to-global-ID reference

This evaluation consumes the completed output of `scripts/run_mtmc.py`.
It runs on CPU, does not execute the models again, and never modifies runtime
predictions, embeddings, history or global identity assignments.

## Commands

From the project root in WSL with the project virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_mtmc_pipeline_evaluation.py

PYTHONPATH="$PWD/src" python scripts/evaluate_mtmc_pipeline.py \
  --run-report artifacts/mtmc_pipeline/20261007T131031285535Z/report.json
```

The evaluator currently accepts exactly 300 consecutive rounds of the pinned
scene_001, cameras 4/5/8, 30 FPS protocol. Longer recordings and other scenes
require an explicit extension of the evaluation/data manifests rather than
silently changing the meaning of the present metric.

## Observation and identity protocol

Validate all frames 0..299, including exact scene time, camera coverage, raw
track/crop-box consistency, consecutive embedding rows, identity scope and
settings. Join global outputs to local boxes using camera/local-ID/frame keys.
Require an assignment for every local prediction, including unencoded tracks.
An unencoded fully outside track must have a null embedding row and remain
in the prediction denominator; lack of a crop must not improve the score by
removing a false prediction.

Evaluate frames **2..299**, which have the audited GT coverage. Runtime global
identity state starts at frame 0. Timing warmup does not exclude predictions
from quality evaluation: frames 2..29 are included as usual. Earlier offline
identity experiments started their manager at frame 2; do not interpret their
86.02% as an exact paired reference for the new frame-0, freshly inferred run.

For each camera/frame slot, clip both GT and predicted boxes to 1920x1080.
Exclude zero-area GT boxes but retain every prediction, including zero-area
clipped boxes. Every same-camera/same-frame IoU >= 0.5 is an admissible spatial
candidate. There is no preliminary greedy or Hungarian per-frame GT labeling
for the identity metric.

Accumulate these candidates across all evaluated cameras and frames. Compute
one shared, one-to-one GT/global-ID assignment maximizing identity-correct
observations. This is offline evaluation, not a runtime identity decision.

Let G be evaluated GT observations, P all predicted observations and IDTP the
observations compatible with that shared identity mapping:

- IDFP = P - IDTP
- IDFN = G - IDTP
- IDP = IDTP / P
- IDR = IDTP / G
- IDF1 = 2 * IDTP / (G + P)

The evaluator reuses the already tested `IdentityCounts` implementation in
`evaluate_global_identity.py`. It also checks both reported variants against
the existing pinned `motmetrics==1.4.0` engine. The camera/time slot ordering is
used only for ID metrics, not for CLEAR ID switches, MOTA or temporal fragments.

## Same-run control

The no-cross-camera control uses **the same boxes and frames** with each
`(camera_id, local_id)` treated as a distinct predicted identity. It performs no
cross-camera linking, extra expiry or relabeling. Both variants must have the
same G, P and camera/frame slot counts.

The difference from this control measures the effect of global identity
assignment on the new frozen local predictions. It is not the difference from
the previous run's pooled per-camera score. A high per-camera IDF1 does not
establish consistent identity across cameras.

The pipeline report's `ever_emitted_global_ids` covers frames 0..299. The metric's
`predicted_identities` covers frames 2..299. These need not be equal and neither
is a count of distinct real people.

## Merge diagnostic

For each recorded merge, label visible members only where IoU matching is
mutually unique: one candidate GT for that prediction and one prediction for
that GT within the camera/frame. Classify the event as:

- `all_visible_members_same_gt`;
- `different_known_gt`;
- `unresolved`, when at least one member lacks a unique spatial label;
- `unannotated_frame`, for events before frame 2.

These labels never enter runtime or the identity metric. Agreement at the
acceptance frame does not prove that the entire merged trajectory is correct.

## Outputs and provenance

Each evaluation creates `artifacts/mtmc_pipeline_evaluation/<UTC-run-id>/`:

- `report.json`: metrics, same-run control, denominators, settings, source hashes,
  merge categories, runtime performance summary, evaluator versions and hashes;
- `comparison.csv`: control and full-pipeline metrics;
- `identity_matching.json`: the shared offline mappings;
- `merge_diagnostics.json`: individual recorded merge events with GT diagnostics.

All five pipeline artifacts are checksum-verified, along with the source report,
the scene source manifest and its pinned GT file. Their hashes are checked again
after evaluation. Embedding arrays must match the verified trace row count and
expected float32 `(N,512)` layout. Ground truth is opened only by this evaluator.

This remains a short, repeatedly inspected training-scene integration fragment.
No deployment threshold is selected. A longer sequence and an independent scene
are required before claiming generalization or final quality.

Reference: Ristani et al., *Performance Measures and a Data Set for Multi-Target,
Multi-Camera Tracking*, ECCV Workshops 2016, https://arxiv.org/abs/1609.01775.
