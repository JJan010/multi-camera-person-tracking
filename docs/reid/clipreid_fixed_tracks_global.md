# CLIP-ReID: fixed-track global identity experiment

## Question

Does substituting CLIP-ReID mean descriptors for OSNet mean descriptors improve
global identity association on the same frozen observations and segment cuts?

This is a controlled cross-camera descriptor experiment. Local detection,
tracking, box geometry and appearance-continuity cuts retain their OSNet-derived
outputs. CLIP does not rerun or alter these upstream components.

## Declared policy

- One numerical threshold transfer: mean cosine similarity strictly above 0.7.
- The existing geometry gates, grouping and controlled-merge settings are reused.
- Both histories contain at most eight observations and have a one-second age
  limit. Stored CLIP means use the same sample participation and segment keys.
- Dormant identity recovery remains disabled.
- No threshold sweep, tuning or deployment model selection occurs.

The same cosine number does not imply the same false-match rate in two models.
A negative result would not establish that CLIP is an inferior encoder; a
positive result would not establish that 0.7 is a calibrated threshold.

## Run

The scripts depend on the dimension-explicit history and feature-identity bridge
already installed in the project. No new package installation is required.

```bash
PYTHONPATH="$PWD/src" python scripts/check_clipreid_global_experiment.py

PYTHONPATH="$PWD/src" python scripts/experiment_clipreid_global.py \
  --compatibility-report artifacts/feature_identity_checks/20261009T190727645206Z/report.json

PYTHONPATH="$PWD/src" python scripts/experiment_clipreid_global.py \
  --compatibility-report artifacts/feature_identity_checks/20261009T190915647970Z/report.json
```

The first full run is scene 001; the second is scene 041. Settings do not change
between them. Both scenes have already been inspected during development;
scene 041 is a transfer check, not an untouched final test set.

## Causality and verification

Phase 1 reads verified frozen local observations, segment mappings and features.
It replays two independent identity managers on CPU, without models, decoding or
GT input. Every OSNet full identity record must exactly reproduce the enabled
continuity reference. All observations, including unencoded outside boxes,
remain represented. New traces retain internal segment keys and a separate
projection to original observation keys for evaluation.

The prediction trace and its SHA-256 are frozen before GT loading. Phase 2
evaluates full frames 2..3599 and disjoint windows 2..1799 and 1800..3599. Each
scope uses one shared GT-to-global-ID assignment across all cameras. The runtime
state never resets at the window boundary. All camera/time slots are included,
including empty GT slots, using the scene's existing IoU and clipping protocol.

Both models' IDTP, IDFP, IDFN, IDF1, IDP and IDR are checked against motmetrics.
OSNet full/window metrics and lifecycle must reproduce the source report.
Predicted and GT observation counts must agree between variants. Merge labels
describe visible event members; they do not establish whole-identity purity.

Synthetic checks cover dimension compatibility, frozen segment cuts, empty
rounds, expiration, outside observations, identical-feature parity and a known
feature change that degrades cross-camera identity quality. They also reject
changed source-row provenance and invalid CLIP vectors. Padding in these checks
is confined to synthetic vectors; it is not used on actual model features.

## Outputs and limits

New runs are written under `artifacts/clipreid_global_experiment/<run>/`:

- `global_tracks.jsonl.gz`: common observations and both identity outputs;
- `prediction_freeze.json`: prediction checksum and declared settings;
- `identity_matching.json`: offline shared identity mappings per scope;
- `report.json`: metrics, lifecycle, inputs, checksums and limitations;
- `run_status.json`: explicit success/failure status.

The experiment does not change the runtime baseline. Its wall-clock duration is
not a model benchmark: both encoders have already produced their feature caches.
The result estimates a cross-camera descriptor substitution conditional on
OSNet-derived local tracking and segment cuts, not a full CLIP-based pipeline.
