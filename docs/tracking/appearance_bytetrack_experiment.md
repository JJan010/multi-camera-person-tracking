# Controlled appearance-assisted local tracking experiment

## Reference

The user verified all 5400 camera updates of the original 1800-frame run:
96282 tracked observations reproduced exactly, including IDs, boxes, scores
and order. Gate report:
`artifacts/frozen_bytetrack_replay/20261007T160430502858Z/report.json`.
Candidate cache: `20261007T154414673309Z`, 113250 observations.

The L11 takeover demonstrates a potential failure mechanism: a high-score
detection of another person can take an existing track before its weaker
correct detection is considered. The local appearance change was visible in
OSNet features. This motivates an experiment, not a claimed general solution.

## Three quality variants and one exact reproduction requirement

| Variant | Internal association | Output mapping |
|---|---|---|
| baseline | Original ByteTrack | Original final IoU remapping |
| direct_iou | Original ByteTrack | Exact candidate selected internally |
| direct_appearance | Original costs plus appearance eligibility mask | Exact candidate selected internally |

All variants use an isolated source adaptation under
`src/mtmc/tracking/_vendor/experimental_bytetrack.py`. The original project
adapter and installed package remain unchanged. Before evaluation, every
baseline camera/frame output must reproduce the frozen original exactly.

The direct-output control separates output-provenance effects from appearance
effects. Returned boxes and counts may differ between baseline and direct
output. Therefore, precision, recall, FP and FN are evaluated as well as IDF1
and ID switches. Detection candidates are fixed, but output boxes are not
assumed identical across variants.

## Experimental policy

- Retain original Kalman filter, score thresholds, matching solver, score-fused
  IoU costs, lifecycle and duplicate-track removal from supervision 0.30.7.
- Before high-score and unconfirmed-track assignment, consider pairs already
  eligible under the original cost limit. If both descriptors are available,
  require cosine >= the experimental threshold (default 0.6).
- Reject incompatible pairs before assignment by setting their cost to
  infinity. The original solver internally clips over-limit costs and rejects
  over-limit assignments afterwards. This experiment preserves that solver;
  it does not claim a new globally optimal partial-assignment objective.
- Preserve original costs for all permitted pairs. No appearance-based cost
  ranking or weighted appearance/motion fusion is added.
- The low-score second stage remains IoU-only. A track rejected by the first
  stage can still be recovered there.
- Keep at most eight accepted high-score appearance samples per internal
  track, with an inclusive maximum age of 30 scene frames. Compute their
  normalized float64 mean before the current association. Near cancellation
  falls back to the last eligible sample.
- New-track activation initializes appearance history. Low-score updates do
  not update or refresh its samples. Missing or expired appearance explicitly
  falls back to the original motion/score cost; no fabricated vector is used.
- Each accepted observation propagates its exact candidate index through
  activation, update and reactivation. Direct output returns original candidate
  boxes and scores in detector order, without another IoU remapping.

Only `update_candidates()` is the supported entry point for the experimental
class. Inputs are validated before mutation; an internal runtime failure
marks the instance failed and requires a fresh instance. Each camera/variant
has an independent tracker. No global IDs are assigned in this experiment.

The threshold 0.6 is an explicit development setting for the initial paired
experiment. It is not a probability, a calibrated deployment threshold, or
evidence of generalization. Neither the fixed-camera cross-view threshold nor
one inspected local event establishes a suitable local appearance threshold.

## Run

From the project root in the existing virtual environment:

```bash
PYTHONPATH="$PWD/src" python scripts/check_appearance_bytetrack.py

PYTHONPATH="$PWD/src" python scripts/experiment_appearance_bytetrack.py \
  --replay-report artifacts/frozen_bytetrack_replay/20261007T160430502858Z/report.json \
  --appearance-threshold 0.6
```

No dependency installation, video decoding or model execution is required.
The whole experiment runs on CPU using the candidate cache.

Phase 1 processes all frames 0..1799 causally and writes all three variants to
`tracks.jsonl.gz`. GT is not loaded or passed to the trackers. Direct variants
include detection indices and cache embedding rows for downstream replay.

Phase 2 loads pinned GT and evaluates the now-frozen outputs on frames
2..1799, including predictions from timing warmup. It uses the existing clipped
box/IoU >= 0.5 convention and motmetrics 1.4.0. IDF1 pools independently
matched per-camera identity counts. It is LOCAL IDF1; compare the baseline
with the previous approximately 64.45%, not the global 71.75%.

Results under `artifacts/appearance_bytetrack/<timestamp>/` include the frozen
trace, `report.json`, `case_tracks.json`, and completion status. The case file
labels selected camera-5 observations near the inspected transition using
mutually unique GT overlap, only after replay. It does not guide runtime
decisions or represent a general evaluation sample.

## Verification and limits

Synthetic checks cover inclusive cosine gating, motion-cost preservation,
missing appearance, weak-detection recovery, exact selected-candidate output,
the next frame's identity continuity, history aging, malformed inputs and
baseline fork parity. In the authoring environment these were run with the
uploaded ByteTrack mathematical source and lightweight Detections/IoU/import
shims because the installed supervision package was unavailable. The user-run
checks exercise the actual installed package.

The disabled fork reproduced 900 camera updates from an uploaded real
300-frame trace in that technical harness. A full 300-frame runner/evaluator
test used constant synthetic features and real early boxes/GT: direct output
and appearance output matched exactly when every feature pair was admissible.
These checks validate implementation behavior, not OSNet tracking quality.

Remaining limitations include unreliable occluded crops, appearance-history
contamination, appearance changes caused by pose, fallback when history is
unavailable, motion-only low-score errors, and the original lifecycle/solver
tradeoffs. This is a small ByteTrack-based experimental variant, not a full
Deep SORT or BoT-SORT reproduction. Improvement on the development minute
must be followed by downstream global-ID evaluation and separate validation.

## Attribution

The adapted core originates from the user-supplied supervision 0.30.7 file,
SHA256 `018911df3dd600c7a0abbd059138b116185a458f2b4877b4c1fd32f5c341c90d`.
Copyright (c) 2022 Roboflow, MIT License, preserved in
`src/mtmc/tracking/_vendor/LICENSE.supervision`.
The installed supervision Kalman filter, STrack, matching and utility helpers
are reused. The copied core has explicit candidate-provenance and appearance
hooks; there is no runtime monkey-patching of the installed package.

References:

- Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every Detection
  Box*, ECCV 2022, https://arxiv.org/abs/2110.06864.
- Wojke et al., *Simple Online and Realtime Tracking with a Deep Association
  Metric*, ICIP 2017, https://arxiv.org/abs/1703.07402.
- Supervision license: https://github.com/roboflow/supervision/blob/develop/LICENSE.md.
