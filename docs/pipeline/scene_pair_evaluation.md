# Paired evaluation on a configured scene

The scene_041 candidate cache completed 3600 rounds / 10800 images, recording
119933 candidates and 119933 normalized embeddings, with zero wholly outside
candidates. Camera counts: 361 = 22276, 362 = 50113, 364 = 47544.
Source report: `artifacts/scene_candidates/20261007T212800237556Z/report.json`.
These are candidate observations over time, not unique people or accepted tracks.
Successful cache construction establishes input integrity, not tracking accuracy.

## Experiment boundary

`evaluate_scene_pair.py` verifies the cache, scene configuration and frozen paired
policy. It checks runtime code against the passed scene_001 parity gate and
keeps the selected package versions and settings. It uses two distinct phases:

1. **Causal replay without GT.** Both variants receive every frame and candidate,
   from frame zero. Each owns independent local trackers, appearance histories,
   geometric associations and global identity state. A missing detection or
   all-empty round is still processed. No runtime state is reset at one minute.
2. **Offline evaluation.** After closing the compressed result trace, its SHA-256
   and configuration are written to `predictions_frozen.json`. Only then is GT
   loaded. Evaluation never changes local/global predictions or earlier IDs.

This is CPU replay of previously computed model outputs. It does not decode
videos, run RF-DETR/OSNet or measure end-to-end system throughput.

The candidate-to-runtime adapter uses recorded detection indices and embedding
rows. It validates every row against the scene-aware cache contract before
updating trackers. It does not use IoU to guess which embedding belongs to an
accepted detection. The previously checked `PairedAssociation` module remains
unchanged.

## Three different identity measurements

| Measurement | ID scope and assignment |
| --- | --- |
| Local pooled IDF1 | Independent GT/local-ID assignment per camera; sum identity counts across cameras |
| Global IDF1 | One shared GT/global-ID assignment over all evaluated camera/time slots |
| No-cross-camera control | Give every `(camera, local_id)` a distinct identity, then evaluate globally |

The control is built independently for each variant from its own frozen local
outputs. The two variants share input candidates, GT and protocol, but may emit
different boxes, numbers of observations and local IDs. No forced equal prediction
denominator or numeric-ID correspondence is assumed between variants.

The global evaluator retains all admissible same-camera/same-frame IoU pairs
before whole-identity assignment. It uses the existing `IdentityCounts` engine
and verifies both global and control ID metrics against motmetrics 1.4.0.
Local metrics use separate motmetrics accumulators per camera, actual scene frame
indices and `1 - IoU` costs. Binary admissibility costs are not substituted for
local costs because that could change temporal matching and ID-switch counts.

For each variant, GT count, prediction count and camera/frame-slot count must
agree between local, global and control evaluation. The shared GT denominator
must also agree across variants.

## Spatial and empty-slot policy

Evaluation uses the scene configuration's interval, explicit GT/video frame
mapping, image dimensions and inclusive IoU gate. For the current scene this is
frames 2..3599 and IoU >= 0.5. Runtime frames 0 and 1 still initialize state.

Both GT and prediction boxes are clipped to the corresponding image. Zero-area
GT after clipping is excluded; every prediction is retained, including wholly
outside predictions. GT identity zero is an ordinary valid identity.

Every camera/time slot in the interval remains in evaluation. If its supplied GT
is empty, predictions count as false observations under this protocol. The report
separately records empty slots and predictions in them per camera. For scene_041,
the audited raw empty-slot counts are 21 / 320 / 0 for cameras 361 / 362 / 364.
Sparse visual checks do not establish annotation completeness throughout every
empty interval; this limitation is not resolved by dropping those intervals.

## Merge diagnostics and outputs

Accepted merge members are labeled only offline, using mutually unique IoU
overlaps. Categories are same known GT, different known GT, unresolved, and
outside the evaluation interval. These diagnostic labels do not drive tracking.
Same-GT visible members do not establish purity of all retained identity members
or future observations.

Outputs live in a fresh `artifacts/scene_pair_evaluation/<run>/` directory:

- `paired_tracks.jsonl.gz`: both variants' local observations, global assignments,
  decisions, retained state and refinement events for every runtime frame.
- `predictions_frozen.json`: prediction checksum, fixed configuration and replay
  counters, written before GT access.
- `identity_matching.json`: offline optimal GT/global and GT/control mappings.
- `report.json`: per-camera/pooled local metrics, global/control metrics, lifecycle,
  merge diagnostics, empty-slot counts, input fingerprints and limits.
- `run_status.json`: completion/failure status.

## Verification and limits

The synthetic checker covers a known perfect shared identity (local/global 100%
but control 50%), GT identity zero, inclusive IoU boundary, an empty-GT false
observation, wholly outside prediction, excluded clipped GT, all-empty sequence,
mixed scopes and missing assignments. It also replays a real pair of tracker
instances through an empty round, freezes outputs and checks a known metric
result. These are integration tests, not estimates of scene_041 quality.

Global identity measurement follows the motivation of Ristani et al.,
*Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking*,
ECCV Workshops 2016: https://arxiv.org/abs/1609.01775 . The project's 2D IoU
protocol is not the official AI City world-coordinate HOTA evaluation.

No threshold or winning policy is automatically selected. A single selected
validation scene, three overlapping cameras and a two-minute interval do not
establish general performance on all camera arrangements. Any subsequent tuning
in response to these results makes this scene development evidence; a final
generalization claim needs separate untouched data.
