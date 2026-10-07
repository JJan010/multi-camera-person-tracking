# Shared multi-camera identity evaluation

Protocol: `scene_001_global_2d_identity_v1`.
Scope: scene_001, cameras 4/5/8, frames 2..299, IoU >= 0.5.
This is a project integration protocol, not official AI City evaluation.

## Run

From the repository root, with the project virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_global_identity_evaluation.py
PYTHONPATH="$PWD/src" python scripts/evaluate_global_identity.py \
  --identity-report artifacts/global_identity/20261007T093510175809Z/report.json
```

Uses existing numpy/scipy and motmetrics 1.4.0 for independent parity verification.
No detector, tracker, embedding model or identity manager is rerun. No new project
dependency is introduced.

## Inputs and provenance

The evaluator hashes the identity report and verifies its compressed assignment
trace. It also verifies the earlier tracked-box trace and original GT file against
their inherited hashes before using them. These two paths are read from the
identity report, so the recorded files must still exist at those locations.

Each output assignment must match a source observation key and embedding row.
Source Re-ID boxes must equal the saved tracker boxes. The current integration
protocol requires a global assignment for every local prediction; any missing
assignment causes failure, rather than silently excluding it from evaluation.
Previously saved `diagnostic_gt_id` labels are not used to calculate ID metrics.

All settings must use the same GT and prediction observation counts. All cameras
and all expected frames must be present. Input hashes are checked again after
computation. Images and embeddings do not need to be loaded.

## Spatial gate and counting unit

An observation means a box in one camera at one frame. A person visible in three
cameras contributes three GT observations at that scene time. Coordinates from
different cameras are never compared.

Both GT and prediction boxes are clipped to 1920x1080. Zero-area GT boxes after
clipping are excluded. Every prediction is retained in its denominator, including
zero-area clipped predictions, which have no valid spatial overlap. This follows
the earlier local box policy.

For each camera/frame, all GT/prediction pairs with IoU >= 0.5 are admissible.
There is no preliminary one-to-one assignment of boxes to fixed GT labels. This
preserves alternatives in ambiguous overlaps for the whole-trajectory identity
assignment. An identity must occur at most once per camera/frame, while it may
legitimately occur in several cameras at the same scene time.

## One identity assignment across cameras and time

For each setting, accumulate admissible overlap counts for every GT ID and
predicted global ID over the entire evaluated fragment. Find a single one-to-one
identity mapping maximizing the supported observation count, IDTP. Then:

- IDFN = number of GT observations - IDTP;
- IDFP = number of prediction observations - IDTP;
- IDP = IDTP / prediction observations;
- IDR = IDTP / GT observations;
- IDF1 = 2 * IDTP / (GT observations + prediction observations).

Undefined ratios are represented by null. The optimal mapping is an offline
scoring operation only. It neither modifies runtime identities nor feeds future
information to the tracker.

The implementation uses a nonnegative overlap-count matrix and maximizes IDTP
with SciPy assignment. Since GT and prediction totals are constant, maximizing
IDTP is equivalent to minimizing IDFN + IDFP. Zero-support assigned pairs are
omitted from the exported matching; unassigned identities keep their full error
counts. Equally optimal matchings may differ while yielding identical metrics.

Known-answer tests and ambiguous random examples are checked against motmetrics
1.4.0. The complete no-cross-camera control is also compared with motmetrics on
every evaluation run. For this reference, each camera/frame is a separate
observation slot, but GT and predicted ID namespaces remain shared across cameras.
Only identity metrics are requested. CLEAR ID switches, MOTA and temporal
fragmentation are not calculated from this artificial slot ordering.

## Why local IDF1 is insufficient

If one person appears equally often in two cameras and receives different IDs in
each, each camera can score 100% when evaluated independently. One shared global
identity matching can match the GT person to only one predicted identity; global
IDF1 is 50% in this example. The earlier pooled local-camera result is therefore
not the right direct baseline for global identity quality.

The evaluator creates a separate control: retain each frozen `(camera, local_id)`
as its own identity over the clip, without any cross-camera association or new
expiry logic. Evaluate that control with exactly the same shared global protocol
and the same boxes/GT. `delta_idf1_vs_no_cross_camera_pp` compares each manager
setting with this control.

## Outputs and interpretation

`artifacts/global_identity_evaluation/<UTC run ID>/` contains:

- `report.json`: protocol, source/code hashes, GT exclusions, control and all results;
- `summary.csv`: global IDTP/IDFP/IDFN, IDP/IDR/IDF1 and control differences;
- `identity_matching.csv`: positively supported whole-clip GT/global-ID pairings.

Lower IDF1 can reflect missed detections, false observations, localization errors,
identity fragmentation or incorrect identity merging. IDFP is not simply the
number of false detector boxes. No threshold is automatically selected.

This evaluates only a short, reused training-scene fragment with registries that
started empty at frame 2. It establishes an integration baseline, not generalization
to new scenes. The no-merge registry may preserve both fragmentation and incorrect
initial groupings. Further changes must retain this baseline for comparison.

## Sources

- Ristani et al., *Performance Measures and a Data Set for Multi-Target,
  Multi-Camera Tracking* (2016), Section 3: https://arxiv.org/abs/1609.01775.
- TrackEval identity metric implementation:
  https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/metrics/identity.py.
- Independent comparison engine: motmetrics 1.4.0,
  https://github.com/cheind/py-motmetrics.

The overlap gate and evaluated scene fragment are project-specific. The evaluator
is not a copied official challenge submission/evaluation wrapper.
