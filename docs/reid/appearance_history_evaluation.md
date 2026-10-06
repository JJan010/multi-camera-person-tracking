# Latest versus mean appearance: paired retrieval diagnostic

This experiment measures whether the causal mean descriptor improves cross-camera
retrieval on the frozen local tracking + Re-ID run. It uses saved arrays and
trace metadata on CPU. It does not decode videos, execute models, change local
IDs, or assign global IDs. No additional packages are required.

## Fixed comparison

- Scene: `MTMC_Tracking_2024/train/scene_001`, cameras 4, 5, 8.
- Retrieve on video frames 2 through 299, at 30 FPS, in all six camera directions.
- `latest`: current source embedding for both query and gallery.
- `mean`: saved L2-normalized causal mean for both query and gallery.
- Preserve the history policy recorded in the input report: initially up to eight
  observations, age at most one scene second, including the current observation.
- Both variants have identical boxes, queries, candidate galleries and GT labels.
  There is no similarity threshold and no crop-quality filtering.

The history mean is `normalize(sum(z_i) / n)` for normalized observation vectors
`z_i`. Averaging may reduce variation within a correctly tracked person. A local
ID switch can instead mix descriptors of different people. Improvement is a
hypothesis to measure, not a property implied by smoothing or normalization.

## Labels, eligibility and counting

Assign GT once per camera/frame using the existing snapshot helpers: clip boxes,
exclude zero-area GT, accept IoU >= 0.5, maximize valid assignment count before
maximizing total IoU. Appearance does not influence label assignment. All encoded
observations remain in the gallery, including observations without a GT match.

A directed query is eligible when its assigned GT identity has an assigned
observation in the target camera at the same time. Queries with missing labels
or no positive gallery observation are counted separately. They do not enter
the Rank-1 denominator. Rank-1 is the fraction whose positive is ranked first;
Rank-3 is the fraction whose positive is among the first three candidates.
The one-to-one GT assignment permits at most one positive per target camera.

Cosine similarity uses the saved normalized vectors and float64 accumulation.
Exact score ties use original source row order, independently of GT. The older
single-frame diagnostic used float32 accumulation; this report is its own paired
experiment and should not be subtracted directly from the old snapshot result.

Pool hit counts and eligible query counts; do not average camera percentages.
Report Rank-1 improvements, regressions, both-correct and both-wrong queries on
the same denominator. The percentage-point change equals
`100 * (improved - worsened) / eligible_queries`.

Frames 0 and 1 lack GT in this dataset. They are excluded from retrieval but may
remain in early histories. Their history labels stay unknown; they are never
silently treated as matching the current person.

## History-label diagnostic

For each current observation, inspect the GT labels of exactly the source rows
recorded by the history producer. Classify it as:

- `current_unmatched`: current observation has no assigned GT.
- `conflict_with_current`: at least one known member has a different assigned GT.
- `unknown_members`: no known conflict, but at least one member lacks a label.
- `consistent_with_current`: every member has the same assigned GT as the current observation.

These are diagnostic IoU label relationships, not authoritative purity labels or
motmetrics IDSW events. They never reset history or remove observations. Report
conditional retrieval by query/positive history status separately from the main
unfiltered result. Distractor histories can also affect ranks.

## Reproducibility and outputs

The evaluator checks source and history checksums, normalized matrix shapes,
observation keys, exact history window membership, and reconstructed means. It
checks the GT checksum against `configs/datasets/scene_001_source.json`. Existing
`evaluate_reid_snapshot.py` and `preview_appearance_history.py` are imported as
helpers; their hashes are recorded. Each evaluation creates a new directory:

`artifacts/appearance_history_evaluation/<UTC>/`

- `report.json`: protocol, provenance, pooled and directed metrics, history strata.
- `rankings.csv`: paired query-level results, including excluded-query reasons.
- `by_frame.csv`: paired counts and Rank-1 per frame.
- `history_labels.csv`: every source observation and the labels of its history members.
- `gt_matching.csv`: matching coverage for each camera/frame.

The fixed camera-8 examples at frames 210, 213, 214, 217 and 221 remain in the
report to inspect local IDs 10 and 12 around the previously diagnosed swap.
They do not select which frames contribute to the main result.

## Interpretation limits and references

This is a diagnostic on one short integration clip. Adjacent observations and
camera directions are correlated, so query counts are not independent people or
evidence of statistical significance. It is not a validation/test benchmark,
global IDF1 evaluation, or evaluation of rejecting people absent from the gallery.
Do not tune the history window or an association threshold to maximize this clip.

Wojke, Bewley and Paulus, *Simple Online and Realtime Tracking with a Deep
Association Metric* (2017), motivates using learned appearance for association:
https://arxiv.org/abs/1703.07402
Deep SORT stores an appearance gallery and uses nearest-neighbor distances; this
project's normalized eight-observation mean is a separate experiment, not its algorithm.
https://github.com/nwojke/deep_sort/blob/master/deep_sort/nn_matching.py

Torchreid's official rank evaluation source illustrates CMC counting and excluding
queries without a positive gallery sample. Our per-frame gallery and IoU-derived
labels define a custom integration protocol, not Market-1501 evaluation:
https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/metrics/rank.html
