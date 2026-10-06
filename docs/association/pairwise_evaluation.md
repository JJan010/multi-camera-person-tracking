# Frozen pairwise association: diagnostic threshold sweep

Replay scene_001 frames 2..299, cameras 4/5/8, using the saved latest and causal
mean OSNet embeddings. No models, videos, CUDA, new packages, or tracker reruns
are required. The underlying source trace and appearance-history artifacts are
verified by checksum, window-membership and mean-reconstruction checks.

## Experiment scope

The fixed exploratory grid is stored in `configs/association/pairwise_diagnostic.json`:
`0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0`. These values were specified before
inspecting this sweep's results. They are diagnostic points, not calibrated
operating thresholds. Similarity must be strictly above the threshold. The 1.0
case rejects all links and checks the zero-acceptance accounting.

Both variants use identical observations and GT labels. Evaluate unordered camera
pairs (4,5), (4,8), (5,8) once per frame, since the matching is symmetric. Earlier
retrieval evaluated six directed queries; its counts must not be compared directly
with these undirected link counts. GT is assigned once using the existing clipped
IoU >= 0.5 protocol, independently of appearance, thresholds and association.

The actual `associate_camera_pair` API is called for every pair/threshold/frame.
It receives only camera envelopes, keys, timestamps and descriptor arrays. Evaluation
reads GT after the decision. No threshold is selected automatically, no runtime
policy is changed, and no global IDs are produced.

## Pair metrics and denominators

- `correct_links`: accepted endpoints have equal assigned GT IDs.
- `wrong_known_links`: accepted endpoints have different assigned GT IDs.
- `unresolved_gt_links`: at least one accepted endpoint lacks an assigned GT ID.
  These are retained and reported, not silently treated as correct or wrong.
- `precision_labeled`: correct / (correct + wrong_known). Always inspect the
  unresolved count and accepted-link label coverage alongside this metric.
- `verified_correct_fraction_all_links`: correct / all accepted links, including
  unresolved links in the denominator. This is a conservative diagnostic fraction,
  not a probability calibrated against perfect ground-truth assignments.
- `available_positive_pairs`: number of shared assigned GT identities in the
  two current observation sets, summed across camera pairs and frames.
- `recall_available_pairs`: correct / available positive pairs.
- `missed_positive_pairs`: available positive pairs minus correct links. A missing
  pair can result from abstention or an incorrect association to another endpoint.

Recall is conditional on both observations existing and receiving GT labels.
It does not include people completely missed upstream by detection/tracking.
This is not the recall of the complete MTMC system.

Also count labeled endpoints with no assigned positive in the opposite camera:
`no_positive_endpoints`, partitioned into `no_positive_linked` and
`no_positive_abstained`. Their ratio reports abstention when no assigned positive
exists. Absence of an assigned positive does not establish physical absence of
the person: a missed or unlabelled observation can also cause it.

The count identity `2 * accepted_links + unmatched_endpoints == endpoints` is
checked per pair. Undefined ratios are JSON null / CSV empty, not 100%.
Counts are pooled before ratios, and the positive-pair denominator must be the
same for every threshold and both descriptor variants. For this frozen run the
expected denominator is 12,031, half of the prior 24,062 directed retrieval queries.

## Three-camera consistency audit

Build an undirected graph of the accepted pair links in each frame, with nodes
identified by (camera, local ID, frame). Inspect connected components without
assigning a global ID to them:

- `camera_conflict_components`: a component contains multiple observations from
  one camera. Naively merging it would violate the one-person-per-camera constraint,
  assuming duplicate detections/tracks are handled upstream.
- `open_three_camera_components`: exactly three nodes from different cameras,
  with two of the three possible pair links. The third pair has not confirmed the
  chain; its absence alone does not establish that the people differ.
- `closed_three_camera_components`: exactly three nodes from different cameras,
  with all three links present. This establishes pairwise agreement, not correctness.

The audit also reports frames containing conflicts, components with different
known GT labels, and components with unknown labels. These last two are evaluation
diagnostics only. No component is accepted, rejected, split or merged using GT.
Examples retain the first five frames with structural issues for each variant and
threshold; counts cover all frames and have no example-based selection.

## Outputs and reproducibility

New output directory: `artifacts/pairwise_association/<UTC>/`.

- `report.json`: protocol, full summary and per-pair counts, component examples,
  source/code/config hashes, NumPy/SciPy versions, explicit null selected threshold.
- `summary.csv`: all-camera results for each variant/threshold.
- `by_pair.csv`: results for each unordered camera pair.
- `by_frame.csv`: framewise pooled counts and graph diagnostics.
- `gt_matching.csv`: the fixed GT assignment coverage.
- `decisions.jsonl.gz`: compressed full accepted/unmatched decision trace with
  frame/time, IDs, source embedding rows, scores, operational reasons and GT outcomes.

This is a quality diagnostic, not a runtime benchmark. One short integration clip
contains strongly correlated observations and is unsuitable for claiming statistical
significance or generalization. Preserve the complete grid; do not cherry-pick
its best row as a validated deployment threshold. Calibration requires separately
designated validation data, with independent final evaluation.

## Relation to published work

Deep SORT (Wojke et al., 2017), section 2.2, combines appearance costs with
admissibility gating and describes separate data for choosing its appearance
threshold. Our cross-camera partial-matching objective is documented in
`pairwise_contract.md`; this sweep does not reproduce the full Deep SORT algorithm.
https://arxiv.org/abs/1703.07402

Ristani et al. (2016), *Performance Measures and a Data Set for Multi-Target,
Multi-Camera Tracking*, motivates identity-level evaluation across cameras.
The link precision/recall here is an intermediate diagnostic, not that paper's
ID precision/recall or global IDF1, which require persistent identity outputs.
https://arxiv.org/abs/1609.01775
