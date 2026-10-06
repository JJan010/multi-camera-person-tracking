# Pairwise appearance association: integration results

## Frozen experiment

Source local pipeline: `20261006T204237314810Z`.
History replay: `20261006T211035258446Z`.
Pairwise evaluation: `20261006T214019883005Z`.

Frames 2..299, cameras 4/5/8, three unordered camera pairs per frame: 894 pair-frame
instances per descriptor variant and threshold. The denominator is 12,031 available
GT-assigned positive pairs. Both latest and mean use identical observations and
GT assignments; the mean has at most eight observations of age at most one second.

The experiment uses the fixed grid 0.0/0.5/0.6/0.7/0.8/0.9/0.95/1.0. The gate is
strict `similarity > threshold`. Partial assignment maximizes the sum of score
gains over the threshold, with zero gain for leaving endpoints unmatched. The
threshold is an exploratory parameter, not a calibrated operating point.

`pairwise_metrics.json` preserves the uploaded report byte-for-byte.
`pairwise_summary.csv` is regenerated from its complete results and matches the
original summary artifact's SHA-256. Available source/config/code hashes and
pooled/per-pair counts were verified. The original evaluator verified full GT and
embedding files on the WSL machine. Detailed decision and frame CSV artifacts
remain under the report's original output directory in `artifacts/`.

## Aggregate results

Precision below is computed only among accepted links with both GT labels known.
Unknown links are reported separately and are not assumed to be correct. Recall
is conditional on both observations existing and being assigned a GT identity.

| Variant | Threshold | Correct | Wrong known | Unknown | Precision known | Recall available | Conflict frames / 298 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| latest | 0.00 | 11415 | 2626 | 287 | 81.30% | 94.88% | 290 |
| latest | 0.50 | 11412 | 1503 | 160 | 88.36% | 94.85% | 241 |
| latest | 0.60 | 11144 | 694 | 111 | 94.14% | 92.63% | 141 |
| latest | 0.70 | 9996 | 272 | 46 | 97.35% | 83.09% | 114 |
| latest | 0.80 | 5720 | 38 | 5 | 99.34% | 47.54% | 5 |
| latest | 0.90 | 1458 | 1 | 0 | 99.93% | 12.12% | 0 |
| latest | 0.95 | 132 | 0 | 0 | 100.00% | 1.10% | 0 |
| latest | 1.00 | 0 | 0 | 0 | undefined | 0.00% | 0 |
| mean | 0.00 | 11494 | 2533 | 301 | 81.94% | 95.54% | 286 |
| mean | 0.50 | 11508 | 1669 | 201 | 87.33% | 95.65% | 258 |
| mean | 0.60 | 11317 | 700 | 139 | 94.17% | 94.07% | 144 |
| mean | 0.70 | 10507 | 363 | 98 | 96.66% | 87.33% | 128 |
| mean | 0.80 | 6904 | 87 | 22 | 98.76% | 57.39% | 46 |
| mean | 0.90 | 1995 | 10 | 1 | 99.50% | 16.58% | 0 |
| mean | 0.95 | 271 | 0 | 0 | 100.00% | 2.25% | 0 |
| mean | 1.00 | 0 | 0 | 0 | undefined | 0.00% | 0 |

Higher thresholds substantially reduce accepted errors in this sweep, while losing
many correct opportunities. Mean at 0.95 has no observed accepted error but finds
only 271 of 12,031 available positive pairs. This is not evidence of a generally
perfect identity matcher. Threshold 1.0 correctly rejects everything.

At threshold 0.8, mean accepts 1,184 more correct links than latest, but also 49
more known wrong links and 17 more unresolved links. Camera-conflict frames rise
from 5 to 46. A higher mean retrieval Rank-1 does not establish safer thresholded
association at the same numeric threshold. Ranking depends on relative ordering;
gating and assignment also depend on absolute scores and candidate competition.
Changing descriptor aggregation does not guarantee comparable calibration of the
same threshold. These grid points do not establish that either variant dominates
at matched precision/recall operating points.

Correct-link counts need not decrease at every individual threshold step: mean
improves from 11,494 correct at 0.0 to 11,508 at 0.5 as the partial assignment changes.
Thresholding can change the selected pairing, not merely delete a fixed list of links.

## Camera-pair differences

Mean at threshold 0.8 illustrates how a pooled metric can hide different behavior:

| Camera pair | Correct | Wrong known | Unknown | Precision known | Recall available |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4-5 | 3249 | 9 | 12 | 99.72% | 87.24% |
| 4-8 | 1542 | 61 | 4 | 96.19% | 37.72% |
| 5-8 | 2113 | 17 | 6 | 99.20% | 50.08% |

Most known wrong links at this threshold occur in pair 4-8. This is a diagnostic
finding, not sufficient evidence for choosing camera-specific thresholds. The
cause of each error has not been established from the summary alone.

## Structural consistency is distinct from identity correctness

Connected components are inspected without assigning global IDs. A component
containing two local IDs from one camera would violate a naive global merge's
same-camera uniqueness assumption, unless upstream duplicate tracks are involved.

At threshold 0.8, latest has 5 such components across 5 frames and mean has 46
across 46 frames. Counts concern repeated frame observations, not necessarily
distinct people or independent failure events.

At threshold 0.9, both variants have zero camera-conflict frames, yet latest still
has one known wrong link and mean has ten, plus one unresolved link. A wrong
two-node component can satisfy all camera-uniqueness constraints. Structural
consistency is necessary for the intended grouping contract but not proof of GT correctness.

Mean at 0.8 also has 1,144 open three-camera components: three distinct camera
observations connected by two edges without a third supporting edge. These occur
in 297 of 298 frames. An open chain is not automatically a false match. In the
first saved example (frame 2), camera/local IDs (4,6), (5,4), (8,5) all have GT ID
13 despite forming an open chain. Requiring closed triangles would reject some
correct groups; blindly accepting transitive chains can create other errors.
The grouping policy therefore needs explicit semantics and its own comparison.

## Unmatched observations

Across each full variant/threshold pass there are 7,270 labeled endpoints whose
assigned identity has no assigned counterpart in the opposite camera. At mean
threshold 0.8, 7,212 remain unmatched and 58 are linked. This is 99.20% abstention
for this diagnostic category. A missing assigned counterpart can result from a
missed or unlabeled observation, so this is not proof of physical absence.

At this same setting, 5,127 available positive pairs are missed. Good rejection
of absent assigned positives does not imply good recovery of available pairs.
Unknown accepted links, conditional recall and the complete denominator must
remain visible when comparing policies.

## Working decision

Preserve both descriptor variants and the entire threshold grid. Mean remains a
candidate working descriptor; the earlier retrieval result is not promoted to a
claim that mean is the final superior association policy. No threshold is selected,
no GT-based reset/filter is added, and no persistent global IDs are claimed here.

The next implementation step is explicit multi-camera grouping with conflict
handling, evaluated on the same frozen decisions where possible. Define how
open chains, conflicting candidates and unmatched observations are handled before
adding global identity lifecycle/state. Structural and GT quality should be
reported separately. Operating thresholds require separately designated validation
data; the final assessment must remain independent of that selection.

This single short synthetic integration clip does not establish statistical
significance, generalization, delayed camera handover, global IDF1, or runtime
performance. Frames and camera pairs reuse the same people. The report's
`selected_threshold` remains null.
