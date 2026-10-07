# Frozen global identity failure analysis

This diagnostic explains one explicitly chosen setting from the global identity
evaluation. It does not modify predictions, merge IDs, rerun models, or select a
deployment threshold. The initial case is `mean` with threshold `0.70`, chosen
because it had the highest observed IDF1 in the reused integration fragment.

## Run

From the project root, in its virtual environment:

```bash
PYTHONPATH="$PWD/src" python scripts/check_global_identity_diagnostic.py
PYTHONPATH="$PWD/src" python scripts/diagnose_global_identity.py \
  --evaluation-report artifacts/global_identity_evaluation/20261007T094804760266Z/report.json \
  --variant mean \
  --threshold 0.70
```

The script verifies the evaluated identity trace, tracked boxes and GT against
their recorded hashes, reconstructs camera/frame IoU gates, and reproduces the
selected global metric counts exactly. Evaluation and replay reports remain
linked by their source replay ID. Input files are hashed again at completion.

## Exact aggregate decomposition

Let S be the sum of maximum-cardinality IoU >= 0.5 box matchings, optimized
independently in each camera/frame. Let I be IDTP from the one shared whole-clip
identity matching, G all evaluated GT observations, and P all predictions.

Every match counted by I is also spatially admissible, so I <= S. Therefore:

- IDFN = (G - S) + (S - I);
- IDFP = (P - S) + (S - I).

The first terms are the minimum FN and FP possible with the existing boxes and
IoU gate, even if identities were unrestricted across frames/cameras. The shared
term `S - I` is the additional loss under the common identity assignment.
`2*S/(G+P)` is a framewise spatial F1 ceiling with these fixed predictions.

This is an explanatory, GT-assisted upper bound, not a promised achievable online
tracker score. The gap is not exclusively a defect in the global-ID module:
local identity switches, duplicate compatible boxes and matching competition can
also contribute. Fixing identities cannot repair the fixed-box minimum without
changing the predictions or evaluation protocol.

Unlike the global ID assignment, the spatial ceiling deliberately allows the
GT/prediction pairing to vary freely each frame. It cannot serve as an alternative
reported tracking result. The same clipped-box and zero-area policies are reused
from the evaluated baseline.

## Identity evidence

For cautious examples, use only overlaps that are mutually unique in their
camera/frame: the prediction has exactly one eligible GT box, and that GT box has
exactly one eligible prediction. Ambiguous overlaps are excluded from this
histogram evidence, although they remain in the complete IDF1 and ceiling scores.

For each GT ID, count supporting observations by predicted global ID. More than
one supported global ID indicates diagnostic fragmentation. Rank examples by
support outside the largest global-ID bucket. This is not the CLEAR fragmentation
metric and does not sum to IDFN.

For each global ID, count supporting observations by GT ID. More than one supported
GT ID indicates diagnostic mixing. Rank by support outside the largest GT bucket.
This is not IDFP, and the largest bucket is not necessarily the identity selected
by the global one-to-one evaluation mapping.

The report also counts frames in which a GT identity has several supported global
IDs simultaneously across cameras, or one global ID has several supported GT
identities simultaneously. Per-camera local-track histograms help locate upstream
identity instability. These are observation diagnostics, not definitive causal
attribution or visual proof; even unique spatial overlap can be misleading.

## Existing-ID conflicts

For every `rejected_multiple_existing_ids` group decision, inspect these diagnostic
labels after inference. Categories are disjoint:

- `all_members_same_gt`: all members have mutually unique overlap evidence for
  the same GT identity;
- `different_known_gt`: at least two supported GT identities differ, even if
  another member lacks evidence;
- `unresolved`: remaining cases with missing or ambiguous evidence.

Counts include repeated conflicts over successive frames. They are not counts of
unique people or unique errors. Same-GT conflicts motivate inspecting a future
causal merge policy; GT labels themselves must never become runtime merge input.
The first five examples in each category include frame, camera/local ID, output
global ID and diagnostic GT ID.

## Outputs

Each run creates `artifacts/global_identity_diagnostic/<UTC run ID>/`:

- `report.json`: verified metrics, decomposition, evidence counts and top examples;
- `gt_fragmentation.csv`: GT-centric global-ID support;
- `global_identity_mixing.csv`: global-ID-centric GT support;
- `local_track_labels.csv`: per-camera/local-track GT support;
- `support_by_track.csv`: observation counts and first/last frames for each
  `(GT ID, global ID, camera, local ID)` combination;
- `conflict_examples.json`: representative rejected existing-ID groups.

CSV evidence files are omitted if the corresponding evidence set is empty.
First/last frame spans can contain gaps; their lengths are not support counts.
The console is a summary; the report and CSVs contain the audit detail.

## Interpretation boundary

This uses the same short training-scene fragment as earlier development. It is
failure analysis, not independent validation or final parameter calibration. A
subsequent behavioral change must remain causal and be compared with the frozen
baseline using the same evaluated observations and denominators.
