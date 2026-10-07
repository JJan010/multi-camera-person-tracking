# Frozen multi-camera grouping diagnostic

This experiment evaluates `complete_support_greedy_v1` on the previously saved
pairwise decisions for scene_001, cameras 4/5/8, frames 2 through 299. It uses the
same `latest`/`mean` variants and threshold grid from the source report. It does
not select a deployment threshold, run inference, or create persistent IDs.

## Run

From the repository root, with the project virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_grouping_evaluation.py
PYTHONPATH="$PWD/src" python scripts/evaluate_multicamera_grouping.py \
  --pairwise-report artifacts/pairwise_association/20261006T214019883005Z/report.json
```

Both commands run on CPU. No new dependencies are needed beyond the existing
evaluation environment. Keep the source `decisions.jsonl.gz` alongside its report.
The scripts import the existing pairwise evaluator and its metric helpers.

## What is compared

The source graph contains one node per current local-track observation and one
edge per accepted pairwise association. Independent pairwise assignments can
produce contradictory multi-camera components. The grouping policy partitions
all nodes into singletons, pairs, or triples. Every group must contain at most
one observation from each camera, and every pair of members must have a source
edge. It is a deterministic greedy policy, not an optimal graph partition.

For each variant and threshold, compare the source edges against all within-group
pairs after grouping. The latter must be a subset of the source edges. The
report records retained and removed correct, wrong-known and unresolved links.
A link is unresolved if either endpoint lacks a diagnostic GT identity.

- Labeled pair precision: correct / (correct + wrong-known).
- Conditional pair recall: correct / original available positive pairs.
- Label coverage: (correct + wrong-known) / all accepted links.
- Verified correct fraction: correct / all accepted links, including unresolved.

An available positive pair is an assigned GT identity present in both current
camera observation sets. The denominator is inherited from the source protocol
and reconstructed from its labels. It remains unchanged after grouping. Recall
is not measured against all GT people, including those absent from the pipeline.
With no accepted known links, precision is undefined (`null` / `n/a`), not 100%.

A correct triangle contributes three correct links. If a true three-camera
person has only two accepted links (an open chain), this policy retains one pair
and a singleton. Its recall in this example decreases from 2/3 to 1/3. Missing
support is not evidence that the people differ. Grouping can remove correct
links; precision improvement is not guaranteed either.

## Groups and decisions

`singletons`, `two_member_groups`, and `three_member_groups` count groups across
frames, not unique people. All source observations, including unmatched ones,
remain in the partition exactly once per setting. The accounting identities are:

- observations = singletons + 2 * pairs + 3 * triples;
- retained links = pairs + 3 * triples.

Only groups of size two or three receive an identity diagnostic:

- `linked_groups_all_known_same_gt`: every member has the same assigned GT ID;
- `linked_groups_different_known_gt`: at least two known GT IDs differ, even if
  another member is unresolved;
- `linked_groups_unresolved_gt`: at least one unknown label and no known conflict.

These three categories are disjoint. Singletons do not count as successful
identity matches. Zero repeated-camera groups is enforced by the policy and
checked during evaluation; it is not a measure of identity correctness.

Every source edge has a recorded grouping decision: `merged`,
`already_in_same_group`, `rejected_camera_conflict`, or
`rejected_missing_pair_support`. The output includes conflict cameras and missing
support edges. A triangle has two merges and one already-in-group decision.

## Provenance and verification boundary

The evaluator verifies the compressed source decision checksum, then reconstructs
GT-free `PairAssociation` objects for the actual grouping function. Saved GT
labels are passed only to diagnostic scoring after grouping.

It checks expected record order and scope, exact timestamps, source row/label
consistency across camera pairs and experiment settings, one-to-one GT labeling
per camera, and unchanged observation coverage. It reproduces source per-pair
counts, aggregate metrics and connected-component counts from the decisions and
compares them with the source report. Input checksums are checked again at the end.

The source report hash and decision hash are recorded. Earlier dataset/model/
embedding hashes are copied under `inherited_source_inputs_not_reverified`:
this step does not reread raw GT, recalculate IoU assignments, or verify earlier
model inputs. Repeating those operations would be a different evaluation step.

Known-answer checks cover a correct triangle, a true open chain, a camera
conflict, unresolved labels, reject-all, empty rounds and malformed trace data.

## Outputs

Each invocation creates `artifacts/multicamera_grouping/<UTC run ID>/`:

- `report.json`: protocol, provenance, completed-run checks and aggregate results;
- `summary.csv`: one row per descriptor variant and threshold;
- `by_frame.csv`: the same counters per scene frame;
- `groups.jsonl.gz`: all memberships, source embedding rows, diagnostic GT labels
  and edge decisions for each frame and setting.

Group list indices are temporary positions within one scene time, never global
IDs. The detailed trace enables later inspection without rerunning models.
Raw outputs remain ignored by Git; selected results can later be documented.

## Interpretation and next boundary

This is a paired integration diagnostic on a short, reused training-scene clip.
Observations across frames are dependent. Do not infer statistical significance,
choose a final threshold, or call these metrics global IDF1. Persistent identity
management across time and its trajectory-level evaluation are later steps.
This script is not a performance benchmark.

Ristani et al., *Performance Measures and a Data Set for Multi-Target,
Multi-Camera Tracking* (2016), introduce identity-based trajectory evaluation:
https://arxiv.org/abs/1609.01775. The pair precision/recall here is a separate
project diagnostic, not that paper's ID precision/recall or IDF1.
