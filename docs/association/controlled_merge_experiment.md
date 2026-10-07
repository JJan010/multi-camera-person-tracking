# Paired controlled-merge experiment

This experiment compares `local_anchor_no_merge_v1` with
`whole_visible_group_confirmed_merge_v1` on identical frozen frame groups,
tracked boxes and observations. It does not rerun detection, tracking or OSNet.
The original baseline implementation and previous artifacts remain unchanged.

## Fixed integration settings

- Scene 001, cameras 4/5/8, frames 2..299 at 30 FPS.
- Mean appearance descriptors and cosine threshold 0.70.
- Binding idle duration inherited from the baseline report (1 second).
- Minimum support: 3 eligible rounds AND 1/5 second of scene time.
- Maximum gap between supporting updates: 1/10 second of scene time.

These are experimental settings, not a deployment calibration. With support
on every frame at 30 FPS, spanning 1/5 second requires 7 supporting rounds:
the first round starts the interval and six further frame intervals complete it.
Elapsed processing time is irrelevant to confirmation.

The selected mean/0.70 baseline was already inspected on this fragment.
Consequently, this is an in-sample diagnostic, not independent validation.
Do not choose new settings solely to maximize this fragment's score.

## Procedure

1. Verify source checksums and reconstruct the original no-merge manager.
   Require every selected baseline output record to match the frozen record.
2. Feed the same runtime groups to a separate controlled-merge manager.
   Runtime objects contain no GT labels. Source diagnostic labels are carried
   separately for provenance and source-counter checks.
3. Write and close the new assignment trace before quality evaluation.
4. Load tracked boxes and GT; evaluate both traces with the existing shared
   identity assignment across all cameras and frames. Reproduce the previously
   recorded baseline metrics exactly.
5. Verify equal observation coverage, identity lifecycle accounting and the
   unchanged checksum of the newly frozen assignment trace.

Merges are causal: accepted merges affect the current and later
rounds. Earlier assignments remain unchanged. An absorbed ID is not reused.
The metric's offline identity matching does not rewrite the prediction trace.

## Run in the WSL repository

```bash
PYTHONPATH="$PWD/src" python scripts/check_controlled_merge_experiment.py
PYTHONPATH="$PWD/src" python scripts/run_controlled_merge_experiment.py \
  --baseline-evaluation-report artifacts/global_identity_evaluation/20261007T094804760266Z/report.json \
  --variant mean \
  --threshold 0.70 \
  --min-support-rounds 3 \
  --min-support-seconds 1/5 \
  --max-evidence-gap 1/10
```

No additional dependencies or GPU execution are required. The scripts reuse
the existing grouping, replay and identity-evaluation helpers, plus the
previously installed controlled-merge module and its smoke-test fixtures.

## Outputs

Each run creates a new `artifacts/controlled_merge/<UTC-run>/` directory:

| File | Purpose |
| --- | --- |
| `report.json` | Settings, source hashes, paired metrics, lifecycle and checks |
| `assignments.jsonl.gz` | Final assignments, base-stage decisions and merge evidence |
| `comparison.csv` | Baseline and controlled identity metrics |
| `by_frame.csv` | Allocations, confirmations, pending candidates and retirements |
| `merge_diagnostics.json` | All accepted merge events with offline GT evidence |
| `identity_matching.json` | Both metric-level one-to-one identity assignments |

`base_assignments` and inherited `decisions` describe the base stage before
merging. `assignments` describes the final output of that round. Thus a
base-stage refusal to merge existing IDs can coexist with a confirmed merge
in the subsequent merge stage of the same round.

Accepted-event GT evidence uses only mutually unique same-camera/same-frame
IoU matches at the confirmation time. Categories are
`all_visible_members_same_gt`, `different_known_gt`, and `unresolved`.
They do not certify the entire retained identity history or absent members.

## Interpretation

The real mean/0.70 baseline to reproduce is IDF1=72.73%, IDTP=11690,
IDFP=4138 and IDFN=4627. Report the controlled result and its IDF1 change
without assuming improvement. Fewer allocated or emitted IDs alone do not
establish better tracking: incorrect merges can also reduce the ID count.

Confirmation can reduce fragmentation but cannot repair every local identity
switch, split a mixed identity or retroactively fix early assignments. Repeated
support is correlated across nearby frames and is not independent evidence.
No accepted merges is a valid result that calls for inspecting rejection and
reset reasons, not automatically weakening safeguards.

The synthetic smoke test deliberately yields baseline IDF1=50% and controlled
IDF1=62.5%, rather than 100%, demonstrating that early fragmentation remains
in the evaluation. Those values are test answers, not scene-quality results.
