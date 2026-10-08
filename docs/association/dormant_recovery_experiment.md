# Paired dormant identity recovery experiment

Status: one declared development hypothesis. Enabled recovery has no measured
quality result on the real two-minute trace until the WSL experiment completes.

## Hypothesis and fixed inputs

Can a bounded memory of departed identities reduce identity fragmentation when
a wholly new local group appears near the person's previous location? Replay
the same frozen scene_001 staged local tracks twice, with recovery disabled and
enabled. Detections, local IDs/boxes/scores, selected candidate embedding rows,
all-update appearance history, geometry and current-frame grouping are identical.
Only persistent identity behavior can change after reactivation. Later merge
decisions may consequently differ; they are not artificially kept fixed.

Scene: train/scene_001; cameras 4/5/8; runtime 0..3599 at 30 FPS; evaluation
2..3599. This is reused development footage, not independent validation.

## Configuration fixed before measuring

`configs/association/dormant_recovery_experiment.json` records:

| Setting | Experimental value | Meaning |
|---|---:|---|
| Maximum archive/evidence age | 5 s | Inclusive; scene time and true source times |
| Minimum cosine similarity | 0.8 | Strictly greater; not a probability |
| Minimum ranking margin | 0.05 | Inclusive margin in both query and identity directions |
| Position slack | 2 native units | Fixed return radius around the last ground sample |
| Radius expansion speed | 0 | No motion prediction or assumed walking speed |
| Archive capacity | 128 | Resource limit; at most 256 KiB of archived float32 vectors |

This is a single configuration, not a sweep. The radius reuses the frozen
association distance tolerance, but its use after a time gap is still a new
unvalidated assumption. A moving person returning farther away is deliberately
not eligible in this first experiment. Five-second retention, appearance gate
and margin are explicit engineering hypotheses, not calibrated deployment
parameters. Native calibration units are not asserted to be meters.

Only wholly new, unanchored groups can request recovery in their birth round.
An ambiguous birth keeps its fresh ID without repeated recovery attempts in
later anchored rounds. Active local-ID switches, wrong earlier merges and
confident false detections are not repaired by this policy. A return decision
uses one round of evidence; there is no multi-round confirmation or automatic
undo of a wrong reactivation.

## Evidence adapter

`RecoveryIdentityStage` composes the original geometry/grouping stage with the
transactional recovery registry. The frozen modules are not edited.

The adapter requires the complete `HistoryBatch`, rather than a mean embedding
without provenance. It verifies key coverage, normalized arrays, increasing
source frames and exact times, history age/count bounds, current-sample inclusion
and cancellation metadata. An ordinary mean retains the earliest contributing
source time. When the original history's cancellation policy selects the latest
vector, the effective time is that vector's current source time. Neither case
silently refreshes an old feature at retirement.

Ground points come from the same raw box-bottom projection and pinned calibration
used for association. Each point has the current scene time. Outside/unencoded
observations retain their assignment but supply explicit unavailable evidence.
Before writing a round the runner verifies exact feature-row equality, source
time mapping and projection equality against the original grouping stage.

The actual archive snapshot policy remains the bridge's complete-current-group
policy. Missing or split support preserves old source times; a confirmed merge
requires new jointly supported evidence for its canonical snapshot.

## Two-phase protocol

1. Verify the completed disabled-bridge parity report and its entire frozen input
   and code chain, including the original scene-aware runtime parity report.
2. Replay the pinned candidate features and staged local trace on CPU, without
   models, decoding or raw GT. The disabled branch must reproduce every complete
   prior staged identity record, changing only its run scope. Both branches must
   preserve the same local observations, appearance inputs, points and grouping.
3. Save and hash global predictions, registry decisions and history provenance
   before loading GT. Failed partial runs remain explicitly incomplete.
4. Evaluate one shared global identity mapping across all cameras and frames,
   then separately for 2..1799 and 1800..3599. Runtime state remains continuous;
   only offline evaluation mappings are separate. Verify all metrics against
   motmetrics 1.4.0 and reproduce unchanged local metrics.
5. Diagnose each accepted return offline, comparing the returning members' unique
   spatial GT labels with the same global ID's LAST VISIBLE member labels before
   the return. Report same, different, mixed or unresolved evidence. This is a
   continuity diagnostic, not a label for the archived vector or proof that the
   entire earlier identity history was correct.
6. Recheck frozen hashes, event/lifecycle denominators and control metrics.

Expected actual disabled control: global IDF1 53.03% full, 73.57% first window,
60.64% second window. Full evaluation denominators are 184207 GT observations
and 180461 prediction observations. Local pooled IDF1 remains 44.86% in both
variants. These are reference checks, not forecasts for enabled recovery.

## Run

In Ubuntu WSL, from the project root with its virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_recovery_evidence.py

PYTHONPATH="$PWD/src" python scripts/experiment_dormant_recovery.py \
  --bridge-report artifacts/dormant_registry_checks/20261008T091625171355Z/report.json
```

The evidence checker covers disabled record/group parity, projection and row
mapping, real source age including cancellation fallback, unavailable crops,
enabled archive-to-registry integration and malformed-provenance rejection.
Local technical end-to-end verification also uses 300 rounds of saved early
boxes with synthetic constant features. That fixture validates plumbing and
control/evaluation parity; its scores do not measure actual OSNet quality.

## Outputs and interpretation

New outputs: `artifacts/dormant_recovery/<run>/`:

- `global_tracks.jsonl.gz`: identical local records and both final global traces.
- `registry_audit.jsonl.gz`: provisional baseline decisions, archive pair evidence,
  claims, snapshot decisions and lifecycle counters for both branches.
- `history_provenance.jsonl.gz`: per-observation source frames/times, effective
  descriptor time and projected ground point, with checked shared input mapping.
- `predictions_frozen.json`: pre-GT hashes and the declared configuration.
- `identity_matching.json`: full/window shared assignments used by the evaluator.
- `reactivation_diagnostics.json`: every return's last-visible GT comparison.
- `report.json` and `run_status.json`: measurements, inputs, checks and completion.

Distinguish allocated ID slots from IDs ever emitted: a provisional new ID can
be superseded by an old one before emission and must never be reused. Expiration
events can recur for a reactivated identity. Archive size differs from all
currently inactive identities because old evidence can expire or be evicted.
Use the bridge's conservation equation and counters, not the older lifecycle
helper that assumes no reactivation.

Inspect full/window IDF1, identity counts, return decisions, last-visible label
diagnostics and merge diagnostics together. More reactivations or fewer emitted
IDs alone do not establish better identity tracking. No threshold should be
silently changed in response to a disappointing score.

Keep the staged/all-updates baseline unchanged. A promising result needs a
fixed-policy transfer check; scene_041 has already been inspected, so a future
untouched final test must remain separate. This experiment establishes no
TensorRT/NVDEC acceleration or end-to-end throughput improvement.
