# Paired appearance-continuity evaluation on frozen observations

Runner: `scripts/experiment_appearance_continuity.py`.
Declared policy: `configs/tracking/appearance_continuity_experiment.json`.
The original staged tracker and all-updates global appearance history are the
reference; dormant recovery is OFF for both variants.

## Input gate and scope

The runner consumes the existing verified disabled registry replay report. That
report supplies the pinned source/cache paths and the staged competitive-global
reference, not an instruction to enable dormant recovery. All pinned input hashes
and the relevant package versions must still match. New code and the declared
configuration are fingerprinted in the new experiment report.

This runner deliberately accepts one declared development experiment only:
scene_001, cameras 4/5/8, 30 FPS, runtime 0..3599, evaluation 2..3599. It does not
perform a parameter sweep or choose thresholds using GT. The sparse-reference
policy is documented separately in `appearance_continuity_contract.md`.

## Phase 1: causal replay

Read the frozen original detector candidates, their cached OSNet features and
the already selected staged local tracks. No detector, local tracker, decoder,
GT loader or model inference is run. The candidate cache is checked against
the original detector rows, and each selected box/score/feature is joined by
its recorded candidate index and embedding row.

Two independent policy/history/global-state instances start at frame zero:

- Disabled: original local keys flow through unchanged; no continuity features
  are retained. Every full global runtime record must match the old reference,
  including decisions, pending merges, retained state and expiry, except run ID.
- Enabled: the declared appearance-continuity policy may introduce new segment
  keys. The unchanged appearance history and global association then consume
  those keys. A split's first downstream history must contain only its current
  frame. All boxes, scores, feature values, timestamps and original observation
  keys remain exactly accounted for.

The global history/geometry/grouping/identity settings remain fixed. Their
RESULTS may change because segment keys and histories change. Both branches keep
continuous state through both minutes; there is no reset at the window boundary.

## Persisted representations

`global_tracks.jsonl.gz` contains both variants, each with:

- `cameras`: the unchanged original staged local observations;
- `segment_bindings`: original observation keys, effective keys, generation and
  segment start provenance;
- `identity_runtime`: the complete global state/output with internal segment keys;
- `identity`: a separate evaluation view with original observation keys for
  assignments and current merge members. This view is not a runtime state object.

Only the explicit binding map is used to project runtime assignments back to
source boxes. Internal segment IDs are never mistaken for ByteTrack IDs.
`continuity_audit.jsonl.gz` records policy decisions, source reference frames and
times, confirmation evidence counts, reset reasons and actual segment events.
Both outputs are hashed into `predictions_frozen.json` before opening GT.

## Phase 2: offline evaluation

The runner reuses the verified shared-identity evaluator from
`experiment_dormant_recovery.py`; that evaluator uses only the explicit evaluation
view and fixed source cameras, not the dormant archive. Both variants are checked
against motmetrics for the full sequence and each disjoint minute. All prediction
and GT denominators are identical between variants. Disabled full quality and
merge diagnostics must reproduce the earlier staged result.

Original local metrics are evaluated once and checked against the frozen staged
metrics. They are unchanged by construction because original tracker IDs and
boxes are unchanged. This is NOT a measurement of local IDF1 after replacing
original IDs with segment IDs; that metric is not reported here.

For every split, the diagnostic uses the actual prior-reference source frames,
the confirmation interval and the current original observation. Each unique GT
label uses the existing mutual spatial uniqueness rule over ALL predictions in
its camera/frame. ID zero is valid. Categories are:

- `different_reference_and_current_gt`: every reference sample has one common
  known GT, and the current known GT differs. Evidence supports a person change.
- `same_reference_and_current_gt`: all reference samples and the current sample
  have the same known GT. Evidence suggests possible unnecessary fragmentation.
- `mixed_reference_gt`: reference samples have multiple known GT identities.
- `unresolved_reference_or_current`: at least one relevant label is unknown.

These are event diagnostics, not accuracy certificates or an attribution of the
change in global IDF1. A different-GT event may be too late to recover earlier
errors; a same-GT event may occur in an already mixed history. Unknown events are
never counted as successes. Full per-event evidence is in `split_diagnostics.json`.

## Running in WSL

From the project root with its venv active:

```bash
PYTHONPATH="$PWD/src" python scripts/experiment_appearance_continuity.py \
  --bridge-report artifacts/dormant_registry_checks/20261008T091625171355Z/report.json
```

This requires the previously delivered dormant experiment evaluator, segment
module, continuity module/config, and the frozen source files. The output is a
new timestamped `artifacts/appearance_continuity_experiment/` directory. Failed
runs retain `run_status.json` with the error and are not marked completed.

Expected disabled development reference: full global IDF1 53.03%, first minute
73.57%, second minute 60.64%. The enabled result is unknown until the user runs
the real frozen data. Do not promote the policy based on synthetic tests or the
number of cuts. Compare quality and error evidence first; then evaluate a
promising unchanged policy on another scene. No end-to-end speed result is
claimed for this CPU replay or its clone-based experimental policy owner.
