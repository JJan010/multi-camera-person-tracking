# Unchanged-policy appearance continuity transfer

Run in the WSL project root with the project virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/validate_appearance_continuity.py \
  --development-report artifacts/appearance_continuity_experiment/20261008T105229140571Z/report.json \
  --validation-report artifacts/scene_pair_evaluation/20261007T214235770374Z/report.json
```

The script imports the continuity settings and implementation used by the completed
development experiment. It offers no threshold overrides. It verifies the recorded
code and source checksums and pinned package versions before replay.

Scope: scene_041, cameras 361/362/364, 3600 runtime rounds, evaluation frames
2..3599. This scene has already been inspected, including false positives and
missed detections. The result is an unchanged-policy transfer check, not an untouched
final test. Camera selection and both scene lengths limit generalization claims.

## Execution and comparison

1. Check the generic camera/size-aware selected-candidate join against the old
   scene_001 join over its complete frozen development trace. Every original
   observation key, box, score, crop, timestamp and selected embedding must match.
2. Replay the frozen scene_041 staged observations through two causal global
   pipelines, with continuity disabled and enabled. No decoder, model or local
   tracker runs. Runtime does not receive GT.
3. Require the disabled pipeline's complete internal records and lifecycle to
   match the previous scene_041 staged reference, except for the new run scope.
4. Preserve original source keys for evaluation. New segment keys are internal;
   cuts start fresh history and leave previously emitted outputs unchanged.
5. Freeze global outputs and audit logs, then load GT for offline full-sequence
   and disjoint first/second-minute evaluation. Check all metrics against
   motmetrics and reproduce the previous full disabled score (56.32%, rounded).
6. Verify original local metrics, observation denominators, merge diagnostics,
   per-event reference/current GT evidence and all source/output checksums.

Both variants use the same saved boxes, scores and embeddings. Local tracking,
normal history aggregation, geometry, association and merge settings remain the
frozen staged configuration. Dormant recovery stays disabled. Empty GT slots
remain in evaluation; predictions there count as errors. No ROI or GT exclusion
is added. Reference evidence for the continuity guard is separate from the short
rolling history used for cross-camera association.

Output directory: artifacts/appearance_continuity_validation/<run-id>/.
It contains report.json, global_tracks.jsonl.gz, continuity_audit.jsonl.gz,
predictions_frozen.json, identity_matching.json and split_diagnostics.json.
A failed run keeps run_status.json with completed=false and must not be treated
as a completed result.

## Interpretation

Compare full-sequence global IDF1 and integer IDTP/IDFP/IDFN first, then windows,
cut diagnostics and lifecycle. Do not average window IDF1 or interpret more
allocated IDs as an automatic regression. Do not classify unknown cuts as correct.
Original local IDF1 must remain unchanged because original tracker IDs are frozen;
this experiment measures the effect of segment boundaries on global identity.

A positive development result alone is insufficient for promotion. Transfer can
improve, remain neutral or regress. Even improvement on both scenes calls for a
further independently reserved scene or sequence before broad quality claims.
This CPU replay is not an end-to-end throughput benchmark.

## Technical verification before delivery

Synthetic 120-round fixtures exercised unchanged-policy replay with nonstandard
camera IDs, different image sizes, ID zero, an empty camera, empty rounds, empty
GT slots and fully outside predictions. Constant appearance produced no cuts and
exact enabled/disabled parity. Artificial rotating appearance produced real cuts,
verified fresh histories and original-key provenance, and complete event diagnostics.
The previous scene evaluator and the new evaluator agreed on full disabled global
and original local metrics; full/window identity counts agreed with motmetrics.
A separate fixture verified exact old/generic adapter parity with unsorted local
IDs, an empty camera and outside boxes. Fixtures use synthetic features and
available dependency shims; they are not GPU or real-data quality measurements.
