# Selective global appearance history: confidence-only ablation

Status: isolated memory implementation and synthetic checks. The runtime
baseline is unchanged; there is no quality or speed improvement claim.

## Hypothesis and scope

Current global AppearanceHistory accepts every available tracked embedding.
The local experimental tracker already has a separate strong-sample history.
This experiment changes only the global history; local associations, boxes,
scores and IDs must remain frozen. Do not confuse these two memories.

Compare accepting all global updates with accepting updates at confidence
at least 0.5. This reuses the existing local high-stage boundary instead of
introducing a threshold sweep. Confidence is an imperfect proxy for crop
quality. The pallet false detections on validation sometimes exceed 0.7,
and genuinely visible fragments can score below 0.5. This hypothesis does
not solve either case by definition and may worsen identity association.

Configuration is recorded before the experiment in
configs/reid/selective_history_experiment.json. Develop on scene_001,
frames 0..3599 with evaluation 2..3599, staged local outputs. No scene_041
retuning or identity-specific filtering. Keep 8 samples, maximum scene age
1 second, mean descriptor, and the existing global association settings.

## Exact memory semantics

- Acceptance is an explicit boolean for each ObservationKey, joined by key.
- Rejecting an update does not discard the tracked observation.
- A rejected current vector never enters the gallery or cancellation fallback.
- Existing, unexpired accepted vectors may provide a descriptor for the
  current observation of the same camera/local ID. Source times remain old;
  the output observation time is current, with old source times recorded.
- Individual samples expire by scene time, inclusive age <=1 second.
  Reuse does not refresh their age; partial expiry recomputes the mean.
- Without retained history, appearance is explicitly unavailable. No zero
  vector or rejected current-vector fallback is supplied.
- With no current observation, no descriptor is emitted, even if memory
  remains. Empty rounds still expire samples.
- If accepted vectors numerically cancel, use the latest accepted vector.
- Inputs and output vectors cannot mutate internal memory. Full validation
  precedes state changes. All-accepted output must match the existing history.

All current input keys are returned in decisions; mean.keys can be a subset.
This is a deliberate availability distinction, not a missing observation.
The existing IdentityStage requires exact crop/descriptor coverage, so this
output must NOT be plugged directly into the baseline stage. The next
integration step must explicitly preserve unavailable observations as
singletons while retaining their local/global bindings. It must not mark
valid crops as outside-image just to bypass the existing coverage contract.

## Next paired experiment and acceptance criteria

1. Add explicit descriptor availability to an isolated experimental identity
   adapter, preserving every local observation and source crop metadata.
2. Replay all-updates to reproduce the complete staged reference trace,
   not just rounded IDF1. Use identical frozen local observations for the
   selective branch and freeze both outputs before reading GT.
3. Verify local metrics and denominators are unchanged. Compare global
   IDF1/IDTP/IDFP/IDFN, lifecycle, merges and descriptor unavailable/reuse
   counts across full sequence and fixed time windows.
4. Do not promote a variant on one improved example. If promising, freeze
   its configuration and evaluate an additional untouched scene. If it
   regresses, retain the negative result and the current staged reference.

This is not a learned occlusion detector, blur filter, local ID-switch
repair, new Re-ID model or TensorRT optimization. Reusing an old descriptor
may itself propagate a local ID switch and must be measured.

## Sources

Zhang et al., ByteTrack (ECCV 2022), https://arxiv.org/abs/2110.06864,
motivates retaining weak observations for tracking rather than simply
discarding them. Aharon et al., BoT-SORT (2022),
https://arxiv.org/abs/2206.14651, studies motion and appearance association.
Neither publication establishes the effectiveness of this project's exact
confidence-gated, one-second, eight-sample global-history policy; that is
the hypothesis of the paired experiment.

Run the contract checks from the project root with the active venv:

```bash
PYTHONPATH="$PWD/src" python scripts/check_selective_history.py
```
