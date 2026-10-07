# Longer sequence with unchanged MTMC settings

## Purpose

Extend the current scene_001 reference from 300 to 1800 rounds: 60 seconds of
30-FPS video from each of cameras 4/5/8, or 5400 camera images. The existing
pipeline runner already supports this duration. Model backends, thresholds,
history limits, geometry policy and merge confirmation settings are unchanged.

This examines behavior over a longer trajectory. It is not independent-scene
validation, a final held-out test or a TensorRT/NVDEC performance comparison.
The first 300 rounds overlap the previously inspected reference clip.

## Check the evaluator first

From the project root in WSL, with the project virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_mtmc_sequence_evaluation.py \
  --baseline-evaluation artifacts/mtmc_pipeline_evaluation/20261007T132132828417Z/report.json
```

The check validates variable-duration GT parsing, exact final-frame boundaries,
the meaning of cumulative IDF1 and the workflow's report selection. It also
reevaluates the **same frozen 300-round run** and requires exact equality of
the original metrics, denominators, merge diagnostic counts and control map.
It writes a new CPU evaluation report but does not execute either GPU model.

## Run and evaluate one minute

```bash
PYTHONPATH="$PWD/src" python scripts/run_mtmc_sequence.py \
  --reference-report artifacts/geometry_identity/20261007T124001420051Z/report.json \
  --rounds 1800
```

The workflow invokes the existing `run_mtmc.py` with `--rounds 1800`,
`--warmup 30` and `--torch-threads 1`. It relays progress immediately, requires
successful process completion, validates the exact emitted report path, and
passes that report to `evaluate_mtmc_sequence.py`. It never selects the latest
directory, which might belong to a different or incomplete run.

GPU models are owned by the pipeline child process. After it exits, the second
phase evaluates frozen output on CPU. Evaluation failure does not invalidate
or delete the completed pipeline artifacts; rerun the evaluator with that
explicit report path after resolving the error.

Output directories:

- `artifacts/mtmc_pipeline/<run-id>/`: the existing video-run artifact schema;
- `artifacts/mtmc_sequence_evaluation/<evaluation-id>/`: full-duration metrics,
  `comparison.csv`, `prefixes.csv`, shared identity matching and merge diagnostics.

No additional dependencies or replacement runtime modules are introduced.
The old 300-round evaluator remains unchanged as a regression reference.

## Evaluation protocol

Read the duration from the completed pipeline report. The supported scope is
the same pinned scene_001 camera subset, from 3 to 23994 runtime rounds.
Evaluate GT frames 2 through `rounds - 1`; validate runtime traces from frame 0.
These cameras have annotated observations on every frame 2..23993 in the
previously audited scene. The GT loader enforces that scene-specific coverage;
it does not claim that empty GT frames are generally invalid in other datasets.

Reuse the original evaluator's observation-key joins, box clipping, IoU gate,
unencoded/outside prediction handling and shared identity metric engine.
Both whole-sequence variants are checked against `motmetrics`. All source
checksums, row counts and assignment coverage are verified. Runtime predictions
are never modified, and no GT labels enter the runtime managers.

The no-cross-camera control uses the exact same boxes, with each camera/local
ID treated as a distinct predicted identity. GT and prediction observation
denominators must agree between the control and full pipeline.

## Growing-prefix metrics

By default, calculate metrics after 300, 600, 900, 1200, 1500 and 1800 runtime
rounds. Include the final round count even if it is not a multiple of 300.
Each prefix has its own **single shared identity matching over its entire
evaluated history**. For example:

| Runtime prefix | Quality evaluation frames |
|---|---|
| First 10 seconds | 2..299 |
| First 20 seconds | 2..599 |
| First 60 seconds | 2..1799 |

There is one uninterrupted runtime state starting at frame 0. Prefix scoring
does not rerun the tracker, reset global identities or process separate clips.
The offline metric may recompute its GT-to-predicted identity mapping when more
observations are added; this never rewrites runtime IDs.

Do not average these overlapping scores. They are also not independent
10-second-window scores. A GT person assigned ID 1 in one segment and ID 2 in
another can look perfect when each segment is scored separately, while receiving
a lower whole-sequence IDF1 because one predicted identity must be selected for
that GT identity across the whole evaluated prefix.

A cumulative score need not decrease monotonically. New frames can improve or
worsen it due to visibility, detection coverage, identity errors or a changed
optimal offline mapping. Interpret changes alongside IDTP/IDFP/IDFN, observation
counts, predicted identity counts and the same-run control in `prefixes.csv`.

The first 10-second prefix of the **new video run** is not guaranteed to be
bitwise identical to the older run: model inference is executed again. The
separate regression check uses the old frozen outputs to verify metric parity.

## Performance and interpretation

Warmup stays at 30 rounds for timing only. Tracker/history/global state persists
through it and all annotated predictions remain in quality evaluation. Longer
run timing can differ because crowd size and crop counts vary; that alone is
not evidence of a backend optimization.

Inspect the final global IDF1, cumulative prefix trajectory, merge diagnostics,
allocated/absorbed/expired IDs, history size and runtime timings. A good first
minute does not establish recovery from every type of occlusion or camera
transition. Diagnose observed failures before choosing another algorithmic
change. Reserve a separate scene for validation and an untouched test scope
for final claims.

Reference for shared multi-camera identity evaluation: Ristani et al.,
*Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking*,
ECCV Workshops 2016, https://arxiv.org/abs/1609.01775.
