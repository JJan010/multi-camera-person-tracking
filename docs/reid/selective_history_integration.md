# Paired global-history integration experiment

Status: implementation and technical checks; no new tracking-quality result yet.

Technical verification covers synthetic descriptor-availability cases and a
300-round fixture using saved early boxes with synthetic constant features.
That fixture checks control parity and evaluation plumbing, not Re-ID quality.
The WSL experiment below verifies the actual two-minute frozen embedding archive.

## Question and fixed scope

On the frozen scene_001 two-minute `staged` tracks, does excluding low-confidence
observations from global appearance history improve persistent identity quality?
Compare `all_updates` to `strong_updates` (score >= 0.5). The boundary reuses the
frozen local tracker's high-score boundary. It is an experimental choice, not a
validated quality threshold. Confidence measures the detector's person score;
it does not establish visibility, crop purity, or correct track identity.

This targets the global appearance history. The local staged tracker already
uses its own strong-observation history. Local matching, its history, detections,
tracked boxes, scores, candidate embeddings and output observation keys remain
unchanged. Geometry, assignment, grouping and persistent identity policy retain
their frozen settings. There is no score sweep or validation-scene adjustment.

## Availability contract

`AvailableIdentityStage` is a separate adapter. `core.py` and every previously
fingerprinted implementation remain unchanged.

For a current observation with a weak score, reuse only accepted history samples
that have not expired. Reuse does not refresh their age. With no retained samples,
the appearance descriptor is explicitly unavailable. The local observation still
enters grouping as a singleton and receives or retains a global ID through the
existing identity policy. It cannot create a new appearance link. Valid crop
metadata is preserved; an unavailable descriptor is not an outside-image crop.

Age is scene time, with the inclusive one-second boundary. Keep at most eight
accepted samples, use the existing normalized mean, and preserve the latest
accepted-vector fallback on cancellation. State advances through empty rounds.

## Protocol

1. Verify the scene-aware paired-runtime parity report, its source hashes, code
   and policy. This report already connects the frozen candidate archive to the
   `staged` local/global trace. Use the exact two-minute scene_001 sources.
2. Replay the selected candidate embeddings and frozen local tracks on CPU.
   Do not run models, decode video, rerun local tracking or read GT in this phase.
3. At every round, compare the all-updates mean against the original history and
   compare the complete all-updates global result against the frozen staged
   reference, changing only the run scope. Check lifecycle and observation counts.
4. Persist and hash both global traces and their accepted/reused/unavailable
   history decisions before opening the ground truth.
5. Evaluate frames 2..3599 with one shared identity assignment over all cameras
   and frames. Also evaluate disjoint windows 2..1799 and 1800..3599. Runtime
   state is continuous; each window has its own offline identity matching.
6. Verify each global result against motmetrics 1.4.0, reproduce unchanged local
   metrics, reproduce the control metrics and merge diagnostics, and recheck
   input/output hashes. Fail the run if any check differs.

Expected development control: global IDF1 53.03% over two minutes, 73.57% in
the first window and 60.64% in the second. Shared local IDF1 remains 44.86%.
These are checks against earlier measured results, not expected gains for the
selective branch. Full-sequence IDF1 is not the average of window IDF1 values.

## Run in WSL

From `/home/jakjan/projects/multi-camera-person-tracking`, with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_selective_identity.py

PYTHONPATH="$PWD/src" python scripts/experiment_selective_history.py \
  --parity-report artifacts/paired_scene_runtime_checks/20261007T211730398428Z/report.json
```

New outputs live under `artifacts/selective_history/<run>/`:

- `global_tracks.jsonl.gz`: frozen local records and both identity outputs.
- `history_decisions.jsonl.gz`: every encoded observation's update decision,
  accepted source frames and scene times, mean norm and cancellation fallback.
- `predictions_frozen.json`: pre-evaluation output hashes and fixed configuration.
- `identity_matching.json`: each variant/window's shared identity assignment.
- `report.json`: metrics, provenance, lifecycle, history counts and verified checks.
- `run_status.json`: completion status; partial failed outputs are not results.

## Interpretation and next decision

Compare global IDF1 on the entire interval and both windows, unavailable/reused
descriptor counts, fragmentation-related lifecycle counts and merge evidence.
A reduced sample count is not by itself a quality improvement. Locally swapped
IDs and confident false detections can still contaminate the history. This test
does not change those upstream observations.

The reused development scene establishes a paired engineering comparison. A
promising result still needs a fixed-policy transfer evaluation; scene_041 has
already been inspected and should not be described as an untouched final test.
Do not promote this policy or claim a deployment threshold from this run alone.
No inference was skipped and this CPU replay is not an end-to-end speed test.

## Scientific context

ByteTrack explains why low-score detections can preserve tracks through
occlusion: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating Every
Detection Box*, ECCV 2022, https://arxiv.org/abs/2110.06864.
Keeping a weak observation for continuity and accepting it into an appearance
memory are different decisions. The confidence-only global-history rule here is
our ablation; it is not a quality guarantee from the ByteTrack paper.
