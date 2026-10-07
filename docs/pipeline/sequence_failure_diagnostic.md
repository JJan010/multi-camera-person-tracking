# Diagnose quality loss over a longer frozen sequence

## Observed motivation

The real 1800-round run `20261007T133915524799Z`, evaluated by
`20261007T134209413468Z`, produced:

| Runtime duration | Global pipeline IDF1 | Global no-cross-camera control IDF1 |
|---|---:|---:|
| 10 seconds | 85.99% | 42.63% |
| 20 seconds | 82.79% | 38.16% |
| 30 seconds | 81.99% | 36.91% |
| 40 seconds | 80.30% | 34.93% |
| 50 seconds | 75.16% | 32.35% |
| 60 seconds | 71.75% | 29.74% |

Full-minute IDTP=69634, IDFP=26562, IDFN=28281; G=97915, P=96196.
Mean throughput remained approximately 12.108 rounds/s (36.325 images/s),
with mean processing core 79.268 ms and core plus recording 82.507 ms.
These are aggregate timings; they do not prove every time window is equally fast.

Accepted merges: 79 have mutually unique visible-member GT agreement and
2 are unresolved. This does not prove the whole trajectories are unmixed.
Runtime allocated 138 IDs, absorbed 82, expired 26 and retained 30:
138 = 82 + 26 + 30. The 81 merge events can absorb more than 81 IDs because
an event can join more than two identities. Allocated IDs are not person counts.

The falling no-cross-camera control is still a **global** identity score using
camera-scoped local IDs. It is not a per-camera ByteTrack score, so its decline
alone does not prove a local-tracker failure. Diagnose before changing thresholds.

## Run

From the project root in WSL, with the project environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_mtmc_sequence_diagnostic.py

PYTHONPATH="$PWD/src" python scripts/diagnose_mtmc_sequence.py \
  --evaluation-report artifacts/mtmc_sequence_evaluation/20261007T134209413468Z/report.json
```

This is CPU-only analysis. It loads already recorded boxes and global IDs,
uses GT only for offline diagnostics, and changes no runtime settings or outputs.
It requires exact reproduction of the published global/control metrics and
every growing-prefix metric before writing a completed diagnostic report.

## Three distinct measurements

1. **Framewise spatial F1 ceiling.** Find the maximum number of IoU >= 0.5
   matches independently at each camera/frame using joint maximum-cardinality
   assignment. Disregard temporal and cross-camera identity consistency. This
   GT-assisted quantity describes the best observation matching possible with
   the fixed recorded boxes. It is not detector AP or an achievable runtime goal.
2. **Independent-camera pooled local IDF1.** Compute identity matching separately
   within each camera using original local IDs; then sum IDTP, IDFP and IDFN to
   derive the pooled score. Do not average camera percentages. This answers a
   different question from the global camera-scoped local-ID control.
3. **Global IDF1.** Reproduce the same shared identity mapping over all cameras
   and evaluated frames as the published pipeline evaluation.

The difference between local pooled and global IDF1 is not a causal decomposition
of module errors: these metrics impose different identity constraints. Shared
global IDs can sometimes join fragmented local tracks, while incorrect global
associations can introduce other errors.

## Exact fixed-box accounting

Let S be the sum of independent maximum spatial match counts, G the evaluated
GT observations, P the prediction observations and T the global IDTP:

- fixed-box minimum FN = G - S;
- fixed-box minimum FP = P - S;
- shared-identity assignment gap = S - T;
- IDFN = (G - S) + (S - T);
- IDFP = (P - S) + (S - T).

The identities are exact accounting relations on the fixed evaluation data.
The shared-ID gap can involve temporal/cross-camera mapping constraints and
competing compatible boxes. It is not an attribution of all those errors to
Re-ID, ByteTrack or the global manager individually.

## Evidence for selecting failure examples

Evidence labels are retained only for mutually unique IoU overlaps: exactly
one GT candidate for that prediction and exactly one prediction candidate for
that GT within a camera/frame. Ambiguous and unmatched observations still
contribute to the metrics and ceiling, but not these evidence histograms.

- `gt_fragmentation.csv`: which emitted global IDs have evidence for each GT
  person, including frames with simultaneous splits across cameras;
- `global_mixing.csv`: which GT persons have evidence for each global ID,
  including simultaneous mixing;
- `local_mixing.csv`: which GT persons have evidence for each camera/local ID;
- `local_label_transitions.csv`: changes between successive available evidence
  labels for one camera/local track, including the frame gap;
- `evidence_segments.csv`: contiguous runs of one camera/local ID, GT label and
  global ID. A missing evidence frame breaks a segment;
- `blocked_merge_examples.json`: bounded examples of blocked candidates whose
  listed members have the same unique GT evidence;
- `prefix_diagnostics.csv`: all three measurements and exact decomposition for
  the published growing prefixes;
- `report.json`: reproduced metrics, summaries, counts, hashes and limitations.

These evidence histograms are not IDTP/IDFP/IDFN and label transitions are not
official CLEAR ID switches. Unique spatial matching is not visual proof. Local
mixing could reflect an actual track switch or erroneous spatial labeling;
inspect selected frames before changing the tracker.

Correct causal merges also cause one GT person to appear under several emitted
global IDs because earlier outputs are intentionally not rewritten. Therefore
multiple global IDs per GT are a symptom to inspect, not sufficient proof that
each merge or initial association was wrong.

Blocked-candidate labels cover the listed candidate members. They do not prove
that merging is safe: other visible members or absent retained camera bindings
can correctly prevent the merge. The diagnostic does not use GT to bypass those
checks and does not apply retrospective identity aliases.

All evaluated trace/GT checksums are verified again after analysis. Runtime
inference is not rerun. A separate scene remains necessary for validation;
this diagnostic does not select a deployment threshold.

The identity metric follows the previously validated evaluator and Ristani et
al., *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking*,
ECCV Workshops 2016, https://arxiv.org/abs/1609.01775. The decomposition and
evidence tables are project-specific diagnostics, not additional standard metrics.
