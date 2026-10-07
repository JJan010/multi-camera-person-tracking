# Two-minute appearance experiment and disjoint-window evaluation

Source local report: `20261007T171955616342Z`.
Source full pipeline: `20261007T164721073889Z`.
Scene_001, cameras 4/5/8, runtime frames 0..3599, 30 FPS.

## Observed local results for frames 2..3599

| Variant | Local IDF1 | Precision | Recall | IDSW | Framewise FP | Framewise FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 43.83% | 98.29% | 96.26% | 312 | 3093 | 6890 |
| direct_iou | 43.82% | 98.27% | 96.28% | 312 | 3114 | 6857 |
| direct_appearance | 44.86% | 98.28% | 96.28% | 324 | 3105 | 6851 |

Compared with direct_iou, direct_appearance adds 1894 pooled local identity
true positives (79904 -> 81798), with an IDF1 gain of approximately 1.04
percentage points. Camera 5 and camera 8 improve; camera 4 is essentially
unchanged. Total distinct camera-local IDs increase from 293 to 307. These
are mixed aggregate results, not proof of improvement in the unseen interval.

The first 1800 frames of the original long pipeline/cache have already been
verified exactly against the short run, including the source records, global
decisions, 96282 tracked/mean embeddings and 113250 candidate embeddings.
Prefix audit: `20261007T171712998154Z`.
All 10800 baseline camera updates reproduce exactly in frozen replay:
`20261007T171726125980Z`.

## Extended evaluator

`evaluate_appearance_global.py` retains its existing full-run replay and
evaluation when called without the new options. With both `--split-frame`
and `--reference-global-report`, it additionally evaluates disjoint intervals
from its frozen outputs. It changes no tracker or global identity algorithm.

The global managers and appearance histories run continuously from frame zero.
GT is not supplied to them. The resulting predictions are frozen before scoring.
The evaluator then uses fresh metric accumulators for each interval:

- First interval: frames 2..1799. Frames 0 and 1 have no evaluation GT.
- Second interval: frames 1800..3599.
- Full-run metrics continue to cover frames 2..3599.

Every interval has separate per-camera local metrics and one shared global
identity assignment across all its camera/frame slots. Spatial matching,
clipping and the inclusive IoU >= 0.5 gate are unchanged. Global ID metrics
are checked against motmetrics in every variant and interval.

Additional consistency checks require:

- Identical local and global settings to the completed short experiment.
- Exact first-interval local output records for all three variants, including
  candidate index and embedding-row provenance.
- Reproduction of the first-interval local and global metrics from the short
  experiment.
- Additive frame/GT/prediction observation counts across the two intervals.

IDTP/IDFP/IDFN and IDF1 are not obtained by subtracting prefix results: each
interval solves its own identity assignment. Local window IDSW starts with
empty evaluator history, so a switch across the interval boundary is not
counted by the isolated second-window metric. Full-run scores remain necessary
to measure continuity across that boundary.

## Run in WSL from the project directory

```bash
PYTHONPATH="$PWD/src" python scripts/check_tracking_windows.py

PYTHONPATH="$PWD/src" python scripts/evaluate_appearance_global.py \
  --local-report artifacts/appearance_bytetrack/20261007T171955616342Z/report.json \
  --baseline-evaluation artifacts/mtmc_sequence_evaluation/20261007T165247973222Z/report.json \
  --split-frame 1800 \
  --reference-global-report artifacts/appearance_global/20261007T163919105755Z/report.json
```

The reference-global report points to the previous short local report and its
frozen outputs, which must remain available with their original hashes.
All three full-run and interval evaluations use CPU; models and video decoding
are not run. Results go to a new `artifacts/appearance_global/<run_id>/` directory.
The report adds `temporal_evaluation.windows.first` and `.second`, each containing
per-variant `local` and `global` metrics. No previous report is overwritten.

## Validation and interpretation

The synthetic check demonstrates perfect scores in each separate interval but
50% global IDF1 over their concatenation when an ID changes at the boundary.
It also checks a switch inside the second interval, unchanged first-interval
results, exact local-prefix comparison and rejected prefix alterations.
A 300-round technical replay with synthetic appearance features/calibration
exercises the optional CLI path and first-prefix metric reproduction. Its
scores are not performance results for the user's models.

Keep local appearance threshold 0.6 and all other settings fixed. The primary
comparison is direct_appearance versus direct_iou in the second interval.
Treat full-run quality, per-camera results and switches as complementary
evidence, not interchangeable measures. Report deterioration if it occurs.
The same scene and people appear in both intervals, so this is temporal
transfer, not independent-scene validation. No production threshold or runtime
promotion is selected by this experiment.
