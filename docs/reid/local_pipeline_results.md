# First continuous local tracking and Re-ID run

Run: `20261006T204237314810Z`, executed by the user on the RTX 5080 in WSL.
The pipeline processed frames 0–299 from cameras 4, 5 and 8 and produced 15,914
normalized 512-dimensional appearance vectors. All rounds completed and no
crop was skipped as fully outside. No global identity association was run.

## Evidence and file locations

- [Original pipeline report](benchmarks/local_pipeline_fp32.json), including
  raw timing samples, source hashes, runtime versions and artifact hashes.
- [Trace comparison](local_pipeline_trace_comparison.json), including both
  trace hashes and the frames where output counts differ.
- [New local tracking evaluation](local_pipeline_evaluation.json), copied from
  the user's evaluator output for this run.

The input code's Git HEAD was
`bf5e26eaece918e2f87ff5d791e24ab522962d77`, with the six new integration files
still untracked. The report's seven source hashes match the supplied module
and runner files. A subsequent commit records this integration and its results.

The older trace is `artifacts/tracking_preview/tracks.jsonl`; the new trace is
`artifacts/local_reid/20261006T204237314810Z/tracks.jsonl`. They are independent
files. The runner intentionally creates a new run directory and does not
overwrite the older baseline. The uploaded new trace has SHA-256
`2a9b529c1a0e6cb6202d6467d64214a917d9d2403a7be201fcfdc77405faa736`, matching
the pipeline report. Its 15,914 encoded observations refer to every matrix row
0–15,913 exactly once. This checks row coverage, not the actual matrix contents;
the matrix was validated by the runner in WSL and was not uploaded here.

## Initial integrated performance

All 300 rounds were saved; timing summaries exclude warmup frames 0–29.
The 270 measured rounds represent nine seconds of source video and took
22.4227 seconds, or 12.0413 rounds/s and 36.1240 images/s across three cameras.
This is about 12 frames/s per camera, not 36 frames/s per camera.

| Phase | Mean ms per round | Share of core time |
| --- | ---: | ---: |
| Replay/decode/RGB | 8.468 | 10.26% |
| Detector adapter | 23.189 | 28.11% |
| Local trackers | 2.727 | 3.31% |
| Crop builder | 0.657 | 0.80% |
| Complete OSNet adapter | 47.469 | 57.53% |
| Core total | 82.510 | 100% |
| Recording work, outside core | 0.483 | — |

Core p95 was 94.653 ms. Measured rounds contained 44–59 crops, mean 54.144.
OSNet's adapter time includes CPU preprocessing and transfers, not just the
network forward pass. It is the largest measured phase in this configuration.

PyTorch's measured peak allocated memory was 853.719 MiB and peak reserved
memory was 7,186 MiB. Reserved memory includes allocator caches; these maxima
alone neither explain the reservation nor establish a leak. No allocator
tuning or per-round cache clearing was performed.

This is one integration run with both models active and variable Re-ID batches.
It does not establish an acceleration ratio versus the fixed-batch OSNet
microbenchmark or the older detector benchmark. The process used one CPU
intra-op thread. See [the protocol](local_pipeline.md) for synchronization,
warmup, storage and filesystem-cache limits.

## Local tracking quality

Both evaluations use scene_001, cameras 4/5/8, frames 2–299 and the same local
IoU-based evaluator. Percentages below are the rounded console values supplied
by the user. The older baseline remains in `docs/tracking/baseline_metrics.json`.

| Sequence | Previous IDF1 | Integrated IDF1 | Previous IDSW | Integrated IDSW |
| --- | ---: | ---: | ---: | ---: |
| Camera 4 | 95.5% | 95.5% | 1 | 1 |
| Camera 5 | 91.7% | 93.1% | 7 | 6 |
| Camera 8 | 90.5% | 88.3% | 8 | 10 |
| All cameras, local identities | 92.3% | 92.1% | 16 | 17 |

Total FP changed from 171 to 167 and FN from 668 to 656. Fewer observation
errors therefore coexist with slightly worse identity consistency. These are
local metrics, not cross-camera/global IDF1.

Raw detector outputs differ between runs from frame 0 in every camera. The
number of detections differs in 24, 53 and 54 frames for cameras 4, 5 and 8,
respectively. This is not a pure experiment changing only Re-ID while holding
all detections fixed. The tracker module source is unchanged, and the runner
does not feed embeddings back into ByteTrack. The root cause of the detector
differences has not been isolated; do not attribute the metric change to OSNet.

Inspection of the traces with the previously uploaded GT slice found:

- Camera 4: GT 18 is missing from the new visible output at frame 38, while the
  old trace contains local ID 13 there. This accounts for the one additional FN.
- Camera 5: GT 21 keeps local ID 22 in the new run. The old run loses spatial
  matching around frames 220–230 and uses a new ID 26 from frame 231. The new
  run also matches GT 4 at frames 248–249 where the old run does not.
- Camera 8: around frame 214, local IDs 10 and 12 become assigned differently
  to GT 9 and GT 18. The new evaluation reports two additional identity switches.

These examples use independent frame-wise matching after clipping boxes, with
IoU >= 0.5, maximum valid-match cardinality and then maximum summed IoU.
They explain specific trace differences; they do not replace the continuity-
aware motmetrics evaluation. Numeric local IDs can also shift after a new
track is created; label-number differences alone are not errors.

## Decision

Keep both baselines and record this run as the first integrated reference.
Use its frozen detections, local tracks and embeddings for upcoming association
experiments so changes in appearance history/global logic can be compared on
the same observations. Local ID switches remain a known source of mixed-person
appearance history. No thresholds were tuned to improve this integration scene.

## References

- [PyTorch 2.11 memory management](https://docs.pytorch.org/docs/2.11/notes/cuda.html#memory-management)
  explains allocated versus reserved memory and caching.
- [Ristani et al., Performance Measures and a Data Set for Multi-Target,
  Multi-Camera Tracking, 2016](https://arxiv.org/abs/1609.01775) discusses
  identity-based tracking evaluation. The observations and numbers above are
  project measurements, not results reported in that paper.
