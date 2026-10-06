# Causal appearance history: integration results

## Experiment and preserved inputs

The frozen local pipeline run is `20261006T204237314810Z`, with 300 rounds,
cameras 4/5/8, and 15,914 normalized OSNet FP32 observation embeddings.
History replay `20261006T211035258446Z` stores at most eight observations per
camera/local ID, with maximum age one scene second, including the current
observation. At continuous 30 FPS, eight observations span 7/30 seconds.

The paired evaluation `20261006T212226930067Z` uses frames 2..299 and all six
directed camera pairs. Latest/latest and mean/mean have identical observations,
GT assignments, galleries and eligible queries. No GT is used to construct or
reset descriptors, and no similarity threshold or crop-quality filtering is applied.

The full reports are preserved byte-for-byte:

- `appearance_history_metrics.json`: evaluation protocol, hashes and metrics.
- `benchmarks/appearance_history_cpu.json`: history construction and CPU timing.

See `appearance_history_evaluation.md` for eligibility, matching and ranking details.
The uploaded reports were checked against the available frozen trace, source
report, history report, dataset manifest and evaluator source hashes. Directed
and conditional counts reconcile with the pooled paired counts. Full embedding
and GT checksums were verified by the evaluator on the WSL machine.

## Retrieval results

| Metric | Latest | Mean | Change |
| --- | ---: | ---: | ---: |
| Rank-1 | 22,236 / 24,062 (92.4113%) | 22,427 / 24,062 (93.2051%) | +0.7938 pp |
| Rank-3 | 23,443 / 24,062 (97.4275%) | 23,555 / 24,062 (97.8929%) | +0.4655 pp |

Among eligible directed queries, 21,893 are Rank-1 correct for both variants,
534 improve, 343 worsen, and 1,292 are wrong for both. The net improvement is
191 queries. The Rank-1 error count falls from 1,826 to 1,635, a 10.46% relative
reduction on this clip. This is not a 10.46 percentage-point accuracy gain.

| Query camera -> gallery camera | Eligible | Latest Rank-1 | Mean Rank-1 | Change pp |
| --- | ---: | ---: | ---: | ---: |
| 4 -> 5 | 3,724 | 98.09% | 98.36% | +0.27 |
| 4 -> 8 | 4,088 | 84.78% | 84.88% | +0.10 |
| 5 -> 4 | 3,724 | 96.59% | 96.83% | +0.24 |
| 5 -> 8 | 4,219 | 93.86% | 94.93% | +1.07 |
| 8 -> 4 | 4,088 | 89.82% | 91.54% | +1.71 |
| 8 -> 5 | 4,219 | 92.15% | 93.41% | +1.26 |

Each direction improves on aggregate. However, 4 -> 8 contains 148 improvements
and 144 regressions, giving only four net extra correct queries. Symmetric pairwise
similarity does not imply symmetric retrieval: reversing the cameras changes the
query and gallery sets and the competitors for each positive match.

There are 31,656 total directed queries: 24,062 eligible, 324 without an assigned
query label, and 7,270 with an assigned query label but no assigned positive in
the target gallery. Rank-1 does not evaluate rejection of those absent positives.
These counts reuse observations across frames and directions; they are not counts
of distinct people or independent trials.

## History conflicts and regressions

The following groups partition eligible queries. A conflict means the query or
its correct gallery observation has a history member with a different assigned
GT identity. The consistent group requires all members of both histories to
have the current assigned identity. Unknown members remain explicit.

| Query / positive history group | Queries | Latest Rank-1 | Mean Rank-1 | Change pp | Improved / worsened |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both histories consistent | 23,204 | 92.97% | 93.86% | +0.89 | 488 / 281 |
| At least one conflict | 222 | 73.42% | 62.61% | -10.81 | 17 / 41 |
| Unknown members, no known conflict | 636 | 78.62% | 79.87% | +1.26 | 29 / 21 |

The conflict group has 24 net fewer correct queries under averaging. This supports
keeping mixed histories as an explicit failure mode. It does not explain all
regressions: 281 of 343 occur in the group with consistent query/positive labels.
The cause of each regression has not been established from this summary report.
Distractor descriptors also change between variants, and an IoU-consistent label
does not establish the visual quality of the crop.

There are 15,828 current observations in evaluated frames: 15,195 have consistent
history labels, 77 have a conflict with the current label, 394 have unknown members
without a known conflict, and 162 have no assigned current label. These are
observation counts, not numbers of people or motmetrics ID-switch events.
Eighty histories contain multiple known GT IDs; three of them have an unmatched
current observation and therefore fall outside the 77 current-label conflicts.

For example, camera 8 / local ID 10 at frame 214 has current assigned GT ID 18,
but history labels `[9, 9, 9, 9, 9, 9, null, 18]`. The mean contains six vectors
associated with ID 9 and one associated with ID 18, plus one with an unknown
label. At this frame the current IoU is approximately 0.506 and there are two
GT candidates above the gate, so the assignment is itself diagnostically ambiguous.
Keep the full fixed examples in the report rather than treating this as a clean
causal experiment isolating the source of all errors.

## Construction correctness and cost

All 15,914 observations received descriptors. Latest vectors match source vectors
exactly; causality, age limits and normalization passed. There were no cancellation
fallbacks. History sizes 1..7 each occur 73 times; size 8 occurs 15,403 times.

One CPU replay measured history-update mean 1.144 ms, median 1.112 ms and p95
1.371 ms per round across the three cameras. Peaks were 62 cached tracks and
484 vectors, equivalent to 0.945 MiB of vector payload. This excludes Python
metadata, offline trace loading and other arrays. It is not total process memory.
The timings exclude surrounding validation/serialization and are not a new
end-to-end pipeline benchmark.

Latest-to-mean cosine has minimum 0.66364, median 0.98316 and p05 0.93400.
It measures descriptor change, not probability of a correct identity.

## Working decision and next experiment

Keep `mean` as a provisional working variant for the first association experiment
and retain `latest` as the paired reference. Preserve the current eight-observation,
one-second policy without tuning it against this integration clip. Keep the frozen
inputs and both output variants so later association decisions can be compared
without rerunning detection, tracking or OSNet.

Do not reset histories using GT labels in normal operation. The next association
stage must allow unmatched observations, preserve session/camera/local-ID scope,
and evaluate global identity consistency over time. The 7,270 queries without an
assigned gallery positive make unconditional nearest-neighbor merging unsuitable
as the final decision rule. Similarity gating and other association choices need
their own explicit evaluation, with separate data for tuning and final assessment.

This short synthetic integration clip does not establish generalization, statistical
significance, performance on disjoint camera views, or global IDF1. A gain in
framewise Rank-1 alone does not demonstrate stable global IDs.
