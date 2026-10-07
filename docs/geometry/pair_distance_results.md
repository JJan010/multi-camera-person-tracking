# Geometry pair-distance diagnostic results

Run: `20261007T113317868426Z`.
Scene 001, cameras 4/5/8, frames 2..299.
Source appearance setting: mean descriptors, cosine threshold 0.70.
All results use the same frozen tracked boxes and offline IoU diagnostic
labels. Distances are native dataset coordinate units; physical units have
not been independently established. No identity assignments were changed.

## Population counts

| Population | Same GT | Different known GT | Unknown GT |
| --- | ---: | ---: | ---: |
| All cross-camera pairs | 12031 | 262570 | 5854 |
| Appearance-assigned links | 10507 | 363 | 98 |
| Group-retained links | 10140 | 210 | 81 |

These are repeated pair observations across frames, not independent trials
or numbers of people. Frozen labels, source rows, positive-pair denominators,
appearance-link counts and grouping counts were reproduced. There were no
invalid projections in this run; unavailable-geometry handling was therefore
not exercised by these real observations.

## Distance distributions

| Population / diagnostic class | Median | p95 | Maximum |
| --- | ---: | ---: | ---: |
| All pairs / same | 0.262 | 0.712 | 1.825 |
| All pairs / different | 7.026 | 15.147 | 26.022 |
| Appearance links / same | 0.250 | 0.642 | 1.825 |
| Appearance links / different | 9.552 | 17.567 | 18.141 |
| Grouped links / same | 0.248 | 0.638 | 1.825 |
| Grouped links / different | 15.856 | 17.617 | 18.141 |

Distances provide useful discrimination in this diagnostic. Group support
does not enforce spatial consistency: some retained wrong links remain far
apart on the ground plane.

## Hypothetical filtering of appearance links

Retain finite distance <= the tested value, reject larger distance. Results
below pool all cameras and include all boxes, without a visibility filter.

| Distance | Same retained | Same rejected | Different retained | Different rejected | Unknown rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 5259 | 5248 | 14 | 349 | 65 |
| 0.5 | 9182 | 1325 | 38 | 325 | 31 |
| 1 | 10365 | 142 | 46 | 317 | 21 |
| 2 | 10507 | 0 | 52 | 311 | 8 |
| 3 | 10507 | 0 | 55 | 308 | 7 |
| 5 | 10507 | 0 | 61 | 302 | 5 |
| 10 | 10507 | 0 | 188 | 175 | 0 |

At distance 2, none of the 10507 diagnostically correct appearance links is
rejected, while 311/363 (85.67%) of known-wrong links are rejected. The 98
unknown-label links remain a separate category: 8 rejected and 90 retained.
No conclusion about their correctness follows from that split.

At distance 1, 317 known-wrong links are rejected, but so are 142 correct
links (1.35%). At 0.5, 325 known-wrong links are rejected together with 1325
correct links (12.61%). The very tight 0.25 check rejects nearly half the
correct appearance links.

All diagnostically same-person pairs, including those not accepted by the
appearance assignment, have distance <= 1.825 in this fragment. This is an
observed maximum, not a future error bound. Ground-truth matching is itself
diagnostic and the current observations do not cover all possible occlusion
or camera configurations.

## Next integration experiment

Distance 2 is a candidate for a clearly labelled in-sample integration
experiment, motivated by this inspected diagnostic. It is not an independently
calibrated deployment setting. The diagnostic report itself selected no
threshold. Preserve the original appearance-only and controlled-merge outputs.

The intended next test is a geometric admissibility mask on cross-camera
pairs before joint appearance assignment. This lets assignment account for
forbidden candidates instead of deleting a winner only after the optimization.
Projection failures and unreliable-geometry policy must have an explicit
contract; unknown GT labels must never enter the runtime gate.

Reassignment can change winners, subsequent groups and global identity
history. Therefore the table above cannot predict the resulting global IDF1
or exact new error count. A paired causal replay and shared global identity
evaluation are required. The current controlled baseline remains 85.03%.

Further evaluation should include camera-specific and quality-stratified
results, remaining wrong links and independent recordings. Selecting the
candidate after viewing this reused training-scene fragment makes the next
experiment in-sample, not a generalization test.

Full provenance and stratified summaries are in `pair_distance_metrics.json`.
The copied `pair_distance_summary.csv` and `pair_distance_sweep.csv` preserve
all diagnostic populations and strata. Large observation and pair traces
remain under the ignored timestamped artifact directory.
