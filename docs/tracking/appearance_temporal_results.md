# Appearance-aware tracking: frozen temporal transfer results

Global/window evaluation: `20261007T174247370251Z`.
Local experiment: `20261007T171955616342Z`.
Two-minute pipeline: `20261007T164721073889Z`.
Candidate cache: `20261007T165444228264Z`.
Exact shared-prefix audit: `20261007T171712998154Z`.
Exact installed-ByteTrack replay: `20261007T171726125980Z`.

Scene_001, cameras 4/5/8, runtime frames 0..3599 at 30 FPS.
The second interval was evaluated with the previously frozen local appearance
threshold 0.6 and unchanged downstream settings. Trackers, histories and global
identity managers ran continuously from frame zero. GT was used only for offline
evaluation of frozen predictions.

## Principal controlled contrast

Both direct variants use the same candidate mapping and cached appearance
features. The appearance compatibility rule is enabled only in
`direct_appearance`.

| Evaluation range | direct_iou global IDF1 | direct_appearance global IDF1 | Delta, pp |
| --- | ---: | ---: | ---: |
| First minute, frames 2..1799 | 71.75% | 73.57% | +1.82 |
| Second minute, frames 1800..3599 | 53.86% | 60.64% | +6.78 |
| Full sequence, frames 2..3599 | 48.94% | 53.03% | +4.09 |

The second-minute global IDTP increases from 45928 to 51709 (+5781).
Full-sequence global IDTP increases from 89242 to 96696 (+7454).
This supports temporal transfer of the observed benefit within this scene.
It does not establish independent-scene generalization or statistical
significance across recordings.

## Detailed identity counts

| Range | Variant | IDTP | IDFP | IDFN |
| --- | --- | ---: | ---: | ---: |
| First | baseline | 69634 | 26562 | 28281 |
| First | direct_iou | 69642 | 26567 | 28273 |
| First | direct_appearance | 71413 | 24802 | 26502 |
| Second | baseline | 45928 | 38286 | 40364 |
| Second | direct_iou | 45928 | 38327 | 40364 |
| Second | direct_appearance | 51709 | 32537 | 34583 |
| Full | baseline | 89232 | 91178 | 94975 |
| Full | direct_iou | 89242 | 91222 | 94965 |
| Full | direct_appearance | 96696 | 83765 | 87511 |

Every variant is evaluated against the same GT population in a given interval,
but prediction counts can differ. Interval identity assignments are solved
separately. ID scores and identity TP/FP/FN are not additive across intervals.

## Complementary local and lifecycle evidence

| Second-minute metric | direct_iou | direct_appearance |
| --- | ---: | ---: |
| Pooled local IDF1 | 59.18% | 60.15% |
| Local ID switches | 169 | 178 |

Full-run allocated global IDs increase from 253 to 267. Absorbed IDs increase
from 150 to 155, expired IDs from 75 to 84, and merge events from 145 to 151.
Both finish with 28 retained IDs. These counters are not person counts and
are not quality objectives by themselves.

Full-run accepted merge evidence changes from 142 to 148 events whose visible
members share one known GT identity; both have 3 unresolved events. No event
was classified as different_known_gt. These diagnostics do not establish
historical purity of a global identity or rule out later local-ID takeovers.

## Remaining continuity problem

For direct_appearance, the sum of independently matched interval IDTP is:

`71413 + 51709 = 123122`.

One shared assignment over both intervals yields only `96696` IDTP. The
difference is `26426` observations, equivalent to approximately 14.49 IDF1
percentage points with this variant's full-run GT/prediction denominator.
The analogous count gaps are 26330 for baseline and 26328 for direct_iou.

This quantifies incompatibility between the best separate-interval identity
assignments and a single full-sequence assignment. It is not a count of
switch events, not proof that errors occur at frame 1800, and not a guarantee
that a particular runtime change can recover all of these observations.
Fragmentation and identity mixing can both contribute. Inspect frozen
trajectories and mappings before changing lifecycle or merge rules.

## Decision and next step

Retain direct_appearance as a promising experimental candidate, with the
original baseline available for comparison. Do not select a new threshold
from this result. Full-run global IDF1 of 53.03% still leaves substantial
identity errors; the experiment does not close quality work.

Checkpoint the code, frozen metric summaries and this result. Next, diagnose
long-term identity continuity on the existing frozen outputs, distinguishing
fragmentation, local-track takeovers and identity mixing. Do not infer that
increasing the idle timeout or accepting more merges is necessarily beneficial.
Any subsequent tuning on the second minute turns it into development data;
further confirmation must then use an untouched interval or another scene.

There is no end-to-end speed result for the experimental variant yet. The full
detector-candidate encoder workload differs from encoding only returned tracks.
Runtime integration and later TensorRT/NVDEC optimization require paired
quality and throughput measurements.

Reference for identity-based MTMC evaluation: Ristani et al., *Performance
Measures and a Data Set for Multi-Target, Multi-Camera Tracking* (2016),
https://arxiv.org/abs/1609.01775.
