# Controlled merge: paired results and observed failure

Run: `20261007T104647560154Z`.
Source baseline evaluation: `20261007T094804760266Z`.
Scene 001, cameras 4/5/8, frames 2..299, 30 FPS.
Policy: `whole_visible_group_confirmed_merge_v1`.
Appearance: mean descriptors, cosine threshold 0.70.
Binding idle duration: 1 second.
Confirmation: at least 3 supporting rounds AND 1/5 second of scene time;
maximum evidence gap 1/10 second. These are integration settings.

## Paired quality

| Metric | No-merge baseline | Controlled merge |
| --- | ---: | ---: |
| Global IDF1 | 72.73% | 85.03% |
| IDP | 73.86% | 86.34% |
| IDR | 71.64% | 83.75% |
| IDTP | 11690 | 13666 |
| IDFP | 4138 | 2162 |
| IDFN | 4627 | 2651 |

The gain, computed before rounding, is 12.29 percentage points of IDF1.
There are 1976 additional identity-correct observations under the shared
identity metric. Both variants use the same 15828 predictions and 16317 GT
observations. Detection, local tracking, boxes and grouping inputs are fixed.
Full baseline output records and metrics were reproduced. Outputs were frozen
before quality evaluation; GT was not passed to either runtime manager.
Earlier assignments were not rewritten after a merge.

## Lifecycle and decisions

- Allocated and ever-emitted global IDs: 49.
- Absorbed IDs: 19; expired IDs: 3; retained IDs at the end: 27.
- Lifecycle accounting: 49 = 19 + 3 + 27.
- Accepted merge events: 19.
- Decision counts: 223 retained-camera conflicts, 57 incomplete-visible-support
  rejections, 153 pending decisions, 19 confirmations.
- Confirmation-state clearing: 19 after acceptance, 12 after a round without
  eligible support.

Decision counts are repeated round-level events, not counts of independent
people or candidates. Retained identities include temporarily absent tracks;
their count is not the number of true people.

At acceptance time, mutually unique spatial GT evidence labels 18 merge
events as all-visible-members-same-GT and one as different-known-GT. None is
unresolved. This describes visible evidence at acceptance, not the correctness
of complete identity histories or absent retained members.

## The incorrect merge

Frame 149, timestamp 149/30 seconds: global ID 26 was absorbed into ID 13.
Support began at 143/30 seconds and lasted 7 rounds spanning exactly 1/5
second. The retained memberships before the merge were:

| Global ID | Camera | Local ID | Diagnostic GT at acceptance |
| --- | ---: | ---: | ---: |
| 13 | 5 | 9 | 24 |
| 26 | 8 | 17 | 23 |

The frozen trace shows:

| Camera/local ID | Frames | Output global ID | Source diagnostic GT |
| --- | --- | ---: | ---: |
| 5/9 | 2..299 | 13 | 24 |
| 8/17 | 2..148 | 26 | 23 |
| 8/17 | 149..299 | 13 | 23 |

The source diagnostic GT labels stay constant for both local tracks throughout
the inspected range. This supports interpreting this event as a cross-camera
association error between two stable local tracks, rather than a local
identity switch in these tracks. Source labels come from spatial matching;
this statement does not replace a visual audit.

The two different GT people share global ID 13 for all 151 rounds from frame
149 through 299 inclusive. No retrospective ID replacement occurs: camera
8/local 17 keeps ID 26 in the earlier saved output. The failure passes the
policy because the cameras differ, both retained identities contain only the
listed member, and incorrect support persists long enough. The manager has
no mechanism to split the mixed identity in response to later evidence.

The 151-round duration is not an isolated IDFP/IDFN contribution. Shared
identity matching depends on the complete sequence; attributing a metric
change to this single event would require a separate counterfactual replay.

## Conclusions and next step

Controlled merging improves this fragment substantially, but temporal support
and camera uniqueness are insufficient to reject a persistent incorrect
appearance association. Increasing the confirmation duration alone is not
a correctness guarantee. This result does not establish a deployment setting.

Keep this run and the original no-merge baseline frozen. A next development
step is to validate the scene calibration and investigate spatial consistency
as an additional source of evidence. Calibration direction, ground-plane
projection and uncertainty must be checked before geometry can gate matches.
Do not assume geometry will resolve this event without measuring it.

These experiments reuse a short training-scene fragment. Independent
validation remains necessary. The 18 apparently consistent merge events do
not establish general merge precision or whole-history correctness.

Full experiment provenance and settings are in
`controlled_merge_metrics.json`; all accepted events with offline labels are
in `controlled_merge_events.json`. Large assignment traces remain in the
ignored artifact run directory.
