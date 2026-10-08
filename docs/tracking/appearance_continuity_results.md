# Appearance continuity: development result

Source: artifacts/appearance_continuity_experiment/20261008T105229140571Z/report.json.
Scene: MTMC_Tracking_2024/train/scene_001; cameras 4, 5, 8.
Runtime: frames 0..3599; quality evaluation: frames 2..3599.
Local observations: frozen staged tracker output; appearance history: all updates.
Dormant recovery: disabled. Runtime baseline: unchanged.

| Evaluation interval | Disabled global IDF1 | Enabled global IDF1 |
| --- | ---: | ---: |
| Full, 2..3599 | 53.03% | 58.79% |
| First, 2..1799 | 73.57% | 78.87% |
| Second, 1800..3599 | 60.64% | 68.52% |

Full-run identity counts:

| Variant | IDTP | IDFP | IDFN |
| --- | ---: | ---: | ---: |
| Disabled | 96696 | 83765 | 87511 |
| Enabled | 107199 | 73262 | 77008 |

The shared full-sequence assignment gains 10503 identity-correct observations;
IDFP and IDFN each decrease by the same amount. Boxes, confidence, feature values,
original local tracker IDs and observation denominators remain unchanged.
The full-run IDF1 gain is approximately 5.76 percentage points.
Window scores use separate identity assignments and must not be averaged into
the full-run score.

There are 30 confirmed splits. Comparing the actual prior reference evidence
with the current observation gives:

- 10 events with different known reference/current GT identities;
- 6 events with the same known reference/current GT identity;
- 12 events with unresolved reference or current evidence;
- 2 events whose reference has mixed known GT identities.

These categories are diagnostic evidence, not a split-accuracy estimate. Same-GT
cuts indicate a risk of unnecessary fragmentation. A different-GT reference does
not alone prove the whole downstream identity is correct. Unknown and mixed
cases remain unresolved. Event evidence is not a counterfactual measurement of
each split's contribution to the aggregate improvement.

Allocated/emitted global IDs increase from 267 to 291, while absorbed IDs increase
from 155 to 170. Expired IDs increase from 84 to 94; merge events from 151 to 165;
retained IDs at the end change from 28 to 27. ID count alone is not quality: cuts
may prevent a contaminated local track from continuing to carry an old identity.
They can also fragment a correct track. The evaluated identity metrics capture
part of that tradeoff over this fixed sequence.

The disabled full trace, lifecycle and metrics reproduce the frozen reference.
Enabled/disabled full and window metrics agree with motmetrics. Predictions are
frozen before offline GT evaluation. These are development results on a repeatedly
inspected two-minute fragment, not independent evidence of generalization.

Next: transfer the exact saved policy and code to scene_041, using its frozen
staged observations. Do not retune thresholds from these 30 events, combine with
dormant recovery, or promote the policy to the default runtime yet.
