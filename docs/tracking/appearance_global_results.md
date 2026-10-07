# Appearance-aware local tracking: downstream global results

Local experiment: `20261007T162104712584Z`.
Global experiment: `20261007T163919105755Z`.
Original pipeline: `20261007T133915524799Z`.
Scene: scene_001, cameras 4/5/8, runtime frames 0..1799,
evaluation frames 2..1799, 30 FPS.

| Variant | Local IDF1 | Local IDSW | Global IDF1 | IDTP | IDFP | IDFN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 64.45% | 120 | 71.75% | 69634 | 26562 | 28281 |
| direct_iou | 64.44% | 120 | 71.75% | 69642 | 26567 | 28273 |
| direct_appearance | 65.56% | 123 | 73.57% | 71413 | 24802 | 26502 |

Global IDF1 computed from the integer counts is 71.74657799%, 71.75001545%,
and 73.57234843%, respectively. The appearance variant improves by
1.82233297 percentage points versus direct_iou and 1.82577044 points versus
the original baseline. Rounding displayed percentages before subtraction
would obscure the latter difference.

The principal controlled contrast is direct_appearance versus direct_iou:
1771 additional identity-correct observations, 1765 fewer IDFP and
1771 fewer IDFN. Evaluation GT observations remain 97915. Prediction counts
are respectively 96215 and 96209, so denominators are not identical.
Runtime observation counts (including frames 0 and 1) are 96301 and 96295.

| Variant | Allocated | Absorbed | Expired | Retained at end | Merge events |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 138 | 82 | 26 | 30 | 81 |
| direct_iou | 138 | 82 | 26 | 30 | 81 |
| direct_appearance | 137 | 84 | 23 | 30 | 83 |

Accepted merge evidence: baseline and direct_iou each have 79 events whose
visible members share one known GT identity and 2 unresolved events.
The appearance variant has 81 such same-GT events and 2 unresolved events.
No event was classified as different_known_gt. This is evidence about the
current visible members, not a guarantee of identity purity across history.

The inspected camera-5/local-11 takeover is prevented by the appearance
variant. The aggregate improvement is modest and does not establish that
all remaining switches or fragmentation have been solved. The experimental
tracker is not yet promoted to the runtime baseline.

## Next experiment: frozen temporal transfer

Before inspecting the next interval, retain the same three variants and all
current settings. In particular, local appearance threshold=0.6, the original
local gallery rules, and the original downstream history, geometry, grouping,
identity expiry and merge settings must remain unchanged.

Collect an unchanged full-pipeline run from frame 0 through 3599. Running
from frame zero preserves causal state and avoids introducing a cold start
at the interval boundary. This run provides the longer frozen detector trace;
it does not activate the experimental local tracker.

Next, cache candidate embeddings and replay all three local/global variants
on the same new trace. Verify the shared first-minute inputs against the
earlier run. Any differences must be reported before attributing differences
to the local appearance rule.

The primary new interval will be frames 1800..3599 (scene times 60..120 s,
right endpoint excluded). Compute local and shared global identity metrics
separately on this interval, with fresh metric accumulators but without
resetting the trackers or global identity managers at frame 1800. Also retain
full-prefix metrics over frames 2..3599 to measure continuity across the boundary.
The separate interval and full prefix answer different questions; neither
should silently replace the other.

The existing cumulative sequence evaluator is sufficient for collecting and
checking the unchanged baseline, but its cumulative scores alone are not the
planned interval comparison. That comparison follows after candidate replay.

This is an unseen temporal interval from the same scene and people, not an
independent-scene generalization test. An additional scene/camera validation
is still needed. Do not tune thresholds after inspecting this interval and
then continue calling it held-out validation.

No end-to-end throughput claim is made for the experimental tracker: encoding
all candidate detections changes the workload and must be benchmarked when
integrating the selected variant into the runtime pipeline.
