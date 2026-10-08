# Dormant recovery: fixed-radius development result

Source experiment: `20261008T094135187275Z`.
Offline context diagnostic: `20261008T095643711175Z`.
Scene: scene_001, cameras 4/5/8, runtime frames 0..3599,
evaluation frames 2..3599. Local variant: staged; history: all updates.

## Hypothesis and controls

A newly unanchored group may recover a fully inactive global ID using a bounded
archive. Settings were declared before the run: age <= 5 seconds, cosine > 0.8,
mutual ambiguity margin >= 0.05, fixed radius <= 2 native scene units,
capacity 128 identities. No physical speed or metre interpretation was assumed.

The disabled branch reproduced the frozen control exactly. Both branches used
identical local tracks, appearance histories, geometry and current grouping.
GT was loaded only after both causal prediction streams were frozen. No
threshold sweep, runtime state reset, model inference or decoding was performed.

## Identity metrics

| Window | Disabled IDF1 | Enabled IDF1 | Disabled IDTP/IDFP/IDFN | Enabled IDTP/IDFP/IDFN |
|---|---:|---:|---|---|
| Full | 53.03% | 53.03% | 96696/83765/87511 | 96696/83765/87511 |
| First minute | 73.57% | 73.57% | 71413/24802/26502 | 71413/24802/26502 |
| Second minute | 60.64% | 60.84% | 51709/32537/34583 | 51879/32367/34413 |

The full result is exactly unchanged, not merely equal after rounding. Each
window solves its own shared identity assignment; the 170 additional IDTP in
the second window cannot be added to the full-run result. These counts alone
do not determine each return's individual contribution or downstream effects.

## Recovery and lifecycle

Five returns were accepted; 234 birth queries had no eligible archived identity.
There were 84 retirements, 77 archive expirations, no skipped retirements or
capacity evictions, and two archived IDs at the end. Ever-emitted IDs fell from
267 to 262; allocator slots stayed at 267 because five provisional new IDs were
superseded and never reused. Retained IDs stayed at 28 and merge events at 151.
Inactive identity counts (84 versus 79) are distinct from archive occupancy.

## Offline event evidence

| Global ID | Camera | Local IDs | Gap | Endpoint evidence | Half-second context |
|---|---:|---|---:|---|---|
| 154 | 5 | 72 -> 46 | 38 frames | GT 9 -> 9 | 5 unique GT-9 samples before, 16 after |
| 186 | 4 | 65 -> 69 | 94 frames | Unknown -> unknown | No admissible GT for 3 visible samples on either side |
| 175 | 5 | 84 -> 96 | 46 frames | Unknown -> GT 8 | 8 unique GT-8 samples before, 16 after |
| 202 | 4 | 71 -> 74 | 31 frames | Unknown -> GT 5 | 12 unique GT-5 samples before, 14 after |
| 213 | 8 | 85 -> 91 | 38 frames | GT 14 -> 14 | 16 unique GT-14 samples on either side |

All five events connect tracks within the same camera. Thus this sample gives
no demonstrated cross-camera dormant return. Same-person continuity is
supported by local context in four events, but this is not a calibrated return
accuracy estimate or a certificate of archived descriptor purity.

For GID 175 the nearest known earlier label is GT 8 at frame 2562, eight frames
before its last visible observation. For GID 202 it is GT 5 at frame 2641, two
frames before the last observation. Context does not relabel those unresolved
endpoints, and future observations must never enter causal runtime decisions.

GID 186 has small raw boxes at the bottom image boundary (approximately
36x13 and 53x29 pixels). Both endpoint boxes have zero IoU with all usable GT
boxes. This identifies weak diagnostic evidence; it does not establish the
visual contents, a GT error, or a correct/incorrect person return. No visual
review of these two endpoint images was performed for this result.

## Decision

Keep staged/all-updates with dormant recovery disabled as the reference policy.
Retain the archive implementation and this experiment for reproducibility, but
do not promote it as an accuracy improvement. Capacity was not limiting, so
increasing it is unsupported by this run. Do not tune age, radius or appearance
thresholds to these five events.

This birth-time recovery mechanism does not repair identity takeover in a
still-active local track, nor does it revisit already anchored return groups.
Those are separate causal continuity problems. Further quality work should
address them as separate controlled hypotheses, with paired full-sequence
metrics and subsequent unchanged-policy validation on another scene.

No independent-validation or end-to-end performance claim is made here. The
training-scene fragment has already been used repeatedly during development.
