# Sparse-reference appearance continuity hypothesis

Status: synthetic contract verified; real frozen replay integration and quality
evaluation pending. No change to the default runtime. This builds on the explicit
track-segment mechanism, using original keys and candidate features from the
staged local tracker. Dormant recovery remains disabled.

## Why a separate reference?

The staged local tracker already compares high-confidence candidates with a
short strong-appearance history. Applying the same mean and threshold again to
its selected outputs would largely duplicate that existing gate. This experiment
instead maintains an independent, more slowly sampled reference. It does not
change the original local tracker or the normal eight-observation global history.

The policy is inspired by appearance-based association in Deep SORT [1] and the
combination of tracking cues in BoT-SORT [2], but is NOT an implementation of
either paper's method. Sparse-reference segmentation and its numerical settings
below are our unvalidated development hypothesis. Neither paper establishes
their effectiveness for these recordings.

1. Wojke, Bewley, Paulus (2017), *Simple Online and Realtime Tracking with a Deep
   Association Metric*: https://arxiv.org/abs/1703.07402
2. Aharon, Orfaig, Bobrovsky (2022), *BoT-SORT: Robust Associations Multi-Pedestrian
   Tracking*: https://arxiv.org/abs/2206.14651

## Declared single experiment

Configuration: `configs/tracking/appearance_continuity_experiment.json`.

| Parameter | Value and boundary |
|---|---|
| Reference capacity | Last 8 accepted, temporally sampled features |
| Minimum sampling interval | >= 0.2 seconds of scene time |
| Reference age | <= 3 seconds; never refreshed by merely reusing a sample |
| Bootstrap | At least 3 sampled features before comparisons can start |
| Detector confidence | >= 0.5 for sampling or support |
| Contradiction | Current cosine to prior reference < 0.6 |
| Candidate coherence | Cosine to the first pending candidate >= 0.8 |
| Confirmation | At least 3 support rounds AND span >= 0.2 seconds |
| Maximum support gap | <= 0.1 seconds between consecutive support calls |

At uninterrupted 30 FPS the time condition requires seven supporting
observations, not three. Eight references sampled exactly every 0.2 seconds
span 1.4 seconds; three seconds is the maximum permissible evidence age, not
the typical span. The minimum bootstrap spans at least 0.4 seconds.

These settings are declared before running on the saved scene. They are not
deployment calibration. Several values reuse existing project settings to keep
the hypothesis small; that does not establish that their new use is optimal.

## Causal state transitions

1. Validate all keys, boxes, feature mappings, confidence values and timestamps.
   External break requests are forbidden for this automatic policy owner.
2. Age reference samples using scene time. Empty/missing observations break any
   pending confirmation; they neither synthesize observations nor reset the
   segment generation. Low confidence or missing features also interrupt support.
3. Bootstrap by sampling sufficiently confident observations at the minimum
   interval. Bootstrap samples are not certified to represent one person.
4. With an established reference, compare the current feature against the prior
   reference BEFORE considering that feature for reference updates.
5. Agreement clears pending evidence. The reference accepts the feature only
   when its sampling interval is due. Unsampled agreement still clears evidence.
6. Contradiction starts a candidate and freezes the prior reference. Pending
   candidates cannot contaminate that reference. Each subsequent support must
   disagree with that same frozen reference and agree with the candidate seed.
   Incoherent candidate appearance starts a new confirmation span.
7. A missing round, excessive time gap, or expiration of any effective frozen
   reference source clears confirmation. Both support count and elapsed scene
   time must be satisfied by the same uninterrupted candidate episode.
8. On confirmation, emit an explicit break with the current segment generation.
   `LocalTrackSegments` allocates a new effective key now. The policy's own new
   reference starts with the current feature only; previous pending samples are
   not inserted into the new reference. A downstream history will similarly
   start fresh when the integration supplies the transformed keys.

Reference means use float64 accumulation and normalized float32 output. In the
degenerate cancellation case the latest accepted feature is used, with its actual
single-source frame/time provenance. A pending frozen mean retains its original
source times even if the underlying rolling reference later loses old samples.

## Runtime API and ownership

`AppearanceContinuity(run_id, camera_ids, enabled=..., settings=...)` owns its
segmenter and policy state. `.update(SegmentRound(...))` returns a
`ContinuityFrame` containing the `SegmentedRound`, per-observation decisions,
and reset reasons. Record/feature ordering and values are preserved by the
segmenter; decision and event ordering is canonical. No GT, files, videos or
global identity labels enter the policy. Configuration must stay fixed per run.

Disabled mode invokes only the pass-through segmenter and retains no appearance
vectors or confirmation state. Enabled updates are planned in a cloned owner;
failure during policy or segment application commits neither. This does not
provide a transaction for external history/global modules. A later failure in a
complete pipeline must still abort/replay that run.

Reference memory has at most eight vectors per recent original track, plus two
vectors (frozen reference and candidate seed) per pending track. Empty rounds
evict stale vectors. The underlying segment metadata grows with original tracks
seen in the run, as documented in the segment contract. No end-to-end performance
claim is made for this deepcopy-based experimental implementation.

## Limitations and evaluation gate

Appearance discrepancy is not proof of a person switch. Pose, occlusion, poor
crops, lighting and similar clothes can cause false cuts or missed switches.
Gradual drift may still enter the sparse reference, and bootstrap can already
be mixed. Weak/missing features cannot support a cut; confirmation introduces
latency. This policy does not revise past outputs or clean other cameras' old
bindings. Geometry is unchanged downstream and is not a second confirmation
signal in this first appearance-only hypothesis.

The synthetic drift fixture deliberately demonstrates both the difference from
a short rolling history and the possibility of false cuts after pose changes.
It is not an accuracy test. Final quality must be compared with disabled control
on frozen detections/boxes, with full and disjoint-window global IDF1, unchanged
observation denominators, split-event diagnostics and generation provenance.
False-cut evidence must be reported alongside apparent switch repairs. Test a
promising unchanged policy on scene_041; neither reused scene is an untouched
final test after iterative development.

Run this contract check in WSL at the project root with its venv:

```bash
PYTHONPATH="$PWD/src" python scripts/check_appearance_continuity.py
```

The script reuses synthetic scene builders from the previously delivered
`scripts/check_track_segments.py`. It executes no models or video decoding.
