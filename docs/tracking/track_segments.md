# Causal local-track segments: plumbing contract

Status: synthetic contract verified; automatic break policy, frozen real-data
integration and paired quality evaluation are pending. Default runtime unchanged.

## Motivation and scientific context

The current local key `(camera_id, local_id)` is used both for rolling appearance
history and as an anchor in the global identity manager. A persistent local
identity switch can therefore carry earlier appearance samples and a previous
global binding into another person's observations. Dormant recovery addresses
expired identities only and cannot resolve that active-track case.

Deep SORT uses appearance information in online measurement-to-track association
to reduce identity switches [1]. BoT-SORT combines appearance and motion with
other tracking improvements [2]. These publications motivate checking identity
continuity beyond local ID equality. The segment mechanism here is a separate
engineering component, not an implementation of either paper or a validated
switch-detection method. No published accuracy is claimed for it.

1. Wojke, Bewley, Paulus. *Simple Online and Realtime Tracking with a Deep
   Association Metric* (2017). https://arxiv.org/abs/1703.07402
2. Aharon, Orfaig, Bobrovsky. *BoT-SORT: Robust Associations Multi-Pedestrian
   Tracking* (2022). https://arxiv.org/abs/2206.14651

## Component placement

Proposed order: local tracker selects candidates; their frozen/current OSNet
features and boxes enter a continuity policy; an explicit break decision enters
`LocalTrackSegments`; transformed keys feed appearance history and the existing
geometry/grouping/global identity stage. Geometry and boxes themselves do not
change. This package supplies the segment mechanism only.

`src/mtmc/tracking/segments.py` accepts a `SegmentRound` with the run scope,
strictly increasing frame and rational scene time, all current `CropRecord`s,
the encoded subset `ReIDBatch`, and zero or more `SegmentBreak` requests.

A request contains the original current observation key, expected generation,
and reason. It must target an already observed track. The mechanism neither
reads GT nor computes a similarity threshold. Only the separate future decision
policy can emit a runtime request. A synthetic request in the smoke test is not
a prediction of an identity switch.

With no requests, both enabled and disabled modes preserve all original keys
and values. Disabled mode rejects break requests rather than silently applying
or ignoring them. Every frame, including empty rounds, must be delivered by the
eventual pipeline owner; this component accepts strictly increasing frames and
timestamps and does not invent omitted observations.

At a cut, a fresh internal local ID takes effect on the current observation and
later observations of that original track. `SegmentBinding` records the original
key, effective identity key, generation and start time. Feature rows and records
keep their input ordering and values; features are copied. `source_key()` maps
current downstream keys back to original observations for trace/evaluation joins.
Full stored outputs must include these bindings, not hide the source tracker ID.

Old history and old global bindings are not edited or force-expired. The existing
modules retain their ordinary age and camera-slot rules. A new segment therefore
starts a clean history but is not guaranteed immediate attachment to the old
identity, or even a different final global identity: current grouping and merge
rules still apply. It can attach to another visible identity when those rules
allow it. Past outputs are immutable. No retroactive correction is performed.

## Namespace, lifecycle and validation

Original local IDs must be nonnegative Python integers below `2**31` and must not
be reused for unrelated tracks in a run. Split IDs use a disjoint monotonically
allocated namespace starting at `2**31`, capped at `2**53 - 1` for exact integer
interchange. Allocation is canonical by camera and original local ID and never
reuses a split ID. Namespace exhaustion or an out-of-range source ID is rejected.
These are identifier representation constraints, not calibrated model settings.

State is run-scoped and retains scalar metadata per original track. It has no
vector cache or full event history. Empty/absent rounds do not reset a generation.
Memory grows with the number of original tracks seen; a long-running service
would need an explicit session/retention design before deployment.

All camera keys, records, feature mappings, timestamps and break generations are
validated before committing the state. A bad later camera cannot leave an earlier
camera partially segmented. Stale break requests are rejected and a corrected
round may be retried. Calls are single-owner/sequential. Failures in downstream
history or identity processing must fail the complete run or use a future
pipeline-level transaction; this component cannot roll back external modules.

## Validation and next gate

```bash
PYTHONPATH="$PWD/src" python scripts/check_track_segments.py
```

The CPU-only synthetic check covers 120-round no-action parity through the real
appearance history and identity stage, a split that joins another existing ID,
fresh history provenance, unchanged old bindings/past outputs, original box/score
and feature mapping, array ownership, input-order invariance, ID zero, empty and
outside observations, gaps, repeated cuts, namespace boundaries, invalid inputs,
stale generation rejection and run isolation. It is not an accuracy benchmark.

Next steps: define one causal automatic break hypothesis with declared evidence
and confirmation requirements; verify disabled integration on frozen outputs;
then compare enabled/disabled quality with identical detections and local boxes,
including false splits and full-sequence global IDF1. Test a promising unchanged
policy on scene_041. More segments alone do not mean better tracking: false cuts
can increase fragmentation, and a bad reassociation can undo a useful cut.
