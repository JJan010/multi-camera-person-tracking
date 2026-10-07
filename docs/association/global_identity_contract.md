# Causal global identity baseline

Policy: `local_anchor_no_merge_v1`.
Module: `src/mtmc/association/global_identity.py`.

## Purpose and boundary

`grouping.py` produces independent partitions at each scene time. This module
adds memory: the same retained camera/local-track key continues to carry the
same global ID across later rounds, even when frame-local grouping changes.

This is an explicit, limited baseline. It is not a complete identity resolver.
There is no evidence accumulation for merging existing global IDs, no recovery
from a local-track identity switch, no gallery search after expiry, and no
geometry. The registry never consumes GT or changes earlier outputs.

Its assumptions and limitations must be evaluated before selecting an operational
policy. In particular, a conservative name does not imply high identity accuracy.
A wrong initial group or attachment can persist through local-track continuity.

## Scope and API

```python
from fractions import Fraction
from mtmc.association.global_identity import GlobalIdentityManager

manager = GlobalIdentityManager(
    run_id="example-session",
    max_idle=Fraction(1),
    descriptor_variant="mean",
    min_similarity=0.5,
)
result = manager.update(frame_groups)
```

The numeric parameters above are examples, not calibrated deployment settings.
Use a separate manager for each run, descriptor variant and threshold. A restart
or local-ID namespace reset requires a new run/session-generation identifier.
The caller must not reuse a still-retained local ID for a different local-track
generation. Such reuse cannot be inferred from the numeric ID alone.

Input is a `FrameGroups` from the upstream complete-support grouping module.
The registry checks run, variant, threshold, policy, frame/time monotonicity,
unique observations and unique cameras per group. It trusts the upstream
producer's complete-support evidence; it does not rerun pairwise matching or
revalidate the `FrameGroups.decisions` audit trail.

Both frame index and exact `Fraction` scene timestamp must strictly increase.
Skipped frames are allowed. Elapsed scene time, not frame count or wall-clock
processing time, drives expiry. Empty rounds must be passed to advance time.
The class is intended for sequential calls from one pipeline coordinator.

## Keys

- Observation: `(camera_id, local_id, frame_index)` within the current run.
- Retained local binding: `(camera_id, local_id)` within the current run.
- Global identity: `(run_id, global_id)`.

Different cameras may use the same numeric local ID. These keys stay distinct.
Global integers begin at 1 and increase. Expired global IDs are not reused within
one manager. Do not treat them as universal person identifiers or as compatible
across independent experiments.

## Round processing

All decisions use a snapshot of prior retained bindings. New assignments from
one group cannot become anchors for another group in the same round.

1. Validate the whole input before touching live state.
2. Stage removal of bindings whose scene-time age is greater than `max_idle`.
3. Collect prior global IDs present in each current group.
4. Plan attachments, including competition across all groups.
5. Preserve existing assignments and assign all new observations.
6. Validate coverage and per-camera uniqueness, then commit the staged state.

A failed update leaves counters, bindings and the last accepted time unchanged.
Input group/member order is canonicalized, including fresh-ID allocation.

## Group decisions

| Existing IDs in a group | Action |
| --- | --- |
| None | Allocate one new global ID to the group, including a singleton. |
| One, no new tracks | Preserve the existing ID. |
| One, with new tracks | Attach new members if all requested camera slots are available. |
| Two or more | Preserve all existing IDs and report a conflict; never merge them. |

New members of a rejected group receive one fresh ID together: they retain their
mutual upstream pair support but do not borrow an ambiguous established ID.
Rejected attachment is all-or-nothing for the group's new members.

A retained binding reserves `(global_id, camera_id)` even while absent from the
current round. A new local track from that camera cannot claim this slot until
the old binding expires. If multiple otherwise eligible groups simultaneously
request the same free identity/camera slot, every competing group is rejected.
There is no first-come winner and no score-based tiebreak in this baseline.

Decision outcomes:

- `new_identity`;
- `continued`;
- `attached`;
- `rejected_multiple_existing_ids`;
- `rejected_reserved_camera`;
- `rejected_competing_attachments`.

Each observation separately records `local_continuity`, `group_attachment`, or
`new_identity`. Thus group conflict decisions do not hide how individual nodes
were assigned. All observations receive exactly one output ID.

## Lifetime

Each binding has its own `last_seen`. It survives while
`current_time - last_seen <= max_idle`, including the exact boundary.
A currently visible track refreshes only its own binding. It does not indefinitely
retain absent bindings from other cameras.

An identity is `visible` if it has at least one observation in the current round,
and `lost` if it has retained bindings but no current observations. These states
refer to visibility, not confidence or correctness. There is no `confirmed`
identity state in this baseline.

When all bindings expire, the global ID is reported in `expired_global_ids` and
removed from active registry state. Expired bindings are separately recorded.
A later return without a live anchor receives a fresh ID, unless current grouping
can attach it to another still-retained identity with a free camera slot.

## Invariants and known costs

- Every current observation appears exactly once in assignments.
- A global ID has at most one current observation per camera.
- A global ID retains at most one local binding per camera, including absent ones.
- Existing live anchors are not relabeled by current grouping.
- Past outputs are immutable; there are no retrospective identity corrections.
- No GT, future frames, image pixels, embeddings or wall clock enter the registry.

Observations with the same retained global ID may be in different current groups.
Memory can therefore preserve an earlier relationship without a current pair
edge. Conversely, a current group may contain multiple output global IDs when
its proposed association conflicts with established anchors. These are deliberate
baseline semantics, not violations of the frame-grouping partition.

The no-merge policy can permanently split a true person's visible tracks if they
first received separate global IDs. Reserved camera slots can also delay or block
recovery from local-track fragmentation. At the opposite extreme, an incorrect
initial group or attachment can propagate an identity error. This module cannot
repair either failure; future changes need measured merge/reassignment policies.

## Check and evaluation boundary

Run from the repository root in the project virtual environment:

```bash
PYTHONPATH="$PWD/src" python scripts/check_global_identity.py
```

The CPU-only known-answer test uses the actual pairwise-result -> grouping API,
then checks continuity, new-camera attachment, existing-ID conflict, temporary
absence, exact expiry boundaries, per-binding ageing, simultaneous competing
attachments, order invariance, run scope and transactional rejection.

Threshold 0.5 and idle time 1 second in the test are synthetic fixtures. No
threshold or idle duration for real recordings is selected by passing this test.

The next integration step is replay on frozen grouping output and recording
per-observation global IDs and conflict/expiry events. Quality must then be
assessed over time, including fragmentation and wrong identity propagation;
framewise pair precision alone cannot characterize persistent identity quality.

Scientific context: Ristani et al., *Performance Measures and a Data Set for
Multi-Target, Multi-Camera Tracking* (2016), motivates identity-based evaluation
in MTMC: https://arxiv.org/abs/1609.01775. This registry is a project baseline,
not a reproduction of the paper's tracking system.
