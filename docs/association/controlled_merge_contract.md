# Confirmed merging with whole-visible-group support

Policy: `whole_visible_group_confirmed_merge_v1`.
Module: `src/mtmc/association/controlled_merge.py`.
The frozen `local_anchor_no_merge_v1` implementation is unchanged.

## Motivation and scope

The selected integration diagnostic showed both repeated correct proposals blocked
by no-merge and repeated incorrect proposals. Persistence alone therefore cannot
justify an identity merge. This experimental policy adds whole-identity camera
constraints and coverage of currently visible members before counting evidence.

It is a project heuristic, not a guaranteed identity verifier or a reproduction
of a published method. No deployment parameters or quality improvement are claimed
by its synthetic contract tests.

## API

`ControlledMergeIdentityManager` subclasses the existing `GlobalIdentityManager`.
It takes the same run, idle duration, descriptor variant and pairwise threshold,
plus three required, explicit confirmation settings:

- `min_support_rounds`: integer at least 2;
- `min_support_seconds`: positive exact `Fraction` scene-time span;
- `max_evidence_gap`: positive exact `Fraction` gap between supporting updates.

Input remains a validated upstream `FrameGroups`. It contains no GT. Its group
partition is trusted to have the upstream complete pair support; the registry does
not recompute embeddings or pairwise matching. One manager serves one session,
variant, threshold and confirmation configuration. Calls are sequential.

## Two stages in one atomic update

1. Stage an ordinary no-merge update, including allocation, expiry and local-track
   continuity, on a copy of the state.
2. Consider each current group spanning at least two resulting global IDs.
3. Apply the gates and update pending evidence.
4. Revalidate and apply accepted merges, then commit the whole staged state.

The subclass uses the baseline's protected binding and sequence state. It does not
patch the baseline module. Copying state provides a clear transaction boundary at
this stage; its cost has not yet been benchmarked. Invalid updates commit neither
base registry changes nor confirmation evidence.

## Eligibility gates

A candidate must satisfy both conditions:

- The union of ALL retained bindings of its global IDs contains at most one local
  track from each camera. Absent but not-yet-expired bindings also reserve a slot.
- ALL currently visible observations belonging to those IDs are in the triggering
  group. A member visible outside the group blocks the candidate.

For example, if ID 1 is visible in cameras 4 and 5, and ID 2 in camera 8, a group
linking only cameras 4 and 8 is insufficient while the camera-5 member remains in
another group. A complete group containing all three can pass, provided retained
camera bindings are compatible.

Absent members cannot provide current appearance support. They are checked for
camera compatibility only. This limitation must remain explicit.

Eligible candidates cannot share an ID in a round: current groups are disjoint,
and whole-visible coverage requires each candidate to contain every visible member
of each involved ID. The implementation checks this invariant and revalidates each
candidate against staged state before committing. There is no transitive chain of
partially supported merges within one round.

## Temporal confirmation

Evidence is one count per eligible group per delivered observation round, not one
count per camera edge. A three-camera group still supplies one confirmation.
The candidate fingerprint consists of its ordered global IDs and their complete
retained `(camera, local_id)` memberships.

Both the required count and elapsed scene-time span must be reached. The span runs
from the first support time to the current time. The exact boundaries are inclusive.
Processing/wall-clock duration never contributes.

Evidence restarts when retained membership changes or the gap since last support
exceeds `max_evidence_gap`. It is cleared if the candidate lacks eligible support
in any delivered round, including an empty round. A changed candidate-ID set starts
new evidence. A skipped frame index is allowed within the explicit time-gap limit:
consecutive evidence means consecutive delivered updates, not every video frame.

These repeated observations are correlated. Their count is not a calibrated
probability of identity correctness.

## Accepted merge and lifecycle

The smallest participating global ID becomes canonical. All retained bindings of
absorbed IDs move to it, preserving each binding's original last-seen time. Only
assignments from the acceptance round onward use the canonical ID. Previously
returned outputs remain unchanged.

Absorbed IDs are retired by merging, not reported as expired. They are never reused;
the original monotonically increasing allocation counter is retained. Future
returns through still-retained local bindings use the canonical ID. Once all its
bindings expire, the canonical identity expires normally.

Allocated identities must be accounted for as:

`allocated = expired + absorbed by merges + currently retained`.

All observations remain assigned exactly once. At most one current observation and
one retained binding per camera belong to a global ID. There is no automatic split
or rollback of an accepted identity merge in this version.

## Output contract

The frozen `ControlledGlobalFrame` extends `GlobalFrame`:

- `assignments`: final current assignments after accepted merges; changed rows have
  reason `confirmed_merge`;
- `base_assignments`: assignments before the merge stage;
- inherited `decisions`: BASE no-merge group decisions before the merge stage;
- `merge_decisions`: blocked/pending/confirmed decisions with evidence and reasons;
- `merge_events`: canonical and absorbed IDs, acceptance time, support count,
  first support time, current members and prior retained memberships;
- `pending_candidates`: surviving candidate fingerprints and support state;
- `candidate_resets`: cleared or restarted evidence and its reason;
- `identities`: post-merge visible/lost registry state;
- inherited expiry fields: genuine binding/identity expiry from the base stage.

A base `rejected_multiple_existing_ids` and a merge-stage `confirmed` can occur in
the same round: the former explains stage one, the latter explains the subsequent
controlled decision. Consumers must not mistake the base decisions for final
merge outcomes. The dedicated policy name distinguishes this schema and behavior.

The old replay/evaluation commands deliberately recognize the old policy. Their
adapter for this new result, configuration, lifecycle counts and provenance will
be introduced in the next integration step. Do not relabel the policy string to
force the old adapter to accept it.

## Checks

From the repository root in its virtual environment:

```bash
PYTHONPATH="$PWD/src" python scripts/check_controlled_merge.py
```

Tests exercise the actual upstream grouping API, preservation of old outputs,
whole-visible-member gating, retained camera conflicts, missing evidence, time
limits, membership changes, three-ID merges, deterministic ordering, no ID reuse,
per-binding expiry after merge and transactional rejection.

The test uses threshold 0.5, three supporting rounds and small rational time spans
as known-answer fixtures. Those values are not selected deployment settings.

## Limits to evaluate next

Stable but incorrect complete groups can still pass all gates. Local-track identity
switches can still contaminate an accepted global identity. Missing observations
can remove useful contradictory evidence. Retained camera conflicts can also block
correct merges after local fragmentation. The conservative reset rule can prevent
true merges when support flickers.

Compare causal replay results against the unchanged baseline with identical boxes,
GT and evaluation denominators. Report incorrect merges as well as fragmentation
improvement, and calibrate eventual parameters on separate data. GT remains strictly
outside runtime decisions.
