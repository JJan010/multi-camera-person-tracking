# Dormant identity registry bridge

Status: synthetic enabled-recovery checks and technical disabled-control replay.
No enabled run on the actual two-minute embeddings, quality result or calibrated
archive settings are claimed by this package.

## What changed

`RecoveryIdentityManager` owns a fresh copy of the existing
`ControlledMergeIdentityManager`. The frozen manager and pipeline code are
unchanged. With recovery disabled, `update` returns the original complete
`ControlledGlobalFrame` exactly. No archive, snapshot memory or recovery
parameters are created in that mode.

With recovery enabled, a round is processed on a private copy containing the
registry, archive, evidence snapshots and counters. Either the entire round
commits, or none of them changes. A synthetic fault after archive consumption
tests that a failed claim cannot leak into the real state. An outer video
pipeline may already have advanced other components; its fail-stop contract
still applies even though this component is internally transactional.

## Enabled round

1. Validate the original grouping contract and every evidence key, vector,
   source time and position. Each observation must have an explicit evidence
   row, including missing descriptors or positions.
2. Run the existing controlled registry on the private copy. Existing bindings,
   current grouping, merge confirmation and idle expiry retain their semantics.
3. Archive snapshots only for global IDs reported as fully expired. A merely
   absent identity with retained bindings is not eligible for recovery. Supply
   all retained IDs and absorbed aliases as blocked archive IDs.
4. Consider only whole current groups whose members ALL received `new_identity`
   assignments and together own one new provisional global ID. An anchored or
   mixed group cannot recover another ID through this path.
5. The archive makes its causal appearance/position and mutual-margin decisions.
   A successful claim remaps that whole provisional group to the old ID and
   installs the recovered binding into the registry. Future local continuity
   uses the old ID. No earlier output is rewritten.
6. Refresh supported snapshots, verify camera uniqueness, observation coverage
   and lifecycle accounting, then commit registry and archive together.

This is a birth-time recovery policy. If a newly created group cannot recover an
ID in that round, it keeps its fresh ID; this version does not keep retrying an
already anchored group. There is no multi-round return confirmation, automatic
correction of a wrong return or repair of an active local-ID switch. Those are
limitations to assess before any policy promotion.

## Snapshot provenance

The bridge accepts `RecoveryEvidence` scoped to run, coordinate system, frame
and exact scene time. `ObservationEvidence` holds a current observation key,
its normalized 512-dimensional float32 descriptor and real descriptor-source
time, plus projected ground position and its real source time. Arrays are copied.
There are no GT inputs or image/model calls.

A snapshot is refreshed only if all currently visible members of the identity
form exactly one supplied group with complete evidence. The mean appearance is
normalized across those members; its source time is the earliest contributing
descriptor time. A cancelled mean does not fabricate a descriptor or fall back
to an arbitrary member. Ground position is the arithmetic mean of current-time
projected member positions. Older positions cannot masquerade as current ones.

When evidence is incomplete or visible members split across groups, preserve
the last snapshot with its original evidence times. Actual `last_seen` can
advance independently; this does not make old features or positions fresh.
After a confirmed merge, discard the canonical identity's previous snapshot
and require new jointly supported evidence. Absorbed aliases lose their snapshots
and never enter the inactive archive. The bridge cannot prove that its inputs
depict one person; upstream association errors can still contaminate snapshots.

## Allocation and output semantics

The baseline allocator first gives an unanchored group a provisional new number.
If recovery succeeds, that number is superseded before emission. It remains
consumed in the monotonic allocator and must never be used for a later person.
Thus `allocated_id_slots` and `ever_emitted_ids` are distinct counters.

Enabled output uses policy `dormant_return_bridge_v1`. Final `assignments` and
`identities` describe the committed recovered state; recovered observations use
reason `dormant_reactivation`. Existing `base_assignments` and `decisions` still
describe the provisional baseline phase. `last_audit.baseline` preserves its
entire unmodified result, and `last_audit.reactivations` explicitly links each
old ID, superseded new number and current observation group.

Do not apply the previous cumulative lifecycle helper to enabled outputs.
The bridge checks:

```text
currently_inactive_ids = expiration_events - reactivation_events

allocated_id_slots = retained_ids + absorbed_ids
                   + superseded_new_ids + currently_inactive_ids
```

Inactive includes expired identities no longer present in the bounded archive,
for example after evidence expiry or capacity eviction. Archive size is therefore
not the number of all inactive identities. An ID can expire and reactivate more
than once, so expiration events are not a count of distinct people. Neither are
allocated or ever-emitted tracker IDs.

## Checks to run in WSL

From `/home/jakjan/projects/multi-camera-person-tracking`, with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_dormant_registry.py

PYTHONPATH="$PWD/src" python scripts/check_dormant_registry_replay.py \
  --parity-report artifacts/paired_scene_runtime_checks/20261007T211730398428Z/report.json
```

The first command tests disabled full-record parity over 120 synthetic rounds,
enabled return and continuity, non-reused provisional numbers, active reservations,
ambiguous returns, repeated expiry/return cycles, absorbed aliases, stale/missing
evidence, cancellation, canonical ordering and transaction rollback. Its numeric
archive settings are synthetic examples only.

The second command verifies the prior gate and all its frozen inputs/code. It
uses the existing exact replay checker with a test-local dependency replacement
that wraps both `staged` and `competitive_iou` registries with DISABLED recovery.
The replacement is restored in `finally`; no original file or policy is modified.
It compares full local/global records, refinements, lifecycle and the prior
summary. Testing both policies is integration coverage, not a promotion of the
competitive policy or a new comparison of tracking quality.

Results are saved in `artifacts/dormant_registry_checks/<run>/report.json`, with
hashes for the new bridge and its checker as well as the frozen inputs. This
replay does not load raw GT, run models, decode videos or recompute IDF1.

Local technical verification uses a 300-round fixture with saved early boxes and
synthetic constant features. It does not substitute for the user's full frozen
two-minute replay or measure the usefulness of enabled reactivation.

## Next gate

Before an enabled paired experiment, build and verify real snapshot/query
provenance from the frozen current groups, appearance history and projected
boxes. Preserve descriptor source times. Define the enabled experimental settings
explicitly, without silently converting native units to meters or tuning them
against inspected validation output. Compare full shared-ID metrics and disjoint
windows against staged/all-updates, with unchanged local tracks and no GT in
causal replay. No quality gain or end-to-end speedup is established yet.
