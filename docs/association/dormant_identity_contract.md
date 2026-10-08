# Dormant identity archive: experimental component contract

Status: synthetic contract tests only. No registry integration, frozen-video
replay, tracking-quality measurement, runtime promotion or deployment calibration.

## Purpose

A local track and a person's global identity have different lifetimes. An
inactive person's recent appearance and ground position can be kept after the
local track bindings expire. A newly observed unanchored group can then request
that previous global ID instead of always starting a new identity.

This component implements the archive and its matching decisions. It does not
allocate fresh IDs, modify local tracks, merge live identities, revise earlier
outputs or repair active local-ID switches. It is separate from all existing
fingerprinted implementations.

## Inputs and ownership

Create one `DormantIdentityArchive` per run and calibrated coordinate space.
Pass explicit `ArchiveSettings`; there are no deployment defaults.

`ArchiveRound` carries a strictly increasing frame and exact Fraction scene time:

- `retirements`: final snapshots of identities that have no remaining live or
  retained local bindings. Each contains ID, true last-seen time, normalized
  512-dimensional float32 descriptor, its actual source time, ground point and
  the point's source time. Missing evidence is represented by None, not zeros.
- `queries`: disjoint, currently unanchored observation groups. Members are
  current-frame `ObservationKey`s, at most one per camera. Descriptors and points
  must be produced causally by the owner from the current observation group.
- `blocked_global_ids`: ALL currently live or retained global IDs, including
  temporarily absent ones, and all IDs absorbed into another canonical identity.
  This prevents resurrecting an ID that is still active or is an obsolete alias.

The caller must validate lifecycle provenance. The archive cannot infer from a
vector that a query already has a live binding or that a retirement is a replay
of an old lifecycle event. A later retirement of a reactivated ID must be a real
new event. Snapshot timestamps must not be replaced with the retirement time.
For an aggregate descriptor, its source time must conservatively account for
its contributing samples rather than pretending an old mean is newly observed.

The archive owns copies of retained vectors and exposes no mutable stored array.
Persistent descriptor storage is limited to `max_identities * 512 * 4` bytes,
plus bounded per-entry metadata. There is no growing per-identity tombstone map.
The global registry remains the owner of permanent ID allocation/alias lineage.

## Lifecycle within a round

1. Validate all input scopes, records, keys, normalized vectors, geometry and
   times before modifying state.
2. Remove archived entries newly listed as blocked.
3. Add valid retirement snapshots. Missing evidence is explicitly skipped;
   malformed evidence raises an error. Duplicate/already-archived retirements
   and blocked retirements are errors.
4. Expire entries when last-seen age OR either evidence-source age exceeds
   `max_age`. Equality is retained. Empty rounds also perform expiration.
5. Enforce capacity before matching. Evict oldest last-seen entries first,
   breaking exact ties by numeric ID. This deterministic memory policy is not
   a claim that those entries are least useful for tracking.
6. Evaluate all eligible appearance/position pairs and resolve return requests.
7. Remove successfully claimed identities from the inactive archive and commit
   the new archive state once. Return canonical, immutable decision records.

An archived identity can be claimed at most once in a round and stays unavailable
for another claim unless the owner later supplies a genuine new retirement.
Returning in the same camera under a different local ID is allowed once its
previous binding has ended; the archive does not reserve camera slots.

## Matching policy

For normalized descriptors, compute cosine similarity. A pair must satisfy:

```text
similarity > min_similarity
distance <= position_slack + max_speed * (now - ground_sample_time)
```

The appearance boundary is strict and the distance boundary inclusive. Position
uses the dataset's calibrated native units; speed is native units per second,
not assumed meters per second. Missing appearance or geometry produces an
explicit unmatched decision. There is no appearance-only fallback for recovery.

Among eligible pairs, a match must be the best choice in both directions: the
query's best archived ID and that archived ID's best current query. In both
directions, the advantage over the next eligible candidate must be at least
`min_margin`. Exact ties are rejected. A sole eligible candidate has no ranking
competitor but still has to pass the absolute appearance and geometry gates.

This policy is deliberately conservative. It can leave a valid alternative
unused and does not maximize total assignment score or cardinality. A test
demonstrates that limitation. A large similarity score or a passed gate is not
proof that two observations depict the same person.

## Integration boundary

A successful `ReturnDecision` is a claim to be applied by the future owner, not
an already-applied global assignment. Once the component commits a claim, the
owner must install its binding in the global registry before the next round.
Use a transaction across both components or fail the whole run after a partial
failure; do not retry the same pipeline round against a consumed archive.

Integration must preserve fresh-ID uniqueness, absorbed-ID aliases and current
camera uniqueness. It must explicitly account for reactivations in lifecycle
statistics: the old equation using cumulative expired IDs cannot simply be
copied once expired identities may become active again. No past identity output
is rewritten. A disabled-archive control must reproduce the existing baseline.

Do not implement this by merely increasing the registry's local-binding idle
timeout. That would also retain camera-slot reservations, with different effects
on replacement tracks and merging.

## Verification

Run from the project root, with its virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_dormant_identity.py
```

Checks cover ID-zero handling, claims and later explicit retirements, blocked
IDs, appearance/geometry/age boundaries, competing requests and alternatives,
margin boundaries, unavailable evidence, owned arrays, canonical input ordering,
capacity eviction, empty-round expiry and malformed-input atomicity. All numeric
values in the checker are synthetic fixtures. They are not camera calibration,
person-speed estimates or recommended production parameters.

Next: build the registry bridge and archive-snapshot provenance on the frozen
staged/all-updates development trace; prove disabled-control parity before
evaluating an enabled variant. Parameters need an explicit experimental policy.
The inspected scene_041 is not an untouched final test. This module adds no
model execution, GT access, image decoding or TensorRT/NVDEC optimization.

## Scientific context

Ristani and Tomasi, *Features for Multi-Target Multi-Camera Tracking and
Re-Identification*, CVPR 2018, https://arxiv.org/abs/1803.10859, distinguish
appearance retrieval from multi-camera tracking and study their relationship.
The causal archive and mutual-best rule here are an engineering experiment,
not a reproduction of that paper's complete tracker or an asserted quality gain.
