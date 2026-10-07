# Ground-distance admissibility before appearance assignment

Policy: `ground_distance_before_assignment_v1`.
Module: `src/mtmc/association/geometry.py`.
The original appearance-only solver, grouping module and identity managers
are unchanged. This module is stateless and contains no GT labels.

## Purpose

A geometry condition restricts candidate pairs before joint assignment.
For finite projected positions, a pair is geometrically admissible exactly
when its Euclidean distance is <= the explicit `max_distance`. Appearance
still requires cosine similarity strictly greater than `min_similarity`.
Both gates must pass. Among admissible pairs, retain the original partial
assignment objective: maximize the sum of similarity minus the appearance
threshold, with an unmatched option for every observation. Distance does not
change the ranking or become an extra cost in this baseline policy.

Consider one observation with appearance scores 0.99 and 0.90 against two
candidates. If the first candidate is too far away and the second is near,
filtering the appearance winner after assignment leaves no match. Applying
geometry before assignment allows selection of the second candidate.

The low-level solver uses the existing partial-assignment implementation.
Geometrically forbidden scores are replaced by -1, which cannot pass the
strict appearance gate for any supported threshold in [-1,1]. Validate all
original scores before masking so forbidden entries cannot hide NaN values.

## Inputs and scope

`CameraAppearance` is reused unchanged. A corresponding
`CameraGroundPositions` envelope carries the run/session ID, camera ID,
frame, exact Fraction scene timestamp, shared `coordinate_space` identifier
and one `GroundObservation` per appearance observation.

Each point is a finite `(X,Y)` tuple or explicit `None`. Mapping is by
`ObservationKey`, not array position. Missing keys, extra keys, duplicate
keys, nonfinite coordinates and mixed scopes are rejected. All appearance
observations, including those with unavailable geometry, must be covered.

`coordinate_space` identifies the common scene, ground plane, coordinate
units and calibration generation, for example a scene identifier plus a
pinned calibration checksum and `Z0/native`. Both cameras must declare the
same space. It must identify the shared world frame, not an individual
camera's matrix hash. The caller is responsible for truthful metadata and
for supplying positions from the declared calibration.

`project_box_foot(H, xyxy)` supplies a minimal projection helper: raw box
bottom center through inverse H, homogeneous division, no clipping, rounding
or resizing. H must map world Z=0 to the original image coordinates. The
helper returns None at a numerical near-horizon projection; malformed boxes
and invalid/singular matrices raise. It does not establish foot visibility,
perform lens undistortion, or assess physical calibration accuracy.

## Explicit unavailable-geometry behavior

The caller must choose `unavailable_policy` on every association call:

- `appearance_only`: if either position is None, allow appearance matching
  for that pair and record `unavailable_appearance_fallback`.
- `reject`: block that pair and record `unavailable_rejected`.

There is no default. These behaviors apply to explicitly unavailable points,
not invalid inputs. A finite projection of an occluded upper-body box is
still finite; the current helper does not automatically flag that ambiguity.
Being fully inside an image is not used as a visibility guarantee or gate.

The initial real replay can explicitly use `appearance_only` to preserve
appearance matching when the minimal projector cannot provide a position.
This is a stated experimental choice, not a demonstrated optimal policy.
The previous distance diagnostic had zero unavailable projections, so it
could not evaluate the choice's real-world consequences.

## Output and downstream use

`GeometryPairResult` stores policy, space, distance threshold, unavailable
policy, a compatible `PairAssociation` and a diagnostic entry for every
cross-camera candidate. Candidate entries contain keys, finite distance or
None, geometry decision, appearance eligibility and selection status.

Finite geometry decisions are `within_distance` and `outside_distance`.
Unmatched observations preserve existing reasons and add `geometry_blocked`
when above-threshold appearance candidates exist but all fail geometry.
`best_similarity` remains the highest raw appearance score, including
blocked candidates; it is not the best geometrically admissible score.

Use `group_geometry_associations(results)` to validate that all camera pairs
share one geometry configuration before invoking the original grouper.
The original grouping checks require complete camera-pair coverage and
consistent observations/session/time/appearance setting. The returned
FrameGroups retains its original type and grouping policy; callers must
preserve geometry metadata separately in experiment configuration and audit
records. Do not strip metadata and combine incompatible policies silently.

Camera orientation and local IDs are canonicalized before solving, matching
the original order/tie handling. Output keys follow the requested camera
orientation. No global identity creation, merge, expiry or history rewrite
occurs in this module.

## Verification and next step

```bash
PYTHONPATH="$PWD/src" python scripts/check_geometry_association.py
```

The smoke test covers a valid alternative lost by post-filtering, exhaustive
small masked assignment objectives, all-admissible parity, inclusive distance
and strict appearance boundaries, empty cameras, explicit unavailability,
key mapping, camera orientation, exact-score ties, raw projection and invalid
input rejection. It also exercises compatible three-camera grouping and
rejection of mixed geometry settings.

Distances in these tests are synthetic fixtures. The diagnostic on the real
training fragment motivates distance 2 native units as an in-sample replay
candidate. This module does not select a deployment threshold, run a new
real-data replay or claim an improvement over global IDF1=85.03%.

Next: feed frozen mean appearance descriptors and projected tracked boxes
through this module, group the new pair assignments, then run the controlled
identity manager causally with its existing confirmation settings. Compare
against reproduced appearance-only outputs on the same observations. A new
shared global identity evaluation is necessary because alternative pair
assignments can change subsequent groups and identity history.
