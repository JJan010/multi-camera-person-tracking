# Conservative grouping of simultaneous camera observations

Policy name: `complete_support_greedy_v1`.

This is a deterministic, stateless baseline between pairwise appearance matching
and the future global-identity lifecycle. It partitions observations at one scene
time into candidate person groups. It does not assign persistent IDs, remember
past groups, correct local tracking, use GT, or guarantee identity correctness.

## Input and output

`group_pair_associations(results)` accepts the full set of `PairAssociation`
objects for one run/session-generation, frame, exact Fraction timestamp,
descriptor variant and shared threshold. Every unordered pair among the represented
cameras must occur exactly once, including pairs with no accepted matches. The
current caller supplies pairs (4,5), (4,8), (5,8), even when a camera is empty.
At least two camera envelopes are represented by at least one pair result.

The matched and unmatched keys for a camera must describe the same observations
in every pair containing that camera. Repeated observations, mixed scopes, missing
camera pairs and accepted edges at or below the threshold are rejected. This
function trusts the producer's descriptor scores; it does not recompute embeddings
or verify the optimality of pairwise assignment. Source/config hashes belong in
the replay report, as in the existing pairwise evaluator.

Output `FrameGroups` contains the same context, the policy name, a sorted tuple
of groups (each is a tuple of ObservationKey values), and a decision for every
accepted input edge. All observations appear exactly once, including singletons.
If all represented cameras are empty, the result contains no groups. Observations
from one nonempty camera remain singletons when other cameras are empty.

Group tuple positions are only an output ordering within a frame. They must not
be interpreted or displayed as persistent global IDs.

## Rule

1. Start with one singleton group per observation.
2. Visit accepted edges by descending cosine score; break exact ties by canonical
   camera/local-ID/frame key pairs, independently of argument order.
3. If both endpoints already belong to one group, record `already_in_same_group`.
4. Otherwise reject the merge if the groups share a camera: `rejected_camera_conflict`.
5. Otherwise require an accepted input edge between every member of the first
   group and every member of the second. If any is missing, record
   `rejected_missing_pair_support` and the missing key pairs.
6. If all checks pass, merge the groups and record `merged`.

Each output group therefore has at most one observation per camera, and every
pair of members has an accepted input edge. In graph terminology each group is
a clique. A pair with no accepted edge is missing support, not proven to be a
different person. An accepted edge between groups can be rejected by grouping
because other necessary edges are missing.

The complete set of pair decisions from the same scene time is available before
grouping starts. Inspecting a supporting edge before its turn in the score order
does not use future video frames. No past outputs are revised by this function.

## Examples and learning points

With A-B=0.95, B-C=0.94 and A-C=0.93 all accepted, output is {A,B,C}.
The second merge checks its support against both existing group members.

With A-B=0.95 and B-C=0.94 accepted but no A-C edge, output is {A,B}, {C}.
The missing A-C support is reported. Even if all three observations belong to
one true person, the policy splits them. This is an explicit expected limitation,
not a test failure. Different views can have unequal appearance similarities.

With A1-B=0.99, A2-C=0.98 and B-C=0.97 accepted, where A1/A2 are from one camera,
output is {A1,B}, {A2,C}. The final attempted merge is rejected for camera conflict.

Greedy score ordering is not a global optimization of partition quality or total
retained edge gain. With B-C=0.99, A1-B=0.98 and A2-C=0.97, the first edge creates
{B,C}; both extensions lack support, leaving {A1}, {B,C}, {A2}. Two disjoint outer
pairs would have greater total retained score gain. A known-answer test preserves
this counterexample so the implementation cannot be presented as globally optimal.

These scores are synthetic teaching examples, not calibrated operating values.

## Guarantees and limitations

The final partition is checked for complete observation coverage, unique cameras
within groups and full pair support. Because no unsupported within-group links
are introduced, retained links form a subset of accepted pairwise links. Relative
to pairwise decisions, this stage cannot increase the count of recovered true
pairs on fixed evaluation labels. It can remove correct and incorrect links;
precision is not guaranteed to improve either. The replay must measure both.

Zero camera conflicts is an enforced structural property, not a quality score.
A closed triangle of incorrect observations can still pass. The camera uniqueness
rule assumes that duplicate tracks/detections are handled upstream.

The greedy policy's choices can change after small score changes. Canonical tie
handling makes the result deterministic for the same inputs, but does not provide
temporal identity stability. The next stateful stage must define tentative,
confirmed, lost and expired identities, rejection/reassignment behavior and the
scope of identity IDs. Reusing local IDs after a restart requires a new generation.

This baseline is intentionally more conservative than merely accepting an open
chain when cameras do not conflict. That alternative can be evaluated separately.
No production threshold, ambiguity margin or GT-driven rule is added here.

## Verification and next replay

`scripts/check_multicamera_grouping.py` checks supported triples, open chains,
camera conflicts, singleton/empty inputs, permutations/orientation, tie behavior,
mixed-context rejection, and the documented greedy counterexample. It runs on CPU
without detector/OSNet inference or new dependencies.

Next, replay the frozen pairwise decisions across the already declared threshold
grid for both variants, preserving every input observation. Compare retained correct,
wrong and unresolved links, missed positive pairs, group sizes and reasons for
rejected merges. Structural guarantees and GT identity correctness are distinct.
Do not tune grouping or thresholds to maximize the short integration clip.

## Sources and algorithm attribution

SciPy's description of complete linkage uses the maximum distance across all
cross-cluster member pairs, providing a useful connection to the idea of requiring
support for every member pair. This implementation does not call SciPy linkage
and is not the same hierarchical clustering algorithm: it processes accepted
assignment edges by score and adds camera constraints.
https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.cluster.hierarchy.linkage.html

Huang et al. (CVPR Workshops 2023), *Enhancing Multi-Camera People Tracking With
Anchor-Guided Clustering and Spatio-Temporal Consistency ID Re-Assignment*, uses
clustering for cross-camera association and additional consistency information.
It motivates treating cross-camera grouping as a separate design problem; the
greedy full-support policy here is a project baseline, not a reproduction of that method.
https://openaccess.thecvf.com/content/CVPR2023W/AICity/html/Huang_Enhancing_Multi-Camera_People_Tracking_With_Anchor-Guided_Clustering_and_Spatio-Temporal_Consistency_CVPRW_2023_paper.html
