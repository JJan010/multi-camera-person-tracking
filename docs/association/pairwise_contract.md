# Same-time pairwise appearance association

This is the first association primitive after causal appearance history. It
produces candidate links between two synchronized camera views. It does not yet
create global IDs, maintain global identity state, or merge three-camera groups.

## Contract

`CameraAppearance` wraps one camera's `ReIDBatch` with:

- `run_id`: scene/session and tracker generation scope; use a new value after a
  restart that might reuse local IDs.
- `camera_id`, `frame_index`, and exact `Fraction` scene timestamp.
- `descriptor_variant`: `latest` or `mean`, the same for both cameras.
- Finite, normalized float32 descriptors with shape `(N, 512)`, including `(0, 512)`.

Every observation key/time must agree with its envelope; local IDs must be unique
within the camera/frame. Camera IDs must differ. Both sides must have the same
run, frame and timestamp. Model provenance and the validity of the camera-pair
topology are caller responsibilities. No GT enters the function.

`associate_camera_pair(left, right, *, min_similarity)` returns `PairAssociation`:
source context, threshold, matched observation-key pairs and cosine scores,
plus unmatched observations on both sides. Input arrays are not mutated, and
results hold scalar scores/keys rather than borrowed feature arrays. This is a
CPU-only stateless operation; the history component owns temporal aggregation.

Unmatched reasons are operational, not identity judgments:

- `empty_opposite_camera`: no candidates exist in the other camera.
- `no_candidate_above_threshold`: every candidate has score at or below the gate.
- `assignment_competition`: admissible candidates exist, but the joint optimum
  uses their capacity elsewhere. This does not prove the rejected person is absent.

The same-camera uniqueness constraint assumes duplicate detections are handled
upstream. The primitive addresses simultaneous observations in overlapping views.
It does not handle delayed transitions between disjoint views.

## Objective and rejection

For normalized vectors, compute `s_ij = dot(z_i, z_j)` in float64 and clip numerical
rounding to `[-1, 1]`. The threshold `tau` is required and has no default.
An edge is admissible only when `s_ij > tau`; exact equality remains unmatched.

Among admissible edges, maximize:

`sum((s_ij - tau) * x_ij)`

Each row and column participates at most once. Unmatched observations contribute
zero. Implement this with real-column costs `tau - s_ij`, positive forbidden-edge
costs, and one zero-cost dummy column per row. Enough dummy columns exist to
leave every row unmatched. Unused real columns are unmatched on the other side.

This objective balances additional links against their similarity gain above the
gate. It does not first maximize match count. These are explicit project design
choices, not a probabilistic model of identity or a claim that tau is calibrated.

For example, with synthetic `tau = 0.8` and scores:

| | B1 | B2 |
| --- | ---: | ---: |
| A1 | 0.95 | 0.94 |
| A2 | 0.93 | 0.10 |

Independent nearest neighbors choose B1 twice. Joint matching chooses A1-B2 and
A2-B1: total gain 0.27, instead of 0.15 for A1-B1 alone.

With scores `[[0.95, 0.81], [0.81, 0.0]]`, the same objective prefers A1-B1 alone:
gain 0.15 exceeds the gain 0.02 from two weak links. A square matrix never forces
a full real-to-real match.

The gate is applied before optimization. Full assignment followed by filtering
can lose valid links: for `[[0.81, 0.79], [0.79, 0.0]]` at tau 0.8, unconstrained
assignment favors the two invalid off-diagonal edges; post-filtering leaves none.
The partial solver correctly retains A1-B1.

## Ordering, ties and limits

Sort camera orientation by camera ID and each camera's observations by local ID
before solving, then return keys in the caller's orientation. This removes input
list order and camera argument order as sources of different tie decisions for
the same solver version. Multiple optimum assignments can still exist: the
chosen one is not evidence that appearance uniquely identifies a person. We do
not add an uncalibrated ambiguity margin in this step. Record SciPy versions in
future experimental reports; cross-version tie identity is not promised.

One-to-one matching applies to a specified camera pair. Running it for A-B, B-C
and A-C does not automatically produce consistent global groups. In particular,
A1-B1 and B1-C1 do not establish that A1-C1 is valid. Global grouping must check
camera conflicts and consistency, and temporal logic must manage tentative,
confirmed, lost and expired states before persistent global IDs are claimed.

## Verification and next experiment

`scripts/check_pairwise_association.py` tests competing candidates, pre-solve
gating, optional unmatched observations, exact boundary behavior, empty inputs,
source-key mapping, argument/input-order invariance, and rejected mixed contexts.
Small exhaustive matching cases independently verify the objective. All vectors,
scores and thresholds in these tests are synthetic. No detector, OSNet execution,
CUDA, videos or additional dependencies are needed.

Next, replay the frozen observations through this primitive and measure accepted
link correctness, missed eligible links, unmatched cases and cross-pair conflicts.
Keep latest and mean variants on identical inputs. A threshold sweep on the short
integration clip is diagnostic only: threshold selection requires separately
designated validation data, with final assessment held apart. Do not silently
turn the smoke-test threshold into a deployment configuration.

## Sources and relationship to this implementation

Wojke, Bewley and Paulus (2017), *Simple Online and Realtime Tracking with a Deep
Association Metric*, section 2.2, combines appearance/motion costs with admissibility
gates for association and discusses separate data for selecting its appearance
threshold. It is a single-camera tracking method; the partial cross-camera objective
above and the mean-history policy are our own explicit baseline choices.
https://arxiv.org/abs/1703.07402

SciPy 1.15.3 `linear_sum_assignment` solves rectangular linear assignment with a
modified Jonker-Volgenant algorithm. The dummy-column construction supplies the
unmatched option used here. The primitive does not claim to be an implementation
of the Hungarian algorithm simply because both solve assignment problems.
https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.linear_sum_assignment.html

Crouse (2016), *On implementing 2D rectangular assignment algorithms*, IEEE TAES
52(4), 1679-1696, is the algorithm reference cited by SciPy:
https://doi.org/10.1109/TAES.2016.140952
