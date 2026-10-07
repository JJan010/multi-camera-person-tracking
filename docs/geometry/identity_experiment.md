# Paired geometry association and controlled identity experiment

## Comparison

Compare the frozen appearance-only association plus controlled-merge baseline
against ground-distance-gated association plus the same controlled identity
manager. The intended real baseline is run `20261007T104647560154Z`, whose
shared global IDF1 is 85.03%, with IDTP=13666, IDFP=2162 and IDFN=2651.

Use the same source run, tracked boxes, observations, appearance descriptors,
frames 2..299 and cameras 4/5/8. No detector, local tracker or OSNet forward
pass is run. Mean descriptors are loaded from the original causal-history
artifacts and verified against their exact source membership, count and age
policy by the existing history loader.

## Experimental configuration

- Appearance variant and threshold: inherited from the baseline (mean, 0.70).
- Geometry: `ground_distance_before_assignment_v1`, max distance 2 native
  dataset units, inclusive boundary.
- Unavailable geometry: explicit `appearance_only` fallback.
- Grouping: existing complete-support greedy policy.
- Identity: existing `whole_visible_group_confirmed_merge_v1` policy.
- Idle, support-round count, support time and maximum evidence gap: inherited
  unchanged (1 second, 3 rounds, 1/5 second and 1/10 second in the real baseline).

The distance was chosen as an integration candidate after inspecting the same
training fragment. This is an in-sample experiment, not independent calibration
or a deployment threshold. The physical scale of native coordinates has not
been independently established in these audits.

Raw tracked-box bottom centers are projected without clipping. The projector
does not detect torso points caused by occlusion. Finite but unreliable
positions still enter the gate. None means unavailable projection; malformed
matrices or inputs raise instead of triggering fallback. The fallback remains
an explicit experiment choice, not an empirically established best policy.

## Phase 1: paired causal replay

1. Verify the controlled baseline and its assignment trace, grouping report
   and trace, upstream pairwise report, appearance-history inputs, calibration
   and tracked-box source hashes.
2. Recompute original appearance assignments and groups from frozen vectors.
   Require the partition and full grouping decisions to equal the frozen
   grouping records, including scores and ordering.
3. Run a fresh copy of the original controlled manager. Require every complete
   output record to equal the frozen baseline record and reproduce lifecycle
   counts. This is stronger than comparing only final metrics.
4. Project tracked-box bottom centers into the shared calibrated plane. Gate
   candidate pairs before assignment, recompute groups and update a separate
   controlled manager with the same confirmation configuration.
5. Freeze new assignment and geometry-decision traces before quality scoring.

The association and identity APIs receive no GT labels or world GT positions.
Historical diagnostic labels are carried separately in audit records and
source-counter checks. Scene timestamps govern confirmation and expiry;
processing speed is irrelevant. No earlier output is rewritten after a merge.

## Phase 2: offline quality

Load GT and build the same camera/frame IoU gates for both outputs. Recompute
fixed diagnostic IoU labels and verify they agree with the source. Evaluate
one shared identity assignment over all cameras and frames with the existing
tested identity metric. Require exact reproduction of baseline metrics and
equal camera/time and observation denominators.

Evaluate baseline and candidate pair links and group links. Separately count
removed old pair links and newly selected pair links. These can differ from
the previous fixed-pair distance sweep because a forbidden candidate frees
an observation for a different partner.

Accepted-merge diagnostics use mutually unique GT overlap for visible members
at the merge time, as in the earlier controlled baseline. They do not certify
the whole retained identity history. Source and frozen output hashes are
checked again after evaluation.

Geometry restricts current pair association. It is not a new veto on existing
local-to-global continuity and cannot split an already mixed identity. Thus
improved pair precision does not by itself guarantee better global IDF1.

## Run in WSL

From `/home/jakjan/projects/multi-camera-person-tracking` with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_geometry_identity_experiment.py

PYTHONPATH="$PWD/src" python scripts/run_geometry_identity_experiment.py \
  --baseline-report artifacts/controlled_merge/20261007T104647560154Z/report.json \
  --geometry-report artifacts/geometry_audit/20261007T110015275411Z/report.json \
  --max-distance 2 \
  --unavailable-policy appearance_only
```

No new dependencies are required. The existing association, controlled-merge,
geometry and evaluation modules/scripts must be present. The smoke-test
50%-versus-100% outcome is a synthetic known answer, not a real-scene result.

## Output

Each run creates `artifacts/geometry_identity/<UTC-run>/`:

| File | Contents |
| --- | --- |
| `report.json` | Exact configuration, input/code hashes, paired metrics, link transitions and lifecycle |
| `assignments.jsonl.gz` | Causal global outputs with separate identity and association policy metadata |
| `decisions.jsonl.gz` | Ground positions, every candidate's geometry decision, selected pairs and groups |
| `comparison.csv` | Baseline and candidate identity metrics |
| `by_frame.csv` | Observation, pair-transition and merge-event counts |
| `merge_diagnostics.json` | All new accepted merge events with offline GT diagnostics |
| `identity_matching.json` | Both metric-level shared identity assignments |

The identity policy field remains the controlled-manager policy; separate
association policy and geometry configuration identify the new pipeline.
The experiment has its own identity namespace. Numeric IDs should not be
compared directly between runs as if they were universal person identifiers.

Inspect new global metrics and changed error cases before drawing conclusions.
This run records extensive diagnostics and is not a performance benchmark.
