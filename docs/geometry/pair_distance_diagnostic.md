# Frozen tracked-box pair-distance diagnostic

## Question

How do ground-plane distances differ between same-person, different-person
and unresolved cross-camera pairs? How many fixed appearance links would a
hypothetical distance check retain or reject? This diagnostic does not modify
any identity assignment, rerun matching, or measure a new global IDF1.

Use scene 001, cameras 4/5/8, frames 2..299 and the exact frozen tracked boxes.
The source appearance setting is read from the controlled identity report
(mean descriptors, cosine threshold 0.70 for the real baseline).

## Geometry and labeling

For each raw tracked xyxy box, project `((x1+x2)/2, y2)` through inverse H
onto world Z=0. Do not clip or round the box before projection. This image
point is a ground-contact approximation and can be wrong under occlusion.
After homogeneous division, pair distance is Euclidean distance between
these two world-plane points. Units remain native dataset coordinates.

GT labels are assigned offline using the existing one-to-one IoU procedure:
clip boxes for matching, require IoU >= 0.5, maximize valid match count then
summed IoU. Labeling is recomputed and must reproduce the labels and embedding
row mapping saved in the grouping trace. GT world coordinates and global
identity assignments are not used to calculate projected pair distances.

- Both GT labels known and equal: `same`.
- Both known and different: `different`.
- Either unknown: `unknown`, never automatically a negative pair.

These are diagnostic labels from 2D matching, not manually verified truth for
every prediction. A separate subset requires mutually unique IoU overlap:
the predicted box has exactly one eligible GT box and that GT box has exactly
one eligible prediction. This is stricter spatial evidence, not proof of
visible feet or reliable projection.

## Populations and strata

Each unordered camera pair is counted once at each frame. Repeated frames
remain repeated observations; do not interpret sample counts as independent
people or independent trials.

| Population | Contents |
| --- | --- |
| `all_pairs` | Cartesian product of frozen observations across cameras |
| `appearance_links` | Links accepted by frozen pairwise appearance assignment, before grouping |
| `grouped_links` | Links retained in the frozen complete-support groups |

The appearance subset is important because all-pair negatives contain many
easy, unrelated observations. Reducing all-pair negatives does not establish
that actual appearance mistakes are reduced.

For each population, output all observations, the subset where both boxes
are fully inside their images, and the subset with mutually unique GT
matches. These subsets overlap; they are diagnostic strata, not a combined
runtime quality filter. Results are available per camera pair and pooled.

Source positive-pair denominators, accepted-link categories and grouping
counters must reproduce the earlier report. Source files are checksum-checked
before and after the run. Image decoding, models and embeddings are not run.

## Hypothetical distance checks

The initial explicit grid is 0.25, 0.5, 1, 2, 3, 5 and 10 in native units.
It is a broad diagnostic grid, not calibrated deployment parameters.
A finite distance <= the candidate threshold is retained; larger is rejected.
If a projection is unavailable, the pair is recorded as deferred. In particular,
missing geometry is not counted as successful rejection of a wrong pair.

For each class, report total, measurable, retained, rejected and deferred
counts. `rejected_fraction_measurable` uses only finite distances of that
class as its denominator. Empty denominators give null, not zero. Report
unknown-labeled pairs separately even when their distance is measurable.

The sweep filters fixed pairs only. Applying geometry before pairwise
assignment could change the winning matches. Applying it before grouping
could change groups. Applying it to merge confirmation could change later
identity state. None of these runtime effects is simulated by this diagnostic.
Do not translate its counts into a claimed global IDF1 improvement.

## Run

From the WSL repository root with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_geometry_pairs.py

PYTHONPATH="$PWD/src" python scripts/evaluate_geometry_pairs.py \
  --geometry-report artifacts/geometry_audit/20261007T110015275411Z/report.json \
  --identity-report artifacts/controlled_merge/20261007T104647560154Z/report.json \
  --distances 0.25 0.5 1 2 3 5 10
```

Uses the existing NumPy/SciPy and project helper scripts; no new installation.

## Artifacts

A new `artifacts/geometry_pairs/<UTC-run>/` directory contains:

- `report.json`: inputs, checksums, protocol, summaries, limits and the known
  frame-149 camera-5/local-9 versus camera-8/local-17 pair, if present.
- `observations.jsonl.gz`: raw boxes, projected footpoints, diagnostic labels
  and quality-stratum flags for each observation.
- `pairs.csv.gz`: every pair with projected distance, labels, source rows,
  flags and membership in the appearance/grouped populations.
- `distance_summary.csv`: distance quantiles by population, stratum, camera
  pair and diagnostic class.
- `threshold_sweep.csv`: explicit hypothetical retention/rejection counts.

## Interpretation and limitations

Compare true-pair rejection with different-person rejection, especially within
appearance links. Examine tails and camera-specific results. A tight threshold
may reject correct associations when footpoints are unreliable. A loose one
may retain nearby different people. Being inside the image or uniquely matched
to GT does not establish visible ground contact.

The GT-box localization quantiles from the earlier audit are not pair-distance
thresholds for tracker boxes: the inputs differ, and pair distances depend on
both projected points. This experiment reuses a short training fragment;
threshold selection and final evaluation require a separate protocol and
independent data. No deployment threshold is selected here.

Reference for the camera/plane mapping: Hartley and Zisserman (2004),
*Multiple View Geometry in Computer Vision*, second edition:
https://robots.ox.ac.uk/~vgg/publications/2004/Hartley04c/
