# CLIP retired-reference probe: G11 to G76

This read-only development diagnostic investigates the first emitted birth of
CLIP global ID 76 and the previously expired ID 11 in the frozen scene-001 run.
The case was selected retrospectively from GT-15 evidence. It is not a recovery
policy and does not establish that two IDs belong to the same person.

## Run

```bash
PYTHONPATH="$PWD/src" python scripts/check_clipreid_return_probe.py

PYTHONPATH="$PWD/src" python scripts/probe_clipreid_return.py \
  --experiment-report artifacts/clipreid_global_experiment/20261009T191501187432Z/report.json \
  --old-gid 11 --new-gid 76
```

No GPU, inference, decoding or tracker replay is required. The script joins the
frozen global outputs, original observation keys, internal segment keys, CLIP
raw rows and stored history means. Every encoded row must be consumed exactly
once. Boxes and scores must match the saved observations. Source checksums and
unit-normalized feature matrices are verified.

## Reference alternatives

The query is the normalized mean of the current stored CLIP history descriptors
for observations assigned to the new GID at birth. The three diagnostic
reference alternatives are:

| Mode | Reference |
|---|---|
| `last_mean` | One latest encoded observation's stored history mean; simultaneous ties use camera and local ID order. |
| `confidence_gallery` | Normalized average of up to eight most recent raw CLIP samples per camera with confidence >= 0.5. |
| `confidence_border_gallery` | Same raw gallery, also requiring normalized raw-box clearance >= 0.01 on every image edge. |

The admission values reuse the existing sample-admission hypothesis. They are
not newly calibrated here and do not prove that a crop is suitable. The gallery
modes differ from `last_mean` in selection, time span and aggregation; results
cannot isolate crop quality alone. Gallery samples are equally weighted, not
camera-balanced. Sample timestamps are preserved and rejected observations do
not refresh accepted samples. Cancellation produces an unavailable reference.

References include only observations actually emitted under that exact GID.
Historical samples from absorbed IDs are not inherited or retroactively relabeled.
The last-seen timestamp includes unencoded observations; descriptor timestamps
refer to encoded sample rows. Stored history means may include earlier rows,
whose frame provenance is retained in the output.

At the query frame, ranking considers all previously expired IDs except live
or absorbed ones. This deliberately has no archive-age, motion, similarity or
ambiguity gate: it measures appearance separability, not recovery eligibility.
All gallery observations predate the return. The candidate set is finite for
this recorded run; no bounded production archive is claimed.

Ground distances compare query points with the retired ID's last encoded
observation point. They use raw box-bottom projection and native calibration
units. They are context only, not an allowed displacement or speed threshold.

## Offline annotation and interpretation

Rankings and selected samples are frozen in memory before GT loading. A second
pass annotates those rows using mutually unique IoU matches while retaining all
same-camera prediction candidates. It cannot change ranking or selection, and
an explicit equality check verifies that only diagnostic labels were added.
A label on a stored mean row labels its current observation, not all samples
that contributed to its history mean. Unknown labels remain unknown.

The output gives actual birth/last-seen/expiry frames, per-camera last sightings,
reference ages and quality flags, target cosine/rank, the top five competitors,
and target cosine minus the best other candidate. That margin is descriptive;
no acceptance threshold is chosen.

New files under `artifacts/clipreid_return_probe/<run>/` are `report.json` and
`rankings_before_gt.json`. Assignments and the runtime baseline remain unchanged.

Synthetic checks cover ID zero, a degraded final descriptor, earlier accepted
samples, competitor ranks, future-sample exclusion, no GT access during ranking,
offline label separation, gallery bounds, cancellation and missing references.
