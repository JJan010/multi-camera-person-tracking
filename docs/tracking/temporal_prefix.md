# Shared-prefix repeatability before temporal transfer

The 3600-round run extends the original 1800-round development run. Matching
printed counts or rounded IDF1 values do not prove that its shared prefix is
identical. Check the frozen inputs and outputs before attributing differences
in later tracker experiments to the appearance rule.

Inputs:

- Short pipeline: `20261007T133915524799Z`.
- Short detection cache: `20261007T154414673309Z` (113250 encoded candidates).
- Long pipeline: `20261007T164721073889Z`.
- Long detection cache: `20261007T165444228264Z` (220063 encoded candidates).

The new cache reported 185 identical-crop reference samples with minimum
cosine 0.9999999999999654 and maximum absolute difference
1.4901161193847656e-07. This is sampled agreement with tracked-crop embeddings;
it is distinct from the complete cached-prefix comparison below.

`scripts/check_temporal_prefix.py` verifies the report and artifact checksums,
unchanged pipeline and encoder configuration, recorded source asset hashes,
and hashes of the pre-existing pipeline source files. Newly added source files
do not invalidate the check if all pre-existing files remain unchanged.

It compares all frames of the shorter run against the longer prefix:

- Local records, including detector boxes/scores, tracked boxes/IDs and feature
  row mapping.
- Global identity records and association decision records.
- Candidate records and their exact embedding-row mapping.
- Original tracked embeddings, causal mean embeddings and cached candidate
  embeddings, using bounded chunks and exact float32 equality.

Only the declared run identifiers are normalized. No scores, identity numbers,
decision fields, timestamps or feature values are removed or rounded. Comparison
failure writes `passed: false` when the comparison completes; malformed inputs
or checksum/configuration violations fail immediately. Differences should be
inspected, not hidden by loosening a tolerance after seeing the result.

Run from the project directory in WSL:

```bash
PYTHONPATH="$PWD/src" python scripts/check_temporal_prefix.py \
  --short-cache-report artifacts/detection_embeddings/20261007T154414673309Z/report.json \
  --long-cache-report artifacts/detection_embeddings/20261007T165444228264Z/report.json
```

Outputs go to a new `artifacts/temporal_prefix/<run_id>/report.json`.
The check uses CPU and disk reads, with no decoder, model execution or GT.
It establishes repeatability of the shared prefix, not quality on the unseen
suffix. A `--self-test` option exercises scope normalization, changed IDs and
a one-ULP vector difference. A separate synthetic end-to-end report fixture
also exercised the complete script during development.

After a passed prefix check, reproduce the installed tracker on the complete
3600-round frozen cache using the existing exact-replay gate:

```bash
PYTHONPATH="$PWD/src" python scripts/check_frozen_bytetrack_replay.py \
  --cache-report artifacts/detection_embeddings/20261007T165444228264Z/report.json
```

Expected source baseline coverage is 10800 camera updates, 220063 detections
and 180496 returned track observations. This includes runtime frames 0 and 1;
evaluation excludes them. The gate produces the input report required by
`experiment_appearance_bytetrack.py`. No threshold is changed here.

Subsequent paired evaluation must separately score frames 1800..3599 with
fresh metric accumulators, preserving tracker/identity state established from
frame zero. Full-prefix metrics remain useful for identity continuity across
the boundary. The second minute is a temporal transfer check within the same
scene, not an independent-scene test.
