# Retired appearance references: frozen scene_041 probe

This diagnostic reconstructs the enabled continuity experiment without enabling
identity recovery. It reuses saved candidate embeddings, original tracked boxes
and recorded segment bindings. The original history and global association code
must reproduce EVERY saved enabled runtime record exactly over all 3600 rounds.
It runs on CPU, without decoding, model inference or GT inputs.

Snapshot evidence follows the existing RecoveryIdentityStage evidence adapter
and RecoveryIdentityManager snapshot update method. A snapshot is refreshed only
when the identity's complete currently visible membership is a supported current
group and complete appearance/geometry evidence exists. Otherwise its old evidence
source times are preserved. The same aggregation and cancellation behavior apply.
History provenance, original observation keys and snapshot update frame are saved.
No claim is made that the resulting reference is a pure description of one person.

At identity expiration, keep its snapshot in a diagnostic catalogue. Unlike the
real archive, this catalogue deliberately does NOT evict by age or capacity. This
lets us inspect references the current policy would have discarded. Appearance
rankings from this catalogue are not live recovery decisions or benchmark scores.

The source hypothesis remains: maximum evidence age 5 seconds, strict cosine
similarity > 0.8, ambiguity margin 0.05, fixed radius 2 native coordinate units,
zero motion expansion, capacity 128. The current recovery registry considers
only wholly new unanchored groups at birth. It does not reconsider already bound
tracks when later GT evidence becomes clearer.

Cases selected offline from the earlier diagnostic:

| New enabled GID | Old reference GID |
| --- | --- |
| 81, 89, 96 | 44 |
| 108, 127 | 4 |
| 165 | 96 |

This is case investigation after seeing GT, not an independent retrieval test.
Runtime receives only these diagnostic numeric ID selectors, never GT labels.
Missing unanchored births are reported rather than fabricating a query at an
arbitrary later frame.

For each observed target birth, the report contains original source members,
reference retirement/provenance, cosine similarity, ground distance, source ages,
individual age/appearance/geometry gates, and the top five retired references by
appearance with gates ignored. Query descriptor age is also checked separately.
No mutual-best margin or competition between simultaneous return groups is tested.
Passing these individual gates therefore does not prove a return would be accepted.

Run in the WSL project root:

```bash
PYTHONPATH="$PWD/src" python scripts/probe_retired_descriptors.py --self-check
PYTHONPATH="$PWD/src" python scripts/probe_retired_descriptors.py \
  --validation-report artifacts/appearance_continuity_validation/20261008T111045386342Z/report.json
```

Output: artifacts/retired_descriptor_probe/<run-id>/report.json.
Frozen inputs are checksummed before and after replay. Predictions and the default
runtime stay unchanged. No threshold is fitted and archive recovery stays off.

Technical checks before delivery cover inclusive age/distance, strict similarity,
missing evidence and ID zero. Synthetic integration fixtures reproduce complete
enabled state with constant and rotating embeddings, segment cuts, empty cameras,
outside observations and a long absence that expires IDs. They exercise actual
retirement snapshot comparisons. They are not model-quality measurements.
