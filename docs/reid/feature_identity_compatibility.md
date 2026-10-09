# Dimension-explicit geometry association and global identity

This bridge prepares the association stage for CLIP-ReID's 1280-dimensional
features while preserving the frozen 512-dimensional OSNet reference. It adds
new modules; the legacy runtime and saved predictions remain unchanged.

## Contract

- Each batch carries a `FeatureSpace`: pinned model/preprocessing configuration
  SHA-256, feature dimension and L2 normalization.
- Pairwise comparisons and grouping reject mixed feature spaces, even when the
  dimensions match. Observation keys and exact scene timestamps remain required.
- Geometry gates, partial-assignment objective, complete-support grouping and
  controlled identity merging retain the existing policy.
- The global manager consumes scalar associations and groups. It does not need
  a fixed embedding dimension.
- Fully outside, unencoded observations remain singleton observations, using
  the existing identity-stage behavior.

`FeatureIdentityStage` is a separate adapter. Its use does not modify
`IdentityStage`, the local tracker, continuity decisions or production settings.
The geometry adapter intentionally mirrors the frozen algorithm and reuses its
solver and grouping implementation. This small duplication isolates the change;
the full replay gate detects divergence from the legacy implementation.

## Verification

```bash
PYTHONPATH="$PWD/src" python scripts/check_feature_association.py

PYTHONPATH="$PWD/src" python scripts/check_feature_identity_replay.py \
  --history-report artifacts/feature_history/20261009T185808088973Z/report.json

PYTHONPATH="$PWD/src" python scripts/check_feature_identity_replay.py \
  --history-report artifacts/feature_history/20261009T185608930923Z/report.json
```

The first command uses synthetic fixtures for model-space isolation, empty
cameras, missing geometry, merges, expiry and 1280-dimensional input. Padding
appears only in this synthetic fixture to preserve known cosine similarities;
real CLIP and OSNet embeddings are never padded or compared with each other.

Each full replay verifies the completed history report, its input and output
checksums, and frozen observation-to-segment mappings. It reconstructs OSNet
histories on the exact saved local observations. Legacy and new adapters must
produce identical pairs, groups, projected points, global decisions and manager
state. Every full global output must also equal the saved enabled-continuity
reference. Timings are excluded from equality comparisons.

Stored CLIP means are checked for dimension, normalization and matching sample
participation. They are **not** used for identity decisions in this gate. No
models, video decoding, local tracking or GT evaluation run here. Results are
written to a new `artifacts/feature_identity_checks/<run>/` directory; failed
runs have `completed: false` and cannot serve as a successful gate.

During preparation, a separate serialized 50-round fixture also exercised the
replay reader with segment cuts, empty rounds, expiry and outside observations.
Changed source-row mapping and changed frozen global assignments were rejected.
The real full-sequence checks must be run against the local project artifacts.

## Interpretation and next experiment

A passing result establishes implementation compatibility, not CLIP-ReID global
tracking quality. The next paired experiment can substitute CLIP descriptors in
the cross-camera stage while keeping local observations and segment cuts fixed.
That will measure the effect of cross-camera descriptors, not replacement of
OSNet inside the local tracker or continuity policy.

OSNet cosine thresholds do not become calibrated CLIP thresholds merely because
both feature spaces are L2-normalized. Any threshold transfer must be labeled
as an experimental setting; policy development and final validation remain
separate. These changes select no model or deployment threshold.
