# Confirmed CLIP identity return: one development hypothesis

This experiment adds a causal recovery path to the frozen CLIP cross-camera
pipeline. It preserves detections, local IDs, segment cuts, CLIP embeddings,
history means, geometry association and controlled-merge settings. Recovery
changes the live registry, so later global decisions may legitimately differ.
Earlier output records are never rewritten.

## Declared settings

`configs/association/clipreid_confirmed_return.json` records one retrospective
development hypothesis informed by the scene-001 G11/G76 probe. It is not a
calibrated deployment configuration. No threshold sweep is performed.

- Reference gallery: up to 8 accepted raw samples per camera per live identity,
  confidence >= 0.5, normalized image-edge clearance >= 0.01, age <= 30 seconds.
- Archive: at most 128 identities, with last-seen, descriptor and ground-source
  timestamps all within 30 seconds. An old source timestamp is never refreshed
  on reuse or retirement. Reference descriptor time is the oldest selected raw
  sample; ground position uses accepted samples at their latest timestamp.
- Appearance: cosine strictly > 0.9, mutual unique best match and margin >= 0.05
  in both query and identity directions. All geometry-feasible rivals count
  toward ambiguity, including rivals below the acceptance threshold.
- Motion: distance <= 2 + 1 * elapsed time since the archived ground sample,
  in native calibration units. These units are not assumed to be meters, and
  these motion values are experimental, not physically calibrated.
- Confirmation: at least 3 support rounds AND 0.2 seconds of scene time; gaps
  <= 0.1 seconds; query appearance remains coherent with its initial support
  descriptor (cosine >= 0.8). At 30 FPS, 0.2 seconds requires seven observations.
- Retry: queries are eligible for up to 2 seconds after the provisional ID's
  allocation, including retries following a rejected first observation.
- Query evidence: all currently visible members must be in one supported group
  and have current raw CLIP features, confidence >= 0.5 and clearance >= 0.01.
  A query is not a subset of another identity. Its full retained membership is
  part of the confirmation fingerprint. Identities already participating in
  successful or pending controlled merges are not eligible young return queries.

The archive has no missing-geometry fallback. Weak, absent, missing, incoherent,
ambiguous or interrupted evidence clears confirmation. Membership/reference
changes restart confirmation. An accepted claim removes the old ID from the
archive and remaps every retained binding of the provisional ID atomically.
Superseded numbers are not reused. They may already appear in earlier outputs;
those outputs retain their original IDs.

## Gallery and lifecycle ownership

Stored feature arrays are owned copies. Accepted raw samples remain bounded per
camera, and at most 128 live gallery identities and 128 queries per round are
allowed by this development configuration. Budget excess fails the run rather
than silently dropping identities. Pending confirmation stores one seed vector
per query. Scalar lifecycle bookkeeping grows with identities encountered during
the run; this is not a claim of constant total process memory for unlimited runs.

Samples from absorbed IDs are not retroactively inherited. On a return, only
the provisional identity's actual recent samples transfer to the recovered ID;
the old archived mean is never inserted as a fresh sample. A recovered identity
can subsequently expire and retire again with new timestamps from real evidence.

Allocation accounting includes active IDs, ordinary absorbed IDs, superseded
provisional IDs, and inactive IDs. Inactive accounting is expiration events
minus reactivation events. The last-visible-label diagnostic does not prove
whole-gallery or whole-lifetime identity purity.

## Checks and experiment

```bash
PYTHONPATH="$PWD/src" python scripts/check_confirmed_return.py
PYTHONPATH="$PWD/src" python scripts/check_confirmed_return_registry.py

PYTHONPATH="$PWD/src" python scripts/experiment_confirmed_clip_return.py \
  --experiment-report artifacts/clipreid_global_experiment/20261009T191501187432Z/report.json

PYTHONPATH="$PWD/src" python scripts/experiment_confirmed_clip_return.py \
  --experiment-report artifacts/clipreid_global_experiment/20261009T191621658298Z/report.json
```

The synthetic archive checks exercise confirmation boundaries, ID zero,
competition, below-threshold rivals, explicit missing evidence, capacity,
source ages, invalid-input atomicity, order invariance and gallery admission.
The registry check runs 150 rounds, proving disabled output/state parity,
retry after a weak birth observation, delayed whole-registry remapping, unchanged
past outputs, repeated retirement and failure atomicity.

A separate local serialized fixture verified the experiment reader, a known
full-sequence IDTP gain from recovery, matching full/window metrics against
motmetrics, and the offline return diagnostic. Real recording quality remains
to be measured by the commands above.

The full experiment first verifies frozen source hashes and runs two causal
registries without models, video decoding or GT input. Every disabled CLIP
record must reproduce the source exactly. Predictions are frozen before GT
loading. Evaluation includes all scene camera/time slots, the shared full-run
identity assignment and independent assignments for the two disjoint windows.
Windows do not reset runtime state. Both variants are checked against motmetrics;
disabled metrics, lifecycle and merge diagnostics must reproduce the source.

Outputs are new files under `artifacts/confirmed_clip_return/<run>/`:
`global_tracks.jsonl.gz`, `prediction_freeze.json`, `identity_matching.json`,
`reactivation_diagnostics.json`, `report.json` and `run_status.json`.
Failed runs are explicitly incomplete. The operational baseline is unchanged.

Scene 001 and previously inspected scene 041 are development/transfer evidence.
Do not tune separately on scene 041 or call it an untouched final test. Future
validation must include additional unseen data after policy choices are frozen.
