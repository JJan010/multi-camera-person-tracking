# Frozen detector-candidate embeddings for local tracking experiments

## Inspected tracker

The user supplied the installed `supervision==0.30.7` ByteTrack sources.
The local adapter uses activation threshold 0.5, matching threshold 0.8,
lost buffer 30 and frame rate 30. In this implementation:

- High-confidence association uses scores >= 0.5 and score-fused IoU cost.
- Low-confidence association uses 0.1 < score < 0.5 and only unmatched,
  currently tracked objects from the first association. Its cost limit is 0.5.
- New-track initialization uses activation threshold + 0.1, hence 0.6 here.
- Unconfirmed-track association has its own score-fused IoU step, limit 0.7.
- `update_with_detections()` performs another IoU assignment between returned
  Kalman-filter track boxes and input detections to assign external IDs.

The output remapping must be considered separately when adding appearance
constraints. An internal appearance gate alone does not guarantee the returned
detection obeys that gate. A future implementation must retain the identity
of each selected detection. A control should isolate changes in output mapping
from changes in appearance-based association.

The current implementation does not consume appearance descriptors. These
source facts support an appearance-assisted local-association experiment;
they do not prove which internal match caused the inspected failure.

## Purpose of this step

The baseline encoded tracked boxes after ByteTrack had selected observations.
An experiment that compares competing detections needs their descriptors
before local association. This script decodes the same videos and computes
OSNet features for every candidate in the frozen detector trace. It does not
rerun RF-DETR, run a tracker, read GT, or modify baseline outputs.

Input example:
`artifacts/mtmc_pipeline/20261007T133915524799Z/report.json`.

Scope: all 1800 frames from frame zero, all three cameras, original detector
box order and confidence values. No extra confidence filter or crop-quality
gate is introduced. Candidates rejected by the original detector threshold
cannot be recovered from this trace.

## Candidate identity and output

The persistent key is `(source_run_id, frame_index, camera_id, detection_index)`.
The detection index is only meaningful inside that camera and frame. It is
not a local track ID. The existing OSNet adapter temporarily receives this
index in its `ObservationKey.local_id` field; cache records explicitly call it
`detection_index` and remain separate from tracked-observation archives.

Crop bounds reuse the baseline clipping and floor/ceil convention. Duplicate
boxes retain distinct candidate indices. Fully outside boxes retain a record
with `embedding_row: null`; no fabricated feature is supplied.

CPU decoding produces original RGB frames. One persistent OSNet CUDA FP32
encoder processes candidates from all cameras in bounded batches (default 64).
Features are normalized 512-dimensional float32 vectors. Appended features
are finalized into an NPY archive without accumulating the sequence in RAM.

Outputs under `artifacts/detection_embeddings/<run_id>/`:

- `detections.jsonl`: one scene-time record per frame, preserving all candidates.
- `embeddings.npy`: feature rows joined by explicit `embedding_row`.
- `reference_parity.json`: selected comparisons against identical integer
  crop bounds in the frozen baseline; no track-to-detection assignment claim.
- `report.json`: input/output/code checksums, counters and configuration.
- `run_status.json`: completion state; interrupted output is not a valid cache.

Parity samples come from frames 0, N/4, N/2 and N-1 (integer indices). Changes
in GPU batch composition may cause small numerical differences, so identical
crops require cosine >= 0.9999, rather than bitwise equality. This is a
numerical parity check, not a same-person threshold. Maximum absolute error
is also reported. At least one matched crop sample is required.

## Execution

From the project root with its virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_detection_embedding_cache.py

PYTHONPATH="$PWD/src" python scripts/cache_detection_embeddings.py \
  --run-report artifacts/mtmc_pipeline/20261007T133915524799Z/report.json \
  --batch-size 64 \
  --torch-threads 1
```

No dependency installation is required. All intermediate predictions and
features are stored under ignored `artifacts/`. Runtime includes decoding,
encoding, checks and writes; it is not a throughput comparison of trackers.
This cache deliberately covers all frames, including warmup predictions.

## Verification and subsequent comparison

CPU known-answer checks cover camera/order mapping, duplicate candidates,
empty input, fully outside boxes, exact RGB slices, bounded batch mapping,
identical-crop parity and streamed feature archives. The full command flow was
also exercised with injected synthetic decoder/encoder implementations,
including hashes, final row coverage and completion reporting. This does not
constitute a test of the user's real CUDA inference.

Next, reproduce the existing local tracker using the frozen detections, then
compare a separate appearance-assisted variant on these same candidates.
No appearance threshold is selected here. The inspected L11 failure is a
development case; any claimed general improvement requires broader evaluation
and a separate validation sequence. Global ID quality must be measured again
because changed local IDs affect downstream association and history.

Background: Zhang et al., *ByteTrack: Multi-Object Tracking by Associating
Every Detection Box*, ECCV 2022, https://arxiv.org/abs/2110.06864.
Wojke et al., *Simple Online and Realtime Tracking with a Deep Association
Metric*, ICIP 2017, https://arxiv.org/abs/1703.07402.
