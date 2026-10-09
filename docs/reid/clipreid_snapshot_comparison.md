# Paired CLIP-ReID / OSNet snapshot comparison

Status: integration diagnostic on existing saved crops. No runtime model selection.

The initial scene_001 frame-150 snapshot contains 57 saved PNG crops from
cameras 4, 5 and 8. Its original OSNet result is 88/96 eligible directed
queries at Rank-1. This comparison must reproduce that result before it can
report CLIP-ReID alongside it.

## Fixed inputs and protocol

The script verifies the existing OSNet embedding manifest and matrix against
the original retrieval report, verifies the crop manifest, every RGB PNG hash,
observation key, timestamp and row index, and requires the successful matching
CLIP-ReID model check. No detector, tracker, crop-selection rule or OSNet
inference is rerun.

The same crop bytes are encoded with CLIP-ReID's official image preprocessing
and 1280-D representation. Its outputs are saved before GT is read for labeling.
The OSNet reference remains 512-D with its original preprocessing. Vectors
from different models are never compared directly or averaged together.

GT labeling reuses the original snapshot evaluator: clipped boxes, inclusive
IoU >= 0.5, maximum admissible assignment cardinality then total IoU. Original
labels and all OSNet direction/pooled metrics must reproduce exactly.

For each ordered pair of cameras:

- every crop in the gallery stays present, including unmatched distractors;
- a query is eligible only if it has assigned GT and a corresponding assigned
  gallery observation;
- similarity is the dot product of normalized vectors from the same model;
- ties retain original embedding-row order;
- no similarity threshold or re-ranking is applied.

Rank-1 and Rank-3 measure whether the assigned matching person appears first
or within the first three results. This snapshot has at most one assigned
positive per eligible query, so AP is 1/(positive rank), and mean AP equals
mean reciprocal rank. This is not a standard full-dataset multi-positive
Re-ID benchmark. The report names the field `mAP_single_positive` explicitly.

Pooled directed queries share observations; they are not independent trials.
Do not infer statistical significance or global-ID improvement from this
single frame. A higher or lower result motivates broader multi-time and
cross-scene evaluation, not immediate replacement of OSNet.

## Running in WSL

From `/home/jakjan/projects/multi-camera-person-tracking`, with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_paired_reid_snapshot.py

PYTHONPATH="$PWD/src" python scripts/compare_clipreid_snapshot.py \
  --reference artifacts/reid_embeddings/20261006T194219632494Z/manifest.json \
  --reference-evaluation artifacts/reid_evaluation/20261006T194834726351Z/report.json \
  --model-check-report artifacts/clipreid_model_checks/20261008T133550336467Z/report.json
```

Default: CUDA FP32, batches of at most 16 across camera records, no TF32.
A small batch bounds activation memory. Output:
`artifacts/reid_model_comparison/<run_id>/` containing:

- `report.json`: common population, both models' results, Rank-1 transitions,
  changed ranks, source/code hashes and limitations;
- `clipreid_embeddings.npy`: (57,1280) for the original snapshot;
- `observations.json`: original keyed crop records in exactly the embedding order;
- `osnet_rankings.csv` and `clipreid_rankings.csv`: per-query results.

No latency or throughput claim is made; this run has no dedicated timing
protocol. The optional CPU mode is for technical fixtures and requires a
successful CPU model-check report. It is not the recommended real run.

## Technical checks

Known-answer checks cover ID zero, unknown distractors, absent positives,
stable ties, AP, paired improvements/regressions, different feature dimensions,
invalid vectors and changed query mappings. An end-to-end CPU fixture with
the actual pinned CLIP-ReID weights checked PNG inputs, original reference
reproduction, partial final batches, artifact creation and tampered-crop
rejection before new inference. These synthetic results do not measure
person retrieval on the user's scene.

The model source, paper and pinned asset provenance are documented in
`docs/reid/clipreid_setup.md`. Further tests must include other scene times,
returns after gaps and an unused scene, with model-specific threshold
calibration separated from final evaluation.
