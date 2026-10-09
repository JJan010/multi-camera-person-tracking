# Fixed-grid OSNet / CLIP-ReID comparison

The first real snapshot improved from OSNet 88/96 Rank-1 (91.67%) to
CLIP-ReID 93/96 (96.88%). Paired transitions: 86 both correct, 7 improved,
2 worsened, 1 both wrong. Single-positive mAP increased from 93.92% to
98.44%. Source comparison run: `20261008T134442155626Z`.
This motivates broader measurement; it does not establish model superiority.

## Fixed protocol before execution

Use the existing two-minute scene_001 and scene_041 continuity traces.
The original local camera records are frozen and identical across enabled
and disabled continuity variants. We read the enabled trace's original
camera/local IDs and accepted detection boxes; global assignments and cuts
are not inputs to the new encoders or retrieval.

Sampling is defined in `configs/reid/model_comparison_temporal.json`:
120 frames, `2,32,...,3572`, selected every second at 30 FPS. No sample
selection by GT, confidence, model output, identity or observed difficulty.
All positive-area crops from the selected frozen local observations are used.
Fully outside observations retain provenance but have no embedding.

Decode the same pinned videos sequentially with exact PTS and dimension
checks. Crop geometry follows the existing floor/ceil clipping rule. Both
encoders receive the same original RGB crop objects; each applies its own
model-specific preprocessing. Encode in batches of at most 16, FP32, TF32 off.
RF-DETR, ByteTrack, global history and identity managers are not executed.

OSNet is recomputed on every selected crop and checked against its original
candidate-cache row (max absolute error <= 1e-5; cosine >= 1-1e-5). Retrieval
uses the original cached OSNet values after this parity check. CLIP-ReID
features and the keyed crop records are frozen before GT labeling.
Thus the original 512-D baseline and new 1280-D features remain distinct.

## Three retrieval protocols

| Protocol | Query | Gallery |
| --- | --- | --- |
| same_time_cross_camera | camera A at t | camera B at t, A != B |
| plus_10s_cross_camera | camera A at t | camera B at t+10s, A != B |
| plus_10s_same_camera | camera A at t | camera A at t+10s |

Only already sampled frames form galleries. A 10-second protocol therefore
has 110 query times, rather than 120. All camera directions are considered.
The future gallery is an offline quality measurement, not a causal identity
recovery algorithm; it does not imply the person disappeared in between.

GT matching is shared between models and follows the original snapshot
protocol: clip boxes, IoU >= 0.5, maximize admissible assignment cardinality
then total IoU. All original local predictions participate in labeling.
The assigned labels are spatial diagnostics, not appearance-model inputs.

Queries require assigned GT and one assigned positive in the gallery to
contribute to rank metrics. Unmatched queries and missing gallery positives
are counted separately. Unmatched gallery crops remain distractors. Stable
ties use ascending original local ID within the camera/frame, never GT.
Each gallery is one camera/frame with at most one assigned positive, so
AP = reciprocal positive rank and mAP = MRR for this protocol.

Reports include pooled and directional Rank-1/Rank-3/mAP, paired Rank-1
transitions, known same/different-pair similarity distributions, and summaries
by the query's first/second minute. A +10-second gallery may fall in the next
minute; those summaries are not independently reset tracking windows.

## WSL execution

From `/home/jakjan/projects/multi-camera-person-tracking` with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_reid_temporal.py

PYTHONPATH="$PWD/src" python scripts/compare_reid_temporal.py \
  --source-report artifacts/appearance_continuity_experiment/20261008T105229140571Z/report.json \
  --model-check-report artifacts/clipreid_model_checks/20261008T133550336467Z/report.json

PYTHONPATH="$PWD/src" python scripts/compare_reid_temporal.py \
  --source-report artifacts/appearance_continuity_validation/20261008T111045386342Z/report.json \
  --model-check-report artifacts/clipreid_model_checks/20261008T133550336467Z/report.json
```

Run the scenes sequentially. Outputs go to
`artifacts/reid_temporal_comparison/<run_id>/`:
`report.json`, `run_status.json`, both embedding matrices, keyed observations,
GT-labeled observations, frozen feature checksums and compressed per-query
rankings. RGB crop hashes are recorded, but no large PNG collection is saved.
Input, model-asset, source-code and output hashes are retained.

## Limits and decision

The frozen local tracks were produced with OSNet-aware logic. This comparison
measures replacement appearance features conditional on that fixed population;
it does not estimate the end-to-end effect of CLIP-ReID on local tracking.

Both scenes have informed development. Sampling more frames does not turn
them into an independent test set. Queries share people and frames and are
not independent statistical trials. Do not tune cosine thresholds from this
report and claim held-out performance. No latency benchmark, threshold
selection or global-ID improvement is produced here.

If CLIP-ReID's retrieval improvement persists across times/scenes, the next
step is a dimension-aware history/association integration with exact disabled
OSNet parity, model-specific threshold calibration on development data and
then global-ID evaluation. ONNX/TensorRT remains a later measured optimization.

Synthetic contract tests cover arbitrary cameras/sizes, ID zero, empty
camera/GT slots, outside crops, deterministic sampling, keyed row mapping,
reference parity, unknown distractors, temporal positives and rejection of
changed feature mappings. Real video/GPU runs remain the WSL execution step.

Model sources and scientific reference: `docs/reid/clipreid_setup.md` and
https://arxiv.org/abs/2211.13977. This temporal sampling/evaluation protocol is
our declared project diagnostic, not the paper's official benchmark.
