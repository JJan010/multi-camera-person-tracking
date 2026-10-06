# OSNet snapshot baseline and failure analysis

## Scope and implementation

This is an integration diagnostic on `MTMC_Tracking_2024/train/scene_001`,
frame 150 (5.0 seconds), cameras 4, 5 and 8. Dataset revision:
`2cbe9563cbe9f47f846e5c871ee994572bbbc60e`.
The result describes retrieval within one synchronized synthetic frame.
Sequence-level global tracking quality and generalization remain to be evaluated.

- OSNet x1.0 with the authors' MSMT17 combineall Re-ID checkpoint.
- Upstream source revision: `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`.
- Pinned source, license and weight hashes: `configs/models/osnet_x1_0_msmt17.json`.
- Source module is vendored unmodified with its MIT license; weights remain under ignored `artifacts/`.
- Strict state-dict loading, evaluation mode, CUDA FP32, TF32 disabled.
- RGB PNG crops resized directly to H=256, W=128 using PIL bilinear interpolation.
- Pixel values divided by 255, then channel mean `[0.485, 0.456, 0.406]`
  and standard deviation `[0.229, 0.224, 0.225]` applied.
- All 57 saved observations processed in one batch: 16 / 22 / 19 per camera.
- Features have 512 dimensions and are L2-normalized per observation.
- Observed normalized norms: 0.99999988 through 1.00000012.
- Every embedding retains its camera, local ID, frame, crop provenance and checksum.

## Evaluation protocol

GT is used only by the evaluator. A crop's input track box is matched to a GT
box within the same camera and frame. Boxes are clipped to the 1920x1080 image;
zero-area GT boxes are excluded. The assignment first maximizes the number of
valid matches at IoU >= 0.5, then maximizes total IoU, with one-to-one matching.
This labels each observation independently of ByteTrack identity continuity.
All 57 crops were matched; no GT boxes were left unmatched in this frame.
Minimum assigned IoU was 0.7565365951.

A query is compared with all observed crops in one other camera using the dot
product of normalized embeddings. GT does not determine the ranking.
Unmatched gallery observations would remain as distractors. A query contributes
to Rank-k only if it has an assigned GT identity and an observation assigned to
that identity exists in the target gallery. Others are counted separately.
Ties use embedding row order. No similarity threshold or global IDs are used.
This is our snapshot protocol, not an official benchmark implementation.

## Results

| Query camera -> gallery camera | Eligible / all queries | Rank-1 | Rank-3 |
| --- | ---: | ---: | ---: |
| 4 -> 5 | 15 / 16 | 14/15 = 93.3% | 14/15 = 93.3% |
| 4 -> 8 | 16 / 16 | 14/16 = 87.5% | 16/16 = 100.0% |
| 5 -> 4 | 15 / 22 | 14/15 = 93.3% | 14/15 = 93.3% |
| 5 -> 8 | 17 / 22 | 16/17 = 94.1% | 16/17 = 94.1% |
| 8 -> 4 | 16 / 19 | 14/16 = 87.5% | 16/16 = 100.0% |
| 8 -> 5 | 17 / 19 | 16/17 = 94.1% | 16/17 = 94.1% |
| Pooled | 96 / 114 | 88/96 = 91.7% | 92/96 = 95.8% |

18 directed queries had no positive in their target gallery; no queries lacked
an assigned GT label. Observations are reused across directions. The 96 queries
are not 96 distinct people or independent statistical samples. The 18 cases
without a positive require an eventual no-match decision; this diagnostic does
not measure the quality of that decision.

## Eight failed directed queries

`cam4:17` denotes a local tracker ID in camera 4, scoped to this tracking run.
The ranks below refer to the true GT-matched candidate.

| Correct observation pair / direction | Same GT | True-match similarity | True-match rank | Failed queries |
| --- | ---: | ---: | --- | ---: |
| cam4:17 <-> cam5:16 | 10 | 0.358163 | 11 / 12 | 2 |
| cam4:17 <-> cam8:14 | 10 | 0.516321 | 2 / 3 | 2 |
| cam5:24 <-> cam8:13 | 11 | 0.392509 | 12 / 14 | 2 |
| cam4:1 -> cam8:18 | 8 | 0.649190 | 2 | 1 |
| cam8:3 -> cam4:8 | 5 | 0.736854 | 2 | 1 |

Six of eight errors involve `cam4:17` or `cam5:24`, either as query or as the
true gallery candidate. Contact-sheet inspection shows severe image-boundary
truncation: `cam4:17` contains a body fragment at the left edge; `cam5:24`
contains an upper-body fragment at the bottom edge. Their original crops are
57x161 and 81x95 pixels respectively. This supports crop quality as a hypothesis
to test, rather than demonstrating a causal improvement from filtering.
Both had `inside_image_fraction=1.0`: a box already clipped by an upstream stage
can be fully inside the image while the person is truncated. The statistic is
not a measure of person visibility.

Useful controls within this snapshot:

- GT 10 matches correctly between `cam5:16` and `cam8:14` in both directions,
  with similarity 0.797112, while comparisons with `cam4:17` fail.
- The truncated `cam5:24` still matches correctly with `cam4:15` in both
  directions (0.688593). Discarding border crops can also remove useful evidence.
- `cam4:1` prefers `cam8:19` (different GT, 0.707137) over `cam8:18`
  (same GT, 0.649190). Perspective and visible body regions differ.
- `cam8:3` prefers `cam4:4` (different GT, 0.775869) over `cam4:8`
  (same GT, 0.736854). Dark clothing is visually similar in these crops.

The last two visual explanations are hypotheses, not measured causal effects.
Full per-query top-3 results are preserved in `snapshot_rankings.csv`.

## Similarity overlap

Each unordered cross-camera pair with assigned GT is counted once.

| Pair label | Count | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Same GT identity | 48 | 0.358163 | 0.799581 | 0.941972 |
| Different GT identities | 1026 | 0.216157 | 0.406044 | 0.775869 |

The score ranges overlap. A single cosine threshold cannot perfectly separate
these observed positive and negative pairs. Cosine similarity is not a calibrated
probability of identity. Threshold selection requires a separate validation
protocol, including pairs with no counterpart and the cost of false merges.

## Decisions and next steps

Preserve this baseline with all original crops and settings. Do not remove
specific failed identities or tune a threshold to this frame. The next engineering
step is a reusable OSNet adapter with the same preprocessing and observation
mapping, followed by a controlled latency/throughput measurement. Any experiment
with crop selection or temporal feature aggregation must report both retrieval
quality and coverage, and must consider local ID switches before averaging track
features. Extend evaluation in time and to held-out scenes before making quality
claims about the full system.

No OSNet performance figure is established by this functional snapshot test.

## Reproduction

Run from the project root in WSL with the project virtual environment active.
Existing crop and embedding manifests are immutable inputs. A new run writes a
new timestamped artifact directory.

    python scripts/prepare_osnet_assets.py
    PYTHONPATH="$PWD/src" python scripts/preview_osnet_embeddings.py --manifest artifacts/reid_crops/20261006T191941008562Z/manifest.json

Use the embedding manifest printed by that run for a new evaluation. The recorded
baseline used:

    python scripts/evaluate_reid_snapshot.py --manifest artifacts/reid_embeddings/20261006T194219632494Z/manifest.json

Original evaluation artifact:
`artifacts/reid_evaluation/20261006T194834726351Z/report.json`.
Input hashes, evaluator hash, package versions and exact metrics are preserved
in `snapshot_metrics.json`; raw crops and weights remain local artifacts.

## References

- Zhou et al., *Omni-Scale Feature Learning for Person Re-Identification*, ICCV 2019:
  https://arxiv.org/abs/1905.00953
- Authors' test preprocessing:
  https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/data/transforms.html
- Authors' ranking metrics implementation (query/gallery and valid-query concepts):
  https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/metrics/rank.html
