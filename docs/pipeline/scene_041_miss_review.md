# Frozen camera 364 missed-observation review

Source: scene_pair_evaluation/20261007T214235770374Z, staged.
The local evaluation contains 2988 missed GT observations in camera 364.
The diagnostic reproduces the camera's local CLEAR and identity metrics,
then reads actual `MISS` events from motmetrics 1.4.0. All observed
predictions are joined to the detector cache by their exact saved candidate
index, with box and confidence equality checked. No models or trackers run.

Three mutually exclusive descriptions of each missed GT observation:

1. `no_admissible_cached_detection`: no cached detection meets the fixed
   evaluation IoU gate with that GT box.
2. `admissible_detection_not_emitted`: at least one cached detection meets
   the gate, but no emitted tracker observation does.
3. `admissible_prediction_unmatched`: an emitted observation meets the
   gate, but this GT was left unmatched by the chronological evaluator.

The candidate cache was collected at a fixed detector threshold. Category 1
does not establish the absence of predictions below that threshold.
Categories 2 and 3 are not automatic proof of an incorrect tracking decision:
one box can overlap multiple people, especially during occlusion.
The evaluation's assignment is distinct from the tracker's association.

Clipped GT dimensions, partial image-boundary clipping, the highest-IoU
candidate and maximum score among admissible candidates are recorded.
These are descriptions, not fitted filtering rules. Small boxes alone do
not establish occlusion or poor detection quality.

For each category, take the two GT identities with the most misses (ties by
GT ID), then their median missed frame in that category. Up to six previews
show the full scene, raw context and annotated context. Green is GT, red is
the selected missed GT, cyan is emitted tracks, yellow is the cached
detection with the highest IoU to the target (even if below the gate).
The chosen examples are not a random prevalence sample.

From the project root in WSL:

```bash
PYTHONPATH="$PWD/src" python scripts/check_scene_misses.py
PYTHONPATH="$PWD/src" python scripts/diagnose_scene_misses.py \
  --evaluation-report artifacts/scene_pair_evaluation/20261007T214235770374Z/report.json \
  --camera 364 --variant staged
```

Outputs are saved in a fresh `artifacts/scene_misses/<run>` directory:
report.json, misses.csv, up to six JPEGs and visual_review.zip.
The existing FP helper is reused without changing it. No dependency
installation, annotation change, threshold change or ROI selection is needed.

Reference for the use of low-score detections to recover occluded objects:
Zhang et al., ByteTrack, ECCV 2022, https://arxiv.org/abs/2110.06864.
The diagnostic categories above are project-specific, not paper metrics.
