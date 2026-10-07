# Causal global identity baseline results

Policy: local_anchor_no_merge_v1.
Source replay: 20261007T093510175809Z.
Evaluation: 20261007T094804760266Z.
Scene: scene_001; cameras 4/5/8; frames 2..299.
Idle duration: 1 second, an integration setting.

Evaluation uses one shared identity assignment across all cameras
and frames, with same-camera/same-frame IoU >= 0.5.
GT observations: 16317. Prediction observations: 15828.

The no-cross-camera local-ID control achieved global IDF1=42.63%.

Selected diagnostic results:
- latest, threshold 0.50: IDF1=70.23%, +27.60 percentage points.
- mean, threshold 0.60: IDF1=71.17%, +28.53 percentage points.
- mean, threshold 0.70: IDF1=72.73%, +30.10 percentage points.
- mean, threshold 0.80: IDF1=54.63%, +12.00 percentage points.

The highest observed score was mean/0.70:
IDTP=11690, IDFP=4138, IDFN=4627.
It recovered 4838 more identity-correct observations than the control.

High framewise link precision did not guarantee high global IDF1.
The no-merge policy can preserve fragmentation and propagate wrong
initial associations or local-track identity switches. Aggregate
metrics alone do not establish the contribution of each failure mode.

No deployment threshold was selected. This reused training-scene
fragment is an integration diagnostic, not independent validation.
The earlier pooled local-camera IDF1 is a different evaluation.

Next: inspect identity fragmentation and incorrect identity
associations in the frozen outputs before changing merge policy.
