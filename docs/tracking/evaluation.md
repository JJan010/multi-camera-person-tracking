# Local tracking evaluation

## Run

From the repository root, with the project virtual environment active:

    python scripts/check_tracking_metrics.py
    python scripts/evaluate_local_tracking.py

Each evaluation creates a new directory under artifacts/tracking_evaluation,
containing metrics.csv, report.json and per-camera event CSV files.

## Protocol

- Integration subset: scene_001, cameras 4, 5 and 8.
- Evaluate video frames 2–299, inclusive: 298 frames per camera.
- Ground-truth frame index equals the zero-based video frame index.
- Clip GT and predicted boxes to the 1920x1080 image.
- Exclude GT boxes with zero area after clipping.
- Retain fully outside predictions as unmatched predictions.
- Spatial matching requires IoU >= 0.5.
- Evaluator: motmetrics 1.4.0, SciPy assignment solver.
- ID switch history has no time limit.
- Evaluate each camera independently.
- ALL_CAMERAS_LOCAL pools counts from the independent evaluations.
  It is not cross-camera IDF1 or an arithmetic mean of camera IDF1 scores.
- Precision and recall describe tracker output boxes.

## Baseline

See baseline_metrics.json for results, versions and input/script hashes.

Pooled local IDF1: approximately 92.35%.
ID switches: 16. False positives: 171. Misses: 668.

These results describe a short integration sequence, not held-out test
performance or the official AI City Challenge evaluation.

The earlier per-frame diagnostic uses different matching rules.
Its event counts should not be treated as evaluator ID switch counts.

## References

- Identity metrics: https://arxiv.org/abs/1609.01775
- Evaluator: https://pypi.org/project/motmetrics/1.4.0/
