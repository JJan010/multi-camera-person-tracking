# Scene 041: frozen camera-local false-positive review

The paired evaluation `20261007T214235770374Z` favored staged on this selected
validation subset: global IDF1 56.32% versus competitive_iou 55.10%.
Staged remains the reference for further work; this does not establish
universal superiority or constitute an untouched final test.

For staged, camera 362 accounts for 4047/5728 local false observations (70.7%).
Only 325/4047 occurred in slots with no GT rows. Camera 364 accounts for
2988/4211 local misses (71.0%). These are observation counts across time,
not counts of distinct people or independent errors.

The frozen diagnostic first reproduces the selected camera's local metrics
with motmetrics 1.4.0, using the same clipped boxes, inclusive IoU gate and
chronological CLEAR assignment as the evaluator. It reads actual FP events;
it does not replace temporal CLEAR matching with independent greedy matching.

Mutually exclusive FP categories:

- `empty_gt_slot`: no raw GT row in the camera/frame.
- `no_admissible_gt`: GT rows exist, but no retained GT box has IoU at least
  the existing evaluation gate with this prediction.
- `admissible_but_unmatched`: a spatially admissible GT box exists, but this
  prediction is unmatched in the evaluator's one-to-one assignment.

The latter suggests competition for annotations, not proof of a duplicate.
None of these categories alone establishes background hallucination or
annotation incompleteness. Visual inspection is needed. Local FP events
are different from the IDFP count of the shared identity metric.

For each category, select at most two local tracks with the most FP
observations (ties by local ID), then the median FP frame of each track in
that category. The resulting examples are deliberately diagnostic, not
random samples for estimating prevalence. Full annotated views and both
raw/annotated context crops are saved. Green boxes are GT; the selected FP
is red. Other predictions are omitted to keep the target readable.

Only recorded predictions, GT and sequential CPU video decoding are used.
No models, tracker updates, threshold fitting, ROI exclusion or edits to
annotations occur. Checksums are verified before and after the diagnostic.

Run from the project root in WSL:

```bash
PYTHONPATH="$PWD/src" python scripts/check_scene_false_positives.py
PYTHONPATH="$PWD/src" python scripts/diagnose_scene_false_positives.py \
  --evaluation-report artifacts/scene_pair_evaluation/20261007T214235770374Z/report.json \
  --camera 362 --variant staged
```

Artifacts: report.json, false_positives.csv, up to six context JPEGs and
visual_review.zip. The completed diagnostic leaves visual review pending.

References: Bernardin and Stiefelhagen, *Evaluating Multiple Object Tracking
Performance: The CLEAR MOT Metrics* (2008); motmetrics event-based
evaluation implementation, https://github.com/cheind/py-motmetrics.
