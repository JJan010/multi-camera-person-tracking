# Competitive tracking: local result and global evaluation

## Observed local result

Frozen local experiment: `20261007T184813424022Z`.
Scene 001, cameras 4/5/8, runtime frames 0..3599; evaluate frames 2..3599.
The staged and disabled implementations reproduced the frozen appearance
baseline exactly. The candidate/embedding cache remained fixed.

| Variant | Pooled local IDF1 | Precision | Recall | IDSW | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| staged | 44.86% | 98.28% | 96.28% | 324 | 3105 | 6851 |
| competitive_iou | 43.30% | 98.14% | 96.39% | 441 | 3359 | 6650 |
| competitive_appearance | 43.05% | 97.99% | 96.36% | 446 | 3642 | 6709 |

Values above are rounded console results, not substitutes for the complete JSON.
There were 641 IoU replacements and 813 appearance replacements. Relative to
staged, the appearance variant had 122 additional ID switches and 537 additional
false observations, while missing 142 fewer observations. Local IDF1 fell about
1.81 percentage points. The IoU variant also regressed.

The local results do not support promoting this modification. More recovered
observations do not automatically imply better identity continuity. Releasing
high-score incumbents may create extra tracks, but the aggregate FP/IDSW counts
alone do not prove how much of the regression was caused by this mechanism.

Local IDs are allocated independently in each variant. L44 representing GT 17
or GT 14 in another variant does not establish an incorrect switch at frame 1438;
the numeric IDs are not aligned across variants. In the case excerpt, follow
diagnostic GT 1 over scene time (e.g. L58 then L62/L64), while recognizing that
mutually unique IoU labels are partial evidence and can change with box geometry.

## Global experiment

`evaluate_competitive_global.py` runs three fresh, isolated downstream managers
continuously from frame zero. It reuses the existing appearance-history and
geometry/grouping/global-identity implementation and the exact original settings.
Every accepted local detection is joined to its cached embedding by explicit
candidate index and embedding row, never by GT or a nearest-box heuristic.

`staged` is replayed and compared at **every frame** to `direct_appearance` from
the existing global experiment `20261007T174247370251Z`. Only the explicit run-scope
string may change. Camera records, assignments, retained state, merge decisions,
expiry and other identity fields must all be identical. Final lifecycle, merge
diagnostics and global metrics must also reproduce the reference (IDF1 53.03%).

After all outputs are frozen, GT is loaded for evaluation. For each variant,
solve one shared identity assignment across all cameras and frames using the
existing same-camera/same-frame IoU >= 0.5 convention. Validate every result
against motmetrics. Local/global GT and prediction denominators must agree.
Prediction counts may differ between variants; do not force equal denominators.

The local tracker's strong-only gallery and the downstream appearance history
are different components. The downstream history retains its original policy:
all encoded tracked observations, including accepted weak ones. This is held
constant across variants, not silently changed for the new experiment.

## WSL commands

Run from `/home/jakjan/projects/multi-camera-person-tracking` with `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_competitive_global_inputs.py
```

Then:

```bash
PYTHONPATH="$PWD/src" python scripts/evaluate_competitive_global.py \
  --local-report artifacts/competitive_bytetrack/20261007T184813424022Z/report.json \
  --reference-global-report artifacts/appearance_global/20261007T174247370251Z/report.json
```

Outputs: `artifacts/competitive_global/<UTC-run>/report.json`,
`global_tracks.jsonl.gz`, `identity_matching.json` and completion status.
No model inference, video decoding, tracker modification or deployment occurs.

## Interpretation

Close this experiment using full local and global metrics rather than success
on the originally selected few frames. If global identity also regresses, retain
the negative result and the unchanged reference policy. Do not tune thresholds
against that same case to rescue the hypothesis. An improvement on this reused
development fragment would still require independent validation.

The CLI was tested with a complete 300-frame technical fixture and synthetic
constant embeddings. This verifies source joins, exact staged state reproduction,
metrics, lifecycle and artifact hashes, not actual OSNet tracking quality.
