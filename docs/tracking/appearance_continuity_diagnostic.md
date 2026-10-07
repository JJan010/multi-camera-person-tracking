# Frozen appearance continuity diagnostic

Target experiment: `20261007T174247370251Z`, variant `direct_appearance`.
The two-minute evaluation reproduced its first minute, then found a global
IDF1 gain of approximately 6.78 percentage points on the second minute versus
direct_iou. Full-sequence global IDF1 remains 53.03%.

This diagnostic uses the already frozen paired global trace and pinned GT.
It does not decode video, run models, replay trackers, change identity state,
tune thresholds or repair past outputs. It requires only CPU computation.

## Questions answered

1. How many identity errors remain beyond the maximum framewise spatial matches
   possible with the fixed boxes? The existing spatial-ceiling decomposition
   is reused and checked against the reproduced identity counts.
2. Which GT identities have evidence under multiple global IDs, and how often
   does this happen simultaneously across cameras?
3. Which global IDs and camera-local tracks have evidence of multiple GT people?
4. Which GT identities contribute most to the difference between independent
   interval assignments and one shared assignment over the full sequence?

For the current variant, the expected separate-minus-shared IDTP is
`71413 + 51709 - 96696 = 26426`. This is not an event count and does not localize
errors to the frame-1800 boundary. Both fragmentation and mixing can contribute.
Per-GT contributions depend on the selected optimal assignments and may be
negative because of competition in the one-to-one assignment. Tied optima can
change attribution without changing the total.

## Evidence and outputs

Global and pooled local identity metrics are recomputed for the full sequence
and both intervals and must reproduce the completed evaluation. Diagnostic
labels use mutually unique same-camera/same-frame IoU matches. Ambiguous and
unmatched observations remain outside this evidence; histograms do not label
every observation. Local evidence-label transitions may span gaps and are not
the CLEAR IDSW metric.

Outputs in `artifacts/appearance_continuity/<run_id>/`:

- `report.json`: metrics, spatial decompositions, per-interval fragmentation
  and mixing tables, and per-GT assignment contributions.
- `evidence_segments.json`: contiguous unique-evidence segments by local track,
  GT identity and emitted global ID.
- `local_label_transitions.json`: changes in the GT evidence label of a local
  track, including timestamps of the surrounding evidence.

These outputs support selection of a small number of cases for inspection.
They do not by themselves establish whether a bad detection, local matching,
expiry or merge policy caused an error.

## Run from the WSL project directory

```bash
PYTHONPATH="$PWD/src" python scripts/check_appearance_continuity.py

PYTHONPATH="$PWD/src" python scripts/diagnose_appearance_continuity.py \
  --evaluation-report artifacts/appearance_global/20261007T174247370251Z/report.json \
  --variant direct_appearance
```

The check covers three known-answer cases: stable identity, one person split
into two identities across the interval boundary, and one local/global ID taken
over by a different GT person. It verifies spatial ceilings, assignment gaps,
fragmentation/mixing evidence and local label transitions.

## Subsequent project milestones

1. Diagnose dominant continuity failures and test a targeted change only when
   supported by the frozen evidence. Preserve the baseline and compare paired
   metrics. Further tuning makes the second minute development data.
2. Integrate the selected tracking policy into the full pipeline. The appearance
   variant needs candidate crops/embeddings before local association. Preserve
   explicit candidate-to-track mapping and avoid redundant encoding where the
   same crop representation can be reused. Measure real end-to-end cost.
3. Validate on an untouched scene/interval with frozen configuration and explicit
   dataset, synchronization and calibration checks. State domain limitations.
4. Export and validate OSNet through ONNX/TensorRT, then RF-DETR. Check numerical
   behavior, tracking quality and throughput after each backend/precision change.
5. Evaluate NVDEC and GPU preprocessing, transfer costs, batching and bounded
   scheduling. Measure the complete pipeline rather than isolated inference only.
6. Finish a reproducible CLI/demo, camera/global-ID visualization, architecture
   and experiment documentation, model/data provenance and installation steps.

GPU optimization does not require moving all association/state logic onto GPU.
Use profiling to decide which CPU operations matter. A target of three streams
at 30 FPS means 30 synchronized rounds/s (90 images/s), and remains a measured
engineering target rather than a promised performance result.
