# Frozen ByteTrack reproduction gate

## Verified candidate cache reported by the user

Cache run: `20261007T154414673309Z`.
Source pipeline run: `20261007T133915524799Z`.
Frames: 0..1799, cameras 4/5/8, 30 FPS.

All 113250 candidates have embeddings; no candidate is fully outside the image.
Per-camera counts: camera 4 = 32078, camera 5 = 40509, camera 8 = 40663.
For 205 identical-crop comparisons, minimum cosine was
0.9999999999999505 and maximum absolute error was 1.4901161193847656e-07.
These results establish sampled numerical consistency with the previous OSNet
output; they do not establish that any tracking policy has improved.

## Why replay the original tracker first?

Before changing association, the frozen candidates must reproduce the
reference outputs. This separates implementation/input errors from effects
of an appearance-assisted tracking policy.

The script verifies candidate equality against the original trace, including
box coordinates, confidence, camera/frame/detection keys, clipped crop bounds,
inside-image fraction and every embedding-row index. All cached descriptors
are checked for shape, float32 type, finite values and unit norm.

The installed `supervision==0.30.7` tracker sources and local adapter are
checked against `docs/tracking/supervision_source_audit.json`. The adapter hash
is also compared with the original pipeline's recorded code hash.

The reference replay creates independent `sv.ByteTrack` instances per camera,
uses the source run's settings, and calls `update_with_detections()` with the
same float32 inputs and zero person-class IDs as the original adapter. It
starts at frame zero and never resets state at the warmup boundary. It retains
the original public API's output remapping, which will be a separate concern
in a later controlled change.

For every camera and frame, exact equality is required for:

- Local IDs and their output order.
- Returned bounding boxes.
- Returned confidence scores.

No GT, video decoding, detection model, OSNet inference or GPU computation is
needed. Cached embeddings are validated but are not used for association.

## Run

From the project root in the existing virtual environment:

```bash
PYTHONPATH="$PWD/src" python scripts/check_frozen_tracking_contract.py

PYTHONPATH="$PWD/src" python scripts/check_frozen_bytetrack_replay.py \
  --cache-report artifacts/detection_embeddings/20261007T154414673309Z/report.json
```

The first command uses synthetic data to check changed scores, IDs, boxes,
feature rows, frame scopes and crop geometry are detected, and that duplicate
candidate keys are rejected. It also checks empty camera/output handling and
candidate-record order independence.

The second command is the real reproduction gate. Its outcomes are written to
`artifacts/frozen_bytetrack_replay/<timestamp>/report.json` and
`frame_parity.csv`. Success requires 5400 matching camera updates, zero
mismatches, all 113250 candidates accounted for, and 96282 reference track
observations accounted for. Counts include frames 0 and 1, unlike GT quality
evaluation which begins at frame 2.

If outputs differ, the completed diagnostic report has `passed: false`, stores
the first three differing camera/frame outputs, and exits with status 1. Stop
and investigate the first mismatch before introducing appearance constraints.
Input corruption or incompatible code fails earlier with an explicit error.

Exact reproduction includes known tracking errors. It establishes a stable
control, not correctness of the tracker. This gate does not recompute quality
metrics or measure tracker throughput.

Next: isolate output-mapping changes, then add appearance to a separate local
association variant and compare against the same frozen candidates. Evaluate
whole-sequence local and downstream global quality; do not select a deployment
threshold from the inspected L11 event alone.
