# Scene-aware paired association runtime

This is an integration step before independent-scene quality evaluation. The
existing frozen experiment scripts and production FP32 runner are unchanged.

## Responsibility and ownership

`mtmc.pipeline.paired.PairedAssociation` consumes one `CandidateRound` per scene
frame. The input contains detector candidates and their already computed OSNet
embeddings. It owns independent trackers, appearance histories and global
identity managers for `staged` and `competitive_iou`.

The module receives only `RuntimeScene`. It opens no files and receives no GT,
ground-truth world coordinates, evaluation labels or future observations.
Decoding, RF-DETR, OSNet, artifact persistence and offline evaluation remain
outside this class. A later candidate producer can feed the same contract from
video inference instead of the frozen cache.

`competitive_iou` still uses the local appearance gate. Its name describes the
objective for competition with weak detections, not absence of appearance.
Both variants preserve their previously evaluated policies; this change does
not claim to implement a new published tracker.

## Scene configuration versus algorithm policy

| Input | Responsibility |
| --- | --- |
| `configs/scenes/scene_001_two_minutes.json` | Development data, camera set, timing and evaluation interval |
| `configs/scenes/scene_041_two_minutes.json` | Validation data with its own cameras and calibration |
| `configs/pipeline/paired_validation_policy.json` | Frozen local/global algorithm parameters copied from the prior paired experiment |

The policy file is created only after the real-data parity check passes. An
existing different policy is never overwritten. Its provenance and checksum
are recorded in the parity report. No parameters are fitted during this step.

Camera IDs, image sizes and homographies come from the scene. The runtime
supports arbitrary nonnegative camera IDs and per-camera image dimensions.
It deliberately retains **30 FPS** for this frozen experiment: the local
strong-appearance history is implemented as eight samples with age at most
30 frames. A different FPS is rejected rather than silently changing its
physical time window. The more general scene loader can represent other FPS;
that does not make this algorithm policy valid at every rate.

## Candidate contract

- Every round contains exactly one entry for each configured camera, including
  cameras with no detections. All-empty rounds still advance every state owner.
- Time is an exact `Fraction(frame_index, fps)`; frames start at zero and are
  consecutive. A source-run identifier prevents mixing candidate streams.
- Boxes are finite positive-area `float32` raw image `xyxy`; scores are finite
  `float32` values in `[0,1]`. Boxes are not clipped before tracking/projection.
- Every positive-intersection crop has a finite, L2-normalized 512-dimensional
  `float32` vector and a unique row in the candidate embedding stream. A wholly
  outside candidate has neither an embedding nor a row.
- Rows are contiguous from zero in sorted-camera/original-candidate order over
  the run. A detector candidate index is its position within that camera/frame;
  it is distinct from the global embedding row and the assigned local track ID.
- Camera entries can arrive in any order. Candidate order within a camera is
  preserved because it is part of the frozen tracker/tie-breaking behavior.

The accepted detection index joins a new local ID to its exact candidate box,
confidence and embedding. The join does not use a second approximate IoU match.
Global history inputs are sorted by camera/local ID, as in the old runner.

All cameras are validated and copied before a state update. Input rejection
does not advance any tracker, history or manager and permits a corrected call.
An internal exception after stateful work starts fails the entire paired run;
there is no rollback or retry of that partially processed round. Reproduction
requires a new instance replayed from frame zero. The object is single-owner,
sequential and not thread-safe.

Local strong histories continue learning only accepted high-score samples.
Global histories continue learning every currently encoded accepted observation,
including weak candidates. These are deliberately different existing policies.

## Verification

`check_paired_runtime_contract.py` uses small synthetic inputs to check arbitrary
cameras/sizes, reordered cameras, empty cameras/rounds, outside candidates,
feature mapping, invalid-input atomicity, copied policy/output ownership and
fail-stop handling after an injected internal error. Test thresholds are fixtures.

`check_paired_scene_runtime.py` consumes the existing scene_001 candidate cache
and the `competitive_global` reference. It regenerates both variants in one
runtime and compares, frame by frame:

1. Local IDs, exact boxes/scores, selected candidate indices, embedding rows and
   output order against the local and global reference traces.
2. Competitive refinement events and aggregated counters.
3. All serialized global assignments, decisions and retained identity state;
   only the explicitly checked top-level run namespace may differ.
4. Lifecycle counters and complete candidate/embedding consumption.

Reference quality metrics are copied into a field explicitly called
`reference_quality_not_recomputed`; this checker does not read GT or recompute
IDF1. Full output parity is a regression result, not proof that the underlying
tracking decisions are correct. No second large output trace is written merely
to duplicate an exact reference.

The checker uses the old scene_001-specific cache reader solely as the legacy
regression bridge. No legacy camera IDs or fixed image sizes enter the new
runtime module. A scene-aware candidate producer is the next integration step.

Development verification includes a 300-round technical fixture replay against
the old local/global runners. It uses recorded boxes, synthetic constant
embeddings and a minimal Supervision API shim around the pinned tracker math;
it is not validation on the user's CUDA environment. The WSL checks use the
installed pinned Supervision and the real cached embeddings for all 3600 rounds.

## Next step

After parity passes, collect candidates/embeddings for scene_041 using the
frozen RF-DETR/OSNet FP32 setup, then run both policies and evaluate the frozen
outputs with the already checked scene-aware GT adapter. Parameters remain
unchanged. In this architecture OSNet encodes detector candidates before local
association; the extra candidate workload must be included in later end-to-end
benchmarks. This check makes no GPU, TensorRT or throughput claim.
