# Frozen candidate collection for a configured scene

Prerequisite: scene-input parity and complete paired-runtime parity on scene_001.
The real paired-runtime check passed all 3600 frames for both `staged` and
`competitive_iou`, including local outputs, refinement events, global decisions,
retained identity state and lifecycle counters. Evidence:
`artifacts/paired_scene_runtime_checks/20261007T211730398428Z/report.json`.
That establishes integration parity, not validation-scene quality.

## Purpose

`collect_scene_candidates.py` produces a single immutable input cache for later
paired tracking experiments. It uses the configured videos and the frozen
RF-DETR Small and OSNet x1.0 FP32 adapters. No tracker or global identity manager
runs at this step. Ground truth is neither loaded nor passed to inference.

The detector's person candidates are encoded before local association so both
experimental trackers can inspect candidate appearance. This is the same
information order used in the frozen development experiments. Encoding only
already accepted tracks would remove appearance information needed to compare
alternative detections.

Zhang et al., **ByteTrack: Multi-Object Tracking by Associating Every Detection
Box**, ECCV 2022, motivate retaining lower-score detections because occlusion can
reduce detection confidence. Tracklet association can recover useful weak
detections. Our extra OSNet candidate gate and competitive refinement are
project experiments, not claims of implementing an unmodified ByteTrack method.
Primary source: https://arxiv.org/abs/2110.06864 .

## Frozen setup and execution

The command takes `--scene-config` and the passed `--parity-report`. From the
verified report chain it obtains the model weights/configuration, original
detector threshold (0.1), decoder thread count, torch thread count and the prior
candidate-cache OSNet chunk size. It verifies model assets, reused adapter code,
reference package versions and the frozen paired policy. No command-line
threshold tuning is offered.

For scene_041 the selected cameras are 361, 362 and 364, with frames 0..3599.
The first 30 frames are recorded normally and excluded only from warm timing
summaries. Annotation gaps cannot affect collection because the producer does
not read GT.

| Operation | Backend |
| --- | --- |
| Sequential decode and RGB conversion | Existing PyAV CPU reader |
| RF-DETR Small inference | Existing CUDA FP32 adapter; one batch of camera images per round |
| Cropping and OSNet preprocessing | CPU, same crop/resize/normalization conventions |
| OSNet inference | Existing CUDA FP32 encoder, sorted-camera candidate order, bounded chunks |
| Normalized embedding storage | CPU float32 NPY, streamed through a bounded temporary spool |

This step is not TensorRT or NVDEC integration. Both models remain persistent
for the full run. The script creates a fresh timestamped output directory and
does not overwrite earlier experiments. Failed runs remain explicitly incomplete.

## Data contract

`mtmc.data.candidates` validates the camera set, actual image sizes, exact
rational timestamps and detection arrays. Raw float32 boxes and scores are
preserved, including duplicate boxes, weak detections and wholly outside boxes.
Within-camera candidate order is unchanged. Camera entries are sorted by ID.

Positive-intersection crops use floor/ceil clipping of original RGB pixels, as
before. Wholly outside detections remain in the trace with `embedding_row=null`.
There is no crop-quality or GT-dependent visibility filter.

The candidate key is `(cache_run_id, frame_index, camera_id, detection_index)`.
`detection_index` starts at zero for each camera/frame and is not a person ID.
The existing encoder's `ObservationKey.local_id` field temporarily transports
that index; persisted records use the explicit `detection_index` name.
Embedding rows are contiguous across all frames in camera/candidate order.
Every frame explicitly lists all cameras, even if some or all have zero boxes.

Outputs under `artifacts/scene_candidates/<run>/`:

- `detections.jsonl`: frozen candidate boxes, scores, crop bounds and embedding rows.
- `embeddings.npy`: L2-normalized float32 rows with 512 components.
- `timings.jsonl`: all frame timings, including warmup and candidate counts.
- `report.json`: pinned inputs, model provenance, settings, counts, timing summaries,
  versions, artifact checksums and limits.
- `run_status.json`: completion or failure status.

After collection, the actual files are read back. The checker verifies frame
coverage, scopes, camera coverage, candidate order, exact float32 representation,
crop geometry, contiguous row mapping, all vector norms and final counts.

## Interpreting the measurements

`encode_ms` includes CPU preprocessing, transfers, CUDA forward, normalization
and returning CPU vectors. Detector timing ends at a CUDA synchronization point.
Warm collection-loop throughput includes recording and progress output, while
excluding final stream closure, NPY finalization, post-write audit and checksums.
The report documents these boundaries explicitly.

The workload includes every returned detector candidate, not just accepted
tracks, and omits tracking/global association. Its FPS must therefore not be
presented as an improvement over the earlier full MTMC pipeline. The result is
a useful component measurement, not a final performance benchmark.

## Verification and next step

`check_scene_candidate_collection.py` uses fake detector/encoder outputs on CPU
to exercise exact crop pixels, weak/duplicate/outside candidates, empty rounds,
chunk boundaries, persistent mapping, recording/read-back and interrupted input.
It does not test CUDA inference or estimate model accuracy. Actual CUDA execution
must pass on the user's machine with the pinned assets and environment.

Once the new cache is complete, replay `staged` and `competitive_iou` from the
same frozen candidates and evaluate their local/global outputs on scene_041.
Use the already checked scene-aware GT adapter, including empty camera/frame
slots, and keep the chosen algorithm settings unchanged. No quality result for
scene_041 is claimed by successful candidate collection alone.
