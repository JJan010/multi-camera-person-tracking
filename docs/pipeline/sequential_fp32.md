# Full sequential MTMC reference pipeline

The entry point is `scripts/run_mtmc.py`. Runtime composition lives in
`src/mtmc/pipeline/core.py`; it calls the existing adapters without changing
their model, crop, assignment or merge algorithms. The implementation is a
single-owner, causal reference for subsequent optimization and evaluation.

## Data flow and devices

| Stage | Implementation | Device |
|---|---|---|
| Synchronized decode and RGB conversion | Existing PyAV replay, exact PTS | CPU |
| Person detection | Persistent RF-DETR Small, batch of three frames | PyTorch CUDA FP32, CPU pre/postprocessing |
| Local tracks | Independent ByteTrack per camera | CPU |
| Crops | Original RGB box crops, all cameras in one batch | CPU views |
| Appearance | Persistent OSNet, normalized 512-D vectors | PyTorch CUDA FP32, CPU preprocessing and output |
| History | Causal normalized mean, inherited count/age limits | CPU |
| Ground positions | Inverse homography of raw box bottom-center | CPU |
| Camera association | Geometry mask before partial assignment | CPU |
| Multi-camera groups | Existing complete-support greedy policy | CPU |
| Global identities | Existing controlled confirmation/merge manager | CPU |

Both models already execute on CUDA. This command does not use TensorRT,
NVDEC, frame dropping, adaptive crop selection, asynchronous queues or rendering.
Small stateful CPU modules are not required to move to CUDA simply because the
models will use TensorRT.

## Run

From the project root in WSL, with the project virtual environment active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_mtmc_pipeline.py \
  --reference-report artifacts/geometry_identity/20261007T124001420051Z/report.json

PYTHONPATH="$PWD/src" python scripts/run_mtmc.py \
  --reference-report artifacts/geometry_identity/20261007T124001420051Z/report.json \
  --rounds 300 --warmup 30 --torch-threads 1
```

No additional packages are required beyond the existing environment.
The geometry experiment report supplies the appearance threshold/variant,
distance/fallback policy, identity confirmation settings, and pinned calibration.
Its pinned history report supplies the original history count/age settings.
These values are recorded in the new run report. Ground truth is not opened by
the video runner and never enters the runtime pipeline.

The video runner additionally checks the dataset revision, scene and camera
manifest, all video hashes, calibration provenance against the scene manifest,
matrix direction, detector weight checksum and pinned OSNet assets.
This entry point intentionally supports the current scene_001/cameras 4,5,8,
30 FPS protocol. General scene configuration is a subsequent change.

## State, clocks and observation coverage

- Create fresh trackers, history and identity state for each run.
- Feed every consecutive frame from zero, including empty camera outputs.
- All age/confirmation calculations use exact scene-time `Fraction` values,
  never elapsed processing time. Slow inference does not accelerate expiry.
- The first 30 rounds are excluded only from performance summaries. Their
  state and predictions are retained, and all rounds are saved for evaluation.
- Global identity starts at frame **0**. Earlier frozen identity experiments
  started at frame **2**, although their appearance history included frames 0/1.
  The new run is therefore a new reference, not a claim of identical old IDs.
- The CPU parity check deliberately recreates the old start-frame boundary and
  checks every old state/merge/assignment record plus recomputed mean vectors.
- Fresh detector execution can also change boxes/local tracks. The old 86.02%
  IDF1 is not a measurement of this new run; evaluate the new saved trace.
- A fully outside track has no valid image crop. Preserve it as a singleton
  with `embedding_row: null`, allowing retained local/global continuity.
  Never fabricate a zero descriptor or silently remove the prediction.
- Missing geometry for an encoded observation uses the explicitly inherited
  policy. Malformed inputs fail rather than silently falling back.
- Earlier outputs are never rewritten after a merge. Global IDs must be read
  with `run_id`/`identity_scope`; allocated or emitted IDs are not people counts.
- An error makes a pipeline instance unusable. Some upstream adapters may have
  advanced; restart as a fresh run rather than retrying the same state.

## Artifacts

Each run creates `artifacts/mtmc_pipeline/<UTC-run-id>/`:

| File | Contents |
|---|---|
| `tracks.jsonl` | Existing local trace schema: detector/track boxes, scores, crop geometry and embedding rows |
| `global_tracks.jsonl` | Global assignments, bindings, merge/expiry decisions and events, missing-embedding flags |
| `decisions.jsonl` | Projected points, selected pair links/unmatched reasons, groups, causal history membership |
| `embeddings.npy` | Current normalized OSNet vectors, float32 `(N,512)` |
| `mean_embeddings.npy` | Causal mean vectors with exactly the same row mapping |
| `report.json` | Completed-run configuration, hashes, code provenance, timings, lifecycle and runtime versions |
| `run_status.json` | Completion status; an interrupted run is not a valid completed reference |

Use `(run_id, camera_id, local_id, frame_index)` as an observation scope.
An assignment's `embedding_row` addresses both NPY files. It is null only for
an unencoded observation. Global boxes are obtained by joining to the same
camera/local-ID/frame in `tracks.jsonl`; JSON line position is not an identity.

Feature arrays are appended to temporary `*.f32.partial` files during replay,
then converted to NPY in bounded chunks. Successful finalization removes these
temporary files. Incomplete runs can retain them for diagnosis. The loop does
not accumulate all historical vectors or RGB images in memory. Small timing
records and the set of emitted IDs still grow with the run length.

Every possible pair is considered by the existing association implementation,
but only candidate reason counts, accepted matches and unmatched reasons are
serialized here. All-pairs diagnostic traces remain in the earlier experiment.

## Timing contract

Timers measure host wall time in milliseconds. The detector timing boundary
synchronizes CUDA; OSNet returns CPU-ready vectors, completing its required GPU
work before the encoder timer ends. This is deliberately sequential.

`core_ms` covers video replay through final global identity assignments.
Its principal components are replay, detection, tracking, cropping, encoding,
history and the full association block. The latter is additionally split into
projection, pair assignment, grouping and identity management. Validation and
small call overhead are included in the complete block/core timers.

`read_decode_ms` and `rgb_conversion_ms` are **nested inside** replay time;
per-camera statistics are also saved for a later NVDEC comparison. Likewise,
do not add `association_block_ms` to its four subcomponents or add component
timers to `core_ms` a second time.

`record_ms` measures JSON construction/serialization, buffered writes, embedding
spooling and counters. `core_and_record_ms` is the sum of core and record time.
Throughput is measured loop wall time including intermediate progress prints.
One round contains three camera images: 30 rounds/s would service these three
30-FPS recordings without dropping frames in sustained throughput terms. This
does not by itself establish live latency or queue stability.

Excluded costs: checksum checks, model constructors, replay opening, closing/
final flushing of output streams, final NPY conversion, hashes and report writes.
Buffered recording does not imply data is durable on disk when its timer ends.
Video checksums read entire files first, so the filesystem cache is uncontrolled
and likely warm. NPY finalization time is recorded separately.

PyTorch peak-memory counters reset after warmup. Reserved memory can still
include caches retained from warmup; it is not the total process/device VRAM.
Allocator measurements are not comparable to full TensorRT/NVDEC device usage
without a broader GPU-memory measurement later.

A single instrumented 300-round run establishes integration behavior. It is
not a final speed benchmark or an independent quality validation. Subsequent
backend comparisons must keep the data, initial state, warmup, crop policy,
precision, recording policy and evaluation protocol explicit.

## Checks and next steps

The CPU check covers adapter composition using synthetic model outputs,
non-positional camera/key mapping, a confirmed merge, immutable older outputs,
fully outside crop handling, empty rounds, expiry, fail-stop behavior and NPY
row parity. With `--reference-report`, it also recomputes the original history
and requires exact equality of all 298 frozen identity records. It does not
replace the real CUDA run on the target machine.

After the video run, evaluate its frozen local boxes and global IDs together
using one shared identity mapping. Inspect the timing and quality changes,
then use a longer recording and a separate scene before more threshold tuning.
TensorRT will replace model backends while preserving these observation and
identity contracts; quality must be measured again after optimization.

References:

- Ristani et al., *Performance Measures and a Data Set for Multi-Target,
  Multi-Camera Tracking*, ECCV Workshops 2016:
  https://arxiv.org/abs/1609.01775. Identity evaluation should capture identity
  consistency across cameras, not only the number of per-frame links.
- NVIDIA, *TensorRT Best Practices*:
  https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html.
  Establish a reproducible measurement baseline, optimize, then measure again.
