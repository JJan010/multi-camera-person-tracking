# Sequential local tracking and Re-ID pipeline

`scripts/run_local_reid.py` connects the existing replay, RF-DETR person
detector, independent per-camera ByteTrack instances, in-memory crop builder
and persistent OSNet encoder. Each round processes one synchronized image from
each of cameras 4, 5 and 8. All available person crops share one OSNet call.

This is the first continuous integration run. It creates appearance observations
for local tracks; it does not assign global identities or aggregate appearance
history. No ground-truth information is passed to inference.

## Run

From the project root, with its `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/run_local_reid.py \
  --rounds 300 --warmup 30 --torch-threads 1
```

The default run processes frames 0–299, ten seconds of source video. Detector
and tracker updates occur every frame, with no dropped or skipped rounds.
RF-DETR uses confidence threshold 0.1, retaining the low-confidence detections
used by the existing ByteTrack baseline. Tracker settings are imported unchanged.
Both networks use FP32; TF32 and cuDNN benchmark are disabled.

Re-ID processes every visible local track in every round, except boxes fully
outside the image. Partially outside and low-confidence output tracks remain.
The batch size varies with visible tracks. Empty crop batches use the encoder's
existing `(0, 512)` behavior without a model forward. This deliberately simple
policy is the reference for later selective appearance updates.

CPU intra-op threads are explicitly set to 1 for the whole process, including
the detector. This setting was useful in the isolated OSNet benchmark but has
not been demonstrated to be optimal for the integrated pipeline.

## Outputs

Every invocation writes a new `artifacts/local_reid/<UTC>/` directory:

- `tracks.jsonl`: one line per frame, with the existing detector and local-track
  fields accepted by `evaluate_local_tracking.py`. Additional fields contain a
  run ID, exact timestamp, and `reid_observations` with crop metadata and
  `embedding_row`. Fully outside observations have a null row and explicit status.
- `embeddings.npy`: CPU float32 `(N, 512)` matrix of normalized descriptors.
  Rows include warmup frames and are linked from the trace. Frame crops never
  round-trip through PNG files. Only the final features and metadata are saved.
- `report.json`: settings, provenance, source hashes, environment, per-round
  timings/counts, memory peaks and summary. `completed=true` is written only
  after the loop and final output validation succeed.

Local IDs are unique within a camera and run, not globally across cameras or
restarts. Distinct local IDs are not counts of unique people. An observation is
identified by `(run_id, camera, local_id, frame_index)`.

The script retains CPU feature arrays until the final NPY write. At the default
300 rounds this supports a compact diagnostic recording. It does not retain
RGB crops across rounds. A long-running service will need a separate bounded
history/output policy.

## Timing protocol

Models and trackers are constructed once. The first 30 rounds are processed and
saved normally, but excluded from the timing summary. Tracker state is not
reset after warmup. Measured frames are 30–299: 270 rounds, one repeat.

| Field | Scope |
| --- | --- |
| `replay_ms` | Three-camera read/decode and RGB conversion |
| `detect_ms` | Full detector adapter plus CUDA synchronization |
| `track_ms` | Local tracker updates for all cameras |
| `crop_ms` | Join, validation and borrowed RGB crop construction |
| `encode_ms` | Complete OSNet adapter: preprocessing, transfers, inference, normalization and CPU results |
| `core_ms` | Sum of the five immediately preceding stages |
| `record_ms` | Metadata/JSONL work, CPU feature retention and counters |

All phase measurements use `perf_counter_ns` on the host. The detector boundary
explicitly synchronizes CUDA. OSNet returns a blocking CPU result, so its call
already waits for the relevant GPU work. These are sequential diagnostic
measurements; CUDA overlap is not being attempted.

Reported throughput uses the measured loop's elapsed wall time, including
recording and intermediate progress prints. It excludes initial input checksum
verification, model construction, video opening, final trace close/flush and
final NPY/report writes. JSONL writes are buffered; this is not a storage
durability benchmark. Constructor timings are separate; RF-DETR's first CUDA
placement/first inference occurs during the initial round.

Peak memory covers the measured interval and is reported by the PyTorch
allocator; reserved memory can include caches populated during warmup. It is
not total GPU memory usage. Video checksum verification reads all files before
replay, influencing the filesystem cache. Raw timing samples include warmup
frames with `measured=false` so startup behavior remains visible.

Treat the output as an initial integrated performance diagnostic, not a
definitive speedup result. The older detector benchmark used a different
confidence threshold and frame range; the isolated OSNet benchmark reused a
fixed 57-crop batch. Direct ratios to those runs would mix different workloads.

After a successful run, evaluate this trace with the existing local evaluator
and inspect the appearance observations before introducing global association.

## References

- [PyTorch 2.11 CUDA semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html):
  asynchronous GPU execution and timing synchronization.
- [Zhang et al., ByteTrack, ECCV 2022](https://arxiv.org/abs/2110.06864):
  association using high- and low-confidence detections.
- [Zhou et al., OSNet, ICCV 2019](https://arxiv.org/abs/1905.00953):
  appearance feature extraction for person re-identification.

The integration and measurement protocol above are project implementation
choices, not a reproduction of the papers' training/evaluation protocols.
