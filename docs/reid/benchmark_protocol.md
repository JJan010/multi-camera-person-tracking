# OSNet FP32 benchmark protocol

The first benchmark reuses the 57 verified RGB crops from scene_001 frame 150.
They are loaded into RAM and hashed before timing. Model assets and the reference
embedding matrix are also verified. One encoder instance remains loaded throughout.
This measures isolated components at a fixed batch size and fixed crop content.

Run from the project root with its virtual environment active:

    PYTHONPATH="$PWD/src" python scripts/benchmark_osnet.py --reference artifacts/reid_embeddings/20261006T194219632494Z/manifest.json --warmup 20 --iterations 100 --repeats 3

For every phase in every repeat, discard 20 warmup iterations and collect 100
measured iterations. The default preserves PyTorch CPU thread counts, which are
printed and recorded. `--torch-threads N` enables a separate, explicitly recorded
intra-op thread experiment. Do not run competing GPU jobs during measurement.

## Measurement boundaries

| Phase | Clock | Included |
| --- | --- | --- |
| `preprocess_cpu` | `perf_counter_ns` | Actual `prepare_inputs`: input checks, NumPy-to-PIL, bilinear resize, tensor conversion, channel normalization, stacking |
| `forward_cuda` | CUDA events | Raw network forward on an already resident FP32 tensor; no preprocessing, H2D, L2 normalization or D2H |
| `encode_total` | `perf_counter_ns` | Actual `encode`: input checks, preprocessing, H2D, model, L2 normalization, blocking D2H, output checks and returned metadata |

The CUDA-event pair is initialized outside measurement. Each end event is
synchronized before reading its duration, so this is a serialized per-call latency
test. The event interval can include device-stream gaps between kernel launches;
it is not a sum of kernel-only durations. The host loop duration is also saved.
Raw GPU input remains resident during the forward phase and is released before
the full adapter phase. FP32, disabled TF32 and disabled cuDNN benchmarking match
the adapter baseline. Both GPU paths use eval mode and inference mode.

The returned CPU embeddings make encode completion observable without adding
per-stage synchronization inside the adapter. Before a full encode measurement
loop, synchronize the device and reset PyTorch peak-memory statistics. Peaks
include model memory; reserved memory can retain allocator cache from earlier
phases. These are PyTorch memory statistics, not total GPU process memory.

## Recorded results and interpretation

- Mean, median and p95 milliseconds **per batch of 57**, with every raw timing sample.
- Completed batches/s and crops/s for the full adapter, based on host loop elapsed
  time. This includes small loop/timer/result-release overhead.
- PyTorch peak allocated and reserved memory for GPU phases.
- One constructor duration and one first-encode duration, separate from steady state.
  Constructor timing begins after imports/crop loading and includes asset hashing,
  weight loading and model transfer; it is not process-start-to-first-result latency.
- Input/code/config/weight provenance, numerical parity, software versions, CPU/GPU,
  thread settings, Git revision and working-tree status.

The phases are separate measurements. Their medians should not be added or
subtracted to claim an exact transfer/postprocessing cost. This microbenchmark
excludes decoding, detection, local tracking, extracting crops from video, global
association and display. Batches/s is a Re-ID service rate under this load, not
whole-system video FPS. Later benchmarks must cover varying crop counts and the
integrated pipeline. The fixed workload provides a repeatable FP32 reference for
future backend comparisons, with explicit preprocessing and output boundaries.

Reports are written to `artifacts/benchmarks/osnet_fp32_<UTC>.json`.

## References

- PyTorch 2.11 CUDA asynchronous execution and timing:
  https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution
- PyTorch 2.11 CUDA events:
  https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html
