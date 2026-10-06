# Replay CPU baseline

Initial measurement for scene_001, cameras 4, 5 and 8.

## Configuration

- PyAV 16.0.1, CPU decoding.
- One decoder thread per camera, SLICE threading.
- Cameras processed sequentially within each round.
- Output: three 1920x1080 RGB uint8 arrays in CPU memory.
- Each repeat starts at the beginning of the videos.
- 60 warmup rounds, then 900 measured rounds.
- Three repeats, without playback pacing or visualization.
- Filesystem cache is uncontrolled; this is not a cold-cache benchmark.
- No detection, tracking, Re-ID or GPU transfer.

## Observed results

- Throughput: 92.09-96.32 rounds/s.
- Aggregate throughput: 276.28-288.96 images/s.
- Round p95: 14.05-14.73 ms.
- Mean read/decode time summed across cameras: 6.155 ms.
- Mean RGB conversion time summed across cameras: 4.210 ms.

Read/decode includes the work needed to obtain the next decoded frame,
including input and demuxing. It is not an isolated decoder measurement.

Per-frame timings measure reader call latency, not live capture-to-output
latency. Overall throughput also includes validation and timing bookkeeping.

## Reproduction

From the repository root, with the project virtual environment active:

    PYTHONPATH="$PWD/src" python scripts/benchmark_replay.py

Full reports, including timing samples, are written to artifacts/benchmarks/.
A compact copy of the initial report is retained beside this document.

The initial measurement was taken with uncommitted reader and benchmark
files. Its recorded Git HEAD identifies the preceding commit; the original
working-tree status is retained in the report. The implementation is
committed together with this summary.

Future GPU comparisons must explicitly state their output memory location,
pixel format, conversion costs and transfer costs.
