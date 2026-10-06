# OSNet FP32 benchmark results: 24 vs 1 CPU intra-op thread

## Scope

Measurements from 2026-10-06 on WSL Ubuntu 22.04, Intel Core Ultra 7 270K Plus
and NVIDIA RTX 5080. PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NumPy
2.2.6, Pillow 12.3.0. See the archived reports for full provenance.

Both configurations use the same 57 RGB crops from scene_001 frame 150,
the same OSNet weights, CUDA FP32, disabled TF32 and disabled cuDNN benchmarking.
Each phase has 20 warmup iterations and 100 measured iterations, repeated
three times. Input PNG loading and hashing occur before timing.

The input-manifest, model-configuration, benchmark-script and adapter hashes
match between reports. Software/hardware metadata also match. Intra-op CPU
threads changed from 24 to 1; inter-op threads remained 24. Both first-encode
checks exactly match the saved reference: maximum absolute error 0.0.

## Pooled measurements

Statistics below are recomputed from 300 raw samples per phase and configuration.
P95 is computed from the combined samples, not by averaging three percentiles.
All times are milliseconds per batch of 57 crops.

| Phase | CPU threads | Mean | Median | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| preprocess_cpu | 24 | 19.209 | 17.199 | 23.696 | 107.431 |
| preprocess_cpu | 1 | 18.263 | 18.319 | 18.857 | 20.023 |
| forward_cuda | 24 | 20.829 | 20.666 | 21.908 | 24.360 |
| forward_cuda | 1 | 20.721 | 20.696 | 20.973 | 21.836 |
| encode_total | 24 | 45.709 | 43.345 | 56.408 | 130.531 |
| encode_total | 1 | 42.289 | 42.150 | 43.819 | 45.053 |

| CPU threads | Complete batches/s | Complete crops/s | Calls >60 ms / 300 |
| ---: | ---: | ---: | ---: |
| 24 | 21.876 | 1246.904 | 12 |
| 1 | 23.645 | 1347.757 | 0 |

Throughput uses the summed elapsed wall time of the three full-encode loops.
Peak allocated memory was 703.099 MiB and peak reserved memory was 766 MiB
for full encode in every repeat of both configurations. These are PyTorch
allocator statistics, not total GPU memory use.

## Interpretation

- Full-encode mean latency was 7.48% lower in the one-thread run;
  median latency was 2.76% lower and pooled p95 was 22.32% lower.
- The main observed benefit is reduced latency variability. The 24-thread
  configuration had 12/300 full-encode calls above 60 ms; the one-thread
  configuration had none and its maximum was 45.053 ms.
- CPU preprocessing median increased from 17.199 to 18.319 ms, even though
  its mean and p95 improved. The result does not show that one thread makes
  every typical preprocessing call faster.
- Mean raw GPU forward times were close: 20.829 vs 20.721 ms. No substantial
  model-forward speedup is demonstrated by this change.
- The fastest 24-thread repeat had a full-encode mean of 42.252 ms, close
  to the pooled one-thread mean of 42.289 ms. Much of the pooled difference
  comes from slow calls in later repeats of the original measurement.
- Stage measurements were collected separately. Do not subtract their
  means or medians to infer exact transfer or postprocessing costs.

Thread-management/scheduling overhead is a plausible hypothesis for the
observed variation. The runs were performed sequentially (24 then 1), without
alternating configurations or profiling OS scheduling, background load or GPU
clocks. These measurements support a practical working setting, but do not
isolate the causal mechanism or establish that one thread is universally optimal.

Constructor and first-encode times are one observation per configuration:
756.801 / 503.880 ms for 24 threads, and 1066.830 / 747.099 ms for one thread.
They are excluded from steady-state statistics and do not establish a reliable
startup-performance comparison.

## Working decision

Use an explicit `--torch-threads 1` for subsequent isolated OSNet benchmark
comparisons on this workload. Preserve both reports. This does not change the
adapter implementation or the benchmark default, which still preserves the
process default when no thread argument is supplied.

PyTorch intra-op thread count applies to CPU operations within the process.
Reassess it when integrating the detector and other pipeline stages; the best
setting for this component may differ from the best setting for the full process.
The benchmark reports crop-encoding service rates, not whole-system video FPS.
Different batch sizes, crop content and precision require separate measurements.

## Reproduction

Run from the project root in WSL with the project virtual environment active:

    PYTHONPATH="$PWD/src" python scripts/benchmark_osnet.py --reference artifacts/reid_embeddings/20261006T194219632494Z/manifest.json --warmup 20 --iterations 100 --repeats 3 --torch-threads 1

For the comparison configuration, replace the final value with `24`.
The measurement boundaries are documented in `benchmark_protocol.md`.

## Archived evidence

The following JSON files are exact copies of the original reports, including
raw timing samples and original execution paths. They record Git HEAD
`65defd81fb1c10ad5f776f22e186b0ef800ad1cb` plus the then-uncommitted adapter and
benchmark files. Source hashes identify the code actually measured.

- `benchmarks/osnet_fp32_threads24.json`
  - Original: `artifacts/benchmarks/osnet_fp32_20261006T201023432200Z.json`
  - SHA256: `6024bc0cb3a0dd60cf9eea72e370cd0bcb2563533a300294757d4a2031454542`
- `benchmarks/osnet_fp32_threads1.json`
  - Original: `artifacts/benchmarks/osnet_fp32_20261006T201513323764Z.json`
  - SHA256: `6117b669309f751e7d8a5ea92cbb215cf22239cc4d609f7b61f82af6788b2fdc`

## References

- PyTorch CPU intra-op threads:
  https://docs.pytorch.org/docs/2.11/generated/torch.set_num_threads.html
- PyTorch asynchronous CUDA execution and timing:
  https://docs.pytorch.org/docs/2.11/notes/cuda.html#asynchronous-execution
