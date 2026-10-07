# Full sequential MTMC reference results

## Scope and provenance

- Pipeline run: `20261007T131031285535Z`.
- Evaluation run: `20261007T132132828417Z`.
- Dataset: `nvidia/PhysicalAI-SmartSpaces`.
- Revision: `2cbe9563cbe9f47f846e5c871ee994572bbbc60e`.
- Scene: `MTMC_Tracking_2024/train/scene_001`; cameras 4, 5, 8.
- Runtime: frames 0..299, 300 rounds / 900 images at 30 FPS scene time.
- Quality evaluation: frames 2..299, including timing-warmup predictions.
- Performance measurement: frames 30..299, 270 rounds after 30 warmup rounds.
- Models: RF-DETR Small and OSNet, PyTorch CUDA FP32.
- Decoder, local tracking, history, geometry and identity management: CPU.
- Appearance: causal mean, 8 observations / 1 scene second, threshold 0.7.
- Geometry: maximum distance 2 native dataset units, before assignment;
  unavailable projection policy `appearance_only`.
- Grouping: complete-support greedy; identities: controlled confirmed merge.
- Global identity state starts at frame 0. Runtime does not use GT.
- No TensorRT, NVDEC, dropped frames, rendering or retroactive ID rewriting.

Exact configuration, input/code hashes and runtime versions are in
`reference_performance.json`; evaluation provenance is in
`reference_metrics.json`. These are copies of the completed run reports.

## Identity quality

Evaluation uses same-camera/same-frame IoU >= 0.5 gates and one shared
one-to-one GT/predicted-identity assignment across all cameras and frames.
Both reported variants use exactly the same local boxes and observations.

| Variant | IDF1 | IDP | IDR | IDTP | IDFP | IDFN |
|---|---:|---:|---:|---:|---:|---:|
| No-cross-camera local-ID control | 42.63% | 43.29% | 41.99% | 6852 | 8976 | 9465 |
| Full MTMC pipeline | 85.99% | 87.32% | 84.70% | 13821 | 2007 | 2496 |

GT observations: **16317**. Prediction observations: **15828**.
Global assignment improves IDF1 by **43.36 percentage points** over the
same-run control, with **6969** additional identity-correct observations.
Both variants agree with the independent `motmetrics` computation.

The earlier frozen geometry experiment achieved 86.02% IDF1 / 13826 IDTP.
The full pipeline has five fewer IDTP, approximately -0.0311 percentage points
on these equal denominators. This is not a controlled regression comparison:
identity initialization changed from frame 2 to frame 0 and model inference
was rerun. Equal counts alone do not establish equal boxes or embeddings.
No specific cause is inferred without a direct trace comparison.

All **19** accepted merges have diagnostic category
`all_visible_members_same_gt`. This describes mutually unique spatial GT
matches for visible members at the merge frame. It does not establish that
whole merged trajectories are correct or that identity errors are absent.

IDF1 measures identity-correct observations; it is not the percentage of
distinct people recognized correctly. IDFP/IDFN combine identity and detection/
localization effects under the stated matching protocol.

## Performance of the same pipeline run

| Component | Mean ms / three-camera round | p95 ms |
|---|---:|---:|
| Replay including RGB conversion | 7.725 | 10.549 |
| RF-DETR adapter | 19.650 | 25.423 |
| ByteTrack | 2.434 | 2.830 |
| Crop preparation | 0.604 | 0.681 |
| OSNet adapter | 43.052 | 50.448 |
| Appearance history | 1.347 | 1.639 |
| Complete geometry/association/identity block | 4.722 | 5.058 |
| Complete processing core | 79.533 | 93.513 |
| Recording | 3.129 | 3.377 |
| Core plus recording | 82.662 | 97.243 |

The association block includes projection (mean 1.323 ms), pair assignment
(1.700 ms), grouping (0.319 ms), identity management (1.133 ms), validation and
call overhead. These are nested timings, not additional costs to add again.
Likewise, decoder read/decode (6.417 ms) and RGB conversion (1.229 ms) are
subtimings within replay. The OSNet adapter includes preprocessing and
transfers, not only the GPU forward pass. p95 values are not additive.

- Measured wall time: **22.337926518 s** for 270 rounds.
- Throughput: **12.0871 rounds/s**, **36.2612 camera images/s**.
- Encoded crops: **654.4475/s**; mean **54.1444** per measured round.
- Total encoded observations over 300 rounds: **15914**.
- PyTorch peak allocated: **853.719 MiB**; peak reserved: **7186 MiB**.
- History peak: **62** retained tracks / **484** vectors.

One round contains three images. Sustained processing of all three 30-FPS
streams requires 30 rounds/s, about 2.48 times this measured throughput, with
additional latency margin needed for a live system. No speedup is promised.

This is a single instrumented sequential run with CPU decoding and buffered
artifact recording. Constructor time, input hashing, final stream close/flush,
NPY finalization and report writes are excluded from measured loop throughput.
Input hashing also warms the filesystem cache. PyTorch reserved memory includes
allocator caches and is not the total process/device VRAM footprint.

## Runtime accounting

- Allocated / ever-emitted global IDs: **50 / 50**.
- Absorbed IDs: **19**; expired IDs: **3**; retained at end: **28**.
- Lifecycle accounting: **50 = 19 + 3 + 28**.
- Unencoded observations: **0**.
- Unavailable projected observations: **0**.

Allocated and retained ID counts are not unique-person counts. Fragmentation
and incorrect associations can remain despite successful lifecycle accounting.

## Interpretation and next work

This establishes a working video-to-global-ID quality/performance reference.
It remains a repeatedly inspected, approximately ten-second training-scene
fragment. The current parameters are integration candidates, not independently
validated deployment settings, and the result is not an official AI City score.

Next, extend the evaluation protocol explicitly to a longer recording with
unchanged settings, inspect persistent identity failures, and define a separate
scene for validation. Do not tune thresholds on a final held-out test scene.

OSNet is the largest measured component and is the first model-backend
optimization candidate. Future TensorRT and decoding changes must be evaluated
for both quality and performance against recorded references under matched
conditions. Identical aggregate IDF1 alone does not prove identical decisions.
