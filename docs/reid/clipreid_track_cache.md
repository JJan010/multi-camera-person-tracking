# Full-rate CLIP-ReID cache on frozen local observations

Purpose: prepare a controlled global-association encoder experiment. This stage
runs no detector, tracker, continuity guard, appearance history, global manager
or ground-truth evaluation. It preserves the frozen local boxes and original
camera/local/frame observation keys from the enabled continuity trace.

The local population originated from OSNet-aware tracking. This cache is
conditional on those tracks; it is not a cache of every detector candidate and
cannot establish the performance of replacing Re-ID inside the local tracker.

## Inputs and scope

Supply the matching continuity source report, the successful paired temporal
retrieval report for that same source, and the CLIP-ReID CUDA model smoke report.
The script verifies report, scene, video, source trace, source candidate vector,
model asset and relevant code checksums. It does not read GT contents or use
retrieval scores to admit crops. Both scenes have already informed development.

Process frames 0..3599 at 30 FPS, including the two initial frames outside the
usual GT evaluation range. The frame-zero warmup observations remain available.
Keep all positive-area crops, irrespective of confidence or border clearance.
Fully outside observations remain in the mapping with null vector rows.
Original local IDs are not segment IDs and must not be treated as global IDs.

## Inference and memory

CPU PyAV decoding uses the existing reader and its timestamp checks. One
persistent CLIP-ReID visual encoder runs CUDA FP32 with batches of at most 16,
TF32 disabled, and the pinned 256x128 RGB preprocessing. Each feature is the
jointly L2-normalized 768+512 vector, dimension 1280. No OSNet inference runs.

A metadata preflight counts rows before allocating a disk-backed NumPy array.
Only current-round crops and features are held for inference; metadata and
features are written incrementally. The full feature array is not accumulated
as Python lists. OS page cache may still use RAM for memory-mapped pages.
Estimated vector payload at the known observation counts is about 882 MiB for
scene_001 and 296 MiB for scene_041, plus mapping files. Files are local ignored
artifacts; do not commit the feature arrays to Git.

## Checks

At every frame, verify scope, time, cameras, original local IDs, accepted
candidate indices, source vector rows and crop geometry. Enforce uniqueness of
source candidate rows across the trace. The encoder must preserve keys and
timestamps and return finite normalized float32 vectors.

At the existing 120-frame grid (2,32,...3572), compare the exact raw RGB crop
SHA-256 and full observation provenance with the earlier paired comparison.
Recomputed CLIP features must satisfy max absolute difference <= 1e-5 and cosine
>= 1-1e-5. All reference rows and sampled empty frames must be accounted for.
This is sampled inference parity, not independent recomputation of every crop.

After writing, read the complete persisted mapping against the original trace
and inspect every stored vector for shape, finiteness and normalization. Recheck
inputs and hash outputs before marking the run completed. An interrupted run
has no successful completed report; rerunning creates a fresh directory and does
not resume or overwrite partial output.

## Output

`artifacts/clipreid_track_cache/<UTC run>/` contains:

- `clipreid_embeddings.npy`: N x 1280 float32 vectors; load using mmap_mode='r'.
- `observations.jsonl.gz`: one record per frame, including empty frames. Each
  observation stores original camera/local/frame key, timestamp, raw box, score,
  integer crop bounds, inside-image fraction, RGB SHA-256, detection index,
  source OSNet candidate-cache row and this cache's vector row (or null).
- `report.json`: completed provenance, counts, sampled parity and artifact hashes.
- `run_status.json`: completion or failure status.

Do not compare raw OSNet and CLIP vectors against each other: they have different
spaces and dimensions. The source OSNet row is a provenance link, not an index
into this new matrix.

## Next step

Introduce an explicitly dimension-aware feature/history/association interface,
prove that the 512-dimensional OSNet path retains its frozen behavior, and then
measure CLIP's effect on global identity assignment. Similarity thresholds from
OSNet are not automatically calibrated for CLIP. No runtime model promotion,
threshold selection, global IDF1 claim or speed benchmark is made by this cache.
