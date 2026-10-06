# In-memory crop integration

This step connects decoded RGB frames and local track observations to the
existing OSNet adapter. It introduces `src/mtmc/reid/crops.py` without changing
the reader, detector, tracker, model weights or previous reference scripts.

## Runtime contract

`build_person_crops(batch: FrameBatch, tracks: Sequence[CameraTracks])` returns a
`CropBatch`. `crops` contains `PersonCrop` objects accepted directly by
`OSNetEncoder.encode`. `records` contains geometry and confidence for every
track observation, including fully outside boxes with `crop_xyxy_int=None`.

Camera IDs determine the frame-to-track join; input list positions do not.
Each frame camera needs exactly one track result, which may contain zero tracks.
Frame indices and exact `Fraction` timestamps must agree across the batch.
Output order is camera ID, then local ID. Repeated local IDs across different
cameras are valid; duplicates within one camera are rejected. Keys belong to
one tracking run; a future session owner must isolate restarts/generations.

Geometry matches `preview_reid_crops.py`: clip the float box to image bounds,
floor left/top, ceil right/bottom, and slice `rgb[top:bottom, left:right]`.
The upper slice bounds are exclusive. Nonpositive or nonfinite original boxes
are errors. Fully outside boxes are recorded and skipped. Partial boxes remain,
without a confidence/size/visibility filter. `inside_image_fraction` is the
clipped area divided by the supplied box area; already-clipped tracker boxes
cannot reveal the portion of a person outside the camera image.

Crops preserve RGB, uint8 and the original crop resolution. They are read-only
NumPy views into the CPU image, not resized copies. The caller must keep the
source buffer unchanged until `encode()` returns; making the view read-only
does not prevent another owner from changing the source. Retaining even a small
view keeps its underlying frame alive. Release crops after encoding and retain
embeddings/metadata for future appearance history. Later GPU preprocessing will
have its own buffer ownership and parity checks.

Usage inside a frame-processing loop:

```python
prepared = build_person_crops(batch, tracks)
features = encoder.encode(prepared.crops)
```

`batch`, `tracks` and the persistent `encoder` are owned by the caller. There is
no file I/O or model execution inside the crop builder. Empty visible-track
sets yield an empty crop tuple, supported by the existing encoder.

## Verification

Run from the project root with its `.venv` active:

```bash
PYTHONPATH="$PWD/src" python scripts/check_reid_crops.py
PYTHONPATH="$PWD/src" python scripts/check_reid_integration.py \
  --reference artifacts/reid_embeddings/20261006T194219632494Z/manifest.json \
  --torch-threads 1
```

The CPU check covers pixel axes, rounding, clipping, empty inputs, camera-local
IDs, unordered camera inputs, view ownership and rejection of mismatched times.
It uses small structural fixtures, so it requires neither video files nor CUDA.

The integration check uses the real `synchronized_replay`, `CameraTracks` and
`OSNetEncoder`. It verifies video/trace/reference checksums, replays from frame
zero through the reference frame and reconstructs that frame's track objects
from the recorded trace. This isolates the new crop boundary from detector and
tracker reruns. Ground truth is not loaded.

All 57 reference crops at frame 150 must match the saved PNG pixels exactly:
camera 4: 16, camera 5: 22, camera 8: 19. PNGs are verification fixtures only;
the encoder receives views of replayed frames. Geometry, confidence, selection,
observation coverage and row mapping must also match. All cameras use one OSNet
model call. Embeddings must meet the existing, unchanged tolerances:
maximum absolute error <= 1e-5 and minimum cosine >= 0.99999.

The script sets CPU intra-op threads to 1 before model work; this is a local
diagnostic setting from the isolated benchmark, not a proven setting for the
complete detector/tracker pipeline. Default OSNet behavior remains unchanged.

Results are saved in a new `artifacts/reid_integration_checks/<UTC>/` directory,
with `report.json` and `embeddings.npy`. Reports include source hashes, input
provenance, versions, checks and embedding row keys. This is a one-frame
correctness check, not an end-to-end speed or global identity evaluation.
The user ran both checks successfully in WSL on 2026-10-06. For frame 150,
all 57 crops matched the PNG pixels exactly, reordered camera/track inputs
preserved the mapping, and one OSNet call returned matching observation keys
and timestamps. Reported maximum absolute embedding error was 0; the printed
minimum cosine was 0.999999821 (rounded console output).

Evidence reported by the user:
`artifacts/reid_integration_checks/20261006T203515268716Z/report.json`.
This confirms the specified snapshot boundary; it does not establish behavior
or speed for the complete continuous detector/tracker/Re-ID loop.

Next: use this same crop builder after live detector and local tracker updates
in a sequential replay loop, then measure the integrated path. Appearance
history, selective updates and global association remain later steps.

## References

- [NumPy 2.2: copies and views](https://numpy.org/doc/2.2/user/basics.copies.html):
  memory sharing and the difference between basic slicing and copying.
- [Zhou et al., Omni-Scale Feature Learning for Person Re-Identification,
  ICCV 2019](https://arxiv.org/abs/1905.00953): OSNet architecture. The join,
  geometry policy and verification protocol above are project implementation
  choices, not claims that the paper specifies this pipeline.
