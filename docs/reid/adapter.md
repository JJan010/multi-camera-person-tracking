# OSNet adapter contract

`src/mtmc/reid/osnet.py` exposes a persistent CUDA FP32 encoder. Construct one
`OSNetEncoder` when starting the pipeline, then call `encode(crops)` for each
batch. Construction verifies the pinned source, license and weights and loads
the model with `strict=True`. Encoding performs no file reads or writes.

## Input and output

Each `PersonCrop` contains:

- `ObservationKey(camera_id, local_id, frame_index)` with nonnegative Python integers;
- `timestamp`, a nonnegative `fractions.Fraction` in seconds;
- `rgb`, an HWC `numpy.uint8` RGB array at original crop resolution.

RGB channel semantics are the caller's responsibility. Repeated keys in a batch
are rejected. Equal local IDs in different cameras are allowed. Non-contiguous
views, such as `frame.rgb[top:bottom, left:right]`, are supported. All supplied
crops are retained, and the input order is preserved.

`ReIDBatch` returns `keys`, `timestamps` and CPU `float32` `embeddings` with shape
`(N, 512)`. Row `i` belongs to `keys[i]` and `timestamps[i]`. Rows are L2-normalized.
Empty input produces `(0, 512)` without running the network.

Keys are scoped to one tracking run. The pipeline owner must keep separate
session/generation context when restarting trackers or combining runs.
The encoder does not assign global IDs, aggregate track history or filter crops.

## Numerical reference

The adapter keeps the snapshot baseline preprocessing: PIL bilinear resize to
H=256, W=128; values divided by 255; ImageNet channel means/standard deviations;
one model forward pass for the combined camera batch; per-row L2 normalization.
The original snapshot script remains an independent reference implementation.

The model runs in `eval()` and `torch.inference_mode()`. Evaluation mode controls
modules such as BatchNorm; inference mode disables autograd bookkeeping and does
not itself switch a model to evaluation mode. Autocast is disabled inside encode.

Initialization sets process-wide CUDA matmul/cuDNN TF32 flags to false and cuDNN
benchmark to false. Nonempty encode calls check that these flags and model
evaluation mode have not changed. The default PyTorch dtype must be FP32.
This adapter is intended for one pipeline owner, not concurrent callers changing
backend flags. Its host output is ready after the blocking device-to-host copy.

## Verification

Run from the project root in WSL with the project virtual environment active:

    PYTHONPATH="$PWD/src" python scripts/check_osnet_adapter.py --reference artifacts/reid_embeddings/20261006T194219632494Z/manifest.json

The check verifies crop, model and reference provenance, then compares all 57
embeddings with the saved FP32 snapshot. It also checks reversed input order,
a single-crop batch, a non-contiguous crop view, an empty batch, duplicate keys,
and accidental floating-point pixels. A forward hook checks that all cameras
use one network call, and that empty/invalid input causes no network call.

Predeclared tolerances: maximum absolute component error <= 1e-5 and minimum
corresponding-row cosine >= 0.99999. The script prints actual differences and
saves a report under `artifacts/reid_adapter_checks/<UTC>/`. A mismatch should
be investigated before changing tolerances. A success confirms this refactor's
numerical behavior on the reference data; it does not establish runtime speed.

`prepare_inputs(crops)` exposes CPU preprocessing for a subsequent benchmark.
Measure the GPU forward pass and full encode separately, after warmup, with
matching input batches and CUDA-aware timing. The first adapter step contains
no performance measurement.

## References

- PyTorch 2.11 model evaluation mode:
  https://docs.pytorch.org/docs/2.11/generated/torch.nn.Module.html#torch.nn.Module.eval
- PyTorch 2.11 inference mode:
  https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad_mode.inference_mode.html
- Authors' OSNet test preprocessing:
  https://kaiyangzhou.github.io/deep-person-reid/_modules/torchreid/data/transforms.html
