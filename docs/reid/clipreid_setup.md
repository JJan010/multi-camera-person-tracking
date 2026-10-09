# CLIP-ReID ViT-B/16: pinned visual inference candidate

Status: separate experimental model; existing OSNet pipeline unchanged.

## Source and selected variant

Paper: Siyuan Li, Li Sun, Qingli Li, *CLIP-ReID: Exploiting Vision-Language
Model for Image Re-Identification without Concrete Text Labels*, AAAI 2023.
https://arxiv.org/abs/2211.13977

Official repository: https://github.com/Syliz517/CLIP-ReID
Pinned revision: `eb1898b72c882875f478bebfc6d41644eece0a5d`.
Checkpoint: author-linked MSMT17 **ViT-CLIP-ReID**, not the baseline or
SIE-OLP variant. File: `MSMT17_clipreid_ViT-B-16_60.pth`.

- Official checkpoint has 315 tensors, including 152 image-encoder tensors.
- Classifiers have 1041 training classes. They are not person IDs for new scenes.
- Image encoder has 86,140,416 parameters.
- Patch size and stride are 16. Input resolution is 256 x 128 (height x width).
- No camera/view embeddings are present or required.
- `TEST.NECK_FEAT='before'`: concatenate final image CLS (768 dimensions)
  and its learned projection (512 dimensions), then normalize the combined
  1280-D vector to unit L2 norm. Do not normalize the branches separately.
- Preprocessing: PIL RGB, bilinear resize, ToTensor, mean/std both (0.5,0.5,0.5).
- The 512-D `get_image=True` training helper is NOT the configured final
  retrieval representation. Taking that shortcut would change the experiment.

The repository code is MIT-licensed. The downloader retains the author's
license and the underlying OpenAI CLIP license. No separate license grant
for the checkpoint was verified; the code license alone is not evidence of
one. We record the official download provenance without redistributing weights.

## Asset integrity

`configs/models/clipreid_vit_b16_msmt17.json` pins URLs, byte counts and SHA256
for nine assets. The checkpoint is 511,251,513 bytes (487.57 MiB):

`9974ef57c418e8c88e59581f66f3f53fbe02f879311f7438009f84fd150b1c63`

Hashes were computed from the official author-linked downloads on 2026-10-08;
they are reproducibility pins, not author-published signed checksums.
Code comes from a fixed commit, not a moving branch.
Downloads use temporary files and verify size/SHA before promotion. Existing
mismatched assets are rejected rather than replaced. HTML download pages are
rejected. Models and upstream reference files stay under ignored
`artifacts/models/clipreid/` and can be recreated from the manifest.

## WSL commands

From `/home/jakjan/projects/multi-camera-person-tracking`:

```bash
source .venv/bin/activate
python scripts/prepare_clipreid_assets.py
PYTHONPATH="$PWD/src" python scripts/check_clipreid_model.py
```

No package installation, downgrade, base CLIP download or tokenizer is needed.
The existing torch, torchvision, NumPy and Pillow packages suffice for this
visual path. We load all image-encoder tensors with `strict=True` and load
the checkpoint on CPU using `weights_only=True` before moving the model to CUDA.
Classifier, BN and text-training branches are intentionally excluded from the
production visual wrapper; they are not silently missing visual parameters.

The smoke test defaults to `cuda:0` and writes
`artifacts/clipreid_model_checks/<run_id>/report.json`. For asset verification
without inference use `python scripts/prepare_clipreid_assets.py --verify-only`.

## Scope of verification

The test executes the exact pinned upstream `build_transformer.forward`
method in its image-only evaluation branch, with the trained BN branches and
the same strictly loaded visual encoder. Its original constructor initializes
training-only text machinery and downloads base CLIP, so it is not invoked.
The smoke test compares raw and normalized outputs with this reference,
checks RGB normalization, finite unit vectors, batch permutations and single
versus batch inference. It uses synthetic pixels, not actual people.

Local technical verification: Python 3.12, torch 2.6.0+cpu,
torchvision 0.21.0+cpu; output (3,1280), raw reference max absolute error 0.
This separate temporary test environment is not a change to the user's WSL
requirements. Python 3.10 syntax was checked. Actual CUDA verification on
WSL torch 2.11.0+cu128 remains the user's next execution step.

Reference parity is within the tested software environment. It does not
reproduce the authors' published benchmark or establish cross-environment
bitwise equivalence, Re-ID quality, TensorRT compatibility or performance.

## Next experiment

1. Encode the same recorded person crops with OSNet and CLIP-ReID, preserving
   exact observation keys and model-specific preprocessing.
2. Evaluate retrieval on fixed observations, including cross-camera and
   separated-time examples. Keep crop selection unchanged for this comparison.
3. Record feature dimension and model identity explicitly. Existing OSNet
   history/association contracts assume 512-D; do not insert or truncate 1280-D
   outputs into those APIs. Generalization must preserve frozen OSNet parity.
4. Estimate model-specific similarity thresholds on development data, then
   freeze them before another held-out scene. Never mix OSNet and CLIP-ReID
   vectors within one history or compare them directly.
5. Compare global ID quality and full preprocessing/encoding cost; only then
   assess ONNX/TensorRT and precision trade-offs.

Scene_001 and scene_041 have both informed development. No improvement on
these sequences alone establishes generalization. Larger embeddings and a
larger model do not guarantee better identity tracking; background-dominated
crops and registry errors remain separate limitations.
