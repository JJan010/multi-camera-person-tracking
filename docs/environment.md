# Development environment

GPU baseline verified on 2026-10-06.

## Verified configuration

| Component | Value |
| --- | --- |
| Host | Windows 11 |
| Linux | Ubuntu 22.04.5 LTS in WSL 2 |
| WSL kernel | 6.18.40.1-microsoft-standard-WSL2 |
| Python | 3.10.12 |
| Environment | Project-local .venv |
| GPU | NVIDIA GeForce RTX 5080 |
| GPU memory reported by nvidia-smi | 16303 MiB |
| Windows NVIDIA KMD | 617.42 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| PyTorch CUDA build | 12.8 |
| GPU compute capability | 12.0 |

## Validation

- pip check: passed.
- CUDA availability: confirmed.
- Matrix multiplication on cuda:0: passed.
- torchvision NMS on CUDA: passed.

These checks validate basic execution, not application performance.
RF-DETR, OSNet, TensorRT and NVDEC have not yet been validated.

## Dependency files

- constraints.txt pins the selected GPU package versions.
- requirements-gpu.txt selects the CUDA 12.8 package index.
- requirements-frozen.txt records installed dependency versions.
  It is a package snapshot, not a complete system lockfile.

## Recreate the baseline

Run from the repository root in a fresh Python 3.10 virtual environment
on compatible Linux x86_64 with an NVIDIA driver supporting CUDA 12.8:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -c constraints.txt -r requirements-frozen.txt
python -m pip check
python scripts/check_environment.py
```

The clean-environment recreation procedure has not yet been tested.

Use -c constraints.txt when adding dependencies. Review conflicts before
changing the pinned GPU baseline.

## References

- https://pytorch.org/get-started/previous-versions/
- https://pip.pypa.io/en/stable/cli/pip_freeze/
