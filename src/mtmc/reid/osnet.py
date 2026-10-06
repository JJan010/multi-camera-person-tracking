"""Reusable CUDA FP32 OSNet encoder for in-memory RGB person crops."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ObservationKey:
    """An observation within one tracking run; local IDs are camera-scoped."""

    camera_id: int
    local_id: int
    frame_index: int


@dataclass(frozen=True)
class PersonCrop:
    key: ObservationKey
    timestamp: Fraction
    rgb: np.ndarray  # HWC uint8, RGB, original crop resolution


@dataclass(frozen=True)
class ReIDBatch:
    keys: tuple[ObservationKey, ...]
    timestamps: tuple[Fraction, ...]
    embeddings: np.ndarray  # CPU float32, (N, 512), L2-normalized rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_crops(crops: Sequence[PersonCrop]) -> tuple[PersonCrop, ...]:
    """Validate the caller's contract before starting GPU work."""
    items = tuple(crops)
    seen = set()
    for item in items:
        if not isinstance(item, PersonCrop) or not isinstance(item.key, ObservationKey):
            raise TypeError("Expected PersonCrop with an ObservationKey")
        key = item.key
        if any(type(v) is not int or v < 0 for v in
               (key.camera_id, key.local_id, key.frame_index)):
            raise ValueError("Camera, local ID and frame must be nonnegative Python integers")
        if key in seen:
            raise ValueError(f"Duplicate observation: {key}")
        seen.add(key)
        if not isinstance(item.timestamp, Fraction) or item.timestamp < 0:
            raise ValueError("Timestamp must be a nonnegative Fraction in seconds")
        rgb = item.rgb
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
                or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0):
            raise ValueError(f"Expected a nonempty HWC uint8 RGB crop: {key}")
    return items


class OSNetEncoder:
    """One persistent model; one GPU batch per encode call; no crop filtering.

    Initialization verifies pinned assets and sets process-wide baseline flags:
    TF32 disabled for CUDA matmul/cuDNN, cuDNN benchmark disabled. Calls require
    those flags to remain unchanged. The adapter is intended for a single owner
    in the pipeline. Observation keys are scoped to the caller's tracking run.
    """

    def __init__(self, config_path: Path, *, project_root: Path, device: str = "cuda:0"):
        import torch
        from torchvision import transforms

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("OSNetEncoder requires a CUDA device")
        if torch.get_default_dtype() != torch.float32:
            raise RuntimeError("The FP32 baseline requires the default torch dtype to be float32")
        self.config_path = Path(config_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.configuration = json.loads(self.config_path.read_text(encoding="utf-8"))
        config = self.configuration
        if (config["architecture"] != "osnet_x1_0" or config["feature_dim"] != 512
                or config["input_size_hw"] != [256, 128]):
            raise ValueError("Expected OSNet x1.0, 512 output features and H=256 W=128")
        for spec in config["assets"].values():
            if sha256(self.project_root / spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"Model asset checksum mismatch: {spec['path']}")
        from ._vendor import osnet as upstream
        source_path = self.project_root / config["assets"]["source"]["path"]
        if Path(upstream.__file__).resolve() != source_path.resolve():
            raise RuntimeError("Imported OSNet source differs from the configured source path")

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        state = torch.load(self.project_root / config["assets"]["weights"]["path"],
                           map_location="cpu", weights_only=True)
        model = upstream.osnet_x1_0(num_classes=config["checkpoint_num_classes"], pretrained=False)
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(device=self.device, dtype=torch.float32)
        self.transform = transforms.Compose([
            transforms.Resize((256, 128), interpolation=transforms.InterpolationMode.BILINEAR,
                              antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def prepare_inputs(self, crops: Sequence[PersonCrop]) -> torch.Tensor:
        """CPU preprocessing only; output NCHW FP32, in caller-supplied order.

        This boundary can also be used by a benchmark to separate preprocessing
        from transfers and the model forward pass. Empty input returns (0,3,256,128).
        """
        items = validate_crops(crops)
        torch = self._torch
        if not items:
            return torch.empty((0, 3, 256, 128), dtype=torch.float32, device="cpu")
        if torch.get_default_dtype() != torch.float32:
            raise RuntimeError("Default torch dtype changed after OSNet initialization")
        return torch.stack([self.transform(Image.fromarray(item.rgb)) for item in items])

    def encode(self, crops: Sequence[PersonCrop]) -> ReIDBatch:
        """Return features and their observation mapping; performs no file I/O."""
        items = validate_crops(crops)
        keys = tuple(item.key for item in items)
        timestamps = tuple(item.timestamp for item in items)
        if not items:
            return ReIDBatch(keys, timestamps, np.empty((0, 512), dtype=np.float32))
        torch = self._torch
        if (self.model.training or torch.backends.cuda.matmul.allow_tf32
                or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark):
            raise RuntimeError("OSNet evaluation mode or baseline backend flags have changed")
        inputs = self.prepare_inputs(items).to(device=self.device, dtype=torch.float32)
        # eval() controls BatchNorm behavior; inference_mode() disables autograd work.
        # Explicitly disable any ambient autocast to preserve this FP32 reference.
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            features = self.model(inputs)
            if features.shape != (len(items), 512) or features.dtype != torch.float32:
                raise RuntimeError("Invalid OSNet output shape or dtype")
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            embeddings = features.cpu().numpy()
        # The CPU result is ready here. No separate global CUDA sync is needed.
        if (not np.isfinite(embeddings).all()
                or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, rtol=0, atol=1e-5)):
            raise RuntimeError("Invalid or zero OSNet embedding")
        return ReIDBatch(keys, timestamps, embeddings)
