"""Check OSNet adapter parity and row mapping against saved snapshot features."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from mtmc.reid.osnet import ObservationKey, OSNetEncoder, PersonCrop, sha256

ROOT = Path(__file__).resolve().parents[1]
ATOL = 1e-5


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != np.float32:
        raise AssertionError("Unexpected embedding shape/dtype")
    error = float(np.max(np.abs(actual - expected)))
    cosine = float(np.min(np.clip(np.sum(actual * expected, axis=1), -1, 1)))
    if not np.isfinite(error) or not np.isfinite(cosine) or error > ATOL or cosine < 1 - ATOL:
        raise AssertionError(f"Embedding parity failed: max_abs={error}, min_cosine={cosine}")
    return {"max_absolute_error": error, "minimum_cosine": cosine}


def check_mapping(batch, crops):
    if batch.keys != tuple(c.key for c in crops) or batch.timestamps != tuple(c.timestamp for c in crops):
        raise AssertionError("Embedding-to-observation mapping changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Saved embedding manifest.json")
    args = parser.parse_args()
    reference_path = args.reference.resolve()
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    feature_path = reference_path.parent / reference["embeddings"]["path"]
    if sha256(feature_path) != reference["embeddings"]["sha256"]:
        raise ValueError("Reference embedding checksum mismatch")
    expected = np.load(feature_path, allow_pickle=False)
    crop_manifest_path = Path(reference["source_manifest"]["path"])
    if sha256(crop_manifest_path) != reference["source_manifest"]["sha256"]:
        raise ValueError("Source crop manifest checksum mismatch")
    crop_manifest = json.loads(crop_manifest_path.read_text(encoding="utf-8"))
    fps = crop_manifest["configuration"]["fps"]
    records = reference["records"]
    if ([r["embedding_row"] for r in records] != list(range(len(records)))
            or len(records) < 2 or expected.shape != (len(records), 512)
            or expected.dtype != np.float32 or not np.isfinite(expected).all()
            or not np.allclose(np.linalg.norm(expected, axis=1), 1, rtol=0, atol=ATOL)):
        raise ValueError("Invalid reference matrix or observation mapping")
    crops = []
    for record in records:
        original = crop_manifest["crops"][record["source_manifest_index"]]
        if any(record[key] != value for key, value in original.items()):
            raise ValueError("Reference differs from the source crop record")
        path = crop_manifest_path.parent / record["crop_path"]
        if sha256(path) != record["crop_sha256"]:
            raise ValueError(f"Crop checksum mismatch: {path}")
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (record["width"], record["height"]):
                raise ValueError(f"Invalid crop image: {path}")
            rgb = np.array(image, dtype=np.uint8, copy=True)
        timestamp = Fraction(record["frame_index"], fps)
        if abs(float(timestamp) - record["timestamp_seconds"]) > 1e-9:
            raise ValueError("Unexpected source timestamp")
        crops.append(PersonCrop(ObservationKey(record["camera"], record["local_id"],
                                              record["frame_index"]), timestamp, rgb))
    config_path = ROOT / "configs/models/osnet_x1_0_msmt17.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config != reference["model_configuration"]:
        raise ValueError("Adapter configuration differs from the saved reference")
    print(f"Reference crops: {len(crops)}; cameras: {dict(sorted(Counter(c.key.camera_id for c in crops).items()))}")
    print("Loading persistent OSNet encoder...", flush=True)
    encoder = OSNetEncoder(config_path, project_root=ROOT)
    calls = []
    hook = encoder.model.register_forward_pre_hook(lambda _module, _inputs: calls.append(1))
    checks = {}
    try:
        result = encoder.encode(crops)
        check_mapping(result, crops)
        checks["reference_parity"] = compare(result.embeddings, expected)
        if len(calls) != 1:
            raise AssertionError("Expected one model call for the full cross-camera batch")
        print("Reference parity:", json.dumps(checks["reference_parity"]))

        reverse = list(reversed(crops))
        reversed_result = encoder.encode(reverse)
        check_mapping(reversed_result, reverse)
        checks["permuted_order"] = compare(reversed_result.embeddings, expected[::-1])
        print("Changed input order preserves observation mapping: OK")

        single = encoder.encode(crops[:1])
        check_mapping(single, crops[:1])
        checks["single_crop"] = compare(single.embeddings, expected[:1])
        print("Single-crop batch matches reference: OK")

        # A crop sliced from a full camera image is normally non-contiguous.
        source = crops[0]
        h, w, _ = source.rgb.shape
        backing = np.zeros((h, w + 2, 3), dtype=np.uint8)
        backing[:, 1:w + 1] = source.rgb
        view = backing[:, 1:w + 1]
        if view.flags.c_contiguous:
            raise AssertionError("The non-contiguous fixture is invalid")
        strided = PersonCrop(source.key, source.timestamp, view)
        strided_result = encoder.encode([strided])
        check_mapping(strided_result, [strided])
        checks["noncontiguous_crop"] = compare(strided_result.embeddings, expected[:1])
        print("Non-contiguous RGB crop matches reference: OK")

        calls_before_empty = len(calls)
        empty = encoder.encode([])
        if (empty.keys or empty.timestamps or empty.embeddings.shape != (0, 512)
                or empty.embeddings.dtype != np.float32 or len(calls) != calls_before_empty):
            raise AssertionError("Empty input must return (0, 512) without model execution")
        checks["empty_batch"] = {"shape": [0, 512], "model_calls": 0}
        print("Empty batch returns (0, 512) without model execution: OK")

        for label, invalid in (
            ("duplicate_keys", [source, source]),
            ("float_pixels", [PersonCrop(source.key, source.timestamp, source.rgb.astype(np.float32))]),
        ):
            before = len(calls)
            try:
                encoder.encode(invalid)
            except ValueError:
                if len(calls) != before:
                    raise AssertionError("Invalid input reached the model")
                checks[label] = "rejected_before_inference"
            else:
                raise AssertionError(f"Invalid input was accepted: {label}")
        print("Duplicate keys and float pixels rejected before inference: OK")
    finally:
        hook.remove()

    output = ROOT / "artifacts/reid_adapter_checks" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
        "adapter_sha256": sha256(ROOT / "src/mtmc/reid/osnet.py"),
        "script_sha256": sha256(Path(__file__)),
        "configuration_sha256": sha256(config_path),
        "tolerances": {"max_absolute_error": ATOL, "minimum_cosine": 1 - ATOL},
        "checks": checks, "passed": True, "crop_count": len(crops),
        "device": str(encoder.device), "dtype": str(next(encoder.model.parameters()).dtype),
        "gpu": torch.cuda.get_device_name(encoder.device), "cuda_build": torch.version.cuda,
        "versions": {p: version(p) for p in ("torch", "torchvision", "numpy", "pillow")},
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print("OSNet adapter smoke test: PASSED")


if __name__ == "__main__":
    main()
