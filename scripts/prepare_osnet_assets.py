"""Download pinned upstream OSNet code, its license and MSMT17 Re-ID weights."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

REVISION = "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50"
REPOSITORY = "https://github.com/KaiyangZhou/deep-person-reid"
RAW = f"https://raw.githubusercontent.com/KaiyangZhou/deep-person-reid/{REVISION}"
WEIGHT_ID = "1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"
ASSETS = {
    "source": {
        "path": "src/mtmc/reid/_vendor/osnet.py",
        "url": f"{RAW}/torchreid/models/osnet.py",
        "sha256": "c7c1c29187d6330f859c91da229271531920464c7011aec13842a086b2263cae",
    },
    "license": {
        "path": "src/mtmc/reid/_vendor/LICENSE.osnet",
        "url": f"{RAW}/LICENSE",
        "sha256": "3ac8ce2a83d170cb1c7c84152e0c1faca1f187794303514383960d2441716247",
    },
    "weights": {
        "path": "artifacts/models/osnet/osnet_x1_0_msmt17_combineall.pth",
        "url": f"https://drive.usercontent.google.com/download?id={WEIGHT_ID}&export=download&confirm=t",
        "sha256": "48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(root, name, spec):
    target = root / spec["path"]
    if target.exists():
        if sha256(target) != spec["sha256"]:
            raise RuntimeError(f"Existing file has a different checksum: {target}")
        print(f"{name}: existing file verified", flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        print(f"Downloading {name}...", flush=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "mtmc-osnet-setup/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(block)
        if sha256(temporary) != spec["sha256"]:
            raise RuntimeError(f"Checksum mismatch for {name}; refusing downloaded content")
        os.replace(temporary, target)
        temporary = None
        print(f"{name}: SHA256 verified ({target.stat().st_size} bytes)", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().project_root.resolve()
    for name, spec in ASSETS.items():
        fetch(root, name, spec)
    for relative, docstring in {
        "src/mtmc/reid/__init__.py": '"""Person re-identification components."""\n',
        "src/mtmc/reid/_vendor/__init__.py": '"""Upstream model implementations; see adjacent license files."""\n',
    }.items():
        path = root / relative
        if not path.exists():
            path.write_text(docstring, encoding="utf-8")
    configuration = {
        "architecture": "osnet_x1_0", "feature_dim": 512,
        "checkpoint_num_classes": 4101, "training_dataset": "MSMT17",
        "combineall": True, "input_size_hw": [256, 128],
        "upstream_repository": REPOSITORY, "upstream_revision": REVISION,
        "model_zoo": "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
        "weights_page": f"https://drive.google.com/file/d/{WEIGHT_ID}/view",
        "code_license": "MIT", "assets": ASSETS,
        "note": "Architecture source is unmodified. Use pretrained=False and load these Re-ID weights explicitly.",
    }
    config_path = root / "configs/models/osnet_x1_0_msmt17.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != configuration:
        raise RuntimeError(f"Existing configuration differs: {config_path}")
    config_path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    print(f"Configuration: {config_path}")
    print("OSNet assets: VERIFIED")


if __name__ == "__main__":
    main()
