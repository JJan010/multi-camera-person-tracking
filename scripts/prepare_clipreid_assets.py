"""Download the pinned official CLIP-ReID assets; do not install packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/models/clipreid_vit_b16_msmt17.json'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(root, name, item):
    target = root / item['path']
    if target.exists():
        if target.stat().st_size != item['bytes'] or sha256(target) != item['sha256']:
            raise RuntimeError(f'Existing asset differs; not overwriting: {target}')
        print(f'{name}: existing asset VERIFIED', flush=True)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        print(f'Downloading {name}: {item["bytes"] / 1024**2:.2f} MiB...', flush=True)
        request = urllib.request.Request(item['url'], headers={'User-Agent': 'mtmc-clipreid-assets/1.0'})
        with urllib.request.urlopen(request, timeout=45) as response:
            if 'text/html' in response.headers.get('Content-Type', '').lower():
                raise RuntimeError('Received an HTML page instead of an asset; check the official weights link')
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                last = time.monotonic()
                for block in iter(lambda: response.read(4 * 1024 * 1024), b''):
                    size += len(block)
                    if size > item['bytes']:
                        raise RuntimeError('Downloaded asset exceeds pinned size')
                    stream.write(block)
                    if time.monotonic() - last > 10:
                        print(f'  {size / 1024**2:.1f} / {item["bytes"] / 1024**2:.1f} MiB', flush=True)
                        last = time.monotonic()
        if temporary.stat().st_size != item['bytes'] or sha256(temporary) != item['sha256']:
            raise RuntimeError(f'Pinned size/SHA256 mismatch: {name}')
        os.replace(temporary, target)
        temporary = None
        print(f'{name}: size and SHA256 VERIFIED', flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    for name, item in config['assets'].items():
        if args.verify_only:
            path = ROOT / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                raise RuntimeError(f'Missing or changed asset: {path}')
            print(f'{name}: VERIFIED')
        else:
            fetch(ROOT, name, item)
    print('CLIP-ReID assets: VERIFIED; installed packages and existing pipeline unchanged')


if __name__ == '__main__':
    main()
