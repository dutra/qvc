#!/usr/bin/env python3
"""Fetch pinned public Pantheon+ reproduction inputs and verify SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/pantheon_reproduction/reference'))
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name('pantheon_reproduction_reference.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for filename, info in manifest['files'].items():
        path = args.output/filename
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']:
            continue
        with urllib.request.urlopen(info['url'], timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != info['sha256']:
            raise ValueError(f'Checksum mismatch: {filename}')
        path.write_bytes(content)
        print(path)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
