#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

from tqdm import tqdm


def count_files(directory: Path) -> int:
    return sum(1 for p in directory.iterdir() if p.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch a directory and show a tqdm progress bar as files appear."
    )
    parser.add_argument("N", type=int, help="Target total number of files")
    parser.add_argument("directory", type=Path, help="Directory to watch")
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()

    if args.N < 0:
        print("N must be non-negative", file=sys.stderr)
        return 1
    if not args.directory.is_dir():
        print(f"Not a directory: {args.directory}", file=sys.stderr)
        return 1

    last_count = min(count_files(args.directory), args.N)

    with tqdm(total=args.N, initial=last_count, desc="Files", unit="file") as pbar:
        while last_count < args.N:
            time.sleep(args.interval)
            current_count = min(count_files(args.directory), args.N)
            if current_count > last_count:
                pbar.update(current_count - last_count)
                last_count = current_count

    return 0


if __name__ == "__main__":
    raise SystemExit(main())