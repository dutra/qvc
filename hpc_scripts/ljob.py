#!/usr/bin/env python3
import argparse
import glob
import os
import subprocess
import sys
import time
from collections import OrderedDict

def find_files(patterns):
    paths = []
    for pat in patterns:
        paths.extend(glob.iglob(pat, recursive=True))
    uniq = list(OrderedDict.fromkeys(paths))
    return [p for p in uniq if os.path.isfile(p)]

def run_with_sh(path) -> int:
    try:
        proc = subprocess.run(
            ["/bin/sh", path],
            stdout=sys.stdout,
            stderr=sys.stderr,
            check=False,
        )
        return proc.returncode
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"ERROR: failed to run {path}: {e}", file=sys.stderr)
        return 127

def main():
    ap = argparse.ArgumentParser(
        description="Run all scripts matching a glob with /bin/sh, showing progress."
    )
    ap.add_argument("pattern", nargs="+", help="Glob(s), e.g. 'tests/*.sh' or 'scripts/**/*.sh'")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="Stop after the first failing script.")
    args = ap.parse_args()

    files = find_files(args.pattern)
    if not files:
        print("No files matched.", file=sys.stderr)
        sys.exit(2)

    total = len(files)
    passed = failed = 0
    start = time.time()

    for i, path in enumerate(files, start=1):
        label = f"[{i}/{total}]"
        print(f"{label} {path}")
        rc = run_with_sh(path)

        if rc == 0:
            passed += 1
            status = "OK"
        elif rc == 130:
            failed += 1
            print(f"{label} {path}  -> INTERRUPTED (rc=130)")
            break
        else:
            failed += 1
            status = f"FAIL (rc={rc})"
            print(f"{label} {path}  -> {status}")
            if args.stop_on_fail:
                break

    dur = time.time() - start
    print("-" * 60)
    print(f"Ran {passed + failed}/{total} in {dur:.1f}s  |  PASS: {passed}  FAIL: {failed}")
    sys.exit(0 if failed == 0 and (passed + failed) == total else 1)

if __name__ == "__main__":
    main()
