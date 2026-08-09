#!/usr/bin/env python3
"""nanokimi - capture_bounds verification: runs the in-module synthetic
selftest (tiny safetensors root, full capture pipeline, per-boundary parity
against a directly loaded reference model).

usage: python3 test_capture_bounds.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "..", "capture_bounds.py"),
                        "--selftest"],
                       capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0 or "capture_bounds selftest OK" not in r.stdout:
        print(r.stderr, end="", file=sys.stderr)
        raise SystemExit("test_capture_bounds FAILED")
    print("test_capture_bounds OK")


if __name__ == "__main__":
    main()
