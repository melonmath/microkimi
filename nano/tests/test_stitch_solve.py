#!/usr/bin/env python3
"""nanokimi - stitch_solve verification: runs the in-module synthetic
selftest (ridge recovery of a known rank-8 correction map).

usage: python3 test_stitch_solve.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    r = subprocess.run([sys.executable,
                        os.path.join(HERE, "..", "stitch_solve.py"),
                        "--selftest"],
                       capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0 or "stitch_solve selftest OK" not in r.stdout:
        print(r.stderr, end="", file=sys.stderr)
        raise SystemExit("test_stitch_solve FAILED")
    print("test_stitch_solve OK")


if __name__ == "__main__":
    main()
