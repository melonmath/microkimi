#!/usr/bin/env python3
"""Wrapper test: runs the selftest of every tool in nano/graft/ as a
subprocess and greps for its OK sentinel (house style, like
test_stitch_solve.py)."""
import os
import subprocess
import sys

GRAFT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "graft")

TOOLS = [
    ("graftlib.py", "graftlib selftest OK"),
    ("capture_donor.py", "capture_donor selftest OK"),
    ("capture_host.py", "capture_host selftest OK"),
    ("expert_solve.py", "expert_solve selftest OK"),
    ("inject_experts.py", "inject_experts selftest OK"),
    ("graft_heal.py", "graft_heal selftest OK"),
    ("route_solve.py", "route_solve selftest OK"),
    ("als_refit.py", "als_refit selftest OK"),
    ("tokens_to_text.py", "tokens_to_text selftest OK"),
]


def main():
    for tool, sentinel in TOOLS:
        p = subprocess.run(
            [sys.executable, os.path.join(GRAFT, tool), "--selftest"],
            capture_output=True, text=True)
        if p.returncode != 0 or sentinel not in p.stdout:
            print(p.stdout)
            print(p.stderr, file=sys.stderr)
            raise SystemExit(f"{tool} selftest failed")
        print(f"{tool}: OK")
    print("test_graft OK")


if __name__ == "__main__":
    main()
