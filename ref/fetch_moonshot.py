#!/usr/bin/env python3
"""Download Moonshot AI's Kimi K3 reference files at runtime (never vendored).

These files are distributed by Moonshot AI under Moonshot's own license at
<https://huggingface.co/moonshotai/Kimi-K3>. This helper fetches them on demand
into a cache directory (default: ~/.cache/microkimi/moonshot) and returns the
package directory to add to sys.path.
"""
import os
import urllib.request

BASE = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"
FILES = [
    "modeling_kimi_linear.py",
    "configuration_kimi_k3.py",
    "encoding_k3.py",
    "tokenization_kimi.py",
]


def ensure_moonshot(cache_dir=None, files=FILES, verbose=True):
    """Download the reference files if absent → path of the moonshot package dir."""
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "microkimi", "moonshot")
    pkg = os.path.join(cache_dir, "moonshot")
    os.makedirs(pkg, exist_ok=True)
    init = os.path.join(pkg, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()
    for name in files:
        dst = os.path.join(pkg, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = BASE + name
        if verbose:
            print(f"[fetch_moonshot] {url} → {dst}")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    return cache_dir


if __name__ == "__main__":
    ensure_moonshot()
