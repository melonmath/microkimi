#!/usr/bin/env python3
"""nanokimi - token frequency counting (count_freq.py)

Counts token id occurrences in an ALREADY tokenized corpus (binary stream of
token ids + .meta.json sidecar, the format written by nano/prepare.py:
{"count": N, "dtype": "uint16le"|"uint32le", ...}) and writes a freqfile for
`microkimi slice --vocab-top N <freqfile>`.

freqfile format (text): one "<token_id> <count>" per line, most frequent
first; '#' comments allowed (the slicer also accepts a flat JSON object
{"<id>": <count>, ...}). Only ids with count > 0 are listed.

usage:
  python3 count_freq.py --tokens tokens.bin --out freq.txt
      [--meta tokens.meta.json] [--dtype uint32le] [--vocab 163840]

The sidecar defaults to <tokens with .bin stripped>.meta.json, then
<tokens>.meta.json. --dtype overrides the sidecar, --vocab only sizes the
histogram (default: max seen id + 1).
"""
import argparse
import array
import json
import os
import sys

DTYPES = {"uint16le": ("H", 2), "uint32le": ("I", 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True, help="binary stream of token ids")
    ap.add_argument("--out", required=True, help="freqfile to write")
    ap.add_argument("--meta", default=None, help=".meta.json sidecar (default: next to --tokens)")
    ap.add_argument("--dtype", default=None, choices=sorted(DTYPES), help="override the sidecar dtype")
    ap.add_argument("--vocab", type=int, default=None, help="histogram size (default: max id + 1)")
    args = ap.parse_args()

    meta = {}
    meta_path = args.meta
    if meta_path is None:
        stem = args.tokens[: -len(".bin")] if args.tokens.endswith(".bin") else args.tokens
        for cand in (stem + ".meta.json", args.tokens + ".meta.json"):
            if os.path.exists(cand):
                meta_path = cand
                break
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"meta: {meta_path} -> {meta}", flush=True)
    dtype = args.dtype or meta.get("dtype")
    if dtype not in DTYPES:
        sys.exit(f"error: unknown dtype {dtype!r} (pass --dtype uint16le|uint32le or a .meta.json sidecar)")

    code, width = DTYPES[dtype]
    size = os.path.getsize(args.tokens)
    if size % width:
        sys.exit(f"error: {args.tokens} is {size} bytes, not a multiple of {width} ({dtype})")
    a = array.array(code)
    with open(args.tokens, "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder == "big":
        a.byteswap()  # corpus is little-endian
    declared = meta.get("count")
    if declared is not None and declared != len(a):
        print(f"warning: meta count {declared} != {len(a)} ids in the file - counting the file", flush=True)

    vocab = args.vocab or (max(a) + 1 if a else 0)
    hist = [0] * vocab
    bad = 0
    for t in a:
        if t < vocab:
            hist[t] += 1
        else:
            bad += 1
    if bad:
        sys.exit(f"error: {bad} ids >= --vocab {vocab} (the corpus does not match the model vocab)")

    order = [(c, i) for i, c in enumerate(hist) if c > 0]
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# microkimi freqfile v1: {len(a)} tokens, {len(order)} distinct ids, source {args.tokens} ({dtype})\n")
        for c, i in sorted(order, key=lambda x: (-x[0], x[1])):  # count desc, id asc
            f.write(f"{i} {c}\n")
    print(f"-> {args.out}: {len(order)} distinct ids over {len(a)} tokens", flush=True)


if __name__ == "__main__":
    main()
