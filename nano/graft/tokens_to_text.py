#!/usr/bin/env python3
"""nanokimi - tokens_to_text: recover a raw-text jsonl corpus from a
tokens.bin stream (the inverse of prepare_fineweb.py's tokenization).

Reads a flat token-id stream (uint16 or uint32, auto-detected from the
sibling tokens.meta.json like train.load_tokens), splits documents at the
BOS/EOS ids declared in the meta, decodes each document with the K3
tiktoken model and writes one {"text": ...} json line per document.

The recovered JSONL preserves document boundaries and provides a stable raw
text representation for downstream consumers.

usage:
  python3 tokens_to_text.py --tokens data/tokens.bin --out corpus.jsonl \
      [--max-docs 2000] [--tiktoken ~/tiktoken.model]
  python3 tokens_to_text.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)


def split_docs(tokens, bos, eos):
    """Yields id arrays, one per document, specials stripped."""
    doc = []
    for t in tokens:
        if t == bos:
            if doc:
                yield np.asarray(doc)
            doc = []
        elif t == eos:
            if doc:
                yield np.asarray(doc)
            doc = []
        else:
            doc.append(int(t))
    if doc:
        yield np.asarray(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--tiktoken", default=None)
    args = ap.parse_args()

    meta_path = args.tokens.replace(".bin", ".meta.json")
    if not os.path.exists(meta_path):
        meta_path = args.tokens + ".meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    dtype = np.uint32 if "32" in str(meta.get("dtype", "uint16")) \
        else np.uint16
    bos, eos = meta.get("bos"), meta.get("eos")

    if args.tiktoken:
        os.environ["KIMI_TIKTOKEN"] = args.tiktoken
    from prepare import make_encoder
    enc = make_encoder()

    toks = np.memmap(args.tokens, dtype, "r")
    n = 0
    with open(args.out, "w") as out:
        for doc in split_docs(toks, bos, eos):
            text = enc.decode(doc.tolist())
            out.write(json.dumps({"text": text}) + "\n")
            n += 1
            if args.max_docs and n >= args.max_docs:
                break
    print(f"-> {args.out}: {n} documents")


def selftest():
    toks = [9, 1, 2, 3, 10, 9, 4, 5, 10, 9, 6]
    docs = list(split_docs(np.asarray(toks), 9, 10))
    assert [d.tolist() for d in docs] == [[1, 2, 3], [4, 5], [6]]
    # robust to missing BOS and truncated tail
    docs = list(split_docs(np.asarray([1, 2, 10, 3, 4]), 9, 10))
    assert [d.tolist() for d in docs] == [[1, 2], [3, 4]]
    print("claim 1: document splitting at BOS/EOS, tail kept")
    print("tokens_to_text selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
