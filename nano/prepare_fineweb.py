#!/usr/bin/env python3
"""nanokimi - fineweb-edu corpus preparation (prepare_fineweb.py)

Streams fineweb-edu (default: the public HuggingFaceFW mirror, sample-10BT
subset; HuggingFaceTB/fineweb-edu itself is gated) with the datasets
streaming API, tokenizes with the REAL Kimi K3 BPE (tiktoken.model,
full 163840-id vocab, NO remap) and writes:
  - tokens.bin      : stream of K3 ids in uint32 LE (BOS/EOS per document)
  - tokens.meta.json: {count, dtype, vocab, bos, eos, source}

train.py auto-detects the uint32 dtype from the meta file (see load_tokens).

usage:
  python3 prepare_fineweb.py --out ~/fineweb_k3 --max-tokens 100000000
  python3 prepare_fineweb.py --out ~/fineweb_k3 --max-docs 1000   # smoke

Bandwidth discipline: streaming only downloads the shards actually consumed;
use --max-tokens / --max-docs to keep the run bounded (default cap 500 M).
"""
import argparse
import json
import os
import time

import numpy as np
from datasets import load_dataset

from prepare import make_encoder

# Real K3 special ids (ref/config.json: bos_token_id / eos_token_id).
BOS_K3 = 163584
EOS_K3 = 163586
VOCAB_K3 = 163840


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT", help="dataset subset (config)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-tokens", type=int, default=500_000_000,
                    help="cap of the final stream (K3 tokens, BOS/EOS included)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="stop after K documents (smoke runs)")
    ap.add_argument("--batch-docs", type=int, default=64,
                    help="documents per encode batch")
    ap.add_argument("--threads", type=int, default=4,
                    help="tiktoken encode threads (keep modest on shared boxes)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    enc = make_encoder()
    t0 = time.time()

    ds = load_dataset(args.dataset, args.name, split=args.split, streaming=True)
    source = f"{args.dataset}:{args.name}:{args.split}"
    out_bin = os.path.join(args.out, "tokens.bin")
    total = 0
    ndocs = 0
    batch = []

    def flush(f, batch):
        """Encodes and writes one batch of documents, returns tokens written."""
        nonlocal total
        written = 0
        for ids in enc.encode_ordinary_batch(batch, num_threads=args.threads):
            seq = np.empty(len(ids) + 2, dtype=np.uint32)
            seq[0] = BOS_K3
            seq[1:-1] = ids
            seq[-1] = EOS_K3
            if total + len(seq) > args.max_tokens:
                seq = seq[: args.max_tokens - total]
            seq.tofile(f)
            total += len(seq)
            written += len(seq)
            if total >= args.max_tokens:
                break
        return written

    print(f"streaming {source} -> {out_bin} (cap {args.max_tokens / 1e6:.0f} M tokens)", flush=True)
    with open(out_bin, "wb") as f:
        for doc in ds:
            text = doc.get("text")
            if not text:
                continue
            batch.append(text)
            if len(batch) >= args.batch_docs:
                flush(f, batch)
                ndocs += len(batch)
                batch = []
                if ndocs % (args.batch_docs * 20) == 0:
                    print(f"  {ndocs} docs, {total / 1e6:.1f} M tokens, "
                          f"{time.time() - t0:.0f}s", flush=True)
                if (args.max_docs is not None and ndocs >= args.max_docs) \
                        or total >= args.max_tokens:
                    break
        if batch and total < args.max_tokens \
                and (args.max_docs is None or ndocs < args.max_docs):
            flush(f, batch)
            ndocs += len(batch)

    print(f"  total: {total / 1e6:.2f} M K3 tokens over {ndocs} docs", flush=True)
    with open(os.path.join(args.out, "tokens.meta.json"), "w") as f:
        json.dump({
            "count": int(total),
            "dtype": "uint32le",
            "vocab": VOCAB_K3,
            "bos": BOS_K3,
            "eos": EOS_K3,
            "docs": int(ndocs),
            "source": source,
        }, f)
    size_mb = total * 4 / 1e6
    print(f"-> {out_bin} ({size_mb:.0f} MB), tokens.meta.json", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
