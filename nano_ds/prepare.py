#!/usr/bin/env python3
"""nanodeepseek - corpus preparation (prepare.py)

Tokenizes the TinyStories corpus with the REAL DeepSeek-V4 tokenizer
(tokenizer.json, HF `tokenizers` runtime), builds a remap towards the top-N
most frequent tokens (+ specials), and writes:
  - tokens.bin         : stream of nano ids in uint16 LE (BOS/EOS per story)
  - tokens.meta.json   : {count, dtype, bos, eos}
  - vocab_ds_nano.json : remap table exported for the Rust engine
      { nano_to_ds: [ds_id x N], specials: {bos, eos, unk, pad}, vocab_size }

usage:
  python3 prepare.py --data ~/data --out ~/nanods_out --top 8192 \
      [--tokenizer ../microdeepseek.tokenizer.json] [--max-stories K] [--max-tokens M]
"""
import argparse
import base64
import json
import os
import time

import numpy as np
import tokenizers

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOK = os.path.join(_HERE, "..", "microdeepseek.tokenizer.json")

SPECIAL_NANO = {"bos": 8192, "eos": 8193, "unk": 8194, "pad": 8195}
VOCAB_NANO = 8200
DS_VOCAB = 129_280


def iter_stories(path, max_stories=None):
    """Iterates the 'story' fields of data*.json files (or of a single file)."""
    files = []
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.endswith(".json") and not f.startswith("tokens") and not f.startswith("vocab")
        )
    else:
        files = [path]
    n = 0
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            story = item.get("story")
            if story:
                yield story
                n += 1
                if max_stories is not None and n >= max_stories:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK if os.path.exists(DEFAULT_TOK) else None)
    ap.add_argument("--top", type=int, default=8192)
    ap.add_argument("--max-stories", type=int, default=None)
    ap.add_argument("--count-stories", type=int, default=None,
                    help="number of stories for frequency counting (default = whole subset)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap of the final stream (in nano tokens, BOS/EOS included)")
    args = ap.parse_args()
    if not args.tokenizer:
        ap.error("--tokenizer required (microdeepseek.tokenizer.json not found next to the repo)")
    os.makedirs(args.out, exist_ok=True)
    tk = tokenizers.Tokenizer.from_file(args.tokenizer)
    t0 = time.time()

    # ── pass 1: frequencies ──
    n_count = args.count_stories or args.max_stories
    print(f"[1/2] counting frequencies ({n_count or 'all'} stories) …", flush=True)
    freq = np.zeros(DS_VOCAB, dtype=np.int64)
    ntok = 0
    for i, story in enumerate(iter_stories(args.data, n_count)):
        ids = tk.encode(story, add_special_tokens=False).ids
        freq += np.bincount(np.asarray(ids, dtype=np.int64), minlength=DS_VOCAB)
        ntok += len(ids)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1} stories, {ntok / 1e6:.1f} M tokens, {time.time() - t0:.0f}s", flush=True)
    top_ds = np.argsort(-freq)[: args.top]
    seen = top_ds[freq[top_ds] > 0]
    if len(seen) < args.top:
        missing = args.top - len(seen)
        unseen = np.nonzero(freq == 0)[0][:missing]
        top_ds = np.concatenate([seen, unseen])
        print(f"  ⚠ only {len(seen)} distinct tokens seen - {missing} slots filled with unused ids", flush=True)
    top_ds.sort()
    nano_to_ds = top_ds.astype(np.int64)
    ds_to_nano = np.full(DS_VOCAB, SPECIAL_NANO["unk"], dtype=np.int64)
    ds_to_nano[nano_to_ds] = np.arange(args.top, dtype=np.int64)
    kept = freq[nano_to_ds].sum()
    print(f"  top-{args.top} coverage: {kept / max(ntok, 1) * 100:.2f}% of counted tokens", flush=True)

    # ── pass 2: stream encoding ──
    print("[2/2] encoding the corpus → tokens.bin …", flush=True)
    out_bin = os.path.join(args.out, "tokens.bin")
    total = 0
    cap = args.max_tokens
    bos, eos = SPECIAL_NANO["bos"], SPECIAL_NANO["eos"]
    with open(out_bin, "wb") as f:
        for i, story in enumerate(iter_stories(args.data, args.max_stories)):
            ids = tk.encode(story, add_special_tokens=False).ids
            seq = np.empty(len(ids) + 2, dtype=np.uint16)
            seq[0] = bos
            seq[1:-1] = ds_to_nano[np.asarray(ids, dtype=np.int64)]
            seq[-1] = eos
            if cap is not None and total + len(seq) > cap:
                seq = seq[: cap - total]
            seq.tofile(f)
            total += len(seq)
            if (i + 1) % 20000 == 0:
                print(f"  {i + 1} stories, {total / 1e6:.1f} M tokens, {time.time() - t0:.0f}s", flush=True)
            if cap is not None and total >= cap:
                break
    print(f"  total: {total / 1e6:.2f} M nano tokens", flush=True)

    # ── exports ──
    with open(os.path.join(args.out, "tokens.meta.json"), "w") as f:
        json.dump({"count": int(total), "dtype": "uint16le", "bos": bos, "eos": eos}, f)
    vocab = {
        "nano_to_ds": nano_to_ds.tolist(),
        "specials": SPECIAL_NANO,
        "vocab_size": VOCAB_NANO,
        "ds_special_ids": {"bos": 0, "eos": 1, "pad": 2},
    }
    with open(os.path.join(args.out, "vocab_ds_nano.json"), "w") as f:
        json.dump(vocab, f)
    # small readable sample for human verification
    sample = {int(i): base64.b64encode(tk.decode([int(k)]).encode()).decode()
              for i, k in enumerate(nano_to_ds[:64])}
    with open(os.path.join(args.out, "vocab_sample.json"), "w") as f:
        json.dump(sample, f, indent=1)
    print(f"→ {out_bin} ({total * 2 / 1e6:.0f} MB), vocab_ds_nano.json, tokens.meta.json", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
