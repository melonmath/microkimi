#!/usr/bin/env python3
"""nanokimi - corpus preparation (prepare.py)

Tokenizes the TinyStories corpus with the REAL Kimi tokenizer (tiktoken),
builds a remap towards the top-N most frequent tokens (+ specials),
and writes:
  - tokens.bin      : stream of nano ids in uint16 LE (BOS/EOS per story)
  - tokens.meta.json: {count, dtype, bos, eos}
  - vocab_nano.json : remap table exported for the Rust engine
      { nano_to_kimi: [kimi_id × N], specials: {bos,eos,open,close,sep,end_of_msg,unk,pad} }

usage:
  python3 prepare.py --data ~/data --out ~/nano_out --top 8192 \
      [--max-stories K] [--max-tokens M] [--count-stories K2]
  python3 prepare.py --data slice.json --json-repair --out ./out --max-stories 3000

--json-repair : tolerates a truncated JSON file (cuts at the last complete object).
"""
import argparse
import base64
import json
import os
import sys
import time

import numpy as np
import tiktoken
from tiktoken.load import load_tiktoken_bpe

# Official pat_str of the Kimi K3 tokenizer (copied verbatim from
# ref/tokenization_kimi.py:54-63, Moonshot AI - avoids depending on the file).
PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])

TIKTOKEN_MODEL = os.environ.get(
    "KIMI_TIKTOKEN",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ref", "tiktoken.model"),
)
if not os.path.exists(TIKTOKEN_MODEL):
    TIKTOKEN_MODEL = os.path.expanduser("~/tiktoken.model")

SPECIAL_NANO = {
    "bos": 8192, "eos": 8193, "open": 8194, "close": 8195,
    "sep": 8196, "end_of_msg": 8197, "unk": 8198, "pad": 8199,
}
VOCAB_NANO = 8200


def make_encoder():
    mergeable = load_tiktoken_bpe(TIKTOKEN_MODEL)
    return tiktoken.Encoding(
        name="kimi", pat_str=PAT_STR,
        mergeable_ranks=mergeable, special_tokens={},
    )


def iter_stories(path, json_repair=False, max_stories=None):
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
            text = f.read()
        if json_repair:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                cut = text.rfind("}")
                data = json.loads(text[:cut + 1] + "]")
        else:
            data = json.loads(text)
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
    ap.add_argument("--top", type=int, default=8192)
    ap.add_argument("--max-stories", type=int, default=None)
    ap.add_argument("--count-stories", type=int, default=None,
                    help="number of stories for frequency counting (default = whole subset)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap of the final stream (in nano tokens, BOS/EOS included)")
    ap.add_argument("--json-repair", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    enc = make_encoder()
    t0 = time.time()

    # ── pass 1: frequencies ──
    n_count = args.count_stories or args.max_stories
    print(f"[1/2] counting frequencies ({n_count or 'all'} stories) …", flush=True)
    freq = np.zeros(163_584, dtype=np.int64)
    ntok = 0
    for i, story in enumerate(iter_stories(args.data, args.json_repair, n_count)):
        ids = enc.encode(story, disallowed_special=())
        freq += np.bincount(np.asarray(ids, dtype=np.int64), minlength=163_584)
        ntok += len(ids)
        if (i + 1) % 20000 == 0:
            print(f"  {i + 1} stories, {ntok / 1e6:.1f} M tokens, {time.time() - t0:.0f}s", flush=True)
    top_kimi = np.argsort(-freq)[: args.top]
    seen = top_kimi[freq[top_kimi] > 0]
    if len(seen) < args.top:
        # corpus too small to fill the top: pad with the smallest
        # Kimi ids never seen (consistent slots, simply never emitted)
        missing = args.top - len(seen)
        unseen = np.nonzero(freq == 0)[0][:missing]
        top_kimi = np.concatenate([seen, unseen])
        print(f"  ⚠ only {len(seen)} distinct tokens seen - {missing} slots filled with unused ids", flush=True)
    top_kimi.sort()
    nano_to_kimi = top_kimi.astype(np.int64)
    kimi_to_nano = np.full(163_584, SPECIAL_NANO["unk"], dtype=np.int64)
    kimi_to_nano[nano_to_kimi] = np.arange(args.top, dtype=np.int64)
    kept = freq[nano_to_kimi].sum()
    print(f"  top-{args.top} coverage: {kept / max(ntok, 1) * 100:.2f}% of counted tokens", flush=True)

    # ── pass 2: stream encoding ──
    print("[2/2] encoding the corpus → tokens.bin …", flush=True)
    out_bin = os.path.join(args.out, "tokens.bin")
    total = 0
    cap = args.max_tokens
    bos, eos = SPECIAL_NANO["bos"], SPECIAL_NANO["eos"]
    with open(out_bin, "wb") as f:
        for i, story in enumerate(iter_stories(args.data, args.json_repair, args.max_stories)):
            ids = enc.encode(story, disallowed_special=())
            seq = np.empty(len(ids) + 2, dtype=np.uint16)
            seq[0] = bos
            seq[1:-1] = kimi_to_nano[np.asarray(ids, dtype=np.int64)]
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
        "nano_to_kimi": nano_to_kimi.tolist(),
        "specials": SPECIAL_NANO,
        "vocab_size": VOCAB_NANO,
        "kimi_special_ids": {  # ids of the markers in the full Kimi vocab (for encoding on the Rust side)
            "open": 163587, "close": 163588, "sep": 163589, "end_of_msg": 163586,
        },
    }
    with open(os.path.join(args.out, "vocab_nano.json"), "w") as f:
        json.dump(vocab, f)
    # small readable sample for human verification
    sample = {int(i): base64.b64encode(enc.decode_single_token_bytes(int(k))).decode()
              for i, k in enumerate(nano_to_kimi[:64])}
    with open(os.path.join(args.out, "vocab_sample.json"), "w") as f:
        json.dump(sample, f, indent=1)
    print(f"→ {out_bin} ({total * 2 / 1e6:.0f} MB), vocab_nano.json, tokens.meta.json", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
