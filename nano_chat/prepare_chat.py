#!/usr/bin/env python3
"""nanokimi chat — SFT corpus preparation (prepare_chat.py)

Downloads a SmolTalk-style dialogue dataset (default: HuggingFaceTB/smol-smoltalk,
the mix designed for small models), renders each dialogue in the EXACT XTML
template the Rust engine builds at inference (src/tokenizer.rs: build_chat /
message / open_tag / close_tag), tokenizes with the real Kimi BPE and remaps to
the nano vocab using the EXISTING vocab_nano.json (the remap the pretrained
checkpoint was trained with — it is loaded, never recomputed here).

Wire format per turn (nano ids), mirroring build_chat:
  user:      <|open|> enc('message role="user"') <|sep|> enc(q)
             <|close|> enc('message') <|sep|> <|end_of_msg|>
  assistant: <|open|> enc('message role="assistant"') <|sep|>
             <|open|> enc('think') <|sep|> enc(a) <|end_of_msg|>

The assistant turn deliberately stops right after the (empty) think channel:
the engine's generation prompt ends with '<|open|>think<|sep|>', so the model
learns to emit the answer text immediately, then <|end_of_msg|> (the stop
token). That keeps the decoded answer clean for display and for the chat
history. No BOS: build_chat does not prepend one at inference.

Dialogues are filtered: system messages dropped, strict user/assistant
alternation required, per-dialogue cap in nano tokens, and a max UNK rate
(the nano vocab is the TinyStories top-8192 — dialogues too far from simple
English would train the model to emit [UNK]).

usage:
  python3 prepare_chat.py --out out_chat --max-dialogues 2000          # smoke
  python3 prepare_chat.py --out out_chat --max-dialogues 60000         # full
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nano"))
from prepare import make_encoder, SPECIAL_NANO  # noqa: E402  (shared Kimi BPE encoder)

VOCAB_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "release-assets", "vocab_nano.json")

USER_TAG = 'message role="user"'
ASSISTANT_TAG = 'message role="assistant"'


def load_remap(path):
    with open(path) as f:
        v = json.load(f)
    nano_to_kimi = np.asarray(v["nano_to_kimi"], dtype=np.int64)
    kimi_to_nano = np.full(163_840, SPECIAL_NANO["unk"], dtype=np.int64)
    kimi_to_nano[nano_to_kimi] = np.arange(len(nano_to_kimi), dtype=np.int64)
    return v, kimi_to_nano


def iter_dialogues(dataset, config, split):
    from datasets import load_dataset
    ds = load_dataset(dataset, config, split=split, streaming=True)
    for item in ds:
        msgs = item.get("messages")
        if msgs:
            yield msgs


def normalize(msgs):
    """Drop system messages, require strict user/assistant alternation starting
    with user, ≥1 full turn. Returns [(q, a), ...] or None."""
    turns = []
    pending = None
    for m in msgs:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role == "system":
            continue
        if not content:
            return None
        if role == "user":
            if pending is not None:
                return None  # two user messages in a row
            pending = content
        elif role == "assistant":
            if pending is None:
                return None  # assistant without a preceding user
            turns.append((pending, content))
            pending = None
        else:
            return None
    if pending is not None or not turns:
        return None
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", default=VOCAB_DEFAULT)
    ap.add_argument("--dataset", default="HuggingFaceTB/smol-smoltalk")
    ap.add_argument("--config", default=None, help="dataset config (None = default 'all')")
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-dialogues", type=int, default=2000)
    ap.add_argument("--max-tokens", type=int, default=None, help="cap of the final stream")
    ap.add_argument("--max-len", type=int, default=1024, help="max nano tokens per dialogue")
    ap.add_argument("--max-unk-rate", type=float, default=0.08,
                    help="max UNK rate on the assistant side (nano vocab = TinyStories top-8192)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    enc = make_encoder()
    vocab, kimi_to_nano = load_remap(args.vocab)
    sp = SPECIAL_NANO
    open_id, close_id, sep_id, eom_id = sp["open"], sp["close"], sp["sep"], sp["end_of_msg"]
    unk = sp["unk"]

    # constant tag segments (encoded once): open TAG sep / close 'message' sep
    def open_tag(text):
        return [open_id] + kimi_to_nano[np.asarray(enc.encode(text), dtype=np.int64)].tolist() + [sep_id]

    USER_OPEN = open_tag(USER_TAG)
    ASSIST_OPEN = open_tag(ASSISTANT_TAG) + open_tag("think")
    MSG_CLOSE = [close_id] + kimi_to_nano[np.asarray(enc.encode("message"), dtype=np.int64)].tolist() + [sep_id]

    def enc_text(text):
        ids = kimi_to_nano[np.asarray(enc.encode(text), dtype=np.int64)]
        return ids

    out_bin = os.path.join(args.out, "tokens_chat.bin")
    t0 = time.time()
    n_seen = n_kept = n_skip_struct = n_skip_len = n_skip_unk = 0
    total = 0
    with open(out_bin, "wb") as f:
        for msgs in iter_dialogues(args.dataset, args.config, args.split):
            n_seen += 1
            if n_kept >= args.max_dialogues:
                break
            turns = normalize(msgs)
            if turns is None:
                n_skip_struct += 1
                continue
            ids = []
            n_unk = n_txt = 0
            for q, a in turns:
                q_ids, a_ids = enc_text(q), enc_text(a)
                # UNK rate measured on the ASSISTANT side only: that is the text
                # the model learns to emit (a few UNK in the user turn is OK —
                # the model just learns to read them).
                n_unk += int((a_ids == unk).sum())
                n_txt += len(a_ids)
                ids += USER_OPEN + q_ids.tolist() + MSG_CLOSE + [eom_id]
                ids += ASSIST_OPEN + a_ids.tolist() + [eom_id]
            if len(ids) > args.max_len:
                n_skip_len += 1
                continue
            if n_txt > 0 and n_unk / n_txt > args.max_unk_rate:
                n_skip_unk += 1
                continue
            if args.max_tokens is not None and total + len(ids) > args.max_tokens:
                break
            np.asarray(ids, dtype=np.uint16).tofile(f)
            total += len(ids)
            n_kept += 1
            if n_kept % 5000 == 0:
                print(f"  {n_kept} dialogues kept ({n_seen} seen), {total / 1e6:.1f} M tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)

    print(f"kept {n_kept}/{n_seen} dialogues "
          f"(skipped: {n_skip_struct} structure, {n_skip_len} length, {n_skip_unk} unk-rate)", flush=True)
    print(f"total: {total / 1e6:.2f} M nano tokens → {out_bin} ({total * 2 / 1e6:.0f} MB)", flush=True)
    with open(os.path.join(args.out, "tokens_chat.meta.json"), "w") as f:
        json.dump({"count": int(total), "dtype": "uint16le", "dialogues": n_kept,
                   "dataset": args.dataset, "config": args.config,
                   "format": "build_chat wire format (no BOS, empty think channel)",
                   "specials": sp}, f, indent=1)
    # provenance copy of the remap used (identical to the pretrained model's)
    with open(os.path.join(args.out, "vocab_chat.json"), "w") as f:
        json.dump(vocab, f)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
