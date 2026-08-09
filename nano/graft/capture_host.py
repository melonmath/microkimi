#!/usr/bin/env python3
"""nanokimi - capture_host: per-MoE-layer activation capture of a microkimi
.bin, byte-anchored to raw text.

Loads a .bin (bin2pt + model_nano, materialized fp32 - sized for nano-class
models; use the streamed healer machinery for full-size ones), runs it over
a corpus of documents (jsonl with a "text" field) and records, for each MoE
layer:

  - the router input stream (post-attention-layernorm hidden), plane
    L{l}.pln [hidden]
  - the latent expert stream (routed_expert_down_proj output), plane
    L{l}.lat [routed_hidden]

plus per-token byte end offsets and a validity mask (BOS and UNK positions
masked out). These tokenizer-independent anchors permit position pairing
across activation streams that use different tokenizers.

Tokenization: the K3 tiktoken model (--tiktoken, default the usual
locations), optionally remapped through a vocab_nano.json (--vocab-nano)
for models with a remapped vocabulary; out-of-vocab positions become UNK
and are masked.

usage:
  python3 capture_host.py --bin model.bin --text corpus.jsonl \
      --out cap/host --seq 512 --max-tokens 500000 [--vocab-nano v.json]
  python3 capture_host.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)
from graftlib import CaptureWriter, pack_ends  # noqa: E402
from capture_donor import iter_docs  # noqa: E402


def moe_layers_of(cfg):
    n = cfg["n_layers"]
    v = cfg.get("dense_layers")  # may be null in the embedded config
    dense = set(v if v is not None else range(cfg.get("first_k_dense", 0)))
    return [l for l in range(n) if l not in dense]


class HostTaps:
    """Hooks on each MoE layer: routed_expert_down_proj's input is the
    router/post-attention-layernorm stream and its output the latent
    stream; routed_expert_norm's input is the latent output of the routed
    expert mix (what the bank already produces, before norm + up_proj)."""

    def __init__(self, model, layers):
        self.rec = {}
        self.handles = []
        for l in layers:
            blk = model.layers[l].block_sparse_moe
            self.handles.append(
                blk.routed_expert_down_proj.register_forward_hook(
                    self._mk(l)))
            norm = getattr(blk, "routed_expert_norm", None)
            if norm is not None:
                self.handles.append(
                    norm.register_forward_hook(self._mk_moe(l)))

    def _mk(self, l):
        def hook(_mod, inp, out):
            self.rec[f"L{l}.pln"] = inp[0].detach().float().cpu()
            self.rec[f"L{l}.lat"] = out.detach().float().cpu()
        return hook

    def _mk_moe(self, l):
        def hook(_mod, inp, _out):
            self.rec[f"L{l}.moe"] = inp[0].detach().float().cpu() \
                .reshape(-1, inp[0].shape[-1])
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def make_host_encoder(tiktoken_path, vocab_nano=None):
    """Returns (encode(text) -> (ids, byte_ends, valid), bos_id).
    With vocab_nano, kimi ids are remapped and OOV becomes UNK (masked)."""
    if tiktoken_path:
        os.environ["KIMI_TIKTOKEN"] = tiktoken_path
    from prepare import make_encoder  # reads KIMI_TIKTOKEN
    enc = make_encoder()
    remap = None
    unk = bos = None
    if vocab_nano:
        with open(vocab_nano) as f:
            v = json.load(f)
        remap = {k: i for i, k in enumerate(v["nano_to_kimi"])}
        unk = v["specials"]["unk"]
        bos = v["specials"]["bos"]

    def encode(text):
        kimi = enc.encode_ordinary(text)
        lens = np.fromiter((len(b) for b in enc.decode_tokens_bytes(kimi)),
                           np.int64, count=len(kimi))
        ends = np.cumsum(lens)
        if remap is not None:
            ids = np.fromiter((remap.get(t, unk) for t in kimi), np.int64,
                              count=len(kimi))
            valid = ids != unk
        else:
            ids = np.asarray(kimi, np.int64)
            valid = np.ones(len(ids), bool)
        return ids, ends, valid

    return encode, bos


def run_capture(model, layers, docs, encode, bos_id, out_prefix, seq,
                max_tokens, extra_meta, device="cpu"):
    import torch
    taps = HostTaps(model, layers)
    writer = None
    total = 0
    for doc_idx, text in docs:
        ids, ends, valid = encode(text)
        if bos_id is not None:
            ids = np.concatenate([[bos_id], ids])
            ends = np.concatenate([[0], ends])
            valid = np.concatenate([[False], valid])
        for w0 in range(0, len(ids), seq):
            w1 = min(w0 + seq, len(ids))
            if w1 - w0 < 2:
                continue
            window = torch.tensor(ids[w0:w1], dtype=torch.long,
                                  device=device).unsqueeze(0)
            with torch.no_grad():
                model(window)
            if writer is None:
                dims = {k: v.shape[-1] for k, v in taps.rec.items()}
                writer = CaptureWriter(out_prefix, dims, extra_meta)
            rows = {k: taps.rec[k].numpy() for k in writer.plane_dims}
            writer.add(pack_ends(doc_idx, ends[w0:w1]), valid[w0:w1], rows)
            total += w1 - w0
            if max_tokens and total >= max_tokens:
                break
        if max_tokens and total >= max_tokens:
            break
    taps.close()
    if writer is None:
        raise SystemExit("no tokens captured")
    return writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--text", required=True, help="jsonl with a text field")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--layers", default="moe",
                    help="'moe' (all MoE layers) or comma-separated indices")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiktoken", default=None)
    ap.add_argument("--vocab-nano", default=None)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    from bin2pt import convert
    from model_nano import NanoModel

    sd, cfg = convert(args.bin)
    model = NanoModel(cfg)
    model.load_state_dict(sd)
    model.to(args.device).eval()

    layers = (moe_layers_of(cfg) if args.layers == "moe"
              else [int(x) for x in args.layers.split(",")])
    encode, bos = make_host_encoder(args.tiktoken, args.vocab_nano)
    if bos is None:
        bos = cfg.get("specials", {}).get("bos") if "specials" in cfg else None

    extra = {"kind": "host", "bin": os.path.abspath(args.bin),
             "layers": layers, "seq": args.seq,
             "hidden": cfg["hidden"], "routed_hidden": cfg["routed_hidden"],
             "moe_inter": cfg["moe_inter"], "n_experts": cfg["n_experts"],
             "top_k": cfg["top_k"]}
    meta = run_capture(model, layers, iter_docs(args.text, args.max_docs),
                       encode, bos, args.out, args.seq, args.max_tokens,
                       extra, args.device)
    print(f"-> {args.out}: {meta['n_tokens']} tokens, planes "
          f"{sorted(meta['planes'])}")


# ------------------------------------------------------------------ selftest

TINY = {
    "n_layers": 4, "hidden": 64, "vocab": 256, "n_experts": 8, "top_k": 2,
    "n_shared": 1, "kda_heads": 2, "kda_dim": 16, "kda_conv": 4,
    "kda_fa_rank": 16, "gate_lower_bound": -5.0, "mla_heads": 2,
    "mla_q_lora": 16, "mla_kv_lora": 16, "mla_nope": 16, "mla_rope": 8,
    "mla_v": 16, "routed_hidden": 32, "moe_inter": 32, "shared_inter": 32,
    "dense_inter": 64, "attn_res_block": 4, "first_k_dense": 1,
    "rms_eps": 1e-5,
}


def selftest():
    import tempfile
    import torch

    from graftlib import open_capture
    from model_nano import NanoModel

    torch.manual_seed(5)
    model = NanoModel(dict(TINY)).eval()
    layers = moe_layers_of(TINY)
    assert layers == [1, 2, 3]

    def encode(text):
        ids = np.frombuffer(text.encode(), np.uint8).astype(np.int64) % 256
        ends = np.arange(1, len(ids) + 1, dtype=np.int64)
        return ids, ends, np.ones(len(ids), bool)

    docs = [(0, "abcdefghij"), (1, "klmnop")]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "host")
        meta = run_capture(model, layers, docs, encode, bos_id=7,
                           out_prefix=p, seq=6, max_tokens=0,
                           extra_meta={"kind": "host"})
        # doc0: bos+10 tokens = 11 -> windows 6+5; doc1: bos+6 = 7 -> 6+... 1
        # (window of 1 dropped)
        assert meta["n_tokens"] == 17, meta
        print("claim 1: windowing + BOS bookkeeping "
              f"({meta['n_tokens']} tokens)")

        _m, ends, mask, planes = open_capture(p)
        assert set(planes) == {f"L{l}.{k}" for l in layers
                               for k in ("pln", "lat", "moe")}
        assert planes["L1.pln"].shape == (17, 64)
        assert planes["L1.lat"].shape == (17, 32)
        assert planes["L1.moe"].shape == (17, 32)
        assert not mask[0] and mask[1]  # BOS masked
        print("claim 2: planes have router-input, latent and MoE-mix "
              "widths, BOS masked")

        # latent plane must equal down_proj(router plane) exactly
        w = model.layers[1].block_sparse_moe.routed_expert_down_proj
        pln = torch.tensor(np.asarray(planes["L1.pln"], np.float32))
        lat = np.asarray(planes["L1.lat"], np.float32)
        ref = w(pln).detach().numpy()
        assert np.allclose(lat, ref, atol=2e-3), np.abs(lat - ref).max()
        print("claim 3: lat == routed_expert_down_proj(pln)")

    print("capture_host selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
