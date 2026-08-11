#!/usr/bin/env python3
"""nanokimi - capture_donor: per-layer FFN activations of an external
transformers checkpoint, byte-anchored to raw text.

Runs a causal LM over a corpus of documents (jsonl with a "text" field)
and records, for each requested decoder layer:

  - the FFN input stream (output of the pre-FFN norm), plane L{l}.in
  - the FFN residual delta (what the FFN adds to the residual stream,
    including any post-FFN norm the architecture applies), plane L{l}.dz

plus, per token, the byte offset in the document where the token ends and
a validity mask (special tokens are masked out). Byte ends are what makes
this capture pairable with a capture of ANY other model over the same
text, whatever its tokenizer (see graftlib.match_anchors).

Documents are processed in independent non-overlapping windows of --seq
tokens; requires the `transformers` package (this tool is the only one in
the directory that does).

Module hook points per architecture style (--style, default auto from
config.json model_type):
  llama (also qwen*, mistral): in = model.layers.{l}.post_attention_layernorm
                               dz = model.layers.{l}.mlp
  gemma3:                      in = model.layers.{l}.pre_feedforward_layernorm
                               dz = model.layers.{l}.post_feedforward_layernorm

usage:
  python3 capture_donor.py --model /path/to/checkpoint --text corpus.jsonl \
      --layers 6,12,18 --out cap/donor --seq 512 --device cuda \
      --max-tokens 500000
  python3 capture_donor.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graftlib import CaptureWriter, char_to_byte_ends, pack_ends  # noqa: E402

STYLES = {
    "llama": {
        "in": "model.layers.{l}.post_attention_layernorm",
        "dz": "model.layers.{l}.mlp",
        "weights": {
            "gate": "model.layers.{l}.mlp.gate_proj.weight",
            "up": "model.layers.{l}.mlp.up_proj.weight",
            "down": "model.layers.{l}.mlp.down_proj.weight",
        },
    },
    "gemma3": {
        "in": "model.layers.{l}.pre_feedforward_layernorm",
        "dz": "model.layers.{l}.post_feedforward_layernorm",
        "weights": {
            "gate": "model.layers.{l}.mlp.gate_proj.weight",
            "up": "model.layers.{l}.mlp.up_proj.weight",
            "down": "model.layers.{l}.mlp.down_proj.weight",
        },
    },
    # MoE decoders (qwen3_5_moe family): "in" is the expert/router input
    # stream, "dz" the whole MoE block output (routed mix + shared).
    # Wrapped multimodal checkpoints expose the text stack under
    # model.language_model (handled by --module-prefix).
    "qwen3_5_moe": {
        "in": "model.layers.{l}.post_attention_layernorm",
        "dz": "model.layers.{l}.mlp",
        "weights": {
            "gate_up": "model.layers.{l}.mlp.experts.gate_up_proj",
            "down": "model.layers.{l}.mlp.experts.down_proj",
        },
    },
    # dense always-on shared expert of a qwen3_5_moe checkpoint as the
    # donor FFN (same in/dz planes as the block)
    "qwen3_5_moe_shared": {
        "in": "model.layers.{l}.post_attention_layernorm",
        "dz": "model.layers.{l}.mlp",
        "weights": {
            "gate": "model.layers.{l}.mlp.shared_expert.gate_proj.weight",
            "up": "model.layers.{l}.mlp.shared_expert.up_proj.weight",
            "down": "model.layers.{l}.mlp.shared_expert.down_proj.weight",
        },
    },
    "deepseek_v4": {
        "in": "model.layers.{l}.post_attention_layernorm",
        "dz": "model.layers.{l}.mlp",
        "weights": {
            "gate_up": "model.layers.{l}.mlp.experts.gate_up_proj",
            "down": "model.layers.{l}.mlp.experts.down_proj",
        },
    },
}
_MODEL_TYPE_STYLE = {
    "llama": "llama", "qwen2": "llama", "qwen3": "llama", "mistral": "llama",
    "gemma3": "gemma3", "gemma3_text": "gemma3",
    "qwen3_5_moe": "qwen3_5_moe", "qwen3_5_moe_text": "qwen3_5_moe",
    "deepseek_v4": "deepseek_v4",
}


def detect_style(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    mt = cfg.get("model_type", "")
    if mt not in _MODEL_TYPE_STYLE:
        raise SystemExit(f"unknown model_type {mt!r}: pass --style explicitly")
    return _MODEL_TYPE_STYLE[mt]


def _reroot(tmpl, root):
    """Points a 'model.layers.{l}...' template at another decoder root."""
    return tmpl.replace("model.", root + ".", 1) if root != "model" else tmpl


def get_module(model, path):
    m = model
    for part in path.split("."):
        m = getattr(m, part)
    return m


class Taps:
    """Forward hooks recording the OUTPUT of one module per (layer, plane)."""

    def __init__(self, model, layers, style):
        self.rec = {}
        self.handles = []
        for l in layers:
            for plane, tmpl in (("in", style["in"]), ("dz", style["dz"])):
                mod = get_module(model, tmpl.format(l=l))
                self.handles.append(mod.register_forward_hook(
                    self._mk(f"L{l}.{plane}")))

    def _mk(self, key):
        def hook(_mod, _inp, out):
            if isinstance(out, tuple):
                out = out[0]
            self.rec[key] = out.detach().to("cpu", dtype=None).float()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def iter_docs(path, max_docs=None):
    with open(path) as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                return
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text")
            if text:
                yield i, text


def run_capture(model, layers, style, docs, encode, out_prefix, seq,
                max_tokens, extra_meta, device="cpu"):
    """Capture core, testable with any model-like object. `encode` maps a
    document text to (ids [n], byte_ends [n], valid [n])."""
    import torch
    taps = Taps(model, layers, style)
    dims = None
    writer = None
    total = 0
    for doc_idx, text in docs:
        ids, ends, valid = encode(text)
        for w0 in range(0, len(ids), seq):
            w1 = min(w0 + seq, len(ids))
            if w1 - w0 < 2:
                continue
            window = torch.tensor(ids[w0:w1], dtype=torch.long,
                                  device=device).unsqueeze(0)
            with torch.no_grad():
                model(window)
            if dims is None:
                dims = {k: v.shape[-1] for k, v in taps.rec.items()}
                writer = CaptureWriter(out_prefix, dims, extra_meta)
            rows = {k: taps.rec[k][0].numpy() for k in dims}
            writer.add(pack_ends(doc_idx, ends[w0:w1]), valid[w0:w1], rows)
            total += w1 - w0
            if max_tokens and total >= max_tokens:
                break
        if max_tokens and total >= max_tokens:
            break
    taps.close()
    if writer is None:
        raise SystemExit("no tokens captured")
    meta = writer.close()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--text", required=True, help="jsonl with a text field")
    ap.add_argument("--layers", required=True,
                    help="comma-separated decoder layer indices")
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--style", default="auto", choices=["auto", *STYLES])
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--module-root", default="model",
                    help="root of the decoder stack in the module tree "
                    "(wrapped multimodal checkpoints: model.language_model)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="quantized load (bitsandbytes); activations stay "
                    "full precision")
    args = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    style_name = (detect_style(args.model) if args.style == "auto"
                  else args.style)
    style = {
        "in": _reroot(STYLES[style_name]["in"], args.module_root),
        "dz": _reroot(STYLES[style_name]["dz"], args.module_root),
        "weights": STYLES[style_name]["weights"],
    }
    layers = [int(x) for x in args.layers.split(",")]

    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {"attn_implementation": "eager"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = "auto" if args.device == "auto" else args.device
    else:
        kw["dtype"] = getattr(torch, args.dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    except ValueError:
        import json as _json
        with open(os.path.join(args.model, "config.json")) as f:
            arch = _json.load(f)["architectures"][0]
        model = getattr(transformers, arch).from_pretrained(args.model, **kw)
    if not args.load_4bit:
        model.to(args.device)
    model.eval()

    special_ids = set(tok.all_special_ids)

    def encode(text):
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
        ids = np.asarray(enc["input_ids"], np.int64)
        offs = np.asarray(enc["offset_mapping"], np.int64)
        valid = np.array([i not in special_ids for i in ids], bool)
        valid &= offs[:, 1] > offs[:, 0]
        ends = np.zeros(len(ids), np.int64)
        ends[valid] = char_to_byte_ends(text, offs[valid, 1])
        return ids, ends, valid

    extra = {
        "kind": "donor", "model": os.path.abspath(args.model),
        "style": style_name, "layers": layers, "seq": args.seq,
        "weights": {k: v for k, v in style["weights"].items()},
    }
    meta = run_capture(model, layers, style,
                       iter_docs(args.text, args.max_docs), encode,
                       args.out, args.seq, args.max_tokens, extra,
                       args.device)
    print(f"-> {args.out}: {meta['n_tokens']} tokens, planes "
          f"{sorted(meta['planes'])}")


# ------------------------------------------------------------------ selftest

def selftest():
    """Runs the capture core against a stub model (no transformers needed):
    checks hook routing, windowing, byte anchoring and the written layout."""
    import tempfile
    import torch
    from torch import nn

    from graftlib import open_capture

    d = 8

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.post_attention_layernorm = nn.LayerNorm(d)
            self.mlp = nn.Linear(d, d)

        def forward(self, h):
            return h + self.mlp(self.post_attention_layernorm(h))

    class Stub(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Layer() for _ in range(3)])
            self.emb = nn.Embedding(64, d)

        def forward(self, ids):
            h = self.emb(ids)
            for l in self.model.layers:
                h = l(h)
            return h

    torch.manual_seed(3)
    stub = Stub().eval()

    def encode(text):
        # one token per word, ends at each word's byte end
        words = text.split(" ")
        ids, ends, pos = [], [], 0
        for w in words:
            pos += len(w.encode()) + 1
            ids.append(hash(w) % 64)
            ends.append(pos - 1)
        return (np.asarray(ids, np.int64), np.asarray(ends, np.int64),
                np.ones(len(ids), bool))

    docs = [(0, "a bb ccc dd e ff"), (1, "gg hh iii")]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "don")
        meta = run_capture(stub, [0, 2], STYLES["llama"], docs, encode, p,
                           seq=4, max_tokens=0, extra_meta={"kind": "donor"})
        assert meta["n_tokens"] == 9, meta
        assert set(meta["planes"]) == {"L0.in", "L0.dz", "L2.in", "L2.dz"}
        print("claim 1: hooks captured both planes for both layers, "
              "windows of 4 covered 6+3 tokens")

        _meta, ends, mask, planes = open_capture(p)
        assert mask.all()
        # doc 0: "a bb ccc dd" -> ends 1,4,8,11 then window 2 "e ff"
        assert (ends[:4] >> np.uint64(40) == 0).all()
        assert (ends[6:] >> np.uint64(40) == 1).all()
        print("claim 2: byte ends packed per document")

        # the captured planes must equal a direct forward's intermediates
        ids = torch.tensor([[hash(w) % 64 for w in "a bb ccc dd".split()]])
        h = stub.emb(ids)
        ref_in = stub.model.layers[0].post_attention_layernorm(h)
        got = np.asarray(planes["L0.in"][:4], np.float32)
        assert np.allclose(got, ref_in[0].detach().numpy(), atol=1e-3)
        print("claim 3: captured plane matches a direct forward")

    print("capture_donor selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
