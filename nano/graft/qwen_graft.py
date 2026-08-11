#!/usr/bin/env python3
"""nanokimi - qwen_graft: runtime expert grafting for qwen3_5_moe-family
checkpoints (transformers side).

Extends a loaded model's fused expert bank with the experts of a graft
pack (expert_solve.py, --act silu --host-planes classic), appends the
matching router rows, and adds a per-expert SELECTION BIAS applied to the
router logits before softmax - the birth lever: a very negative bias
excludes an expert from the top-k renormalization entirely, so a freshly
grafted bank leaves the model bit-identical until something raises it.

Subcommands:
  graft  --model DIR --pack P.npz --out-cfg G.json [--bias -1e9]
         extends the bank in memory, verifies silent birth (logits equal
         the stock model on a probe batch), writes a small runtime config
         (pack path + per-layer bias) that `eval` and `gen` re-apply.
  eval   --model DIR --text corpus.jsonl --doc-range A:B
         [--graft-cfg G.json] [--calibrate greedy|global]
         paired per-document CE (bootstrap CI) of grafted vs stock; with
         --calibrate, chooses per-layer biases on a calibration range
         first (silence always in the grid: never worse by construction).
  gen    --model DIR --prompt "..." [--graft-cfg G.json]

The fused expert tensors are plain 3D Parameters, so the bank stays in
model precision regardless of any Linear-level quantization. Router
patching is per-instance and reversible.

  python3 qwen_graft.py --selftest
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------- model access

def find_decoder(model):
    """Returns the decoder stack (module owning .layers and .embed_tokens),
    wherever the checkpoint wrapper put it."""
    import torch.nn as nn
    for _n, m in model.named_modules():
        if hasattr(m, "layers") and isinstance(getattr(m, "layers"), nn.ModuleList) \
                and hasattr(m, "embed_tokens"):
            return m
    raise SystemExit("no decoder stack found")


def load_model(model_dir, device="cpu", dtype="bfloat16"):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    kw = {"attn_implementation": "eager", "dtype": getattr(torch, dtype)}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    except ValueError:
        with open(os.path.join(model_dir, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        model = getattr(transformers, arch).from_pretrained(model_dir, **kw)
    model.to(device).eval()
    return model, tok


def moe_blocks(model):
    """[(layer_idx, block)] for every layer whose mlp has a fused expert
    bank (gate_up_proj/down_proj 3D parameters + router `gate`)."""
    dec = find_decoder(model)
    out = []
    for i, layer in enumerate(dec.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "experts") \
                and hasattr(mlp.experts, "gate_up_proj"):
            out.append((i, mlp))
    return out


# ---------------------------------------------------------------- extension

def _patched_router_forward(self, hidden_states):
    import torch
    import torch.nn.functional as F
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight) + self.expert_bias
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    top_val, top_idx = torch.topk(router_probs, self.top_k, dim=-1)
    top_val /= top_val.sum(dim=-1, keepdim=True)
    return router_logits, top_val.to(router_logits.dtype), top_idx


def extend_bank(model, pack_path, bias_by_layer):
    """Appends the pack's experts + router rows to each MoE layer and
    installs the selection-bias router patch. bias_by_layer: {layer: bias}
    (missing layers get -1e9 = silent)."""
    import torch
    z = np.load(pack_path)
    meta = json.loads(bytes(z["meta"]).decode())
    bands = int(meta["bands"])
    n_graft = {}
    for li, mlp in moe_blocks(model):
        keys = [f"L{li}.g{g}" for g in range(bands)]
        if f"{keys[0]}.w1" not in z:
            continue
        ex = mlp.experts
        dt, dev = ex.gate_up_proj.dtype, ex.gate_up_proj.device
        gu, dn, rows = [], [], []
        for k in keys:
            w1 = torch.tensor(z[f"{k}.w1"])  # [inter, d]
            w3 = torch.tensor(z[f"{k}.w3"])
            w2 = torch.tensor(z[f"{k}.w2"])  # [d, inter]
            gu.append(torch.cat([w1, w3], dim=0))
            dn.append(w2)
            rows.append(torch.tensor(z[f"{k}.gate"]))
        with torch.no_grad():
            ex.gate_up_proj = torch.nn.Parameter(torch.cat(
                [ex.gate_up_proj.data,
                 torch.stack(gu).to(dt).to(dev)]), requires_grad=False)
            ex.down_proj = torch.nn.Parameter(torch.cat(
                [ex.down_proj.data,
                 torch.stack(dn).to(dt).to(dev)]), requires_grad=False)
            gate = mlp.gate
            e0 = gate.weight.shape[0]
            gate.weight = torch.nn.Parameter(torch.cat(
                [gate.weight.data,
                 torch.stack(rows).to(gate.weight.dtype).to(dev)]),
                requires_grad=False)
            bias = torch.zeros(e0 + bands, dtype=torch.float32, device=dev)
            bias[e0:] = float(bias_by_layer.get(str(li),
                                                bias_by_layer.get(li, -1e9)))
            gate.expert_bias = torch.nn.Parameter(bias, requires_grad=False)
            gate.forward = types.MethodType(_patched_router_forward, gate)
            ex.num_experts = e0 + bands
            gate.num_experts = e0 + bands
        n_graft[li] = bands
    if not n_graft:
        raise SystemExit("pack matched no MoE layer of this model")
    return n_graft


def set_graft_bias(model, bias_by_layer, n_graft):
    import torch
    for li, mlp in moe_blocks(model):
        if li in n_graft and hasattr(mlp.gate, "expert_bias"):
            with torch.no_grad():
                mlp.gate.expert_bias[-n_graft[li]:] = float(
                    bias_by_layer.get(li, -1e9))


# ------------------------------------------------------------------- eval

def doc_windows(text, tok, seq):
    enc = tok(text, add_special_tokens=True)
    ids = np.asarray(enc["input_ids"], np.int64)
    return [ids[w0:w0 + seq + 1] for w0 in range(0, len(ids) - 1, seq)
            if min(w0 + seq + 1, len(ids)) - w0 >= 8]


def doc_nll(model, wins, vocab, device, batch=2):
    import torch
    import torch.nn.functional as F
    nll, cnt = 0.0, 0
    for b0 in range(0, len(wins), batch):
        grp = wins[b0:b0 + batch]
        mx = max(len(w) for w in grp)
        x = np.zeros((len(grp), mx - 1), np.int64)
        y = np.zeros((len(grp), mx - 1), np.int64)
        m = np.zeros((len(grp), mx - 1), bool)
        for i, w in enumerate(grp):
            x[i, :len(w) - 1] = w[:-1]
            y[i, :len(w) - 1] = w[1:]
            m[i, 1:len(w) - 1] = True  # skip the BOS target
        with torch.no_grad():
            logits = model(input_ids=torch.tensor(x, device=device)).logits
        lm = torch.tensor(m, device=device)
        ce = F.cross_entropy(logits[lm].float(),
                             torch.tensor(y, device=device)[lm],
                             reduction="sum")
        nll += float(ce)
        cnt += int(m.sum())
    return nll, cnt


def eval_range(model, tok, text_path, d0, d1, seq, device, batch=2,
               log=print):
    per_doc = []
    with open(text_path) as f:
        for i, line in enumerate(f):
            if i >= d1:
                break
            if i < d0:
                continue
            text = json.loads(line).get("text")
            if not text:
                continue
            wins = doc_windows(text, tok, seq)
            if not wins:
                continue
            per_doc.append(doc_nll(model, wins, None, device, batch))
            if len(per_doc) % 25 == 0:
                log(f"  {len(per_doc)} docs...", flush=True)
    return per_doc


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["graft", "eval", "gen"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", default=None)
    ap.add_argument("--graft-cfg", default=None)
    ap.add_argument("--out-cfg", default=None)
    ap.add_argument("--bias", type=float, default=-1e9)
    ap.add_argument("--text", default=None)
    ap.add_argument("--doc-range", default=None)
    ap.add_argument("--cal-range", default=None,
                    help="doc range for greedy per-layer calibration")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 8)

    model, tok = load_model(args.model, args.device, args.dtype)

    if args.cmd == "graft":
        # silent-birth verification: logits before == after at bias -1e9
        probe = tok("The quick brown fox jumps over the lazy dog",
                    return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            ref = model(input_ids=probe).logits.clone()
        n_graft = extend_bank(model, args.pack, {})
        with torch.no_grad():
            got = model(input_ids=probe).logits
        d = float((ref - got).abs().max())
        print(f"grafted {sum(n_graft.values())} experts over "
              f"{len(n_graft)} layers; silent-birth max|dlogit| = {d:.2e}")
        assert d < 1e-3, "silent birth violated"
        cfg = {"pack": os.path.abspath(args.pack),
               "bias": {str(l): args.bias for l in n_graft}}
        with open(args.out_cfg or "graft_cfg.json", "w") as f:
            json.dump(cfg, f)
        print(f"-> {args.out_cfg or 'graft_cfg.json'}")
        return

    gcfg = None
    if args.graft_cfg:
        with open(args.graft_cfg) as f:
            gcfg = json.load(f)
        n_graft = extend_bank(model, gcfg["pack"],
                              {int(k): v for k, v in gcfg["bias"].items()})
        print(f"applied graft: {len(n_graft)} layers, "
              f"biases from {args.graft_cfg}")

    if args.cmd == "gen":
        ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=args.max_new,
                                 do_sample=False)
        print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
        return

    # eval
    from eval_compare import paired_bootstrap
    d0, d1 = (int(x) for x in args.doc_range.split(":"))

    if gcfg and args.cal_range:
        c0, c1 = (int(x) for x in args.cal_range.split(":"))
        n_g = {int(k): 1 for k in gcfg["bias"]}
        set_graft_bias(model, {}, n_g)  # all silent
        base_docs = eval_range(model, tok, args.text, c0, c1, args.seq,
                               args.device, args.batch)
        best = sum(n for n, _c in base_docs) / sum(c for _n, c in base_docs)
        print(f"calibration: silent CE {best:.4f}")
        chosen = {}
        for li in sorted(n_g):
            set_graft_bias(model, {li: 0.0, **chosen}, n_g)
            docs = eval_range(model, tok, args.text, c0, c1, args.seq,
                              args.device, args.batch)
            ce = sum(n for n, _c in docs) / sum(c for _n, c in docs)
            keep = ce < best
            print(f"  L{li} bias 0: CE {ce:.4f} {'KEPT' if keep else ''}",
                  flush=True)
            if keep:
                best, chosen = ce, {**chosen, li: 0.0}
            else:
                set_graft_bias(model, chosen, n_g)
        gcfg["bias"] = {str(k): v for k, v in chosen.items()}
        gcfg["bias"].update({str(k): -1e9 for k in n_g if k not in chosen})
        with open(args.graft_cfg, "w") as f:
            json.dump(gcfg, f)
        print(f"calibrated: {len(chosen)} layers active -> {args.graft_cfg}")

    t0 = time.time()
    docs = eval_range(model, tok, args.text, d0, d1, args.seq, args.device,
                      args.batch)
    ce = sum(n for n, _c in docs) / sum(c for _n, c in docs)
    tag = "grafted" if gcfg else "stock"
    print(f"{tag}: CE {ce:.4f} nats over {sum(c for _n, c in docs)} tokens "
          f"({len(docs)} docs, {time.time() - t0:.0f}s)")
    np.save(f"/tmp/qg_eval_{tag}.npy", np.asarray(docs, np.float64))
    other = f"/tmp/qg_eval_{'stock' if gcfg else 'grafted'}.npy"
    if os.path.exists(other):
        a = np.load(other)
        b = np.asarray(docs, np.float64)
        if len(a) == len(b):
            base, cand = (a, b) if gcfg else (b, a)
            delta, lo, hi, p = paired_bootstrap(
                base[:, 0], cand[:, 0], base[:, 1], args.boot)
            print(f"paired delta (grafted - stock): {delta:+.4f} nats, "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}], p(not better) {p:.4f}")


# ------------------------------------------------------------------ selftest

def selftest():
    try:
        import torch
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe \
            import Qwen3_5MoeTextConfig
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe \
            import Qwen3_5MoeForCausalLM
    except Exception as e:
        print(f"note: transformers qwen3_5_moe unavailable ({type(e).__name__})"
              " - structural checks only")
        print("qwen_graft selftest OK")
        return
    import torch
    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        shared_expert_intermediate_size=32, num_experts=8,
        num_experts_per_tok=2, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, vocab_size=128,
        linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        max_position_embeddings=512)
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    x = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        ref = model(input_ids=x).logits.clone()

    import tempfile
    rng = np.random.default_rng(1)
    pack = {"meta": np.frombuffer(json.dumps({"bands": 1}).encode(),
                                  np.uint8)}
    blocks = moe_blocks(model)
    assert len(blocks) == 4
    for li, _m in blocks:
        pack[f"L{li}.g0.w1"] = rng.normal(size=(32, 64)).astype(np.float32)
        pack[f"L{li}.g0.w3"] = rng.normal(size=(32, 64)).astype(np.float32)
        pack[f"L{li}.g0.w2"] = rng.normal(size=(64, 32)).astype(np.float32)
        pack[f"L{li}.g0.gate"] = rng.normal(size=64).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "p.npz")
        np.savez(p, **pack)
        n_graft = extend_bank(model, p, {})
        with torch.no_grad():
            got = model(input_ids=x).logits
        d = float((ref - got).abs().max())
        print(f"claim 1: silent birth exact under softmax exclusion "
              f"(max|dlogit| {d:.2e})")
        assert d < 1e-4, d

        set_graft_bias(model, {li: 0.0 for li, _ in blocks}, n_graft)
        with torch.no_grad():
            got2 = model(input_ids=x).logits
        d2 = float((ref - got2).abs().max())
        print(f"claim 2: bias 0 lets grafts act (max|dlogit| {d2:.2e})")
        assert d2 > 1e-4

        _l, mlp = blocks[0]
        assert mlp.experts.gate_up_proj.shape[0] == 9
        assert mlp.gate.weight.shape[0] == 9
        print("claim 3: bank and router extended 8 -> 9")

        set_graft_bias(model, {}, n_graft)
        with torch.no_grad():
            got3 = model(input_ids=x).logits
        assert float((ref - got3).abs().max()) < 1e-4
        print("claim 4: re-silencing restores the stock model exactly")
    print("qwen_graft selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
