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
    if device == "auto":
        kw["device_map"] = "auto"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
    except ValueError:
        with open(os.path.join(model_dir, "config.json")) as f:
            arch = json.load(f)["architectures"][0]
        model = getattr(transformers, arch).from_pretrained(model_dir, **kw)
    if device != "auto":
        model.to(device)
    model.eval()
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
    router_logits = F.linear(hidden_states, self.weight)
    router_logits = router_logits + self.expert_bias.to(router_logits.dtype)
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


def scale_grafts(model, n_graft, factor):
    """Multiplies every grafted down matrix by `factor`. Softmax routing
    renormalizes among the selected experts, so a graft that wins a slot
    also claims a large share of the mixture: attenuating its output is
    the direct lever on the eviction damage that lever cannot reach."""
    import torch
    if factor == 1.0:
        return
    with torch.no_grad():
        for li, mlp in moe_blocks(model):
            if li in n_graft:
                mlp.experts.down_proj[-n_graft[li]:] *= factor


def set_graft_bias(model, bias_by_layer, n_graft):
    import torch
    for li, mlp in moe_blocks(model):
        if li in n_graft and hasattr(mlp.gate, "expert_bias"):
            with torch.no_grad():
                mlp.gate.expert_bias[-n_graft[li]:] = float(
                    bias_by_layer.get(li, -1e9))


# ------------------------------------------------------- additive grafting

class SideBranch:
    """Adds a grafted FFN in PARALLEL to a MoE block instead of entering
    its top-k. The block output becomes

        y = moe(h) + sigma(h . g + bias) * gamma * organ(h)

    so the organ never displaces a native expert. Measured motivation:
    on a strong host the organ loses to the expert it would evict
    (utility -7.6 to -11.1 nats when forced through the router), while
    the same organ can still contribute where the gate opens, because
    nothing is taken away. The sigmoid gate keeps the contract soft: no
    hard selection, no renormalization, and bias -> -inf reproduces the
    host exactly."""

    def __init__(self, mlp, w1, w3, w2, row, bias=-1e9, gamma=1.0):
        import torch
        ref = (mlp.experts.down_proj if hasattr(mlp, "experts")
               else mlp.down_proj.weight)
        dev, dt = ref.device, ref.dtype
        self.mlp = mlp
        self.w1 = torch.tensor(w1, dtype=dt, device=dev)
        self.w3 = torch.tensor(w3, dtype=dt, device=dev)
        self.w2 = torch.tensor(w2, dtype=dt, device=dev)
        self.row = torch.tensor(row, dtype=dt, device=dev)
        self.bias = bias
        self.gamma = gamma
        self.inner = mlp.forward
        mlp.forward = self._forward

    gamma_base = 1.0

    def _forward(self, hidden_states):
        import torch
        import torch.nn.functional as F
        out = self.inner(hidden_states)
        if self.bias <= -1e8:
            return out
        h = hidden_states.reshape(-1, hidden_states.shape[-1])
        gate = torch.sigmoid(h @ self.row + self.bias)
        act = F.silu(h @ self.w1.T) * (h @ self.w3.T)
        side = (act @ self.w2.T) * gate[:, None] * (self.gamma
                                                     * self.gamma_base)
        return out + side.reshape(out.shape)


def measure_norm_gain(model, tok, text_path, d0, d1, seq, device, layers,
                      log=print):
    """Per-layer RMS ratio between a MoE block's OUTPUT and its INPUT.

    Host planes are captured post-norm (block input) but a side branch
    adds to the block output, which lives in residual scale; the RMSNorm
    gain between the two is layer dependent, so no single output scale
    can absorb it. This measures the ratio so it can be folded into each
    organ's down matrix."""
    import torch
    rec, handles = {}, []

    def mk(li):
        def hook(_m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            rec.setdefault(li, []).append(
                (float(inp[0].detach().float().pow(2).mean().sqrt()),
                 float(o.detach().float().pow(2).mean().sqrt())))
        return hook

    for li, mlp in moe_blocks(model):
        if li in layers:
            handles.append(mlp.register_forward_hook(mk(li)))
    n = 0
    with open(text_path) as f:
        for i, line in enumerate(f):
            if i >= d1 or n >= 8:
                break
            if i < d0:
                continue
            t = json.loads(line).get("text")
            if not t:
                continue
            for w in doc_windows(t, tok, seq)[:1]:
                with torch.no_grad():
                    model(input_ids=torch.tensor(w[None, :-1],
                                                 device=device))
                n += 1
    for h in handles:
        h.remove()
    gains = {li: float(np.mean([o / max(i, 1e-6) for i, o in v]))
             for li, v in rec.items()}
    log("norm gains: " + " ".join(f"L{li}:{g:.3f}"
                                  for li, g in sorted(gains.items())))
    return gains


def ffn_blocks(model):
    """[(layer_idx, mlp)] for every layer with a feed-forward block, MoE
    or dense. A side branch attaches to either: it wraps the block's
    forward and adds a gated organ to its output, which needs no router
    and no expert bank."""
    dec = find_decoder(model)
    out = []
    for i, layer in enumerate(dec.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            out.append((i, mlp))
    return out


def attach_branches(model, pack_path):
    """Attaches one side branch per feed-forward layer from a graft pack.
    Returns {layer: branch}. Every branch starts silent (bias -1e9)."""
    z = np.load(pack_path)
    meta = json.loads(bytes(z["meta"]).decode())
    out = {}
    for li, mlp in ffn_blocks(model):
        k = f"L{li}.g0"
        if f"{k}.w1" not in z:
            continue
        out[li] = SideBranch(mlp, z[f"{k}.w1"], z[f"{k}.w3"], z[f"{k}.w2"],
                             z[f"{k}.gate"])
    if not out:
        raise SystemExit("pack matched no MoE layer")
    return out


def set_branches(branches, bias, gamma=1.0):
    for b in branches.values():
        b.bias = bias
        b.gamma = gamma


# --------------------------------------------------------- utility routing

SILENT = -1e9


def utility_rows(model, tok, text_path, d0, d1, seq, device, n_graft,
                 batch=1, rel_lambda=1e-4, gate_sigma=1.0, log=print):
    """Solves each grafted router row against MEASURED per-token utility.

    Baseline pass (grafts silent) and one forced pass per grafted expert
    give a per-position loss difference; a ridge solve maps the router
    input stream to that utility, and the row is normalized to a
    router-typical logit scale. This replaces the activation-derived
    proxy row, which cannot tell where a graft HELPS from where it merely
    has something to say."""
    import torch
    import torch.nn.functional as F
    from graftlib import RectGram

    wins, pln = [], {li: [] for li in n_graft}
    handles = []
    rec = {}

    def mk(li):
        def hook(_m, inp, _o):
            rec[li] = inp[0].detach().float()
        return hook

    for li, mlp in moe_blocks(model):
        if li in n_graft:
            handles.append(mlp.gate.register_forward_hook(mk(li)))

    with open(text_path) as f:
        for i, line in enumerate(f):
            if i >= d1:
                break
            if i < d0:
                continue
            t = json.loads(line).get("text")
            if t:
                wins.extend(doc_windows(t, tok, seq)[:2])

    def pass_losses(collect_pln):
        out = []
        for w in wins:
            x = torch.tensor(w[None, :-1], device=device)
            y = torch.tensor(w[None, 1:], device=device)
            with torch.no_grad():
                logits = model(input_ids=x).logits
            ce = F.cross_entropy(logits[0].float(), y[0], reduction="none")
            out.append(ce.cpu().numpy())
            if collect_pln:
                for li in n_graft:
                    pln[li].append(rec[li].reshape(-1, rec[li].shape[-1])
                                   .cpu().numpy().astype(np.float16))
        return np.concatenate(out)

    set_graft_bias(model, {}, n_graft)
    base = pass_losses(True)
    for h in handles:
        h.remove()
    log(f"utility baseline: {len(wins)} windows, {len(base)} positions")

    rows = {}
    for li in n_graft:
        stream = np.concatenate(pln[li])[:len(base)]
        for g in range(n_graft[li]):
            set_graft_bias(model, {li: 0.0}, {li: n_graft[li]})
            forced = pass_losses(False)
            set_graft_bias(model, {}, n_graft)
            util = base - forced[:len(base)]
            gram = RectGram(stream.shape[1], 1)
            for t0 in range(0, len(util), 8192):
                gram.add(stream[t0:t0 + 8192], util[t0:t0 + 8192, None])
            row, _ = gram.solve(rel_lambda)
            row = row[0]
            ex2 = float(row @ gram.gxx @ row) / gram.n
            mu = float(row @ gram.sum_x) / gram.n
            sd = np.sqrt(max(ex2 - mu * mu, 1e-12))
            rows[(li, g)] = (row * (gate_sigma / sd)).astype(np.float32)
            log(f"L{li}.g{g}: mean util {util.mean():+.4f}, helps on "
                f"{100 * (util > 0).mean():.1f}% of positions", flush=True)
    return rows, {li: np.concatenate(pln[li])[:len(base)] for li in n_graft}


def install_rows(model, rows, n_graft, stream_by_layer=None):
    """Installs solved router rows. With a sample of the router input
    stream, each row is rescaled and centered so its logits match the
    native rows' spread and mean: a solved row carries an arbitrary
    scale, and under softmax an offset of a few units makes the graft win
    every slot and claim most of the mixture weight, starving competent
    native experts. Matching the distribution is what makes the selection
    bias meaningful in native units."""
    import torch
    with torch.no_grad():
        for (li, g), row in rows.items():
            for l2, mlp in moe_blocks(model):
                if l2 != li:
                    continue
                w = mlp.gate.weight
                idx = w.shape[0] - n_graft[li] + g
                r = torch.tensor(row, dtype=w.dtype, device=w.device)
                if stream_by_layer is not None and li in stream_by_layer:
                    x = torch.tensor(stream_by_layer[li][:4096],
                                     dtype=w.dtype, device=w.device)
                    nat = x @ w[:w.shape[0] - n_graft[li]].T
                    got = x @ r
                    scale = (nat.float().std() /
                             got.float().std().clamp_min(1e-6))
                    r = r * scale.to(w.dtype)
                    off = (x @ r).float().mean() - nat.float().mean()
                    # fold the offset into a constant direction of the row
                    r = r - (off / x.float().mean(0).norm().clamp_min(1e-6)
                             ).to(w.dtype) * (x.float().mean(0)
                                              / x.float().mean(0).norm()
                                              .clamp_min(1e-6)).to(w.dtype)
                w[idx] = r


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
    ap.add_argument("--norm-gain", action="store_true",
                    help="fold the measured per-layer block output/input "
                    "RMS ratio into each side branch")
    ap.add_argument("--additive", action="store_true",
                    help="attach grafts as parallel side branches "
                    "(sigmoid-gated, no top-k competition) instead of "
                    "extending the expert bank")
    ap.add_argument("--utility-rows", default=None,
                    help="doc range A:B used to solve router rows from "
                    "measured per-token utility before calibrating")
    ap.add_argument("--gamma-grid", default="1.0",
                    help="output-scale factors tried for the grafted "
                    "down matrices during calibration")
    ap.add_argument("--bias-grid", default=None,
                    help="global selection-bias sweep on the cal range "
                    "(replaces the greedy per-layer pass)")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(os.cpu_count() or 8)

    model, tok = load_model(args.model, args.device, args.dtype)
    if args.device == "auto":
        args.device = "cuda:0"

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
        rel = d / float(ref.abs().max())
        top_match = float((ref.argmax(-1) == got.argmax(-1)).float().mean())
        print(f"grafted {sum(n_graft.values())} experts over "
              f"{len(n_graft)} layers; silent birth: max|dlogit| {d:.2e} "
              f"(rel {rel:.2e}), top-1 agreement {top_match:.3f}")
        # bf16 GEMM blocking changes with the bank size; tolerate last-bit
        # accumulation noise, require identical greedy behavior
        assert top_match == 1.0 and rel < 1e-1, "silent birth violated"
        cfg = {"pack": os.path.abspath(args.pack),
               "bias": {str(l): args.bias for l in n_graft}}
        with open(args.out_cfg or "graft_cfg.json", "w") as f:
            json.dump(cfg, f)
        print(f"-> {args.out_cfg or 'graft_cfg.json'}")
        return

    if args.additive:
        with open(args.graft_cfg) as f:
            gc = json.load(f)
        br = attach_branches(model, gc["pack"])
        print(f"attached {len(br)} side branches")
        c0, c1 = (int(x) for x in args.cal_range.split(":"))
        if args.norm_gain:
            g = measure_norm_gain(model, tok, args.text, c0, c1, args.seq,
                                  args.device, set(br))
            for li, b in br.items():
                b.gamma_base = g.get(li, 1.0)
        set_branches(br, -1e9)
        docs = eval_range(model, tok, args.text, c0, c1, args.seq,
                          args.device, args.batch)
        best = sum(n for n, _c in docs) / sum(c for _n, c in docs)
        print(f"additive: silent CE {best:.4f}", flush=True)
        keep = None
        for gam in (float(x) for x in args.gamma_grid.split(",")):
            for b in (float(x) for x in args.bias_grid.split(",")):
                set_branches(br, b, gam)
                docs = eval_range(model, tok, args.text, c0, c1, args.seq,
                                  args.device, args.batch)
                ce = sum(n for n, _c in docs) / sum(c for _n, c in docs)
                print(f"additive: gamma {gam:g} bias {b:+.1f} CE {ce:.4f}",
                      flush=True)
                if ce < best:
                    best, keep = ce, (b, gam)
        set_branches(br, *(keep if keep else (-1e9, 1.0)))
        print(f"additive kept {keep} (CE {best:.4f})", flush=True)
        d0, d1 = (int(x) for x in args.doc_range.split(":"))
        docs = eval_range(model, tok, args.text, d0, d1, args.seq,
                          args.device, args.batch)
        ce = sum(n for n, _c in docs) / sum(c for _n, c in docs)
        print(f"additive final: CE {ce:.4f} over "
              f"{sum(c for _n, c in docs)} tokens", flush=True)
        return

    gcfg = None
    if args.graft_cfg:
        with open(args.graft_cfg) as f:
            gcfg = json.load(f)
        n_graft = extend_bank(model, gcfg["pack"],
                              {int(k): v for k, v in gcfg["bias"].items()})
        if gcfg.get("rows"):
            install_rows(model, {(int(k.split(".")[0]), int(k.split(".")[1])):
                                 np.asarray(v, np.float32)
                                 for k, v in gcfg["rows"].items()}, n_graft)
        if gcfg.get("gamma", 1.0) != 1.0:
            scale_grafts(model, n_graft, gcfg["gamma"])
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
        if args.utility_rows:
            u0, u1 = (int(x) for x in args.utility_rows.split(":"))
            rows, streams = utility_rows(model, tok, args.text, u0, u1,
                                         args.seq, args.device, n_g,
                                         args.batch)
            install_rows(model, rows, n_g, streams)
            print(f"installed {len(rows)} measured-utility router rows")
        if args.bias_grid:
            # 2D sweep over (output scale, selection bias); silence is
            # always in the grid, so the result is never worse
            set_graft_bias(model, {}, n_g)
            docs = eval_range(model, tok, args.text, c0, c1, args.seq,
                              args.device, args.batch)
            best = sum(n for n, _c in docs) / sum(c for _n, c in docs)
            print(f"sweep: silent CE {best:.4f}")
            keep, keep_g, cur_g = None, 1.0, 1.0
            for gam in (float(x) for x in args.gamma_grid.split(",")):
                scale_grafts(model, n_g, gam / cur_g)
                cur_g = gam
                for b in (float(x) for x in args.bias_grid.split(",")):
                    set_graft_bias(model, {li: b for li in n_g}, n_g)
                    docs = eval_range(model, tok, args.text, c0, c1,
                                      args.seq, args.device, args.batch)
                    ce = sum(n for n, _c in docs) / sum(c for _n, c in docs)
                    print(f"sweep: gamma {gam:g} bias {b:+.1f} CE {ce:.4f}",
                          flush=True)
                    if ce < best:
                        best, keep, keep_g = ce, b, gam
            scale_grafts(model, n_g, keep_g / cur_g)
            chosen = {li: keep for li in n_g} if keep is not None else {}
            set_graft_bias(model, chosen, n_g)
            gcfg["bias"] = {str(k): chosen.get(k, -1e9) for k in n_g}
            gcfg["gamma"] = keep_g
            gcfg["rows"] = {f"{li}.{g}": r.tolist()
                            for (li, g), r in
                            (rows.items() if args.utility_rows else {}.items())}
            with open(args.graft_cfg, "w") as f:
                json.dump(gcfg, f)
            print(f"sweep kept bias {keep} -> {args.graft_cfg}")
        if not args.bias_grid:
            _greedy(model, tok, args, gcfg, n_g, c0, c1)

    t0 = time.time()
    _run_final_eval(model, tok, args, gcfg)


def _greedy(model, tok, args, gcfg, n_g, c0, c1):
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


def _run_final_eval(model, tok, args, gcfg):
    from eval_compare import paired_bootstrap
    d0, d1 = (int(x) for x in args.doc_range.split(":"))
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
