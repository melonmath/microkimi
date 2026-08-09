#!/usr/bin/env python3
"""nanokimi - graft_heal: integrate freshly injected experts by training
ONLY them, host frozen.

Newly added experts may approximate an external FFN without having been
trained against the host's actual loss, and their router rows may be a
closed-form estimate. This tool runs a short SGD pass where the only trainable
parameters are, per MoE layer:

  - the grafted rows of block_sparse_moe.gate.weight (row-masked grad);
  - the grafted experts' w1/w3/w2.

Everything else (trunk, attention, norms, original experts and router
rows) is frozen, so the host cannot drift: in the worst case the router
learns to never pick the grafts and the model returns to its baseline.
The e_score_correction_bias of the grafted experts is set to --heal-bias
during training (they must be selectable for gradients to flow) and
written as --final-bias in the output.

The output .bin is a byte copy of the input with only the touched
tensors rewritten in place (same sizes: gate/bias are fp32, experts are
re-quantized MXFP4), so everything else stays byte-identical.

usage:
  python3 graft_heal.py --bin grafted.bin --text corpus.jsonl \
      --out healed.bin --steps 300 --device cuda \
      [--vocab-nano v.json] [--tiktoken path] [--heal-bias 0] \
      [--final-bias 0] [--lr 1e-3] [--batch 4] [--seq 512]
  python3 graft_heal.py --selftest
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)
from bin2pt import read_bin, convert  # noqa: E402
from export import quantize_mxfp4  # noqa: E402
from capture_host import moe_layers_of  # noqa: E402


# ------------------------------------------------------------- param surgery

def freeze_except_grafts(model, moe, e0):
    """Freezes every parameter, then re-enables the grafted expert weights
    and the grafted rows of each gate (via a row-masking gradient hook).
    Returns the list of trainable parameters."""
    import torch
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for l in moe:
        blk = model.layers[l].block_sparse_moe
        n_e = len(blk.experts)
        gw = blk.gate.weight
        gw.requires_grad_(True)
        mask = torch.zeros(n_e, 1, device=gw.device, dtype=gw.dtype)
        mask[e0:] = 1.0
        gw.register_hook(lambda g, m=mask: g * m)
        trainable.append(gw)
        for e in range(e0, n_e):
            for mod in (blk.experts[e].w1, blk.experts[e].w2,
                        blk.experts[e].w3):
                mod.weight.requires_grad_(True)
                trainable.append(mod.weight)
    return trainable


def set_graft_bias(model, moe, e0, value):
    import torch
    with torch.no_grad():
        for l in moe:
            model.layers[l].block_sparse_moe.gate \
                 .e_score_correction_bias[e0:] = value


# ------------------------------------------------------------------ batching

def make_windows(docs, encode, bos, seq, max_tokens, unk=None):
    """Tokenizes documents into full-length windows: (ids [n, seq+1],
    mask [n, seq]) where mask marks target positions that count in the
    loss (drops BOS/UNK/invalid targets)."""
    xs, ms = [], []
    total = 0
    for _i, text in docs:
        ids, _ends, valid = encode(text)
        if bos is not None:
            ids = np.concatenate([[bos], ids])
            valid = np.concatenate([[False], valid])
        for w0 in range(0, len(ids) - seq - 1, seq):
            xs.append(ids[w0:w0 + seq + 1])
            m = valid[w0 + 1:w0 + seq + 1].copy()
            if unk is not None:
                m &= ids[w0 + 1:w0 + seq + 1] != unk
            ms.append(m)
            total += seq
            if max_tokens and total >= max_tokens:
                break
        if max_tokens and total >= max_tokens:
            break
    if not xs:
        raise SystemExit("corpus produced no full windows")
    return np.asarray(xs, np.int64), np.asarray(ms, bool)


def ce_eval(model, xs, ms, vocab, device, batch=4):
    import torch
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for b0 in range(0, len(xs), batch):
            x = torch.tensor(xs[b0:b0 + batch, :-1], device=device)
            y = torch.tensor(xs[b0:b0 + batch, 1:], device=device)
            m = torch.tensor(ms[b0:b0 + batch], device=device)
            logits = model(x).view(x.shape[0], x.shape[1], vocab)
            ce = torch.nn.functional.cross_entropy(
                logits[m], y[m], reduction="sum")
            nll += float(ce)
            cnt += int(m.sum())
    return nll / max(cnt, 1), cnt


def train(model, xs, ms, vocab, device, steps, lr, warmup, clip, batch,
          trainable, seed=1234, log=print, log_every=20):
    import torch
    opt = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.0)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        idx = rng.integers(0, len(xs), batch)
        x = torch.tensor(xs[idx, :-1], device=device)
        y = torch.tensor(xs[idx, 1:], device=device)
        m = torch.tensor(ms[idx], device=device)
        logits = model(x).view(batch, -1, vocab)
        loss = torch.nn.functional.cross_entropy(logits[m], y[m])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, clip)
        s = (step + 1) / max(warmup, 1)
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, s)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            log(f"  step {step}: loss {float(loss.detach()):.4f}", flush=True)
    return model


# ------------------------------------------------------------------ patching

def patch_bin(src, dst, updates):
    """Copies src to dst, then rewrites the given tensors in place.
    updates: {name: float32 array}. fp32 entries must match dims; MXFP4
    entries are re-quantized (same blob size by construction)."""
    config, entries, f = read_bin(src)
    f.close()
    shutil.copyfile(src, dst)
    by_name = {n: (dt, dims, off, size) for n, dt, dims, off, size in entries}
    with open(dst, "r+b") as out:
        for name, w in updates.items():
            dt, dims, off, size = by_name[name]
            w = np.ascontiguousarray(w, np.float32)
            assert list(w.shape) == list(dims), (name, w.shape, dims)
            if dt == 0:
                blob = w.tobytes()
            elif dt == 1:
                p, s = quantize_mxfp4(w)
                blob = p.tobytes() + s.tobytes()
            else:
                raise SystemExit(f"{name}: unsupported dtype {dt}")
            assert len(blob) == size, (name, len(blob), size)
            out.seek(off)
            out.write(blob)
    return len(updates)


def collect_updates(model, config, moe, e0, final_bias):
    import torch
    upd = {}
    with torch.no_grad():
        for l in moe:
            blk = model.layers[l].block_sparse_moe
            m = f"layers.{l}.block_sparse_moe."
            upd[m + "gate.weight"] = blk.gate.weight.detach().cpu().numpy()
            bias = blk.gate.e_score_correction_bias.detach().cpu() \
                      .numpy().copy()
            bias[e0:] = final_bias
            upd[m + "gate.e_score_correction_bias"] = bias
            for e in range(e0, len(blk.experts)):
                for nm in ("w1", "w2", "w3"):
                    w = getattr(blk.experts[e], nm).weight
                    upd[m + f"experts.{e}.{nm}"] = w.detach().cpu().numpy()
    return upd


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--text", required=True, help="jsonl with a text field")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-experts", type=int, default=None,
                    help="pre-graft bank size (default: the "
                    "graft_base_experts config key)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiktoken", default=None)
    ap.add_argument("--vocab-nano", default=None)
    ap.add_argument("--max-tokens", type=int, default=500_000)
    ap.add_argument("--holdout-windows", type=int, default=32)
    ap.add_argument("--heal-bias", type=float, default=0.0)
    ap.add_argument("--final-bias", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    import torch
    from model_nano import NanoModel
    from capture_host import make_host_encoder
    from capture_donor import iter_docs

    sd, cfg = convert(args.bin)
    config, _entries, f = read_bin(args.bin)
    f.close()
    e0 = args.base_experts or config.get("graft_base_experts")
    if not e0 or e0 >= cfg["n_experts"]:
        raise SystemExit("need --base-experts (or a graft_base_experts "
                         "config key) smaller than n_experts")
    moe = moe_layers_of(cfg)
    torch.manual_seed(args.seed)
    model = NanoModel(cfg)
    model.load_state_dict(sd)
    model.to(args.device).eval()

    encode, bos = make_host_encoder(args.tiktoken, args.vocab_nano)
    unk = None
    if args.vocab_nano:
        with open(args.vocab_nano) as fv:
            unk = json.load(fv)["specials"]["unk"]
    xs, ms = make_windows(iter_docs(args.text), encode, bos, args.seq,
                          args.max_tokens, unk)
    hold = min(args.holdout_windows, len(xs) // 4)
    xs_h, ms_h = xs[:hold], ms[:hold]
    xs_t, ms_t = xs[hold:], ms[hold:]
    print(f"{len(xs_t)} train windows, {hold} holdout windows "
          f"({cfg['n_experts'] - e0} grafted experts on {len(moe)} layers)")

    set_graft_bias(model, moe, e0, args.heal_bias)
    ce0, cnt = ce_eval(model, xs_h, ms_h, cfg["vocab"], args.device,
                       args.batch)
    print(f"holdout CE before heal: {ce0:.4f} nats ({cnt} tokens, "
          f"heal bias {args.heal_bias})")

    trainable = freeze_except_grafts(model, moe, e0)
    n_par = sum(p.numel() for p in trainable)
    print(f"training {len(trainable)} tensors, {n_par} params")
    train(model, xs_t, ms_t, cfg["vocab"], args.device, args.steps,
          args.lr, args.warmup, args.clip, args.batch, trainable,
          args.seed)

    ce1, _ = ce_eval(model, xs_h, ms_h, cfg["vocab"], args.device,
                     args.batch)
    print(f"holdout CE after heal:  {ce1:.4f} nats (delta {ce1 - ce0:+.4f})")

    n = patch_bin(args.bin, args.out, collect_updates(model, config, moe,
                                                      e0, args.final_bias))
    print(f"-> {args.out}: {n} tensors rewritten in place, final bias "
          f"{args.final_bias}")


# ------------------------------------------------------------------ selftest

def selftest():
    import subprocess
    import tempfile
    import torch
    from model_nano import NanoModel
    from capture_host import TINY

    cfg = dict(TINY)
    # export.py assumes kda channels == hidden; make the tiny config honor
    # that (4 heads x 16 = 64 = hidden)
    cfg.update(n_experts=9, kda_heads=4, kda_dim=16)
    e0 = 8
    torch.manual_seed(9)
    model = NanoModel(cfg).eval()
    moe = moe_layers_of(cfg)

    # learnable synthetic stream: a skewed unigram over few symbols (a
    # uniform stream has nothing to learn and the heal would rightly be a
    # no-op)
    rng = np.random.default_rng(1)
    probs = 1.0 / np.arange(1, 17)
    probs /= probs.sum()
    xs = rng.choice(16, size=(24, 65), p=probs)
    ms = np.ones((24, 64), bool)

    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    set_graft_bias(model, moe, e0, 0.0)
    trainable = freeze_except_grafts(model, moe, e0)
    ce_a, _ = ce_eval(model, xs[:8], ms[:8], cfg["vocab"], "cpu", batch=4)
    train(model, xs[8:], ms[8:], cfg["vocab"], "cpu", steps=60, lr=1e-2,
          warmup=5, clip=1.0, batch=4, trainable=trainable,
          log=lambda *a, **k: None)
    ce_b, _ = ce_eval(model, xs[:8], ms[:8], cfg["vocab"], "cpu", batch=4)
    print(f"claim 1: heal reduces holdout CE on learnable data "
          f"({ce_a:.3f} -> {ce_b:.3f})")
    assert ce_b < ce_a, (ce_a, ce_b)

    changed, frozen_ok = [], True
    for n, p in model.named_parameters():
        same = torch.equal(before[n], p.detach())
        is_graft = any(
            n == f"layers.{l}.block_sparse_moe.gate.weight" or
            (".experts." in n and int(n.split(".experts.")[1].split(".")[0])
             >= e0) for l in moe if f"layers.{l}." in n)
        if is_graft and not same:
            changed.append(n)
        if not is_graft and not same:
            frozen_ok = False
            print("  LEAKED:", n)
    assert frozen_ok, "a frozen parameter moved"
    assert changed, "no grafted parameter moved"
    gw = model.layers[moe[0]].block_sparse_moe.gate.weight
    assert torch.equal(before[f"layers.{moe[0]}.block_sparse_moe"
                              f".gate.weight"][:e0], gw.detach()[:e0])
    print(f"claim 2: only grafted params moved ({len(changed)} tensors), "
          "base gate rows bit-exact")

    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "m.pt")
        b0 = os.path.join(td, "m.bin")
        b1 = os.path.join(td, "healed.bin")
        sd = {k: v.detach() for k, v in model.state_dict().items()}
        torch.save({"model": sd, "cfg": cfg, "step": 0}, ck)
        r = subprocess.run(
            [sys.executable, os.path.join(_NANO, "export.py"),
             "--ckpt", ck, "--out", b0], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        config, _e, f = read_bin(b0)
        f.close()
        upd = collect_updates(model, config, moe, e0, final_bias=-4.0)
        patch_bin(b0, b1, upd)
        sd2, cfg2 = convert(b1)
        gw2 = sd2[f"layers.{moe[0]}.block_sparse_moe.gate.weight"]
        assert torch.allclose(gw2, gw.detach(), atol=1e-6)
        bias2 = sd2[f"layers.{moe[0]}.block_sparse_moe.gate"
                    ".e_score_correction_bias"]
        assert float(bias2[e0]) == -4.0
        w1_mem = model.layers[moe[0]].block_sparse_moe.experts[e0] \
                      .w1.weight.detach().numpy()
        p, s = quantize_mxfp4(w1_mem)
        from bin2pt import dequant_mxfp4
        w1_ref = dequant_mxfp4(p.tobytes() + s.tobytes(),
                               list(w1_mem.shape))
        w1_bin = sd2[f"layers.{moe[0]}.block_sparse_moe.experts."
                     f"{e0}.w1.weight"].numpy()
        assert np.array_equal(w1_bin, w1_ref)
        print("claim 3: patch round trip - healed gate exact, final bias "
              "applied, experts equal their MXFP4 image")

    print("graft_heal selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
