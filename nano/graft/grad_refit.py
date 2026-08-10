#!/usr/bin/env python3
"""nanokimi - grad_refit: one-pass gradient healing of grafted experts,
no optimizer.

SGD healing spends hundreds of steps rediscovering the same information:
with the host frozen, the loss gradient at each MoE layer's output tells
every grafted expert how its contribution should move, everywhere at
once. This tool turns that observation into a closed-form update:

  1. one forward+backward pass over calibration windows, harvesting per
     position: the latent stream, the routing decisions, and the loss
     gradient at the expert-mix output (full-backward hook on
     routed_expert_norm);
  2. per grafted expert, the Gauss-Newton direction for its down matrix
     restricted to the positions it actually served:
       D = (sum A^T A + lambda I)^-1  sum A^T (w_g * dL/dy)
     with A the SiTU response and w_g the mixing weight - the same
     normal equations as a ridge solve, sourced from gradients;
  3. a trust-region line search over the RELATIVE step size (the
     direction is normalized to the current |w2|), evaluated on holdout
     cross-entropy; the zero step is always in the grid, so the output
     can never be worse than the input on the calibration holdout;
  4. optionally iterate (gradients are recomputed at the new point);
  5. patch the final w2 tensors into a copy of the .bin.

Wall-clock cost is a few forward+backward passes over the calibration
set - minutes, versus hundreds of optimizer steps. Loads the model
through bin2pt (materialized): sized for nano-class .bins.

usage:
  python3 grad_refit.py --bin routed.bin --text corpus.jsonl \
      --out healed.bin [--iters 2] [--windows 160] [--holdout-windows 48] \
      [--eta-grid "0.01,0.03,0.1,0.3"] [--device cuda] \
      [--tiktoken p] [--vocab-nano v]
  python3 grad_refit.py --selftest
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)
from bin2pt import convert, read_bin  # noqa: E402
from graft_heal import make_windows, ce_eval, patch_bin  # noqa: E402
from capture_host import moe_layers_of  # noqa: E402


def _situ_t(g, u):
    import torch
    return (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) \
        * (25.0 * torch.tanh(u / 25.0))


class GradTaps:
    """Per MoE layer: latent stream + routing (forward hooks) and the
    loss gradient at the expert-mix output (full backward hook)."""

    def __init__(self, model, moe):
        self.lat = {}
        self.route = {}
        self.grad = {}
        self.handles = []
        for l in moe:
            blk = model.layers[l].block_sparse_moe
            self.handles.append(
                blk.routed_expert_down_proj.register_forward_hook(
                    self._fwd_lat(l)))
            self.handles.append(
                blk.gate.register_forward_hook(self._fwd_gate(l)))
            self.handles.append(
                blk.routed_expert_norm.register_full_backward_hook(
                    self._bwd(l)))

    def _fwd_lat(self, l):
        def hook(_m, _i, out):
            self.lat[l] = out.detach()
        return hook

    def _fwd_gate(self, l):
        def hook(_m, _i, out):
            self.route[l] = (out[0].detach(), out[1].detach())
        return hook

    def _bwd(self, l):
        def hook(_m, grad_input, _go):
            self.grad[l] = grad_input[0].detach()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def harvest(model, moe, e0, xs, ms, vocab, device, batch=4,
            rel_lambda=1e-4):
    """One forward+backward pass over the windows; returns per (layer,
    expert) the Gauss-Newton direction D [mi, rh] and the fired count."""
    import torch
    taps = GradTaps(model, moe)
    n_exp = len(model.layers[moe[0]].block_sparse_moe.experts)
    acc = {}
    for l in moe:
        blk = model.layers[l].block_sparse_moe
        mi = blk.experts[e0].w1.weight.shape[0]
        rh = blk.experts[e0].w2.weight.shape[0]
        for e in range(e0, n_exp):
            acc[(l, e)] = [np.zeros((mi, mi)), np.zeros((mi, rh)), 0]
    for b0 in range(0, len(xs), batch):
        x = torch.tensor(xs[b0:b0 + batch, :-1], device=device)
        y = torch.tensor(xs[b0:b0 + batch, 1:], device=device)
        m = torch.tensor(ms[b0:b0 + batch], device=device)
        logits = model(x).view(x.shape[0], x.shape[1], vocab)
        loss = torch.nn.functional.cross_entropy(
            logits[m], y[m], reduction="sum")
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            for l in moe:
                blk = model.layers[l].block_sparse_moe
                lat = taps.lat[l]
                g = taps.grad[l].reshape(lat.shape)
                ti, tw = taps.route[l]
                for e in range(e0, n_exp):
                    sel = ti == e
                    rows = sel.any(dim=1)
                    if not bool(rows.any()):
                        continue
                    wg = (tw * sel).sum(dim=1)[rows]
                    h = lat[rows]
                    a = _situ_t(h @ blk.experts[e].w1.weight.T,
                                h @ blk.experts[e].w3.weight.T)
                    t = g[rows] * wg[:, None]
                    a64 = a.double().cpu().numpy()
                    acc[(l, e)][0] += a64.T @ a64
                    acc[(l, e)][1] += a64.T @ t.double().cpu().numpy()
                    acc[(l, e)][2] += int(rows.sum())
    taps.close()
    dirs = {}
    for (l, e), (ga, c, n) in acc.items():
        if n == 0:
            continue
        lam = rel_lambda * np.trace(ga) / ga.shape[0] + 1e-12
        d = np.linalg.solve(ga + lam * np.eye(ga.shape[0]), c)
        dirs[(l, e)] = (d, n)
    return dirs


def apply_step(model, dirs, eta):
    """w2 <- w2 - eta * |w2|/|D| * D^T for every direction; returns the
    tensors' previous values for revert."""
    import torch
    prev = {}
    with torch.no_grad():
        for (l, e), (d, _n) in dirs.items():
            w2 = model.layers[l].block_sparse_moe.experts[e].w2.weight
            prev[(l, e)] = w2.detach().clone()
            if eta == 0.0:
                continue
            step = torch.tensor(d.T, dtype=w2.dtype, device=w2.device)
            scale = w2.norm() / step.norm().clamp_min(1e-12)
            w2 -= eta * scale * step
    return prev


def revert(model, prev):
    import torch
    with torch.no_grad():
        for (l, e), w in prev.items():
            model.layers[l].block_sparse_moe.experts[e].w2.weight.copy_(w)


def refit(model, moe, e0, xs_t, ms_t, xs_h, ms_h, vocab, device, batch=4,
          iters=2, eta_grid=(0.01, 0.03, 0.1, 0.3), rel_lambda=1e-4,
          log=print):
    """Gradient harvest + trust-region line search, `iters` times.
    Monotone on the holdout by construction (eta=0 in every search)."""
    ce0, _ = ce_eval(model, xs_h, ms_h, vocab, device, batch)
    log(f"holdout CE before: {ce0:.4f}")
    best_ce = ce0
    for it in range(iters):
        t0 = time.time()
        dirs = harvest(model, moe, e0, xs_t, ms_t, vocab, device, batch,
                       rel_lambda)
        if not dirs:
            log("no grafted expert fired; nothing to refit")
            break
        fired = sum(n for _d, n in dirs.values())
        best_eta = 0.0
        for eta in eta_grid:
            prev = apply_step(model, dirs, eta)
            ce, _ = ce_eval(model, xs_h, ms_h, vocab, device, batch)
            revert(model, prev)
            log(f"  iter {it}: eta {eta:g} -> CE {ce:.4f}")
            if ce < best_ce:
                best_ce, best_eta = ce, eta
        apply_step(model, dirs, best_eta)
        log(f"iter {it}: {len(dirs)} experts, {fired} fired rows, kept "
            f"eta {best_eta:g}, CE {best_ce:.4f} "
            f"({time.time() - t0:.0f}s)")
        if best_eta == 0.0:
            break
    return ce0, best_ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-experts", type=int, default=None)
    ap.add_argument("--windows", type=int, default=160)
    ap.add_argument("--holdout-windows", type=int, default=48)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--eta-grid", default="0.01,0.03,0.1,0.3")
    ap.add_argument("--rel-lambda", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiktoken", default=None)
    ap.add_argument("--vocab-nano", default=None)
    ap.add_argument("--max-tokens", type=int, default=300_000)
    args = ap.parse_args()

    import torch
    from model_nano import NanoModel
    from capture_host import make_host_encoder
    from capture_donor import iter_docs

    t_start = time.time()
    sd, cfg = convert(args.bin)
    config, _e, f = read_bin(args.bin)
    f.close()
    e0 = args.base_experts or config.get("graft_base_experts")
    if not e0 or e0 >= cfg["n_experts"]:
        raise SystemExit("need --base-experts (or graft_base_experts key)")
    moe = moe_layers_of(cfg)
    model = NanoModel(cfg)
    model.load_state_dict(sd)
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for l in moe:
        blk = model.layers[l].block_sparse_moe
        for e in range(e0, len(blk.experts)):
            blk.experts[e].w2.weight.requires_grad_(True)

    encode, bos = make_host_encoder(args.tiktoken, args.vocab_nano)
    unk = None
    if args.vocab_nano:
        with open(args.vocab_nano) as fv:
            unk = json.load(fv)["specials"]["unk"]
    xs, ms = make_windows(iter_docs(args.text), encode, bos, args.seq,
                          args.max_tokens, unk)
    hold = min(args.holdout_windows, len(xs) // 4)
    xs_h, ms_h = xs[:hold], ms[:hold]
    take = min(args.windows, len(xs) - hold)
    xs_t, ms_t = xs[hold:hold + take], ms[hold:hold + take]
    print(f"{take} gradient windows, {hold} holdout windows")

    eta_grid = tuple(float(x) for x in args.eta_grid.split(","))
    ce0, ce1 = refit(model, moe, e0, xs_t, ms_t, xs_h, ms_h, cfg["vocab"],
                     args.device, args.batch, args.iters, eta_grid,
                     args.rel_lambda)

    upd = {}
    import torch as _t
    with _t.no_grad():
        for l in moe:
            blk = model.layers[l].block_sparse_moe
            for e in range(e0, len(blk.experts)):
                upd[f"layers.{l}.block_sparse_moe.experts.{e}.w2"] = \
                    blk.experts[e].w2.weight.detach().cpu().numpy()
    n = patch_bin(args.bin, args.out, upd)
    print(f"-> {args.out}: {n} w2 tensors, CE {ce0:.4f} -> {ce1:.4f}, "
          f"total {time.time() - t_start:.0f}s, zero optimizer steps",
          flush=True)


# ------------------------------------------------------------------ selftest

def selftest():
    import torch
    from model_nano import NanoModel
    from capture_host import TINY
    from graft_heal import set_graft_bias

    cfg = dict(TINY)
    cfg.update(n_experts=9, kda_heads=4, kda_dim=16)
    e0 = 8
    torch.manual_seed(11)
    model = NanoModel(cfg).eval()
    moe = moe_layers_of(cfg)
    for p in model.parameters():
        p.requires_grad_(False)
    for l in moe:
        model.layers[l].block_sparse_moe.experts[e0].w2.weight \
            .requires_grad_(True)
    set_graft_bias(model, moe, e0, 0.0)

    rng = np.random.default_rng(4)
    probs = 1.0 / np.arange(1, 17)
    probs /= probs.sum()
    xs = rng.choice(16, size=(24, 65), p=probs)
    ms = np.ones((24, 64), bool)

    # claim 1: the harvested normal-equation RHS matches autograd's
    # w2.grad on the same backward (C^T == dL/dw2)
    import torch.nn.functional as F
    x = torch.tensor(xs[:4, :-1])
    y = torch.tensor(xs[:4, 1:])
    m = torch.tensor(ms[:4])
    taps = GradTaps(model, moe)
    logits = model(x).view(4, 64, cfg["vocab"])
    loss = F.cross_entropy(logits[m], y[m], reduction="sum")
    model.zero_grad(set_to_none=True)
    loss.backward()
    checked = 0
    for l in moe:
        blk = model.layers[l].block_sparse_moe
        w2 = blk.experts[e0].w2.weight
        if w2.grad is None or float(w2.grad.abs().sum()) == 0:
            continue
        lat = taps.lat[l]
        g = taps.grad[l].reshape(lat.shape)
        ti, tw = taps.route[l]
        sel = ti == e0
        rows = sel.any(dim=1)
        wg = (tw * sel).sum(dim=1)[rows]
        h = lat[rows]
        a = _situ_t(h @ blk.experts[e0].w1.weight.T,
                    h @ blk.experts[e0].w3.weight.T)
        t = g[rows] * wg[:, None]
        c = (a.double().T @ t.double()).numpy()
        ref = w2.grad.double().numpy()
        rel = np.abs(c.T - ref).max() / (np.abs(ref).max() + 1e-12)
        assert rel < 1e-3, rel
        checked += 1
    taps.close()
    assert checked > 0
    print(f"claim 1: harvested normal equations match autograd's w2.grad "
          f"on {checked} layers (hooks, routing and weights all agree)")

    # claim 2: refit reduces holdout CE on learnable data, and eta=0 in
    # the grid makes it monotone
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    ce0, ce1 = refit(model, moe, e0, xs[8:], ms[8:], xs[:8], ms[:8],
                     cfg["vocab"], "cpu", batch=4, iters=2,
                     eta_grid=(0.03, 0.1, 0.3, 1.0),
                     log=lambda *a, **k: None)
    assert ce1 <= ce0, (ce0, ce1)
    print(f"claim 2: gradient refit is monotone on holdout "
          f"({ce0:.4f} -> {ce1:.4f})")

    moved, leaked = 0, []
    for n, p in model.named_parameters():
        same = torch.equal(before[n], p.detach())
        is_g = ".experts.8.w2" in n
        if is_g and not same:
            moved += 1
        if not is_g and not same:
            leaked.append(n)
    assert not leaked, leaked
    assert moved > 0
    print(f"claim 3: only grafted w2 tensors moved ({moved}), everything "
          "else bit-exact")

    print("grad_refit selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
