#!/usr/bin/env python3
"""nanokimi - route_solve: router rows for grafted experts from MEASURED
per-token utility, plus a forward-only (bias, gain) calibration. The method
uses forward passes and closed-form solves without gradient steps.

An activation-derived router row predicts where the donor has something to
say. This tool replaces it with where the donor actually HELPS: for each
grafted expert it runs the model over a probe set twice
(expert silenced / expert forced into the top-k) and records the
per-token loss difference. A ridge solve then maps the router input
stream to that measured utility, and the row is normalized to a
router-typical logit range.

A final calibration sweeps a small grid of (selection bias, output gain)
by evaluating holdout cross-entropy per combination, picks the best, and
folds the gain into the experts' down matrices (w2). The whole procedure
is forward passes + closed-form solves. Safety: the silent combination
(strongly negative bias) is always in the grid, so the output can never
be worse than the input model on the calibration holdout.

Output: a byte copy of the input .bin with only the router rows, the
selection biases and (if gain != 1) the grafted w2 tensors rewritten.

usage:
  python3 route_solve.py --bin grafted.bin --text corpus.jsonl \
      --out routed.bin --device cuda [--vocab-nano v.json] [--tiktoken p] \
      [--probe-windows 160] [--holdout-windows 32] \
      [--bias-grid " -1,-0.5,0,0.5"] [--gamma-grid "0.5,1,1.5"]
  python3 route_solve.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)
from bin2pt import read_bin, convert  # noqa: E402
from graftlib import RectGram  # noqa: E402
from graft_heal import (make_windows, ce_eval, patch_bin, set_graft_bias)  # noqa: E402
from capture_host import moe_layers_of  # noqa: E402

SILENT = -9.0


def token_losses(model, xs, ms, vocab, device, batch=4, taps=None):
    """Per-token CE at masked positions, flattened in window order. With
    taps (list of layers), also returns {l: fp16 [n_masked, d]} of the
    router input stream at the same positions, same order."""
    import torch
    rec = {}
    handles = []
    if taps:
        def mk(l):
            def hook(_m, inp, _o):
                rec[l] = inp[0].detach().float()
            return hook
        for l in taps:
            mod = model.layers[l].block_sparse_moe.routed_expert_down_proj
            handles.append(mod.register_forward_hook(mk(l)))
    losses = []
    plns = {l: [] for l in (taps or [])}
    with torch.no_grad():
        for b0 in range(0, len(xs), batch):
            x = torch.tensor(xs[b0:b0 + batch, :-1], device=device)
            y = torch.tensor(xs[b0:b0 + batch, 1:], device=device)
            m = torch.tensor(ms[b0:b0 + batch], device=device)
            logits = model(x).view(x.shape[0], x.shape[1], vocab)
            ce = torch.nn.functional.cross_entropy(
                logits[m], y[m], reduction="none")
            losses.append(ce.cpu().numpy())
            for l in (taps or []):
                h = rec[l].view(x.shape[0], x.shape[1], -1)
                plns[l].append(h[m].cpu().numpy().astype(np.float16))
    for h in handles:
        h.remove()
    losses = np.concatenate(losses)
    return (losses, {l: np.concatenate(v) for l, v in plns.items()}) \
        if taps else losses


def solve_rows(model, moe, e0, xs, ms, vocab, device, batch=4,
               gate_sigma=1.0, rel_lambda=1e-4, log=print):
    """Measures per-expert utility and solves normalized router rows.
    Returns {(l, e): row} and a report dict."""
    import torch
    n_g = len(next(iter([model.layers[moe[0]].block_sparse_moe
                         .gate.e_score_correction_bias]))) - e0
    set_graft_bias(model, moe, e0, SILENT)
    base, plns = token_losses(model, xs, ms, vocab, device, batch, taps=moe)
    rows = {}
    report = {}
    for l in moe:
        bias = model.layers[l].block_sparse_moe.gate.e_score_correction_bias
        for e in range(e0, e0 + n_g):
            with torch.no_grad():
                bias[e] = 1e4  # force into the top-k on every token
            forced = token_losses(model, xs, ms, vocab, device, batch)
            with torch.no_grad():
                bias[e] = SILENT
            util = base - forced  # >0 where the expert helps
            g = RectGram(plns[l].shape[1], 1)
            for t0 in range(0, len(util), 8192):
                g.add(plns[l][t0:t0 + 8192],
                      util[t0:t0 + 8192, None])
            row, _ = g.solve(rel_lambda)
            row = row[0]
            ex2 = float(row @ g.gxx @ row) / g.n
            mu = float(row @ g.sum_x) / g.n
            sd = np.sqrt(max(ex2 - mu * mu, 1e-12))
            rows[(l, e)] = (row * (gate_sigma / sd)).astype(np.float32)
            report[f"L{l}.e{e}"] = {
                "mean_util": float(util.mean()),
                "p95_util": float(np.percentile(util, 95)),
                "help_frac": float((util > 0).mean()),
            }
            log(f"L{l} e{e}: mean util {util.mean():+.4f}, "
                f"helps on {100 * (util > 0).mean():.1f}% of tokens, "
                f"p95 {np.percentile(util, 95):+.4f}")
    return rows, report


def install_rows(model, rows):
    import torch
    with torch.no_grad():
        for (l, e), row in rows.items():
            gw = model.layers[l].block_sparse_moe.gate.weight
            gw[e] = torch.tensor(row, dtype=gw.dtype, device=gw.device)


def scale_grafts(model, moe, e0, factor):
    import torch
    if factor == 1.0:
        return
    with torch.no_grad():
        for l in moe:
            blk = model.layers[l].block_sparse_moe
            for e in range(e0, len(blk.experts)):
                blk.experts[e].w2.weight *= factor


def sweep(model, moe, e0, xs, ms, vocab, device, bias_grid, gamma_grid,
          batch=4, log=print):
    """Evaluates holdout CE over the (bias, gamma) grid; the silent
    combination is always included. Leaves the model at the best combo
    and returns (best_bias, best_gamma, best_ce, base_ce)."""
    set_graft_bias(model, moe, e0, SILENT)
    base_ce, _ = ce_eval(model, xs, ms, vocab, device, batch)
    log(f"holdout CE, grafts silent: {base_ce:.4f}")
    best = (SILENT, 1.0, base_ce)
    cur_gamma = 1.0
    for gamma in gamma_grid:
        scale_grafts(model, moe, e0, gamma / cur_gamma)
        cur_gamma = gamma
        for b in bias_grid:
            set_graft_bias(model, moe, e0, b)
            ce, _ = ce_eval(model, xs, ms, vocab, device, batch)
            log(f"  bias {b:+.2f} gamma {gamma:.2f}: CE {ce:.4f}")
            if ce < best[2]:
                best = (b, gamma, ce)
    scale_grafts(model, moe, e0, best[1] / cur_gamma)
    set_graft_bias(model, moe, e0, best[0])
    return best[0], best[1], best[2], base_ce


def collect_updates(model, moe, e0, gamma):
    import torch
    upd = {}
    with torch.no_grad():
        for l in moe:
            blk = model.layers[l].block_sparse_moe
            m = f"layers.{l}.block_sparse_moe."
            upd[m + "gate.weight"] = blk.gate.weight.detach().cpu().numpy()
            upd[m + "gate.e_score_correction_bias"] = \
                blk.gate.e_score_correction_bias.detach().cpu().numpy()
            if gamma != 1.0:
                for e in range(e0, len(blk.experts)):
                    upd[m + f"experts.{e}.w2"] = \
                        blk.experts[e].w2.weight.detach().cpu().numpy()
    return upd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-experts", type=int, default=None)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiktoken", default=None)
    ap.add_argument("--vocab-nano", default=None)
    ap.add_argument("--max-tokens", type=int, default=300_000)
    ap.add_argument("--probe-windows", type=int, default=160)
    ap.add_argument("--holdout-windows", type=int, default=32)
    ap.add_argument("--gate-sigma", type=float, default=1.0)
    ap.add_argument("--rel-lambda", type=float, default=1e-4)
    ap.add_argument("--bias-grid", default="-1,-0.5,0,0.5")
    ap.add_argument("--gamma-grid", default="0.5,1,1.5")
    args = ap.parse_args()

    import torch
    from model_nano import NanoModel
    from capture_host import make_host_encoder
    from capture_donor import iter_docs

    sd, cfg = convert(args.bin)
    config, _e, f = read_bin(args.bin)
    f.close()
    e0 = args.base_experts or config.get("graft_base_experts")
    if not e0 or e0 >= cfg["n_experts"]:
        raise SystemExit("need --base-experts (or graft_base_experts key) "
                         "smaller than n_experts")
    moe = moe_layers_of(cfg)
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
    probe = min(args.probe_windows, len(xs) - hold)
    xs_p, ms_p = xs[hold:hold + probe], ms[hold:hold + probe]
    print(f"{probe} probe windows, {hold} holdout windows, "
          f"{cfg['n_experts'] - e0} grafted expert(s) x {len(moe)} layers")

    rows, _rep = solve_rows(model, moe, e0, xs_p, ms_p, cfg["vocab"],
                            args.device, args.batch, args.gate_sigma,
                            args.rel_lambda)
    install_rows(model, rows)
    bias_grid = [float(x) for x in args.bias_grid.split(",")]
    gamma_grid = [float(x) for x in args.gamma_grid.split(",")]
    b, gamma, ce, base_ce = sweep(model, moe, e0, xs_h, ms_h, cfg["vocab"],
                                  args.device, bias_grid, gamma_grid,
                                  args.batch)
    print(f"best: bias {b:+.2f} gamma {gamma:.2f} -> holdout CE {ce:.4f} "
          f"(silent {base_ce:.4f})")
    n = patch_bin(args.bin, args.out, collect_updates(model, moe, e0, gamma))
    print(f"-> {args.out}: {n} tensors rewritten, zero training steps",
          flush=True)


# ------------------------------------------------------------------ selftest

def selftest():
    import torch
    from model_nano import NanoModel
    from capture_host import TINY

    cfg = dict(TINY)
    cfg.update(n_experts=9, kda_heads=4, kda_dim=16)
    e0 = 8
    torch.manual_seed(3)
    model = NanoModel(cfg).eval()
    moe = moe_layers_of(cfg)

    rng = np.random.default_rng(5)
    probs = 1.0 / np.arange(1, 17)
    probs /= probs.sum()
    xs = rng.choice(16, size=(20, 65), p=probs)
    ms = np.ones((20, 64), bool)
    v = cfg["vocab"]

    set_graft_bias(model, moe, e0, SILENT)
    ce_silent, _ = ce_eval(model, xs[:6], ms[:6], v, "cpu", 4)

    base, plns = token_losses(model, xs[6:], ms[6:], v, "cpu", 4, taps=moe)
    assert base.shape == (14 * 64,) and plns[moe[0]].shape == (14 * 64, 64)
    print("claim 1: per-token losses and router stream aligned "
          f"({base.shape[0]} positions)")

    rows, rep = solve_rows(model, moe, e0, xs[6:], ms[6:], v, "cpu", 4,
                           log=lambda *a, **k: None)
    assert len(rows) == len(moe)
    ce_after_probe, _ = ce_eval(model, xs[:6], ms[:6], v, "cpu", 4)
    assert abs(ce_after_probe - ce_silent) < 1e-6
    print("claim 2: probing restores the model exactly "
          f"(CE {ce_silent:.4f} == {ce_after_probe:.4f})")

    logit_sd = float(np.std(plns[moe[0]].astype(np.float32)
                            @ rows[(moe[0], e0)]))
    assert 0.5 < logit_sd < 2.0, logit_sd
    print(f"claim 3: solved rows have router-typical logits "
          f"(std {logit_sd:.2f})")

    install_rows(model, rows)
    b, gamma, ce, base_ce = sweep(model, moe, e0, xs[:6], ms[:6], v, "cpu",
                                  [-1.0, 0.0], [0.5, 1.0], 4,
                                  log=lambda *a, **k: None)
    assert ce <= base_ce + 1e-9, (ce, base_ce)
    print(f"claim 4: sweep never worse than silent on its holdout "
          f"({ce:.4f} <= {base_ce:.4f}, bias {b:+.1f} gamma {gamma})")

    upd = collect_updates(model, moe, e0, gamma)
    for l in moe:
        gw = upd[f"layers.{l}.block_sparse_moe.gate.weight"]
        assert gw.shape == (9, cfg["hidden"])
        bias = upd[f"layers.{l}.block_sparse_moe.gate"
                   ".e_score_correction_bias"]
        assert float(bias[e0]) == b
    print("claim 5: update set carries solved rows and chosen bias")

    print("route_solve selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
