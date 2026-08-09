#!/usr/bin/env python3
"""nanokimi - als_refit: closed-form re-solve of grafted expert down
matrices on their ROUTED distribution.

expert_solve fits each grafted expert's w2 over every anchor position.
Once router rows exist (route_solve), the expert only ever runs on the
slice of tokens the router sends it, whose distribution differs from the
global one. This tool re-solves w2 by weighted ridge restricted to that
slice - one alternating-least-squares step with the routing held fixed:

  1. simulate the routing offline from the captures: full router keys
     (sigmoid(pln @ gate.weight^T) + selection bias) for every anchor,
     top-k selection, and the expert's renormalized mixing weight;
  2. rebuild the expert's target from the pack's donor-to-latent stitch
     (m_out) and the donor capture (optionally differential, minus the
     host bank's own mix output);
  3. weighted ridge of the SiTU response against the target over the
     fired anchors only (weights = mixing weights);
  4. patch the refitted w2 tensors into a copy of the .bin.

Everything is linear algebra on captured activations; no model forward
and no gradient step. Experts whose selection bias silences them can be
refit under a hypothetical bias (--assume-bias), so a later calibration
pass may activate them with a slice-tuned w2. Loads the model tensors
through bin2pt (materialized): sized for nano-class .bins.

usage:
  python3 als_refit.py --bin routed.bin --pack graft.npz \
      --host-capture cap/host --donor-capture cap/don --out refit.bin \
      [--assume-bias 0.0] [--target donor|diff] [--rel-lambda 1e-4] \
      [--min-fired 2048]
  python3 als_refit.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)
from bin2pt import convert, read_bin  # noqa: E402
from graftlib import RectGram, match_anchors, open_capture, situ  # noqa: E402
from graft_heal import patch_bin  # noqa: E402
from capture_host import moe_layers_of  # noqa: E402


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def routed_slice(pln, gate_w, bias_vec, e_idx, top_k, assume_bias=None,
                 chunk=8192):
    """Simulates the router over `pln` rows: returns (fired mask [n],
    mixing weight [n]) for expert e_idx. bias affects selection keys
    only; the mixing weight is the raw sigmoid score renormalized over
    the selected set (noaux_tc semantics)."""
    n = pln.shape[0]
    bias = np.asarray(bias_vec, np.float64).copy()
    if assume_bias is not None:
        bias[e_idx] = assume_bias
    fired = np.zeros(n, bool)
    weight = np.zeros(n, np.float64)
    gwt = np.asarray(gate_w, np.float64).T
    for t0 in range(0, n, chunk):
        p = np.asarray(pln[t0:t0 + chunk], np.float64)
        scores = sigmoid(p @ gwt)
        keys = scores + bias[None, :]
        kth = np.partition(keys, -top_k, axis=1)[:, -top_k]
        sel = keys >= kth[:, None]
        f = sel[:, e_idx]
        denom = (scores * sel).sum(axis=1)
        fired[t0:t0 + chunk] = f
        weight[t0:t0 + chunk] = np.where(f, scores[:, e_idx]
                                         / np.maximum(denom, 1e-20), 0.0)
    return fired, weight


def refit_w2(lat, dz_rows, m_out, w1, w3, weights, y_moe_rows=None,
             rel_lambda=1e-4, holdout=2048, chunk=8192):
    """Weighted ridge of the SiTU response against the projected donor
    delta over pre-selected (already fired) anchor rows. Returns
    (w2 [rh, mi], fit residual, holdout rel err)."""
    n = lat.shape[0]
    holdout = min(holdout, n // 4)
    g = RectGram(w1.shape[0], m_out.shape[0])
    w1t = w1.T.astype(np.float64)
    w3t = w3.T.astype(np.float64)
    for t0 in range(holdout, n, chunk):
        h = np.asarray(lat[t0:t0 + chunk], np.float64)
        a = situ(h @ w1t, h @ w3t)
        y = np.asarray(dz_rows[t0:t0 + chunk], np.float64) @ m_out.T
        if y_moe_rows is not None:
            y = y - np.asarray(y_moe_rows[t0:t0 + chunk], np.float64)
        sw = np.sqrt(weights[t0:t0 + chunk])[:, None]
        g.add(a * sw, y * sw)
    w2, res = g.solve(rel_lambda)
    hh = np.asarray(lat[:holdout], np.float64)
    yy = np.asarray(dz_rows[:holdout], np.float64) @ m_out.T
    if y_moe_rows is not None:
        yy = yy - np.asarray(y_moe_rows[:holdout], np.float64)
    pred = situ(hh @ w1t, hh @ w3t) @ w2.T
    den = float((yy ** 2).sum())
    rel = float(((pred - yy) ** 2).sum() / den) if den > 0 else 0.0
    return w2.astype(np.float32), res, rel


def refit_all(bin_path, pack_path, host_prefix, donor_prefix, out_path,
              assume_bias=0.0, target="donor", rel_lambda=1e-4,
              min_fired=2048, log=print):
    sd, cfg = convert(bin_path)
    config, _e, f = read_bin(bin_path)
    f.close()
    e0 = config.get("graft_base_experts")
    if not e0:
        raise SystemExit("bin carries no graft_base_experts key")
    moe = moe_layers_of(cfg)
    top_k = cfg["top_k"]
    z = np.load(pack_path)
    meta = json.loads(bytes(z["meta"]).decode())
    bands = int(meta["bands"])

    hmeta, hends, hmask, hplanes = open_capture(host_prefix)
    dmeta, dends, dmask, dplanes = open_capture(donor_prefix)
    ih, idz = match_anchors(hends, hmask, dends, dmask)

    updates = {}
    report = {}
    for l in moe:
        m = f"layers.{l}.block_sparse_moe."
        gate_w = sd[m + "gate.weight"].numpy()
        bias_vec = sd[m + "gate.e_score_correction_bias"].numpy()
        pln = hplanes[f"L{l}.pln"]
        lat = hplanes[f"L{l}.lat"]
        y_moe = hplanes.get(f"L{l}.moe") if target == "diff" else None
        for g_i in range(bands):
            e_idx = e0 + g_i
            dl = int(z[f"L{l}.g{g_i}.donor_layer"])
            fired, wgt = routed_slice(pln[ih], gate_w, bias_vec, e_idx,
                                      top_k, assume_bias)
            n_f = int(fired.sum())
            frac = n_f / len(ih)
            if n_f < min_fired:
                log(f"L{l} e{e_idx}: fired {n_f} ({100 * frac:.2f}%) "
                    f"< {min_fired}, keeping global w2")
                report[f"L{l}.e{e_idx}"] = {"fired": n_f, "refit": False}
                continue
            fi = np.nonzero(fired)[0]
            w2, res, rel = refit_w2(
                lat[ih[fi]], dplanes[f"L{dl}.dz"][idz[fi]],
                z[f"L{l}.g{g_i}.m_out"].astype(np.float64),
                z[f"L{l}.g{g_i}.w1"], z[f"L{l}.g{g_i}.w3"], wgt[fi],
                None if y_moe is None else y_moe[ih[fi]],
                rel_lambda)
            updates[m + f"experts.{e_idx}.w2"] = w2
            report[f"L{l}.e{e_idx}"] = {"fired": n_f, "frac": frac,
                                        "refit": True, "slice_rel": rel}
            log(f"L{l} e{e_idx}: fired {n_f} ({100 * frac:.2f}%), "
                f"slice holdout rel {rel:.3f}, w2 refit")
    if not updates:
        raise SystemExit("nothing refit: no expert fires enough "
                         "(try --assume-bias or lower --min-fired)")
    n = patch_bin(bin_path, out_path, updates)
    log(f"-> {out_path}: {n} w2 tensor(s) refit on the routed slice, "
        f"zero training steps")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="routed .bin (rows and "
                    "selection biases already installed)")
    ap.add_argument("--pack", required=True)
    ap.add_argument("--host-capture", required=True)
    ap.add_argument("--donor-capture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assume-bias", type=float, default=0.0,
                    help="selection bias assumed for the refit slice "
                    "(lets silenced experts be refit for later "
                    "calibration)")
    ap.add_argument("--target", default="donor", choices=["donor", "diff"])
    ap.add_argument("--rel-lambda", type=float, default=1e-4)
    ap.add_argument("--min-fired", type=int, default=2048)
    args = ap.parse_args()
    refit_all(args.bin, args.pack, args.host_capture, args.donor_capture,
              args.out, args.assume_bias, args.target, args.rel_lambda,
              args.min_fired)


# ------------------------------------------------------------------ selftest

def selftest():
    rng = np.random.default_rng(31)
    n, d, rh, mi, d_d, e0, top_k = 12000, 24, 16, 8, 20, 6, 2

    # two-mode stream: the router sends the grafted expert only mode B,
    # where the target map DIFFERS from the global average
    mode = rng.random(n) < 0.35
    pln = rng.normal(size=(n, d))
    pln[mode, :4] += 2.5  # mode B is linearly visible to the router
    pln[:, 4] = 1.0       # constant column: lets the row carry an offset
    lat = rng.normal(size=(n, rh))
    map_a = rng.normal(size=(d_d, rh)) / np.sqrt(rh)
    map_b = rng.normal(size=(d_d, rh)) / np.sqrt(rh)
    dz = np.where(mode[:, None], lat @ map_b.T, lat @ map_a.T)
    m_out = np.eye(rh, d_d, dtype=np.float64)  # project first rh dims

    # router: host experts with random rows, graft row aligned to mode B
    gate_w = rng.normal(size=(e0 + 1, d)) * 0.3
    gate_w[e0] = 0.0
    gate_w[e0, :4] = 2.0
    gate_w[e0, 4] = -5.0  # offset through the constant column
    bias = np.zeros(e0 + 1)
    bias[e0] = -9.0
    fired, wgt = routed_slice(pln, gate_w, bias, e0, top_k, assume_bias=0.0)
    assert fired[mode].mean() > 0.8 and fired[~mode].mean() < 0.2, \
        (fired[mode].mean(), fired[~mode].mean())
    got_w = wgt[fired]
    assert (got_w > 0).all() and (got_w <= 1.0).all()
    print(f"claim 1: simulated routing fires on the intended mode "
          f"({100 * fired[mode].mean():.0f}% of B, "
          f"{100 * fired[~mode].mean():.0f}% of A)")

    # keys must honor the top-k rule exactly on a small manual check
    p0 = pln[:1]
    scores = sigmoid(p0 @ gate_w.T)[0]
    keys = scores + np.where(np.arange(e0 + 1) == e0, 0.0, bias[:e0 + 1])
    manual = keys[e0] >= np.sort(keys)[-top_k]
    assert bool(fired[0]) == bool(manual)
    print("claim 2: selection matches a manual top-k computation")

    # global vs slice-refit w2: the slice fit must beat the global fit on
    # the routed slice
    w1 = rng.normal(size=(mi, rh)).astype(np.float32)
    w3 = rng.normal(size=(mi, rh)).astype(np.float32)
    a_all = situ(lat @ w1.T.astype(np.float64), lat @ w3.T.astype(np.float64))
    y_all = dz @ m_out.T
    g_glob = RectGram(mi, rh)
    g_glob.add(a_all, y_all)
    w2_glob, _ = g_glob.solve(1e-4)

    fi = np.nonzero(fired)[0]
    w2_slice, _res, rel_slice = refit_w2(lat[fi], dz[fi], m_out, w1, w3,
                                         wgt[fi], holdout=1000)
    hh = lat[fi[:1000]]
    yy = dz[fi[:1000]] @ m_out.T
    a_h = situ(hh @ w1.T.astype(np.float64), hh @ w3.T.astype(np.float64))
    rel_glob = float(((a_h @ w2_glob.T - yy) ** 2).sum() / (yy ** 2).sum())
    print(f"claim 3: slice refit beats the global fit on the routed slice "
          f"({rel_slice:.3f} < {rel_glob:.3f})")
    assert rel_slice < rel_glob

    print("als_refit selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
