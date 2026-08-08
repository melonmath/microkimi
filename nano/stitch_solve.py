#!/usr/bin/env python3
"""nanokimi - closed-form seam adapter (ridge regression + SVD truncation).

The gradient-trained seam adapter (heal_stream.py --seam-adapter) learns a
low-rank correction h' = h + B A h that maps the residual dialect at the
cut point of a sliced model onto the dialect the next kept layer expects.
That same map can be SOLVED instead of learned: one calibration pass over
the full model collects paired statistics at the seam, and the optimal
linear correction follows in closed form.

    x = residual stream after the last layer before the cut   (source side)
    y = residual stream entering the first layer after the cut (target side)
    d = y - x   (what the deleted segment added)

    C = argmin_W  sum_tokens ||x W^T - d||^2 + lambda ||W||_F^2
      =>  W^T = (Gxx + lambda I)^-1 Gxd      (ridge, solved exactly)

The full-rank W is then SVD-truncated to the adapter rank and split the
usual way: W ~ B A with A = sqrt(S) V^T, B = U sqrt(S) - exactly the
SeamAdapter layout (model_nano.SeamAdapter, .bin tensors seam.A/seam.B),
so the result deploys through apply_lora_bin.write_seam_bin unchanged.

No gradient step, no hyperparameter search, deterministic. The price: the
correction is linear, so it cannot reproduce the cross-token mixing the
deleted KDA layers performed - expect most (not all) of the gap closed.
If a gap remains, the solved A/B is also a far better INIT for the
gradient heal than zero.

Interchange format (.npz, written by the capture tool or accumulated
here): Gxx [H,H], Gxd [H,H], Gdd [H,H] (float64), n (token count), and
optionally sample_x/sample_d held-out rows for an honest residual report.
Gram files merge by plain addition, so captures can be sharded.

usage:
  python3 stitch_solve.py --gram seam_gram.npz --rank 64 --out stitch.pt
  python3 stitch_solve.py --gram a.npz --gram b.npz --out stitch.pt
  python3 stitch_solve.py --raw cap.bounds.npy --src-boundary 12 \
      --dst-boundary 83 --gram-out seam_gram.npz --out stitch.pt
  python3 stitch_solve.py --gram seam_gram.npz --write-bin model.bin \
      --out stitched.bin --seam-after 11
  python3 stitch_solve.py --selftest

Boundary convention (capture_bounds raw files): boundary b is the residual
ENTERING layer b (0 = embedding output). The v3 cut "0-11,83-92" stitches
--src-boundary 12 (output of layer 11) to --dst-boundary 83 (input of
layer 83). The report also prints two diagnostics: the affine variant
(what a bias term would buy - the .bin seam format has no bias slot) and
the diagonal fit (what pure per-channel gain would buy).
"""
import argparse
import sys

import numpy as np


class GramAccumulator:
    """Streaming sufficient statistics for the ridge solve. float64: the
    residual stream has a few huge-norm directions and float32 accumulation
    over millions of tokens would drown the small ones. The first moments
    (sum_x, sum_d) are kept too: they make the affine diagnostic possible."""

    def __init__(self, hidden):
        self.h = hidden
        self.gxx = np.zeros((hidden, hidden), np.float64)
        self.gxd = np.zeros((hidden, hidden), np.float64)
        self.gdd = np.zeros((hidden, hidden), np.float64)
        self.sum_x = np.zeros(hidden, np.float64)
        self.sum_d = np.zeros(hidden, np.float64)
        self.n = 0

    def add(self, x, d):
        """x, d: [tokens, hidden] float arrays (any float dtype)."""
        x = np.asarray(x, np.float64)
        d = np.asarray(d, np.float64)
        assert x.shape == d.shape and x.shape[1] == self.h
        self.gxx += x.T @ x
        self.gxd += x.T @ d
        self.gdd += d.T @ d
        self.sum_x += x.sum(axis=0)
        self.sum_d += d.sum(axis=0)
        self.n += x.shape[0]

    def save(self, path, sample_x=None, sample_d=None):
        out = {"Gxx": self.gxx, "Gxd": self.gxd, "Gdd": self.gdd,
               "sum_x": self.sum_x, "sum_d": self.sum_d,
               "n": np.int64(self.n)}
        if sample_x is not None:
            out["sample_x"] = np.asarray(sample_x, np.float32)
            out["sample_d"] = np.asarray(sample_d, np.float32)
        np.savez(path, **out)


def grams_from_raw(raw_path, src, dst, chunk=4096, holdout=4096,
                   gram_out=None):
    """Grams from a capture_bounds raw file: fp16 memmap
    (n_tokens, n_bounds, hidden), token-major, with PREFIX.meta.json next to
    it. Boundary semantics: boundary b is the residual ENTERING layer b
    (0 = embedding output), so a cut keeping layers 0..a and b..L-1 stitches
    src=a+1 (output of layer a) to dst=b (input of layer b).
    `holdout` leading tokens are excluded from the grams and returned as the
    eval sample."""
    import json
    raw = np.load(raw_path, mmap_mode="r")
    meta_path = raw_path.replace(".bounds.npy", ".meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    n_tok, n_bounds, hidden = (int(meta["n_tokens"]), int(meta["n_bounds"]),
                               int(meta["hidden"]))
    if raw.shape != (n_tok, n_bounds, hidden):
        raise SystemExit(f"{raw_path}: shape {raw.shape}, meta says "
                         f"({n_tok}, {n_bounds}, {hidden})")
    if not (0 <= src < n_bounds and 0 <= dst < n_bounds):
        raise SystemExit(f"boundaries must be in [0, {n_bounds - 1}]")
    holdout = min(holdout, n_tok // 4)  # never starve the fit
    acc = GramAccumulator(hidden)
    for t0 in range(holdout, n_tok, chunk):
        t1 = min(t0 + chunk, n_tok)
        x = np.asarray(raw[t0:t1, src, :], np.float64)
        y = np.asarray(raw[t0:t1, dst, :], np.float64)
        acc.add(x, y - x)
    sx = np.asarray(raw[:holdout, src, :], np.float32)
    sd = np.asarray(raw[:holdout, dst, :], np.float32) - sx
    print(f"raw {raw_path}: {acc.n} tokens into grams, boundaries "
          f"{src} -> {dst}, holdout {holdout}")
    if gram_out:
        acc.save(gram_out, sample_x=sx, sample_d=sd)
        print(f"grams written: {gram_out}")
    return acc.gxx, acc.gxd, acc.gdd, acc.sum_x, acc.sum_d, acc.n, \
        {"sample_x": sx, "sample_d": sd}


def load_grams(paths):
    """Load and merge one or more gram .npz files (plain addition)."""
    gxx = gxd = gdd = sx = sd = None
    n = 0
    samples = {}
    for p in paths:
        z = np.load(p)
        n += int(z["n"])
        if gxx is None:
            gxx, gxd, gdd = z["Gxx"], z["Gxd"], z["Gdd"]
        else:
            gxx += z["Gxx"]
            gxd += z["Gxd"]
            gdd += z["Gdd"]
        if "sum_x" in z:
            sx = z["sum_x"] if sx is None else sx + z["sum_x"]
            sd = z["sum_d"] if sd is None else sd + z["sum_d"]
        for k in ("sample_x", "sample_d"):
            if k in z:
                samples[k] = z[k]
    if gxx is None:
        raise SystemExit("no gram files given")
    return gxx, gxd, gdd, sx, sd, n, samples


def ridge_solve(gxx, gxd, rel_lambda=1e-4):
    """W^T = (Gxx + lambda I)^-1 Gxd, lambda scaled to the average diagonal
    so the same rel_lambda works at any token count / activation scale."""
    lam = rel_lambda * np.trace(gxx) / gxx.shape[0]
    return np.linalg.solve(gxx + lam * np.eye(gxx.shape[0]), gxd).T, lam


def affine_solve(gxx, gxd, gdd, sum_x, sum_d, n, rel_lambda=1e-4):
    """Affine variant: d ~ x W^T + b, solved by centering. Returns W, b and
    the exact relative residual. DIAGNOSTIC ONLY: the .bin seam format has
    no bias slot, so this quantifies what a bias would buy before we ever
    extend the format."""
    mux = sum_x / n
    mud = sum_d / n
    gxx_c = gxx - n * np.outer(mux, mux)
    gxd_c = gxd - n * np.outer(mux, mud)
    w, _ = ridge_solve(gxx_c, gxd_c, rel_lambda)
    b = mud - mux @ w.T
    # exact: ||X W^T + b - D||^2 =
    #   ||X W^T - D||^2 + n |b|^2 + 2 b . (W sum_x - sum_d)
    num = (np.trace(w @ gxx @ w.T) - 2.0 * np.trace(w @ gxd)
           + np.trace(gdd) + n * float(b @ b)
           + 2.0 * float(b @ (w @ sum_x - sum_d)))
    rel = float(num / np.trace(gdd))
    return w, b, rel


def diagonal_fit(gxx, gxd, gdd, rel_lambda=1e-8):
    """Per-channel scale s_c = argmin sum (s_c x_c - d_c)^2 (diagonal ridge).
    DIAGNOSTIC: how much of the gap is pure per-channel gain (foldable into
    the next layer's input-norm gains, approximately)."""
    lam = rel_lambda * np.trace(gxx) / gxx.shape[0]
    s = np.diag(gxd) / (np.diag(gxx) + lam)
    num = (np.sum(s**2 * np.diag(gxx)) - 2.0 * np.sum(s * np.diag(gxd))
           + np.trace(gdd))
    return s, float(num / np.trace(gdd))


def svd_truncate(w, rank):
    """W ~ B A with A [rank, H], B [H, rank], the SeamAdapter layout."""
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    r = min(rank, s.shape[0])
    root = np.sqrt(s[:r])
    a = root[:, None] * vt[:r]      # [r, H]
    b = u[:, :r] * root[None, :]    # [H, r]
    return a.astype(np.float32), b.astype(np.float32), s


def gram_residual(w, gxx, gxd, gdd):
    """Exact ||X W^T - D||_F^2 / ||D||_F^2 from the sufficient statistics.
    ||XW^T - D||^2 = tr(W Gxx W^T) - 2 tr(W Gxd) + tr(Gdd)."""
    num = np.trace(w @ gxx @ w.T) - 2.0 * np.trace(w @ gxd) + np.trace(gdd)
    den = np.trace(gdd)
    return float(num / den) if den > 0 else float("nan")


def report(w_full, a, b, s, grams, n, samples, rank, sums=None,
           rel_lambda=1e-4):
    gxx, gxd, gdd = grams
    rel_full = gram_residual(w_full, gxx, gxd, gdd)
    rel_rank = gram_residual(b.astype(np.float64) @ a.astype(np.float64),
                             gxx, gxd, gdd)
    energy = np.cumsum(s**2) / np.sum(s**2)
    marks = {r: energy[r - 1] for r in (8, 16, 32, 64, 128, 256)
             if r <= s.shape[0]}
    print(f"tokens accumulated : {n}")
    print(f"rel residual, full-rank W   : {rel_full:.4f}")
    print(f"rel residual, rank-{rank:<4}     : {rel_rank:.4f}")
    print("energy kept by rank     : "
          + "  ".join(f"r{r}={v:.3f}" for r, v in marks.items()))
    if sums is not None and sums[0] is not None:
        _, _, rel_aff = affine_solve(gxx, gxd, gdd, sums[0], sums[1], n,
                                     rel_lambda)
        _, rel_diag = diagonal_fit(gxx, gxd, gdd)
        print(f"diagnostics  affine (bias)   : {rel_aff:.4f} "
              f"({'bias buys nothing' if rel_aff > rel_full - 1e-3 else 'a bias term would help - consider a format extension'})")
        print(f"diagnostics  diagonal scale  : {rel_diag:.4f} "
              f"({'not a scale problem' if rel_diag > 0.9 else 'a large share is per-channel gain'})")
    if "sample_x" in samples:
        sx = samples["sample_x"].astype(np.float64)
        sd = samples["sample_d"].astype(np.float64)
        pred = sx @ (b.astype(np.float64) @ a.astype(np.float64)).T
        rel = np.sum((pred - sd)**2) / np.sum(sd**2)
        print(f"held-out rel residual (rank-{rank}): {rel:.4f} "
              f"on {sx.shape[0]} pairs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gram", action="append", default=[],
                    help="gram .npz from a previous capture (repeatable, merged)")
    ap.add_argument("--raw", default=None, metavar="PREFIX.bounds.npy",
                    help="capture_bounds raw file: compute grams from raw "
                         "boundary residuals (needs --src-boundary/--dst-boundary)")
    ap.add_argument("--src-boundary", type=int, default=None,
                    help="x-side boundary: output of the last kept layer "
                         "before the cut (v3 cut 0-11,83-92: 12)")
    ap.add_argument("--dst-boundary", type=int, default=None,
                    help="y-side boundary: input of the first kept layer "
                         "after the cut (v3 cut: 83)")
    ap.add_argument("--gram-out", default=None,
                    help="also write the computed grams to this .npz")
    ap.add_argument("--rank", type=int, default=64,
                    help="adapter rank after SVD truncation (default 64, the "
                         "rank the v3 heal uses)")
    ap.add_argument("--rel-lambda", type=float, default=1e-4,
                    help="ridge penalty relative to the average diagonal of "
                         "Gxx (default 1e-4)")
    ap.add_argument("--out", default="stitch.pt",
                    help="output: state dict with seam_adapter.A/B (+ seam_after)")
    ap.add_argument("--seam-after", type=int, default=11,
                    help="0-based index of the layer after which the adapter "
                         "sits (default 11, the v3 cut)")
    ap.add_argument("--write-bin", default=None, metavar="SRC.bin",
                    help="deploy immediately: rewrite SRC.bin into --out with "
                         "the adapter embedded (exact --write-seam path)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.raw:
        if args.src_boundary is None or args.dst_boundary is None:
            ap.error("--raw needs --src-boundary and --dst-boundary")
        gxx, gxd, gdd, sx, sd, n, samples = grams_from_raw(
            args.raw, args.src_boundary, args.dst_boundary,
            gram_out=args.gram_out)
    elif args.gram:
        gxx, gxd, gdd, sx, sd, n, samples = load_grams(args.gram)
    else:
        ap.error("--gram or --raw is required (or --selftest)")

    h = gxx.shape[0]
    print(f"solving ridge on {h}x{h} from {n} tokens ...")
    w, lam = ridge_solve(gxx, gxd, args.rel_lambda)
    a, b, s = svd_truncate(w, args.rank)
    report(w, a, b, s, (gxx, gxd, gdd), n, samples, args.rank,
           sums=(sx, sd), rel_lambda=args.rel_lambda)

    if args.write_bin:
        import torch  # write_seam_bin expects torch tensors
        from apply_lora_bin import write_seam_bin
        sd = {"seam_adapter.A": torch.from_numpy(a),
              "seam_adapter.B": torch.from_numpy(b)}
        write_seam_bin(args.write_bin, args.out,
                       {"rank": args.rank, "after": args.seam_after}, sd)
        print(f"written: {args.out} (seam adapter rank {args.rank} after "
              f"layer {args.seam_after})")
    else:
        import torch
        torch.save({"seam_adapter.A": torch.from_numpy(a),
                    "seam_adapter.B": torch.from_numpy(b),
                    "seam_after": args.seam_after, "rank": args.rank},
                   args.out)
        print(f"written: {args.out} (deploy with apply_lora_bin.py "
              f"--write-seam, or use as heal init)")


def selftest():
    """Synthetic recovery, four checks: (1) ridge recovers a true rank-8
    map d = x C^T + noise; (2) gram shards merge; (3) raw boundary files
    yield the same grams as direct accumulation; (4) affine and diagonal
    diagnostics fire on data that truly has a bias / per-channel scale."""
    rng = np.random.default_rng(0)
    h, rank_true, n = 256, 8, 40000
    # anisotropic x, like a residual stream (a few directions dominate)
    scales = np.geomspace(1.0, 1e-2, h)
    x = rng.normal(size=(n, h)) * scales
    uc, _, vc = np.linalg.svd(rng.normal(size=(h, h)), full_matrices=False)
    c_true = (uc[:, :rank_true] * 3.0) @ vc[:rank_true]
    d = x @ c_true.T + 0.01 * rng.normal(size=(n, h))

    import tempfile, os, json
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
        half = GramAccumulator(h)
        half.add(x[: n // 2], d[: n // 2])
        half.save(p1)
        other = GramAccumulator(h)
        other.add(x[n // 2:], d[n // 2:])
        other.save(p2, sample_x=x[:512], sample_d=d[:512])
        gxx, gxd, gdd, sum_x, sum_d, n_loaded, samples = load_grams([p1, p2])
        assert n_loaded == n and sum_x is not None

        # (3) raw roundtrip: fake a 3-boundary capture, y = x + d at dst
        raw = np.zeros((n, 3, h), np.float16)
        raw[:, 0, :] = x
        raw[:, 2, :] = x + d
        raw_path = os.path.join(td, "cap.bounds.npy")
        np.save(raw_path, raw)
        with open(raw_path.replace(".bounds.npy", ".meta.json"), "w") as f:
            json.dump({"n_tokens": n, "n_bounds": 3, "hidden": h,
                       "n_layers": 2}, f)
        g2 = grams_from_raw(raw_path, 0, 2, holdout=512)
        rel_diff = np.linalg.norm(g2[0] - (gxx - x[:512].T @ x[:512])) \
            / np.linalg.norm(gxx)
        assert rel_diff < 0.02, f"raw grams diverge ({rel_diff:.4f})"

    w, _ = ridge_solve(gxx, gxd)
    err_map = np.linalg.norm(w - c_true) / np.linalg.norm(c_true)
    a8, b8, s = svd_truncate(w, 8)
    rel8 = gram_residual(b8.astype(np.float64) @ a8.astype(np.float64),
                         gxx, gxd, gdd)
    a64, b64, _ = svd_truncate(w, 64)
    rel64 = gram_residual(b64.astype(np.float64) @ a64.astype(np.float64),
                          gxx, gxd, gdd)
    noise_floor = 0.01**2 * h * n / np.trace(gdd)  # E||noise||^2 / ||D||^2
    print(f"map recovery |W - C_true|/|C_true| : {err_map:.4f}")
    print(f"rel residual rank-8                : {rel8:.5f}")
    print(f"rel residual rank-64               : {rel64:.5f}")
    print(f"noise floor                        : {noise_floor:.5f}")
    assert err_map < 0.08, f"ridge did not recover the map ({err_map:.4f})"
    assert rel8 < 0.02, f"rank-8 truncation lost too much ({rel8:.5f})"
    assert rel64 <= rel8 + 1e-9

    # (4a) affine: d = x C^T + b_true -> affine residual must beat linear
    b_true = rng.normal(size=h) * 0.5
    d_aff = d + b_true
    acc_a = GramAccumulator(h)
    acc_a.add(x, d_aff)
    rel_lin = gram_residual(w, acc_a.gxx, acc_a.gxd, acc_a.gdd)
    _, _, rel_aff = affine_solve(acc_a.gxx, acc_a.gxd, acc_a.gdd,
                                 acc_a.sum_x, acc_a.sum_d, acc_a.n)
    print(f"affine check: linear {rel_lin:.4f} -> affine {rel_aff:.4f}")
    assert rel_aff < rel_lin * 0.9, "affine did not capture the bias"

    # (4b) diagonal: d = s_true * x -> diagonal residual near zero
    s_true = rng.uniform(0.5, 2.0, size=h)
    d_diag = x * s_true + 0.01 * rng.normal(size=(n, h))
    acc_d = GramAccumulator(h)
    acc_d.add(x, d_diag)
    _, rel_diag = diagonal_fit(acc_d.gxx, acc_d.gxd, acc_d.gdd)
    print(f"diagonal check: rel residual {rel_diag:.5f}")
    assert rel_diag < 0.02, f"diagonal fit failed ({rel_diag:.5f})"
    print("stitch_solve selftest OK")


if __name__ == "__main__":
    main()
