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
  python3 stitch_solve.py --gram seam_gram.npz --write-bin model.bin \
      --out stitched.bin --seam-after 11
  python3 stitch_solve.py --selftest
"""
import argparse
import sys

import numpy as np


class GramAccumulator:
    """Streaming sufficient statistics for the ridge solve. float64: the
    residual stream has a few huge-norm directions and float32 accumulation
    over millions of tokens would drown the small ones."""

    def __init__(self, hidden):
        self.h = hidden
        self.gxx = np.zeros((hidden, hidden), np.float64)
        self.gxd = np.zeros((hidden, hidden), np.float64)
        self.gdd = np.zeros((hidden, hidden), np.float64)
        self.n = 0

    def add(self, x, d):
        """x, d: [tokens, hidden] float arrays (any float dtype)."""
        x = np.asarray(x, np.float64)
        d = np.asarray(d, np.float64)
        assert x.shape == d.shape and x.shape[1] == self.h
        self.gxx += x.T @ x
        self.gxd += x.T @ d
        self.gdd += d.T @ d
        self.n += x.shape[0]

    def save(self, path, sample_x=None, sample_d=None):
        out = {"Gxx": self.gxx, "Gxd": self.gxd, "Gdd": self.gdd,
               "n": np.int64(self.n)}
        if sample_x is not None:
            out["sample_x"] = np.asarray(sample_x, np.float32)
            out["sample_d"] = np.asarray(sample_d, np.float32)
        np.savez(path, **out)


def load_grams(paths):
    """Load and merge one or more gram .npz files (plain addition)."""
    gxx = gxd = gdd = None
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
        for k in ("sample_x", "sample_d"):
            if k in z:
                samples[k] = z[k]
    if gxx is None:
        raise SystemExit("no gram files given")
    return gxx, gxd, gdd, n, samples


def ridge_solve(gxx, gxd, rel_lambda=1e-4):
    """W^T = (Gxx + lambda I)^-1 Gxd, lambda scaled to the average diagonal
    so the same rel_lambda works at any token count / activation scale."""
    h = gxx.shape[0]
    lam = rel_lambda * np.trace(gxx) / h
    return np.linalg.solve(gxx + lam * np.eye(gxx.shape[0]), gxd).T, lam


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


def report(w_full, a, b, s, grams, n, samples, rank):
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
                    help="gram .npz from the capture tool (repeatable, merged)")
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
    if not args.gram:
        ap.error("--gram is required (or --selftest)")

    gxx, gxd, gdd, n, samples = load_grams(args.gram)
    h = gxx.shape[0]
    print(f"solving ridge on {h}x{h} from {n} tokens "
          f"({len(args.gram)} gram file(s)) ...")
    w, lam = ridge_solve(gxx, gxd, args.rel_lambda)
    a, b, s = svd_truncate(w, args.rank)
    report(w, a, b, s, (gxx, gxd, gdd), n, samples, args.rank)

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
    """Synthetic recovery: d = x C^T + noise with a true rank-8 C. The
    ridge solve must recover it, rank-8 truncation must keep nearly all
    energy, and the A/B split must reproduce the truncated map."""
    rng = np.random.default_rng(0)
    h, rank_true, n = 256, 8, 40000
    # anisotropic x, like a residual stream (a few directions dominate)
    scales = np.geomspace(1.0, 1e-2, h)
    x = rng.normal(size=(n, h)) * scales
    uc, _, vc = np.linalg.svd(rng.normal(size=(h, h)), full_matrices=False)
    c_true = (uc[:, :rank_true] * 3.0) @ vc[:rank_true]
    d = x @ c_true.T + 0.01 * rng.normal(size=(n, h))

    acc = GramAccumulator(h)
    acc.add(x[: n // 2], d[: n // 2])   # two shards, to exercise merging
    acc.add(x[n // 2:], d[n // 2:])
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p1, p2 = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
        half = GramAccumulator(h)
        half.add(x[: n // 2], d[: n // 2])
        half.save(p1)
        other = GramAccumulator(h)
        other.add(x[n // 2:], d[n // 2:])
        other.save(p2, sample_x=x[:512], sample_d=d[:512])
        gxx, gxd, gdd, n_loaded, samples = load_grams([p1, p2])
    assert n_loaded == n
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
    print("stitch_solve selftest OK")


if __name__ == "__main__":
    main()
