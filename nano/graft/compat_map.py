#!/usr/bin/env python3
"""nanokimi - compat_map: cross-model compatibility map.

Measures, for every ordered pair of captured models and every pair of
their layers, how well one model's FFN delta is predictable from the
other's stream. The score is the ridge R2 of that prediction, computed
in closed form from streaming Gram statistics - the same criterion
expert_solve uses to choose a source layer, applied exhaustively.

What the number means: 1.0 means the second model's contribution at that
layer is a linear function of the first model's state, so anything the
second model computes there is expressible in the first one's geometry;
0.0 means the two carry unrelated information. Directional by
construction: predicting A from B is not predicting B from A, and the
asymmetry is informative (a larger model's delta is usually harder to
predict from a smaller one's stream than the reverse).

Inputs are captures written by capture_donor.py / capture_host.py over
THE SAME TEXT; positions are paired by byte anchors, so tokenizers may
differ.

usage:
  python3 compat_map.py --captures a=cap/qwen08 b=cap/q122 c=cap/gemma \
      --out map.json [--sample 30000] [--layers-per-model 8]
  python3 compat_map.py --report map.json
  python3 compat_map.py --selftest
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graftlib import RectGram, match_anchors, open_capture  # noqa: E402


def layer_ids(meta, planes, limit=None):
    ids = sorted({int(k.split(".")[0][1:]) for k in planes
                  if k.endswith(".dz") or k.endswith(".in")})
    if limit and len(ids) > limit:
        step = len(ids) / limit
        ids = [ids[int(i * step)] for i in range(limit)]
    return ids


def pair_scores(src, dst, sample=30000, chunk=8192, rel_lambda=1e-4,
                limit=None, log=print):
    """R2 of predicting dst's FFN delta from src's stream, for every
    (src layer, dst layer). Returns {"L{a}->L{b}": r2}."""
    (_ma, ea, ka, pa), (_mb, eb, kb, pb) = src, dst
    ia, ib = match_anchors(ea, ka, eb, kb)
    n = min(sample, len(ia))
    if n < 1000:
        log(f"  only {n} anchor pairs, scores will be noisy")
    la = layer_ids(_ma, pa, limit)
    lb = layer_ids(_mb, pb, limit)
    out = {}
    for a in la:
        xs = pa.get(f"L{a}.in")
        if xs is None:
            continue
        for b in lb:
            ys = pb.get(f"L{b}.dz")
            if ys is None:
                continue
            g = RectGram(xs.shape[1], ys.shape[1])
            for t0 in range(0, n, chunk):
                g.add(np.asarray(xs[ia[t0:t0 + chunk]], np.float64),
                      np.asarray(ys[ib[t0:t0 + chunk]], np.float64))
            _w, res = g.solve(rel_lambda)
            out[f"L{a}->L{b}"] = round(1.0 - res, 4)
    return out, len(ia)


def build_map(captures, sample, limit, log=print):
    caps = {name: open_capture(p) for name, p in captures.items()}
    result = {"models": {n: {"tokens": c[0]["n_tokens"],
                             "source": c[0].get("model") or c[0].get("bin")}
                         for n, c in caps.items()},
              "pairs": {}}
    for a, b in itertools.permutations(caps, 2):
        scores, n_anchor = pair_scores(caps[a], caps[b], sample, limit=limit,
                                       log=log)
        if not scores:
            continue
        best = max(scores, key=scores.get)
        result["pairs"][f"{a}->{b}"] = {
            "anchors": int(n_anchor), "scores": scores,
            "best": best, "best_r2": scores[best],
            "mean_r2": round(float(np.mean(list(scores.values()))), 4)}
        log(f"{a} -> {b}: best {best} R2 {scores[best]:.3f}, "
            f"mean {np.mean(list(scores.values())):.3f} "
            f"({n_anchor} anchors)", flush=True)
    return result


def report(path):
    with open(path) as f:
        m = json.load(f)
    names = sorted(m["models"])
    print("models: " + ", ".join(f"{n} ({m['models'][n]['tokens']} tok)"
                                 for n in names))
    w = max(len(n) for n in names) + 1
    print("\nbest R2, row = source of the stream, column = model whose "
          "delta is predicted\n")
    print(" " * w + "".join(f"{n:>8}" for n in names))
    for a in names:
        row = f"{a:<{w}}"
        for b in names:
            k = f"{a}->{b}"
            row += f"{m['pairs'][k]['best_r2']:>8.3f}" if k in m["pairs"] \
                else f"{'-':>8}"
        print(row)
    print("\nasymmetry (best R2 a->b minus b->a):")
    for a, b in itertools.combinations(names, 2):
        ab, ba = f"{a}->{b}", f"{b}->{a}"
        if ab in m["pairs"] and ba in m["pairs"]:
            d = m["pairs"][ab]["best_r2"] - m["pairs"][ba]["best_r2"]
            print(f"  {a} vs {b}: {d:+.3f}"
                  + ("" if abs(d) < 0.05 else
                     f"  ({b if d > 0 else a} is the easier target)"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", nargs="*", default=[],
                    help="name=prefix pairs")
    ap.add_argument("--out", default="compat_map.json")
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--layers-per-model", type=int, default=8)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    caps = dict(c.split("=", 1) for c in args.captures)
    if not caps:
        raise SystemExit("give --captures name=prefix ...")
    m = build_map(caps, args.sample, args.layers_per_model)
    with open(args.out, "w") as f:
        json.dump(m, f, indent=1)
    print(f"-> {args.out}: {len(m['pairs'])} ordered pairs")
    report(args.out)


# ------------------------------------------------------------------ selftest

def selftest():
    import tempfile
    from graftlib import CaptureWriter, pack_ends

    rng = np.random.default_rng(17)
    n, d = 4000, 24
    lat = rng.normal(size=(n, d))
    ends = np.arange(1, n + 1, dtype=np.int64) * 3

    def write(prefix, dz_map, extra_layer_noise):
        w = CaptureWriter(prefix, {"L1.in": d, "L1.dz": d,
                                   "L5.in": d, "L5.dz": d},
                          {"kind": "test", "layers": [1, 5], "model": prefix})
        w.add(pack_ends(0, ends), np.ones(n, bool),
              {"L1.in": lat, "L1.dz": lat @ dz_map,
               "L5.in": lat + 0.1 * rng.normal(size=(n, d)),
               "L5.dz": extra_layer_noise})
        w.close()

    with tempfile.TemporaryDirectory() as td:
        a = os.path.join(td, "a")
        b = os.path.join(td, "b")
        # b's L1 delta is a clean linear image of the shared cause;
        # its L5 delta is pure noise
        write(a, np.eye(d), rng.normal(size=(n, d)))
        write(b, rng.normal(size=(d, d)) / np.sqrt(d),
              rng.normal(size=(n, d)))
        m = build_map({"a": a, "b": b}, 4000, None, log=lambda *x, **k: None)

        s = m["pairs"]["a->b"]["scores"]
        print(f"claim 1: predictable delta scores high "
              f"(a L1 -> b L1: {s['L1->L1']:.3f})")
        assert s["L1->L1"] > 0.9, s["L1->L1"]

        print(f"claim 2: noise delta scores low "
              f"(a L1 -> b L5: {s['L1->L5']:.3f})")
        assert s["L1->L5"] < 0.2, s["L1->L5"]

        assert m["pairs"]["a->b"]["best"] == "L1->L1"
        print("claim 3: the map names the best source/target layer pair")

        assert set(m["pairs"]) == {"a->b", "b->a"}
        print("claim 4: both directions are measured (the score is "
              "directional, so the pair is ordered)")

        out = os.path.join(td, "m.json")
        with open(out, "w") as f:
            json.dump(m, f)
        report(out)
        print("claim 5: report renders the matrix and the asymmetries")

    print("compat_map selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
