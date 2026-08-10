#!/usr/bin/env python3
"""nanokimi - expert_solve: closed-form solve of host-shaped MoE experts
from a host capture, a donor capture and the donor FFN weights.

For each requested (host layer -> donor layer) pair this tool:
  1. pairs host and donor tokens by byte anchor (positions where both
     tokenizations close a token at the same byte offset);
  2. solves the two state stitches by ridge regression on the paired
     activations: latent->donor-input (folded into the donor gate/up
     matrices) and donor->latent (projects the donor FFN delta into the
     host latent space as the target);
  3. ranks donor FFN neurons by contribution on the host stream and keeps
     the top moe_inter per expert band;
  4. re-solves the down matrix against the projected donor delta THROUGH
     the engine's SiTU activation - absorbing the output map, the donor/
     host activation mismatch and the neuron slice in one regression;
  5. solves a router row predicting the size of the projected donor
     contribution from the router input stream.

Everything is closed form (streamed grams + ridge); no gradient step. The
result is a graft pack (npz) of host-native tensors. `n_experts` is global in
the .bin format, so a complete pack covers every MoE layer of the host (one
`--map` entry per layer).

usage:
  python3 expert_solve.py --host-capture cap/host --donor-capture cap/don \
      --donor-weights /path/to/checkpoint --map 1:6,2:12,3:18 \
      --bands 1 --out graft.npz
  python3 expert_solve.py --selftest

Diagnostics per layer: CKA between the streams (how alignable they are at
all), relative residuals of each solve, and the holdout error of the final
expert against the projected donor delta (1.0 = no better than silence).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graftlib import (cka_scan, match_anchors, open_capture,
                      read_safetensors_dir, solve_graft)  # noqa: E402


def parse_map(s):
    out = []
    for part in s.split(","):
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def scan_map(hmeta, hplanes, dmeta, dplanes, ih, idz, sample, log=print):
    """Chooses one donor layer per host port by centered linear CKA of the
    host latent stream against each donor FFN DELTA, over a subsample of
    anchor pairs (closed form, no solve). The delta is the quantity the
    grafted expert must reproduce, so its predictability from the host
    stream is the binding constraint; state streams are misleading here
    (early-layer states align with everything while their deltas carry
    nothing transferable). Returns the host->donor map."""
    donor_layers = sorted(dmeta["layers"])
    donor_dz = {dl: dplanes[f"L{dl}.dz"] for dl in donor_layers}
    layer_map = []
    for hl in hmeta["layers"]:
        scores = cka_scan(hplanes[f"L{hl}.lat"], donor_dz, ih, idz, sample)
        best = max(scores, key=scores.get)
        row = " ".join(f"L{dl}:{scores[dl]:.3f}" for dl in donor_layers)
        log(f"scan L{hl} (delta cka): {row} -> donor L{best}")
        layer_map.append((hl, best))
    return layer_map


def solve_all(host_prefix, donor_prefix, donor_weights_path, layer_map,
              bands=1, rel_lambda=1e-4, holdout=4096, max_pairs=0,
              moe_inter=None, log=print, target="donor",
              scan_sample=30000):
    hmeta, hends, hmask, hplanes = open_capture(host_prefix)
    dmeta, dends, dmask, dplanes = open_capture(donor_prefix)
    mi = moe_inter or hmeta["moe_inter"]
    wtmpl = dmeta["weights"]

    ih, idz = match_anchors(hends, hmask, dends, dmask)
    if max_pairs and len(ih) > max_pairs:
        ih, idz = ih[:max_pairs], idz[:max_pairs]
    log(f"anchors: {len(ih)} pairs "
        f"({hmeta['n_tokens']} host / {dmeta['n_tokens']} donor tokens)")
    if len(ih) < 10 * holdout:
        log(f"  note: few pairs for holdout {holdout}")

    if layer_map == "scan":
        layer_map = scan_map(hmeta, hplanes, dmeta, dplanes, ih, idz,
                             scan_sample, log)

    pack = {}
    report = {}
    g_next = {}
    for hl, dl in layer_map:
        names = {k: wtmpl[k].format(l=dl) for k in ("gate", "up", "down")}
        w = read_safetensors_dir(donor_weights_path, list(names.values()))
        donor_w = {k: w[names[k]] for k in names}
        y_moe = None
        if target == "diff":
            if f"L{hl}.moe" not in hplanes:
                raise SystemExit(f"--target diff needs the L{hl}.moe plane "
                                 "(re-capture the host with a tool version "
                                 "that records the MoE mix output)")
            y_moe = hplanes[f"L{hl}.moe"]
        out = solve_graft(
            hplanes[f"L{hl}.lat"], hplanes[f"L{hl}.pln"],
            dplanes[f"L{dl}.in"], dplanes[f"L{dl}.dz"], donor_w, mi,
            bands=bands, rel_lambda=rel_lambda, holdout=holdout,
            ih=ih, idz=idz, y_moe=y_moe)
        g0 = g_next.get(hl, 0)
        for g, (w1, w3, w2, gate_row) in enumerate(out["experts"], g0):
            pack[f"L{hl}.g{g}.w1"] = w1
            pack[f"L{hl}.g{g}.w3"] = w3
            pack[f"L{hl}.g{g}.w2"] = w2
            pack[f"L{hl}.g{g}.gate"] = gate_row
            # donor-to-latent stitch: lets later closed-form passes rebuild
            # the target (e.g. a re-solve restricted to routed positions)
            pack[f"L{hl}.g{g}.m_out"] = out["m_out"]
            pack[f"L{hl}.g{g}.donor_layer"] = np.int64(dl)
        g_next[hl] = g0 + len(out["experts"])
        d = out["diag"]
        report[f"{hl}:{dl}"] = {**{k: d[k] for k in
                                ("cka", "res_s_in", "res_m_out", "bands")}}
        band_txt = " ".join(f"g{g}:hold={b['rel_holdout']:.3f}"
                            for g, b in enumerate(d["bands"], g0))
        log(f"L{hl} <- donor L{dl}: cka {d['cka']:.3f}, "
            f"stitch res in/out {d['res_s_in']:.3f}/{d['res_m_out']:.3f}, "
            f"{band_txt}")
    counts = set(g_next.values())
    if len(counts) != 1:
        raise SystemExit(f"uneven experts per layer {g_next}: n_experts is "
                         "global, every mapped layer needs the same count")
    total_bands = counts.pop()
    meta = {"bands": total_bands, "rel_lambda": rel_lambda,
            "host": hmeta.get("bin"), "donor": dmeta.get("model"),
            "map": layer_map, "target": target, "report": report}
    pack["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    return pack, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-capture", required=True)
    ap.add_argument("--donor-capture", required=True)
    ap.add_argument("--donor-weights", required=True,
                    help="safetensors file or checkpoint directory")
    ap.add_argument("--map", required=True,
                    help="hostLayer:donorLayer,... A host layer may appear "
                    "several times (one expert per occurrence); an "
                    "injectable pack needs every MoE layer with the same "
                    "expert count. 'scan' picks the best donor layer per "
                    "port by CKA over the captured donor layers")
    ap.add_argument("--scan-sample", type=int, default=30000)
    ap.add_argument("--bands", type=int, default=1,
                    help="experts per layer, successive neuron bands")
    ap.add_argument("--rel-lambda", type=float, default=1e-4)
    ap.add_argument("--holdout", type=int, default=4096)
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--moe-inter", type=int, default=None)
    ap.add_argument("--target", default="donor", choices=["donor", "diff"],
                    help="donor: projected donor FFN delta; diff: that "
                    "delta MINUS the host bank's own mix output (the "
                    "expert only encodes what the host lacks)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lm = "scan" if args.map == "scan" else parse_map(args.map)
    pack, meta = solve_all(args.host_capture, args.donor_capture,
                           args.donor_weights, lm,
                           args.bands, args.rel_lambda, args.holdout,
                           args.max_pairs, args.moe_inter,
                           target=args.target, scan_sample=args.scan_sample)
    np.savez(args.out, **pack)
    print(f"-> {args.out}: {len(pack) - 1} tensors, bands {meta['bands']}")


# ------------------------------------------------------------------ selftest

def _write_st(path, tensors):
    header = {}
    off = 0
    blobs = []
    for n, a in tensors.items():
        b = np.ascontiguousarray(a, np.float32).tobytes()
        header[n] = {"dtype": "F32", "shape": list(a.shape),
                     "data_offsets": [off, off + len(b)]}
        off += len(b)
        blobs.append(b)
    hb = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(len(hb).to_bytes(8, "little"))
        f.write(hb)
        for b in blobs:
            f.write(b)


def selftest():
    import tempfile
    from graftlib import CaptureWriter, pack_ends

    rng = np.random.default_rng(23)
    n, rh, d, d_d, d_ff, mi = 5000, 16, 32, 24, 64, 16

    # a host stream and a donor stream that are linear images of a common
    # latent cause, written as two captures with interleaved anchors: the
    # donor tokenization "splits" every 5th host token in two
    basis = rng.normal(size=(rh, rh)) * np.linspace(2, 0.3, rh)
    lat = rng.normal(size=(n, rh)) @ basis
    tmap = rng.normal(size=(d_d, rh)) / np.sqrt(rh)
    z = lat @ tmap.T + 0.03 * rng.normal(size=(n, d_d))
    pln = np.repeat(lat, 2, axis=1)[:, :d]
    w_gate = rng.normal(size=(d_ff, d_d)) / np.sqrt(d_d)
    w_up = rng.normal(size=(d_ff, d_d)) / np.sqrt(d_d)
    w_down = rng.normal(size=(d_d, d_ff)) / np.sqrt(d_ff)
    a = z @ w_gate.T
    dz = ((a / (1 + np.exp(-a))) * (z @ w_up.T)) @ w_down.T  # silu donor

    with tempfile.TemporaryDirectory() as td:
        hp = os.path.join(td, "host")
        dp = os.path.join(td, "don")
        hw = CaptureWriter(hp, {"L1.lat": rh, "L1.pln": d},
                           {"kind": "host", "moe_inter": mi, "bin": "h"})
        ends = np.arange(1, n + 1, dtype=np.int64) * 3
        hw.add(pack_ends(0, ends), np.ones(n, bool),
               {"L1.lat": lat, "L1.pln": pln})
        hw.close()
        # donor: same stream plus an extra token between anchors (unmatched
        # byte end) every 5 positions
        keep = np.ones(n, bool)
        extra_at = np.arange(0, n, 5)
        d_ends, d_in, d_dz, d_mask = [], [], [], []
        for i in range(n):
            d_ends.append(ends[i])
            d_in.append(z[i])
            d_dz.append(dz[i])
            d_mask.append(True)
            if i in extra_at:
                d_ends.append(ends[i] + 1)  # never a host end (ends are *3)
                d_in.append(rng.normal(size=d_d))
                d_dz.append(rng.normal(size=d_d))
                d_mask.append(True)
        # layer 4 carries the real stream; layer 9 is pure noise, so a CKA
        # scan must prefer 4
        dw = CaptureWriter(
            dp, {"L4.in": d_d, "L4.dz": d_d, "L9.in": d_d, "L9.dz": d_d},
            {"kind": "donor", "model": "toy", "layers": [4, 9],
             "weights": {"gate": "l{l}.gate", "up": "l{l}.up",
                         "down": "l{l}.down"}})
        nz = len(d_in)
        dw.add(np.asarray(d_ends, np.uint64) | np.uint64(0),
               d_mask, {"L4.in": np.asarray(d_in),
                        "L4.dz": np.asarray(d_dz),
                        "L9.in": rng.normal(size=(nz, d_d)),
                        "L9.dz": rng.normal(size=(nz, d_d))})
        dw.close()

        st = os.path.join(td, "donor.safetensors")
        _write_st(st, {"l4.gate": w_gate, "l4.up": w_up, "l4.down": w_down})

        pack, meta = solve_all(hp, dp, st, [(1, 4)], bands=1,
                               holdout=800, log=lambda *a: None)
        rep = meta["report"]["1:4"]
        print(f"claim 1: anchors filtered the unmatched donor tokens, solve "
              f"ran (cka {rep['cka']:.3f})")
        assert rep["cka"] > 0.4

        hold = rep["bands"][0]["rel_holdout"]
        print(f"claim 2: solved expert beats silence on holdout "
              f"({hold:.3f} < 1)")
        assert hold < 0.7, hold

        assert pack["L1.g0.w1"].shape == (mi, rh)
        assert pack["L1.g0.w2"].shape == (rh, mi)
        assert pack["L1.g0.gate"].shape == (d,)
        print("claim 3: pack tensors are host-native shapes")

        # determinism: same inputs, same pack
        pack2, _ = solve_all(hp, dp, st, [(1, 4)], bands=1, holdout=800,
                             log=lambda *a: None)
        assert np.array_equal(pack["L1.g0.w2"], pack2["L1.g0.w2"])
        print("claim 4: solve is deterministic")

        # a repeated host layer stacks experts g0, g1
        pack3, meta3 = solve_all(hp, dp, st, [(1, 4), (1, 4)], bands=1,
                                 holdout=800, log=lambda *a: None)
        assert meta3["bands"] == 2
        assert "L1.g1.w1" in pack3
        assert np.array_equal(pack3["L1.g0.w1"], pack3["L1.g1.w1"])
        print("claim 5: repeated host layer stacks experts (bands 2)")

        assert pack["L1.g0.m_out"].shape == (rh, d_d)
        assert int(pack["L1.g0.donor_layer"]) == 4
        print("claim 6: pack carries the donor-to-latent stitch and the "
              "donor layer")

        # host meta needs its layer list for the scan entry point
        with open(hp + ".meta.json") as f:
            hm = json.load(f)
        hm["layers"] = [1]
        with open(hp + ".meta.json", "w") as f:
            json.dump(hm, f)
        pack4, meta4 = solve_all(hp, dp, st, "scan", bands=1, holdout=800,
                                 scan_sample=2000, log=lambda *a: None)
        assert meta4["map"] == [(1, 4)], meta4["map"]
        print("claim 7: CKA scan picks the informative donor layer "
              "(4, not the noise layer 9)")

    print("expert_solve selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
