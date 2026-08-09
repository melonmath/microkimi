#!/usr/bin/env python3
"""nanokimi - inject_experts: splice extra routed experts into a microkimi
.bin (MKIM0002).

Takes a graft pack containing host-shaped expert tensors and router rows per
MoE layer, then rewrites the container with:

  - G new MXFP4 experts appended to the expert bank
    (block_sparse_moe.experts.{E+g}.{w1,w2,w3});
  - G new rows appended to block_sparse_moe.gate.weight;
  - G new entries appended to gate.e_score_correction_bias, set to
    --birth-bias: strongly negative means the new experts are (almost)
    never in the top-k until something raises the bias, so the model's
    behavior is preserved at injection time (the correction bias affects
    selection only, not the mixing weights).

The config's n_experts is bumped by G. n_experts is a global of the
format, so the pack must cover EVERY MoE layer with the same expert
count; expert blobs keep the host's [moe_inter, routed_hidden] /
[routed_hidden, moe_inter] shapes, hence the engine needs no change and
expert streaming keeps working. Original blobs are copied byte for byte;
offsets are recomputed with the container's alignment rule (experts
4096 or 64, detected from the source, everything else 64).

usage:
  python3 inject_experts.py --bin host.bin --pack graft.npz --out output.bin \
      [--birth-bias -8.0] [--gate-scale 1.0]
  python3 inject_experts.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)
from bin2pt import DTYPE_F32, DTYPE_MXFP4, read_bin, dequant_mxfp4  # noqa: E402
from export import quantize_mxfp4  # noqa: E402


def moe_layers_of(config):
    n = config["n_layers"]
    v = config.get("dense_layers")  # may be null in the embedded config
    dense = set(v if v is not None else range(config.get("first_k_dense", 0)))
    return [l for l in range(n) if l not in dense]


def load_pack(path):
    """Graft pack: npz with, per MoE layer l and band g,
    L{l}.g{g}.w1 [mi, rh], .w3, .w2 [rh, mi], .gate [d]; plus meta json
    under key "meta" (bands, source note)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(bytes(z["meta"]).decode())
    return z, meta


def mxfp4_blob(w):
    packed, scales = quantize_mxfp4(np.ascontiguousarray(w, np.float32))
    return packed.tobytes() + scales.tobytes()


def inject(src, dst, pack_path, birth_bias, gate_scale):
    config, entries, f = read_bin(src)
    z, meta = load_pack(pack_path)
    bands = int(meta["bands"])
    moe = moe_layers_of(config)
    e0 = config["n_experts"]
    mi, rh, d = config["moe_inter"], config["routed_hidden"], config["hidden"]

    for l in moe:
        for g in range(bands):
            for nm, shape in (("w1", (mi, rh)), ("w3", (mi, rh)),
                              ("w2", (rh, mi))):
                k = f"L{l}.g{g}.{nm}"
                if k not in z:
                    raise SystemExit(
                        f"pack misses {k}: n_experts is global, every MoE "
                        f"layer needs the same {bands} grafted expert(s)")
                if z[k].shape != shape:
                    raise SystemExit(f"{k}: shape {z[k].shape} != {shape}")
            if z[f"L{l}.g{g}.gate"].shape != (d,):
                raise SystemExit(f"L{l}.g{g}.gate: bad shape")

    by_name = {n: (n, dt, dims, off, size)
               for n, dt, dims, off, size in entries}

    def is_expert(name):
        return (".block_sparse_moe.experts." in name
                and name.rsplit(".", 1)[-1] in ("w1", "w2", "w3"))

    ex_offsets = [o for n, _, _, o, _ in entries if is_expert(n)]
    expert_align = 4096 if ex_offsets and all(o % 4096 == 0
                                              for o in ex_offsets) else 64

    # output tensor list: original order, gate tensors replaced, new
    # experts appended at the end. source = old offset (copy) or bytes.
    out = []
    for n, dt, dims, off, size in entries:
        if n.endswith("block_sparse_moe.gate.weight"):
            l = int(n.split(".")[1])
            old = np.frombuffer(_read(f, off, size), np.float32).reshape(dims)
            rows = np.stack([z[f"L{l}.g{g}.gate"] * gate_scale
                             for g in range(bands)]).astype(np.float32)
            neww = np.concatenate([old, rows], axis=0)
            out.append((n, dt, [e0 + bands, dims[1]], neww.tobytes()))
        elif n.endswith("gate.e_score_correction_bias"):
            old = np.frombuffer(_read(f, off, size), np.float32)
            newb = np.concatenate([old, np.full(bands, birth_bias,
                                                np.float32)])
            out.append((n, dt, [e0 + bands], newb.tobytes()))
        else:
            out.append((n, dt, dims, (off, size)))
    for l in moe:
        p = f"layers.{l}.block_sparse_moe.experts."
        for g in range(bands):
            e = e0 + g
            out.append((p + f"{e}.w1", DTYPE_MXFP4, [mi, rh],
                        mxfp4_blob(z[f"L{l}.g{g}.w1"])))
            out.append((p + f"{e}.w2", DTYPE_MXFP4, [rh, mi],
                        mxfp4_blob(z[f"L{l}.g{g}.w2"])))
            out.append((p + f"{e}.w3", DTYPE_MXFP4, [mi, rh],
                        mxfp4_blob(z[f"L{l}.g{g}.w3"])))

    cfg = dict(config)
    cfg["n_experts"] = e0 + bands
    # first injection records the pre-graft bank size; later tools (heal)
    # use it to tell grafted experts from the original bank. The engine
    # ignores unknown config keys.
    cfg.setdefault("graft_base_experts", e0)
    cfg_bytes = json.dumps(cfg).encode()
    dir_size = sum(2 + len(n.encode()) + 1 + 1 + 4 * len(dims) + 8 + 8
                   for n, _, dims, _ in out)
    pos = 8 + 4 + len(cfg_bytes) + 4 + dir_size
    offsets = []
    for n, _, _, src_ in out:
        size = len(src_) if isinstance(src_, bytes) else src_[1]
        align = expert_align if is_expert(n) else 64
        pos = (pos + align - 1) // align * align
        offsets.append(pos)
        pos += size
    with open(dst, "wb") as o:
        o.write(b"MKIM0002")
        o.write(len(cfg_bytes).to_bytes(4, "little"))
        o.write(cfg_bytes)
        o.write(len(out).to_bytes(4, "little"))
        for (n, dt, dims, src_), off in zip(out, offsets):
            nb = n.encode()
            size = len(src_) if isinstance(src_, bytes) else src_[1]
            o.write(len(nb).to_bytes(2, "little"))
            o.write(nb)
            o.write(bytes([dt, len(dims)]))
            for dim in dims:
                o.write(int(dim).to_bytes(4, "little"))
            o.write(off.to_bytes(8, "little"))
            o.write(size.to_bytes(8, "little"))
        for (n, _, _, src_), off in zip(out, offsets):
            if o.tell() < off:
                o.write(b"\0" * (off - o.tell()))
            if isinstance(src_, bytes):
                o.write(src_)
            else:
                f.seek(src_[0])
                left = src_[1]
                while left:
                    chunk = f.read(min(left, 1 << 26))
                    o.write(chunk)
                    left -= len(chunk)
    f.close()
    return e0 + bands, len(moe), pos


def _read(f, off, size):
    f.seek(off)
    return f.read(size)


# ------------------------------------------------------------------ selftest

def selftest():
    import tempfile
    sys.path.insert(0, _NANO)
    from export import write_bin

    rng = np.random.default_rng(11)
    # mxfp4 groups are 32 wide: rh and mi must be multiples of 32
    n_layers, d, rh, mi, e0, v = 3, 48, 32, 32, 4, 64
    cfg = {"format": 2, "n_layers": n_layers, "hidden": d, "vocab": v,
           "n_experts": e0, "top_k": 2, "n_shared": 1,
           "routed_hidden": rh, "moe_inter": mi,
           "mla_layers": [2], "dense_layers": [0], "first_k_dense": 1}

    def t(name, shape, dtype=DTYPE_F32):
        w = rng.normal(size=shape).astype(np.float32)
        if dtype == DTYPE_MXFP4:
            p, s = quantize_mxfp4(w)
            return (name, dtype, list(shape), p.tobytes() + s.tobytes()), w
        return (name, dtype, list(shape), w.tobytes()), w

    tensors, ref = [], {}
    for name, shape in [("embed_tokens.weight", (v, d)),
                        ("norm.weight", (d,))]:
        e, w = t(name, shape)
        tensors.append(e)
        ref[name] = w
    for l in (1, 2):
        m = f"layers.{l}.block_sparse_moe."
        e, w = t(m + "gate.weight", (e0, d))
        tensors.append(e)
        ref[m + "gate.weight"] = w
        e, w = t(m + "gate.e_score_correction_bias", (e0,))
        tensors.append(e)
        ref[m + "gate.e_score_correction_bias"] = w
        for x in range(e0):
            for nm, shape in (("w1", (mi, rh)), ("w2", (rh, mi)),
                              ("w3", (mi, rh))):
                e, w = t(m + f"experts.{x}.{nm}", shape, DTYPE_MXFP4)
                tensors.append(e)
                ref[m + f"experts.{x}.{nm}"] = w

    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "host.bin")
        dst = os.path.join(td, "graft.bin")
        pk = os.path.join(td, "pack.npz")
        write_bin(src, cfg, tensors)

        pack = {"meta": np.frombuffer(json.dumps({"bands": 2}).encode(),
                                      np.uint8)}
        want = {}
        for l in (1, 2):
            for g in range(2):
                for nm, shape in (("w1", (mi, rh)), ("w3", (mi, rh)),
                                  ("w2", (rh, mi))):
                    w = rng.normal(size=shape).astype(np.float32)
                    pack[f"L{l}.g{g}.{nm}"] = w
                    want[(l, e0 + g, nm)] = w
                pack[f"L{l}.g{g}.gate"] = rng.normal(size=d).astype(np.float32)
        np.savez(pk, **pack)

        n_new, n_moe, _ = inject(src, dst, pk, birth_bias=-9.0,
                                 gate_scale=0.5)
        assert (n_new, n_moe) == (6, 2)

        cfg2, entries2, f2 = read_bin(dst)
        assert cfg2["n_experts"] == 6
        assert cfg2["graft_base_experts"] == 4
        print("claim 1: config n_experts bumped 4 -> 6, base bank recorded")

        e2 = {n: (dt, dims, off, size) for n, dt, dims, off, size in entries2}
        for name, w in ref.items():
            dt, dims, off, size = e2[name]
            f2.seek(off)
            blob = f2.read(size)
            if dt == DTYPE_MXFP4:
                got = dequant_mxfp4(blob, dims)
                exp = dequant_mxfp4(
                    quantize_mxfp4(w)[0].tobytes()
                    + quantize_mxfp4(w)[1].tobytes(), list(w.shape))
                if ".gate." not in name:
                    assert np.array_equal(got, exp), name
            else:
                got = np.frombuffer(blob, np.float32).reshape(dims)
                if name.endswith("gate.weight"):
                    assert np.array_equal(got[:e0], w), name
                    l = int(name.split(".")[1])
                    for g in range(2):
                        assert np.allclose(
                            got[e0 + g],
                            pack[f"L{l}.g{g}.gate"] * 0.5), name
                elif name.endswith("e_score_correction_bias"):
                    assert np.array_equal(got[:e0], w)
                    assert np.all(got[e0:] == -9.0)
                else:
                    assert np.array_equal(got, w), name
        print("claim 2: original tensors byte-preserved, gate rows and "
              "birth bias appended as specified")

        for (l, e, nm), w in want.items():
            dt, dims, off, size = e2[f"layers.{l}.block_sparse_moe."
                                     f"experts.{e}.{nm}"]
            assert dt == DTYPE_MXFP4 and tuple(dims) == w.shape
            f2.seek(off)
            got = dequant_mxfp4(f2.read(size), dims)
            p, s = quantize_mxfp4(w)
            exp = dequant_mxfp4(p.tobytes() + s.tobytes(), list(w.shape))
            assert np.array_equal(got, exp), (l, e, nm)
        print("claim 3: grafted experts round-trip through MXFP4 exactly")

        ex_off = [off for nn, (dt, dims, off, size) in e2.items()
                  if ".experts." in nn]
        assert all(o % 64 == 0 for o in ex_off)
        print("claim 4: alignment rule preserved")
        f2.close()

        # partial pack must be refused
        bad = {k: v for k, v in pack.items() if not k.startswith("L2")}
        pkb = os.path.join(td, "bad.npz")
        np.savez(pkb, **bad)
        try:
            inject(src, os.path.join(td, "x.bin"), pkb, -9.0, 1.0)
            raise AssertionError("partial pack accepted")
        except SystemExit as ex:
            assert "every MoE layer" in str(ex)
        print("claim 5: partial pack refused (n_experts is global)")

    print("inject_experts selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--birth-bias", type=float, default=-8.0,
                    help="e_score_correction_bias of the new experts; "
                    "strongly negative keeps them out of the top-k")
    ap.add_argument("--gate-scale", type=float, default=1.0)
    args = ap.parse_args()
    n_new, n_moe, size = inject(args.bin, args.out, args.pack,
                                args.birth_bias, args.gate_scale)
    print(f"-> {args.out}: {size / 1e6:.0f} MB, n_experts {n_new} over "
          f"{n_moe} MoE layers, birth bias {args.birth_bias}")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
