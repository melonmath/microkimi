#!/usr/bin/env python3
"""nanokimi - graftlib: numerical and I/O primitives for expert grafts.

This module provides the SiTU activation used by the engine, rectangular Gram
accumulation for streaming ridge solves, a dependency-free safetensors reader,
neuron selection, byte-anchor packing, capture archive helpers, and the core
host-shaped expert solve.

All solves are closed-form ridge regressions on calibration activations;
no gradient step is taken anywhere. Convention: ridge_solve returns W such
that x @ W.T ~ y (same as stitch_solve.py, which provides the solver).

  python3 graftlib.py --selftest
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stitch_solve import ridge_solve  # noqa: E402


# ---------------------------------------------------------------- activation

def situ(g, u):
    """The engine's expert activation (model_nano SiTU, fixed in Rust):
    act = (4 tanh(g/4) sigmoid(g)) * (25 tanh(u/25)). g: gate branch (w1),
    u: up branch (w3). Any float dtype, elementwise."""
    g = np.asarray(g)
    u = np.asarray(u)
    sig = 1.0 / (1.0 + np.exp(-g))
    return (4.0 * np.tanh(g / 4.0) * sig) * (25.0 * np.tanh(u / 25.0))


# ---------------------------------------------------------------- gram solve

class RectGram:
    """Streaming sufficient statistics for a RECTANGULAR ridge solve
    x [n, din] -> y [n, dout] (stitch_solve.GramAccumulator is square-only).
    Keeps Gyy in full so the exact relative residual and linear CKA can be
    reported. float64 throughout (same rationale as stitch_solve)."""

    def __init__(self, din, dout):
        self.din, self.dout = din, dout
        self.gxx = np.zeros((din, din), np.float64)
        self.gxy = np.zeros((din, dout), np.float64)
        self.gyy = np.zeros((dout, dout), np.float64)
        self.sum_x = np.zeros(din, np.float64)
        self.sum_y = np.zeros(dout, np.float64)
        self.n = 0

    def add(self, x, y):
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        assert x.shape[0] == y.shape[0] and x.shape[1] == self.din \
            and y.shape[1] == self.dout
        self.gxx += x.T @ x
        self.gxy += x.T @ y
        self.gyy += y.T @ y
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.n += x.shape[0]

    def solve(self, rel_lambda=1e-4):
        """Returns (W [dout, din], exact relative residual in [0, ~1])."""
        w, _ = ridge_solve(self.gxx, self.gxy, rel_lambda)
        num = (np.trace(w @ self.gxx @ w.T) - 2.0 * np.trace(w @ self.gxy)
               + np.trace(self.gyy))
        den = np.trace(self.gyy)
        return w, float(num / den) if den > 0 else 0.0

    def cka(self):
        """Centered linear CKA between the two streams: how alignable they
        are by ANY linear map (1 = perfectly, 0 = not at all). Cheap
        pre-filter before solving."""
        n = self.n
        gxx = self.gxx - np.outer(self.sum_x, self.sum_x) / n
        gxy = self.gxy - np.outer(self.sum_x, self.sum_y) / n
        gyy = self.gyy - np.outer(self.sum_y, self.sum_y) / n
        num = np.linalg.norm(gxy) ** 2
        den = np.linalg.norm(gxx) * np.linalg.norm(gyy)
        return float(num / den) if den > 0 else 0.0


def chunked_pairs(x, y, ix=None, iy=None, chunk=8192):
    """Yields aligned float64 chunks from two row sources (arrays or
    memmaps), optionally through index arrays (anchor pairing)."""
    n = len(ix) if ix is not None else x.shape[0]
    for t0 in range(0, n, chunk):
        t1 = min(t0 + chunk, n)
        xs = x[ix[t0:t1]] if ix is not None else x[t0:t1]
        ys = y[iy[t0:t1]] if iy is not None else y[t0:t1]
        yield np.asarray(xs, np.float64), np.asarray(ys, np.float64)


# ------------------------------------------------------------- byte anchors

DOC_SHIFT = 40  # packed anchor = (doc_index << 40) | byte_end_in_doc


def pack_ends(doc_idx, byte_ends):
    """Packs per-token byte end offsets of one document into globally
    comparable uint64 keys."""
    e = np.asarray(byte_ends, np.uint64)
    assert e.size == 0 or int(e.max()) < (1 << DOC_SHIFT), "document too large"
    return (np.uint64(doc_idx) << np.uint64(DOC_SHIFT)) | e


def match_anchors(ends_a, mask_a, ends_b, mask_b):
    """Intersects two packed-end streams; returns (idx_a, idx_b) index
    arrays of equal length: positions where both tokenizations close a
    token at the same byte offset of the same document."""
    ia = np.nonzero(mask_a)[0]
    ib = np.nonzero(mask_b)[0]
    common, ca, cb = np.intersect1d(ends_a[ia], ends_b[ib],
                                    return_indices=True)
    return ia[ca], ib[cb]


def char_to_byte_ends(text, char_ends):
    """Converts char-offset token ends (HF fast tokenizers) to byte offsets
    (tiktoken-comparable). O(len(text)) once per document."""
    blen = np.fromiter((len(c.encode("utf-8")) for c in text), np.int64,
                       count=len(text))
    cum = np.concatenate([[0], np.cumsum(blen)])
    return cum[np.asarray(char_ends, np.int64)]


# ---------------------------------------------------------------- capture io

def open_capture(prefix):
    """Loads the shared capture format: metadata, packed ends, validity mask,
    and a dictionary of lazily memory-mapped activation planes."""
    with open(prefix + ".meta.json") as f:
        meta = json.load(f)
    n = meta["n_tokens"]
    ends = np.memmap(prefix + ".ends.u64", np.uint64, "r", shape=(n,))
    mask = np.memmap(prefix + ".mask.u8", np.uint8, "r", shape=(n,))
    planes = {}
    for name, dim in meta["planes"].items():
        planes[name] = np.memmap(f"{prefix}.{name}.f16", np.float16, "r",
                                 shape=(n, dim))
    return meta, ends, mask.astype(bool), planes


class CaptureWriter:
    """Writes the capture layout read by open_capture. Planes are raw fp16
    row streams (no npy header) so the final token count can stay unknown
    until the end."""

    def __init__(self, prefix, planes, extra_meta=None):
        self.prefix = prefix
        self.plane_dims = dict(planes)
        os.makedirs(os.path.dirname(os.path.abspath(prefix)) or ".",
                    exist_ok=True)
        self.f_ends = open(prefix + ".ends.u64", "wb")
        self.f_mask = open(prefix + ".mask.u8", "wb")
        self.f_planes = {k: open(f"{prefix}.{k}.f16", "wb") for k in planes}
        self.n = 0
        self.extra = extra_meta or {}

    def add(self, ends_u64, mask_bool, plane_rows):
        k = len(ends_u64)
        self.f_ends.write(np.asarray(ends_u64, np.uint64).tobytes())
        self.f_mask.write(np.asarray(mask_bool, np.uint8).tobytes())
        for name, rows in plane_rows.items():
            rows = np.asarray(rows, np.float16)
            assert rows.shape == (k, self.plane_dims[name]), \
                f"{name}: {rows.shape} != ({k}, {self.plane_dims[name]})"
            self.f_planes[name].write(rows.tobytes())
        self.n += k

    def close(self):
        for f in [self.f_ends, self.f_mask, *self.f_planes.values()]:
            f.close()
        meta = {"n_tokens": self.n, "planes": self.plane_dims, **self.extra}
        with open(self.prefix + ".meta.json", "w") as f:
            json.dump(meta, f)
        return meta


# ----------------------------------------------------------- safetensors io

_ST_DTYPES = {
    "F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (np.uint16, 2),
    "F64": (np.float64, 8), "I64": (np.int64, 8), "I32": (np.int32, 4),
    "U8": (np.uint8, 1),
}


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def read_safetensors_dir(path, names):
    """Reads the given tensor names from a safetensors file or directory
    (with model.safetensors.index.json for sharded checkpoints), without
    the safetensors package. Returns {name: float32 array}."""
    if os.path.isdir(path):
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                wmap = json.load(f)["weight_map"]
            by_shard = {}
            for n in names:
                by_shard.setdefault(wmap[n], []).append(n)
            out = {}
            for shard, ns in by_shard.items():
                out.update(_read_st_file(os.path.join(path, shard), ns))
            return out
        path = os.path.join(path, "model.safetensors")
    return _read_st_file(path, names)


def _read_st_file(path, names):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        base = 8 + hlen
        out = {}
        for n in names:
            e = header[n]
            dt, _ = _ST_DTYPES[e["dtype"]]
            f.seek(base + e["data_offsets"][0])
            raw = f.read(e["data_offsets"][1] - e["data_offsets"][0])
            a = np.frombuffer(raw, dt).reshape(e["shape"]).copy()
            if e["dtype"] == "BF16":
                a = _bf16_to_f32(a)
            out[n] = a.astype(np.float32)
    return out


# ------------------------------------------------------------- neuron slice

def neuron_scores(w1_full, w3_full, w_down, h_sample, chunk=2048):
    """Importance of each donor FFN neuron seen from the host latent stream:
    mean |activation| on the sample times the norm of the neuron's output
    column. w1_full/w3_full: [d_ff, rh] (input-folded), w_down: [d_D, d_ff],
    h_sample: [m, rh]."""
    d_ff = w1_full.shape[0]
    acc = np.zeros(d_ff, np.float64)
    n = 0
    for t0 in range(0, h_sample.shape[0], chunk):
        h = np.asarray(h_sample[t0:t0 + chunk], np.float32)
        a = situ(h @ w1_full.T, h @ w3_full.T)
        acc += np.abs(a).sum(axis=0)
        n += h.shape[0]
    mean_act = acc / max(n, 1)
    return mean_act * np.linalg.norm(w_down, axis=0)


# ---------------------------------------------------------------- core solve

def solve_graft(h_lat, h_pln, z_in, dz, donor_w, moe_inter, bands=1,
                rel_lambda=1e-4, holdout=4096, score_sample=16384,
                chunk=8192, ih=None, idz=None, gate_sigma=1.0):
    """Solves host-shaped expert tensors from aligned activations.

    h_lat [n, rh]   host latent stream (routed_expert_down_proj output)
    h_pln [n, d]    host router input (post-attention-layernorm stream)
    z_in  [n, d_D]  donor FFN input (post-norm)
    dz    [n, d_D]  donor FFN residual delta
    donor_w         {"gate": [d_ff, d_D], "up": [d_ff, d_D],
                     "down": [d_D, d_ff]}
    ih, idz         optional anchor index arrays pairing host rows with
                    donor rows (from match_anchors)

    Returns {"experts": [(w1, w3, w2, gate_row) x bands], "diag": {...}}
    with w1/w3 [moe_inter, rh], w2 [rh, moe_inter], gate_row [d].
    Steps: S_in ridge (latent -> donor input), input fold, importance
    slice into `bands` successive neuron bands, SiTU response, per-band
    w2 ridge against the projected donor delta, per-band router row ridge.
    The w2 re-solve absorbs the output map, the activation mismatch and
    the slice error in one regression."""
    n = len(ih) if ih is not None else h_lat.shape[0]
    if n <= holdout:
        holdout = n // 4
    ih = np.asarray(ih) if ih is not None else np.arange(n)
    idz = np.asarray(idz) if idz is not None else np.arange(n)
    ih_fit, ih_hold = ih[holdout:], ih[:holdout]
    idz_fit, idz_hold = idz[holdout:], idz[:holdout]
    rh = h_lat.shape[1]
    d_d = z_in.shape[1]
    d_ff = donor_w["gate"].shape[0]

    # 1) state stitches, both directions, streamed grams (fit rows only:
    # the leading `holdout` pairs never enter any solve)
    g_in = RectGram(rh, d_d)
    g_out = RectGram(d_d, rh)
    for x, y in chunked_pairs(h_lat, z_in, ih_fit, idz_fit, chunk):
        g_in.add(x, y)
        g_out.add(y, x)
    s_in, res_in = g_in.solve(rel_lambda)      # [d_D, rh]
    m_out, res_out = g_out.solve(rel_lambda)   # [rh, d_D]
    cka = g_in.cka()

    # 2) input fold + importance slice
    w1_full = (donor_w["gate"].astype(np.float64) @ s_in).astype(np.float32)
    w3_full = (donor_w["up"].astype(np.float64) @ s_in).astype(np.float32)
    m = min(score_sample, n - holdout)
    h_sample = np.asarray(h_lat[ih_fit[:m]], np.float32)
    scores = neuron_scores(w1_full, w3_full, donor_w["down"], h_sample)
    order = np.argsort(-scores)
    need = moe_inter * bands
    assert d_ff >= need, f"donor d_ff {d_ff} < moe_inter*bands {need}"

    experts = []
    diag_bands = []
    for b in range(bands):
        idx = np.sort(order[b * moe_inter:(b + 1) * moe_inter])
        w1 = w1_full[idx]
        w3 = w3_full[idx]
        # 3) w2 re-solve: SiTU response -> projected donor delta
        g_w2 = RectGram(moe_inter, rh)
        g_gate = RectGram(h_pln.shape[1], 1)
        for (h, d), (p, _) in zip(
                chunked_pairs(h_lat, dz, ih_fit, idz_fit, chunk),
                chunked_pairs(h_pln, dz, ih_fit, idz_fit, chunk)):
            a = situ(h @ w1.T.astype(np.float64), h @ w3.T.astype(np.float64))
            y = d @ m_out.T
            g_w2.add(a, y)
            t = np.linalg.norm(y, axis=1, keepdims=True)
            g_gate.add(p, t)
        w2, res_w2 = g_w2.solve(rel_lambda)     # [rh, moe_inter]
        # 4) router row: predict the (standardized) size of the projected
        # donor contribution from the router's own input stream
        gate_row, _ = g_gate.solve(rel_lambda)
        gate_row = gate_row[0]
        # normalize the row so its logits sit in a router-typical range:
        # solved against an arbitrary-scale target, the raw row can put
        # every logit far outside the host distribution (a sigmoid score
        # saturated at 1.0 on every token routes the expert everywhere).
        # exact logit second moment from the grams: E[(g.h)^2] = g Gxx g / n
        ex2 = float(gate_row @ g_gate.gxx @ gate_row) / g_gate.n
        mu = float(gate_row @ g_gate.sum_x) / g_gate.n
        sd = np.sqrt(max(ex2 - mu * mu, 1e-12))
        gate_row = (gate_row * (gate_sigma / sd)).astype(np.float32)
        logit_mu = mu * gate_sigma / sd

        # 5) holdout: relative error of the final expert vs the target
        hh = np.asarray(h_lat[ih_hold], np.float64)
        dd = np.asarray(dz[idz_hold], np.float64)
        yy = dd @ m_out.T
        pred = situ(hh @ w1.T, hh @ w3.T) @ w2.T
        denom = float((yy ** 2).sum())
        rel_hold = float(((pred - yy) ** 2).sum() / denom) if denom > 0 else 0.0
        experts.append((w1.astype(np.float32), w3.astype(np.float32),
                        w2.astype(np.float32), gate_row))
        diag_bands.append({"res_w2": res_w2, "rel_holdout": rel_hold,
                           "gate_logit_mu": logit_mu,
                           "gate_logit_sd": gate_sigma})

    diag = {"cka": cka, "res_s_in": res_in, "res_m_out": res_out,
            "n_pairs": n, "holdout": holdout, "bands": diag_bands}
    return {"experts": experts, "diag": diag}


# ------------------------------------------------------------------ selftest

def selftest():
    rng = np.random.default_rng(7)
    n, rh, d, d_d, d_ff, mi = 6000, 24, 48, 32, 96, 32

    # host latent stream with anisotropic covariance; donor stream is a
    # noisy linear image of it (the platonic-alignment assumption)
    basis = rng.normal(size=(rh, rh)) * np.linspace(3, 0.2, rh)
    h_lat = rng.normal(size=(n, rh)) @ basis
    true_map = rng.normal(size=(d_d, rh)) / np.sqrt(rh)
    z_in = h_lat @ true_map.T + 0.05 * rng.normal(size=(n, d_d))
    h_pln = np.repeat(h_lat, 2, axis=1)[:, :d] + 0.1 * rng.normal(size=(n, d))

    # donor FFN: GELU-gated (unlike the host's SiTU - the mismatch is the
    # point), producing the residual delta the donor would add
    w_gate = rng.normal(size=(d_ff, d_d)) / np.sqrt(d_d)
    w_up = rng.normal(size=(d_ff, d_d)) / np.sqrt(d_d)
    w_down = rng.normal(size=(d_d, d_ff)) / np.sqrt(d_ff)
    g = z_in @ w_gate.T
    gelu = 0.5 * g * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (g + 0.044715 * g**3)))
    dz = (gelu * (z_in @ w_up.T)) @ w_down.T
    donor_w = {"gate": w_gate.astype(np.float32),
               "up": w_up.astype(np.float32),
               "down": w_down.astype(np.float32)}

    out = solve_graft(h_lat, h_pln, z_in, dz, donor_w, mi, bands=2,
                      rel_lambda=1e-4, holdout=1000, score_sample=2000)
    dg = out["diag"]
    print(f"claim 1: state stitch recovers the planted map - "
          f"res_s_in {dg['res_s_in']:.4f}, cka {dg['cka']:.3f}")
    assert dg["res_s_in"] < 0.05, dg["res_s_in"]
    assert dg["cka"] > 0.5, dg["cka"]

    b0 = dg["bands"][0]
    print(f"claim 2: sliced SiTU expert explains most of the projected "
          f"donor delta - band0 holdout rel err {b0['rel_holdout']:.3f}")
    assert b0["rel_holdout"] < 0.5, b0["rel_holdout"]

    zero_rel = 1.0  # predicting 0 has relative error exactly 1
    assert b0["rel_holdout"] < zero_rel
    b1 = dg["bands"][1]
    print(f"claim 3: second band still beats the zero predictor - "
          f"band1 holdout rel err {b1['rel_holdout']:.3f}")
    assert b1["rel_holdout"] < zero_rel, b1["rel_holdout"]

    w1, w3, w2, gate_row = out["experts"][0]
    assert w1.shape == (mi, rh) and w3.shape == (mi, rh) \
        and w2.shape == (rh, mi) and gate_row.shape == (d,)
    print("claim 4: host-native shapes "
          f"w1 {w1.shape} w2 {w2.shape} gate {gate_row.shape}")

    sd_log = float(np.std(h_pln[1000:] @ gate_row))
    print(f"claim 4b: gate logits normalized (std {sd_log:.3f})")
    assert 0.7 < sd_log < 1.4, sd_log

    # anchors: two tokenizations of the same byte stream must pair exactly
    ends_a = np.concatenate([pack_ends(0, [2, 5, 9]), pack_ends(1, [4, 8])])
    ends_b = np.concatenate([pack_ends(0, [2, 3, 5, 6, 9]), pack_ends(1, [8])])
    ia, ib = match_anchors(ends_a, np.ones(5, bool),
                           ends_b, np.ones(6, bool))
    assert list(ends_a[ia]) == list(ends_b[ib]) and len(ia) == 4
    mask = np.ones(5, bool)
    mask[0] = False  # e.g. a BOS or UNK position drops out
    ia2, _ = match_anchors(ends_a, mask, ends_b, np.ones(6, bool))
    assert len(ia2) == 3
    print("claim 5: byte-anchor pairing exact, masks respected")

    # char->byte ends on multibyte text
    text = "hélloé"
    be = char_to_byte_ends(text, [1, 2, 6])
    assert list(be) == [1, 3, 8], be
    print("claim 6: char->byte end conversion handles multibyte")

    # capture round trip
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "cap")
        w = CaptureWriter(p, {"lat": 4}, {"kind": "test"})
        w.add(pack_ends(0, [1, 2]), [1, 0], {"lat": rng.normal(size=(2, 4))})
        w.add(pack_ends(1, [3]), [1], {"lat": rng.normal(size=(1, 4))})
        w.close()
        meta, ends, mask, planes = open_capture(p)
        assert meta["n_tokens"] == 3 and planes["lat"].shape == (3, 4)
        assert mask.tolist() == [True, False, True]
    print("claim 7: capture writer/reader round trip")

    print("graftlib selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        raise SystemExit("usage: python3 graftlib.py --selftest")
