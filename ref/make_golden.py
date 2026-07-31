#!/usr/bin/env python3
"""Generates ref/golden.json: reference values for `microkimi selftest`.

TEST tool only (not a product dependency). Uses CPU torch +
the vendored fla shim (nano/vendor/fla, MIT) + ref/mxfp4_numpy.py.
Fixed-seed random inputs → reproducible.
"""
import json
import sys

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "nano", "vendor"))  # shim fla (MIT, vendored)
sys.path.insert(0, _HERE)

import numpy as np
import torch

from fla.ops.kda import chunk_kda
from mxfp4_numpy import dequant_mxfp4

OUT = os.path.join(_HERE, "golden.json")
golden = {}

# ── 1) KDA recurrence (B=1, T=3, H=4, K=V=128, per-channel A_log, lower_bound=-5) ──
g = torch.Generator().manual_seed(42)
T, H, K, V = 3, 4, 128, 128
q = torch.randn(1, T, H, K, generator=g)
k = torch.randn(1, T, H, K, generator=g)
v = torch.randn(1, T, H, V, generator=g)
graw = torch.randn(1, T, H, K, generator=g)
beta = torch.randn(1, T, H, generator=g)
A_log = torch.randn(K, generator=g)          # per channel (broadcast across heads)
dt_bias = torch.randn(H * K, generator=g)    # [H*K] → view(H,K)

o, S = chunk_kda(
    q, k, v, graw, beta,
    A_log=A_log, dt_bias=dt_bias, scale=None,
    initial_state=None, output_final_state=True,
    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
)
golden["kda"] = {
    "T": T, "H": H, "K": K, "V": V,
    "q": q.flatten().tolist(),
    "k": k.flatten().tolist(),
    "v": v.flatten().tolist(),
    "g": graw.flatten().tolist(),
    "beta": beta.flatten().tolist(),
    "A_log": A_log.flatten().tolist(),
    "dt_bias": dt_bias.flatten().tolist(),
    "o": o.flatten().tolist(),       # [T,H,V]
    "S": S.flatten().tolist(),       # [H,K,V]
}

# ── 2) SiTU: a = 4·tanh(g/4)·sigmoid(g) ; u = 25·tanh(u/25) ; out = a·u ──
gs = torch.randn(16, generator=g) * 6.0
us = torch.randn(16, generator=g) * 30.0
situ = 4.0 * torch.tanh(gs / 4.0) * torch.sigmoid(gs) * (25.0 * torch.tanh(us / 25.0))
golden["situ"] = {"g": gs.tolist(), "u": us.tolist(), "out": situ.tolist()}

# ── 3) MXFP4 dequant: packed [4,32] u8 (cols=64), scales [4,2] u8 ──
rng = np.random.default_rng(1234)
packed = rng.integers(0, 256, size=(4, 32), dtype=np.uint8)
scales = rng.integers(118, 136, size=(4, 2), dtype=np.uint8)
W = dequant_mxfp4(packed, scales)
golden["mxfp4"] = {
    "rows": 4, "cols": 64,
    "packed": packed.flatten().tolist(),
    "scales": scales.flatten().tolist(),
    "W": W.flatten().tolist(),
}

# ── 4) attn_res: 3 blocks + prefix, D=512, eps 1e-5 ──
D, B = 512, 3
blocks = torch.randn(B, D, generator=g)
prefix = torch.randn(D, generator=g)
norm_w = torch.randn(D, generator=g)
proj_w = torch.randn(D, generator=g)
v = torch.cat([blocks, prefix.unsqueeze(0)], dim=0)          # [B+1, D]
kk = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-5)  # RMS-norm without weights
w = norm_w * proj_w
scores = (kk * w).sum(-1)
probs = scores.softmax(-1)
out = (probs.unsqueeze(-1) * v).sum(0)
golden["attn_res"] = {
    "D": D, "B": B,
    "blocks": blocks.flatten().tolist(),
    "prefix": prefix.tolist(),
    "norm_w": norm_w.tolist(),
    "proj_w": proj_w.tolist(),
    "out": out.tolist(),
}

with open(OUT, "w") as f:
    json.dump(golden, f)
print(f"golden.json written: {len(json.dumps(golden)) / 1e6:.1f} MB")
print(f"  kda o[{T},{H},{V}] S[{H},{K},{V}] | situ[16] | mxfp4[4,64] | attn_res D={D} B={B}")
