#!/usr/bin/env python3
"""Unit parity: chunked KDA (NANO_KDA_CHUNKED=1, _kda_recur_chunked) vs the
reference per-token recurrence - forward output, final state AND input grads,
on real training dims (H = 4..64 heads, K = V = 128, T = 1..600, odd T on
purpose). The chunked form is NOT bit-identical (per-chunk exp(cumsum) decays
instead of per-token products, different triangular-inverse operation order):
the deviation must stay at float-noise level, bounded here at 1e-4 relative.
Also checks the chunk_kda wrapper end to end (qk l2norm + gate + beta sigmoid,
per-channel A_log) with the flag toggled at runtime, and the automatic
fallback to the reference path when the chunked path raises.
Run from anywhere: python3 nano/test_kda_chunked.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import torch  # noqa: E402
import vendor.fla.ops.kda as kda_mod  # noqa: E402
from vendor.fla.ops.kda import _kda_core, chunk_kda  # noqa: E402

TOL = 1e-4


def rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-20)).item()


def run_core(chunked, dims, seed, with_state):
    B, T, H, K, V = dims
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=g, requires_grad=True)
    k = torch.randn(B, T, H, K, generator=g, requires_grad=True)
    v = torch.randn(B, T, H, V, generator=g, requires_grad=True)
    graw = torch.randn(B, T, H, K, generator=g)
    beta = torch.randn(B, T, H, generator=g, requires_grad=True)
    A_log = torch.log(torch.empty(K).uniform_(1.0, 16.0, generator=g))
    dt_bias = torch.zeros(H * K)
    S0 = torch.randn(B, H, K, V, generator=g, requires_grad=True) if with_state else None
    w_o = torch.randn(B, T, H, V, generator=g)
    w_s = torch.randn(B, H, K, V, generator=g)
    o, S = _kda_core(q, k, v, graw, beta, A_log, dt_bias, None, S0,
                     True, True, True, -5.0, chunked=chunked)
    # random projection: a scalar loss that touches every output/state element
    ((o * w_o).sum() + (S * w_s).sum()).backward()
    grads = [t.grad for t in (q, k, v, beta)] + ([S0.grad] if with_state else [])
    return o, S, grads


def check(name, ref, new, worst):
    r = rel(ref, new)
    worst[0] = max(worst[0], r)
    status = "ok" if r < TOL else "FAIL"
    print(f"    {name}: rel={r:.3e} {status}")
    return r < TOL


ok = True
worst = [0.0]
cases = [
    # (B, T, H, K, V) - K=V=128, H from 4 to 64, T from 1 to 600
    (1, 1, 4, 128, 128),
    (2, 7, 4, 128, 128),
    (1, 64, 4, 128, 128),
    (2, 65, 8, 128, 128),
    (1, 130, 16, 128, 128),
    (1, 600, 4, 128, 128),
    (1, 600, 64, 128, 128),
    (2, 333, 64, 128, 128),
]
for with_state in (False, True):
    for i, dims in enumerate(cases):
        print(f"  dims B,T,H,K,V={dims} initial_state={with_state}")
        o0, S0, g0 = run_core(False, dims, 100 + i, with_state)
        o1, S1, g1 = run_core(True, dims, 100 + i, with_state)
        ok &= check("o", o0, o1, worst)
        ok &= check("S", S0, S1, worst)
        for n, a, b in zip(["dq", "dk", "dv", "dbeta"] + (["dS0"] if with_state else []), g0, g1):
            ok &= check(n, a, b, worst)

# end-to-end through the chunk_kda wrapper, flag toggled at runtime
print("  chunk_kda wrapper (NANO_KDA_CHUNKED toggled)")
g = torch.Generator().manual_seed(7)
B, T, H, K, V = 1, 200, 4, 128, 128
q = torch.randn(B, T, H, K, generator=g)
k = torch.randn(B, T, H, K, generator=g)
v = torch.randn(B, T, H, V, generator=g)
graw = torch.randn(B, T, H, K, generator=g)
beta = torch.randn(B, T, H, generator=g)
A_log = torch.log(torch.empty(K).uniform_(1.0, 16.0, generator=g))
dt_bias = torch.zeros(H * K)
kw = dict(A_log=A_log, dt_bias=dt_bias, output_final_state=True,
          use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
          use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0)
kda_mod.KDA_CHUNKED = False
o0, S0 = chunk_kda(q, k, v, graw, beta, **kw)
kda_mod.KDA_CHUNKED = True
o1, S1 = chunk_kda(q, k, v, graw, beta, **kw)
ok &= check("o", o0, o1, worst)
ok &= check("S", S0, S1, worst)

# automatic fallback: a broken chunked path must disable itself, not crash
print("  fallback on error")
def boom(*a, **kw_):
    raise RuntimeError("injected failure")
real = kda_mod._kda_recur_chunked
kda_mod._kda_recur_chunked = boom
kda_mod._chunked_available = True
o2, S2 = chunk_kda(q, k, v, graw, beta, **kw)
ok &= torch.equal(o2, o0) and torch.equal(S2, S0)
ok &= kda_mod._chunked_available is False
kda_mod._kda_recur_chunked = real
kda_mod._chunked_available = True
print(f"    fallback: {'ok' if ok else 'FAIL'} (bit-identical to reference)")

print(f"worst relative deviation: {worst[0]:.3e} (tolerance {TOL:.0e})")
print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
