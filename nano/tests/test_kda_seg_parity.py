#!/usr/bin/env python3
"""Unit parity: KDA recurrence plain (KDA_SEG=0) vs time-segment checkpointed
(KDA_SEG=64) - forward output, final state and input grads must match.
Run from the nano/ dir with NANO_KDA_SEG_DEVICES=cpu.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "vendor"))

import torch  # noqa: E402
import vendor.fla.ops.kda as kda_mod  # noqa: E402
from vendor.fla.ops.kda import _kda_core  # noqa: E402


def run(seg):
    kda_mod.KDA_SEG = seg
    kda_mod.KDA_SEG_DEVICES = ("cpu",)  # force the segment path on CPU
    g = torch.Generator().manual_seed(11)
    B, T, H, K, V = 2, 130, 4, 128, 128  # T not a multiple of 64 on purpose
    q = torch.randn(B, T, H, K, generator=g, requires_grad=True)
    k = torch.randn(B, T, H, K, generator=g, requires_grad=True)
    v = torch.randn(B, T, H, V, generator=g, requires_grad=True)
    graw = torch.randn(B, T, H, K, generator=g)
    beta = torch.rand(B, T, H, generator=g, requires_grad=True)
    A_log = torch.log(torch.empty(K).uniform_(1.0, 16.0, generator=g))
    dt_bias = torch.zeros(H * K)
    o, S = _kda_core(q, k, v, graw, beta, A_log, dt_bias, 0.5, None,
                     True, True, False, -5.0)  # qk l2norm + gate, as in the model
    loss = o.sum() + S.sum()
    loss.backward()
    return o, S, q.grad, k.grad, v.grad, beta.grad


o0, S0, *g0 = run(0)
o1, S1, *g1 = run(64)
checks = [("o", o0, o1), ("S", S0, S1)] + [
    (n, a, b) for n, a, b in zip(["dq", "dk", "dv", "dbeta"], g0, g1)
]
ok = True
for n, a, b in checks:
    d = (a - b).abs().max().item()
    bit = torch.equal(a, b)
    ok &= d < 1e-6
    print(f"{n}: max|diff|={d:.3e} bit_identical={bit}")
print("OK" if ok else "MISMATCH")
