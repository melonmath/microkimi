#!/usr/bin/env python3
"""Unit parity: KDA_SEG checkpointing plain vs NANO_ACT_OFFLOAD (host-stashed
segment inputs) - forward output, final state and input grads must be
BIT-identical (same recompute, only the storage of the segment inputs
differs). Also checks the automatic fallback to the plain checkpoint when the
offloaded path raises, and the NANO_PRETRANSPOSE LoRALinear path (x @ W_t
forward, gy @ W backward) against the classic F.linear one.
Run from the nano/ dir: python3 test_kda_offload_parity.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import vendor.fla.ops.kda as kda_mod  # noqa: E402
from vendor.fla.ops.kda import _kda_core  # noqa: E402

sys.path.pop(0)
from model_nano import LoRALinear, _WTLinear  # noqa: E402


def run(offload):
    kda_mod.KDA_SEG = 64
    kda_mod.KDA_SEG_DEVICES = ("cpu",)  # force the segment path on CPU
    kda_mod.ACT_OFFLOAD = offload
    kda_mod.ACT_OFFLOAD_DEVICES = ("cpu",)  # host-clone ring on CPU
    kda_mod._offload_available = True
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


o0, S0, *g0 = run(False)
o1, S1, *g1 = run(True)
checks = [("o", o0, o1), ("S", S0, S1)] + [
    (n, a, b) for n, a, b in zip(["dq", "dk", "dv", "dbeta"], g0, g1)
]
ok = True
for n, a, b in checks:
    d = (a - b).abs().max().item()
    bit = torch.equal(a, b)
    ok &= bit
    print(f"{n}: max|diff|={d:.3e} bit_identical={bit}")

# automatic fallback: a broken offload path must disable itself, not crash
class BoomRing:
    def __init__(self, dev):
        pass

    def stash(self, tensors):
        raise RuntimeError("injected failure")


real_ring = kda_mod._offload_ring
kda_mod._offload_ring = BoomRing
try:
    o2, S2, *_ = run(True)
finally:
    kda_mod._offload_ring = real_ring
ok &= torch.equal(o2, o0) and torch.equal(S2, S0)
ok &= kda_mod._offload_available is False
print(f"fallback: bit_identical={torch.equal(o2, o0)} disabled={not kda_mod._offload_available}")

# NANO_PRETRANSPOSE: _WTLinear (x @ W_t forward, gy @ W backward) vs F.linear
g = torch.Generator().manual_seed(3)
base = torch.nn.Linear(96, 64, bias=True)
with torch.no_grad():
    base.weight.copy_(torch.randn(64, 96, generator=g))
    base.bias.copy_(torch.randn(64, generator=g))
w_t = base.weight.t().contiguous()
x = torch.randn(5, 7, 96, generator=g, requires_grad=True)
y_ref = F.linear(x, base.weight, base.bias)
y_new = _WTLinear.apply(x, base.weight, w_t, base.bias)
# x @ W_t (contiguous) vs x @ W.T (view): same math, different BLAS summation
# order on CPU - float-noise tolerance, like the chunked KDA path
def rel(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-20)).item()


d = rel(y_ref, y_new)
ok &= d < 1e-4
print(f"pretranspose fwd: rel={d:.3e}")
gy = torch.randn(5, 7, 64, generator=g)
gx_ref, = torch.autograd.grad((y_ref * gy).sum(), x, retain_graph=False)
x2 = x.detach().requires_grad_(True)
y_new2 = _WTLinear.apply(x2, base.weight, w_t, base.bias)
gx_new, = torch.autograd.grad((y_new2 * gy).sum(), x2)
d = rel(gx_ref, gx_new)
ok &= d < 1e-4
print(f"pretranspose bwd: rel={d:.3e}")

# end-to-end through LoRALinear with the flag toggled at runtime
import model_nano  # noqa: E402

lora = LoRALinear(base, rank=8, alpha=8.0)
with torch.no_grad():
    lora.lora_A.copy_(torch.randn(8, 96, generator=g))
    lora.lora_B.copy_(torch.randn(64, 8, generator=g))
x3 = torch.randn(5, 7, 96, generator=g, requires_grad=True)
model_nano.PRETRANSPOSE = False
y_ref = lora(x3)
gx_ref, = torch.autograd.grad((y_ref * gy).sum(), x3)
model_nano.PRETRANSPOSE = True
lora.base.weight._w_t = w_t
x4 = x3.detach().requires_grad_(True)
y_new = lora(x4)
gx_new, = torch.autograd.grad((y_new * gy).sum(), x4)
d = max(rel(y_ref, y_new), rel(gx_ref, gx_new))
ok &= d < 1e-4
print(f"LoRALinear pretranspose: rel={d:.3e}")
del lora.base.weight._w_t
model_nano.PRETRANSPOSE = False

print("OK" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
