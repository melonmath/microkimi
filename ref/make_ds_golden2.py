#!/usr/bin/env python3
"""Generate ref/ds_golden2.json: reference values for DeepSeek-V4 MoE routing
(sqrtsoftplus + noaux_tc bias + renorm + route_scale, hash routing) and
Hyper-Connections (hc_pre/hc_post/hc_head with Sinkhorn), plain torch,
math copied verbatim from /tmp/dsv4/model.py and /tmp/dsv4/kernel.py
(DeepSeek AI reference code).

Test tool only. Run: /home/node/venv/bin/python3 ref/make_ds_golden2.py
"""
import json
import torch
import torch.nn.functional as F

OUT = "/workspace/microkimi-oss/ref/ds_golden2.json"
g = torch.Generator().manual_seed(31337)
golden = {}

# ── Gate (model.py:551-589) : sqrtsoftplus, bias for selection only, renorm ×1.5 ──
N, D, E, TOPK, ROUTE_SCALE = 3, 16, 32, 6, 1.5
x = torch.randn(N, D, generator=g)
gw = torch.randn(E, D, generator=g)
bias = torch.randn(E, generator=g)
scores = F.linear(x.float(), gw.float())
scores = F.softplus(scores).sqrt()
original_scores = scores
sel = scores + bias
indices = sel.topk(TOPK, dim=-1)[1]
weights = original_scores.gather(1, indices)
weights = weights / weights.sum(dim=-1, keepdim=True) * ROUTE_SCALE
golden["gate"] = {
    "x": x.flatten().tolist(),
    "gate_w": gw.flatten().tolist(),
    "bias": bias.flatten().tolist(),
    "indices": indices.flatten().tolist(),
    "weights": weights.flatten().tolist(),
}

# ── Expert (model.py:592-611) : silu(gate)*up, clamp up±10, gate≤10 ──
H, I = 8, 16
ex = torch.randn(1, H, generator=g) * 4.0  # large values to exercise clamps
w1 = torch.randn(I, H, generator=g)
w2 = torch.randn(H, I, generator=g)
w3 = torch.randn(I, H, generator=g)
LIMIT = 10.0
gate = F.linear(ex, w1).float()
up = F.linear(ex, w3).float()
up = torch.clamp(up, min=-LIMIT, max=LIMIT)
gate = torch.clamp(gate, max=LIMIT)
act = F.silu(gate) * up
out = F.linear(act, w2)
golden["expert"] = {
    "x": ex.flatten().tolist(),
    "w1": w1.flatten().tolist(),
    "w2": w2.flatten().tolist(),
    "w3": w3.flatten().tolist(),
    "act": act.flatten().tolist(),
    "out": out.flatten().tolist(),
}

# ── Hyper-Connections (model.py:680-716 + kernel.py:371-438) ──
HC, DD, ITERS, EPS = 4, 6, 20, 1e-6
mix_hc = (2 + HC) * HC
xs = torch.randn(1, HC, DD, generator=g)          # [1,hc,d] (one token)
hc_fn = torch.randn(mix_hc, HC * DD, generator=g)
hc_scale = torch.randn(3, generator=g)
hc_base = torch.randn(mix_hc, generator=g)

xf = xs.flatten(1).float()                        # [1, hc*d]
rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
mixes = F.linear(xf, hc_fn) * rsqrt

# hc_split_sinkhorn (kernel.py:392-425) — comb is [b, s, hc, hc]: the sinkhorn
# operates on the 4x4 MATRIX (sums over matrix rows/cols), so reshape first.
pre = torch.sigmoid(mixes[..., :HC] * hc_scale[0] + hc_base[:HC]) + EPS
post = 2 * torch.sigmoid(mixes[..., HC:2 * HC] * hc_scale[1] + hc_base[HC:2 * HC])
comb = (mixes[..., 2 * HC:] * hc_scale[2] + hc_base[2 * HC:]).view(1, HC, HC)
comb = comb.softmax(-1) + EPS
comb = comb / (comb.sum(-2, keepdim=True) + EPS)
for _ in range(ITERS - 1):
    comb = comb / (comb.sum(-1, keepdim=True) + EPS)
    comb = comb / (comb.sum(-2, keepdim=True) + EPS)

# hc_pre / hc_post (model.py:680-693) — keep the exact [b, s] leading dims of
# model.py so that torch.sum(..., dim=2) hits the FIRST hc index, as upstream.
xs_bs = xs.view(1, 1, HC, DD)
pre_bs = pre.view(1, 1, HC)
post_bs = post.view(1, 1, HC)
comb_bs = comb.view(1, 1, HC, HC)
y_pre = (pre_bs.unsqueeze(-1) * xs_bs).sum(dim=2)                    # [1, 1, d]
y_post = post_bs.unsqueeze(-1) * y_pre.unsqueeze(-2) + \
    torch.sum(comb_bs.unsqueeze(-1) * xs_bs.unsqueeze(-2), dim=2)
y_pre = y_pre.view(-1)
y_post = y_post.view(-1)
comb = comb.view(-1)

# hc_head (model.py:709-716): sigmoid + eps, no sinkhorn
pre_head = torch.sigmoid(mixes[..., :HC] * hc_scale[0] + hc_base[:HC]) + EPS
y_head = (pre_head.unsqueeze(-1) * xs).sum(dim=1)

golden["hc"] = {
    "x": xs.flatten().tolist(),
    "hc_fn": hc_fn.flatten().tolist(),
    "hc_scale": hc_scale.flatten().tolist(),
    "hc_base": hc_base.flatten().tolist(),
    "mixes": mixes.flatten().tolist(),
    "pre": pre.flatten().tolist(),
    "post": post.flatten().tolist(),
    "comb": comb.flatten().tolist(),
    "y_pre": y_pre.flatten().tolist(),
    "y_post": y_post.flatten().tolist(),
    "y_head": y_head.flatten().tolist(),
}

with open(OUT, "w") as f:
    json.dump(golden, f)
print(f"written {OUT}")
