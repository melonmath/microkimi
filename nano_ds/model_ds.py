#!/usr/bin/env python3
"""nanodeepseek - PyTorch training model (model_ds.py)

DeepSeek-V4 architecture at nano dims, trainable with autograd, batched
prefill form [B, T]. The math follows src/deepseek.rs (the Rust inference
engine, validated 1:1 against DeepSeek's reference model.py) EXACTLY:

  embed -> expand x hc_mult (Hyper-Connections) -> n_layers x Block
    Block: hc_pre (Sinkhorn 20 iters) -> attn_norm -> sparse attention
           (window 128 ring, overlap/dense KV compressors, lightning indexer
           with per-head Hadamard, attentional sink, grouped low-rank O proj,
           RoPE theta 10000 / compress 160000+YaRN 16, FP8/FP4 QAT round-trips)
           -> hc_post ; hc_pre -> ffn_norm -> MoE (sqrtsoftplus router,
           noaux_tc bias, top-6, renorm x1.5, hash routing with tid2eid on the
           first n_hash_layers, SwiGLU clamp +/-10, fp4 experts trained in
           fp32 and quantized at export) -> hc_post
  -> hc_head -> RMSNorm -> lm_head

Batched-prefill and sequential-decode forms are mathematically identical
(proven for the Rust engine by dsparity); nano_ds/selfcheck.py re-proves the
equivalence for THIS torch implementation against the sequential replica of
ref/make_ds_parity.py.

Training specifics:
  - discrete selections (router top-k, hash table, indexer top-k): no gradient
    (standard MoE); routing weights DO propagate gradients to gate.weight.
  - QAT round-trips (act_quant fp8 / fp4) are applied with a straight-through
    estimator: x + (rt(x) - x).detach() - same values in eval, identity
    gradient in training (the reference trains QAT the same way).
  - MoE experts: capacity-padded grouped bmm (same approach as
    nano/model_nano.py's moe_train_bmm, validated 1:1 against the naive loop).
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6       # norm_eps (V4 RMSNorm)
HC_EPS = 1e-6    # hc_eps
HC_ITERS = 20    # hc_sinkhorn_iters
HC = 4           # hc_mult

NANO_DS = dict(
    n_layers=8, hidden=512, vocab=8200,
    n_heads=8, head_dim=128, rope_head_dim=64,
    q_lora_rank=128, o_lora_rank=128, o_groups=8,
    window_size=128,
    compress_ratios=[0, 0, 4, 128, 4, 128, 4, 128],
    rope_theta=10000.0, compress_rope_theta=160000.0,
    yarn_factor=16.0, yarn_beta_fast=32, yarn_beta_slow=1, yarn_orig_seq_len=65536,
    index_n_heads=8, index_head_dim=128, index_topk=64,
    n_routed_experts=256, top_k=6, moe_inter=64,
    n_hash_layers=3, route_scale=1.5, swiglu_limit=10.0,
    max_seq_len=512,
)

# ── RoPE / YaRN (same formulas as deepseek.rs::precompute_freqs_cis) ──

def precompute_angles(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def cdim(num_rot):
            return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(cdim(beta_fast)), 0)
        high = min(math.ceil(cdim(beta_slow)), dim - 1)
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (max(high, low + 1) - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return torch.outer(torch.arange(seqlen), freqs)  # [seqlen, dim/2]


def apply_rope(x, angles, inverse=False):
    """x [B, T, H, D]; angles [T, rd/2]. Rotates the trailing 2*len dims in
    adjacent-pairs convention (view_as_complex), like the reference."""
    rd2 = angles.shape[-1]
    xc = torch.view_as_complex(x[..., -2 * rd2:].float().unflatten(-1, (-1, 2)))
    f = torch.polar(torch.ones_like(angles), angles)  # [T, rd2]
    if inverse:
        f = f.conj()
    xc = xc * f.unsqueeze(0).unsqueeze(2)  # broadcast over B and H
    out = x.clone()
    out[..., -2 * rd2:] = torch.view_as_real(xc).flatten(-2)
    return out


def hadamard(x):
    """Sylvester Walsh-Hadamard rotation on the last dim (power of 2), x n^-0.5."""
    n = x.shape[-1]
    H = torch.ones(1, 1, device=x.device, dtype=x.dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (x @ H) * n ** -0.5


# ── QAT round-trips (kernel.py act_quant / fp4_act_quant) with STE ──

def _fp8_rt(x, block=64):
    y = x.reshape(*x.shape[:-1], -1, block)
    amax = y.abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 8)
    s = 2.0 ** e
    return (torch.clamp(y / s, -448.0, 448.0).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                      -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

def _fp4_rt(x, block=32):
    y = x.reshape(*x.shape[:-1], -1, block)
    amax = y.abs().amax(dim=-1, keepdim=True).clamp(min=6e-38)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 128)
    s = 2.0 ** e
    q = torch.clamp(y / s, -6.0, 6.0)
    lut = _E2M1.to(x.device)
    idx = (q.abs().unsqueeze(-1) - lut.abs()).abs().argmin(-1)
    idx = torch.where(q < 0, idx + 8, idx)
    return (lut[idx] * s).reshape(x.shape)


def ste(rt_fn, x, *args):
    """Straight-through estimator: forward = rt_fn(x), gradient = identity."""
    with torch.no_grad():
        rt = rt_fn(x, *args)
    return x + (rt - x.detach() if torch.is_grad_enabled() else rt - x)


# ── shared math ──

def rms(x, w):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w


def sinkhorn(m):
    """m [B, T, HC*HC] -> normalized combination matrix [B, T, HC, HC].
    softmax(rows)+eps; colnorm+eps; (iters-1) x (rownorm+eps, colnorm+eps)."""
    m = m.unflatten(-1, (HC, HC)).softmax(-1) + HC_EPS
    m = m / (m.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(1, HC_ITERS):
        m = m / (m.sum(-1, keepdim=True) + HC_EPS)
        m = m / (m.sum(-2, keepdim=True) + HC_EPS)
    return m


def hc_pre(state, fn, scale, base):
    """state [B,T,HC,D] -> (y [B,T,D], post [B,T,HC], comb [B,T,HC,HC])."""
    xf = state.flatten(2)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    mixes = torch.einsum("mc,btc->btm", fn, xf) * rsqrt
    pre = torch.sigmoid(mixes[..., :HC] * scale[0] + base[:HC]) + HC_EPS
    post = 2.0 * torch.sigmoid(mixes[..., HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = sinkhorn(mixes[..., 2 * HC:] * scale[2] + base[2 * HC:])
    y = (pre.unsqueeze(-1) * state).sum(2)
    return y, post, comb


def hc_post(x, state, post, comb):
    """y[j] = post[j] * x + sum_k comb[k, j] * residual[k] -> [B,T,HC,D]."""
    return post.unsqueeze(-1) * x.unsqueeze(2) + torch.einsum("btkj,btkd->btjd", comb, state)


def hc_head(state, fn, scale, base):
    xf = state.flatten(2)
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)
    mixes = torch.einsum("mc,btc->btm", fn, xf) * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + HC_EPS
    return (pre.unsqueeze(-1) * state).sum(2)


def overlap_pool(kv, score, ratio):
    """Overlap compressor pooling (ratio=4, coff=2), batched:
    kv/score [B, T, 2*S] -> pooled [B, T//ratio, S].
    Window w: [prev_window first S | current window second S], softmax PER
    COLUMN over the 2*ratio entries (torch softmax(dim=0) equivalent)."""
    B, T, S2 = kv.shape
    S = S2 // 2
    E = T // ratio
    kv = kv[:, : E * ratio]
    score = score[:, : E * ratio]
    kvw = kv.view(B, E, ratio, S2)
    scw = score.view(B, E, ratio, S2)
    prev_kv = torch.cat([torch.zeros_like(kvw[:, :1]), kvw[:, :-1]], 1)
    prev_sc = torch.cat([torch.full_like(scw[:, :1], float("-inf")), scw[:, :-1]], 1)
    kvf = torch.cat([prev_kv[..., :S], kvw[..., S:]], 2)   # [B, E, 2*ratio, S]
    scf = torch.cat([prev_sc[..., :S], scw[..., S:]], 2)
    return (kvf * scf.softmax(2)).sum(2)


def dense_pool(kv, score, ratio):
    """Dense compressor pooling (coff=1): [B,T,S] -> [B,T//ratio,S],
    per-column softmax over the window entries."""
    B, T, S = kv.shape
    E = T // ratio
    kv = kv[:, : E * ratio]
    score = score[:, : E * ratio]
    kvw = kv.view(B, E, ratio, S)
    scw = score.view(B, E, ratio, S)
    return (kvw * scw.softmax(2)).sum(2)


def softplus_sqrt(x):
    return F.softplus(x).sqrt()


# ── sparse attention layer (model.py:442-548) ──

class DsAttention(nn.Module):
    def __init__(self, c, ratio):
        super().__init__()
        self.c = c
        self.ratio = ratio
        D, HD, QL = c["hidden"], c["head_dim"], c["q_lora_rank"]
        NH, OG, OL = c["n_heads"], c["o_groups"], c["o_lora_rank"]
        self.wq_a = nn.Parameter(torch.empty(QL, D))
        self.q_norm = nn.Parameter(torch.ones(QL))
        self.wq_b = nn.Parameter(torch.empty(NH * HD, QL))
        self.wkv = nn.Parameter(torch.empty(HD, D))
        self.kv_norm = nn.Parameter(torch.ones(HD))
        self.wo_a = nn.Parameter(torch.empty(OG * OL, NH * HD // OG))
        self.wo_b = nn.Parameter(torch.empty(D, OG * OL))
        self.attn_sink = nn.Parameter(torch.zeros(NH))
        if ratio > 0:
            coff = 2 if ratio == 4 else 1
            self.comp_wkv = nn.Parameter(torch.empty(coff * HD, D))
            self.comp_wgate = nn.Parameter(torch.empty(coff * HD, D))
            self.comp_ape = nn.Parameter(torch.zeros(ratio, coff * HD))
            self.comp_norm = nn.Parameter(torch.ones(HD))
        if ratio == 4:
            INH, IHD = c["index_n_heads"], c["index_head_dim"]
            self.idx_wq_b = nn.Parameter(torch.empty(INH * IHD, QL))
            self.idx_weights_proj = nn.Parameter(torch.empty(INH, D))
            self.idx_comp_wkv = nn.Parameter(torch.empty(2 * IHD, D))
            self.idx_comp_wgate = nn.Parameter(torch.empty(2 * IHD, D))
            self.idx_comp_ape = nn.Parameter(torch.zeros(4, 2 * IHD))
            self.idx_comp_norm = nn.Parameter(torch.ones(IHD))

    def forward(self, x, angles):
        c = self.c
        B, T, D = x.shape
        NH, HD, RD = c["n_heads"], c["head_dim"], c["rope_head_dim"]
        WIN = c["window_size"]
        # q: wq_a -> q_norm -> wq_b -> per-head rsqrt norm -> rope (model.py:502-505)
        qr = rms(x @ self.wq_a.t(), self.q_norm)
        q = (qr @ self.wq_b.t()).view(B, T, NH, HD)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS)
        q = apply_rope(q, angles)
        # kv: wkv -> kv_norm -> rope -> fp8 QAT on non-rope dims (model.py:508-512)
        kv = rms(x @ self.wkv.t(), self.kv_norm)
        kv = apply_rope(kv.unsqueeze(2), angles).squeeze(2)
        kv = torch.cat([ste(_fp8_rt, kv[..., :-RD]), kv[..., -RD:]], -1)

        # window scores [B, NH, T, T] with causal sliding-window mask
        scale = HD ** -0.5
        sw = torch.einsum("bthd,bsd->bhts", q, kv) * scale  # kv shared across heads (V4: 1 kv head)
        pos = torch.arange(T, device=x.device)
        win_mask = (pos.view(1, -1) > pos.view(-1, 1)) | (pos.view(1, -1) <= pos.view(-1, 1) - WIN)
        sw = sw.masked_fill(win_mask, float("-inf"))

        scores_list, vals_list = [sw], [kv]
        # compressed section: complete windows only (a trailing partial window
        # simply has not compressed yet — exactly the decode behavior)
        n_win = T // self.ratio if self.ratio > 0 else 0
        if n_win > 0:
            ratio = self.ratio
            coff = 2 if ratio == 4 else 1
            ck = x @ self.comp_wkv.t()                       # [B,T,coff*HD]
            cs = x @ self.comp_wgate.t() + self.comp_ape[pos % ratio]
            if ratio == 4:
                comp = overlap_pool(ck, cs, ratio)           # [B,E,HD]
            else:
                comp = dense_pool(ck, cs, ratio)
            comp = rms(comp, self.comp_norm)
            # RoPE angle of the FIRST token of the window (model.py:372)
            ang_c = angles[(torch.arange(T // ratio, device=x.device)) * ratio]
            comp = apply_rope(comp.unsqueeze(2), ang_c).squeeze(2)
            comp = torch.cat([ste(_fp8_rt, comp[..., :-RD]), comp[..., -RD:]], -1)
            E = comp.shape[1]
            sc_c = torch.einsum("bthd,bsd->bhts", q, comp) * scale  # [B,NH,T,E]
            # causal: entry e covers tokens up to (e+1)*ratio-1 -> need t >= that
            e_idx = torch.arange(E, device=x.device)
            comp_valid = e_idx.view(1, -1) <= (pos.view(-1, 1) + 1) // ratio - 1  # [T,E]
            if ratio == 4:
                # indexer: its own overlap compressor (Hadamard + fp4)
                INH, IHD = c["index_n_heads"], c["index_head_dim"]
                ik = x @ self.idx_comp_wkv.t()
                isc = x @ self.idx_comp_wgate.t() + self.idx_comp_ape[pos % 4]
                ikv = overlap_pool(ik, isc, 4)               # [B,E,IHD]
                ikv = rms(ikv, self.idx_comp_norm)
                ikv = apply_rope(ikv.unsqueeze(2), ang_c).squeeze(2)
                ikv = ste(_fp4_rt, hadamard(ikv))
                iq = (qr @ self.idx_wq_b.t()).view(B, T, INH, IHD)
                iq = apply_rope(iq, angles)
                iq = ste(_fp4_rt, hadamard(iq))
                iw = (x @ self.idx_weights_proj.t()) * ((IHD ** -0.5) * (INH ** -0.5))
                idx_sc = torch.einsum("bthd,bsd->bhts", iq, ikv).relu_()
                idx_sc = torch.einsum("bhts,bth->bts", idx_sc, iw)
                idx_sc = idx_sc.masked_fill(~comp_valid, float("-inf"))
                k = min(c["index_topk"], E)
                sel = idx_sc.topk(k, dim=-1).indices         # [B,T,K] (discrete)
                sel_mask = torch.zeros(B, T, E, dtype=torch.bool, device=x.device)
                sel_mask.scatter_(-1, sel, True)
                comp_valid = comp_valid & sel_mask
            else:
                comp_valid = comp_valid.unsqueeze(0).expand(B, -1, -1)
            scores_list.append(sc_c.masked_fill(~comp_valid.unsqueeze(1), float("-inf")))
            vals_list.append(comp)

        # sparse attention with attentional sink (kernel.py:277-368)
        scores = torch.cat(scores_list, -1)                  # [B,NH,T,J]
        vals = torch.cat(vals_list, 1)                       # [B,J,HD]
        m = scores.amax(-1, keepdim=True)
        z = torch.exp(self.attn_sink.view(1, NH, 1, 1) - m) + torch.exp(scores - m).sum(-1, keepdim=True)
        o = torch.einsum("bhtj,bjd->bthd", torch.exp(scores - m) / z, vals)
        # derotation (model.py:539) + grouped O projection (model.py:542-547)
        o = apply_rope(o, angles, inverse=True)
        OG, OL = c["o_groups"], c["o_lora_rank"]
        lat = torch.einsum("grd,btgd->btgr", self.wo_a.view(OG, OL, NH * HD // OG), o.view(B, T, OG, -1))
        return lat.flatten(2) @ self.wo_b.t()


# ── MoE (model.py:551-649) ──

class DsMoe(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.c = c
        D, E, I = c["hidden"], c["n_routed_experts"], c["moe_inter"]
        self.gate_w = nn.Parameter(torch.empty(E, D))
        self.hash = layer < c["n_hash_layers"]
        if self.hash:
            g = torch.Generator().manual_seed(0xD5E4_0000 + layer)
            tid2eid = torch.randint(0, E, (c["vocab"], c["top_k"]), generator=g, dtype=torch.int32)
            self.register_buffer("tid2eid", tid2eid)
            self.gate_bias = None
        else:
            self.tid2eid = None
            self.gate_bias = nn.Parameter(torch.zeros(E))
        # stacked expert weights [E, I, D] / [E, D, I] (bmm-friendly)
        self.w1 = nn.Parameter(torch.empty(E, I, D))
        self.w3 = nn.Parameter(torch.empty(E, I, D))
        self.w2 = nn.Parameter(torch.empty(E, D, I))
        self.sh1 = nn.Parameter(torch.empty(I, D))
        self.sh3 = nn.Parameter(torch.empty(I, D))
        self.sh2 = nn.Parameter(torch.empty(D, I))

    def expert_math(self, g, u):
        limit = self.c["swiglu_limit"]
        return F.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)

    def forward(self, x, ids):
        c = self.c
        B, T, D = x.shape
        E, K = c["n_routed_experts"], c["top_k"]
        xf = x.reshape(B * T, D)
        scores = softplus_sqrt(xf @ self.gate_w.t())          # [N,E]
        if self.hash:
            sel = self.tid2eid[ids.reshape(-1)].long()        # [N,K]
        else:
            sel = (scores + self.gate_bias).topk(K, dim=-1).indices
        w = scores.gather(1, sel)
        w = w / w.sum(-1, keepdim=True) * c["route_scale"]
        y = self.moe_bmm(xf, sel, w)
        # shared expert (single, unweighted)
        y = y + (self.expert_math(xf @ self.sh1.t(), xf @ self.sh3.t()) @ self.sh2.t())
        return y.view(B, T, D)

    def moe_bmm(self, x, sel, w):
        """Capacity-padded grouped bmm (same approach as nano/model_nano.py's
        moe_train_bmm): sort assignments by expert, scatter into [E, cap, D],
        3 bmm, gather back. Routing weight applied BEFORE w2 (Expert.forward).
        The capacity is capped (~3x the average); above that, extra rounds
        (identical result) instead of a giant dense allocation."""
        c = self.c
        N, K = sel.shape
        E, I, D = self.w1.shape[0], self.w1.shape[1], x.shape[-1]
        flat_ids = sel.reshape(-1)
        flat_w = w.reshape(-1, 1)
        rep = x.unsqueeze(1).expand(N, K, D).reshape(N * K, D)
        order = flat_ids.argsort()
        sorted_tokens = rep[order]
        counts = torch.bincount(flat_ids, minlength=E)
        starts = counts.cumsum(0) - counts
        sorted_eids = flat_ids[order]
        slot = torch.arange(N * K, device=x.device) - starts[sorted_eids]
        cap = max(64, (3 * N * K + E - 1) // E)
        out_sorted = torch.empty_like(sorted_tokens)
        round_of = slot // cap
        for r in range(int(round_of.max().item()) + 1):
            mask = round_of == r
            eids_r = sorted_eids[mask]
            slot_r = slot[mask] - r * cap
            dense = x.new_zeros(E, cap, D)
            dense[eids_r, slot_r] = sorted_tokens[mask]
            wdense = x.new_zeros(E, cap, 1)
            wdense[eids_r, slot_r] = flat_w[mask]
            g = torch.bmm(dense, self.w1.transpose(1, 2))      # [E,cap,I]
            u = torch.bmm(dense, self.w3.transpose(1, 2))
            act = self.expert_math(g, u) * wdense              # weight BEFORE w2
            y = torch.bmm(act, self.w2.transpose(1, 2))        # [E,cap,D]
            out_sorted[mask] = y[eids_r, slot_r]
        unsorted = torch.empty_like(out_sorted)
        unsorted[order] = out_sorted
        return unsorted.view(N, K, D).sum(1)


# ── block + full model ──

class DsBlock(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        mix = (2 + HC) * HC
        self.attn_norm = nn.Parameter(torch.ones(c["hidden"]))
        self.ffn_norm = nn.Parameter(torch.ones(c["hidden"]))
        for kind in ("attn", "ffn"):
            setattr(self, f"hc_{kind}_fn", nn.Parameter(torch.empty(mix, HC * c["hidden"])))
            setattr(self, f"hc_{kind}_base", nn.Parameter(torch.zeros(mix)))
            setattr(self, f"hc_{kind}_scale", nn.Parameter(torch.ones(3)))
        self.attn = DsAttention(c, c["compress_ratios"][layer])
        self.moe = DsMoe(c, layer)

    def forward(self, state, ids, angles):
        y, post, comb = hc_pre(state, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        a = self.attn(rms(y, self.attn_norm), angles)
        state = hc_post(a, state, post, comb)
        y, post, comb = hc_pre(state, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        m = self.moe(rms(y, self.ffn_norm), ids)
        return hc_post(m, state, post, comb)


class NanoDsModel(nn.Module):
    def __init__(self, cfg=None, grad_ckpt=False):
        super().__init__()
        c = dict(NANO_DS) if cfg is None else {**NANO_DS, **cfg}
        assert len(c["compress_ratios"]) == c["n_layers"]
        self.c = c
        self.grad_ckpt = grad_ckpt
        D, V = c["hidden"], c["vocab"]
        self.embed = nn.Embedding(V, D)
        self.layers = nn.ModuleList([DsBlock(c, l) for l in range(c["n_layers"])])
        self.norm_w = nn.Parameter(torch.ones(D))
        self.hc_head_fn = nn.Parameter(torch.empty(HC, HC * D))
        self.hc_head_base = nn.Parameter(torch.zeros(HC))
        self.hc_head_scale = nn.Parameter(torch.ones(1))
        self.head = nn.Linear(D, V, bias=False)
        ms = c["max_seq_len"]
        self.register_buffer("ang_win", precompute_angles(
            c["rope_head_dim"], ms, 0, c["rope_theta"], c["yarn_factor"], c["yarn_beta_fast"], c["yarn_beta_slow"]), persistent=False)
        self.register_buffer("ang_cmp", precompute_angles(
            c["rope_head_dim"], ms, c["yarn_orig_seq_len"], c["compress_rope_theta"],
            c["yarn_factor"], c["yarn_beta_fast"], c["yarn_beta_slow"]), persistent=False)
        self.apply(self._init)
        self._init_2d()

    def _init_2d(self):
        # every 2D parameter (projections, hc_*_fn, gate, expert stacks) gets
        # N(0, 0.02); 1D parameters keep their explicit init (norms = 1,
        # sinks/bases = 0, scales = 1)
        for _, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, mean=0.0, std=0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        """ids [B, T] -> logits [B, T, V] (Transformer.forward, model.py:913-926)."""
        B, T = ids.shape
        x = self.embed(ids)
        state = x.unsqueeze(2).repeat(1, 1, HC, 1)
        for l, blk in enumerate(self.layers):
            angles = self.ang_win if self.c["compress_ratios"][l] == 0 else self.ang_cmp
            angles = angles[:T]
            if self.grad_ckpt and torch.is_grad_enabled():
                state = torch.utils.checkpoint.checkpoint(
                    lambda s, blk=blk, angles=angles: blk(s, ids, angles),
                    state, use_reentrant=False)
            else:
                state = blk(state, ids, angles)
        h = hc_head(state, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return self.head(rms(h, self.norm_w))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    experts = sum(p.numel() for n, p in model.named_parameters() if n.endswith((".moe.w1", ".moe.w2", ".moe.w3")))
    return total, experts


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    m = NanoDsModel().float().eval()
    total, experts = count_params(m)
    print(f"params : {total / 1e6:.1f} M (of which routed experts {experts / 1e6:.1f} M)")
    ids = torch.randint(0, 8200, (2, 256))
    out = m(ids)
    print("logits :", tuple(out.shape), "mean", out.mean().item(), "std", out.std().item())
