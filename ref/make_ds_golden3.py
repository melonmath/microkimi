#!/usr/bin/env python3
"""Generate ref/ds_golden3.json: DeepSeek-V4 sparse attention reference for
sequential decode over T tokens, on 3 layer types (window-only, overlap
compressor+indexer, dense compressor), plain torch, math copied verbatim from
/tmp/dsv4/model.py (Attention, Compressor, Indexer, precompute_freqs_cis,
apply_rotary_emb) with sparse_attn reimplemented without tilelang
(index-gather + softmax with attentional sink — same math as kernel.py:277-368).

Test tool only. Run: /home/node/venv/bin/python3 ref/make_ds_golden3.py
"""
import json
import math
import torch
import torch.nn.functional as F

OUT = "/workspace/microkimi-oss/ref/ds_golden3.json"
torch.manual_seed(4242)
g = torch.Generator().manual_seed(4242)

# micro dims (same as the Rust engine's microdeepseek config)
D, NH, HD, RD = 512, 8, 128, 64
QL, OL, OG, WIN = 128, 128, 8, 128
INH, IHD, ITOPK = 8, 128, 64
T = 10
EPS = 1e-6
YARN_FACTOR, BF, BS, ORIG = 16.0, 32, 1, 65536


def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)
    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        return torch.clamp(linear_func, 0, 1)
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(seqlen)
    return torch.outer(t, freqs)  # [seqlen, dim/2] angles


def apply_rotary_emb(x, angles, inverse=False):
    # x: [n_heads, head_dim]; angles: [dim/2] for one position (model.py:238-250)
    rd = angles.shape[0]
    x = x.clone()
    xc = torch.view_as_complex(x[..., -2 * rd:].float().unflatten(-1, (-1, 2)))
    f = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        f = f.conj()
    xc = xc * f
    x[..., -2 * rd:] = torch.view_as_real(xc).flatten(-2)
    return x


def fp8_rt(x, block=64):
    # act_quant(inplace) round-trip, ue8m0 pow2 scales, ±448 (kernel.py:40-125)
    y = x.clone().view(-1, block)
    amax = y.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 8)
    s = 2.0 ** e
    y = torch.clamp(y / s, -448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.view_as(x)


def fp4_rt(x, block=32):
    E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    y = x.clone().view(-1, block)
    amax = y.abs().amax(dim=1, keepdim=True).clamp(min=6e-38)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 128)
    s = 2.0 ** e
    q = torch.clamp(y / s, -6.0, 6.0)
    idx = (q.abs().unsqueeze(-1) - E2M1.abs()).abs().argmin(-1)
    idx = torch.where(q < 0, idx + 8, idx)
    return (E2M1[idx] * s).view_as(x)


def hadamard(x):
    # rotate_activation: x · H × n^-0.5, Sylvester construction (power of 2)
    n = x.shape[-1]
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (x @ H) * n ** -0.5


def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


class CompressorT:
    def __init__(self, ratio, head_dim, overlap, rotate):
        self.ratio, self.overlap, self.rotate = ratio, overlap, rotate
        coff = 1 + overlap
        self.coff = coff
        self.wkv = torch.randn(coff * head_dim, D, generator=g)
        self.wgate = torch.randn(coff * head_dim, D, generator=g)
        self.ape = torch.randn(ratio, coff * head_dim, generator=g)
        self.norm_w = torch.randn(head_dim, generator=g)
        self._dump = {}
        self.kv_state = torch.zeros(coff * ratio, coff * head_dim)
        self.score_state = torch.full((coff * ratio, coff * head_dim), float("-inf"))

    def step(self, x, pos, angles):
        # model.py:349-368 (decode path)
        ratio, coff, d = self.ratio, self.coff, self.wkv.shape[0]
        kv = self.wkv @ x
        score = self.wgate @ x + self.ape[pos % ratio]
        should = (pos + 1) % ratio == 0
        if self.overlap:
            self.kv_state[ratio + pos % ratio] = kv
            self.score_state[ratio + pos % ratio] = score
            if not should:
                return None
            kvf = torch.cat([self.kv_state[:ratio, :d // 2], self.kv_state[ratio:, d // 2:]], 0)
            scf = torch.cat([self.score_state[:ratio, :d // 2], self.score_state[ratio:, d // 2:]], 0)
            kv = (kvf * scf.softmax(0)).sum(0)
            self.kv_state[:ratio] = self.kv_state[ratio:]
            self.score_state[:ratio] = self.score_state[ratio:]
        else:
            self.kv_state[pos % ratio] = kv
            self.score_state[pos % ratio] = score
            if not should:
                return None
            kv = (self.kv_state * self.score_state.softmax(0)).sum(0)
        kv = rms(kv, self.norm_w)
        kv = apply_rotary_emb(kv.unsqueeze(0), angles[pos + 1 - ratio]).squeeze(0)
        if self.rotate:
            kv = fp4_rt(hadamard(kv))
        else:
            kv[:-RD] = fp8_rt(kv[:-RD], 64)
        return kv


class IndexerT(CompressorT):
    def __init__(self, ratio):
        super().__init__(ratio, IHD, True, True)
        self.wq_b = torch.randn(INH * IHD, QL, generator=g)
        self.weights_proj = torch.randn(INH, D, generator=g)
        self.cache = []
        self.softmax_scale = IHD ** -0.5
        self._dump = {"idx_wq_b": self.wq_b, "idx_weights_proj": self.weights_proj}

    def step(self, qr, x, pos, angles):
        done = super().step(x, pos, angles)
        if done is not None:
            self.cache.append(done)
        q = (self.wq_b @ qr).view(INH, IHD)
        q = apply_rotary_emb(q, angles[pos])
        q = fp4_rt(hadamard(q))
        w = self.weights_proj @ x * (self.softmax_scale * INH ** -0.5)
        n = len(self.cache)
        if n == 0:
            return torch.full((0,), -1, dtype=torch.long)
        kv = torch.stack(self.cache)
        sc = torch.einsum("hd,td->ht", q, kv).relu_() * w.unsqueeze(-1)
        sc = sc.sum(0)
        limit = pos // 4
        sc[limit + 1:] = float("-inf")
        k = min(ITOPK, n)
        return sc.topk(k)[1]


def sparse_attn(q, kv, sink, idxs, scale):
    # kernel.py:277-368 without tilelang: gather + softmax with sink
    nh, d = q.shape
    o = torch.zeros(nh, d)
    for h in range(nh):
        valid = idxs >= 0
        kvv = kv[idxs[valid]]
        s = (kvv @ q[h]) * scale
        m = s.max()
        z = (sink[h] - m).exp() + (s - m).exp().sum()
        o[h] = ((s - m).exp().unsqueeze(-1) * kvv).sum(0) / z
    return o


def layer_forward(x_tokens, ratio, with_indexer, angles, names):
    T = len(x_tokens)
    win = WIN
    ring = torch.zeros(win, HD)
    compressed = []
    comp = CompressorT(ratio, HD, ratio == 4, False) if ratio > 0 else None
    idx = IndexerT(4) if with_indexer else None
    W = {}
    W["wq_a"] = torch.randn(QL, D, generator=g)
    W["q_norm_w"] = torch.randn(QL, generator=g)
    W["wq_b"] = torch.randn(NH * HD, QL, generator=g)
    W["wkv"] = torch.randn(HD, D, generator=g)
    W["kv_norm_w"] = torch.randn(HD, generator=g)
    W["wo_a"] = torch.randn(OG * OL, NH * HD // OG, generator=g)
    W["wo_b"] = torch.randn(D, OG * OL, generator=g)
    W["attn_sink"] = torch.randn(NH, generator=g)
    outs = []
    for pos, x in enumerate(x_tokens):
        qr = rms(W["wq_a"] @ x, W["q_norm_w"])
        q = (W["wq_b"] @ qr).view(NH, HD)
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS)
        q = apply_rotary_emb(q, angles[pos])
        kv = rms(W["wkv"] @ x, W["kv_norm_w"])
        kv = apply_rotary_emb(kv.unsqueeze(0), angles[pos]).squeeze(0)
        kv[:-2 * angles.shape[1]] = fp8_rt(kv[:-2 * angles.shape[1]], 64)
        ring[pos % win] = kv
        topk = list(range(pos % win + 1, win)) if pos >= win - 1 else []
        if pos >= win - 1:
            start = pos % win
            topk = list(range(start + 1, win)) + list(range(0, start + 1))
        else:
            topk = list(range(0, pos + 1)) + [-1] * (win - pos - 1)
        if comp is not None:
            done = comp.step(x, pos, angles)
            if done is not None:
                compressed.append(done)
            if with_indexer:
                ctop = idx.step(qr, x, pos, angles)
                topk = topk + (ctop + win).tolist()
            else:
                topk = topk + [t + win for t in range((pos + 1) // ratio)]
        kvfull = torch.cat([ring, torch.stack(compressed) if compressed else torch.zeros(0, HD)], 0)
        o = sparse_attn(q, kvfull, W["attn_sink"], torch.tensor(topk), HD ** -0.5)
        o = apply_rotary_emb(o, angles[pos], inverse=True)
        lat = torch.einsum("grd,gd->gr", W["wo_a"].view(OG, OL, NH * HD // OG), o.view(OG, -1))
        outs.append((W["wo_b"] @ lat.flatten()).tolist())
    dump = {k: v.flatten().tolist() for k, v in W.items()}
    if comp is not None:
        dump.update({"comp_wkv": comp.wkv.flatten().tolist(), "comp_wgate": comp.wgate.flatten().tolist(),
                     "comp_ape": comp.ape.flatten().tolist(), "comp_norm_w": comp.norm_w.flatten().tolist()})
    if with_indexer:
        dump.update({"idx_wq_b": idx.wq_b.flatten().tolist(), "idx_weights_proj": idx.weights_proj.flatten().tolist(),
                     "idx_comp_wkv": idx.wkv.flatten().tolist(), "idx_comp_wgate": idx.wgate.flatten().tolist(),
                     "idx_comp_ape": idx.ape.flatten().tolist(), "idx_comp_norm_w": idx.norm_w.flatten().tolist()})
    return outs, dump


x_tokens = torch.randn(T, D, generator=g)
x_tokens = [x_tokens[t] for t in range(T)]
MAXSEQ = 4096
golden = {"T": T, "D": D, "x": torch.stack(x_tokens).flatten().tolist()}

# L0: window-only, theta 10000, no YaRN
ang0 = precompute_freqs_cis(RD, MAXSEQ, 0, 10000.0, YARN_FACTOR, BF, BS)
golden["rope_theta10000"] = {"cos": torch.cos(ang0).flatten().tolist(), "sin": torch.sin(ang0).flatten().tolist()}
o0, d0 = layer_forward(x_tokens, 0, False, ang0, "L0")
golden["layer_window"] = {"out": torch.tensor(o0).flatten().tolist(), "weights": d0}

# L1: overlap compressor + indexer, theta 160000 + YaRN
ang1 = precompute_freqs_cis(RD, MAXSEQ, ORIG, 160000.0, YARN_FACTOR, BF, BS)
golden["rope_compress"] = {"cos": torch.cos(ang1).flatten().tolist(), "sin": torch.sin(ang1).flatten().tolist()}
o1, d1 = layer_forward(x_tokens, 4, True, ang1, "L1")
golden["layer_overlap_indexer"] = {"out": torch.tensor(o1).flatten().tolist(), "weights": d1}

# L2: dense compressor (ratio 8 for the test), theta 160000 + YaRN
o2, d2 = layer_forward(x_tokens, 8, False, ang1, "L2")
golden["layer_dense"] = {"out": torch.tensor(o2).flatten().tolist(), "weights": d2}

with open(OUT, "w") as f:
    json.dump(golden, f)
print(f"written {OUT}")
