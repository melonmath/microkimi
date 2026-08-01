#!/usr/bin/env python3
"""End-to-end microdeepseek parity (TEST tool, not a product dependency).

Plain-torch replica of the full DeepSeek-V4 Transformer (/tmp/dsv4/model.py:
Transformer/Block/Attention/Compressor/Indexer/Gate/MoE/Expert, plus
hc_split_sinkhorn from kernel.py) on the micro dims of microdeepseek.bin,
driven by the weights READ FROM THAT BIN (MKIM0002), strict fp32, sequential
decode (which is mathematically identical to the reference prefill thanks to
the -inf score masking). No tilelang: sparse_attn is reimplemented as
index-gather + online softmax with attentional sink (kernel.py:277-368).

Dumps ref/ds_parity_golden.json: hidden HC states after blocks
[0, 1, 2, 3, 21, 42] at positions [0, 3, 7, 127, 129], router selections for
layers [1, 3, 42] at every position, final logits (last position) and the
top-16 logits ids/values at every position.

Run: /home/node/venv/bin/python3 ref/make_ds_parity.py
"""
import json
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from mxfp4_numpy import dequant_mxfp4

BIN = os.environ.get("MICRODEEPSEEK_BIN", os.path.join(_HERE, "..", "microdeepseek.bin"))
OUT = os.path.join(_HERE, "ds_parity_golden.json")

T = 132
POS_DUMP = [0, 3, 7, 127, 129]
LAYER_DUMP = [0, 1, 2, 3, 21, 42]
ROUTER_LAYERS = [1, 3, 42]
EPS = 1e-6
HC_EPS = 1e-6
HC_ITERS = 20
RD = 64  # qk_rope_head_dim

# ── MKIM0002 parser ──

def load_bin(path):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert data[:8].tobytes() == b"MKIM0002", "bad magic"
    clen = int.from_bytes(data[8:12], "little")
    cfg = json.loads(data[12:12 + clen].tobytes())
    off = 12 + clen
    n = int.from_bytes(data[off:off + 4], "little"); off += 4
    tensors = {}
    for _ in range(n):
        nl = int.from_bytes(data[off:off + 2], "little"); off += 2
        name = data[off:off + nl].tobytes().decode(); off += nl
        dt, nd = int(data[off]), int(data[off + 1]); off += 2
        dims = [int.from_bytes(data[off + 4 * i:off + 4 * i + 4], "little") for i in range(nd)]
        off += 4 * nd
        o = int.from_bytes(data[off:off + 8], "little")
        s = int.from_bytes(data[off + 8:off + 16], "little"); off += 16
        tensors[name] = (dt, dims, o, s)
    return data, cfg, tensors


def t_f32(data, t):
    dt, dims, o, s = t
    assert dt == 0
    cnt = 1
    for d in dims:
        cnt *= d
    arr = np.frombuffer(data, dtype=np.float32, count=cnt, offset=o)
    return torch.from_numpy(np.ascontiguousarray(arr.reshape(dims)))


def t_i32(data, t):
    dt, dims, o, s = t
    assert dt == 2
    cnt = 1
    for d in dims:
        cnt *= d
    arr = np.frombuffer(data, dtype=np.int32, count=cnt, offset=o)
    return torch.from_numpy(np.ascontiguousarray(arr.reshape(dims)).astype(np.int64))


def t_fp4(data, t):
    # mxfp4 blob: packed (R×C/2) then scales (R×C/32), dims = logical [R, C]
    dt, dims, o, s = t
    assert dt == 1
    r, c = dims
    packed = np.frombuffer(data, dtype=np.uint8, count=r * c // 2, offset=o).reshape(r, c // 2)
    scales = np.frombuffer(data, dtype=np.uint8, count=r * c // 32, offset=o + r * c // 2).reshape(r, c // 32)
    return torch.from_numpy(dequant_mxfp4(packed, scales))


# ── math (verbatim replicas — see deepseek.rs for the Rust mirror) ──

def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations):
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
        ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (max(high, low + 1) - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return torch.outer(torch.arange(seqlen), freqs)  # [seqlen, dim/2] angles


def apply_rotary_emb(x, angles, inverse=False):
    # x: [n_heads, head_dim]; angles: [rd/2] for one position
    x = x.clone()
    xc = torch.view_as_complex(x[..., -2 * angles.shape[0]:].float().unflatten(-1, (-1, 2)))
    f = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        f = f.conj()
    xc = xc * f
    x[..., -2 * angles.shape[0]:] = torch.view_as_real(xc).flatten(-2)
    return x


def fp8_rt(x, block=64):
    y = x.clone().view(-1, block)
    amax = y.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 8)
    s = 2.0 ** e
    y = torch.clamp(y / s, -448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.view_as(x)


_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

def fp4_rt(x, block=32):
    y = x.clone().view(-1, block)
    amax = y.abs().amax(dim=1, keepdim=True).clamp(min=6e-38)
    e = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 128)
    s = 2.0 ** e
    q = torch.clamp(y / s, -6.0, 6.0)
    idx = (q.abs().unsqueeze(-1) - _E2M1.abs()).abs().argmin(-1)
    idx = torch.where(q < 0, idx + 8, idx)
    return (_E2M1[idx] * s).view_as(x)


def hadamard(x):
    n = x.shape[-1]
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (x @ H) * n ** -0.5


def rms(x, w, eps=EPS):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * w


def sinkhorn(comb, hc):
    # softmax rows + eps; colnorm+eps; then (iters-1) × (rownorm+eps, colnorm+eps)
    m = torch.softmax(comb.view(hc, hc), dim=1) + HC_EPS
    m = m / (m.sum(0, keepdim=True) + HC_EPS)
    for _ in range(1, HC_ITERS):
        m = m / (m.sum(1, keepdim=True) + HC_EPS)
        m = m / (m.sum(0, keepdim=True) + HC_EPS)
    return m


def hc_pre(state, fn, scale, base, hc):
    # state: [hc*d]; fn: [mix, hc*d]; returns (y [d], post [hc], comb [hc, hc])
    x = state.float()
    rsqrt = torch.rsqrt(x.square().mean() + EPS)
    mixes = fn @ x * rsqrt
    pre = torch.sigmoid(mixes[:hc] * scale[0] + base[:hc]) + HC_EPS
    post = 2.0 * torch.sigmoid(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc])
    comb = sinkhorn(mixes[2 * hc:] * scale[2] + base[2 * hc:], hc)
    y = (pre.unsqueeze(-1) * state.view(hc, -1)).sum(0)
    return y, post, comb


def hc_post(x, residual, post, comb, hc):
    # y[j] = post[j]·x + Σ_k comb[k, j]·residual[k]
    return post.unsqueeze(-1) * x.unsqueeze(0) + (comb.t().unsqueeze(-1) * residual.view(hc, -1).unsqueeze(0)).sum(1)


def hc_head(state, fn, scale, base, hc):
    x = state.float()
    rsqrt = torch.rsqrt(x.square().mean() + EPS)
    mixes = fn @ x * rsqrt
    pre = torch.sigmoid(mixes * scale + base) + HC_EPS
    return (pre.unsqueeze(-1) * state.view(hc, -1)).sum(0)


def softplus_sqrt(x):
    return torch.nn.functional.softplus(x).sqrt()


def sparse_attn_sink(q, kvfull, sink, idxs, scale):
    nh, d = q.shape
    o = torch.zeros(nh, d)
    for h in range(nh):
        valid = idxs >= 0
        kvv = kvfull[idxs[valid]]
        s = (kvv @ q[h]) * scale
        m = s.max()
        z = (sink[h] - m).exp() + (s - m).exp().sum()
        o[h] = ((s - m).exp().unsqueeze(-1) * kvv).sum(0) / z
    return o


class Compressor:
    def __init__(self, wkv, wgate, ape, norm_w, ratio, overlap, rotate):
        self.wkv, self.wgate, self.ape, self.norm_w = wkv, wgate, ape, norm_w
        self.ratio, self.overlap, self.rotate = ratio, overlap, rotate
        coff = 2 if overlap else 1
        self.coff = coff
        d = wkv.shape[0]
        self.kv_state = torch.zeros(coff * ratio, d)
        self.score_state = torch.full((coff * ratio, d), float("-inf"))

    def step(self, x, pos, angles):
        ratio, coff, cd = self.ratio, self.coff, self.wkv.shape[0]
        hd = cd // coff
        kv = self.wkv @ x
        score = self.wgate @ x + self.ape[pos % ratio]
        should = (pos + 1) % ratio == 0
        if self.overlap:
            self.kv_state[ratio + pos % ratio] = kv
            self.score_state[ratio + pos % ratio] = score
            if not should:
                return None
            kvf = torch.cat([self.kv_state[:ratio, :hd], self.kv_state[ratio:, hd:]], 0)
            scf = torch.cat([self.score_state[:ratio, :hd], self.score_state[ratio:, hd:]], 0)
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


class LayerState:
    def __init__(self, cfg, l, tensors, data, angles):
        self.l = l
        self.ratio = cfg["ds"]["compress_ratios"][l]
        self.angles = angles
        p = f"layers.{l}."
        g = lambda n: t_f32(data, tensors[p + n])
        self.w = {
            "attn_norm": g("attn_norm.weight"), "ffn_norm": g("ffn_norm.weight"),
            "hc_attn_fn": g("hc_attn_fn"), "hc_attn_base": g("hc_attn_base"), "hc_attn_scale": g("hc_attn_scale"),
            "hc_ffn_fn": g("hc_ffn_fn"), "hc_ffn_base": g("hc_ffn_base"), "hc_ffn_scale": g("hc_ffn_scale"),
            "wq_a": g("attn.wq_a.weight"), "q_norm": g("attn.q_norm.weight"),
            "wq_b": g("attn.wq_b.weight"), "wkv": g("attn.wkv.weight"),
            "kv_norm": g("attn.kv_norm.weight"), "wo_a": g("attn.wo_a.weight"),
            "wo_b": g("attn.wo_b.weight"), "attn_sink": g("attn.attn_sink"),
            "gate": g("ffn.gate.weight"),
            "sh1": g("ffn.shared_experts.w1.weight"), "sh2": g("ffn.shared_experts.w2.weight"),
            "sh3": g("ffn.shared_experts.w3.weight"),
        }
        self.gate_bias = g("ffn.gate.bias") if l >= cfg["ds"]["num_hash_layers"] else None
        self.tid2eid = t_i32(data, tensors[p + "ffn.gate.tid2eid"]) if l < cfg["ds"]["num_hash_layers"] else None
        self.ring = torch.zeros(cfg["ds"]["sliding_window"], cfg["ds"]["head_dim"])
        self.compressed = []
        if self.ratio > 0:
            self.comp = Compressor(g("attn.compressor.wkv.weight"), g("attn.compressor.wgate.weight"),
                                   g("attn.compressor.ape"), g("attn.compressor.norm.weight"),
                                   self.ratio, self.ratio == 4, False)
        else:
            self.comp = None
        if self.ratio == 4:
            self.idx_comp = Compressor(g("attn.indexer.compressor.wkv.weight"), g("attn.indexer.compressor.wgate.weight"),
                                       g("attn.indexer.compressor.ape"), g("attn.indexer.compressor.norm.weight"),
                                       4, True, True)
            self.idx_wq_b = g("attn.indexer.wq_b.weight")
            self.idx_weights_proj = g("attn.indexer.weights_proj.weight")
            self.idx_cache = []
        # fp4 experts stay packed in the memmap; dequantized on demand (cached)
        self._exp_cache = {}
        self._tensors = tensors
        self._data = data
        self._p = p

    def expert(self, eid):
        if eid not in self._exp_cache:
            p = f"{self._p}ffn.experts.{eid}."
            self._exp_cache[eid] = (
                t_fp4(self._data, self._tensors[p + "w1"]),
                t_fp4(self._data, self._tensors[p + "w2"]),
                t_fp4(self._data, self._tensors[p + "w3"]),
            )
        return self._exp_cache[eid]


def expert_forward(w1, w2, w3, x, limit, weight=None):
    gate = w1 @ x
    up = (w3 @ x).clamp(-limit, limit)
    gate = gate.clamp(max=limit)
    act = torch.nn.functional.silu(gate) * up
    if weight is not None:
        act = weight * act
    return w2 @ act


def main():
    data, cfg, tensors = load_bin(BIN)
    ds = cfg["ds"]
    D = ds["hidden"]
    NH, HD = ds["n_heads"], ds["head_dim"]
    QL, OL, OG = ds["q_lora_rank"], ds["o_lora_rank"], ds["o_groups"]
    WIN = ds["sliding_window"]
    INH, IHD, ITOPK = ds["index_n_heads"], ds["index_head_dim"], ds["index_topk"]
    NLAY = ds["n_layers"]
    HC = 4
    TOPK = ds["num_experts_per_tok"]
    RSCALE = ds["routed_scaling_factor"]
    LIMIT = ds["swiglu_limit"]
    MAXSEQ = 4096

    angles_win = precompute_freqs_cis(RD, MAXSEQ, 0, ds["rope_theta"], 16.0, 32, 1)
    angles_cmp = precompute_freqs_cis(RD, MAXSEQ, 65536, ds["compress_rope_theta"], 16.0, 32, 1)

    embed = t_f32(data, tensors["embed.weight"])
    head = t_f32(data, tensors["head.weight"])
    norm_w = t_f32(data, tensors["norm.weight"])
    hc_head_fn = t_f32(data, tensors["hc_head_fn"])
    hc_head_base = t_f32(data, tensors["hc_head_base"])
    hc_head_scale = t_f32(data, tensors["hc_head_scale"])[0]

    layers = []
    for l in range(NLAY):
        ratio = ds["compress_ratios"][l]
        angles = angles_win if ratio == 0 else angles_cmp
        layers.append(LayerState(cfg, l, tensors, data, angles))
        if l % 10 == 0:
            print(f"  layer {l}/{NLAY} loaded", flush=True)

    rng = np.random.default_rng(20260731)
    ids = [0] + rng.integers(3, 16384, T - 5).tolist() + [50000, 100000, 129279, 2]
    assert len(ids) == T

    hiddens = {}
    router = {}
    logits_top = []
    logits_last = None

    for pos, tok in enumerate(ids):
        state = embed[tok].repeat(HC)  # [HC*D]
        for l, st in enumerate(layers):
            w = st.w
            # attention sublayer
            y, post, comb = hc_pre(state, w["hc_attn_fn"], w["hc_attn_scale"], w["hc_attn_base"], HC)
            yn = rms(y, w["attn_norm"])
            angles = st.angles
            qr = rms(w["wq_a"] @ yn, w["q_norm"])
            q = (w["wq_b"] @ qr).view(NH, HD)
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + EPS)
            q = apply_rotary_emb(q, angles[pos])
            kv = rms(w["wkv"] @ yn, w["kv_norm"])
            kv = apply_rotary_emb(kv.unsqueeze(0), angles[pos]).squeeze(0)
            kv[:-RD] = fp8_rt(kv[:-RD], 64)
            st.ring[pos % WIN] = kv
            if pos >= WIN - 1:
                start = pos % WIN
                topk = list(range(start + 1, WIN)) + list(range(0, start + 1))
            else:
                topk = list(range(0, pos + 1)) + [-1] * (WIN - pos - 1)
            if st.comp is not None:
                done = st.comp.step(yn, pos, angles)
                if done is not None:
                    st.compressed.append(done)
                if st.ratio == 4:
                    idone = st.idx_comp.step(yn, pos, angles)
                    if idone is not None:
                        st.idx_cache.append(idone)
                    iq = (st.idx_wq_b @ qr).view(INH, IHD)
                    iq = apply_rotary_emb(iq, angles[pos])
                    iq = fp4_rt(hadamard(iq))
                    iw = st.idx_weights_proj @ yn * ((IHD ** -0.5) * INH ** -0.5)
                    n = len(st.idx_cache)
                    if n > 0:
                        ikvm = torch.stack(st.idx_cache)
                        sc = torch.einsum("hd,td->ht", iq, ikvm).relu_() * iw.unsqueeze(-1)
                        sc = sc.sum(0)
                        sc[(pos // 4) + 1:] = float("-inf")
                        ctop = sc.topk(min(ITOPK, n))[1] + WIN
                        topk = topk + ctop.tolist()
                else:
                    topk = topk + [t + WIN for t in range((pos + 1) // st.ratio)]
            kvfull = torch.cat([st.ring, torch.stack(st.compressed) if st.compressed else torch.zeros(0, HD)], 0)
            o = sparse_attn_sink(q, kvfull, w["attn_sink"], torch.tensor(topk), HD ** -0.5)
            o = apply_rotary_emb(o, angles[pos], inverse=True)
            lat = torch.einsum("grd,gd->gr", w["wo_a"].view(OG, OL, NH * HD // OG), o.reshape(OG, -1))
            attn_out = w["wo_b"] @ lat.flatten()
            state = hc_post(attn_out, state, post, comb, HC).flatten()
            # ffn sublayer
            y, post, comb = hc_pre(state, w["hc_ffn_fn"], w["hc_ffn_scale"], w["hc_ffn_base"], HC)
            yn = rms(y, w["ffn_norm"])
            scores = softplus_sqrt(w["gate"] @ yn)
            if st.tid2eid is not None:
                sel = st.tid2eid[tok].tolist()
            else:
                sel = (scores + st.gate_bias).topk(TOPK)[1].tolist()
            wsel = scores[sel]
            wsel = wsel / wsel.sum() * RSCALE
            if l in ROUTER_LAYERS:
                router[f"{pos},{l}"] = sorted(sel)
            moe = torch.zeros(D)
            for eid, wgt in zip(sel, wsel):
                w1, w2, w3 = st.expert(eid)
                moe += expert_forward(w1, w2, w3, yn, LIMIT, wgt)
            moe += expert_forward(w["sh1"], w["sh2"], w["sh3"], yn, LIMIT)
            state = hc_post(moe, state, post, comb, HC).flatten()
            if l in LAYER_DUMP and pos in POS_DUMP:
                hiddens[f"{pos},{l}"] = state.tolist()
        h = hc_head(state, hc_head_fn, hc_head_scale, hc_head_base, HC)
        xn = rms(h, norm_w)
        logits = head @ xn
        top = logits.topk(16)
        logits_top.append({"ids": top.indices.tolist(), "vals": top.values.tolist()})
        logits_last = logits.tolist()
        if pos % 20 == 0:
            print(f"  pos {pos}/{T} done", flush=True)

    golden = {
        "ids": ids, "T": T, "pos_dump": POS_DUMP, "layer_dump": LAYER_DUMP,
        "hiddens": hiddens, "router": router,
        "logits_last": logits_last, "logits_top": logits_top,
    }
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
