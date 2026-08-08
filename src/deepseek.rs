// DeepSeek-V4 building blocks (microdeepseek): MoE routing (sqrtsoftplus,
// noaux_tc bias, hash layers) and Hyper-Connections with Sinkhorn.
// Math follows /tmp/dsv4/model.py (DeepSeek AI reference) and
// /tmp/dsv4/kernel.py (hc_split_sinkhorn) exactly, f32 throughout.

use crate::model::{dot, matvec};

// ── MoE Gate (model.py:551-589) ──
//
// scores = sqrt(softplus(x @ gate_w)) ("sqrtsoftplus"); the bias is added ONLY
// for top-k selection (noaux_tc), routing weights are gathered from the
// ORIGINAL (bias-free) scores, renormalized to sum 1, then × route_scale.
// For hash layers (layer < n_hash_layers), indices come from tid2eid[token_id]
// instead of top-k (weights still gathered from the original scores).
#[allow(clippy::too_many_arguments)]
pub fn gate_forward(
    x: &[f32],
    gate_w: &[f32], // [n_experts, d]
    bias: Option<&[f32]>,
    tid2eid: Option<&[i32]>, // [vocab, topk] when hash routing
    token_id: u32,
    n_experts: usize,
    topk: usize,
    route_scale: f32,
) -> (Vec<(u32, f32)>, Vec<f32>) {
    let d = x.len();
    let mut logits = vec![0f32; n_experts];
    matvec(gate_w, n_experts, d, x, &mut logits);
    let mut orig = vec![0f32; n_experts];
    for (o, &l) in orig.iter_mut().zip(&logits) {
        *o = softplus(l).sqrt();
    }
    let indices: Vec<u32> = match tid2eid {
        Some(table) => (0..topk).map(|i| table[token_id as usize * topk + i] as u32).collect(),
        None => {
            let mut sel: Vec<(u32, f32)> = Vec::with_capacity(topk);
            for (i, &s) in orig.iter().enumerate() {
                let key = s + bias.map_or(0.0, |b| b[i]);
                if sel.len() < topk {
                    sel.push((i as u32, key));
                    if sel.len() == topk {
                        sel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    }
                } else if key > sel[topk - 1].1 {
                    sel[topk - 1] = (i as u32, key);
                    sel.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                }
            }
            sel.iter().map(|s| s.0).collect()
        }
    };
    let mut weights: Vec<f32> = indices.iter().map(|&i| orig[i as usize]).collect();
    let sum: f32 = weights.iter().sum();
    for w in weights.iter_mut() {
        *w = *w / sum * route_scale;
    }
    (indices.into_iter().zip(weights).collect(), orig)
}

#[inline]
fn softplus(x: f32) -> f32 {
    // torch.nn.functional.softplus: log1p(exp(x)) with the threshold at 20
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

// ── Expert (model.py:592-611): silu(w1·x)·clamp(w3·x), then w2 ──
//
// gate = w1·x (f32), up = clamp(w3·x, ±limit); gate clamped to ≤ limit;
// act = silu(gate) * up; out = w2·act. limit = swiglu_limit (10 in V4).
pub fn expert_forward(
    w1: &[f32], // [inter, d]
    w2: &[f32], // [d, inter]
    w3: &[f32], // [inter, d]
    x: &[f32],
    inter: usize,
    limit: f32,
    out: &mut [f32],
) {
    let d = x.len();
    let mut gate = vec![0f32; inter];
    let mut up = vec![0f32; inter];
    matvec(w1, inter, d, x, &mut gate);
    matvec(w3, inter, d, x, &mut up);
    let mut act = vec![0f32; inter];
    for i in 0..inter {
        let g = gate[i].min(limit);
        let u = up[i].clamp(-limit, limit);
        act[i] = silu(g) * u;
    }
    matvec(w2, d, inter, &act, out);
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── Hyper-Connections (model.py:680-716, kernel.py:371-438) ──

/// Sinkhorn normalization of the combination matrix (kernel.py:401-423):
/// comb = softmax(rows) + eps; then colnorm+eps; then (iters-1) ×
/// (rownorm+eps, colnorm+eps). hc_mult = 4, iters = 20, eps = 1e-6 in V4.
pub fn sinkhorn(mixes_comb: &mut [f32], hc: usize, iters: usize, eps: f32) {
    // softmax over rows + eps
    for j in 0..hc {
        let row = &mut mixes_comb[j * hc..(j + 1) * hc];
        let m = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut z = 0f32;
        for v in row.iter_mut() {
            *v = (*v - m).exp();
            z += *v;
        }
        for v in row.iter_mut() {
            *v = *v / z + eps;
        }
    }
    // column normalize + eps
    colnorm(mixes_comb, hc, eps);
    for _ in 1..iters {
        rownorm(mixes_comb, hc, eps);
        colnorm(mixes_comb, hc, eps);
    }
}

fn rownorm(m: &mut [f32], hc: usize, eps: f32) {
    for j in 0..hc {
        let row = &mut m[j * hc..(j + 1) * hc];
        let z: f32 = row.iter().sum();
        for v in row.iter_mut() {
            *v = *v / (z + eps);
        }
    }
}

fn colnorm(m: &mut [f32], hc: usize, eps: f32) {
    for k in 0..hc {
        let z: f32 = (0..hc).map(|j| m[j * hc + k]).sum();
        for j in 0..hc {
            m[j * hc + k] = m[j * hc + k] / (z + eps);
        }
    }
}

/// hc_pre (model.py:680-688): mixes = (x_flat @ hc_fn) * rsqrt(mean(x_flat²)+norm_eps);
/// then split into pre/post/comb through sigmoids and Sinkhorn.
/// Returns (y [d], post [hc], comb [hc*hc]).
pub fn hc_pre(
    x_flat: &[f32],     // [hc*d]
    x_state: &[f32],    // same data viewed as [hc, d] (row-major)
    hc_fn: &[f32],      // [mix_hc, hc*d]
    hc_scale: &[f32],   // [3]
    hc_base: &[f32],    // [mix_hc]
    hc: usize,
    norm_eps: f32,
    sinkhorn_iters: usize,
    hc_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = x_flat.len() / hc;
    let mix_hc = (2 + hc) * hc;
    let ss = dot(x_flat, x_flat) / x_flat.len() as f32;
    let rsqrt = 1.0 / (ss + norm_eps).sqrt();
    let mut mixes = vec![0f32; mix_hc];
    matvec(hc_fn, mix_hc, x_flat.len(), x_flat, &mut mixes);
    for m in mixes.iter_mut() {
        *m *= rsqrt;
    }
    let pre: Vec<f32> = (0..hc)
        .map(|j| sigmoid(mixes[j] * hc_scale[0] + hc_base[j]) + hc_eps)
        .collect();
    let post: Vec<f32> = (0..hc)
        .map(|j| 2.0 * sigmoid(mixes[j + hc] * hc_scale[1] + hc_base[j + hc]))
        .collect();
    let mut comb: Vec<f32> = (0..hc * hc)
        .map(|i| mixes[i + 2 * hc] * hc_scale[2] + hc_base[i + 2 * hc])
        .collect();
    sinkhorn(&mut comb, hc, sinkhorn_iters, hc_eps);
    // y = Σ_j pre[j] * x_state[j]
    let mut y = vec![0f32; d];
    for j in 0..hc {
        for i in 0..d {
            y[i] += pre[j] * x_state[j * d + i];
        }
    }
    (y, post, comb)
}

/// hc_post (model.py:690-693): y = post ⊗ x + Σ comb ⊗ residual.
/// NB: torch.sum(..., dim=2) sums over the FIRST index of comb:
/// y[j, i] = post[j]·x[i] + Σ_k comb[k, j]·residual[k, i].
pub fn hc_post(x: &[f32], residual: &[f32], post: &[f32], comb: &[f32], hc: usize) -> Vec<f32> {
    let d = x.len();
    let mut y = vec![0f32; hc * d];
    for j in 0..hc {
        for i in 0..d {
            let mut acc = post[j] * x[i];
            for k in 0..hc {
                acc += comb[k * hc + j] * residual[k * d + i];
            }
            y[j * d + i] = acc;
        }
    }
    y
}

/// hc_head (model.py:709-716): like hc_pre's pre branch without Sinkhorn:
/// pre = sigmoid(mixes * scale + base) + eps; y = Σ pre ⊗ x.
pub fn hc_head(x_flat: &[f32], x_state: &[f32], hc_fn: &[f32], hc_scale: f32, hc_base: &[f32], hc: usize, norm_eps: f32, hc_eps: f32) -> Vec<f32> {
    let d = x_flat.len() / hc;
    let ss = dot(x_flat, x_flat) / x_flat.len() as f32;
    let rsqrt = 1.0 / (ss + norm_eps).sqrt();
    let mut mixes = vec![0f32; hc];
    matvec(hc_fn, hc, x_flat.len(), x_flat, &mut mixes);
    for m in mixes.iter_mut() {
        *m *= rsqrt;
    }
    let mut y = vec![0f32; d];
    for j in 0..hc {
        let p = sigmoid(mixes[j] * hc_scale + hc_base[j]) + hc_eps;
        for i in 0..d {
            y[i] += p * x_state[j * d + i];
        }
    }
    y
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ════════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 sparse attention (model.py:205-548, kernel.py:277-368)
// ════════════════════════════════════════════════════════════════════════════

use crate::config::DsConfig;

/// RMSNorm with explicit eps (V4 uses 1e-6; the K3 rmsnorm is 1e-5).
pub fn ds_rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ss = crate::model::dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

// ── RoPE / YaRN (model.py:205-250) ──
// Interleaved-pair convention (view_as_complex): pairs are ADJACENT elements
// (x[2i], x[2i+1]), rotated by pos·freq[i]. NOT the half-split convention.

/// Precompute (cos, sin) tables: [n_pos][dim/2]. YaRN applied when
/// original_seq_len > 0 (compressed layers: base=160000, factor=16).
/// Everything is computed in f32 to match the torch reference bit-closely
/// (angle rounding at large positions dominates the comparison).
pub fn precompute_freqs_cis(dim: usize, n_pos: usize, original_seq_len: i32, base: f64, factor: f64, beta_fast: i32, beta_slow: i32) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let base32 = base as f32;
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / base32.powf(2.0 * i as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 {
        let cdim = |num_rot: f64| {
            dim as f64 * ((original_seq_len as f64) / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
        };
        let low = (cdim(beta_fast as f64).floor() as i32).max(0) as usize;
        let high = (cdim(beta_slow as f64).ceil() as i32).min(dim as i32 - 1) as usize;
        for i in 0..half {
            let ramp = ((i as f32 - low as f32) / ((high.max(low + 1)) as f32 - low as f32)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            freqs[i] = freqs[i] / factor as f32 * (1.0 - smooth) + freqs[i] * smooth;
        }
    }
    let mut cos = vec![0f32; n_pos * half];
    let mut sin = vec![0f32; n_pos * half];
    for p in 0..n_pos {
        for i in 0..half {
            let angle = p as f32 * freqs[i];
            cos[p * half + i] = angle.cos();
            sin[p * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// apply_rotary_emb on the trailing `rope_dim` dims of each head vector:
/// pairs (x[2i], x[2i+1]) rotated by (cos[p], sin[p]); inverse = conjugate.
pub fn apply_rotary(x: &mut [f32], n_heads: usize, head_dim: usize, rope_dim: usize, cos: &[f32], sin: &[f32], pos: usize, inverse: bool) {
    let half = rope_dim / 2;
    let off = head_dim - rope_dim;
    for h in 0..n_heads {
        let base = h * head_dim + off;
        for i in 0..half {
            let c = cos[pos * half + i];
            let s = if inverse { -sin[pos * half + i] } else { sin[pos * half + i] };
            let x0 = x[base + 2 * i];
            let x1 = x[base + 2 * i + 1];
            x[base + 2 * i] = x0 * c - x1 * s;
            x[base + 2 * i + 1] = x0 * s + x1 * c;
        }
    }
}

// ── FP8/FP4 QAT round-trips (act_quant / fp4_act_quant, inplace=True) ──
// The reference quantizes then dequantizes in place to simulate QAT; for
// parity we replicate the exact round-trip.

/// act_quant(x, block, ue8m0, fp8, inplace=True): per-row-block pow2 scale,
/// clamp ±448, e4m3 round-trip. Block size 64 in the reference attention path.
pub fn fp8_roundtrip(x: &mut [f32], block: usize) {
    for chunk in x.chunks_mut(block) {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-4);
        let e = (amax / 448.0).log2().ceil() as i32;
        let e = e.clamp(-127, 8);
        let s = crate::quant::mxfp4::exp2_i(e);
        for v in chunk.iter_mut() {
            *v = crate::quant::dequant::e4m3_to_f32(crate::quant::dequant::f32_to_e4m3((*v / s).clamp(-448.0, 448.0))) * s;
        }
    }
}

/// fp4_act_quant(x, 32, inplace=True): pow2 scale per 32, e2m1 round-trip.
pub fn fp4_roundtrip(x: &mut [f32]) {
    const LUT: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];
    for chunk in x.chunks_mut(32) {
        let amax = chunk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(6.0 * f32::MIN_POSITIVE);
        let e = (amax / 6.0).log2().ceil() as i32;
        let e = e.clamp(-127, 128);
        let s = crate::quant::mxfp4::exp2_i(e);
        for v in chunk.iter_mut() {
            let q = (*v / s).clamp(-6.0, 6.0);
            // nearest e2m1 level (same rule as mxfp4::quantize)
            let mag = q.abs();
            let mut idx = 0usize;
            const B: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
            while idx < 7 && mag >= B[idx] {
                idx += 1;
            }
            if q.is_sign_negative() {
                idx += 8;
            }
            *v = LUT[idx] * s;
        }
    }
}

// ── Sparse attention layer state (per layer) ──

pub struct DsAttention {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub ring: Vec<f32>,       // [window, head_dim] circular window cache
    pub compressed: Vec<f32>, // [max_compressed, head_dim] compressed cache
    pub n_compressed: usize,
    pub comp_kv_state: Vec<f32>,    // [coff*ratio, coff*head_dim]
    pub comp_score_state: Vec<f32>, // same, init -inf
    // indexer state (ratio==4 layers): its own compressor
    pub idx_kv_state: Vec<f32>,
    pub idx_score_state: Vec<f32>,
    pub idx_compressed: Vec<f32>, // [max_compressed, index_head_dim]
    pub idx_n_compressed: usize,
    pub pos: usize, // absolute position of next token
}

impl DsAttention {
    pub fn new(cfg: &DsConfig, layer: usize) -> Self {
        let ratio = cfg.compress_ratio(layer);
        let (orig, base) = if ratio != 0 {
            (cfg.yarn_orig_seq_len as i32, cfg.compress_rope_theta)
        } else {
            (0, cfg.rope_theta)
        };
        let (cos, sin) = precompute_freqs_cis(cfg.rope_head_dim, cfg.max_seq_len, orig, base, cfg.yarn_factor, cfg.yarn_beta_fast, cfg.yarn_beta_slow);
        let coff = if ratio == 4 { 2 } else { 1 };
        let max_comp = if ratio > 0 { cfg.max_seq_len / ratio as usize } else { 0 };
        let comp_dim = coff * cfg.head_dim * (coff * ratio as usize).max(1);
        DsAttention {
            cos,
            sin,
            ring: vec![0.0; cfg.window_size * cfg.head_dim],
            compressed: vec![0.0; max_comp * cfg.head_dim],
            n_compressed: 0,
            comp_kv_state: vec![0.0; comp_dim],
            comp_score_state: vec![f32::NEG_INFINITY; comp_dim],
            // state dims: coff*ratio entries of coff*index_head_dim each (coff=2 for overlap)
            idx_kv_state: vec![0.0; if ratio == 4 { 2 * ratio as usize * (2 * cfg.index_head_dim) } else { 0 }],
            idx_score_state: vec![f32::NEG_INFINITY; if ratio == 4 { 2 * ratio as usize * (2 * cfg.index_head_dim) } else { 0 }],
            idx_compressed: vec![0.0; if ratio == 4 { max_comp * cfg.index_head_dim } else { 0 }],
            idx_n_compressed: 0,
            pos: 0,
        }
    }
}

/// Compressor forward for ONE token at absolute position `pos` (decode-path
/// equivalent of model.py:322-383 — prefill processes tokens sequentially,
/// which is mathematically identical thanks to -inf score masking).
/// Returns Some(compressed kv [head_dim]) when a window completes, else None.
#[allow(clippy::too_many_arguments)]
pub fn compressor_step(
    ratio: usize,
    overlap: bool,
    head_dim: usize,
    rope_dim: usize,
    wkv: &[f32],   // [coff*head_dim, d]
    wgate: &[f32], // [coff*head_dim, d]
    ape: &[f32],   // [ratio, coff*head_dim]
    norm_w: &[f32],
    x: &[f32],
    pos: usize,
    kv_state: &mut [f32],
    score_state: &mut [f32],
    norm_eps: f32,
    cos: &[f32],
    sin: &[f32],
    out: &mut Vec<f32>,
) -> Option<()> {
    let coff = if overlap { 2 } else { 1 };
    let cd = coff * head_dim;
    let d = x.len();
    let mut kv = vec![0f32; cd];
    let mut score = vec![0f32; cd];
    crate::model::matvec(wkv, cd, d, x, &mut kv);
    crate::model::matvec(wgate, cd, d, x, &mut score);
    for i in 0..cd {
        score[i] += ape[(pos % ratio) * cd + i];
    }
    let should_compress = (pos + 1) % ratio == 0;
    if overlap {
        // entries [ratio, 2*ratio) = current window; [0, ratio) = previous (overlap)
        kv_state[(ratio + pos % ratio) * cd..(ratio + pos % ratio + 1) * cd].copy_from_slice(&kv);
        score_state[(ratio + pos % ratio) * cd..(ratio + pos % ratio + 1) * cd].copy_from_slice(&score);
        if !should_compress {
            return None;
        }
        // full window: [prev.first_half, current.second_half]
        let n = 2 * ratio;
        let mut kvf = vec![0f32; n * head_dim];
        let mut scf = vec![f32::NEG_INFINITY; n * head_dim];
        for e in 0..n {
            let (src, off) = if e < ratio { (e, 0) } else { (e, head_dim) };
            for i in 0..head_dim {
                kvf[e * head_dim + i] = kv_state[src * cd + off + i];
                scf[e * head_dim + i] = score_state[src * cd + off + i];
            }
        }
        let mut out_kv = vec![0f32; head_dim];
        softmax_weighted_sum(&scf, &kvf, n, head_dim, &mut out_kv);
        // shift: previous := current
        let (l, r) = kv_state.split_at_mut(ratio * cd);
        l.copy_from_slice(&r[..]);
        let (l2, r2) = score_state.split_at_mut(ratio * cd);
        l2.copy_from_slice(&r2[..]);
        finish_compressed(out_kv, norm_w, norm_eps, head_dim, rope_dim, cos, sin, pos, ratio, out);
        Some(())
    } else {
        kv_state[(pos % ratio) * cd..(pos % ratio + 1) * cd].copy_from_slice(&kv);
        score_state[(pos % ratio) * cd..(pos % ratio + 1) * cd].copy_from_slice(&score);
        if !should_compress {
            return None;
        }
        let mut out_kv = vec![0f32; head_dim];
        softmax_weighted_sum(score_state, kv_state, ratio, head_dim, &mut out_kv);
        finish_compressed(out_kv, norm_w, norm_eps, head_dim, rope_dim, cos, sin, pos, ratio, out);
        Some(())
    }
}

fn softmax_weighted_sum(scores: &[f32], kvs: &[f32], n: usize, dim: usize, out: &mut [f32]) {
    // softmax ALONG the entry axis, PER dimension (torch softmax(dim=0)):
    // per-column max and per-column normalizer.
    for i in 0..dim {
        let m = (0..n).map(|e| scores[e * dim + i]).fold(f32::NEG_INFINITY, |a, b| a.max(b));
        let mut z = 0f32;
        for e in 0..n {
            z += (scores[e * dim + i] - m).exp();
        }
        let mut acc = 0f32;
        for e in 0..n {
            acc += (scores[e * dim + i] - m).exp() / z * kvs[e * dim + i];
        }
        out[i] = acc;
    }
}

fn finish_compressed(
    mut kv: Vec<f32>,
    norm_w: &[f32],
    norm_eps: f32,
    head_dim: usize,
    rope_dim: usize,
    cos: &[f32],
    sin: &[f32],
    pos: usize,
    ratio: usize,
    out: &mut Vec<f32>,
) {
    // RMSNorm over head_dim with weight (Compressor.norm, model.py:305,368)
    let ss = kv.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
    let inv = 1.0 / (ss + norm_eps).sqrt();
    for i in 0..head_dim {
        kv[i] *= inv * norm_w[i];
    }
    // RoPE on the trailing rope_dim with the angle of the LAST token of the
    // window (model.py:372: freqs_cis[start_pos + 1 - ratio])
    let rope_pos = pos + 1 - ratio;
    apply_rotary(&mut kv, 1, head_dim, rope_dim, cos, sin, rope_pos, false);
    *out = kv;
}

// ── sparse attention with attentional sink (kernel.py:277-368) ──
//
// Online-softmax over the gathered topk positions; the denominator gets an
// extra exp(attn_sink - max) term (the "sink" logit), so a head can dump
// probability mass onto the sink instead of any real position.
// topk idx: < window → ring[idx], >= window → compressed[idx - window]. -1 = masked.
#[allow(clippy::too_many_arguments)]
pub fn sparse_attn_sink(
    q: &[f32],         // [n_heads, head_dim]
    ring: &[f32],      // [window, head_dim]
    compressed: &[f32],// [n_comp, head_dim]
    window: usize,
    topk: &[i32],
    sink: &[f32],
    scale: f32,
    out: &mut [f32],   // [n_heads, head_dim]
) {
    let n_heads = sink.len();
    let head_dim = q.len() / n_heads;
    for h in 0..n_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![f32::NEG_INFINITY; topk.len()];
        let mut m = f32::NEG_INFINITY;
        for (t, &idx) in topk.iter().enumerate() {
            if idx < 0 {
                continue;
            }
            let kv = if (idx as usize) < window {
                &ring[(idx as usize) * head_dim..(idx as usize + 1) * head_dim]
            } else {
                &compressed[(idx as usize - window) * head_dim..(idx as usize - window + 1) * head_dim]
            };
            let s = crate::model::dot(qh, kv) * scale;
            scores[t] = s;
            m = m.max(s);
        }
        let mut z = (sink[h] - m).exp();
        for i in 0..head_dim {
            out[h * head_dim + i] = 0.0;
        }
        for (t, &idx) in topk.iter().enumerate() {
            if scores[t] == f32::NEG_INFINITY {
                continue;
            }
            let p = (scores[t] - m).exp();
            z += p;
            let kv = if (idx as usize) < window {
                &ring[(idx as usize) * head_dim..(idx as usize + 1) * head_dim]
            } else {
                &compressed[(idx as usize - window) * head_dim..(idx as usize - window + 1) * head_dim]
            };
            for i in 0..head_dim {
                out[h * head_dim + i] += p * kv[i];
            }
        }
        for i in 0..head_dim {
            out[h * head_dim + i] /= z;
        }
    }
}

/// get_window_topk_idxs decode branch (model.py:260-271): circular window
/// positions for the query at absolute position `pos`.
pub fn window_topk(pos: usize, window: usize) -> Vec<i32> {
    if pos >= window - 1 {
        let start = pos % window;
        let mut v: Vec<i32> = (start as i32 + 1..window as i32).collect();
        v.extend(0..start as i32 + 1);
        v
    } else {
        let mut v: Vec<i32> = (0..pos as i32 + 1).collect();
        v.resize(window, -1);
        v
    }
}

/// get_compress_topk_idxs for ratio != 4 (dense compressor, model.py:274-282):
/// all compressed positions [0, (pos+1)//ratio) + window offset.
pub fn compress_topk_dense(pos: usize, ratio: usize, window: usize) -> Vec<i32> {
    let n = (pos + 1) / ratio;
    (0..n as i32).map(|i| i + window as i32).collect()
}

/// Hadamard rotation (rotate_activation, model.py:253-257): y = x · H_d × d^-0.5
/// where H_d is the Sylvester Walsh-Hadamard matrix (unnormalized ±1 entries).
/// d must be a power of two.
pub fn hadamard_rotate(x: &mut [f32]) {
    let n = x.len();
    debug_assert!(n.is_power_of_two());
    let mut h = 1usize;
    let scale = (n as f32).powf(-0.5);
    while h < n {
        for base in (0..n).step_by(h * 2) {
            for i in 0..h {
                let a = x[base + i];
                let b = x[base + i + h];
                x[base + i] = a + b;
                x[base + i + h] = a - b;
            }
        }
        h *= 2;
    }
    for v in x.iter_mut() {
        *v *= scale;
    }
}

/// Indexer forward for one token (model.py:408-439, ratio==4 layers only).
/// Returns topk compressed indices (with `window` offset applied).
#[allow(clippy::too_many_arguments)]
pub fn indexer_step(
    wq_b: &[f32],          // [index_n_heads*index_head_dim, q_lora_rank]
    weights_proj: &[f32],  // [index_n_heads, d]
    index_n_heads: usize,
    index_head_dim: usize,
    rope_dim: usize,
    qr: &[f32],
    x: &[f32],
    pos: usize,
    cos: &[f32],
    sin: &[f32],
    idx_compressed: &[f32],
    n_idx_compressed: usize,
    index_topk: usize,
    window: usize,
    softmax_scale: f32,
) -> Vec<i32> {
    let qd = index_n_heads * index_head_dim;
    let ql = qr.len();
    let mut q = vec![0f32; qd];
    crate::model::matvec(wq_b, qd, ql, qr, &mut q);
    apply_rotary(&mut q, index_n_heads, index_head_dim, rope_dim, cos, sin, pos, false);
    // Hadamard rotation is applied per head (last dim = index_head_dim),
    // scaled by index_head_dim^-0.5 — NOT on the flattened head-concat vector.
    for h in 0..index_n_heads {
        hadamard_rotate(&mut q[h * index_head_dim..(h + 1) * index_head_dim]);
    }
    fp4_roundtrip(&mut q);
    // weights per head for this token
    let mut w = vec![0f32; index_n_heads];
    crate::model::matvec(weights_proj, index_n_heads, x.len(), x, &mut w);
    let wscale = softmax_scale * (index_n_heads as f32).powf(-0.5);
    // index_score[t] = Σ_h relu(q_h · kv_t) · w_h  (model.py:426-427)
    let mut scores = vec![0f32; n_idx_compressed];
    for t in 0..n_idx_compressed {
        let kv = &idx_compressed[t * index_head_dim..(t + 1) * index_head_dim];
        let mut s = 0f32;
        for h in 0..index_n_heads {
            let qh = &q[h * index_head_dim..(h + 1) * index_head_dim];
            s += crate::model::dot(qh, kv).max(0.0) * w[h] * wscale;
        }
        scores[t] = s;
    }
    // causal mask: compressed position t covers tokens up to (t+1)*ratio-1;
    // current query is at pos → allowed t <= pos // ratio... reference masks
    // positions >= (pos+1)//ratio (model.py:431-432)
    let ratio = 4usize;
    let limit = pos / ratio; // last fully-compressed window index
    for (t, s) in scores.iter_mut().enumerate() {
        if t > limit {
            *s = f32::NEG_INFINITY;
        }
    }
    let k = index_topk.min(n_idx_compressed);
    let mut idx: Vec<i32> = (0..n_idx_compressed as i32).collect();
    idx.sort_by(|&a, &b| scores[b as usize].partial_cmp(&scores[a as usize]).unwrap());
    idx.truncate(k);
    // keep reference order? the reference takes topk()[1] (sorted by score desc)
    for i in idx.iter_mut() {
        *i += window as i32;
    }
    idx
}

// ── full attention layer forward (model.py:490-548) ──

#[allow(clippy::too_many_arguments)]
pub fn attention_step(
    cfg: &DsConfig,
    layer: usize,
    w: &DsAttentionW,
    st: &mut DsAttention,
    x: &[f32],
    out: &mut [f32],
) {
    let d = cfg.d;
    let hd = cfg.head_dim;
    let rd = cfg.rope_head_dim;
    let nh = cfg.n_heads;
    let win = cfg.window_size;
    let pos = st.pos;
    let ratio = cfg.compress_ratio(layer) as usize;

    // q: wq_a → q_norm → wq_b → per-head rsqrt norm → rope (model.py:502-505)
    let mut qr = vec![0f32; cfg.q_lora_rank];
    crate::model::matvec(&w.wq_a, cfg.q_lora_rank, d, x, &mut qr);
    let mut qn = vec![0f32; cfg.q_lora_rank];
    ds_rmsnorm(&qr, &w.q_norm_w, cfg.norm_eps as f32, &mut qn);
    let mut q = vec![0f32; nh * hd];
    crate::model::matvec(&w.wq_b, nh * hd, cfg.q_lora_rank, &qn, &mut q);
    for h in 0..nh {
        let qh = &mut q[h * hd..(h + 1) * hd];
        let ss = qh.iter().map(|v| v * v).sum::<f32>() / hd as f32; // mean of squares
        let inv = 1.0 / (ss + cfg.norm_eps as f32).sqrt();
        for v in qh.iter_mut() {
            *v *= inv;
        }
    }
    apply_rotary(&mut q, nh, hd, rd, &st.cos, &st.sin, pos, false);

    // kv: wkv → kv_norm → rope → fp8 QAT on non-rope dims (model.py:508-512)
    let mut kv = vec![0f32; hd];
    crate::model::matvec(&w.wkv, hd, d, x, &mut kv);
    let mut kvn = vec![0f32; hd];
    ds_rmsnorm(&kv, &w.kv_norm_w, cfg.norm_eps as f32, &mut kvn);
    apply_rotary(&mut kvn, 1, hd, rd, &st.cos, &st.sin, pos, false);
    fp8_roundtrip(&mut kvn[..hd - rd], 64);

    // ring write + window topk
    st.ring[(pos % win) * hd..(pos % win + 1) * hd].copy_from_slice(&kvn);
    let mut topk = window_topk(pos, win);

    // compressor (+ optionally indexer) for compressed layers
    if ratio > 0 {
        let coff = if ratio == 4 { 2 } else { 1 };
        let mut compressed_kv = Vec::new();
        let done = compressor_step(
            ratio, ratio == 4, hd, rd, &w.comp_wkv, &w.comp_wgate, &w.comp_ape, &w.comp_norm_w,
            x, pos, &mut st.comp_kv_state, &mut st.comp_score_state, cfg.norm_eps as f32,
            &st.cos, &st.sin, &mut compressed_kv,
        );
        if done.is_some() {
            let n = st.n_compressed;
            fp8_roundtrip(&mut compressed_kv[..hd - rd], 64);
            st.compressed[n * hd..(n + 1) * hd].copy_from_slice(&compressed_kv);
            st.n_compressed += 1;
        }
        let _ = coff;
        if ratio == 4 {
            // indexer (own compressor with Hadamard + fp4)
            let mut ikv = Vec::new();
            let idone = compressor_step(
                4, true, cfg.index_head_dim, rd, &w.idx_comp_wkv, &w.idx_comp_wgate, &w.idx_comp_ape,
                &w.idx_comp_norm_w, x, pos, &mut st.idx_kv_state, &mut st.idx_score_state,
                cfg.norm_eps as f32, &st.cos, &st.sin, &mut ikv,
            );
            if idone.is_some() {
                hadamard_rotate(&mut ikv);
                fp4_roundtrip(&mut ikv);
                let n = st.idx_n_compressed;
                st.idx_compressed[n * cfg.index_head_dim..(n + 1) * cfg.index_head_dim].copy_from_slice(&ikv);
                st.idx_n_compressed += 1;
            }
            let ctop = indexer_step(
                &w.idx_wq_b, &w.idx_weights_proj, cfg.index_n_heads, cfg.index_head_dim, rd,
                &qn, x, pos, &st.cos, &st.sin, &st.idx_compressed, st.idx_n_compressed,
                cfg.index_topk, win, 1.0 / (cfg.index_head_dim as f32).sqrt(),
            );
            topk.extend(ctop);
        } else {
            topk.extend(compress_topk_dense(pos, ratio, win));
        }
    }

    // sparse attention with sink (kernel.py:277-368)
    let scale = 1.0 / (hd as f32).sqrt();
    sparse_attn_sink(&q, &st.ring, &st.compressed, win, &topk, &w.attn_sink, scale, out);

    // derotation (model.py:539)
    apply_rotary(out, nh, hd, rd, &st.cos, &st.sin, pos, true);
    st.pos += 1;
}

pub struct DsAttentionW {
    pub wq_a: Vec<f32>,      // [q_lora, d]
    pub q_norm_w: Vec<f32>,  // [q_lora]
    pub wq_b: Vec<f32>,      // [n_heads*head_dim, q_lora]
    pub wkv: Vec<f32>,       // [head_dim, d]
    pub kv_norm_w: Vec<f32>, // [head_dim]
    pub wo_a: Vec<f32>,      // [o_groups*o_lora, n_heads*head_dim/o_groups]
    pub wo_b: Vec<f32>,      // [d, o_groups*o_lora]
    pub attn_sink: Vec<f32>, // [n_heads]
    pub comp_wkv: Vec<f32>, pub comp_wgate: Vec<f32>, pub comp_ape: Vec<f32>, pub comp_norm_w: Vec<f32>,
    pub idx_wq_b: Vec<f32>, pub idx_weights_proj: Vec<f32>, pub idx_comp_wkv: Vec<f32>, pub idx_comp_wgate: Vec<f32>, pub idx_comp_ape: Vec<f32>, pub idx_comp_norm_w: Vec<f32>,
}

/// Grouped O projection (model.py:542-547): per group g, out_g = wo_a[g] @ o_g,
/// then wo_b over the flattened result.
pub fn grouped_o_proj(cfg: &DsConfig, wo_a: &[f32], wo_b: &[f32], o: &[f32], out: &mut [f32]) {
    let gdim = cfg.n_heads * cfg.head_dim / cfg.o_groups;
    let r = cfg.o_lora_rank;
    let mut lat = vec![0f32; cfg.o_groups * r];
    for g in 0..cfg.o_groups {
        crate::model::matvec(
            &wo_a[g * r * gdim..(g + 1) * r * gdim],
            r,
            gdim,
            &o[g * gdim..(g + 1) * gdim],
            &mut lat[g * r..(g + 1) * r],
        );
    }
    crate::model::matvec(wo_b, cfg.d, cfg.o_groups * r, &lat, out);
}

// ════════════════════════════════════════════════════════════════════════════
// DsModel: full microdeepseek forward (model.py Transformer, lines 877-926)
// embed → expand ×hc_mult → n_layers × Block(hc_pre/attn/hc_post, hc_pre/MoE/
// hc_post) → hc_head → RMSNorm → lm_head. f32 throughout, experts stay fp4
// packed (MXFP4 layout) and are dequantized on the fly by matvec_packed.
// ════════════════════════════════════════════════════════════════════════════

use crate::quant::weights::{BinFile, Entry};

fn as_f32(bytes: &[u8]) -> &[f32] {
    let (pre, mid, post) = unsafe { bytes.align_to::<f32>() };
    assert!(pre.is_empty() && post.is_empty(), "unexpected f32 alignment");
    mid
}

/// Per-layer weight bundle (small tensors copied out of the bin at load,
/// routed experts kept as packed-blob offsets).
pub struct DsLayerW {
    pub attn_norm_w: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub hc_attn_fn: Vec<f32>,   // [mix_hc, hc*d]
    pub hc_attn_base: Vec<f32>, // [mix_hc]
    pub hc_attn_scale: Vec<f32>, // [3]
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub attn: DsAttentionW,
    pub gate_w: Vec<f32>,               // [n_routed, d]
    pub gate_bias: Option<Vec<f32>>,    // score-routing layers (>= n_hash_layers)
    pub tid2eid: Option<Vec<i32>>,      // hash layers (< n_hash_layers) [vocab, topk]
    pub shared: [Vec<f32>; 3],          // w1 [inter,d], w2 [d,inter], w3 [inter,d]
    pub experts: Vec<[u64; 3]>,         // packed blob offsets [w1, w2, w3] per expert
}

pub struct DsModel {
    pub cfg: DsConfig,
    bin: BinFile,
    embed: Entry, // f32 [vocab, d]
    head: Entry,  // f32 [vocab, d]
    norm_w: Vec<f32>,
    hc_head_fn: Vec<f32>,   // [hc, hc*d]
    hc_head_base: Vec<f32>, // [hc]
    hc_head_scale: f32,
    layers: Vec<DsLayerW>,
    states: Vec<DsAttention>,
}

impl DsModel {
    fn f32v(bin: &BinFile, name: &str) -> Vec<f32> {
        bin.f32_vec(name)
    }

    pub fn load(path: &str) -> Self {
        let bin = BinFile::open(path);
        let cfg = bin
            .config
            .ds
            .clone()
            .unwrap_or_else(|| panic!("{}: not a deepseek_v4 model (no ds config block)", path));
        let get = |name: &str| -> Entry {
            bin.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name)).clone()
        };
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = format!("layers.{}.", l);
            let ratio = cfg.compress_ratio(l);
            let attn = DsAttentionW {
                wq_a: Self::f32v(&bin, &format!("{}attn.wq_a.weight", p)),
                q_norm_w: Self::f32v(&bin, &format!("{}attn.q_norm.weight", p)),
                wq_b: Self::f32v(&bin, &format!("{}attn.wq_b.weight", p)),
                wkv: Self::f32v(&bin, &format!("{}attn.wkv.weight", p)),
                kv_norm_w: Self::f32v(&bin, &format!("{}attn.kv_norm.weight", p)),
                wo_a: Self::f32v(&bin, &format!("{}attn.wo_a.weight", p)),
                wo_b: Self::f32v(&bin, &format!("{}attn.wo_b.weight", p)),
                attn_sink: Self::f32v(&bin, &format!("{}attn.attn_sink", p)),
                comp_wkv: if ratio > 0 { Self::f32v(&bin, &format!("{}attn.compressor.wkv.weight", p)) } else { Vec::new() },
                comp_wgate: if ratio > 0 { Self::f32v(&bin, &format!("{}attn.compressor.wgate.weight", p)) } else { Vec::new() },
                comp_ape: if ratio > 0 { Self::f32v(&bin, &format!("{}attn.compressor.ape", p)) } else { Vec::new() },
                comp_norm_w: if ratio > 0 { Self::f32v(&bin, &format!("{}attn.compressor.norm.weight", p)) } else { Vec::new() },
                idx_wq_b: if ratio == 4 { Self::f32v(&bin, &format!("{}attn.indexer.wq_b.weight", p)) } else { Vec::new() },
                idx_weights_proj: if ratio == 4 { Self::f32v(&bin, &format!("{}attn.indexer.weights_proj.weight", p)) } else { Vec::new() },
                idx_comp_wkv: if ratio == 4 { Self::f32v(&bin, &format!("{}attn.indexer.compressor.wkv.weight", p)) } else { Vec::new() },
                idx_comp_wgate: if ratio == 4 { Self::f32v(&bin, &format!("{}attn.indexer.compressor.wgate.weight", p)) } else { Vec::new() },
                idx_comp_ape: if ratio == 4 { Self::f32v(&bin, &format!("{}attn.indexer.compressor.ape", p)) } else { Vec::new() },
                idx_comp_norm_w: if ratio == 4 { Self::f32v(&bin, &format!("{}attn.indexer.compressor.norm.weight", p)) } else { Vec::new() },
            };
            let hash = l < cfg.n_hash_layers;
            let experts: Vec<[u64; 3]> = (0..cfg.n_routed_experts)
                .map(|e| {
                    ["w1", "w2", "w3"].map(|wn| get(&format!("{}ffn.experts.{}.{}", p, e, wn)).offset)
                })
                .collect();
            layers.push(DsLayerW {
                attn_norm_w: Self::f32v(&bin, &format!("{}attn_norm.weight", p)),
                ffn_norm_w: Self::f32v(&bin, &format!("{}ffn_norm.weight", p)),
                hc_attn_fn: Self::f32v(&bin, &format!("{}hc_attn_fn", p)),
                hc_attn_base: Self::f32v(&bin, &format!("{}hc_attn_base", p)),
                hc_attn_scale: Self::f32v(&bin, &format!("{}hc_attn_scale", p)),
                hc_ffn_fn: Self::f32v(&bin, &format!("{}hc_ffn_fn", p)),
                hc_ffn_base: Self::f32v(&bin, &format!("{}hc_ffn_base", p)),
                hc_ffn_scale: Self::f32v(&bin, &format!("{}hc_ffn_scale", p)),
                attn,
                gate_w: Self::f32v(&bin, &format!("{}ffn.gate.weight", p)),
                gate_bias: if hash { None } else { Some(Self::f32v(&bin, &format!("{}ffn.gate.bias", p))) },
                tid2eid: if hash { Some(bin.i32_vec(&format!("{}ffn.gate.tid2eid", p))) } else { None },
                shared: [
                    Self::f32v(&bin, &format!("{}ffn.shared_experts.w1.weight", p)),
                    Self::f32v(&bin, &format!("{}ffn.shared_experts.w2.weight", p)),
                    Self::f32v(&bin, &format!("{}ffn.shared_experts.w3.weight", p)),
                ],
                experts,
            });
        }
        let states = (0..cfg.n_layers).map(|l| DsAttention::new(&cfg, l)).collect();
        let hc_head_scale = Self::f32v(&bin, "hc_head_scale");
        DsModel {
            cfg,
            embed: get("embed.weight"),
            head: get("head.weight"),
            norm_w: Self::f32v(&bin, "norm.weight"),
            hc_head_fn: Self::f32v(&bin, "hc_head_fn"),
            hc_head_base: Self::f32v(&bin, "hc_head_base"),
            hc_head_scale: hc_head_scale[0],
            layers,
            states,
            bin,
        }
    }

    pub fn reset(&mut self) {
        self.states = (0..self.cfg.n_layers).map(|l| DsAttention::new(&self.cfg, l)).collect();
    }

    /// MoE forward for one token (model.py:614-649): gate routing, fp4 packed
    /// routed experts (routing weight applied BEFORE w2, as in Expert.forward),
    /// plus the single unweighted shared expert.
    #[allow(dead_code)]
    fn moe(&self, w: &DsLayerW, x: &[f32], token: u32) -> Vec<f32> {
        ds_moe(&self.cfg, &self.bin.data[..], w, x, token, 0, 0)
    }

    /// One decode step: token id → logits [vocab] (model.py Transformer.forward).
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        let Self { cfg, bin, embed, head, norm_w, hc_head_fn, hc_head_base, hc_head_scale, layers, states } = self;
        let cfg = &*cfg;
        let d = cfg.d;
        let hc = cfg.hc_mult;
        let data = &bin.data[..];
        // embed + expand to hc_mult identical copies (model.py:914-916)
        let erow = &as_f32(&data[embed.offset as usize..(embed.offset + embed.size) as usize])
            [token as usize * d..(token as usize + 1) * d];
        let mut state = vec![0f32; hc * d];
        for j in 0..hc {
            state[j * d..(j + 1) * d].copy_from_slice(erow);
        }

        for l in 0..cfg.n_layers {
            let w = &layers[l];
            // ── attention sublayer (model.py:696-700) ──
            let (y, post, comb) = hc_pre(&state, &state, &w.hc_attn_fn, &w.hc_attn_scale, &w.hc_attn_base, hc, cfg.norm_eps as f32, cfg.hc_sinkhorn_iters, cfg.hc_eps as f32);
            let mut yn = vec![0f32; d];
            ds_rmsnorm(&y, &w.attn_norm_w, cfg.norm_eps as f32, &mut yn);
            let mut attn_pre = vec![0f32; cfg.n_heads * cfg.head_dim];
            attention_step(cfg, l, &w.attn, &mut states[l], &yn, &mut attn_pre);
            let mut attn_out = vec![0f32; d];
            grouped_o_proj(cfg, &w.attn.wo_a, &w.attn.wo_b, &attn_pre, &mut attn_out);
            state = hc_post(&attn_out, &state, &post, &comb, hc);
            // ── ffn sublayer (model.py:702-706) ──
            let (y, post, comb) = hc_pre(&state, &state, &w.hc_ffn_fn, &w.hc_ffn_scale, &w.hc_ffn_base, hc, cfg.norm_eps as f32, cfg.hc_sinkhorn_iters, cfg.hc_eps as f32);
            ds_rmsnorm(&y, &w.ffn_norm_w, cfg.norm_eps as f32, &mut yn);
            let moe_out = ds_moe(cfg, data, w, &yn, token, l, pos);
            state = hc_post(&moe_out, &state, &post, &comb, hc);
            ds_parity_rec(pos, l, &state);
        }

        // head: hc_head → final rmsnorm → lm_head (model.py:922-923)
        let h = hc_head(&state, &state, hc_head_fn, *hc_head_scale, hc_head_base, hc, cfg.norm_eps as f32, cfg.hc_eps as f32);
        let mut xn = vec![0f32; d];
        ds_rmsnorm(&h, norm_w, cfg.norm_eps as f32, &mut xn);
        let mut logits = vec![0f32; cfg.vocab];
        let hrow = as_f32(&data[head.offset as usize..(head.offset + head.size) as usize]);
        crate::model::matvec(hrow, cfg.vocab, d, &xn, &mut logits);
        logits
    }
}

// ── DS parity dump (thread-local, inactive during normal inference) ──

// ── DS parity dump (thread-local, inactive during normal inference) ──
// Records the full HC state [hc*d] after selected blocks and the router
// selections of selected layers, for the end-to-end comparison against the
// torch replica (ref/make_ds_parity.py).

pub const DS_DUMP_LAYERS: [usize; 6] = [0, 1, 2, 3, 21, 42];
pub const DS_ROUTER_LAYERS: [usize; 3] = [1, 3, 42];

#[derive(Default)]
pub struct DsParityDump {
    pub hiddens: std::collections::HashMap<(usize, usize), Vec<f32>>, // (pos, layer)
    pub router: std::collections::HashMap<(usize, usize), Vec<u32>>,  // (pos, layer) sorted experts
}

thread_local! {
    pub static DS_PARITY: std::cell::RefCell<Option<DsParityDump>> = std::cell::RefCell::new(None);
}

fn ds_parity_rec(pos: usize, layer: usize, state: &[f32]) {
    if !DS_DUMP_LAYERS.contains(&layer) {
        return;
    }
    DS_PARITY.with(|p| {
        if let Some(d) = p.borrow_mut().as_mut() {
            d.hiddens.insert((pos, layer), state.to_vec());
        }
    });
}

/// MoE forward for one token (model.py:614-649): gate routing, fp4 packed
/// routed experts (routing weight applied BEFORE w2, as in Expert.forward),
/// plus the single unweighted shared expert. The top-k experts run as
/// independent pool jobs (like the K3 engine).
fn ds_moe(cfg: &DsConfig, data: &[u8], w: &DsLayerW, x: &[f32], token: u32, layer: usize, pos: usize) -> Vec<f32> {
    let d = cfg.d;
    let inter = cfg.moe_inter_dim;
    let (sel, _) = gate_forward(
        x,
        &w.gate_w,
        w.gate_bias.as_deref(),
        w.tid2eid.as_deref(),
        token,
        cfg.n_routed_experts,
        cfg.n_activated_experts,
        cfg.route_scale as f32,
    );
    if DS_ROUTER_LAYERS.contains(&layer) {
        let mut ids: Vec<u32> = sel.iter().map(|s| s.0).collect();
        ids.sort();
        DS_PARITY.with(|p| {
            if let Some(d) = p.borrow_mut().as_mut() {
                d.router.insert((pos, layer), ids);
            }
        });
    }
    let packed = inter * d / 2;
    let blob = packed + inter * d / 32;
    let limit = cfg.swiglu_limit as f32;
    let mut outs = vec![0f32; sel.len() * d];
    {
        let dp = crate::model::pool::SPtrU8(data.as_ptr());
        let dlen = data.len();
        let xp = crate::model::pool::SPtr(x.as_ptr());
        let op = crate::model::pool::MPtr(outs.as_mut_ptr());
        let mut jobs: Vec<crate::model::pool::Job> = Vec::with_capacity(sel.len());
        for (ei, &(eid, wgt)) in sel.iter().enumerate() {
            let offs = w.experts[eid as usize];
            jobs.push(Box::new(move || {
                let (dp, xp, op) = (dp, xp, op);
                unsafe {
                    let data = std::slice::from_raw_parts(dp.0, dlen);
                    let x = std::slice::from_raw_parts(xp.0, d);
                    let blob_at = |i: usize| &data[offs[i] as usize..offs[i] as usize + blob];
                    let mut gate = vec![0f32; inter];
                    let mut up = vec![0f32; inter];
                    crate::quant::mxfp4::matvec_packed(&blob_at(0)[..packed], &blob_at(0)[packed..], inter, d, x, &mut gate, 1);
                    crate::quant::mxfp4::matvec_packed(&blob_at(2)[..packed], &blob_at(2)[packed..], inter, d, x, &mut up, 1);
                    let mut act = vec![0f32; inter];
                    for j in 0..inter {
                        let g = gate[j].min(limit);
                        let u = up[j].clamp(-limit, limit);
                        act[j] = wgt * (g / (1.0 + (-g).exp())) * u; // wgt * silu(g) * u
                    }
                    let o = std::slice::from_raw_parts_mut(op.0.add(ei * d), d);
                    crate::quant::mxfp4::matvec_packed(&blob_at(1)[..packed], &blob_at(1)[packed..], d, inter, &act, o, 1);
                }
            }));
        }
        crate::model::pool::pool().run(jobs);
    }
    let mut out = vec![0f32; d];
    for ei in 0..sel.len() {
        for j in 0..d {
            out[j] += outs[ei * d + j];
        }
    }
    // shared expert (unweighted, f32)
    let mut sout = vec![0f32; d];
    expert_forward(&w.shared[0], &w.shared[1], &w.shared[2], x, inter, limit, &mut sout);
    for j in 0..d {
        out[j] += sout[j];
    }
    out
}

/// Greedy turn on a DsModel (shares model::run_turn_core with the K3 engine).
/// Always greedy: the sampling flags are a no-op on the DeepSeek path.
pub fn ds_run_turn(ids: &[u32], max_new: usize, tok: &crate::tokenizer::AnyTokenizer, model: &mut DsModel, debug: bool, debug_routing: bool, stop_id: u32) -> String {
    model.reset();
    let mut pos = 0usize;
    crate::model::run_turn_core(
        ids,
        max_new,
        tok,
        &mut |id| {
            let l = model.forward(id, pos);
            pos += 1;
            l
        },
        debug,
        debug_routing,
        stop_id,
        &mut crate::model::Sampler::greedy(),
    )
}
