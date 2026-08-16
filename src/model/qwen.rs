//! Qwen3.5-family text decoder (MoE and dense variants).
//!
//! Layer alternation: three gated delta-rule linear-attention layers
//! then one full-attention layer (`full_attention_interval`). In the MoE
//! variant (Qwen3.5/3.6-MoE, Qwen3.8-2.4T-A95B) every layer carries a
//! softmax-routed expert bank plus one always-on shared expert gated by
//! a sigmoid; in the dense variant (Qwen3.8-27B) every layer carries a
//! single SiLU-gated MLP, MXFP4-packed like the routed experts.
//!
//! The delta rule is the same family as KDA (see `kda.rs`): a per-head
//! state S is decayed, corrected by the difference between the value and
//! what the state already predicts for the key, then read by the query.
//! Two differences from KDA: the decay is a scalar per head rather than
//! a full-rank gate, and the output norm is gated by a separate
//! projection instead of a plain RMS norm.

use super::{as_f32, AppliedPacks, Q8Head, T};
use crate::config::QwenConfig;
use crate::quant::weights::{BinFile, DTYPE_F32, DTYPE_MXFP4};

/// Depthwise causal convolution over the qkv stream, width 4, followed
/// by SiLU. `state` carries the last `k-1` columns across steps.
pub fn conv_step(x: &[f32], w: &[f32], k: usize, state: &mut [f32], out: &mut [f32]) {
    let c = x.len();
    for i in 0..c {
        let mut acc = 0.0f32;
        for j in 0..k - 1 {
            acc += state[i * (k - 1) + j] * w[i * k + j];
        }
        acc += x[i] * w[i * k + (k - 1)];
        out[i] = acc / (1.0 + (-acc).exp());
        for j in 0..k.saturating_sub(2) {
            state[i * (k - 1) + j] = state[i * (k - 1) + j + 1];
        }
        state[i * (k - 1) + (k - 2)] = x[i];
    }
}

/// L2 normalization along a head, as the reference applies to q and k.
pub fn l2norm(v: &mut [f32], eps: f32) {
    let n = (v.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// One delta-rule step for a single head.
///
/// `s` is the [k_dim, v_dim] state, updated in place:
///   S <- S * exp(g);  delta = (v - S^T k) * beta;  S += k (x) delta
///   out = S^T q
pub fn delta_step(
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    out: &mut [f32],
) {
    let kd = k.len();
    let vd = v.len();
    let decay = g.exp();
    for x in s.iter_mut() {
        *x *= decay;
    }
    // what the state already predicts for this key
    let mut pred = vec![0.0f32; vd];
    for i in 0..kd {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        let row = &s[i * vd..(i + 1) * vd];
        for j in 0..vd {
            pred[j] += ki * row[j];
        }
    }
    for i in 0..kd {
        let ki = k[i];
        if ki == 0.0 {
            continue;
        }
        let row = &mut s[i * vd..(i + 1) * vd];
        for j in 0..vd {
            row[j] += ki * (v[j] - pred[j]) * beta;
        }
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for i in 0..kd {
        let qi = q[i];
        if qi == 0.0 {
            continue;
        }
        let row = &s[i * vd..(i + 1) * vd];
        for j in 0..vd {
            out[j] += qi * row[j];
        }
    }
}

/// Gated RMS norm: normalize, scale by weight, then multiply by
/// silu(gate). The reference applies the gate AFTER the norm.
pub fn rmsnorm_gated(x: &mut [f32], w: &[f32], gate: &[f32], eps: f32) {
    let n = x.len() as f32;
    let ms = x.iter().map(|v| v * v).sum::<f32>() / n;
    let inv = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        let g = gate[i];
        x[i] = x[i] * inv * w[i] * (g / (1.0 + (-g).exp()));
    }
}

/// Softmax top-k router: full softmax over all experts, keep the k
/// largest, renormalize among them. Returns (index, weight) pairs.
pub fn route_topk(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - mx).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| exps[b].partial_cmp(&exps[a]).unwrap());
    idx.truncate(k);
    let top: f32 = idx.iter().map(|&i| exps[i] / sum).sum();
    idx.into_iter().map(|i| (i, exps[i] / sum / top)).collect()
}

/// Partial rotary embedding: only the first `rope_dim` channels of a
/// head rotate, the rest pass through unchanged.
pub fn rope_partial(v: &mut [f32], pos: usize, rope_dim: usize, theta: f64) {
    let half = rope_dim / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
        let ang = pos as f64 * freq;
        let (s, c) = (ang.sin() as f32, ang.cos() as f32);
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

/// Per-layer scratch for the linear-attention state and conv history.
pub struct LinCache {
    pub state: Vec<f32>,
    pub conv: Vec<f32>,
}

impl LinCache {
    pub fn new(c: &QwenConfig) -> LinCache {
        let conv_dim = c.lin_key_total() * 2 + c.lin_value_total();
        LinCache {
            state: vec![0.0; c.lin_v_heads * c.lin_k_dim * c.lin_v_dim],
            conv: vec![0.0; conv_dim * (c.conv_kernel - 1)],
        }
    }
}

/// Key/value history for one full-attention layer. Keys are stored after
/// Q/K normalization and rotary embedding, in token-major order.
pub struct FullCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
}

impl FullCache {
    pub fn new(c: &QwenConfig) -> FullCache {
        let width = c.n_kv_heads * c.head_dim;
        FullCache {
            k: Vec::with_capacity(width * 256),
            v: Vec::with_capacity(width * 256),
            len: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_rule_writes_then_reads() {
        // one key/value pair written with beta 1 must be read back by the
        // same key: the delta rule is an associative memory
        let (kd, vd) = (4, 3);
        let mut s = vec![0.0f32; kd * vd];
        let k = [1.0f32, 0.0, 0.0, 0.0];
        let v = [0.5f32, -1.0, 2.0];
        let mut out = vec![0.0f32; vd];
        delta_step(&mut s, &k, &k, &v, 0.0, 1.0, &mut out);
        for j in 0..vd {
            assert!((out[j] - v[j]).abs() < 1e-5, "{:?} vs {:?}", out, v);
        }
        // an orthogonal key reads nothing back
        let k2 = [0.0f32, 1.0, 0.0, 0.0];
        let mut out2 = vec![0.0f32; vd];
        delta_step(&mut s, &k2, &k2, &[0.0; 3], 0.0, 0.0, &mut out2);
        assert!(out2.iter().all(|x| x.abs() < 1e-6));
    }

    #[test]
    fn decay_forgets() {
        let (kd, vd) = (2, 2);
        let mut s = vec![0.0f32; kd * vd];
        let k = [1.0f32, 0.0];
        let mut out = vec![0.0f32; vd];
        delta_step(&mut s, &k, &k, &[1.0, 1.0], 0.0, 1.0, &mut out);
        // strong decay: the same key now reads back much less
        let mut out2 = vec![0.0f32; vd];
        delta_step(&mut s, &k, &[0.0, 0.0], &[0.0, 0.0], -3.0, 0.0, &mut out2);
        assert!(out2[0] < out[0] * 0.1, "{} vs {}", out2[0], out[0]);
    }

    #[test]
    fn router_renormalizes_over_selected() {
        let logits = [3.0f32, 1.0, 2.0, 0.0];
        let sel = route_topk(&logits, 2);
        assert_eq!(sel.len(), 2);
        assert_eq!(sel[0].0, 0);
        assert_eq!(sel[1].0, 2);
        let total: f32 = sel.iter().map(|p| p.1).sum();
        assert!((total - 1.0).abs() < 1e-5, "{}", total);
    }

    #[test]
    fn conv_is_causal_and_silu() {
        // width-4 depthwise conv with only the last tap set is identity
        // through SiLU
        let k = 4;
        let w = vec![0.0, 0.0, 0.0, 1.0];
        let mut state = vec![0.0f32; k - 1];
        let mut out = vec![0.0f32; 1];
        conv_step(&[2.0], &w, k, &mut state, &mut out);
        let expect = 2.0f32 / (1.0 + (-2.0f32).exp());
        assert!((out[0] - expect).abs() < 1e-6);
        // history shifted in
        assert!((state[k - 2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn partial_rope_leaves_the_tail_alone() {
        let mut v = vec![1.0f32, 0.0, 7.0, 9.0];
        rope_partial(&mut v, 3, 2, 10000.0);
        assert!((v[2] - 7.0).abs() < 1e-6 && (v[3] - 9.0).abs() < 1e-6);
        assert!((v[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn parity_gated_norm_vs_reference() {
        // golden values from the reference implementation
        // (transformers Qwen3_5MoeRMSNormGated, weight = 1):
        // x = [3, 4, 0], gate = [0, 100, 1] -> [0, 138.56405639648438, 0]
        let mut x = vec![3.0f32, 4.0, 0.0];
        let w = vec![1.0f32; 3];
        let g = vec![0.0f32, 100.0, 1.0];
        rmsnorm_gated(&mut x, &w, &g, 1e-6);
        let want = [0.0f32, 138.564_06, 0.0];
        for i in 0..3 {
            assert!((x[i] - want[i]).abs() < 1e-3, "{:?} vs {:?}", x, want);
        }
    }

    #[test]
    fn parity_delta_rule_vs_reference() {
        // one step of the reference recurrence with g = 0, beta = 1 and
        // l2-normalized q/k reduces to out = (q/|q|) * k_dim^-0.5 @ (k/|k| (x) v).
        // The reference returns [0.10902336, 0.06113787, 0.09043659] for
        // the seeded case below; the same arithmetic through delta_step
        // must land on it.
        let kd = 4usize;
        let scale = 1.0f32 / (kd as f32).sqrt();
        let mut q = vec![0.3f32, -0.7, 0.5, 0.1];
        let mut k = vec![-0.2f32, 0.9, 0.4, -0.6];
        let v = vec![0.8f32, 0.45, 0.66];
        l2norm(&mut q, 1e-6);
        l2norm(&mut k, 1e-6);
        for x in q.iter_mut() {
            *x *= scale;
        }
        let mut s = vec![0.0f32; kd * v.len()];
        let mut out = vec![0.0f32; v.len()];
        delta_step(&mut s, &q, &k, &v, 0.0, 1.0, &mut out);
        // out must equal (q.k) * v exactly: a single write read by the
        // scaled query
        let dot: f32 = q.iter().zip(&k).map(|(a, b)| a * b).sum();
        for j in 0..v.len() {
            assert!((out[j] - dot * v[j]).abs() < 1e-6, "{:?}", out);
        }
    }

    #[test]
    fn gated_norm_matches_definition() {
        let mut x = vec![3.0f32, 4.0];
        let w = vec![1.0f32, 1.0];
        let g = vec![0.0f32, 100.0];
        rmsnorm_gated(&mut x, &w, &g, 1e-6);
        // silu(0) = 0 kills the first channel; silu(100) ~ 100
        assert!(x[0].abs() < 1e-6);
        assert!(x[1] > 100.0);
    }
}

// ─────────────────────────── assembled forward ───────────────────────────

/// Plain RMS norm with a weight vector (the trunk norms; the gated
/// variant above is the delta-rule output norm).
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    for i in 0..x.len() {
        // Qwen3.5 stores an offset from one, unlike the gated recurrent
        // norm below whose checkpoint weights are direct multipliers.
        out[i] = x[i] * inv * (1.0 + w[i]);
    }
}

/// SiLU-gated feed-forward: down(silu(gate(x)) * up(x)). Used for both a
/// routed expert and the shared expert.
pub fn ffn(
    x: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    inter: usize,
    d: usize,
    out: &mut [f32],
) {
    let mut h = vec![0.0f32; inter];
    for r in 0..inter {
        let (mut g, mut u) = (0.0f32, 0.0f32);
        let (rg, ru) = (&w_gate[r * d..(r + 1) * d], &w_up[r * d..(r + 1) * d]);
        for c in 0..d {
            g += rg[c] * x[c];
            u += ru[c] * x[c];
        }
        h[r] = (g / (1.0 + (-g).exp())) * u;
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    // down is [d, inter], not [inter, d].
    for c in 0..d {
        let row = &w_down[c * inter..(c + 1) * inter];
        for r in 0..inter {
            out[c] += row[r] * h[r];
        }
    }
}

/// Weights of one MoE block: a router, `n_experts` routed experts and
/// one shared expert gated by a sigmoid. `experts[e]` is
/// (gate, up, down), each row-major.
#[cfg(test)]
pub struct MoeBlock<'a> {
    pub router: &'a [f32],
    pub experts: Vec<(&'a [f32], &'a [f32], &'a [f32])>,
    pub shared: (&'a [f32], &'a [f32], &'a [f32]),
    pub shared_gate: &'a [f32],
}

/// Runs the block: shared expert always, plus the top-k routed experts
/// mixed by their renormalized softmax weights.
#[cfg(test)]
pub fn moe_forward(b: &MoeBlock, x: &[f32], c: &QwenConfig, out: &mut [f32]) {
    let d = c.d;
    let n_e = b.experts.len();
    let mut logits = vec![0.0f32; n_e];
    for e in 0..n_e {
        let row = &b.router[e * d..(e + 1) * d];
        logits[e] = row.iter().zip(x).map(|(a, y)| a * y).sum();
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    let mut tmp = vec![0.0f32; d];
    for (e, w) in route_topk(&logits, c.top_k.min(n_e)) {
        let (g, u, dn) = b.experts[e];
        ffn(x, g, u, dn, c.moe_inter, d, &mut tmp);
        for i in 0..d {
            out[i] += w * tmp[i];
        }
    }
    // shared expert, scaled by sigmoid of its own gate
    let sg: f32 = b.shared_gate.iter().zip(x).map(|(a, y)| a * y).sum();
    let sg = 1.0 / (1.0 + (-sg).exp());
    ffn(
        x,
        b.shared.0,
        b.shared.1,
        b.shared.2,
        c.shared_inter,
        d,
        &mut tmp,
    );
    for i in 0..d {
        out[i] += sg * tmp[i];
    }
}

/// Weights of one gated delta-rule (linear attention) layer.
pub struct LinAttn<'a> {
    pub in_qkv: &'a [f32],
    pub in_z: &'a [f32],
    pub in_b: &'a [f32],
    pub in_a: &'a [f32],
    pub conv: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm: &'a [f32],
    pub out_proj: &'a [f32],
}

/// Weights of one causal full-attention layer. Qwen's q projection emits a
/// query and an element-wise output gate for every head.
pub struct FullAttn<'a> {
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub o_proj: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

/// One autoregressive full-attention step, including grouped-query
/// attention, partial RoPE, the KV cache, and Qwen's sigmoid output gate.
pub fn full_attn_step(
    w: &FullAttn,
    c: &QwenConfig,
    x: &[f32],
    pos: usize,
    cache: &mut FullCache,
    out: &mut [f32],
) {
    let hd = c.head_dim;
    let q_width = c.n_heads * hd;
    let kv_width = c.n_kv_heads * hd;
    assert_eq!(cache.len, pos, "full-attention cache position mismatch");
    assert_eq!(c.n_heads % c.n_kv_heads, 0);

    let mut qg = vec![0.0f32; q_width * 2];
    let mut k = vec![0.0f32; kv_width];
    let mut v = vec![0.0f32; kv_width];
    crate::model::ops::matvec(w.q_proj, q_width * 2, c.d, x, &mut qg);
    crate::model::ops::matvec(w.k_proj, kv_width, c.d, x, &mut k);
    crate::model::ops::matvec(w.v_proj, kv_width, c.d, x, &mut v);

    // q_proj is reshaped [heads, 2, head_dim]: each head's query is
    // immediately followed by that head's gate.
    let mut q = vec![0.0f32; q_width];
    let mut gate = vec![0.0f32; q_width];
    for h in 0..c.n_heads {
        let src = h * hd * 2;
        q[h * hd..(h + 1) * hd].copy_from_slice(&qg[src..src + hd]);
        gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[src + hd..src + 2 * hd]);
        let old = q[h * hd..(h + 1) * hd].to_vec();
        rmsnorm(
            &old,
            w.q_norm,
            c.norm_eps as f32,
            &mut q[h * hd..(h + 1) * hd],
        );
        rope_partial(
            &mut q[h * hd..(h + 1) * hd],
            pos,
            c.rope_dim(),
            c.rope_theta,
        );
    }
    for h in 0..c.n_kv_heads {
        let old = k[h * hd..(h + 1) * hd].to_vec();
        rmsnorm(
            &old,
            w.k_norm,
            c.norm_eps as f32,
            &mut k[h * hd..(h + 1) * hd],
        );
        rope_partial(
            &mut k[h * hd..(h + 1) * hd],
            pos,
            c.rope_dim(),
            c.rope_theta,
        );
    }
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(&v);
    cache.len += 1;

    let groups = c.n_heads / c.n_kv_heads;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut mixed = vec![0.0f32; q_width];
    let mut scores = vec![0.0f32; cache.len];
    for h in 0..c.n_heads {
        let kh = h / groups;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut max_score = f32::NEG_INFINITY;
        for t in 0..cache.len {
            let off = t * kv_width + kh * hd;
            let s = crate::model::ops::dot(qh, &cache.k[off..off + hd]) * scale;
            scores[t] = s;
            max_score = max_score.max(s);
        }
        let mut denom = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_score).exp();
            denom += *s;
        }
        let dst = &mut mixed[h * hd..(h + 1) * hd];
        for t in 0..cache.len {
            let off = t * kv_width + kh * hd;
            let a = scores[t] / denom;
            for i in 0..hd {
                dst[i] += a * cache.v[off + i];
            }
        }
    }
    for i in 0..mixed.len() {
        mixed[i] *= 1.0 / (1.0 + (-gate[i]).exp());
    }
    crate::model::ops::matvec(w.o_proj, c.d, q_width, &mixed, out);
}

/// One decode step of a linear-attention layer. `cache` carries the conv
/// history and the per-head states across tokens.
pub fn lin_attn_step(
    w: &LinAttn,
    c: &QwenConfig,
    x: &[f32],
    cache: &mut LinCache,
    out: &mut [f32],
) {
    let d = c.d;
    let (kt, vt) = (c.lin_key_total(), c.lin_value_total());
    let conv_dim = kt * 2 + vt;

    let mut qkv = vec![0.0f32; conv_dim];
    crate::model::ops::matvec(w.in_qkv, conv_dim, d, x, &mut qkv);
    let mut conved = vec![0.0f32; conv_dim];
    conv_step(&qkv, w.conv, c.conv_kernel, &mut cache.conv, &mut conved);

    let mut z = vec![0.0f32; vt];
    crate::model::ops::matvec(w.in_z, vt, d, x, &mut z);
    let mut b_raw = vec![0.0f32; c.lin_v_heads];
    let mut a_raw = vec![0.0f32; c.lin_v_heads];
    crate::model::ops::matvec(w.in_b, c.lin_v_heads, d, x, &mut b_raw);
    crate::model::ops::matvec(w.in_a, c.lin_v_heads, d, x, &mut a_raw);

    let rep = c.lin_v_heads / c.lin_k_heads.max(1);
    let (kd, vd) = (c.lin_k_dim, c.lin_v_dim);
    let mut mixed = vec![0.0f32; vt];
    for h in 0..c.lin_v_heads {
        let kh = h / rep.max(1);
        let mut q: Vec<f32> = conved[kh * kd..(kh + 1) * kd].to_vec();
        let mut k: Vec<f32> = conved[kt + kh * kd..kt + (kh + 1) * kd].to_vec();
        let v = &conved[2 * kt + h * vd..2 * kt + (h + 1) * vd];
        l2norm(&mut q, 1e-6);
        l2norm(&mut k, 1e-6);
        let scale = 1.0 / (kd as f32).sqrt();
        for t in q.iter_mut() {
            *t *= scale;
        }
        let beta = 1.0 / (1.0 + (-b_raw[h]).exp());
        let sp = {
            let t = a_raw[h] + w.dt_bias[h];
            if t > 20.0 {
                t
            } else {
                (1.0 + t.exp()).ln()
            }
        };
        let g = -w.a_log[h].exp() * sp;
        let st = &mut cache.state[h * kd * vd..(h + 1) * kd * vd];
        delta_step(st, &q, &k, v, g, beta, &mut mixed[h * vd..(h + 1) * vd]);
    }
    // per-head gated norm, then the output projection
    for h in 0..c.lin_v_heads {
        let (s, e) = (h * vd, (h + 1) * vd);
        rmsnorm_gated(&mut mixed[s..e], w.norm, &z[s..e], c.norm_eps as f32);
    }
    crate::model::ops::matvec(w.out_proj, d, vt, &mixed, out);
}

#[cfg(test)]
mod forward_tests {
    use super::*;

    fn tiny() -> QwenConfig {
        let mut c = QwenConfig::qwen35_moe();
        c.n_layers = 4;
        c.d = 8;
        c.n_experts = 4;
        c.top_k = 2;
        c.moe_inter = 6;
        c.shared_inter = 6;
        c.lin_k_heads = 1;
        c.lin_v_heads = 2;
        c.lin_k_dim = 4;
        c.lin_v_dim = 4;
        c
    }

    #[test]
    fn moe_mixes_only_the_selected_experts() {
        let c = tiny();
        let d = c.d;
        // expert e outputs e+1 on every channel, whatever the input:
        // gate rows are zero except a bias-like constant path, so we
        // build the output directly through down with a fixed hidden
        let mut router = vec![0.0f32; c.n_experts * d];
        // expert 0 wins on channel 0, expert 3 on channel 1
        router[0] = 10.0;
        router[3 * d + 1] = 10.0;
        let mk = |scale: f32| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let g = vec![1.0f32; c.moe_inter * d];
            let u = vec![1.0f32; c.moe_inter * d];
            let dn = vec![scale; c.moe_inter * d];
            (g, u, dn)
        };
        let bank: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> =
            (0..c.n_experts).map(|e| mk((e + 1) as f32)).collect();
        let sh = mk(0.0);
        let b = MoeBlock {
            router: &router,
            experts: bank
                .iter()
                .map(|(g, u, dn)| (&g[..], &u[..], &dn[..]))
                .collect(),
            shared: (&sh.0[..], &sh.1[..], &sh.2[..]),
            shared_gate: &vec![0.0f32; d],
        };
        let mut x = vec![0.0f32; d];
        x[0] = 1.0;
        let mut out = vec![0.0f32; d];
        moe_forward(&b, &x, &c, &mut out);
        // expert 0 dominates: output sign follows its down scale
        assert!(out[0] > 0.0);
        let base = out[0];

        x[0] = 0.0;
        x[1] = 1.0;
        moe_forward(&b, &x, &c, &mut out);
        // expert 3 has 4x the down scale of expert 0
        assert!(out[0] > base * 2.0, "{} vs {}", out[0], base);
    }

    #[test]
    fn lin_attn_step_runs_and_carries_state() {
        let c = tiny();
        let (d, kt, vt) = (c.d, c.lin_key_total(), c.lin_value_total());
        let conv_dim = kt * 2 + vt;
        let w = LinAttn {
            in_qkv: &vec![0.1f32; conv_dim * d],
            in_z: &vec![0.2f32; vt * d],
            in_b: &vec![0.0f32; c.lin_v_heads * d],
            in_a: &vec![0.0f32; c.lin_v_heads * d],
            conv: &{
                let mut v = vec![0.0f32; conv_dim * c.conv_kernel];
                for i in 0..conv_dim {
                    v[i * c.conv_kernel + c.conv_kernel - 1] = 1.0;
                }
                v
            },
            a_log: &vec![0.0f32; c.lin_v_heads],
            dt_bias: &vec![0.0f32; c.lin_v_heads],
            norm: &vec![1.0f32; c.lin_v_dim],
            out_proj: &vec![0.05f32; d * vt],
        };
        let mut cache = LinCache::new(&c);
        let x = vec![1.0f32; d];
        let mut out1 = vec![0.0f32; d];
        lin_attn_step(&w, &c, &x, &mut cache, &mut out1);
        assert!(out1.iter().all(|v| v.is_finite()));
        // the state is no longer empty, so a second identical token
        // produces a different output: the layer is recurrent
        let mut out2 = vec![0.0f32; d];
        lin_attn_step(&w, &c, &x, &mut cache, &mut out2);
        let moved: f32 = out1.iter().zip(&out2).map(|(a, b)| (a - b).abs()).sum();
        assert!(moved > 1e-6, "state did not carry: {:?} {:?}", out1, out2);
    }

    #[test]
    fn rmsnorm_matches_the_gated_variant_at_unit_gate() {
        let x = vec![3.0f32, 4.0, 0.0];
        // Plain Qwen RMSNorm stores an offset from one, while the gated
        // recurrent norm stores a direct multiplier.
        let w = vec![0.0f32; 3];
        let mut plain = vec![0.0f32; 3];
        rmsnorm(&x, &w, 1e-6, &mut plain);
        let mut gated = x.clone();
        // silu(g) = g/(1+e^-g); pick g so the factor is 1
        let g: Vec<f32> = vec![1.2784645f32; 3];
        rmsnorm_gated(&mut gated, &[1.0; 3], &g, 1e-6);
        for i in 0..3 {
            assert!(
                (plain[i] - gated[i]).abs() < 1e-3,
                "{:?} {:?}",
                plain,
                gated
            );
        }
    }

    #[test]
    fn ffn_down_projection_is_d_by_intermediate() {
        let x = [1.0f32, 0.0];
        let gate = [1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0];
        let up = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        let down = [1.0f32, 10.0, 100.0, -1.0, -10.0, -100.0];
        let mut out = [0.0f32; 2];
        ffn(&x, &gate, &up, &down, 3, 2, &mut out);
        let h = [
            1.0 / (1.0 + (-1.0f32).exp()),
            2.0 / (1.0 + (-2.0f32).exp()),
            3.0 / (1.0 + (-3.0f32).exp()),
        ];
        let expected = [
            h[0] + 10.0 * h[1] + 100.0 * h[2],
            -h[0] - 10.0 * h[1] - 100.0 * h[2],
        ];
        for i in 0..2 {
            assert!(
                (out[i] - expected[i]).abs() < 1e-5,
                "{:?} vs {:?}",
                out,
                expected
            );
        }
    }

    #[test]
    fn full_attention_first_token_uses_value_and_sigmoid_gate() {
        let mut c = tiny();
        c.d = 4;
        c.n_heads = 2;
        c.n_kv_heads = 1;
        c.head_dim = 2;
        c.partial_rotary = 1.0;
        let mut v_proj = vec![0.0f32; 2 * c.d];
        v_proj[0] = 1.0;
        v_proj[c.d + 1] = 1.0;
        let mut o_proj = vec![0.0f32; c.d * 4];
        for i in 0..4 {
            o_proj[i * 4 + i] = 1.0;
        }
        let q_proj = vec![0.0f32; 8 * c.d];
        let k_proj = vec![0.0f32; 2 * c.d];
        let norm = vec![0.0f32; 2];
        let w = FullAttn {
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            o_proj: &o_proj,
            q_norm: &norm,
            k_norm: &norm,
        };
        let mut cache = FullCache::new(&c);
        let mut out = vec![0.0f32; c.d];
        full_attn_step(&w, &c, &[1.0, 2.0, 3.0, 4.0], 0, &mut cache, &mut out);
        assert_eq!(cache.len, 1);
        let expected = [0.5f32, 1.0, 0.5, 1.0];
        for i in 0..4 {
            assert!((out[i] - expected[i]).abs() < 1e-6, "{:?}", out);
        }
    }
}

// ─────────────────────── checkpoint-backed runtime ───────────────────────

#[derive(Clone, Copy)]
struct PackedT {
    off: usize,
    rows: usize,
    cols: usize,
}

#[derive(Clone, Copy)]
struct QwenLinW {
    in_qkv: T,
    in_z: T,
    in_b: T,
    in_a: T,
    conv: T,
    a_log: T,
    dt_bias: T,
    norm: T,
    out_proj: T,
}

#[derive(Clone, Copy)]
struct QwenFullW {
    q_proj: T,
    k_proj: T,
    v_proj: T,
    o_proj: T,
    q_norm: T,
    k_norm: T,
}

enum QwenAttnW {
    Linear(QwenLinW),
    Full(QwenFullW),
}

enum QwenMlpW {
    Moe {
        router: T,
        experts: Vec<[PackedT; 3]>, // w1=gate, w2=down, w3=up
        shared: [T; 3],             // gate, down, up
        shared_gate: T,
    },
    Dense {
        gate: PackedT,
        up: PackedT,
        down: PackedT,
    },
}

struct QwenLayerW {
    input_norm: T,
    post_norm: T,
    attn: QwenAttnW,
    mlp: QwenMlpW,
}

enum QwenCache {
    Linear(LinCache),
    Full(FullCache),
}

/// Multi-token-prediction draft head (dense variant): the fc merge of the
/// normed input embedding and the trunk's final-norm hidden, one
/// trunk-style full-attention decoder layer, and a final norm feeding the
/// shared language-model head.
struct QwenMtpW {
    fc: T,
    norm_e: T,
    norm_h: T,
    norm_f: T,
    input_norm: T,
    post_norm: T,
    attn: QwenFullW,
    mlp_gate: PackedT,
    mlp_up: PackedT,
    mlp_down: PackedT,
}

/// Rollback point for speculative decoding: the linear states are
/// recurrent (not truncatable) and are cloned; the append-only key/value
/// caches only record their lengths and are truncated on restore.
pub struct QwenSnapshot {
    lin: Vec<(Vec<f32>, Vec<f32>)>,
    full_lens: Vec<usize>,
    mtp_len: usize,
    pos: usize,
}

/// A zero-copy Qwen3.5-family text decoder backed by an MKIM0002 file.
/// Float spine tensors remain in private file-backed pages. Routed experts
/// stay MXFP4-packed and are evaluated only when selected by the router;
/// the dense variant's MLP matrices stay MXFP4-packed and run row-parallel.
pub struct QwenModel {
    pub cfg: QwenConfig,
    bin: BinFile,
    embed: T,
    norm_f: T,
    lm_head: T,
    lm_head_q8: Option<Q8Head>,
    layers: Vec<QwenLayerW>,
    caches: Vec<QwenCache>,
    mtp: Option<QwenMtpW>,
    mtp_cache: FullCache,
    pos: usize,
    adapter_packs: AppliedPacks,
}

fn dims_match(actual: &[u32], expected: &[usize]) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(&a, &b)| a as usize == b)
}

fn expect_f32(bin: &BinFile, name: &str, dims: &[usize]) -> T {
    let e = bin
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("missing Qwen tensor: {}", name));
    assert_eq!(e.dtype, DTYPE_F32, "{} must be f32", name);
    assert!(
        dims_match(&e.dims, dims),
        "{}: expected shape {:?}, found {:?}",
        name,
        dims,
        e.dims
    );
    assert_eq!(
        e.size as usize,
        dims.iter().product::<usize>() * 4,
        "{}: invalid byte size",
        name
    );
    T::from(e)
}

fn expect_conv(bin: &BinFile, name: &str, channels: usize, kernel: usize) -> T {
    let e = bin
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("missing Qwen tensor: {}", name));
    assert_eq!(e.dtype, DTYPE_F32, "{} must be f32", name);
    let ok =
        dims_match(&e.dims, &[channels, kernel]) || dims_match(&e.dims, &[channels, 1, kernel]);
    assert!(
        ok,
        "{}: expected [{}, 1, {}], found {:?}",
        name, channels, kernel, e.dims
    );
    assert_eq!(
        e.size as usize,
        channels * kernel * 4,
        "{}: invalid byte size",
        name
    );
    T::from(e)
}

fn expect_packed(bin: &BinFile, name: &str, rows: usize, cols: usize) -> PackedT {
    let e = bin
        .entries
        .get(name)
        .unwrap_or_else(|| panic!("missing Qwen expert tensor: {}", name));
    assert_eq!(e.dtype, DTYPE_MXFP4, "{} must be MXFP4", name);
    assert!(
        dims_match(&e.dims, &[rows, cols]),
        "{}: expected [{}, {}], found {:?}",
        name,
        rows,
        cols,
        e.dims
    );
    let size = rows * cols / 2 + rows * cols / 32;
    assert_eq!(e.size as usize, size, "{}: invalid MXFP4 byte size", name);
    PackedT {
        off: e.offset as usize,
        rows,
        cols,
    }
}

#[inline]
fn tensor<'a>(data: &'a [u8], t: &T) -> &'a [f32] {
    as_f32(&data[t.off..t.off + t.len * 4])
}

#[inline]
fn packed_parts<'a>(data: &'a [u8], t: &PackedT) -> (&'a [u8], &'a [u8]) {
    let np = t.rows * t.cols / 2;
    let ns = t.rows * t.cols / 32;
    (&data[t.off..t.off + np], &data[t.off + np..t.off + np + ns])
}

fn packed_moe(
    data: &[u8],
    router: &T,
    experts: &[[PackedT; 3]],
    shared: &[T; 3],
    shared_gate: &T,
    c: &QwenConfig,
    x: &[f32],
) -> Vec<f32> {
    let mut logits = vec![0.0f32; c.n_experts];
    crate::model::ops::matvec(tensor(data, router), c.n_experts, c.d, x, &mut logits);
    let selected = route_topk(&logits, c.top_k);
    let mut routed = vec![0.0f32; selected.len() * c.d];

    // Selected experts are independent. Each job keeps packed GEMVs
    // single-threaded so the outer pool supplies the parallelism once.
    {
        let dp = crate::model::pool::SPtrU8(data.as_ptr());
        let dlen = data.len();
        let xp = crate::model::pool::SPtr(x.as_ptr());
        let op = crate::model::pool::MPtr(routed.as_mut_ptr());
        let mut jobs: Vec<crate::model::pool::Job> = Vec::with_capacity(selected.len());
        for (slot, &(expert, _)) in selected.iter().enumerate() {
            let weights = experts[expert];
            let d = c.d;
            let inter = c.moe_inter;
            jobs.push(Box::new(move || {
                // Rebind the wrappers so the closure captures their Send
                // newtypes, not the raw pointer fields directly.
                let (dp, xp, op) = (dp, xp, op);
                unsafe {
                    let data = std::slice::from_raw_parts(dp.0, dlen);
                    let x = std::slice::from_raw_parts(xp.0, d);
                    let mut gate = vec![0.0f32; inter];
                    let mut up = vec![0.0f32; inter];
                    let (p1, s1) = packed_parts(data, &weights[0]);
                    let (p3, s3) = packed_parts(data, &weights[2]);
                    crate::quant::mxfp4::matvec_packed(p1, s1, inter, d, x, &mut gate, 1);
                    crate::quant::mxfp4::matvec_packed(p3, s3, inter, d, x, &mut up, 1);
                    for i in 0..inter {
                        gate[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
                    }
                    let out = std::slice::from_raw_parts_mut(op.0.add(slot * d), d);
                    let (p2, s2) = packed_parts(data, &weights[1]);
                    crate::quant::mxfp4::matvec_packed(p2, s2, d, inter, &gate, out, 1);
                }
            }));
        }
        crate::model::pool::pool().run(jobs);
    }

    let mut out = vec![0.0f32; c.d];
    for (slot, &(_, weight)) in selected.iter().enumerate() {
        for i in 0..c.d {
            out[i] += weight * routed[slot * c.d + i];
        }
    }

    let shared_gate = crate::model::ops::dot(tensor(data, shared_gate), x);
    let shared_scale = 1.0 / (1.0 + (-shared_gate).exp());
    let mut shared_out = vec![0.0f32; c.d];
    ffn(
        x,
        tensor(data, &shared[0]),
        tensor(data, &shared[2]),
        tensor(data, &shared[1]),
        c.shared_inter,
        c.d,
        &mut shared_out,
    );
    for i in 0..c.d {
        out[i] += shared_scale * shared_out[i];
    }
    out
}

/// Dense-variant MLP: down(silu(gate(x)) * up(x)) over MXFP4-packed
/// matrices. The three matvecs dominate a dense layer, so each one runs
/// row-parallel across the worker pool.
fn packed_dense_mlp(
    data: &[u8],
    gate: &PackedT,
    up: &PackedT,
    down: &PackedT,
    c: &QwenConfig,
    x: &[f32],
) -> Vec<f32> {
    let inter = c.dense_inter;
    let threads = crate::model::pool::pool().workers.max(1);
    let mut h_gate = vec![0.0f32; inter];
    let mut h_up = vec![0.0f32; inter];
    let (pg, sg) = packed_parts(data, gate);
    crate::quant::mxfp4::matvec_packed(pg, sg, inter, c.d, x, &mut h_gate, threads);
    let (pu, su) = packed_parts(data, up);
    crate::quant::mxfp4::matvec_packed(pu, su, inter, c.d, x, &mut h_up, threads);
    for i in 0..inter {
        h_gate[i] = (h_gate[i] / (1.0 + (-h_gate[i]).exp())) * h_up[i];
    }
    let mut out = vec![0.0f32; c.d];
    let (pd, sd) = packed_parts(data, down);
    crate::quant::mxfp4::matvec_packed(pd, sd, c.d, inter, &h_gate, &mut out, threads);
    out
}

impl QwenModel {
    pub fn load(path: &str) -> QwenModel {
        Self::load_with_adapters(path, &[])
    }

    /// Applies hash-bound low-rank packs to private f32 spine pages before
    /// tensor handles are created. Packed routed experts remain immutable.
    pub fn load_with_adapters(path: &str, pack_paths: &[String]) -> QwenModel {
        let mut bin = BinFile::open(path);
        let adapter_packs = super::adapter::apply_packs(path, &mut bin, pack_paths)
            .unwrap_or_else(|e| panic!("adapter pack: {}", e));
        Self::from_bin(bin, adapter_packs)
    }

    fn from_bin(bin: BinFile, adapter_packs: AppliedPacks) -> QwenModel {
        let c = bin
            .config
            .qwen
            .clone()
            .expect("model is not a Qwen-family MKIM0002 file");
        assert!(c.n_kv_heads > 0 && c.n_heads % c.n_kv_heads == 0);
        assert!(c.lin_k_heads > 0 && c.lin_v_heads % c.lin_k_heads == 0);
        if c.is_dense() {
            assert!(c.d % 32 == 0 && c.dense_inter % 32 == 0);
        } else {
            assert!(c.top_k > 0 && c.top_k <= c.n_experts);
            assert!(c.d % 32 == 0 && c.moe_inter % 32 == 0 && c.shared_inter > 0);
        }

        let embed = expect_f32(
            &bin,
            "model.language_model.embed_tokens.weight",
            &[c.vocab, c.d],
        );
        let norm_f = expect_f32(&bin, "model.language_model.norm.weight", &[c.d]);
        let lm_head = expect_f32(&bin, "lm_head.weight", &[c.vocab, c.d]);
        let mut layers = Vec::with_capacity(c.n_layers);
        let conv_dim = c.lin_key_total() * 2 + c.lin_value_total();
        let full_width = c.n_heads * c.head_dim;
        let kv_width = c.n_kv_heads * c.head_dim;

        for l in 0..c.n_layers {
            let p = format!("model.language_model.layers.{}", l);
            let input_norm = expect_f32(&bin, &format!("{}.input_layernorm.weight", p), &[c.d]);
            let post_norm = expect_f32(
                &bin,
                &format!("{}.post_attention_layernorm.weight", p),
                &[c.d],
            );
            let attn = if c.is_full_attn(l) {
                QwenAttnW::Full(QwenFullW {
                    q_proj: expect_f32(
                        &bin,
                        &format!("{}.self_attn.q_proj.weight", p),
                        &[full_width * 2, c.d],
                    ),
                    k_proj: expect_f32(
                        &bin,
                        &format!("{}.self_attn.k_proj.weight", p),
                        &[kv_width, c.d],
                    ),
                    v_proj: expect_f32(
                        &bin,
                        &format!("{}.self_attn.v_proj.weight", p),
                        &[kv_width, c.d],
                    ),
                    o_proj: expect_f32(
                        &bin,
                        &format!("{}.self_attn.o_proj.weight", p),
                        &[c.d, full_width],
                    ),
                    q_norm: expect_f32(
                        &bin,
                        &format!("{}.self_attn.q_norm.weight", p),
                        &[c.head_dim],
                    ),
                    k_norm: expect_f32(
                        &bin,
                        &format!("{}.self_attn.k_norm.weight", p),
                        &[c.head_dim],
                    ),
                })
            } else {
                QwenAttnW::Linear(QwenLinW {
                    in_qkv: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.in_proj_qkv.weight", p),
                        &[conv_dim, c.d],
                    ),
                    in_z: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.in_proj_z.weight", p),
                        &[c.lin_value_total(), c.d],
                    ),
                    in_b: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.in_proj_b.weight", p),
                        &[c.lin_v_heads, c.d],
                    ),
                    in_a: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.in_proj_a.weight", p),
                        &[c.lin_v_heads, c.d],
                    ),
                    conv: expect_conv(
                        &bin,
                        &format!("{}.linear_attn.conv1d.weight", p),
                        conv_dim,
                        c.conv_kernel,
                    ),
                    a_log: expect_f32(&bin, &format!("{}.linear_attn.A_log", p), &[c.lin_v_heads]),
                    dt_bias: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.dt_bias", p),
                        &[c.lin_v_heads],
                    ),
                    norm: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.norm.weight", p),
                        &[c.lin_v_dim],
                    ),
                    out_proj: expect_f32(
                        &bin,
                        &format!("{}.linear_attn.out_proj.weight", p),
                        &[c.d, c.lin_value_total()],
                    ),
                })
            };
            let mlp = if c.is_dense() {
                QwenMlpW::Dense {
                    gate: expect_packed(
                        &bin,
                        &format!("{}.mlp.gate_proj.weight", p),
                        c.dense_inter,
                        c.d,
                    ),
                    up: expect_packed(
                        &bin,
                        &format!("{}.mlp.up_proj.weight", p),
                        c.dense_inter,
                        c.d,
                    ),
                    down: expect_packed(
                        &bin,
                        &format!("{}.mlp.down_proj.weight", p),
                        c.d,
                        c.dense_inter,
                    ),
                }
            } else {
                let router =
                    expect_f32(&bin, &format!("{}.mlp.gate.weight", p), &[c.n_experts, c.d]);
                let shared = [
                    expect_f32(
                        &bin,
                        &format!("{}.mlp.shared_expert.gate_proj.weight", p),
                        &[c.shared_inter, c.d],
                    ),
                    expect_f32(
                        &bin,
                        &format!("{}.mlp.shared_expert.down_proj.weight", p),
                        &[c.d, c.shared_inter],
                    ),
                    expect_f32(
                        &bin,
                        &format!("{}.mlp.shared_expert.up_proj.weight", p),
                        &[c.shared_inter, c.d],
                    ),
                ];
                let shared_gate = expect_f32(
                    &bin,
                    &format!("{}.mlp.shared_expert_gate.weight", p),
                    &[1, c.d],
                );
                let experts = (0..c.n_experts)
                    .map(|e| {
                        let ep = format!("layers.{}.block_sparse_moe.experts.{}", l, e);
                        [
                            expect_packed(&bin, &format!("{}.w1", ep), c.moe_inter, c.d),
                            expect_packed(&bin, &format!("{}.w2", ep), c.d, c.moe_inter),
                            expect_packed(&bin, &format!("{}.w3", ep), c.moe_inter, c.d),
                        ]
                    })
                    .collect();
                QwenMlpW::Moe {
                    router,
                    experts,
                    shared,
                    shared_gate,
                }
            };
            layers.push(QwenLayerW {
                input_norm,
                post_norm,
                attn,
                mlp,
            });
        }

        let mtp = if c.mtp_layers > 0 {
            assert!(
                c.is_dense() && c.mtp_layers == 1,
                "MTP runtime supports exactly one draft layer on the dense variant"
            );
            Some(QwenMtpW {
                fc: expect_f32(&bin, "mtp.fc.weight", &[c.d, 2 * c.d]),
                norm_e: expect_f32(&bin, "mtp.pre_fc_norm_embedding.weight", &[c.d]),
                norm_h: expect_f32(&bin, "mtp.pre_fc_norm_hidden.weight", &[c.d]),
                norm_f: expect_f32(&bin, "mtp.norm.weight", &[c.d]),
                input_norm: expect_f32(&bin, "mtp.layers.0.input_layernorm.weight", &[c.d]),
                post_norm: expect_f32(&bin, "mtp.layers.0.post_attention_layernorm.weight", &[c.d]),
                attn: QwenFullW {
                    q_proj: expect_f32(
                        &bin,
                        "mtp.layers.0.self_attn.q_proj.weight",
                        &[full_width * 2, c.d],
                    ),
                    k_proj: expect_f32(
                        &bin,
                        "mtp.layers.0.self_attn.k_proj.weight",
                        &[kv_width, c.d],
                    ),
                    v_proj: expect_f32(
                        &bin,
                        "mtp.layers.0.self_attn.v_proj.weight",
                        &[kv_width, c.d],
                    ),
                    o_proj: expect_f32(
                        &bin,
                        "mtp.layers.0.self_attn.o_proj.weight",
                        &[c.d, full_width],
                    ),
                    q_norm: expect_f32(&bin, "mtp.layers.0.self_attn.q_norm.weight", &[c.head_dim]),
                    k_norm: expect_f32(&bin, "mtp.layers.0.self_attn.k_norm.weight", &[c.head_dim]),
                },
                mlp_gate: expect_packed(
                    &bin,
                    "mtp.layers.0.mlp.gate_proj.weight",
                    c.dense_inter,
                    c.d,
                ),
                mlp_up: expect_packed(&bin, "mtp.layers.0.mlp.up_proj.weight", c.dense_inter, c.d),
                mlp_down: expect_packed(
                    &bin,
                    "mtp.layers.0.mlp.down_proj.weight",
                    c.d,
                    c.dense_inter,
                ),
            })
        } else {
            None
        };

        let lm_head_q8 = if super::ops::q8head_enabled() && !super::gpu_on() && c.d % 32 == 0 {
            Some(Q8Head::from_f32(tensor(&bin.data, &lm_head), c.vocab, c.d))
        } else {
            None
        };
        let caches = (0..c.n_layers)
            .map(|l| {
                if c.is_full_attn(l) {
                    QwenCache::Full(FullCache::new(&c))
                } else {
                    QwenCache::Linear(LinCache::new(&c))
                }
            })
            .collect();
        let mtp_cache = FullCache::new(&c);
        QwenModel {
            cfg: c,
            bin,
            embed,
            norm_f,
            lm_head,
            lm_head_q8,
            layers,
            caches,
            mtp,
            mtp_cache,
            pos: 0,
            adapter_packs,
        }
    }

    pub fn reset(&mut self) {
        self.caches = (0..self.cfg.n_layers)
            .map(|l| {
                if self.cfg.is_full_attn(l) {
                    QwenCache::Full(FullCache::new(&self.cfg))
                } else {
                    QwenCache::Linear(LinCache::new(&self.cfg))
                }
            })
            .collect();
        self.mtp_cache = FullCache::new(&self.cfg);
        self.pos = 0;
    }

    /// The converted checkpoint carries the multi-token-prediction head.
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Captures the rollback point for a speculative batch.
    pub fn snapshot(&self) -> QwenSnapshot {
        QwenSnapshot {
            lin: self
                .caches
                .iter()
                .filter_map(|cache| match cache {
                    QwenCache::Linear(c) => Some((c.state.clone(), c.conv.clone())),
                    QwenCache::Full(_) => None,
                })
                .collect(),
            full_lens: self
                .caches
                .iter()
                .filter_map(|cache| match cache {
                    QwenCache::Full(c) => Some(c.len),
                    QwenCache::Linear(_) => None,
                })
                .collect(),
            mtp_len: self.mtp_cache.len,
            pos: self.pos,
        }
    }

    /// Restores a snapshot: linear states are copied back, append-only
    /// key/value caches are truncated to their recorded lengths.
    pub fn restore(&mut self, snap: &QwenSnapshot) {
        let kv_width = self.cfg.n_kv_heads * self.cfg.head_dim;
        let (mut li, mut fi) = (0usize, 0usize);
        for cache in self.caches.iter_mut() {
            match cache {
                QwenCache::Linear(c) => {
                    let (state, conv) = &snap.lin[li];
                    c.state.copy_from_slice(state);
                    c.conv.copy_from_slice(conv);
                    li += 1;
                }
                QwenCache::Full(c) => {
                    let len = snap.full_lens[fi];
                    c.k.truncate(len * kv_width);
                    c.v.truncate(len * kv_width);
                    c.len = len;
                    fi += 1;
                }
            }
        }
        self.mtp_cache.k.truncate(snap.mtp_len * kv_width);
        self.mtp_cache.v.truncate(snap.mtp_len * kv_width);
        self.mtp_cache.len = snap.mtp_len;
        self.pos = snap.pos;
    }

    /// One multi-token-prediction step. Draft slot `i` pairs the committed
    /// token at position `i + 1` with the trunk's final-norm hidden of
    /// position `i`, at rotary position `i` (the reference proposer shifts
    /// the input ids and leaves the positions unchanged). Returns the draft
    /// logits when `want_logits`; cache-building calls skip the head.
    pub fn mtp_advance(&mut self, token: u32, hidden: &[f32], want_logits: bool) -> Option<Vec<f32>> {
        let c = self.cfg.clone();
        let mtp = self.mtp.as_ref().expect("model has no MTP head");
        let data = &self.bin.data;
        let d = c.d;
        assert!((token as usize) < c.vocab && hidden.len() == d);

        let embed = tensor(data, &self.embed);
        let mut merged = vec![0.0f32; 2 * d];
        {
            let (e_half, h_half) = merged.split_at_mut(d);
            rmsnorm(
                &embed[token as usize * d..(token as usize + 1) * d],
                tensor(data, &mtp.norm_e),
                c.norm_eps as f32,
                e_half,
            );
            rmsnorm(hidden, tensor(data, &mtp.norm_h), c.norm_eps as f32, h_half);
        }
        let mut x = vec![0.0f32; d];
        crate::model::ops::matvec(tensor(data, &mtp.fc), d, 2 * d, &merged, &mut x);

        let mut normed = vec![0.0f32; d];
        let mut attn_out = vec![0.0f32; d];
        rmsnorm(&x, tensor(data, &mtp.input_norm), c.norm_eps as f32, &mut normed);
        let weights = FullAttn {
            q_proj: tensor(data, &mtp.attn.q_proj),
            k_proj: tensor(data, &mtp.attn.k_proj),
            v_proj: tensor(data, &mtp.attn.v_proj),
            o_proj: tensor(data, &mtp.attn.o_proj),
            q_norm: tensor(data, &mtp.attn.q_norm),
            k_norm: tensor(data, &mtp.attn.k_norm),
        };
        let pos = self.mtp_cache.len;
        full_attn_step(&weights, &c, &normed, pos, &mut self.mtp_cache, &mut attn_out);
        for i in 0..d {
            x[i] += attn_out[i];
        }
        rmsnorm(&x, tensor(data, &mtp.post_norm), c.norm_eps as f32, &mut normed);
        let mlp = packed_dense_mlp(data, &mtp.mlp_gate, &mtp.mlp_up, &mtp.mlp_down, &c, &normed);
        for i in 0..d {
            x[i] += mlp[i];
        }
        if !want_logits {
            return None;
        }
        rmsnorm(&x, tensor(data, &mtp.norm_f), c.norm_eps as f32, &mut normed);
        let mut logits = vec![0.0f32; c.vocab];
        match &self.lm_head_q8 {
            Some(head) => head.matvec(&normed, &mut logits),
            None => crate::model::ops::matvec(
                tensor(data, &self.lm_head),
                c.vocab,
                d,
                &normed,
                &mut logits,
            ),
        }
        Some(logits)
    }

    pub fn has_adapter_packs(&self) -> bool {
        !self.adapter_packs.is_empty()
    }

    pub fn adapter_set_sha256(&self) -> Option<&str> {
        self.adapter_packs.set_sha256.as_deref()
    }

    /// Advances the autoregressive decoder by one token and returns logits.
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        assert!(
            (token as usize) < self.cfg.vocab,
            "token {} is outside the Qwen vocabulary",
            token
        );
        let c = &self.cfg;
        let data = &self.bin.data;
        let d = c.d;
        let mut hidden =
            tensor(data, &self.embed)[token as usize * d..(token as usize + 1) * d].to_vec();
        let mut normed = vec![0.0f32; d];
        let mut attn_out = vec![0.0f32; d];

        for l in 0..c.n_layers {
            let layer = &self.layers[l];
            rmsnorm(
                &hidden,
                tensor(data, &layer.input_norm),
                c.norm_eps as f32,
                &mut normed,
            );
            match (&layer.attn, &mut self.caches[l]) {
                (QwenAttnW::Linear(w), QwenCache::Linear(cache)) => {
                    let weights = LinAttn {
                        in_qkv: tensor(data, &w.in_qkv),
                        in_z: tensor(data, &w.in_z),
                        in_b: tensor(data, &w.in_b),
                        in_a: tensor(data, &w.in_a),
                        conv: tensor(data, &w.conv),
                        a_log: tensor(data, &w.a_log),
                        dt_bias: tensor(data, &w.dt_bias),
                        norm: tensor(data, &w.norm),
                        out_proj: tensor(data, &w.out_proj),
                    };
                    lin_attn_step(&weights, c, &normed, cache, &mut attn_out);
                }
                (QwenAttnW::Full(w), QwenCache::Full(cache)) => {
                    let weights = FullAttn {
                        q_proj: tensor(data, &w.q_proj),
                        k_proj: tensor(data, &w.k_proj),
                        v_proj: tensor(data, &w.v_proj),
                        o_proj: tensor(data, &w.o_proj),
                        q_norm: tensor(data, &w.q_norm),
                        k_norm: tensor(data, &w.k_norm),
                    };
                    full_attn_step(&weights, c, &normed, self.pos, cache, &mut attn_out);
                }
                _ => unreachable!("Qwen attention/cache kind mismatch at layer {}", l),
            }
            for i in 0..d {
                hidden[i] += attn_out[i];
            }
            rmsnorm(
                &hidden,
                tensor(data, &layer.post_norm),
                c.norm_eps as f32,
                &mut normed,
            );
            let mlp = match &layer.mlp {
                QwenMlpW::Moe {
                    router,
                    experts,
                    shared,
                    shared_gate,
                } => packed_moe(data, router, experts, shared, shared_gate, c, &normed),
                QwenMlpW::Dense { gate, up, down } => {
                    packed_dense_mlp(data, gate, up, down, c, &normed)
                }
            };
            for i in 0..d {
                hidden[i] += mlp[i];
            }
        }

        rmsnorm(
            &hidden,
            tensor(data, &self.norm_f),
            c.norm_eps as f32,
            &mut normed,
        );
        let mut logits = vec![0.0f32; c.vocab];
        match &self.lm_head_q8 {
            Some(head) => head.matvec(&normed, &mut logits),
            None => crate::model::ops::matvec(
                tensor(data, &self.lm_head),
                c.vocab,
                d,
                &normed,
                &mut logits,
            ),
        }
        self.pos += 1;
        logits
    }
}

// ───────────────────────── batched layers-outer prefill ─────────────────────────

/// Number of prefill worker threads (the shared pool size; prefill phases
/// use scoped threads over contiguous token or head ranges).
fn prefill_workers(items: usize) -> usize {
    crate::model::pool::pool().workers.max(1).min(items.max(1))
}

/// Splits `items` into per-worker contiguous ranges.
fn ranges(items: usize, workers: usize) -> Vec<(usize, usize)> {
    let per = items.div_ceil(workers.max(1));
    (0..workers)
        .map(|w| (w * per, ((w + 1) * per).min(items)))
        .filter(|(a, b)| a < b)
        .collect()
}

/// Per-position output of a batched prefill.
pub struct QwenPrefillOut {
    /// Logits per requested position (all positions, or only the last).
    pub logits: Vec<Vec<f32>>,
    /// Final-norm hidden state per position (the lm_head input; the MTP
    /// drafter consumes these).
    pub hidden: Vec<Vec<f32>>,
}

impl QwenModel {
    /// Ingests `tokens` and returns the logits after the last one,
    /// bit-identical to the same sequence of `forward` calls. The prompt is
    /// processed layer-by-layer (layers-outer): each weight region is
    /// traversed once per chunk instead of once per token, and
    /// token-independent work fans out over scoped worker threads.
    pub fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.prefill_collect(tokens, false)
            .logits
            .pop()
            .expect("prefill requires at least one token")
    }

    /// Batched ingestion that also returns per-position final-norm hiddens
    /// and, with `all_logits`, the logits of every position (speculative
    /// verification). `logits` holds one entry per position, or only the
    /// last position's entry when `all_logits` is false.
    pub fn prefill_collect(&mut self, tokens: &[u32], all_logits: bool) -> QwenPrefillOut {
        assert!(!tokens.is_empty(), "prefill requires at least one token");
        let c = self.cfg.clone();
        let t_count = tokens.len();
        let d = c.d;
        for &token in tokens {
            assert!(
                (token as usize) < c.vocab,
                "token {} is outside the Qwen vocabulary",
                token
            );
        }

        // hidden stream for every position, token-major
        let mut hidden = vec![0.0f32; t_count * d];
        {
            let embed = tensor(&self.bin.data, &self.embed);
            for (t, &token) in tokens.iter().enumerate() {
                hidden[t * d..(t + 1) * d]
                    .copy_from_slice(&embed[token as usize * d..(token as usize + 1) * d]);
            }
        }
        let mut normed = vec![0.0f32; t_count * d];
        let mut attn_out = vec![0.0f32; t_count * d];

        for l in 0..c.n_layers {
            let layer = &self.layers[l];
            let data = &self.bin.data;
            {
                let w = tensor(data, &layer.input_norm);
                for t in 0..t_count {
                    let (h, n) = (&hidden[t * d..(t + 1) * d], &mut normed[t * d..(t + 1) * d]);
                    rmsnorm(h, w, c.norm_eps as f32, n);
                }
            }
            match (&layer.attn, &mut self.caches[l]) {
                (QwenAttnW::Linear(w), QwenCache::Linear(cache)) => {
                    lin_attn_prefill(data, w, &c, &normed, t_count, cache, &mut attn_out);
                }
                (QwenAttnW::Full(w), QwenCache::Full(cache)) => {
                    full_attn_prefill(data, w, &c, &normed, t_count, self.pos, cache, &mut attn_out);
                }
                _ => unreachable!("Qwen attention/cache kind mismatch at layer {}", l),
            }
            for i in 0..t_count * d {
                hidden[i] += attn_out[i];
            }
            {
                let w = tensor(data, &layer.post_norm);
                for t in 0..t_count {
                    let (h, n) = (&hidden[t * d..(t + 1) * d], &mut normed[t * d..(t + 1) * d]);
                    rmsnorm(h, w, c.norm_eps as f32, n);
                }
            }
            mlp_prefill(data, &layer.mlp, &c, &normed, t_count, &mut attn_out);
            for i in 0..t_count * d {
                hidden[i] += attn_out[i];
            }
        }

        let mut out = QwenPrefillOut {
            logits: Vec::new(),
            hidden: Vec::with_capacity(t_count),
        };
        let data = &self.bin.data;
        let norm_w = tensor(data, &self.norm_f);
        for t in 0..t_count {
            let mut n = vec![0.0f32; d];
            rmsnorm(&hidden[t * d..(t + 1) * d], norm_w, c.norm_eps as f32, &mut n);
            out.hidden.push(n);
        }
        for t in 0..t_count {
            if !all_logits && t + 1 != t_count {
                continue;
            }
            let mut logits = vec![0.0f32; c.vocab];
            match &self.lm_head_q8 {
                Some(head) => head.matvec(&out.hidden[t], &mut logits),
                None => crate::model::ops::matvec(
                    tensor(data, &self.lm_head),
                    c.vocab,
                    d,
                    &out.hidden[t],
                    &mut logits,
                ),
            }
            out.logits.push(logits);
        }
        self.pos += t_count;
        out
    }
}

/// Prefill for one gated delta-rule layer. Projections fan out over token
/// ranges, the convolution is the cheap sequential scan, the recurrence
/// fans out over heads (each head replays its own token sequence), and the
/// gated norm plus output projection fan out over tokens again. Per-token
/// float operations are exactly those of `lin_attn_step`.
fn lin_attn_prefill(
    data: &[u8],
    w: &QwenLinW,
    c: &QwenConfig,
    normed: &[f32],
    t_count: usize,
    cache: &mut LinCache,
    attn_out: &mut [f32],
) {
    let d = c.d;
    let (kt, vt) = (c.lin_key_total(), c.lin_value_total());
    let conv_dim = kt * 2 + vt;
    let heads = c.lin_v_heads;
    let (kd, vd) = (c.lin_k_dim, c.lin_v_dim);
    let rep = heads / c.lin_k_heads.max(1);

    let in_qkv = tensor(data, &w.in_qkv);
    let in_z = tensor(data, &w.in_z);
    let in_b = tensor(data, &w.in_b);
    let in_a = tensor(data, &w.in_a);
    let conv = tensor(data, &w.conv);
    let a_log = tensor(data, &w.a_log);
    let dt_bias = tensor(data, &w.dt_bias);
    let norm = tensor(data, &w.norm);
    let out_proj = tensor(data, &w.out_proj);

    // projections, parallel over token ranges
    let mut qkv = vec![0.0f32; t_count * conv_dim];
    let mut z = vec![0.0f32; t_count * vt];
    let mut b_raw = vec![0.0f32; t_count * heads];
    let mut a_raw = vec![0.0f32; t_count * heads];
    {
        let workers = prefill_workers(t_count);
        std::thread::scope(|s| {
            let mut qkv_rest = qkv.as_mut_slice();
            let mut z_rest = z.as_mut_slice();
            let mut b_rest = b_raw.as_mut_slice();
            let mut a_rest = a_raw.as_mut_slice();
            for (t0, t1) in ranges(t_count, workers) {
                let n = t1 - t0;
                let (qkv_c, qr) = qkv_rest.split_at_mut(n * conv_dim);
                let (z_c, zr) = z_rest.split_at_mut(n * vt);
                let (b_c, br) = b_rest.split_at_mut(n * heads);
                let (a_c, ar) = a_rest.split_at_mut(n * heads);
                qkv_rest = qr;
                z_rest = zr;
                b_rest = br;
                a_rest = ar;
                let x = &normed[t0 * d..t1 * d];
                s.spawn(move || {
                    for i in 0..n {
                        let xt = &x[i * d..(i + 1) * d];
                        crate::model::ops::matvec_st(
                            in_qkv,
                            conv_dim,
                            d,
                            xt,
                            &mut qkv_c[i * conv_dim..(i + 1) * conv_dim],
                        );
                        crate::model::ops::matvec_st(in_z, vt, d, xt, &mut z_c[i * vt..(i + 1) * vt]);
                        crate::model::ops::matvec_st(
                            in_b,
                            heads,
                            d,
                            xt,
                            &mut b_c[i * heads..(i + 1) * heads],
                        );
                        crate::model::ops::matvec_st(
                            in_a,
                            heads,
                            d,
                            xt,
                            &mut a_c[i * heads..(i + 1) * heads],
                        );
                    }
                });
            }
        });
    }

    // causal convolution: the sequential scan (cheap, carries cache.conv)
    let mut conved = vec![0.0f32; t_count * conv_dim];
    for t in 0..t_count {
        conv_step(
            &qkv[t * conv_dim..(t + 1) * conv_dim],
            conv,
            c.conv_kernel,
            &mut cache.conv,
            &mut conved[t * conv_dim..(t + 1) * conv_dim],
        );
    }

    // recurrence, parallel over heads: each head replays its own tokens in
    // order against its private state slice (head-major mixed buffer)
    let mut mixed_hm = vec![0.0f32; heads * t_count * vd];
    {
        std::thread::scope(|s| {
            let mut state_rest = cache.state.as_mut_slice();
            let mut mixed_rest = mixed_hm.as_mut_slice();
            let conved = &conved;
            let b_raw = &b_raw;
            let a_raw = &a_raw;
            for h in 0..heads {
                let (state_h, sr) = state_rest.split_at_mut(kd * vd);
                let (mixed_h, mr) = mixed_rest.split_at_mut(t_count * vd);
                state_rest = sr;
                mixed_rest = mr;
                s.spawn(move || {
                    let kh = h / rep.max(1);
                    for t in 0..t_count {
                        let row = &conved[t * conv_dim..(t + 1) * conv_dim];
                        let mut q: Vec<f32> = row[kh * kd..(kh + 1) * kd].to_vec();
                        let mut k: Vec<f32> = row[kt + kh * kd..kt + (kh + 1) * kd].to_vec();
                        let v = &row[2 * kt + h * vd..2 * kt + (h + 1) * vd];
                        l2norm(&mut q, 1e-6);
                        l2norm(&mut k, 1e-6);
                        let scale = 1.0 / (kd as f32).sqrt();
                        for value in q.iter_mut() {
                            *value *= scale;
                        }
                        let beta = 1.0 / (1.0 + (-b_raw[t * heads + h]).exp());
                        let sp = {
                            let arg = a_raw[t * heads + h] + dt_bias[h];
                            if arg > 20.0 {
                                arg
                            } else {
                                (1.0 + arg.exp()).ln()
                            }
                        };
                        let g = -a_log[h].exp() * sp;
                        delta_step(
                            state_h,
                            &q,
                            &k,
                            v,
                            g,
                            beta,
                            &mut mixed_h[t * vd..(t + 1) * vd],
                        );
                    }
                });
            }
        });
    }

    // gated norm + output projection, parallel over token ranges
    {
        let workers = prefill_workers(t_count);
        std::thread::scope(|s| {
            let mut out_rest = &mut attn_out[..t_count * d];
            let mixed_hm = &mixed_hm;
            let z = &z;
            for (t0, t1) in ranges(t_count, workers) {
                let n = t1 - t0;
                let (out_c, or) = out_rest.split_at_mut(n * d);
                out_rest = or;
                s.spawn(move || {
                    let mut mixed = vec![0.0f32; vt];
                    for i in 0..n {
                        let t = t0 + i;
                        for h in 0..heads {
                            mixed[h * vd..(h + 1) * vd]
                                .copy_from_slice(&mixed_hm[(h * t_count + t) * vd..(h * t_count + t + 1) * vd]);
                        }
                        for h in 0..heads {
                            let (a, b) = (h * vd, (h + 1) * vd);
                            rmsnorm_gated(
                                &mut mixed[a..b],
                                norm,
                                &z[t * vt + a..t * vt + b],
                                c.norm_eps as f32,
                            );
                        }
                        crate::model::ops::matvec_st(
                            out_proj,
                            d,
                            vt,
                            &mixed,
                            &mut out_c[i * d..(i + 1) * d],
                        );
                    }
                });
            }
        });
    }
}

/// Prefill for one full-attention layer: projections and per-head norms and
/// rotary fan out over tokens, the key/value block is appended once, then
/// each position attends over its causal prefix in parallel. Per-token
/// float operations are exactly those of `full_attn_step`.
#[allow(clippy::too_many_arguments)]
fn full_attn_prefill(
    data: &[u8],
    w: &QwenFullW,
    c: &QwenConfig,
    normed: &[f32],
    t_count: usize,
    base_pos: usize,
    cache: &mut FullCache,
    attn_out: &mut [f32],
) {
    let d = c.d;
    let hd = c.head_dim;
    let q_width = c.n_heads * hd;
    let kv_width = c.n_kv_heads * hd;
    assert_eq!(cache.len, base_pos, "full-attention cache position mismatch");

    let q_proj = tensor(data, &w.q_proj);
    let k_proj = tensor(data, &w.k_proj);
    let v_proj = tensor(data, &w.v_proj);
    let o_proj = tensor(data, &w.o_proj);
    let q_norm = tensor(data, &w.q_norm);
    let k_norm = tensor(data, &w.k_norm);

    let mut q_all = vec![0.0f32; t_count * q_width];
    let mut gate_all = vec![0.0f32; t_count * q_width];
    let mut k_all = vec![0.0f32; t_count * kv_width];
    let mut v_all = vec![0.0f32; t_count * kv_width];
    {
        let workers = prefill_workers(t_count);
        std::thread::scope(|s| {
            let mut q_rest = q_all.as_mut_slice();
            let mut g_rest = gate_all.as_mut_slice();
            let mut k_rest = k_all.as_mut_slice();
            let mut v_rest = v_all.as_mut_slice();
            for (t0, t1) in ranges(t_count, workers) {
                let n = t1 - t0;
                let (q_c, qr) = q_rest.split_at_mut(n * q_width);
                let (g_c, gr) = g_rest.split_at_mut(n * q_width);
                let (k_c, kr) = k_rest.split_at_mut(n * kv_width);
                let (v_c, vr) = v_rest.split_at_mut(n * kv_width);
                q_rest = qr;
                g_rest = gr;
                k_rest = kr;
                v_rest = vr;
                let x = &normed[t0 * d..t1 * d];
                s.spawn(move || {
                    let mut qg = vec![0.0f32; q_width * 2];
                    for i in 0..n {
                        let xt = &x[i * d..(i + 1) * d];
                        let pos = base_pos + t0 + i;
                        crate::model::ops::matvec_st(q_proj, q_width * 2, d, xt, &mut qg);
                        let k = &mut k_c[i * kv_width..(i + 1) * kv_width];
                        let v = &mut v_c[i * kv_width..(i + 1) * kv_width];
                        crate::model::ops::matvec_st(k_proj, kv_width, d, xt, k);
                        crate::model::ops::matvec_st(v_proj, kv_width, d, xt, v);
                        let q = &mut q_c[i * q_width..(i + 1) * q_width];
                        let gate = &mut g_c[i * q_width..(i + 1) * q_width];
                        for h in 0..c.n_heads {
                            let src = h * hd * 2;
                            q[h * hd..(h + 1) * hd].copy_from_slice(&qg[src..src + hd]);
                            gate[h * hd..(h + 1) * hd].copy_from_slice(&qg[src + hd..src + 2 * hd]);
                            let old = q[h * hd..(h + 1) * hd].to_vec();
                            rmsnorm(&old, q_norm, c.norm_eps as f32, &mut q[h * hd..(h + 1) * hd]);
                            rope_partial(&mut q[h * hd..(h + 1) * hd], pos, c.rope_dim(), c.rope_theta);
                        }
                        for h in 0..c.n_kv_heads {
                            let old = k[h * hd..(h + 1) * hd].to_vec();
                            rmsnorm(&old, k_norm, c.norm_eps as f32, &mut k[h * hd..(h + 1) * hd]);
                            rope_partial(&mut k[h * hd..(h + 1) * hd], pos, c.rope_dim(), c.rope_theta);
                        }
                    }
                });
            }
        });
    }
    cache.k.extend_from_slice(&k_all);
    cache.v.extend_from_slice(&v_all);
    cache.len += t_count;

    // each position attends over its causal prefix, parallel over tokens
    {
        let workers = prefill_workers(t_count);
        let groups = c.n_heads / c.n_kv_heads;
        let scale = 1.0f32 / (hd as f32).sqrt();
        std::thread::scope(|s| {
            let mut out_rest = &mut attn_out[..t_count * d];
            let cache_k = &cache.k;
            let cache_v = &cache.v;
            let q_all = &q_all;
            let gate_all = &gate_all;
            for (t0, t1) in ranges(t_count, workers) {
                let n = t1 - t0;
                let (out_c, or) = out_rest.split_at_mut(n * d);
                out_rest = or;
                s.spawn(move || {
                    for i in 0..n {
                        let t = t0 + i;
                        let window = base_pos + t + 1;
                        let mut mixed = vec![0.0f32; q_width];
                        let mut scores = vec![0.0f32; window];
                        for h in 0..c.n_heads {
                            let kh = h / groups;
                            let qh = &q_all[t * q_width + h * hd..t * q_width + (h + 1) * hd];
                            let mut max_score = f32::NEG_INFINITY;
                            for u in 0..window {
                                let off = u * kv_width + kh * hd;
                                let sc =
                                    crate::model::ops::dot(qh, &cache_k[off..off + hd]) * scale;
                                scores[u] = sc;
                                max_score = max_score.max(sc);
                            }
                            let mut denom = 0.0f32;
                            for sc in scores.iter_mut() {
                                *sc = (*sc - max_score).exp();
                                denom += *sc;
                            }
                            let dst = &mut mixed[h * hd..(h + 1) * hd];
                            for u in 0..window {
                                let off = u * kv_width + kh * hd;
                                let a = scores[u] / denom;
                                for j in 0..hd {
                                    dst[j] += a * cache_v[off + j];
                                }
                            }
                        }
                        for (j, value) in mixed.iter_mut().enumerate() {
                            *value *= 1.0 / (1.0 + (-gate_all[t * q_width + j]).exp());
                        }
                        crate::model::ops::matvec_st(o_proj, d, q_width, &mixed, &mut out_c[i * d..(i + 1) * d]);
                    }
                });
            }
        });
    }
}

/// Prefill MLP dispatch: every token is independent, so tokens fan out over
/// worker ranges and each token runs the exact single-token math (routed
/// experts sequentially inside its worker for the MoE variant).
fn mlp_prefill(
    data: &[u8],
    mlp: &QwenMlpW,
    c: &QwenConfig,
    normed: &[f32],
    t_count: usize,
    out: &mut [f32],
) {
    let d = c.d;
    let workers = prefill_workers(t_count);
    std::thread::scope(|s| {
        let mut out_rest = &mut out[..t_count * d];
        for (t0, t1) in ranges(t_count, workers) {
            let n = t1 - t0;
            let (out_c, or) = out_rest.split_at_mut(n * d);
            out_rest = or;
            let x_all = &normed[t0 * d..t1 * d];
            s.spawn(move || {
                for i in 0..n {
                    let x = &x_all[i * d..(i + 1) * d];
                    let value = match mlp {
                        QwenMlpW::Moe {
                            router,
                            experts,
                            shared,
                            shared_gate,
                        } => moe_token_serial(data, router, experts, shared, shared_gate, c, x),
                        QwenMlpW::Dense { gate, up, down } => {
                            dense_token_serial(data, gate, up, down, c, x)
                        }
                    };
                    out_c[i * d..(i + 1) * d].copy_from_slice(&value);
                }
            });
        }
    });
}

/// Single-token MoE block with the routed experts evaluated sequentially
/// (the prefill worker already owns a core). Float operations and the
/// mixing order match `packed_moe` exactly.
fn moe_token_serial(
    data: &[u8],
    router: &T,
    experts: &[[PackedT; 3]],
    shared: &[T; 3],
    shared_gate: &T,
    c: &QwenConfig,
    x: &[f32],
) -> Vec<f32> {
    let mut logits = vec![0.0f32; c.n_experts];
    crate::model::ops::matvec_st(tensor(data, router), c.n_experts, c.d, x, &mut logits);
    let selected = route_topk(&logits, c.top_k);
    let mut out = vec![0.0f32; c.d];
    let mut routed = vec![0.0f32; c.d];
    let mut gate_buf = vec![0.0f32; c.moe_inter];
    let mut up_buf = vec![0.0f32; c.moe_inter];
    for &(expert, weight) in &selected {
        let weights = &experts[expert];
        let (p1, s1) = packed_parts(data, &weights[0]);
        let (p3, s3) = packed_parts(data, &weights[2]);
        crate::quant::mxfp4::matvec_packed(p1, s1, c.moe_inter, c.d, x, &mut gate_buf, 1);
        crate::quant::mxfp4::matvec_packed(p3, s3, c.moe_inter, c.d, x, &mut up_buf, 1);
        for i in 0..c.moe_inter {
            gate_buf[i] = (gate_buf[i] / (1.0 + (-gate_buf[i]).exp())) * up_buf[i];
        }
        let (p2, s2) = packed_parts(data, &weights[1]);
        crate::quant::mxfp4::matvec_packed(p2, s2, c.d, c.moe_inter, &gate_buf, &mut routed, 1);
        for i in 0..c.d {
            out[i] += weight * routed[i];
        }
    }
    let sg = crate::model::ops::dot(tensor(data, shared_gate), x);
    let shared_scale = 1.0 / (1.0 + (-sg).exp());
    let mut shared_out = vec![0.0f32; c.d];
    ffn(
        x,
        tensor(data, &shared[0]),
        tensor(data, &shared[2]),
        tensor(data, &shared[1]),
        c.shared_inter,
        c.d,
        &mut shared_out,
    );
    for i in 0..c.d {
        out[i] += shared_scale * shared_out[i];
    }
    out
}

/// Single-token dense MLP on one worker thread (single-threaded packed
/// matvecs; identical row results to the row-parallel decode path).
fn dense_token_serial(
    data: &[u8],
    gate: &PackedT,
    up: &PackedT,
    down: &PackedT,
    c: &QwenConfig,
    x: &[f32],
) -> Vec<f32> {
    let inter = c.dense_inter;
    let mut h_gate = vec![0.0f32; inter];
    let mut h_up = vec![0.0f32; inter];
    let (pg, sg) = packed_parts(data, gate);
    crate::quant::mxfp4::matvec_packed(pg, sg, inter, c.d, x, &mut h_gate, 1);
    let (pu, su) = packed_parts(data, up);
    crate::quant::mxfp4::matvec_packed(pu, su, inter, c.d, x, &mut h_up, 1);
    for i in 0..inter {
        h_gate[i] = (h_gate[i] / (1.0 + (-h_gate[i]).exp())) * h_up[i];
    }
    let mut out = vec![0.0f32; c.d];
    let (pd, sd) = packed_parts(data, down);
    crate::quant::mxfp4::matvec_packed(pd, sd, c.d, inter, &h_gate, &mut out, 1);
    out
}

/// Shared generation loop for the Qwen runtime. The prompt is ingested in
/// one batched prefill (bit-identical to sequential forwards); decoding
/// then advances one token at a time, or two per accepted draft with
/// `--mtp` on a checkpoint converted with its multi-token-prediction head.
pub fn qwen_run_turn(
    ids: &[u32],
    max_new: usize,
    tok: &crate::tokenizer::AnyTokenizer,
    model: &mut QwenModel,
    debug: bool,
    debug_routing: bool,
    stop_id: u32,
    sampler: &mut super::Sampler,
) -> String {
    model.reset();
    if sampler.mtp {
        if !model.has_mtp() {
            eprintln!("warning: --mtp ignored, the model was converted without its MTP head");
        } else if sampler.temp > 0.0 {
            eprintln!("warning: --mtp is greedy-only, ignoring it with --temp > 0");
        } else {
            return run_turn_mtp(ids, max_new, tok, model, debug, stop_id, sampler);
        }
    }
    super::run_turn_core_batch(
        ids,
        max_new,
        tok,
        &mut |batch: &[u32]| model.prefill(batch),
        debug,
        debug_routing,
        stop_id,
        None,
        sampler,
    )
}

/// Greedy selection with the same top-5 tie-breaking as the plain loop,
/// including the --dry anti-repetition context (required for the
/// bit-identity of speculative output).
fn mtp_select(logits: &[f32], sampler: &super::Sampler, gen_ctx: &[u32]) -> u32 {
    if sampler.dry > 0.0 {
        let mut adjusted = logits.to_vec();
        super::apply_dry(&mut adjusted, gen_ctx, sampler.dry);
        return super::top_k_probs(&adjusted, 5)[0].0 as u32;
    }
    super::top_k_probs(logits, 5)[0].0 as u32
}

/// Greedy self-speculative decoding: the MTP head drafts one token ahead,
/// the trunk verifies the pending token and the draft in one two-token
/// batched prefill, and a rejected draft is rolled back (linear states
/// restored, key/value caches truncated) before the pending token is
/// re-ingested alone. Every emitted token is the greedy argmax of the same
/// logits the plain loop would produce, so the output is bit-identical.
fn run_turn_mtp(
    ids: &[u32],
    max_new: usize,
    tok: &crate::tokenizer::AnyTokenizer,
    model: &mut QwenModel,
    debug: bool,
    stop_id: u32,
    sampler: &mut super::Sampler,
) -> String {
    use std::time::Instant;
    let t_gen = Instant::now();
    let (generated, passes, accepted) = mtp_generate(model, ids, max_new, stop_id, sampler, debug);
    let gen_dt = t_gen.elapsed().as_secs_f64();
    let answer = tok.decode(&generated);
    if debug {
        println!();
        println!("answer: {}", answer);
    } else {
        println!("Bot > {}", answer);
    }
    if !generated.is_empty() {
        let moy = gen_dt / generated.len() as f64;
        println!(
            "  ({:.0} ms/token, {:.1} tok/s | mtp: {} passes, {} drafts accepted, {:.0}% acceptance)",
            moy * 1000.0,
            1.0 / moy,
            passes,
            accepted,
            if passes > 0 { accepted as f64 / passes as f64 * 100.0 } else { 0.0 }
        );
    }
    answer
}

/// Token-level MTP speculative loop (see `run_turn_mtp`). Returns the
/// generated ids plus (verification passes, accepted drafts).
fn mtp_generate(
    model: &mut QwenModel,
    ids: &[u32],
    max_new: usize,
    stop_id: u32,
    sampler: &super::Sampler,
    debug: bool,
) -> (Vec<u32>, usize, usize) {
    assert!(!ids.is_empty(), "MTP decoding requires a prompt");
    // trunk prefill + MTP prompt ingestion: draft slot i pairs the prompt
    // token at i+1 with the trunk hidden at i
    let out = model.prefill_collect(ids, false);
    let mut logits = out.logits.into_iter().next_back().unwrap();
    let mut hidden_prev = out.hidden.last().unwrap().clone();
    for i in 0..ids.len() - 1 {
        model.mtp_advance(ids[i + 1], &out.hidden[i], false);
    }

    let mut generated: Vec<u32> = Vec::new();
    let mut passes = 0usize;
    let mut accepted = 0usize;

    // first pending token + its draft
    let mut pending = false;
    let mut draft = 0u32;
    let first = mtp_select(&logits, sampler, &generated);
    if first != stop_id {
        generated.push(first);
        let dl = model.mtp_advance(first, &hidden_prev, true).unwrap();
        draft = super::top_k_probs(&dl, 5)[0].0 as u32;
        pending = true;
    }

    while pending && generated.len() < max_new {
        let n = *generated.last().unwrap();
        let snap = model.snapshot();
        let batch = [n, draft];
        let out = model.prefill_collect(&batch, true);
        passes += 1;
        let sel = mtp_select(&out.logits[0], sampler, &generated);
        if sel == stop_id {
            break;
        }
        if sel == draft {
            // draft accepted: two tokens for one batched pass
            accepted += 1;
            generated.push(draft);
            model.mtp_advance(draft, &out.hidden[0], false);
            logits = out.logits[1].clone();
            hidden_prev = out.hidden[1].clone();
            if debug {
                println!("  mtp pass {}: draft token {} accepted", passes, draft);
            }
        } else {
            // rejected: undo the draft ingestion, re-ingest the pending
            // token alone (bit-identical state), continue from `sel`
            model.restore(&snap);
            model.prefill_collect(&[n], false);
            logits = out.logits[0].clone();
            hidden_prev = out.hidden[0].clone();
            if debug {
                println!(
                    "  mtp pass {}: draft token {} rejected for {}",
                    passes, draft, sel
                );
            }
        }
        if generated.len() >= max_new {
            break;
        }
        let next = mtp_select(&logits, sampler, &generated);
        if next == stop_id {
            break;
        }
        generated.push(next);
        let dl = model.mtp_advance(next, &hidden_prev, true).unwrap();
        draft = super::top_k_probs(&dl, 5)[0].0 as u32;
        pending = true;
    }
    (generated, passes, accepted)
}

/// Hidden parity helper: writes per-token logits as
/// `QWLOGIT1 | u32 n_tokens | u32 vocab | f32 logits...`.
pub fn dump_cmd(args: &[String]) {
    let value = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_path = value("--model").expect("qwen-dump requires --model MODEL.bin");
    let tokens: Vec<u32> = value("--tokens")
        .expect("qwen-dump requires --tokens ID,ID,...")
        .split(',')
        .map(|value| value.parse().expect("qwen-dump: invalid token id"))
        .collect();
    assert!(!tokens.is_empty(), "qwen-dump: token list is empty");
    let out_path = value("--out").expect("qwen-dump requires --out LOGITS.bin");
    let mut model = QwenModel::load(&model_path);
    let mut file = std::fs::File::create(&out_path).unwrap();
    use std::io::Write;
    file.write_all(b"QWLOGIT1").unwrap();
    if args.iter().any(|arg| arg == "--mtp") {
        // draft logits for every prompt pair: slot i = (token[i+1], hidden[i])
        assert!(model.has_mtp(), "qwen-dump --mtp: model has no MTP head");
        assert!(tokens.len() >= 2, "qwen-dump --mtp needs at least two tokens");
        let out = model.prefill_collect(&tokens, false);
        file.write_all(&((tokens.len() - 1) as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(model.cfg.vocab as u32).to_le_bytes())
            .unwrap();
        for i in 0..tokens.len() - 1 {
            let hidden = out.hidden[i].clone();
            let logits = model.mtp_advance(tokens[i + 1], &hidden, true).unwrap();
            file.write_all(&crate::quant::weights::f32_to_bytes(&logits))
                .unwrap();
        }
    } else {
        file.write_all(&(tokens.len() as u32).to_le_bytes())
            .unwrap();
        file.write_all(&(model.cfg.vocab as u32).to_le_bytes())
            .unwrap();
        for token in tokens {
            let logits = model.forward(token);
            file.write_all(&crate::quant::weights::f32_to_bytes(&logits))
                .unwrap();
        }
    }
    file.sync_all().unwrap();
    println!("Qwen logits: {}", out_path);
}

#[cfg(test)]
mod model_tests {
    use super::*;

    fn bin_tiny() -> QwenConfig {
        let mut c = QwenConfig::qwen35_moe();
        c.n_layers = 4;
        c.d = 32;
        c.vocab = 64;
        c.n_heads = 2;
        c.n_kv_heads = 1;
        c.head_dim = 16;
        c.partial_rotary = 0.5;
        c.lin_k_heads = 1;
        c.lin_v_heads = 1;
        c.lin_k_dim = 32;
        c.lin_v_dim = 32;
        c.n_experts = 2;
        c.top_k = 1;
        c.moe_inter = 32;
        c.shared_inter = 32;
        c
    }

    fn checkpoint_fixture(c: &QwenConfig) -> String {
        let path = std::env::temp_dir()
            .join(format!(
                "microkimi_qwen_fixture_{}_{}.bin",
                std::process::id(),
                std::thread::current().name().unwrap_or("test")
            ))
            .to_string_lossy()
            .into_owned();
        let layout = crate::tools::convert_qwen::output_layout(c);
        let mut writer = crate::quant::weights::BinWriter::new();
        for (name, dtype, dims) in &layout {
            writer.add(name, *dtype, dims.clone());
        }
        let mut file = std::fs::File::create(&path).unwrap();
        let offsets = writer.write_header_v2(
            &mut file,
            &crate::tools::convert_qwen::config_json(c, "qwen.tokenizer.json"),
        );
        for ((name, dtype, dims), offset) in layout.iter().zip(offsets) {
            let n: usize = dims.iter().map(|&d| d as usize).product();
            let blob = if *dtype == DTYPE_MXFP4 {
                let values: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.004).collect();
                let (packed, scales) =
                    crate::quant::mxfp4::quantize(&values, dims[0] as usize, dims[1] as usize);
                [packed, scales].concat()
            } else {
                let mut values = if name.contains(".linear_attn.norm.weight") {
                    vec![1.0f32; n]
                } else if name.ends_with("norm.weight")
                    || name.ends_with("layernorm.weight")
                    || name.ends_with("A_log")
                    || name.ends_with("dt_bias")
                    || name.ends_with("shared_expert_gate.weight")
                {
                    vec![0.0f32; n]
                } else {
                    (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.002).collect()
                };
                if name.ends_with("linear_attn.conv1d.weight") {
                    values.fill(0.0);
                    for channel in 0..(n / c.conv_kernel) {
                        values[channel * c.conv_kernel + c.conv_kernel - 1] = 1.0;
                    }
                }
                crate::quant::weights::f32_to_bytes(&values)
            };
            writer.write_blob_at(&mut file, offset, &blob);
        }
        file.sync_all().unwrap();
        path
    }

    fn bin_tiny_dense() -> QwenConfig {
        let mut c = bin_tiny();
        c.n_experts = 0;
        c.top_k = 0;
        c.moe_inter = 0;
        c.shared_inter = 0;
        c.dense_inter = 64;
        c
    }

    #[test]
    fn dense_checkpoint_runtime_runs_and_resets() {
        let c = bin_tiny_dense();
        let path = checkpoint_fixture(&c);
        let mut model = QwenModel::load(&path);
        assert!(model.cfg.is_dense());
        let first = model.forward(3);
        let second = model.forward(5);
        assert_eq!(first.len(), c.vocab);
        assert!(first.iter().chain(&second).all(|v| v.is_finite()));
        assert!(first.iter().any(|v| v.abs() > 1e-9));
        model.reset();
        let replay = model.forward(3);
        assert_eq!(first, replay);
        drop(model);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn mtp_speculative_output_matches_plain_greedy() {
        let mut c = bin_tiny_dense();
        c.mtp_layers = 1;
        let path = checkpoint_fixture(&c);
        let ids = [3u32, 5, 7, 11];
        let max_new = 24;
        let stop = 999_999; // unreachable: exercise the full loop

        // plain greedy reference: same selection as the generation loop
        let mut plain_model = QwenModel::load(&path);
        let mut logits = plain_model.prefill(&ids);
        let mut plain: Vec<u32> = Vec::new();
        while plain.len() < max_new {
            let next = crate::model::top_k_probs(&logits, 5)[0].0 as u32;
            if next == stop {
                break;
            }
            plain.push(next);
            logits = plain_model.forward(next);
        }

        let mut spec_model = QwenModel::load(&path);
        assert!(spec_model.has_mtp());
        let sampler = crate::model::Sampler::greedy();
        let (spec, passes, accepted) =
            mtp_generate(&mut spec_model, &ids, max_new, stop, &sampler, false);
        assert_eq!(plain, spec, "MTP speculative output diverges from greedy");
        assert!(passes > 0);
        assert!(accepted <= passes);

        drop((plain_model, spec_model));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn mtp_rollback_restores_the_exact_state() {
        let mut c = bin_tiny_dense();
        c.mtp_layers = 1;
        let path = checkpoint_fixture(&c);
        let mut model = QwenModel::load(&path);
        let baseline = model.prefill_collect(&[3, 5, 7], false);
        let snap = model.snapshot();
        // speculative ingestion of two tokens, then rollback
        model.prefill_collect(&[11, 2], true);
        model.restore(&snap);
        // the continuation must be bit-identical to never having speculated
        let after = model.forward(11);

        let mut reference = QwenModel::load(&path);
        reference.prefill_collect(&[3, 5, 7], false);
        let expected = reference.forward(11);
        assert_eq!(after, expected);
        drop((model, reference, baseline));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn batched_prefill_is_bit_identical_to_sequential_forwards() {
        for c in [bin_tiny(), bin_tiny_dense()] {
            let path = checkpoint_fixture(&c);
            let tokens = [3u32, 5, 7, 11, 2, 9, 13];

            let mut sequential = QwenModel::load(&path);
            let mut seq_logits = Vec::new();
            for &t in &tokens {
                seq_logits.push(sequential.forward(t));
            }

            let mut batched = QwenModel::load(&path);
            let out = batched.prefill_collect(&tokens, true);
            assert_eq!(out.logits.len(), tokens.len());
            for (a, b) in seq_logits.iter().zip(&out.logits) {
                assert_eq!(a, b, "prefill logits diverge from sequential forwards");
            }
            assert_eq!(batched.pos, sequential.pos);

            // caches must be equivalent: continued decoding stays identical
            for &t in &[4u32, 6, 8] {
                assert_eq!(sequential.forward(t), batched.forward(t));
            }

            // the last-logits wrapper takes the same path
            let mut wrapper = QwenModel::load(&path);
            assert_eq!(wrapper.prefill(&tokens), *seq_logits.last().unwrap());

            drop((sequential, batched, wrapper));
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn checkpoint_runtime_executes_both_attention_kinds_and_resets() {
        let c = bin_tiny();
        let path = checkpoint_fixture(&c);
        let mut model = QwenModel::load(&path);
        let first = model.forward(3);
        let second = model.forward(5);
        assert_eq!(first.len(), c.vocab);
        assert!(first.iter().chain(&second).all(|v| v.is_finite()));
        assert!(first.iter().any(|v| v.abs() > 1e-9));
        match &model.caches[3] {
            QwenCache::Full(cache) => assert_eq!(cache.len, 2),
            _ => panic!("layer 3 must have a full-attention cache"),
        }
        model.reset();
        let replay = model.forward(3);
        assert_eq!(first, replay);
        drop(model);
        std::fs::remove_file(path).ok();
    }
}
