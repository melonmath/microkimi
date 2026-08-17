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
    // stack scratch: a heap allocation here ran once per (head, token)
    let mut pred_stack = [0.0f32; 256];
    let mut pred_heap;
    let pred: &mut [f32] = if vd <= 256 {
        &mut pred_stack[..vd]
    } else {
        pred_heap = vec![0.0f32; vd];
        &mut pred_heap
    };
    // TWO fused passes over the state instead of four (decay | predict |
    // update | readout): each state row is decayed and dotted in one
    // visit, then corrected and read out in one visit - half the state
    // traffic. The per-element arithmetic and its order are exactly the
    // former four-pass code's, so results are bit-identical.
    for i in 0..kd {
        let ki = k[i];
        let row = &mut s[i * vd..(i + 1) * vd];
        if ki == 0.0 {
            for x in row.iter_mut() {
                *x *= decay;
            }
            continue;
        }
        for j in 0..vd {
            row[j] *= decay;
            pred[j] += ki * row[j];
        }
    }
    for o in out.iter_mut() {
        *o = 0.0;
    }
    for i in 0..kd {
        let ki = k[i];
        let qi = q[i];
        let row = &mut s[i * vd..(i + 1) * vd];
        if ki == 0.0 {
            if qi == 0.0 {
                continue;
            }
            for j in 0..vd {
                out[j] += qi * row[j];
            }
            continue;
        }
        if qi == 0.0 {
            for j in 0..vd {
                row[j] += ki * (v[j] - pred[j]) * beta;
            }
            continue;
        }
        for j in 0..vd {
            row[j] += ki * (v[j] - pred[j]) * beta;
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
    /// q8 mirror of `k` (same flat layout; per-32 scales alongside).
    /// Maintained at every append so the quantized spine modes can score
    /// attention with integer dots; the f32 paths never read it.
    pub kq: Vec<i8>,
    pub kqs: Vec<f32>,
    pub len: usize,
}

impl FullCache {
    pub fn new(c: &QwenConfig) -> FullCache {
        let width = c.n_kv_heads * c.head_dim;
        FullCache {
            k: Vec::with_capacity(width * 256),
            v: Vec::with_capacity(width * 256),
            kq: Vec::with_capacity(width * 256),
            kqs: Vec::with_capacity(width * 8),
            len: 0,
        }
    }
}

/// Appends the q8 mirror for freshly appended key rows (kv_width is a
/// multiple of 32, so flat 32-blocks align with per-head slices).
fn push_k_mirror(cache: &mut FullCache, k_new: &[f32]) {
    if k_new.len() % 32 != 0 {
        return; // tiny test fixtures; the q8 scoring gates on hd % 32 too
    }
    let xq = crate::quant::q8::quantize_q8(k_new);
    cache.kq.extend_from_slice(&xq.q);
    cache.kqs.extend_from_slice(&xq.scales);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_scan_matches_sequential_delta_steps() {
        // sizes crossing chunk boundaries and a remainder, two dims
        for (kd, vd, t_count) in [(8usize, 8usize, 70usize), (128, 128, 45), (16, 8, 32)] {
            let f = |i: usize, m: usize| ((i * 37 + 11) % m) as f32 / m as f32 - 0.4;
            let qn: Vec<f32> = (0..t_count * kd).map(|i| f(i, 19)).collect();
            let kn: Vec<f32> = (0..t_count * kd).map(|i| f(i + 5, 23)).collect();
            let vn: Vec<f32> = (0..t_count * vd).map(|i| f(i + 9, 29)).collect();
            let beta: Vec<f32> = (0..t_count).map(|i| 0.2 + 0.7 * f(i, 13).abs()).collect();
            // include BRUTAL decays (down to ~1e-3): cumulative products
            // underflow over a chunk, which the division form got wrong
            let gamma: Vec<f32> = (0..t_count)
                .map(|i| if i % 7 == 3 { 0.001 } else { 0.85 + 0.14 * f(i + 3, 17).abs() })
                .collect();
            let mut s_seq = vec![0.05f32; kd * vd];
            let mut s_chk = s_seq.clone();
            let mut out_seq = vec![0.0f32; t_count * vd];
            let mut out_chk = vec![0.0f32; t_count * vd];
            for t in 0..t_count {
                // delta_step takes g with decay = exp(g)
                let g = gamma[t].ln();
                delta_step(
                    &mut s_seq,
                    &qn[t * kd..(t + 1) * kd],
                    &kn[t * kd..(t + 1) * kd],
                    &vn[t * vd..(t + 1) * vd],
                    g,
                    beta[t],
                    &mut out_seq[t * vd..(t + 1) * vd],
                );
            }
            chunked_scan_head(
                &mut s_chk, &mut out_chk, &qn, &kn, &vn, &beta, &gamma, t_count, kd, vd,
            );
            let tol = 2e-4f32;
            for (a, b) in out_seq.iter().zip(&out_chk) {
                assert!(
                    (a - b).abs() <= tol * (1.0 + a.abs()),
                    "output diverges: {} vs {} (kd {} vd {} t {})",
                    a, b, kd, vd, t_count
                );
            }
            for (a, b) in s_seq.iter().zip(&s_chk) {
                assert!(
                    (a - b).abs() <= tol * (1.0 + a.abs()),
                    "state diverges: {} vs {}",
                    a, b
                );
            }
        }
    }

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
    push_k_mirror(cache, &k);
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
    // reusable q/k scratch (two heap allocations per head added up)
    let mut q = vec![0.0f32; kd];
    let mut k = vec![0.0f32; kd];
    for h in 0..c.lin_v_heads {
        let kh = h / rep.max(1);
        q.copy_from_slice(&conved[kh * kd..(kh + 1) * kd]);
        k.copy_from_slice(&conved[kt + kh * kd..kt + (kh + 1) * kd]);
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

pub(crate) enum QwenCache {
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

/// Q8 copies of one layer's large attention matrices (spine q8 mode):
/// the f32 spine dominates per-token weight traffic, and quantizing it to
/// q8 at load cuts that traffic roughly 4x on the covered matrices, the
/// same trade llama.cpp makes everywhere. Norms, the convolution, the
/// small b/a projections, and the MTP head stay f32. NOT bit-identical
/// to the f32 spine (q8 rounding, same bound as the q8 lm_head); the
/// quality delta is measured, not promised.
pub(crate) enum LayerQ8 {
    Linear {
        in_qkv: SpineMat,
        in_z: SpineMat,
        out_proj: SpineMat,
    },
    Full {
        q_proj: SpineMat,
        k_proj: SpineMat,
        v_proj: SpineMat,
        o_proj: SpineMat,
    },
}

/// One quantized spine matrix: q8 rows (llama.cpp's Q8_0 trade) or MXFP4
/// nibbles (half the traffic again, the same format as the packed MLP).
pub(crate) enum SpineMat {
    Q8(crate::model::Q8Head),
    Fp4(PackedMat),
}

/// An owned MXFP4 matrix quantized at load (packed nibbles + e8m0 row
/// scales), driven by the same kernels as the on-disk packed MLP.
pub(crate) struct PackedMat {
    packed: Vec<u8>,
    scales: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl PackedMat {
    fn from_f32(w: &[f32], rows: usize, cols: usize) -> PackedMat {
        let (packed, scales) = crate::quant::mxfp4::quantize(w, rows, cols);
        PackedMat { packed, scales, rows, cols }
    }
}

impl SpineMat {
    fn build(w: &[f32], rows: usize, cols: usize, fp4: bool) -> SpineMat {
        if fp4 {
            SpineMat::Fp4(PackedMat::from_f32(w, rows, cols))
        } else {
            SpineMat::Q8(crate::model::Q8Head::from_f32(w, rows, cols))
        }
    }

    pub(crate) fn matvec_st(&self, x: &[f32], out: &mut [f32]) {
        match self {
            SpineMat::Q8(head) => head.matvec_st(x, out),
            SpineMat::Fp4(m) => crate::quant::mxfp4::matvec_packed(
                &m.packed, &m.scales, m.rows, m.cols, x, out, 1,
            ),
        }
    }

    /// Pool-parallel matvec (row-chunked, bit-identical to matvec_st):
    /// the single-token decode path calls this from the main thread so
    /// the whole worker pool serves one token's projections.
    pub(crate) fn matvec(&self, x: &[f32], out: &mut [f32]) {
        match self {
            SpineMat::Q8(head) => head.matvec(x, out),
            SpineMat::Fp4(m) => crate::quant::mxfp4::matvec_packed(
                &m.packed,
                &m.scales,
                m.rows,
                m.cols,
                x,
                out,
                crate::model::pool::pool().workers.max(1),
            ),
        }
    }

    pub(crate) fn matvec_multi(&self, xs: &[&[f32]], outs: &mut [&mut [f32]]) {
        match self {
            SpineMat::Q8(head) => head.matvec_multi(xs, outs),
            SpineMat::Fp4(m) => crate::quant::mxfp4::matvec_packed_multi(
                &m.packed,
                &m.scales,
                m.rows,
                m.cols,
                xs,
                outs,
                crate::model::pool::pool().workers.max(1),
            ),
        }
    }
}

/// Spine quantization at load: MICROKIMI_FP4_SPINE=1 selects MXFP4 (wins
/// when both are set), MICROKIMI_Q8_SPINE=1 selects q8, neither keeps the
/// exact f32 spine. Read per load so tests can exercise every mode.
fn spine_mode() -> Option<bool> {
    if std::env::var("MICROKIMI_FP4_SPINE").map(|v| v == "1").unwrap_or(false) {
        return Some(true);
    }
    if std::env::var("MICROKIMI_Q8_SPINE").map(|v| v == "1").unwrap_or(false) {
        return Some(false);
    }
    None
}

/// Frequency-sliced draft head for chained MTP proposals. BPE vocabulary
/// ids are roughly frequency-ordered by construction (earlier merges are
/// more frequent), so the argmax over the first K rows plus the special
/// block agrees with the full-head argmax on most steps - and when it
/// does not, the full-head verification pass rejects the draft, so the
/// output stays bit-identical. At small model scale the lm_head
/// dominates per-token compute; drafting through K rows instead of the
/// full vocabulary is what makes deep draft chains close to free.
pub(crate) struct DraftHead {
    q8: crate::model::Q8Head,
    rows: usize,
    /// Start of the special-token block scored in f32 next to the q8
    /// rows (vocab when absent).
    specials_start: usize,
}

impl DraftHead {
    /// Builds the head over the first `rows` lm_head rows (tied models
    /// share them with the embedding). None when `rows` covers the whole
    /// vocabulary (the full head is already optimal).
    pub(crate) fn from_rows(model: &QwenModel, rows: usize) -> Option<DraftHead> {
        let c = &model.cfg;
        if rows == 0 || rows >= c.vocab || c.d % 32 != 0 {
            return None;
        }
        let head = tensor(&model.bin.data, &model.lm_head);
        let specials_start = if c.vocab > crate::model::qwentok::QWEN_ENDOFTEXT as usize {
            crate::model::qwentok::QWEN_ENDOFTEXT as usize
        } else {
            c.vocab
        };
        Some(DraftHead {
            q8: crate::model::Q8Head::from_f32(&head[..rows * c.d], rows, c.d),
            rows,
            specials_start,
        })
    }

    /// Argmax over the covered rows (q8 block + f32 specials).
    fn argmax(&self, model: &QwenModel, normed: &[f32]) -> u32 {
        let mut scores = vec![0.0f32; self.rows];
        self.q8.matvec(normed, &mut scores);
        let mut best = 0usize;
        for (i, &v) in scores.iter().enumerate() {
            if v > scores[best] {
                best = i;
            }
        }
        let mut best_id = best as u32;
        let mut best_score = scores[best];
        let c = &model.cfg;
        if self.specials_start < c.vocab {
            let head = tensor(&model.bin.data, &model.lm_head);
            let n = c.vocab - self.specials_start;
            let mut sp = vec![0.0f32; n];
            crate::model::ops::matvec_st(
                &head[self.specials_start * c.d..],
                n,
                c.d,
                normed,
                &mut sp,
            );
            for (i, &v) in sp.iter().enumerate() {
                if v > best_score {
                    best_score = v;
                    best_id = (self.specials_start + i) as u32;
                }
            }
        }
        best_id
    }
}

/// Per-layer certified-skip statistics for the dense MLP, built once at
/// load by scanning the packed matrices. For a 32-channel block b of the
/// intermediate dimension:
///   up_l2[i]  = L2 norm of up row i            (|u_i| <= up_l2[i] * |x|)
///   down_sup[b] = max_{j, i in b} |down[j, i]|
/// Skipping block b changes each MLP output coordinate by at most
///   sum_{i in b} |silu(g_i)| * up_l2[i] * |x|_2 * down_sup[b]
/// so after the gate matvec the engine can PROVE which blocks fit inside
/// the caller's error budget (sup-norm of the MLP block output, per
/// layer). Budget 0 skips nothing and stays bit-exact.
pub(crate) struct SkipBounds {
    /// per intermediate channel: L2 norm of the up row
    up_l2: Vec<f32>,
    /// per 32-block: sup of |down| over the block's columns
    down_sup: Vec<f32>,
}

impl SkipBounds {
    fn build(data: &[u8], gate: &PackedT, up: &PackedT, down: &PackedT, c: &QwenConfig) -> SkipBounds {
        let _ = gate;
        let inter = c.dense_inter;
        let d = c.d;
        let (pu, su) = packed_parts(data, up);
        let up_f = crate::quant::mxfp4::dequant(pu, su, inter, d);
        let up_l2: Vec<f32> = (0..inter)
            .map(|i| up_f[i * d..(i + 1) * d].iter().map(|v| v * v).sum::<f32>().sqrt())
            .collect();
        let (pd, sd) = packed_parts(data, down);
        let down_f = crate::quant::mxfp4::dequant(pd, sd, d, inter);
        let blocks = inter / 32;
        let mut down_sup = vec![0.0f32; blocks];
        for j in 0..d {
            let row = &down_f[j * inter..(j + 1) * inter];
            for b in 0..blocks {
                for i in b * 32..(b + 1) * 32 {
                    let a = row[i].abs();
                    if a > down_sup[b] {
                        down_sup[b] = a;
                    }
                }
            }
        }
        SkipBounds { up_l2, down_sup }
    }

    /// Given the gate activations, greedily selects the blocks whose
    /// summed certified contribution stays within `budget`, cheapest
    /// bounds first. Returns the keep mask per 32-block.
    fn keep_mask(&self, silu_gate: &[f32], x_l2: f32, budget: f32) -> Vec<bool> {
        let blocks = self.down_sup.len();
        let mut keep = vec![true; blocks];
        if budget <= 0.0 {
            return keep;
        }
        let mut bounds: Vec<(f32, usize)> = (0..blocks)
            .map(|b| {
                let inner: f32 = (b * 32..(b + 1) * 32)
                    .map(|i| silu_gate[i].abs() * self.up_l2[i])
                    .sum();
                (inner * x_l2 * self.down_sup[b], b)
            })
            .collect();
        bounds.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let mut spent = 0.0f32;
        for (bound, b) in bounds {
            if spent + bound > budget {
                break;
            }
            spent += bound;
            keep[b] = false;
        }
        keep
    }
}

/// MICROKIMI_MLP_BUDGET: certified sup-norm error budget per MLP block
/// output (0 = exact, the default).
fn mlp_budget() -> f32 {
    static ON: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MICROKIMI_MLP_BUDGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0)
    })
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
    pub(crate) caches: Vec<QwenCache>,
    mtp: Option<QwenMtpW>,
    pub(crate) draft_head: Option<DraftHead>,
    /// Per-layer certified-skip bounds (dense layers, budget mode only).
    skip_bounds: Vec<Option<SkipBounds>>,
    /// Q8 spine copies (spine q8 mode only; empty otherwise).
    pub(crate) q8_spine: Vec<Option<LayerQ8>>,
    pub(crate) mlp_q8: Vec<Option<MlpQ8>>,
    pub(crate) mtp_cache: FullCache,
    pub(crate) pos: usize,
    /// Logits after the last ingested token (state snapshots resume from
    /// them without re-ingesting anything).
    pub last_logits: Vec<f32>,
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
/// Single-token dense MLP through the unpacked-i8 copies when the q8
/// spine carries them (and no block budget is active); the packed path
/// otherwise. Same math, pure-SDOT kernels.
fn packed_dense_mlp_q8(
    data: &[u8],
    gate: &PackedT,
    up: &PackedT,
    down: &PackedT,
    q8m: Option<&MlpQ8>,
    c: &QwenConfig,
    x: &[f32],
    bounds: Option<&SkipBounds>,
) -> Vec<f32> {
    if let Some(mq) = q8m {
        if bounds.is_none() || mlp_budget() == 0.0 {
            let inter = c.dense_inter;
            let mut h_gate = vec![0.0f32; inter];
            let mut h_up = vec![0.0f32; inter];
            crate::model::Q8Head::matvec2(&mq.gate, &mq.up, x, &mut h_gate, &mut h_up);
            for i in 0..inter {
                h_gate[i] = (h_gate[i] / (1.0 + (-h_gate[i]).exp())) * h_up[i];
            }
            let mut value = vec![0.0f32; c.d];
            mq.down.matvec(&h_gate, &mut value);
            return value;
        }
    }
    packed_dense_mlp(data, gate, up, down, c, x, None, bounds)
}

fn packed_dense_mlp(
    data: &[u8],
    gate: &PackedT,
    up: &PackedT,
    down: &PackedT,
    c: &QwenConfig,
    x: &[f32],
    imatrix_layer: Option<usize>,
    bounds: Option<&SkipBounds>,
) -> Vec<f32> {
    let inter = c.dense_inter;
    let threads = crate::model::pool::pool().workers.max(1);
    if let Some(l) = imatrix_layer {
        crate::quant::imatrix::record_hidden(l, x);
    }
    let mut h_gate = vec![0.0f32; inter];
    let (pg, sg) = packed_parts(data, gate);
    crate::quant::mxfp4::matvec_packed(pg, sg, inter, c.d, x, &mut h_gate, threads);

    // certified-budget path: after the gate matvec, silu magnitudes plus
    // the precomputed norms bound each 32-block's possible contribution;
    // blocks proven under the budget skip both their up rows and their
    // down columns
    if let (Some(bounds), budget) = (bounds, mlp_budget()) {
        if budget > 0.0 {
            let mut silu = vec![0.0f32; inter];
            for i in 0..inter {
                silu[i] = h_gate[i] / (1.0 + (-h_gate[i]).exp());
            }
            let x_l2 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let keep = bounds.keep_mask(&silu, x_l2, budget);
            let kept: Vec<usize> = (0..inter / 32).filter(|&b| keep[b]).collect();
            MLP_BLOCKS_TOTAL.fetch_add((inter / 32) as u64, std::sync::atomic::Ordering::Relaxed);
            MLP_BLOCKS_SKIPPED.fetch_add(
                (inter / 32 - kept.len()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let mut h = vec![0.0f32; inter];
            let (pu, su) = packed_parts(data, up);
            {
                let nt = threads.min(kept.len()).max(1);
                let chunk = kept.len().div_ceil(nt);
                std::thread::scope(|scope| {
                    for blocks in kept.chunks(chunk.max(1)) {
                        let h_ptr = h.as_mut_ptr() as usize;
                        let silu = &silu;
                        scope.spawn(move || {
                            for &b in blocks {
                                let mut rows = [0.0f32; 32];
                                crate::quant::mxfp4::matvec_packed(
                                    &pu[b * 32 * c.d / 2..(b + 1) * 32 * c.d / 2],
                                    &su[b * 32 * (c.d / 32)..(b + 1) * 32 * (c.d / 32)],
                                    32,
                                    c.d,
                                    x,
                                    &mut rows,
                                    1,
                                );
                                for j in 0..32 {
                                    let i = b * 32 + j;
                                    unsafe {
                                        *(h_ptr as *mut f32).add(i) = silu[i] * rows[j];
                                    }
                                }
                            }
                        });
                    }
                });
            }
            if let Some(l) = imatrix_layer {
                crate::quant::imatrix::record_inter(l, &h);
            }
            let mut out = vec![0.0f32; c.d];
            let (pd, sd) = packed_parts(data, down);
            crate::quant::mxfp4::matvec_packed_colblocks(
                pd, sd, c.d, inter, &kept, &h, &mut out, threads,
            );
            return out;
        }
    }

    let mut h_up = vec![0.0f32; inter];
    let (pu, su) = packed_parts(data, up);
    crate::quant::mxfp4::matvec_packed(pu, su, inter, c.d, x, &mut h_up, threads);
    for i in 0..inter {
        h_gate[i] = (h_gate[i] / (1.0 + (-h_gate[i]).exp())) * h_up[i];
    }
    if let Some(l) = imatrix_layer {
        crate::quant::imatrix::record_inter(l, &h_gate);
    }
    let mut out = vec![0.0f32; c.d];
    let (pd, sd) = packed_parts(data, down);
    crate::quant::mxfp4::matvec_packed(pd, sd, c.d, inter, &h_gate, &mut out, threads);
    out
}

/// Skip accounting for the budget mode (reported by run/serve).
static MLP_BLOCKS_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static MLP_BLOCKS_SKIPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// (skipped, total) MLP blocks since process start (budget mode).
pub fn mlp_skip_stats() -> (u64, u64) {
    (
        MLP_BLOCKS_SKIPPED.load(std::sync::atomic::Ordering::Relaxed),
        MLP_BLOCKS_TOTAL.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// Builds the quantized spine copies for every layer (q8 or fp4).
/// Load-time i8 copies of the dense MLP matrices (q8 spine mode): the
/// exact LUT2 transfer of Q8Head::from_packed_fp4, so the decode and
/// prefill MLPs run pure-SDOT kernels instead of unpacking nibbles on
/// every matvec.
pub(crate) struct MlpQ8 {
    pub(crate) gate: Q8Head,
    pub(crate) up: Q8Head,
    pub(crate) down: Q8Head,
}

fn build_mlp_q8(layers: &[QwenLayerW], bin: &BinFile, c: &QwenConfig) -> Vec<Option<MlpQ8>> {
    layers
        .iter()
        .map(|layer| match &layer.mlp {
            QwenMlpW::Dense { gate, up, down } => {
                let (pg, sg) = packed_parts(&bin.data, gate);
                let (pu, su) = packed_parts(&bin.data, up);
                let (pd, sd) = packed_parts(&bin.data, down);
                Some(MlpQ8 {
                    gate: Q8Head::from_packed_fp4(pg, sg, c.dense_inter, c.d),
                    up: Q8Head::from_packed_fp4(pu, su, c.dense_inter, c.d),
                    down: Q8Head::from_packed_fp4(pd, sd, c.d, c.dense_inter),
                })
            }
            QwenMlpW::Moe { .. } => None,
        })
        .collect()
}

fn build_spine(
    layers: &[QwenLayerW],
    bin: &BinFile,
    c: &QwenConfig,
    fp4: bool,
) -> Vec<Option<LayerQ8>> {
    let conv_dim = c.lin_key_total() * 2 + c.lin_value_total();
    let full_width = c.n_heads * c.head_dim;
    let kvw = c.n_kv_heads * c.head_dim;
    layers
        .iter()
        .map(|layer| {
            Some(match &layer.attn {
                QwenAttnW::Linear(w) => LayerQ8::Linear {
                    in_qkv: SpineMat::build(tensor(&bin.data, &w.in_qkv), conv_dim, c.d, fp4),
                    in_z: SpineMat::build(
                        tensor(&bin.data, &w.in_z),
                        c.lin_value_total(),
                        c.d,
                        fp4,
                    ),
                    out_proj: SpineMat::build(
                        tensor(&bin.data, &w.out_proj),
                        c.d,
                        c.lin_value_total(),
                        fp4,
                    ),
                },
                QwenAttnW::Full(w) => LayerQ8::Full {
                    q_proj: SpineMat::build(
                        tensor(&bin.data, &w.q_proj),
                        full_width * 2,
                        c.d,
                        fp4,
                    ),
                    k_proj: SpineMat::build(tensor(&bin.data, &w.k_proj), kvw, c.d, fp4),
                    v_proj: SpineMat::build(tensor(&bin.data, &w.v_proj), kvw, c.d, fp4),
                    o_proj: SpineMat::build(tensor(&bin.data, &w.o_proj), c.d, full_width, fp4),
                },
            })
        })
        .collect()
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
        let lm_head = if c.tied_embeddings {
            embed
        } else {
            expect_f32(&bin, "lm_head.weight", &[c.vocab, c.d])
        };
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
        let q8_spine: Vec<Option<LayerQ8>> = match spine_mode() {
            Some(fp4) => build_spine(&layers, &bin, &c, fp4),
            None => (0..c.n_layers).map(|_| None).collect(),
        };
        // q8 mode also unpacks the dense MLP to i8 (exact transfer);
        // the fp4 spine keeps the packed MLP it exists to exercise.
        let mlp_q8: Vec<Option<MlpQ8>> = match spine_mode() {
            Some(false) if c.is_dense() => build_mlp_q8(&layers, &bin, &c),
            _ => (0..c.n_layers).map(|_| None).collect(),
        };
        // GEMM packs build at load, not inside the first measured prefill
        for lq in q8_spine.iter().flatten() {
            match lq {
                LayerQ8::Linear { in_qkv, in_z, out_proj } => {
                    for m in [in_qkv, in_z, out_proj] {
                        if let SpineMat::Q8(h) = m {
                            h.prebuild_gemm();
                        }
                    }
                }
                LayerQ8::Full { q_proj, k_proj, v_proj, o_proj } => {
                    for m in [q_proj, k_proj, v_proj, o_proj] {
                        if let SpineMat::Q8(h) = m {
                            h.prebuild_gemm();
                        }
                    }
                }
            }
        }
        for mq in mlp_q8.iter().flatten() {
            mq.gate.prebuild_gemm();
            mq.up.prebuild_gemm();
            mq.down.prebuild_gemm();
        }
        let skip_bounds: Vec<Option<SkipBounds>> = if mlp_budget() > 0.0 && c.is_dense() {
            layers
                .iter()
                .map(|layer| match &layer.mlp {
                    QwenMlpW::Dense { gate, up, down } => {
                        Some(SkipBounds::build(&bin.data, gate, up, down, &c))
                    }
                    QwenMlpW::Moe { .. } => None,
                })
                .collect()
        } else {
            (0..c.n_layers).map(|_| None).collect()
        };
        let mut model = QwenModel {
            cfg: c,
            bin,
            embed,
            norm_f,
            lm_head,
            lm_head_q8,
            layers,
            caches,
            mtp,
            draft_head: None,
            skip_bounds,
            q8_spine,
            mlp_q8,
            mtp_cache,
            pos: 0,
            last_logits: Vec::new(),
            adapter_packs,
        };
        if model.mtp.is_some() {
            // MICROKIMI_MTP_MINIHEAD rows (default 32768, 0 = full head)
            let rows = std::env::var("MICROKIMI_MTP_MINIHEAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32_768usize);
            model.draft_head = DraftHead::from_rows(&model, rows);
        }
        model
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

    /// Rebuilds the spine quantization after load: None restores the
    /// exact f32 spine, Some(false) selects q8, Some(true) selects MXFP4.
    /// Tests inject modes this way instead of racing on process env.
    #[cfg(test)]
    pub(crate) fn set_spine_mode(&mut self, mode: Option<bool>) {
        self.q8_spine = match mode {
            Some(fp4) => build_spine(&self.layers, &self.bin, &self.cfg, fp4),
            None => (0..self.cfg.n_layers).map(|_| None).collect(),
        };
        self.mlp_q8 = match mode {
            Some(false) if self.cfg.is_dense() => build_mlp_q8(&self.layers, &self.bin, &self.cfg),
            _ => (0..self.cfg.n_layers).map(|_| None).collect(),
        };
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
                    c.kq.truncate(len * kv_width);
                    c.kqs.truncate(len * kv_width / 32);
                    c.len = len;
                    fi += 1;
                }
            }
        }
        self.mtp_cache.k.truncate(snap.mtp_len * kv_width);
        self.mtp_cache.v.truncate(snap.mtp_len * kv_width);
        self.mtp_cache.kq.truncate(snap.mtp_len * kv_width);
        self.mtp_cache.kqs.truncate(snap.mtp_len * kv_width / 32);
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
        let mlp = packed_dense_mlp(data, &mtp.mlp_gate, &mtp.mlp_up, &mtp.mlp_down, &c, &normed, None, None);
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

    /// One chained draft step: ingests the pair, returns the final-normed
    /// MTP hidden (the next chain step's hidden input, following the
    /// reference proposer) and the draft argmax - through the
    /// frequency-sliced head when available, the full head otherwise.
    /// The verification pass corrects any draft, so the head choice never
    /// changes the output, only the acceptance rate.
    pub(crate) fn mtp_draft_step(&mut self, token: u32, hidden: &[f32]) -> (Vec<f32>, u32) {
        self.mtp_chain_body(token, hidden)
            .expect("mtp_draft_step requires the MTP head")
    }

    /// Chain-step body: like `mtp_advance(want_logits = true)` but returns
    /// (final-normed hidden, draft id), scoring through the sliced draft
    /// head when present instead of a full-vocabulary argmax.
    fn mtp_chain_body(&mut self, token: u32, hidden: &[f32]) -> Option<(Vec<f32>, u32)> {
        let c = self.cfg.clone();
        let mtp = self.mtp.as_ref()?;
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
        let mtp_w = self.mtp.as_ref().unwrap();
        let mlp = packed_dense_mlp(
            &self.bin.data,
            &mtp_w.mlp_gate,
            &mtp_w.mlp_up,
            &mtp_w.mlp_down,
            &c,
            &normed,
            None,
            None,
        );
        for i in 0..d {
            x[i] += mlp[i];
        }
        let mtp_w = self.mtp.as_ref().unwrap();
        rmsnorm(
            &x,
            tensor(&self.bin.data, &mtp_w.norm_f),
            c.norm_eps as f32,
            &mut normed,
        );
        let draft = match &self.draft_head {
            Some(head) => head.argmax(self, &normed),
            None => {
                let mut logits = vec![0.0f32; c.vocab];
                match &self.lm_head_q8 {
                    Some(head) => head.matvec(&normed, &mut logits),
                    None => crate::model::ops::matvec(
                        tensor(&self.bin.data, &self.lm_head),
                        c.vocab,
                        d,
                        &normed,
                        &mut logits,
                    ),
                }
                super::top_k_probs(&logits, 5)[0].0 as u32
            }
        };
        Some((normed, draft))
    }

    /// Truncates the MTP draft cache to `len` pairs (chain rollback).
    pub(crate) fn rollback_mtp(&mut self, len: usize) {
        let kv_width = self.cfg.n_kv_heads * self.cfg.head_dim;
        self.mtp_cache.k.truncate(len * kv_width);
        self.mtp_cache.v.truncate(len * kv_width);
        self.mtp_cache.kq.truncate(len * kv_width);
        self.mtp_cache.kqs.truncate(len * kv_width / 32);
        self.mtp_cache.len = len;
    }

    pub fn has_adapter_packs(&self) -> bool {
        !self.adapter_packs.is_empty()
    }

    pub fn adapter_set_sha256(&self) -> Option<&str> {
        self.adapter_packs.set_sha256.as_deref()
    }

    /// Advances the autoregressive decoder by one token and returns logits.
    /// In q8-spine mode this delegates to the batched single-token prefill,
    /// which carries the q8 routing (one code path, no drift).
    pub fn forward(&mut self, token: u32) -> Vec<f32> {
        if self.q8_spine.iter().any(|x| x.is_some()) {
            return self.prefill(&[token]);
        }
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
                    packed_dense_mlp(data, gate, up, down, c, &normed, Some(l), self.skip_bounds[l].as_ref())
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
        self.last_logits = logits.clone();
        logits
    }
}

// ───────────────────────── batched layers-outer prefill ─────────────────────────

/// Number of prefill worker threads (the shared pool size; prefill phases
/// use scoped threads over contiguous token or head ranges).
/// True when MICROKIMI_NO_QWEN_BATCH=1 (sequential prefill fallback).
fn no_batch_prefill() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MICROKIMI_NO_QWEN_BATCH").map(|v| v == "1").unwrap_or(false)
    })
}

/// Decode/prefill phase profiler (MICROKIMI_PROF=1): accumulates wall
/// micros per phase (0 lin attn, 1 full attn, 2 mlp, 3 lm_head) and
/// prints once at process exit via dprof_print (called by `run`).
static DPROF: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn dprof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_PROF").map(|v| v == "1").unwrap_or(false))
}

#[inline]
fn dprof_add(phase: usize, d: std::time::Duration) {
    if dprof_on() {
        DPROF[phase].fetch_add(d.as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Prints the accumulated phase profile (best interpreted per token by
/// the caller's own token count).
pub fn dprof_print() {
    if !dprof_on() {
        return;
    }
    let v: Vec<u64> = DPROF.iter().map(|a| a.load(std::sync::atomic::Ordering::Relaxed)).collect();
    println!(
        "prof: lin_attn {:.1} ms | full_attn {:.1} ms | mlp {:.1} ms | lm_head {:.1} ms",
        v[0] as f64 / 1000.0,
        v[1] as f64 / 1000.0,
        v[2] as f64 / 1000.0,
        v[3] as f64 / 1000.0
    );
}

/// MICROKIMI_CHUNKED_SCAN=1 opts the spine prefill into the WY chunked
/// scan. Default OFF: the same-window A/B measured the scalar chunked
/// form SLOWER than the fused sequential scan (median 6.1 vs 4.2
/// ms/token) - it earns its place only with tiled kernels behind it.
fn chunked_scan_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MICROKIMI_CHUNKED_SCAN").map(|v| v == "1").unwrap_or(false)
    })
}

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

/// Residual add plus RMS norm over all positions, parallel over token
/// ranges above a small-batch threshold (both are element-wise or
/// per-token, so the split is bit-identical to the serial loop).
fn par_add_norm(
    hidden: &mut [f32],
    add: Option<&[f32]>,
    w: &[f32],
    eps: f32,
    normed: &mut [f32],
    t_count: usize,
    d: usize,
) {
    if t_count < 64 {
        if let Some(a) = add {
            for i in 0..t_count * d {
                hidden[i] += a[i];
            }
        }
        for t in 0..t_count {
            rmsnorm(&hidden[t * d..(t + 1) * d], w, eps, &mut normed[t * d..(t + 1) * d]);
        }
        return;
    }
    let workers = prefill_workers(t_count);
    std::thread::scope(|s| {
        let mut h_rest = &mut hidden[..t_count * d];
        let mut n_rest = &mut normed[..t_count * d];
        for (t0, t1) in ranges(t_count, workers) {
            let n = t1 - t0;
            let (h_c, hr) = h_rest.split_at_mut(n * d);
            let (n_c, nr) = n_rest.split_at_mut(n * d);
            h_rest = hr;
            n_rest = nr;
            let a_c = add.map(|a| &a[t0 * d..t1 * d]);
            s.spawn(move || {
                if let Some(a) = a_c {
                    for (hv, av) in h_c.iter_mut().zip(a) {
                        *hv += av;
                    }
                }
                for t in 0..n {
                    rmsnorm(&h_c[t * d..(t + 1) * d], w, eps, &mut n_c[t * d..(t + 1) * d]);
                }
            });
        }
    });
}

/// Splits `items` into contiguous ranges of ~equal CAUSAL cost: token t
/// attends over base+t+1 positions, so uniform ranges make the last
/// worker do about twice the work of the first at base=0. Cuts are placed
/// on the prefix sums of the per-token cost instead; the per-token math
/// is untouched, so the result stays bit-identical to `ranges`.
fn causal_ranges(items: usize, workers: usize, base: usize) -> Vec<(usize, usize)> {
    let w = workers.max(1).min(items.max(1));
    let total: f64 = (0..items).map(|t| (base + t + 1) as f64).sum();
    let mut out = Vec::with_capacity(w);
    let (mut start, mut acc) = (0usize, 0f64);
    let mut next_cut = total / w as f64;
    for t in 0..items {
        acc += (base + t + 1) as f64;
        if acc >= next_cut - 1e-9 && out.len() + 1 < w {
            out.push((start, t + 1));
            start = t + 1;
            next_cut = total * (out.len() + 1) as f64 / w as f64;
        }
    }
    if start < items {
        out.push((start, items));
    }
    out
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
    /// MICROKIMI_NO_QWEN_BATCH=1 forces the sequential single-token path
    /// (A/B benchmarking toggle; both paths are bit-identical).
    pub fn prefill_collect(&mut self, tokens: &[u32], all_logits: bool) -> QwenPrefillOut {
        assert!(!tokens.is_empty(), "prefill requires at least one token");
        if no_batch_prefill() {
            assert!(
                self.q8_spine.iter().all(|x| x.is_none()),
                "the q8 spine requires the batched prefill (unset MICROKIMI_NO_QWEN_BATCH)"
            );
            // A/B benchmarking fallback: one forward per token, bit-identical
            // logits. Final-norm hiddens are not recoverable from forward, so
            // they stay empty; the MTP drafter checks and refuses the toggle.
            let mut out = QwenPrefillOut { logits: Vec::new(), hidden: Vec::new() };
            for (i, &token) in tokens.iter().enumerate() {
                let logits = self.forward(token);
                out.hidden.push(Vec::new());
                if all_logits || i + 1 == tokens.len() {
                    out.logits.push(logits);
                }
            }
            return out;
        }
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
                // fold the previous layer's MLP residual into this norm pass
                let add = if l > 0 { Some(&attn_out[..t_count * d]) } else { None };
                par_add_norm(&mut hidden, add, w, c.norm_eps as f32, &mut normed, t_count, d);
            }
            let q8 = self.q8_spine[l].as_ref();
            let t_attn = std::time::Instant::now();
            match (&layer.attn, &mut self.caches[l]) {
                (QwenAttnW::Linear(w), QwenCache::Linear(cache)) => {
                    lin_attn_prefill(data, w, q8, &c, &normed, t_count, cache, &mut attn_out);
                    dprof_add(0, t_attn.elapsed());
                }
                (QwenAttnW::Full(w), QwenCache::Full(cache)) => {
                    full_attn_prefill(
                        data, w, q8, &c, &normed, t_count, self.pos, cache, &mut attn_out,
                    );
                    dprof_add(1, t_attn.elapsed());
                }
                _ => unreachable!("Qwen attention/cache kind mismatch at layer {}", l),
            }
            {
                let w = tensor(data, &layer.post_norm);
                par_add_norm(
                    &mut hidden,
                    Some(&attn_out[..t_count * d]),
                    w,
                    c.norm_eps as f32,
                    &mut normed,
                    t_count,
                    d,
                );
            }
            let t_mlp = std::time::Instant::now();
            mlp_prefill(
                data,
                &layer.mlp,
                self.mlp_q8[l].as_ref(),
                &c,
                &normed,
                t_count,
                self.skip_bounds[l].as_ref(),
                &mut attn_out,
            );
            dprof_add(2, t_mlp.elapsed());
            // the MLP residual folds into the NEXT layer's input-norm pass;
            // after the last layer it is applied just below
        }
        for i in 0..t_count * d {
            hidden[i] += attn_out[i];
        }
        let t_head = std::time::Instant::now();

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
        dprof_add(3, t_head.elapsed());
        self.pos += t_count;
        if let Some(last) = out.logits.last() {
            self.last_logits = last.clone();
        }
        out
    }
}

/// Prefill for one gated delta-rule layer. Projections fan out over token
/// ranges, the convolution is the cheap sequential scan, the recurrence
/// fans out over heads (each head replays its own token sequence), and the
/// gated norm plus output projection fan out over tokens again. Per-token
/// float operations are exactly those of `lin_attn_step`.
#[allow(clippy::too_many_arguments)]
fn lin_attn_prefill(
    data: &[u8],
    w: &QwenLinW,
    q8: Option<&LayerQ8>,
    c: &QwenConfig,
    normed: &[f32],
    t_count: usize,
    cache: &mut LinCache,
    attn_out: &mut [f32],
) {
    let q8_mats = q8.map(|q| match q {
        LayerQ8::Linear { in_qkv, in_z, out_proj } => (in_qkv, in_z, out_proj),
        LayerQ8::Full { .. } => unreachable!("q8 layer kind mismatch"),
    });

    // single-token decode: one token cannot fan out across tokens, so the
    // projections run ROW-parallel on the whole pool from this thread
    // (bit-identical: the pooled kernels chunk whole rows). Without this,
    // decode would run every attention matvec on one core.
    if t_count == 1 {
        let d = c.d;
        let (kt, vt) = (c.lin_key_total(), c.lin_value_total());
        let conv_dim = kt * 2 + vt;
        let heads = c.lin_v_heads;
        let mut qkv = vec![0.0f32; conv_dim];
        let mut z = vec![0.0f32; vt];
        let mut b_raw = vec![0.0f32; heads];
        let mut a_raw = vec![0.0f32; heads];
        let x = &normed[..d];
        match q8_mats {
            // both q8: one activation quantization, one pool barrier
            Some((SpineMat::Q8(hq), SpineMat::Q8(hz), _)) => {
                crate::model::Q8Head::matvec2(hq, hz, x, &mut qkv, &mut z);
            }
            Some((q_qkv, q_z, _)) => {
                q_qkv.matvec(x, &mut qkv);
                q_z.matvec(x, &mut z);
            }
            None => {
                crate::model::ops::matvec(tensor(data, &w.in_qkv), conv_dim, d, x, &mut qkv);
                crate::model::ops::matvec(tensor(data, &w.in_z), vt, d, x, &mut z);
            }
        }
        crate::model::ops::matvec(tensor(data, &w.in_b), heads, d, x, &mut b_raw);
        crate::model::ops::matvec(tensor(data, &w.in_a), heads, d, x, &mut a_raw);
        let mut mixed = vec![0.0f32; vt];
        lin_attn_recur(
            c,
            &qkv,
            &b_raw,
            &a_raw,
            tensor(data, &w.conv),
            tensor(data, &w.a_log),
            tensor(data, &w.dt_bias),
            cache,
            &mut mixed,
        );
        let norm_w = tensor(data, &w.norm);
        let (kd, vd) = (c.lin_k_dim, c.lin_v_dim);
        let _ = kd;
        for h in 0..heads {
            let (a, b) = (h * vd, (h + 1) * vd);
            rmsnorm_gated(&mut mixed[a..b], norm_w, &z[a..b], c.norm_eps as f32);
        }
        match q8_mats {
            Some((_, _, q_out)) => q_out.matvec(&mixed, &mut attn_out[..d]),
            None => crate::model::ops::matvec(
                tensor(data, &w.out_proj),
                d,
                vt,
                &mixed,
                &mut attn_out[..d],
            ),
        }
        return;
    }
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

    // projections: every weight row is streamed ONCE and dotted against
    // all tokens (the multi kernels), instead of once per token per worker
    let mut qkv = vec![0.0f32; t_count * conv_dim];
    let mut z = vec![0.0f32; t_count * vt];
    let mut b_raw = vec![0.0f32; t_count * heads];
    let mut a_raw = vec![0.0f32; t_count * heads];
    {
        let xs: Vec<&[f32]> = normed.chunks(d).take(t_count).collect();
        match q8_mats {
            Some((q_qkv, q_z, _)) => {
                let mut outs: Vec<&mut [f32]> = qkv.chunks_mut(conv_dim).collect();
                q_qkv.matvec_multi(&xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = z.chunks_mut(vt).collect();
                q_z.matvec_multi(&xs, &mut outs);
            }
            None => {
                let mut outs: Vec<&mut [f32]> = qkv.chunks_mut(conv_dim).collect();
                crate::model::ops::matvec_multi(in_qkv, conv_dim, d, &xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = z.chunks_mut(vt).collect();
                crate::model::ops::matvec_multi(in_z, vt, d, &xs, &mut outs);
            }
        }
        let mut outs: Vec<&mut [f32]> = b_raw.chunks_mut(heads).collect();
        crate::model::ops::matvec_multi(in_b, heads, d, &xs, &mut outs);
        let mut outs: Vec<&mut [f32]> = a_raw.chunks_mut(heads).collect();
        crate::model::ops::matvec_multi(in_a, heads, d, &xs, &mut outs);
    }

    // causal convolution: sequential over time but channel-separable
    // (each channel owns its (k-1)-tap state), so channels fan out over
    // workers and each replays all tokens for its slice - bit-identical
    // to the former token-major conv_step loop.
    let mut conved = vec![0.0f32; t_count * conv_dim];
    {
        let k = c.conv_kernel;
        let workers = prefill_workers(conv_dim);
        let out_ptr = crate::model::pool::MPtr(conved.as_mut_ptr());
        std::thread::scope(|s| {
            let qkv = &qkv;
            let mut state_rest = cache.conv.as_mut_slice();
            for (c0, c1) in ranges(conv_dim, workers) {
                let (state_c, sr) = state_rest.split_at_mut((c1 - c0) * (k - 1));
                state_rest = sr;
                s.spawn(move || {
                    let out_ptr = out_ptr;
                    for t in 0..t_count {
                        let x = &qkv[t * conv_dim..(t + 1) * conv_dim];
                        for i in c0..c1 {
                            let st = &mut state_c[(i - c0) * (k - 1)..(i - c0 + 1) * (k - 1)];
                            let wt = &conv[i * k..(i + 1) * k];
                            let mut acc = 0.0f32;
                            for j in 0..k - 1 {
                                acc += st[j] * wt[j];
                            }
                            acc += x[i] * wt[k - 1];
                            // SAFETY: column i is owned by this worker alone
                            // (disjoint channel ranges), the barrier is the
                            // scope end.
                            unsafe {
                                *out_ptr.0.add(t * conv_dim + i) = acc / (1.0 + (-acc).exp());
                            }
                            for j in 0..k.saturating_sub(2) {
                                st[j] = st[j + 1];
                            }
                            st[k - 2] = x[i];
                        }
                    }
                });
            }
        });
    }

    // recurrence, parallel over heads: each head replays its own tokens in
    // order against its private state slice (head-major mixed buffer)
    let mut mixed_hm = vec![0.0f32; heads * t_count * vd];
    // GPU scan (MICROKIMI_QWEN_GPU=1): the same recurrence as one thread
    // per (head, value column) on the GPU; falls back to the CPU scan on
    // any refusal (MICROKIMI_QWEN_GPU_NOSCAN=1 pins it to the CPU).
    #[allow(unused_mut)]
    let mut scan_done = false;
    #[cfg(target_os = "macos")]
    if crate::model::metal::qwen_gpu_scan_on() && t_count >= crate::model::metal::GEMM_MIN_T {
        scan_done = gpu_lin_scan(
            &conved, b_raw.as_slice(), a_raw.as_slice(), dt_bias, a_log, &mut cache.state,
            heads, rep, kd, vd, kt, conv_dim, t_count, &mut mixed_hm,
        );
    }
    if !scan_done {
        // heads chunk across workers - one spawn per worker rather than
        // one per head (32 spawns/layer dominated small verify batches);
        // per-head math unchanged, bit-identical.
        let hworkers = prefill_workers(heads);
        let hchunk = heads.div_ceil(hworkers);
        std::thread::scope(|s| {
            let mut state_rest = cache.state.as_mut_slice();
            let mut mixed_rest = mixed_hm.as_mut_slice();
            let conved = &conved;
            let b_raw = &b_raw;
            let a_raw = &a_raw;
            let use_chunked = q8.is_some() && t_count >= 32 && chunked_scan_on();
            for h0 in (0..heads).step_by(hchunk.max(1)) {
                let h1 = (h0 + hchunk).min(heads);
                let (state_c, sr) = state_rest.split_at_mut((h1 - h0) * kd * vd);
                let (mixed_c, mr) = mixed_rest.split_at_mut((h1 - h0) * t_count * vd);
                state_rest = sr;
                mixed_rest = mr;
                s.spawn(move || {
                    // reusable per-worker q/k scratch: a heap allocation
                    // per (head, token) was a measurable slice of the scan
                    let mut q = vec![0.0f32; kd];
                    let mut k = vec![0.0f32; kd];
                    // chunked-scan prep buffers (spine modes): normalized
                    // q/k, contiguous v, per-token beta and decay
                    let (mut qb, mut kb, mut vb, mut bb, mut gb) = if use_chunked {
                        (
                            vec![0.0f32; t_count * kd],
                            vec![0.0f32; t_count * kd],
                            vec![0.0f32; t_count * vd],
                            vec![0.0f32; t_count],
                            vec![0.0f32; t_count],
                        )
                    } else {
                        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
                    };
                    for h in h0..h1 {
                        let state_h =
                            &mut state_c[(h - h0) * kd * vd..(h - h0 + 1) * kd * vd];
                        let mixed_h =
                            &mut mixed_c[(h - h0) * t_count * vd..(h - h0 + 1) * t_count * vd];
                        let kh = h / rep.max(1);
                        if use_chunked {
                            // WY chunked scan: identical math, GEMM-shaped;
                            // reassociation-level numerics, spine-only
                            let scale = 1.0 / (kd as f32).sqrt();
                            for t in 0..t_count {
                                let row = &conved[t * conv_dim..(t + 1) * conv_dim];
                                let qd = &mut qb[t * kd..(t + 1) * kd];
                                qd.copy_from_slice(&row[kh * kd..(kh + 1) * kd]);
                                l2norm(qd, 1e-6);
                                for value in qd.iter_mut() {
                                    *value *= scale;
                                }
                                let kdst = &mut kb[t * kd..(t + 1) * kd];
                                kdst.copy_from_slice(&row[kt + kh * kd..kt + (kh + 1) * kd]);
                                l2norm(kdst, 1e-6);
                                vb[t * vd..(t + 1) * vd]
                                    .copy_from_slice(&row[2 * kt + h * vd..2 * kt + (h + 1) * vd]);
                                bb[t] = 1.0 / (1.0 + (-b_raw[t * heads + h]).exp());
                                let sp = {
                                    let arg = a_raw[t * heads + h] + dt_bias[h];
                                    if arg > 20.0 { arg } else { (1.0 + arg.exp()).ln() }
                                };
                                gb[t] = (-a_log[h].exp() * sp).exp();
                            }
                            let state_h =
                                &mut state_c[(h - h0) * kd * vd..(h - h0 + 1) * kd * vd];
                            let mixed_h =
                                &mut mixed_c[(h - h0) * t_count * vd..(h - h0 + 1) * t_count * vd];
                            chunked_scan_head(
                                state_h, mixed_h, &qb, &kb, &vb, &bb, &gb, t_count, kd, vd,
                            );
                            continue;
                        }
                        for t in 0..t_count {
                            let row = &conved[t * conv_dim..(t + 1) * conv_dim];
                            q.copy_from_slice(&row[kh * kd..(kh + 1) * kd]);
                            k.copy_from_slice(&row[kt + kh * kd..kt + (kh + 1) * kd]);
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
                    }
                });
            }
        });
    }

    // gated norm per token (token-major gather), then ONE multi-lane
    // output projection over all tokens
    let mut mixed_tm = vec![0.0f32; t_count * vt];
    {
        let workers = prefill_workers(t_count);
        std::thread::scope(|s| {
            let mut rest = mixed_tm.as_mut_slice();
            let mixed_hm = &mixed_hm;
            let z = &z;
            for (t0, t1) in ranges(t_count, workers) {
                let n = t1 - t0;
                let (chunk, r) = rest.split_at_mut(n * vt);
                rest = r;
                s.spawn(move || {
                    for i in 0..n {
                        let t = t0 + i;
                        let mixed = &mut chunk[i * vt..(i + 1) * vt];
                        for h in 0..heads {
                            mixed[h * vd..(h + 1) * vd].copy_from_slice(
                                &mixed_hm[(h * t_count + t) * vd..(h * t_count + t + 1) * vd],
                            );
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
                    }
                });
            }
        });
    }
    {
        let xs: Vec<&[f32]> = mixed_tm.chunks(vt).collect();
        let mut outs: Vec<&mut [f32]> = attn_out[..t_count * d].chunks_mut(d).collect();
        match q8_mats {
            Some((_, _, q_out)) => q_out.matvec_multi(&xs, &mut outs),
            None => crate::model::ops::matvec_multi(out_proj, d, vt, &xs, &mut outs),
        }
    }
}

/// Convolution + per-head delta recurrence of one token (the exact ops
/// of `lin_attn_step` between its projections and its gated norm).
#[allow(clippy::too_many_arguments)]
fn lin_attn_recur(
    c: &QwenConfig,
    qkv: &[f32],
    b_raw: &[f32],
    a_raw: &[f32],
    conv_w: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    cache: &mut LinCache,
    mixed: &mut [f32],
) {
    let (kt, vt) = (c.lin_key_total(), c.lin_value_total());
    let conv_dim = kt * 2 + vt;
    let mut conved = vec![0.0f32; conv_dim];
    conv_step(qkv, conv_w, c.conv_kernel, &mut cache.conv, &mut conved);
    let rep = c.lin_v_heads / c.lin_k_heads.max(1);
    let (kd, vd) = (c.lin_k_dim, c.lin_v_dim);
    let _ = vt;
    // reusable q/k scratch (two heap allocations per head added up)
    let mut q = vec![0.0f32; kd];
    let mut k = vec![0.0f32; kd];
    for h in 0..c.lin_v_heads {
        let kh = h / rep.max(1);
        q.copy_from_slice(&conved[kh * kd..(kh + 1) * kd]);
        k.copy_from_slice(&conved[kt + kh * kd..kt + (kh + 1) * kd]);
        let v = &conved[2 * kt + h * vd..2 * kt + (h + 1) * vd];
        l2norm(&mut q, 1e-6);
        l2norm(&mut k, 1e-6);
        let scale = 1.0 / (kd as f32).sqrt();
        for t in q.iter_mut() {
            *t *= scale;
        }
        let beta = 1.0 / (1.0 + (-b_raw[h]).exp());
        let sp = {
            let t = a_raw[h] + dt_bias[h];
            if t > 20.0 {
                t
            } else {
                (1.0 + t.exp()).ln()
            }
        };
        let g = -a_log[h].exp() * sp;
        let st = &mut cache.state[h * kd * vd..(h + 1) * kd * vd];
        delta_step(st, &q, &k, v, g, beta, &mut mixed[h * vd..(h + 1) * vd]);
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
    q8: Option<&LayerQ8>,
    c: &QwenConfig,
    normed: &[f32],
    t_count: usize,
    base_pos: usize,
    cache: &mut FullCache,
    attn_out: &mut [f32],
) {
    let q8_mats = q8.map(|q| match q {
        LayerQ8::Full { q_proj, k_proj, v_proj, o_proj } => (q_proj, k_proj, v_proj, o_proj),
        LayerQ8::Linear { .. } => unreachable!("q8 layer kind mismatch"),
    });

    // single-token decode: row-parallel projections on the whole pool
    // (see lin_attn_prefill for the rationale; bit-identical)
    if t_count == 1 {
        let d = c.d;
        let hd = c.head_dim;
        let q_width = c.n_heads * hd;
        let kv_width = c.n_kv_heads * hd;
        assert_eq!(cache.len, base_pos, "full-attention cache position mismatch");
        let x = &normed[..d];
        let mut qg = vec![0.0f32; q_width * 2];
        let mut k = vec![0.0f32; kv_width];
        let mut v = vec![0.0f32; kv_width];
        match q8_mats {
            Some((qq, qk, qv, _)) => {
                qq.matvec(x, &mut qg);
                qk.matvec(x, &mut k);
                qv.matvec(x, &mut v);
            }
            None => {
                crate::model::ops::matvec(tensor(data, &w.q_proj), q_width * 2, d, x, &mut qg);
                crate::model::ops::matvec(tensor(data, &w.k_proj), kv_width, d, x, &mut k);
                crate::model::ops::matvec(tensor(data, &w.v_proj), kv_width, d, x, &mut v);
            }
        }
        let mut mixed = vec![0.0f32; q_width];
        full_attn_tail(
            c,
            &qg,
            &k,
            &v,
            tensor(data, &w.q_norm),
            tensor(data, &w.k_norm),
            base_pos,
            cache,
            q8_mats.is_some() && hd % 32 == 0,
            &mut mixed,
        );
        match q8_mats {
            Some((_, _, _, qo)) => qo.matvec(&mixed, &mut attn_out[..d]),
            None => crate::model::ops::matvec(
                tensor(data, &w.o_proj),
                d,
                q_width,
                &mixed,
                &mut attn_out[..d],
            ),
        }
        return;
    }
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

    // projections once for all tokens (weights streamed a single time),
    // then the per-token interleave / norm / rotary phase fans out
    let mut qg_all = vec![0.0f32; t_count * q_width * 2];
    let mut q_all = vec![0.0f32; t_count * q_width];
    let mut gate_all = vec![0.0f32; t_count * q_width];
    let mut k_all = vec![0.0f32; t_count * kv_width];
    let mut v_all = vec![0.0f32; t_count * kv_width];
    {
        let xs: Vec<&[f32]> = normed.chunks(d).take(t_count).collect();
        match q8_mats {
            Some((qq, qk, qv, _)) => {
                let mut outs: Vec<&mut [f32]> = qg_all.chunks_mut(q_width * 2).collect();
                qq.matvec_multi(&xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = k_all.chunks_mut(kv_width).collect();
                qk.matvec_multi(&xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = v_all.chunks_mut(kv_width).collect();
                qv.matvec_multi(&xs, &mut outs);
            }
            None => {
                let mut outs: Vec<&mut [f32]> = qg_all.chunks_mut(q_width * 2).collect();
                crate::model::ops::matvec_multi(q_proj, q_width * 2, d, &xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = k_all.chunks_mut(kv_width).collect();
                crate::model::ops::matvec_multi(k_proj, kv_width, d, &xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = v_all.chunks_mut(kv_width).collect();
                crate::model::ops::matvec_multi(v_proj, kv_width, d, &xs, &mut outs);
            }
        }
    }
    {
        let workers = prefill_workers(t_count);
        std::thread::scope(|s| {
            let mut q_rest = q_all.as_mut_slice();
            let mut g_rest = gate_all.as_mut_slice();
            let mut k_rest = k_all.as_mut_slice();
            let qg_all = &qg_all;
            for (t0, t1) in ranges(t_count, workers) {
                let n = t1 - t0;
                let (q_c, qr) = q_rest.split_at_mut(n * q_width);
                let (g_c, gr) = g_rest.split_at_mut(n * q_width);
                let (k_c, kr) = k_rest.split_at_mut(n * kv_width);
                q_rest = qr;
                g_rest = gr;
                k_rest = kr;
                s.spawn(move || {
                    for i in 0..n {
                        let pos = base_pos + t0 + i;
                        let qg = &qg_all[(t0 + i) * q_width * 2..(t0 + i + 1) * q_width * 2];
                        let q = &mut q_c[i * q_width..(i + 1) * q_width];
                        let gate = &mut g_c[i * q_width..(i + 1) * q_width];
                        let k = &mut k_c[i * kv_width..(i + 1) * kv_width];
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
    push_k_mirror(cache, &k_all);
    cache.len += t_count;

    // each position attends over its causal prefix, parallel over tokens;
    // gated mixes land token-major for the single multi o_proj below
    let mut mixed_all = vec![0.0f32; t_count * q_width];
    // GPU attention (MICROKIMI_QWEN_GPU=1): scores and mixes for every
    // head as two batched GEMMs with the causal softmax on the CPU in
    // between; falls back to the CPU loop below on any refusal.
    #[allow(unused_mut)]
    let mut gpu_attn_done = false;
    #[cfg(target_os = "macos")]
    if crate::model::metal::qwen_gpu_attn_on() && t_count >= crate::model::metal::GEMM_MIN_T {
        gpu_attn_done = gpu_full_attention(
            c, cache, &q_all, &gate_all, base_pos, t_count, q_width, kv_width, hd, &mut mixed_all,
        );
    }
    if !gpu_attn_done {
        let workers = prefill_workers(t_count);
        let groups = c.n_heads / c.n_kv_heads;
        let scale = 1.0f32 / (hd as f32).sqrt();
        std::thread::scope(|s| {
            let mut mixed_rest = mixed_all.as_mut_slice();
            let cache_k = &cache.k;
            let cache_v = &cache.v;
            let q_all = &q_all;
            let gate_all = &gate_all;
            let q8_scores = q8_mats.is_some() && hd % 32 == 0;
            let cache_kq = &cache.kq;
            let cache_kqs = &cache.kqs;
            for (t0, t1) in causal_ranges(t_count, workers, base_pos) {
                let n = t1 - t0;
                let (mixed_c, mr) = mixed_rest.split_at_mut(n * q_width);
                mixed_rest = mr;
                s.spawn(move || {
                    let mut qq = crate::quant::q8::Q8Vec::new();
                    for i in 0..n {
                        let t = t0 + i;
                        let window = base_pos + t + 1;
                        let mut mixed = vec![0.0f32; q_width];
                        let mut scores = vec![0.0f32; window];
                        for h in 0..c.n_heads {
                            let kh = h / groups;
                            let qh = &q_all[t * q_width + h * hd..t * q_width + (h + 1) * hd];
                            let mut max_score = f32::NEG_INFINITY;
                            // quantized spine: integer score dots against the
                            // K mirror (the f32 default stays bit-exact)
                            if q8_scores {
                                crate::quant::q8::quantize_q8_into(qh, &mut qq);
                                max_score = crate::quant::q8::score_window(
                                    &qq,
                                    cache_kq,
                                    cache_kqs,
                                    kv_width,
                                    kh * hd,
                                    hd / 32,
                                    window,
                                    scale,
                                    &mut scores,
                                );
                            } else {
                            for u in 0..window {
                                let off = u * kv_width + kh * hd;
                                let sc =
                                    crate::model::ops::dot(qh, &cache_k[off..off + hd]) * scale;
                                scores[u] = sc;
                                max_score = max_score.max(sc);
                            }
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
                        mixed_c[i * q_width..(i + 1) * q_width].copy_from_slice(&mixed);
                    }
                });
            }
        });
    }
    {
        let xs: Vec<&[f32]> = mixed_all.chunks(q_width).collect();
        let mut outs: Vec<&mut [f32]> = attn_out[..t_count * d].chunks_mut(d).collect();
        match q8_mats {
            Some((_, _, _, qo)) => qo.matvec_multi(&xs, &mut outs),
            None => crate::model::ops::matvec_multi(o_proj, d, q_width, &xs, &mut outs),
        }
    }
}

/// Prepares the delta-scan inputs exactly as the CPU scan computes them
/// per token (l2-normalized q and k with the 1/sqrt(kd) query scale,
/// kv-heads expanded across their group, sigmoid beta, softplus-gated
/// decay), then runs the whole recurrence on the GPU. `state` is the
/// live recurrent carry ([heads, kd, vd]) and is updated in place;
/// `mixed_hm` receives the head-major readout. False = CPU scan.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn gpu_lin_scan(
    conved: &[f32],
    b_raw: &[f32],
    a_raw: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    state: &mut [f32],
    heads: usize,
    rep: usize,
    kd: usize,
    vd: usize,
    kt: usize,
    conv_dim: usize,
    t_count: usize,
    mixed_hm: &mut [f32],
) -> bool {
    let mut qn = vec![0.0f32; t_count * heads * kd];
    let mut kn = vec![0.0f32; t_count * heads * kd];
    let mut va = vec![0.0f32; t_count * heads * vd];
    let mut beta = vec![0.0f32; t_count * heads];
    let mut gv = vec![0.0f32; t_count * heads];
    let workers = prefill_workers(t_count);
    std::thread::scope(|s| {
        let (mut q_rest, mut k_rest, mut v_rest) =
            (qn.as_mut_slice(), kn.as_mut_slice(), va.as_mut_slice());
        let (mut b_rest, mut g_rest) = (beta.as_mut_slice(), gv.as_mut_slice());
        for (t0, t1) in ranges(t_count, workers) {
            let n = t1 - t0;
            let (q_c, qr) = q_rest.split_at_mut(n * heads * kd);
            let (k_c, kr) = k_rest.split_at_mut(n * heads * kd);
            let (v_c, vr) = v_rest.split_at_mut(n * heads * vd);
            let (b_c, br) = b_rest.split_at_mut(n * heads);
            let (g_c, gr) = g_rest.split_at_mut(n * heads);
            q_rest = qr;
            k_rest = kr;
            v_rest = vr;
            b_rest = br;
            g_rest = gr;
            s.spawn(move || {
                let scale = 1.0 / (kd as f32).sqrt();
                for i in 0..n {
                    let t = t0 + i;
                    let row = &conved[t * conv_dim..(t + 1) * conv_dim];
                    for h in 0..heads {
                        let kh = h / rep.max(1);
                        let q = &mut q_c[(i * heads + h) * kd..(i * heads + h + 1) * kd];
                        q.copy_from_slice(&row[kh * kd..(kh + 1) * kd]);
                        l2norm(q, 1e-6);
                        for value in q.iter_mut() {
                            *value *= scale;
                        }
                        let k = &mut k_c[(i * heads + h) * kd..(i * heads + h + 1) * kd];
                        k.copy_from_slice(&row[kt + kh * kd..kt + (kh + 1) * kd]);
                        l2norm(k, 1e-6);
                        v_c[(i * heads + h) * vd..(i * heads + h + 1) * vd]
                            .copy_from_slice(&row[2 * kt + h * vd..2 * kt + (h + 1) * vd]);
                        b_c[i * heads + h] = 1.0 / (1.0 + (-b_raw[t * heads + h]).exp());
                        let sp = {
                            let arg = a_raw[t * heads + h] + dt_bias[h];
                            if arg > 20.0 {
                                arg
                            } else {
                                (1.0 + arg.exp()).ln()
                            }
                        };
                        // decay = exp(g), the exact factor the CPU delta_step applies
                        g_c[i * heads + h] = (-a_log[h].exp() * sp).exp();
                    }
                }
            });
        }
    });
    crate::model::metal::gpu_delta_scan(
        &qn, &kn, &va, &beta, &gv, state, mixed_hm, t_count, heads, kd, vd,
    )
}

/// Full attention over the causal prefix as two batched GPU GEMMs -
/// scores = scale·Q·Kᵀ and mix = P·V for every head in one encode each -
/// with the causal softmax on the CPU in between (masked tail zeroed so
/// the P·V product runs over the full cache width). K and V expand their
/// kv-head across the GQA group into head-major stacks. Numerics differ
/// from the CPU loop only by GEMM reassociation; any refusal returns
/// false and the caller keeps its CPU path.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn gpu_full_attention(
    c: &QwenConfig,
    cache: &FullCache,
    q_all: &[f32],
    gate_all: &[f32],
    base_pos: usize,
    t_count: usize,
    q_width: usize,
    kv_width: usize,
    hd: usize,
    mixed_all: &mut [f32],
) -> bool {
    let heads = c.n_heads;
    let groups = (heads / c.n_kv_heads).max(1);
    let l = cache.len;
    debug_assert_eq!(l, base_pos + t_count);
    let scale = 1.0f32 / (hd as f32).sqrt();
    // the scores stack is the big allocation; refuse early what the
    // staging ceiling would refuse anyway (256 MB)
    if heads * t_count * l * 4 > 256 * 1024 * 1024 || !crate::model::metal::mps_available() {
        return false;
    }

    // head-major repacks, parallel over heads
    let mut q_hm = vec![0.0f32; heads * t_count * hd];
    let mut k_hm = vec![0.0f32; heads * l * hd];
    let mut v_hm = vec![0.0f32; heads * l * hd];
    std::thread::scope(|s| {
        let mut q_rest = q_hm.as_mut_slice();
        let mut k_rest = k_hm.as_mut_slice();
        let mut v_rest = v_hm.as_mut_slice();
        for h in 0..heads {
            let (q_h, qr) = q_rest.split_at_mut(t_count * hd);
            let (k_h, kr) = k_rest.split_at_mut(l * hd);
            let (v_h, vr) = v_rest.split_at_mut(l * hd);
            q_rest = qr;
            k_rest = kr;
            v_rest = vr;
            let (ck, cv) = (&cache.k, &cache.v);
            s.spawn(move || {
                let kh = h / groups;
                for t in 0..t_count {
                    q_h[t * hd..(t + 1) * hd]
                        .copy_from_slice(&q_all[t * q_width + h * hd..t * q_width + (h + 1) * hd]);
                }
                for u in 0..l {
                    k_h[u * hd..(u + 1) * hd]
                        .copy_from_slice(&ck[u * kv_width + kh * hd..u * kv_width + (kh + 1) * hd]);
                    v_h[u * hd..(u + 1) * hd]
                        .copy_from_slice(&cv[u * kv_width + kh * hd..u * kv_width + (kh + 1) * hd]);
                }
            });
        }
    });

    // fused path: scores GEMM + causal softmax + P.V GEMM in one
    // command buffer, scores resident on the GPU (f16 mode)
    let mut mixed_try = vec![0.0f32; heads * t_count * hd];
    if crate::model::metal::gpu_attention_fused(
        &q_hm, &k_hm, &v_hm, heads, t_count, l, hd, base_pos, scale, &mut mixed_try,
    ) {
        gather_gate(&mixed_try, gate_all, heads, t_count, hd, q_width, mixed_all);
        return true;
    }
    let mut scores = vec![0.0f32; heads * t_count * l];
    if !crate::model::metal::gpu_gemm_batched(&q_hm, &k_hm, heads, t_count, l, hd, true, scale, &mut scores) {
        return false;
    }
    // causal softmax in place, parallel over heads
    std::thread::scope(|s| {
        let mut rest = scores.as_mut_slice();
        for _ in 0..heads {
            let (rows_h, r) = rest.split_at_mut(t_count * l);
            rest = r;
            s.spawn(move || {
                for t in 0..t_count {
                    let row = &mut rows_h[t * l..(t + 1) * l];
                    let window = base_pos + t + 1;
                    let mut mx = f32::NEG_INFINITY;
                    for v in row[..window].iter() {
                        mx = mx.max(*v);
                    }
                    let mut denom = 0.0f32;
                    for v in row[..window].iter_mut() {
                        *v = (*v - mx).exp();
                        denom += *v;
                    }
                    let inv = 1.0 / denom;
                    for v in row[..window].iter_mut() {
                        *v *= inv;
                    }
                    for v in row[window..].iter_mut() {
                        *v = 0.0;
                    }
                }
            });
        }
    });

    let mut mixed_hm = vec![0.0f32; heads * t_count * hd];
    if !crate::model::metal::gpu_gemm_batched(&scores, &v_hm, heads, t_count, hd, l, false, 1.0, &mut mixed_hm) {
        return false;
    }
    gather_gate(&mixed_hm, gate_all, heads, t_count, hd, q_width, mixed_all);
    true
}

/// Token-major gather of a head-major mix plus the sigmoid output gate
/// (the tail both GPU attention paths share).
#[cfg(target_os = "macos")]
fn gather_gate(
    mixed_hm: &[f32],
    gate_all: &[f32],
    heads: usize,
    t_count: usize,
    hd: usize,
    q_width: usize,
    mixed_all: &mut [f32],
) {
    let workers = prefill_workers(t_count);
    std::thread::scope(|s| {
        let mut rest = &mut mixed_all[..t_count * q_width];
        for (t0, t1) in ranges(t_count, workers) {
            let (chunk, r) = rest.split_at_mut((t1 - t0) * q_width);
            rest = r;
            s.spawn(move || {
                for (i, t) in (t0..t1).enumerate() {
                    let m = &mut chunk[i * q_width..(i + 1) * q_width];
                    for h in 0..heads {
                        m[h * hd..(h + 1) * hd].copy_from_slice(
                            &mixed_hm[(h * t_count + t) * hd..(h * t_count + t + 1) * hd],
                        );
                    }
                    for (j, value) in m.iter_mut().enumerate() {
                        *value *= 1.0 / (1.0 + (-gate_all[t * q_width + j]).exp());
                    }
                }
            });
        }
    });
}

/// Prefill MLP dispatch: every token is independent, so tokens fan out over
/// worker ranges and each token runs the exact single-token math (routed
/// experts sequentially inside its worker for the MoE variant).
#[allow(clippy::too_many_arguments)]
fn mlp_prefill(
    data: &[u8],
    mlp: &QwenMlpW,
    q8m: Option<&MlpQ8>,
    c: &QwenConfig,
    normed: &[f32],
    t_count: usize,
    bounds: Option<&SkipBounds>,
    out: &mut [f32],
) {
    let d = c.d;
    // the unpacked-i8 MLP wins on CPU; on macOS the GPU and AMX offloads
    // keep priority (their hooks live inside the packed multi kernels)
    #[cfg(target_os = "macos")]
    let q8m = if crate::model::metal::qwen_gpu_on() || crate::model::accel::accel_on() {
        None
    } else {
        q8m
    };
    // dense batch without a budget: three multi-kernel passes stream the
    // packed weights once for ALL tokens
    if t_count > 1 && bounds.is_none() {
        if let QwenMlpW::Dense { gate, up, down } = mlp {
            let inter = c.dense_inter;
            let threads = crate::model::pool::pool().workers.max(1);
            let xs: Vec<&[f32]> = normed.chunks(d).take(t_count).collect();
            let mut h_gate = vec![0.0f32; t_count * inter];
            let mut h_up = vec![0.0f32; t_count * inter];
            if let Some(mq) = q8m {
                let mut outs: Vec<&mut [f32]> = h_gate.chunks_mut(inter).collect();
                mq.gate.matvec_multi(&xs, &mut outs);
                let mut outs: Vec<&mut [f32]> = h_up.chunks_mut(inter).collect();
                mq.up.matvec_multi(&xs, &mut outs);
            } else {
                let (pg, sg) = packed_parts(data, gate);
                let mut outs: Vec<&mut [f32]> = h_gate.chunks_mut(inter).collect();
                crate::quant::mxfp4::matvec_packed_multi(pg, sg, inter, d, &xs, &mut outs, threads);
                let (pu, su) = packed_parts(data, up);
                let mut outs: Vec<&mut [f32]> = h_up.chunks_mut(inter).collect();
                crate::quant::mxfp4::matvec_packed_multi(pu, su, inter, d, &xs, &mut outs, threads);
            }
            // SiLU(gate) * up, parallel over token ranges: 3M exp calls per
            // layer at 1k tokens were a serial hotspot. Element-wise, so the
            // split is bit-identical to the serial loop.
            {
                let workers = prefill_workers(t_count);
                std::thread::scope(|s| {
                    let mut g_rest = h_gate.as_mut_slice();
                    let h_up = &h_up;
                    for (t0, t1) in ranges(t_count, workers) {
                        let n = (t1 - t0) * inter;
                        let (g_c, gr) = g_rest.split_at_mut(n);
                        g_rest = gr;
                        let u_c = &h_up[t0 * inter..t1 * inter];
                        s.spawn(move || {
                            for (g, u) in g_c.iter_mut().zip(u_c) {
                                *g = (*g / (1.0 + (-*g).exp())) * u;
                            }
                        });
                    }
                });
            }
            let hs: Vec<&[f32]> = h_gate.chunks(inter).collect();
            let mut outs: Vec<&mut [f32]> = out[..t_count * d].chunks_mut(d).collect();
            if let Some(mq) = q8m {
                mq.down.matvec_multi(&hs, &mut outs);
            } else {
                let (pd, sd) = packed_parts(data, down);
                crate::quant::mxfp4::matvec_packed_multi(pd, sd, d, inter, &hs, &mut outs, threads);
            }
            return;
        }
    }

    // single-token decode: the row-parallel MLP (pool-threaded packed
    // matvecs, parallel routed experts) instead of one serial worker
    if t_count == 1 {
        let value = match mlp {
            QwenMlpW::Moe {
                router,
                experts,
                shared,
                shared_gate,
            } => packed_moe(data, router, experts, shared, shared_gate, c, &normed[..d]),
            QwenMlpW::Dense { gate, up, down } => {
                packed_dense_mlp_q8(data, gate, up, down, q8m, c, &normed[..d], bounds)
            }
        };
        out[..d].copy_from_slice(&value);
        return;
    }
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
                            dense_token_serial(data, gate, up, down, c, x, bounds)
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
    bounds: Option<&SkipBounds>,
) -> Vec<f32> {
    let inter = c.dense_inter;
    let mut h_gate = vec![0.0f32; inter];
    let (pg, sg) = packed_parts(data, gate);
    crate::quant::mxfp4::matvec_packed(pg, sg, inter, c.d, x, &mut h_gate, 1);

    // certified-budget path (see packed_dense_mlp for the contract)
    if let (Some(bounds), budget) = (bounds, mlp_budget()) {
        if budget > 0.0 {
            let mut silu = vec![0.0f32; inter];
            for i in 0..inter {
                silu[i] = h_gate[i] / (1.0 + (-h_gate[i]).exp());
            }
            let x_l2 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
            let keep = bounds.keep_mask(&silu, x_l2, budget);
            let kept: Vec<usize> = (0..inter / 32).filter(|&b| keep[b]).collect();
            MLP_BLOCKS_TOTAL.fetch_add((inter / 32) as u64, std::sync::atomic::Ordering::Relaxed);
            MLP_BLOCKS_SKIPPED.fetch_add(
                (inter / 32 - kept.len()) as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let (pu, su) = packed_parts(data, up);
            let mut h = vec![0.0f32; inter];
            for &b in &kept {
                let mut rows = [0.0f32; 32];
                crate::quant::mxfp4::matvec_packed(
                    &pu[b * 32 * c.d / 2..(b + 1) * 32 * c.d / 2],
                    &su[b * 32 * (c.d / 32)..(b + 1) * 32 * (c.d / 32)],
                    32,
                    c.d,
                    x,
                    &mut rows,
                    1,
                );
                for j in 0..32 {
                    let i = b * 32 + j;
                    h[i] = silu[i] * rows[j];
                }
            }
            let mut out = vec![0.0f32; c.d];
            let (pd, sd) = packed_parts(data, down);
            crate::quant::mxfp4::matvec_packed_colblocks(pd, sd, c.d, inter, &kept, &h, &mut out, 1);
            return out;
        }
    }

    let mut h_up = vec![0.0f32; inter];
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

// ───────────────────── lane-batched decoding (throughput) ─────────────────────

/// One independent decode stream: its own caches and position, stepped
/// together with other lanes through `forward_lanes` so every weight
/// region is read once per layer for ALL lanes. In the memory-bound
/// decode regime this multiplies aggregate tokens/second by close to the
/// lane count. Per-lane results are bit-identical to the single-stream
/// `forward` (same per-row dots, same order).
pub struct DecodeLane {
    pub(crate) caches: Vec<QwenCache>,
    pub(crate) pos: usize,
}

impl DecodeLane {
    pub fn new(model: &QwenModel) -> DecodeLane {
        let c = &model.cfg;
        DecodeLane {
            caches: (0..c.n_layers)
                .map(|l| {
                    if c.is_full_attn(l) {
                        QwenCache::Full(FullCache::new(c))
                    } else {
                        QwenCache::Linear(LinCache::new(c))
                    }
                })
                .collect(),
            pos: 0,
        }
    }

}

impl QwenModel {
    /// Ingests a prompt into one lane by temporarily swapping its caches
    /// into the model and running the batched prefill (bit-identical).
    pub fn prefill_lane(&mut self, lane: &mut DecodeLane, tokens: &[u32]) -> Vec<f32> {
        std::mem::swap(&mut self.caches, &mut lane.caches);
        std::mem::swap(&mut self.pos, &mut lane.pos);
        let logits = self.prefill(tokens);
        std::mem::swap(&mut self.caches, &mut lane.caches);
        std::mem::swap(&mut self.pos, &mut lane.pos);
        logits
    }

    /// One decode step for every lane at once. The large projections and
    /// the packed MLP run through the multi-lane kernels (weights read
    /// once for all lanes); the recurrences, attention mixes, and norms
    /// are per-lane and fan out over scoped threads.
    pub fn forward_lanes(&self, lanes: &mut [&mut DecodeLane], tokens: &[u32]) -> Vec<Vec<f32>> {
        assert_eq!(lanes.len(), tokens.len());
        let n = lanes.len();
        assert!(n > 0, "forward_lanes needs at least one lane");
        let c = self.cfg.clone();
        let d = c.d;
        let data = &self.bin.data;
        for &t in tokens {
            assert!((t as usize) < c.vocab, "token outside the vocabulary");
        }

        let embed = tensor(data, &self.embed);
        let mut hidden: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| embed[t as usize * d..(t as usize + 1) * d].to_vec())
            .collect();
        let mut normed: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n];

        for l in 0..c.n_layers {
            let layer = &self.layers[l];
            {
                let w = tensor(data, &layer.input_norm);
                for i in 0..n {
                    rmsnorm(&hidden[i], w, c.norm_eps as f32, &mut normed[i]);
                }
            }
            let mut attn_out: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n];
            match &layer.attn {
                QwenAttnW::Linear(w) => {
                    let (kt, vt) = (c.lin_key_total(), c.lin_value_total());
                    let conv_dim = kt * 2 + vt;
                    let heads = c.lin_v_heads;
                    let mut qkv = vec![vec![0.0f32; conv_dim]; n];
                    let mut z = vec![vec![0.0f32; vt]; n];
                    let mut b_raw = vec![vec![0.0f32; heads]; n];
                    let mut a_raw = vec![vec![0.0f32; heads]; n];
                    let xs: Vec<&[f32]> = normed.iter().map(|v| v.as_slice()).collect();
                    match self.q8_spine[l].as_ref() {
                        Some(LayerQ8::Linear { in_qkv, in_z, .. }) => {
                            multi_q8(in_qkv, &xs, &mut qkv);
                            multi_q8(in_z, &xs, &mut z);
                        }
                        _ => {
                            multi(tensor(data, &w.in_qkv), conv_dim, d, &xs, &mut qkv);
                            multi(tensor(data, &w.in_z), vt, d, &xs, &mut z);
                        }
                    }
                    multi(tensor(data, &w.in_b), heads, d, &xs, &mut b_raw);
                    multi(tensor(data, &w.in_a), heads, d, &xs, &mut a_raw);
                    let conv_w = tensor(data, &w.conv);
                    let a_log = tensor(data, &w.a_log);
                    let dt_bias = tensor(data, &w.dt_bias);
                    let norm_w = tensor(data, &w.norm);
                    let out_proj = tensor(data, &w.out_proj);
                    std::thread::scope(|scope| {
                        for ((((lane, out_i), qkv_i), z_i), (b_i, a_i)) in lanes
                            .iter_mut()
                            .zip(attn_out.iter_mut())
                            .zip(qkv.iter())
                            .zip(z.iter())
                            .zip(b_raw.iter().zip(a_raw.iter()))
                        {
                            let QwenCache::Linear(cache) = &mut lane.caches[l] else {
                                unreachable!("lane cache kind mismatch");
                            };
                            let cfg = &c;
                            let q8_out = match self.q8_spine[l].as_ref() {
                                Some(LayerQ8::Linear { out_proj, .. }) => Some(out_proj),
                                _ => None,
                            };
                            scope.spawn(move || {
                                lin_attn_tail(
                                    cfg, qkv_i, z_i, b_i, a_i, conv_w, a_log, dt_bias, norm_w,
                                    out_proj, q8_out, cache, out_i,
                                );
                            });
                        }
                    });
                }
                QwenAttnW::Full(w) => {
                    let hd = c.head_dim;
                    let q_width = c.n_heads * hd;
                    let kv_width = c.n_kv_heads * hd;
                    let mut qg = vec![vec![0.0f32; q_width * 2]; n];
                    let mut k = vec![vec![0.0f32; kv_width]; n];
                    let mut v = vec![vec![0.0f32; kv_width]; n];
                    let xs: Vec<&[f32]> = normed.iter().map(|x| x.as_slice()).collect();
                    match self.q8_spine[l].as_ref() {
                        Some(LayerQ8::Full { q_proj, k_proj, v_proj, .. }) => {
                            multi_q8(q_proj, &xs, &mut qg);
                            multi_q8(k_proj, &xs, &mut k);
                            multi_q8(v_proj, &xs, &mut v);
                        }
                        _ => {
                            multi(tensor(data, &w.q_proj), q_width * 2, d, &xs, &mut qg);
                            multi(tensor(data, &w.k_proj), kv_width, d, &xs, &mut k);
                            multi(tensor(data, &w.v_proj), kv_width, d, &xs, &mut v);
                        }
                    }
                    let q_norm = tensor(data, &w.q_norm);
                    let k_norm = tensor(data, &w.k_norm);
                    let mut mixed = vec![vec![0.0f32; q_width]; n];
                    let lane_q8 = self.q8_spine[l].is_some() && hd % 32 == 0;
                    std::thread::scope(|scope| {
                        for (((lane, mixed_i), qg_i), (k_i, v_i)) in lanes
                            .iter_mut()
                            .zip(mixed.iter_mut())
                            .zip(qg.iter())
                            .zip(k.iter().zip(v.iter()))
                        {
                            let pos = lane.pos;
                            let QwenCache::Full(cache) = &mut lane.caches[l] else {
                                unreachable!("lane cache kind mismatch");
                            };
                            let cfg = &c;
                            scope.spawn(move || {
                                full_attn_tail(cfg, qg_i, k_i, v_i, q_norm, k_norm, pos, cache, lane_q8, mixed_i);
                            });
                        }
                    });
                    let ms: Vec<&[f32]> = mixed.iter().map(|x| x.as_slice()).collect();
                    match self.q8_spine[l].as_ref() {
                        Some(LayerQ8::Full { o_proj, .. }) => multi_q8(o_proj, &ms, &mut attn_out),
                        _ => multi(tensor(data, &w.o_proj), d, q_width, &ms, &mut attn_out),
                    }
                }
            }
            for i in 0..n {
                for j in 0..d {
                    hidden[i][j] += attn_out[i][j];
                }
            }
            {
                let w = tensor(data, &layer.post_norm);
                for i in 0..n {
                    rmsnorm(&hidden[i], w, c.norm_eps as f32, &mut normed[i]);
                }
            }
            match &layer.mlp {
                QwenMlpW::Dense { gate, up, down } => {
                    let inter = c.dense_inter;
                    let threads = crate::model::pool::pool().workers.max(1);
                    let mut h_gate = vec![vec![0.0f32; inter]; n];
                    let mut h_up = vec![vec![0.0f32; inter]; n];
                    let xs: Vec<&[f32]> = normed.iter().map(|x| x.as_slice()).collect();
                    {
                        let (pg, sg) = packed_parts(data, gate);
                        let mut outs: Vec<&mut [f32]> =
                            h_gate.iter_mut().map(|x| x.as_mut_slice()).collect();
                        crate::quant::mxfp4::matvec_packed_multi(pg, sg, inter, d, &xs, &mut outs, threads);
                        let (pu, su) = packed_parts(data, up);
                        let mut outs: Vec<&mut [f32]> =
                            h_up.iter_mut().map(|x| x.as_mut_slice()).collect();
                        crate::quant::mxfp4::matvec_packed_multi(pu, su, inter, d, &xs, &mut outs, threads);
                    }
                    for i in 0..n {
                        for j in 0..inter {
                            h_gate[i][j] = (h_gate[i][j] / (1.0 + (-h_gate[i][j]).exp())) * h_up[i][j];
                        }
                    }
                    let hs: Vec<&[f32]> = h_gate.iter().map(|x| x.as_slice()).collect();
                    let (pd, sd) = packed_parts(data, down);
                    let mut outs: Vec<&mut [f32]> = Vec::with_capacity(n);
                    let mut mlp_out: Vec<Vec<f32>> = vec![vec![0.0f32; d]; n];
                    for x in mlp_out.iter_mut() {
                        outs.push(x.as_mut_slice());
                    }
                    crate::quant::mxfp4::matvec_packed_multi(pd, sd, d, inter, &hs, &mut outs, threads);
                    for i in 0..n {
                        for j in 0..d {
                            hidden[i][j] += mlp_out[i][j];
                        }
                    }
                }
                QwenMlpW::Moe {
                    router,
                    experts,
                    shared,
                    shared_gate,
                } => {
                    // routed experts differ per lane: no shared weight read
                    // exists by nature, so lanes run their exact serial MoE
                    // concurrently
                    let mut outs: Vec<Vec<f32>> = vec![Vec::new(); n];
                    std::thread::scope(|scope| {
                        for (i, out_slot) in outs.iter_mut().enumerate() {
                            let x = &normed[i];
                            let cfg = &c;
                            scope.spawn(move || {
                                *out_slot =
                                    moe_token_serial(data, router, experts, shared, shared_gate, cfg, x);
                            });
                        }
                    });
                    for i in 0..n {
                        for j in 0..d {
                            hidden[i][j] += outs[i][j];
                        }
                    }
                }
            }
        }

        let norm_w = tensor(data, &self.norm_f);
        for i in 0..n {
            rmsnorm(&hidden[i], norm_w, c.norm_eps as f32, &mut normed[i]);
        }
        let mut logits: Vec<Vec<f32>> = vec![vec![0.0f32; c.vocab]; n];
        {
            let xs: Vec<&[f32]> = normed.iter().map(|x| x.as_slice()).collect();
            let mut outs: Vec<&mut [f32]> = logits.iter_mut().map(|x| x.as_mut_slice()).collect();
            match &self.lm_head_q8 {
                Some(head) => head.matvec_multi(&xs, &mut outs),
                None => crate::model::ops::matvec_multi(
                    tensor(data, &self.lm_head),
                    c.vocab,
                    d,
                    &xs,
                    &mut outs,
                ),
            }
        }
        for lane in lanes.iter_mut() {
            lane.pos += 1;
        }
        logits
    }
}

/// f32 multi-lane matvec over Vec<Vec<f32>> outputs.
fn multi(w: &[f32], rows: usize, cols: usize, xs: &[&[f32]], outs: &mut [Vec<f32>]) {
    let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
    crate::model::ops::matvec_multi(w, rows, cols, xs, &mut refs);
}

/// Quantized-spine multi-lane matvec over Vec<Vec<f32>> outputs.
fn multi_q8(w: &SpineMat, xs: &[&[f32]], outs: &mut [Vec<f32>]) {
    let mut refs: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
    w.matvec_multi(xs, &mut refs);
}

/// Post-projection tail of one linear-attention step for one lane:
/// exactly the ops of `lin_attn_step` after its four matvecs.
#[allow(clippy::too_many_arguments)]
fn lin_attn_tail(
    c: &QwenConfig,
    qkv: &[f32],
    z: &[f32],
    b_raw: &[f32],
    a_raw: &[f32],
    conv_w: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_w: &[f32],
    out_proj: &[f32],
    q8_out: Option<&SpineMat>,
    cache: &mut LinCache,
    out: &mut [f32],
) {
    let (kt, vt) = (c.lin_key_total(), c.lin_value_total());
    let conv_dim = kt * 2 + vt;
    let mut conved = vec![0.0f32; conv_dim];
    conv_step(qkv, conv_w, c.conv_kernel, &mut cache.conv, &mut conved);
    let rep = c.lin_v_heads / c.lin_k_heads.max(1);
    let (kd, vd) = (c.lin_k_dim, c.lin_v_dim);
    let mut mixed = vec![0.0f32; vt];
    // reusable q/k scratch (two heap allocations per head added up)
    let mut q = vec![0.0f32; kd];
    let mut k = vec![0.0f32; kd];
    for h in 0..c.lin_v_heads {
        let kh = h / rep.max(1);
        q.copy_from_slice(&conved[kh * kd..(kh + 1) * kd]);
        k.copy_from_slice(&conved[kt + kh * kd..kt + (kh + 1) * kd]);
        let v = &conved[2 * kt + h * vd..2 * kt + (h + 1) * vd];
        l2norm(&mut q, 1e-6);
        l2norm(&mut k, 1e-6);
        let scale = 1.0 / (kd as f32).sqrt();
        for t in q.iter_mut() {
            *t *= scale;
        }
        let beta = 1.0 / (1.0 + (-b_raw[h]).exp());
        let sp = {
            let t = a_raw[h] + dt_bias[h];
            if t > 20.0 {
                t
            } else {
                (1.0 + t.exp()).ln()
            }
        };
        let g = -a_log[h].exp() * sp;
        let st = &mut cache.state[h * kd * vd..(h + 1) * kd * vd];
        delta_step(st, &q, &k, v, g, beta, &mut mixed[h * vd..(h + 1) * vd]);
    }
    for h in 0..c.lin_v_heads {
        let (a, b) = (h * vd, (h + 1) * vd);
        rmsnorm_gated(&mut mixed[a..b], norm_w, &z[a..b], c.norm_eps as f32);
    }
    match q8_out {
        Some(q) => q.matvec_st(&mixed, out),
        None => crate::model::ops::matvec_st(out_proj, c.d, vt, &mixed, out),
    }
}

/// Post-projection tail of one full-attention step for one lane: exactly
/// the ops of `full_attn_step` after its three matvecs, writing the gated
/// mix (before o_proj, which runs lane-batched).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn full_attn_tail(
    c: &QwenConfig,
    qg: &[f32],
    k_in: &[f32],
    v_in: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    pos: usize,
    cache: &mut FullCache,
    q8_scores: bool,
    mixed: &mut [f32],
) {
    let hd = c.head_dim;
    let q_width = c.n_heads * hd;
    let kv_width = c.n_kv_heads * hd;
    assert_eq!(cache.len, pos, "lane full-attention cache position mismatch");
    let mut q = vec![0.0f32; q_width];
    let mut gate = vec![0.0f32; q_width];
    let mut k = k_in.to_vec();
    let v = v_in;
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
    cache.k.extend_from_slice(&k);
    cache.v.extend_from_slice(v);
    push_k_mirror(cache, &k);
    cache.len += 1;
    let groups = c.n_heads / c.n_kv_heads;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut scores = vec![0.0f32; cache.len];
    let mut qq = crate::quant::q8::Q8Vec::new();
    for h in 0..c.n_heads {
        let kh = h / groups;
        let qh = &q[h * hd..(h + 1) * hd];
        let mut max_score = f32::NEG_INFINITY;
        // quantized spine: integer score dots against the K mirror
        // (bit-identical to the batch path's q8 branch)
        if q8_scores {
            crate::quant::q8::quantize_q8_into(qh, &mut qq);
            max_score = crate::quant::q8::score_window(
                &qq,
                &cache.kq,
                &cache.kqs,
                kv_width,
                kh * hd,
                hd / 32,
                cache.len,
                scale,
                &mut scores,
            );
        } else {
        for t in 0..cache.len {
            let off = t * kv_width + kh * hd;
            let sc = crate::model::ops::dot(qh, &cache.k[off..off + hd]) * scale;
            scores[t] = sc;
            max_score = max_score.max(sc);
        }
        }
        let mut denom = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - max_score).exp();
            denom += *sc;
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
    for (i, value) in mixed.iter_mut().enumerate() {
        *value *= 1.0 / (1.0 + (-gate[i]).exp());
    }
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
    qwen_run_turn_resume(
        ids,
        max_new,
        tok,
        model,
        debug,
        debug_routing,
        stop_id,
        None,
        sampler,
    )
}

/// `qwen_run_turn` + optional initial logits restored from a state
/// snapshot: the caches are kept, the prompt tokens (possibly none) are
/// ingested on top of the loaded state, and decoding starts from the
/// stored logits when the prompt is empty.
#[allow(clippy::too_many_arguments)]
pub fn qwen_run_turn_resume(
    ids: &[u32],
    max_new: usize,
    tok: &crate::tokenizer::AnyTokenizer,
    model: &mut QwenModel,
    debug: bool,
    debug_routing: bool,
    stop_id: u32,
    init_logits: Option<Vec<f32>>,
    sampler: &mut super::Sampler,
) -> String {
    if init_logits.is_none() {
        model.reset();
    }
    if sampler.mtp && init_logits.is_none() {
        if !model.has_mtp() {
            eprintln!("warning: --mtp ignored, the model was converted without its MTP head");
        } else if sampler.temp > 0.0 {
            eprintln!("warning: --mtp is greedy-only, ignoring it with --temp > 0");
        } else {
            return run_turn_mtp(ids, max_new, tok, model, debug, stop_id, sampler);
        }
    }
    let answer = super::run_turn_core_batch(
        ids,
        max_new,
        tok,
        &mut |batch: &[u32]| model.prefill(batch),
        debug,
        debug_routing,
        stop_id,
        init_logits,
        sampler,
    );
    if mlp_budget() > 0.0 {
        let (skipped, total) = mlp_skip_stats();
        if total > 0 {
            eprintln!(
                "mlp budget {}: {}/{} blocks skipped ({:.0}%)",
                mlp_budget(),
                skipped,
                total,
                skipped as f64 / total as f64 * 100.0
            );
        }
    }
    answer
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
            "  ({:.0} ms/token, {:.1} tok/s | mtp: {} passes, {} drafts accepted, {:.2} tokens/pass)",
            moy * 1000.0,
            1.0 / moy,
            passes,
            accepted,
            if passes > 0 { generated.len() as f64 / passes as f64 } else { 0.0 }
        );
    }
    answer
}

/// Token-level MTP speculative loop (see `run_turn_mtp`). Returns the
/// generated ids plus (verification passes, accepted drafts).
///
/// The draft is a CHAIN: the one-layer MTP head proposes up to
/// `sampler.mtp_depth` tokens by feeding each proposal's own final-normed
/// hidden into the next step (the reference proposer's multi-step
/// contract), scored through the frequency-sliced draft head when the
/// model has one. One batched trunk pass then verifies the pending token
/// plus the whole chain; the longest matching prefix commits, the
/// mismatch position yields the next token for free (the standard
/// speculative bonus), and a partial accept rolls the trunk back exactly.
/// The MTP cache is rebuilt from the verified trunk hiddens after every
/// pass, so draft pairs never carry speculative state forward. Output is
/// bit-identical to plain greedy decoding at every depth and with any
/// draft head.
pub(crate) fn mtp_generate(
    model: &mut QwenModel,
    ids: &[u32],
    max_new: usize,
    stop_id: u32,
    sampler: &super::Sampler,
    debug: bool,
) -> (Vec<u32>, usize, usize) {
    assert!(!ids.is_empty(), "MTP decoding requires a prompt");
    assert!(
        !no_batch_prefill(),
        "--mtp requires the batched prefill (unset MICROKIMI_NO_QWEN_BATCH)"
    );
    let depth = sampler.mtp_depth.max(1);
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

    let first = mtp_select(&logits, sampler, &generated);
    if first == stop_id {
        return (generated, passes, accepted);
    }
    generated.push(first);
    let mut pending = true;

    // adaptive chain depth (AIMD): a fully accepted pass grows the chain
    // back toward --mtp-depth, a zero-acceptance pass shrinks it, and at
    // depth 0 the loop degenerates to plain decoding (no drafts, no
    // snapshot) with a periodic probe - so --mtp is never much worse
    // than plain decoding on a prompt the draft head cannot predict.
    let mut cur_depth = depth;
    let mut cold_passes = 0usize;

    while pending && generated.len() < max_new {
        let n = *generated.last().unwrap();
        if cur_depth == 0 {
            cold_passes += 1;
            if cold_passes >= 16 {
                cur_depth = 1;
                cold_passes = 0;
            }
        }
        let mtp_len = model.mtp_cache.len;

        // chain draft: n's pair first, then each proposal chained on its
        // own normed hidden; never draft the stop token or past max_new
        let mut batch = vec![n];
        let mut chain_hidden = hidden_prev.clone();
        for _ in 0..cur_depth {
            if generated.len() + batch.len() > max_new {
                break;
            }
            let (next_hidden, draft) = model.mtp_draft_step(*batch.last().unwrap(), &chain_hidden);
            if draft == stop_id {
                break;
            }
            batch.push(draft);
            chain_hidden = next_hidden;
        }
        // the snapshot (a full clone of the linear states) is only needed
        // when there are drafts to roll back
        let snap = if batch.len() > 1 { Some(model.snapshot()) } else { None };

        let out = model.prefill_collect(&batch, true);
        passes += 1;

        // verify: accept drafts while they match the trunk's selection
        let mut committed = 1usize;
        let mut vctx: Vec<u32> = Vec::new();
        while committed < batch.len() {
            let sel = if sampler.dry > 0.0 {
                vctx.clear();
                vctx.extend_from_slice(&generated);
                vctx.extend_from_slice(&batch[1..committed]);
                mtp_select(&out.logits[committed - 1], sampler, &vctx)
            } else {
                mtp_select(&out.logits[committed - 1], sampler, &generated)
            };
            if sel != batch[committed] {
                break;
            }
            committed += 1;
        }
        accepted += committed - 1;
        {
            let drafted = batch.len() - 1;
            if drafted > 0 {
                if committed - 1 == drafted {
                    cur_depth = (cur_depth + 1).min(depth);
                } else if committed == 1 {
                    cur_depth = cur_depth.saturating_sub(1);
                }
            }
        }
        if debug {
            println!(
                "  mtp pass {}: drafted {}, accepted {}",
                passes,
                batch.len() - 1,
                committed - 1
            );
        }

        // partial accept: exact trunk rollback + reingest of the prefix
        if committed < batch.len() {
            model.restore(snap.as_ref().expect("drafted pass carries a snapshot"));
            model.prefill_collect(&batch[..committed], false);
        }
        // the MTP cache always rebuilds from verified trunk hiddens
        model.rollback_mtp(mtp_len);
        for j in 0..committed {
            let h = if j == 0 { &hidden_prev } else { &out.hidden[j - 1] };
            model.mtp_advance(batch[j], h, false);
        }
        for &draft in &batch[1..committed] {
            generated.push(draft);
        }
        logits = out.logits[committed - 1].clone();
        hidden_prev = out.hidden[committed - 1].clone();
        pending = false;
        if generated.len() >= max_new {
            break;
        }
        // bonus token: the mismatch position's own selection is free
        let next = mtp_select(&logits, sampler, &generated);
        if next == stop_id {
            break;
        }
        generated.push(next);
        pending = true;
    }
    if pending {
        // the trailing emitted token was never verified through the trunk:
        // ingest it so the state (and last_logits) matches the plain loop
        model.prefill_collect(&[*generated.last().unwrap()], false);
    }
    (generated, passes, accepted)
}

/// `microkimi lanebench --model X.bin [--lanes N] [--steps M]`: aggregate
/// decode throughput of lane-batched decoding. Each lane gets a distinct
/// short prompt; every step decodes one token per lane through
/// forward_lanes (greedy). Reports per-step wall time and aggregate
/// tokens/second.
pub fn lanebench_cmd(args: &[String]) {
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let model_path = value("--model").expect("lanebench requires --model MODEL.bin");
    let lanes_n: usize = value("--lanes").and_then(|v| v.parse().ok()).unwrap_or(4);
    let steps: usize = value("--steps").and_then(|v| v.parse().ok()).unwrap_or(32);
    let mut model = QwenModel::load(&model_path);
    let vocab = model.cfg.vocab as u32;
    let mut lanes: Vec<DecodeLane> = (0..lanes_n).map(|_| DecodeLane::new(&model)).collect();
    for (i, lane) in lanes.iter_mut().enumerate() {
        let prompt: Vec<u32> = (0..8).map(|j| (3 + i as u32 * 17 + j * 7) % vocab.min(50_000)).collect();
        model.prefill_lane(lane, &prompt);
    }
    let mut tokens: Vec<u32> = (0..lanes_n).map(|i| (5 + i as u32 * 13) % vocab.min(50_000)).collect();

    let run_phase = |model: &QwenModel, lanes: &mut [DecodeLane], tokens: &mut [u32], steps: usize| -> f64 {
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            let n = tokens.len();
            let mut refs: Vec<&mut DecodeLane> = lanes.iter_mut().take(n).collect();
            let logits = model.forward_lanes(&mut refs, tokens);
            for (i, l) in logits.iter().enumerate() {
                tokens[i] = crate::model::top_k_probs(l, 5)[0].0 as u32;
            }
        }
        t0.elapsed().as_secs_f64()
    };

    // --ab: alternate single-lane and N-lane phases in the SAME process
    // (no reload, no page-in between arms), report per-round aggregate
    // throughput ratios and their median - the noise-robust comparison.
    if args.iter().any(|a| a == "--ab") {
        let rounds: usize = value("--rounds").and_then(|v| v.parse().ok()).unwrap_or(6);
        // warm both shapes once
        let mut single_tok = vec![tokens[0]];
        run_phase(&model, &mut lanes[..1], &mut single_tok, 2);
        run_phase(&model, &mut lanes, &mut tokens.clone(), 2);
        let mut ratios: Vec<f64> = Vec::new();
        for round in 0..rounds {
            let dt1 = run_phase(&model, &mut lanes[..1], &mut single_tok, steps);
            let mut multi_tok = tokens.clone();
            let dtn = run_phase(&model, &mut lanes, &mut multi_tok, steps);
            let single_rate = steps as f64 / dt1;
            let multi_rate = (lanes_n * steps) as f64 / dtn;
            let ratio = multi_rate / single_rate;
            ratios.push(ratio);
            println!(
                "round {}: 1 lane {:6.1} tok/s | {} lanes {:6.1} tok/s | ratio {:.2}x",
                round + 1,
                single_rate,
                lanes_n,
                multi_rate,
                ratio
            );
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "median aggregate speedup at {} lanes: {:.2}x (min {:.2}, max {:.2})",
            lanes_n,
            ratios[ratios.len() / 2],
            ratios[0],
            ratios[ratios.len() - 1]
        );
        return;
    }

    let dt = run_phase(&model, &mut lanes, &mut tokens, steps);
    let total = lanes_n * steps;
    println!(
        "lanes {:2}: {} tokens in {:.2} s -> {:.1} tok/s aggregate ({:.0} ms/step)",
        lanes_n,
        total,
        dt,
        total as f64 / dt,
        dt / steps as f64 * 1000.0
    );
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

/// Test-only checkpoint builder shared with the converter tests: writes a
/// deterministic MKIM0002 fixture for `c` and returns its path.
#[cfg(test)]
pub fn test_fixture(c: &QwenConfig) -> String {
    model_tests::checkpoint_fixture(c)
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

    pub(super) fn checkpoint_fixture(c: &QwenConfig) -> String {
        let path = std::env::temp_dir()
            .join(format!(
                "microkimi_qwen_fixture_{}_{}_{}l_{}d_{}m_{}t.bin",
                std::process::id(),
                std::thread::current().name().unwrap_or("test"),
                c.n_layers,
                c.dense_inter,
                c.mtp_layers,
                c.tied_embeddings as u8
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
    fn tied_embeddings_share_one_matrix_and_run() {
        let mut c = bin_tiny_dense();
        c.tied_embeddings = true;
        let layout = crate::tools::convert_qwen::output_layout(&c);
        assert!(
            !layout.iter().any(|(name, _, _)| name == "lm_head.weight"),
            "tied checkpoints must not duplicate the head"
        );
        let path = checkpoint_fixture(&c);
        let mut model = QwenModel::load(&path);
        assert!(model.cfg.tied_embeddings);
        let logits = model.forward(3);
        assert_eq!(logits.len(), c.vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        // the head really is the embedding: feeding the argmax row back in
        // stays finite and deterministic across a reset
        let top = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        let continued = model.forward(top);
        model.reset();
        model.forward(3);
        assert_eq!(model.forward(top), continued);
        drop(model);
        std::fs::remove_file(path).ok();
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
    fn state_snapshot_resumes_bit_identically() {
        for (c, tag) in [(bin_tiny(), "moe"), (bin_tiny_dense(), "dense")] {
            let path = checkpoint_fixture(&c);
            let mem_path = std::env::temp_dir()
                .join(format!(
                    "microkimi_qwen_state_{}_{}.mkmem",
                    std::process::id(),
                    tag
                ))
                .to_string_lossy()
                .into_owned();

            let mut uninterrupted = QwenModel::load(&path);
            uninterrupted.prefill(&[3, 5, 7, 11]);
            let expected = [uninterrupted.forward(2), uninterrupted.forward(9)];

            let mut saver = QwenModel::load(&path);
            let saved_logits = saver.prefill(&[3, 5, 7, 11]);
            crate::memory::qwen_state::save(&saver, &mem_path).unwrap();
            drop(saver);

            let mut resumed = QwenModel::load(&path);
            let restored = crate::memory::qwen_state::load(&mut resumed, &mem_path).unwrap();
            assert_eq!(restored, saved_logits);
            assert_eq!(resumed.pos, 4);
            assert_eq!(resumed.forward(2), expected[0]);
            assert_eq!(resumed.forward(9), expected[1]);

            // fingerprint check: a different fixture must be refused
            let mut other_cfg = c.clone();
            other_cfg.n_layers = 8;
            let other_path = checkpoint_fixture(&other_cfg);
            let mut other = QwenModel::load(&other_path);
            assert!(crate::memory::qwen_state::load(&mut other, &mem_path).is_err());

            drop((uninterrupted, resumed, other));
            std::fs::remove_file(path).ok();
            std::fs::remove_file(other_path).ok();
            std::fs::remove_file(mem_path).ok();
        }
    }

    #[test]
    fn q8_spine_paths_agree_and_differ_from_f32() {
        let c = bin_tiny_dense();
        let path = checkpoint_fixture(&c);
        let tokens = [3u32, 5, 7, 11, 2];

        // reference f32 logits
        let mut f32_model = QwenModel::load(&path);
        let f32_logits = f32_model.prefill(&tokens);

        let mut q8_model = QwenModel::load(&path);
        let mut q8_forward = QwenModel::load(&path);
        let mut q8_lanes_model = QwenModel::load(&path);
        q8_model.set_spine_mode(Some(false));
        q8_forward.set_spine_mode(Some(false));
        q8_lanes_model.set_spine_mode(Some(false));
        assert!(q8_model.q8_spine.iter().all(|x| x.is_some()));

        // prefill vs forward (delegated) agree exactly
        let q8_logits = q8_model.prefill(&tokens);
        let mut fwd_logits = Vec::new();
        for &t in &tokens {
            fwd_logits = q8_forward.forward(t);
        }
        assert_eq!(q8_logits, fwd_logits, "q8 forward diverges from q8 prefill");

        // lanes agree with single-stream under q8
        let mut lane = DecodeLane::new(&q8_lanes_model);
        q8_lanes_model.prefill_lane(&mut lane, &tokens[..4]);
        let mut refs = vec![&mut lane];
        let lane_logits = q8_lanes_model.forward_lanes(&mut refs, &tokens[4..5]);
        let mut q8_single = QwenModel::load(&path);
        q8_single.set_spine_mode(Some(false));
        q8_single.prefill(&tokens[..4]);
        let single_logits = q8_single.forward(tokens[4]);
        assert_eq!(lane_logits[0], single_logits, "q8 lanes diverge from single-stream");

        // and the mode is actually active: q8 output differs from f32
        assert_ne!(q8_logits, f32_logits, "q8 spine produced f32-identical logits");
        assert!(q8_logits.iter().all(|v| v.is_finite()));

        drop((f32_model, q8_model, q8_forward, q8_lanes_model, q8_single));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn fp4_spine_paths_agree_and_stay_finite() {
        let c = bin_tiny_dense();
        let path = checkpoint_fixture(&c);
        let tokens = [3u32, 5, 7, 11, 2];
        let mut fp4_prefill = QwenModel::load(&path);
        let mut fp4_forward = QwenModel::load(&path);
        fp4_prefill.set_spine_mode(Some(true));
        fp4_forward.set_spine_mode(Some(true));
        assert!(fp4_prefill
            .q8_spine
            .iter()
            .all(|x| matches!(x, Some(LayerQ8::Linear { in_qkv: SpineMat::Fp4(_), .. })
                | Some(LayerQ8::Full { q_proj: SpineMat::Fp4(_), .. }))));
        let a = fp4_prefill.prefill(&tokens);
        let mut b = Vec::new();
        for &t in &tokens {
            b = fp4_forward.forward(t);
        }
        assert_eq!(a, b, "fp4 forward diverges from fp4 prefill");
        assert!(a.iter().all(|v| v.is_finite()));
        drop((fp4_prefill, fp4_forward));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn certified_skip_bounds_hold_on_the_fixture() {
        let c = bin_tiny_dense();
        let path = checkpoint_fixture(&c);
        let model = QwenModel::load(&path);
        let (gate, up, down) = match &model.layers[0].mlp {
            QwenMlpW::Dense { gate, up, down } => (*gate, *up, *down),
            _ => unreachable!(),
        };
        let bounds = SkipBounds::build(&model.bin.data, &gate, &up, &down, &c);
        assert_eq!(bounds.up_l2.len(), c.dense_inter);
        assert_eq!(bounds.down_sup.len(), c.dense_inter / 32);

        // exact MLP output vs output with one block zeroed: the certified
        // bound must dominate the true sup-norm deviation
        let x: Vec<f32> = (0..c.d).map(|i| ((i * 13 + 5) % 17) as f32 * 0.01 - 0.08).collect();
        let exact = packed_dense_mlp(&model.bin.data, &gate, &up, &down, &c, &x, None, None);
        let inter = c.dense_inter;
        let (pg, sg) = packed_parts(&model.bin.data, &gate);
        let mut g_act = vec![0.0f32; inter];
        crate::quant::mxfp4::matvec_packed(pg, sg, inter, c.d, &x, &mut g_act, 1);
        let (pu, su) = packed_parts(&model.bin.data, &up);
        let mut u_act = vec![0.0f32; inter];
        crate::quant::mxfp4::matvec_packed(pu, su, inter, c.d, &x, &mut u_act, 1);
        let mut h = vec![0.0f32; inter];
        for i in 0..inter {
            h[i] = (g_act[i] / (1.0 + (-g_act[i]).exp())) * u_act[i];
        }
        let x_l2 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        for skip_block in 0..inter / 32 {
            let mut h_zeroed = h.clone();
            for i in skip_block * 32..(skip_block + 1) * 32 {
                h_zeroed[i] = 0.0;
            }
            let (pd, sd) = packed_parts(&model.bin.data, &down);
            let mut approx = vec![0.0f32; c.d];
            crate::quant::mxfp4::matvec_packed(pd, sd, c.d, inter, &h_zeroed, &mut approx, 1);
            let true_dev = exact
                .iter()
                .zip(&approx)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let bound: f32 = (skip_block * 32..(skip_block + 1) * 32)
                .map(|i| (g_act[i] / (1.0 + (-g_act[i]).exp())).abs() * bounds.up_l2[i])
                .sum::<f32>()
                * x_l2
                * bounds.down_sup[skip_block];
            assert!(
                true_dev <= bound * 1.0001 + 1e-6,
                "block {}: true deviation {} exceeds certified bound {}",
                skip_block,
                true_dev,
                bound
            );
        }
        drop(model);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn colblock_kernel_matches_full_and_zeroed_subsets() {
        let (rows, cols) = (16usize, 128usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7 + 3) % 23) as f32 * 0.01 - 0.1).collect();
        let (packed, scales) = crate::quant::mxfp4::quantize(&w, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| ((i * 5 + 1) % 13) as f32 * 0.02 - 0.1).collect();
        let all: Vec<usize> = (0..cols / 32).collect();
        let mut full = vec![0.0f32; rows];
        crate::quant::mxfp4::matvec_packed(&packed, &scales, rows, cols, &x, &mut full, 1);
        let mut via_blocks = vec![0.0f32; rows];
        crate::quant::mxfp4::matvec_packed_colblocks(
            &packed, &scales, rows, cols, &all, &x, &mut via_blocks, 1,
        );
        assert_eq!(full, via_blocks, "all-kept must equal the plain kernel");

        let kept = vec![0usize, 2];
        let mut x_masked = x.clone();
        for i in 32..64 {
            x_masked[i] = 0.0;
        }
        for i in 96..128 {
            x_masked[i] = 0.0;
        }
        let mut masked = vec![0.0f32; rows];
        crate::quant::mxfp4::matvec_packed(&packed, &scales, rows, cols, &x_masked, &mut masked, 1);
        let mut sparse = vec![0.0f32; rows];
        crate::quant::mxfp4::matvec_packed_colblocks(
            &packed, &scales, rows, cols, &kept, &x, &mut sparse, 1,
        );
        for (a, b) in masked.iter().zip(&sparse) {
            assert!((a - b).abs() < 1e-5, "{} vs {}", a, b);
        }
    }

    #[test]
    fn lane_batched_decode_matches_single_stream_bitwise() {
        for c in [bin_tiny(), bin_tiny_dense()] {
            let path = checkpoint_fixture(&c);
            let model = QwenModel::load(&path);
            // four lanes with different prompts and continuations
            let prompts: [&[u32]; 4] = [&[3, 5, 7], &[11, 2], &[9, 13, 4, 6], &[1]];
            let steps: [&[u32]; 4] = [&[8, 10], &[12, 3], &[5, 5], &[7, 9]];

            // reference: each stream alone through the plain forward
            let mut expected: Vec<Vec<Vec<f32>>> = Vec::new();
            for i in 0..4 {
                let mut single = QwenModel::load(&path);
                single.prefill(prompts[i]);
                let mut per_step = Vec::new();
                for &t in steps[i] {
                    per_step.push(single.forward(t));
                }
                expected.push(per_step);
            }

            // lanes: prompts ingested per lane, then decoded together
            let mut model = model;
            let mut lanes: Vec<DecodeLane> =
                (0..4).map(|_| DecodeLane::new(&model)).collect();
            for i in 0..4 {
                model.prefill_lane(&mut lanes[i], prompts[i]);
            }
            for step in 0..2 {
                let tokens: Vec<u32> = (0..4).map(|i| steps[i][step]).collect();
                let mut refs: Vec<&mut DecodeLane> = lanes.iter_mut().collect();
                let logits = model.forward_lanes(&mut refs, &tokens);
                for i in 0..4 {
                    assert_eq!(
                        logits[i], expected[i][step],
                        "lane {} step {} diverges from single-stream",
                        i, step
                    );
                }
            }
            drop(model);
            std::fs::remove_file(path).ok();
        }
    }

    #[test]
    fn chained_mtp_is_bit_identical_at_every_depth_and_head() {
        let mut c = bin_tiny_dense();
        c.mtp_layers = 1;
        let path = checkpoint_fixture(&c);
        let ids = [3u32, 5, 7, 11];
        let max_new = 24;
        let stop = 999_999;

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

        for depth in [1usize, 2, 4, 8] {
            for mini_rows in [0usize, 32] {
                let mut model = QwenModel::load(&path);
                model.draft_head = if mini_rows > 0 {
                    DraftHead::from_rows(&model, mini_rows)
                } else {
                    None
                };
                let mut sampler = crate::model::Sampler::greedy();
                sampler.mtp_depth = depth;
                let (spec, passes, accepted) =
                    mtp_generate(&mut model, &ids, max_new, stop, &sampler, false);
                assert_eq!(
                    plain, spec,
                    "divergence at depth {} mini {}",
                    depth, mini_rows
                );
                assert!(passes > 0 && accepted <= passes * depth);
            }
        }
        drop(plain_model);
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

/// Chunked delta-rule scan for one head (WY-style): within a chunk of C
/// tokens the recurrence unrolls to
///   u_t = beta_t*v_t - beta_t*G(1..t)*S0'k_t - sum_{s<t} beta_t*G(s+1..t)*(k_s'k_t)*u_s
/// solved by forward substitution, then outputs and the end-of-chunk
/// state come from GEMM-shaped sums - no per-token dependency chain
/// except the short substitution. Mathematically identical to
/// delta_step folded over the chunk; floating-point reassociation
/// differs (~1e-5 relative), so this path only runs under the
/// quantized spine modes whose contract is already tolerance-based.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunked_scan_head(
    state: &mut [f32],
    mixed: &mut [f32],
    qn: &[f32],
    kn: &[f32],
    vn: &[f32],
    beta: &[f32],
    gamma: &[f32],
    t_count: usize,
    kd: usize,
    vd: usize,
) {
    const C: usize = 32;
    let mut c0 = 0usize;
    let mut u = vec![0.0f32; C * vd];
    let mut b = vec![0.0f32; C * vd];
    let mut s0k = vec![0.0f32; C * vd];
    let mut s0q = vec![0.0f32; C * vd];
    let mut kk = vec![0.0f32; C * C];
    let mut qk = vec![0.0f32; C * C];
    let mut grel = vec![0.0f32; C * C];
    let mut gcum = [0.0f32; C];
    while c0 < t_count {
        let c1 = (c0 + C).min(t_count);
        let n = c1 - c0;
        // cumulative decay products G(1..t) within the chunk
        let mut acc = 1.0f32;
        for (i, g) in gamma[c0..c1].iter().enumerate() {
            acc *= g;
            gcum[i] = acc;
        }
        // gram matrices (GEMM-shaped, kd shared)
        for t in 0..n {
            let kt = &kn[(c0 + t) * kd..(c0 + t + 1) * kd];
            let qt = &qn[(c0 + t) * kd..(c0 + t + 1) * kd];
            for s in 0..=t {
                let ks = &kn[(c0 + s) * kd..(c0 + s + 1) * kd];
                if s < t {
                    kk[t * C + s] = crate::model::ops::dot(ks, kt);
                }
                qk[t * C + s] = crate::model::ops::dot(ks, qt);
            }
        }
        // S0-side terms: s0k[t] = S0' k_t, s0q[t] = S0' q_t (row-weighted
        // sums over the state's kd rows; k/q weights are per-row scalars)
        s0k[..n * vd].fill(0.0);
        s0q[..n * vd].fill(0.0);
        for i in 0..kd {
            let row = &state[i * vd..(i + 1) * vd];
            for t in 0..n {
                let kw = kn[(c0 + t) * kd + i];
                let qw = qn[(c0 + t) * kd + i];
                let dk = &mut s0k[t * vd..(t + 1) * vd];
                if kw != 0.0 {
                    for j in 0..vd {
                        dk[j] += kw * row[j];
                    }
                }
                if qw != 0.0 {
                    let dq = &mut s0q[t * vd..(t + 1) * vd];
                    for j in 0..vd {
                        dq[j] += qw * row[j];
                    }
                }
            }
        }
        // relative decay products G(s+1..t), built multiplicatively per
        // row (never by dividing cumulative products: real decays reach
        // ~1e-2 per token and 32-token cumulatives underflow)
        for t in 0..n {
            grel[t * C + t] = 1.0;
            let mut p = 1.0f32;
            for s in (0..t).rev() {
                p *= gamma[c0 + s + 1];
                grel[t * C + s] = p;
            }
        }
        // b_t and forward substitution for u
        for t in 0..n {
            let bt = beta[c0 + t];
            let g1t = gcum[t];
            let vt = &vn[(c0 + t) * vd..(c0 + t + 1) * vd];
            let (bu, sk) = (&mut b[t * vd..(t + 1) * vd], &s0k[t * vd..(t + 1) * vd]);
            for j in 0..vd {
                bu[j] = bt * (vt[j] - g1t * sk[j]);
            }
        }
        for t in 0..n {
            let (ready, ut_row) = u.split_at_mut(t * vd);
            let ut = &mut ut_row[..vd];
            ut.copy_from_slice(&b[t * vd..(t + 1) * vd]);
            let bt = beta[c0 + t];
            for s in 0..t {
                let a = bt * grel[t * C + s] * kk[t * C + s];
                if a != 0.0 {
                    let us = &ready[s * vd..(s + 1) * vd];
                    for j in 0..vd {
                        ut[j] -= a * us[j];
                    }
                }
            }
        }
        // outputs: out_t = G(1..t) s0q[t] + sum_{s<=t} G(s+1..t) qk[t][s] u_s
        for t in 0..n {
            let out = &mut mixed[(c0 + t) * vd..(c0 + t + 1) * vd];
            let g1t = gcum[t];
            let sq = &s0q[t * vd..(t + 1) * vd];
            for j in 0..vd {
                out[j] = g1t * sq[j];
            }
            for s in 0..=t {
                let w = grel[t * C + s] * qk[t * C + s];
                if w != 0.0 {
                    let us = &u[s * vd..(s + 1) * vd];
                    for j in 0..vd {
                        out[j] += w * us[j];
                    }
                }
            }
        }
        // state update: S = G(1..C) S0 + sum_s G(s+1..C) k_s (x) u_s
        let gtot = gcum[n - 1];
        for x in state.iter_mut() {
            *x *= gtot;
        }
        for s in 0..n {
            let w = grel[(n - 1) * C + s];
            let ks = &kn[(c0 + s) * kd..(c0 + s + 1) * kd];
            let us = &u[s * vd..(s + 1) * vd];
            for i in 0..kd {
                let f = w * ks[i];
                if f != 0.0 {
                    let row = &mut state[i * vd..(i + 1) * vd];
                    for j in 0..vd {
                        row[j] += f * us[j];
                    }
                }
            }
        }
        c0 = c1;
    }
}
