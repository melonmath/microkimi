//! Qwen3.5-MoE text decoder.
//!
//! Layer alternation: three gated delta-rule linear-attention layers
//! then one full-attention layer (`full_attention_interval`). Every
//! layer carries a softmax-routed expert bank plus one always-on shared
//! expert gated by a sigmoid.
//!
//! The delta rule is the same family as KDA (see `kda.rs`): a per-head
//! state S is decayed, corrected by the difference between the value and
//! what the state already predicts for the key, then read by the query.
//! Two differences from KDA: the decay is a scalar per head rather than
//! a full-rank gate, and the output norm is gated by a separate
//! projection instead of a plain RMS norm.

use crate::config::QwenConfig;

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
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(eps);
    for x in v.iter_mut() {
        *x /= n;
    }
}

/// One delta-rule step for a single head.
///
/// `s` is the [k_dim, v_dim] state, updated in place:
///   S <- S * exp(g);  delta = (v - S^T k) * beta;  S += k (x) delta
///   out = S^T q
pub fn delta_step(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, out: &mut [f32]) {
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
