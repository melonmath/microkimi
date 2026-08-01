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

/// Full MoE forward for one token: route, run selected experts with f32
/// accumulation, add the (single, unweighted) shared expert.
#[allow(clippy::too_many_arguments)]
pub fn moe_forward(
    x: &[f32],
    gate_w: &[f32],
    bias: Option<&[f32]>,
    tid2eid: Option<&[i32]>,
    token_id: u32,
    experts: &dyn Fn(u32, &[f32], &mut [f32]), // (expert_id, x, out[d])
    shared: &dyn Fn(&[f32], &mut [f32]),
    topk: usize,
    route_scale: f32,
    out: &mut [f32],
) {
    let (sel, _) = gate_forward(x, gate_w, bias, tid2eid, token_id, gate_w.len() / x.len(), topk, route_scale);
    let d = x.len();
    let mut acc = vec![0f32; d];
    for (eid, w) in &sel {
        let mut eo = vec![0f32; d];
        experts(*eid, x, &mut eo);
        for j in 0..d {
            acc[j] += w * eo[j];
        }
    }
    let mut so = vec![0f32; d];
    shared(x, &mut so);
    for j in 0..d {
        out[j] = acc[j] + so[j];
    }
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
