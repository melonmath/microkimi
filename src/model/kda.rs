// KDA linear attention: short conv on q/k/v, per-head gated delta-rule
// recurrence (state s updated token by token), gated output norm.
// kda_prefill walks a whole chunk (optionally chunked via kda_chunk) and must
// stay bit-identical to stepping kda_forward one token at a time.

use super::*;

/// Depthwise causal conv (kernel 4) + SiLU for one position.
/// cache_raw = last 3 raw inputs (left shift, push the current raw input).
fn kda_conv_step(cfg: &Config, data: &[u8], vec: &mut [f32], conv_w: &T, cache_raw: &mut Vec<f32>) {
    let w_conv = Model::t(data, conv_w);
    // window = 3 previous raw inputs + current input (weight j=0 → oldest)
    let mut out = vec![0f32; cfg.kda_proj()];
    for c in 0..cfg.kda_proj() {
        let mut acc = 0f32;
        for j in 0..3 {
            acc += w_conv[c * cfg.kda_conv + j] * cache_raw[j * cfg.kda_proj() + c];
        }
        acc += w_conv[c * cfg.kda_conv + 3] * vec[c];
        out[c] = silu(acc);
    }
    cache_raw.copy_within(cfg.kda_proj()..3 * cfg.kda_proj(), 0);
    cache_raw[2 * cfg.kda_proj()..3 * cfg.kda_proj()].copy_from_slice(vec);
    vec.copy_from_slice(&out);
}

/// Per-head L2-norm over kda_dim (eps 1e-12), scaled.
fn kda_norm_head(cfg: &Config, vec: &mut [f32], scale: f32) {
    for h in 0..cfg.kda_heads {
        let head = &mut vec[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let n2 = dot(head, head);
        let inv = scale / n2.sqrt().max(1e-12);
        for x in head.iter_mut() {
            *x *= inv;
        }
    }
}

/// Log-decay gate: g = gate_lb · sigmoid(exp(A_log) · (g_low + dt_bias)).
fn kda_gate(cfg: &Config, a_log: &[f32], dt_bias: &[f32], g_low: &[f32]) -> Vec<f32> {
    let mut g = vec![0f32; cfg.kda_proj()];
    for h in 0..cfg.kda_heads {
        for c in 0..cfg.kda_dim {
            let i = h * cfg.kda_dim + c;
            g[i] = cfg.gate_lb * sigmoid(a_log[c].exp() * (g_low[i] + dt_bias[i]));
        }
    }
    g
}

/// One step of the KDA recurrence on the persistent state S[heads,dim,dim]:
/// decay along K, δ = v - kᵀS, S += (βk)⊗δ, o = qᵀS (read AFTER the update).
#[allow(clippy::too_many_arguments)]
fn kda_recur_step(cfg: &Config, s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], o: &mut [f32]) {
    for h in 0..cfg.kda_heads {
        let sh = &mut s[h * cfg.kda_dim * cfg.kda_dim..(h + 1) * cfg.kda_dim * cfg.kda_dim];
        let gh = &g[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let kh = &k[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let vh = &v[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let qh = &q[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        // decay along K
        for i in 0..cfg.kda_dim {
            let decay = gh[i].exp();
            let row = &mut sh[i * cfg.kda_dim..(i + 1) * cfg.kda_dim];
            for x in row.iter_mut() {
                *x *= decay;
            }
        }
        // δ = v - kᵀ S
        let mut delta = vh.to_vec();
        for i in 0..cfg.kda_dim {
            let row = &sh[i * cfg.kda_dim..(i + 1) * cfg.kda_dim];
            let ki = kh[i];
            for j in 0..cfg.kda_dim {
                delta[j] -= ki * row[j];
            }
        }
        // S += (β k) ⊗ δ ; o = qᵀ S  →  o[j] = Σ_i q[i]·S[i][j]
        let bh = beta[h];
        let oh = &mut o[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        for j in 0..cfg.kda_dim {
            oh[j] = 0.0;
        }
        for i in 0..cfg.kda_dim {
            let row = &mut sh[i * cfg.kda_dim..(i + 1) * cfg.kda_dim];
            let bk = bh * kh[i];
            for j in 0..cfg.kda_dim {
                row[j] += bk * delta[j];
            }
            let qi = qh[i];
            for j in 0..cfg.kda_dim {
                oh[j] += qi * row[j];
            }
        }
    }
}

/// Test-only handle on the sequential recurrence (parity reference for the
/// chunked prefill form in src/kda_chunk.rs).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn kda_recur_step_pub(cfg: &Config, s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], o: &mut [f32]) {
    kda_recur_step(cfg, s, q, k, v, g, beta, o)
}

/// Per-head gated rmsnorm: y = o·rsqrt(mean(o²)+eps)·o_norm ; o = y·sigmoid(g2).
fn kda_gated_onorm(cfg: &Config, o: &mut [f32], o_norm: &[f32], g2: &[f32]) {
    for h in 0..cfg.kda_heads {
        let oh = &mut o[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let ss = dot(oh, oh) / cfg.kda_dim as f32;
        let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
        for c in 0..cfg.kda_dim {
            oh[c] = oh[c] * inv * o_norm[c] * sigmoid(g2[h * cfg.kda_dim + c]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kda_forward(
    cfg: &Config,
    data: &[u8],
    w: &KdaW,
    cache: &mut KdaCache,
    x: &[f32],
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    let mut q = vec![0f32; cfg.kda_proj()];
    let mut k = vec![0f32; cfg.kda_proj()];
    let mut v = vec![0f32; cfg.kda_proj()];
    matvec(Model::t(data, &w.q_proj), cfg.kda_proj(), cfg.d, x, &mut q);
    matvec(Model::t(data, &w.k_proj), cfg.kda_proj(), cfg.d, x, &mut k);
    matvec(Model::t(data, &w.v_proj), cfg.kda_proj(), cfg.d, x, &mut v);
    let mut g_low = vec![0f32; cfg.kda_proj()];
    {
        let mut fa = vec![0f32; cfg.kda_fa];
        matvec(Model::t(data, &w.f_a), cfg.kda_fa, cfg.d, x, &mut fa);
        matvec(Model::t(data, &w.f_b), cfg.kda_proj(), cfg.kda_fa, &fa, &mut g_low);
    }
    let mut beta = vec![0f32; cfg.kda_heads];
    matvec(Model::t(data, &w.b_proj), cfg.kda_heads, cfg.d, x, &mut beta);
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }
    let mut g2 = vec![0f32; cfg.kda_proj()];
    matvec(Model::t(data, &w.g_proj), cfg.kda_proj(), cfg.d, x, &mut g2);
    prof.t_kda_proj += tm.elapsed().as_secs_f64();

    // depthwise causal conv kernel 4 + SiLU; cache = last 3 raw inputs
    let tm = Instant::now();
    kda_conv_step(cfg, data, &mut q, &w.q_conv, &mut cache.conv_q);
    kda_conv_step(cfg, data, &mut k, &w.k_conv, &mut cache.conv_k);
    kda_conv_step(cfg, data, &mut v, &w.v_conv, &mut cache.conv_v);
    prof.t_kda_conv += tm.elapsed().as_secs_f64();

    let tm = Instant::now();
    // per-head L2-norm over 128, q × 128^-0.5
    kda_norm_head(cfg, &mut q, (cfg.kda_dim as f32).powf(-0.5));
    kda_norm_head(cfg, &mut k, 1.0);
    let g = kda_gate(cfg, Model::t(data, &w.a_log), Model::t(data, &w.dt_bias), &g_low);
    // recurrence: persistent S[4,128,128]
    let mut o = vec![0f32; cfg.kda_proj()];
    kda_recur_step(cfg, &mut cache.s, &q, &k, &v, &g, &beta, &mut o);
    prof.t_kda_recur += tm.elapsed().as_secs_f64();

    let tm = Instant::now();
    kda_gated_onorm(cfg, &mut o, Model::t(data, &w.o_norm), &g2);
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.o_proj), cfg.d, cfg.kda_proj(), &o, &mut out);
    prof.t_kda_proj += tm.elapsed().as_secs_f64();
    out
}

/// Batched KDA for prefill: `x` = n position rows [n * d], returns [n * d].
/// All projections run as gemm_batch (weights streamed once for the whole
/// prompt); the conv stays sequential over positions (tiny elementwise work)
/// and the recurrence runs chunked (WY/UT, src/kda_chunk.rs) for n >= MIN_LEN,
/// sequential below. Both paths update the cache exactly like n single-token
/// calls; the sequential one is bit-identical to kda_forward per position,
/// the chunked one deviates < 1e-4 (unit-tested).
#[allow(clippy::too_many_arguments)]
pub(super) fn kda_prefill(
    cfg: &Config,
    data: &[u8],
    w: &KdaW,
    cache: &mut KdaCache,
    x: &[f32],
    n: usize,
    prof: &mut Prof,
) -> Vec<f32> {
    let kp = cfg.kda_proj();
    let tm = Instant::now();
    let mut q = vec![0f32; n * kp];
    let mut k = vec![0f32; n * kp];
    let mut v = vec![0f32; n * kp];
    gemm_batch(Model::t(data, &w.q_proj), kp, cfg.d, x, n, &mut q);
    gemm_batch(Model::t(data, &w.k_proj), kp, cfg.d, x, n, &mut k);
    gemm_batch(Model::t(data, &w.v_proj), kp, cfg.d, x, n, &mut v);
    let mut fa = vec![0f32; n * cfg.kda_fa];
    gemm_batch(Model::t(data, &w.f_a), cfg.kda_fa, cfg.d, x, n, &mut fa);
    let mut g_low = vec![0f32; n * kp];
    gemm_batch(Model::t(data, &w.f_b), kp, cfg.kda_fa, &fa, n, &mut g_low);
    let mut beta = vec![0f32; n * cfg.kda_heads];
    gemm_batch(Model::t(data, &w.b_proj), cfg.kda_heads, cfg.d, x, n, &mut beta);
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }
    let mut g2 = vec![0f32; n * kp];
    gemm_batch(Model::t(data, &w.g_proj), kp, cfg.d, x, n, &mut g2);
    prof.t_kda_proj += tm.elapsed().as_secs_f64();

    // conv, sequential over positions: the cache shifts one step per position
    let tm = Instant::now();
    for t in 0..n {
        kda_conv_step(cfg, data, &mut q[t * kp..(t + 1) * kp], &w.q_conv, &mut cache.conv_q);
        kda_conv_step(cfg, data, &mut k[t * kp..(t + 1) * kp], &w.k_conv, &mut cache.conv_k);
        kda_conv_step(cfg, data, &mut v[t * kp..(t + 1) * kp], &w.v_conv, &mut cache.conv_v);
    }
    prof.t_kda_conv += tm.elapsed().as_secs_f64();

    let tm = Instant::now();
    let a_log = Model::t(data, &w.a_log);
    let dt_bias = Model::t(data, &w.dt_bias);
    let mut o = vec![0f32; n * kp];
    if n >= crate::kda_chunk::MIN_LEN && !crate::kda_chunk::disabled() {
        // chunked WY/UT form (src/kda_chunk.rs): same recurrence, deviation
        // < 1e-4 vs the sequential loop (unit-tested), much faster on long
        // prompts. Short batches (e.g. the --spec verify passes) keep the
        // sequential step below, bit-identical per position.
        for t in 0..n {
            kda_norm_head(cfg, &mut q[t * kp..(t + 1) * kp], (cfg.kda_dim as f32).powf(-0.5));
            kda_norm_head(cfg, &mut k[t * kp..(t + 1) * kp], 1.0);
        }
        let mut g = vec![0f32; n * kp];
        for t in 0..n {
            g[t * kp..(t + 1) * kp].copy_from_slice(&kda_gate(cfg, a_log, dt_bias, &g_low[t * kp..(t + 1) * kp]));
        }
        crate::kda_chunk::kda_recur_chunked(cfg, &mut cache.s, &q, &k, &v, &g, &beta, n, &mut o);
    } else {
        for t in 0..n {
            kda_norm_head(cfg, &mut q[t * kp..(t + 1) * kp], (cfg.kda_dim as f32).powf(-0.5));
            kda_norm_head(cfg, &mut k[t * kp..(t + 1) * kp], 1.0);
            let g = kda_gate(cfg, a_log, dt_bias, &g_low[t * kp..(t + 1) * kp]);
            let (qr, kr, vr) = (&q[t * kp..(t + 1) * kp], &k[t * kp..(t + 1) * kp], &v[t * kp..(t + 1) * kp]);
            kda_recur_step(cfg, &mut cache.s, qr, kr, vr, &g, &beta[t * cfg.kda_heads..(t + 1) * cfg.kda_heads], &mut o[t * kp..(t + 1) * kp]);
        }
    }
    prof.t_kda_recur += tm.elapsed().as_secs_f64();

    let tm = Instant::now();
    let o_norm = Model::t(data, &w.o_norm);
    for t in 0..n {
        kda_gated_onorm(cfg, &mut o[t * kp..(t + 1) * kp], o_norm, &g2[t * kp..(t + 1) * kp]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.o_proj), cfg.d, kp, &o, n, &mut out);
    prof.t_kda_proj += tm.elapsed().as_secs_f64();
    out
}

// ── MLA: full NoPE ──

// ── MLA attention kernel: flash (online softmax) vs materialized scores ──
//
// Both mla_forward (decode) and mla_prefill compute, for one (query, head),
// attention over the cache positions 0..=pos. The historical kernel
// materializes the whole score row (pos+1 floats per (query, head), fresh
// allocation each time) and makes three passes over it (max, exp+sum,
// normalize) plus a fourth over V. The flash kernel below never materializes
// it: scores are computed one KV tile at a time and folded into an online
// softmax (running max m, running normalizer l, output accumulator rescaled
// on every max update). Same math, different f32 summation order (running
// rescales instead of a final single division) - the A/B test in selftest
// bounds the deviation (measured ~1e-6, tol 1e-5). Causal masking needs no
// matrix anywhere: the loop bound IS the mask (NoPE has no positional terms
// to adjust). MICROKIMI_NO_FLASH=1 restores the materialized path.
