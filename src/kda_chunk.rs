// kda_chunk.rs - chunked KDA recurrence for prefill (WY/UT transform).
//
// The sequential recurrence (kda_recur_step, one token at a time) is the
// prefill bottleneck: per token it touches the whole S[H,K,V] state with
// scalar loops, and tokens cannot be parallelized (S carries over). The
// chunked form (fla naive_chunk_kda semantics) processes 64-token chunks:
// within a chunk everything becomes small dense GEMM-shaped passes, only a
// mini-scan over chunk boundaries stays sequential (T/64 steps).
//
// Per chunk c (length L <= 64), per head (gates g are per-K-channel, beta
// per head; q/k already L2-normed, q pre-scaled by K^-0.5):
//   eg_t[d]  = exp(g_t[d])                                   (L x K exps)
//   EG_t[d]  = prod_{u<=t} eg_u[d]         (cumprod, = exp(cumsum g))
//   A[r,i]   = -beta_i * sum_d k_r[d] k_i[d] exp(G_r[d]-G_i[d])   (i < r)
//   T        = (A + I) after forward substitution T[r,:] += T[r,:] @ A[:, :]
//              (the UT transform, exactly the fla reference recurrence)
//   A'       = T with column i scaled by beta_i
//   w        = A' @ (EG .* k)      u = A' @ v
//   Aqk[r,j] = sum_d q_r[d] k_j[d] exp(G_r[d]-G_j[d])             (j <= r)
// then the inter-chunk scan (sequential over chunks, parallel over heads):
//   v~ = u - w @ S
//   o  = (q .* EG) @ S + Aqk @ v~
//   S  = S .* EG_last[:,None] + (exp(G_last - G_j) .* k_j)^T @ v~
//
// Numerics: the decays exp(G_r - G_i) are evaluated as RUNNING PRODUCTS of
// the per-token eg_t (walk i down from r: D *= eg_{i+1}) instead of
// exp(cumsum differences): L*K exps per chunk instead of L^2*K, and the
// product order matches the sequential recurrence's repeated decay
// multiplications (deviation measured < 1e-4 in the unit test). All decays
// are <= 1 by construction (g < 0), no overflow path.
//
// Parallelism: phase 1 (per-chunk matrices) is independent per (chunk,
// head) and fanned out over the pool; phase 2 (the state scan) is
// sequential over chunks but independent over heads. Decode is untouched
// (single-token forward keeps the sequential step); prompts shorter than
// MIN_LEN keep the sequential loop (fixed costs do not pay off).
// MICROKIMI_NO_KDACHUNK=1 forces the sequential path everywhere.

use crate::config::Config;
use crate::pool::{Job, MPtr, SPtr};

/// Chunk size: 64 like the fla reference (T/64 scan steps, 64x64 intra
/// matrices fit L1 with room for the K=128 row buffers).
const BT: usize = 64;

/// Below this many tokens the sequential loop is kept (scratch allocation +
/// pool barriers dominate on tiny batches, e.g. the --spec verify passes).
pub(crate) const MIN_LEN: usize = 16;

/// True when MICROKIMI_NO_KDACHUNK=1 (A/B toggle, sequential fallback) or when
/// the sequential path was pinned at runtime (force_sequential).
pub(crate) fn disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    FORCE_SEQ.load(std::sync::atomic::Ordering::Relaxed)
        || *OFF.get_or_init(|| std::env::var("MICROKIMI_NO_KDACHUNK").map(|v| v == "1").unwrap_or(false))
}

// Runtime pin for src/pck.rs (prefix cache): a snapshot resume splits the
// prefill into two calls, and the chunked form reassociates the recurrence
// per 64-token chunk, so chunk boundaries moving between the cold and the
// resumed run would break bit-identity. The sequential loop applies the exact
// same per-position operations however the sequence is split, so the pck path
// pins it for the whole process.
static FORCE_SEQ: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn force_sequential() {
    FORCE_SEQ.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Chunked KDA recurrence over n positions. Same inputs/outputs as n calls
/// to kda_recur_step: `s` [H*K*K] is the persistent state (read/updated),
/// q/k/v/g [n * H*K] the per-position projections (q/k normed, q scaled,
/// g the negative log-decays), beta [n * H], `o` [n * H*K] the output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn kda_recur_chunked(
    cfg: &Config,
    s: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    n: usize,
    o: &mut [f32],
) {
    let (hn, kd) = (cfg.kda_heads, cfg.kda_dim);
    let kp = hn * kd;
    let nc = n.div_ceil(BT);
    let len_of = |c: usize| BT.min(n - c * BT);

    // flat scratch, indexed by global position where possible:
    //   w/u/qeg/dk: [n * kp] (token-major, like the inputs)
    //   aqk:        [nc * hn * BT * BT] (per (chunk, head) L x L, lower incl diag)
    //   sdecay:     [nc * hn * kd] (EG of the last chunk row)
    let mut w = vec![0f32; n * kp];
    let mut u = vec![0f32; n * kp];
    let mut qeg = vec![0f32; n * kp];
    let mut dk = vec![0f32; n * kp];
    let mut aqk = vec![0f32; nc * hn * BT * BT];
    let mut sdecay = vec![0f32; nc * hn * kd];

    // ── phase 1: per (chunk, head) matrices, independent -> pool ──
    {
        let p = crate::pool::pool();
        let (qp, kp_, vp, gp, bp) = (SPtr(q.as_ptr()), SPtr(k.as_ptr()), SPtr(v.as_ptr()), SPtr(g.as_ptr()), SPtr(beta.as_ptr()));
        let (wp, up, qegp, dkp, aqkp, sdp) = (
            MPtr(w.as_mut_ptr()),
            MPtr(u.as_mut_ptr()),
            MPtr(qeg.as_mut_ptr()),
            MPtr(dk.as_mut_ptr()),
            MPtr(aqk.as_mut_ptr()),
            MPtr(sdecay.as_mut_ptr()),
        );
        let mut jobs: Vec<Job> = Vec::new();
        for c in 0..nc {
            for h in 0..hn {
                let (qp, kp_, vp, gp, bp) = (qp, kp_, vp, gp, bp);
                let (wp, up, qegp, dkp, aqkp, sdp) = (wp, up, qegp, dkp, aqkp, sdp);
                let (l, s0) = (len_of(c), c * BT);
                jobs.push(Box::new(move || unsafe {
                    // rebind so the closure captures the whole Send wrapper,
                    // not the raw pointer field (edition-2021 precise capture)
                    let (qp, kp_, vp, gp, bp) = (qp, kp_, vp, gp, bp);
                    let (wp, up, qegp, dkp, aqkp, sdp) = (wp, up, qegp, dkp, aqkp, sdp);
                    let q = std::slice::from_raw_parts(qp.0, n * kp);
                    let k = std::slice::from_raw_parts(kp_.0, n * kp);
                    let v = std::slice::from_raw_parts(vp.0, n * kp);
                    let g = std::slice::from_raw_parts(gp.0, n * kp);
                    let beta = std::slice::from_raw_parts(bp.0, n * hn);
                    let w = std::slice::from_raw_parts_mut(wp.0, n * kp);
                    let u = std::slice::from_raw_parts_mut(up.0, n * kp);
                    let qeg = std::slice::from_raw_parts_mut(qegp.0, n * kp);
                    let dk = std::slice::from_raw_parts_mut(dkp.0, n * kp);
                    let aqk = std::slice::from_raw_parts_mut(aqkp.0, nc * hn * BT * BT);
                    let sdecay = std::slice::from_raw_parts_mut(sdp.0, nc * hn * kd);
                    chunk_matrices(hn, kd, s0, l, h, q, k, v, g, beta, w, u, qeg, dk, aqk, sdecay);
                }));
            }
        }
        p.run(jobs);
    }

    // ── phase 2: inter-chunk state scan, sequential over chunks ──
    // (parallel over heads: each head owns a disjoint S slice)
    {
        let p = crate::pool::pool();
        let (wp, up, qegp, dkp, aqkp, sdp) = (
            SPtr(w.as_ptr()),
            SPtr(u.as_ptr()),
            SPtr(qeg.as_ptr()),
            SPtr(dk.as_ptr()),
            SPtr(aqk.as_ptr()),
            SPtr(sdecay.as_ptr()),
        );
        let (sp, op) = (MPtr(s.as_mut_ptr()), MPtr(o.as_mut_ptr()));
        let mut jobs: Vec<Job> = Vec::new();
        for h in 0..hn {
            let (wp, up, qegp, dkp, aqkp, sdp) = (wp, up, qegp, dkp, aqkp, sdp);
            let (sp, op) = (sp, op);
            jobs.push(Box::new(move || unsafe {
                // rebind so the closure captures the whole Send wrapper,
                // not the raw pointer field (edition-2021 precise capture)
                let (wp, up, qegp, dkp, aqkp, sdp) = (wp, up, qegp, dkp, aqkp, sdp);
                let (sp, op) = (sp, op);
                let w = std::slice::from_raw_parts(wp.0, n * kp);
                let u = std::slice::from_raw_parts(up.0, n * kp);
                let qeg = std::slice::from_raw_parts(qegp.0, n * kp);
                let dk = std::slice::from_raw_parts(dkp.0, n * kp);
                let aqk = std::slice::from_raw_parts(aqkp.0, nc * hn * BT * BT);
                let sdecay = std::slice::from_raw_parts(sdp.0, nc * hn * kd);
                let s = std::slice::from_raw_parts_mut(sp.0, hn * kd * kd);
                let o = std::slice::from_raw_parts_mut(op.0, n * kp);
                let sh = &mut s[h * kd * kd..(h + 1) * kd * kd];
                let mut vt = vec![0f32; BT * kd]; // v~ rows of the current chunk
                for c in 0..nc {
                    let l = BT.min(n - c * BT);
                    let s0 = c * BT;
                    let aqk_h = &aqk[(c * hn + h) * BT * BT..(c * hn + h) * BT * BT + l * l];
                    let sd = &sdecay[(c * hn + h) * kd..(c * hn + h + 1) * kd];
                    // v~ = u - w @ S  (row r: v~[r] = u[r] - sum_d w[r,d] S[d,:])
                    for r in 0..l {
                        let (ur, wr) = (&u[(s0 + r) * kp + h * kd..(s0 + r) * kp + (h + 1) * kd], &w[(s0 + r) * kp + h * kd..(s0 + r) * kp + (h + 1) * kd]);
                        let vr = &mut vt[r * kd..(r + 1) * kd];
                        vr.copy_from_slice(ur);
                        for (d, &wd) in wr.iter().enumerate() {
                            if wd != 0.0 {
                                let srow = &sh[d * kd..(d + 1) * kd];
                                for (x, &sv) in vr.iter_mut().zip(srow.iter()) {
                                    *x -= wd * sv;
                                }
                            }
                        }
                    }
                    // o = (q .* EG) @ S + Aqk @ v~
                    for r in 0..l {
                        let orow = &mut o[(s0 + r) * kp + h * kd..(s0 + r) * kp + (h + 1) * kd];
                        for x in orow.iter_mut() {
                            *x = 0.0;
                        }
                        let qr = &qeg[(s0 + r) * kp + h * kd..(s0 + r) * kp + (h + 1) * kd];
                        for (d, &qd) in qr.iter().enumerate() {
                            if qd != 0.0 {
                                let srow = &sh[d * kd..(d + 1) * kd];
                                for (x, &sv) in orow.iter_mut().zip(srow.iter()) {
                                    *x += qd * sv;
                                }
                            }
                        }
                        for (j, &a) in aqk_h[r * l..(r + 1) * l].iter().enumerate() {
                            if a != 0.0 {
                                let vj = &vt[j * kd..(j + 1) * kd];
                                for (x, &y) in orow.iter_mut().zip(vj.iter()) {
                                    *x += a * y;
                                }
                            }
                        }
                    }
                    // S = S .* sd[:,None] + dk^T @ v~
                    for d in 0..kd {
                        let srow = &mut sh[d * kd..(d + 1) * kd];
                        let dec = sd[d];
                        for x in srow.iter_mut() {
                            *x *= dec;
                        }
                        for j in 0..l {
                            let coef = dk[(s0 + j) * kp + h * kd + d];
                            if coef != 0.0 {
                                let vj = &vt[j * kd..(j + 1) * kd];
                                for (x, &y) in srow.iter_mut().zip(vj.iter()) {
                                    *x += coef * y;
                                }
                            }
                        }
                    }
                }
            }));
        }
        p.run(jobs);
    }
}

/// Phase 1 for one (chunk, head): builds w, u, qeg, dk, aqk, sdecay.
/// `s0` = first global position of the chunk, `l` its length, `h` the head.
#[allow(clippy::too_many_arguments)]
fn chunk_matrices(
    hn: usize,
    kd: usize,
    s0: usize,
    l: usize,
    h: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    w: &mut [f32],
    u: &mut [f32],
    qeg: &mut [f32],
    dk: &mut [f32],
    aqk: &mut [f32],
    sdecay: &mut [f32],
) {
    let kp = hn * kd;
    // first flat index of the (token t, head h) row in a [n * kp] buffer
    let r0 = |t: usize| (s0 + t) * kp + h * kd;

    // eg[t][d] = exp(g_t[d]); EG[t][d] = cumprod (running, same product
    // order as the sequential decay multiplications)
    let mut eg = vec![0f32; l * kd];
    let mut ec = vec![0f32; l * kd]; // EG
    for t in 0..l {
        let gr = &g[r0(t)..r0(t) + kd];
        let er = &mut eg[t * kd..(t + 1) * kd];
        for (e, &x) in er.iter_mut().zip(gr.iter()) {
            *e = x.exp();
        }
    }
    for t in 0..l {
        if t == 0 {
            ec[..kd].copy_from_slice(&eg[..kd]);
        } else {
            let (prev, cur) = ec.split_at_mut(t * kd);
            let pr = &prev[(t - 1) * kd..t * kd];
            let er = &eg[t * kd..(t + 1) * kd];
            let cr = &mut cur[..kd];
            for ((c, &p), &e) in cr.iter_mut().zip(pr.iter()).zip(er.iter()) {
                *c = p * e;
            }
        }
    }
    // sdecay = EG of the last row (written at this (chunk, head)'s slot of
    // the shared [nc * hn * kd] buffer)
    let c = s0 / BT;
    sdecay[(c * hn + h) * kd..(c * hn + h + 1) * kd].copy_from_slice(&ec[(l - 1) * kd..l * kd]);
    // qeg[t] = q[t] .* EG[t]
    for t in 0..l {
        let qr = &mut qeg[r0(t)..r0(t) + kd];
        let (qi, ei) = (&q[r0(t)..r0(t) + kd], &ec[t * kd..(t + 1) * kd]);
        for ((x, &a), &b) in qr.iter_mut().zip(qi.iter()).zip(ei.iter()) {
            *x = a * b;
        }
    }
    // dk[t] = k[t] .* exp(G_last - G_t): D walks UP from the last row
    // (D_{l-1} = 1, D_{t} = D_{t+1} .* eg_{t+1})
    {
        let mut d = vec![1f32; kd];
        for t in (0..l).rev() {
            let dr = &mut dk[r0(t)..r0(t) + kd];
            let ki = &k[r0(t)..r0(t) + kd];
            for ((x, &dd), &kk) in dr.iter_mut().zip(d.iter()).zip(ki.iter()) {
                *x = dd * kk;
            }
            if t > 0 {
                let et = &eg[t * kd..(t + 1) * kd];
                for (dd, &e) in d.iter_mut().zip(et.iter()) {
                    *dd *= e;
                }
            }
        }
    }

    // ── intra-chunk A (strictly lower, negated) with running decays ──
    // A[r,i] = -beta_r * sum_d k_r[d] k_i[d] D[d] (beta on the ROW: the
    // column scaling by beta_i happens in A' below, and the WY derivation
    // against the sequential recurrence requires exactly one of each), D =
    // exp(G_r - G_i) walks down with i (D_{r-1} = eg_r, D_{i-1} = D_i .* eg_i)
    let mut a = vec![0f32; l * l];
    for r in 1..l {
        let kr = k[r0(r)..r0(r) + kd].to_vec();
        let mut d = eg[r * kd..(r + 1) * kd].to_vec();
        for i in (0..r).rev() {
            let ki = &k[r0(i)..r0(i) + kd];
            let mut acc = 0f32;
            for dd in 0..kd {
                acc += kr[dd] * ki[dd] * d[dd];
            }
            a[r * l + i] = -beta[(s0 + r) * hn + h] * acc;
            if i > 0 {
                let ei = &eg[i * kd..(i + 1) * kd];
                for (dd, &e) in d.iter_mut().zip(ei.iter()) {
                    *dd *= e;
                }
            }
        }
    }
    // ── UT forward substitution: T[r,:] += T[r,:] @ A[:,:] over strict lower ──
    for r in 1..l {
        for j in 0..r {
            let mut acc = 0f32;
            for m in (j + 1)..r {
                acc += a[r * l + m] * a[m * l + j];
            }
            a[r * l + j] += acc;
        }
    }
    // A' = (A + I), column i scaled by beta_i
    for r in 0..l {
        for i in 0..r {
            a[r * l + i] *= beta[(s0 + i) * hn + h];
        }
        a[r * l + r] = beta[(s0 + r) * hn + h];
    }
    // w = A' @ (EG .* k) ; u = A' @ v   (row r: sum over i <= r)
    for r in 0..l {
        let wr = &mut w[r0(r)..r0(r) + kd];
        let ur = &mut u[r0(r)..r0(r) + kd];
        for x in wr.iter_mut() {
            *x = 0.0;
        }
        for x in ur.iter_mut() {
            *x = 0.0;
        }
        for i in 0..=r {
            let coef = a[r * l + i];
            if coef == 0.0 {
                continue;
            }
            let ki = &k[r0(i)..r0(i) + kd];
            let vi = &v[r0(i)..r0(i) + kd];
            let ei = &ec[i * kd..(i + 1) * kd];
            for d in 0..kd {
                wr[d] += coef * ei[d] * ki[d];
                ur[d] += coef * vi[d];
            }
        }
    }
    // ── Aqk[r,j] = sum_d q_r[d] k_j[d] D[d] (j <= r), D walks down like A ──
    let aqk_h = &mut aqk[(c * hn + h) * BT * BT..(c * hn + h) * BT * BT + l * l];
    for r in 0..l {
        let qr = q[r0(r)..r0(r) + kd].to_vec();
        let mut d = vec![1f32; kd]; // j == r: exp(0)
        for j in (0..=r).rev() {
            let kj = &k[r0(j)..r0(j) + kd];
            let mut acc = 0f32;
            for dd in 0..kd {
                acc += qr[dd] * kj[dd] * d[dd];
            }
            aqk_h[r * l + j] = acc;
            if j > 0 {
                let ej = &eg[j * kd..(j + 1) * kd];
                for (dd, &e) in d.iter_mut().zip(ej.iter()) {
                    *dd *= e;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// deterministic filler (splitmix64), no rand crate
    struct Rng(u64);
    impl Rng {
        fn f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64 - 0.5) as f32
        }
    }

    /// Chunked recurrence vs the sequential kda_recur_step on random inputs
    /// at the nanokimi dims (H=4, K=V=128), lengths around the chunk size.
    /// Inputs mimic the real distributions: q/k per-head L2-normed, g in
    /// (-5, 0) (gate_lb), beta in (0, 1).
    #[test]
    fn chunk_vs_sequential() {
        let cfg = Config::microkimi();
        let (hn, kd) = (cfg.kda_heads, cfg.kda_dim);
        let kp = hn * kd;
        let mut rng = Rng(0xC0FFEE);
        for n in [1usize, 63, 64, 65, 128, 300] {
            let mut q: Vec<f32> = (0..n * kp).map(|_| rng.f32()).collect();
            let mut k: Vec<f32> = (0..n * kp).map(|_| rng.f32()).collect();
            let v: Vec<f32> = (0..n * kp).map(|_| rng.f32()).collect();
            let g: Vec<f32> = (0..n * kp).map(|_| -5.0 * (rng.f32() + 0.5).abs()).collect();
            let beta: Vec<f32> = (0..n * hn).map(|_| (rng.f32() + 0.5).abs()).collect();
            // per-head L2 norm (q scaled by K^-0.5 like the engine)
            for t in 0..n {
                for h in 0..hn {
                    let qh = &mut q[t * kp + h * kd..t * kp + (h + 1) * kd];
                    let nq = qh.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                    qh.iter_mut().for_each(|x| *x *= (kd as f32).powf(-0.5) / nq);
                    let kh = &mut k[t * kp + h * kd..t * kp + (h + 1) * kd];
                    let nk = kh.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-12);
                    kh.iter_mut().for_each(|x| *x /= nk);
                }
            }
            let mut s_seq = vec![0f32; hn * kd * kd];
            let mut s_chk = s_seq.clone();
            let mut o_seq = vec![0f32; n * kp];
            let mut o_chk = vec![0f32; n * kp];
            for t in 0..n {
                crate::model::kda_recur_step_pub(
                    &cfg,
                    &mut s_seq,
                    &q[t * kp..(t + 1) * kp],
                    &k[t * kp..(t + 1) * kp],
                    &v[t * kp..(t + 1) * kp],
                    &g[t * kp..(t + 1) * kp],
                    &beta[t * hn..(t + 1) * hn],
                    &mut o_seq[t * kp..(t + 1) * kp],
                );
            }
            kda_recur_chunked(&cfg, &mut s_chk, &q, &k, &v, &g, &beta, n, &mut o_chk);
            let max_o = o_seq.iter().zip(o_chk.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let max_s = s_seq.iter().zip(s_chk.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            let scale = o_seq.iter().map(|x| x.abs()).fold(0f32, f32::max).max(1e-6);
            eprintln!("n={:4}  max|dO|={:.3e}  max|dS|={:.3e}  (max|o|={:.3e})", n, max_o, max_s, scale);
            assert!(max_o < 1e-4, "output deviation {} at n={}", max_o, n);
            assert!(max_s < 1e-4, "state deviation {} at n={}", max_s, n);
        }
    }
}
