// Math kernels: SIMD dot products (scalar reference plus AVX2/NEON, dispatch is
// bit-identical to scalar on every length), f32 matvec/GEMM, q8 lm-head matvec,
// packed-weight matvec for the MoE prefill, rmsnorm, SiLU/SiTU, AttnRes.
// Pure functions over slices: no model state, no I/O, no caching.

use super::*;

/// Scalar dot: the historical path, reference for the SIMD kernels and
/// fallback when no SIMD feature is present.
#[inline]
#[allow(dead_code)] // on aarch64 the dispatched dot() never reaches this
pub(super) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    loop {
        match (ca.next(), cb.next()) {
            (Some(av), Some(bv)) => {
                for j in 0..8 {
                    acc[j] += av[j] * bv[j];
                }
            }
            _ => break,
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x * y;
    }
    s
}

/// NEON dot (aarch64): the 8 accumulators live in two float32x4 registers
/// (lanes = acc[0..4], acc[4..8]); each lane sees the same mul-then-add as
/// the scalar loop (vaddq of vmulq, never vfmaq). The horizontal reduction
/// replays the scalar reduction order exactly (see the contract above).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut lo = vdupq_n_f32(0.0);
        let mut hi = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 8 <= n {
            lo = vaddq_f32(lo, vmulq_f32(vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i))));
            hi = vaddq_f32(hi, vmulq_f32(vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4))));
            i += 8;
        }
        let mut acc = [0f32; 8];
        vst1q_f32(acc.as_mut_ptr(), lo);
        vst1q_f32(acc.as_mut_ptr().add(4), hi);
        let (p01, p23) = (acc[0] + acc[1], acc[2] + acc[3]);
        let (p45, p67) = (acc[4] + acc[5], acc[6] + acc[7]);
        let mut s = ((p01 + p23) + p45) + p67;
        while i < n {
            s += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        s
    }
}

/// AVX2 dot (x86_64): the 8 accumulators are the 8 lanes of one __m256
/// (_mm256_add_ps of _mm256_mul_ps, never _mm256_fmadd_ps). Same reduction
/// replay as NEON. Bit-identical to dot_scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let n = a.len().min(b.len());
        let (pa, pb) = (a.as_ptr(), b.as_ptr());
        let mut vacc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= n {
            vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i))));
            i += 8;
        }
        let mut acc = [0f32; 8];
        _mm256_storeu_ps(acc.as_mut_ptr(), vacc);
        let (p01, p23) = (acc[0] + acc[1], acc[2] + acc[3]);
        let (p45, p67) = (acc[4] + acc[5], acc[6] + acc[7]);
        let mut s = ((p01 + p23) + p45) + p67;
        while i < n {
            s += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        s
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_available() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| is_x86_feature_detected!("avx2"))
}

/// Chunk size for dynamically scheduled row loops: ~8 pulls per worker
/// bound the straggler tail, a 16-row floor keeps the atomic traffic
/// negligible, and the multiple-of-4 rounding feeds the quad-row kernels.
/// MICROKIMI_NO_DYNROWS=1 is the A/B arm: one pull per worker, which
/// degenerates to the old fixed contiguous chunking.
#[inline]
pub(crate) fn dyn_step(rows: usize, workers: usize) -> usize {
    static FIXED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let pulls = if *FIXED.get_or_init(|| {
        std::env::var("MICROKIMI_NO_DYNROWS").map(|v| v == "1").unwrap_or(false)
    }) {
        1
    } else {
        8
    };
    (rows.div_ceil(workers.max(1) * pulls).max(16) + 3) & !3
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64: unconditional, zero dispatch cost
        return unsafe { dot_neon(a, b) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

/// f32 matrix × vector. Entry point for the whole engine.
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_os = "macos")]
    {
        if gpu_on() && rows * cols >= GPU_MIN_ELEMS && crate::model::metal::gpu_available() {
            crate::model::metal::gpu_matvec(w, rows, cols, x, out);
            return;
        }
    }
    matvec_cpu(w, rows, cols, x, out);
}

/// f32 matrix × vector on the persistent pool (std::thread). Adaptive job
/// count (~200k MACs/job): small matvecs stay inline, large ones are split
/// into rows. The pool barrier guarantees the validity of the raw pointers
/// captured by the jobs.
pub fn matvec_cpu(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let p = crate::model::pool::pool();
    let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
    if njobs <= 1 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(&w[r * cols..(r + 1) * cols], x);
        }
        return;
    }
    // dynamic row scheduling (see Q8Head::matvec): fine chunks off a
    // shared counter, bit-identical per row.
    let step = dyn_step(rows, njobs);
    let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wp = crate::model::pool::SPtr(w.as_ptr());
    let xp = crate::model::pool::SPtr(x.as_ptr());
    let op = crate::model::pool::MPtr(out.as_mut_ptr());
    let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
    for _ in 0..njobs {
        let ctr = ctr.clone();
        jobs.push(Box::new(move || {
            // rebind → capture whole structs (Send), not fields
            let (wp, xp, op) = (wp, xp, op);
            unsafe {
                let w = std::slice::from_raw_parts(wp.0, rows * cols);
                let x = std::slice::from_raw_parts(xp.0, cols);
                let out = std::slice::from_raw_parts_mut(op.0, rows);
                loop {
                    let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                    if r0 >= rows {
                        break;
                    }
                    for r in r0..(r0 + step).min(rows) {
                        out[r] = dot(&w[r * cols..(r + 1) * cols], x);
                    }
                }
            }
        }));
    }
    p.run(jobs);
}

/// Single-threaded f32 matrix × vector for callers that already own a
/// worker thread (batched prefill): per-row results are bit-identical to
/// the pooled `matvec`, which chunks whole rows across jobs.
pub fn matvec_st(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(out.len(), rows);
    for (r, o) in out.iter_mut().enumerate() {
        *o = dot(&w[r * cols..(r + 1) * cols], x);
    }
}

/// Multi-lane f32 matrix × vectors: each weight row is read ONCE and
/// dotted against every lane's input. In the memory-bound decode regime
/// the weight traffic dominates, so n lanes cost close to one - this is
/// the kernel behind lane-batched decoding. Per-lane results are
/// bit-identical to `matvec` (same row, same dot). Pool-parallel over
/// row chunks like matvec_cpu.
pub fn matvec_multi(w: &[f32], rows: usize, cols: usize, xs: &[&[f32]], outs: &mut [&mut [f32]]) {
    assert_eq!(xs.len(), outs.len());
    let lanes = xs.len();
    if lanes == 0 {
        return;
    }
    // Qwen prefill offload (MICROKIMI_QWEN_GPU=1, macOS): one MPS GEMM for
    // the whole batch. Only worth it above the lane/size thresholds; any
    // failure falls through to the CPU kernels below.
    #[cfg(target_os = "macos")]
    if crate::model::metal::qwen_gpu_on()
        && lanes >= crate::model::metal::GEMM_MIN_T
        && rows * cols >= crate::model::metal::GEMM_MIN_ELEMS
        && crate::model::metal::gpu_gemm_xwt(w, rows, cols, xs, outs)
    {
        return;
    }
    let p = crate::model::pool::pool();
    let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
    if njobs <= 1 {
        for r in 0..rows {
            let row = &w[r * cols..(r + 1) * cols];
            for l in 0..lanes {
                outs[l][r] = dot(row, xs[l]);
            }
        }
        return;
    }
    // dynamic row scheduling (see Q8Head::matvec): fine chunks off a
    // shared counter, bit-identical per row.
    let step = dyn_step(rows, njobs);
    let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let wp = crate::model::pool::SPtr(w.as_ptr());
    let xps: Vec<crate::model::pool::SPtr> =
        xs.iter().map(|x| crate::model::pool::SPtr(x.as_ptr())).collect();
    let ops: Vec<crate::model::pool::MPtr> = outs
        .iter_mut()
        .map(|o| crate::model::pool::MPtr(o.as_mut_ptr()))
        .collect();
    let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
    for _ in 0..njobs {
        let xps = xps.clone();
        let ops = ops.clone();
        let ctr = ctr.clone();
        jobs.push(Box::new(move || {
            let (wp, xps, ops) = (wp, xps, ops);
            unsafe {
                let w = std::slice::from_raw_parts(wp.0, rows * cols);
                loop {
                    let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                    if r0 >= rows {
                        break;
                    }
                    for r in r0..(r0 + step).min(rows) {
                        let row = &w[r * cols..(r + 1) * cols];
                        for l in 0..xps.len() {
                            let x = std::slice::from_raw_parts(xps[l].0, cols);
                            *ops[l].0.add(r) = dot(row, x);
                        }
                    }
                }
            }
        }));
    }
    p.run(jobs);
}


/// Packed-quad variant of `multi_rows_q8`: full row quads read the
/// interleaved GEMM layout (one address stream), remainder rows fall
/// back to the row-major kernels. Same math, same order - bit-identical.
///
/// SAFETY: same contract as `multi_rows_q8`, plus `pq`/`ps` hold the
/// quad-interleaved copy of the same matrix.
#[allow(clippy::too_many_arguments)]
unsafe fn multi_rows_q8_packed(
    q: &[i8],
    scales: &[f32],
    pq: &[i8],
    ps: &[f32],
    rows: usize,
    cols: usize,
    r0: usize,
    r1: usize,
    lanes: &[(&[i8], &[f32])],
    out_ptrs: &[usize],
) {
    let nb = cols / 32;
    // L2 lane-blocking as in multi_rows_q8
    if lanes.len() > 64 {
        let mut l0 = 0usize;
        while l0 < lanes.len() {
            let l1 = (l0 + 64).min(lanes.len());
            // SAFETY: forwarded contract; lane blocks are disjoint.
            unsafe {
                multi_rows_q8_packed(
                    q, scales, pq, ps, rows, cols, r0, r1, &lanes[l0..l1], &out_ptrs[l0..l1],
                );
            }
            l0 = l1;
        }
        return;
    }
    let mut r = r0;
    #[cfg(target_arch = "aarch64")]
    if crate::quant::q8::sdot4_available() && r % 4 == 0 {
        while r + 4 <= r1 {
            let qd = r / 4;
            let wq = &pq[qd * nb * 128..(qd + 1) * nb * 128];
            let wsq = &ps[qd * nb * 4..(qd + 1) * nb * 4];
            let mut l0 = 0usize;
            while l0 + 2 <= lanes.len() {
                let width = (lanes.len() - l0).min(8);
                let tile = unsafe {
                    crate::quant::q8::rows4_dot_fma_x4_packed(wq, wsq, &lanes[l0..l0 + width])
                };
                for (dl, lane_out) in tile.iter().take(width).enumerate() {
                    for k in 0..4 {
                        unsafe { *(out_ptrs[l0 + dl] as *mut f32).add(r + k) = lane_out[k] };
                    }
                }
                l0 += width;
            }
            for (l, lane) in lanes.iter().enumerate().skip(l0) {
                let w4 = [
                    &q[r * cols..(r + 1) * cols],
                    &q[(r + 1) * cols..(r + 2) * cols],
                    &q[(r + 2) * cols..(r + 3) * cols],
                    &q[(r + 3) * cols..(r + 4) * cols],
                ];
                let s4 = [
                    &scales[r * nb..(r + 1) * nb],
                    &scales[(r + 1) * nb..(r + 2) * nb],
                    &scales[(r + 2) * nb..(r + 3) * nb],
                    &scales[(r + 3) * nb..(r + 4) * nb],
                ];
                let sums = unsafe { crate::quant::q8::rows4_dot_fma(w4, s4, lane.0, lane.1) };
                for k in 0..4 {
                    unsafe { *(out_ptrs[l] as *mut f32).add(r + k) = sums[k] };
                }
            }
            r += 4;
        }
    }
    if r < r1 {
        // remainder rows (or misaligned chunk start): row-major path
        // SAFETY: forwarded contract.
        unsafe { multi_rows_q8(q, scales, rows, cols, r, r1, lanes, out_ptrs) };
    }
}

/// One row range against every lane, shared by the pooled and scoped
/// multi paths: 4x4 register tiles, then rows4 per remaining lane, then
/// q8_rows_dot single rows - the same kernels and order everywhere.
///
/// SAFETY: `out_ptrs[l] + r` cells are exclusive to the caller's row
/// range; slices hold full rows/blocks.
#[allow(clippy::too_many_arguments)]
unsafe fn multi_rows_q8(
    q: &[i8],
    scales: &[f32],
    _rows: usize,
    cols: usize,
    r0: usize,
    r1: usize,
    lanes: &[(&[i8], &[f32])],
    out_ptrs: &[usize],
) {
    let nb = cols / 32;
    // L2 lane-blocking: with lanes outermost the whole activation set
    // re-streamed from DRAM once per row quad (~740 MB per 3072-row
    // matrix at 1k lanes). A 64-lane block stays cache-resident while
    // the rows stream past it, so activations are read from DRAM once
    // and weights once per block.
    if lanes.len() > 64 {
        let mut l0 = 0usize;
        while l0 < lanes.len() {
            let l1 = (l0 + 64).min(lanes.len());
            // SAFETY: forwarded contract; lane blocks are disjoint.
            unsafe {
                multi_rows_q8(q, scales, _rows, cols, r0, r1, &lanes[l0..l1], &out_ptrs[l0..l1]);
            }
            l0 = l1;
        }
        return;
    }
    let mut r = r0;
    #[cfg(target_arch = "aarch64")]
    if crate::quant::q8::sdot4_available() {
        while r + 4 <= r1 {
            let w4 = [
                &q[r * cols..(r + 1) * cols],
                &q[(r + 1) * cols..(r + 2) * cols],
                &q[(r + 2) * cols..(r + 3) * cols],
                &q[(r + 3) * cols..(r + 4) * cols],
            ];
            let s4 = [
                &scales[r * nb..(r + 1) * nb],
                &scales[(r + 1) * nb..(r + 2) * nb],
                &scales[(r + 2) * nb..(r + 3) * nb],
                &scales[(r + 3) * nb..(r + 4) * nb],
            ];
            let mut l0 = 0usize;
            while l0 + 2 <= lanes.len() {
                let width = (lanes.len() - l0).min(8);
                let tile =
                    unsafe { crate::quant::q8::rows4_dot_fma_x4(w4, s4, &lanes[l0..l0 + width]) };
                for (dl, lane_out) in tile.iter().take(width).enumerate() {
                    for k in 0..4 {
                        unsafe { *(out_ptrs[l0 + dl] as *mut f32).add(r + k) = lane_out[k] };
                    }
                }
                l0 += width;
            }
            for (l, lane) in lanes.iter().enumerate().skip(l0) {
                let sums = unsafe { crate::quant::q8::rows4_dot_fma(w4, s4, lane.0, lane.1) };
                for k in 0..4 {
                    unsafe { *(out_ptrs[l] as *mut f32).add(r + k) = sums[k] };
                }
            }
            r += 4;
        }
    }
    while r < r1 {
        for (l, lane) in lanes.iter().enumerate() {
            let mut buf = [0f32; 1];
            q8_rows_dot(q, scales, cols, r, 1, lane.0, lane.1, &mut buf);
            unsafe { *(out_ptrs[l] as *mut f32).add(r) = buf[0] };
        }
        r += 1;
    }
}

/// Row-range q8 dot against one quantized activation, four rows at a
/// time through the fused SDOT kernel when available: the activation
/// blocks are loaded once per quad and the four accumulator chains are
/// independent. Integer sums are exact, so results are bit-identical to
/// the single-row loop. Shared by the Q8Head methods and the pool jobs.
#[allow(clippy::too_many_arguments)]
pub(super) fn q8_rows_dot(
    q: &[i8],
    scales: &[f32],
    cols: usize,
    r0: usize,
    n: usize,
    xq_q: &[i8],
    xq_scales: &[f32],
    out: &mut [f32],
) {
    let nb = cols / 32;
    let mut r = 0usize;
    #[cfg(target_arch = "aarch64")]
    if crate::quant::q8::sdot4_available() {
        while r + 4 <= n {
            let base = r0 + r;
            // SAFETY: dotprod checked above; the row and scale slices
            // hold exactly nb blocks each.
            let sums = unsafe {
                crate::quant::q8::rows4_dot_fma(
                    [
                        &q[base * cols..(base + 1) * cols],
                        &q[(base + 1) * cols..(base + 2) * cols],
                        &q[(base + 2) * cols..(base + 3) * cols],
                        &q[(base + 3) * cols..(base + 4) * cols],
                    ],
                    [
                        &scales[base * nb..(base + 1) * nb],
                        &scales[(base + 1) * nb..(base + 2) * nb],
                        &scales[(base + 2) * nb..(base + 3) * nb],
                        &scales[(base + 3) * nb..(base + 4) * nb],
                    ],
                    xq_q,
                    xq_scales,
                )
            };
            out[r..r + 4].copy_from_slice(&sums);
            r += 4;
        }
    }
    while r < n {
        let row = r0 + r;
        let mut sum = 0f32;
        for g in 0..nb {
            let idot = crate::quant::q8::block_dot_i8(
                &q[row * cols + g * 32..row * cols + (g + 1) * 32],
                &xq_q[g * 32..(g + 1) * 32],
            );
            // fused, block-sequential: the exact order of rows4_dot_fma
            sum = (idot as f32).mul_add(scales[row * nb + g] * xq_scales[g], sum);
        }
        out[r] = sum;
        r += 1;
    }
}

// ── q8_0 lm_head (runtime copy, built once at load) ──
//
// The final logits projection re-reads the whole f32 lm_head tensor every
// token (vocab x d: the largest single matvec of the engine). Keeping a
// row-wise q8_0 copy (same convention as q8.rs: int8 values + one f32 scale
// per block of 32) shrinks that stream ~3.5x and moves the dot to the integer
// SIMD kernel (block_dot_i8). NOT bit-identical to the f32 matvec: q8
// rounding of both the weights and the input, error bounded by dx/2 per
// element. Greedy token parity is validated on the nanokimi smoke model;
// MICROKIMI_Q8HEAD=0 at load time keeps the exact f32 path.

/// Row-wise q8_0 quantized matrix (built from an f32 [rows, cols] tensor).
pub struct Q8Head {
    q: Vec<i8>,      // rows x cols, row-major
    scales: Vec<f32>, // rows x cols/32
    rows: usize,
    cols: usize,
    /// Quad-interleaved GEMM layout, built lazily on the first pooled
    /// multi-lane call: per (row-quad, block) the four rows' 32 bytes sit
    /// consecutively and their four scales together - one address stream
    /// for the tile kernel instead of four plus a scalar scale gather.
    gemm_pack: std::sync::OnceLock<(Vec<i8>, Vec<f32>)>,
}

/// Builds the quad-interleaved copy (rows past the last full quad stay
/// on the row-major path).
fn build_gemm_pack(q: &[i8], scales: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let nb = cols / 32;
    let quads = rows / 4;
    let mut wq = vec![0i8; quads * nb * 128];
    let mut ws = vec![0.0f32; quads * nb * 4];
    let workers = crate::model::pool::pool().workers.max(1).min(quads.max(1));
    let chunk = quads.div_ceil(workers.max(1)).max(1);
    std::thread::scope(|s| {
        let mut wq_rest = wq.as_mut_slice();
        let mut ws_rest = ws.as_mut_slice();
        let mut q0 = 0usize;
        while q0 < quads {
            let q1 = (q0 + chunk).min(quads);
            let (wq_c, wr) = wq_rest.split_at_mut((q1 - q0) * nb * 128);
            let (ws_c, sr) = ws_rest.split_at_mut((q1 - q0) * nb * 4);
            wq_rest = wr;
            ws_rest = sr;
            s.spawn(move || {
                for qd in q0..q1 {
                    for g in 0..nb {
                        let dst = (qd - q0) * nb + g;
                        for r in 0..4 {
                            let row = qd * 4 + r;
                            wq_c[dst * 128 + r * 32..dst * 128 + (r + 1) * 32]
                                .copy_from_slice(&q[row * cols + g * 32..row * cols + (g + 1) * 32]);
                            ws_c[dst * 4 + r] = scales[row * nb + g];
                        }
                    }
                }
            });
            q0 = q1;
        }
    });
    (wq, ws)
}

/// MICROKIMI_Q8HEAD=0 disables the q8 lm_head copy (exact f32 fallback).
pub(super) fn q8head_enabled() -> bool {
    std::env::var("MICROKIMI_Q8HEAD").map(|v| v != "0").unwrap_or(true)
}

impl Q8Head {
    pub(super) fn from_f32(w: &[f32], rows: usize, cols: usize) -> Q8Head {
        assert!(cols % 32 == 0, "q8 blocks are 32 wide");
        let nb = cols / 32;
        let mut q = vec![0i8; rows * cols];
        let mut scales = vec![0f32; rows * nb];
        let mut scratch = crate::quant::q8::Q8Vec::new();
        for r in 0..rows {
            crate::quant::q8::quantize_q8_into(&w[r * cols..(r + 1) * cols], &mut scratch);
            q[r * cols..(r + 1) * cols].copy_from_slice(&scratch.q);
            scales[r * nb..(r + 1) * nb].copy_from_slice(&scratch.scales);
        }
        Q8Head { q, scales, rows, cols, gemm_pack: std::sync::OnceLock::new() }
    }

    /// Single-threaded q8 matvec for callers that already own a worker
    /// thread (prefill workers): per-row results are bit-identical to the
    /// pooled `matvec`.
    pub(super) fn matvec_st(&self, x: &[f32], out: &mut [f32]) {
        let xq = crate::quant::q8::quantize_q8(x);
        self.rows_dot(0, out.len(), &xq, out);
    }

    /// Dots rows [r0, r0+n) against one quantized activation, four rows at
    /// a time through the fused SDOT kernel when available.
    fn rows_dot(&self, r0: usize, n: usize, xq: &crate::quant::q8::Q8Vec, out: &mut [f32]) {
        q8_rows_dot(
            &self.q,
            &self.scales,
            self.cols,
            r0,
            n,
            &xq.q,
            &xq.scales,
            out,
        );
    }

    /// Multi-lane head matvec: each q8 row is read once and dotted
    /// against every lane's quantized input; per-lane results are
    /// bit-identical to `matvec`.
    pub(super) fn matvec_multi(&self, xs: &[&[f32]], outs: &mut [&mut [f32]]) {
        assert_eq!(xs.len(), outs.len());
        let lanes = xs.len();
        if lanes == 0 {
            return;
        }
        let (rows, cols) = (self.rows, self.cols);
        let xqs: Vec<crate::quant::q8::Q8Vec> =
            xs.iter().map(|x| crate::quant::q8::quantize_q8(x)).collect();
        let p = crate::model::pool::pool();
        let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
        if njobs <= 1 {
            for r in 0..rows {
                for l in 0..lanes {
                    outs[l][r] = self.row_dot(r, &xqs[l]);
                }
            }
            return;
        }
        // main thread: run on the PERSISTENT pool (the scoped path below
        // spawns OS threads per call, which dominated small verify
        // batches). Same kernels, same order - bit-identical.
        if !crate::model::pool::in_pool_worker() {
            let step = dyn_step(rows, njobs);
            let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let qp = crate::model::pool::SPtrU8(self.q.as_ptr() as *const u8);
            let sp = crate::model::pool::SPtr(self.scales.as_ptr());
            let (pack_q, pack_s) = self
                .gemm_pack
                .get_or_init(|| build_gemm_pack(&self.q, &self.scales, rows, cols));
            let pqp = crate::model::pool::SPtrU8(pack_q.as_ptr() as *const u8);
            let psp = crate::model::pool::SPtr(pack_s.as_ptr());
            let (pq_len, ps_len) = (pack_q.len(), pack_s.len());
            let lane_ptrs: Vec<(usize, usize, usize)> = xqs
                .iter()
                .map(|x| (x.q.as_ptr() as usize, x.scales.as_ptr() as usize, x.scales.len()))
                .collect();
            let out_ptrs: Vec<usize> = outs.iter_mut().map(|o| o.as_mut_ptr() as usize).collect();
            let nb = cols / 32;
            let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
            for _ in 0..njobs {
                let ctr = ctr.clone();
                let lane_ptrs = lane_ptrs.clone();
                let out_ptrs = out_ptrs.clone();
                jobs.push(Box::new(move || {
                    let (qp, sp, pqp, psp) = (qp, sp, pqp, psp);
                    // SAFETY: the pool barrier outlives every borrow; each
                    // (row, lane) cell is written exactly once.
                    unsafe {
                        let q = std::slice::from_raw_parts(qp.0 as *const i8, rows * cols);
                        let sc = std::slice::from_raw_parts(sp.0, rows * nb);
                        let lanes_v: Vec<(&[i8], &[f32])> = lane_ptrs
                            .iter()
                            .map(|&(qq, ss, n)| {
                                (
                                    std::slice::from_raw_parts(qq as *const i8, n * 32),
                                    std::slice::from_raw_parts(ss as *const f32, n),
                                )
                            })
                            .collect();
                        let pq = std::slice::from_raw_parts(pqp.0 as *const i8, pq_len);
                        let ps = std::slice::from_raw_parts(psp.0, ps_len);
                        loop {
                            let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                            if r0 >= rows {
                                break;
                            }
                            let r1 = (r0 + step).min(rows);
                            multi_rows_q8_packed(
                                q, sc, pq, ps, rows, cols, r0, r1, &lanes_v, &out_ptrs,
                            );
                        }
                    }
                }));
            }
            p.run(jobs);
            return;
        }
        // dynamic row scheduling (see Q8Head::matvec): fine chunks off a
        // shared counter. Row quads run the same rows4_dot_fma as the
        // pooled matvec, so every path stays bit-identical per row.
        let step = dyn_step(rows, njobs);
        let ctr = std::sync::atomic::AtomicUsize::new(0);
        let out_ptrs: Vec<usize> = outs.iter_mut().map(|o| o.as_mut_ptr() as usize).collect();
        let nb = cols / 32;
        std::thread::scope(|scope| {
            for _ in 0..njobs {
                let this = &*self;
                let xqs = &xqs;
                let ctr = &ctr;
                let out_ptrs = out_ptrs.clone();
                scope.spawn(move || {
                    loop {
                        let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                        if r0 >= rows {
                            break;
                        }
                        let r1 = (r0 + step).min(rows);
                        let mut r = r0;
                        #[cfg(target_arch = "aarch64")]
                        if crate::quant::q8::sdot4_available() {
                            while r + 4 <= r1 {
                                // 4x4 register tile: weights load once per
                                // block for four lanes at a time
                                let mut l0 = 0usize;
                                while l0 + 2 <= xqs.len() {
                                    let width = (xqs.len() - l0).min(8);
                                    let lane_refs: Vec<(&[i8], &[f32])> = xqs[l0..l0 + width]
                                        .iter()
                                        .map(|x| (&x.q[..], &x.scales[..]))
                                        .collect();
                                    // SAFETY: dotprod checked; slices hold nb blocks
                                    let tile = unsafe {
                                        crate::quant::q8::rows4_dot_fma_x4(
                                            [
                                                &this.q[r * cols..(r + 1) * cols],
                                                &this.q[(r + 1) * cols..(r + 2) * cols],
                                                &this.q[(r + 2) * cols..(r + 3) * cols],
                                                &this.q[(r + 3) * cols..(r + 4) * cols],
                                            ],
                                            [
                                                &this.scales[r * nb..(r + 1) * nb],
                                                &this.scales[(r + 1) * nb..(r + 2) * nb],
                                                &this.scales[(r + 2) * nb..(r + 3) * nb],
                                                &this.scales[(r + 3) * nb..(r + 4) * nb],
                                            ],
                                            &lane_refs,
                                        )
                                    };
                                    for (dl, lane_out) in tile.iter().take(width).enumerate() {
                                        for k in 0..4 {
                                            unsafe {
                                                *(out_ptrs[l0 + dl] as *mut f32).add(r + k) =
                                                    lane_out[k];
                                            }
                                        }
                                    }
                                    l0 += width;
                                }
                                for (l, xq) in xqs.iter().enumerate().skip(l0) {
                                    // SAFETY: dotprod checked; slices hold nb blocks
                                    let sums = unsafe {
                                        crate::quant::q8::rows4_dot_fma(
                                            [
                                                &this.q[r * cols..(r + 1) * cols],
                                                &this.q[(r + 1) * cols..(r + 2) * cols],
                                                &this.q[(r + 2) * cols..(r + 3) * cols],
                                                &this.q[(r + 3) * cols..(r + 4) * cols],
                                            ],
                                            [
                                                &this.scales[r * nb..(r + 1) * nb],
                                                &this.scales[(r + 1) * nb..(r + 2) * nb],
                                                &this.scales[(r + 2) * nb..(r + 3) * nb],
                                                &this.scales[(r + 3) * nb..(r + 4) * nb],
                                            ],
                                            &xq.q,
                                            &xq.scales,
                                        )
                                    };
                                    for k in 0..4 {
                                        unsafe { *(out_ptrs[l] as *mut f32).add(r + k) = sums[k] };
                                    }
                                }
                                r += 4;
                            }
                        }
                        while r < r1 {
                            for l in 0..xqs.len() {
                                let v = this.row_dot(r, &xqs[l]);
                                unsafe { *(out_ptrs[l] as *mut f32).add(r) = v };
                            }
                            r += 1;
                        }
                    }
                });
            }
        });
    }

    /// out[r] = <row r, x> computed in integer per 32-block, rescaled to f32.
    /// Same pool split as matvec_cpu.
    pub(super) fn matvec(&self, x: &[f32], out: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        let nb = cols / 32;
        let xq = crate::quant::q8::quantize_q8(x);
        let p = crate::model::pool::pool();
        let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
        if njobs <= 1 {
            for (r, o) in out.iter_mut().enumerate() {
                *o = self.row_dot(r, &xq);
            }
            return;
        }
        // dynamic row scheduling: the workers pull fine chunks from a
        // shared counter instead of owning one fixed range each, so a
        // straggler (E-core, interrupt) delays one chunk, not a quarter of
        // the matrix. Per-row math unchanged - bit-identical results.
        let step = dyn_step(rows, njobs);
        let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let qp = crate::model::pool::SPtrU8(self.q.as_ptr() as *const u8);
        let sp = crate::model::pool::SPtr(self.scales.as_ptr());
        let xp = crate::model::pool::SPtrU8(xq.q.as_ptr() as *const u8);
        let xsp = crate::model::pool::SPtr(xq.scales.as_ptr());
        let op = crate::model::pool::MPtr(out.as_mut_ptr());
        let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
        for _ in 0..njobs {
            let ctr = ctr.clone();
            jobs.push(Box::new(move || {
                // rebind → capture whole structs (Send), not fields
                let (qp, sp, xp, xsp, op) = (qp, sp, xp, xsp, op);
                unsafe {
                    let q = std::slice::from_raw_parts(qp.0 as *const i8, rows * cols);
                    let ws = std::slice::from_raw_parts(sp.0, rows * nb);
                    let xq8 = std::slice::from_raw_parts(xp.0 as *const i8, cols);
                    let xs = std::slice::from_raw_parts(xsp.0, nb);
                    let out = std::slice::from_raw_parts_mut(op.0, rows);
                    loop {
                        let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                        if r0 >= rows {
                            break;
                        }
                        let r1 = (r0 + step).min(rows);
                        q8_rows_dot(q, ws, cols, r0, r1 - r0, xq8, xs, &mut out[r0..r1]);
                    }
                }
            }));
        }
        p.run(jobs);
    }

    /// Exact i8 transfer of an MXFP4 matrix: the e2m1 nibbles decode to
    /// the LUT2 integers (E2M1 x 2) and the per-block scale absorbs the
    /// halving (2^(sb-128)), so integer dots and scales are IDENTICAL to
    /// the packed kernels' - only the nibble unpacking moves from every
    /// matvec to this one load-time pass. Rows convert in parallel.
    pub(crate) fn from_packed_fp4(packed: &[u8], scales_b: &[u8], rows: usize, cols: usize) -> Q8Head {
        assert!(cols % 32 == 0, "q8 blocks are 32 wide");
        let nb = cols / 32;
        let mut q = vec![0i8; rows * cols];
        let mut scales = vec![0f32; rows * nb];
        let workers = crate::model::pool::pool().workers.max(1).min(rows.max(1));
        let chunk = rows.div_ceil(workers);
        std::thread::scope(|s| {
            let mut q_rest = q.as_mut_slice();
            let mut s_rest = scales.as_mut_slice();
            let mut r0 = 0usize;
            while r0 < rows {
                let r1 = (r0 + chunk).min(rows);
                let (q_c, qr) = q_rest.split_at_mut((r1 - r0) * cols);
                let (s_c, sr) = s_rest.split_at_mut((r1 - r0) * nb);
                q_rest = qr;
                s_rest = sr;
                s.spawn(move || {
                    for (i, r) in (r0..r1).enumerate() {
                        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
                        for c in 0..cols {
                            let byte = prow[c / 2];
                            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                            q_c[i * cols + c] = crate::quant::q8::E2M1_X2[nib as usize];
                        }
                        for g in 0..nb {
                            s_c[i * nb + g] =
                                crate::quant::mxfp4::exp2_i(scales_b[r * nb + g] as i32 - 128);
                        }
                    }
                });
                r0 = r1;
            }
        });
        Q8Head { q, scales, rows, cols, gemm_pack: std::sync::OnceLock::new() }
    }

    /// Two heads, one input: the activation is quantized ONCE and both
    /// matrices run under ONE pool barrier (a shared dynamic counter
    /// walks a's rows then b's). Per-row math is row_dot/q8_rows_dot
    /// exactly, so results are bit-identical to two separate matvecs -
    /// only the sync and the quantization are halved. Callers: the
    /// gate/up MLP pair and the in_qkv/in_z projection pair.
    pub(crate) fn matvec2(a: &Q8Head, b: &Q8Head, x: &[f32], out_a: &mut [f32], out_b: &mut [f32]) {
        debug_assert_eq!(a.cols, b.cols);
        let xq = crate::quant::q8::quantize_q8(x);
        let p = crate::model::pool::pool();
        let total = a.rows + b.rows;
        let njobs = ((a.rows * a.cols + b.rows * b.cols) / 60_000).clamp(1, p.workers).min(total);
        if njobs <= 1 {
            for (r, o) in out_a.iter_mut().enumerate() {
                *o = a.row_dot(r, &xq);
            }
            for (r, o) in out_b.iter_mut().enumerate() {
                *o = b.row_dot(r, &xq);
            }
            return;
        }
        let step = dyn_step(total, njobs);
        let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (nba, nbb) = (a.cols / 32, b.cols / 32);
        let aq = crate::model::pool::SPtrU8(a.q.as_ptr() as *const u8);
        let asc = crate::model::pool::SPtr(a.scales.as_ptr());
        let bq = crate::model::pool::SPtrU8(b.q.as_ptr() as *const u8);
        let bsc = crate::model::pool::SPtr(b.scales.as_ptr());
        let xp = crate::model::pool::SPtrU8(xq.q.as_ptr() as *const u8);
        let xsp = crate::model::pool::SPtr(xq.scales.as_ptr());
        let oa = crate::model::pool::MPtr(out_a.as_mut_ptr());
        let ob = crate::model::pool::MPtr(out_b.as_mut_ptr());
        let (ar, ac, br, bc) = (a.rows, a.cols, b.rows, b.cols);
        let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
        for _ in 0..njobs {
            let ctr = ctr.clone();
            jobs.push(Box::new(move || {
                let (aq, asc, bq, bsc, xp, xsp, oa, ob) = (aq, asc, bq, bsc, xp, xsp, oa, ob);
                // SAFETY: the pool barrier outlives every borrow; row
                // ranges are disjoint across pulls.
                unsafe {
                    let qa = std::slice::from_raw_parts(aq.0 as *const i8, ar * ac);
                    let sa = std::slice::from_raw_parts(asc.0, ar * nba);
                    let qb = std::slice::from_raw_parts(bq.0 as *const i8, br * bc);
                    let sb = std::slice::from_raw_parts(bsc.0, br * nbb);
                    let xq8 = std::slice::from_raw_parts(xp.0 as *const i8, ac);
                    let xs = std::slice::from_raw_parts(xsp.0, nba);
                    loop {
                        let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                        if r0 >= total {
                            break;
                        }
                        let r1 = (r0 + step).min(total);
                        // the range may straddle the a/b boundary
                        if r0 < ar {
                            let ra1 = r1.min(ar);
                            let out = std::slice::from_raw_parts_mut(oa.0, ar);
                            q8_rows_dot(qa, sa, ac, r0, ra1 - r0, xq8, xs, &mut out[r0..ra1]);
                        }
                        if r1 > ar {
                            let rb0 = r0.max(ar) - ar;
                            let rb1 = r1 - ar;
                            let out = std::slice::from_raw_parts_mut(ob.0, br);
                            q8_rows_dot(qb, sb, bc, rb0, rb1 - rb0, xq8, xs, &mut out[rb0..rb1]);
                        }
                    }
                }
            }));
        }
        p.run(jobs);
    }

    /// Builds the GEMM pack eagerly (load time): the lazy build was
    /// measured inside the first prefill - up to a second of interleaving
    /// billed to prompt reading.
    pub(crate) fn prebuild_gemm(&self) {
        let _ = self
            .gemm_pack
            .get_or_init(|| build_gemm_pack(&self.q, &self.scales, self.rows, self.cols));
    }

    fn row_dot(&self, r: usize, xq: &crate::quant::q8::Q8Vec) -> f32 {
        let nb = self.cols / 32;
        let wq = &self.q[r * self.cols..(r + 1) * self.cols];
        let ws = &self.scales[r * nb..(r + 1) * nb];
        let mut acc = 0f32;
        for g in 0..nb {
            let idot = crate::quant::q8::block_dot_i8(&wq[g * 32..g * 32 + 32], &xq.q[g * 32..g * 32 + 32]);
            // fused, block-sequential: the exact order of rows4_dot_fma,
            // so every q8 path (pooled quad, st, multi) stays bit-equal
            acc = (idot as f32).mul_add(ws[g] * xq.scales[g], acc);
        }
        acc
    }
}

/// dot() for 8 positions at once against the same weight row, positions in
/// vector lanes: the row is loaded once for all eight and the vector unit
/// works across positions, while each position keeps the exact 8-lane
/// accumulation order of dot() (bit-identical results).
/// `xt` is x transposed: xt[c * n + t].
/// dot8t scalar (reference + fallback): 8 positions at once against the
/// same weight row. Each position keeps the exact 8-accumulator order of
/// dot() (bit-identical); the SIMD kernels below replay the same per-lane
/// mul-then-add and the same reduction order, so they are bit-identical to
/// this reference BY CONSTRUCTION (see the contract above dot()).
#[inline]
#[allow(dead_code)] // on aarch64 the dispatched dot8t never reaches this
fn dot8t_scalar(wr: &[f32], xt: &[f32], n: usize, t0: usize) -> [f32; 8] {
    let mut acc = [[0f32; 8]; 8]; // acc[lane][position]
    let mut cw = wr.chunks_exact(8);
    let mut c = 0;
    for w8 in &mut cw {
        for (j, &wv) in w8.iter().enumerate() {
            let xc = &xt[(c + j) * n + t0..(c + j) * n + t0 + 8];
            for p in 0..8 {
                acc[j][p] += wv * xc[p];
            }
        }
        c += 8;
    }
    let mut s = [0f32; 8];
    for p in 0..8 {
        s[p] = (acc[0][p] + acc[1][p]) + (acc[2][p] + acc[3][p]) + (acc[4][p] + acc[5][p]) + (acc[6][p] + acc[7][p]);
    }
    for (i, &wv) in cw.remainder().iter().enumerate() {
        let xc = &xt[(c + i) * n + t0..(c + i) * n + t0 + 8];
        for p in 0..8 {
            s[p] += wv * xc[p];
        }
    }
    s
}

/// NEON dot8t: the 8 column-accumulators live in 16 float32x4 registers
/// (positions in lanes, acc[j][0] = positions 0-3, acc[j][1] = 4-7). Each
/// lane sees the same mul-then-add as the scalar kernel (vaddq of vmulq,
/// never fma). The per-position reduction replays the scalar reduction
/// order exactly, so the result is bit-identical to dot8t_scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot8t_neon(wr: &[f32], xt: &[f32], n: usize, t0: usize) -> [f32; 8] {
    use std::arch::aarch64::*;
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); 2]; 8];
        let xp = xt.as_ptr();
        let mut cw = wr.chunks_exact(8);
        let mut c = 0usize;
        for w8 in &mut cw {
            for (j, &wv) in w8.iter().enumerate() {
                let w = vdupq_n_f32(wv);
                let p = xp.add((c + j) * n + t0);
                acc[j][0] = vaddq_f32(acc[j][0], vmulq_f32(w, vld1q_f32(p)));
                acc[j][1] = vaddq_f32(acc[j][1], vmulq_f32(w, vld1q_f32(p.add(4))));
            }
            c += 8;
        }
        let mut accs = [[0f32; 8]; 8];
        for j in 0..8 {
            vst1q_f32(accs[j].as_mut_ptr(), acc[j][0]);
            vst1q_f32(accs[j].as_mut_ptr().add(4), acc[j][1]);
        }
        let mut s = [0f32; 8];
        for p in 0..8 {
            s[p] = (accs[0][p] + accs[1][p]) + (accs[2][p] + accs[3][p]) + (accs[4][p] + accs[5][p]) + (accs[6][p] + accs[7][p]);
        }
        for (i, &wv) in cw.remainder().iter().enumerate() {
            let xc = &xt[(c + i) * n + t0..(c + i) * n + t0 + 8];
            for p in 0..8 {
                s[p] += wv * xc[p];
            }
        }
        s
    }
}

/// AVX2 dot8t: the 8 column-accumulators live in 8 __m256 registers (the 8
/// positions in lanes). Same mul-then-add discipline (never fmadd), same
/// reduction replay: bit-identical to dot8t_scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot8t_avx2(wr: &[f32], xt: &[f32], n: usize, t0: usize) -> [f32; 8] {
    use std::arch::x86_64::*;
    unsafe {
        let mut acc = [_mm256_setzero_ps(); 8];
        let xp = xt.as_ptr();
        let mut cw = wr.chunks_exact(8);
        let mut c = 0usize;
        for w8 in &mut cw {
            for (j, &wv) in w8.iter().enumerate() {
                let w = _mm256_set1_ps(wv);
                acc[j] = _mm256_add_ps(acc[j], _mm256_mul_ps(w, _mm256_loadu_ps(xp.add((c + j) * n + t0))));
            }
            c += 8;
        }
        let mut accs = [[0f32; 8]; 8];
        for j in 0..8 {
            _mm256_storeu_ps(accs[j].as_mut_ptr(), acc[j]);
        }
        let mut s = [0f32; 8];
        for p in 0..8 {
            s[p] = (accs[0][p] + accs[1][p]) + (accs[2][p] + accs[3][p]) + (accs[4][p] + accs[5][p]) + (accs[6][p] + accs[7][p]);
        }
        for (i, &wv) in cw.remainder().iter().enumerate() {
            let xc = &xt[(c + i) * n + t0..(c + i) * n + t0 + 8];
            for p in 0..8 {
                s[p] += wv * xc[p];
            }
        }
        s
    }
}

/// dot8t: 8 positions at once against the same weight row, dispatched to
/// the widest bit-identical SIMD kernel (positions in vector lanes: the row
/// is loaded once for all eight).
#[inline]
pub(super) fn dot8t(wr: &[f32], xt: &[f32], n: usize, t0: usize) -> [f32; 8] {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is baseline on aarch64: unconditional, zero dispatch cost
        return unsafe { dot8t_neon(wr, xt, n, t0) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_available() {
            return unsafe { dot8t_avx2(wr, xt, n, t0) };
        }
    }
    #[allow(unreachable_code)]
    dot8t_scalar(wr, xt, n, t0)
}

/// dot4t: dot8t for a tail block of 4 positions.
#[inline]
fn dot4t(wr: &[f32], xt: &[f32], n: usize, t0: usize) -> [f32; 4] {
    let mut acc = [[0f32; 4]; 8];
    let mut cw = wr.chunks_exact(8);
    let mut c = 0;
    for w8 in &mut cw {
        for (j, &wv) in w8.iter().enumerate() {
            let xc = &xt[(c + j) * n + t0..(c + j) * n + t0 + 4];
            for p in 0..4 {
                acc[j][p] += wv * xc[p];
            }
        }
        c += 8;
    }
    let mut s = [0f32; 4];
    for p in 0..4 {
        s[p] = (acc[0][p] + acc[1][p]) + (acc[2][p] + acc[3][p]) + (acc[4][p] + acc[5][p]) + (acc[6][p] + acc[7][p]);
    }
    for (i, &wv) in cw.remainder().iter().enumerate() {
        let xc = &xt[(c + i) * n + t0..(c + i) * n + t0 + 4];
        for p in 0..4 {
            s[p] += wv * xc[p];
        }
    }
    s
}

/// gemm_batch over the row range r0..r1 (shared by the inline and the
/// pooled paths). `x` is position-major [n * cols], `xt` its transpose
/// [cols * n] (None when n < 4: plain per-position dots).
#[allow(clippy::too_many_arguments)]
fn gemm_rows(w: &[f32], rows: usize, cols: usize, x: &[f32], xt: Option<&[f32]>, n: usize, out: &mut [f32], r0: usize, r1: usize) {
    for r in r0..r1 {
        let wr = &w[r * cols..(r + 1) * cols];
        let mut t = 0;
        if let Some(xt) = xt {
            while t + 8 <= n {
                let r8 = dot8t(wr, xt, n, t);
                for p in 0..8 {
                    out[(t + p) * rows + r] = r8[p];
                }
                t += 8;
            }
            if t + 4 <= n {
                let r4 = dot4t(wr, xt, n, t);
                for p in 0..4 {
                    out[(t + p) * rows + r] = r4[p];
                }
                t += 4;
            }
        }
        while t < n {
            out[t * rows + r] = dot(wr, &x[t * cols..(t + 1) * cols]);
            t += 1;
        }
    }
}

/// Batched matvec (position-major GEMM) for prefill:
/// out[t * rows + r] = dot(w_row_r, x_t) for t in 0..n.
/// Loop order is row-major: each weight row is read once and dotted against
/// all n position vectors, so the weights are streamed from RAM once per
/// layer instead of once per token. Bit-identical to n separate matvec
/// calls: each (row, position) dot keeps the exact same accumulation order.
pub fn gemm_batch(w: &[f32], rows: usize, cols: usize, x: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), n * cols);
    debug_assert_eq!(out.len(), n * rows);
    // transposed copy of x for the position-in-lane kernel (n >= 4 only)
    let xt: Option<Vec<f32>> = if n >= 4 {
        let mut xt = vec![0f32; n * cols];
        for t in 0..n {
            for c in 0..cols {
                xt[c * n + t] = x[t * cols + c];
            }
        }
        Some(xt)
    } else {
        None
    };
    let p = crate::model::pool::pool();
    let njobs = (rows * cols * n / 240_000).clamp(1, p.workers).min(rows);
    if njobs <= 1 {
        gemm_rows(w, rows, cols, x, xt.as_deref(), n, out, 0, rows);
        return;
    }
    let chunk = rows.div_ceil(njobs);
    let wp = crate::model::pool::SPtr(w.as_ptr());
    let xp = crate::model::pool::SPtr(x.as_ptr());
    let xtp = xt.as_ref().map(|v| crate::model::pool::SPtr(v.as_ptr()));
    let op = crate::model::pool::MPtr(out.as_mut_ptr());
    let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
    for j in 0..njobs {
        let (r0, r1) = (j * chunk, ((j + 1) * chunk).min(rows));
        if r0 >= r1 {
            break;
        }
        jobs.push(Box::new(move || {
            // rebind → capture whole structs (Send), not fields
            let (wp, xp, xtp, op) = (wp, xp, xtp, op);
            unsafe {
                let w = std::slice::from_raw_parts(wp.0, rows * cols);
                let x = std::slice::from_raw_parts(xp.0, n * cols);
                let xt = xtp.map(|p| std::slice::from_raw_parts(p.0, n * cols));
                let out = std::slice::from_raw_parts_mut(op.0, n * rows);
                gemm_rows(w, rows, cols, x, xt, n, out, r0, r1);
            }
        }));
    }
    p.run(jobs);
}

pub fn rmsnorm(cfg: &Config, x: &[f32], w: &[f32], out: &mut [f32]) {
    let ss = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

#[inline]
pub(super) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub(super) fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// SiTU: a = 4·tanh(g/4)·sigmoid(g) ; u = 25·tanh(u/25) ; out = a·u
#[inline]
pub fn situ(g: f32, u: f32) -> f32 {
    4.0 * (g / 4.0).tanh() * sigmoid(g) * (25.0 * (u / 25.0).tanh())
}

/// AttnRes: softmax over the RMS-normed scores of blocks + prefix,
/// RAW values as output. `w` = norm.weight · proj.weight (pre-combined).
pub fn attn_res(cfg: &Config, prefix: &[f32], blocks: &[Vec<f32>], w: &[f32], out: &mut [f32]) {
    let refs: Vec<&[f32]> = blocks.iter().map(|b| b.as_slice()).collect();
    attn_res_refs(cfg, prefix, &refs, w, out);
}

/// Slice-based core of attn_res, shared by the single-token path and the
/// batched prefill (each block slice is one position of a [n * d] buffer).
pub(super) fn attn_res_refs(cfg: &Config, prefix: &[f32], blocks: &[&[f32]], w: &[f32], out: &mut [f32]) {
    let b = blocks.len();
    let mut scores = vec![0f32; b + 1];
    let mut kbuf = vec![0f32; cfg.d];
    let mut score_of = |v: &[f32]| {
        let ss = dot(v, v) / cfg.d as f32;
        let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
        for j in 0..cfg.d {
            kbuf[j] = v[j] * inv;
        }
        dot(&kbuf, w)
    };
    for (i, v) in blocks.iter().enumerate() {
        scores[i] = score_of(v);
    }
    scores[b] = score_of(prefix);
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let mut z = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    for s in scores.iter_mut() {
        *s /= z;
    }
    for j in 0..cfg.d {
        out[j] = 0.0;
    }
    for (i, v) in blocks.iter().enumerate() {
        for j in 0..cfg.d {
            out[j] += scores[i] * v[j];
        }
    }
    let p = scores[b];
    for j in 0..cfg.d {
        out[j] += p * prefix[j];
    }
}

// ── weights: (offset, dims) descriptors into the file ──

/// MXFP4 packed matvec for a batch of m inputs (positions in vector lanes,
/// blocks of 8): the packed weights are read and dequantized ONCE per
/// element for the whole block instead of once per position.
/// `xt` = the m inputs transposed [cols * m], zero-padded so that m is a
/// multiple of 8 (pad columns produce zero outputs, ignored by the caller);
/// `out` is position-major [m * rows]. Per position the accumulation is the
/// exact per-group sequence of mxfp4::matvec_packed (gsum over the 32
/// columns of a group in order, then sum += gsum * scale): bit-identical
/// results.
/// Loop order is row-outer: each row's nibbles are decoded to their f32 LUT
/// values ONCE (they were re-decoded for every 8-position tile - 75x
/// redundant work on a 600-token prompt), then all tiles run against the
/// decoded row. Outputs are independent per (row, tile), so the swap and
/// the hoisted decode keep the result bit-identical.
pub(crate) fn matvec_packed_nt(packed: &[u8], scales: &[u8], rows: usize, cols: usize, xt: &[f32], m: usize, out: &mut [f32]) {
    use crate::quant::mxfp4::{E2M1, exp2_i};
    debug_assert_eq!(cols % 32, 0);
    debug_assert_eq!(m % 8, 0);
    let mut wrow = vec![0f32; cols];
    for r in 0..rows {
        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
        for c in 0..cols {
            let byte = prow[c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            wrow[c] = E2M1[nib as usize];
        }
        for t0 in (0..m).step_by(8) {
            let mut sum = [0f32; 8];
            for g in 0..cols / 32 {
                let mut gsum = [0f32; 8];
                for j in 0..32 {
                    let c = g * 32 + j;
                    let wv = wrow[c];
                    let xc = &xt[c * m + t0..c * m + t0 + 8];
                    for p in 0..8 {
                        gsum[p] += wv * xc[p];
                    }
                }
                let sc = exp2_i(srow[g] as i32 - 127);
                for p in 0..8 {
                    sum[p] += gsum[p] * sc;
                }
            }
            for p in 0..8 {
                out[(t0 + p) * rows + r] = sum[p];
            }
        }
    }
}

/// `microkimi kernbench`: isolates the multi-lane q8 GEMM from the
/// engine - synthetic 3072x1024 head against 256 lanes, plus the
/// single-thread tile and the single-lane pooled matvec, in GMAC/s.
/// The datum that decides whether a prompt-reading gap lives in the
/// kernel, the threading, or the host.
pub fn kernbench_cmd(_args: &[String]) {
    let (rows, cols, lanes_n) = (3072usize, 1024usize, 256usize);
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7 + 3) % 23) as f32 * 0.01 - 0.1).collect();
    let head = Q8Head::from_f32(&w, rows, cols);
    let xs_data: Vec<Vec<f32>> = (0..lanes_n)
        .map(|l| (0..cols).map(|i| ((i * 5 + l * 11 + 1) % 13) as f32 * 0.02 - 0.1).collect())
        .collect();
    let macs_multi = (rows * cols * lanes_n) as f64;

    // multi-lane pooled GEMM (the prefill path)
    let mut outs_data: Vec<Vec<f32>> = vec![vec![0.0f32; rows]; lanes_n];
    for round in 0..3 {
        let xs: Vec<&[f32]> = xs_data.iter().map(|x| x.as_slice()).collect();
        let mut outs: Vec<&mut [f32]> = outs_data.iter_mut().map(|o| o.as_mut_slice()).collect();
        let t0 = std::time::Instant::now();
        head.matvec_multi(&xs, &mut outs);
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "multi ({} lanes, pooled)   round {}: {:>7.1} GMAC/s  ({:.1} ms)",
            lanes_n,
            round,
            macs_multi / dt / 1e9,
            dt * 1000.0
        );
    }

    // single-thread 4x8 tile (the raw kernel ceiling per core)
    #[cfg(target_arch = "aarch64")]
    if crate::quant::q8::sdot4_available() {
        let nb = cols / 32;
        let xqs: Vec<crate::quant::q8::Q8Vec> =
            xs_data[..8].iter().map(|x| crate::quant::q8::quantize_q8(x)).collect();
        let lanes: Vec<(&[i8], &[f32])> =
            xqs.iter().map(|x| (&x.q[..], &x.scales[..])).collect();
        let mut sink = 0.0f32;
        let t0 = std::time::Instant::now();
        let mut r = 0usize;
        while r + 4 <= rows {
            let w4 = [
                &head.q[r * cols..(r + 1) * cols],
                &head.q[(r + 1) * cols..(r + 2) * cols],
                &head.q[(r + 2) * cols..(r + 3) * cols],
                &head.q[(r + 3) * cols..(r + 4) * cols],
            ];
            let s4 = [
                &head.scales[r * nb..(r + 1) * nb],
                &head.scales[(r + 1) * nb..(r + 2) * nb],
                &head.scales[(r + 2) * nb..(r + 3) * nb],
                &head.scales[(r + 3) * nb..(r + 4) * nb],
            ];
            // SAFETY: dotprod checked above; slices hold nb blocks each.
            let tile = unsafe { crate::quant::q8::rows4_dot_fma_x4(w4, s4, &lanes) };
            sink += tile[0][0];
            r += 4;
        }
        let dt = t0.elapsed().as_secs_f64();
        println!(
            "tile 4x8 (single thread)  : {:>7.1} GMAC/s  ({:.1} ms, sink {:.3})",
            (rows * cols * 8) as f64 / dt / 1e9,
            dt * 1000.0,
            sink
        );
    }

    // single-lane pooled matvec (the decode shape)
    let mut out1 = vec![0.0f32; rows];
    let t0 = std::time::Instant::now();
    for _ in 0..64 {
        head.matvec(&xs_data[0], &mut out1);
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "matvec x64 (pooled, 1 lane): {:>7.1} GMAC/s  ({:.1} ms total)",
        (rows * cols * 64) as f64 / dt / 1e9,
        dt * 1000.0
    );
}
