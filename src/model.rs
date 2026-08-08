// microkimi forward pass: 93 layers, AttnRes block 12, KDA (69),
// MLA NoPE (24), latent MoE 896 experts top-16 + 2 shared (layers 1..92),
// dense MLP layer 0, SiTU everywhere, MXFP4 experts dequantized on the fly.
// All in f32, zero-copy: f32 tensors are read in slices directly from the
// mmap-like file (Vec<u8> + align_to), experts stay packed.

use crate::config::Config;
use crate::tokenizer::AnyTokenizer;
use crate::weights::{BinFile, Entry};
use std::time::Instant;

// ── default microkimi dims - used ONLY by build.rs (micro builder)
// and tests (selftest/parity are micro-specific). The inference engine
// is entirely driven by Config (config.rs, MKIM0002 block or microkimi default).
pub const D: usize = 512;

pub const fn is_mla(l: usize) -> bool {
    l % 4 == 3 || l == 92
}
pub const fn is_moe(l: usize) -> bool {
    l >= 1
}

pub fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

// ── zero-copy bytes → f32 conversion (64-byte alignment guaranteed by the format) ──
fn as_f32(bytes: &[u8]) -> &[f32] {
    let (pre, mid, post) = unsafe { bytes.align_to::<f32>() };
    assert!(pre.is_empty() && post.is_empty(), "unexpected f32 alignment");
    mid
}

/// (bytes actually read from disk, major page faults) from /proc, Linux only.
/// Used to show the mmap demand-paging cost of a prefill.
fn io_stats() -> Option<(u64, u64)> {
    let io = std::fs::read_to_string("/proc/self/io").ok()?;
    let read_bytes = io.lines().find_map(|l| l.strip_prefix("read_bytes:"))?.trim().parse().ok()?;
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // after the ") " of comm, fields are numbered from 3: majflt is field 12
    let majflt = stat.rsplit_once(") ")?.1.split_whitespace().nth(9)?.parse().ok()?;
    Some((read_bytes, majflt))
}

// ── math kernels ──
//
// Bit-exactness contract of dot(): every path (scalar fallback, NEON, AVX2)
// computes the SAME IEEE operations in the SAME order:
//   - 8 parallel accumulators; element j of each 8-wide chunk goes to acc[j]
//     (mul, then add - NEVER a fused multiply-add: FMA skips the intermediate
//     rounding and would drift from the scalar path);
//   - fixed reduction: pairs p01=(a0+a1), p23=(a2+a3), p45=(a4+a5),
//     p67=(a6+a7), then ((p01 + p23) + p45) + p67, left-associative;
//   - the remainder (< 8 elements) is accumulated sequentially into s.
// The SIMD kernels keep the accumulators in vector lanes and replay this
// exact reduction, so they are bit-identical to dot_scalar BY CONSTRUCTION.

/// Scalar dot: the historical path, reference for the SIMD kernels and
/// fallback when no SIMD feature is present.
#[inline]
#[allow(dead_code)] // on aarch64 the dispatched dot() never reaches this
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
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

// ── --gpu flag: routes large matvecs to Metal on macOS ──
//
// GPU_ENABLED is set once from main.rs when --gpu is passed. matvec() is the
// single entry point used by every projection in the engine (KDA/MLA/MoE/
// dense/router/lm_head); when the flag is on and the matvec is large enough,
// it is dispatched to Metal (see metal.rs). Without the flag, behavior is
// bit-identical to the pure-CPU path.
//
// Threshold rationale (64k MACs): encoding + dispatching a Metal command
// buffer costs ~50-100 µs of latency; below ~64k multiply-accumulates the
// CPU finishes faster than the round trip, so small matvecs stay on the CPU.
// Tune later with real measurements on device.
pub static GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(target_os = "macos")]
// GPU threshold measured on Apple M5: each Metal dispatch costs ~0.25 ms of
// sync latency, and the micro model runs ~1 200 matvecs per token - only
// genuinely large matvecs (lm_head: 163840×512 = 84M MACs) are a net win on
// GPU. Smaller ones stay on the CPU thread pool.
pub const GPU_MIN_ELEMS: usize = 2 * 1024 * 1024;

pub fn set_gpu(on: bool) {
    GPU_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn gpu_on() -> bool {
    GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// f32 matrix × vector. Entry point for the whole engine.
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_os = "macos")]
    {
        if gpu_on() && rows * cols >= GPU_MIN_ELEMS && crate::metal::gpu_available() {
            crate::metal::gpu_matvec(w, rows, cols, x, out);
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
    let p = crate::pool::pool();
    let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
    if njobs <= 1 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(&w[r * cols..(r + 1) * cols], x);
        }
        return;
    }
    let chunk = rows.div_ceil(njobs);
    let wp = crate::pool::SPtr(w.as_ptr());
    let xp = crate::pool::SPtr(x.as_ptr());
    let op = crate::pool::MPtr(out.as_mut_ptr());
    let mut jobs: Vec<crate::pool::Job> = Vec::new();
    for j in 0..njobs {
        let (r0, r1) = (j * chunk, ((j + 1) * chunk).min(rows));
        if r0 >= r1 {
            break;
        }
        jobs.push(Box::new(move || {
            // rebind → capture whole structs (Send), not fields
            let (wp, xp, op) = (wp, xp, op);
            unsafe {
                let w = std::slice::from_raw_parts(wp.0, rows * cols);
                let x = std::slice::from_raw_parts(xp.0, cols);
                let out = std::slice::from_raw_parts_mut(op.0, rows);
                for r in r0..r1 {
                    out[r] = dot(&w[r * cols..(r + 1) * cols], x);
                }
            }
        }));
    }
    p.run(jobs);
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
}

/// MICROKIMI_Q8HEAD=0 disables the q8 lm_head copy (exact f32 fallback).
fn q8head_enabled() -> bool {
    std::env::var("MICROKIMI_Q8HEAD").map(|v| v != "0").unwrap_or(true)
}

impl Q8Head {
    fn from_f32(w: &[f32], rows: usize, cols: usize) -> Q8Head {
        assert!(cols % 32 == 0, "q8 blocks are 32 wide");
        let nb = cols / 32;
        let mut q = vec![0i8; rows * cols];
        let mut scales = vec![0f32; rows * nb];
        let mut scratch = crate::q8::Q8Vec::new();
        for r in 0..rows {
            crate::q8::quantize_q8_into(&w[r * cols..(r + 1) * cols], &mut scratch);
            q[r * cols..(r + 1) * cols].copy_from_slice(&scratch.q);
            scales[r * nb..(r + 1) * nb].copy_from_slice(&scratch.scales);
        }
        Q8Head { q, scales, rows, cols }
    }

    /// out[r] = <row r, x> computed in integer per 32-block, rescaled to f32.
    /// Same pool split as matvec_cpu.
    fn matvec(&self, x: &[f32], out: &mut [f32]) {
        let (rows, cols) = (self.rows, self.cols);
        let nb = cols / 32;
        let xq = crate::q8::quantize_q8(x);
        let p = crate::pool::pool();
        let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
        if njobs <= 1 {
            for (r, o) in out.iter_mut().enumerate() {
                *o = self.row_dot(r, &xq);
            }
            return;
        }
        let chunk = rows.div_ceil(njobs);
        let qp = crate::pool::SPtrU8(self.q.as_ptr() as *const u8);
        let sp = crate::pool::SPtr(self.scales.as_ptr());
        let xp = crate::pool::SPtrU8(xq.q.as_ptr() as *const u8);
        let xsp = crate::pool::SPtr(xq.scales.as_ptr());
        let op = crate::pool::MPtr(out.as_mut_ptr());
        let mut jobs: Vec<crate::pool::Job> = Vec::new();
        for j in 0..njobs {
            let (r0, r1) = (j * chunk, ((j + 1) * chunk).min(rows));
            if r0 >= r1 {
                break;
            }
            jobs.push(Box::new(move || {
                // rebind → capture whole structs (Send), not fields
                let (qp, sp, xp, xsp, op) = (qp, sp, xp, xsp, op);
                unsafe {
                    let q = std::slice::from_raw_parts(qp.0 as *const i8, rows * cols);
                    let ws = std::slice::from_raw_parts(sp.0, rows * nb);
                    let xq8 = std::slice::from_raw_parts(xp.0 as *const i8, cols);
                    let xs = std::slice::from_raw_parts(xsp.0, nb);
                    let out = std::slice::from_raw_parts_mut(op.0, rows);
                    for r in r0..r1 {
                        let mut acc = 0f32;
                        for g in 0..nb {
                            let idot = crate::q8::block_dot_i8(&q[r * cols + g * 32..r * cols + g * 32 + 32], &xq8[g * 32..g * 32 + 32]);
                            acc += ws[r * nb + g] * xs[g] * idot as f32;
                        }
                        out[r] = acc;
                    }
                }
            }));
        }
        p.run(jobs);
    }

    fn row_dot(&self, r: usize, xq: &crate::q8::Q8Vec) -> f32 {
        let nb = self.cols / 32;
        let wq = &self.q[r * self.cols..(r + 1) * self.cols];
        let ws = &self.scales[r * nb..(r + 1) * nb];
        let mut acc = 0f32;
        for g in 0..nb {
            let idot = crate::q8::block_dot_i8(&wq[g * 32..g * 32 + 32], &xq.q[g * 32..g * 32 + 32]);
            acc += ws[g] * xq.scales[g] * idot as f32;
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
fn dot8t(wr: &[f32], xt: &[f32], n: usize, t0: usize) -> [f32; 8] {
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
    let p = crate::pool::pool();
    let njobs = (rows * cols * n / 240_000).clamp(1, p.workers).min(rows);
    if njobs <= 1 {
        gemm_rows(w, rows, cols, x, xt.as_deref(), n, out, 0, rows);
        return;
    }
    let chunk = rows.div_ceil(njobs);
    let wp = crate::pool::SPtr(w.as_ptr());
    let xp = crate::pool::SPtr(x.as_ptr());
    let xtp = xt.as_ref().map(|v| crate::pool::SPtr(v.as_ptr()));
    let op = crate::pool::MPtr(out.as_mut_ptr());
    let mut jobs: Vec<crate::pool::Job> = Vec::new();
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
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
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
fn attn_res_refs(cfg: &Config, prefix: &[f32], blocks: &[&[f32]], w: &[f32], out: &mut [f32]) {
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

#[derive(Clone, Copy)]
struct T {
    off: usize,
    len: usize, // in f32
}

impl T {
    fn from(e: &Entry) -> T {
        let len: usize = e.dims.iter().map(|&d| d as usize).product();
        T { off: e.offset as usize, len }
    }
}

struct KdaW {
    q_proj: T,
    k_proj: T,
    v_proj: T,
    q_conv: T,
    k_conv: T,
    v_conv: T,
    f_a: T,
    f_b: T,
    a_log: T,
    dt_bias: T,
    b_proj: T,
    g_proj: T,
    o_norm: T,
    o_proj: T,
}

struct MlaW {
    q_a: T,
    q_a_ln: T,
    q_b: T,
    kv_a: T,
    kv_a_ln: T,
    kv_b: T,
    g_proj: T,
    o_proj: T,
}

enum AttnW {
    Kda(KdaW),
    Mla(MlaW),
}

struct MoeW {
    gate_w: T,
    gate_b: T,
    routed_down: T,
    routed_up: T,
    routed_norm: T,
    shared_gate: T,
    shared_up: T,
    shared_down: T,
    experts: Vec<[u64; 3]>, // offsets of the mxfp4/vq1 blobs [w1, w2, w3] per expert
    experts_vq: Vec<bool>,  // true = cold expert stored as VQ1 (DTYPE_VQ1 indices)
    vq_cb: Vec<f32>,        // global VQ codebook [256*16] (empty when no VQ1 tensors)
}

struct DenseW {
    gate: T,
    up: T,
    down: T,
}

enum FfnW {
    Dense(DenseW),
    Moe(MoeW),
}

struct LayerW {
    input_ln: T,
    post_ln: T,
    sa_res_w: Vec<f32>,  // pre-combined norm·proj [512]
    mlp_res_w: Vec<f32>, // pre-combined norm·proj [512]
    attn: AttnW,
    ffn: FfnW,
}

// ── caches ──

// pub(crate): read/rewritten by mkmem.rs (.mkmem state snapshots)
#[derive(Clone)]
pub(crate) struct KdaCache {
    pub conv_q: Vec<f32>, // 3 × 512 (raw pre-conv)
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    pub s: Vec<f32>, // 4 × 128 × 128
}

#[derive(Clone)]
pub(crate) struct MlaCache {
    /// f32 layout (MICROKIMI_NO_KVQ8=1): k = pos × H×(nope+rope), v = pos × H×v
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    // q8_0 layout (default): the latent (nope) part of K and all of V are
    // q8_0 (i8 + one f32 scale per 32); the rope part of K stays f32
    // (position-sensitive) and is stored ONCE per position - the f32 layout
    // duplicates it per head. With MICROKIMI_KV_HADAMARD=1 the latent K and
    // V rows are kept in the 64-point Hadamard domain (see hadamard64).
    pub kq: Vec<i8>,  // pos × H × nope (Hadamard-rotated when `had`)
    pub ks: Vec<f32>, // pos × H × nope/32 scales
    pub kr: Vec<f32>, // pos × rope, never quantized
    pub vq: Vec<i8>,  // pos × H × v (Hadamard-rotated when `had`)
    pub vs: Vec<f32>, // pos × H × v/32 scales
    pub q8: bool,
    pub had: bool,
}

#[derive(Clone)]
pub(crate) enum Cache {
    Kda(KdaCache),
    Mla(MlaCache),
}

// ── profiler ──

#[derive(Default)]
pub struct Prof {
    pub t_norm_res: f64,
    pub t_kda_proj: f64,
    pub t_kda_conv: f64,
    pub t_kda_recur: f64,
    pub t_mla: f64,
    pub t_router: f64,
    pub t_experts: f64,
    pub t_lm_head: f64,
}

impl Prof {
    #[allow(dead_code)]
    pub fn print(&self) {
        self.print_cfg(&Config::microkimi());
    }

    pub fn print_cfg(&self, cfg: &Config) {
        let tot = self.t_norm_res + self.t_kda_proj + self.t_kda_conv + self.t_kda_recur + self.t_mla + self.t_router + self.t_experts + self.t_lm_head;
        if tot == 0.0 {
            return;
        }
        let lm_label = format!("lm_head ({} x {})", cfg.d, cfg.vocab);
        let router_label = format!("MoE router ({})", cfg.n_experts);
        let mla_label = format!(
            "MLA attention (H={}, {}/head, d={})",
            cfg.mla_heads, cfg.mla_v, cfg.d
        );
        let rows = [
            ("RMSNorm + AttnRes".to_string(), self.t_norm_res),
            ("KDA projections (qkv/f/g/o)".to_string(), self.t_kda_proj),
            ("causal KDA conv1d".to_string(), self.t_kda_conv),
            ("KDA recurrence (state S)".to_string(), self.t_kda_recur),
            (mla_label, self.t_mla),
            (router_label, self.t_router),
            ("MoE experts + shared + dense".to_string(), self.t_experts),
            (lm_label, self.t_lm_head),
        ];
        println!("Compute time breakdown (cumulative over processed tokens):");
        for (name, t) in rows {
            println!("  {:<32} {:6.1}%  ({:.2} s)", name, t / tot * 100.0, t);
        }
    }
}

// ── model ──

pub struct Model {
    pub cfg: Config,
    bin: BinFile,
    embed: T,
    lm_head: T,
    /// q8_0 runtime copy of lm_head (see the Q8Head section above): None when
    /// MICROKIMI_Q8HEAD=0, when --gpu serves the large matvecs, or when the
    /// dims do not divide into 32-wide blocks.
    lm_head_q8: Option<Q8Head>,
    norm_f: T,
    out_res_w: Vec<f32>,
    layers: Vec<LayerW>,
    pub(crate) caches: Vec<Cache>, // pub(crate): saved/restored by mkmem.rs
    pub last_logits: Vec<f32>,     // logits of the last forward (source for mkmem --save)
    pub prof: Prof,
    /// --stream: RAM LRU of packed expert bytes over the disk/HTTP tiers
    /// (stream.rs). None = historical full-load path, byte-identical behavior.
    stream: Option<crate::stream::ExpertCache>,
}

impl Model {
    fn t<'a>(data: &'a [u8], t: &T) -> &'a [f32] {
        as_f32(&data[t.off..t.off + t.len * 4])
    }

    /// Final logits projection: the q8_0 copy of lm_head when it was built at
    /// load (default), the exact f32 matvec otherwise (MICROKIMI_Q8HEAD=0).
    /// The q8 path is not bit-identical to f32 (q8 rounding, see q8.rs).
    fn logits_project(data: &[u8], lm_head: &T, q8: Option<&Q8Head>, cfg: &Config, x: &[f32], out: &mut [f32]) {
        match q8 {
            Some(h) => h.matvec(x, out),
            None => matvec(Self::t(data, lm_head), cfg.vocab, cfg.d, x, out),
        }
    }

    pub fn load(path: &str) -> Self {
        Self::from_bin(BinFile::open(path), None)
    }

    /// Streaming load (--stream): the spine is loaded compacted (expert MXFP4
    /// blobs excluded), experts are served on demand by the three-tier cache
    /// (RAM LRU of `ram_mb` MB -> the .bin on disk). Same weights, same
    /// dequant, same matvec: the output is bit-identical to `load`.
    /// `fallback` (--stream-fallback): the VQ1 expert shadows sidecar
    /// (<path>.shadows, shadow.rs) is loaded resident in RAM and the stream
    /// engine serves it on expert cache misses while refilling full
    /// precision in the background - a DEGRADED latency mode, NOT
    /// bit-identical, off by default.
    pub fn load_streaming(path: &str, ram_mb: usize, fallback: bool) -> Self {
        let bin = BinFile::open_spine(path);
        let shadows = if fallback {
            let cfg = &bin.config;
            let moe_layers: Vec<usize> = (0..cfg.n_layers).filter(|&l| cfg.is_moe(l)).collect();
            assert!(!moe_layers.is_empty(), "--stream-fallback on a MoE-less model is meaningless");
            crate::stream::set_fallback_shape(moe_layers.len() * cfg.top_k);
            Some(crate::shadow::Shadows::load(
                &crate::shadow::sidecar_path(path),
                &moe_layers,
                cfg.n_experts,
                cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM,
            ))
        } else {
            None
        };
        Self::from_bin(bin, Some(crate::stream::ExpertCache::local(path, ram_mb, shadows)))
    }

    fn from_bin(bin: BinFile, stream: Option<crate::stream::ExpertCache>) -> Self {
        let cfg = bin.config.clone();
        let get = |name: &str| -> T {
            T::from(bin.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name)))
        };
        let combine = |norm: &T, proj: &T| -> Vec<f32> {
            let n = Self::t(&bin.data, norm);
            let p = Self::t(&bin.data, proj);
            (0..cfg.d).map(|i| n[i] * p[i]).collect()
        };
        let embed = get("embed_tokens.weight");
        let lm_head = get("lm_head.weight");
        // q8_0 runtime copy of lm_head (default on): the .bin format is
        // unchanged, this is a load-time requantization of the f32 tensor.
        // Skipped under --gpu (Metal already serves the large matvecs).
        let lm_head_q8 = if q8head_enabled() && !gpu_on() && cfg.d % 32 == 0 && lm_head.len == cfg.vocab * cfg.d {
            Some(Q8Head::from_f32(Self::t(&bin.data, &lm_head), cfg.vocab, cfg.d))
        } else {
            None
        };
        let norm_f = get("norm.weight");
        let out_res_w = combine(&get("output_attn_res_norm.weight"), &get("output_attn_res_proj.weight"));
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = format!("layers.{}.", l);
            let input_ln = get(&format!("{}input_layernorm.weight", p));
            let post_ln = get(&format!("{}post_attention_layernorm.weight", p));
            let sa_res_w = combine(
                &get(&format!("{}self_attention_res_norm.weight", p)),
                &get(&format!("{}self_attention_res_proj.weight", p)),
            );
            let mlp_res_w = combine(
                &get(&format!("{}mlp_res_norm.weight", p)),
                &get(&format!("{}mlp_res_proj.weight", p)),
            );
            let attn = if cfg.is_mla(l) {
                caches.push(Cache::Mla(MlaCache::new()));
                AttnW::Mla(MlaW {
                    q_a: get(&format!("{}self_attn.q_a_proj.weight", p)),
                    q_a_ln: get(&format!("{}self_attn.q_a_layernorm.weight", p)),
                    q_b: get(&format!("{}self_attn.q_b_proj.weight", p)),
                    kv_a: get(&format!("{}self_attn.kv_a_proj_with_mqa.weight", p)),
                    kv_a_ln: get(&format!("{}self_attn.kv_a_layernorm.weight", p)),
                    kv_b: get(&format!("{}self_attn.kv_b_proj.weight", p)),
                    g_proj: get(&format!("{}self_attn.g_proj.weight", p)),
                    o_proj: get(&format!("{}self_attn.o_proj.weight", p)),
                })
            } else {
                caches.push(Cache::Kda(KdaCache {
                    conv_q: vec![0.0; 3 * cfg.kda_proj()],
                    conv_k: vec![0.0; 3 * cfg.kda_proj()],
                    conv_v: vec![0.0; 3 * cfg.kda_proj()],
                    s: vec![0.0; cfg.kda_heads * cfg.kda_dim * cfg.kda_dim],
                }));
                AttnW::Kda(KdaW {
                    q_proj: get(&format!("{}self_attn.q_proj.weight", p)),
                    k_proj: get(&format!("{}self_attn.k_proj.weight", p)),
                    v_proj: get(&format!("{}self_attn.v_proj.weight", p)),
                    q_conv: get(&format!("{}self_attn.q_conv1d.weight", p)),
                    k_conv: get(&format!("{}self_attn.k_conv1d.weight", p)),
                    v_conv: get(&format!("{}self_attn.v_conv1d.weight", p)),
                    f_a: get(&format!("{}self_attn.f_a_proj.weight", p)),
                    f_b: get(&format!("{}self_attn.f_b_proj.weight", p)),
                    a_log: get(&format!("{}self_attn.A_log", p)),
                    dt_bias: get(&format!("{}self_attn.dt_bias", p)),
                    b_proj: get(&format!("{}self_attn.b_proj.weight", p)),
                    g_proj: get(&format!("{}self_attn.g_proj.weight", p)),
                    o_norm: get(&format!("{}self_attn.o_norm.weight", p)),
                    o_proj: get(&format!("{}self_attn.o_proj.weight", p)),
                })
            };
            let ffn = if cfg.is_moe(l) {
                let pfx = format!("{}block_sparse_moe.experts.", p);
                let mut experts_vq = Vec::with_capacity(cfg.n_experts);
                let experts: Vec<[u64; 3]> = (0..cfg.n_experts)
                    .map(|e| {
                        ["w1", "w2", "w3"].map(|wn| {
                            let entry = bin
                                .entries
                                .get(&format!("{}{}.{}", pfx, e, wn))
                                .unwrap_or_else(|| panic!("missing expert: {}{}.{}", pfx, e, wn));
                            if wn == "w1" {
                                experts_vq.push(entry.dtype == crate::weights::DTYPE_VQ1);
                            }
                            entry.offset
                        })
                    })
                    .collect();
                // one global codebook shared by every VQ1 tensor (16 KB, L1-resident)
                let vq_cb = if experts_vq.iter().any(|&v| v) {
                    let cb = bin.f32_vec("vq_codebook");
                    assert_eq!(cb.len(), crate::quant::VQ_K * crate::quant::VQ_DIM, "vq_codebook: bad dims");
                    cb
                } else {
                    Vec::new()
                };
                FfnW::Moe(MoeW {
                    gate_w: get(&format!("{}block_sparse_moe.gate.weight", p)),
                    gate_b: get(&format!("{}block_sparse_moe.gate.e_score_correction_bias", p)),
                    routed_down: get(&format!("{}block_sparse_moe.routed_expert_down_proj.weight", p)),
                    routed_up: get(&format!("{}block_sparse_moe.routed_expert_up_proj.weight", p)),
                    routed_norm: get(&format!("{}block_sparse_moe.routed_expert_norm.weight", p)),
                    shared_gate: get(&format!("{}block_sparse_moe.shared_experts.gate_proj.weight", p)),
                    shared_up: get(&format!("{}block_sparse_moe.shared_experts.up_proj.weight", p)),
                    shared_down: get(&format!("{}block_sparse_moe.shared_experts.down_proj.weight", p)),
                    experts,
                    experts_vq,
                    vq_cb,
                })
            } else {
                FfnW::Dense(DenseW {
                    gate: get(&format!("{}mlp.gate_proj.weight", p)),
                    up: get(&format!("{}mlp.up_proj.weight", p)),
                    down: get(&format!("{}mlp.down_proj.weight", p)),
                })
            };
            layers.push(LayerW { input_ln, post_ln, sa_res_w, mlp_res_w, attn, ffn });
        }
        Model { cfg, bin, embed, lm_head, lm_head_q8, norm_f, out_res_w, layers, caches, last_logits: Vec::new(), prof: Prof::default(), stream }
    }

    /// Number of tokens already represented in the caches (from the first MLA
    /// layer; 0 for a KDA-only model). Only seeds the debug position counter:
    /// K3 has no positional encoding, so the math does not depend on it.
    pub fn cached_tokens(&self) -> usize {
        for c in &self.caches {
            if let Cache::Mla(m) = c {
                return m.positions(&self.cfg);
            }
        }
        0
    }

    pub fn reset_cache(&mut self) {
        for c in &mut self.caches {
            match c {
                Cache::Kda(k) => {
                    k.conv_q.iter_mut().for_each(|x| *x = 0.0);
                    k.conv_k.iter_mut().for_each(|x| *x = 0.0);
                    k.conv_v.iter_mut().for_each(|x| *x = 0.0);
                    k.s.iter_mut().for_each(|x| *x = 0.0);
                }
                Cache::Mla(m) => {
                    m.k.clear();
                    m.v.clear();
                    m.kq.clear();
                    m.ks.clear();
                    m.kr.clear();
                    m.vq.clear();
                    m.vs.clear();
                }
            }
        }
        // positions restart at 0: the routing history of the previous turn
        // (draft-aware prefetch) would alias the new positions
        crate::stream::route_hist_clear();
    }
}

// ── KDA ──

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
fn kda_forward(
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
fn kda_prefill(
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

/// True when MICROKIMI_NO_FLASH=1 (A/B toggle for the flash attention kernel).
fn no_flash() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MICROKIMI_NO_FLASH").map(|v| v == "1").unwrap_or(false))
}

/// KV tile size of the flash kernel: a tile's scores live in a 64-float
/// stack buffer and its K rows span 64 x 192 f32 = 48 KB (micro dims), L1/L2
/// friendly; 64 also keeps the running-rescale corrections rare enough that
/// the numerics stay within ~1e-6 of the materialized path.
const FLASH_KV: usize = 64;

/// Attention output for one (query, head) over cache positions 0..=pos:
/// materialized-score reference (the historical kernel, kept for the
/// MICROKIMI_NO_FLASH toggle and the selftest A/B). `oh` must be zeroed.
pub(crate) fn mla_attn_ref(cfg: &Config, k: &[f32], v: &[f32], qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let mut scores = vec![0f32; pos + 1];
    for j in 0..=pos {
        let kj = &k[(j * cfg.mla_heads + h) * cfg.mla_qh()..(j * cfg.mla_heads + h + 1) * cfg.mla_qh()];
        scores[j] = dot(qh, kj) * scale;
    }
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let mut z = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    for s in scores.iter_mut() {
        *s /= z;
    }
    for j in 0..=pos {
        let vj = &v[(j * cfg.mla_heads + h) * cfg.mla_v..(j * cfg.mla_heads + h + 1) * cfg.mla_v];
        let p = scores[j];
        for d in 0..cfg.mla_v {
            oh[d] += p * vj[d];
        }
    }
}

/// Flash attention for one (query, head) over cache positions 0..=pos:
/// online softmax in KV tiles of FLASH_KV positions, no score row
/// materialized (a 64-float stack buffer is the only working memory, vs
/// pos+1 floats allocated per (query, head) in the reference). `oh` must be
/// zeroed. Numerics: identical math to mla_attn_ref, different f32
/// association (the accumulator and the normalizer are rescaled whenever the
/// running max grows instead of normalizing once at the end); the deviation
/// is bounded by the selftest A/B (tol 1e-5, measured ~1e-6).
pub(crate) fn mla_attn_flash(cfg: &Config, k: &[f32], v: &[f32], qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let (hd, vd, nh) = (cfg.mla_qh(), cfg.mla_v, cfg.mla_heads);
    let mut m = f32::NEG_INFINITY; // running max
    let mut l = 0f32; // running normalizer (sum of exp(s - m) so far)
    let mut scores = [0f32; FLASH_KV];
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        // tile scores (causal mask = the loop bound; NoPE: nothing positional)
        let mut tm = f32::NEG_INFINITY;
        for (i, j) in (t..=end).enumerate() {
            let kj = &k[(j * nh + h) * hd..(j * nh + h + 1) * hd];
            let s = dot(qh, kj) * scale;
            scores[i] = s;
            tm = tm.max(s);
        }
        let m_new = m.max(tm);
        let corr = (m - m_new).exp(); // 1 when the max did not move, 0 on the first tile
        let mut tile_l = 0f32;
        for s in scores.iter_mut().take(end - t + 1) {
            *s = (*s - m_new).exp();
            tile_l += *s;
        }
        for d in 0..vd {
            oh[d] *= corr;
        }
        l = l * corr + tile_l;
        for (i, j) in (t..=end).enumerate() {
            let vj = &v[(j * nh + h) * vd..(j * nh + h + 1) * vd];
            let p = scores[i];
            for d in 0..vd {
                oh[d] += p * vj[d];
            }
        }
        m = m_new;
        t = end + 1;
    }
    for d in 0..vd {
        oh[d] /= l;
    }
}

/// True when MICROKIMI_NO_MQA=1 (A/B toggle: per-head flash loops instead of
/// the all-heads MQA-style kernels, f32 and q8 alike).
fn no_mqa() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MICROKIMI_NO_MQA").map(|v| v == "1").unwrap_or(false))
}

/// MQA-style flash attention for ALL heads at once over cache positions
/// 0..=pos. The per-head loop of mla_forward streams the whole KV cache
/// once PER HEAD (each pass strided over the full cache: ~H cold
/// re-traversals of L2/TLB per token). MLA decodes like MQA: the KV row of
/// one position (all heads' slices, contiguous) is consumed by every head
/// together, so the cache is streamed exactly ONCE per token, tile by tile.
///
/// Bit-identical to the per-head mla_attn_flash loop BY CONSTRUCTION: each
/// head keeps its own online-softmax state (m, l, accumulator) and sees the
/// exact same sequence of tiles and operations; only the interleaving
/// across heads changes, and the heads are independent. `q` is [H * qh],
/// `attn` (zeroed) is [H * vd].
pub(crate) fn mla_attn_flash_mqa(cfg: &Config, k: &[f32], v: &[f32], q: &[f32], pos: usize, scale: f32, attn: &mut [f32]) {
    let (hd, vd, nh) = (cfg.mla_qh(), cfg.mla_v, cfg.mla_heads);
    let mut m = vec![f32::NEG_INFINITY; nh]; // running max per head
    let mut l = vec![0f32; nh]; // running normalizer per head
    let mut scores = vec![0f32; nh * FLASH_KV]; // head-major tile scores
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        let tn = end - t + 1;
        // scores of every head over the tile: each KV row is read once
        for (i, j) in (t..=end).enumerate() {
            let kj = &k[j * nh * hd..(j + 1) * nh * hd];
            for h in 0..nh {
                scores[h * FLASH_KV + i] = dot(&q[h * hd..(h + 1) * hd], &kj[h * hd..(h + 1) * hd]) * scale;
            }
        }
        // per-head online-softmax update: the exact tile body of
        // mla_attn_flash, same order, same values
        for h in 0..nh {
            let sh = &mut scores[h * FLASH_KV..h * FLASH_KV + tn];
            let tm = sh.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
            let m_new = m[h].max(tm);
            let corr = (m[h] - m_new).exp();
            let mut tile_l = 0f32;
            for s in sh.iter_mut() {
                *s = (*s - m_new).exp();
                tile_l += *s;
            }
            let oh = &mut attn[h * vd..(h + 1) * vd];
            for d in 0..vd {
                oh[d] *= corr;
            }
            l[h] = l[h] * corr + tile_l;
            m[h] = m_new;
        }
        // V accumulation: each V row is read once for all heads
        for (i, j) in (t..=end).enumerate() {
            let vj = &v[j * nh * vd..(j + 1) * nh * vd];
            for h in 0..nh {
                let p = scores[h * FLASH_KV + i];
                let oh = &mut attn[h * vd..(h + 1) * vd];
                let vh = &vj[h * vd..(h + 1) * vd];
                for d in 0..vd {
                    oh[d] += p * vh[d];
                }
            }
        }
        t = end + 1;
    }
    for h in 0..nh {
        for d in 0..vd {
            attn[h * vd + d] /= l[h];
        }
    }
}

// ── MLA KV cache: f32 rows or q8_0 (latent quantized, rope f32) ──
//
// The MLA cache is the only engine state that grows with the context (KDA
// states are fixed-size). By default it is stored q8_0: the latent (nope)
// part of K and all of V are quantized per block of 32 (i8 + one f32
// scale), the rope part of K stays f32 (position-sensitive; a farm
// reference confirms rope must not be quantized) and is stored ONCE per
// position instead of duplicated per head. Bytes per position per layer:
//   f32: H*(nope+v)*4 + (H-1)*rope*4 (rope duplicated)   [micro: 5248]
//   q8:  H*(nope+v) + H*(nope+v)/32*4 + rope*4           [micro: 1936, ÷2.7]
// The Q.K dot of the latent part runs in INTEGER (q8 query x q8 cache row,
// q8::block_dot_i8); the rope dot, the softmax and the V accumulation stay
// f32 (V rows are dequantized tile by tile, never in full). Quantization
// happens at append (MlaCache::push). MICROKIMI_NO_KVQ8=1 restores the f32
// cache. MICROKIMI_KV_HADAMARD=1 rotates the latent K and V rows with an
// unnormalized 64-point Walsh-Hadamard transform before quantization
// (smearing outliers over the block makes q8 near-lossless; measured in
// selftest::run_kvq8) and inverts the rotation at read: H = H^T with
// H.H = 64 I, so one butterfly routine serves both directions.

/// True when the MLA KV cache quantizes to q8_0 (default).
/// MICROKIMI_NO_KVQ8=1 keeps the historical f32 cache.
fn kvq8_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_NO_KVQ8").map(|v| v != "1").unwrap_or(true))
}

/// True when MICROKIMI_KV_HADAMARD=1 (Hadamard rotation before KV
/// quantization; default off - the measured gain on nanokimi is marginal,
/// see selftest::run_kvq8).
fn kv_hadamard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_KV_HADAMARD").map(|v| v == "1").unwrap_or(false))
}

/// Unnormalized 64-point Walsh-Hadamard butterfly, in place (H = H^T and
/// H.H = 64 I: applying it twice with a 1/64 scale is the identity, so this
/// one routine rotates before quantization and de-rotates after).
pub(crate) fn hadamard64(x: &mut [f32]) {
    debug_assert_eq!(x.len(), 64);
    let mut h = 1usize;
    while h < 64 {
        for i in (0..64).step_by(2 * h) {
            for j in i..i + h {
                let a = x[j];
                let b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

impl MlaCache {
    fn new() -> MlaCache {
        MlaCache {
            k: Vec::new(),
            v: Vec::new(),
            kq: Vec::new(),
            ks: Vec::new(),
            kr: Vec::new(),
            vq: Vec::new(),
            vs: Vec::new(),
            q8: kvq8_on(),
            had: kv_hadamard(),
        }
    }

    /// Positions held by the cache.
    fn positions(&self, cfg: &Config) -> usize {
        if self.q8 {
            self.kr.len() / cfg.mla_rope
        } else {
            self.k.len() / (cfg.mla_heads * cfg.mla_qh())
        }
    }

    /// Appends one position's rows (k_row: H x (nope+rope) with the latent
    /// part first per head, v_row: H x v) - the f32 layout the attention
    /// builders produce. Quantizes on the fly in q8 mode.
    pub(crate) fn push(&mut self, cfg: &Config, k_row: &[f32], v_row: &[f32]) {
        if !self.q8 {
            self.k.extend_from_slice(k_row);
            self.v.extend_from_slice(v_row);
            return;
        }
        let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
        let had = self.had && nope % 64 == 0 && vd % 64 == 0;
        let qh = nope + rope;
        // rope is shared across heads: stored once, from head 0
        self.kr.extend_from_slice(&k_row[nope..qh]);
        let mut scratch = vec![0f32; nope.max(vd)];
        let mut qv = crate::q8::Q8Vec::new();
        for h in 0..nh {
            let row = &k_row[h * qh..h * qh + nope];
            scratch[..nope].copy_from_slice(row);
            if had {
                for b in scratch[..nope].chunks_mut(64) {
                    hadamard64(b);
                }
            }
            crate::q8::quantize_q8_into(&scratch[..nope], &mut qv);
            self.kq.extend_from_slice(&qv.q);
            self.ks.extend_from_slice(&qv.scales);
            let rowv = &v_row[h * vd..(h + 1) * vd];
            scratch[..vd].copy_from_slice(rowv);
            if had {
                for b in scratch[..vd].chunks_mut(64) {
                    hadamard64(b);
                }
            }
            crate::q8::quantize_q8_into(&scratch[..vd], &mut qv);
            self.vq.extend_from_slice(&qv.q);
            self.vs.extend_from_slice(&qv.scales);
        }
    }

    /// Dequantizes back to the f32 row layout (mkmem snapshot; the .mkmem
    /// format stays f32 whatever the runtime cache mode is). In f32 mode
    /// this is a plain clone.
    pub(crate) fn to_f32(&self, cfg: &Config) -> (Vec<f32>, Vec<f32>) {
        if !self.q8 {
            return (self.k.clone(), self.v.clone());
        }
        let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
        let had = self.had && nope % 64 == 0 && vd % 64 == 0;
        let qh = nope + rope;
        let n = self.positions(cfg);
        let nb_n = nope / 32;
        let nb_v = vd / 32;
        let mut k = vec![0f32; n * nh * qh];
        let mut v = vec![0f32; n * nh * vd];
        for j in 0..n {
            for h in 0..nh {
                // latent K: dequant (+ de-rotate), then rope from the shared row
                let base_q = (j * nh + h) * nope;
                let base_s = (j * nh + h) * nb_n;
                let out = &mut k[(j * nh + h) * qh..(j * nh + h) * qh + nope];
                for g in 0..nb_n {
                    let s = self.ks[base_s + g];
                    for i in 0..32 {
                        out[g * 32 + i] = self.kq[base_q + g * 32 + i] as f32 * s;
                    }
                }
                if had {
                    for b in out.chunks_mut(64) {
                        hadamard64(b);
                        for x in b.iter_mut() {
                            *x /= 64.0;
                        }
                    }
                }
                k[(j * nh + h) * qh + nope..(j * nh + h) * qh + qh].copy_from_slice(&self.kr[j * rope..(j + 1) * rope]);
                let base_q = (j * nh + h) * vd;
                let base_s = (j * nh + h) * nb_v;
                let out = &mut v[(j * nh + h) * vd..(j * nh + h) * vd + vd];
                for g in 0..nb_v {
                    let s = self.vs[base_s + g];
                    for i in 0..32 {
                        out[g * 32 + i] = self.vq[base_q + g * 32 + i] as f32 * s;
                    }
                }
                if had {
                    for b in out.chunks_mut(64) {
                        hadamard64(b);
                        for x in b.iter_mut() {
                            *x /= 64.0;
                        }
                    }
                }
            }
        }
        (k, v)
    }

    /// Replaces the cache contents by f32 rows (mkmem restore; requantizes
    /// when the runtime cache is in q8 mode).
    pub(crate) fn assign_f32(&mut self, cfg: &Config, k: Vec<f32>, v: Vec<f32>) {
        *self = MlaCache::new();
        if !self.q8 {
            self.k = k;
            self.v = v;
            return;
        }
        let (nh, qh, vd) = (cfg.mla_heads, cfg.mla_qh(), cfg.mla_v);
        let n = k.len() / (nh * qh);
        for j in 0..n {
            self.push(cfg, &k[j * nh * qh..(j + 1) * nh * qh], &v[j * nh * vd..(j + 1) * nh * vd]);
        }
    }
}

/// Flash attention over the q8_0 KV cache for one (query, head), positions
/// 0..=pos. Same online-softmax tile structure as mla_attn_flash; only the
/// dot inputs differ: the latent part of the score runs in INTEGER (q8
/// query x q8 K row via q8::block_dot_i8, per-block scale product), the rope
/// part is an f32 dot on the shared rope rows. V rows are dequantized per
/// tile (never in full). With the Hadamard option the query is rotated once
/// (dot(Hq, Hk) = 64 * dot(q, k), folded back as 1/64) and the V accumulator
/// is de-rotated once at the end (linearity: sum of p.V in the Hadamard
/// domain, then H./64). `oh` must be zeroed. NOT bit-identical to the f32
/// path: the q8 rounding of K, V and the query is the deal (measured in
/// selftest::run_kvq8, max rel << 1e-3).
pub(crate) fn mla_attn_flash_q8(cfg: &Config, c: &MlaCache, qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
    let had = c.had && nope % 64 == 0 && vd % 64 == 0;
    let nb = nope / 32;
    // query prep: latent part (Hadamard-rotated when enabled) quantized to q8
    let mut qnope = qh[..nope].to_vec();
    if had {
        for b in qnope.chunks_mut(64) {
            hadamard64(b);
        }
    }
    let qq = crate::q8::quantize_q8(&qnope);
    let q_rope = &qh[nope..];
    let had_k = if had { 1.0 / 64.0 } else { 1.0 };
    let mut m = f32::NEG_INFINITY;
    let mut l = 0f32;
    let mut scores = [0f32; FLASH_KV];
    let mut vtile = vec![0f32; FLASH_KV * vd];
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        let tn = end - t + 1;
        let mut tm = f32::NEG_INFINITY;
        for (i, j) in (t..=end).enumerate() {
            // rope dot (f32) + latent dot (integer, per-block scale product)
            let mut s = dot(q_rope, &c.kr[j * rope..(j + 1) * rope]);
            let mut acc = 0f32;
            let bq = (j * nh + h) * nope;
            let bs = (j * nh + h) * nb;
            for g in 0..nb {
                let d = crate::q8::block_dot_i8(&c.kq[bq + g * 32..bq + g * 32 + 32], &qq.q[g * 32..g * 32 + 32]);
                acc += qq.scales[g] * c.ks[bs + g] * d as f32;
            }
            s += had_k * acc;
            let s = s * scale;
            scores[i] = s;
            tm = tm.max(s);
        }
        let m_new = m.max(tm);
        let corr = (m - m_new).exp();
        let mut tile_l = 0f32;
        for s in scores.iter_mut().take(tn) {
            *s = (*s - m_new).exp();
            tile_l += *s;
        }
        for d in 0..vd {
            oh[d] *= corr;
        }
        l = l * corr + tile_l;
        // dequant the tile's V rows (Hadamard domain when enabled: the
        // accumulator is de-rotated once at the end)
        for (i, j) in (t..=end).enumerate() {
            let bq = (j * nh + h) * vd;
            let bs = (j * nh + h) * (vd / 32);
            let out = &mut vtile[i * vd..(i + 1) * vd];
            for g in 0..vd / 32 {
                let sc = c.vs[bs + g];
                for d2 in 0..32 {
                    out[g * 32 + d2] = c.vq[bq + g * 32 + d2] as f32 * sc;
                }
            }
            let p = scores[i];
            for d in 0..vd {
                oh[d] += p * out[d];
            }
        }
        m = m_new;
        t = end + 1;
    }
    if had {
        for b in oh[..vd].chunks_mut(64) {
            hadamard64(b);
            for x in b.iter_mut() {
                *x /= 64.0;
            }
        }
    }
    for d in 0..vd {
        oh[d] /= l;
    }
}

/// MQA-style flash attention over the q8_0 KV cache for ALL heads at once:
/// the tile-outer / head-inner restructure of mla_attn_flash_q8, exactly
/// like mla_attn_flash_mqa is for mla_attn_flash. The per-head q8 loop
/// re-streams the whole quantized cache once PER HEAD; here one position's
/// row (all heads' latent slices contiguous in kq/ks, the shared rope row,
/// then all heads' V slices in vq/vs) is consumed while it is hot, so the
/// cache is streamed exactly ONCE per token. The integer latent dot is
/// unchanged (q8 query x q8 cache row via q8::block_dot_i8, per-block scale
/// product): for a fixed position the head-inner loop keeps every head's q8
/// query (H * nope bytes, L1-resident) against the streamed row.
///
/// Bit-identical to the per-head mla_attn_flash_q8 loop BY CONSTRUCTION:
/// each head keeps its own online-softmax state and sees the exact same
/// tile sequence, the same integer dots (exact int32) and the same f32
/// operations in the same order; only the interleaving across independent
/// heads changes. `q` is [H * (nope+rope)], `attn` (zeroed) is [H * vd].
pub(crate) fn mla_attn_flash_q8_mqa(cfg: &Config, c: &MlaCache, q: &[f32], pos: usize, scale: f32, attn: &mut [f32]) {
    let (nh, nope, rope, vd) = (cfg.mla_heads, cfg.mla_nope, cfg.mla_rope, cfg.mla_v);
    let had = c.had && nope % 64 == 0 && vd % 64 == 0;
    let hd = nope + rope;
    let nb = nope / 32;
    let nbv = vd / 32;
    // query prep, per head: latent part (Hadamard-rotated when enabled)
    // quantized to q8 - the exact prep of mla_attn_flash_q8
    let mut qqs = Vec::with_capacity(nh);
    for h in 0..nh {
        let mut qnope = q[h * hd..h * hd + nope].to_vec();
        if had {
            for b in qnope.chunks_mut(64) {
                hadamard64(b);
            }
        }
        qqs.push(crate::q8::quantize_q8(&qnope));
    }
    let had_k = if had { 1.0 / 64.0 } else { 1.0 };
    let mut m = vec![f32::NEG_INFINITY; nh]; // running max per head
    let mut l = vec![0f32; nh]; // running normalizer per head
    let mut scores = vec![0f32; nh * FLASH_KV]; // head-major tile scores
    let mut vtile = vec![0f32; vd]; // one head's dequantized V row
    let mut t = 0usize;
    while t <= pos {
        let end = (t + FLASH_KV - 1).min(pos);
        let tn = end - t + 1;
        // scores of every head over the tile: the position's K row (rope
        // shared + all latent slices, contiguous) is read once
        for (i, j) in (t..=end).enumerate() {
            let kr = &c.kr[j * rope..(j + 1) * rope];
            let kq = &c.kq[j * nh * nope..(j + 1) * nh * nope];
            let ks = &c.ks[j * nh * nb..(j + 1) * nh * nb];
            for h in 0..nh {
                // rope dot (f32) + latent dot (integer, per-block scale
                // product): the same operations, in the same order, as
                // mla_attn_flash_q8
                let mut s = dot(&q[h * hd + nope..(h + 1) * hd], kr);
                let qq = &qqs[h];
                let mut acc = 0f32;
                for g in 0..nb {
                    let d = crate::q8::block_dot_i8(&kq[h * nope + g * 32..h * nope + g * 32 + 32], &qq.q[g * 32..g * 32 + 32]);
                    acc += qq.scales[g] * ks[h * nb + g] * d as f32;
                }
                s += had_k * acc;
                scores[h * FLASH_KV + i] = s * scale;
            }
        }
        // per-head online-softmax update: the exact tile body of
        // mla_attn_flash_q8, same order, same values
        for h in 0..nh {
            let sh = &mut scores[h * FLASH_KV..h * FLASH_KV + tn];
            let tm = sh.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
            let m_new = m[h].max(tm);
            let corr = (m[h] - m_new).exp();
            let mut tile_l = 0f32;
            for s in sh.iter_mut() {
                *s = (*s - m_new).exp();
                tile_l += *s;
            }
            let oh = &mut attn[h * vd..(h + 1) * vd];
            for d in 0..vd {
                oh[d] *= corr;
            }
            l[h] = l[h] * corr + tile_l;
            m[h] = m_new;
        }
        // V accumulation: the position's V row (all heads' slices,
        // contiguous) is read once, dequantized head by head (Hadamard
        // domain when enabled: the accumulators are de-rotated at the end)
        for (i, j) in (t..=end).enumerate() {
            let vq = &c.vq[j * nh * vd..(j + 1) * nh * vd];
            let vs = &c.vs[j * nh * nbv..(j + 1) * nh * nbv];
            for h in 0..nh {
                for g in 0..nbv {
                    let sc = vs[h * nbv + g];
                    for d2 in 0..32 {
                        vtile[g * 32 + d2] = vq[h * vd + g * 32 + d2] as f32 * sc;
                    }
                }
                let p = scores[h * FLASH_KV + i];
                let oh = &mut attn[h * vd..(h + 1) * vd];
                for d in 0..vd {
                    oh[d] += p * vtile[d];
                }
            }
        }
        t = end + 1;
    }
    if had {
        for h in 0..nh {
            for b in attn[h * vd..(h + 1) * vd].chunks_mut(64) {
                hadamard64(b);
                for x in b.iter_mut() {
                    *x /= 64.0;
                }
            }
        }
    }
    for h in 0..nh {
        for d in 0..vd {
            attn[h * vd + d] /= l[h];
        }
    }
}

/// Materialized-score reference over the q8_0 cache (MICROKIMI_NO_FLASH
/// debug path): dequantizes the cache (to_f32) and runs the historical
/// three-pass structure. Same q8 rounding as mla_attn_flash_q8, same f32
/// reassociation as mla_attn_ref.
pub(crate) fn mla_attn_ref_q8(cfg: &Config, c: &MlaCache, qh: &[f32], h: usize, pos: usize, scale: f32, oh: &mut [f32]) {
    let (nh, hd, vd) = (cfg.mla_heads, cfg.mla_qh(), cfg.mla_v);
    let (k, v) = c.to_f32(cfg);
    let mut scores = vec![0f32; pos + 1];
    for j in 0..=pos {
        scores[j] = dot(qh, &k[(j * nh + h) * hd..(j * nh + h + 1) * hd]) * scale;
    }
    let mx = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let mut z = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - mx).exp();
        z += *s;
    }
    for s in scores.iter_mut() {
        *s /= z;
    }
    for j in 0..=pos {
        let vj = &v[(j * nh + h) * vd..(j * nh + h + 1) * vd];
        let p = scores[j];
        for d in 0..vd {
            oh[d] += p * vj[d];
        }
    }
}

fn mla_forward(
    cfg: &Config,
    data: &[u8],
    w: &MlaW,
    cache: &mut MlaCache,
    x: &[f32],
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    // q = q_b(rmsnorm(q_a(x))) [H*(nope+rope)]
    let mut qa = vec![0f32; cfg.mla_qa];
    matvec(Model::t(data, &w.q_a), cfg.mla_qa, cfg.d, x, &mut qa);
    let mut qa_n = vec![0f32; cfg.mla_qa];
    rmsnorm(cfg, &qa, Model::t(data, &w.q_a_ln), &mut qa_n);
    let mut q = vec![0f32; cfg.mla_qb()];
    matvec(Model::t(data, &w.q_b), cfg.mla_qb(), cfg.mla_qa, &qa_n, &mut q);
    // c = kv_a(x) [kva+rope] ; k_pass [kva] ; k_rot [rope] (shared across heads)
    let mut c = vec![0f32; cfg.mla_c_dim()];
    matvec(Model::t(data, &w.kv_a), cfg.mla_c_dim(), cfg.d, x, &mut c);
    let k_rot: Vec<f32> = c[cfg.mla_kva..cfg.mla_kva + cfg.mla_rope].to_vec();
    let mut kp_n = vec![0f32; cfg.mla_kva];
    rmsnorm(cfg, &c[..cfg.mla_kva], Model::t(data, &w.kv_a_ln), &mut kp_n);
    let mut kb = vec![0f32; cfg.mla_kvb()];
    matvec(Model::t(data, &w.kv_b), cfg.mla_kvb(), cfg.mla_kva, &kp_n, &mut kb);
    // K[h] = kb[h][..nope] ++ k_rot ; V[h] = kb[h][nope..nope+v]
    let mut k_new = vec![0f32; cfg.mla_heads * cfg.mla_qh()];
    let mut v_new = vec![0f32; cfg.mla_hv()];
    for h in 0..cfg.mla_heads {
        k_new[h * cfg.mla_qh()..h * cfg.mla_qh() + cfg.mla_nope]
            .copy_from_slice(&kb[h * (cfg.mla_nope + cfg.mla_v)..h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope]);
        k_new[h * cfg.mla_qh() + cfg.mla_nope..(h + 1) * cfg.mla_qh()].copy_from_slice(&k_rot);
        v_new[h * cfg.mla_v..(h + 1) * cfg.mla_v].copy_from_slice(
            &kb[h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope..(h + 1) * (cfg.mla_nope + cfg.mla_v)],
        );
    }
    cache.push(cfg, &k_new, &v_new);
    let pos = cache.positions(cfg) - 1;
    // causal attention, scale (nope+rope)^-0.5
    let scale = (cfg.mla_qh() as f32).powf(-0.5);
    let mut attn = vec![0f32; cfg.mla_heads * cfg.mla_v];
    let flash = !no_flash();
    if cache.q8 {
        if flash && !no_mqa() {
            // q8_0 cache, MQA-style: all heads together, the quantized
            // cache is streamed once (integer latent dot, bit-identical
            // to the per-head q8 loop)
            mla_attn_flash_q8_mqa(cfg, cache, &q, pos, scale, &mut attn);
        } else {
            // q8_0 cache: per-head kernels with the integer latent dot
            for h in 0..cfg.mla_heads {
                let qh = &q[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
                let oh = &mut attn[h * cfg.mla_v..(h + 1) * cfg.mla_v];
                if flash {
                    mla_attn_flash_q8(cfg, cache, qh, h, pos, scale, oh);
                } else {
                    mla_attn_ref_q8(cfg, cache, qh, h, pos, scale, oh);
                }
            }
        }
    } else if flash && !no_mqa() {
        // MQA-style: all heads together, the KV cache is streamed once
        mla_attn_flash_mqa(cfg, &cache.k, &cache.v, &q, pos, scale, &mut attn);
    } else {
        for h in 0..cfg.mla_heads {
            let qh = &q[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
            let oh = &mut attn[h * cfg.mla_v..(h + 1) * cfg.mla_v];
            if flash {
                mla_attn_flash(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            } else {
                mla_attn_ref(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            }
        }
    }
    // output gate + o_proj (g_proj is [H*v, d]: H*v == d only in the micro
    // config; real K3 is [12288, 7168])
    let hv = cfg.mla_hv();
    let mut g = vec![0f32; hv];
    matvec(Model::t(data, &w.g_proj), hv, cfg.d, x, &mut g);
    for i in 0..hv {
        attn[i] *= sigmoid(g[i]);
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.o_proj), cfg.d, hv, &attn, &mut out);
    prof.t_mla += tm.elapsed().as_secs_f64();
    out
}

/// Batched MLA for prefill: `x` = n position rows [n * d], returns [n * d].
/// Projections run as gemm_batch; the n new latent K/V rows are appended to
/// the cache in position order (identical layout to n single-token calls),
/// then attention runs per query position over the cache entries 0..=pos
/// (parallel causal attention, NoPE: no positional encoding anywhere).
/// Bit-identical to mla_forward per position.
#[allow(clippy::too_many_arguments)]
fn mla_prefill(
    cfg: &Config,
    data: &[u8],
    w: &MlaW,
    cache: &mut MlaCache,
    x: &[f32],
    n: usize,
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    let (qa_dim, qb, kvb, c_dim) = (cfg.mla_qa, cfg.mla_qb(), cfg.mla_kvb(), cfg.mla_c_dim());
    // q = q_b(rmsnorm(q_a(x))) for all positions
    let mut qa = vec![0f32; n * qa_dim];
    gemm_batch(Model::t(data, &w.q_a), qa_dim, cfg.d, x, n, &mut qa);
    let mut qa_n = vec![0f32; n * qa_dim];
    for t in 0..n {
        rmsnorm(cfg, &qa[t * qa_dim..(t + 1) * qa_dim], Model::t(data, &w.q_a_ln), &mut qa_n[t * qa_dim..(t + 1) * qa_dim]);
    }
    let mut q = vec![0f32; n * qb];
    gemm_batch(Model::t(data, &w.q_b), qb, qa_dim, &qa_n, n, &mut q);
    // c = kv_a(x) [kva+rope] ; k_rot = c[kva..kva+rope] ; kp_n = rmsnorm(c[..kva])
    let mut c = vec![0f32; n * c_dim];
    gemm_batch(Model::t(data, &w.kv_a), c_dim, cfg.d, x, n, &mut c);
    let mut kp_n = vec![0f32; n * cfg.mla_kva];
    for t in 0..n {
        rmsnorm(cfg, &c[t * c_dim..t * c_dim + cfg.mla_kva], Model::t(data, &w.kv_a_ln), &mut kp_n[t * cfg.mla_kva..(t + 1) * cfg.mla_kva]);
    }
    let mut kb = vec![0f32; n * kvb];
    gemm_batch(Model::t(data, &w.kv_b), kvb, cfg.mla_kva, &kp_n, n, &mut kb);
    // build K[h] = kb[h][..nope] ++ k_rot ; V[h] = kb[h][nope..nope+v] per
    // position and append in order: same cache state as the sequential path
    let p0 = cache.positions(cfg);
    for t in 0..n {
        let k_rot = &c[t * c_dim + cfg.mla_kva..t * c_dim + cfg.mla_kva + cfg.mla_rope];
        let kbt = &kb[t * kvb..(t + 1) * kvb];
        let mut k_new = vec![0f32; cfg.mla_heads * cfg.mla_qh()];
        let mut v_new = vec![0f32; cfg.mla_heads * cfg.mla_v];
        for h in 0..cfg.mla_heads {
            k_new[h * cfg.mla_qh()..h * cfg.mla_qh() + cfg.mla_nope]
                .copy_from_slice(&kbt[h * (cfg.mla_nope + cfg.mla_v)..h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope]);
            k_new[h * cfg.mla_qh() + cfg.mla_nope..(h + 1) * cfg.mla_qh()].copy_from_slice(k_rot);
            v_new[h * cfg.mla_v..(h + 1) * cfg.mla_v].copy_from_slice(
                &kbt[h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope..(h + 1) * (cfg.mla_nope + cfg.mla_v)],
            );
        }
        cache.push(cfg, &k_new, &v_new);
    }
    // causal attention per query position, scale qh^-0.5
    let scale = (cfg.mla_qh() as f32).powf(-0.5);
    let mut attn = vec![0f32; n * cfg.mla_heads * cfg.mla_v];
    let flash = !no_flash();
    for t in 0..n {
        let pos = p0 + t;
        let qt = &q[t * qb..(t + 1) * qb];
        for h in 0..cfg.mla_heads {
            let qh = &qt[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
            let oh = &mut attn[(t * cfg.mla_heads + h) * cfg.mla_v..(t * cfg.mla_heads + h + 1) * cfg.mla_v];
            if cache.q8 {
                if flash {
                    mla_attn_flash_q8(cfg, cache, qh, h, pos, scale, oh);
                } else {
                    mla_attn_ref_q8(cfg, cache, qh, h, pos, scale, oh);
                }
            } else if flash {
                mla_attn_flash(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            } else {
                mla_attn_ref(cfg, &cache.k, &cache.v, qh, h, pos, scale, oh);
            }
        }
    }
    // output gate + o_proj (g_proj is [H*v, d]: H*v == d only in the micro
    // config; real K3 is [12288, 7168])
    let hv = cfg.mla_hv();
    let mut g = vec![0f32; n * hv];
    gemm_batch(Model::t(data, &w.g_proj), hv, cfg.d, x, n, &mut g);
    for i in 0..n * hv {
        attn[i] *= sigmoid(g[i]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.o_proj), cfg.d, hv, &attn, n, &mut out);
    prof.t_mla += tm.elapsed().as_secs_f64();
    out
}

// ── MoE ──

/// Router-lookahead prefetch (--stream-predict N; MICROKIMI_LOOKAHEAD=0
/// reverts to the Markov predictor): runs the NEXT MoE layer's router (one
/// small GEMV, n_experts x d) on the CURRENT MoE input x - the closest state
/// available to what that router will actually see (its true input is the
/// post-attention layernorm of the next layer, not yet computed) - and
/// background-prefetches its top-N predicted experts through the stream
/// cache while the current layer's experts compute. The selection replicates
/// the noaux_tc rule (sigmoid + correction bias, ranked by key) without
/// touching moe_forward's own selection; only WHEN bytes are fetched
/// changes, never WHICH experts run: the output is bit-identical.
fn moe_lookahead(cfg: &Config, data: &[u8], w: &MoeW, layer2: usize, x: &[f32], n: usize, cache: &crate::stream::ExpertCache) {
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; cfg.n_experts];
    matvec(gate_w, cfg.n_experts, cfg.d, x, &mut logits);
    let mut ids: Vec<(u32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as u32, sigmoid(l) + gate_b[i])).collect();
    ids.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
    let jobs: Vec<(u32, u32, [u64; 3], usize)> = ids
        .iter()
        .take(n)
        .map(|&(e, _)| {
            let eblob = if w.experts_vq[e as usize] { expert_vq_blob } else { expert_blob };
            (layer2 as u32, e, w.experts[e as usize], eblob)
        })
        .collect();
    cache.prefetch(jobs);
}

fn moe_forward(cfg: &Config, data: &[u8], w: &MoeW, x: &[f32], prof: &mut Prof, layer: usize, pos: usize, stream: Option<&crate::stream::ExpertCache>) -> Vec<f32> {
    // noaux_tc router: sigmoid, +bias for selection, weights without bias
    let tm = Instant::now();
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; cfg.n_experts];
    matvec(gate_w, cfg.n_experts, cfg.d, x, &mut logits);
    let mut sel: Vec<(u32, f32, f32)> = Vec::with_capacity(cfg.top_k); // (expert, score, key)
    for (i, &l) in logits.iter().enumerate() {
        let sc = sigmoid(l);
        let key = sc + gate_b[i];
        let item = (i as u32, sc, key);
        if sel.len() < cfg.top_k {
            sel.push(item);
            if sel.len() == cfg.top_k {
                sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        } else if key > sel[cfg.top_k - 1].2 {
            sel[cfg.top_k - 1] = item;
            sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        }
    }
    let sumw: f32 = sel.iter().map(|s| s.1).sum::<f32>() + 1e-20;
    let weights: Vec<f32> = sel.iter().map(|s| s.1 / sumw).collect();
    if ROUTER_LAYERS.contains(&layer) {
        let mut ids: Vec<u32> = sel.iter().map(|s| s.0).collect();
        ids.sort();
        parity_rec(|d| {
            d.router.insert((pos, layer), ids);
        });
    }
    // --debug-routing: top-3 by renormalized weight + count of top-16 appearances
    ROUTING.with(|r| {
        if let Some(d) = r.borrow_mut().as_mut() {
            let mut top3: Vec<(u32, f32)> = sel.iter().map(|s| (s.0, s.1 / sumw)).collect();
            top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top3.truncate(3);
            d.cur.push((layer, top3));
            for s in &sel {
                *d.counts.entry((layer, s.0)).or_insert(0) += 1;
            }
        }
    });
    // count-min routing statistics (no-op unless routestats/MICROKIMI_ROUTECMS)
    for s in &sel {
        crate::cms::record(layer, s.0);
    }
    // routing history for the draft-aware prefetch (streaming runs only;
    // no-op unless MICROKIMI_DRAFTPREFETCH is on)
    if stream.is_some() {
        crate::stream::route_record(pos as u32, layer as u32, sel.iter().map(|s| s.0).collect());
    }
    let mut h = vec![0f32; cfg.routed_hidden];
    matvec(Model::t(data, &w.routed_down), cfg.routed_hidden, cfg.d, x, &mut h);
    prof.t_router += tm.elapsed().as_secs_f64();
    // imatrix calibration hook (no-op unless `calibrate` is running)
    crate::imatrix::record_hidden(layer, &h);

    // MXFP4 experts (dequantized on the fly): SiTU(cat(w1 h, w3 h)) then w2.
    // The 16 experts are independent → one pool job per expert (offsets
    // precomputed at load time, zero lookup). Combination in fixed order after
    // the barrier → deterministic.
    let tm = Instant::now();
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
    let (erh, emi) = (cfg.routed_hidden, cfg.moe_inter); // copies for the 'static closures
    let cbp = crate::pool::SPtr(w.vq_cb.as_ptr());
    let cblen = w.vq_cb.len();
    let mut outs = vec![0f32; cfg.top_k * cfg.routed_hidden];
    match stream {
        // historical full-load path: expert blobs read straight from the file image
        None => {
            let dp = crate::pool::SPtrU8(data.as_ptr());
            let dlen = data.len();
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(cfg.top_k);
            for (ei, _) in weights.iter().enumerate() {
                let offs = w.experts[sel[ei].0 as usize];
                let vq = w.experts_vq[sel[ei].0 as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (dp, hp, op, cbp) = (dp, hp, op, cbp);
                    unsafe {
                        let data = std::slice::from_raw_parts(dp.0, dlen);
                        let h = std::slice::from_raw_parts(hp.0, erh);
                        let blob = |i: usize| &data[offs[i] as usize..offs[i] as usize + eblob];
                        let mut a = vec![0f32; emi];
                        let mut u = vec![0f32; emi];
                        if vq {
                            // cold expert: gather from the L1-resident codebook
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq(cb, blob(0), emi, erh, h, &mut a);
                            crate::quant::matvec_vq(cb, blob(2), emi, erh, h, &mut u);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, h, &mut a, 1);
                            crate::mxfp4::matvec_packed(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, h, &mut u, 1);
                        }
                        let mut act = vec![0f32; emi];
                        for j in 0..emi {
                            act[j] = situ(a[j], u[j]);
                        }
                        crate::imatrix::record_inter(layer, &act);
                        let o = std::slice::from_raw_parts_mut(op.0.add(ei * erh), erh);
                        if vq {
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq(cb, blob(1), erh, emi, &act, o);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act, o, 1);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
        // --stream: router-first prefetch. The router above already selected
        // the 16 experts; each pool job pulls its packed bytes through the
        // three-tier cache (RAM LRU -> disk -> HTTP) and computes as soon as
        // its bytes land. Same bytes, same matvec sequence: bit-identical.
        Some(cache) => {
            let cp = cache as *const crate::stream::ExpertCache as usize;
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let layer32 = layer as u32;
            // Submit the reads sorted by file offset: the top-k experts are
            // not stored in id order, so offset order turns scattered seeks
            // into a near-sequential sweep. Each job writes its own output
            // slot, so the submission order cannot change the result.
            let mut order: Vec<usize> = (0..weights.len()).collect();
            if crate::stream::offset_sort() {
                order.sort_by_key(|&ei| w.experts[sel[ei].0 as usize][0]);
            }
            // fused run fetch: the missing experts of this layer are pulled
            // through the cache in as few physical reads as possible (one
            // span read per run of file-adjacent experts); the compute jobs
            // below then hit the RAM LRU. No-op when fusion is off or the
            // source is remote.
            let items: Vec<(u32, [u64; 3], usize)> = order
                .iter()
                .map(|&ei| {
                    let e = sel[ei].0;
                    (e, w.experts[e as usize], if w.experts_vq[e as usize] { expert_vq_blob } else { expert_blob })
                })
                .collect();
            cache.warm_batch(layer32, &items);
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(cfg.top_k);
            for &ei in &order {
                let e = sel[ei].0;
                let offs = w.experts[e as usize];
                let vq = w.experts_vq[e as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (hp, op, cbp) = (hp, op, cbp);
                    unsafe {
                        let cache = &*(cp as *const crate::stream::ExpertCache);
                        let served = cache.get(layer32, e, offs, eblob);
                        let h = std::slice::from_raw_parts(hp.0, erh);
                        // shadow fallback (--stream-fallback): a cache miss
                        // comes back as the resident VQ1 shadow (shadow
                        // codebook, 3 x expert_vq_blob bytes) - degraded,
                        // refilled in the background. Served::Full is the
                        // historical bit-identical path.
                        let (bytes, vq, eb, cb): (&[u8], bool, usize, &[f32]) = match &served {
                            crate::stream::Served::Full(b) => (&b[..], vq, eblob, std::slice::from_raw_parts(cbp.0, cblen)),
                            crate::stream::Served::Shadow(s, off) => {
                                (&s.data[*off..*off + 3 * expert_vq_blob], true, expert_vq_blob, &s.cb[..])
                            }
                        };
                        let blob = |i: usize| &bytes[i * eb..(i + 1) * eb];
                        let mut a = vec![0f32; emi];
                        let mut u = vec![0f32; emi];
                        if vq {
                            crate::quant::matvec_vq(cb, blob(0), emi, erh, h, &mut a);
                            crate::quant::matvec_vq(cb, blob(2), emi, erh, h, &mut u);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, h, &mut a, 1);
                            crate::mxfp4::matvec_packed(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, h, &mut u, 1);
                        }
                        let mut act = vec![0f32; emi];
                        for j in 0..emi {
                            act[j] = situ(a[j], u[j]);
                        }
                        crate::imatrix::record_inter(layer, &act);
                        let o = std::slice::from_raw_parts_mut(op.0.add(ei * erh), erh);
                        if vq {
                            crate::quant::matvec_vq(cb, blob(1), erh, emi, &act, o);
                        } else {
                            crate::mxfp4::matvec_packed(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act, o, 1);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
    }
    let mut y = vec![0f32; cfg.routed_hidden];
    for (ei, &wi) in weights.iter().enumerate() {
        for j in 0..cfg.routed_hidden {
            y[j] += wi * outs[ei * cfg.routed_hidden + j];
        }
    }
    // norm BEFORE up-proj
    let mut yn = vec![0f32; cfg.routed_hidden];
    rmsnorm(cfg, &y, Model::t(data, &w.routed_norm), &mut yn);
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.routed_up), cfg.d, cfg.routed_hidden, &yn, &mut out);
    // shared experts (2): SiTU MLP on the pre-down input
    let mut sa = vec![0f32; cfg.shared_inter];
    let mut su = vec![0f32; cfg.shared_inter];
    matvec(Model::t(data, &w.shared_gate), cfg.shared_inter, cfg.d, x, &mut sa);
    matvec(Model::t(data, &w.shared_up), cfg.shared_inter, cfg.d, x, &mut su);
    let mut sact = vec![0f32; cfg.shared_inter];
    for j in 0..cfg.shared_inter {
        sact[j] = situ(sa[j], su[j]);
    }
    let mut sout = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.shared_down), cfg.d, cfg.shared_inter, &sact, &mut sout);
    if layer == 1 {
        let (routed, shared) = (out.clone(), sout.clone());
        parity_rec(|d| {
            d.l1_routed.insert(pos, routed);
            d.l1_shared.insert(pos, shared);
        });
    }
    for j in 0..cfg.d {
        out[j] += sout[j];
    }
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

fn dense_forward(cfg: &Config, data: &[u8], w: &DenseW, x: &[f32], prof: &mut Prof) -> Vec<f32> {
    let tm = Instant::now();
    let mut a = vec![0f32; cfg.dense_inter];
    let mut u = vec![0f32; cfg.dense_inter];
    matvec(Model::t(data, &w.gate), cfg.dense_inter, cfg.d, x, &mut a);
    matvec(Model::t(data, &w.up), cfg.dense_inter, cfg.d, x, &mut u);
    let mut act = vec![0f32; cfg.dense_inter];
    for j in 0..cfg.dense_inter {
        act[j] = situ(a[j], u[j]);
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.down), cfg.d, cfg.dense_inter, &act, &mut out);
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

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
fn matvec_packed_nt(packed: &[u8], scales: &[u8], rows: usize, cols: usize, xt: &[f32], m: usize, out: &mut [f32]) {
    use crate::mxfp4::{E2M1, exp2_i};
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

/// Batched MoE for prefill: `x` = n position rows [n * d], returns [n * d].
/// Router and shared/latent projections run as gemm_batch; the top-k
/// selection is per position (same code as the sequential path); the expert
/// work is grouped by expert id, one pool job per used expert over all its
/// assigned (position, slot) pairs, so a packed expert blob is read once per
/// prompt instead of once per token. Bit-identical to moe_forward per
/// position: each expert evaluation is the same matvec_packed sequence and
/// the combination keeps the slot order.
#[allow(clippy::too_many_arguments)]
fn moe_prefill(
    cfg: &Config,
    data: &[u8],
    w: &MoeW,
    x: &[f32],
    n: usize,
    prof: &mut Prof,
    layer: usize,
    pos0: usize,
    stream: Option<&crate::stream::ExpertCache>,
) -> Vec<f32> {
    // noaux_tc router: sigmoid, +bias for selection, weights without bias
    let tm = Instant::now();
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; n * cfg.n_experts];
    gemm_batch(gate_w, cfg.n_experts, cfg.d, x, n, &mut logits);
    // top-k selection per position (identical to moe_forward)
    let mut sels: Vec<Vec<(u32, f32)>> = Vec::with_capacity(n); // (expert, renormalized weight) in slot order
    for t in 0..n {
        let logits_t = &logits[t * cfg.n_experts..(t + 1) * cfg.n_experts];
        let mut sel: Vec<(u32, f32, f32)> = Vec::with_capacity(cfg.top_k); // (expert, score, key)
        for (i, &l) in logits_t.iter().enumerate() {
            let sc = sigmoid(l);
            let key = sc + gate_b[i];
            let item = (i as u32, sc, key);
            if sel.len() < cfg.top_k {
                sel.push(item);
                if sel.len() == cfg.top_k {
                    sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                }
            } else if key > sel[cfg.top_k - 1].2 {
                sel[cfg.top_k - 1] = item;
                sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        }
        let sumw: f32 = sel.iter().map(|s| s.1).sum::<f32>() + 1e-20;
        if ROUTER_LAYERS.contains(&layer) {
            let mut ids: Vec<u32> = sel.iter().map(|s| s.0).collect();
            ids.sort();
            parity_rec(|d| {
                d.router.insert((pos0 + t, layer), ids);
            });
        }
        // --debug-routing: top-3 by renormalized weight + count of top-16 appearances
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                let mut top3: Vec<(u32, f32)> = sel.iter().map(|s| (s.0, s.1 / sumw)).collect();
                top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                top3.truncate(3);
                d.cur.push((layer, top3));
                for s in &sel {
                    *d.counts.entry((layer, s.0)).or_insert(0) += 1;
                }
            }
        });
        sels.push(sel.iter().map(|s| (s.0, s.1 / sumw)).collect());
        // count-min routing statistics (no-op unless routestats/MICROKIMI_ROUTECMS)
        for s in &sel {
            crate::cms::record(layer, s.0);
        }
        // routing history for the draft-aware prefetch (streaming runs only)
        if stream.is_some() {
            crate::stream::route_record((pos0 + t) as u32, layer as u32, sel.iter().map(|s| s.0).collect());
        }
    }
    let mut h = vec![0f32; n * cfg.routed_hidden];
    gemm_batch(Model::t(data, &w.routed_down), cfg.routed_hidden, cfg.d, x, n, &mut h);
    prof.t_router += tm.elapsed().as_secs_f64();

    // MXFP4 experts grouped by expert id: one job per used expert. Inside a
    // job the expert's (position, slot) pairs are evaluated as a batch with
    // matvec_packed_nt (packed blob read and dequantized once per element
    // for up to 8 positions), then scattered to their output slots.
    let tm = Instant::now();
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
    let (erh, emi, topk) = (cfg.routed_hidden, cfg.moe_inter, cfg.top_k); // copies for the 'static closures
    let cbp = crate::pool::SPtr(w.vq_cb.as_ptr());
    let cblen = w.vq_cb.len();
    let mut by_expert: std::collections::HashMap<u32, Vec<(usize, usize)>> = std::collections::HashMap::new();
    for (t, sel) in sels.iter().enumerate() {
        for (slot, &(e, _)) in sel.iter().enumerate() {
            by_expert.entry(e).or_default().push((t, slot));
        }
    }
    let mut outs = vec![0f32; n * cfg.top_k * cfg.routed_hidden];
    match stream {
        // historical full-load path: expert blobs read straight from the file image
        None => {
            let dp = crate::pool::SPtrU8(data.as_ptr());
            let dlen = data.len();
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(by_expert.len());
            for (e, pairs) in by_expert {
                let offs = w.experts[e as usize];
                let vq = w.experts_vq[e as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (dp, hp, op, cbp) = (dp, hp, op, cbp);
                    unsafe {
                        let data = std::slice::from_raw_parts(dp.0, dlen);
                        let blob = |i: usize| &data[offs[i] as usize..offs[i] as usize + eblob];
                        let m = pairs.len();
                        let m8 = m.next_multiple_of(8); // zero-padded lanes (outputs ignored)
                        // gather the inputs of this expert's pairs, transposed [erh][m8]
                        let mut ht = vec![0f32; m8 * erh];
                        for (i, (t, _)) in pairs.iter().enumerate() {
                            let h = std::slice::from_raw_parts(hp.0.add(t * erh), erh);
                            for c in 0..erh {
                                ht[c * m8 + i] = h[c];
                            }
                        }
                        let mut a = vec![0f32; m8 * emi];
                        let mut u = vec![0f32; m8 * emi];
                        if vq {
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq_nt(cb, blob(0), emi, erh, &ht, m8, &mut a);
                            crate::quant::matvec_vq_nt(cb, blob(2), emi, erh, &ht, m8, &mut u);
                        } else {
                            matvec_packed_nt(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, &ht, m8, &mut a);
                            matvec_packed_nt(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, &ht, m8, &mut u);
                        }
                        // SiTU, transposed [emi][m8]
                        let mut act_t = vec![0f32; m8 * emi];
                        for i in 0..m {
                            for j in 0..emi {
                                act_t[j * m8 + i] = situ(a[i * emi + j], u[i * emi + j]);
                            }
                        }
                        let mut o = vec![0f32; m8 * erh];
                        if vq {
                            let cb = std::slice::from_raw_parts(cbp.0, cblen);
                            crate::quant::matvec_vq_nt(cb, blob(1), erh, emi, &act_t, m8, &mut o);
                        } else {
                            matvec_packed_nt(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act_t, m8, &mut o);
                        }
                        for (i, (t, slot)) in pairs.iter().enumerate() {
                            let dst = std::slice::from_raw_parts_mut(op.0.add((t * topk + slot) * erh), erh);
                            dst.copy_from_slice(&o[i * erh..(i + 1) * erh]);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
        // --stream: one job per used expert, bytes pulled through the
        // three-tier cache before the same batched matvec sequence runs.
        Some(cache) => {
            let cp = cache as *const crate::stream::ExpertCache as usize;
            let hp = crate::pool::SPtr(h.as_ptr());
            let op = crate::pool::MPtr(outs.as_mut_ptr());
            let layer32 = layer as u32;
            // same offset-sorted submission as moe_forward (each job scatters
            // to its own (position, slot) outputs: order cannot leak into the
            // result)
            let mut order: Vec<(u32, Vec<(usize, usize)>)> = by_expert.into_iter().collect();
            if crate::stream::offset_sort() {
                order.sort_by_key(|(e, _)| w.experts[*e as usize][0]);
            }
            // fused run fetch, same as moe_forward: missing experts of this
            // layer land in the RAM LRU with one span read per file-adjacent
            // run, the compute jobs below then hit the cache
            let items: Vec<(u32, [u64; 3], usize)> = order
                .iter()
                .map(|(e, _)| (*e, w.experts[*e as usize], if w.experts_vq[*e as usize] { expert_vq_blob } else { expert_blob }))
                .collect();
            cache.warm_batch(layer32, &items);
            let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(order.len());
            for (e, pairs) in order {
                let offs = w.experts[e as usize];
                let vq = w.experts_vq[e as usize];
                let eblob = if vq { expert_vq_blob } else { expert_blob };
                jobs.push(Box::new(move || {
                    let (hp, op, cbp) = (hp, op, cbp);
                    unsafe {
                        let cache = &*(cp as *const crate::stream::ExpertCache);
                        let served = cache.get(layer32, e, offs, eblob);
                        // shadow fallback, same contract as moe_forward:
                        // Served::Shadow = VQ1 shadow bytes + shadow codebook
                        // (degraded); Served::Full = historical path.
                        let (bytes, vq, eb, cb): (&[u8], bool, usize, &[f32]) = match &served {
                            crate::stream::Served::Full(b) => (&b[..], vq, eblob, std::slice::from_raw_parts(cbp.0, cblen)),
                            crate::stream::Served::Shadow(s, off) => {
                                (&s.data[*off..*off + 3 * expert_vq_blob], true, expert_vq_blob, &s.cb[..])
                            }
                        };
                        let blob = |i: usize| &bytes[i * eb..(i + 1) * eb];
                        let m = pairs.len();
                        let m8 = m.next_multiple_of(8); // zero-padded lanes (outputs ignored)
                        // gather the inputs of this expert's pairs, transposed [erh][m8]
                        let mut ht = vec![0f32; m8 * erh];
                        for (i, (t, _)) in pairs.iter().enumerate() {
                            let h = std::slice::from_raw_parts(hp.0.add(t * erh), erh);
                            for c in 0..erh {
                                ht[c * m8 + i] = h[c];
                            }
                        }
                        let mut a = vec![0f32; m8 * emi];
                        let mut u = vec![0f32; m8 * emi];
                        if vq {
                            crate::quant::matvec_vq_nt(cb, blob(0), emi, erh, &ht, m8, &mut a);
                            crate::quant::matvec_vq_nt(cb, blob(2), emi, erh, &ht, m8, &mut u);
                        } else {
                            matvec_packed_nt(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, &ht, m8, &mut a);
                            matvec_packed_nt(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, &ht, m8, &mut u);
                        }
                        // SiTU, transposed [emi][m8]
                        let mut act_t = vec![0f32; m8 * emi];
                        for i in 0..m {
                            for j in 0..emi {
                                act_t[j * m8 + i] = situ(a[i * emi + j], u[i * emi + j]);
                            }
                        }
                        let mut o = vec![0f32; m8 * erh];
                        if vq {
                            crate::quant::matvec_vq_nt(cb, blob(1), erh, emi, &act_t, m8, &mut o);
                        } else {
                            matvec_packed_nt(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act_t, m8, &mut o);
                        }
                        for (i, (t, slot)) in pairs.iter().enumerate() {
                            let dst = std::slice::from_raw_parts_mut(op.0.add((t * topk + slot) * erh), erh);
                            dst.copy_from_slice(&o[i * erh..(i + 1) * erh]);
                        }
                    }
                }));
            }
            crate::pool::pool().run(jobs);
        }
    }
    // combination per position in slot order, norm BEFORE up-proj
    let mut yn = vec![0f32; n * cfg.routed_hidden];
    for (t, sel) in sels.iter().enumerate() {
        let mut y = vec![0f32; cfg.routed_hidden];
        for (slot, &(_, wi)) in sel.iter().enumerate() {
            for j in 0..cfg.routed_hidden {
                y[j] += wi * outs[(t * cfg.top_k + slot) * cfg.routed_hidden + j];
            }
        }
        rmsnorm(cfg, &y, Model::t(data, &w.routed_norm), &mut yn[t * cfg.routed_hidden..(t + 1) * cfg.routed_hidden]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.routed_up), cfg.d, cfg.routed_hidden, &yn, n, &mut out);
    // shared experts (2): SiTU MLP on the pre-down input
    let mut sa = vec![0f32; n * cfg.shared_inter];
    let mut su = vec![0f32; n * cfg.shared_inter];
    gemm_batch(Model::t(data, &w.shared_gate), cfg.shared_inter, cfg.d, x, n, &mut sa);
    gemm_batch(Model::t(data, &w.shared_up), cfg.shared_inter, cfg.d, x, n, &mut su);
    let mut sact = vec![0f32; n * cfg.shared_inter];
    for j in 0..n * cfg.shared_inter {
        sact[j] = situ(sa[j], su[j]);
    }
    let mut sout = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.shared_down), cfg.d, cfg.shared_inter, &sact, n, &mut sout);
    if layer == 1 {
        for t in 0..n {
            let routed = out[t * cfg.d..(t + 1) * cfg.d].to_vec();
            let shared = sout[t * cfg.d..(t + 1) * cfg.d].to_vec();
            parity_rec(|d| {
                d.l1_routed.insert(pos0 + t, routed);
                d.l1_shared.insert(pos0 + t, shared);
            });
        }
    }
    for j in 0..n * cfg.d {
        out[j] += sout[j];
    }
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

/// Batched dense MLP for prefill: `x` = n position rows [n * d], returns
/// [n * d]. Bit-identical to dense_forward per position.
fn dense_prefill(cfg: &Config, data: &[u8], w: &DenseW, x: &[f32], n: usize, prof: &mut Prof) -> Vec<f32> {
    let tm = Instant::now();
    let mut a = vec![0f32; n * cfg.dense_inter];
    let mut u = vec![0f32; n * cfg.dense_inter];
    gemm_batch(Model::t(data, &w.gate), cfg.dense_inter, cfg.d, x, n, &mut a);
    gemm_batch(Model::t(data, &w.up), cfg.dense_inter, cfg.d, x, n, &mut u);
    let mut act = vec![0f32; n * cfg.dense_inter];
    for j in 0..n * cfg.dense_inter {
        act[j] = situ(a[j], u[j]);
    }
    let mut out = vec![0f32; n * cfg.d];
    gemm_batch(Model::t(data, &w.down), cfg.d, cfg.dense_inter, &act, n, &mut out);
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

// ── parity dumps (thread-local, inactive during normal inference) ──

#[derive(Default)]
pub struct ParityDump {
    pub hiddens: std::collections::HashMap<(usize, usize), Vec<f32>>, // (pos, layer)
    pub l1_attn: std::collections::HashMap<usize, Vec<f32>>,
    pub l1_routed: std::collections::HashMap<usize, Vec<f32>>,
    pub l1_shared: std::collections::HashMap<usize, Vec<f32>>,
    pub router: std::collections::HashMap<(usize, usize), Vec<u32>>, // (pos, layer) sorted top-16
}

pub const DUMP_LAYERS: [usize; 7] = [0, 1, 3, 4, 12, 47, 92];
pub const ROUTER_LAYERS: [usize; 3] = [1, 47, 92];

thread_local! {
    pub static PARITY: std::cell::RefCell<Option<ParityDump>> = std::cell::RefCell::new(None);
}

// ── --debug-routing collection (thread-local, inactive by default) ──

#[derive(Default)]
pub struct RoutingDebug {
    pub cur: Vec<(usize, Vec<(u32, f32)>)>, // layer → top-3 (expert, renormalized weight)
    pub counts: std::collections::HashMap<(usize, u32), u32>, // (layer, expert) → times in top-16
}

thread_local! {
    pub static ROUTING: std::cell::RefCell<Option<RoutingDebug>> = std::cell::RefCell::new(None);
}

fn parity_rec(f: impl FnOnce(&mut ParityDump)) {
    PARITY.with(|p| {
        if let Some(d) = p.borrow_mut().as_mut() {
            f(d);
        }
    });
}

// ── --dump-hidden collection (inactive by default) ──
//
// Diagnostic instrument for pruned-model collapse: the RMS of the hidden
// state after each layer, printed ONCE at the end of the first prefill (or
// the first generated token when the prefill is empty). Read-only: without
// the flag the forward paths are untouched (bit-exact).
pub static DUMP_HIDDEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static DUMP_HIDDEN_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_dump_hidden(on: bool) {
    DUMP_HIDDEN.store(on, std::sync::atomic::Ordering::Relaxed);
    DUMP_HIDDEN_DONE.store(false, std::sync::atomic::Ordering::Relaxed);
}

fn dump_hidden_on() -> bool {
    use std::sync::atomic::Ordering;
    DUMP_HIDDEN.load(Ordering::Relaxed) && !DUMP_HIDDEN_DONE.load(Ordering::Relaxed)
}

/// RMS (sqrt of the mean of squares): the same norm rmsnorm rescales to ~1.
fn vec_rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// Prints the per-layer table once, then disarms (subsequent tokens of the
/// same run are not re-dumped).
fn dump_hidden_print(per_layer: &[(usize, &'static str, f64)], residual_rms: f64, logits: &[f32]) {
    let mean = logits.iter().map(|&x| x as f64).sum::<f64>() / logits.len().max(1) as f64;
    let std = (logits.iter().map(|&x| (x as f64 - mean) * (x as f64 - mean)).sum::<f64>() / logits.len().max(1) as f64).sqrt();
    println!("── dump-hidden: rms of the hidden state after each layer ──");
    for (l, kind, rms) in per_layer {
        println!("  layer {:>2} ({}): rms={:.4}", l, kind, rms);
    }
    println!("  final residual: rms={:.4}", residual_rms);
    println!("  logits: std={:.4} mean={:.4}", std, mean);
    DUMP_HIDDEN_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
}

// ── --logit-lens collection (inactive by default) ──
//
// Logit lens: each post-layer hidden state is projected through the FINAL
// norm + lm_head and the top-5 softmax tokens are printed, showing where in
// depth the next-token semantics emerge (or die). Read-only diagnostic: the
// captured hiddens are clones, the forward math is untouched (bit-exact).
pub static LOGIT_LENS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOGIT_LENS_ALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOGIT_LENS_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// on: lens active at the last prefill position. all: also on every
/// generated token (--logit-lens-all).
pub fn set_logit_lens(on: bool, all: bool) {
    LOGIT_LENS.store(on, std::sync::atomic::Ordering::Relaxed);
    LOGIT_LENS_ALL.store(all, std::sync::atomic::Ordering::Relaxed);
    LOGIT_LENS_DONE.store(false, std::sync::atomic::Ordering::Relaxed);
}

fn logit_lens_on() -> bool {
    use std::sync::atomic::Ordering;
    LOGIT_LENS.load(Ordering::Relaxed) && (LOGIT_LENS_ALL.load(Ordering::Relaxed) || !LOGIT_LENS_DONE.load(Ordering::Relaxed))
}

/// One lens row: (layer index, attention kind, top-5 (token id, softmax prob)).
type LensRows = Vec<(usize, &'static str, Vec<(usize, f32)>)>;
static LENS_PENDING: std::sync::Mutex<Option<LensRows>> = std::sync::Mutex::new(None);

/// Projects every captured post-layer hidden through norm_f + lm_head (the
/// same rmsnorm/matvec pair as the normal path) and stashes the top-5 rows
/// for logit_lens_print_maybe. The last layer reuses the real logits: the
/// output residual mix (out_res_w) sits between it and the final norm, so
/// this keeps the bottom row bit-identical to the model's own candidates.
fn logit_lens_compute(cfg: &Config, lm_head: &[f32], norm_f: &[f32], per_layer: &[(usize, &'static str, Vec<f32>)], final_logits: &[f32]) {
    let mut rows: LensRows = Vec::with_capacity(per_layer.len());
    let mut xf = vec![0f32; cfg.d];
    let mut logits = vec![0f32; cfg.vocab];
    for (i, (l, kind, h)) in per_layer.iter().enumerate() {
        if i + 1 == per_layer.len() {
            rows.push((*l, kind, top_k_probs(final_logits, 5)));
        } else {
            rmsnorm(cfg, h, norm_f, &mut xf);
            matvec(lm_head, cfg.vocab, cfg.d, &xf, &mut logits);
            rows.push((*l, kind, top_k_probs(&logits, 5)));
        }
    }
    LOGIT_LENS_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
    *LENS_PENDING.lock().unwrap() = Some(rows);
}

/// Prints the pending lens table (called by the generation loop, which owns
/// the tokenizer, right after each forward/prefill). No-op without a dump.
pub fn logit_lens_print_maybe(tok: &AnyTokenizer, label: &str) {
    let rows = LENS_PENDING.lock().unwrap().take();
    let Some(rows) = rows else { return };
    println!("── logit lens ({}): top-5 of each layer through final norm + lm_head ──", label);
    for (l, kind, top) in rows {
        let segs: Vec<String> = top.iter().map(|&(id, p)| format!("{} {:.1}%", py_repr(&tok.decode_id(id as u32)), p * 100.0)).collect();
        println!("  layer {:>2} ({}): {}", l, kind, segs.join("  "));
    }
}

// ── full forward ──

impl Model {
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // destructuring: independent per-field borrows (data immutable,
        // caches/prof mutable) - no raw pointers.
        let Self { cfg, bin, embed, lm_head, lm_head_q8, norm_f, out_res_w, layers, caches, last_logits, prof, stream } = self;
        let cfg = &*cfg;
        let data = &bin.data[..];
        let embed = Self::t(data, embed);
        let mut hidden = embed[token as usize * cfg.d..(token as usize + 1) * cfg.d].to_vec();
        let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(8);
        let mut buf_res = vec![0f32; cfg.d];
        let mut x = vec![0f32; cfg.d];
        let hd_on = dump_hidden_on();
        let mut hd: Vec<(usize, &'static str, f64)> = Vec::new();
        let lens_on = logit_lens_on();
        let mut lens: Vec<(usize, &'static str, Vec<f32>)> = Vec::new();

        for l in 0..cfg.n_layers {
            let prefix: Option<Vec<f32>> = Some(hidden.clone());
            let layer = &layers[l];
            let tm = Instant::now();
            if !blocks.is_empty() {
                attn_res(cfg, prefix.as_ref().unwrap(), &blocks, &layer.sa_res_w, &mut buf_res);
                hidden.copy_from_slice(&buf_res);
            }
            let prefix: Option<Vec<f32>> = if l % cfg.attn_res_block == 0 {
                blocks.push(prefix.unwrap());
                None
            } else {
                prefix
            };
            rmsnorm(cfg, &hidden, Self::t(data, &layer.input_ln), &mut x);
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let attn_out = match (&layer.attn, &mut caches[l]) {
                (AttnW::Kda(w), Cache::Kda(c)) => {
                    let mut p = Prof::default();
                    let out = kda_forward(cfg, data, w, c, &x, &mut p);
                    prof.t_kda_proj += p.t_kda_proj;
                    prof.t_kda_conv += p.t_kda_conv;
                    prof.t_kda_recur += p.t_kda_recur;
                    out
                }
                (AttnW::Mla(w), Cache::Mla(c)) => {
                    let mut p = Prof::default();
                    let out = mla_forward(cfg, data, w, c, &x, &mut p);
                    prof.t_mla += p.t_mla;
                    out
                }
                _ => unreachable!(),
            };
            if l == 1 {
                let a = attn_out.clone();
                parity_rec(|d| {
                    d.l1_attn.insert(pos, a);
                });
            }
            let prefix2: Vec<f32> = match prefix {
                Some(p) => {
                    let mut p = p;
                    for j in 0..cfg.d {
                        p[j] += attn_out[j];
                    }
                    p
                }
                None => attn_out,
            };

            let tm = Instant::now();
            attn_res(cfg, &prefix2, &blocks, &layer.mlp_res_w, &mut buf_res);
            hidden.copy_from_slice(&buf_res);
            rmsnorm(cfg, &hidden, Self::t(data, &layer.post_ln), &mut x);
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let mlp_out = match &layer.ffn {
                FfnW::Dense(w) => {
                    let mut p = Prof::default();
                    let out = dense_forward(cfg, data, w, &x, &mut p);
                    prof.t_experts += p.t_experts;
                    out
                }
                FfnW::Moe(w) => {
                    let mut p = Prof::default();
                    // router-lookahead: while this layer's experts compute,
                    // predict the NEXT MoE layer's experts with its own
                    // router on the current MoE input x and prefetch them
                    if let Some(cache) = stream.as_ref() {
                        let n = crate::stream::predict_n();
                        if n > 0 && crate::stream::lookahead_on() {
                            let next = (l + 1..cfg.n_layers).find_map(|l2| match &layers[l2].ffn {
                                FfnW::Moe(w2) => Some((l2, w2)),
                                _ => None,
                            });
                            if let Some((l2, w2)) = next {
                                let tml = Instant::now();
                                moe_lookahead(cfg, data, w2, l2, &x, n, cache);
                                p.t_router += tml.elapsed().as_secs_f64();
                            }
                        }
                    }
                    let out = moe_forward(cfg, data, w, &x, &mut p, l, pos, stream.as_ref());
                    prof.t_router += p.t_router;
                    prof.t_experts += p.t_experts;
                    out
                }
            };
            for j in 0..cfg.d {
                hidden[j] = prefix2[j] + mlp_out[j];
            }
            if hd_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                hd.push((l, kind, vec_rms(&hidden)));
            }
            if lens_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                lens.push((l, kind, hidden.clone()));
            }
            if DUMP_LAYERS.contains(&l) {
                let h = hidden.clone();
                parity_rec(|d| {
                    d.hiddens.insert((pos, l), h);
                });
            }
        }

        let tm = Instant::now();
        attn_res(cfg, &hidden, &blocks, &out_res_w, &mut buf_res);
        hidden.copy_from_slice(&buf_res);
        let mut xf = vec![0f32; cfg.d];
        rmsnorm(cfg, &hidden, Self::t(data, &norm_f), &mut xf);
        prof.t_norm_res += tm.elapsed().as_secs_f64();
        let tm = Instant::now();
        let mut logits = vec![0f32; cfg.vocab];
        Self::logits_project(data, &lm_head, lm_head_q8.as_ref(), cfg, &xf, &mut logits);
        prof.t_lm_head += tm.elapsed().as_secs_f64();
        if hd_on {
            dump_hidden_print(&hd, vec_rms(&hidden), &logits);
        }
        if lens_on {
            logit_lens_compute(cfg, Self::t(data, &lm_head), Self::t(data, &norm_f), &lens, &logits);
        }
        *last_logits = logits.clone();
        logits
    }

    /// Batched prefill: ingests `ids` (absolute positions pos0..pos0+n) in
    /// one pass and returns the logits of the last position. Every layer is
    /// applied to all positions at once: projections/router/experts run as
    /// gemm_batch over the [n * d] hidden buffer, the KDA conv + recurrence
    /// stay sequential over positions (cheap elementwise work), MLA appends
    /// its n latent K/V rows in order and attends causally per query
    /// position. lm_head runs only on the last position (the only logits the
    /// generation loop consumes). Caches end in exactly the same state as n
    /// sequential forward calls, and every per-position computation keeps
    /// the same accumulation order, so the result is bit-identical.
    pub fn prefill(&mut self, ids: &[u32], pos0: usize) -> Vec<f32> {
        self.prefill_impl(ids, pos0, false).pop().unwrap()
    }

    /// Batched prefill returning the logits of EVERY position, not just the
    /// last (consumed by the --spec verification pass). Same pass as prefill
    /// with lm_head applied per position: each logits vector is bit-identical
    /// to what a sequential forward of that prefix would produce.
    pub fn prefill_all(&mut self, ids: &[u32], pos0: usize) -> Vec<Vec<f32>> {
        self.prefill_impl(ids, pos0, true)
    }

    /// Draft-aware expert prefetch (--spec / --spec-rosa with --stream;
    /// MICROKIMI_DRAFTPREFETCH=0 disables): the batched verification pass
    /// that ingests `toks` (pending token + drafted proposals) will route
    /// them for real, so the experts it will pull are predictable before
    /// the pass starts. Both proposers draft tokens that ALREADY occurred
    /// in the committed context, and every ingested position had its router
    /// picks recorded (stream::route_record, real hidden states, not an
    /// embedding proxy - measured ~0% recall), so the routing of the source
    /// occurrence, `srcs[t]` = the context position toks[t] was lifted
    /// from, is the prediction: union of the recorded top-k sets over the
    /// source positions, background-fetched through the stream cache so the
    /// pass finds its experts in the RAM LRU. Same contract as the
    /// router-lookahead prefetch: only WHEN bytes land in the cache
    /// changes, never WHICH experts the model computes - mispredictions are
    /// harmless LRU fills and the greedy output stays bit-identical.
    /// No-op without --stream.
    pub fn draft_prefetch(&self, toks: &[u32], srcs: &[Option<usize>]) {
        let Some(cache) = &self.stream else { return };
        if toks.is_empty() || !crate::stream::draft_prefetch_on() {
            return;
        }
        let cfg = &self.cfg;
        let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
        let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
        let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::VQ_DIM;
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut jobs: Vec<(u32, u32, [u64; 3], usize)> = Vec::new();
        for src in srcs.iter().flatten() {
            let Some(layers) = crate::stream::route_lookup(*src as u32) else { continue };
            for (layer, experts) in layers {
                let FfnW::Moe(w) = &self.layers[layer as usize].ffn else { continue };
                for e in experts {
                    if seen.insert((layer, e)) {
                        let eblob = if w.experts_vq[e as usize] { expert_vq_blob } else { expert_blob };
                        jobs.push((layer, e, w.experts[e as usize], eblob));
                    }
                }
            }
        }
        cache.prefetch_draft(jobs);
    }

    fn prefill_impl(&mut self, ids: &[u32], pos0: usize, all_logits: bool) -> Vec<Vec<f32>> {
        if ids.len() == 1 {
            return vec![self.forward(ids[0], pos0)];
        }
        let Self { cfg, bin, embed, lm_head, lm_head_q8, norm_f, out_res_w, layers, caches, last_logits, prof, stream } = self;
        let cfg = &*cfg;
        let data = &bin.data[..];
        let n = ids.len();
        let d = cfg.d;
        let embed = Self::t(data, embed);
        let mut hidden = vec![0f32; n * d];
        for (t, &id) in ids.iter().enumerate() {
            hidden[t * d..(t + 1) * d].copy_from_slice(&embed[id as usize * d..(id as usize + 1) * d]);
        }
        let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(8); // each [n * d]
        let mut buf_res = vec![0f32; n * d];
        let mut x = vec![0f32; n * d];
        let hd_on = dump_hidden_on();
        let mut hd: Vec<(usize, &'static str, f64)> = Vec::new();
        let lens_on = logit_lens_on();
        let mut lens: Vec<(usize, &'static str, Vec<f32>)> = Vec::new();

        for l in 0..cfg.n_layers {
            let layer = &layers[l];
            let tm = Instant::now();
            let mut prefix: Option<Vec<f32>> = Some(hidden.clone());
            if !blocks.is_empty() {
                for t in 0..n {
                    let brefs: Vec<&[f32]> = blocks.iter().map(|b| &b[t * d..(t + 1) * d]).collect();
                    attn_res_refs(cfg, &prefix.as_ref().unwrap()[t * d..(t + 1) * d], &brefs, &layer.sa_res_w, &mut buf_res[t * d..(t + 1) * d]);
                }
                hidden.copy_from_slice(&buf_res);
            }
            let prefix: Option<Vec<f32>> = if l % cfg.attn_res_block == 0 {
                blocks.push(prefix.take().unwrap());
                None
            } else {
                prefix
            };
            for t in 0..n {
                rmsnorm(cfg, &hidden[t * d..(t + 1) * d], Self::t(data, &layer.input_ln), &mut x[t * d..(t + 1) * d]);
            }
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let attn_out = match (&layer.attn, &mut caches[l]) {
                (AttnW::Kda(w), Cache::Kda(c)) => {
                    let mut p = Prof::default();
                    let out = kda_prefill(cfg, data, w, c, &x, n, &mut p);
                    prof.t_kda_proj += p.t_kda_proj;
                    prof.t_kda_conv += p.t_kda_conv;
                    prof.t_kda_recur += p.t_kda_recur;
                    out
                }
                (AttnW::Mla(w), Cache::Mla(c)) => {
                    let mut p = Prof::default();
                    let out = mla_prefill(cfg, data, w, c, &x, n, &mut p);
                    prof.t_mla += p.t_mla;
                    out
                }
                _ => unreachable!(),
            };
            if l == 1 {
                for t in 0..n {
                    let a = attn_out[t * d..(t + 1) * d].to_vec();
                    parity_rec(|d| {
                        d.l1_attn.insert(pos0 + t, a);
                    });
                }
            }
            let prefix2: Vec<f32> = match prefix {
                Some(mut p) => {
                    for j in 0..n * d {
                        p[j] += attn_out[j];
                    }
                    p
                }
                None => attn_out,
            };

            let tm = Instant::now();
            for t in 0..n {
                let brefs: Vec<&[f32]> = blocks.iter().map(|b| &b[t * d..(t + 1) * d]).collect();
                attn_res_refs(cfg, &prefix2[t * d..(t + 1) * d], &brefs, &layer.mlp_res_w, &mut buf_res[t * d..(t + 1) * d]);
            }
            hidden.copy_from_slice(&buf_res);
            for t in 0..n {
                rmsnorm(cfg, &hidden[t * d..(t + 1) * d], Self::t(data, &layer.post_ln), &mut x[t * d..(t + 1) * d]);
            }
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let mlp_out = match &layer.ffn {
                FfnW::Dense(w) => {
                    let mut p = Prof::default();
                    let out = dense_prefill(cfg, data, w, &x, n, &mut p);
                    prof.t_experts += p.t_experts;
                    out
                }
                FfnW::Moe(w) => {
                    let mut p = Prof::default();
                    let out = moe_prefill(cfg, data, w, &x, n, &mut p, l, pos0, stream.as_ref());
                    prof.t_router += p.t_router;
                    prof.t_experts += p.t_experts;
                    out
                }
            };
            for j in 0..n * d {
                hidden[j] = prefix2[j] + mlp_out[j];
            }
            if hd_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                hd.push((l, kind, vec_rms(&hidden[(n - 1) * d..n * d])));
            }
            if lens_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                lens.push((l, kind, hidden[(n - 1) * d..n * d].to_vec()));
            }
            if DUMP_LAYERS.contains(&l) {
                for t in 0..n {
                    let h = hidden[t * d..(t + 1) * d].to_vec();
                    parity_rec(|d| {
                        d.hiddens.insert((pos0 + t, l), h);
                    });
                }
            }
        }

        let tm = Instant::now();
        for t in 0..n {
            let brefs: Vec<&[f32]> = blocks.iter().map(|b| &b[t * d..(t + 1) * d]).collect();
            attn_res_refs(cfg, &hidden[t * d..(t + 1) * d], &brefs, &out_res_w, &mut buf_res[t * d..(t + 1) * d]);
        }
        hidden.copy_from_slice(&buf_res);
        if all_logits {
            // --spec verification: rmsnorm + lm_head on EVERY position (the
            // same matvec as the single-token forward, so per-position
            // logits are bit-identical to a sequential run)
            let tm = Instant::now();
            let mut out = Vec::with_capacity(n);
            for t in 0..n {
                let mut xf = vec![0f32; d];
                rmsnorm(cfg, &hidden[t * d..(t + 1) * d], Self::t(data, &norm_f), &mut xf);
                let mut logits = vec![0f32; cfg.vocab];
                Self::logits_project(data, &lm_head, lm_head_q8.as_ref(), cfg, &xf, &mut logits);
                out.push(logits);
            }
            prof.t_norm_res += tm.elapsed().as_secs_f64();
            if lens_on {
                logit_lens_compute(cfg, Self::t(data, &lm_head), Self::t(data, &norm_f), &lens, out.last().unwrap());
            }
            *last_logits = out.last().unwrap().clone();
            return out;
        }
        let mut xf = vec![0f32; d];
        rmsnorm(cfg, &hidden[(n - 1) * d..n * d], Self::t(data, &norm_f), &mut xf);
        prof.t_norm_res += tm.elapsed().as_secs_f64();
        let tm = Instant::now();
        let mut logits = vec![0f32; cfg.vocab];
        Self::logits_project(data, &lm_head, lm_head_q8.as_ref(), cfg, &xf, &mut logits);
        prof.t_lm_head += tm.elapsed().as_secs_f64();
        if hd_on {
            // per-layer rms was taken on the LAST position (the one the
            // logits are computed from)
            dump_hidden_print(&hd, vec_rms(&hidden[(n - 1) * d..n * d]), &logits);
        }
        if lens_on {
            logit_lens_compute(cfg, Self::t(data, &lm_head), Self::t(data, &norm_f), &lens, &logits);
        }
        *last_logits = logits.clone();
        vec![logits]
    }
}

// ── greedy generation + display (rustgpt style) ──

pub(crate) fn top_k_probs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut z = 0f32;
    for &l in logits {
        z += (l - m).exp();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp() / z;
        if top.len() < k {
            top.push((i, p));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        } else if p > top[k - 1].1 {
            top[k - 1] = (i, p);
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
    }
    top
}

// ── sampling: temperature + top-p nucleus, xorshift64* RNG (build.rs style) ──

/// xorshift64* RNG, same generator style as build::Rng, seedable via --seed
/// for reproducible sampling (same seed + same prompt = same output).
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> XorShift {
        XorShift(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Decoding policy: temp <= 0 keeps the exact greedy argmax path; temp > 0
/// samples from softmax(logits / temp) restricted to the top-p nucleus.
/// spec > 0 enables n-gram speculative decoding (src/spec.rs, greedy only);
/// spec_rosa > 0 swaps the proposer for the suffix automaton (src/rosa.rs).
/// dry > 0 subtracts a DRY-style anti-repetition penalty from the logits
/// (apply_dry; 0 = off, the historical bit-exact path).
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub rng: XorShift,
    pub spec: usize,
    pub spec_rosa: usize,
    pub dry: f32,
}

impl Sampler {
    pub fn new(temp: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler { temp, top_p, rng: XorShift::new(seed), spec: 0, spec_rosa: 0, dry: 0.0 }
    }
    /// Default no-op decoding: the historical greedy behavior.
    pub fn greedy() -> Sampler {
        Sampler::new(0.0, 1.0, 0)
    }
}

/// Nucleus (top-p) sampling from softmax(logits / temp): sort the candidates
/// by probability desc, keep the smallest set covering `top_p` of the mass,
/// renormalize, draw with `rng`. Returns (token id, probability under the
/// truncated renormalized distribution).
fn sample_top_p(logits: &[f32], temp: f32, top_p: f32, rng: &mut XorShift) -> (u32, f32) {
    let inv_t = 1.0 / temp;
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - m) * inv_t).exp()))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f32 = probs.iter().map(|&(_, p)| p).sum();
    // smallest prefix covering top_p of the mass
    let top_p = top_p.clamp(0.0, 1.0);
    let mut keep = probs.len();
    let mut cum = 0f32;
    for (i, &(_, p)) in probs.iter().enumerate() {
        cum += p / total;
        if cum >= top_p {
            keep = i + 1;
            break;
        }
    }
    let nucleus = &probs[..keep.max(1)];
    let nsum: f32 = nucleus.iter().map(|&(_, p)| p).sum();
    let mut r = rng.uniform() as f32 * nsum;
    for &(id, p) in nucleus {
        r -= p;
        if r <= 0.0 {
            return (id, p / nsum);
        }
    }
    let (id, p) = nucleus[nucleus.len() - 1];
    (id, p / nsum)
}

fn py_repr(s: &str) -> String {
    let mut out = String::from("'");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\'' => out.push_str("\\'"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// DRY-style anti-repetition penalty (--dry P): a token that would EXTEND an
/// n-gram (length >= 3) already present earlier in the GENERATION gets
/// P x DECAY^distance subtracted from its logit, where distance is the
/// number of tokens between the end of the earlier occurrence and the
/// current tail (DECAY = 0.9: a repetition starting far back hurts less).
/// Only the last 64 generated tokens are scanned, and the prompt is never
/// scanned: what matters is the text the model itself produced. Plain
/// quadratic scan over the 64-token window, negligible next to a forward.
/// Shared by the run_turn_core_batch loop and the --spec verification.
pub(crate) fn apply_dry(logits: &mut [f32], generated: &[u32], pen: f32) {
    const WIN: usize = 64;
    const DECAY: f32 = 0.9;
    let w = &generated[generated.len().saturating_sub(WIN)..];
    if w.len() < 3 {
        return;
    }
    for n in 3..=8usize.min(w.len()) {
        let m = n - 1; // matched suffix length (the n-gram completes with the next token)
        let suffix = &w[w.len() - m..];
        for i in 0..w.len() - m {
            // the final occurrence (the suffix itself, at i == w.len() - m) is excluded
            if w[i..i + m] == *suffix {
                let culprit = w[i + m] as usize; // token that followed the earlier occurrence
                let dist = (w.len() - m - i) as f32;
                if culprit < logits.len() {
                    logits[culprit] -= pen * DECAY.powf(dist);
                }
            }
        }
    }
}

pub fn run_turn(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, sampler: &mut Sampler) -> String {
    model.reset_cache();
    run_turn_impl(ids, max_new, tok, model, debug, debug_routing, stop_id, false, None, sampler)
}

/// Same as run_turn but keeps the current caches (restored from a .mkmem
/// snapshot via --memory): the prompt tokens are fed on top of the loaded
/// state and `init_logits` (the logits stored in the snapshot) seed the
/// decoding when the prompt is empty - a pure continuation.
pub fn run_turn_resume(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    run_turn_impl(ids, max_new, tok, model, debug, debug_routing, stop_id, true, init_logits, sampler)
}

fn run_turn_impl(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, resumed: bool, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    model.prof = Prof::default();
    let mut pos = if resumed { model.cached_tokens() } else { 0 };
    // --spec N / --spec-rosa N: speculative decoding, greedy only (rejection
    // sampling for temp > 0 is future work; the flags are ignored there)
    if sampler.spec > 0 || sampler.spec_rosa > 0 {
        if sampler.temp > 0.0 {
            eprintln!("warning: --spec/--spec-rosa are greedy-only, ignoring them with --temp > 0");
        } else {
            let answer = crate::spec::run_turn_spec(ids, max_new, tok, model, pos, init_logits, debug, stop_id, sampler);
            model.prof.print_cfg(&model.cfg);
            return answer;
        }
    }
    let answer = run_turn_core_batch(
        ids,
        max_new,
        tok,
        &mut |batch: &[u32]| {
            let l = model.prefill(batch, pos);
            pos += batch.len();
            l
        },
        debug,
        debug_routing,
        stop_id,
        init_logits,
        sampler,
    );
    model.prof.print_cfg(&model.cfg);
    answer
}

/// Generic greedy generation loop: prefill then argmax decode through the
/// `fwd` closure (one forward per token, position tracked by the caller).
/// With `sampler.temp > 0` the argmax becomes top-p nucleus sampling.
/// Shared by the K3 Model (run_turn) and the DeepSeek DsModel (ds_run_turn).
pub fn run_turn_core(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(u32) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, sampler: &mut Sampler) -> String {
    run_turn_core_resume(ids, max_new, tok, fwd, debug, debug_routing, stop_id, None, sampler)
}

/// run_turn_core + optional initial logits restored from a .mkmem snapshot:
/// with an empty prompt the decoding starts straight from them (pure
/// continuation, no token is re-ingested).
pub fn run_turn_core_resume(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(u32) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    run_turn_core_batch(
        ids,
        max_new,
        tok,
        &mut |batch: &[u32]| {
            // sequential prefill, one forward per token
            let mut l = Vec::new();
            for &id in batch {
                l = fwd(id);
            }
            l
        },
        debug,
        debug_routing,
        stop_id,
        init_logits,
        sampler,
    )
}

/// Batch variant of run_turn_core_resume: `fwd` ingests a slice of tokens
/// and returns the logits of its last token. The whole prompt is handed
/// over in ONE call (batched prefill on the K3 Model); during decoding each
/// call carries exactly one token. Bit-identical generation as long as the
/// closure's prefill matches n sequential single-token forwards.
pub fn run_turn_core_batch(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(&[u32]) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    if debug_routing {
        ROUTING.with(|r| *r.borrow_mut() = Some(RoutingDebug::default()));
    }

    if debug {
        println!("{}", "=".repeat(64));
        println!("STEP 0 - TOKENIZATION  ({} tokens)", ids.len());
        println!("{}", "=".repeat(64));
        for (i, &id) in ids.iter().enumerate() {
            println!("  position {:2} : token {:6} = {}", i, id, py_repr(&tok.decode_id(id)));
        }
    }

    // ── prefill: the whole prompt in one batched call ──
    let t2 = Instant::now();
    let io0 = io_stats();
    let mut logits = init_logits.unwrap_or_default();
    if !ids.is_empty() {
        logits = fwd(ids);
        logit_lens_print_maybe(tok, "last prefill position");
    }
    if logits.is_empty() {
        eprintln!("error: nothing to continue from (empty prompt and no logits stored in the .mkmem snapshot)");
        std::process::exit(1);
    }
    let t_prefill = t2.elapsed();
    if debug {
        println!();
        println!("{}", "=".repeat(64));
        println!("STEP 1 - PREFILL  (caches filled)");
        println!("{}", "=".repeat(64));
        if ids.is_empty() {
            println!("⏱  skipped: pure continuation from the .mkmem snapshot");
        } else {
            println!("⏱  {:.2} s  for {} tokens ({:.1} ms/token)", t_prefill.as_secs_f64(), ids.len(), t_prefill.as_secs_f64() / ids.len() as f64 * 1000.0);
            if let (Some((b0, f0)), Some((b1, f1))) = (io0, io_stats()) {
                let gb = (b1 - b0) as f64 / 1e9;
                if gb > 0.01 {
                    println!("💾 {:.1} GB paged in from disk during prefill ({} major faults)", gb, f1 - f0);
                }
            }
        }
        println!();
        println!("{}", "=".repeat(64));
        if sampler.temp > 0.0 {
            println!("STEP 2 - GENERATION  (sampling: temp = {}, top-p = {}, stop = token {})", sampler.temp, sampler.top_p, stop_id);
        } else {
            println!("STEP 2 - GENERATION  (greedy: softmax → argmax, stop = token {})", stop_id);
        }
        println!("{}", "=".repeat(64));
    }

    let mut generated: Vec<u32> = Vec::new();
    let mut gen_times: Vec<f64> = Vec::new();
    if debug_routing {
        // ignore prefill in the routing display (generated tokens only)
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                d.cur.clear();
            }
        });
    }
    for i in 0..max_new {
        // --dry P: anti-repetition penalty on the tokens that would extend
        // an already-seen n-gram of the generation. Off by default: the
        // selection below is bit-identical when P == 0.
        let mut dry_logits;
        let sel_logits: &Vec<f32> = if sampler.dry > 0.0 {
            dry_logits = logits.clone();
            apply_dry(&mut dry_logits, &generated, sampler.dry);
            &dry_logits
        } else {
            &logits
        };
        let top = top_k_probs(sel_logits, 5);
        // temp <= 0: exact historical greedy path (argmax of the top-5);
        // temp > 0: top-p nucleus sampling over softmax(logits / temp).
        let (next_id, sampled_p) = if sampler.temp > 0.0 {
            sample_top_p(sel_logits, sampler.temp, sampler.top_p, &mut sampler.rng)
        } else {
            (top[0].0 as u32, top[0].1)
        };
        if debug {
            let candidats: Vec<String> = top
                .iter()
                .map(|&(tid, p)| format!("{} {:.1}%", py_repr(&tok.decode_id(tid as u32)), p * 100.0))
                .collect();
            println!();
            println!("token {:2} → {}", i + 1, py_repr(&tok.decode_id(next_id)));
            println!("  candidates: {}", candidats.join("  "));
            if sampler.temp > 0.0 {
                println!("  sampled: p = {:.1}% (temp = {}, top-p = {})", sampled_p * 100.0, sampler.temp, sampler.top_p);
            }
        }
        if next_id == stop_id {
            if debug {
                println!("  [end: stop token {}]", stop_id);
            }
            break;
        }
        let ta = Instant::now();
        logits = fwd(&[next_id]);
        logit_lens_print_maybe(tok, &format!("generated token {}", i + 1));
        let dt = ta.elapsed().as_secs_f64();
        gen_times.push(dt);
        generated.push(next_id);
        if debug_routing {
            ROUTING.with(|r| {
                if let Some(d) = r.borrow_mut().as_mut() {
                    let segs: Vec<String> = d
                        .cur
                        .iter()
                        .map(|(l, top3)| {
                            let exps: Vec<String> = top3
                                .iter()
                                .map(|(e, w)| format!("E{}({:.2})", e, w))
                                .collect();
                            format!("L{}: {}", l, exps.join(" "))
                        })
                        .collect();
                    println!("tok {} | {}", py_repr(&tok.decode_id(next_id)), segs.join(" | "));
                    d.cur.clear();
                }
            });
        }
        if debug {
            println!("  ⏱  {:.0} ms for this token", dt * 1000.0);
        }
    }

    if debug_routing {
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                let mut all: Vec<((usize, u32), u32)> = d.counts.iter().map(|(k, v)| (*k, *v)).collect();
                all.sort_by(|a, b| b.1.cmp(&a.1));
                println!();
                println!("Most-used experts of the run (top-10, top-16 appearances):");
                for ((l, e), n) in all.iter().take(10) {
                    println!("  L{} E{} : {}×", l, e, n);
                }
            }
        });
    }

    let answer = tok.decode(&generated);
    if debug {
        println!();
        println!("{}", "=".repeat(64));
        println!("SUMMARY");
        println!("{}", "=".repeat(64));
        println!("answer: {}", answer);
    } else {
        println!("Bot > {}", answer);
    }
    if !gen_times.is_empty() {
        let moy = gen_times.iter().sum::<f64>() / gen_times.len() as f64;
        if debug {
            println!("prefill: {:.2} s  |  generation: {:.0} ms/token average ({:.1} tok/s)",
                t_prefill.as_secs_f64(), moy * 1000.0, 1.0 / moy);
        } else {
            println!("  ({:.0} ms/token, {:.1} tok/s)", moy * 1000.0, 1.0 / moy);
        }
    }
    answer
}

#[cfg(test)]
mod dot_simd_tests {
    use super::{dot, dot_scalar};

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

    /// The dispatched dot() must be BIT-IDENTICAL to the scalar reference on
    /// every length (8-chunks, awkward remainders, degenerate cases).
    #[test]
    fn dot_simd_bit_exact() {
        let mut rng = Rng(0x1234567890ABCDEF);
        for n in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 100, 127, 128, 129, 1024, 1025, 4096, 16384, 16387] {
            let a: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let (want, got) = (dot_scalar(&a, &b), dot(&a, &b));
            assert_eq!(want.to_bits(), got.to_bits(), "bit mismatch at n={}", n);
        }
        // a few pathological values too (infinities, subnormals, zeros)
        for n in [8usize, 9, 64, 1000] {
            let a: Vec<f32> = (0..n)
                .map(|i| match i % 5 {
                    0 => 0.0,
                    1 => f32::MIN_POSITIVE,
                    2 => -f32::MIN_POSITIVE,
                    3 => 1e30,
                    _ => -1e-30,
                })
                .collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 1e-10).collect();
            let (want, got) = (dot_scalar(&a, &b), dot(&a, &b));
            assert_eq!(want.to_bits(), got.to_bits(), "bit mismatch (pathological) at n={}", n);
        }
    }

    /// gemm_batch (the batched-prefill GEMM) must be BIT-IDENTICAL to n
    /// separate matvec calls: every (row, position) dot keeps the same
    /// accumulation order in every kernel (dot, dot8t tiles, tail dots,
    /// pooled row split).
    #[test]
    fn gemm_matches_matvec_bit_exact() {
        use super::{dot, gemm_batch};
        let mut rng = Rng(0x0F0F0F0F0F0F0F0F);
        for (rows, cols, n) in [
            (5usize, 3usize, 1usize),
            (7, 8, 3),
            (16, 64, 4),
            (33, 96, 7),
            (64, 128, 8),
            (64, 130, 13), // awkward cols remainder
            (128, 512, 32), // exercises the pooled row split
        ] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
            let x: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
            let mut out = vec![0f32; n * rows];
            gemm_batch(&w, rows, cols, &x, n, &mut out);
            for t in 0..n {
                for r in 0..rows {
                    let want = dot(&w[r * cols..(r + 1) * cols], &x[t * cols..(t + 1) * cols]);
                    assert_eq!(
                        want.to_bits(),
                        out[t * rows + r].to_bits(),
                        "bit mismatch at (t={}, r={}) for {}x{}x{}",
                        t,
                        r,
                        rows,
                        cols,
                        n
                    );
                }
            }
        }
    }

    /// Integration: kda_prefill over n = 200 positions (chunked recurrence,
    /// n >= kda_chunk::MIN_LEN) vs n single-token kda_forward calls (the
    /// sequential step) on one synthetic KDA layer. The conv caches must be
    /// bit-identical (same sequential code path); the recurrence state and
    /// the layer outputs must match within the chunk transform tolerance.
    #[test]
    fn kda_prefill_chunked_matches_forward() {
        use super::{kda_forward, kda_prefill, KdaCache, KdaW, Prof, T};
        let cfg = crate::config::Config::microkimi();
        let (d, kp, fa, hn, kd) = (cfg.d, cfg.kda_proj(), cfg.kda_fa, cfg.kda_heads, cfg.kda_dim);
        let mut rng = Rng(0xDA7A_DA7A_DA7A_DA7A);
        // f32 weights laid out back to back in one byte buffer, KdaW field order
        let lens = [
            kp * d, // q_proj
            kp * d, // k_proj
            kp * d, // v_proj
            kp * cfg.kda_conv, // q_conv
            kp * cfg.kda_conv, // k_conv
            kp * cfg.kda_conv, // v_conv
            fa * d, // f_a
            kp * fa, // f_b
            kp,     // a_log
            kp,     // dt_bias
            hn * d, // b_proj
            kp * d, // g_proj
            kp,     // o_norm
            d * kp, // o_proj
        ];
        let mut buf: Vec<f32> = Vec::new();
        let mut offs: Vec<usize> = Vec::new();
        for &len in &lens {
            offs.push(buf.len() * 4);
            buf.extend((0..len).map(|_| rng.f32() * 0.1));
        }
        let data: Vec<u8> = buf.iter().flat_map(|f| f.to_le_bytes()).collect();
        let t = |i: usize| T { off: offs[i], len: lens[i] };
        let w = KdaW {
            q_proj: t(0),
            k_proj: t(1),
            v_proj: t(2),
            q_conv: t(3),
            k_conv: t(4),
            v_conv: t(5),
            f_a: t(6),
            f_b: t(7),
            a_log: t(8),
            dt_bias: t(9),
            b_proj: t(10),
            g_proj: t(11),
            o_norm: t(12),
            o_proj: t(13),
        };
        let new_cache = || KdaCache {
            conv_q: vec![0.0; 3 * kp],
            conv_k: vec![0.0; 3 * kp],
            conv_v: vec![0.0; 3 * kp],
            s: vec![0.0; hn * kd * kd],
        };
        let n = 200usize; // > kda_chunk::MIN_LEN, spans 4 chunks
        assert!(n >= crate::kda_chunk::MIN_LEN);
        let x: Vec<f32> = (0..n * d).map(|_| rng.f32()).collect();
        let mut prof = Prof::default();
        let (mut c_chk, mut c_seq) = (new_cache(), new_cache());
        let out_chk = kda_prefill(&cfg, &data, &w, &mut c_chk, &x, n, &mut prof);
        let mut out_seq = vec![0f32; n * d];
        for t in 0..n {
            let o = kda_forward(&cfg, &data, &w, &mut c_seq, &x[t * d..(t + 1) * d], &mut prof);
            out_seq[t * d..(t + 1) * d].copy_from_slice(&o);
        }
        // conv caches: identical sequential code in both paths
        for (a, b) in c_chk.conv_q.iter().zip(c_seq.conv_q.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_q not bit-identical");
        }
        for (a, b) in c_chk.conv_k.iter().zip(c_seq.conv_k.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_k not bit-identical");
        }
        for (a, b) in c_chk.conv_v.iter().zip(c_seq.conv_v.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_v not bit-identical");
        }
        let max_o = out_chk.iter().zip(out_seq.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let max_s = c_chk.s.iter().zip(c_seq.s.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("kda_prefill chunked vs forward: max|dO|={:.3e}  max|dS|={:.3e}", max_o, max_s);
        assert!(max_o < 1e-4, "layer output deviation {}", max_o);
        assert!(max_s < 1e-4, "recurrence state deviation {}", max_s);
    }
}

#[cfg(test)]
mod q8head_tests {
    use super::Q8Head;

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

    /// The q8 lm_head projection must track the f32 matvec: max error small
    /// relative to the logit span, and identical argmax (the greedy token).
    #[test]
    fn q8head_matches_f32() {
        let (rows, cols) = (512usize, 1024usize); // multi-job split (>60k MACs)
        let mut rng = Rng(0xC0FFEE);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32() * 0.02).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        let h = Q8Head::from_f32(&w, rows, cols);
        let mut got = vec![0f32; rows];
        let mut want = vec![0f32; rows];
        h.matvec(&x, &mut got);
        super::matvec_cpu(&w, rows, cols, &x, &mut want);
        let span = want.iter().map(|v| v.abs()).fold(0f32, f32::max) as f64;
        let max_err = got.iter().zip(&want).map(|(&a, &b)| (a as f64 - b as f64).abs()).fold(0f64, f64::max);
        assert!(max_err / span < 1e-2, "q8 head max err {} vs span {}", max_err, span);
        let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(am(&got), am(&want), "q8 head argmax differs from f32");
    }
}
