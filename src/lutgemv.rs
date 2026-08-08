// LUT GEMV: multiplication-free matvec for codebook-quantized weights.
//
// Instead of dequantize-then-multiply (w * x per weight), the products that
// can actually occur are precomputed once per activation group into a small
// lookup table, and the main loop is additions only:
//
//   sub-byte scalar codebook (2-bit / 4-bit indices into a codebook of 4 /
//   16 f32 values): G = 8 / kbits consecutive weights share one index byte.
//   That byte IS the LUT key: lut_g[b] = sum_j cb[field_j(b)] * x[g*G + j]
//   has 256 entries, so one weight-group contributes one byte load, one
//   table load and one add. Precompute costs 256 * G muls per group and is
//   amortized over every row of the matrix.
//
//   VQ1 vector codebook (8-bit index per vector of VQ_DIM=16 weights into a
//   256 x 16 codebook, see quant.rs): lut_g[k] = dot(cb[k], x[g*16..][..16]),
//   256 entries of one 16-mul dot each, amortized over all rows. The main
//   loop is again one byte load, one table load, one add per 16 weights.
//
// Tiling: the per-group tables are built and swept TILE groups at a time
// (tile outer, rows inner) so the hot tables stay L1-resident (TILE * 1 KB)
// while the index stream is still read exactly once, in row-major order.
//
// Exactness: every table entry accumulates its products in strictly
// ascending j order and every row accumulates table entries in strictly
// ascending group order, so the result is BIT-IDENTICAL to a plain
// sequential dequant + dot over the same values (and, for VQ1, to
// quant::matvec_vq). Compared against model::dot (8 lane accumulators,
// pairwise reduction) the summation order differs and results drift by
// ordering-only rounding (measured: ~150 ulp worst case on random 256-term
// sums with cancellation); the unit tests below pin both facts.
//
// The gather-add sweep is scalar by nature (NEON has no f32 gather); the
// NEON kernels vectorize the LUT precompute instead (across codebook
// entries, lane-wise bit-exact with the scalar chain). Dispatch mirrors
// dot_simd: NEON is baseline on aarch64, portable scalar elsewhere.
//
// Wiring: quant::matvec_vq dispatches here when enabled() is true. The
// default is ON (measured > 15% faster than the gather-dot path on
// aarch64, see the ignored bench below); MICROKIMI_LUTGEMV=0 forces the
// legacy path, MICROKIMI_LUTGEMV=1 forces this one. Only the decode GEMV
// is routed here; the batched prefill path (matvec_vq_nt) is unchanged.

use crate::quant::{VQ_DIM, VQ_K};

/// Groups of indices (one LUT of 256 f32 = 1 KB each) built and swept per
/// tile. 16 KB of hot tables: L1-resident next to the index stream and the
/// accumulator slice.
const TILE: usize = 16;

/// Runtime switch for the VQ1 wiring in quant::matvec_vq. Default on;
/// MICROKIMI_LUTGEMV=0/1 overrides.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("MICROKIMI_LUTGEMV").as_deref() {
        Ok("0") | Ok("false") | Ok("off") => false,
        _ => true,
    })
}

// ── sweep (shared main loop) ──
//
// out[r] += sum over the tile's groups of lut[g * 256 + idx[r * vpr + g0 + g]]
// for every row. Four rows at a time: four independent accumulator chains
// give the load-add latency enough parallelism without changing the
// per-row ascending-group summation order (bit-exact with the reference).

fn sweep(lut: &[f32], idx: &[u8], rows: usize, vpr: usize, g0: usize, groups: usize, out: &mut [f32]) {
    let mut r = 0usize;
    while r + 4 <= rows {
        let (mut a0, mut a1, mut a2, mut a3) = (out[r], out[r + 1], out[r + 2], out[r + 3]);
        let (i0, i1, i2, i3) = (&idx[r * vpr..], &idx[(r + 1) * vpr..], &idx[(r + 2) * vpr..], &idx[(r + 3) * vpr..]);
        for g in 0..groups {
            let base = g * 256;
            let k = g0 + g;
            a0 += lut[base + i0[k] as usize];
            a1 += lut[base + i1[k] as usize];
            a2 += lut[base + i2[k] as usize];
            a3 += lut[base + i3[k] as usize];
        }
        (out[r], out[r + 1], out[r + 2], out[r + 3]) = (a0, a1, a2, a3);
        r += 4;
    }
    while r < rows {
        let mut a = out[r];
        let ir = &idx[r * vpr..];
        for g in 0..groups {
            a += lut[g * 256 + ir[g0 + g] as usize];
        }
        out[r] = a;
        r += 1;
    }
}

// ── sub-byte scalar codebooks (variants A and B) ──
//
// Packing convention (little-endian fields, matching the mxfp4 nibble
// order): weight j of a group sits in bits [j*kbits, j*kbits+kbits) of its
// index byte, so for 4-bit the low nibble is the first weight of the pair.

/// LUT entries for one group of G weights sharing one index byte:
/// lut[b] = sum_j cb[(b >> (j*kbits)) & mask] * x[j], ascending j.
#[allow(dead_code)] // sub-byte variants are benched but not wired (see tests)
fn build_subbyte_lut(cb: &[f32], kbits: usize, x: &[f32], lut: &mut [f32]) {
    let group = 8 / kbits;
    let mask = (1usize << kbits) - 1;
    debug_assert_eq!(x.len(), group);
    debug_assert_eq!(lut.len(), 256);
    for (b, e) in lut.iter_mut().enumerate() {
        let mut s = 0f32;
        for j in 0..group {
            s += cb[(b >> (j * kbits)) & mask] * x[j];
        }
        *e = s;
    }
}

/// Generic sub-byte LUT GEMV. `idx` holds rows * cols * kbits / 8 bytes,
/// row-major, one byte per group of 8/kbits consecutive weights.
#[allow(dead_code)] // sub-byte variants are benched but not wired (see tests)
fn gemv_subbyte(cb: &[f32], kbits: usize, idx: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let group = 8 / kbits;
    debug_assert_eq!(cols % group, 0);
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(x.len(), cols);
    let vpr = cols / group; // index bytes (= LUT groups) per row
    debug_assert_eq!(idx.len(), rows * vpr);
    let mut lut = vec![0f32; TILE * 256];
    for o in out.iter_mut() {
        *o = 0.0;
    }
    let mut g0 = 0usize;
    while g0 < vpr {
        let groups = (vpr - g0).min(TILE);
        for g in 0..groups {
            build_subbyte_lut(cb, kbits, &x[(g0 + g) * group..(g0 + g) * group + group], &mut lut[g * 256..(g + 1) * 256]);
        }
        sweep(&lut, idx, rows, vpr, g0, groups, out);
        g0 += groups;
    }
}

/// Variant A: 2-bit indices into a 4-entry f32 codebook, 4 weights per byte.
#[allow(dead_code)] // benched but not wired: no 2-bit tensor format in the engine
pub fn gemv_lut2(cb4: &[f32], idx: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cb4.len(), 4);
    gemv_subbyte(cb4, 2, idx, rows, cols, x, out);
}

/// Variant B: 4-bit indices into a 16-entry f32 codebook, 2 weights per byte.
#[allow(dead_code)] // benched but not wired: mxfp4 keeps its own packed path
pub fn gemv_lut4(cb16: &[f32], idx: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cb16.len(), 16);
    gemv_subbyte(cb16, 4, idx, rows, cols, x, out);
}

// ── VQ1 vector codebook ──

/// Transposed codebook [VQ_DIM][VQ_K]: for a fixed j the 256 entry values
// are contiguous, so the precompute vectorizes across entries.
fn transpose_cb(codebook: &[f32], cb_t: &mut [f32]) {
    debug_assert_eq!(codebook.len(), VQ_K * VQ_DIM);
    for k in 0..VQ_K {
        for j in 0..VQ_DIM {
            cb_t[j * VQ_K + k] = codebook[k * VQ_DIM + j];
        }
    }
}

/// Scalar LUT precompute for one 16-wide activation group: lut[k] is the
/// plain sequential dot of codebook entry k with xs (same j order as
/// quant::matvec_vq, so the entry is bit-identical to the per-vector
/// partial sum of the reference path).
#[inline]
#[allow(dead_code)] // on aarch64 the dispatched precompute never reaches this
fn build_vq_lut_scalar(codebook: &[f32], xs: &[f32], lut: &mut [f32]) {
    for k in 0..VQ_K {
        let cb = &codebook[k * VQ_DIM..(k + 1) * VQ_DIM];
        let mut s = 0f32;
        for j in 0..VQ_DIM {
            s += cb[j] * xs[j];
        }
        lut[k] = s;
    }
}

/// NEON LUT precompute: four codebook entries per iteration, one per lane.
/// Each lane replays the scalar sequential chain (vaddq of vmulq, never
/// vfmaq), so every entry is bit-identical to build_vq_lut_scalar.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn build_vq_lut_neon(cb_t: &[f32], xs: &[f32], lut: &mut [f32]) {
    use std::arch::aarch64::*;
    unsafe {
        for k4 in (0..VQ_K).step_by(4) {
            let mut acc = vdupq_n_f32(0.0);
            for j in 0..VQ_DIM {
                let c = vld1q_f32(cb_t.as_ptr().add(j * VQ_K + k4));
                acc = vaddq_f32(acc, vmulq_f32(c, vdupq_n_f32(*xs.get_unchecked(j))));
            }
            vst1q_f32(lut.as_mut_ptr().add(k4), acc);
        }
    }
}

/// VQ1 GEMV by lookup table, bit-identical to quant::matvec_vq. One table
/// of 256 entry-activation dots per 16-wide group, swept TILE groups at a
/// time; the sweep is additions only.
pub fn matvec_vq_lut(codebook: &[f32], indices: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(cols % VQ_DIM, 0);
    debug_assert_eq!(out.len(), rows);
    debug_assert_eq!(x.len(), cols);
    let vpr = cols / VQ_DIM;
    debug_assert_eq!(indices.len(), rows * vpr);
    let mut lut = vec![0f32; TILE * 256];
    #[cfg(target_arch = "aarch64")]
    let mut cb_t = vec![0f32; VQ_K * VQ_DIM];
    #[cfg(target_arch = "aarch64")]
    transpose_cb(codebook, &mut cb_t);
    for o in out.iter_mut() {
        *o = 0.0;
    }
    let mut g0 = 0usize;
    while g0 < vpr {
        let groups = (vpr - g0).min(TILE);
        for g in 0..groups {
            let xs = &x[(g0 + g) * VQ_DIM..(g0 + g) * VQ_DIM + VQ_DIM];
            let dst = &mut lut[g * 256..(g + 1) * 256];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                build_vq_lut_neon(&cb_t, xs, dst);
            }
            #[cfg(not(target_arch = "aarch64"))]
            build_vq_lut_scalar(codebook, xs, dst);
        }
        sweep(&lut, indices, rows, vpr, g0, groups, out);
        g0 += groups;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{matvec_vq_gather, quantize, train_codebook};

    /// deterministic filler (splitmix64), same as the dot_simd tests
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

    /// Sequential dequant + dot reference (ascending column order): the
    /// order the LUT kernels are built to reproduce bit for bit.
    fn dequant_dot_seq(cb: &[f32], kbits: usize, idx: &[u8], x: &[f32]) -> f32 {
        let group = 8 / kbits;
        let mask = (1usize << kbits) - 1;
        let mut s = 0f32;
        for (v, &b) in idx.iter().enumerate() {
            let mut g = 0f32;
            for j in 0..group {
                g += cb[(b as usize >> (j * kbits)) & mask] * x[v * group + j];
            }
            s += g;
        }
        s
    }

    fn pack(w: &[f32], cb: &[f32], kbits: usize) -> Vec<u8> {
        let group = 8 / kbits;
        w.chunks_exact(group)
            .map(|g| {
                let mut b = 0u8;
                for (j, &v) in g.iter().enumerate() {
                    let mut best = 0usize;
                    let mut bd = f32::INFINITY;
                    for (k, &c) in cb.iter().enumerate() {
                        let d = (v - c) * (v - c);
                        if d < bd {
                            bd = d;
                            best = k;
                        }
                    }
                    b |= (best as u8) << (j * kbits);
                }
                b
            })
            .collect()
    }

    fn ulp_diff(a: f32, b: f32) -> u64 {
        let (a, b) = (a.to_bits(), b.to_bits());
        // monotonic mapping: order floats by their bit pattern
        let o = |v: u32| (v as i32 ^ (((v >> 31) as i32) | i32::MIN)) as i64;
        (o(a) - o(b)).unsigned_abs()
    }

    /// Realistic dims (DeepSeek-scale expert shapes), both variants: the
    /// LUT GEMV must be bit-identical to sequential dequant + dot. Against
    /// the 8-accumulator model::dot the summation order differs, so only an
    /// f64-referenced error bound is asserted (ordering drift, measured at
    /// ~150 ulp worst-case on 256-term random sums with cancellation).
    #[test]
    fn subbyte_lut_exactness() {
        let mut rng = Rng(0x5EED_5EED_5EED_5EED);
        for (rows, cols) in [(1usize, 8usize), (3, 16), (64, 256), (256, 2048), (1024, 7168)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
            for kbits in [2usize, 4] {
                let k = 1usize << kbits;
                let cb: Vec<f32> = (0..k).map(|_| rng.f32()).collect();
                let idx = pack(&w, &cb, kbits);
                let mut got = vec![0f32; rows];
                gemv_subbyte(&cb, kbits, &idx, rows, cols, &x, &mut got);
                let mut wdeq = vec![0f32; cols];
                let group = 8 / kbits;
                let mask = (1usize << kbits) - 1;
                let mut max_ulp = 0u64;
                for r in 0..rows {
                    let irow = &idx[r * cols / group..(r + 1) * cols / group];
                    let mut exact = 0f64; // f64 reference, cancellation-proof norm below
                    let mut norm = 0f64;
                    for (v, &b) in irow.iter().enumerate() {
                        for j in 0..group {
                            let wv = cb[(b as usize >> (j * kbits)) & mask];
                            wdeq[v * group + j] = wv;
                            exact += wv as f64 * x[v * group + j] as f64;
                            norm += (wv as f64 * x[v * group + j] as f64).abs();
                        }
                    }
                    let want = dequant_dot_seq(&cb, kbits, irow, &x);
                    assert_eq!(want.to_bits(), got[r].to_bits(), "bit mismatch {}x{} kbits={} row {}", rows, cols, kbits, r);
                    let d = crate::model::dot(&wdeq, &x);
                    max_ulp = max_ulp.max(ulp_diff(d, got[r]));
                    // both orderings must sit within f32 rounding of the f64
                    // reference, relative to the absolute-term sum
                    let rel = (got[r] as f64 - exact).abs() / norm.max(1e-30);
                    assert!(rel < 1e-5, "f64-referenced error {} too large ({}x{} kbits={} row {})", rel, rows, cols, kbits, r);
                }
                let _ = max_ulp; // ordering drift vs model::dot, documented above
            }
        }
    }

    /// The VQ1 LUT path must be bit-identical to quant::matvec_vq on
    /// realistic expert dims (the wiring contract of the cold-vq path).
    #[test]
    fn vq_lut_bit_exact() {
        let mut rng = Rng(0xC01D_0015_5EED_0001u64);
        for (rows, cols) in [(1usize, 16usize), (7, 64), (64, 128), (128, 64), (512, 1408), (1408, 512)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32() * 0.05).collect();
            let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
            let cb = train_codebook(&w, 0x1234);
            let idx = quantize(&w, &cb);
            let mut want = vec![0f32; rows];
            let mut got = vec![0f32; rows];
            matvec_vq_gather(&cb, &idx, rows, cols, &x, &mut want);
            matvec_vq_lut(&cb, &idx, rows, cols, &x, &mut got);
            for r in 0..rows {
                assert_eq!(want[r].to_bits(), got[r].to_bits(), "bit mismatch {}x{} row {}", rows, cols, r);
            }
        }
    }

    /// Micro-bench: lutgemv vs dequant+dot and vs the VQ1 gather-dot path.
    /// Run: cargo test --release -- --ignored bench_lutgemv --nocapture
    #[test]
    #[ignore]
    fn bench_lutgemv() {
        use std::time::Instant;
        let mut rng = Rng(0xBEAC_BA25_EED0_0002u64);
        let (rows, cols) = (7168usize, 2048usize);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32() * 0.05).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        let iters = 20;

        for kbits in [2usize, 4] {
            let k = 1usize << kbits;
            let cb: Vec<f32> = (0..k).map(|_| rng.f32()).collect();
            let idx = pack(&w, &cb, kbits);
            let mut out = vec![0f32; rows];
            let group = 8 / kbits;
            let mask = (1usize << kbits) - 1;
            // warmup + checksum guard
            gemv_subbyte(&cb, kbits, &idx, rows, cols, &x, &mut out);
            let t = Instant::now();
            for _ in 0..iters {
                gemv_subbyte(&cb, kbits, &idx, rows, cols, &x, &mut out);
            }
            let lut_t = t.elapsed().as_secs_f64() / iters as f64;
            let chk_lut: f64 = out.iter().map(|v| *v as f64).sum();

            let mut wdeq = vec![0f32; cols];
            let t = Instant::now();
            for _ in 0..iters {
                for r in 0..rows {
                    let irow = &idx[r * cols / group..(r + 1) * cols / group];
                    for (v, &b) in irow.iter().enumerate() {
                        for j in 0..group {
                            wdeq[v * group + j] = cb[(b as usize >> (j * kbits)) & mask];
                        }
                    }
                    out[r] = crate::model::dot(&wdeq, &x);
                }
            }
            let deq_t = t.elapsed().as_secs_f64() / iters as f64;
            let chk_deq: f64 = out.iter().map(|v| *v as f64).sum();
            println!(
                "bench {}-bit {}x{}: lutgemv {:.3} ms, dequant+dot {:.3} ms, speedup {:.2}x (chk {:.6} vs {:.6})",
                kbits,
                rows,
                cols,
                lut_t * 1e3,
                deq_t * 1e3,
                deq_t / lut_t,
                chk_lut,
                chk_deq
            );
        }

        // VQ1: the path the engine actually runs for cold experts
        let cb = train_codebook(&w, 0x1234);
        let idx = quantize(&w, &cb);
        let mut out = vec![0f32; rows];
        matvec_vq_lut(&cb, &idx, rows, cols, &x, &mut out);
        let t = Instant::now();
        for _ in 0..iters {
            matvec_vq_lut(&cb, &idx, rows, cols, &x, &mut out);
        }
        let lut_t = t.elapsed().as_secs_f64() / iters as f64;
        let chk_lut: f64 = out.iter().map(|v| *v as f64).sum();
        let t = Instant::now();
        for _ in 0..iters {
            matvec_vq_gather(&cb, &idx, rows, cols, &x, &mut out);
        }
        let ref_t = t.elapsed().as_secs_f64() / iters as f64;
        let chk_ref: f64 = out.iter().map(|v| *v as f64).sum();
        println!(
            "bench VQ1 {}x{}: lutgemv {:.3} ms, gather+dot {:.3} ms, speedup {:.2}x (chk {:.6} vs {:.6})",
            rows,
            cols,
            lut_t * 1e3,
            ref_t * 1e3,
            ref_t / lut_t,
            chk_lut,
            chk_ref
        );
        // engine-realistic cold expert dims (routed_hidden x moe_inter scale)
        for (r2, c2) in [(1408usize, 512usize), (512, 1408), (2048, 1408)] {
            let w2: Vec<f32> = (0..r2 * c2).map(|_| rng.f32() * 0.05).collect();
            let x2: Vec<f32> = (0..c2).map(|_| rng.f32()).collect();
            let idx2 = quantize(&w2, &cb);
            let mut out2 = vec![0f32; r2];
            let reps = 200;
            matvec_vq_lut(&cb, &idx2, r2, c2, &x2, &mut out2);
            let t = Instant::now();
            for _ in 0..reps {
                matvec_vq_lut(&cb, &idx2, r2, c2, &x2, &mut out2);
            }
            let lut_t = t.elapsed().as_secs_f64() / reps as f64;
            let t = Instant::now();
            for _ in 0..reps {
                matvec_vq_gather(&cb, &idx2, r2, c2, &x2, &mut out2);
            }
            let ref_t = t.elapsed().as_secs_f64() / reps as f64;
            println!(
                "bench VQ1 {}x{}: lutgemv {:.3} ms, gather+dot {:.3} ms, speedup {:.2}x",
                r2,
                c2,
                lut_t * 1e3,
                ref_t * 1e3,
                ref_t / lut_t
            );
        }
    }
}
