// Q8 activation path for the quantized matvecs (mxfp4 experts today, fp4/fp8
// spine later). The f32 activation is quantized on the fly to int8 per block
// of 32 (q8_0 convention: one f32 scale dx = max|x|/127 per block), then the
// dot product runs in INTEGER SIMD between the packed weights and the q8
// activation - the bulk of the work never touches f32.
//
// This path is NOT bit-identical to the f32 reference (int32-exact block
// sums vs f32 accumulation, plus the q8 rounding of x itself): that is the
// deal. The error is bounded by dx/2 per element and measured in
// tools::selftest::run_q8 (max rel << 1e-3). MICROKIMI_NO_Q8=1 restores the exact
// f32 path.
//
// Scale convention for mxfp4: the e2m1 LUT contains half-integers (0.5,
// 1.5), not representable in int8. We therefore use LUT2 = E2M1 * 2 (exact
// int8: 0,1,2,3,4,6,8,12 and negatives) and fold the compensating 0.5 into
// the weight block scale: W[c] = LUT2[nib] * 2^(sb-128). Per 32-block:
//
//   out_block = 2^(sb-128) * dx * <LUT2 block, q8 block>   (exact int32 dot)

/// LUT of the e2m1 values times 2 (see the convention above). Index 8 is
/// e2m1's negative zero, mapped to 0.
pub const E2M1_X2: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
/// E2M1_X2 + 12: the same values as unsigned bytes 0..24 (dpbusd's u8 operand).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub const E2M1_X2_P12: [u8; 16] = [12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0];

/// One q8_0-quantized activation vector: int8 values + one f32 scale per
/// block of 32.
pub struct Q8Vec {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
}

impl Q8Vec {
    pub fn new() -> Q8Vec {
        Q8Vec { q: Vec::new(), scales: Vec::new() }
    }
}

/// q8_0 quantization: per block of 32, dx = max|x|/127, q = round(x/dx)
/// clamped to [-127, 127] (so -128 never occurs - the AVX2 sign trick
/// relies on |q| <= 127).
pub fn quantize_q8(x: &[f32]) -> Q8Vec {
    let mut out = Q8Vec::new();
    quantize_q8_into(x, &mut out);
    out
}

/// quantize_q8 into a caller-owned buffer (the hot path reuses one
/// thread-local scratch: allocating per matvec call dominated the tiny
/// expert matvecs).
pub fn quantize_q8_into(x: &[f32], out: &mut Q8Vec) {
    assert!(x.len() % 32 == 0, "q8 blocks are 32 wide");
    let nb = x.len() / 32;
    out.q.clear();
    out.q.resize(x.len(), 0);
    out.scales.clear();
    out.scales.resize(nb, 0.0);
    let (q, scales) = (&mut out.q, &mut out.scales);
    for g in 0..nb {
        let b = &x[g * 32..(g + 1) * 32];
        let max = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let dx = max / 127.0;
        scales[g] = dx;
        if dx == 0.0 {
            continue; // all-zero block: q stays 0, scale 0 kills the block
        }
        for (j, &v) in b.iter().enumerate() {
            q[g * 32 + j] = (v / dx).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

// ── runtime toggle ──

// -1: follow the env (default, q8 ON unless MICROKIMI_NO_Q8=1);
// 0/1: forced off/on (selftest and dotbench A/B, not for production).
static FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

pub fn q8_enabled() -> bool {
    match FORCE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => std::env::var("MICROKIMI_NO_Q8").map(|v| v != "1").unwrap_or(true),
    }
}

/// Test/bench override of the q8 toggle (-1 restores env-driven behavior).
#[doc(hidden)]
pub fn force_q8(v: i8) {
    FORCE.store(v, std::sync::atomic::Ordering::Relaxed);
}

// ── integer block dot: 16 packed mxfp4 bytes (32 e2m1 nibbles) x 32 q8 bytes ──

// ── integer block dot: 32 q8 weights x 32 q8 activations (KV cache) ──

type I8Dot = unsafe fn(&[i8], &[i8]) -> i32;

/// The widest available i8 x i8 block-dot, resolved once per process.
/// Input: 32 q8 weight bytes and 32 q8 activation bytes (both |q| <= 127 by
/// the quantize_q8 convention). Output: the exact int32 dot. Used by the
/// q8_0 MLA KV cache (model.rs): the Q.K score of the latent part runs in
/// integer between the q8 cache row and the q8 query.
/// MICROKIMI_NO_SDOT=1 keeps the portable NEON kernels (A/B toggle; both
/// paths are bit-identical, only the instruction count differs).
#[cfg(target_arch = "aarch64")]
fn no_sdot() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_NO_SDOT").map(|v| v == "1").unwrap_or(false))
}

pub fn block_dot_i8(w32: &[i8], x32: &[i8]) -> i32 {
    static F: std::sync::OnceLock<I8Dot> = std::sync::OnceLock::new();
    fn pick() -> I8Dot {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return dot_i8_avx2;
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") && !no_sdot() {
                return dot_i8_sdot;
            }
            return dot_i8_neon;
        }
        #[cfg(not(target_arch = "aarch64"))]
        return dot_i8_scalar;
    }
    let f = F.get_or_init(pick);
    unsafe { f(w32, x32) }
}

#[allow(dead_code)] // on aarch64 the scalar kernel is test-only
fn dot_i8_scalar(w32: &[i8], x32: &[i8]) -> i32 {
    let mut s = 0i32;
    for j in 0..32 {
        s += w32[j] as i32 * x32[j] as i32;
    }
    s
}

/// Four-row fused int8 dot: the activation block is loaded ONCE per 32
/// columns and dotted against four consecutive weight rows through four
/// independent SDOT accumulator chains (better instruction-level
/// parallelism, a quarter of the activation loads). Integer sums are
/// exact, so the result equals four scalar dots.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(dead_code)] // reference kernel; the fused tiles superseded it
pub unsafe fn dot_i8_sdot4(
    w0: &[i8],
    w1: &[i8],
    w2: &[i8],
    w3: &[i8],
    x32: &[i8],
) -> [i32; 4] {
    use std::arch::aarch64::*;
    unsafe {
        let x0 = vld1q_s8(x32.as_ptr());
        let x1 = vld1q_s8(x32.as_ptr().add(16));
        let mut a0 = vdupq_n_s32(0);
        let mut a1 = vdupq_n_s32(0);
        let mut a2 = vdupq_n_s32(0);
        let mut a3 = vdupq_n_s32(0);
        let w00 = vld1q_s8(w0.as_ptr());
        let w01 = vld1q_s8(w0.as_ptr().add(16));
        let w10 = vld1q_s8(w1.as_ptr());
        let w11 = vld1q_s8(w1.as_ptr().add(16));
        let w20 = vld1q_s8(w2.as_ptr());
        let w21 = vld1q_s8(w2.as_ptr().add(16));
        let w30 = vld1q_s8(w3.as_ptr());
        let w31 = vld1q_s8(w3.as_ptr().add(16));
        std::arch::asm!(
            ".arch_extension dotprod",
            "sdot {a0:v}.4s, {w00:v}.16b, {x0:v}.16b",
            "sdot {a1:v}.4s, {w10:v}.16b, {x0:v}.16b",
            "sdot {a2:v}.4s, {w20:v}.16b, {x0:v}.16b",
            "sdot {a3:v}.4s, {w30:v}.16b, {x0:v}.16b",
            "sdot {a0:v}.4s, {w01:v}.16b, {x1:v}.16b",
            "sdot {a1:v}.4s, {w11:v}.16b, {x1:v}.16b",
            "sdot {a2:v}.4s, {w21:v}.16b, {x1:v}.16b",
            "sdot {a3:v}.4s, {w31:v}.16b, {x1:v}.16b",
            a0 = inout(vreg) a0,
            a1 = inout(vreg) a1,
            a2 = inout(vreg) a2,
            a3 = inout(vreg) a3,
            w00 = in(vreg) w00, w01 = in(vreg) w01,
            w10 = in(vreg) w10, w11 = in(vreg) w11,
            w20 = in(vreg) w20, w21 = in(vreg) w21,
            w30 = in(vreg) w30, w31 = in(vreg) w31,
            x0 = in(vreg) x0, x1 = in(vreg) x1,
            options(pure, nomem, nostack)
        );
        [vaddvq_s32(a0), vaddvq_s32(a1), vaddvq_s32(a2), vaddvq_s32(a3)]
    }
}

/// True when the four-row fused kernel may be used (dotprod present and
/// not disabled by MICROKIMI_NO_SDOT).
#[cfg(target_arch = "aarch64")]
pub fn sdot4_available() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod") && !no_sdot()
}

#[cfg(not(target_arch = "aarch64"))]
#[allow(dead_code)]
pub fn sdot4_available() -> bool {
    false
}

/// NEON dotprod path: one SDOT per 16 int8 lanes replaces the widening
/// multiply + pairwise accumulate pair. Integer sums are exact whatever
/// the accumulation shape, so this is bit-identical to the scalar
/// reference. vdotq_s32 is nightly-only in std::arch, but the instruction
/// itself is stable silicon: stable inline asm emits it directly, and the
/// dispatcher only selects this kernel when the CPU reports dotprod.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_sdot(w32: &[i8], x32: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let w0 = vld1q_s8(w32.as_ptr());
        let w1 = vld1q_s8(w32.as_ptr().add(16));
        let x0 = vld1q_s8(x32.as_ptr());
        let x1 = vld1q_s8(x32.as_ptr().add(16));
        let mut acc = vdupq_n_s32(0);
        std::arch::asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
            "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
            acc = inout(vreg) acc,
            w0 = in(vreg) w0,
            w1 = in(vreg) w1,
            x0 = in(vreg) x0,
            x1 = in(vreg) x1,
            options(pure, nomem, nostack)
        );
        vaddvq_s32(acc)
    }
}

/// NEON, portable integer path: widening multiply vmull_s8 + pairwise long
/// accumulate vpadalq_s16 (exact int32).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_i8_neon(w32: &[i8], x32: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let w0 = vld1q_s8(w32.as_ptr());
        let w1 = vld1q_s8(w32.as_ptr().add(16));
        let x0 = vld1q_s8(x32.as_ptr());
        let x1 = vld1q_s8(x32.as_ptr().add(16));
        let mut acc = vdupq_n_s32(0);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w0), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w1), vget_high_s8(x1)));
        vaddvq_s32(acc)
    }
}

/// AVX2: the classic maddubs+madd integer dot with the same sign trick as
/// dot_block_avx2 (both sides |q| <= 127, so |x| takes the unsigned side and
/// w is sign-flipped to match; pair sums <= 2 * 127 * 127 = 32258, no i16
/// saturation is possible).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_i8_avx2(w32: &[i8], x32: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        // Widen to i16 and use madd_epi16 (exact: |a*b| <= 128*128 and two
        // such products fit i32). The maddubs sign-trick saturates in i16
        // when both operands reach -128 (128*128*2 = 32768 > 32767), which
        // the exactness test caught on this architecture; production
        // quantizers emit -127..127, but the kernel must be exact anyway.
        let w = _mm256_loadu_si256(w32.as_ptr() as *const __m256i);
        let x = _mm256_loadu_si256(x32.as_ptr() as *const __m256i);
        let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
        let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
        let x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x));
        let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1));
        let quads = _mm256_add_epi32(_mm256_madd_epi16(w_lo, x_lo), _mm256_madd_epi16(w_hi, x_hi));
        // exact i32 horizontal sum
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(quads), _mm256_extracti128_si256(quads, 1));
        let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
        let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0xB1));
        _mm_cvtsi128_si32(s32)
    }
}

type BlockDot = unsafe fn(&[u8], &[i8]) -> i32;

/// The widest available integer block-dot, resolved once per process.
/// Input: 16 packed bytes (low nibble = even column) and the 32 matching
/// q8 activation bytes. Output: the exact int32 dot against LUT2.
/// Note: sdot/vdotq_s32 is still nightly-only in std::arch (feature
/// stdarch_neon_dotprod), so the NEON kernel uses vmull + vpadalq.
pub fn block_dot(packed16: &[u8], x32: &[i8]) -> i32 {
    static F: std::sync::OnceLock<BlockDot> = std::sync::OnceLock::new();
    fn pick() -> BlockDot {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return dot_block_avx2;
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") && !no_sdot() {
                return dot_block_sdot;
            }
            return dot_block_neon;
        }
        #[cfg(not(target_arch = "aarch64"))]
        return dot_block_scalar;
    }
    let f = F.get_or_init(pick);
    unsafe { f(packed16, x32) }
}

#[allow(dead_code)] // on aarch64 the scalar kernel is test-only
fn dot_block_scalar(packed16: &[u8], x32: &[i8]) -> i32 {
    let mut s = 0i32;
    for j in 0..16 {
        let b = packed16[j];
        s += E2M1_X2[(b & 0x0F) as usize] as i32 * x32[2 * j] as i32;
        s += E2M1_X2[(b >> 4) as usize] as i32 * x32[2 * j + 1] as i32;
    }
    s
}

/// NEON dotprod path for packed nibbles: same table-lookup decode as the
/// portable kernel, SDOT tail (see dot_i8_sdot for the asm rationale).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_block_sdot(packed16: &[u8], x32: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let lut = vld1q_s8(E2M1_X2.as_ptr());
        let b = vld1q_u8(packed16.as_ptr());
        let lo = vqtbl1q_s8(lut, vandq_u8(b, vdupq_n_u8(0x0F)));
        let hi = vqtbl1q_s8(lut, vshrq_n_u8::<4>(b));
        let w0 = vzip1q_s8(lo, hi);
        let w1 = vzip2q_s8(lo, hi);
        let x0 = vld1q_s8(x32.as_ptr());
        let x1 = vld1q_s8(x32.as_ptr().add(16));
        let mut acc = vdupq_n_s32(0);
        std::arch::asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
            "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
            acc = inout(vreg) acc,
            w0 = in(vreg) w0,
            w1 = in(vreg) w1,
            x0 = in(vreg) x0,
            x1 = in(vreg) x1,
            options(pure, nomem, nostack)
        );
        vaddvq_s32(acc)
    }
}

/// NEON, portable integer path: nibble decode via the 16-entry table lookup
/// (vqtbl1q), then widening multiply vmull_s8 + pairwise long accumulate
/// vpadalq_s16 (exact int32).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_block_neon(packed16: &[u8], x32: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    unsafe {
        let lut = vld1q_s8(E2M1_X2.as_ptr());
        let b = vld1q_u8(packed16.as_ptr());
        let lo = vqtbl1q_s8(lut, vandq_u8(b, vdupq_n_u8(0x0F)));
        let hi = vqtbl1q_s8(lut, vshrq_n_u8::<4>(b));
        // interleave to column order: w0 = cols 0..15, w1 = cols 16..31
        let w0 = vzip1q_s8(lo, hi);
        let w1 = vzip2q_s8(lo, hi);
        let x0 = vld1q_s8(x32.as_ptr());
        let x1 = vld1q_s8(x32.as_ptr().add(16));
        let mut acc = vdupq_n_s32(0);
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w0), vget_low_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w0), vget_high_s8(x0)));
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(w1), vget_low_s8(x1)));
        acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(w1), vget_high_s8(x1)));
        vaddvq_s32(acc)
    }
}

/// AVX2: nibble decode via in-lane shuffle (the 16-entry LUT is broadcast
/// to both 128-bit lanes), then the classic maddubs+madd integer dot.
/// maddubs is UNSIGNED x SIGNED, so the q8 activation (|q| <= 127, never
/// -128) takes the unsigned side: ax = |x| via sign_epi8(x, x), and the
/// weight side is sign-flipped to match: aw = w * sign(x). Pair sums stay
/// tiny (<= 2 * 127 * 12 = 3048), no i16 saturation is possible.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_block_avx2(packed16: &[u8], x32: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(E2M1_X2.as_ptr() as *const __m128i));
        let b = _mm256_broadcastsi128_si256(_mm_loadu_si128(packed16.as_ptr() as *const __m128i));
        let msk = _mm256_set1_epi8(0x0F);
        let andlo = _mm256_and_si256(b, msk);
        let andhi = _mm256_and_si256(_mm256_srli_epi16(b, 4), msk);
        let lo = _mm256_shuffle_epi8(lut, andlo);
        let hi = _mm256_shuffle_epi8(lut, andhi);
        // in-lane interleave, then take lane 0 of each: cols 0..15, 16..31
        let ilo = _mm256_unpacklo_epi8(lo, hi);
        let ihi = _mm256_unpackhi_epi8(lo, hi);
        let w = _mm256_permute2x128_si256(ilo, ihi, 0x20);
        let x = _mm256_loadu_si256(x32.as_ptr() as *const __m256i);
        let quads = i8_block_dot_vec(w, x);
        // exact i32 horizontal sum
        let s128 = _mm_add_epi32(_mm256_castsi256_si128(quads), _mm256_extracti128_si256(quads, 1));
        let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
        let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0xB1));
        _mm_cvtsi128_si32(s32)
    }
}

#[cfg(test)]
mod q8_tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn u8(&mut self) -> u8 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z ^ (z >> 31)) as u8
        }
    }

    /// The dispatched block dot must equal the scalar kernel EXACTLY
    /// (integer arithmetic: bit-exact by construction, verified anyway).
    #[test]
    fn block_dot_simd_equals_scalar() {
        let mut rng = Rng(0xDEADBEEF);
        for _ in 0..2000 {
            let packed: Vec<u8> = (0..16).map(|_| rng.u8()).collect();
            let x: Vec<i8> = (0..32).map(|_| (rng.u8() % 255) as i8).collect();
            assert_eq!(block_dot(&packed, &x), dot_block_scalar(&packed, &x));
        }
        // extremes: max magnitude weights and activation
        let packed = vec![0x77u8; 16]; // +6 and +6 (x2 = 12)
        let x = vec![127i8; 32];
        assert_eq!(block_dot(&packed, &x), dot_block_scalar(&packed, &x));
        let packed = vec![0xFFu8; 16]; // -6 and -6
        assert_eq!(block_dot(&packed, &x), dot_block_scalar(&packed, &x));
    }

    /// Every architecture's fp4 row kernel must equal the reduction-order
    /// reference bit for bit (integer block dots, fused four-lane order).
    #[test]
    fn row_dot_fp4_kernels_match_reference() {
        let mut rng = Rng(0xC0FFEE);
        for &nb in &[1usize, 3, 4, 7, 8, 13, 64, 65] {
            for _ in 0..40 {
                let prow: Vec<u8> = (0..nb * 16).map(|_| rng.u8()).collect();
                let srow: Vec<u8> = (0..nb).map(|_| 120 + rng.u8() % 16).collect();
                let q: Vec<i8> = (0..nb * 32).map(|_| (rng.u8() % 255) as i8).collect();
                let scales: Vec<f32> = (0..nb).map(|_| 0.001 + (rng.u8() as f32) / 300.0).collect();
                let xq = Q8Vec { q, scales };
                let want = unsafe { row_dot_fp4_generic(&prow, &srow, &xq) };
                let got = row_dot_fp4(&prow, &srow, &xq);
                assert_eq!(got.to_bits(), want.to_bits(), "nb={nb}: {got} vs {want}");
                #[cfg(target_arch = "x86_64")]
                {
                    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                        let v = unsafe { row_dot_fp4_x86(&prow, &srow, &xq) };
                        assert_eq!(v.to_bits(), want.to_bits(), "avx2 nb={nb}");
                    }
                    if vnni512_available() {
                        let v = unsafe { row_dot_fp4_vnni(&prow, &srow, &xq) };
                        assert_eq!(v.to_bits(), want.to_bits(), "vnni nb={nb}");
                        // four-row form: row 0 is this row, rows 1..3 shifted copies
                        let p1: Vec<u8> = prow.iter().rev().cloned().collect();
                        let s1: Vec<u8> = srow.iter().rev().cloned().collect();
                        let xs12 = xsum12(&xq);
                        let t = unsafe { rows4_dot_fp4_vnni([&prow, &p1, &prow, &p1], [&srow, &s1, &srow, &s1], &xq, &xs12) };
                        let want1 = unsafe { row_dot_fp4_generic(&p1, &s1, &xq) };
                        assert_eq!(t[0].to_bits(), want.to_bits(), "vnni4 r0 nb={nb}");
                        assert_eq!(t[1].to_bits(), want1.to_bits(), "vnni4 r1 nb={nb}");
                        assert_eq!(t[2].to_bits(), want.to_bits(), "vnni4 r2 nb={nb}");
                        assert_eq!(t[3].to_bits(), want1.to_bits(), "vnni4 r3 nb={nb}");
                    }
                }
            }
        }
    }

    /// The i8 x i8 KV dot must equal the scalar kernel EXACTLY (integer
    /// arithmetic, including the maddubs sign-trick extremes).
    #[test]
    fn block_dot_i8_simd_equals_scalar() {
        let mut rng = Rng(0x12345678);
        for _ in 0..2000 {
            let w: Vec<i8> = (0..32).map(|_| (rng.u8() % 255) as i8).collect();
            let x: Vec<i8> = (0..32).map(|_| (rng.u8() % 255) as i8).collect();
            assert_eq!(block_dot_i8(&w, &x), dot_i8_scalar(&w, &x));
        }
        // extremes: +-127 on both sides (the i16 saturation boundary)
        let w = vec![127i8; 32];
        let x = vec![127i8; 32];
        assert_eq!(block_dot_i8(&w, &x), dot_i8_scalar(&w, &x));
        let w = vec![-127i8; 32];
        assert_eq!(block_dot_i8(&w, &x), dot_i8_scalar(&w, &x));
        let x = vec![-127i8; 32];
        assert_eq!(block_dot_i8(&w, &x), dot_i8_scalar(&w, &x));
    }

    /// quantize_q8 conventions: scale = max|x|/127, values round-trip within
    /// dx/2, -128 never produced.
    #[test]
    fn quantize_conventions() {
        let mut x = vec![0f32; 64];
        x[3] = 12.7;
        x[40] = -3.3;
        let xq = quantize_q8(&x);
        assert_eq!(xq.scales[0], 12.7 / 127.0);
        assert_eq!(xq.scales[1], 3.3 / 127.0);
        assert_eq!(xq.q[3], 127);
        assert!(xq.q.iter().all(|&v| v >= -127));
        // all-zero block: zero scale, zero values
        let xq = quantize_q8(&vec![0f32; 32]);
        assert_eq!(xq.scales[0], 0.0);
        assert!(xq.q.iter().all(|&v| v == 0));
    }
}

// ── the shared MXFP4-row × q8-activation kernel ──

type RowDotFp4 = unsafe fn(&[u8], &[u8], &Q8Vec) -> f32;

/// One MXFP4 row against one q8 activation - THE row kernel behind every
/// q8-activation packed path (single-token decode, batch prefill lanes,
/// full colblock rows), so those paths stay bit-identical to each other.
/// Four blocks per round: nibble LUT decode + SDOT per block, then ONE
/// pairwise reduction and ONE vector FMA apply the four block sums - the
/// per-block horizontal add, scalar scale multiply and dispatch call of
/// the old shape are gone. The no-dotprod fallback mirrors the reduction
/// order exactly (integer block sums are exact on every kernel, the f32
/// lane math uses the same fused multiply-add and the same
/// ((l0+l1)+(l2+l3)) collapse), so the MICROKIMI_NO_SDOT toggle remains
/// bit-identical.
pub fn row_dot_fp4(prow: &[u8], srow: &[u8], xq: &Q8Vec) -> f32 {
    static F: std::sync::OnceLock<RowDotFp4> = std::sync::OnceLock::new();
    fn pick() -> RowDotFp4 {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") && !no_sdot() {
                return row_dot_fp4_sdot;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if vnni512_available() {
                return row_dot_fp4_vnni;
            }
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return row_dot_fp4_x86;
            }
        }
        row_dot_fp4_generic
    }
    let f = F.get_or_init(pick);
    // SAFETY: the sdot variant is only selected when dotprod is present.
    unsafe { f(prow, srow, xq) }
}

/// Reduction-order reference: 4 f32 lane accumulators, fused
/// multiply-add per block, pairwise collapse - the exact shape of the
/// vector kernel below.
unsafe fn row_dot_fp4_generic(prow: &[u8], srow: &[u8], xq: &Q8Vec) -> f32 {
    let nb = srow.len();
    let mut lanes = [0.0f32; 4];
    let mut g = 0usize;
    while g + 4 <= nb {
        for (b, lane) in lanes.iter_mut().enumerate() {
            let i = g + b;
            let idot = block_dot(&prow[i * 16..(i + 1) * 16], &xq.q[i * 32..(i + 1) * 32]);
            let s = crate::quant::mxfp4::exp2_i(srow[i] as i32 - 128) * xq.scales[i];
            *lane = (idot as f32).mul_add(s, *lane);
        }
        g += 4;
    }
    let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    while g < nb {
        let idot = block_dot(&prow[g * 16..(g + 1) * 16], &xq.q[g * 32..(g + 1) * 32]);
        total += idot as f32 * (crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128) * xq.scales[g]);
        g += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn row_dot_fp4_sdot(prow: &[u8], srow: &[u8], xq: &Q8Vec) -> f32 {
    use std::arch::aarch64::*;
    let nb = srow.len();
    // SAFETY: caller slices are exactly nb blocks wide (16 packed bytes,
    // 32 activation bytes, one scale byte and one f32 activation scale
    // per block); dotprod presence is guaranteed by the dispatcher.
    unsafe {
        let lut = vld1q_s8(E2M1_X2.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let mut accf = vdupq_n_f32(0.0);
        let mut g = 0usize;
        while g + 4 <= nb {
            let mut lanes = [vdupq_n_s32(0); 4];
            for (b, lane) in lanes.iter_mut().enumerate() {
                let i = g + b;
                let bytes = vld1q_u8(prow.as_ptr().add(i * 16));
                let lo = vqtbl1q_s8(lut, vandq_u8(bytes, mask));
                let hi = vqtbl1q_s8(lut, vshrq_n_u8::<4>(bytes));
                let w0 = vzip1q_s8(lo, hi);
                let w1 = vzip2q_s8(lo, hi);
                let x0 = vld1q_s8(xq.q.as_ptr().add(i * 32));
                let x1 = vld1q_s8(xq.q.as_ptr().add(i * 32 + 16));
                let mut acc = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension dotprod",
                    "sdot {acc:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {acc:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    acc = inout(vreg) acc,
                    w0 = in(vreg) w0,
                    w1 = in(vreg) w1,
                    x0 = in(vreg) x0,
                    x1 = in(vreg) x1,
                    options(pure, nomem, nostack)
                );
                *lane = acc;
            }
            // [sumA, sumB, sumC, sumD] in two pairwise adds
            let p01 = vpaddq_s32(lanes[0], lanes[1]);
            let p23 = vpaddq_s32(lanes[2], lanes[3]);
            let sums = vcvtq_f32_s32(vpaddq_s32(p01, p23));
            let ws = [
                crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128),
                crate::quant::mxfp4::exp2_i(srow[g + 1] as i32 - 128),
                crate::quant::mxfp4::exp2_i(srow[g + 2] as i32 - 128),
                crate::quant::mxfp4::exp2_i(srow[g + 3] as i32 - 128),
            ];
            let sv = vmulq_f32(vld1q_f32(ws.as_ptr()), vld1q_f32(xq.scales.as_ptr().add(g)));
            accf = vfmaq_f32(accf, sums, sv);
            g += 4;
        }
        let mut total = (vgetq_lane_f32::<0>(accf) + vgetq_lane_f32::<1>(accf))
            + (vgetq_lane_f32::<2>(accf) + vgetq_lane_f32::<3>(accf));
        while g < nb {
            let idot = block_dot(&prow[g * 16..(g + 1) * 16], &xq.q[g * 32..(g + 1) * 32]);
            total += idot as f32 * (crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128) * xq.scales[g]);
            g += 1;
        }
        total
    }
}

/// Four full rows against one q8 activation, block-fused: SDOT per
/// (row, half-block), the four rows' block sums collapse with two
/// pairwise adds, and ONE vector FMA per block applies the four row
/// scales times the activation scale - no horizontal reduction until
/// the very end, no per-row scalar scale chain. The f32 order per row
/// is fma(block_sum, wscale*xscale) sequentially over blocks, which
/// the scalar mirror in the callers reproduces exactly.
///
/// SAFETY: caller guarantees dotprod, `w[k]`/`s[k]` hold nb blocks
/// (32 i8 / one f32 each), and xq_q/xq_scales hold nb blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rows4_dot_fma(
    w: [&[i8]; 4],
    s: [&[f32]; 4],
    xq_q: &[i8],
    xq_scales: &[f32],
) -> [f32; 4] {
    use std::arch::aarch64::*;
    let nb = xq_scales.len();
    unsafe {
        let mut accf = vdupq_n_f32(0.0);
        for g in 0..nb {
            let x0 = vld1q_s8(xq_q.as_ptr().add(g * 32));
            let x1 = vld1q_s8(xq_q.as_ptr().add(g * 32 + 16));
            let mut a0 = vdupq_n_s32(0);
            let mut a1 = vdupq_n_s32(0);
            let mut a2 = vdupq_n_s32(0);
            let mut a3 = vdupq_n_s32(0);
            let w00 = vld1q_s8(w[0].as_ptr().add(g * 32));
            let w01 = vld1q_s8(w[0].as_ptr().add(g * 32 + 16));
            let w10 = vld1q_s8(w[1].as_ptr().add(g * 32));
            let w11 = vld1q_s8(w[1].as_ptr().add(g * 32 + 16));
            let w20 = vld1q_s8(w[2].as_ptr().add(g * 32));
            let w21 = vld1q_s8(w[2].as_ptr().add(g * 32 + 16));
            let w30 = vld1q_s8(w[3].as_ptr().add(g * 32));
            let w31 = vld1q_s8(w[3].as_ptr().add(g * 32 + 16));
            std::arch::asm!(
                ".arch_extension dotprod",
                "sdot {a0:v}.4s, {w00:v}.16b, {x0:v}.16b",
                "sdot {a1:v}.4s, {w10:v}.16b, {x0:v}.16b",
                "sdot {a2:v}.4s, {w20:v}.16b, {x0:v}.16b",
                "sdot {a3:v}.4s, {w30:v}.16b, {x0:v}.16b",
                "sdot {a0:v}.4s, {w01:v}.16b, {x1:v}.16b",
                "sdot {a1:v}.4s, {w11:v}.16b, {x1:v}.16b",
                "sdot {a2:v}.4s, {w21:v}.16b, {x1:v}.16b",
                "sdot {a3:v}.4s, {w31:v}.16b, {x1:v}.16b",
                a0 = inout(vreg) a0,
                a1 = inout(vreg) a1,
                a2 = inout(vreg) a2,
                a3 = inout(vreg) a3,
                w00 = in(vreg) w00, w01 = in(vreg) w01,
                w10 = in(vreg) w10, w11 = in(vreg) w11,
                w20 = in(vreg) w20, w21 = in(vreg) w21,
                w30 = in(vreg) w30, w31 = in(vreg) w31,
                x0 = in(vreg) x0, x1 = in(vreg) x1,
                options(pure, nomem, nostack)
            );
            // [row0, row1, row2, row3] block sums in two pairwise adds
            let p01 = vpaddq_s32(a0, a1);
            let p23 = vpaddq_s32(a2, a3);
            let sums = vcvtq_f32_s32(vpaddq_s32(p01, p23));
            let ws = [s[0][g], s[1][g], s[2][g], s[3][g]];
            let sv = vmulq_f32(vld1q_f32(ws.as_ptr()), vdupq_n_f32(xq_scales[g]));
            accf = vfmaq_f32(accf, sums, sv);
        }
        let mut out = [0.0f32; 4];
        vst1q_f32(out.as_mut_ptr(), accf);
        out
    }
}

/// One MXFP4 row against MANY q8 activations: the nibbles of each
/// 4-block group unpack ONCE and every lane's dot reuses them - the
/// unpack cost that dominates the per-lane path amortizes over the
/// whole batch. Per lane the accumulation replays row_dot_fp4 exactly
/// (same 4-block lane structure, same fused multiply-add, same
/// pairwise collapse, same tail), so a batched result is bit-identical
/// to the single-activation kernel on the same row.
pub fn row_dot_fp4_multi(prow: &[u8], srow: &[u8], xqs: &[&Q8Vec], out: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") && !no_sdot() {
            // SAFETY: dotprod present; slice widths checked by the callee.
            unsafe { return row_dot_fp4_multi_sdot(prow, srow, xqs, out) };
        }
    }
    for (l, xq) in xqs.iter().enumerate() {
        out[l] = row_dot_fp4(prow, srow, xq);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn row_dot_fp4_multi_sdot(prow: &[u8], srow: &[u8], xqs: &[&Q8Vec], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let nb = srow.len();
    let lanes = xqs.len();
    let n4 = nb / 4 * 4;
    // SAFETY: caller slices hold nb blocks; per-lane layout as in
    // row_dot_fp4_sdot.
    unsafe {
        let lut = vld1q_s8(E2M1_X2.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let mut accs = [vdupq_n_f32(0.0); 16];
        debug_assert!(lanes <= 16, "lane tile too wide");
        let mut g = 0usize;
        while g + 4 <= nb {
            // unpack the 4 blocks once (8 half-block weight vectors)
            let mut w = [vdupq_n_s8(0); 8];
            for b in 0..4 {
                let bytes = vld1q_u8(prow.as_ptr().add((g + b) * 16));
                let lo = vqtbl1q_s8(lut, vandq_u8(bytes, mask));
                let hi = vqtbl1q_s8(lut, vshrq_n_u8::<4>(bytes));
                w[b * 2] = vzip1q_s8(lo, hi);
                w[b * 2 + 1] = vzip2q_s8(lo, hi);
            }
            let ws = [
                crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128),
                crate::quant::mxfp4::exp2_i(srow[g + 1] as i32 - 128),
                crate::quant::mxfp4::exp2_i(srow[g + 2] as i32 - 128),
                crate::quant::mxfp4::exp2_i(srow[g + 3] as i32 - 128),
            ];
            let wsv = vld1q_f32(ws.as_ptr());
            for (l, xq) in xqs.iter().enumerate() {
                let mut b0 = vdupq_n_s32(0);
                let mut b1 = vdupq_n_s32(0);
                let mut b2 = vdupq_n_s32(0);
                let mut b3 = vdupq_n_s32(0);
                let xp = xq.q.as_ptr().add(g * 32);
                let x0 = vld1q_s8(xp);
                let x1 = vld1q_s8(xp.add(16));
                let x2 = vld1q_s8(xp.add(32));
                let x3 = vld1q_s8(xp.add(48));
                let x4 = vld1q_s8(xp.add(64));
                let x5 = vld1q_s8(xp.add(80));
                let x6 = vld1q_s8(xp.add(96));
                let x7 = vld1q_s8(xp.add(112));
                std::arch::asm!(
                    ".arch_extension dotprod",
                    "sdot {b0:v}.4s, {w0:v}.16b, {x0:v}.16b",
                    "sdot {b0:v}.4s, {w1:v}.16b, {x1:v}.16b",
                    "sdot {b1:v}.4s, {w2:v}.16b, {x2:v}.16b",
                    "sdot {b1:v}.4s, {w3:v}.16b, {x3:v}.16b",
                    "sdot {b2:v}.4s, {w4:v}.16b, {x4:v}.16b",
                    "sdot {b2:v}.4s, {w5:v}.16b, {x5:v}.16b",
                    "sdot {b3:v}.4s, {w6:v}.16b, {x6:v}.16b",
                    "sdot {b3:v}.4s, {w7:v}.16b, {x7:v}.16b",
                    b0 = inout(vreg) b0, b1 = inout(vreg) b1,
                    b2 = inout(vreg) b2, b3 = inout(vreg) b3,
                    w0 = in(vreg) w[0], w1 = in(vreg) w[1],
                    w2 = in(vreg) w[2], w3 = in(vreg) w[3],
                    w4 = in(vreg) w[4], w5 = in(vreg) w[5],
                    w6 = in(vreg) w[6], w7 = in(vreg) w[7],
                    x0 = in(vreg) x0, x1 = in(vreg) x1,
                    x2 = in(vreg) x2, x3 = in(vreg) x3,
                    x4 = in(vreg) x4, x5 = in(vreg) x5,
                    x6 = in(vreg) x6, x7 = in(vreg) x7,
                    options(pure, nomem, nostack)
                );
                let p01 = vpaddq_s32(b0, b1);
                let p23 = vpaddq_s32(b2, b3);
                let sums = vcvtq_f32_s32(vpaddq_s32(p01, p23));
                let sv = vmulq_f32(wsv, vld1q_f32(xq.scales.as_ptr().add(g)));
                accs[l] = vfmaq_f32(accs[l], sums, sv);
            }
            g += 4;
        }
        for (l, xq) in xqs.iter().enumerate() {
            let a = accs[l];
            let mut total = (vgetq_lane_f32::<0>(a) + vgetq_lane_f32::<1>(a))
                + (vgetq_lane_f32::<2>(a) + vgetq_lane_f32::<3>(a));
            let mut gg = n4;
            while gg < nb {
                let idot = block_dot(&prow[gg * 16..(gg + 1) * 16], &xq.q[gg * 32..(gg + 1) * 32]);
                total += idot as f32
                    * (crate::quant::mxfp4::exp2_i(srow[gg] as i32 - 128) * xq.scales[gg]);
                gg += 1;
            }
            out[l] = total;
        }
    }
}

/// Four rows x up to eight lanes, register-tiled: each block's eight weight
/// vectors load ONCE and serve all four activations. Per lane the
/// instruction sequence is rows4_dot_fma exactly (same sdot pairs, same
/// pairwise collapse, same scale-vector FMA, block-sequential), so each
/// lane's four results are bit-identical to the narrower kernels.
/// Returns [lane][row].
///
/// SAFETY: caller guarantees dotprod; every slice holds nb full blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rows4_dot_fma_x4(
    w: [&[i8]; 4],
    s: [&[f32]; 4],
    xqs: &[(&[i8], &[f32])],
) -> [[f32; 4]; 8] {
    use std::arch::aarch64::*;
    debug_assert!(xqs.len() <= 8, "lane tile is at most 8 wide");
    let nb = xqs[0].1.len();
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 8]; // per lane, rows in vector lanes
        for g in 0..nb {
            let w00 = vld1q_s8(w[0].as_ptr().add(g * 32));
            let w01 = vld1q_s8(w[0].as_ptr().add(g * 32 + 16));
            let w10 = vld1q_s8(w[1].as_ptr().add(g * 32));
            let w11 = vld1q_s8(w[1].as_ptr().add(g * 32 + 16));
            let w20 = vld1q_s8(w[2].as_ptr().add(g * 32));
            let w21 = vld1q_s8(w[2].as_ptr().add(g * 32 + 16));
            let w30 = vld1q_s8(w[3].as_ptr().add(g * 32));
            let w31 = vld1q_s8(w[3].as_ptr().add(g * 32 + 16));
            let ws = [s[0][g], s[1][g], s[2][g], s[3][g]];
            let wsv = vld1q_f32(ws.as_ptr());
            for (l, xq) in xqs.iter().enumerate() {
                let x0 = vld1q_s8(xq.0.as_ptr().add(g * 32));
                let x1 = vld1q_s8(xq.0.as_ptr().add(g * 32 + 16));
                let mut a0 = vdupq_n_s32(0);
                let mut a1 = vdupq_n_s32(0);
                let mut a2 = vdupq_n_s32(0);
                let mut a3 = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension dotprod",
                    "sdot {a0:v}.4s, {w00:v}.16b, {x0:v}.16b",
                    "sdot {a1:v}.4s, {w10:v}.16b, {x0:v}.16b",
                    "sdot {a2:v}.4s, {w20:v}.16b, {x0:v}.16b",
                    "sdot {a3:v}.4s, {w30:v}.16b, {x0:v}.16b",
                    "sdot {a0:v}.4s, {w01:v}.16b, {x1:v}.16b",
                    "sdot {a1:v}.4s, {w11:v}.16b, {x1:v}.16b",
                    "sdot {a2:v}.4s, {w21:v}.16b, {x1:v}.16b",
                    "sdot {a3:v}.4s, {w31:v}.16b, {x1:v}.16b",
                    a0 = inout(vreg) a0,
                    a1 = inout(vreg) a1,
                    a2 = inout(vreg) a2,
                    a3 = inout(vreg) a3,
                    w00 = in(vreg) w00, w01 = in(vreg) w01,
                    w10 = in(vreg) w10, w11 = in(vreg) w11,
                    w20 = in(vreg) w20, w21 = in(vreg) w21,
                    w30 = in(vreg) w30, w31 = in(vreg) w31,
                    x0 = in(vreg) x0, x1 = in(vreg) x1,
                    options(pure, nomem, nostack)
                );
                let p01 = vpaddq_s32(a0, a1);
                let p23 = vpaddq_s32(a2, a3);
                let sums = vcvtq_f32_s32(vpaddq_s32(p01, p23));
                let sv = vmulq_f32(wsv, vdupq_n_f32(xq.1[g]));
                acc[l] = vfmaq_f32(acc[l], sums, sv);
            }
        }
        let mut out = [[0.0f32; 4]; 8];
        for l in 0..xqs.len() {
            vst1q_f32(out[l].as_mut_ptr(), acc[l]);
        }
        out
    }
}


/// The packed-quad GEMM kernel: same math and reduction order as
/// rows4_dot_fma_x4, but the four rows' bytes are interleaved per block
/// (8 consecutive 16-byte vectors) and the four scales sit together -
/// one address stream instead of four plus a scalar gather. Layout per
/// (quad, block): [r0lo r0hi r1lo r1hi r2lo r2hi r3lo r3hi], scales
/// [quad][block][4].
///
/// SAFETY: caller guarantees dotprod; `wq` holds nb*128 bytes, `ws`
/// nb*4 scales, every lane nb blocks.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn rows4_dot_fma_x4_packed(
    wq: &[i8],
    ws: &[f32],
    xqs: &[(&[i8], &[f32])],
) -> [[f32; 4]; 16] {
    use std::arch::aarch64::*;
    debug_assert!(xqs.len() <= 16, "lane tile is at most 16 wide");
    let nb = xqs[0].1.len();
    unsafe {
        // 16 accumulators: one weight-quad load per block serves 16 lanes
        let mut acc = [vdupq_n_f32(0.0); 16];
        for g in 0..nb {
            let wp = wq.as_ptr().add(g * 128);
            let w00 = vld1q_s8(wp);
            let w01 = vld1q_s8(wp.add(16));
            let w10 = vld1q_s8(wp.add(32));
            let w11 = vld1q_s8(wp.add(48));
            let w20 = vld1q_s8(wp.add(64));
            let w21 = vld1q_s8(wp.add(80));
            let w30 = vld1q_s8(wp.add(96));
            let w31 = vld1q_s8(wp.add(112));
            let wsv = vld1q_f32(ws.as_ptr().add(g * 4));
            for (l, xq) in xqs.iter().enumerate() {
                let x0 = vld1q_s8(xq.0.as_ptr().add(g * 32));
                let x1 = vld1q_s8(xq.0.as_ptr().add(g * 32 + 16));
                let mut b0 = vdupq_n_s32(0);
                let mut b1 = vdupq_n_s32(0);
                let mut b2 = vdupq_n_s32(0);
                let mut b3 = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension dotprod",
                    "sdot {b0:v}.4s, {w00:v}.16b, {x0:v}.16b",
                    "sdot {b1:v}.4s, {w10:v}.16b, {x0:v}.16b",
                    "sdot {b2:v}.4s, {w20:v}.16b, {x0:v}.16b",
                    "sdot {b3:v}.4s, {w30:v}.16b, {x0:v}.16b",
                    "sdot {b0:v}.4s, {w01:v}.16b, {x1:v}.16b",
                    "sdot {b1:v}.4s, {w11:v}.16b, {x1:v}.16b",
                    "sdot {b2:v}.4s, {w21:v}.16b, {x1:v}.16b",
                    "sdot {b3:v}.4s, {w31:v}.16b, {x1:v}.16b",
                    b0 = inout(vreg) b0, b1 = inout(vreg) b1,
                    b2 = inout(vreg) b2, b3 = inout(vreg) b3,
                    w00 = in(vreg) w00, w01 = in(vreg) w01,
                    w10 = in(vreg) w10, w11 = in(vreg) w11,
                    w20 = in(vreg) w20, w21 = in(vreg) w21,
                    w30 = in(vreg) w30, w31 = in(vreg) w31,
                    x0 = in(vreg) x0, x1 = in(vreg) x1,
                    options(pure, nomem, nostack)
                );
                let p01 = vpaddq_s32(b0, b1);
                let p23 = vpaddq_s32(b2, b3);
                let sums = vcvtq_f32_s32(vpaddq_s32(p01, p23));
                let sv = vmulq_f32(wsv, vdupq_n_f32(xq.1[g]));
                acc[l] = vfmaq_f32(acc[l], sums, sv);
            }
        }
        let mut out = [[0.0f32; 4]; 16];
        for l in 0..xqs.len() {
            vst1q_f32(out[l].as_mut_ptr(), acc[l]);
        }
        out
    }
}

/// True when the i8mm extension (SMMLA) is available and not disabled.
pub fn smmla_available() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| {
            std::arch::is_aarch64_feature_detected!("i8mm")
                && !no_sdot()
                && std::env::var("MICROKIMI_NO_SMMLA").map(|v| v != "1").unwrap_or(true)
        });
    }
    #[cfg(not(target_arch = "aarch64"))]
    false
}

/// Four rows x up to sixteen lanes through SMMLA (i8mm): one instruction
/// multiplies a 2x8 row-pair block by an 8x2 lane-pair block - 32 MACs,
/// twice SDOT's throughput. Operands come PAIR-INTERLEAVED:
///   weights `wp`, per block: [r0[0..8] r1[0..8] r0[8..16] r1[8..16]
///   ... r2/r3 likewise] (128 bytes per (quad, block));
///   activations `xp`, TILE-major: for the tile's up to 8 pairs, block g
///   holds the pairs contiguously at ((g*8)+pair)*64, each pair
///   [lA[0..8] lB[0..8] lA[8..16] ...] (64 bytes) - one address stream
///   per block for the whole 16-lane tile.
/// Scales apply per block with one fused multiply-add per accumulator
/// element in block order - the same per-(row, lane) arithmetic as
/// rows4_dot_fma, so results are bit-identical to the SDOT kernels.
///
/// SAFETY: caller guarantees i8mm; `wp` holds nb*128 bytes, `ws` nb*4
/// scales, `xp` holds `pairs`*nb*64 bytes, `xs` per-lane nb scales.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rows4_x8_smmla(
    wp: &[i8],
    ws: &[f32],
    xp: &[i8],
    xs: &[&[f32]],
    pairs: usize,
    nb: usize,
    out: &mut [[f32; 4]; 16],
) {
    use std::arch::aarch64::*;
    debug_assert!(pairs <= 8, "tile is at most 16 lanes (8 pairs)");
    unsafe {
        // acc[pair][0] = rows(0,1) x lanes(A,B) tile, acc[pair][1] = rows(2,3);
        // 8 pairs = 16 accumulators, one weight-block load per 16 lanes
        let mut acc = [[vdupq_n_f32(0.0); 2]; 8];
        for g in 0..nb {
            let wb = wp.as_ptr().add(g * 128);
            let a01_0 = vld1q_s8(wb);
            let a01_1 = vld1q_s8(wb.add(16));
            let a01_2 = vld1q_s8(wb.add(32));
            let a01_3 = vld1q_s8(wb.add(48));
            let a23_0 = vld1q_s8(wb.add(64));
            let a23_1 = vld1q_s8(wb.add(80));
            let a23_2 = vld1q_s8(wb.add(96));
            let a23_3 = vld1q_s8(wb.add(112));
            let wsv = [ws[g * 4], ws[g * 4 + 1], ws[g * 4 + 2], ws[g * 4 + 3]];
            for p in 0..pairs {
                // tile-major layout: the tile's pairs of one block sit
                // contiguous (one address stream per block for 16 lanes)
                let xb = xp.as_ptr().add((g * 8 + p) * 64);
                let b0 = vld1q_s8(xb);
                let b1 = vld1q_s8(xb.add(16));
                let b2 = vld1q_s8(xb.add(32));
                let b3 = vld1q_s8(xb.add(48));
                let mut t01 = vdupq_n_s32(0);
                let mut t23 = vdupq_n_s32(0);
                std::arch::asm!(
                    ".arch_extension i8mm",
                    "smmla {t01:v}.4s, {a0:v}.16b, {b0:v}.16b",
                    "smmla {t01:v}.4s, {a1:v}.16b, {b1:v}.16b",
                    "smmla {t01:v}.4s, {a2:v}.16b, {b2:v}.16b",
                    "smmla {t01:v}.4s, {a3:v}.16b, {b3:v}.16b",
                    "smmla {t23:v}.4s, {a4:v}.16b, {b0:v}.16b",
                    "smmla {t23:v}.4s, {a5:v}.16b, {b1:v}.16b",
                    "smmla {t23:v}.4s, {a6:v}.16b, {b2:v}.16b",
                    "smmla {t23:v}.4s, {a7:v}.16b, {b3:v}.16b",
                    t01 = inout(vreg) t01,
                    t23 = inout(vreg) t23,
                    a0 = in(vreg) a01_0, a1 = in(vreg) a01_1,
                    a2 = in(vreg) a01_2, a3 = in(vreg) a01_3,
                    a4 = in(vreg) a23_0, a5 = in(vreg) a23_1,
                    a6 = in(vreg) a23_2, a7 = in(vreg) a23_3,
                    b0 = in(vreg) b0, b1 = in(vreg) b1,
                    b2 = in(vreg) b2, b3 = in(vreg) b3,
                    options(pure, nomem, nostack)
                );
                // tile layout [C(r0,lA) C(r0,lB) C(r1,lA) C(r1,lB)]
                let xa = xs[p * 2][g];
                let xbs = xs[p * 2 + 1][g];
                let s01 = [wsv[0] * xa, wsv[0] * xbs, wsv[1] * xa, wsv[1] * xbs];
                let s23 = [wsv[2] * xa, wsv[2] * xbs, wsv[3] * xa, wsv[3] * xbs];
                acc[p][0] = vfmaq_f32(acc[p][0], vcvtq_f32_s32(t01), vld1q_f32(s01.as_ptr()));
                acc[p][1] = vfmaq_f32(acc[p][1], vcvtq_f32_s32(t23), vld1q_f32(s23.as_ptr()));
            }
        }
        for p in 0..pairs {
            let mut t = [0.0f32; 4];
            vst1q_f32(t.as_mut_ptr(), acc[p][0]);
            out[p * 2][0] = t[0];
            out[p * 2 + 1][0] = t[1];
            out[p * 2][1] = t[2];
            out[p * 2 + 1][1] = t[3];
            vst1q_f32(t.as_mut_ptr(), acc[p][1]);
            out[p * 2][2] = t[0];
            out[p * 2 + 1][2] = t[1];
            out[p * 2][3] = t[2];
            out[p * 2 + 1][3] = t[3];
        }
    }
}

// ── x86: whole-row and four-row q8 kernels (AVX2, VNNI when present) ──
//
// The per-block dispatch + horizontal reduction that made the ARM path
// slow before the fused kernels was still the ONLY x86 path: every
// 32-column block paid an indirect call and a five-instruction
// reduction, and none of the multi-row tiles ran (they are gated on
// SDOT). These are their x86 counterparts: one vector accumulator per
// row across the whole row, one reduction at the end, the per-block
// scale folded as a fused multiply-add in the same block-sequential
// order as rows4_dot_fma - so results are bit-identical to the ARM
// kernels and to the scalar reference.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn i8_block_dot_vec(w: std::arch::x86_64::__m256i, x: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // exact widening form (see dot_i8_avx2)
    let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
    let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
    let x_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(x));
    let x_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(x, 1));
    _mm256_add_epi32(_mm256_madd_epi16(w_lo, x_lo), _mm256_madd_epi16(w_hi, x_hi))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let s128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let s64 = _mm_add_epi32(s128, _mm_shuffle_epi32(s128, 0x4E));
    let s32 = _mm_add_epi32(s64, _mm_shuffle_epi32(s64, 0xB1));
    _mm_cvtsi128_si32(s32)
}

/// Four rows x N lanes on x86: per block, the four rows' i8 words load
/// once and each lane's block dot is an AVX2 maddubs/madd pair; the
/// four block sums collapse to a __m128 and one FMA applies the four
/// row scales times the lane scale - the exact rows4_dot_fma order.
/// Returns [lane][row].
///
/// SAFETY: caller guarantees avx2/fma; slices hold nb blocks each.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn rows4_dot_fma_x86(
    w: [&[i8]; 4],
    s: [&[f32]; 4],
    xqs: &[(&[i8], &[f32])],
) -> [[f32; 4]; 16] {
    use std::arch::x86_64::*;
    let nb = xqs[0].1.len();
    let mut out = [[0.0f32; 4]; 16];
    unsafe {
        let mut acc = [_mm_setzero_ps(); 16];
        for g in 0..nb {
            let w0 = _mm256_loadu_si256(w[0].as_ptr().add(g * 32) as *const __m256i);
            let w1 = _mm256_loadu_si256(w[1].as_ptr().add(g * 32) as *const __m256i);
            let w2 = _mm256_loadu_si256(w[2].as_ptr().add(g * 32) as *const __m256i);
            let w3 = _mm256_loadu_si256(w[3].as_ptr().add(g * 32) as *const __m256i);
            let ws = _mm_set_ps(s[3][g], s[2][g], s[1][g], s[0][g]);
            for (l, xq) in xqs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.0.as_ptr().add(g * 32) as *const __m256i);
                let d0 = hsum_i32(i8_block_dot_vec(w0, x));
                let d1 = hsum_i32(i8_block_dot_vec(w1, x));
                let d2 = hsum_i32(i8_block_dot_vec(w2, x));
                let d3 = hsum_i32(i8_block_dot_vec(w3, x));
                let sums = _mm_cvtepi32_ps(_mm_set_epi32(d3, d2, d1, d0));
                let sv = _mm_mul_ps(ws, _mm_set1_ps(xq.1[g]));
                acc[l] = _mm_fmadd_ps(sums, sv, acc[l]);
            }
        }
        for l in 0..xqs.len() {
            _mm_storeu_ps(out[l].as_mut_ptr(), acc[l]);
        }
    }
    out
}

/// True on x86 with AVX2+FMA (the four-row tiles' gate there).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub fn x86_tiles_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"));
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// x86 whole-row fp4-x-q8 kernel: block nibbles decode with the shuffle
/// LUT, dots through maddubs/madd, and the block sums stay VECTOR (one
/// i32 lane group per block) with the scale folded per block through a
/// scalar-free path: we accumulate f32 lanes of (block_sum * scale) in
/// groups of four blocks - the same 4-lane structure and pairwise
/// collapse as row_dot_fp4_generic, so it is bit-identical to it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn row_dot_fp4_x86(prow: &[u8], srow: &[u8], xq: &Q8Vec) -> f32 {
    use std::arch::x86_64::*;
    let nb = srow.len();
    unsafe {
        let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(E2M1_X2.as_ptr() as *const __m128i));
        let msk = _mm256_set1_epi8(0x0F);
        let mut lanes = [0.0f32; 4];
        let mut g = 0usize;
        while g + 4 <= nb {
            for (b, lane) in lanes.iter_mut().enumerate() {
                let i = g + b;
                let bytes = _mm256_broadcastsi128_si256(_mm_loadu_si128(prow.as_ptr().add(i * 16) as *const __m128i));
                let lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(bytes, msk));
                let hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(bytes, 4), msk));
                let ilo = _mm256_unpacklo_epi8(lo, hi);
                let ihi = _mm256_unpackhi_epi8(lo, hi);
                let w = _mm256_permute2x128_si256(ilo, ihi, 0x20);
                let x = _mm256_loadu_si256(xq.q.as_ptr().add(i * 32) as *const __m256i);
                let idot = hsum_i32(i8_block_dot_vec(w, x));
                let s = crate::quant::mxfp4::exp2_i(srow[i] as i32 - 128) * xq.scales[i];
                *lane = (idot as f32).mul_add(s, *lane);
            }
            g += 4;
        }
        let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        while g < nb {
            let idot = block_dot(&prow[g * 16..(g + 1) * 16], &xq.q[g * 32..(g + 1) * 32]);
            total += idot as f32 * (crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128) * xq.scales[g]);
            g += 1;
        }
        total
    }
}

/// True with AVX-512 VNNI (Cascade Lake and later): vpdpbusd on 512-bit
/// vectors, 64 int8 MACs per instruction.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub fn vnni512_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        return *ON.get_or_init(|| {
            is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
                && is_x86_feature_detected!("avx512vnni")
                && std::env::var("MICROKIMI_NO_VNNI").map(|v| v != "1").unwrap_or(true)
        });
    }
    #[cfg(not(target_arch = "x86_64"))]
    false
}

/// Four rows x N lanes with AVX-512 VNNI: two 32-column blocks per
/// 512-bit vector, one vpdpbusd per (row, lane, 64 columns). VNNI is
/// unsigned x signed, so the row (weights) provides the signed operand
/// and |x| the unsigned one, with sign(x) folded into w as the AVX2 path
/// does (sign_epi8): exact integer sums, identical to every other q8
/// kernel. Block sums are extracted per 32-column half so the per-block
/// scale FMA keeps the shared block-sequential order - bit-identical.
///
/// SAFETY: caller guarantees avx512f/bw/vnni; slices hold nb blocks,
/// nb even (caller routes odd tails to the AVX2 tile).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
pub unsafe fn rows4_dot_fma_vnni(
    w: [&[i8]; 4],
    s: [&[f32]; 4],
    xqs: &[(&[i8], &[f32])],
) -> [[f32; 4]; 16] {
    use std::arch::x86_64::*;
    let nb = xqs[0].1.len();
    let mut out = [[0.0f32; 4]; 16];
    unsafe {
        let mut acc = [_mm_setzero_ps(); 16];
        let mut g = 0usize;
        while g + 2 <= nb {
            // 64 columns = blocks g, g+1
            let w0 = _mm512_loadu_si512(w[0].as_ptr().add(g * 32) as *const _);
            let w1 = _mm512_loadu_si512(w[1].as_ptr().add(g * 32) as *const _);
            let w2 = _mm512_loadu_si512(w[2].as_ptr().add(g * 32) as *const _);
            let w3 = _mm512_loadu_si512(w[3].as_ptr().add(g * 32) as *const _);
            let ws_a = _mm_set_ps(s[3][g], s[2][g], s[1][g], s[0][g]);
            let ws_b = _mm_set_ps(s[3][g + 1], s[2][g + 1], s[1][g + 1], s[0][g + 1]);
            for (l, xq) in xqs.iter().enumerate() {
                let x = _mm512_loadu_si512(xq.0.as_ptr().add(g * 32) as *const _);
                // exact unsigned x signed: u = x + 128 (0..255), then
                // dot(u, w) - 128 * sum(w) = dot(x, w); the correction is a
                // dpbusd against a constant 128 vector - no saturation, no
                // sign folding, exact for the full i8 range.
                let ux = _mm512_xor_si512(x, _mm512_set1_epi8(-128)); // x + 128 as u8
                let c128 = _mm512_set1_epi8(-128); // 128 as u8
                let d = |wv: __m512i| -> __m512i {
                    let a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ux, wv);
                    let b = _mm512_dpbusd_epi32(_mm512_setzero_si512(), c128, wv);
                    _mm512_sub_epi32(a, b)
                };
                let d0 = d(w0);
                let d1 = d(w1);
                let d2 = d(w2);
                let d3 = d(w3);
                // per-row: sum of low 8 i32 lanes = block g, high 8 = block g+1
                let half = |v: __m512i| -> (i32, i32) {
                    let lo = _mm512_castsi512_si256(v);
                    let hi = _mm512_extracti64x4_epi64(v, 1);
                    (hsum_i32(lo), hsum_i32(hi))
                };
                let (a0, b0) = half(d0);
                let (a1, b1) = half(d1);
                let (a2, b2) = half(d2);
                let (a3, b3) = half(d3);
                let sums_a = _mm_cvtepi32_ps(_mm_set_epi32(a3, a2, a1, a0));
                let sums_b = _mm_cvtepi32_ps(_mm_set_epi32(b3, b2, b1, b0));
                let sv_a = _mm_mul_ps(ws_a, _mm_set1_ps(xq.1[g]));
                let sv_b = _mm_mul_ps(ws_b, _mm_set1_ps(xq.1[g + 1]));
                acc[l] = _mm_fmadd_ps(sums_a, sv_a, acc[l]);
                acc[l] = _mm_fmadd_ps(sums_b, sv_b, acc[l]);
            }
            g += 2;
        }
        // odd tail block through the AVX2 pair
        if g < nb {
            let w0 = _mm256_loadu_si256(w[0].as_ptr().add(g * 32) as *const __m256i);
            let w1 = _mm256_loadu_si256(w[1].as_ptr().add(g * 32) as *const __m256i);
            let w2 = _mm256_loadu_si256(w[2].as_ptr().add(g * 32) as *const __m256i);
            let w3 = _mm256_loadu_si256(w[3].as_ptr().add(g * 32) as *const __m256i);
            let ws = _mm_set_ps(s[3][g], s[2][g], s[1][g], s[0][g]);
            for (l, xq) in xqs.iter().enumerate() {
                let x = _mm256_loadu_si256(xq.0.as_ptr().add(g * 32) as *const __m256i);
                let d0 = hsum_i32(i8_block_dot_vec(w0, x));
                let d1 = hsum_i32(i8_block_dot_vec(w1, x));
                let d2 = hsum_i32(i8_block_dot_vec(w2, x));
                let d3 = hsum_i32(i8_block_dot_vec(w3, x));
                let sums = _mm_cvtepi32_ps(_mm_set_epi32(d3, d2, d1, d0));
                let sv = _mm_mul_ps(ws, _mm_set1_ps(xq.1[g]));
                acc[l] = _mm_fmadd_ps(sums, sv, acc[l]);
            }
        }
        for l in 0..xqs.len() {
            _mm_storeu_ps(out[l].as_mut_ptr(), acc[l]);
        }
    }
    out
}

/// One MXFP4 row against a q8 activation with AVX-512 (BW + VNNI): four
/// 32-column blocks per iteration. The 64 packed bytes decode through
/// the LUT2 shuffle in all four 128-bit lanes at once, each lane one
/// block; per lane the exact u8 x s8 form (dot(x + 128, w) - 128 sum(w))
/// gives the block's integer dot in four i32 partials, reduced inside
/// the lane (exact), converted and FMA'd against 2^(e-128) * dx as the
/// scalar kernel does. Bit-identical to row_dot_fp4_x86 / _generic:
/// same integer block dots, same fused four-lane f32 order.
///
/// SAFETY: caller guarantees avx512f/bw/vnni; prow has 16*nb bytes,
/// srow nb bytes, xq nb blocks.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
unsafe fn row_dot_fp4_vnni(prow: &[u8], srow: &[u8], xq: &Q8Vec) -> f32 {
    use std::arch::x86_64::*;
    let nb = srow.len();
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(E2M1_X2.as_ptr() as *const __m128i));
        let msk = _mm512_set1_epi8(0x0F);
        let c128 = _mm512_set1_epi8(-128);
        let mut lanes = [0.0f32; 4];
        let mut g = 0usize;
        while g + 4 <= nb {
            // 64 bytes = 4 blocks of 16 packed bytes: lane k = block g+k
            let bytes = _mm512_loadu_si512(prow.as_ptr().add(g * 16) as *const _);
            let lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(bytes, msk));
            let hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), msk));
            // per lane: bytes 0..16 = even columns' first 16 / odd ... interleave
            let ilo = _mm512_unpacklo_epi8(lo, hi); // per lane: columns 0..15 of the block
            let ihi = _mm512_unpackhi_epi8(lo, hi); // per lane: columns 16..31
            // activation blocks g..g+4: 128 bytes; regroup so that lane k
            // holds block g+k's first 16 (xa) / last 16 (xb) columns
            let x0 = _mm512_loadu_si512(xq.q.as_ptr().add(g * 32) as *const _); // blocks g, g+1
            let x1 = _mm512_loadu_si512(xq.q.as_ptr().add(g * 32 + 64) as *const _); // blocks g+2, g+3
            // x0 lanes: [g:0-15][g:16-31][g+1:0-15][g+1:16-31]; x1 same for g+2,g+3
            let xa = _mm512_permutex2var_epi64(x0, _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0), x1);
            let xb = _mm512_permutex2var_epi64(x0, _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2), x1);
            let uxa = _mm512_xor_si512(xa, c128);
            let uxb = _mm512_xor_si512(xb, c128);
            let d = _mm512_add_epi32(
                _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), uxa, ilo), _mm512_dpbusd_epi32(_mm512_setzero_si512(), c128, ilo)),
                _mm512_sub_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), uxb, ihi), _mm512_dpbusd_epi32(_mm512_setzero_si512(), c128, ihi)),
            );
            // reduce the four i32 partials inside each 128-bit lane (exact)
            let d2 = _mm512_add_epi32(d, _mm512_shuffle_epi32(d, _MM_PERM_BADC));
            let d1 = _mm512_add_epi32(d2, _mm512_shuffle_epi32(d2, _MM_PERM_CDAB));
            // lane k element 0 = idot of block g+k
            let idx = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
            let idots = _mm512_castsi512_si128(_mm512_permutexvar_epi32(idx, d1));
            let sc = _mm_set_ps(
                crate::quant::mxfp4::exp2_i(srow[g + 3] as i32 - 128) * xq.scales[g + 3],
                crate::quant::mxfp4::exp2_i(srow[g + 2] as i32 - 128) * xq.scales[g + 2],
                crate::quant::mxfp4::exp2_i(srow[g + 1] as i32 - 128) * xq.scales[g + 1],
                crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128) * xq.scales[g],
            );
            let acc = _mm_loadu_ps(lanes.as_ptr());
            _mm_storeu_ps(lanes.as_mut_ptr(), _mm_fmadd_ps(_mm_cvtepi32_ps(idots), sc, acc));
            g += 4;
        }
        let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        while g < nb {
            let idot = block_dot(&prow[g * 16..(g + 1) * 16], &xq.q[g * 32..(g + 1) * 32]);
            total += idot as f32 * (crate::quant::mxfp4::exp2_i(srow[g] as i32 - 128) * xq.scales[g]);
            g += 1;
        }
        total
    }
}

/// Four MXFP4 rows against one q8 activation with AVX-512 VNNI. The
/// nibbles decode through a shifted LUT (E2M1 x 2 + 12, all in 0..24) so
/// the weights are the UNSIGNED dpbusd operand and the activation the
/// signed one: dot(w + 12, x) = dot(w, x) + 12 sum(x), and 12 sum(x) per
/// block is a property of the activation, computed once per call
/// (`xsum12`) instead of once per row. Block scales 2^(e-128) come from
/// a vector shift when every e >= 2 (the scalar exp2_i otherwise, for
/// the subnormal cases nobody quantizes into). Same exact integer block
/// dots and the same fused four-lane f32 order as row_dot_fp4: each
/// output equals row_dot_fp4 on that row bit for bit.
///
/// SAFETY: caller guarantees avx512f/bw/vnni; every prow has 16*nb
/// bytes, every srow nb bytes, xq nb blocks; xsum12[g] = 12 * sum of
/// block g of xq.q.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,fma")]
pub unsafe fn rows4_dot_fp4_vnni(prows: [&[u8]; 4], srows: [&[u8]; 4], xq: &Q8Vec, xsum12: &[i32]) -> [f32; 4] {
    use std::arch::x86_64::*;
    let nb = srows[0].len();
    unsafe {
        let lut = _mm512_broadcast_i32x4(_mm_loadu_si128(E2M1_X2_P12.as_ptr() as *const __m128i));
        let msk = _mm512_set1_epi8(0x0F);
        let idx = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 8, 4, 0);
        let pa = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
        let pb = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
        let mut acc = [_mm_setzero_ps(); 4];
        let mut g = 0usize;
        while g + 4 <= nb {
            let x0 = _mm512_loadu_si512(xq.q.as_ptr().add(g * 32) as *const _);
            let x1 = _mm512_loadu_si512(xq.q.as_ptr().add(g * 32 + 64) as *const _);
            let xa = _mm512_permutex2var_epi64(x0, pa, x1);
            let xb = _mm512_permutex2var_epi64(x0, pb, x1);
            let xs = _mm_loadu_ps(xq.scales.as_ptr().add(g));
            let corr = _mm_loadu_si128(xsum12.as_ptr().add(g) as *const __m128i);
            for r in 0..4 {
                let bytes = _mm512_loadu_si512(prows[r].as_ptr().add(g * 16) as *const _);
                let lo = _mm512_shuffle_epi8(lut, _mm512_and_si512(bytes, msk));
                let hi = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), msk));
                let ilo = _mm512_unpacklo_epi8(lo, hi);
                let ihi = _mm512_unpackhi_epi8(lo, hi);
                let d = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), ilo, xa), ihi, xb);
                let d2 = _mm512_add_epi32(d, _mm512_shuffle_epi32(d, _MM_PERM_BADC));
                let d1 = _mm512_add_epi32(d2, _mm512_shuffle_epi32(d2, _MM_PERM_CDAB));
                let idots = _mm_sub_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi32(idx, d1)), corr);
                let s = srows[r];
                let e = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(i32::from_le_bytes([s[g], s[g + 1], s[g + 2], s[g + 3]])));
                // 2^(e-128) = bits ((e-128)+127) << 23 = (e-1) << 23, valid for e >= 2
                let sc2 = if _mm_movemask_epi8(_mm_cmplt_epi32(e, _mm_set1_epi32(2))) == 0 {
                    _mm_castsi128_ps(_mm_slli_epi32(_mm_sub_epi32(e, _mm_set1_epi32(1)), 23))
                } else {
                    _mm_set_ps(
                        crate::quant::mxfp4::exp2_i(s[g + 3] as i32 - 128),
                        crate::quant::mxfp4::exp2_i(s[g + 2] as i32 - 128),
                        crate::quant::mxfp4::exp2_i(s[g + 1] as i32 - 128),
                        crate::quant::mxfp4::exp2_i(s[g] as i32 - 128),
                    )
                };
                acc[r] = _mm_fmadd_ps(_mm_cvtepi32_ps(idots), _mm_mul_ps(sc2, xs), acc[r]);
            }
            g += 4;
        }
        let mut out = [0.0f32; 4];
        for r in 0..4 {
            let mut lanes = [0.0f32; 4];
            _mm_storeu_ps(lanes.as_mut_ptr(), acc[r]);
            let mut total = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
            let mut gg = g;
            while gg < nb {
                let idot = block_dot(&prows[r][gg * 16..(gg + 1) * 16], &xq.q[gg * 32..(gg + 1) * 32]);
                total += idot as f32 * (crate::quant::mxfp4::exp2_i(srows[r][gg] as i32 - 128) * xq.scales[gg]);
                gg += 1;
            }
            out[r] = total;
        }
        out
    }
}

/// 12 * (sum of each 32-block of a q8 activation): the correction term of
/// the shifted-LUT fp4 kernels, one i32 per block.
pub fn xsum12(xq: &Q8Vec) -> Vec<i32> {
    xq.q.chunks_exact(32).map(|b| 12 * b.iter().map(|&v| v as i32).sum::<i32>()).collect()
}
