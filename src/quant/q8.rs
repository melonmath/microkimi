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
        let w = _mm256_loadu_si256(w32.as_ptr() as *const __m256i);
        let x = _mm256_loadu_si256(x32.as_ptr() as *const __m256i);
        let ax = _mm256_sign_epi8(x, x);
        let aw = _mm256_sign_epi8(w, x);
        let pairs = _mm256_maddubs_epi16(ax, aw);
        let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
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
        let ax = _mm256_sign_epi8(x, x);
        let aw = _mm256_sign_epi8(w, x);
        let pairs = _mm256_maddubs_epi16(ax, aw);
        let quads = _mm256_madd_epi16(pairs, _mm256_set1_epi16(1));
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

type ScoreWindow =
    unsafe fn(&Q8Vec, &[i8], &[f32], usize, usize, usize, usize, f32, &mut [f32]) -> f32;

/// Scores a whole causal window against one quantized query in ONE
/// call: the per-block indirect dispatch of the naive loop cost more
/// than the arithmetic (four dispatches per scored position, hundreds
/// of millions per prompt). Per position the four block dots reduce
/// pairwise and the scales apply as one vector FMA; the scalar
/// fallback mirrors that exact order, and every attention path (batch,
/// single-token tail, lanes) calls this same function. Returns the max
/// score; `scores[..window]` receives scale-applied values.
#[allow(clippy::too_many_arguments)]
pub fn score_window(
    qq: &Q8Vec,
    kq_all: &[i8],
    kqs_all: &[f32],
    stride: usize,
    off0: usize,
    nbh: usize,
    window: usize,
    scale: f32,
    scores: &mut [f32],
) -> f32 {
    static F: std::sync::OnceLock<ScoreWindow> = std::sync::OnceLock::new();
    fn pick() -> ScoreWindow {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("dotprod") && !no_sdot() {
                return score_window_sdot;
            }
        }
        score_window_generic
    }
    let f = F.get_or_init(pick);
    // SAFETY: the sdot variant is only selected when dotprod is present.
    unsafe { f(qq, kq_all, kqs_all, stride, off0, nbh, window, scale, scores) }
}

/// Order reference for the vector kernel: per position, per-block fused
/// lane accumulators collapsed pairwise, then one scale multiply.
#[allow(clippy::too_many_arguments)]
unsafe fn score_window_generic(
    qq: &Q8Vec,
    kq_all: &[i8],
    kqs_all: &[f32],
    stride: usize,
    off0: usize,
    nbh: usize,
    window: usize,
    scale: f32,
    scores: &mut [f32],
) -> f32 {
    let mut max_score = f32::NEG_INFINITY;
    for (u, slot) in scores[..window].iter_mut().enumerate() {
        let off = u * stride + off0;
        let mut lanes = [0.0f32; 4];
        let mut g = 0usize;
        while g + 4 <= nbh {
            for (b, lane) in lanes.iter_mut().enumerate() {
                let i = g + b;
                let idot = block_dot_i8(
                    &kq_all[off + i * 32..off + (i + 1) * 32],
                    &qq.q[i * 32..(i + 1) * 32],
                );
                *lane = (idot as f32).mul_add(kqs_all[off / 32 + i] * qq.scales[i], *lane);
            }
            g += 4;
        }
        let mut acc = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
        while g < nbh {
            let idot = block_dot_i8(
                &kq_all[off + g * 32..off + (g + 1) * 32],
                &qq.q[g * 32..(g + 1) * 32],
            );
            acc += idot as f32 * (kqs_all[off / 32 + g] * qq.scales[g]);
            g += 1;
        }
        let sc = acc * scale;
        *slot = sc;
        max_score = max_score.max(sc);
    }
    max_score
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn score_window_sdot(
    qq: &Q8Vec,
    kq_all: &[i8],
    kqs_all: &[f32],
    stride: usize,
    off0: usize,
    nbh: usize,
    window: usize,
    scale: f32,
    scores: &mut [f32],
) -> f32 {
    use std::arch::aarch64::*;
    if nbh != 4 {
        // uncommon head widths keep the mirrored scalar order
        // SAFETY: forwarded contract.
        return unsafe {
            score_window_generic(qq, kq_all, kqs_all, stride, off0, nbh, window, scale, scores)
        };
    }
    // SAFETY: caller guarantees the mirror holds `window` positions of
    // `stride` bytes with per-32 scales, and qq holds nbh blocks.
    unsafe {
        let q0 = vld1q_s8(qq.q.as_ptr());
        let q1 = vld1q_s8(qq.q.as_ptr().add(16));
        let q2 = vld1q_s8(qq.q.as_ptr().add(32));
        let q3 = vld1q_s8(qq.q.as_ptr().add(48));
        let q4 = vld1q_s8(qq.q.as_ptr().add(64));
        let q5 = vld1q_s8(qq.q.as_ptr().add(80));
        let q6 = vld1q_s8(qq.q.as_ptr().add(96));
        let q7 = vld1q_s8(qq.q.as_ptr().add(112));
        let qs = vld1q_f32(qq.scales.as_ptr());
        let mut maxv = vdupq_n_f32(f32::NEG_INFINITY);
        for (u, slot) in scores[..window].iter_mut().enumerate() {
            let off = u * stride + off0;
            let kp = kq_all.as_ptr().add(off);
            let k0 = vld1q_s8(kp);
            let k1 = vld1q_s8(kp.add(16));
            let k2 = vld1q_s8(kp.add(32));
            let k3 = vld1q_s8(kp.add(48));
            let k4 = vld1q_s8(kp.add(64));
            let k5 = vld1q_s8(kp.add(80));
            let k6 = vld1q_s8(kp.add(96));
            let k7 = vld1q_s8(kp.add(112));
            let mut b0 = vdupq_n_s32(0);
            let mut b1 = vdupq_n_s32(0);
            let mut b2 = vdupq_n_s32(0);
            let mut b3 = vdupq_n_s32(0);
            std::arch::asm!(
                ".arch_extension dotprod",
                "sdot {b0:v}.4s, {k0:v}.16b, {q0:v}.16b",
                "sdot {b0:v}.4s, {k1:v}.16b, {q1:v}.16b",
                "sdot {b1:v}.4s, {k2:v}.16b, {q2:v}.16b",
                "sdot {b1:v}.4s, {k3:v}.16b, {q3:v}.16b",
                "sdot {b2:v}.4s, {k4:v}.16b, {q4:v}.16b",
                "sdot {b2:v}.4s, {k5:v}.16b, {q5:v}.16b",
                "sdot {b3:v}.4s, {k6:v}.16b, {q6:v}.16b",
                "sdot {b3:v}.4s, {k7:v}.16b, {q7:v}.16b",
                b0 = inout(vreg) b0, b1 = inout(vreg) b1,
                b2 = inout(vreg) b2, b3 = inout(vreg) b3,
                k0 = in(vreg) k0, k1 = in(vreg) k1, k2 = in(vreg) k2, k3 = in(vreg) k3,
                k4 = in(vreg) k4, k5 = in(vreg) k5, k6 = in(vreg) k6, k7 = in(vreg) k7,
                q0 = in(vreg) q0, q1 = in(vreg) q1, q2 = in(vreg) q2, q3 = in(vreg) q3,
                q4 = in(vreg) q4, q5 = in(vreg) q5, q6 = in(vreg) q6, q7 = in(vreg) q7,
                options(pure, nomem, nostack)
            );
            let p01 = vpaddq_s32(b0, b1);
            let p23 = vpaddq_s32(b2, b3);
            let sums = vcvtq_f32_s32(vpaddq_s32(p01, p23));
            let ks = vld1q_f32(kqs_all.as_ptr().add(off / 32));
            let prod = vmulq_f32(sums, vmulq_f32(ks, qs));
            // ((l0+l1)+(l2+l3)) * scale - the generic collapse order
            let pair = vpadd_f32(vget_low_f32(prod), vget_high_f32(prod));
            let acc = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);
            let sc = acc * scale;
            *slot = sc;
            maxv = vmaxq_f32(maxv, vdupq_n_f32(sc));
        }
        vmaxvq_f32(maxv)
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
) -> [[f32; 4]; 8] {
    use std::arch::aarch64::*;
    debug_assert!(xqs.len() <= 8, "lane tile is at most 8 wide");
    let nb = xqs[0].1.len();
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 8];
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
        let mut out = [[0.0f32; 4]; 8];
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

/// Four rows x eight lanes through SMMLA (i8mm): one instruction
/// multiplies a 2x8 row-pair block by an 8x2 lane-pair block - 32 MACs,
/// twice SDOT's throughput. Operands come PAIR-INTERLEAVED:
///   weights `wp`, per block: [r0[0..8] r1[0..8] r0[8..16] r1[8..16]
///   ... r2/r3 likewise] (128 bytes per (quad, block));
///   activations `xp`, per lane-pair, per block: [lA[0..8] lB[0..8]
///   lA[8..16] ...] (64 bytes per (pair, block)).
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
    out: &mut [[f32; 4]; 8],
) {
    use std::arch::aarch64::*;
    debug_assert!(pairs <= 4);
    unsafe {
        // acc[pair][0] = rows(0,1) x lanes(A,B) tile, acc[pair][1] = rows(2,3)
        let mut acc = [[vdupq_n_f32(0.0); 2]; 4];
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
                let xb = xp.as_ptr().add((p * nb + g) * 64);
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
