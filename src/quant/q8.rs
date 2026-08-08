// Q8 activation path for the quantized matvecs (mxfp4 experts today, fp4/fp8
// spine later). The f32 activation is quantized on the fly to int8 per block
// of 32 (q8_0 convention: one f32 scale dx = max|x|/127 per block), then the
// dot product runs in INTEGER SIMD between the packed weights and the q8
// activation - the bulk of the work never touches f32.
//
// This path is NOT bit-identical to the f32 reference (int32-exact block
// sums vs f32 accumulation, plus the q8 rounding of x itself): that is the
// deal. The error is bounded by dx/2 per element and measured in
// selftest::run_q8 (max rel << 1e-3). MICROKIMI_NO_Q8=1 restores the exact
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
pub fn block_dot_i8(w32: &[i8], x32: &[i8]) -> i32 {
    static F: std::sync::OnceLock<I8Dot> = std::sync::OnceLock::new();
    fn pick() -> I8Dot {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return dot_i8_avx2;
        }
        #[cfg(target_arch = "aarch64")]
        return dot_i8_neon;
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
        return dot_block_neon;
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
