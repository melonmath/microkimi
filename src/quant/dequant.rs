// Dequantization for DeepSeek-V4 checkpoint formats (see /tmp/dsv4/kernel.py):
// - FP8  e4m3fn weights + ue8m0 scales, blocks of 128x128 (dense/spine weights)
// - FP4  e2m1 packed nibbles (low nibble first) + ue8m0 scales per 32 (experts)
//   — the FP4 layout is identical to the MXFP4 one in mxfp4.rs.
//
// ue8m0 (unsigned e8m0): scale = 2^(byte - 127).
// e4m3fn (NVIDIA float8, no inf, NaN at 0x7F/0xFF):
//   normal:    (-1)^s * 2^(e-7) * (1 + m/8), e in 1..=15
//   subnormal: (-1)^s * (m/8) * 2^-6,        e = 0

/// e4m3fn byte → f32 (bit-exact, no lookup table needed).
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = (b >> 3) & 0x0F;
    let m = (b & 0x07) as f32;
    if e == 15 && (b & 0x07) == 7 {
        return f32::NAN; // e4m3fn NaN encoding (unused in practice)
    }
    if e == 0 {
        return sign * (m / 8.0) * 2f32.powi(-6);
    }
    sign * (1.0 + m / 8.0) * crate::quant::mxfp4::exp2_i(e as i32 - 7)
}

/// FP8 e4m3 weight block [rows, cols] u8 + ue8m0 scales [ceil(rows/128), ceil(cols/128)]
/// → f32 [rows, cols]. W[r,c] = e4m3(w[r,c]) * 2^(scale[r/128, c/128] - 127).
pub fn dequant_fp8(w: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let srows = rows.div_ceil(128);
    assert_eq!(w.len(), rows * cols, "fp8: bad weight length");
    assert_eq!(scales.len(), srows * cols.div_ceil(128), "fp8: bad scale length");
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = crate::quant::mxfp4::exp2_i(scales[(r / 128) * cols.div_ceil(128) + c / 128] as i32 - 127);
            out[r * cols + c] = e4m3_to_f32(w[r * cols + c]) * s;
        }
    }
    out
}

/// FP4 experts = MXFP4 layout (e2m1 low-nibble-first + ue8m0/32). Thin wrapper
/// over mxfp4::dequant with the V4 naming for clarity at call sites.
pub fn dequant_fp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    crate::quant::mxfp4::dequant(packed, scales, rows, cols)
}

/// e4m3 quantization (for the micro builder): per-128×128 block,
/// scale_exp = ceil(log2(maxabs / 448)), clamped to [-7, 8] (ue8m0 byte range),
/// value → nearest e4m3 level (round to nearest even on ties).
pub fn quantize_fp8(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(rows % 128 == 0 && cols % 128 == 0, "fp8 quant: dims must be multiples of 128");
    let mut qw = vec![0u8; rows * cols];
    let mut scales = vec![0u8; (rows / 128) * (cols / 128)];
    for br in 0..rows / 128 {
        for bc in 0..cols / 128 {
            let mut maxabs = 0f32;
            for r in br * 128..(br + 1) * 128 {
                for c in bc * 128..(bc + 1) * 128 {
                    maxabs = maxabs.max(w[r * cols + c].abs());
                }
            }
            let e = if maxabs == 0.0 { -127 } else { (maxabs / 448.0).log2().ceil() as i32 }
                .clamp(-127, 8);
            scales[br * (cols / 128) + bc] = (e + 127) as u8;
            let inv = 1.0 / crate::quant::mxfp4::exp2_i(e);
            for r in br * 128..(br + 1) * 128 {
                for c in bc * 128..(bc + 1) * 128 {
                    qw[r * cols + c] = f32_to_e4m3(w[r * cols + c] * inv);
                }
            }
        }
    }
    (qw, scales)
}

/// Round half to even (torch cast semantics) — Rust f32::round is half-away-from-zero.
fn rne(x: f32) -> i32 {
    let f = x.floor();
    let frac = x - f;
    let i = f as i32;
    if frac > 0.5 {
        i + 1
    } else if frac < 0.5 {
        i
    } else {
        // exact tie: round to even
        if i % 2 == 0 { i } else { i + 1 }
    }
}

/// f32 → nearest e4m3fn byte (round to nearest, ties to even mantissa).
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs().min(448.0);
    // round-to-zero threshold = half the smallest subnormal (2^-9 / 2 = 2^-10)
    if a < 2f32.powi(-10) {
        return sign;
    }
    // smallest normal is 2^-6; subnormals are m/8 * 2^-6 with m in 1..=7
    if a < 2f32.powi(-6) {
        let m = rne(a / 2f32.powi(-6) * 8.0);
        // m == 8 carries to the smallest normal (e=1, m=0), not a clamp to 7
        return if m >= 8 { sign | 0x08 } else { sign | (m as u8) };
    }
    // normal: find e, m with value = 2^(e-7) * (1 + m/8)
    let mut e = 1i32;
    while e < 15 && a >= 2f32.powi(e - 6) {
        e += 1;
    }
    let base = 2f32.powi(e - 7);
    let m = rne((a / base - 1.0) * 8.0);
    let (e2, m2) = if m == 8 { (e + 1, 0) } else { (e, m) };
    if e2 >= 15 && m2 > 6 {
        return sign | 0x7E; // clamp to max finite (448)
    }
    sign | ((e2 as u8) << 3) | (m2 as u8)
}
