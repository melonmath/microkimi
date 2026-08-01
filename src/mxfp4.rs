// MXFP4: e2m1 per element + e8m0 scale per group of 32 columns.
// Layout: packed u8 [R, C/2] (low nibble = even column), scales u8 [R, C/32]
// (scale = 2^(byte-127)). W[r,c] = LUT[nibble] × 2^(scale[r,c/32]-127).

pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

#[inline]
pub fn exp2_i(e: i32) -> f32 {
    // exact 2^e for e ∈ [-127, 128] via bit manipulation
    if e < -126 {
        return 2f32.powi(e); // subnormal: rare, slow but exact
    }
    f32::from_bits(((e + 127) as u32) << 23)
}

/// Dequantizes a full matrix (for the pools and the selftest).
pub fn dequant(packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
        for c in 0..cols {
            let byte = prow[c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[r * cols + c] = E2M1[nib as usize] * exp2_i(srow[c / 32] as i32 - 127);
        }
    }
    out
}

/// Quantization: per group of 32, scale_exp = max(-127, ceil(log2(maxabs/6))),
/// each value → nearest e2m1 level to v/2^scale_exp (midpoint cutoffs).
/// Returns (packed, scales).
pub fn quantize(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    assert!(cols % 32 == 0);
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * cols / 32];
    // positive e2m1 levels: 0, .5, 1, 1.5, 2, 3, 4, 6; midpoint boundaries
    const BOUNDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        for g in 0..cols / 32 {
            let group = &row[g * 32..(g + 1) * 32];
            let maxabs = group.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let e = if maxabs == 0.0 {
                -127
            } else {
                (maxabs / 6.0).log2().ceil() as i32
            }
            .max(-127)
            .min(128);
            scales[r * cols / 32 + g] = (e + 127).clamp(0, 255) as u8;
            let inv = 1.0 / exp2_i(e);
            for (j, &v) in group.iter().enumerate() {
                let q = (v * inv).clamp(-6.0, 6.0);
                let mag = q.abs();
                let mut idx = 0usize;
                while idx < 7 && mag >= BOUNDS[idx] {
                    idx += 1;
                }
                if q.is_sign_negative() {
                    idx += 8;
                }
                let c = g * 32 + j;
                let byte = &mut packed[r * cols / 2 + c / 2];
                if c % 2 == 0 {
                    *byte |= idx as u8;
                } else {
                    *byte |= (idx as u8) << 4;
                }
            }
        }
    }
    (packed, scales)
}

/// Matvec with on-the-fly dequantization (weights stay packed in RAM):
/// out[r] = Σ_c W[r,c] · x[c]. Per group of 32: Σ(lut·x) × scale - same
/// mathematical result, one floating-point multiplication per group. Multithreaded over rows.
///
/// --gpu (macOS): at rows*cols ≥ GPU_MIN_ELEMS the fused Metal fp4 kernel
/// takes over (metal::gpu_matvec_fp4, weights cached on device). Below the
/// threshold the CPU path wins — a Metal dispatch costs ~0.25 ms, far more
/// than these small matvecs. Micro models keep every expert on the CPU
/// (128×512 = 65 K params ≪ 2 M); the GPU path only kicks in at real V4
/// expert dims (2048×4096 = 8.4 M). NOTE: the three expert matvecs
/// (w1, w3, w2) are NOT batched into one dispatch — each is routed
/// independently; batching would be the next optimization if real-dim
/// profiles show dispatch overhead dominating.
pub fn matvec_packed(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    out: &mut [f32],
    n_threads: usize,
) {
    #[cfg(target_os = "macos")]
    {
        if crate::model::gpu_on() && rows * cols >= crate::model::GPU_MIN_ELEMS && crate::metal::gpu_available() {
            crate::metal::gpu_matvec_fp4(packed, scales, rows, cols, x, out);
            return;
        }
    }
    let nt = n_threads.min(rows);
    if nt <= 1 {
        // direct single-threaded path (small matvecs: experts)
        for (r, o) in out.iter_mut().enumerate() {
            let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
            let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
            let mut sum = 0f32;
            for g in 0..cols / 32 {
                let mut gsum = 0f32;
                for j in 0..32 {
                    let c = g * 32 + j;
                    let byte = prow[c / 2];
                    let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    gsum += E2M1[nib as usize] * x[c];
                }
                sum += gsum * exp2_i(srow[g] as i32 - 127);
            }
            *o = sum;
        }
        return;
    }
    let chunk = rows.div_ceil(nt);
    std::thread::scope(|s| {
        let mut p_rest = packed;
        let mut sc_rest = scales;
        for out_chunk in out.chunks_mut(chunk) {
            let nrows = out_chunk.len();
            let p_chunk = &p_rest[..nrows * cols / 2];
            let s_chunk = &sc_rest[..nrows * cols / 32];
            p_rest = &p_rest[nrows * cols / 2..];
            sc_rest = &sc_rest[nrows * cols / 32..];
            s.spawn(move || {
                for (r, o) in out_chunk.iter_mut().enumerate() {
                    let prow = &p_chunk[r * cols / 2..(r + 1) * cols / 2];
                    let srow = &s_chunk[r * cols / 32..(r + 1) * cols / 32];
                    let mut sum = 0f32;
                    for g in 0..cols / 32 {
                        let mut gsum = 0f32;
                        for j in 0..32 {
                            let c = g * 32 + j;
                            let byte = prow[c / 2];
                            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                            gsum += E2M1[nib as usize] * x[c];
                        }
                        sum += gsum * exp2_i(srow[g] as i32 - 127);
                    }
                    *o = sum;
                }
            });
        }
    });
}
