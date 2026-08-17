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

/// Quantization: per group of 32 the e8m0 scale exponent is searched over
/// {e-1, e, e+1} around the naive e = max(-127, ceil(log2(maxabs/6)))
/// (search_mx_scale), keeping the candidate with the lowest block squared
/// error; each value → nearest e2m1 level to v/2^scale_exp (midpoint
/// cutoffs). Returns (packed, scales).
pub fn quantize(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    quantize_impl(w, rows, cols, true, None)
}

/// Importance-weighted quantization: the per-group scale search minimizes
/// the squared error weighted by `col_imp` (one weight per input column,
/// e.g. the mean squared activation from a calibration run). The nibble
/// assignment stays nearest-level - per element the nearest grid point is
/// optimal under any positive weight - so only the scale choice moves.
/// Byte layout and dtype are unchanged: the runtime cannot tell a weighted
/// blob from an unweighted one.
pub fn quantize_weighted(w: &[f32], rows: usize, cols: usize, col_imp: &[f32]) -> (Vec<u8>, Vec<u8>) {
    assert_eq!(col_imp.len(), cols, "one importance weight per column");
    quantize_impl(w, rows, cols, true, Some(col_imp))
}

/// The pre-search scale rule (naive e = ceil(log2(maxabs/6)) verbatim, no
/// candidate search). Kept for the A/B measurement in test_cmd and for the
/// never-worse unit test.
pub fn quantize_naive(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
    quantize_impl(w, rows, cols, false, None)
}

/// Squared quantization error of one 32-value group at scale exponent e,
/// under the same nearest-level e2m1 assignment quantize_impl packs with.
/// With `imp`, each element's squared error is weighted by its column
/// importance.
fn group_sse(group: &[f32], e: i32, imp: Option<&[f32]>) -> f64 {
    const BOUNDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let inv = 1.0 / exp2_i(e);
    let s = exp2_i(e) as f64;
    let mut sse = 0f64;
    for (j, &v) in group.iter().enumerate() {
        let q = (v * inv).clamp(-6.0, 6.0);
        let mag = q.abs();
        let mut idx = 0usize;
        while idx < 7 && mag >= BOUNDS[idx] {
            idx += 1;
        }
        let lvl = E2M1[idx] as f64;
        let dq = (if q.is_sign_negative() { -lvl } else { lvl }) * s;
        let d = v as f64 - dq;
        sse += d * d * imp.map_or(1.0, |w| w[j] as f64);
    }
    sse
}

/// SSE-optimal e8m0 scale exponent for one 32-value group, searched over
/// {e-1, e, e+1} (e8m0 scales are powers of two, so the immediate neighbors
/// of the naive e are the only useful candidates: a coarser e-1 clips the
/// group max but doubles the grid step, a finer e+1 wastes range). The naive
/// e is scored first and wins ties, candidates outside [-127, 128] are
/// skipped, so the returned exponent is never worse than e on the group.
pub fn search_mx_scale(group: &[f32], e: i32, imp: Option<&[f32]>) -> i32 {
    debug_assert_eq!(group.len(), 32);
    let mut best = e;
    let mut best_sse = group_sse(group, e, imp);
    for cand in [e - 1, e + 1] {
        if !(-127..=128).contains(&cand) {
            continue;
        }
        let sse = group_sse(group, cand, imp);
        if sse < best_sse {
            best_sse = sse;
            best = cand;
        }
    }
    best
}

fn quantize_impl(
    w: &[f32],
    rows: usize,
    cols: usize,
    search: bool,
    col_imp: Option<&[f32]>,
) -> (Vec<u8>, Vec<u8>) {
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
            let mut e = if maxabs == 0.0 {
                -127
            } else {
                (maxabs / 6.0).log2().ceil() as i32
            }
            .max(-127)
            .min(128);
            if search && maxabs != 0.0 {
                let imp = col_imp.map(|imp| &imp[g * 32..(g + 1) * 32]);
                e = search_mx_scale(group, e, imp);
            }
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

// ── MXFP4SQ: quadratic scale encoding (DTYPE_MXFP4SQ, see weights.rs) ──
//
// Same e2m1 nibbles and group-of-32 layout as MXFP4, but the scale byte
// decodes quadratically against a per-tensor f32 `smax` stored as the last
// 4 bytes of the blob: s(q) = ((q+1)/256)^2 * smax, where smax is the exact
// maximum over groups of maxabs/6. e8m0 scales are powers of two, so a group
// whose ideal scale sits just above 2^e loses up to a full factor 2 of
// range; the quadratic grid's relative step is ~2/(q+1) at index q, finest
// exactly where most groups live (well below smax). Encoding always rounds
// the index UP, so the decoded scale is >= the ideal one and no value clips.
// The runtime packed matvec still reads DTYPE_MXFP4 only; MXFP4SQ is a
// storage/measurement variant at this stage (dequant_any reads both).

/// Scale decoded from a quadratic scale byte: ((q+1)/256)^2 * smax.
#[inline]
pub fn scale_sq(q: u8, smax: f32) -> f32 {
    let t = (q as f32 + 1.0) * (1.0 / 256.0);
    t * t * smax
}

/// MXFP4SQ quantization: per group of 32 the ideal scale is maxabs/6 (as
/// MXFP4), the byte is ceil(256*sqrt(ideal/smax)) - 1 clamped to 0..=255, so
/// the decoded quadratic scale never clips. Returns (packed, scales, smax).
pub fn quantize_sq(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>, f32) {
    assert!(cols % 32 == 0);
    const BOUNDS: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
    let ng = rows * cols / 32;
    let mut ideal = vec![0f32; ng];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        for g in 0..cols / 32 {
            let maxabs = row[g * 32..(g + 1) * 32].iter().fold(0f32, |m, &v| m.max(v.abs()));
            ideal[r * cols / 32 + g] = maxabs / 6.0;
        }
    }
    let smax = ideal.iter().fold(0f32, |m, &v| m.max(v));
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * cols / 32];
    for r in 0..rows {
        let row = &w[r * cols..(r + 1) * cols];
        for g in 0..cols / 32 {
            let gi = r * cols / 32 + g;
            let q = if smax == 0.0 || ideal[gi] == 0.0 {
                0
            } else {
                ((256.0 * (ideal[gi] / smax).sqrt()).ceil() as i32 - 1).clamp(0, 255) as u8
            };
            scales[gi] = q;
            let inv = 1.0 / scale_sq(q, smax);
            for (j, &v) in row[g * 32..(g + 1) * 32].iter().enumerate() {
                let qv = (v * inv).clamp(-6.0, 6.0);
                let mag = qv.abs();
                let mut idx = 0usize;
                while idx < 7 && mag >= BOUNDS[idx] {
                    idx += 1;
                }
                if qv.is_sign_negative() {
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
    (packed, scales, smax)
}

/// MXFP4SQ dequantization (mirror of dequant with the quadratic scale).
pub fn dequant_sq(packed: &[u8], scales: &[u8], smax: f32, rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
        for c in 0..cols {
            let byte = prow[c / 2];
            let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            out[r * cols + c] = E2M1[nib as usize] * scale_sq(srow[c / 32], smax);
        }
    }
    out
}

/// Dequantizes a raw blob of either mxfp4 flavor (reader supporting both
/// formats): packed || scales for DTYPE_MXFP4, packed || scales || smax
/// (trailing f32) for DTYPE_MXFP4SQ.
pub fn dequant_any(dtype: u8, blob: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let np = rows * cols / 2;
    match dtype {
        crate::quant::weights::DTYPE_MXFP4 => dequant(&blob[..np], &blob[np..], rows, cols),
        crate::quant::weights::DTYPE_MXFP4SQ => {
            let smax = f32::from_le_bytes(blob[np + rows * cols / 32..np + rows * cols / 32 + 4].try_into().unwrap());
            dequant_sq(&blob[..np], &blob[np..np + rows * cols / 32], smax, rows, cols)
        }
        _ => panic!("dequant_any: dtype {} is not an mxfp4 flavor", dtype),
    }
}

/// Hidden measurement behind `microkimi mxfp4test --model X.bin [--tensors N]`:
/// takes real f32 matrices from a .bin (2D, cols % 32 == 0, >= 16k elements,
/// first N in name order), quantizes each with the naive e8m0 scale rule, the
/// searched e8m0 scale (search_mx_scale, the default of quantize) and the
/// quadratic (MXFP4SQ) scale encoding, and reports the per-tensor and
/// aggregate relative RMS error ||w - wq|| / ||w|| of each.
pub fn test_cmd(args: &[String]) {
    let mp = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(crate::bin_path);
    let n_max: usize = args
        .iter()
        .position(|a| a == "--tensors")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let bin = crate::quant::weights::BinFile::open(&mp);
    let mut names: Vec<&String> = bin
        .entries
        .iter()
        .filter(|(_, e)| {
            e.dtype == crate::quant::weights::DTYPE_F32
                && e.dims.len() == 2
                && e.dims[1] % 32 == 0
                && e.dims[0] as u64 * e.dims[1] as u64 >= 16384
        })
        .map(|(n, _)| n)
        .collect();
    names.sort();
    names.truncate(n_max);
    let mut err_naive = (0f64, 0f64); // (sum err^2, sum w^2)
    let mut err_e8m0 = (0f64, 0f64);
    let mut err_sq = (0f64, 0f64);
    for name in names {
        let e = &bin.entries[name];
        let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
        let w = bin.f32_vec(name);
        let (p0, s0) = quantize_naive(&w, r, c);
        let wq0 = dequant(&p0, &s0, r, c);
        let (p1, s1) = quantize(&w, r, c);
        let wq1 = dequant(&p1, &s1, r, c);
        let (p2, s2, smax) = quantize_sq(&w, r, c);
        let wq2 = dequant_sq(&p2, &s2, smax, r, c);
        let acc = |wq: &[f32]| {
            let mut num = 0f64;
            let mut den = 0f64;
            for (&a, &b) in w.iter().zip(wq) {
                num += (a as f64 - b as f64) * (a as f64 - b as f64);
                den += a as f64 * a as f64;
            }
            (num, den)
        };
        let (n0, d) = acc(&wq0);
        let (n1, _) = acc(&wq1);
        let (n2, _) = acc(&wq2);
        err_naive.0 += n0;
        err_naive.1 += d;
        err_e8m0.0 += n1;
        err_e8m0.1 += d;
        err_sq.0 += n2;
        err_sq.1 += d;
        println!(
            "{:50} [{:5}x{:5}]  rel RMS  naive {:.4}   search {:.4}   sq {:.4}   ({:+.1}% RMS search vs naive)",
            name,
            r,
            c,
            (n0 / d).sqrt(),
            (n1 / d).sqrt(),
            (n2 / d).sqrt(),
            ((n1 / n0).sqrt() - 1.0) * 100.0
        );
    }
    println!(
        "AGGREGATE  rel RMS  naive {:.4}   search {:.4}   sq {:.4}   ({:+.1}% RMS, {:+.1}% MSE search vs naive)",
        (err_naive.0 / err_naive.1).sqrt(),
        (err_e8m0.0 / err_e8m0.1).sqrt(),
        (err_sq.0 / err_sq.1).sqrt(),
        ((err_e8m0.0 / err_naive.0).sqrt() - 1.0) * 100.0,
        (err_e8m0.0 / err_naive.0 - 1.0) * 100.0
    );
}

/// MICROKIMI_NO_PACKED_GPU=1: keep the packed mxfp4 matvecs on the CPU even
/// with --gpu (A/B toggle for the fused Metal fp4 kernel path).
#[cfg(target_os = "macos")]
fn no_packed_gpu() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MICROKIMI_NO_PACKED_GPU").map(|v| v == "1").unwrap_or(false))
}

/// 2^(sb-127) from a ue8m0 scale byte, as an exact bit pattern - the SAME
/// formula the Metal matvec_fp4 kernel uses (and equivalent to exp2_i):
/// sb >= 1 -> exponent field = sb (normal float), sb == 0 -> 0x00400000
/// (2^-127, subnormal), matching exp2_i(-127)'s 2f32.powi fallback.
#[inline]
pub fn scale_from_byte(sb: u8) -> f32 {
    if sb == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits((sb as u32) << 23)
    }
}

/// Host-side emulation of the Metal matvec_fp4 kernel (model/metal.rs, macOS-only):
/// per-element scaling `lut * s * x[c]` (NOT the CPU's per-group
/// (Σ lut·x)·s) and the kernel's accumulation order - `lanes` strided
/// accumulators per row (lane i takes columns i, i+lanes, ...), then a
/// binary-tree reduction within each 32-lane simdgroup and across the
/// simdgroups, standing in for simd_sum (Metal's simdgroup reduction order
/// is implementation-defined; a butterfly tree is representative of its
/// reassociation noise). selftest uses this to bound the GPU-vs-CPU numeric
/// gap on hosts without a Metal device.
pub fn matvec_packed_shader_emul(packed: &[u8], scales: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32], lanes: usize) {
    assert_eq!(cols % 32, 0);
    assert_eq!(packed.len(), rows * cols / 2);
    assert_eq!(scales.len(), rows * cols / 32);
    assert_eq!(x.len(), cols);
    assert_eq!(out.len(), rows);
    assert!(lanes % 32 == 0 && lanes > 0);
    fn tree_sum(v: &mut [f32]) -> f32 {
        let mut n = v.len();
        while n > 1 {
            for i in 0..n / 2 {
                v[i] += v[n / 2 + i];
            }
            n /= 2;
        }
        v[0]
    }
    for (r, o) in out.iter_mut().enumerate() {
        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
        let mut part = vec![0f32; lanes];
        for (lane, p) in part.iter_mut().enumerate() {
            let mut acc = 0f32;
            let mut c = lane;
            while c < cols {
                let byte = prow[c / 2];
                let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                acc += E2M1[nib as usize] * scale_from_byte(srow[c / 32]) * x[c];
                c += lanes;
            }
            *p = acc;
        }
        // simdgroup trees, then a tree across the simdgroup partials
        let mut sg: Vec<f32> = part.chunks_mut(32).map(|c| tree_sum(c)).collect();
        *o = tree_sum(&mut sg);
    }
}

thread_local! {
    /// Reusable q8 activation buffer: matvec_packed quantizes once per call,
    /// and allocating the scratch per call dominated the tiny expert matvecs.
    static Q8_SCRATCH: std::cell::RefCell<crate::quant::q8::Q8Vec> = std::cell::RefCell::new(crate::quant::q8::Q8Vec::new());
}

/// Integer q8 matvec (see q8.rs for the scale convention): per 32-block,
/// out_block = 2^(sb-128) * dx_g * <LUT2 block, q8 block>, the inner dot in
/// exact int32. NOT bit-identical to the f32 path (that is the deal;
/// MICROKIMI_NO_Q8=1 disables it). Same row splitting as the f32 path.
pub fn matvec_packed_q8(packed: &[u8], scales: &[u8], rows: usize, cols: usize, xq: &crate::quant::q8::Q8Vec, out: &mut [f32], n_threads: usize) {
    assert_eq!(xq.q.len(), cols);
    assert_eq!(xq.scales.len(), cols / 32);
    #[inline]
    fn row(prow: &[u8], srow: &[u8], _cols: usize, xq: &crate::quant::q8::Q8Vec) -> f32 {
        // the shared row kernel keeps every q8-activation packed path
        // bit-identical (see q8::row_dot_fp4)
        crate::quant::q8::row_dot_fp4(prow, srow, xq)
    }
    let nt = n_threads.min(rows);
    if nt <= 1 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = row(&packed[r * cols / 2..(r + 1) * cols / 2], &scales[r * cols / 32..(r + 1) * cols / 32], cols, xq);
        }
        return;
    }
    // From the main thread, run on the PERSISTENT pool: the scoped-thread
    // path below spawns nt OS threads per call, and the dense decode
    // makes 72 of these calls per token - the spawns alone were measured
    // at ~8 ms/token (13 GB/s effective on the MLP against ~45 elsewhere).
    // Inside a pool job (MoE experts) the pool barrier cannot nest, so
    // the scoped path remains.
    if !crate::model::pool::in_pool_worker() {
        let p = crate::model::pool::pool();
        let njobs = nt.min(p.workers);
        if njobs > 1 {
            let step = crate::model::dyn_step(rows, njobs);
            let ctr = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let pp = crate::model::pool::SPtrU8(packed.as_ptr());
            let sp = crate::model::pool::SPtrU8(scales.as_ptr());
            let xqp = xq as *const crate::quant::q8::Q8Vec as usize;
            let op = crate::model::pool::MPtr(out.as_mut_ptr());
            let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
            for _ in 0..njobs {
                let ctr = ctr.clone();
                jobs.push(Box::new(move || {
                    let (pp, sp, op) = (pp, sp, op);
                    // SAFETY: the pool barrier in run() outlives every
                    // borrow captured here; each row is written once.
                    unsafe {
                        let packed = std::slice::from_raw_parts(pp.0, rows * cols / 2);
                        let scales = std::slice::from_raw_parts(sp.0, rows * cols / 32);
                        let xq = &*(xqp as *const crate::quant::q8::Q8Vec);
                        loop {
                            let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                            if r0 >= rows {
                                break;
                            }
                            for r in r0..(r0 + step).min(rows) {
                                *op.0.add(r) = row(
                                    &packed[r * cols / 2..(r + 1) * cols / 2],
                                    &scales[r * cols / 32..(r + 1) * cols / 32],
                                    cols,
                                    xq,
                                );
                            }
                        }
                    }
                }));
            }
            p.run(jobs);
            return;
        }
    }
    // dynamic row scheduling: fine chunks off a shared counter so a
    // straggler thread delays one chunk, not rows/nt of them. Per-row
    // math unchanged - bit-identical. Scoped threads (not the pool):
    // this kernel runs inside pool jobs for MoE experts.
    let step = crate::model::dyn_step(rows, nt);
    let ctr = std::sync::atomic::AtomicUsize::new(0);
    let out_base = crate::model::pool::MPtr(out.as_mut_ptr());
    std::thread::scope(|s| {
        for _ in 0..nt {
            let ctr = &ctr;
            s.spawn(move || {
                let out_base = out_base;
                loop {
                    let r0 = ctr.fetch_add(1, std::sync::atomic::Ordering::Relaxed) * step;
                    if r0 >= rows {
                        break;
                    }
                    for r in r0..(r0 + step).min(rows) {
                        let v = row(&packed[r * cols / 2..(r + 1) * cols / 2], &scales[r * cols / 32..(r + 1) * cols / 32], cols, xq);
                        // SAFETY: each row index is visited exactly once
                        // across the scope (disjoint writes), and the scope
                        // outlives no borrow of `out`.
                        unsafe { *out_base.0.add(r) = v };
                    }
                }
            });
        }
    });
}
/// Block-sparse packed matvec over the COLUMN dimension: only the listed
/// 32-column blocks contribute. Per row, the q8 path walks exactly the
/// kept blocks (the q8 activation blocks share the 32 granularity), so a
/// skipped block costs nothing. Used by the certified-budget MLP down
/// projection; results equal `matvec_packed` when every block is kept.
pub fn matvec_packed_colblocks(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    kept: &[usize],
    x: &[f32],
    out: &mut [f32],
    n_threads: usize,
) {
    if kept.len() == cols / 32 {
        return matvec_packed(packed, scales, rows, cols, x, out, n_threads);
    }
    if crate::quant::q8::q8_enabled() {
        let xq = crate::quant::q8::quantize_q8(x);
        let row_dot = |r: usize| -> f32 {
            let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
            let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
            let mut sum = 0f32;
            for &g in kept {
                let idot = crate::quant::q8::block_dot(
                    &prow[g * 16..(g + 1) * 16],
                    &xq.q[g * 32..(g + 1) * 32],
                );
                sum += idot as f32 * (exp2_i(srow[g] as i32 - 128) * xq.scales[g]);
            }
            sum
        };
        let nt = n_threads.min(rows).max(1);
        if nt <= 1 {
            for (r, o) in out.iter_mut().enumerate() {
                *o = row_dot(r);
            }
            return;
        }
        let chunk = rows.div_ceil(nt);
        std::thread::scope(|scope| {
            for (j, out_chunk) in out.chunks_mut(chunk).enumerate() {
                let r0 = j * chunk;
                scope.spawn(move || {
                    for (i, o) in out_chunk.iter_mut().enumerate() {
                        *o = row_dot(r0 + i);
                    }
                });
            }
        });
        return;
    }
    // exact f32 fallback
    for r in 0..rows {
        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
        let mut sum = 0f32;
        for &g in kept {
            let mut gsum = 0f32;
            for j in 0..32 {
                let c = g * 32 + j;
                let byte = prow[c / 2];
                let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                gsum += E2M1[nib as usize] * x[c];
            }
            sum += gsum * exp2_i(srow[g] as i32 - 127);
        }
        out[r] = sum;
    }
}

/// Multi-lane packed matvec: each packed row (and its scales) is
/// traversed ONCE and dotted against every lane's quantized input, so n
/// decode lanes cost close to one in weight traffic. Per-lane results
/// are bit-identical to `matvec_packed` with the same settings.
pub fn matvec_packed_multi(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    xs: &[&[f32]],
    outs: &mut [&mut [f32]],
    n_threads: usize,
) {
    assert_eq!(xs.len(), outs.len());
    let lanes = xs.len();
    if lanes == 0 {
        return;
    }
    // Qwen prefill offload (MICROKIMI_QWEN_GPU=1, macOS): one f32 GEMM
    // over the device-cached dequantized copy for the whole batch. Note
    // the numerics differ from the CPU path twice over (no q8 activation
    // quantization, GPU reassociation); any failure falls through.
    #[cfg(target_os = "macos")]
    if crate::model::metal::qwen_gpu_on()
        && lanes >= crate::model::metal::GEMM_MIN_T
        && rows * cols >= crate::model::metal::GEMM_MIN_ELEMS
        && crate::model::metal::gpu_gemm_xwt_fp4(packed, scales, rows, cols, xs, outs)
    {
        return;
    }
    if crate::quant::q8::q8_enabled() {
        let xqs: Vec<crate::quant::q8::Q8Vec> =
            xs.iter().map(|x| crate::quant::q8::quantize_q8(x)).collect();
        let xrefs: Vec<&crate::quant::q8::Q8Vec> = xqs.iter().collect();
        // per-lane raw output pointers shared across the scoped rows
        let out_ptrs: Vec<usize> = outs.iter_mut().map(|o| o.as_mut_ptr() as usize).collect();
        // one row against every lane, in tiles of 16: the nibble unpack
        // happens once per 4-block group for the whole tile
        // (q8::row_dot_fp4_multi), per-lane bits equal to row_dot_fp4
        let do_rows = |r0: usize, r1: usize| {
            let mut buf = [0f32; 16];
            // L2 lane-blocking (see ops::multi_rows_q8): a 64-lane block
            // stays cache-resident while the rows stream past it
            let mut b0 = 0usize;
            while b0 < xrefs.len() {
                let b1 = (b0 + 64).min(xrefs.len());
                for r in r0..r1 {
                    let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
                    let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
                    let mut l0 = b0;
                    for tile in xrefs[b0..b1].chunks(16) {
                        crate::quant::q8::row_dot_fp4_multi(prow, srow, tile, &mut buf[..tile.len()]);
                        for (k, v) in buf[..tile.len()].iter().enumerate() {
                            // SAFETY: (r, lane) cells are written exactly once;
                            // the scope barrier below outlives the borrows.
                            unsafe { *(out_ptrs[l0 + k] as *mut f32).add(r) = *v };
                        }
                        l0 += tile.len();
                    }
                }
                b0 = b1;
            }
        };
        let nt = n_threads.min(rows).max(1);
        if nt <= 1 || crate::model::pool::in_pool_worker() {
            do_rows(0, rows);
            return;
        }
        let chunk = rows.div_ceil(nt);
        std::thread::scope(|scope| {
            for j in 0..nt {
                let (r0, r1) = (j * chunk, ((j + 1) * chunk).min(rows));
                if r0 >= r1 {
                    break;
                }
                let do_rows = &do_rows;
                scope.spawn(move || do_rows(r0, r1));
            }
        });
        return;
    }
    // exact f32 fallback (MICROKIMI_NO_Q8=1): row nibbles decoded once per
    // row per lane through the same loop as matvec_packed's slow path
    for r in 0..rows {
        let prow = &packed[r * cols / 2..(r + 1) * cols / 2];
        let srow = &scales[r * cols / 32..(r + 1) * cols / 32];
        for l in 0..lanes {
            let mut sum = 0f32;
            for g in 0..cols / 32 {
                let mut gsum = 0f32;
                for j in 0..32 {
                    let c = g * 32 + j;
                    let byte = prow[c / 2];
                    let nib = if c % 2 == 0 { byte & 0x0F } else { byte >> 4 };
                    gsum += E2M1[nib as usize] * xs[l][c];
                }
                sum += gsum * exp2_i(srow[g] as i32 - 127);
            }
            outs[l][r] = sum;
        }
    }
}

/// out[r] = Σ_c W[r,c] · x[c]. Per group of 32: Σ(lut·x) × scale - same
/// mathematical result, one floating-point multiplication per group. Multithreaded over rows.
///
/// --gpu (macOS): at rows*cols ≥ GPU_MIN_ELEMS the fused Metal fp4 kernel
/// takes over (metal::gpu_matvec_fp4, weights cached on device).
/// MICROKIMI_NO_PACKED_GPU=1 keeps the packed matvecs on the CPU even with
/// --gpu (A/B toggle for the fused kernel). Below the
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
        if crate::model::gpu_on() && !no_packed_gpu() && rows * cols >= crate::model::GPU_MIN_ELEMS && crate::model::metal::gpu_available() {
            crate::model::metal::gpu_matvec_fp4(packed, scales, rows, cols, x, out);
            return;
        }
    }
    // q8 integer path (default; MICROKIMI_NO_Q8=1 keeps the exact f32 path
    // below). The activation is quantized once per call into a thread-local
    // scratch (allocating per call dominated the tiny expert matvecs) -
    // O(cols) work against O(rows*cols) for the matvec, so sharing it
    // across the w1/w3 calls of one expert (same input) was measured not
    // worth the call-site churn (< 2% of an expert matvec).
    if crate::quant::q8::q8_enabled() {
        Q8_SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            crate::quant::q8::quantize_q8_into(x, &mut s);
            matvec_packed_q8(packed, scales, rows, cols, &s, out, n_threads);
        });
        return;
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_quantization_never_loses_on_the_weighted_metric() {
        let (rows, cols) = (8usize, 64usize);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 61 + 17) % 509) as f32 - 254.0) * 0.003)
            .collect();
        // strongly skewed importances: a few hot columns dominate
        let imp: Vec<f32> = (0..cols)
            .map(|c| if c % 16 == 0 { 40.0 } else { 0.2 })
            .collect();
        let weighted_err = |packed: &[u8], scales: &[u8]| -> f64 {
            let dq = dequant(packed, scales, rows, cols);
            (0..rows * cols)
                .map(|i| {
                    let d = (w[i] - dq[i]) as f64;
                    d * d * imp[i % cols] as f64
                })
                .sum()
        };
        let (p0, s0) = quantize(&w, rows, cols);
        let (p1, s1) = quantize_weighted(&w, rows, cols, &imp);
        assert_eq!(p1.len(), p0.len());
        assert_eq!(s1.len(), s0.len());
        let (e0, e1) = (weighted_err(&p0, &s0), weighted_err(&p1, &s1));
        assert!(
            e1 <= e0,
            "weighted search must not lose on its own metric: {} vs {}",
            e1,
            e0
        );
        // uniform weights reproduce the unweighted bytes exactly
        let (pu, su) = quantize_weighted(&w, rows, cols, &vec![1.0; cols]);
        assert_eq!((pu, su), (p0, s0));
    }

    /// Per-32-group squared errors between w and dequant(packed, scales).
    fn block_errors(w: &[f32], rows: usize, cols: usize, packed: &[u8], scales: &[u8]) -> Vec<f64> {
        let wq = dequant(packed, scales, rows, cols);
        (0..rows * cols / 32)
            .map(|g| {
                w[g * 32..(g + 1) * 32]
                    .iter()
                    .zip(&wq[g * 32..(g + 1) * 32])
                    .map(|(&a, &b)| {
                        let d = a as f64 - b as f64;
                        d * d
                    })
                    .sum()
            })
            .collect()
    }

    /// The searched scale never scores worse than the naive one on ANY block:
    /// the naive exponent is one of the three candidates and wins ties.
    fn assert_never_worse(w: &[f32], rows: usize, cols: usize) {
        let (p0, s0) = quantize_naive(w, rows, cols);
        let (p1, s1) = quantize(w, rows, cols);
        let e0 = block_errors(w, rows, cols, &p0, &s0);
        let e1 = block_errors(w, rows, cols, &p1, &s1);
        for (g, (&a, &b)) in e0.iter().zip(&e1).enumerate() {
            assert!(b <= a * (1.0 + 1e-9) + 1e-20, "block {} worse with search: {} > {}", g, b, a);
        }
    }

    #[test]
    fn search_never_worse_synthetic() {
        let pattern = |i: usize| -> f32 {
            let h = (i as u64).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
            ((h >> 13) % 2000) as f32 / 1000.0 - 1.0
        };
        for (rows, cols) in [(64usize, 128usize), (128, 64), (3, 64), (1, 32), (256, 1024)] {
            let w: Vec<f32> = (0..rows * cols).map(&pattern).collect();
            assert_never_worse(&w, rows, cols);
        }
        // spike blocks: one large outlier per group, the e-1 candidate clips
        // it but halves every other error
        let (rows, cols) = (16usize, 64usize);
        let mut w = vec![0.01f32; rows * cols];
        for g in 0..rows * cols / 32 {
            w[g * 32] = if g % 2 == 0 { 3.7 } else { -5.9 };
        }
        assert_never_worse(&w, rows, cols);
        // all-zero groups (scale byte 0) and exact-level groups
        let w = vec![0f32; 64];
        assert_never_worse(&w, 2, 32);
        let w: Vec<f32> = (0..64).map(|i| E2M1[i % 16] * 0.125).collect();
        assert_never_worse(&w, 2, 32);
    }

    #[test]
    fn roundtrip_exact_levels() {
        // values already on the e2m1 grid at the naive group scale round-trip
        // exactly (maxabs = 6 * 0.25, so naive e = -2 and the scale is exact)
        let (rows, cols) = (4usize, 64usize);
        let w: Vec<f32> = (0..rows * cols).map(|i| E2M1[i % 16] * 0.25).collect();
        let (p, s) = quantize(&w, rows, cols);
        let wq = dequant(&p, &s, rows, cols);
        for (&a, &b) in w.iter().zip(&wq) {
            assert_eq!(a, b);
        }
    }

    /// Proof on real weights: every f32 2D tensor of the smoke model
    /// (shared experts, routed projections, embeddings) quantizes with a
    /// per-block error never worse than the naive scale rule. Skipped when
    /// the file is absent (MICROKIMI_SMOKE_BIN overrides the path).
    #[test]
    fn search_never_worse_smoke_model() {
        let path = std::env::var("MICROKIMI_SMOKE_BIN")
            .unwrap_or_else(|_| "/workspace/chat_smoke/nanokimi_chat_smoke.bin".to_string());
        if !std::path::Path::new(&path).exists() {
            eprintln!("smoke model {} not found, skipping", path);
            return;
        }
        let bin = crate::quant::weights::BinFile::open(&path);
        let mut names: Vec<&String> = bin
            .entries
            .iter()
            .filter(|(_, e)| e.dtype == crate::quant::weights::DTYPE_F32 && e.dims.len() == 2 && e.dims[1] % 32 == 0)
            .map(|(n, _)| n)
            .collect();
        names.sort();
        assert!(!names.is_empty(), "no f32 2D tensors in {}", path);
        let (mut sum0, mut sum1, mut sumw) = (0f64, 0f64, 0f64);
        for name in names {
            let e = &bin.entries[name];
            let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
            let w = bin.f32_vec(name);
            assert_never_worse(&w, r, c);
            let (p0, s0) = quantize_naive(&w, r, c);
            let (p1, s1) = quantize(&w, r, c);
            sum0 += block_errors(&w, r, c, &p0, &s0).iter().sum::<f64>();
            sum1 += block_errors(&w, r, c, &p1, &s1).iter().sum::<f64>();
            sumw += w.iter().map(|&v| v as f64 * v as f64).sum::<f64>();
        }
        eprintln!(
            "smoke model f32 tensors: rel RMS naive {:.5} -> search {:.5} ({:+.2}% MSE)",
            (sum0 / sumw).sqrt(),
            (sum1 / sumw).sqrt(),
            (sum1 / sum0 - 1.0) * 100.0
        );
    }
}
