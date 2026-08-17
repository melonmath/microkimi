// Accelerate (AMX) GEMM for the macOS CPU prefill - reached over plain
// C FFI to the Accelerate framework, no crates, exactly like the Metal
// path. llama.cpp's CPU pp rows on Apple silicon go through BLAS (the
// AMX matrix coprocessor, ~4x NEON's GEMM throughput); this module is
// the same weapon: one cblas_sgemm per weight matrix per prompt.
//
// Opt-in (MICROKIMI_ACCEL=1) because sgemm reassociates: results are
// not bit-identical to the sequential kernels, so the default engine
// contract (bit-exact f32) stays intact unless explicitly traded.
// f32 spine weights feed sgemm straight from the mapping; MXFP4 MLP
// matrices dequantize to f32 once and stay cached. Every failure path
// returns false and the caller keeps its CPU kernels.

#![cfg(target_os = "macos")]

use std::sync::{Arc, Mutex, OnceLock};

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

const ROW_MAJOR: i32 = 101;
const NO_TRANS: i32 = 111;
const TRANS: i32 = 112;

/// Lane and size floors: below these the call overhead beats the AMX win.
pub const ACCEL_MIN_T: usize = 16;
pub const ACCEL_MIN_ELEMS: usize = 1 << 20;

/// True when the Accelerate offload is armed (MICROKIMI_ACCEL=1).
pub fn accel_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_ACCEL").map(|v| v == "1").unwrap_or(false))
}

/// Y[t, rows] = X[t, cols] x W[rows, cols]' through Accelerate.
fn sgemm_xwt(x: &[f32], w: &[f32], t: usize, rows: usize, cols: usize, y: &mut [f32]) {
    debug_assert_eq!(x.len(), t * cols);
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(y.len(), t * rows);
    // SAFETY: shapes are asserted above; Accelerate reads A and B, writes
    // C, all within the given leading dimensions.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            NO_TRANS,
            TRANS,
            t as i32,
            rows as i32,
            cols as i32,
            1.0,
            x.as_ptr(),
            cols as i32,
            w.as_ptr(),
            cols as i32,
            0.0,
            y.as_mut_ptr(),
            rows as i32,
        );
    }
}

/// Packs the lanes, runs the GEMM, scatters the rows back.
fn run(w: &[f32], rows: usize, cols: usize, xs: &[&[f32]], outs: &mut [&mut [f32]]) {
    let t = xs.len();
    let mut x = vec![0.0f32; t * cols];
    for (l, lane) in xs.iter().enumerate() {
        x[l * cols..(l + 1) * cols].copy_from_slice(lane);
    }
    let mut y = vec![0.0f32; t * rows];
    sgemm_xwt(&x, w, t, rows, cols, &mut y);
    for (l, out) in outs.iter_mut().enumerate() {
        out.copy_from_slice(&y[l * rows..(l + 1) * rows]);
    }
}

/// f32 multi-lane matvec through AMX. False = caller keeps its kernels.
pub fn gemm_f32(w: &[f32], rows: usize, cols: usize, xs: &[&[f32]], outs: &mut [&mut [f32]]) -> bool {
    if !accel_on() || xs.len() < ACCEL_MIN_T || rows * cols < ACCEL_MIN_ELEMS {
        return false;
    }
    run(w, rows, cols, xs, outs);
    true
}

/// MXFP4 multi-lane matvec through AMX over a dequantized-once cache.
pub fn gemm_fp4(
    packed: &[u8],
    scales: &[u8],
    rows: usize,
    cols: usize,
    xs: &[&[f32]],
    outs: &mut [&mut [f32]],
) -> bool {
    if !accel_on() || xs.len() < ACCEL_MIN_T || rows * cols < ACCEL_MIN_ELEMS {
        return false;
    }
    // (ptr, rows, cols) -> dequantized copy; weights are slices of the
    // session-stable mapping, so the key is stable (same argument as the
    // GPU caches; the first packed byte is kept as an alias tag).
    static CACHE: OnceLock<Mutex<std::collections::HashMap<(usize, usize, usize), (Arc<Vec<f32>>, u8)>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let key = (packed.as_ptr() as usize, rows, cols);
    let tag = packed.first().copied().unwrap_or(0);
    let w = {
        let mut map = cache.lock().unwrap();
        match map.get(&key) {
            Some((w, seen)) if *seen == tag => w.clone(),
            _ => {
                let w = Arc::new(crate::quant::mxfp4::dequant(packed, scales, rows, cols));
                map.insert(key, (w.clone(), tag));
                w
            }
        }
    };
    run(&w, rows, cols, xs, outs);
    true
}
