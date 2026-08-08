// Metal GPU support — step 1: FFI layer to the Objective-C runtime and the
// Metal framework, no crates (std only). Everything in this file is compiled
// ONLY on macOS (`#[cfg(target_os = "macos")]` at the `mod` site in main.rs);
// on Linux this file does not exist as far as the compiler is concerned.
//
// Approach (standard for raw-FFI Objective-C on arm64):
// - `objc_getClass` / `sel_registerName` resolve classes and selectors.
// - `objc_msgSend` is declared once as a variadic C function, then transmuted
//   to typed function pointers matching each method's exact signature. On
//   arm64 this is sound because the objc_msgSend trampolines are handwritten
//   assembly that read arguments from registers following the standard AAPCS
//   (they do not apply the variadic-argument ABI), which is exactly what the
//   typed signatures describe. This is the same technique the `objc` crate
//   and core-foundation use internally.
// - Small structs passed by value (MTLSize = 3 x u64 = 24 bytes) travel in
//   registers on arm64 — no stret trampoline is needed on this architecture.
//
// All `unsafe` is confined to this module; every call site documents why the
// pointers it passes are valid.

#![cfg(target_os = "macos")]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CStr, CString};

// ── raw FFI declarations ──

pub type Id = *mut c_void;
pub type Sel = *mut c_void;

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    // Declared variadic like the real symbol; always used through a typed
    // transmute below, never called directly.
    fn objc_msgSend(receiver: Id, selector: Sel, ...) -> Id;
}

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

/// #[repr(C)] mirror of MTLSize (3 x NSUInteger).
#[repr(C)]
#[derive(Clone, Copy)]
struct MTLSize {
    width: u64,
    height: u64,
    depth: u64,
}

// ── tiny helpers ──

fn sel(name: &str) -> Sel {
    let c = CString::new(name).unwrap();
    // SAFETY: `c` is a valid NUL-terminated C string for the duration of the call.
    unsafe { sel_registerName(c.as_ptr()) }
}

fn class(name: &str) -> Id {
    let c = CString::new(name).unwrap();
    // SAFETY: `c` is a valid NUL-terminated C string for the duration of the call.
    unsafe { objc_getClass(c.as_ptr()) }
}

/// Send a no-argument message returning an object pointer.
/// SAFETY: `obj` must be a valid Objective-C object (or a class for class
/// methods) that responds to `s` with a signature `() -> id`.
unsafe fn msg_id(obj: Id, s: Sel) -> Id {
    let f: extern "C" fn(Id, Sel) -> Id = unsafe { std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend) };
    f(obj, s)
}

/// Send a no-argument message returning nothing.
/// SAFETY: `obj` must respond to `s` with a signature `() -> void`.
unsafe fn msg_void(obj: Id, s: Sel) {
    let f: extern "C" fn(Id, Sel) = unsafe { std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend) };
    f(obj, s)
}

fn ns_string(s: &str) -> Id {
    let c = CString::new(s).unwrap();
    let cls = class("NSString");
    // SAFETY: NSString responds to stringWithUTF8String: with (const char*) -> id;
    // `c` outlives the call; the returned NSString is autoreleased (we hold an
    // autorelease pool for the whole test).
    unsafe {
        let f: extern "C" fn(Id, Sel, *const c_char) -> Id = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        f(cls, sel("stringWithUTF8String:"), c.as_ptr())
    }
}

fn utf8(ns: Id) -> String {
    if ns.is_null() {
        return String::new();
    }
    // SAFETY: `ns` is a valid NSString; UTF8String returns a valid C string
    // pointer owned by the NSString, copied here before the pool is drained.
    unsafe {
        let f: extern "C" fn(Id, Sel) -> *const c_char = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let p = f(ns, sel("UTF8String"));
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

fn err_desc(err: Id) -> String {
    if err.is_null() {
        return String::from("(no error)");
    }
    // SAFETY: `err` is a valid NSError; localizedDescription returns an
    // autoreleased NSString valid until the pool is drained.
    unsafe { utf8(msg_id(err, sel("localizedDescription"))) }
}

// ── Metal shader (MSL), compiled at runtime by newLibraryWithSource: ──

const MATVEC_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// One threadgroup per output row. Threads accumulate strided partial sums
// with COALESCED accesses (consecutive lanes read consecutive addresses of
// both W and x), then reduce across the threadgroup: simd_sum inside each
// 32-lane simdgroup, then a final simd_sum over the per-simdgroup partials.
// (The previous one-thread-per-row kernel had fully uncoalesced accesses:
// at each instruction step, 32 consecutive threads touched 32 different
// cache lines ~2 KB apart.)
kernel void matvec(device const float* W [[buffer(0)]],
                   device const float* x [[buffer(1)]],
                   device float*       y [[buffer(2)]],
                   constant uint&    cols [[buffer(3)]],
                   uint row  [[threadgroup_position_in_grid]],
                   uint lane [[thread_position_in_threadgroup]],
                   uint lanes [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    device const float* w = W + (size_t)row * cols;
    float acc = 0.0f;
    for (uint i = lane; i < cols; i += lanes) {
        acc += w[i] * x[i];
    }
    acc = simd_sum(acc);                        // reduce within the 32-lane simdgroup
    if ((lane & 31u) == 0u) partial[lane / 32u] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32u) {
        uint nsg = (lanes + 31u) / 32u;         // simdgroups in this threadgroup
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) y[row] = v;
    }
}
"#;

// FP4 (e2m1) dequant-matvec, fused on GPU: y[r] = Σ_c dequant(W[r,c]) · x[c].
// EXACT same layout as the CPU mxfp4::matvec_packed: packed u8 [rows, cols/2]
// (LOW nibble = even column), one ue8m0 scale byte per 32 columns
// (scale = 2^(byte-127)). cols is a multiple of 32 (mxfp4 invariant).
// Same threadgroup-per-row + simd_sum structure as the f32 kernel above;
// consecutive lanes read consecutive packed bytes (coalesced).
// Numeric note: the CPU applies the scale AFTER summing each group of 32
// (Σ lut·x)·s; here each element is scaled before the accumulation
// (lut·s·x) — same math, f32-level reassociation noise (≪ 1e-3).
const MATVEC_FP4_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matvec_fp4(device const uchar* packed [[buffer(0)]],
                       device const uchar* scales [[buffer(1)]],
                       device const float* x      [[buffer(2)]],
                       device float*       y      [[buffer(3)]],
                       constant uint&    cols   [[buffer(4)]],
                       uint row  [[threadgroup_position_in_grid]],
                       uint lane [[thread_position_in_threadgroup]],
                       uint lanes [[threads_per_threadgroup]]) {
    // e2m1 value table (sign × 8 magnitudes) — same as mxfp4::E2M1
    constant float LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    threadgroup float partial[32];
    device const uchar* prow = packed + (size_t)row * (cols >> 1);
    device const uchar* srow = scales + (size_t)row * (cols >> 5);
    float acc = 0.0f;
    for (uint c = lane; c < cols; c += lanes) {
        uchar byte = prow[c >> 1];
        uint nib = (c & 1u) ? uint(byte >> 4) : uint(byte & 0x0F);
        uchar sb = srow[c >> 5];
        // 2^(sb-127) as an exact bit pattern (exponent field = sb), no exp call;
        // sb == 0 → 2^-127 (subnormal 0x00400000), matching mxfp4::exp2_i.
        float s = (sb == 0) ? as_type<float>(0x00400000u) : as_type<float>(uint(sb) << 23);
        acc += LUT[nib] * s * x[c];
    }
    acc = simd_sum(acc);
    if ((lane & 31u) == 0u) partial[lane / 32u] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32u) {
        uint nsg = (lanes + 31u) / 32u;
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) y[row] = v;
    }
}
"#;

// ── step 1 test: GPU matvec vs CPU reference on a fixed case ──

const ROWS: usize = 1024;
const COLS: usize = 512;

/// Deterministic pattern (integer hash → [-1, 1]) — no RNG state to keep in
/// sync between the Rust and Metal sides.
fn pattern(i: usize) -> f32 {
    let h = (i as u64).wrapping_mul(2654435761).wrapping_add(0x9E3779B9);
    ((h >> 13) % 2000) as f32 / 1000.0 - 1.0
}

pub fn metaltest() {
    // Fixed deterministic case.
    let w: Vec<f32> = (0..ROWS * COLS).map(pattern).collect();
    let x: Vec<f32> = (0..COLS).map(|i| pattern(i + 12345)).collect();

    // CPU reference (our existing multithreaded matvec).
    let mut y_ref = vec![0f32; ROWS];
    crate::model::matvec(&w, ROWS, COLS, &x, &mut y_ref);

    println!("metaltest — Metal matvec vs CPU reference ({}x{})", ROWS, COLS);

    // SAFETY: the whole block only touches Objective-C objects returned by the
    // runtime itself; raw pointers are dereferenced only through the typed
    // transmutes whose signatures match the Objective-C methods. The single
    // autorelease pool covers every autoreleased object created below.
    unsafe {
        let pool_cls = class("NSAutoreleasePool");
        let pool = msg_id(msg_id(pool_cls, sel("alloc")), sel("init"));

        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            println!("FAIL: MTLCreateSystemDefaultDevice returned nil (no Metal device)");
            msg_void(pool, sel("drain"));
            return;
        }
        let dev_name = msg_id(device, sel("name"));
        println!("device: {}", utf8(dev_name));

        // Compile the MSL source.
        let src = ns_string(MATVEC_MSL);
        let mut err: Id = std::ptr::null_mut();
        let library = {
            let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                device,
                sel("newLibraryWithSource:options:error:"),
                src,
                std::ptr::null_mut(), // options = nil
                &mut err,
            )
        };
        if library.is_null() {
            println!("FAIL: shader compilation error: {}", err_desc(err));
            msg_void(pool, sel("drain"));
            return;
        }

        let fname = ns_string("matvec");
        let function = {
            let f: extern "C" fn(Id, Sel, Id) -> Id = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(library, sel("newFunctionWithName:"), fname)
        };
        if function.is_null() {
            println!("FAIL: kernel 'matvec' not found in library");
            msg_void(pool, sel("drain"));
            return;
        }

        let mut perr: Id = std::ptr::null_mut();
        let pipeline = {
            let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                device,
                sel("newComputePipelineStateWithFunction:error:"),
                function,
                &mut perr,
            )
        };
        if pipeline.is_null() {
            println!("FAIL: pipeline creation error: {}", err_desc(perr));
            msg_void(pool, sel("drain"));
            return;
        }

        let max_ttg = {
            let f: extern "C" fn(Id, Sel) -> u64 = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(pipeline, sel("maxTotalThreadsPerThreadgroup"))
        };

        // Buffers in shared storage mode (0 = MTLResourceStorageModeShared):
        // CPU writes them, GPU reads/writes them, CPU reads the result back.
        let cols_vec = vec![COLS as u32];
        let y_gpu_init = vec![0f32; ROWS];
        let make_buf = |ptr: *const c_void, len: usize| -> Id {
            let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(device, sel("newBufferWithBytes:length:options:"), ptr, len as u64, 0)
        };
        let buf_w = make_buf(w.as_ptr() as *const c_void, std::mem::size_of_val(&w[..]));
        let buf_x = make_buf(x.as_ptr() as *const c_void, std::mem::size_of_val(&x[..]));
        let buf_y = make_buf(y_gpu_init.as_ptr() as *const c_void, std::mem::size_of_val(&y_gpu_init[..]));
        let buf_cols = make_buf(cols_vec.as_ptr() as *const c_void, 4);
        if buf_w.is_null() || buf_x.is_null() || buf_y.is_null() || buf_cols.is_null() {
            println!("FAIL: buffer allocation failed");
            msg_void(pool, sel("drain"));
            return;
        }

        let queue = msg_id(device, sel("newCommandQueue"));
        let cmdbuf = msg_id(queue, sel("commandBuffer"));
        let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));

        // setComputePipelineState: takes the pipeline as its argument.
        {
            let f: extern "C" fn(Id, Sel, Id) = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setComputePipelineState:"), pipeline);
        }
        {
            let f: extern "C" fn(Id, Sel, Id, u64, u64) = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_w, 0, 0);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_x, 0, 1);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_y, 0, 2);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_cols, 0, 3);
        }
        {
            // threadgroup-per-row kernel: GROUP threads per row (multiple of
            // 32 for the simdgroup reductions), grid = ROWS × GROUP threads.
            let mut group = max_ttg.min(256);
            group = (group / 32).max(1) * 32;
            let f: extern "C" fn(Id, Sel, MTLSize, MTLSize) = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                encoder,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { width: ROWS as u64 * group, height: 1, depth: 1 },
                MTLSize { width: group, height: 1, depth: 1 },
            );
        }
        msg_void(encoder, sel("endEncoding"));
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        // Read back the result.
        let out = {
            let f: extern "C" fn(Id, Sel) -> *mut c_void = std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(buf_y, sel("contents")) as *const f32
        };
        let y_gpu: Vec<f32> = std::slice::from_raw_parts(out, ROWS).to_vec();

        let max_abs = y_gpu
            .iter()
            .zip(&y_ref)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let tol = 1e-3;
        println!("rows={} cols={} group={}", ROWS, COLS, max_ttg.min(256));
        println!("y_gpu[0..4]  = {:?}", &y_gpu[..4]);
        println!("y_ref[0..4]  = {:?}", &y_ref[..4]);
        println!("max_abs = {:.3e}  (tolerance {:.0e})", max_abs, tol);
        if max_abs <= tol {
            println!("METALTEST OK — GPU and CPU agree");
        } else {
            println!("METALTEST FAIL — GPU/CPU mismatch above tolerance");
        }

        msg_void(pool, sel("drain"));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Step 2: persistent context + gpu_matvec routed from model::matvec via --gpu
// ════════════════════════════════════════════════════════════════════════════

/// Persistent Metal context: device, command queue and the matvec pipeline,
/// compiled once and reused for the whole process.
///
/// STEP 3 — on-device weight cache: `cache` holds one Metal buffer per weight
/// matrix, uploaded on first use and reused afterwards. Key = (data pointer,
/// rows, cols). This is valid because every weight slice seen by model::matvec
/// points into `BinFile.data` (a single backing - `Vec<u8>` or read-only mmap
/// - allocated/mapped once at load time and NEVER reallocated or remapped
/// during the session), so a (pointer, dims) pair
/// uniquely and stably identifies a weight matrix; keeping dims in the key
/// also guards against two views sharing a base pointer with different shapes.
/// Cache buffers are retained and live until process exit (intentional;
/// releasing them on exit would be cosmetic).
pub struct MetalCtx {
    device: Id,
    queue: Id,
    pipeline: Id,
    pipeline_fp4: Id, // null if the fp4 pipeline failed to build (fp4 path then stays CPU)
    max_ttg: u64,
    max_ttg_fp4: u64,
    cache: std::sync::Mutex<WeightCache>,
    cache_fp4: std::sync::Mutex<WeightCacheFp4>,
    io: std::sync::Mutex<IoBufs>,
}

/// Pre-allocated device buffers for the per-call i/o (x, y, cols), grown on
/// demand — avoids a buffer allocation per matvec call.
pub struct IoBufs {
    x: (Id, usize),    // (buffer, capacity bytes)
    y: (Id, usize),
    cols: (Id, usize),
}

pub struct WeightCache {
    map: std::collections::HashMap<(usize, usize, usize), Id>, // (ptr, rows, cols) → buffer
    bytes: usize,
    cap_bytes: usize, // device-memory budget; usize::MAX = unlimited (default)
}

/// FP4 weight cache: (packed ptr, rows, cols) → (packed buffer, scales buffer).
/// Same stability argument as the f32 cache (engine weights are slices into
/// the session-stable BinFile.data); the anti-alias check compares raw bytes.
pub struct WeightCacheFp4 {
    map: std::collections::HashMap<(usize, usize, usize), (Id, Id)>,
    bytes: usize,
    cap_bytes: usize,
}

// SAFETY: MTLDevice / MTLCommandQueue are documented by Apple as thread-safe.
// The f32 path is only driven from the single forward-pass thread; the fp4
// path CAN be entered from several pool workers at once (MoE experts run as
// parallel jobs) — those calls serialize on the io mutex, which is held for
// the whole encode/wait/readback section, so no two threads ever share a
// transient Metal object (command buffers/encoders are created fresh per call).
unsafe impl Send for MetalCtx {}
unsafe impl Sync for MetalCtx {}

static CTX: std::sync::OnceLock<Option<MetalCtx>> = std::sync::OnceLock::new();

/// Returns the shared context, initializing it on first use. None if Metal is
/// unavailable or shader/pipeline compilation failed (CPU fallback then).
pub fn gpu_available() -> bool {
    ctx().is_some()
}

fn ctx() -> Option<&'static MetalCtx> {
    CTX.get_or_init(init_ctx).as_ref()
}

fn retain(obj: Id) -> Id {
    if !obj.is_null() {
        // SAFETY: `obj` is a valid Objective-C object; retain returns the same object.
        unsafe { msg_id(obj, sel("retain")) }
    } else {
        obj
    }
}

fn init_ctx() -> Option<MetalCtx> {
    // SAFETY: same invariants as metaltest — all pointers come from the
    // Objective-C runtime; signatures match the Objective-C methods.
    unsafe {
        let pool_cls = class("NSAutoreleasePool");
        let pool = msg_id(msg_id(pool_cls, sel("alloc")), sel("init"));

        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            println!("gpu: MTLCreateSystemDefaultDevice returned nil — GPU disabled, using CPU");
            msg_void(pool, sel("drain"));
            return None;
        }
        let src = ns_string(MATVEC_MSL);
        let mut err: Id = std::ptr::null_mut();
        let library = {
            let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
        };
        if library.is_null() {
            println!("gpu: shader compilation error: {} — GPU disabled, using CPU", err_desc(err));
            msg_void(pool, sel("drain"));
            return None;
        }
        let function = {
            let f: extern "C" fn(Id, Sel, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(library, sel("newFunctionWithName:"), ns_string("matvec"))
        };
        let mut perr: Id = std::ptr::null_mut();
        let pipeline = {
            let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
        };
        if pipeline.is_null() {
            println!("gpu: pipeline creation error: {} — GPU disabled, using CPU", err_desc(perr));
            msg_void(pool, sel("drain"));
            return None;
        }
        let max_ttg = {
            let f: extern "C" fn(Id, Sel) -> u64 =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(pipeline, sel("maxTotalThreadsPerThreadgroup"))
        };
        // fp4 dequant-matvec pipeline (separate MSL source); a failure here
        // only disables the fp4 GPU path, not the f32 one.
        let mut pipeline_fp4: Id = std::ptr::null_mut();
        let mut max_ttg_fp4: u64 = 0;
        {
            let src4 = ns_string(MATVEC_FP4_MSL);
            let mut err4: Id = std::ptr::null_mut();
            let library4 = {
                let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                    std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                f(device, sel("newLibraryWithSource:options:error:"), src4, std::ptr::null_mut(), &mut err4)
            };
            if library4.is_null() {
                println!("gpu: fp4 shader compilation error: {} — fp4 path stays on CPU", err_desc(err4));
            } else {
                let function4 = {
                    let f: extern "C" fn(Id, Sel, Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(library4, sel("newFunctionWithName:"), ns_string("matvec_fp4"))
                };
                let mut perr4: Id = std::ptr::null_mut();
                pipeline_fp4 = {
                    let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(device, sel("newComputePipelineStateWithFunction:error:"), function4, &mut perr4)
                };
                if pipeline_fp4.is_null() {
                    println!("gpu: fp4 pipeline creation error: {} — fp4 path stays on CPU", err_desc(perr4));
                } else {
                    max_ttg_fp4 = {
                        let f: extern "C" fn(Id, Sel) -> u64 =
                            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                        f(pipeline_fp4, sel("maxTotalThreadsPerThreadgroup"))
                    };
                    retain(pipeline_fp4);
                    retain(function4);
                }
                retain(library4);
            }
        }
        let queue = msg_id(device, sel("newCommandQueue"));

        // Device-memory budget for the weight cache. Default: unlimited
        // (microkimi ≈ 2.5 GB, nanokimi ≈ 190 MB — fine in unified memory).
        // Override with MICROKIMI_GPU_CACHE_MB if needed; when the budget is
        // exceeded the matvec falls back to the CPU path (never a hard fail).
        let cap_bytes = std::env::var("MICROKIMI_GPU_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(usize::MAX);

        // The context lives in a static for the whole process: retain every
        // long-lived object so per-call autorelease pools can never free them
        // (intentional, bounded "leak" of a handful of singletons).
        let ctx = MetalCtx {
            device: retain(device),
            queue: retain(queue),
            pipeline: retain(pipeline),
            pipeline_fp4,
            max_ttg,
            max_ttg_fp4,
            cache: std::sync::Mutex::new(WeightCache {
                map: std::collections::HashMap::new(),
                bytes: 0,
                cap_bytes,
            }),
            cache_fp4: std::sync::Mutex::new(WeightCacheFp4 {
                map: std::collections::HashMap::new(),
                bytes: 0,
                cap_bytes,
            }),
            io: std::sync::Mutex::new(IoBufs {
                x: (std::ptr::null_mut(), 0),
                y: (std::ptr::null_mut(), 0),
                cols: (std::ptr::null_mut(), 0),
            }),
        };
        retain(library);
        retain(function);
        let dev_name = msg_id(device, sel("name"));
        println!("gpu: Metal context ready on '{}' (max threads/group {})", utf8(dev_name), max_ttg);
        msg_void(pool, sel("drain"));
        Some(ctx)
    }
}

/// Returns the on-device buffer for weight matrix `w` (rows×cols), uploading
/// it on first use. None if the cache budget would be exceeded or allocation
/// failed (caller then falls back to the CPU path).
///
/// ALIAS SAFETY: the cache key is (data pointer, dims), valid because engine
/// weights are slices into the session-stable BinFile.data. But a pointer can
/// be REUSED by the allocator for a different tensor (proved on M5: gputest's
/// per-tensor Vec<f32> copies — q_proj [512,512] freed, k_proj [512,512]
/// reallocated at the same address → false cache hit → wrong weights). On
/// Apple Silicon `contents` of a shared buffer is plain unified memory, so we
/// cheaply verify the first floats on every hit; on mismatch we re-upload and
/// replace the entry (one-shot warning).
fn weight_buffer(ctx: &MetalCtx, w: &[f32], rows: usize, cols: usize) -> Option<Id> {
    let key = (w.as_ptr() as usize, rows, cols);
    let mut cache = ctx.cache.lock().unwrap();
    if let Some(&buf) = cache.map.get(&key) {
        // verify the cached buffer really holds THIS matrix (anti-aliasing)
        let ncheck = w.len().min(16);
        let cached: &[f32] = unsafe {
            let f: extern "C" fn(Id, Sel) -> *mut c_void =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            let p = f(buf, sel("contents")) as *const f32;
            if p.is_null() {
                &[]
            } else {
                std::slice::from_raw_parts(p, ncheck)
            }
        };
        if !cached.is_empty() && cached == &w[..ncheck] {
            return Some(buf);
        }
        if cached.is_empty() {
            return Some(buf); // cannot verify; trust the key
        }
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("gpu: pointer alias detected in weight cache (reused address) - re-uploading affected tensors");
        }
        cache.bytes -= w.len() * 4;
        cache.map.remove(&key);
        // release the aliased buffer, then fall through to a fresh upload
        unsafe { msg_void(buf, sel("release")) };
    }
    let size = w.len() * 4;
    if cache.bytes + size > cache.cap_bytes {
        return None; // budget exceeded → CPU fallback (by design, never fatal)
    }
    // SAFETY: `w` is valid for the whole call; the buffer is retained and
    // owned by the cache until process exit.
    let buf = unsafe {
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let b = f(
            ctx.device,
            sel("newBufferWithBytes:length:options:"),
            w.as_ptr() as *const c_void,
            size as u64,
            0,
        );
        if !b.is_null() {
            retain(b);
        }
        b
    };
    if buf.is_null() {
        return None;
    }
    cache.map.insert(key, buf);
    cache.bytes += size;
    Some(buf)
}

// ── GPU profiling (MICROKIMI_GPU_DEBUG=1) ──


#[derive(Default)]
pub struct GpuProf {
    pub calls: u64,
    pub hits: u64,
    pub misses: u64,
    pub t_cache_ms: f64,
    pub t_io_ms: f64,
    pub t_encode_ms: f64,
    pub t_wait_ms: f64,
    pub t_readback_ms: f64,
    pub printed: u64,
}

static GPU_PROF: std::sync::Mutex<Option<GpuProf>> = std::sync::Mutex::new(None);

fn gpu_debug() -> bool {
    static DBG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DBG.get_or_init(|| std::env::var("MICROKIMI_GPU_DEBUG").map(|v| v == "1").unwrap_or(false))
}

fn prof() -> std::sync::MutexGuard<'static, Option<GpuProf>> {
    GPU_PROF.lock().unwrap()
}

/// Prints the accumulated GPU profile (called at end of run when --gpu is on).
pub fn gpu_prof_print() {
    let guard = prof();
    let Some(p) = guard.as_ref() else { return };
    if p.calls == 0 {
        return;
    }
    let n = p.calls as f64;
    println!("gpu profile: {} matvec calls | weight cache: {} hits, {} misses ({:.1}% hit rate)",
        p.calls, p.hits, p.misses, 100.0 * p.hits as f64 / (p.hits + p.misses).max(1) as f64);
    println!(
        "  avg/call: cache {:.3} ms | io {:.3} ms | encode {:.3} ms | waitUntilCompleted {:.3} ms | readback {:.3} ms",
        p.t_cache_ms / n, p.t_io_ms / n, p.t_encode_ms / n, p.t_wait_ms / n, p.t_readback_ms / n
    );
    if let Some(ctx) = ctx() {
        let cache = ctx.cache.lock().unwrap();
        println!(
            "  weight cache: {} tensors, {:.1} MB on device",
            cache.map.len(),
            cache.bytes as f64 / 1e6
        );
    }
}

/// Grows an i/o device buffer to at least `bytes` (creates it on first use),
/// returns the buffer and its contents pointer.
/// SAFETY: caller must hold the io mutex; returned pointer is valid until the
/// buffer is next reallocated.
unsafe fn ensure_buf(ctx: &MetalCtx, slot: &mut (Id, usize), bytes: usize) -> (*mut c_void, Id) {
    if slot.1 < bytes {
        let f: extern "C" fn(Id, Sel, u64, u64) -> Id =
            unsafe { std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend) };
        let b = f(ctx.device, sel("newBufferWithLength:options:"), bytes as u64, 0);
        if !b.is_null() {
            retain(b);
            if !slot.0.is_null() {
                // release the old, too-small buffer (it is retained by us)
                unsafe { msg_void(slot.0, sel("release")) };
            }
            *slot = (b, bytes);
        } else if slot.0.is_null() {
            return (std::ptr::null_mut(), std::ptr::null_mut());
        }
        // if realloc fails but an old buffer exists, fall through and use it
        // only if it still fits (it doesn't) → caller must check
    }
    let f: extern "C" fn(Id, Sel) -> *mut c_void =
        unsafe { std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend) };
    (f(slot.0, sel("contents")), slot.0)
}

/// Matrix × vector on the GPU: y = W · x (W is rows×cols row-major).
/// Called from model::matvec when --gpu is set and rows*cols >= the threshold.
/// Step 3: the weight matrix is uploaded ONCE and reused (see weight_buffer);
/// only x (tiny) is uploaded and y read back per call, into pre-allocated i/o
/// buffers. Set MICROKIMI_GPU_DEBUG=1 to get per-call phase timings.
pub fn gpu_matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    let t0 = std::time::Instant::now();
    let Some(ctx) = ctx() else {
        // Defensive fallback: callers check gpu_available(), but never leave
        // the caller without a result.
        crate::model::matvec_cpu(w, rows, cols, x, y);
        return;
    };
    assert_eq!(w.len(), rows * cols);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);

    let hit = ctx.cache.lock().unwrap().map.contains_key(&(w.as_ptr() as usize, rows, cols));
    let Some(buf_w) = weight_buffer(ctx, w, rows, cols) else {
        crate::model::matvec_cpu(w, rows, cols, x, y);
        return;
    };
    let t_cache = t0.elapsed();

    // SAFETY: all Metal objects come from the live context or the weight
    // cache (retained, alive until process exit); x/y/cols buffers are
    // pre-allocated per context and grown on demand; waitUntilCompleted
    // guarantees the GPU is done before `contents` is read back. A fresh
    // autorelease pool per call frees transient command buffers/encoders.
    let (enc_ms, wait_ms, io_ms, readback_ms) = unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let t_io = std::time::Instant::now();
        let mut io = ctx.io.lock().unwrap();
        let (x_ptr, buf_x) = ensure_buf(ctx, &mut io.x, x.len() * 4);
        let (y_ptr, buf_y) = ensure_buf(ctx, &mut io.y, y.len() * 4);
        let (c_ptr, buf_cols) = ensure_buf(ctx, &mut io.cols, 4);
        if buf_x.is_null() || buf_y.is_null() || buf_cols.is_null() || x_ptr.is_null() || c_ptr.is_null() || y_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            println!("gpu: i/o buffer allocation failed — falling back to CPU for this matvec");
            crate::model::matvec_cpu(w, rows, cols, x, y);
            return;
        }
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const c_void, x_ptr, x.len() * 4);
        std::ptr::copy_nonoverlapping((&(cols as u32)) as *const u32 as *const c_void, c_ptr, 4);
        let io_ms = t_io.elapsed();

        let t_enc = std::time::Instant::now();
        let cmdbuf = msg_id(ctx.queue, sel("commandBuffer"));
        let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));
        {
            let f: extern "C" fn(Id, Sel, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setComputePipelineState:"), ctx.pipeline);
        }
        {
            let f: extern "C" fn(Id, Sel, Id, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_w, 0, 0);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_x, 0, 1);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_y, 0, 2);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_cols, 0, 3);
        }
        {
            // threadgroup-per-row kernel: GROUP threads per row (multiple of
            // 32 for simdgroup reductions), grid = rows × GROUP threads.
            let mut group = ctx.max_ttg.min(256);
            group = (group / 32).max(1) * 32;
            let f: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                encoder,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { width: rows as u64 * group, height: 1, depth: 1 },
                MTLSize { width: group, height: 1, depth: 1 },
            );
        }
        msg_void(encoder, sel("endEncoding"));
        msg_void(cmdbuf, sel("commit"));
        let enc_ms = t_enc.elapsed();

        let t_wait = std::time::Instant::now();
        msg_void(cmdbuf, sel("waitUntilCompleted"));
        let wait_ms = t_wait.elapsed();

        let t_rb = std::time::Instant::now();
        y.copy_from_slice(std::slice::from_raw_parts(y_ptr as *const f32, rows));
        let readback_ms = t_rb.elapsed();
        drop(io);
        msg_void(pool, sel("drain"));
        (enc_ms, wait_ms, io_ms, readback_ms)
    };

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    {
        let mut g = prof();
        let p = g.get_or_insert_with(GpuProf::default);
        p.calls += 1;
        if hit {
            p.hits += 1;
        } else {
            p.misses += 1;
        }
        p.t_cache_ms += t_cache.as_secs_f64() * 1000.0;
        p.t_io_ms += io_ms.as_secs_f64() * 1000.0;
        p.t_encode_ms += enc_ms.as_secs_f64() * 1000.0;
        p.t_wait_ms += wait_ms.as_secs_f64() * 1000.0;
        p.t_readback_ms += readback_ms.as_secs_f64() * 1000.0;
        if gpu_debug() && p.printed < 10 {
            p.printed += 1;
            println!(
                "gpu#{:03} [{}x{}] {} | cache {:.3} ms | io {:.3} ms | encode {:.3} ms | wait {:.3} ms | readback {:.3} ms | total {:.3} ms",
                p.calls, rows, cols, if hit { "HIT " } else { "MISS" },
                t_cache.as_secs_f64() * 1000.0, io_ms.as_secs_f64() * 1000.0,
                enc_ms.as_secs_f64() * 1000.0, wait_ms.as_secs_f64() * 1000.0,
                readback_ms.as_secs_f64() * 1000.0, total_ms
            );
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// fp4 (e2m1/ue8m0) dequant-matvec on GPU — mxfp4::matvec_packed's counterpart
// ════════════════════════════════════════════════════════════════════════════

/// Returns the on-device buffers (packed, scales) for an fp4 weight matrix,
/// uploading them on first use. None on budget overflow or allocation failure
/// (caller falls back to the CPU path). Same aliasing defense as the f32
/// cache: the first bytes are verified on every hit, a mismatch re-uploads.
fn weight_buffer_fp4(ctx: &MetalCtx, packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Option<(Id, Id)> {
    let key = (packed.as_ptr() as usize, rows, cols);
    let mut cache = ctx.cache_fp4.lock().unwrap();
    if let Some(&(bp, bs)) = cache.map.get(&key) {
        // verify the cached buffers really hold THIS matrix (anti-aliasing)
        let ok = unsafe {
            let f: extern "C" fn(Id, Sel) -> *mut c_void =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            let pp = f(bp, sel("contents")) as *const u8;
            let sp = f(bs, sel("contents")) as *const u8;
            if pp.is_null() || sp.is_null() {
                true // cannot verify; trust the key
            } else {
                let np = packed.len().min(16);
                let ns = scales.len().min(4);
                std::slice::from_raw_parts(pp, np) == &packed[..np]
                    && std::slice::from_raw_parts(sp, ns) == &scales[..ns]
            }
        };
        if ok {
            return Some((bp, bs));
        }
        static WARNED_FP4: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED_FP4.swap(true, std::sync::atomic::Ordering::Relaxed) {
            println!("gpu: pointer alias detected in fp4 weight cache (reused address) - re-uploading affected tensors");
        }
        cache.bytes -= packed.len() + scales.len();
        cache.map.remove(&key);
        unsafe {
            msg_void(bp, sel("release"));
            msg_void(bs, sel("release"));
        }
    }
    let size = packed.len() + scales.len();
    if cache.bytes + size > cache.cap_bytes {
        return None; // budget exceeded → CPU fallback (by design, never fatal)
    }
    // SAFETY: packed/scales are valid for the whole call; the buffers are
    // retained and owned by the cache until process exit.
    let (bp, bs) = unsafe {
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let bp = f(ctx.device, sel("newBufferWithBytes:length:options:"), packed.as_ptr() as *const c_void, packed.len() as u64, 0);
        let bs = f(ctx.device, sel("newBufferWithBytes:length:options:"), scales.as_ptr() as *const c_void, scales.len() as u64, 0);
        (bp, bs)
    };
    if bp.is_null() || bs.is_null() {
        unsafe {
            if !bp.is_null() {
                msg_void(bp, sel("release"));
            }
            if !bs.is_null() {
                msg_void(bs, sel("release"));
            }
        }
        return None;
    }
    retain(bp);
    retain(bs);
    cache.map.insert(key, (bp, bs));
    cache.bytes += size;
    Some((bp, bs))
}

/// FP4 matrix × vector on the GPU: y = dequant(W) · x, fused (weights stay
/// packed on device). Called from mxfp4::matvec_packed when --gpu is set and
/// rows*cols >= GPU_MIN_ELEMS — and directly by dstest at any size.
/// `packed` is [rows, cols/2] (low nibble = even column), `scales` is
/// [rows, cols/32] (ue8m0), cols a multiple of 32 — the mxfp4 invariants.
pub fn gpu_matvec_fp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
    assert_eq!(cols % 32, 0, "fp4: cols must be a multiple of 32");
    assert_eq!(packed.len(), rows * cols / 2);
    assert_eq!(scales.len(), rows * cols / 32);
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    let Some(ctx) = ctx() else {
        crate::quant::mxfp4::matvec_packed(packed, scales, rows, cols, x, y, 1);
        return;
    };
    if ctx.pipeline_fp4.is_null() {
        crate::quant::mxfp4::matvec_packed(packed, scales, rows, cols, x, y, 1);
        return;
    }
    let Some((buf_p, buf_s)) = weight_buffer_fp4(ctx, packed, scales, rows, cols) else {
        crate::quant::mxfp4::matvec_packed(packed, scales, rows, cols, x, y, 1);
        return;
    };

    // SAFETY: same invariants as gpu_matvec — all Metal objects come from the
    // live context or the retained caches; the io mutex serializes concurrent
    // callers (MoE expert pool jobs); waitUntilCompleted before readback; a
    // fresh autorelease pool per call frees transient command buffers/encoders.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = ctx.io.lock().unwrap();
        let (x_ptr, buf_x) = ensure_buf(ctx, &mut io.x, x.len() * 4);
        let (y_ptr, buf_y) = ensure_buf(ctx, &mut io.y, y.len() * 4);
        let (c_ptr, buf_cols) = ensure_buf(ctx, &mut io.cols, 4);
        if buf_x.is_null() || buf_y.is_null() || buf_cols.is_null() || x_ptr.is_null() || y_ptr.is_null() || c_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            println!("gpu: fp4 i/o buffer allocation failed — falling back to CPU for this matvec");
            crate::quant::mxfp4::matvec_packed(packed, scales, rows, cols, x, y, 1);
            return;
        }
        std::ptr::copy_nonoverlapping(x.as_ptr() as *const c_void, x_ptr, x.len() * 4);
        std::ptr::copy_nonoverlapping((&(cols as u32)) as *const u32 as *const c_void, c_ptr, 4);

        let cmdbuf = msg_id(ctx.queue, sel("commandBuffer"));
        let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));
        {
            let f: extern "C" fn(Id, Sel, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setComputePipelineState:"), ctx.pipeline_fp4);
        }
        {
            let f: extern "C" fn(Id, Sel, Id, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_p, 0, 0);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_s, 0, 1);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_x, 0, 2);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_y, 0, 3);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_cols, 0, 4);
        }
        {
            let mut group = ctx.max_ttg_fp4.min(256);
            group = (group / 32).max(1) * 32;
            let f: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                encoder,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { width: rows as u64 * group, height: 1, depth: 1 },
                MTLSize { width: group, height: 1, depth: 1 },
            );
        }
        msg_void(encoder, sel("endEncoding"));
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        y.copy_from_slice(std::slice::from_raw_parts(y_ptr as *const f32, rows));
        drop(io);
        msg_void(pool, sel("drain"));
    }
}

// ════════════════════════════════════════════════════════════════════════════
// gputest: real model matvecs CPU vs GPU
// ════════════════════════════════════════════════════════════════════════════

/// Loads the local model binary and compares a spread of REAL weight matrices
/// (KDA / MLA / MoE / dense projections + lm_head) between the CPU matvec and
/// the GPU one. Tolerance 1e-3 (max_abs and scale-relative).
pub fn gputest() {
    if !gpu_available() {
        println!("gputest FAIL — no usable Metal context (see message above)");
        return;
    }
    let path = crate::bin_path();
    println!("gputest — real model matvecs, CPU vs GPU ({})", path);
    let bin = crate::quant::weights::BinFile::open(&path);

    // Tensors present in BOTH microkimi-debug.bin and nanokimi-0.2b.bin (any config):
    // a KDA projection, an MLA projection, the MoE router, a routed projection,
    // the dense MLP of layer 0, and lm_head (the big one).
    let names = [
        "layers.0.self_attn.q_proj.weight",
        "layers.1.self_attn.k_proj.weight",
        "layers.3.self_attn.q_b_proj.weight",
        "layers.1.block_sparse_moe.gate.weight",
        "layers.1.block_sparse_moe.routed_expert_up_proj.weight",
        "layers.0.mlp.gate_proj.weight",
        "lm_head.weight",
    ];

    let mut all_ok = true;
    let mut keep_alive: Vec<Vec<f32>> = Vec::new(); // weights stay alive: the cache
    // keys on (ptr, dims) — dropping a copy would let the allocator reuse its
    // address for the next tensor (proved: q_proj vs k_proj, both [512,512]).
    for name in names {
        let Some(e) = bin.entries.get(name) else {
            println!("  {:<48} SKIP (not in this model)", name);
            continue;
        };
        let (rows, cols) = (e.dims[0] as usize, e.dims[1] as usize);
        keep_alive.push(bin.f32_vec(name));
        let w = keep_alive.last().unwrap();
        let x: Vec<f32> = (0..cols).map(|i| pattern(i + 777)).collect();
        let mut y_cpu = vec![0f32; rows];
        crate::model::matvec_cpu(&w, rows, cols, &x, &mut y_cpu);
        let mut y_gpu = vec![0f32; rows];
        gpu_matvec(&w, rows, cols, &x, &mut y_gpu);

        let scale = y_cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
        let max_abs = y_gpu
            .iter()
            .zip(&y_cpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let rel = max_abs / scale;
        let ok = rel <= 1e-3;
        all_ok &= ok;
        println!(
            "  {:<48} [{}x{}] max_abs={:.3e} rel={:.3e}  {}",
            name, rows, cols, max_abs, rel, if ok { "OK" } else { "FAIL" }
        );
    }
    if all_ok {
        println!("GPUTEST OK — GPU matches CPU on real model matvecs (tol 1e-3)");
    } else {
        println!("GPUTEST FAIL");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// dstest: fp4 dequant-matvec Metal vs CPU (microdeepseek experts)
// ════════════════════════════════════════════════════════════════════════════

/// Compares the fused fp4 kernel against the CPU mxfp4::matvec_packed on:
/// 1. synthetic matrices at micro dims, real V4 dims and edge shapes,
/// 2. real expert blobs from microdeepseek-debug.bin (SKIPPED if the bin is absent),
/// 3. the DeepSeek lm_head routing through model::matvec (66M ≥ GPU_MIN_ELEMS).
/// Tolerance 1e-3 (max_abs and scale-relative) — the kernel reassociates the
/// f32 sums differently than the sequential CPU path.
pub fn dstest() {
    if !gpu_available() {
        println!("dstest FAIL — no usable Metal context (see message above)");
        return;
    }
    println!("dstest — fp4 dequant-matvec, Metal vs CPU (mxfp4 layout, tol 1e-3)");
    let mut all_ok = true;

    let check = |label: String, packed: &[u8], scales: &[u8], rows: usize, cols: usize| -> bool {
        let x: Vec<f32> = (0..cols).map(|i| pattern(i + 4242)).collect();
        let mut y_cpu = vec![0f32; rows];
        crate::quant::mxfp4::matvec_packed(packed, scales, rows, cols, &x, &mut y_cpu, 1);
        let mut y_gpu = vec![0f32; rows];
        gpu_matvec_fp4(packed, scales, rows, cols, &x, &mut y_gpu);
        let scale = y_cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
        let max_abs = y_gpu.iter().zip(&y_cpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let rel = max_abs / scale;
        let ok = rel <= 1e-3;
        println!(
            "  {:<44} [{}x{}] max_abs={:.3e} rel={:.3e}  {}",
            label, rows, cols, max_abs, rel, if ok { "OK" } else { "FAIL" }
        );
        ok
    };

    // 1) synthetic: micro dims, edge shapes, real V4 dims (2048x4096 experts)
    let mut keep_alive: Vec<(Vec<u8>, Vec<u8>)> = Vec::new(); // cache keys on ptr:
    // keep every blob alive so the allocator can never reuse its address
    for (rows, cols) in [
        (128usize, 512usize),
        (512, 128),
        (64, 128),
        (3, 64),
        (1, 32),
        (256, 1024),
        (2048, 4096),
        (4096, 2048),
    ] {
        let w: Vec<f32> = (0..rows * cols).map(pattern).collect();
        keep_alive.push(crate::quant::mxfp4::quantize(&w, rows, cols));
        let (p, s) = &keep_alive[keep_alive.len() - 1];
        all_ok &= check(format!("synthetic [{rows}x{cols}]"), p, s, rows, cols);
    }

    // 1b) routing proof: with --gpu, a ≥ GPU_MIN_ELEMS fp4 matvec must land in
    // the fp4 device cache; a sub-threshold one must NOT.
    {
        crate::model::set_gpu(true);
        let (rows, cols) = (2048usize, 4096usize); // 8.4M params ≥ 2M → GPU
        let w: Vec<f32> = (0..rows * cols).map(|i| pattern(i + 77)).collect();
        let (p, s) = crate::quant::mxfp4::quantize(&w, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| pattern(i + 88)).collect();
        let mut y_routed = vec![0f32; rows];
        crate::quant::mxfp4::matvec_packed(&p, &s, rows, cols, &x, &mut y_routed, 1);
        let on_device = ctx()
            .map(|c| c.cache_fp4.lock().unwrap().map.contains_key(&(p.as_ptr() as usize, rows, cols)))
            .unwrap_or(false);
        // sub-threshold: 128x512 = 65K params → must stay on the CPU
        let (r2, c2) = (128usize, 512usize);
        let w2: Vec<f32> = (0..r2 * c2).map(|i| pattern(i + 99)).collect();
        let (p2, s2) = crate::quant::mxfp4::quantize(&w2, r2, c2);
        let x2: Vec<f32> = (0..c2).map(|i| pattern(i + 111)).collect();
        let mut y2 = vec![0f32; r2];
        crate::quant::mxfp4::matvec_packed(&p2, &s2, r2, c2, &x2, &mut y2, 1);
        let small_on_device = ctx()
            .map(|c| c.cache_fp4.lock().unwrap().map.contains_key(&(p2.as_ptr() as usize, r2, c2)))
            .unwrap_or(false);
        crate::model::set_gpu(false);
        let mut y_ref = vec![0f32; rows];
        crate::quant::mxfp4::matvec_packed(&p, &s, rows, cols, &x, &mut y_ref, 1);
        let scale = y_ref.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
        let max_abs = y_routed.iter().zip(&y_ref).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let ok = on_device && !small_on_device && max_abs / scale <= 1e-3;
        all_ok &= ok;
        println!(
            "  {:<44} max_abs={:.3e} big_on_gpu={} small_on_gpu={}  {}",
            "matvec_packed routing (--gpu, threshold 2M)", max_abs, on_device, small_on_device,
            if ok { "OK" } else { "FAIL" }
        );
    }

    // 2) real expert blobs + 3) lm_head routing (need microdeepseek-debug.bin)
    let ds_path = ["microdeepseek-debug.bin", "microdeepseek.bin"]
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(|s| s.to_string());
    match ds_path {
        Some(path) => {
            println!("  model: {}", path);
            let bin = crate::quant::weights::BinFile::open(&path);
            if bin.config.ds.is_none() {
                println!("  {} is not a deepseek_v4 model — real-blob checks SKIPPED", path);
            } else {
                for name in [
                    "layers.0.ffn.experts.0.w1",
                    "layers.0.ffn.experts.0.w2",
                    "layers.5.ffn.experts.7.w3",
                    "layers.42.ffn.experts.255.w1",
                ] {
                    let (p, s, rows, cols) = bin.mxfp4_parts(name);
                    all_ok &= check(name.to_string(), p, s, rows, cols);
                }
                // lm_head through model::matvec with --gpu: 129280x512 = 66M
                // MACs ≥ GPU_MIN_ELEMS → must land on the GPU (cache proof).
                let head = bin.f32_vec("head.weight");
                let (rows, cols) = (bin.entries["head.weight"].dims[0] as usize, bin.entries["head.weight"].dims[1] as usize);
                let x: Vec<f32> = (0..cols).map(|i| pattern(i + 31337)).collect();
                let mut y_cpu = vec![0f32; rows];
                crate::model::matvec_cpu(&head, rows, cols, &x, &mut y_cpu);
                crate::model::set_gpu(true);
                let mut y_gpu = vec![0f32; rows];
                crate::model::matvec(&head, rows, cols, &x, &mut y_gpu);
                crate::model::set_gpu(false);
                let on_device = ctx()
                    .map(|c| c.cache.lock().unwrap().map.contains_key(&(head.as_ptr() as usize, rows, cols)))
                    .unwrap_or(false);
                let scale = y_cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
                let max_abs = y_gpu.iter().zip(&y_cpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                let ok = max_abs / scale <= 1e-3 && on_device;
                all_ok &= ok;
                println!(
                    "  {:<44} [{}x{}] max_abs={:.3e} routed_to_gpu={}  {}",
                    "head.weight via model::matvec --gpu", rows, cols, max_abs, on_device,
                    if ok { "OK" } else { "FAIL" }
                );
            }
        }
        None => println!("  microdeepseek-debug.bin not found — real-expert and lm_head checks SKIPPED"),
    }

    if let Some(c) = ctx() {
        let cache = c.cache_fp4.lock().unwrap();
        println!(
            "  fp4 weight cache: {} matrices, {:.1} MB on device",
            cache.map.len(),
            cache.bytes as f64 / 1e6
        );
    }
    if all_ok {
        println!("DSTEST OK — fp4 GPU kernel matches the CPU path (tol 1e-3)");
    } else {
        println!("DSTEST FAIL");
    }
}

// ════════════════════════════════════════════════════════════════════════════
// metaltest-packed: packed mxfp4 GPU kernel vs CPU on REAL K3 expert blobs
// ════════════════════════════════════════════════════════════════════════════

/// Compares the fused packed fp4 kernel (matvec_fp4, weights uploaded as
/// PACKED BYTES - no fp32 staging) against the CPU mxfp4::matvec_packed on:
/// 1. real routed-expert blobs of the local K3 model (nanokimi /
///    microkimi-debug: layers.N.block_sparse_moe.experts.E.w{1,2,3}),
/// 2. synthetic shapes at micro and real V4 dims (2048x4096),
/// 3. the ue8m0 subnormal path (an all-zero block -> scale byte 0).
/// dstest covers the DeepSeek models; this one covers the K3 expert layout.
/// Tolerance 1e-3 relative - the kernel scales per element (lut*s*x) where
/// the CPU scales per group of 32, an f32 reassociation, plus the
/// implementation-defined simd_sum reduction order (bounded host-side by
/// selftest's PACKED-EMUL section).
pub fn metaltest_packed() {
    if !gpu_available() {
        println!("metaltest-packed FAIL — no usable Metal context (see message above)");
        return;
    }
    let Some(c) = ctx() else { return };
    if c.pipeline_fp4.is_null() {
        println!("metaltest-packed FAIL — fp4 pipeline unavailable (see message above)");
        return;
    }
    println!("metaltest-packed — packed mxfp4 Metal kernel vs CPU (K3 experts, tol 1e-3)");
    let mut all_ok = true;

    let check = |label: String, packed: &[u8], scales: &[u8], rows: usize, cols: usize| -> bool {
        let x: Vec<f32> = (0..cols).map(|i| pattern(i + 4242)).collect();
        let mut y_cpu = vec![0f32; rows];
        crate::quant::mxfp4::matvec_packed(packed, scales, rows, cols, &x, &mut y_cpu, 1);
        let mut y_gpu = vec![0f32; rows];
        gpu_matvec_fp4(packed, scales, rows, cols, &x, &mut y_gpu);
        let scale = y_cpu.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
        let max_abs = y_gpu.iter().zip(&y_cpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let rel = max_abs / scale;
        let ok = rel <= 1e-3;
        println!(
            "  {:<52} [{}x{}] max_abs={:.3e} rel={:.3e}  {}",
            label, rows, cols, max_abs, rel, if ok { "OK" } else { "FAIL" }
        );
        ok
    };

    // 1) real K3 routed-expert blobs from the default model (SKIPPED per
    // tensor if absent: microkimi-debug and nanokimi share the naming, dims
    // differ). keep_alive pins every blob: the fp4 cache keys on (ptr, dims).
    let path = crate::bin_path();
    println!("  model: {}", path);
    let bin = crate::quant::weights::BinFile::open(&path);
    let names = [
        "layers.1.block_sparse_moe.experts.0.w1",
        "layers.1.block_sparse_moe.experts.0.w2",
        "layers.1.block_sparse_moe.experts.0.w3",
        "layers.2.block_sparse_moe.experts.17.w1",
        "layers.5.block_sparse_moe.experts.511.w2",
        "layers.7.block_sparse_moe.experts.895.w3",
    ];
    let mut keep_alive: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    for name in names {
        let Some(e) = bin.entries.get(name) else {
            println!("  {:<52} SKIP (not in this model)", name);
            continue;
        };
        let (rows, cols) = (e.dims[0] as usize, e.dims[1] as usize);
        let (p, s, _, _) = bin.mxfp4_parts(name);
        keep_alive.push((p.to_vec(), s.to_vec()));
        let (p, s) = &keep_alive[keep_alive.len() - 1];
        all_ok &= check(name.to_string(), p, s, rows, cols);
    }

    // 2) synthetic: micro dims, edge shapes, real V4 expert dims
    for (rows, cols) in [(128usize, 512usize), (3, 64), (1, 32), (2048, 4096), (4096, 2048)] {
        let w: Vec<f32> = (0..rows * cols).map(pattern).collect();
        keep_alive.push(crate::quant::mxfp4::quantize(&w, rows, cols));
        let (p, s) = &keep_alive[keep_alive.len() - 1];
        all_ok &= check(format!("synthetic [{rows}x{cols}]"), p, s, rows, cols);
    }

    // 3) ue8m0 subnormal path: an all-zero block -> scale byte 0 -> 2^-127
    {
        let (rows, cols) = (64usize, 128usize);
        let mut w: Vec<f32> = (0..rows * cols).map(pattern).collect();
        for v in w[2 * cols..4 * cols].iter_mut() {
            *v = 0.0;
        }
        keep_alive.push(crate::quant::mxfp4::quantize(&w, rows, cols));
        let (p, s) = &keep_alive[keep_alive.len() - 1];
        assert!(s[2 * cols / 32] == 0, "zero block must produce scale byte 0");
        all_ok &= check("zero block (scale byte 0)".to_string(), p, s, rows, cols);
    }

    if all_ok {
        println!("METALTEST-PACKED OK — packed GPU kernel matches the CPU path on K3 experts (tol 1e-3)");
    } else {
        println!("METALTEST-PACKED FAIL");
    }
}
