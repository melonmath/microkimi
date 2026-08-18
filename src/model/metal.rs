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

// e2m1 value table (sign x 8 magnitudes) - program scope: newer Metal
// compilers reject constant-qualified automatic variables
constant float FP4_LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                              -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

kernel void matvec_fp4(device const uchar* packed [[buffer(0)]],
                       device const uchar* scales [[buffer(1)]],
                       device const float* x      [[buffer(2)]],
                       device float*       y      [[buffer(3)]],
                       constant uint&    cols   [[buffer(4)]],
                       uint row  [[threadgroup_position_in_grid]],
                       uint lane [[thread_position_in_threadgroup]],
                       uint lanes [[threads_per_threadgroup]]) {
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
        acc += FP4_LUT[nib] * s * x[c];
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
    z: (Id, usize),    // second input for the batched GEMMs (attention K/V)
    st: (Id, usize),   // read-write state for the delta-scan kernel
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
                z: (std::ptr::null_mut(), 0),
                st: (std::ptr::null_mut(), 0),
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
    let ds_path = ["models/microdeepseek-debug.bin", "models/microdeepseek.bin"]
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

// ════════════════════════════════════════════════════════════════════════════
// MPS GEMM: Qwen batched-prefill offload (MICROKIMI_QWEN_GPU=1)
// ════════════════════════════════════════════════════════════════════════════
//
// The decode regime cannot win on this path: each Metal dispatch costs
// ~0.25 ms of sync latency and a single token walks ~100 matvecs, so
// per-op offload would cost more than the whole CPU token. The batched
// prefill is the opposite regime: ONE GEMM per weight matrix covers the
// whole prompt, so the sync cost amortizes to ~0.03 ms/token at 1k
// tokens while the arithmetic moves to the GPU. The nonlinear tissue
// between matmuls (norms, conv, delta scan, softmax, activations) stays
// on the CPU; activations round-trip through unified memory, which is
// cheap next to the weight traffic the GPU absorbs.
//
// The GEMM itself is MPSMatrixMultiplication (Metal Performance
// Shaders), reached through the same objc_msgSend FFI as the rest of
// this file - still zero crates. Result layout: Y = X · Wᵀ with X as
// [t, cols] row-major, so each token's output row lands contiguous.
//
// MXFP4 weights are dequantized to f32 once at first use and cached on
// device (the prefill regime is compute-bound on the GPU, so the 8x
// traffic increase against packed nibbles is paid from spare bandwidth;
// ~1 GB for the 0.8B MLP stack in unified memory).

#[link(name = "MetalPerformanceShaders", kind = "framework")]
unsafe extern "C" {
    fn MPSSupportsMTLDevice(device: Id) -> bool;
}

const MPS_FLOAT32: u32 = 0x10000020; // MPSDataTypeFloatBit | 32

/// Below this lane count the caller is lane-batched decode or a tiny
/// batch: the per-op sync latency dominates, stay on the CPU kernels.
pub const GEMM_MIN_T: usize = 16;
/// Below this weight size a dispatch is not worth its latency.
pub const GEMM_MIN_ELEMS: usize = 1 << 20;
/// Staging-buffer ceiling for one GEMM result (guards the all-logits
/// lm_head case: t x vocab would want hundreds of MB).
const GEMM_MAX_OUT_BYTES: usize = 256 * 1024 * 1024;

static QWEN_GPU: std::sync::OnceLock<std::sync::atomic::AtomicBool> = std::sync::OnceLock::new();

fn qwen_gpu_flag() -> &'static std::sync::atomic::AtomicBool {
    QWEN_GPU.get_or_init(|| {
        std::sync::atomic::AtomicBool::new(
            std::env::var("MICROKIMI_QWEN_GPU").map(|v| v == "1").unwrap_or(false),
        )
    })
}

/// True when the Qwen GEMM offload is armed (env MICROKIMI_QWEN_GPU=1,
/// or set_qwen_gpu for in-process A/B benchmarks).
pub fn qwen_gpu_on() -> bool {
    qwen_gpu_flag().load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_qwen_gpu(on: bool) {
    qwen_gpu_flag().store(on, std::sync::atomic::Ordering::Relaxed);
}

/// True when single-token decode runs on the GPU: the offload switch is
/// on and MICROKIMI_QWEN_GPU_NODECODE=1 is not set (the A/B arm that keeps
/// decode on the CPU while the prefill offloads).
pub fn qwen_gpu_decode_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    qwen_gpu_on()
        && *ON.get_or_init(|| std::env::var("MICROKIMI_QWEN_GPU_NODECODE").map(|v| v != "1").unwrap_or(true))
}

struct MpsCtx {
    // (m, n, k, transpose_b, alpha bits) → retained MPSMatrixMultiplication
    // kernel (the batch count lives in the matrix descriptors, not here)
    gemms: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize, bool, u32), Id>>,
    // (packed ptr, rows, cols) → (retained device buffer of the dequantized
    // matrix - f16 or f32 per the process-constant mode - and the first
    // packed bytes as alias tag)
    dequant: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), (Id, [u8; 16])>>,
    // (f32 ptr, rows, cols) → (retained f16 device copy, first f32 bits as tag)
    w16: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), (Id, u32)>>,
    // (f32 ptr, rows, cols) → (retained int8 rows, retained f16 block scales, tag)
    wq8: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), (Id, Id, u32)>>,
    // (packed ptr, rows, cols) → (retained raw nibble bytes, retained scale bytes, tag)
    fp4raw: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), (Id, Id, [u8; 16])>>,
}

// SAFETY: same argument as MetalCtx - kernel objects and buffers are
// device-owned, all mutation goes through the mutexes, and encoding is
// serialized on the MetalCtx io mutex.
unsafe impl Send for MpsCtx {}
unsafe impl Sync for MpsCtx {}

static MPS: std::sync::OnceLock<Option<MpsCtx>> = std::sync::OnceLock::new();

fn mps_ctx() -> Option<(&'static MetalCtx, &'static MpsCtx)> {
    let base = ctx()?;
    let mps = MPS
        .get_or_init(|| {
            // SAFETY: class lookups and a device-capability C call only.
            let ok = unsafe { MPSSupportsMTLDevice(base.device) }
                && !class("MPSMatrixMultiplication").is_null()
                && !class("MPSMatrixDescriptor").is_null()
                && !class("MPSMatrix").is_null();
            if !ok {
                println!("gpu: MPS matrix kernels unavailable - qwen prefill stays on CPU");
                return None;
            }
            println!("gpu: MPS GEMM ready (qwen batched-prefill offload)");
            Some(MpsCtx {
                gemms: std::sync::Mutex::new(std::collections::HashMap::new()),
                dequant: std::sync::Mutex::new(std::collections::HashMap::new()),
                w16: std::sync::Mutex::new(std::collections::HashMap::new()),
                wq8: std::sync::Mutex::new(std::collections::HashMap::new()),
                fp4raw: std::sync::Mutex::new(std::collections::HashMap::new()),
            })
        })
        .as_ref()?;
    Some((base, mps))
}

/// True when the full offload stack (Metal device + MPS kernels) is up.
pub fn mps_available() -> bool {
    mps_ctx().is_some()
}

/// Cached MPSMatrixMultiplication for C[m,n] = alpha · A[m,k] · B' where
/// B' is B[n,k]ᵀ when `transpose_b` and B[k,n] otherwise. Batching comes
/// from the matrix descriptors at encode time, not from the kernel.
fn gemm_kernel(base: &MetalCtx, mps: &MpsCtx, m: usize, n: usize, k: usize, transpose_b: bool, alpha: f32) -> Option<Id> {
    let key = (m, n, k, transpose_b, alpha.to_bits());
    let mut cache = mps.gemms.lock().unwrap();
    if let Some(&kern) = cache.get(&key) {
        return Some(kern);
    }
    // SAFETY: alloc/init on a resolved class; the typed signature matches
    // initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:
    // interiorColumns:alpha:beta: (BOOL is one byte on arm64, NSUInteger is
    // u64, alpha/beta are doubles).
    let kernel = unsafe {
        let alloc = msg_id(class("MPSMatrixMultiplication"), sel("alloc"));
        let f: extern "C" fn(Id, Sel, Id, bool, bool, u64, u64, u64, f64, f64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        f(
            alloc,
            sel("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"),
            base.device,
            false,
            transpose_b,
            m as u64,
            n as u64,
            k as u64,
            alpha as f64,
            0.0,
        )
    };
    if kernel.is_null() {
        return None;
    }
    cache.insert(key, kernel); // init gave +1; owned by the cache
    Some(kernel)
}

/// One-shot numeric sanity line, printed on the first successful GEMM:
/// recomputes row 0 of lane 0 on the CPU and reports the relative error.
fn gemm_check_once(kind: &str, w_row0: &[f32], x0: &[f32], got: f32, rows: usize, cols: usize, t: usize) {
    static CHECKED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if CHECKED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let mut acc = 0f64;
    for (a, b) in w_row0.iter().zip(x0) {
        acc += *a as f64 * *b as f64;
    }
    let cpu = acc as f32;
    let scale = cpu.abs().max(1e-6);
    println!(
        "gpu gemm check ({kind}, [{rows}x{cols}] t={t}): row0 cpu {cpu:.6} gpu {got:.6} rel {:.2e}",
        (got - cpu).abs() / scale
    );
}

/// Shared encode/run path: packs `xs` into the staging X buffer, runs the
/// GEMM against `buf_w` (a device-resident [rows, cols] f32 matrix), and
/// scatters Y back into `outs`. Returns false on any allocation failure
/// (caller falls back to the CPU kernels - never fatal).
fn run_gemm(
    base: &MetalCtx,
    mps: &MpsCtx,
    buf_w: Id,
    rows: usize,
    cols: usize,
    xs: &[&[f32]],
    outs: &mut [&mut [f32]],
) -> bool {
    let t = xs.len();
    if t * rows * 4 > GEMM_MAX_OUT_BYTES {
        return false;
    }
    let Some(kernel) = gemm_kernel(base, mps, t, rows, cols, true, 1.0) else {
        return false;
    };
    let t_start = std::time::Instant::now();
    // SAFETY: same invariants as gpu_matvec - Metal objects come from the
    // live context/caches; the io mutex serializes staging-buffer use for
    // the whole encode/wait/readback; waitUntilCompleted precedes readback;
    // a fresh autorelease pool covers the transient objects. The MPSMatrix
    // wrappers are alloc/init-owned and explicitly released below.
    let f16 = gemm_f16_on();
    let (esz, dtype) = if f16 { (2usize, MPS_FLOAT16) } else { (4usize, MPS_FLOAT32) };
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = base.io.lock().unwrap();
        let (x_ptr, buf_x) = ensure_buf(base, &mut io.x, t * cols * esz);
        let (y_ptr, buf_y) = ensure_buf(base, &mut io.y, t * rows * esz);
        if buf_x.is_null() || buf_y.is_null() || x_ptr.is_null() || y_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        for (l, x) in xs.iter().enumerate() {
            debug_assert_eq!(x.len(), cols);
            if f16 {
                let dst = std::slice::from_raw_parts_mut((x_ptr as *mut u16).add(l * cols), cols);
                f32s_to_f16s(x, dst);
            } else {
                std::ptr::copy_nonoverlapping(x.as_ptr(), (x_ptr as *mut f32).add(l * cols), cols);
            }
        }

        let desc = |r: usize, c: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                class("MPSMatrixDescriptor"),
                sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"),
                r as u64,
                c as u64,
                (c * esz) as u64,
                dtype,
            )
        };
        let matrix = |buf: Id, d: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(msg_id(class("MPSMatrix"), sel("alloc")), sel("initWithBuffer:descriptor:"), buf, d)
        };
        let mx = matrix(buf_x, desc(t, cols));
        let mw = matrix(buf_w, desc(rows, cols));
        let my = matrix(buf_y, desc(t, rows));
        if mx.is_null() || mw.is_null() || my.is_null() {
            for m in [mx, mw, my] {
                if !m.is_null() {
                    msg_void(m, sel("release"));
                }
            }
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }

        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        {
            let f: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(kernel, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, mx, mw, my);
        }
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        for (l, out) in outs.iter_mut().enumerate() {
            debug_assert_eq!(out.len(), rows);
            if f16 {
                let src = std::slice::from_raw_parts((y_ptr as *const u16).add(l * rows), rows);
                f16s_to_f32s(src, out);
            } else {
                out.copy_from_slice(std::slice::from_raw_parts((y_ptr as *const f32).add(l * rows), rows));
            }
        }
        for m in [mx, mw, my] {
            msg_void(m, sel("release"));
        }
        drop(io);
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

/// f32 multi-lane matvec on the GPU as one GEMM. Returns false when the
/// offload is unavailable (caller falls back to the CPU kernels).
pub fn gpu_gemm_xwt(w: &[f32], rows: usize, cols: usize, xs: &[&[f32]], outs: &mut [&mut [f32]]) -> bool {
    debug_assert_eq!(w.len(), rows * cols);
    let Some((base, mps)) = mps_ctx() else {
        return false;
    };
    let buf_w = if gemm_f16_on() {
        weight_buffer_f16(base, mps, w, rows, cols)
    } else {
        weight_buffer(base, w, rows, cols)
    };
    let Some(buf_w) = buf_w else {
        return false;
    };
    if !run_gemm(base, mps, buf_w, rows, cols, xs, outs) {
        return false;
    }
    gemm_check_once("f32", &w[..cols], xs[0], outs[0][0], rows, cols, xs.len());
    true
}

/// Device-resident f32 copy of an MXFP4 matrix, dequantized on first use.
/// The alias tag (first packed bytes) guards against allocator address
/// reuse, like the other weight caches.
fn dequant_buffer(base: &MetalCtx, mps: &MpsCtx, packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Option<Id> {
    let key = (packed.as_ptr() as usize, rows, cols);
    let mut tag = [0u8; 16];
    let n = packed.len().min(16);
    tag[..n].copy_from_slice(&packed[..n]);
    let mut cache = mps.dequant.lock().unwrap();
    if let Some(&(buf, seen)) = cache.get(&key) {
        if seen == tag {
            return Some(buf);
        }
        cache.remove(&key);
        // SAFETY: the stale buffer is owned by this cache (retained at insert).
        unsafe { msg_void(buf, sel("release")) };
    }
    let w = crate::quant::mxfp4::dequant(packed, scales, rows, cols);
    // f16 mode stores the dequantized copy in half precision (e2m1 x
    // power-of-two scales convert exactly for every in-range value).
    let (ptr, bytes, _h16);
    if gemm_f16_on() {
        let mut h = vec![0u16; w.len()];
        f32s_to_f16s(&w, &mut h);
        ptr = h.as_ptr() as *const c_void;
        bytes = h.len() * 2;
        _h16 = Some(h);
    } else {
        ptr = w.as_ptr() as *const c_void;
        bytes = w.len() * 4;
        _h16 = None;
    }
    // SAFETY: the source vec is alive for the whole call; the buffer
    // copies its bytes.
    let buf = unsafe {
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let b = f(base.device, sel("newBufferWithBytes:length:options:"), ptr, bytes as u64, 0);
        if !b.is_null() {
            retain(b);
        }
        b
    };
    if buf.is_null() {
        return None;
    }
    cache.insert(key, (buf, tag));
    Some(buf)
}

/// MXFP4 multi-lane matvec on the GPU as one f32 GEMM over the cached
/// dequantized copy. Returns false when the offload is unavailable.
pub fn gpu_gemm_xwt_fp4(packed: &[u8], scales: &[u8], rows: usize, cols: usize, xs: &[&[f32]], outs: &mut [&mut [f32]]) -> bool {
    let Some((base, mps)) = mps_ctx() else {
        return false;
    };
    let Some(buf_w) = dequant_buffer(base, mps, packed, scales, rows, cols) else {
        return false;
    };
    run_gemm(base, mps, buf_w, rows, cols, xs, outs)
}

// ── GEMM time accounting (drives the Amdahl split in qwengpubench) ──

static GEMM_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static GEMM_MICROS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn gemm_account(micros: u64) {
    GEMM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    GEMM_MICROS.fetch_add(micros, std::sync::atomic::Ordering::Relaxed);
}

/// Returns and resets the (calls, milliseconds) spent inside GPU GEMMs
/// since the last take - the difference against wall time is the CPU
/// tissue between matmuls, the phase-2 porting target.
pub fn gemm_stats_take() -> (u64, f64) {
    let calls = GEMM_CALLS.swap(0, std::sync::atomic::Ordering::Relaxed);
    let micros = GEMM_MICROS.swap(0, std::sync::atomic::Ordering::Relaxed);
    (calls, micros as f64 / 1000.0)
}

// ════════════════════════════════════════════════════════════════════════════
// Batched GEMM: the attention offload (scores and mixes for every head in
// one encode each; the causal softmax stays on the CPU between the two)
// ════════════════════════════════════════════════════════════════════════════

/// Batched C[b][m,n] = alpha · A[b][m,k] · B' for every b at once, where
/// B' is B[b][n,k]ᵀ when `transpose_b` and B[b][k,n] otherwise. A, B and
/// C are dense head-major stacks ([batch, rows, cols] contiguous). All
/// staging goes through the shared io buffers (x = A, z = B, y = C).
/// Returns false when the offload is unavailable or the result would
/// exceed the staging ceiling (caller keeps its CPU path).
pub fn gpu_gemm_batched(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    transpose_b: bool,
    alpha: f32,
    out: &mut [f32],
) -> bool {
    assert_eq!(a.len(), batch * m * k);
    assert_eq!(b.len(), batch * n * k);
    assert_eq!(out.len(), batch * m * n);
    if batch * m * n * 4 > GEMM_MAX_OUT_BYTES {
        return false;
    }
    let Some((base, mps)) = mps_ctx() else {
        return false;
    };
    let Some(kernel) = gemm_kernel(base, mps, m, n, k, transpose_b, alpha) else {
        return false;
    };
    let t_start = std::time::Instant::now();
    // SAFETY: same invariants as run_gemm - io mutex serializes staging use
    // for the whole encode/wait/readback, waitUntilCompleted precedes the
    // readback, MPSMatrix wrappers are init-owned and released below.
    let f16 = gemm_f16_on();
    let (esz, dtype) = if f16 { (2usize, MPS_FLOAT16) } else { (4usize, MPS_FLOAT32) };
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = base.io.lock().unwrap();
        let (a_ptr, buf_a) = ensure_buf(base, &mut io.x, a.len() * esz);
        let (b_ptr, buf_b) = ensure_buf(base, &mut io.z, b.len() * esz);
        let (c_ptr, buf_c) = ensure_buf(base, &mut io.y, out.len() * esz);
        if buf_a.is_null() || buf_b.is_null() || buf_c.is_null() || a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        if f16 {
            f32s_to_f16s(a, std::slice::from_raw_parts_mut(a_ptr as *mut u16, a.len()));
            f32s_to_f16s(b, std::slice::from_raw_parts_mut(b_ptr as *mut u16, b.len()));
        } else {
            std::ptr::copy_nonoverlapping(a.as_ptr(), a_ptr as *mut f32, a.len());
            std::ptr::copy_nonoverlapping(b.as_ptr(), b_ptr as *mut f32, b.len());
        }

        // batched descriptor: matrices = batch, matrixBytes = one matrix
        let bdesc = |rows: usize, cols: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                class("MPSMatrixDescriptor"),
                sel("matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:"),
                rows as u64,
                cols as u64,
                batch as u64,
                (cols * esz) as u64,
                (rows * cols * esz) as u64,
                dtype,
            )
        };
        let matrix = |buf: Id, d: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(msg_id(class("MPSMatrix"), sel("alloc")), sel("initWithBuffer:descriptor:"), buf, d)
        };
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        let ma = matrix(buf_a, bdesc(m, k));
        let mb = matrix(buf_b, bdesc(b_rows, b_cols));
        let mc = matrix(buf_c, bdesc(m, n));
        if ma.is_null() || mb.is_null() || mc.is_null() {
            for mm in [ma, mb, mc] {
                if !mm.is_null() {
                    msg_void(mm, sel("release"));
                }
            }
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }

        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        {
            // one kernel run per batch entry, all in one command buffer.
            // batchSize lives on MPSMatrixBinaryKernel; its absence would
            // raise an unrecognized-selector exception, so probe first
            // (the default, 0, already means "the whole batch").
            let responds: extern "C" fn(Id, Sel, Sel) -> bool =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            if responds(kernel, sel("respondsToSelector:"), sel("setBatchSize:")) {
                let f: extern "C" fn(Id, Sel, u64) =
                    std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                f(kernel, sel("setBatchSize:"), batch as u64);
            }
        }
        {
            let f: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(kernel, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, ma, mb, mc);
        }
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        if f16 {
            f16s_to_f32s(std::slice::from_raw_parts(c_ptr as *const u16, out.len()), out);
        } else {
            out.copy_from_slice(std::slice::from_raw_parts(c_ptr as *const f32, out.len()));
        }
        for mm in [ma, mb, mc] {
            msg_void(mm, sel("release"));
        }
        drop(io);
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

/// Kill switch for the attention offload alone (MICROKIMI_QWEN_GPU_NOATTN=1):
/// the projection GEMMs stay offloaded, attention returns to the CPU loop.
pub fn qwen_gpu_attn_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    qwen_gpu_on()
        && *ON.get_or_init(|| {
            std::env::var("MICROKIMI_QWEN_GPU_NOATTN").map(|v| v != "1").unwrap_or(true)
        })
}

// ════════════════════════════════════════════════════════════════════════════
// Half-precision GEMM storage (the offload's default; f32 on demand)
// ════════════════════════════════════════════════════════════════════════════
//
// The measured split at 1k tokens put ~half of the offloaded prefill
// inside the GEMMs, and those GEMMs move mostly weight bytes - f16
// storage halves that traffic (what llama.cpp's Metal path does
// throughout). Weights convert once at upload; activations convert at
// the staging boundary with fcvtn/fcvtl (base AArch64 instructions,
// stable inline asm like the SDOT kernels; scalar fallback elsewhere).
// Accumulation inside MPS stays wider than the storage, and the
// qwengpubench parity line reports the end-to-end numeric cost.
// MICROKIMI_QWEN_GPU_F32=1 restores full f32 storage.

const MPS_FLOAT16: u32 = 0x10000010; // MPSDataTypeFloatBit | 16

/// True when the offload stores GEMM operands in f16 (the default).
pub fn gemm_f16_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_QWEN_GPU_F32").map(|v| v != "1").unwrap_or(true))
}

fn f32_to_f16_scalar(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp8 = (b >> 23) & 0xff;
    let man = b & 0x007f_ffff;
    if exp8 == 0xff {
        return sign | if man != 0 { 0x7e00 } else { 0x7c00 }; // qNaN / inf
    }
    let e = exp8 as i32 - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // underflow -> signed zero
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        let mut h = half as u16;
        if rem > mid || (rem == mid && (half & 1) == 1) {
            h += 1;
        }
        return sign | h;
    }
    let half = (man >> 13) as u16;
    let rem = man & 0x1fff;
    let mut h = sign | ((e as u16) << 10) | half;
    if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        h = h.wrapping_add(1); // a carry into the exponent rounds up correctly
    }
    h
}

fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Bulk f32 -> f16, four lanes per fcvtn on aarch64 (RNE, matching the
/// scalar tail), plain scalar elsewhere.
fn f32s_to_f16s(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    #[allow(unused_mut)]
    let mut done = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        let n4 = src.len() / 4 * 4;
        while done < n4 {
            // SAFETY: 4 f32 reads and 4 u16 writes inside the slices.
            unsafe {
                core::arch::asm!(
                    "ld1 {{v0.4s}}, [{s}]",
                    "fcvtn v0.4h, v0.4s",
                    "st1 {{v0.4h}}, [{d}]",
                    s = in(reg) src.as_ptr().add(done),
                    d = in(reg) dst.as_mut_ptr().add(done),
                    out("v0") _,
                    options(nostack)
                );
            }
            done += 4;
        }
    }
    for i in done..src.len() {
        dst[i] = f32_to_f16_scalar(src[i]);
    }
}

/// Bulk f16 -> f32 (fcvtl on aarch64, scalar elsewhere); exact.
fn f16s_to_f32s(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    #[allow(unused_mut)]
    let mut done = 0usize;
    #[cfg(target_arch = "aarch64")]
    {
        let n4 = src.len() / 4 * 4;
        while done < n4 {
            // SAFETY: 4 u16 reads and 4 f32 writes inside the slices.
            unsafe {
                core::arch::asm!(
                    "ld1 {{v0.4h}}, [{s}]",
                    "fcvtl v0.4s, v0.4h",
                    "st1 {{v0.4s}}, [{d}]",
                    s = in(reg) src.as_ptr().add(done),
                    d = in(reg) dst.as_mut_ptr().add(done),
                    out("v0") _,
                    options(nostack)
                );
            }
            done += 4;
        }
    }
    for i in done..src.len() {
        dst[i] = f16_to_f32_scalar(src[i]);
    }
}

/// Device-resident f16 copy of an f32 weight matrix (spine attention
/// matrices in f16 mode). Tagged with the first f32 bits against
/// allocator address reuse, like the other weight caches.
fn weight_buffer_f16(base: &MetalCtx, mps: &MpsCtx, w: &[f32], rows: usize, cols: usize) -> Option<Id> {
    let key = (w.as_ptr() as usize, rows, cols);
    let tag = w.first().map(|v| v.to_bits()).unwrap_or(0);
    let mut cache = mps.w16.lock().unwrap();
    if let Some(&(buf, seen)) = cache.get(&key) {
        if seen == tag {
            return Some(buf);
        }
        cache.remove(&key);
        // SAFETY: the stale buffer is owned by this cache (retained at insert).
        unsafe { msg_void(buf, sel("release")) };
    }
    let mut h = vec![0u16; w.len()];
    f32s_to_f16s(w, &mut h);
    // SAFETY: `h` is alive for the whole call; the buffer copies its bytes.
    let buf = unsafe {
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let b = f(
            base.device,
            sel("newBufferWithBytes:length:options:"),
            h.as_ptr() as *const c_void,
            (h.len() * 2) as u64,
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
    cache.insert(key, (buf, tag));
    Some(buf)
}

/// Device-resident q8_0 copy of an f32 weight matrix for the decode
/// matvec: int8 rows + one f16 scale per block of 32 (dx = max|w|/127,
/// rounded to f16 first so the reconstruction uses the stored scale).
/// Returns (rows buffer, scales buffer). Cached like weight_buffer_f16.
fn weight_buffer_q8(base: &MetalCtx, mps: &MpsCtx, w: &[f32], rows: usize, cols: usize) -> Option<(Id, Id)> {
    if cols % 32 != 0 || w.len() < rows * cols {
        return None;
    }
    let key = (w.as_ptr() as usize, rows, cols);
    let tag = w.first().map(|v| v.to_bits()).unwrap_or(0);
    let mut cache = mps.wq8.lock().unwrap();
    if let Some(&(bq, bs, seen)) = cache.get(&key) {
        if seen == tag {
            return Some((bq, bs));
        }
        cache.remove(&key);
        // SAFETY: the stale buffers are owned by this cache (retained at insert).
        unsafe {
            msg_void(bq, sel("release"));
            msg_void(bs, sel("release"));
        }
    }
    let nb = cols / 32;
    let mut q = vec![0i8; rows * cols];
    let mut sc = vec![0u16; rows * nb];
    for r in 0..rows {
        for g in 0..nb {
            let blk = &w[r * cols + g * 32..r * cols + (g + 1) * 32];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s16 = f32_to_f16_scalar(amax / 127.0);
            let s = f16_to_f32_scalar(s16);
            sc[r * nb + g] = s16;
            if s > 0.0 {
                let inv = 1.0 / s;
                for (i, v) in blk.iter().enumerate() {
                    q[r * cols + g * 32 + i] = (v * inv).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }
    // SAFETY: the vecs are alive for the whole call; the buffers copy the bytes.
    let (bq, bs) = unsafe {
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let bq = f(base.device, sel("newBufferWithBytes:length:options:"), q.as_ptr() as *const c_void, q.len() as u64, 0);
        let bs = f(base.device, sel("newBufferWithBytes:length:options:"), sc.as_ptr() as *const c_void, (sc.len() * 2) as u64, 0);
        if bq.is_null() || bs.is_null() {
            return None;
        }
        retain(bq);
        retain(bs);
        (bq, bs)
    };
    cache.insert(key, (bq, bs, tag));
    Some((bq, bs))
}

/// Device-resident copy of an MXFP4 packed matrix as stored (nibbles,
/// scale bytes) for the decode matvec. Returns (packed, scales) buffers.
fn packed_buffers_fp4(base: &MetalCtx, mps: &MpsCtx, packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Option<(Id, Id)> {
    if cols % 32 != 0 || packed.len() < rows * cols / 2 || scales.len() < rows * cols / 32 {
        return None;
    }
    let key = (packed.as_ptr() as usize, rows, cols);
    let mut tag = [0u8; 16];
    let n = packed.len().min(16);
    tag[..n].copy_from_slice(&packed[..n]);
    let mut cache = mps.fp4raw.lock().unwrap();
    if let Some(&(bp, bs, seen)) = cache.get(&key) {
        if seen == tag {
            return Some((bp, bs));
        }
        cache.remove(&key);
        // SAFETY: the stale buffers are owned by this cache (retained at insert).
        unsafe {
            msg_void(bp, sel("release"));
            msg_void(bs, sel("release"));
        }
    }
    // SAFETY: the slices are alive for the whole call; the buffers copy the bytes.
    let (bp, bs) = unsafe {
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let bp = f(base.device, sel("newBufferWithBytes:length:options:"), packed.as_ptr() as *const c_void, (rows * cols / 2) as u64, 0);
        let bs = f(base.device, sel("newBufferWithBytes:length:options:"), scales.as_ptr() as *const c_void, (rows * cols / 32) as u64, 0);
        if bp.is_null() || bs.is_null() {
            return None;
        }
        retain(bp);
        retain(bs);
        (bp, bs)
    };
    cache.insert(key, (bp, bs, tag));
    Some((bp, bs))
}

// ════════════════════════════════════════════════════════════════════════════
// Delta-scan kernel: the recurrence itself on the GPU
// ════════════════════════════════════════════════════════════════════════════
//
// The delta rule is sequential over time but COLUMN-SEPARABLE: for a
// fixed value column j, every operation touches only S[.,j], pred[j],
// v[j] and out[j], with k and q shared reads. So the scan launches one
// thread per (head, column) - 4096 independent sequential scans for the
// 0.8B - each carrying its state column (<= 128 f32) in thread-private
// memory, with NO barriers anywhere. All scan traffic stays f32: the
// recurrence is the numerically delicate part and its bytes are small
// next to the GEMM weights. Failures compile-time or dispatch-time fall
// back to the CPU scan; MICROKIMI_QWEN_GPU_NOSCAN=1 is the kill switch.

const DELTA_SCAN_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// The recurrent state lives in registers. Inside a SIMD group, SCAN_L
// consecutive lanes share a set of SCAN_C consecutive value columns and
// split the kd state rows between them (per = kd / SCAN_L rows per lane;
// kd a multiple of SCAN_L, per <= 16); the 32/SCAN_L lane groups of the
// SIMD group take consecutive column sets. No private array beyond the
// register tile, no threadgroup memory, no barrier: the per-token dot
// products (prediction, readout) reduce over the SCAN_L lanes with
// xor-shuffles, and k/q are read once per token per lane for SCAN_C
// columns. Threadgroups of 128 threads carry 4 * (32/SCAN_L) * SCAN_C
// columns; the host dispatches heads x ceil(vd / that) threadgroups.
// Layouts:
//   q, k    [t, heads, kd]   (kv-heads pre-expanded across their group)
//   v       [t, heads, vd]
//   beta    [t, heads]
//   decay   [t, heads]       (exp(g), precomputed on the CPU)
//   state   [heads, kd, vd]  (read-write, f32 - the recurrent object)
//   out     [heads, t, vd]
// Per token, in the CPU delta_step order: decay, prediction, beta-scaled
// delta, q readout.
constant uint SCAN_L = 16;
constant uint SCAN_C = 4;
constant uint SCAN_PER_MAX = 16;

kernel void delta_scan(device const float* q     [[buffer(0)]],
                       device const float* k     [[buffer(1)]],
                       device const float* v     [[buffer(2)]],
                       device const float* beta  [[buffer(3)]],
                       device const float* decay [[buffer(4)]],
                       device float*       state [[buffer(5)]],
                       device float*       out   [[buffer(6)]],
                       constant uint4&     dims  [[buffer(7)]],
                       uint tg    [[threadgroup_position_in_grid]],
                       uint lane  [[thread_position_in_threadgroup]]) {
    uint t_count = dims.x, heads = dims.y, kd = dims.z, vd = dims.w;
    uint sg = lane / 32u, sl = lane & 31u;
    uint grp = sl / SCAN_L, rl = sl % SCAN_L;
    uint groups_per_sg = 32u / SCAN_L;
    uint cols_per_tg = 4u * groups_per_sg * SCAN_C;
    uint groups = (vd + cols_per_tg - 1u) / cols_per_tg;
    uint h = tg / groups;
    uint j0 = (tg % groups) * cols_per_tg + (sg * groups_per_sg + grp) * SCAN_C;
    if (h >= heads || j0 >= vd) { return; }
    uint per = kd / SCAN_L;
    uint nc = min(SCAN_C, vd - j0);
    float s[SCAN_PER_MAX][SCAN_C];
    for (uint i = 0; i < SCAN_PER_MAX; i++) { for (uint c = 0; c < SCAN_C; c++) { s[i][c] = 0.0f; } }
    device float* sbase = state + (size_t)h * kd * vd + j0;
    for (uint i = 0; i < per; i++) { for (uint c = 0; c < nc; c++) { s[i][c] = sbase[(rl * per + i) * vd + c]; } }
    for (uint t = 0; t < t_count; t++) {
        device const float* kt = k + (size_t)(t * heads + h) * kd + rl * per;
        device const float* qt = q + (size_t)(t * heads + h) * kd + rl * per;
        device const float* vt = v + (size_t)(t * heads + h) * vd + j0;
        float dec = decay[t * heads + h];
        float bet = beta[t * heads + h];
        float kk[SCAN_PER_MAX], qq[SCAN_PER_MAX];
        for (uint i = 0; i < per; i++) { kk[i] = kt[i]; qq[i] = qt[i]; }
        float pred[SCAN_C];
        for (uint c = 0; c < SCAN_C; c++) { pred[c] = 0.0f; }
        for (uint i = 0; i < per; i++) {
            for (uint c = 0; c < SCAN_C; c++) { s[i][c] *= dec; pred[c] += kk[i] * s[i][c]; }
        }
        float delta[SCAN_C];
        for (uint c = 0; c < SCAN_C; c++) {
            float p = pred[c];
            for (uint off = SCAN_L / 2u; off > 0u; off >>= 1u) { p += simd_shuffle_xor(p, (ushort)off); }
            delta[c] = ((c < nc) ? (vt[c] - p) : 0.0f) * bet;
        }
        float o[SCAN_C];
        for (uint c = 0; c < SCAN_C; c++) { o[c] = 0.0f; }
        for (uint i = 0; i < per; i++) {
            for (uint c = 0; c < SCAN_C; c++) { s[i][c] += kk[i] * delta[c]; o[c] += qq[i] * s[i][c]; }
        }
        for (uint c = 0; c < SCAN_C; c++) {
            float oc = o[c];
            for (uint off = SCAN_L / 2u; off > 0u; off >>= 1u) { oc += simd_shuffle_xor(oc, (ushort)off); }
            if (rl == 0u && c < nc) { out[((size_t)h * t_count + t) * vd + j0 + c] = oc; }
        }
    }
    for (uint i = 0; i < per; i++) { for (uint c = 0; c < nc; c++) { sbase[(rl * per + i) * vd + c] = s[i][c]; } }
}
"#;

/// Lanes per column set and columns per lane group in `delta_scan`
/// (SCAN_L / SCAN_C in the shader); kd must be a multiple of SCAN_L with
/// kd / SCAN_L <= 16.
const SCAN_L: usize = 16;
const SCAN_C: usize = 4;

/// Threadgroups for the delta scan grid: heads x ceil(vd / columns per threadgroup).
fn scan_groups(heads: usize, vd: usize) -> usize {
    heads * vd.div_ceil(4 * (32 / SCAN_L) * SCAN_C)
}

/// The scan kernel's kd budget: a multiple of SCAN_L, at most 16 rows per lane.
fn scan_kd_ok(kd: usize) -> bool {
    kd % SCAN_L == 0 && kd / SCAN_L <= 16
}

struct ScanCtx {
    pipeline: Id,
}

// SAFETY: the pipeline is a retained device-owned object; encoding is
// serialized on the MetalCtx io mutex like every other path here.
unsafe impl Send for ScanCtx {}
unsafe impl Sync for ScanCtx {}

static SCAN: std::sync::OnceLock<Option<ScanCtx>> = std::sync::OnceLock::new();

fn scan_ctx() -> Option<(&'static MetalCtx, &'static ScanCtx)> {
    let base = ctx()?;
    let scan = SCAN
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(DELTA_SCAN_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: delta-scan shader compilation error: {} - scan stays on CPU", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let function = {
                    let f: extern "C" fn(Id, Sel, Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(library, sel("newFunctionWithName:"), ns_string("delta_scan"))
                };
                let mut perr: Id = std::ptr::null_mut();
                let pipeline = {
                    let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                };
                if pipeline.is_null() {
                    println!("gpu: delta-scan pipeline error: {} - scan stays on CPU", err_desc(perr));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                retain(pipeline);
                retain(function);
                retain(library);
                println!("gpu: delta-scan kernel ready");
                msg_void(pool, sel("drain"));
                Some(ScanCtx { pipeline })
            }
        })
        .as_ref()?;
    Some((base, scan))
}

/// True when the delta scan runs on the GPU (with the offload, unless
/// MICROKIMI_QWEN_GPU_NOSCAN=1 pins it to the CPU).
pub fn qwen_gpu_scan_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    qwen_gpu_on()
        && *ON.get_or_init(|| {
            std::env::var("MICROKIMI_QWEN_GPU_NOSCAN").map(|v| v != "1").unwrap_or(true)
        })
}

/// Runs the whole delta recurrence of one layer on the GPU. `state` is
/// [heads, kd, vd] and is read AND written (the recurrent carry); `out`
/// is [heads, t, vd]. Returns false when the kernel is unavailable or
/// kd exceeds the thread-private column budget (caller keeps its CPU
/// scan).
#[allow(clippy::too_many_arguments)]
pub fn gpu_delta_scan(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    beta: &[f32],
    decay: &[f32],
    state: &mut [f32],
    out: &mut [f32],
    t_count: usize,
    heads: usize,
    kd: usize,
    vd: usize,
) -> bool {
    if !scan_kd_ok(kd) {
        return false; // register-tile scan: kd/SCAN_L state rows per lane, up to 16
    }
    assert_eq!(q.len(), t_count * heads * kd);
    assert_eq!(k.len(), t_count * heads * kd);
    assert_eq!(v.len(), t_count * heads * vd);
    assert_eq!(beta.len(), t_count * heads);
    assert_eq!(decay.len(), t_count * heads);
    assert_eq!(state.len(), heads * kd * vd);
    assert_eq!(out.len(), heads * t_count * vd);
    let Some((base, scan)) = scan_ctx() else {
        return false;
    };
    let t_start = std::time::Instant::now();
    // one staging buffer for the five read-only inputs, offset-addressed
    let sizes = [q.len(), k.len(), v.len(), beta.len(), decay.len()];
    let mut offs = [0usize; 5];
    let mut total = 0usize;
    for (i, len) in sizes.iter().enumerate() {
        offs[i] = total;
        total += len.div_ceil(4) * 4; // keep every section 16-byte aligned
    }
    // SAFETY: same invariants as the GEMM paths - io mutex held across
    // encode/wait/readback, waitUntilCompleted before reads, fresh
    // autorelease pool for the transients.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = base.io.lock().unwrap();
        let (in_ptr, buf_in) = ensure_buf(base, &mut io.x, total * 4);
        let (out_ptr, buf_out) = ensure_buf(base, &mut io.y, out.len() * 4);
        let (st_ptr, buf_st) = ensure_buf(base, &mut io.st, state.len() * 4);
        if buf_in.is_null() || buf_out.is_null() || buf_st.is_null() || in_ptr.is_null() || out_ptr.is_null() || st_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        for (src, off) in [(q, offs[0]), (k, offs[1]), (v, offs[2]), (beta, offs[3]), (decay, offs[4])] {
            std::ptr::copy_nonoverlapping(src.as_ptr(), (in_ptr as *mut f32).add(off), src.len());
        }
        std::ptr::copy_nonoverlapping(state.as_ptr(), st_ptr as *mut f32, state.len());

        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));
        {
            let f: extern "C" fn(Id, Sel, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setComputePipelineState:"), scan.pipeline);
        }
        {
            let f: extern "C" fn(Id, Sel, Id, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            for (idx, off) in offs.iter().enumerate() {
                f(encoder, sel("setBuffer:offset:atIndex:"), buf_in, (off * 4) as u64, idx as u64);
            }
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_st, 0, 5);
            f(encoder, sel("setBuffer:offset:atIndex:"), buf_out, 0, 6);
        }
        {
            let dims: [u32; 4] = [t_count as u32, heads as u32, kd as u32, vd as u32];
            let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(encoder, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 16, 7);
        }
        {
            let f: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MTLSize { width: scan_groups(heads, vd) as u64, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
        }
        msg_void(encoder, sel("endEncoding"));
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        out.copy_from_slice(std::slice::from_raw_parts(out_ptr as *const f32, out.len()));
        state.copy_from_slice(std::slice::from_raw_parts(st_ptr as *const f32, state.len()));
        drop(io);
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

// ════════════════════════════════════════════════════════════════════════════
// Fused attention: GEMM1 -> causal softmax (MSL) -> GEMM2, one command
// buffer, scores never leave the GPU
// ════════════════════════════════════════════════════════════════════════════

const CAUSAL_SOFTMAX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// In-place causal softmax over f16 score rows laid out [head][row][L].
// One threadgroup per (head, row); lanes reduce max and sum with the
// same simd pattern as the matvec kernels; the tail past the causal
// window zeroes so the following P.V GEMM can run the full width.
kernel void causal_softmax_f16(device half* s      [[buffer(0)]],
                               constant uint4& dims [[buffer(1)]],
                               uint tg    [[threadgroup_position_in_grid]],
                               uint lane  [[thread_position_in_threadgroup]],
                               uint lanes [[threads_per_threadgroup]]) {
    uint t_count = dims.x, l = dims.y, base = dims.z;
    uint row = tg % t_count;
    uint window = base + row + 1;
    device half* p = s + (size_t)tg * l;
    threadgroup float partial[32];
    // max over the window
    float m = -INFINITY;
    for (uint i = lane; i < window; i += lanes) { m = max(m, float(p[i])); }
    m = simd_max(m);
    if ((lane & 31u) == 0u) partial[lane / 32u] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (lanes + 31u) / 32u;
    if (lane < 32u) {
        float v = (lane < nsg) ? partial[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0u) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = partial[0];
    // exp and sum
    float acc = 0.0f;
    for (uint i = lane; i < window; i += lanes) {
        float e = exp(float(p[i]) - m);
        p[i] = half(e);
        acc += e;
    }
    acc = simd_sum(acc);
    if ((lane & 31u) == 0u) partial[lane / 32u] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 32u) {
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / partial[0];
    for (uint i = lane; i < window; i += lanes) { p[i] = half(float(p[i]) * inv); }
    for (uint i = window + lane; i < l; i += lanes) { p[i] = half(0.0f); }
}
"#;

static SOFTMAX: std::sync::OnceLock<Option<ScanCtx>> = std::sync::OnceLock::new();

fn softmax_ctx() -> Option<(&'static MetalCtx, &'static ScanCtx)> {
    let base = ctx()?;
    let sm = SOFTMAX
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(CAUSAL_SOFTMAX_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: softmax shader error: {} - attention stays two-step", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let function = {
                    let f: extern "C" fn(Id, Sel, Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(library, sel("newFunctionWithName:"), ns_string("causal_softmax_f16"))
                };
                let mut perr: Id = std::ptr::null_mut();
                let pipeline = {
                    let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                };
                if pipeline.is_null() {
                    println!("gpu: softmax pipeline error: {} - attention stays two-step", err_desc(perr));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                retain(pipeline);
                retain(function);
                retain(library);
                msg_void(pool, sel("drain"));
                Some(ScanCtx { pipeline })
            }
        })
        .as_ref()?;
    Some((base, sm))
}

/// Whole attention for one layer in ONE command buffer: batched
/// scores GEMM, in-place causal softmax, batched P.V GEMM. Only in the
/// f16 storage mode (the kernel is written for half rows); refusals
/// fall back to the caller's two-step path. Layout as gpu_gemm_batched:
/// q [heads, t, hd], k/v [heads, l, hd], out [heads, t, hd].
#[allow(clippy::too_many_arguments)]
pub fn gpu_attention_fused(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    t: usize,
    l: usize,
    hd: usize,
    base_pos: usize,
    scale: f32,
    out: &mut [f32],
) -> bool {
    if !gemm_f16_on() {
        return false;
    }
    let scores_len = heads * t * l;
    if scores_len * 2 > GEMM_MAX_OUT_BYTES {
        return false;
    }
    let Some((base, sm)) = softmax_ctx() else {
        return false;
    };
    let Some((_, mps)) = mps_ctx() else {
        return false;
    };
    let Some(k1) = gemm_kernel(base, mps, t, l, hd, true, scale) else {
        return false;
    };
    let Some(k2) = gemm_kernel(base, mps, t, hd, l, false, 1.0) else {
        return false;
    };
    let t_start = std::time::Instant::now();
    // SAFETY: same invariants as the other offload paths - io mutex held
    // across encode/wait/readback, one autorelease pool, explicit
    // releases of the alloc/init-owned MPSMatrix wrappers. Encoders in
    // one command buffer execute in order with tracked-resource hazards.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = base.io.lock().unwrap();
        let (q_ptr, buf_q) = ensure_buf(base, &mut io.x, q.len() * 2);
        let kv_off = k.len(); // halves: [K | V] in one staging buffer
        let (kv_ptr, buf_kv) = ensure_buf(base, &mut io.z, (k.len() + v.len()) * 2);
        let out_off = scores_len; // halves: [scores | out]
        let (y_ptr, buf_y) = ensure_buf(base, &mut io.y, (scores_len + out.len()) * 2);
        if buf_q.is_null() || buf_kv.is_null() || buf_y.is_null() || q_ptr.is_null() || kv_ptr.is_null() || y_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        f32s_to_f16s(q, std::slice::from_raw_parts_mut(q_ptr as *mut u16, q.len()));
        f32s_to_f16s(k, std::slice::from_raw_parts_mut(kv_ptr as *mut u16, k.len()));
        f32s_to_f16s(v, std::slice::from_raw_parts_mut((kv_ptr as *mut u16).add(kv_off), v.len()));

        let bdesc = |rows: usize, cols: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                class("MPSMatrixDescriptor"),
                sel("matrixDescriptorWithRows:columns:matrices:rowBytes:matrixBytes:dataType:"),
                rows as u64,
                cols as u64,
                heads as u64,
                (cols * 2) as u64,
                (rows * cols * 2) as u64,
                MPS_FLOAT16,
            )
        };
        let matrix_at = |buf: Id, off_bytes: usize, d: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, u64, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                msg_id(class("MPSMatrix"), sel("alloc")),
                sel("initWithBuffer:offset:descriptor:"),
                buf,
                off_bytes as u64,
                d,
            )
        };
        let mq = matrix_at(buf_q, 0, bdesc(t, hd));
        let mk = matrix_at(buf_kv, 0, bdesc(l, hd));
        let mv = matrix_at(buf_kv, kv_off * 2, bdesc(l, hd));
        let ms = matrix_at(buf_y, 0, bdesc(t, l));
        let mo = matrix_at(buf_y, out_off * 2, bdesc(t, hd));
        if [mq, mk, mv, ms, mo].iter().any(|m| m.is_null()) {
            for m in [mq, mk, mv, ms, mo] {
                if !m.is_null() {
                    msg_void(m, sel("release"));
                }
            }
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }

        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        {
            let responds: extern "C" fn(Id, Sel, Sel) -> bool =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            if responds(k1, sel("respondsToSelector:"), sel("setBatchSize:")) {
                let f: extern "C" fn(Id, Sel, u64) =
                    std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                f(k1, sel("setBatchSize:"), heads as u64);
                f(k2, sel("setBatchSize:"), heads as u64);
            }
        }
        {
            let f: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(k1, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, mq, mk, ms);
        }
        {
            // the softmax between the two GEMMs, same command buffer
            let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));
            let f1: extern "C" fn(Id, Sel, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f1(encoder, sel("setComputePipelineState:"), sm.pipeline);
            let f2: extern "C" fn(Id, Sel, Id, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_y, 0, 0);
            let dims: [u32; 4] = [t as u32, l as u32, base_pos as u32, heads as u32];
            let f3: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f3(encoder, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 16, 1);
            let f4: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f4(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MTLSize { width: (heads * t) as u64, height: 1, depth: 1 },
                MTLSize { width: 64, height: 1, depth: 1 },
            );
            msg_void(encoder, sel("endEncoding"));
        }
        {
            let f: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(k2, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, ms, mv, mo);
        }
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        f16s_to_f32s(
            std::slice::from_raw_parts((y_ptr as *const u16).add(out_off), out.len()),
            out,
        );
        for m in [mq, mk, mv, ms, mo] {
            msg_void(m, sel("release"));
        }
        drop(io);
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

// ════════════════════════════════════════════════════════════════════════════
// Fused MLP: gate GEMM, up GEMM, SiLU(gate)*up in MSL, down GEMM - one
// command buffer, the intermediate activation never leaves the GPU
// ════════════════════════════════════════════════════════════════════════════

const SILU_MUL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// h[i] = silu(g[i]) * u[i], in place into g (f16 storage, f32 math)
kernel void silu_mul_f16(device half* g       [[buffer(0)]],
                         device const half* u [[buffer(1)]],
                         constant uint& n     [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= n) { return; }
    float x = float(g[i]);
    float s = x / (1.0f + exp(-x));
    g[i] = half(s * float(u[i]));
}
"#;

static SILU: std::sync::OnceLock<Option<ScanCtx>> = std::sync::OnceLock::new();

fn silu_ctx() -> Option<(&'static MetalCtx, &'static ScanCtx)> {
    let base = ctx()?;
    let sm = SILU
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(SILU_MUL_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: silu shader error: {} - MLP stays three-step", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let function = {
                    let f: extern "C" fn(Id, Sel, Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(library, sel("newFunctionWithName:"), ns_string("silu_mul_f16"))
                };
                let mut perr: Id = std::ptr::null_mut();
                let pipeline = {
                    let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                };
                if pipeline.is_null() {
                    println!("gpu: silu pipeline error: {} - MLP stays three-step", err_desc(perr));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                retain(pipeline);
                retain(function);
                retain(library);
                msg_void(pool, sel("drain"));
                Some(ScanCtx { pipeline })
            }
        })
        .as_ref()?;
    Some((base, sm))
}

/// Whole dense MLP for one layer in ONE command buffer: gate and up
/// GEMMs from the same staged X, SiLU(gate)*up in place on the GPU,
/// down GEMM into the output. Three device-resident MXFP4 dequant
/// copies (f16 mode only). False = caller runs its three-GEMM path.
#[allow(clippy::too_many_arguments)]
pub fn gpu_mlp_fused(
    gate: (&[u8], &[u8]),
    up: (&[u8], &[u8]),
    down: (&[u8], &[u8]),
    inter: usize,
    d: usize,
    xs: &[&[f32]],
    norm: Option<(&[f32], f32)>, // (post-norm weight, eps): xs are hidden rows, norm on GPU
    outs: &mut [&mut [f32]],
) -> bool {
    if !gemm_f16_on() {
        return false;
    }
    let t = xs.len();
    if t * inter * 2 * 2 + t * d * 2 > GEMM_MAX_OUT_BYTES {
        return false;
    }
    let Some((base, sm)) = silu_ctx() else {
        return false;
    };
    let norm_ctx = if norm.is_some() { addnorm_ctx() } else { None };
    if norm.is_some() && norm_ctx.is_none() {
        return false;
    }
    let Some((_, mps)) = mps_ctx() else {
        return false;
    };
    let Some(bg) = dequant_buffer(base, mps, gate.0, gate.1, inter, d) else {
        return false;
    };
    let Some(bu) = dequant_buffer(base, mps, up.0, up.1, inter, d) else {
        return false;
    };
    let Some(bd) = dequant_buffer(base, mps, down.0, down.1, d, inter) else {
        return false;
    };
    let Some(k_up) = gemm_kernel(base, mps, t, inter, d, true, 1.0) else {
        return false;
    };
    let Some(k_down) = gemm_kernel(base, mps, t, d, inter, true, 1.0) else {
        return false;
    };
    let t_start = std::time::Instant::now();
    // SAFETY: same invariants as the other offload paths.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = base.io.lock().unwrap();
        let (x_ptr, buf_x) = ensure_buf(base, &mut io.x, t * d * 2);
        // [gate | up] halves in one staging buffer, output in y
        let (h_ptr, buf_h) = ensure_buf(base, &mut io.z, t * inter * 2 * 2);
        let (y_ptr, buf_y) = ensure_buf(base, &mut io.y, t * d * 2);
        // st: [hidden f16 (t*d) | norm weight f32 (d)] when the norm is fused
        let (st_ptr, buf_st) = if norm.is_some() {
            ensure_buf(base, &mut io.st, t * d * 2 + d * 4)
        } else {
            (std::ptr::null_mut(), std::ptr::null_mut())
        };
        if buf_x.is_null() || buf_h.is_null() || buf_y.is_null() || x_ptr.is_null() || h_ptr.is_null() || y_ptr.is_null()
            || (norm.is_some() && (buf_st.is_null() || st_ptr.is_null()))
        {
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        if let Some((nw, _)) = norm {
            // hidden rows staged f16; the norm kernel produces X on device
            for (l, x) in xs.iter().enumerate() {
                let dst = std::slice::from_raw_parts_mut((st_ptr as *mut u16).add(l * d), d);
                f32s_to_f16s(x, dst);
            }
            std::ptr::copy_nonoverlapping(nw.as_ptr(), (st_ptr as *mut u8).add(t * d * 2) as *mut f32, d);
        } else {
            for (l, x) in xs.iter().enumerate() {
                let dst = std::slice::from_raw_parts_mut((x_ptr as *mut u16).add(l * d), d);
                f32s_to_f16s(x, dst);
            }
        }
        let desc = |r: usize, c: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                class("MPSMatrixDescriptor"),
                sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"),
                r as u64,
                c as u64,
                (c * 2) as u64,
                MPS_FLOAT16,
            )
        };
        let matrix_at = |buf: Id, off: usize, dsc: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, u64, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(msg_id(class("MPSMatrix"), sel("alloc")), sel("initWithBuffer:offset:descriptor:"), buf, off as u64, dsc)
        };
        let mx = matrix_at(buf_x, 0, desc(t, d));
        let mwg = matrix_at(bg, 0, desc(inter, d));
        let mwu = matrix_at(bu, 0, desc(inter, d));
        let mwd = matrix_at(bd, 0, desc(d, inter));
        let mg = matrix_at(buf_h, 0, desc(t, inter));
        let mu = matrix_at(buf_h, t * inter * 2, desc(t, inter));
        let my = matrix_at(buf_y, 0, desc(t, d));
        let all = [mx, mwg, mwu, mwd, mg, mu, my];
        if all.iter().any(|m| m.is_null()) {
            for m in all {
                if !m.is_null() {
                    msg_void(m, sel("release"));
                }
            }
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }

        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        if let (Some((_, eps)), Some((_, an))) = (norm, norm_ctx) {
            // fused post-norm: X = rmsnorm(hidden) * (1 + w), on device
            let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));
            let f1: extern "C" fn(Id, Sel, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f1(encoder, sel("setComputePipelineState:"), an.pipeline);
            let f2: extern "C" fn(Id, Sel, Id, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_st, 0, 0);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_st, 0, 1);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_st, (t * d * 2) as u64, 2);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_x, 0, 3);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_x, 0, 6);
            let dims: [u32; 2] = [d as u32, 0];
            let f3: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f3(encoder, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 8, 4);
            f3(encoder, sel("setBytes:length:atIndex:"), (&eps) as *const f32 as *const c_void, 4, 5);
            let f4: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f4(
                encoder,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MTLSize { width: t as u64, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            msg_void(encoder, sel("endEncoding"));
        }
        let enc: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        enc(k_up, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, mx, mwg, mg);
        enc(k_up, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, mx, mwu, mu);
        {
            let encoder = msg_id(cmdbuf, sel("computeCommandEncoder"));
            let f1: extern "C" fn(Id, Sel, Id) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f1(encoder, sel("setComputePipelineState:"), sm.pipeline);
            let f2: extern "C" fn(Id, Sel, Id, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_h, 0, 0);
            f2(encoder, sel("setBuffer:offset:atIndex:"), buf_h, (t * inter * 2) as u64, 1);
            let n = (t * inter) as u32;
            let f3: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f3(encoder, sel("setBytes:length:atIndex:"), (&n) as *const u32 as *const c_void, 4, 2);
            let f4: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f4(
                encoder,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { width: (t * inter) as u64, height: 1, depth: 1 },
                MTLSize { width: 256, height: 1, depth: 1 },
            );
            msg_void(encoder, sel("endEncoding"));
        }
        enc(k_down, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, mg, mwd, my);
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        for (l, out) in outs.iter_mut().enumerate() {
            let src = std::slice::from_raw_parts((y_ptr as *const u16).add(l * d), d);
            f16s_to_f32s(src, out);
        }
        for m in all {
            msg_void(m, sel("release"));
        }
        drop(io);
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

/// Several f32 weight matrices against ONE staged input, one command
/// buffer, one wait: X uploads once, each GEMM writes its own region of
/// the output staging. For the attention projections (q/k/v, in_qkv/in_z)
/// that share the normed hidden. f16 mode only; false = per-GEMM path.
pub fn gpu_gemm_multi_w(
    ws: &[(&[f32], usize)], // (weights, rows), all with the same cols
    cols: usize,
    xs: &[&[f32]],
    outs: &mut [&mut [&mut [f32]]], // outs[k][lane] -> rows_k
) -> bool {
    if !gemm_f16_on() || ws.is_empty() {
        return false;
    }
    let t = xs.len();
    let total_rows: usize = ws.iter().map(|w| w.1).sum();
    if t * total_rows * 2 > GEMM_MAX_OUT_BYTES {
        return false;
    }
    let Some((base, mps)) = mps_ctx() else {
        return false;
    };
    let mut bufs = Vec::with_capacity(ws.len());
    let mut kernels = Vec::with_capacity(ws.len());
    for (w, rows) in ws {
        let Some(b) = weight_buffer_f16(base, mps, w, *rows, cols) else {
            return false;
        };
        let Some(k) = gemm_kernel(base, mps, t, *rows, cols, true, 1.0) else {
            return false;
        };
        bufs.push(b);
        kernels.push(k);
    }
    let t_start = std::time::Instant::now();
    // SAFETY: same invariants as run_gemm.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mut io = base.io.lock().unwrap();
        let (x_ptr, buf_x) = ensure_buf(base, &mut io.x, t * cols * 2);
        let (y_ptr, buf_y) = ensure_buf(base, &mut io.y, t * total_rows * 2);
        if buf_x.is_null() || buf_y.is_null() || x_ptr.is_null() || y_ptr.is_null() {
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        for (l, x) in xs.iter().enumerate() {
            let dst = std::slice::from_raw_parts_mut((x_ptr as *mut u16).add(l * cols), cols);
            f32s_to_f16s(x, dst);
        }
        let desc = |r: usize, c: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(
                class("MPSMatrixDescriptor"),
                sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"),
                r as u64,
                c as u64,
                (c * 2) as u64,
                MPS_FLOAT16,
            )
        };
        let matrix_at = |buf: Id, off: usize, dsc: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, u64, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(msg_id(class("MPSMatrix"), sel("alloc")), sel("initWithBuffer:offset:descriptor:"), buf, off as u64, dsc)
        };
        let mx = matrix_at(buf_x, 0, desc(t, cols));
        let mut mats = vec![mx];
        let mut y_off = 0usize;
        let mut y_offs = Vec::with_capacity(ws.len());
        for (i, (_, rows)) in ws.iter().enumerate() {
            mats.push(matrix_at(bufs[i], 0, desc(*rows, cols)));
            mats.push(matrix_at(buf_y, y_off * 2, desc(t, *rows)));
            y_offs.push(y_off);
            y_off += t * rows;
        }
        if mats.iter().any(|m| m.is_null()) {
            for m in &mats {
                if !m.is_null() {
                    msg_void(*m, sel("release"));
                }
            }
            drop(io);
            msg_void(pool, sel("drain"));
            return false;
        }
        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        let enc: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        for i in 0..ws.len() {
            enc(
                kernels[i],
                sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"),
                cmdbuf,
                mx,
                mats[1 + 2 * i],
                mats[2 + 2 * i],
            );
        }
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));
        for (i, (_, rows)) in ws.iter().enumerate() {
            let base_off = y_offs[i];
            for (l, out) in outs[i].iter_mut().enumerate() {
                let src = std::slice::from_raw_parts((y_ptr as *const u16).add(base_off + l * rows), *rows);
                f16s_to_f32s(src, out);
            }
        }
        for m in &mats {
            msg_void(*m, sel("release"));
        }
        drop(io);
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

// ════════════════════════════════════════════════════════════════════════════
// Phase 2: residual + RMSNorm on the GPU (f16 activations, f32 math)
// ════════════════════════════════════════════════════════════════════════════

const ADD_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// normed[t] = rmsnorm(hidden[t]) * (1 + w) (Qwen's offset-from-one norm
// weights). hidden arrives as f16 for the norm only - the f32 residual
// stream itself stays on the CPU, so precision accumulates in f32 across
// the layers. One threadgroup per row, simd reductions.
kernel void add_rmsnorm_f16(device const half* hidden [[buffer(0)]],
                            device const half* add    [[buffer(1)]],
                            device const float* w     [[buffer(2)]],
                            device half* normed       [[buffer(3)]],
                            constant uint2& dims      [[buffer(4)]], // d, has_add
                            constant float& eps       [[buffer(5)]],
                            device half* sum_out      [[buffer(6)]], // hidden+add when has_add
                            uint row   [[threadgroup_position_in_grid]],
                            uint lane  [[thread_position_in_threadgroup]],
                            uint lanes [[threads_per_threadgroup]]) {
    uint d = dims.x;
    device const half* h = hidden + (size_t)row * d;
    device const half* a = add + (size_t)row * d;
    device half* so = sum_out + (size_t)row * d;
    threadgroup float partial[32];
    float ss = 0.0f;
    for (uint i = lane; i < d; i += lanes) {
        float v = float(h[i]);
        if (dims.y != 0u) { v += float(a[i]); so[i] = half(v); }
        ss += v * v;
    }
    ss = simd_sum(ss);
    if ((lane & 31u) == 0u) partial[lane / 32u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (lanes + 31u) / 32u;
    if (lane < 32u) {
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(partial[0] / float(d) + eps);
    device half* n = normed + (size_t)row * d;
    for (uint i = lane; i < d; i += lanes) {
        float v = (dims.y != 0u) ? float(so[i]) : float(h[i]);
        n[i] = half(v * inv * (1.0f + w[i]));
    }
}
"#;

static ADDNORM: std::sync::OnceLock<Option<ScanCtx>> = std::sync::OnceLock::new();

fn addnorm_ctx() -> Option<(&'static MetalCtx, &'static ScanCtx)> {
    let base = ctx()?;
    let c = ADDNORM
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(ADD_NORM_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: add_rmsnorm shader error: {} - norms stay on CPU", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let function = {
                    let f: extern "C" fn(Id, Sel, Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(library, sel("newFunctionWithName:"), ns_string("add_rmsnorm_f16"))
                };
                let mut perr: Id = std::ptr::null_mut();
                let pipeline = {
                    let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                };
                if pipeline.is_null() {
                    println!("gpu: add_rmsnorm pipeline error: {} - norms stay on CPU", err_desc(perr));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                retain(pipeline);
                retain(function);
                retain(library);
                msg_void(pool, sel("drain"));
                Some(ScanCtx { pipeline })
            }
        })
        .as_ref()?;
    Some((base, c))
}


// ════════════════════════════════════════════════════════════════════════════
// Phase 3: per-layer command buffers - the linear layer's tissue kernels
// (causal conv + SiLU, gated RMSNorm) so a whole layer encodes at once
// ════════════════════════════════════════════════════════════════════════════

const LAYER_TISSUE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Causal depthwise conv over t rows of conv_dim channels, taps k, then
// SiLU. Sequential over time per channel (each thread owns a channel and
// walks all rows: the state carries the last k-1 inputs). f16 in/out,
// f32 math. state [conv_dim, k-1] in/out (f32).
kernel void causal_conv_silu_f16(device const half* x   [[buffer(0)]],
                                 device const float* w  [[buffer(1)]],
                                 device float* state    [[buffer(2)]],
                                 device half* y         [[buffer(3)]],
                                 constant uint3& dims   [[buffer(4)]], // t, conv_dim, k
                                 uint i [[thread_position_in_grid]]) {
    uint t = dims.x, cd = dims.y, k = dims.z;
    if (i >= cd) { return; }
    float st[8];
    for (uint j = 0; j + 1 < k; j++) { st[j] = state[i * (k - 1) + j]; }
    for (uint r = 0; r < t; r++) {
        float xv = float(x[(size_t)r * cd + i]);
        float acc = 0.0f;
        for (uint j = 0; j + 1 < k; j++) { acc += st[j] * w[i * k + j]; }
        acc += xv * w[i * k + (k - 1)];
        y[(size_t)r * cd + i] = half(acc / (1.0f + exp(-acc)));
        for (uint j = 0; j + 2 < k; j++) { st[j] = st[j + 1]; }
        st[k - 2] = xv;
    }
    for (uint j = 0; j + 1 < k; j++) { state[i * (k - 1) + j] = st[j]; }
}

// Scan prep from the conv output: per (row, value head h) with kv head
// kh = h / rep: q = l2norm(conv[q part, kh]) / sqrt(kd), k = l2norm(conv[k
// part, kh]), v = conv[v part, h]; beta = sigmoid(b_raw), decay =
// exp(-exp(a_log) * softplus(a_raw + dt_bias)). Outputs in the scan's
// layouts: q,k [t, heads, kd], v [t, heads, vd], beta/decay [t, heads].
// One threadgroup per (row, head); lanes stride kd/vd.
kernel void scan_prep_f16(device const half* conved  [[buffer(0)]], // [t, conv_dim]
                          device const half* b_raw   [[buffer(1)]], // [t, heads]
                          device const half* a_raw   [[buffer(2)]], // [t, heads]
                          device const float* a_log  [[buffer(3)]], // [heads]
                          device const float* dt_bias[[buffer(4)]], // [heads]
                          device float* q            [[buffer(5)]],
                          device float* k            [[buffer(6)]],
                          device float* v            [[buffer(7)]],
                          device float* beta         [[buffer(8)]],
                          device float* decay        [[buffer(9)]],
                          constant uint4& d0         [[buffer(10)]], // t, heads, kd, vd
                          constant uint4& d1         [[buffer(11)]], // rep, kt, conv_dim, 0
                          uint tg    [[threadgroup_position_in_grid]],
                          uint lane  [[thread_position_in_threadgroup]],
                          uint lanes [[threads_per_threadgroup]]) {
    uint t = d0.x, heads = d0.y, kd = d0.z, vd = d0.w;
    uint rep = d1.x, kt = d1.y, cd = d1.z;
    uint row = tg / heads, h = tg % heads;
    uint kh = h / max(rep, 1u);
    device const half* rowp = conved + (size_t)row * cd;
    threadgroup float partial[64];
    // q norm
    float sq = 0.0f, sk = 0.0f;
    for (uint i = lane; i < kd; i += lanes) {
        float a = float(rowp[kh * kd + i]);
        float b = float(rowp[kt + kh * kd + i]);
        sq += a * a; sk += b * b;
    }
    sq = simd_sum(sq); sk = simd_sum(sk);
    if ((lane & 31u) == 0u) { partial[lane / 32u] = sq; partial[32u + lane / 32u] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (lanes + 31u) / 32u;
    if (lane < 32u) {
        float a = (lane < nsg) ? partial[lane] : 0.0f;
        float b = (lane < nsg) ? partial[32u + lane] : 0.0f;
        a = simd_sum(a); b = simd_sum(b);
        if (lane == 0u) { partial[0] = a; partial[32] = b; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float nq = sqrt(partial[0] + 1e-6f), nk = sqrt(partial[32] + 1e-6f);
    float qs = rsqrt(float(kd));
    size_t ob = ((size_t)row * heads + h);
    for (uint i = lane; i < kd; i += lanes) {
        q[ob * kd + i] = float(rowp[kh * kd + i]) / nq * qs;
        k[ob * kd + i] = float(rowp[kt + kh * kd + i]) / nk;
    }
    for (uint i = lane; i < vd; i += lanes) {
        v[ob * vd + i] = float(rowp[2u * kt + h * vd + i]);
    }
    if (lane == 0u) {
        float br = float(b_raw[row * heads + h]);
        beta[ob] = 1.0f / (1.0f + exp(-br));
        float arg = float(a_raw[row * heads + h]) + dt_bias[h];
        float sp = (arg > 20.0f) ? arg : log(1.0f + exp(arg));
        decay[ob] = exp(-exp(a_log[h]) * sp);
    }
}

// Gated RMSNorm per (row, head): y = rmsnorm(x_h) * w * silu(z_h), the
// gated-DeltaNet output norm (direct weight, no offset). One threadgroup
// per (row, head); vd <= 256 lanes.
kernel void gated_rmsnorm_f16(device const float* mixed_hm [[buffer(0)]], // [heads, t, vd] (f32: the scan's output)
                              device const half* z        [[buffer(1)]], // [t, heads*vd]
                              device const float* w       [[buffer(2)]], // [vd]
                              device half* out            [[buffer(3)]], // [t, heads*vd]
                              constant uint3& dims        [[buffer(4)]], // t, heads, vd
                              constant float& eps         [[buffer(5)]],
                              uint tg    [[threadgroup_position_in_grid]],
                              uint lane  [[thread_position_in_threadgroup]],
                              uint lanes [[threads_per_threadgroup]]) {
    uint t = dims.x, heads = dims.y, vd = dims.z;
    uint row = tg / heads, h = tg % heads;
    device const float* xin = mixed_hm + ((size_t)h * t + row) * vd;
    device const half* zin = z + (size_t)row * heads * vd + h * vd;
    device half* o = out + (size_t)row * heads * vd + h * vd;
    threadgroup float partial[32];
    float ss = 0.0f;
    for (uint i = lane; i < vd; i += lanes) { float v = xin[i]; ss += v * v; }
    ss = simd_sum(ss);
    if ((lane & 31u) == 0u) partial[lane / 32u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (lanes + 31u) / 32u;
    if (lane < 32u) {
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(partial[0] / float(vd) + eps);
    for (uint i = lane; i < vd; i += lanes) {
        float zv = float(zin[i]);
        float g = zv / (1.0f + exp(-zv));
        o[i] = half(xin[i] * inv * w[i] * g);
    }
}
"#;

#[allow(dead_code)] // pipelines consumed by the per-layer encoder (phase 3 wiring)
pub(crate) struct TissueCtx {
    pub(crate) conv: Id,
    pub(crate) gnorm: Id,
    pub(crate) scanprep: Id,
}
unsafe impl Send for TissueCtx {}
unsafe impl Sync for TissueCtx {}

static TISSUE: std::sync::OnceLock<Option<TissueCtx>> = std::sync::OnceLock::new();

/// Compiles the phase-3 tissue kernels and reports; called by the GPU
/// bench so the next bare-metal run validates the MSL before the layer
/// encoder that uses them is wired.
pub fn tissue_probe() {
    match tissue_ctx() {
        Some(_) => println!("gpu: layer-tissue kernels ready (conv+silu, gated norm, scan prep)"),
        None => println!("gpu: layer-tissue kernels unavailable"),
    }
}

pub(crate) fn tissue_ctx() -> Option<(&'static MetalCtx, &'static TissueCtx)> {
    let base = ctx()?;
    let c = TISSUE
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(LAYER_TISSUE_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: layer-tissue shader error: {} - per-layer graph off", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let mk = |name: &str| -> Id {
                    let function = {
                        let f: extern "C" fn(Id, Sel, Id) -> Id =
                            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                        f(library, sel("newFunctionWithName:"), ns_string(name))
                    };
                    let mut perr: Id = std::ptr::null_mut();
                    let p = {
                        let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                        f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                    };
                    if p.is_null() {
                        println!("gpu: {} pipeline error: {} - per-layer graph off", name, err_desc(perr));
                    } else {
                        retain(p);
                        retain(function);
                    }
                    p
                };
                let conv = mk("causal_conv_silu_f16");
                let gnorm = mk("gated_rmsnorm_f16");
                let scanprep = mk("scan_prep_f16");
                retain(library);
                msg_void(pool, sel("drain"));
                if conv.is_null() || gnorm.is_null() || scanprep.is_null() {
                    return None;
                }
                Some(TissueCtx { conv, gnorm, scanprep })
            }
        })
        .as_ref()?;
    Some((base, c))
}

// ════════════════════════════════════════════════════════════════════════════
// Phase 3: one command buffer per linear layer
// ════════════════════════════════════════════════════════════════════════════
//
// The whole GatedDeltaNet layer - in_qkv/in_z GEMMs, in_b/in_a GEMMs,
// causal conv + SiLU, scan prep, delta scan, gated norm, out_proj GEMM,
// residual add, post-norm, gate/up GEMMs, SiLU*up, down GEMM - encodes
// into ONE command buffer with every intermediate resident on the GPU.
// The CPU sees the layer as: upload hidden (f16 view) once, one wait,
// read back the new hidden delta and the updated conv/scan states. The
// f32 residual stream stays on the CPU (added in f32 there), so precision
// across the 24 layers is unchanged from the CPU path.

/// Device buffers a linear layer's chain needs, grown on demand.
struct LayerArena {
    bufs: Vec<(Id, usize)>,
}

// SAFETY: device buffers are owned by the Metal runtime; the arena only
// hands out pointers under its own lock.
unsafe impl Send for LayerArena {}

static ARENA: std::sync::Mutex<Option<LayerArena>> = std::sync::Mutex::new(None);

/// Slot indices in the arena.
const A_HID: usize = 0; // hidden f16 [t, d]
const A_NORM: usize = 1; // normed f16 [t, d]
const A_QKV: usize = 2; // qkv f16 [t, conv_dim]
const A_Z: usize = 3; // z f16 [t, vt]
const A_BA: usize = 4; // b_raw|a_raw f16 [t, heads]x2
const A_CONV: usize = 5; // conved f16 [t, conv_dim]
const A_SCANIN: usize = 6; // q|k|v|beta|decay f32 (scan layouts)
const A_MIXHM: usize = 7; // mixed head-major f32 [heads, t, vd]
const A_MIXTM: usize = 8; // mixed token-major f16 [t, vt]
const A_ATTN: usize = 9; // attention out f16 [t, d]
const A_HID2: usize = 10; // hidden + attn f16 [t, d] (post-norm input)
const A_NORM2: usize = 11; // post-normed f16 [t, d]
const A_H: usize = 12; // gate|up f16 [t, inter]x2
const A_OUT: usize = 13; // mlp out f16 [t, d]
const A_STATE: usize = 14; // scan state f32 [heads, kd, vd]
const A_CONVST: usize = 15; // conv state f32 [conv_dim, k-1]
const A_SMALL: usize = 16; // small f32 params: norm weights, a_log, dt_bias, conv w
const A_N: usize = 17;

/// Ensures arena slot `i` holds at least `bytes`; returns (contents, buffer).
/// SAFETY: caller holds the ARENA lock for the whole layer encode.
unsafe fn arena_buf(base: &MetalCtx, arena: &mut LayerArena, i: usize, bytes: usize) -> (*mut c_void, Id) {
    while arena.bufs.len() < A_N {
        arena.bufs.push((std::ptr::null_mut(), 0));
    }
    unsafe { ensure_buf(base, &mut arena.bufs[i], bytes) }
}

/// Everything a linear layer needs, by reference (weights as slices of the
/// mapping or the q8/dequant caches).
#[allow(clippy::too_many_arguments)]
pub struct LinLayerRefs<'a> {
    pub in_qkv: &'a [f32],
    pub in_z: &'a [f32],
    pub in_b: &'a [f32],
    pub in_a: &'a [f32],
    pub conv_w: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_w: &'a [f32],
    pub out_proj: &'a [f32],
    pub post_norm_w: &'a [f32],
    pub gate: (&'a [u8], &'a [u8]),
    pub up: (&'a [u8], &'a [u8]),
    pub down: (&'a [u8], &'a [u8]),
}

/// Dimensions of a linear layer.
#[derive(Clone, Copy)]
pub struct LinDims {
    pub d: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub kd: usize,
    pub vd: usize,
    pub conv_k: usize,
    pub inter: usize,
    pub eps: f32,
}

/// One whole linear layer on the GPU. `hidden` [t, d] f32 in; on return
/// `attn_plus_mlp` [t, d] holds (attention output + MLP output) so the
/// caller adds it into its f32 residual stream; `conv_state` and
/// `scan_state` are updated in place. False = the caller runs the CPU
/// layer (nothing was mutated).
#[allow(clippy::too_many_arguments)]
/// Per-stage GPU micros of gpu_linear_layer under MICROKIMI_GPU_LAYER_PROF=1:
/// 0 norm, 1 projections, 2 conv, 3 scan prep, 4 scan, 5 gated norm,
/// 6 out_proj, 7 add+post-norm, 8 MLP.
static LAYER_PROF: [std::sync::atomic::AtomicU64; 9] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn layer_prof_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_GPU_LAYER_PROF").map(|v| v == "1").unwrap_or(false))
}

/// Prints and resets the per-stage split of the linear-layer command
/// buffer (when MICROKIMI_GPU_LAYER_PROF=1), in ms total since the last print.
pub fn layer_prof_print() {
    if !layer_prof_on() {
        return;
    }
    let names = ["norm", "proj", "conv", "scanprep", "scan", "gnorm", "out_proj", "add+norm", "mlp"];
    let v: Vec<u64> = LAYER_PROF.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed)).collect();
    let total: u64 = v.iter().sum();
    let parts: Vec<String> = names.iter().zip(&v).map(|(n, x)| format!("{} {:.2}", n, *x as f64 / 1000.0)).collect();
    println!("  layer stages (ms, GPU, all linear layers): {} | total {:.2}", parts.join(" | "), total as f64 / 1000.0);
}

pub fn gpu_linear_layer(
    w: &LinLayerRefs,
    gated_w: &[f32], // the DeltaNet output norm weight [vd]
    dm: LinDims,
    hidden: &[f32],
    t: usize,
    conv_state: &mut [f32],
    scan_state: &mut [f32],
    attn_plus_mlp: &mut [f32],
) -> bool {
    if !gemm_f16_on() || dm.kd > 128 || !scan_kd_ok(dm.kd) || dm.vd > 128 || dm.conv_k > 9 {
        return false;
    }
    let (d, heads, kd, vd, inter) = (dm.d, dm.heads, dm.kd, dm.vd, dm.inter);
    let kt = dm.kv_heads * kd;
    let vt = heads * vd;
    let conv_dim = 2 * kt + vt;
    let rep = heads / dm.kv_heads.max(1);
    let Some((base, mps)) = mps_ctx() else {
        return false;
    };
    let Some((_, tis)) = tissue_ctx() else {
        return false;
    };
    let Some((_, an)) = addnorm_ctx() else {
        return false;
    };
    let Some((_, sm)) = silu_ctx() else {
        return false;
    };
    let Some((_, sc)) = scan_ctx() else {
        return false;
    };
    // weights on device
    let Some(b_qkv) = weight_buffer_f16(base, mps, w.in_qkv, conv_dim, d) else { return false };
    let Some(b_z) = weight_buffer_f16(base, mps, w.in_z, vt, d) else { return false };
    let Some(b_b) = weight_buffer_f16(base, mps, w.in_b, heads, d) else { return false };
    let Some(b_a) = weight_buffer_f16(base, mps, w.in_a, heads, d) else { return false };
    let Some(b_out) = weight_buffer_f16(base, mps, w.out_proj, d, vt) else { return false };
    let Some(b_g) = dequant_buffer(base, mps, w.gate.0, w.gate.1, inter, d) else { return false };
    let Some(b_u) = dequant_buffer(base, mps, w.up.0, w.up.1, inter, d) else { return false };
    let Some(b_dn) = dequant_buffer(base, mps, w.down.0, w.down.1, d, inter) else { return false };
    // kernels
    let Some(k_qkv) = gemm_kernel(base, mps, t, conv_dim, d, true, 1.0) else { return false };
    let Some(k_z) = gemm_kernel(base, mps, t, vt, d, true, 1.0) else { return false };
    let Some(k_ba) = gemm_kernel(base, mps, t, heads, d, true, 1.0) else { return false };
    let Some(k_out) = gemm_kernel(base, mps, t, d, vt, true, 1.0) else { return false };
    let Some(k_gu) = gemm_kernel(base, mps, t, inter, d, true, 1.0) else { return false };
    let Some(k_dn) = gemm_kernel(base, mps, t, d, inter, true, 1.0) else { return false };
    let t_start = std::time::Instant::now();

    let mut arena_guard = ARENA.lock().unwrap();
    let arena = arena_guard.get_or_insert_with(|| LayerArena { bufs: Vec::new() });
    // SAFETY: the ARENA lock serializes layer encodes; every buffer below is
    // sized before use; waitUntilCompleted precedes the readbacks; the io
    // mutex is NOT taken (the arena is separate), so no deadlock with the
    // per-op paths.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        macro_rules! slot {
            ($i:expr, $bytes:expr) => {{
                let (p, b) = arena_buf(base, arena, $i, $bytes);
                if p.is_null() || b.is_null() {
                    msg_void(pool, sel("drain"));
                    return false;
                }
                (p, b)
            }};
        }
        let (p_hid, b_hid) = slot!(A_HID, t * d * 2);
        let (_, b_norm) = slot!(A_NORM, t * d * 2);
        let (_, b_qkvo) = slot!(A_QKV, t * conv_dim * 2);
        let (_, b_zo) = slot!(A_Z, t * vt * 2);
        let (_, b_ba) = slot!(A_BA, t * heads * 2 * 2);
        let (_, b_conv) = slot!(A_CONV, t * conv_dim * 2);
        let scan_q = 0usize;
        let scan_k = t * heads * kd;
        let scan_v = 2 * t * heads * kd;
        let scan_beta = scan_v + t * heads * vd;
        let scan_decay = scan_beta + t * heads;
        let scan_total = scan_decay + t * heads;
        let (_, b_scanin) = slot!(A_SCANIN, scan_total * 4);
        let (_, b_mixhm) = slot!(A_MIXHM, heads * t * vd * 4);
        let (_, b_mixtm) = slot!(A_MIXTM, t * vt * 2);
        let (p_attn, b_attn) = slot!(A_ATTN, t * d * 2);
        let (_, b_hid2) = slot!(A_HID2, t * d * 2);
        let (_, b_norm2) = slot!(A_NORM2, t * d * 2);
        let (_, b_h) = slot!(A_H, t * inter * 2 * 2);
        let (p_out, b_out_mlp) = slot!(A_OUT, t * d * 2);
        let (p_state, b_state) = slot!(A_STATE, scan_state.len() * 4);
        let (p_convst, b_convst) = slot!(A_CONVST, conv_state.len() * 4);
        // small params packed: [norm_w d | post_norm_w d | a_log heads | dt_bias heads | conv_w conv_dim*k]
        let off_norm = 0usize;
        let off_post = d;
        let off_alog = 2 * d;
        let off_dtb = 2 * d + heads;
        let off_convw = 2 * d + 2 * heads;
        let off_gated = off_convw + conv_dim * dm.conv_k;
        let small_total = off_gated + vd;
        let (p_small, b_small) = slot!(A_SMALL, small_total * 4);

        // uploads
        f32s_to_f16s(hidden, std::slice::from_raw_parts_mut(p_hid as *mut u16, t * d));
        std::ptr::copy_nonoverlapping(scan_state.as_ptr(), p_state as *mut f32, scan_state.len());
        std::ptr::copy_nonoverlapping(conv_state.as_ptr(), p_convst as *mut f32, conv_state.len());
        let sm_ptr = p_small as *mut f32;
        std::ptr::copy_nonoverlapping(w.norm_w.as_ptr(), sm_ptr.add(off_norm), d.min(w.norm_w.len()));
        std::ptr::copy_nonoverlapping(w.post_norm_w.as_ptr(), sm_ptr.add(off_post), d);
        std::ptr::copy_nonoverlapping(w.a_log.as_ptr(), sm_ptr.add(off_alog), heads);
        std::ptr::copy_nonoverlapping(w.dt_bias.as_ptr(), sm_ptr.add(off_dtb), heads);
        std::ptr::copy_nonoverlapping(w.conv_w.as_ptr(), sm_ptr.add(off_convw), conv_dim * dm.conv_k);
        std::ptr::copy_nonoverlapping(gated_w.as_ptr(), sm_ptr.add(off_gated), vd.min(gated_w.len()));

        // helpers
        let desc16 = |r: usize, c: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(class("MPSMatrixDescriptor"), sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"), r as u64, c as u64, (c * 2) as u64, MPS_FLOAT16)
        };
        let mat = |buf: Id, off: usize, dsc: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, u64, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(msg_id(class("MPSMatrix"), sel("alloc")), sel("initWithBuffer:offset:descriptor:"), buf, off as u64, dsc)
        };
        let enc_gemm: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_pipe: extern "C" fn(Id, Sel, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_buf: extern "C" fn(Id, Sel, Id, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_bytes: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_tg: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let sel_enc = sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:");
        let sel_pipe = sel("setComputePipelineState:");
        let sel_sb = sel("setBuffer:offset:atIndex:");
        let sel_by = sel("setBytes:length:atIndex:");
        let sel_dtg = sel("dispatchThreadgroups:threadsPerThreadgroup:");
        let sel_end = sel("endEncoding");
        let mut mats: Vec<Id> = Vec::new();
        let mut m = |buf: Id, off: usize, r: usize, c: usize| -> Id {
            let x = mat(buf, off, desc16(r, c));
            mats.push(x);
            x
        };

        let mut cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        // MICROKIMI_GPU_LAYER_PROF=1: each stage in its own command buffer,
        // GPU time per stage accumulated (relative costs, not the fused time)
        let prof = layer_prof_on();
        macro_rules! stage {
            ($i:expr) => {
                if prof {
                    msg_void(cmdbuf, sel("commit"));
                    msg_void(cmdbuf, sel("waitUntilCompleted"));
                    let getf: extern "C" fn(Id, Sel) -> f64 =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    let dt = getf(cmdbuf, sel("GPUEndTime")) - getf(cmdbuf, sel("GPUStartTime"));
                    LAYER_PROF[$i].fetch_add((dt * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
                    cmdbuf = msg_id(base.queue, sel("commandBuffer"));
                }
            };
        }

        // 1. input norm: normed = rmsnorm(hidden) * (1 + norm_w)   [hidden f16 view]
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, an.pipeline);
            set_buf(e, sel_sb, b_hid, 0, 0);
            set_buf(e, sel_sb, b_hid, 0, 1);
            set_buf(e, sel_sb, b_small, (off_norm * 4) as u64, 2);
            set_buf(e, sel_sb, b_norm, 0, 3);
            set_buf(e, sel_sb, b_norm, 0, 6);
            let dims: [u32; 2] = [d as u32, 0];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
            set_bytes(e, sel_by, (&dm.eps) as *const f32 as *const c_void, 4, 5);
            disp_tg(e, sel_dtg, MTLSize { width: t as u64, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        stage!(0);
        // 2. projections from normed: qkv, z, b, a
        let mx = m(b_norm, 0, t, d);
        enc_gemm(k_qkv, sel_enc, cmdbuf, mx, m(b_qkv, 0, conv_dim, d), m(b_qkvo, 0, t, conv_dim));
        enc_gemm(k_z, sel_enc, cmdbuf, mx, m(b_z, 0, vt, d), m(b_zo, 0, t, vt));
        enc_gemm(k_ba, sel_enc, cmdbuf, mx, m(b_b, 0, heads, d), m(b_ba, 0, t, heads));
        enc_gemm(k_ba, sel_enc, cmdbuf, mx, m(b_a, 0, heads, d), m(b_ba, t * heads * 2, t, heads));
        stage!(1);
        // 3. causal conv + SiLU (state carried)
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, tis.conv);
            set_buf(e, sel_sb, b_qkvo, 0, 0);
            set_buf(e, sel_sb, b_small, (off_convw * 4) as u64, 1);
            set_buf(e, sel_sb, b_convst, 0, 2);
            set_buf(e, sel_sb, b_conv, 0, 3);
            let dims: [u32; 3] = [t as u32, conv_dim as u32, dm.conv_k as u32];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 12, 4);
            let f: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(e, sel("dispatchThreads:threadsPerThreadgroup:"), MTLSize { width: conv_dim as u64, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        stage!(2);
        // 4. scan prep -> q,k,v,beta,decay (f32, scan layouts)
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, tis.scanprep);
            set_buf(e, sel_sb, b_conv, 0, 0);
            set_buf(e, sel_sb, b_ba, 0, 1);
            set_buf(e, sel_sb, b_ba, (t * heads * 2) as u64, 2);
            set_buf(e, sel_sb, b_small, (off_alog * 4) as u64, 3);
            set_buf(e, sel_sb, b_small, (off_dtb * 4) as u64, 4);
            set_buf(e, sel_sb, b_scanin, (scan_q * 4) as u64, 5);
            set_buf(e, sel_sb, b_scanin, (scan_k * 4) as u64, 6);
            set_buf(e, sel_sb, b_scanin, (scan_v * 4) as u64, 7);
            set_buf(e, sel_sb, b_scanin, (scan_beta * 4) as u64, 8);
            set_buf(e, sel_sb, b_scanin, (scan_decay * 4) as u64, 9);
            let d0: [u32; 4] = [t as u32, heads as u32, kd as u32, vd as u32];
            let d1: [u32; 4] = [rep as u32, kt as u32, conv_dim as u32, 0];
            set_bytes(e, sel_by, d0.as_ptr() as *const c_void, 16, 10);
            set_bytes(e, sel_by, d1.as_ptr() as *const c_void, 16, 11);
            disp_tg(e, sel_dtg, MTLSize { width: (t * heads) as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        stage!(3);
        // 5. delta scan (state in place, mixed head-major)
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, sc.pipeline);
            set_buf(e, sel_sb, b_scanin, (scan_q * 4) as u64, 0);
            set_buf(e, sel_sb, b_scanin, (scan_k * 4) as u64, 1);
            set_buf(e, sel_sb, b_scanin, (scan_v * 4) as u64, 2);
            set_buf(e, sel_sb, b_scanin, (scan_beta * 4) as u64, 3);
            set_buf(e, sel_sb, b_scanin, (scan_decay * 4) as u64, 4);
            set_buf(e, sel_sb, b_state, 0, 5);
            set_buf(e, sel_sb, b_mixhm, 0, 6);
            let dims: [u32; 4] = [t as u32, heads as u32, kd as u32, vd as u32];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 16, 7);
            disp_tg(e, sel_dtg, MTLSize { width: scan_groups(heads, vd) as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        stage!(4);
        // 6. gated norm: reads the scan's f32 head-major mix, writes the
        //    token-major f16 GEMM input
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, tis.gnorm);
            set_buf(e, sel_sb, b_mixhm, 0, 0);
            set_buf(e, sel_sb, b_zo, 0, 1);
            set_buf(e, sel_sb, b_small, (off_gated * 4) as u64, 2);
            set_buf(e, sel_sb, b_mixtm, 0, 3);
            let dims: [u32; 3] = [t as u32, heads as u32, vd as u32];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 12, 4);
            set_bytes(e, sel_by, (&dm.eps) as *const f32 as *const c_void, 4, 5);
            disp_tg(e, sel_dtg, MTLSize { width: (t * heads) as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        stage!(5);
        // 7. out_proj -> attn out
        enc_gemm(k_out, sel_enc, cmdbuf, m(b_mixtm, 0, t, vt), m(b_out, 0, d, vt), m(b_attn, 0, t, d));
        stage!(6);
        // 8. hidden2 = hidden + attn (device-side f16 sum, used ONLY as the
        //    post-norm input; the f32 residual add happens on the CPU from the
        //    returned attn+mlp), normed2 = rmsnorm(hidden2) * (1 + post_w)
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, an.pipeline);
            set_buf(e, sel_sb, b_hid, 0, 0);
            set_buf(e, sel_sb, b_attn, 0, 1);
            set_buf(e, sel_sb, b_small, (off_post * 4) as u64, 2);
            set_buf(e, sel_sb, b_norm2, 0, 3);
            set_buf(e, sel_sb, b_hid2, 0, 6);
            let dims: [u32; 2] = [d as u32, 1];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
            set_bytes(e, sel_by, (&dm.eps) as *const f32 as *const c_void, 4, 5);
            disp_tg(e, sel_dtg, MTLSize { width: t as u64, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        stage!(7);
        // 9. MLP: gate/up from normed2, SiLU*up, down -> out
        let mx2 = m(b_norm2, 0, t, d);
        enc_gemm(k_gu, sel_enc, cmdbuf, mx2, m(b_g, 0, inter, d), m(b_h, 0, t, inter));
        enc_gemm(k_gu, sel_enc, cmdbuf, mx2, m(b_u, 0, inter, d), m(b_h, t * inter * 2, t, inter));
        {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel_pipe, sm.pipeline);
            set_buf(e, sel_sb, b_h, 0, 0);
            set_buf(e, sel_sb, b_h, (t * inter * 2) as u64, 1);
            let n = (t * inter) as u32;
            set_bytes(e, sel_by, (&n) as *const u32 as *const c_void, 4, 2);
            let f: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(e, sel("dispatchThreads:threadsPerThreadgroup:"), MTLSize { width: (t * inter) as u64, height: 1, depth: 1 }, MTLSize { width: 256, height: 1, depth: 1 });
            msg_void(e, sel_end);
        }
        enc_gemm(k_dn, sel_enc, cmdbuf, m(b_h, 0, t, inter), m(b_dn, 0, d, inter), m(b_out_mlp, 0, t, d));

        stage!(8);
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));

        // readback: attn + mlp (both f16) summed into the caller's f32 buffer,
        // and the two states
        {
            let a16 = std::slice::from_raw_parts(p_attn as *const u16, t * d);
            let o16 = std::slice::from_raw_parts(p_out as *const u16, t * d);
            let mut tmp = vec![0.0f32; t * d];
            f16s_to_f32s(a16, attn_plus_mlp);
            f16s_to_f32s(o16, &mut tmp);
            for (x, y) in attn_plus_mlp.iter_mut().zip(&tmp) {
                *x += y;
            }
        }
        scan_state.copy_from_slice(std::slice::from_raw_parts(p_state as *const f32, scan_state.len()));
        conv_state.copy_from_slice(std::slice::from_raw_parts(p_convst as *const f32, conv_state.len()));
        for x in &mats {
            msg_void(*x, sel("release"));
        }
        msg_void(pool, sel("drain"));
    }
    gemm_account(t_start.elapsed().as_micros() as u64);
    true
}

// ════════════════════════════════════════════════════════════════════════════
// Phase 4: full-GPU decode - one token, all layers, one command buffer
// ════════════════════════════════════════════════════════════════════════════
//
// The decode regime is the opposite of prefill: ~100 matvecs of one row,
// where each Metal dispatch's sync latency (~0.25 ms) would swamp the
// arithmetic if any op waited on the CPU. So NOTHING waits: the token's
// entire forward - 24 layers, every norm/conv/scan/attention/MLP - is
// encoded into a single command buffer against resident f16 weights,
// resident conv/scan states and a resident KV cache, and the CPU waits
// once per token for the logits. Matvecs are MPS GEMMs with t=1 (rows =
// 1) - correct and simple; a hand matvec kernel would be the next
// refinement. The residual stream lives on the GPU for the token in
// f32, so precision matches the CPU path within f16 GEMM rounding.

const DECODE_TISSUE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// hidden (f32, d) -> normed (f16) with the (1+w) offset norm; optional
// add (f16, from a GEMM output) into hidden first, in f32.
kernel void dec_add_norm(device float* hidden      [[buffer(0)]],
                         device const half* add    [[buffer(1)]],
                         device const float* w     [[buffer(2)]],
                         device half* normed       [[buffer(3)]],
                         constant uint2& dims      [[buffer(4)]], // d, has_add
                         constant float& eps       [[buffer(5)]],
                         uint lane  [[thread_position_in_threadgroup]],
                         uint lanes [[threads_per_threadgroup]]) {
    uint d = dims.x;
    threadgroup float partial[32];
    float ss = 0.0f;
    for (uint i = lane; i < d; i += lanes) {
        float v = hidden[i];
        if (dims.y != 0u) { v += float(add[i]); hidden[i] = v; }
        ss += v * v;
    }
    ss = simd_sum(ss);
    if ((lane & 31u) == 0u) partial[lane / 32u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (lanes + 31u) / 32u;
    if (lane < 32u) {
        float v = (lane < nsg) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0u) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(partial[0] / float(d) + eps);
    for (uint i = lane; i < d; i += lanes) {
        normed[i] = half(hidden[i] * inv * (1.0f + w[i]));
    }
}

// Decode matvec: y[r] = W[r,:] . x, W f16 row-major, x f16, y f16 (f32
// accumulate). One SIMD group (32 lanes) per row, lanes stride the
// columns in 8-wide half loads, simd_sum reduces; 4 rows per threadgroup
// (128 threads). Weight rows stream once from device memory - the
// bandwidth-bound shape a decode wants; MPS GEMM at t=1 is not it.
kernel void dec_matvec(device const half* W   [[buffer(0)]],
                       device const half* x   [[buffer(1)]],
                       device half* y         [[buffer(2)]],
                       constant uint2& dims   [[buffer(3)]], // rows, cols
                       uint tid   [[thread_position_in_grid]],
                       uint lane  [[thread_index_in_simdgroup]]) {
    uint rows = dims.x, cols = dims.y;
    uint row = tid / 32u;
    if (row >= rows) { return; }
    device const half* w = W + (size_t)row * cols;
    float acc = 0.0f;
    // vector loads only when both bases are 8-half aligned (they are for
    // every shape in the model; the scalar loop is the safety net)
    bool aligned = ((cols & 7u) == 0u) && ((((size_t)x) & 15u) == 0u) && ((((size_t)W) & 15u) == 0u);
    uint c8 = aligned ? (cols / 8u) : 0u;
    for (uint i = lane; i < c8; i += 32u) {
        half4 w0 = *(device const half4*)(w + i * 8u);
        half4 w1 = *(device const half4*)(w + i * 8u + 4u);
        half4 x0 = *(device const half4*)(x + i * 8u);
        half4 x1 = *(device const half4*)(x + i * 8u + 4u);
        float4 p = float4(w0) * float4(x0) + float4(w1) * float4(x1);
        acc += p.x + p.y + p.z + p.w;
    }
    for (uint c = c8 * 8u + lane; c < cols; c += 32u) { acc += float(w[c]) * float(x[c]); }
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = half(acc); }
}

// Decode matvec on MXFP4 packed weights read as stored (nibbles + one
// e8m0 scale byte per block of 32): y[r] = W[r,:] . x, x f16, y f16, f32
// accumulate. One simdgroup per row; a lane takes 8 columns per step
// (one 32-bit word of nibbles, one scale byte, two half4 of x). Exact
// against the CPU dequantization (e2m1 x 2^k values are exact in f32).
constant float DEC_FP4_LUT[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                  -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

kernel void dec_matvec_fp4(device const uchar* packed [[buffer(0)]],
                           device const uchar* scales [[buffer(1)]],
                           device const half* x       [[buffer(2)]],
                           device half* y             [[buffer(3)]],
                           constant uint2& dims       [[buffer(4)]], // rows, cols (cols % 8 == 0)
                           uint tid   [[thread_position_in_grid]],
                           uint lane  [[thread_index_in_simdgroup]]) {
    uint rows = dims.x, cols = dims.y;
    uint row = tid / 32u;
    if (row >= rows) { return; }
    device const uchar* prow = packed + (size_t)row * (cols >> 1);
    device const uchar* srow = scales + (size_t)row * (cols >> 5);
    float acc = 0.0f;
    uint c8 = cols / 8u;
    for (uint g = lane; g < c8; g += 32u) {
        uint c0 = g * 8u;
        uint w = *(device const uint*)(prow + g * 4u);
        uchar sb = srow[c0 >> 5];
        float s = (sb == 0) ? as_type<float>(0x00400000u) : as_type<float>(uint(sb) << 23);
        half4 x0 = *(device const half4*)(x + c0);
        half4 x1 = *(device const half4*)(x + c0 + 4u);
        float4 w0 = float4(DEC_FP4_LUT[w & 15u], DEC_FP4_LUT[(w >> 4) & 15u], DEC_FP4_LUT[(w >> 8) & 15u], DEC_FP4_LUT[(w >> 12) & 15u]);
        float4 w1 = float4(DEC_FP4_LUT[(w >> 16) & 15u], DEC_FP4_LUT[(w >> 20) & 15u], DEC_FP4_LUT[(w >> 24) & 15u], DEC_FP4_LUT[(w >> 28) & 15u]);
        float4 p = w0 * float4(x0) + w1 * float4(x1);
        acc += s * (p.x + p.y + p.z + p.w);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = half(acc); }
}

// Decode matvec on q8_0 weights (int8 row-major + one f16 scale per
// block of 32): same shape as dec_matvec_fp4, 8 columns per lane per
// step (two char4 loads).
kernel void dec_matvec_q8(device const char* q      [[buffer(0)]],
                          device const half* sc     [[buffer(1)]],
                          device const half* x      [[buffer(2)]],
                          device half* y            [[buffer(3)]],
                          constant uint2& dims      [[buffer(4)]], // rows, cols (cols % 8 == 0)
                          uint tid   [[thread_position_in_grid]],
                          uint lane  [[thread_index_in_simdgroup]]) {
    uint rows = dims.x, cols = dims.y;
    uint row = tid / 32u;
    if (row >= rows) { return; }
    device const char* qrow = q + (size_t)row * cols;
    device const half* srow = sc + (size_t)row * (cols >> 5);
    float acc = 0.0f;
    uint c8 = cols / 8u;
    for (uint g = lane; g < c8; g += 32u) {
        uint c0 = g * 8u;
        char4 a = *(device const char4*)(qrow + c0);
        char4 b = *(device const char4*)(qrow + c0 + 4u);
        float s = float(srow[c0 >> 5]);
        half4 x0 = *(device const half4*)(x + c0);
        half4 x1 = *(device const half4*)(x + c0 + 4u);
        float4 p = float4(a) * float4(x0) + float4(b) * float4(x1);
        acc += s * (p.x + p.y + p.z + p.w);
    }
    acc = simd_sum(acc);
    if (lane == 0u) { y[row] = half(acc); }
}

// hidden += add (f16 GEMM output), f32 accumulate - the plain residual
kernel void dec_add(device float* hidden [[buffer(0)]], device const half* add [[buffer(1)]],
                    constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) { hidden[i] += float(add[i]); }
}

// One-token causal conv + SiLU: state [cd, k-1] carries the taps.
kernel void dec_conv(device const half* x     [[buffer(0)]], // [cd]
                     device const float* w    [[buffer(1)]], // [cd, k]
                     device float* state      [[buffer(2)]],
                     device half* y           [[buffer(3)]],
                     constant uint2& dims     [[buffer(4)]], // cd, k
                     uint i [[thread_position_in_grid]]) {
    uint cd = dims.x, k = dims.y;
    if (i >= cd) { return; }
    float xv = float(x[i]);
    float acc = 0.0f;
    for (uint j = 0; j + 1 < k; j++) { acc += state[i * (k - 1) + j] * w[i * k + j]; }
    acc += xv * w[i * k + (k - 1)];
    y[i] = half(acc / (1.0f + exp(-acc)));
    for (uint j = 0; j + 2 < k; j++) { state[i * (k - 1) + j] = state[i * (k - 1) + j + 1]; }
    state[i * (k - 1) + (k - 2)] = xv;
}

// One-token full attention for one head over the resident KV cache
// (f16 rows [pos, kv_width]); softmax; mix; sigmoid gate. One
// threadgroup (4 simdgroups) per head. Lanes split the head dim: lane
// l owns dims [l*per, (l+1)*per) with per = hd/32 (<= 8, hd <= 256);
// a simdgroup takes 32 positions per chunk (chunks interleaved across
// simdgroups), the score of position p is a simd_sum across the 32
// lanes, the chunk's online-softmax rescale happens once, and each
// lane accumulates its own dims over the chunk. The four simdgroups'
// (max, denominator, partial mix) combine through threadgroup memory
// at the end. Private state per lane: per q values + per accumulators.
kernel void dec_attn(device const half* q      [[buffer(0)]], // [n_heads*hd]
                     device const half* kc     [[buffer(1)]], // [len, kvw]
                     device const half* vc     [[buffer(2)]], // [len, kvw]
                     device const half* gate   [[buffer(3)]], // [n_heads*hd]
                     device half* out          [[buffer(4)]], // [n_heads*hd]
                     constant uint4& dims      [[buffer(5)]], // len, hd, kvw, groups
                     constant float& scale     [[buffer(6)]],
                     uint h     [[threadgroup_position_in_grid]],
                     uint lane  [[thread_position_in_threadgroup]],
                     uint lanes [[threads_per_threadgroup]]) {
    uint len = dims.x, hd = dims.y, kvw = dims.z, groups = dims.w;
    uint kh = h / groups;
    uint sg = lane / 32u, sl = lane & 31u;
    uint nsg = lanes / 32u;
    uint per = hd / 32u;
    threadgroup float tm[4];
    threadgroup float td[4];
    threadgroup float tacc[4 * 256];
    float qreg[8];
    float acc[8];
    for (uint j = 0; j < 8u; j++) { qreg[j] = 0.0f; acc[j] = 0.0f; }
    for (uint j = 0; j < per; j++) { qreg[j] = float(q[h * hd + sl * per + j]); }
    float m_run = -INFINITY, d_run = 0.0f;
    for (uint c0 = sg * 32u; c0 < len; c0 += nsg * 32u) {
        uint n = min(32u, len - c0);
        float s_mine = -INFINITY;
        for (uint p = 0; p < n; p++) {
            device const half* kr = kc + (size_t)(c0 + p) * kvw + kh * hd + sl * per;
            float s = 0.0f;
            for (uint j = 0; j < per; j++) { s += qreg[j] * float(kr[j]); }
            s = simd_sum(s) * scale;
            if (sl == p) { s_mine = s; }
        }
        float m_new = max(m_run, simd_max(s_mine));
        float r = exp(m_run - m_new);
        float e_mine = (sl < n) ? exp(s_mine - m_new) : 0.0f;
        d_run = d_run * r + simd_sum(e_mine);
        for (uint j = 0; j < per; j++) { acc[j] *= r; }
        for (uint p = 0; p < n; p++) {
            float e = simd_shuffle(e_mine, (ushort)p);
            device const half* vr = vc + (size_t)(c0 + p) * kvw + kh * hd + sl * per;
            for (uint j = 0; j < per; j++) { acc[j] += e * float(vr[j]); }
        }
        m_run = m_new;
    }
    if (sl == 0u) { tm[sg] = m_run; td[sg] = d_run; }
    for (uint j = 0; j < per; j++) { tacc[sg * hd + sl * per + j] = acc[j]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m_all = -INFINITY;
    for (uint g = 0; g < nsg; g++) { m_all = max(m_all, tm[g]); }
    float d_all = 0.0f;
    for (uint g = 0; g < nsg; g++) { d_all += (tm[g] == -INFINITY) ? 0.0f : td[g] * exp(tm[g] - m_all); }
    float inv = 1.0f / d_all;
    for (uint j = lane; j < hd; j += lanes) {
        float v = 0.0f;
        for (uint g = 0; g < nsg; g++) { v += (tm[g] == -INFINITY) ? 0.0f : tacc[g * hd + j] * exp(tm[g] - m_all); }
        float gt = float(gate[h * hd + j]);
        out[h * hd + j] = half(v * inv / (1.0f + exp(-gt)));
    }
}

// q/k per-head RMSNorm (offset weights) + partial RoPE at `pos`, gate
// split, k/v rows appended to the resident cache. One threadgroup per
// head: q heads first, then kv heads.
kernel void dec_qk_prep(device const half* qg      [[buffer(0)]],  // [n_heads, 2*hd]
                        device const half* kin     [[buffer(1)]],  // [kvw]
                        device const half* vin     [[buffer(2)]],  // [kvw]
                        device const float* qn     [[buffer(3)]],  // [hd]
                        device const float* kn     [[buffer(4)]],  // [hd]
                        device half* qout          [[buffer(5)]],  // [n_heads*hd]
                        device half* gout          [[buffer(6)]],  // [n_heads*hd]
                        device half* kc_row        [[buffer(7)]],  // cache row for pos
                        device half* vc_row        [[buffer(8)]],
                        constant uint4& dims       [[buffer(9)]],  // n_heads, n_kv, hd, rope_dim
                        constant uint& pos         [[buffer(10)]],
                        constant float& theta      [[buffer(11)]],
                        constant float& eps        [[buffer(12)]],
                        uint tg    [[threadgroup_position_in_grid]],
                        uint lane  [[thread_position_in_threadgroup]],
                        uint lanes [[threads_per_threadgroup]]) {
    uint nh = dims.x, hd = dims.z, rd = dims.w;
    threadgroup float row[256];
    threadgroup float red[32];
    bool is_q = tg < nh;
    uint h = is_q ? tg : tg - nh;
    for (uint j = lane; j < hd; j += lanes) {
        row[j] = is_q ? float(qg[h * 2 * hd + j]) : float(kin[h * hd + j]);
    }
    if (is_q) { for (uint j = lane; j < hd; j += lanes) { gout[h * hd + j] = qg[h * 2 * hd + hd + j]; } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.0f;
    for (uint j = lane; j < hd; j += lanes) { ss += row[j] * row[j]; }
    ss = simd_sum(ss);
    if ((lane & 31u) == 0u) red[lane / 32u] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint nsg = (lanes + 31u) / 32u;
    if (lane < 32u) { float v = (lane < nsg) ? red[lane] : 0.0f; v = simd_sum(v); if (lane == 0u) red[0] = v; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = rsqrt(red[0] / float(hd) + eps);
    device const float* wn = is_q ? qn : kn;
    for (uint j = lane; j < hd; j += lanes) { row[j] = row[j] * inv * (1.0f + wn[j]); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint half_rd = rd / 2;
    for (uint i = lane; i < half_rd; i += lanes) {
        float freq = pow(theta, -float(2u * i) / float(rd));
        float ang = float(pos) * freq;
        float c = cos(ang), s = sin(ang);
        float a = row[i], b = row[i + half_rd];
        row[i] = a * c - b * s;
        row[i + half_rd] = a * s + b * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_q) {
        for (uint j = lane; j < hd; j += lanes) { qout[h * hd + j] = half(row[j]); }
    } else {
        for (uint j = lane; j < hd; j += lanes) {
            kc_row[h * hd + j] = half(row[j]);
            vc_row[h * hd + j] = vin[h * hd + j];
        }
    }
}
"#;

pub(crate) struct DecodeCtx {
    pub(crate) add_norm: Id,
    pub(crate) add: Id,
    pub(crate) conv: Id,
    pub(crate) attn: Id,
    pub(crate) qk_prep: Id,
    pub(crate) matvec: Id,
    pub(crate) matvec_fp4: Id,
    pub(crate) matvec_q8: Id,
}
unsafe impl Send for DecodeCtx {}
unsafe impl Sync for DecodeCtx {}

static DECODE: std::sync::OnceLock<Option<DecodeCtx>> = std::sync::OnceLock::new();

pub(crate) fn decode_ctx() -> Option<(&'static MetalCtx, &'static DecodeCtx)> {
    let base = ctx()?;
    let c = DECODE
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(DECODE_TISSUE_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: decode shader error: {} - GPU decode off", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let mk = |name: &str| -> Id {
                    let function = {
                        let f: extern "C" fn(Id, Sel, Id) -> Id =
                            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                        f(library, sel("newFunctionWithName:"), ns_string(name))
                    };
                    let mut perr: Id = std::ptr::null_mut();
                    let p = {
                        let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                        f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                    };
                    if p.is_null() {
                        println!("gpu: {} pipeline error: {} - GPU decode off", name, err_desc(perr));
                    } else {
                        retain(p);
                        retain(function);
                    }
                    p
                };
                let add_norm = mk("dec_add_norm");
                let add = mk("dec_add");
                let conv = mk("dec_conv");
                let attn = mk("dec_attn");
                let qk_prep = mk("dec_qk_prep");
                let matvec = mk("dec_matvec");
                let matvec_fp4 = mk("dec_matvec_fp4");
                let matvec_q8 = mk("dec_matvec_q8");
                retain(library);
                msg_void(pool, sel("drain"));
                if [add_norm, add, conv, attn, qk_prep, matvec, matvec_fp4, matvec_q8].iter().any(|p| p.is_null()) {
                    return None;
                }
                Some(DecodeCtx { add_norm, add, conv, attn, qk_prep, matvec, matvec_fp4, matvec_q8 })
            }
        })
        .as_ref()?;
    Some((base, c))
}

/// Decode timing accumulators (MICROKIMI_GPU_DECODE_TIMING=1): encode
/// wall, GPU busy, kernel span, commit-to-completion wall (micros), steps.
static DECODE_TIMING: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

fn decode_timing_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_GPU_DECODE_TIMING").map(|v| v == "1").unwrap_or(false))
}

/// Prints the per-token decode timing split (when enabled) and resets it.
pub fn decode_timing_print() {
    if !decode_timing_on() {
        return;
    }
    let v: Vec<u64> = DECODE_TIMING.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed)).collect();
    let n = v[4].max(1) as f64;
    println!(
        "gpu decode timing ({} steps): encode {:.2} ms | gpu busy {:.2} ms | kernel span {:.2} ms | commit-to-done {:.2} ms (per token)",
        v[4],
        v[0] as f64 / n / 1000.0,
        v[1] as f64 / n / 1000.0,
        v[2] as f64 / n / 1000.0,
        v[3] as f64 / n / 1000.0
    );
}

/// Certifies `dec_matvec_fp4` and `dec_matvec_q8` on a synthetic
/// [rows, cols] matrix against CPU references: fp4 against the exact
/// dequantization of an MXFP4-quantized copy (the same nibbles), q8
/// against the reconstruction from the same int8 rows and f16 scales.
/// Prints one line each; returns the two max relative errors.
pub fn dec_matvec_check(rows: usize, cols: usize) -> Option<(f32, f32)> {
    let (base, dc) = decode_ctx()?;
    let (_, mps) = mps_ctx()?;
    let mut seed = 0x1234_5678u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..rows * cols).map(|_| rnd() * 0.1).collect();
    let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();
    let mut x16 = vec![0u16; cols];
    f32s_to_f16s(&x, &mut x16);
    let mut xr = vec![0.0f32; cols];
    f16s_to_f32s(&x16, &mut xr);
    // fp4 reference: quantize, dequantize exactly, f32 matvec on the f16-rounded x
    let (packed, scales) = crate::quant::mxfp4::quantize(&w, rows, cols);
    let wd = crate::quant::mxfp4::dequant(&packed, &scales, rows, cols);
    let mut ref_fp4 = vec![0.0f32; rows];
    for r in 0..rows {
        ref_fp4[r] = wd[r * cols..(r + 1) * cols].iter().zip(&xr).map(|(a, b)| a * b).sum();
    }
    // q8 reference: the same rows/scales the device holds
    let (bq, bs) = weight_buffer_q8(base, mps, &w, rows, cols)?;
    let (bp, bsc) = packed_buffers_fp4(base, mps, &packed, &scales, rows, cols)?;
    let mut ref_q8 = vec![0.0f32; rows];
    let nb = cols / 32;
    // SAFETY: readback of device buffers this function just built; then a
    // dispatch per kernel, wait, readback.
    unsafe {
        let contents: extern "C" fn(Id, Sel) -> *mut c_void =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let q = std::slice::from_raw_parts(contents(bq, sel("contents")) as *const i8, rows * cols);
        let sc = std::slice::from_raw_parts(contents(bs, sel("contents")) as *const u16, rows * nb);
        for r in 0..rows {
            let mut acc = 0.0f32;
            for g in 0..nb {
                let s = f16_to_f32_scalar(sc[r * nb + g]);
                let mut blk = 0.0f32;
                for i in 0..32 {
                    blk += q[r * cols + g * 32 + i] as f32 * xr[g * 32 + i];
                }
                acc += s * blk;
            }
            ref_q8[r] = acc;
        }
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let bx = f(base.device, sel("newBufferWithBytes:length:options:"), x16.as_ptr() as *const c_void, (cols * 2) as u64, 0);
        let y0 = vec![0u16; rows];
        let by1 = f(base.device, sel("newBufferWithBytes:length:options:"), y0.as_ptr() as *const c_void, (rows * 2) as u64, 0);
        let by2 = f(base.device, sel("newBufferWithBytes:length:options:"), y0.as_ptr() as *const c_void, (rows * 2) as u64, 0);
        if bx.is_null() || by1.is_null() || by2.is_null() {
            msg_void(pool, sel("drain"));
            return None;
        }
        let set_pipe: extern "C" fn(Id, Sel, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_buf: extern "C" fn(Id, Sel, Id, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_bytes: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_th: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
        let dims: [u32; 2] = [rows as u32, cols as u32];
        for (pipe, b0, b1, by) in [(dc.matvec_fp4, bp, bsc, by1), (dc.matvec_q8, bq, bs, by2)] {
            set_pipe(e, sel("setComputePipelineState:"), pipe);
            set_buf(e, sel("setBuffer:offset:atIndex:"), b0, 0, 0);
            set_buf(e, sel("setBuffer:offset:atIndex:"), b1, 0, 1);
            set_buf(e, sel("setBuffer:offset:atIndex:"), bx, 0, 2);
            set_buf(e, sel("setBuffer:offset:atIndex:"), by, 0, 3);
            set_bytes(e, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 8, 4);
            disp_th(
                e,
                sel("dispatchThreads:threadsPerThreadgroup:"),
                MTLSize { width: (rows * 32) as u64, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
        }
        msg_void(e, sel("endEncoding"));
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));
        let mut out_fp4 = vec![0.0f32; rows];
        let mut out_q8 = vec![0.0f32; rows];
        f16s_to_f32s(std::slice::from_raw_parts(contents(by1, sel("contents")) as *const u16, rows), &mut out_fp4);
        f16s_to_f32s(std::slice::from_raw_parts(contents(by2, sel("contents")) as *const u16, rows), &mut out_q8);
        for b in [bx, by1, by2] {
            msg_void(b, sel("release"));
        }
        msg_void(pool, sel("drain"));
        let rel = |name: &str, out: &[f32], reference: &[f32]| -> f32 {
            let mut max_abs = 0.0f32;
            let mut max_ref = 0.0f32;
            let mut worst = 0usize;
            for i in 0..rows {
                let e = (out[i] - reference[i]).abs();
                if e > max_abs || e.is_nan() {
                    max_abs = e;
                    worst = i;
                }
                max_ref = max_ref.max(reference[i].abs());
            }
            let rel = max_abs / max_ref.max(1e-30);
            println!(
                "gpu: {} check [{}x{}]: max abs {:.3e} rel {:.2e} at [{}] gpu {:.5} cpu {:.5} - {}",
                name, rows, cols, max_abs, rel, worst, out[worst], reference[worst],
                if rel < 5e-3 { "MATCH" } else { "MISMATCH" }
            );
            rel
        };
        let r1 = rel("dec_matvec_fp4", &out_fp4, &ref_fp4);
        let r2 = rel("dec_matvec_q8", &out_q8, &ref_q8);
        Some((r1, r2))
    }
}

/// Certifies and times the delta scan alone: synthetic q/k/v/beta/decay
/// for `t` tokens, `heads` heads, kd x vd state; CPU reference in the
/// delta_step order; prints the max relative error of out and state, the
/// GPU time per pass and the per-token-step latency.
pub fn delta_scan_check_bench(t: usize, heads: usize, kd: usize, vd: usize, reps: usize) -> Option<()> {
    let (base, sc) = scan_ctx()?;
    let mut seed = 0x5eed_1234u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..t * heads * kd).map(|_| rnd() * 0.3).collect();
    let k: Vec<f32> = (0..t * heads * kd).map(|_| rnd() * 0.3).collect();
    let v: Vec<f32> = (0..t * heads * vd).map(|_| rnd()).collect();
    let beta: Vec<f32> = (0..t * heads).map(|_| 0.2 + 0.7 * rnd().abs()).collect();
    let decay: Vec<f32> = (0..t * heads).map(|_| 0.9 + 0.1 * rnd().abs()).collect();
    let state0: Vec<f32> = (0..heads * kd * vd).map(|_| rnd() * 0.1).collect();
    // CPU reference
    let mut st = state0.clone();
    let mut out_ref = vec![0.0f32; heads * t * vd];
    for h in 0..heads {
        for tt in 0..t {
            let dec = decay[tt * heads + h];
            let bet = beta[tt * heads + h];
            let kt = &k[(tt * heads + h) * kd..(tt * heads + h + 1) * kd];
            let qt = &q[(tt * heads + h) * kd..(tt * heads + h + 1) * kd];
            for j in 0..vd {
                let vj = v[(tt * heads + h) * vd + j];
                let mut pred = 0.0f32;
                for i in 0..kd {
                    let sv = st[(h * kd + i) * vd + j] * dec;
                    st[(h * kd + i) * vd + j] = sv;
                    pred += kt[i] * sv;
                }
                let delta = (vj - pred) * bet;
                let mut o = 0.0f32;
                for i in 0..kd {
                    let sv = st[(h * kd + i) * vd + j] + kt[i] * delta;
                    st[(h * kd + i) * vd + j] = sv;
                    o += qt[i] * sv;
                }
                out_ref[(h * t + tt) * vd + j] = o;
            }
        }
    }
    // SAFETY: buffer creation, dispatches, waits, readbacks, releases.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let mk = |x: &[f32]| f(base.device, sel("newBufferWithBytes:length:options:"), x.as_ptr() as *const c_void, (x.len() * 4) as u64, 0);
        let (bq, bk, bv, bb, bd) = (mk(&q), mk(&k), mk(&v), mk(&beta), mk(&decay));
        let bs = mk(&state0);
        let bo = mk(&vec![0.0f32; heads * t * vd]);
        if [bq, bk, bv, bb, bd, bs, bo].iter().any(|b| b.is_null()) {
            msg_void(pool, sel("drain"));
            return None;
        }
        let set_pipe: extern "C" fn(Id, Sel, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_buf: extern "C" fn(Id, Sel, Id, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_bytes: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_tg: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let getf: extern "C" fn(Id, Sel) -> f64 =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let contents: extern "C" fn(Id, Sel) -> *mut c_void =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let encode = |cmdbuf: Id, state_buf: Id| {
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            set_pipe(e, sel("setComputePipelineState:"), sc.pipeline);
            let sb = sel("setBuffer:offset:atIndex:");
            set_buf(e, sb, bq, 0, 0);
            set_buf(e, sb, bk, 0, 1);
            set_buf(e, sb, bv, 0, 2);
            set_buf(e, sb, bb, 0, 3);
            set_buf(e, sb, bd, 0, 4);
            set_buf(e, sb, state_buf, 0, 5);
            set_buf(e, sb, bo, 0, 6);
            let dims: [u32; 4] = [t as u32, heads as u32, kd as u32, vd as u32];
            set_bytes(e, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 16, 7);
            disp_tg(
                e,
                sel("dispatchThreadgroups:threadsPerThreadgroup:"),
                MTLSize { width: scan_groups(heads, vd) as u64, height: 1, depth: 1 },
                MTLSize { width: 128, height: 1, depth: 1 },
            );
            msg_void(e, sel("endEncoding"));
        };
        // correctness pass on the real state
        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        encode(cmdbuf, bs);
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));
        let out_gpu = std::slice::from_raw_parts(contents(bo, sel("contents")) as *const f32, heads * t * vd).to_vec();
        let st_gpu = std::slice::from_raw_parts(contents(bs, sel("contents")) as *const f32, heads * kd * vd).to_vec();
        // timing passes on a scratch state (its content does not matter for time)
        let bs2 = mk(&state0);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
            for _ in 0..reps {
                encode(cmdbuf, bs2);
            }
            msg_void(cmdbuf, sel("commit"));
            msg_void(cmdbuf, sel("waitUntilCompleted"));
            best = best.min(getf(cmdbuf, sel("GPUEndTime")) - getf(cmdbuf, sel("GPUStartTime")));
        }
        for b in [bq, bk, bv, bb, bd, bs, bo, bs2] {
            msg_void(b, sel("release"));
        }
        msg_void(pool, sel("drain"));
        let rel = |g: &[f32], c: &[f32]| -> f32 {
            let mut max_abs = 0.0f32;
            let mut max_ref = 0.0f32;
            for i in 0..c.len() {
                let e = (g[i] - c[i]).abs();
                if e > max_abs || e.is_nan() {
                    max_abs = e;
                }
                max_ref = max_ref.max(c[i].abs());
            }
            max_abs / max_ref.max(1e-30)
        };
        let (ro, rs) = (rel(&out_gpu, &out_ref), rel(&st_gpu, &st));
        let per_pass = best / reps as f64;
        println!(
            "gpu: delta_scan check (t {}, heads {}, kd {}, vd {}): out rel {:.2e} state rel {:.2e} - {} | {:.2} ms/pass, {:.2} us per token step",
            t, heads, kd, vd, ro, rs,
            if ro < 1e-4 && rs < 1e-4 { "MATCH" } else { "MISMATCH" },
            per_pass * 1000.0, per_pass * 1e6 / t as f64
        );
        Some(())
    }
}

/// Streams a [rows, cols] f16 matrix through `dec_matvec` `reps` times
/// in one command buffer and prints the achieved GB/s (GPU time).
pub fn dec_matvec_bench(rows: usize, cols: usize, reps: usize) -> Option<f64> {
    let (base, dc) = decode_ctx()?;
    let w: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let x: Vec<f32> = (0..cols).map(|i| ((i * 31) % 100) as f32 / 100.0).collect();
    // SAFETY: buffer creation, dispatches, wait, release.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mkbuf = |x: &[f32]| -> Id {
            let mut h = vec![0u16; x.len()];
            f32s_to_f16s(x, &mut h);
            let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(base.device, sel("newBufferWithBytes:length:options:"), h.as_ptr() as *const c_void, (h.len() * 2) as u64, 0)
        };
        let bw = mkbuf(&w);
        let bx = mkbuf(&x);
        let by = mkbuf(&vec![0.0f32; rows]);
        if [bw, bx, by].iter().any(|b| b.is_null()) {
            msg_void(pool, sel("drain"));
            return None;
        }
        let set_pipe: extern "C" fn(Id, Sel, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_buf: extern "C" fn(Id, Sel, Id, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_bytes: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_th: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let getf: extern "C" fn(Id, Sel) -> f64 =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let mut best = f64::MAX;
        for _ in 0..3 {
            let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            for _ in 0..reps {
                set_pipe(e, sel("setComputePipelineState:"), dc.matvec);
                set_buf(e, sel("setBuffer:offset:atIndex:"), bw, 0, 0);
                set_buf(e, sel("setBuffer:offset:atIndex:"), bx, 0, 1);
                set_buf(e, sel("setBuffer:offset:atIndex:"), by, 0, 2);
                let dims: [u32; 2] = [rows as u32, cols as u32];
                set_bytes(e, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 8, 3);
                disp_th(
                    e,
                    sel("dispatchThreads:threadsPerThreadgroup:"),
                    MTLSize { width: (rows * 32) as u64, height: 1, depth: 1 },
                    MTLSize { width: 128, height: 1, depth: 1 },
                );
            }
            msg_void(e, sel("endEncoding"));
            msg_void(cmdbuf, sel("commit"));
            msg_void(cmdbuf, sel("waitUntilCompleted"));
            let t = getf(cmdbuf, sel("GPUEndTime")) - getf(cmdbuf, sel("GPUStartTime"));
            best = best.min(t);
        }
        for b in [bw, bx, by] {
            msg_void(b, sel("release"));
        }
        msg_void(pool, sel("drain"));
        let bytes = (rows * cols * 2) as f64 * reps as f64;
        let gbs = bytes / best / 1e9;
        println!(
            "gpu: dec_matvec [{}x{}] f16 x{}: {:.2} ms per pass, {:.1} GB/s",
            rows,
            cols,
            reps,
            best * 1000.0 / reps as f64,
            gbs
        );
        Some(gbs)
    }
}

/// Compiles the decode kernels and reports (bench probe).
pub fn decode_probe() {
    match decode_ctx() {
        Some(_) => println!("gpu: decode kernels ready (add_norm, add, conv, attn, qk_prep, matvec f16/fp4/q8)"),
        None => println!("gpu: decode kernels unavailable"),
    }
}

/// Certifies `dec_attn` on synthetic inputs against a CPU reference:
/// n_heads x n_kv heads, head dim `hd`, `len` cached positions (spans
/// several 32-position chunks and every simdgroup). Prints one line and
/// returns the max relative error (f16 inputs and output; ~1e-2 passes).
pub fn dec_attn_check(n_heads: usize, n_kv: usize, hd: usize, len: usize) -> Option<f32> {
    let (base, dc) = decode_ctx()?;
    let qw = n_heads * hd;
    let kvw = n_kv * hd;
    // deterministic pseudo-random inputs
    let mut seed = 0x9E37_79B9u32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let q: Vec<f32> = (0..qw).map(|_| rnd()).collect();
    let gate: Vec<f32> = (0..qw).map(|_| rnd() * 3.0).collect();
    let k: Vec<f32> = (0..len * kvw).map(|_| rnd()).collect();
    let v: Vec<f32> = (0..len * kvw).map(|_| rnd()).collect();
    // f16-rounded copies (what the GPU sees)
    let r16 = |x: &[f32]| -> Vec<f32> {
        let mut h = vec![0u16; x.len()];
        f32s_to_f16s(x, &mut h);
        let mut back = vec![0.0f32; x.len()];
        f16s_to_f32s(&h, &mut back);
        back
    };
    let (q16, g16, k16, v16) = (r16(&q), r16(&gate), r16(&k), r16(&v));
    let groups = n_heads / n_kv;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut reference = vec![0.0f32; qw];
    let mut scores = vec![0.0f32; len];
    for h in 0..n_heads {
        let kh = h / groups;
        let mut mx = f32::NEG_INFINITY;
        for t in 0..len {
            let mut sc = 0.0f32;
            for j in 0..hd {
                sc += q16[h * hd + j] * k16[t * kvw + kh * hd + j];
            }
            sc *= scale;
            scores[t] = sc;
            mx = mx.max(sc);
        }
        let mut den = 0.0f32;
        for sc in scores.iter_mut() {
            *sc = (*sc - mx).exp();
            den += *sc;
        }
        for t in 0..len {
            let a = scores[t] / den;
            for j in 0..hd {
                reference[h * hd + j] += a * v16[t * kvw + kh * hd + j];
            }
        }
        for j in 0..hd {
            let g = g16[h * hd + j];
            reference[h * hd + j] *= 1.0 / (1.0 + (-g).exp());
        }
    }
    let mut out = vec![0.0f32; qw];
    // SAFETY: plain buffer creation, one dispatch, wait, readback, release.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let mkbuf = |x: &[f32]| -> Id {
            let mut h = vec![0u16; x.len()];
            f32s_to_f16s(x, &mut h);
            let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(base.device, sel("newBufferWithBytes:length:options:"), h.as_ptr() as *const c_void, (h.len() * 2) as u64, 0)
        };
        let bq = mkbuf(&q);
        let bg = mkbuf(&gate);
        let bk = mkbuf(&k);
        let bv = mkbuf(&v);
        let bo = mkbuf(&out);
        if [bq, bg, bk, bv, bo].iter().any(|b| b.is_null()) {
            msg_void(pool, sel("drain"));
            return None;
        }
        let set_pipe: extern "C" fn(Id, Sel, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_buf: extern "C" fn(Id, Sel, Id, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_bytes: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_tg: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
        set_pipe(e, sel("setComputePipelineState:"), dc.attn);
        let sb = sel("setBuffer:offset:atIndex:");
        set_buf(e, sb, bq, 0, 0);
        set_buf(e, sb, bk, 0, 1);
        set_buf(e, sb, bv, 0, 2);
        set_buf(e, sb, bg, 0, 3);
        set_buf(e, sb, bo, 0, 4);
        let dims: [u32; 4] = [len as u32, hd as u32, kvw as u32, groups as u32];
        set_bytes(e, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 16, 5);
        set_bytes(e, sel("setBytes:length:atIndex:"), (&scale) as *const f32 as *const c_void, 4, 6);
        disp_tg(
            e,
            sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            MTLSize { width: n_heads as u64, height: 1, depth: 1 },
            MTLSize { width: 128, height: 1, depth: 1 },
        );
        msg_void(e, sel("endEncoding"));
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));
        let contents: extern "C" fn(Id, Sel) -> *mut c_void =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        f16s_to_f32s(std::slice::from_raw_parts(contents(bo, sel("contents")) as *const u16, qw), &mut out);
        for b in [bq, bg, bk, bv, bo] {
            msg_void(b, sel("release"));
        }
        msg_void(pool, sel("drain"));
    }
    let mut max_abs = 0.0f32;
    let mut max_ref = 0.0f32;
    let mut worst = 0usize;
    for i in 0..qw {
        let e = (out[i] - reference[i]).abs();
        if e > max_abs || e.is_nan() {
            max_abs = e;
            worst = i;
        }
        max_ref = max_ref.max(reference[i].abs());
    }
    let rel = max_abs / max_ref.max(1e-30);
    println!(
        "gpu: dec_attn check (heads {}x{}, hd {}, len {}): max abs {:.3e} rel {:.2e} at [{}] gpu {:.5} cpu {:.5} - {}",
        n_heads, n_kv, hd, len, max_abs, rel, worst, out[worst], reference[worst],
        if rel < 2e-2 { "MATCH" } else { "MISMATCH" }
    );
    Some(rel)
}

/// Per-layer resident state for the GPU decoder.
enum GpuLayerState {
    Linear { conv_state: Id, scan_state: Id },
    Full { kc: Id, vc: Id, cap: usize },
}

/// The resident GPU decoder for one model: weights are the shared f16 /
/// dequant caches; per-layer states and the KV cache live here; `hidden`
/// (f32, d) rides across the 24 layers inside one command buffer.
pub struct GpuDecoder {
    layers: Vec<GpuLayerState>,
    hidden: Id,   // f32 [d]
    normed: Id,   // f16 [d]
    tmp_a: Id,    // f16 scratch [max(conv_dim, 2*inter)]
    tmp_b: Id,    // f16 scratch [max(conv_dim, inter)]
    scan_in: Id,  // f32 [q|k|v|beta|decay] one token
    mix_hm: Id,   // f32 [heads*vd]
    mix_tm: Id,   // f16 [vt]
    ba: Id,       // f16 [2*heads]
    small: Id,    // f32 packed per-layer small params (see offsets)
    small_off: Vec<[usize; 6]>, // per layer: [in_norm, post_norm, a_log, dt_bias, conv_w, gated_w] (norm layers: [in_norm, post_norm, q_norm, k_norm, 0, 0])
    logits: Id,   // f16 [vocab]
    sizes: [usize; 3], // element counts of tmp_a, tmp_b, mix_tm (trace readback)
    d: usize,
    vocab: usize,
    pos: usize,
}
unsafe impl Send for GpuDecoder {}

impl GpuDecoder {
    /// Current position (next token index) of the resident state.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// What the decoder needs from the model, by reference.
pub struct DecodeModelRefs<'a> {
    pub layers: Vec<DecodeLayerRefs<'a>>,
    pub embed: &'a [f32],     // [vocab, d]
    pub norm_f: &'a [f32],    // [d]
    pub lm_head: &'a [f32],   // [vocab, d]
    pub d: usize,
    pub vocab: usize,
    pub eps: f32,
}

pub enum DecodeLayerRefs<'a> {
    Linear {
        in_norm: &'a [f32],
        post_norm: &'a [f32],
        w: LinLayerRefs<'a>,
        gated_w: &'a [f32],
        dm: LinDims,
    },
    Full {
        in_norm: &'a [f32],
        post_norm: &'a [f32],
        q_proj: &'a [f32],
        k_proj: &'a [f32],
        v_proj: &'a [f32],
        o_proj: &'a [f32],
        q_norm: &'a [f32],
        k_norm: &'a [f32],
        gate: (&'a [u8], &'a [u8]),
        up: (&'a [u8], &'a [u8]),
        down: (&'a [u8], &'a [u8]),
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        inter: usize,
    },
}

/// Builds the resident decoder for a model whose prefill state (conv,
/// scan, KV caches) is passed in f32 host layouts. `kv_cap` sizes the
/// resident KV cache in positions. False/None = GPU decode unavailable.
pub fn gpu_decoder_new(
    m: &DecodeModelRefs,
    lin_states: &[(&[f32], &[f32])], // per linear layer: (conv_state, scan_state)
    full_kv: &[(&[f32], &[f32], usize)], // per full layer: (k rows f32, v rows f32, len)
    kv_width: usize,
    kv_cap: usize,
    pos: usize,
) -> Option<GpuDecoder> {
    if !gemm_f16_on() {
        return None;
    }
    let (base, _dc) = decode_ctx()?;
    let (_, _mps) = mps_ctx()?;
    let _ = scan_ctx()?;
    let _ = silu_ctx()?;
    let d = m.d;
    let vocab = m.vocab;
    let mut max_a = 0usize;
    let mut max_b = 0usize;
    let mut scan_max = 0usize;
    let mut mix_max = 0usize;
    let mut vt_max = 0usize;
    let mut heads_max = 0usize;
    let mut small_total = 0usize;
    let mut small_off = Vec::new();
    for l in &m.layers {
        match l {
            DecodeLayerRefs::Linear { dm, .. } => {
                let kt = dm.kv_heads * dm.kd;
                let vt = dm.heads * dm.vd;
                let cd = 2 * kt + vt;
                max_a = max_a.max(cd).max(2 * dm.inter);
                max_b = max_b.max(cd).max(dm.inter);
                scan_max = scan_max.max(2 * dm.heads * dm.kd + dm.heads * dm.vd + 2 * dm.heads);
                mix_max = mix_max.max(dm.heads * dm.vd);
                vt_max = vt_max.max(vt);
                heads_max = heads_max.max(dm.heads);
                let offs = [
                    small_total,
                    small_total + d,
                    small_total + 2 * d,
                    small_total + 2 * d + dm.heads,
                    small_total + 2 * d + 2 * dm.heads,
                    small_total + 2 * d + 2 * dm.heads + cd * dm.conv_k,
                ];
                small_total = offs[5] + dm.vd;
                small_off.push(offs);
            }
            DecodeLayerRefs::Full { n_heads, n_kv, hd, inter, .. } => {
                let qw = n_heads * hd;
                max_a = max_a.max(qw * 2).max(2 * inter);
                max_b = max_b.max(qw).max(2 * n_kv * hd).max(*inter);
                // q | gate ride in mix_tm during attention
                vt_max = vt_max.max(2 * qw);
                let offs = [small_total, small_total + d, small_total + 2 * d, small_total + 2 * d + hd, 0, 0];
                small_total = offs[3] + hd;
                small_off.push(offs);
            }
        }
    }
    // SAFETY: allocations only; every buffer is retained for the decoder's life.
    unsafe {
        let newbuf = |bytes: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            let b = f(base.device, sel("newBufferWithLength:options:"), bytes.max(16) as u64, 0);
            if !b.is_null() {
                retain(b);
            }
            b
        };
        let contents = |b: Id| -> *mut c_void {
            let f: extern "C" fn(Id, Sel) -> *mut c_void =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(b, sel("contents"))
        };
        let hidden = newbuf(d * 4);
        let normed = newbuf(d * 2);
        let tmp_a = newbuf(max_a * 2);
        let tmp_b = newbuf(max_b * 2);
        let scan_in = newbuf(scan_max * 4);
        let mix_hm = newbuf(mix_max * 4);
        let mix_tm = newbuf(vt_max * 2);
        let ba = newbuf(2 * heads_max * 2);
        let small = newbuf(small_total * 4);
        let logits = newbuf(vocab * 2);
        if [hidden, normed, tmp_a, tmp_b, scan_in, mix_hm, mix_tm, ba, small, logits].iter().any(|b| b.is_null()) {
            return None;
        }
        // small params
        let sp = contents(small) as *mut f32;
        for (li, l) in m.layers.iter().enumerate() {
            let o = small_off[li];
            match l {
                DecodeLayerRefs::Linear { in_norm, post_norm, w, gated_w, dm } => {
                    let cd = 2 * dm.kv_heads * dm.kd + dm.heads * dm.vd;
                    std::ptr::copy_nonoverlapping(in_norm.as_ptr(), sp.add(o[0]), d);
                    std::ptr::copy_nonoverlapping(post_norm.as_ptr(), sp.add(o[1]), d);
                    std::ptr::copy_nonoverlapping(w.a_log.as_ptr(), sp.add(o[2]), dm.heads);
                    std::ptr::copy_nonoverlapping(w.dt_bias.as_ptr(), sp.add(o[3]), dm.heads);
                    std::ptr::copy_nonoverlapping(w.conv_w.as_ptr(), sp.add(o[4]), cd * dm.conv_k);
                    std::ptr::copy_nonoverlapping(gated_w.as_ptr(), sp.add(o[5]), dm.vd.min(gated_w.len()));
                }
                DecodeLayerRefs::Full { in_norm, post_norm, q_norm, k_norm, hd, .. } => {
                    std::ptr::copy_nonoverlapping(in_norm.as_ptr(), sp.add(o[0]), d);
                    std::ptr::copy_nonoverlapping(post_norm.as_ptr(), sp.add(o[1]), d);
                    std::ptr::copy_nonoverlapping(q_norm.as_ptr(), sp.add(o[2]), *hd);
                    std::ptr::copy_nonoverlapping(k_norm.as_ptr(), sp.add(o[3]), *hd);
                }
            }
        }
        // per-layer states
        let mut layers = Vec::new();
        let (mut li_lin, mut li_full) = (0usize, 0usize);
        for l in &m.layers {
            match l {
                DecodeLayerRefs::Linear { .. } => {
                    let (cs, ss) = lin_states[li_lin];
                    li_lin += 1;
                    let conv_state = newbuf(cs.len() * 4);
                    let scan_state = newbuf(ss.len() * 4);
                    if conv_state.is_null() || scan_state.is_null() {
                        return None;
                    }
                    std::ptr::copy_nonoverlapping(cs.as_ptr(), contents(conv_state) as *mut f32, cs.len());
                    std::ptr::copy_nonoverlapping(ss.as_ptr(), contents(scan_state) as *mut f32, ss.len());
                    layers.push(GpuLayerState::Linear { conv_state, scan_state });
                }
                DecodeLayerRefs::Full { .. } => {
                    let (k, v, len) = full_kv[li_full];
                    li_full += 1;
                    let cap = kv_cap.max(len + 1);
                    let kc = newbuf(cap * kv_width * 2);
                    let vc = newbuf(cap * kv_width * 2);
                    if kc.is_null() || vc.is_null() {
                        return None;
                    }
                    f32s_to_f16s(&k[..len * kv_width], std::slice::from_raw_parts_mut(contents(kc) as *mut u16, len * kv_width));
                    f32s_to_f16s(&v[..len * kv_width], std::slice::from_raw_parts_mut(contents(vc) as *mut u16, len * kv_width));
                    layers.push(GpuLayerState::Full { kc, vc, cap });
                }
            }
        }
        Some(GpuDecoder {
            layers,
            hidden,
            normed,
            tmp_a,
            tmp_b,
            scan_in,
            mix_hm,
            mix_tm,
            ba,
            small,
            small_off,
            logits,
            sizes: [max_a, max_b, vt_max],
            d,
            vocab,
            pos,
        })
    }
}

/// One decode token on the GPU: embed on the CPU (one row), then every
/// layer, the final norm and the lm_head inside ONE command buffer; the
/// f16 logits come back. Returns None when any weight/kernel is missing
/// (the caller falls back to the CPU forward - but note the resident
/// states have then diverged from the caller's; the caller must treat a
/// None as "GPU decode is off for this session").
pub fn gpu_decode_step(dec: &mut GpuDecoder, m: &DecodeModelRefs, token: u32, logits_out: &mut [f32]) -> Option<()> {
    gpu_decode_step_inner(dec, m, token, logits_out, None)
}

/// `gpu_decode_step` that also returns the f32 hidden state after every
/// layer (residual folded), for layer-by-layer comparison against the
/// CPU forward. The residual is folded eagerly per layer here (same
/// arithmetic as the fused fold, one extra encode per layer).
pub fn gpu_decode_step_trace(
    dec: &mut GpuDecoder,
    m: &DecodeModelRefs,
    token: u32,
    logits_out: &mut [f32],
    trace: &mut Vec<Vec<f32>>,
) -> Option<()> {
    gpu_decode_step_inner(dec, m, token, logits_out, Some(trace))
}

/// A decode weight matrix on the device, by storage.
#[derive(Clone, Copy)]
enum W {
    F16(Id),
    Q8(Id, Id),  // int8 rows, f16 block scales
    Fp4(Id, Id), // MXFP4 nibbles, e8m0 scale bytes
}

/// MICROKIMI_QWEN_GPU_DEC_F16=1: f16 storage for every decode matrix (the
/// A/B arm); default is q8_0 rows for f32 tensors and MXFP4 as stored.
fn decode_f16_arm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_QWEN_GPU_DEC_F16").map(|v| v == "1").unwrap_or(false))
}

/// Per-layer device weights of the decode graph.
struct LinW {
    qkv: W, z: W, b: W, a: W, out: W, g: W, u: W, dn: W,
}
struct FullW {
    q: W, k: W, v: W, o: W, g: W, u: W, dn: W,
}
enum LW { L(LinW), F(FullW) }

/// Resolves (builds on first use, then finds in the caches) every device
/// weight buffer of the decode graph. Storage per matrix: f16 copies
/// under MICROKIMI_QWEN_GPU_DEC_F16=1 (the A/B arm), else q8_0 rows for
/// the f32 tensors and the MXFP4 blobs as stored for the MLP. None when
/// any allocation fails.
fn resolve_decode_weights(base: &MetalCtx, mps: &MpsCtx, m: &DecodeModelRefs) -> Option<(Vec<LW>, W)> {
    let d = m.d;
    let vocab = m.vocab;
    let f16_arm = decode_f16_arm();
    let wf = |w: &[f32], rows: usize, cols: usize| -> Option<W> {
        if f16_arm {
            Some(W::F16(weight_buffer_f16(base, mps, w, rows, cols)?))
        } else {
            let (q, sc) = weight_buffer_q8(base, mps, w, rows, cols)?;
            Some(W::Q8(q, sc))
        }
    };
    let wp = |packed: &[u8], scales: &[u8], rows: usize, cols: usize| -> Option<W> {
        if f16_arm {
            Some(W::F16(dequant_buffer(base, mps, packed, scales, rows, cols)?))
        } else {
            let (bp, bs) = packed_buffers_fp4(base, mps, packed, scales, rows, cols)?;
            Some(W::Fp4(bp, bs))
        }
    };
    let mut lw: Vec<LW> = Vec::with_capacity(m.layers.len());
    for l in &m.layers {
        match l {
            DecodeLayerRefs::Linear { w, dm, .. } => {
                let kt = dm.kv_heads * dm.kd;
                let vt = dm.heads * dm.vd;
                let cd = 2 * kt + vt;
                lw.push(LW::L(LinW {
                    qkv: wf(w.in_qkv, cd, d)?,
                    z: wf(w.in_z, vt, d)?,
                    b: wf(w.in_b, dm.heads, d)?,
                    a: wf(w.in_a, dm.heads, d)?,
                    out: wf(w.out_proj, d, vt)?,
                    g: wp(w.gate.0, w.gate.1, dm.inter, d)?,
                    u: wp(w.up.0, w.up.1, dm.inter, d)?,
                    dn: wp(w.down.0, w.down.1, d, dm.inter)?,
                }));
            }
            DecodeLayerRefs::Full { q_proj, k_proj, v_proj, o_proj, gate, up, down, n_heads, n_kv, hd, inter, .. } => {
                let qw = n_heads * hd;
                let kvw = n_kv * hd;
                lw.push(LW::F(FullW {
                    q: wf(q_proj, qw * 2, d)?,
                    k: wf(k_proj, kvw, d)?,
                    v: wf(v_proj, kvw, d)?,
                    o: wf(o_proj, d, qw)?,
                    g: wp(gate.0, gate.1, *inter, d)?,
                    u: wp(up.0, up.1, *inter, d)?,
                    dn: wp(down.0, down.1, d, *inter)?,
                }));
            }
        }
    }
    let head_buf = wf(m.lm_head, vocab, d)?;
    Some((lw, head_buf))
}

/// Builds every device weight buffer of the decode graph ahead of the
/// first token (the q8_0 rows and MXFP4 uploads take seconds for a
/// billion parameters; done at load, the first token costs a token).
/// False when the GPU decode is unavailable.
pub fn gpu_decode_prepare(m: &DecodeModelRefs) -> bool {
    if !gemm_f16_on() {
        return false;
    }
    let Some((base, _dc)) = decode_ctx() else {
        return false;
    };
    let Some((_, mps)) = mps_ctx() else {
        return false;
    };
    resolve_decode_weights(base, mps, m).is_some()
}

// ════════════════════════════════════════════════════════════════════════════
// Tensor GEMM: the Metal 4 tensor ops (matmul2d) on the M5's GPU tensor units
// ════════════════════════════════════════════════════════════════════════════
//
// A prompt read is FLOP-bound; MPSMatrixMultiplication runs on the plain
// ALUs. The tensor-ops path (MetalPerformancePrimitives, M5/A19 and
// later, macOS 26) streams f16 tiles through the GPU's neural
// accelerators. Compiled at runtime like every other shader here; when
// the include or the device refuse, the probe says so and MPS stays.

const TGEMM_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// y[t, rows] (f16) = x[t, cols] (f16) . W[rows, cols]^T (f16), f32
// accumulation on the tensor units. One threadgroup (4 simdgroups) owns
// a TB-token x TA-row tile; K advances by TK per matmul2d run; the f32
// tile lands in threadgroup memory and leaves as f16.
constant int TA = 64;
constant int TB = 64;
constant int TK = 32;

kernel void tgemm_f16(device half* W         [[buffer(0)]], // (non-const: the tensor ops' operand types)
                      device half* X         [[buffer(1)]],
                      device half* Y         [[buffer(2)]],
                      constant uint4& dims   [[buffer(3)]], // rows, cols, t, x row stride (elements)
                      threadgroup float* tile [[threadgroup(0)]],
                      uint2 tgid  [[threadgroup_position_in_grid]],
                      ushort tid  [[thread_index_in_threadgroup]]) {
    const int rows = dims.x, cols = dims.y, t = dims.z, xs = dims.w;
    const int ra = tgid.y * TA;
    const int rb = tgid.x * TB;
    auto tA = tensor(W, dextents<int32_t, 2>(cols, rows), array<int, 2>({1, cols}));
    auto tB = tensor(X, dextents<int32_t, 2>(cols, t), array<int, 2>({1, xs}));
    matmul2d<matmul2d_descriptor(TB, TA, TK, false, true, true, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    for (int k = 0; k < cols; k += TK) {
        auto sB = tB.slice(k, rb);
        auto sA = tA.slice(k, ra);
        mm.run(sB, sA, cT);
    }
    auto tC = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(tile, dextents<int32_t, 2>(TA, TB));
    cT.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < TA * TB; i += 128) {
        int a = i % TA, b = i / TA;
        int r = ra + a, tt = rb + b;
        if (r < rows && tt < t) { Y[(size_t)tt * rows + r] = half(tile[b * TA + a]); }
    }
}
"#;

pub(crate) struct TgemmCtx {
    pub(crate) pipeline: Id,
}
unsafe impl Send for TgemmCtx {}
unsafe impl Sync for TgemmCtx {}

static TGEMM: std::sync::OnceLock<Option<TgemmCtx>> = std::sync::OnceLock::new();

/// The tensor GEMM pipeline, compiled once; None (with a `gpu:` line)
/// when the tensor ops are unavailable on this device or OS.
pub(crate) fn tgemm_ctx() -> Option<(&'static MetalCtx, &'static TgemmCtx)> {
    let base = ctx()?;
    let c = TGEMM
        .get_or_init(|| {
            // SAFETY: same shader-compilation sequence as init_ctx.
            unsafe {
                let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
                let src = ns_string(TGEMM_MSL);
                let mut err: Id = std::ptr::null_mut();
                let library = {
                    let f: extern "C" fn(Id, Sel, Id, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newLibraryWithSource:options:error:"), src, std::ptr::null_mut(), &mut err)
                };
                if library.is_null() {
                    println!("gpu: tensor GEMM unavailable - MPS GEMM stays:\n{}", err_desc(err));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                let function = {
                    let f: extern "C" fn(Id, Sel, Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(library, sel("newFunctionWithName:"), ns_string("tgemm_f16"))
                };
                let mut perr: Id = std::ptr::null_mut();
                let pipeline = {
                    let f: extern "C" fn(Id, Sel, Id, *mut Id) -> Id =
                        std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                    f(base.device, sel("newComputePipelineStateWithFunction:error:"), function, &mut perr)
                };
                if pipeline.is_null() {
                    println!("gpu: tensor GEMM pipeline error: {} - MPS GEMM stays", err_desc(perr));
                    msg_void(pool, sel("drain"));
                    return None;
                }
                retain(pipeline);
                retain(function);
                retain(library);
                msg_void(pool, sel("drain"));
                Some(TgemmCtx { pipeline })
            }
        })
        .as_ref()?;
    Some((base, c))
}

/// Encodes y[t, rows] = x[t, cols] . W[rows, cols]^T on the tensor GEMM
/// (all f16, byte offsets into the buffers, `x_stride` in elements).
/// SAFETY: `e` is a live compute encoder; buffers outlive the command.
unsafe fn tgemm_encode(tg: &TgemmCtx, e: Id, w: Id, w_off: usize, x: Id, x_off: usize, x_stride: usize, y: Id, y_off: usize, t: usize, rows: usize, cols: usize) {
    // SAFETY: typed objc_msgSend signatures matching the Metal selectors.
    let (set_pipe, set_buf, set_bytes, set_tgm, disp_tg) = unsafe {
        (
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, extern "C" fn(Id, Sel, Id)>(objc_msgSend),
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, extern "C" fn(Id, Sel, Id, u64, u64)>(objc_msgSend),
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, extern "C" fn(Id, Sel, *const c_void, u64, u64)>(objc_msgSend),
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, extern "C" fn(Id, Sel, u64, u64)>(objc_msgSend),
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, extern "C" fn(Id, Sel, MTLSize, MTLSize)>(objc_msgSend),
        )
    };
    set_pipe(e, sel("setComputePipelineState:"), tg.pipeline);
    set_buf(e, sel("setBuffer:offset:atIndex:"), w, w_off as u64, 0);
    set_buf(e, sel("setBuffer:offset:atIndex:"), x, x_off as u64, 1);
    set_buf(e, sel("setBuffer:offset:atIndex:"), y, y_off as u64, 2);
    let dims: [u32; 4] = [rows as u32, cols as u32, t as u32, x_stride as u32];
    set_bytes(e, sel("setBytes:length:atIndex:"), dims.as_ptr() as *const c_void, 16, 3);
    set_tgm(e, sel("setThreadgroupMemoryLength:atIndex:"), (64 * 64 * 4) as u64, 0);
    disp_tg(
        e,
        sel("dispatchThreadgroups:threadsPerThreadgroup:"),
        MTLSize { width: t.div_ceil(64) as u64, height: rows.div_ceil(64) as u64, depth: 1 },
        MTLSize { width: 128, height: 1, depth: 1 },
    );
}

/// Certifies and times the tensor GEMM against the CPU reference and the
/// MPS GEMM on one shape (y[t, rows] = x[t, cols] . W^T, all f16 storage):
/// prints the max relative error of both and their TFLOP/s.
pub fn tgemm_check_bench(t: usize, rows: usize, cols: usize, reps: usize) -> Option<()> {
    let (base, tg) = tgemm_ctx()?;
    let (_, mps) = mps_ctx()?;
    let mut seed = 0x0badf00du32;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let w: Vec<f32> = (0..rows * cols).map(|_| rnd() * 0.1).collect();
    let x: Vec<f32> = (0..t * cols).map(|_| rnd()).collect();
    let r16 = |v: &[f32]| -> (Vec<u16>, Vec<f32>) {
        let mut h = vec![0u16; v.len()];
        f32s_to_f16s(v, &mut h);
        let mut back = vec![0.0f32; v.len()];
        f16s_to_f32s(&h, &mut back);
        (h, back)
    };
    let (w16, wr) = r16(&w);
    let (x16, xr) = r16(&x);
    // CPU reference on a sample of rows (the full product is t*rows*cols)
    let sample: Vec<usize> = (0..t).step_by((t / 8).max(1)).collect();
    let mut reference = vec![0.0f32; t * rows];
    for &tt in &sample {
        for r in 0..rows {
            let mut acc = 0.0f32;
            for c in 0..cols {
                acc += xr[tt * cols + c] * wr[r * cols + c];
            }
            reference[tt * rows + r] = acc;
        }
    }
    let Some(k_mps) = gemm_kernel(base, mps, t, rows, cols, true, 1.0) else {
        return None;
    };
    // SAFETY: buffer creation, dispatches, waits, readbacks, releases.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let bw = f(base.device, sel("newBufferWithBytes:length:options:"), w16.as_ptr() as *const c_void, (w16.len() * 2) as u64, 0);
        let bx = f(base.device, sel("newBufferWithBytes:length:options:"), x16.as_ptr() as *const c_void, (x16.len() * 2) as u64, 0);
        let y0 = vec![0u16; t * rows];
        let by1 = f(base.device, sel("newBufferWithBytes:length:options:"), y0.as_ptr() as *const c_void, (y0.len() * 2) as u64, 0);
        let by2 = f(base.device, sel("newBufferWithBytes:length:options:"), y0.as_ptr() as *const c_void, (y0.len() * 2) as u64, 0);
        if [bw, bx, by1, by2].iter().any(|b| b.is_null()) {
            msg_void(pool, sel("drain"));
            return None;
        }
        let getf: extern "C" fn(Id, Sel) -> f64 =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let contents: extern "C" fn(Id, Sel) -> *mut c_void =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        // tensor GEMM: reps in one command buffer, best of 3
        let mut best_t = f64::MAX;
        for _ in 0..3 {
            let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
            let e = msg_id(cmdbuf, sel("computeCommandEncoder"));
            for _ in 0..reps {
                tgemm_encode(tg, e, bw, 0, bx, 0, cols, by1, 0, t, rows, cols);
            }
            msg_void(e, sel("endEncoding"));
            msg_void(cmdbuf, sel("commit"));
            msg_void(cmdbuf, sel("waitUntilCompleted"));
            best_t = best_t.min(getf(cmdbuf, sel("GPUEndTime")) - getf(cmdbuf, sel("GPUStartTime")));
        }
        // MPS GEMM: same
        let desc16 = |r: usize, c: usize| -> Id {
            let f: extern "C" fn(Id, Sel, u64, u64, u64, u32) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(class("MPSMatrixDescriptor"), sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"), r as u64, c as u64, (c * 2) as u64, MPS_FLOAT16)
        };
        let mat = |buf: Id, dsc: Id| -> Id {
            let f: extern "C" fn(Id, Sel, Id, u64, Id) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(msg_id(class("MPSMatrix"), sel("alloc")), sel("initWithBuffer:offset:descriptor:"), buf, 0, dsc)
        };
        let mx = mat(bx, desc16(t, cols));
        let mw = mat(bw, desc16(rows, cols));
        let my = mat(by2, desc16(t, rows));
        let enc_gemm: extern "C" fn(Id, Sel, Id, Id, Id, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let mut best_m = f64::MAX;
        for _ in 0..3 {
            let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
            for _ in 0..reps {
                enc_gemm(k_mps, sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"), cmdbuf, mx, mw, my);
            }
            msg_void(cmdbuf, sel("commit"));
            msg_void(cmdbuf, sel("waitUntilCompleted"));
            best_m = best_m.min(getf(cmdbuf, sel("GPUEndTime")) - getf(cmdbuf, sel("GPUStartTime")));
        }
        let mut out_t = vec![0.0f32; t * rows];
        let mut out_m = vec![0.0f32; t * rows];
        f16s_to_f32s(std::slice::from_raw_parts(contents(by1, sel("contents")) as *const u16, t * rows), &mut out_t);
        f16s_to_f32s(std::slice::from_raw_parts(contents(by2, sel("contents")) as *const u16, t * rows), &mut out_m);
        for b in [mx, mw, my, bw, bx, by1, by2] {
            msg_void(b, sel("release"));
        }
        msg_void(pool, sel("drain"));
        let rel = |out: &[f32]| -> (f32, usize) {
            let mut max_abs = 0.0f32;
            let mut max_ref = 0.0f32;
            let mut worst = 0usize;
            for &tt in &sample {
                for r in 0..rows {
                    let i = tt * rows + r;
                    let e = (out[i] - reference[i]).abs();
                    if e > max_abs || e.is_nan() {
                        max_abs = e;
                        worst = i;
                    }
                    max_ref = max_ref.max(reference[i].abs());
                }
            }
            (max_abs / max_ref.max(1e-30), worst)
        };
        let (rt, wt) = rel(&out_t);
        let (rm, _) = rel(&out_m);
        let flops = 2.0 * t as f64 * rows as f64 * cols as f64 * reps as f64;
        println!(
            "gpu: tensor GEMM check [{}x{}]x[{}x{}]^T: rel {:.2e} (at [{}] gpu {:.4} cpu {:.4}) - {} | tensor {:.2} ms/pass {:.2} TFLOP/s | MPS rel {:.2e} {:.2} ms/pass {:.2} TFLOP/s",
            t, cols, rows, cols, rt, wt, out_t[wt], reference[wt],
            if rt < 5e-3 { "MATCH" } else { "MISMATCH" },
            best_t * 1000.0 / reps as f64, flops / best_t / 1e12,
            rm, best_m * 1000.0 / reps as f64, flops / best_m / 1e12
        );
        Some(())
    }
}

/// Trace stop point: encoding stops after sub-stage `stage` of layer
/// `layer` (full-attention layers: 0 qkv projections, 1 qk prep, 2
/// attention, 3 o_proj), leaving the scratch buffers readable through
/// `GpuDecoder::scratch` (a verifier for the layer's pieces).
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TraceStop {
    pub layer: usize,
    pub stage: u8,
}

/// `gpu_decode_step_trace` that stops encoding at `stop` (see TraceStop);
/// the logits are not computed, `dec.pos` does not advance.
pub fn gpu_decode_step_stop(dec: &mut GpuDecoder, m: &DecodeModelRefs, token: u32, stop: TraceStop) -> Option<()> {
    let mut sink = vec![0.0f32; m.vocab];
    let mut tr = Vec::new();
    gpu_decode_step_impl(dec, m, token, &mut sink, Some(&mut tr), Some(stop))
}

fn gpu_decode_step_inner(
    dec: &mut GpuDecoder,
    m: &DecodeModelRefs,
    token: u32,
    logits_out: &mut [f32],
    trace: Option<&mut Vec<Vec<f32>>>,
) -> Option<()> {
    gpu_decode_step_impl(dec, m, token, logits_out, trace, None)
}

/// f32 copies of the decoder's scratch buffers (trace verifier).
pub struct DecodeScratch {
    pub normed: Vec<f32>,
    pub tmp_a: Vec<f32>,
    pub tmp_b: Vec<f32>,
    pub mix_tm: Vec<f32>,
}

impl GpuDecoder {
    /// Reads the f16 scratch buffers back as f32.
    pub fn scratch(&self) -> DecodeScratch {
        // SAFETY: buffers are retained for the decoder's life; sizes are
        // the allocation sizes recorded at construction.
        unsafe {
            let contents = |b: Id| -> *mut c_void {
                let f: extern "C" fn(Id, Sel) -> *mut c_void =
                    std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                f(b, sel("contents"))
            };
            let rd = |b: Id, n: usize| -> Vec<f32> {
                let mut out = vec![0.0f32; n];
                f16s_to_f32s(std::slice::from_raw_parts(contents(b) as *const u16, n), &mut out);
                out
            };
            DecodeScratch {
                normed: rd(self.normed, self.d),
                tmp_a: rd(self.tmp_a, self.sizes[0]),
                tmp_b: rd(self.tmp_b, self.sizes[1]),
                mix_tm: rd(self.mix_tm, self.sizes[2]),
            }
        }
    }

    /// Copies the resident state back to host layouts: every linear
    /// layer's (conv, scan) state in f32, and for every full layer the KV
    /// rows [from, pos) appended (f16 -> f32) to the given vectors, where
    /// `from` is that cache's current row count. Returns the position.
    pub fn export(
        &self,
        lin_states: &mut [(&mut [f32], &mut [f32])],
        full_kv: &mut [(&mut Vec<f32>, &mut Vec<f32>, usize)],
        kv_width: usize,
    ) -> usize {
        let (mut li, mut fi) = (0usize, 0usize);
        // SAFETY: buffers are retained for the decoder's life; row ranges
        // are within the capacity checked at every step.
        unsafe {
            let contents = |b: Id| -> *mut c_void {
                let f: extern "C" fn(Id, Sel) -> *mut c_void =
                    std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                f(b, sel("contents"))
            };
            for l in &self.layers {
                match l {
                    GpuLayerState::Linear { conv_state, scan_state } => {
                        let (conv, state) = &mut lin_states[li];
                        li += 1;
                        std::ptr::copy_nonoverlapping(contents(*conv_state) as *const f32, conv.as_mut_ptr(), conv.len());
                        std::ptr::copy_nonoverlapping(contents(*scan_state) as *const f32, state.as_mut_ptr(), state.len());
                    }
                    GpuLayerState::Full { kc, vc, cap } => {
                        let (k, v, from) = &mut full_kv[fi];
                        fi += 1;
                        let to = self.pos.min(*cap);
                        if *from >= to {
                            continue;
                        }
                        let n = (to - *from) * kv_width;
                        let mut tmp = vec![0.0f32; n];
                        f16s_to_f32s(std::slice::from_raw_parts((contents(*kc) as *const u16).add(*from * kv_width), n), &mut tmp);
                        k.truncate(*from * kv_width);
                        k.extend_from_slice(&tmp);
                        f16s_to_f32s(std::slice::from_raw_parts((contents(*vc) as *const u16).add(*from * kv_width), n), &mut tmp);
                        v.truncate(*from * kv_width);
                        v.extend_from_slice(&tmp);
                    }
                }
            }
        }
        self.pos
    }

    /// The resident KV cache row `row` of full-attention layer `layer`
    /// (k, v) as f32, or None for a linear layer / out-of-range row.
    pub fn kv_row(&self, layer: usize, row: usize, kv_width: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        match self.layers.get(layer)? {
            GpuLayerState::Full { kc, vc, cap } => {
                if row >= *cap {
                    return None;
                }
                // SAFETY: as above.
                unsafe {
                    let contents = |b: Id| -> *mut c_void {
                        let f: extern "C" fn(Id, Sel) -> *mut c_void =
                            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                        f(b, sel("contents"))
                    };
                    let mut k = vec![0.0f32; kv_width];
                    let mut v = vec![0.0f32; kv_width];
                    f16s_to_f32s(std::slice::from_raw_parts((contents(*kc) as *const u16).add(row * kv_width), kv_width), &mut k);
                    f16s_to_f32s(std::slice::from_raw_parts((contents(*vc) as *const u16).add(row * kv_width), kv_width), &mut v);
                    Some((k, v))
                }
            }
            GpuLayerState::Linear { .. } => None,
        }
    }
}

fn gpu_decode_step_impl(
    dec: &mut GpuDecoder,
    m: &DecodeModelRefs,
    token: u32,
    logits_out: &mut [f32],
    mut trace: Option<&mut Vec<Vec<f32>>>,
    stop: Option<TraceStop>,
) -> Option<()> {
    // kernel budgets: the linear layer's private arrays take head dims up
    // to 128; dec_attn/dec_qk_prep take hd up to 256 (a multiple of 32);
    // refuse cleanly beyond
    for l in &m.layers {
        match l {
            DecodeLayerRefs::Linear { dm, .. } => {
                if dm.kd > 128 || !scan_kd_ok(dm.kd) || dm.vd > 128 || dm.conv_k > 9 {
                    return None;
                }
            }
            DecodeLayerRefs::Full { hd, .. } => {
                if *hd > 256 || *hd % 32 != 0 {
                    return None;
                }
            }
        }
    }
    let (base, dc) = decode_ctx()?;
    let (_, mps) = mps_ctx()?;
    let (_, sc) = scan_ctx()?;
    let (_, sm) = silu_ctx()?;
    let (_, tis) = tissue_ctx()?;
    let d = dec.d;
    let vocab = dec.vocab;
    let pos = dec.pos;
    let t_start = std::time::Instant::now();
    // resolve every weight buffer and kernel BEFORE encoding (no partial
    // state mutation on refusal)
    let (lw, head_buf) = resolve_decode_weights(base, mps, m)?;
    let final_norm = m.norm_f;


    let mut stopped = false;
    // SAFETY: all objects come from the live context; the decoder's buffers
    // are retained for its lifetime; waitUntilCompleted precedes readback.
    unsafe {
        let pool = msg_id(msg_id(class("NSAutoreleasePool"), sel("alloc")), sel("init"));
        let contents = |b: Id| -> *mut c_void {
            let f: extern "C" fn(Id, Sel) -> *mut c_void =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(b, sel("contents"))
        };
        // embed the token on the CPU into the resident f32 hidden
        let hp = contents(dec.hidden) as *mut f32;
        std::ptr::copy_nonoverlapping(m.embed.as_ptr().add(token as usize * d), hp, d);
        // final-norm weight rides in the small buffer's tail? no: stage it
        // into normed's neighbor... keep simple: a dedicated tiny buffer
        let fnw = {
            let f: extern "C" fn(Id, Sel, *const c_void, u64, u64) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            f(base.device, sel("newBufferWithBytes:length:options:"), final_norm.as_ptr() as *const c_void, (d * 4) as u64, 0)
        };
        if fnw.is_null() {
            msg_void(pool, sel("drain"));
            return None;
        }
        let n_layers = m.layers.len();
        let trace_buf = if trace.is_some() {
            let f: extern "C" fn(Id, Sel, u64, u64) -> Id =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            let b = f(base.device, sel("newBufferWithLength:options:"), (n_layers * d * 4).max(16) as u64, 0);
            if b.is_null() {
                msg_void(fnw, sel("release"));
                msg_void(pool, sel("drain"));
                return None;
            }
            b
        } else {
            std::ptr::null_mut()
        };

        let set_pipe: extern "C" fn(Id, Sel, Id) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_buf: extern "C" fn(Id, Sel, Id, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let set_bytes: extern "C" fn(Id, Sel, *const c_void, u64, u64) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_tg: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let disp_th: extern "C" fn(Id, Sel, MTLSize, MTLSize) =
            std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
        let sel_pipe = sel("setComputePipelineState:");
        let sel_sb = sel("setBuffer:offset:atIndex:");
        let sel_by = sel("setBytes:length:atIndex:");
        let sel_dtg = sel("dispatchThreadgroups:threadsPerThreadgroup:");
        let sel_dth = sel("dispatchThreads:threadsPerThreadgroup:");
        let sel_end = sel("endEncoding");
        let sel_ce = sel("computeCommandEncoder");
        let one = MTLSize { width: 1, height: 1, depth: 1 };
        let tg256 = MTLSize { width: 256, height: 1, depth: 1 };

        let cmdbuf = msg_id(base.queue, sel("commandBuffer"));
        // ONE compute encoder for the whole token (serial dispatch type:
        // dispatches execute in order with their writes visible to the
        // next); trace mode ends it around each blit
        let enc: std::cell::Cell<Id> = std::cell::Cell::new(msg_id(cmdbuf, sel_ce));
        // helper: add (optional) + norm from `hidden` into `normed`
        let norm_enc = |add: Option<Id>, w_off: usize| {
            let e = enc.get();
            set_pipe(e, sel_pipe, dc.add_norm);
            set_buf(e, sel_sb, dec.hidden, 0, 0);
            set_buf(e, sel_sb, add.unwrap_or(dec.normed), 0, 1);
            set_buf(e, sel_sb, dec.small, (w_off * 4) as u64, 2);
            set_buf(e, sel_sb, dec.normed, 0, 3);
            let dims: [u32; 2] = [d as u32, if add.is_some() { 1 } else { 0 }];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
            set_bytes(e, sel_by, (&m.eps) as *const f32 as *const c_void, 4, 5);
            disp_tg(e, sel_dtg, one, tg256);
        };
        let add_enc = |add: Id| {
            let e = enc.get();
            set_pipe(e, sel_pipe, dc.add);
            set_buf(e, sel_sb, dec.hidden, 0, 0);
            set_buf(e, sel_sb, add, 0, 1);
            let n = d as u32;
            set_bytes(e, sel_by, (&n) as *const u32 as *const c_void, 4, 2);
            disp_th(e, sel_dth, MTLSize { width: d as u64, height: 1, depth: 1 }, tg256);
        };
        // helper: matvec y[rows] = W[rows, cols] . x through the decode
        // matvec of the weight's storage (one simdgroup per row; replaces
        // MPS GEMM at t=1 for bandwidth)
        let mv_enc = |w: &W, rows: usize, cols: usize, x: Id, x_off: usize, y: Id, y_off: usize| {
            let e = enc.get();
            let dims: [u32; 2] = [rows as u32, cols as u32];
            match *w {
                W::F16(b) => {
                    set_pipe(e, sel_pipe, dc.matvec);
                    set_buf(e, sel_sb, b, 0, 0);
                    set_buf(e, sel_sb, x, x_off as u64, 1);
                    set_buf(e, sel_sb, y, y_off as u64, 2);
                    set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 3);
                }
                W::Q8(q, sc) => {
                    set_pipe(e, sel_pipe, dc.matvec_q8);
                    set_buf(e, sel_sb, q, 0, 0);
                    set_buf(e, sel_sb, sc, 0, 1);
                    set_buf(e, sel_sb, x, x_off as u64, 2);
                    set_buf(e, sel_sb, y, y_off as u64, 3);
                    set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
                }
                W::Fp4(pk, sc) => {
                    set_pipe(e, sel_pipe, dc.matvec_fp4);
                    set_buf(e, sel_sb, pk, 0, 0);
                    set_buf(e, sel_sb, sc, 0, 1);
                    set_buf(e, sel_sb, x, x_off as u64, 2);
                    set_buf(e, sel_sb, y, y_off as u64, 3);
                    set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
                }
            }
            disp_th(e, sel_dth, MTLSize { width: (rows * 32) as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
        };
        // helper: MLP from `normed` -> tmp_a[..d] (f16): gate|up in tmp_a, silu, down into tmp_b[..d]
        let mlp_enc = |g: &W, u: &W, dn: &W, inter: usize| -> Id {
            mv_enc(g, inter, d, dec.normed, 0, dec.tmp_a, 0);
            mv_enc(u, inter, d, dec.normed, 0, dec.tmp_a, inter * 2);
            let e = enc.get();
            set_pipe(e, sel_pipe, sm.pipeline);
            set_buf(e, sel_sb, dec.tmp_a, 0, 0);
            set_buf(e, sel_sb, dec.tmp_a, (inter * 2) as u64, 1);
            let n = inter as u32;
            set_bytes(e, sel_by, (&n) as *const u32 as *const c_void, 4, 2);
            disp_th(e, sel_dth, MTLSize { width: inter as u64, height: 1, depth: 1 }, tg256);
            mv_enc(dn, d, inter, dec.tmp_a, 0, dec.tmp_b, 0);
            dec.tmp_b
        };

        let mut prev_add: Option<Id> = None; // f16 output of the previous layer's MLP, folded by the next norm
        let stop_at = |li: usize, stage: u8| stop == Some(TraceStop { layer: li, stage });
        'layers: for (li, l) in m.layers.iter().enumerate() {
            let o = dec.small_off[li];
            match (l, &lw[li], &dec.layers[li]) {
                (DecodeLayerRefs::Linear { dm, .. }, LW::L(w), GpuLayerState::Linear { conv_state, scan_state }) => {
                    let kt = dm.kv_heads * dm.kd;
                    let vt = dm.heads * dm.vd;
                    let cd = 2 * kt + vt;
                    let heads = dm.heads;
                    // input norm (folds the previous MLP output)
                    norm_enc(prev_add, o[0]);
                    // projections: qkv -> tmp_a[..cd], z -> mix_tm (as z buffer), b|a -> ba
                    mv_enc(&w.qkv, cd, d, dec.normed, 0, dec.tmp_a, 0);
                    mv_enc(&w.z, vt, d, dec.normed, 0, dec.mix_tm, 0);
                    mv_enc(&w.b, heads, d, dec.normed, 0, dec.ba, 0);
                    mv_enc(&w.a, heads, d, dec.normed, 0, dec.ba, heads * 2);
                    // conv + silu: tmp_a[..cd] -> tmp_b[..cd]
                    {
                        let e = enc.get();
                        set_pipe(e, sel_pipe, dc.conv);
                        set_buf(e, sel_sb, dec.tmp_a, 0, 0);
                        set_buf(e, sel_sb, dec.small, (o[4] * 4) as u64, 1);
                        set_buf(e, sel_sb, *conv_state, 0, 2);
                        set_buf(e, sel_sb, dec.tmp_b, 0, 3);
                        let dims: [u32; 2] = [cd as u32, dm.conv_k as u32];
                        set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
                        disp_th(e, sel_dth, MTLSize { width: cd as u64, height: 1, depth: 1 }, tg256);
                                }
                    // scan prep (t=1): conved tmp_b -> scan_in (q|k|v|beta|decay)
                    let sq = 0usize;
                    let sk = heads * dm.kd;
                    let sv = 2 * heads * dm.kd;
                    let sb_ = sv + heads * dm.vd;
                    let sd_ = sb_ + heads;
                    {
                        let e = enc.get();
                        set_pipe(e, sel_pipe, tis.scanprep);
                        set_buf(e, sel_sb, dec.tmp_b, 0, 0);
                        set_buf(e, sel_sb, dec.ba, 0, 1);
                        set_buf(e, sel_sb, dec.ba, (heads * 2) as u64, 2);
                        set_buf(e, sel_sb, dec.small, (o[2] * 4) as u64, 3);
                        set_buf(e, sel_sb, dec.small, (o[3] * 4) as u64, 4);
                        set_buf(e, sel_sb, dec.scan_in, (sq * 4) as u64, 5);
                        set_buf(e, sel_sb, dec.scan_in, (sk * 4) as u64, 6);
                        set_buf(e, sel_sb, dec.scan_in, (sv * 4) as u64, 7);
                        set_buf(e, sel_sb, dec.scan_in, (sb_ * 4) as u64, 8);
                        set_buf(e, sel_sb, dec.scan_in, (sd_ * 4) as u64, 9);
                        let d0: [u32; 4] = [1, heads as u32, dm.kd as u32, dm.vd as u32];
                        let d1: [u32; 4] = [(heads / dm.kv_heads.max(1)) as u32, kt as u32, cd as u32, 0];
                        set_bytes(e, sel_by, d0.as_ptr() as *const c_void, 16, 10);
                        set_bytes(e, sel_by, d1.as_ptr() as *const c_void, 16, 11);
                        disp_tg(e, sel_dtg, MTLSize { width: heads as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
                                }
                    // scan (t=1) -> mix_hm
                    {
                        let e = enc.get();
                        set_pipe(e, sel_pipe, sc.pipeline);
                        set_buf(e, sel_sb, dec.scan_in, (sq * 4) as u64, 0);
                        set_buf(e, sel_sb, dec.scan_in, (sk * 4) as u64, 1);
                        set_buf(e, sel_sb, dec.scan_in, (sv * 4) as u64, 2);
                        set_buf(e, sel_sb, dec.scan_in, (sb_ * 4) as u64, 3);
                        set_buf(e, sel_sb, dec.scan_in, (sd_ * 4) as u64, 4);
                        set_buf(e, sel_sb, *scan_state, 0, 5);
                        set_buf(e, sel_sb, dec.mix_hm, 0, 6);
                        let dims: [u32; 4] = [1, heads as u32, dm.kd as u32, dm.vd as u32];
                        set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 16, 7);
                        disp_tg(e, sel_dtg, MTLSize { width: scan_groups(heads, dm.vd) as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
                                }
                    // gated norm: mix_hm (f32) x z (mix_tm holds z) -> tmp_a[..vt] f16
                    {
                        let e = enc.get();
                        set_pipe(e, sel_pipe, tis.gnorm);
                        set_buf(e, sel_sb, dec.mix_hm, 0, 0);
                        set_buf(e, sel_sb, dec.mix_tm, 0, 1);
                        set_buf(e, sel_sb, dec.small, (o[5] * 4) as u64, 2);
                        set_buf(e, sel_sb, dec.tmp_a, 0, 3);
                        let dims: [u32; 3] = [1, heads as u32, dm.vd as u32];
                        set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 12, 4);
                        set_bytes(e, sel_by, (&dm.eps) as *const f32 as *const c_void, 4, 5);
                        disp_tg(e, sel_dtg, MTLSize { width: heads as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
                                }
                    // out_proj: tmp_a[..vt] -> tmp_b[..d]; residual add; post-norm; MLP
                    mv_enc(&w.out, d, vt, dec.tmp_a, 0, dec.tmp_b, 0);
                    norm_enc(Some(dec.tmp_b), o[1]);
                    prev_add = Some(mlp_enc(&w.g, &w.u, &w.dn, dm.inter));
                }
                (DecodeLayerRefs::Full { n_heads, n_kv, hd, rope_dim, theta, inter, .. }, LW::F(w), GpuLayerState::Full { kc, vc, cap }) => {
                    if pos >= *cap {
                        msg_void(pool, sel("drain"));
                        return None;
                    }
                    let qw = n_heads * hd;
                    let kvw = n_kv * hd;
                    norm_enc(prev_add, o[0]);
                    // qg -> tmp_a[..2qw], k -> tmp_b[..kvw], v -> tmp_b[kvw..2kvw]
                    mv_enc(&w.q, qw * 2, d, dec.normed, 0, dec.tmp_a, 0);
                    mv_enc(&w.k, kvw, d, dec.normed, 0, dec.tmp_b, 0);
                    mv_enc(&w.v, kvw, d, dec.normed, 0, dec.tmp_b, kvw * 2);
                    if stop_at(li, 0) {
                        stopped = true;
                        break 'layers;
                    }
                    // qk prep + rope + cache append (row `pos`): q -> mix_tm[..qw], gate -> mix_tm[qw..2qw]
                    {
                        let e = enc.get();
                        set_pipe(e, sel_pipe, dc.qk_prep);
                        set_buf(e, sel_sb, dec.tmp_a, 0, 0);
                        set_buf(e, sel_sb, dec.tmp_b, 0, 1);
                        set_buf(e, sel_sb, dec.tmp_b, (kvw * 2) as u64, 2);
                        set_buf(e, sel_sb, dec.small, (o[2] * 4) as u64, 3);
                        set_buf(e, sel_sb, dec.small, (o[3] * 4) as u64, 4);
                        set_buf(e, sel_sb, dec.mix_tm, 0, 5);
                        set_buf(e, sel_sb, dec.mix_tm, (qw * 2) as u64, 6);
                        set_buf(e, sel_sb, *kc, (pos * kvw * 2) as u64, 7);
                        set_buf(e, sel_sb, *vc, (pos * kvw * 2) as u64, 8);
                        let dims: [u32; 4] = [*n_heads as u32, *n_kv as u32, *hd as u32, *rope_dim as u32];
                        set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 16, 9);
                        let p32 = pos as u32;
                        set_bytes(e, sel_by, (&p32) as *const u32 as *const c_void, 4, 10);
                        set_bytes(e, sel_by, theta as *const f32 as *const c_void, 4, 11);
                        set_bytes(e, sel_by, (&m.eps) as *const f32 as *const c_void, 4, 12);
                        disp_tg(e, sel_dtg, MTLSize { width: (n_heads + n_kv) as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
                                }
                    if stop_at(li, 1) {
                        stopped = true;
                        break 'layers;
                    }
                    // attention over len = pos+1 -> tmp_a[..qw]
                    {
                        let e = enc.get();
                        set_pipe(e, sel_pipe, dc.attn);
                        set_buf(e, sel_sb, dec.mix_tm, 0, 0);
                        set_buf(e, sel_sb, *kc, 0, 1);
                        set_buf(e, sel_sb, *vc, 0, 2);
                        set_buf(e, sel_sb, dec.mix_tm, (qw * 2) as u64, 3);
                        set_buf(e, sel_sb, dec.tmp_a, 0, 4);
                        let dims: [u32; 4] = [(pos + 1) as u32, *hd as u32, kvw as u32, (n_heads / n_kv.max(&1)) as u32];
                        set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 16, 5);
                        let scale = 1.0f32 / (*hd as f32).sqrt();
                        set_bytes(e, sel_by, (&scale) as *const f32 as *const c_void, 4, 6);
                        disp_tg(e, sel_dtg, MTLSize { width: *n_heads as u64, height: 1, depth: 1 }, MTLSize { width: 128, height: 1, depth: 1 });
                                }
                    if stop_at(li, 2) {
                        stopped = true;
                        break 'layers;
                    }
                    // o_proj -> tmp_b[..d]; residual; post-norm; MLP
                    mv_enc(&w.o, d, qw, dec.tmp_a, 0, dec.tmp_b, 0);
                    if stop_at(li, 3) {
                        stopped = true;
                        break 'layers;
                    }
                    norm_enc(Some(dec.tmp_b), o[1]);
                    prev_add = Some(mlp_enc(&w.g, &w.u, &w.dn, *inter));
                }
                _ => {
                    msg_void(pool, sel("drain"));
                    return None;
                }
            }
            if !trace_buf.is_null() {
                if let Some(a) = prev_add.take() {
                    add_enc(a);
                }
                msg_void(enc.get(), sel_end);
                let blit = msg_id(cmdbuf, sel("blitCommandEncoder"));
                let f: extern "C" fn(Id, Sel, Id, u64, Id, u64, u64) =
                    std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
                f(
                    blit,
                    sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
                    dec.hidden,
                    0,
                    trace_buf,
                    (li * d * 4) as u64,
                    (d * 4) as u64,
                );
                msg_void(blit, sel_end);
                enc.set(msg_id(cmdbuf, sel_ce));
            }
        }
        // final: fold the last MLP, final norm, lm_head
        if let Some(a) = prev_add {
            if !stopped {
                add_enc(a);
            }
        }
        if !stopped {
            let e = enc.get();
            set_pipe(e, sel_pipe, dc.add_norm);
            set_buf(e, sel_sb, dec.hidden, 0, 0);
            set_buf(e, sel_sb, dec.normed, 0, 1);
            set_buf(e, sel_sb, fnw, 0, 2);
            set_buf(e, sel_sb, dec.normed, 0, 3);
            let dims: [u32; 2] = [d as u32, 0];
            set_bytes(e, sel_by, dims.as_ptr() as *const c_void, 8, 4);
            set_bytes(e, sel_by, (&m.eps) as *const f32 as *const c_void, 4, 5);
            disp_tg(e, sel_dtg, one, tg256);
            mv_enc(&head_buf, vocab, d, dec.normed, 0, dec.logits, 0);
        }
        msg_void(enc.get(), sel_end);
        let t_commit = std::time::Instant::now();
        msg_void(cmdbuf, sel("commit"));
        msg_void(cmdbuf, sel("waitUntilCompleted"));
        if decode_timing_on() {
            let getf: extern "C" fn(Id, Sel) -> f64 =
                std::mem::transmute::<unsafe extern "C" fn(Id, Sel, ...) -> Id, _>(objc_msgSend);
            let g0 = getf(cmdbuf, sel("GPUStartTime"));
            let g1 = getf(cmdbuf, sel("GPUEndTime"));
            let k0 = getf(cmdbuf, sel("kernelStartTime"));
            let k1 = getf(cmdbuf, sel("kernelEndTime"));
            DECODE_TIMING[0].fetch_add((t_commit - t_start).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            DECODE_TIMING[1].fetch_add(((g1 - g0) * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
            DECODE_TIMING[2].fetch_add(((k1 - k0) * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
            DECODE_TIMING[3].fetch_add(t_commit.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            DECODE_TIMING[4].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        f16s_to_f32s(std::slice::from_raw_parts(contents(dec.logits) as *const u16, vocab), &mut logits_out[..vocab]);
        if let Some(tr) = trace.as_deref_mut() {
            let tp = contents(trace_buf) as *const f32;
            tr.clear();
            for li in 0..n_layers {
                tr.push(std::slice::from_raw_parts(tp.add(li * d), d).to_vec());
            }
            msg_void(trace_buf, sel("release"));
        }
        msg_void(fnw, sel("release"));
        msg_void(pool, sel("drain"));
    }
    if stopped {
        return Some(());
    }
    dec.pos += 1;
    gemm_account(t_start.elapsed().as_micros() as u64);
    Some(())
}
