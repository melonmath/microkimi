//! CUDA backend for the Qwen runtime (Linux, NVIDIA GPUs).
//!
//! Zero build-time dependencies, zero crates: `libcuda.so.1` (the driver
//! API) and `libnvrtc.so.12` (the runtime compiler) are opened with
//! `dlopen` when a GPU is first asked for, and the kernels below - plain
//! CUDA C in this file - are compiled for the device's own architecture
//! at that moment. Nothing here runs when the libraries are absent: every
//! entry point returns None and the CPU paths take over. The exact
//! counterpart of the Metal backend (`metal.rs`), whose MSL sources are
//! compiled by the OS at run time the same way.
//!
//! Layout on the device: activations f32; the attention spine as q8_0
//! rows (int8 + one f32 scale per block of 32, quantized on the host from
//! the f32 tensors, the same numbers the CPU's q8 spine holds); the MLP
//! as the MXFP4 bytes of the file, untouched. Every dot is an exact int8
//! dot per block of 32 (`dp4a`) times the two block scales, in float -
//! the CPU's q8 arithmetic, summed in a different order.

#![cfg(target_os = "linux")]
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CStr, CString};

// ── dlopen ──

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}
const RTLD_NOW: c_int = 2;

fn open_lib(names: &[&str]) -> Option<*mut c_void> {
    for n in names {
        let c = CString::new(*n).ok()?;
        // SAFETY: valid C string; dlopen returns null on failure.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
        if !h.is_null() {
            return Some(h);
        }
    }
    None
}

fn sym(h: *mut c_void, name: &str) -> Option<*mut c_void> {
    let c = CString::new(name).ok()?;
    // SAFETY: valid handle and C string.
    let p = unsafe { dlsym(h, c.as_ptr()) };
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}

// ── driver API surface ──

type CUresult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
pub type CUdeviceptr = u64;

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

#[allow(dead_code)]
struct Driver {
    init: unsafe extern "C" fn(u32) -> CUresult,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CUresult,
    device_get: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    device_total_mem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    ctx_set_current: unsafe extern "C" fn(CUcontext) -> CUresult,
    ctx_synchronize: unsafe extern "C" fn() -> CUresult,
    module_load_data: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    module_get_function: unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,
    func_set_attribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    memcpy_htod: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    memcpy_htod_async: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    memcpy_dtod_async: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    memset_d8_async: unsafe extern "C" fn(CUdeviceptr, u8, usize, CUstream) -> CUresult,
    launch_kernel: unsafe extern "C" fn(
        CUfunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
    stream_create: unsafe extern "C" fn(*mut CUstream, u32) -> CUresult,
    stream_synchronize: unsafe extern "C" fn(CUstream) -> CUresult,
    event_create: unsafe extern "C" fn(*mut CUevent, u32) -> CUresult,
    event_record: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    event_synchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    event_elapsed: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    get_error_string: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
    mem_host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CUresult,
    mem_free_host: unsafe extern "C" fn(*mut c_void) -> CUresult,
    stream_begin_capture: unsafe extern "C" fn(CUstream, c_int) -> CUresult,
    stream_end_capture: unsafe extern "C" fn(CUstream, *mut CUgraph) -> CUresult,
    graph_instantiate: unsafe extern "C" fn(*mut CUgraphExec, CUgraph, u64) -> CUresult,
    graph_launch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    graph_exec_destroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
    graph_destroy: unsafe extern "C" fn(CUgraph) -> CUresult,
}
type CUgraph = *mut c_void;
type CUgraphExec = *mut c_void;

struct Nvrtc {
    create: unsafe extern "C" fn(*mut *mut c_void, *const c_char, *const c_char, c_int, *const *const c_char, *const *const c_char) -> c_int,
    compile: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int,
    ptx_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    ptx: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    cubin_size: Option<unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int>,
    cubin: Option<unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int>,
    log_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    log: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
}

macro_rules! load {
    ($h:expr, $name:literal) => {{
        // SAFETY: the symbol's C signature is the one declared at the use
        // site (checked against the CUDA 12 headers).
        unsafe { std::mem::transmute(sym($h, $name)?) }
    }};
}

fn load_driver() -> Option<Driver> {
    let h = open_lib(&["libcuda.so.1", "libcuda.so"])?;
    Some(Driver {
        init: load!(h, "cuInit"),
        device_get_count: load!(h, "cuDeviceGetCount"),
        device_get: load!(h, "cuDeviceGet"),
        device_get_name: load!(h, "cuDeviceGetName"),
        device_get_attribute: load!(h, "cuDeviceGetAttribute"),
        device_total_mem: load!(h, "cuDeviceTotalMem_v2"),
        primary_ctx_retain: load!(h, "cuDevicePrimaryCtxRetain"),
        ctx_set_current: load!(h, "cuCtxSetCurrent"),
        ctx_synchronize: load!(h, "cuCtxSynchronize"),
        module_load_data: load!(h, "cuModuleLoadData"),
        module_get_function: load!(h, "cuModuleGetFunction"),
        func_set_attribute: load!(h, "cuFuncSetAttribute"),
        mem_alloc: load!(h, "cuMemAlloc_v2"),
        mem_free: load!(h, "cuMemFree_v2"),
        mem_get_info: load!(h, "cuMemGetInfo_v2"),
        memcpy_htod: load!(h, "cuMemcpyHtoD_v2"),
        memcpy_dtoh: load!(h, "cuMemcpyDtoH_v2"),
        memcpy_htod_async: load!(h, "cuMemcpyHtoDAsync_v2"),
        memcpy_dtoh_async: load!(h, "cuMemcpyDtoHAsync_v2"),
        memcpy_dtod_async: load!(h, "cuMemcpyDtoDAsync_v2"),
        memset_d8_async: load!(h, "cuMemsetD8Async"),
        launch_kernel: load!(h, "cuLaunchKernel"),
        stream_create: load!(h, "cuStreamCreate"),
        stream_synchronize: load!(h, "cuStreamSynchronize"),
        event_create: load!(h, "cuEventCreate"),
        event_record: load!(h, "cuEventRecord"),
        event_synchronize: load!(h, "cuEventSynchronize"),
        event_elapsed: load!(h, "cuEventElapsedTime"),
        get_error_string: load!(h, "cuGetErrorString"),
        mem_host_alloc: load!(h, "cuMemHostAlloc"),
        mem_free_host: load!(h, "cuMemFreeHost"),
        stream_begin_capture: load!(h, "cuStreamBeginCapture_v2"),
        stream_end_capture: load!(h, "cuStreamEndCapture"),
        graph_instantiate: load!(h, "cuGraphInstantiateWithFlags"),
        graph_launch: load!(h, "cuGraphLaunch"),
        graph_exec_destroy: load!(h, "cuGraphExecDestroy"),
        graph_destroy: load!(h, "cuGraphDestroy"),
    })
}

fn load_nvrtc() -> Option<Nvrtc> {
    let mut names: Vec<String> = Vec::new();
    if let Ok(p) = std::env::var("MICROKIMI_NVRTC") {
        names.push(p);
    }
    for n in ["libnvrtc.so.12", "libnvrtc.so", "/usr/local/cuda/lib64/libnvrtc.so.12", "/usr/local/cuda/lib64/libnvrtc.so"] {
        names.push(n.to_string());
    }
    let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
    let h = open_lib(&refs)?;
    Some(Nvrtc {
        create: load!(h, "nvrtcCreateProgram"),
        compile: load!(h, "nvrtcCompileProgram"),
        ptx_size: load!(h, "nvrtcGetPTXSize"),
        ptx: load!(h, "nvrtcGetPTX"),
        // SAFETY: as in load!.
        cubin_size: sym(h, "nvrtcGetCUBINSize").map(|p| unsafe { std::mem::transmute(p) }),
        cubin: sym(h, "nvrtcGetCUBIN").map(|p| unsafe { std::mem::transmute(p) }),
        log_size: load!(h, "nvrtcGetProgramLogSize"),
        log: load!(h, "nvrtcGetProgramLog"),
        destroy: load!(h, "nvrtcDestroyProgram"),
        version: load!(h, "nvrtcVersion"),
    })
}

// ── context ──

pub struct CudaCtx {
    drv: Driver,
    #[allow(dead_code)]
    ctx: CUcontext,
    #[allow(dead_code)]
    module: CUmodule,
    stream: CUstream,
    funcs: std::collections::HashMap<&'static str, CUfunction>,
    pub name: String,
    pub sm_count: usize,
    pub cc: (i32, i32),
    #[allow(dead_code)]
    pub total_mem: usize,
}
unsafe impl Send for CudaCtx {}
unsafe impl Sync for CudaCtx {}

static CTX: std::sync::OnceLock<Option<CudaCtx>> = std::sync::OnceLock::new();

/// The process-wide CUDA context (device 0, or MICROKIMI_CUDA_DEVICE),
/// with the kernels compiled; None when no usable GPU, driver or NVRTC is
/// present. The reason is printed once.
pub fn ctx() -> Option<&'static CudaCtx> {
    CTX.get_or_init(|| match init_ctx() {
        Ok(c) => Some(c),
        Err(e) => {
            if std::env::var("MICROKIMI_QWEN_CUDA").map(|v| v == "1").unwrap_or(false) || std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
                println!("cuda: unavailable - {e}");
            }
            None
        }
    })
    .as_ref()
}

/// True when the CUDA offload is requested (MICROKIMI_QWEN_CUDA=1).
pub fn qwen_cuda_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_QWEN_CUDA").map(|v| v == "1").unwrap_or(false))
}

fn err_str(drv: &Driver, r: CUresult) -> String {
    let mut p: *const c_char = std::ptr::null();
    // SAFETY: valid out pointer; the driver returns a static string.
    unsafe {
        (drv.get_error_string)(r, &mut p);
        if p.is_null() {
            format!("CUDA error {r}")
        } else {
            format!("CUDA error {r}: {}", CStr::from_ptr(p).to_string_lossy())
        }
    }
}

macro_rules! cu {
    ($drv:expr, $call:expr) => {{
        let r = $call;
        if r != 0 {
            return Err(err_str($drv, r));
        }
    }};
}

const KERNEL_NAMES: &[&str] = &[
    "k_quantize_q8",
    "k_matvec_q8",
    "k_matvec_fp4",
    "k_gemm_q8",
    "k_gemm_fp4",
    "k_gemm_q8_mma64",
    "k_gemm_q8_mma128",
    "k_gemm_q8_mma256",
    "k_gemm_q8_mma128x128",
    "k_gemm_fp4_mma64",
    "k_gemm_fp4_mma128",
    "k_gemm_fp4_mma256",
    "k_gemm_fp4_mma128x128",
    "k_gemm_q8_pipe",
    "k_gemm_fp4_pipe",
    "k_gemm_fp4_pipe_h",
    "k_add",
    "k_rmsnorm",
    "k_rmsnorm_rows",
    "k_silu_mul",
    "k_conv_silu",
    "k_delta_step",
    "k_lin_head_step128",
    "k_lin_head_step64",
    "k_qk_prep",
    "k_attn_decode",
    "k_attn_prefill",
    "k_attn_prefill_grouped",
    "k_gated_norm",
    "k_scale_rows",
    "k_lin_prep",
    "k_conv_prefill",
    "k_delta_scan",
    "k_delta_scan128",
    "k_delta_scan4x128",
    "k_delta_scan4x64",
    "k_delta_scan64",
    "k_gated_norm_rows",
    "k_gated_norm_quant_rows",
    "k_silu_mul_rows",
    "k_add_rmsnorm_rows",
    "k_add_rmsnorm_quant_rows",
    "k_silu_mul_quant_rows",
    "k_silu_mul_quant_rows_h",
    "k_qk_prep_rows",
];

fn init_ctx() -> Result<CudaCtx, String> {
    let drv = load_driver().ok_or("libcuda.so.1 not found (no NVIDIA driver)")?;
    let nv = load_nvrtc().ok_or("libnvrtc.so.12 not found (CUDA toolkit; MICROKIMI_NVRTC=/path/to/libnvrtc.so)")?;
    // SAFETY: driver API calls with valid out-pointers, in the documented order.
    unsafe {
        cu!(&drv, (drv.init)(0));
        let mut count: c_int = 0;
        cu!(&drv, (drv.device_get_count)(&mut count));
        if count <= 0 {
            return Err("no CUDA device".into());
        }
        let want: c_int = std::env::var("MICROKIMI_CUDA_DEVICE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let mut dev: CUdevice = 0;
        cu!(&drv, (drv.device_get)(&mut dev, want.min(count - 1)));
        let mut name = [0 as c_char; 128];
        cu!(&drv, (drv.device_get_name)(name.as_mut_ptr(), 128, dev));
        let name = CStr::from_ptr(name.as_ptr()).to_string_lossy().into_owned();
        let (mut major, mut minor, mut sms) = (0 as c_int, 0 as c_int, 0 as c_int);
        cu!(&drv, (drv.device_get_attribute)(&mut major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
        cu!(&drv, (drv.device_get_attribute)(&mut minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
        cu!(&drv, (drv.device_get_attribute)(&mut sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
        let mut total: usize = 0;
        cu!(&drv, (drv.device_total_mem)(&mut total, dev));
        let mut ctx: CUcontext = std::ptr::null_mut();
        cu!(&drv, (drv.primary_ctx_retain)(&mut ctx, dev));
        cu!(&drv, (drv.ctx_set_current)(ctx));
        // compile the kernels for this device
        let t0 = std::time::Instant::now();
        let image = compile(&nv, major, minor)?;
        let mut module: CUmodule = std::ptr::null_mut();
        cu!(&drv, (drv.module_load_data)(&mut module, image.as_ptr() as *const c_void));
        let mut funcs = std::collections::HashMap::new();
        for &n in KERNEL_NAMES {
            let cn = CString::new(n).unwrap();
            let mut f: CUfunction = std::ptr::null_mut();
            let r = (drv.module_get_function)(&mut f, module, cn.as_ptr());
            if r != 0 {
                return Err(format!("kernel {n} missing: {}", err_str(&drv, r)));
            }
            funcs.insert(n, f);
        }
        let mut stream: CUstream = std::ptr::null_mut();
        cu!(&drv, (drv.stream_create)(&mut stream, 0));
        if std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
            println!("cuda: {name} (sm_{major}{minor}, {sms} SMs, {:.1} GB), kernels compiled in {:.2} s", total as f64 / 1e9, t0.elapsed().as_secs_f64());
        }
        Ok(CudaCtx { drv, ctx, module, stream, funcs, name, sm_count: sms as usize, cc: (major, minor), total_mem: total })
    }
}

/// NVRTC-compiles the kernel source for the device (SASS when the
/// toolkit knows the architecture, PTX otherwise), returning the image
/// `cuModuleLoadData` accepts. Cached on disk under
/// $XDG_CACHE_HOME/microkimi (or ~/.cache/microkimi) by source hash.
fn compile(nv: &Nvrtc, major: i32, minor: i32) -> Result<Vec<u8>, String> {
    let (mut nmaj, mut nmin) = (0 as c_int, 0 as c_int);
    // SAFETY: valid out pointers.
    unsafe {
        (nv.version)(&mut nmaj, &mut nmin);
    }
    let src = KERNEL_SRC;
    let key = {
        let mut h = crate::sha256::Sha256::new();
        h.update(src.as_bytes());
        h.update(format!("sm{major}{minor}-nvrtc{nmaj}.{nmin}-v48-{}", std::env::var("MICROKIMI_CUDA_PTXAS_V").is_ok()).as_bytes());
        let d = h.finalize();
        d.iter().take(12).map(|b| format!("{b:02x}")).collect::<String>()
    };
    let cache_dir = std::env::var("XDG_CACHE_HOME")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOME").ok().map(|h| format!("{h}/.cache")))
        .map(|c| format!("{c}/microkimi"));
    let cache_path = cache_dir.as_ref().map(|d| format!("{d}/cuda-{key}.bin"));
    if let Some(p) = &cache_path {
        if let Ok(bytes) = std::fs::read(p) {
            if !bytes.is_empty() {
                return Ok(bytes);
            }
        }
    }
    let csrc = CString::new(src).map_err(|_| "kernel source contains NUL")?;
    let cname = CString::new("microkimi.cu").unwrap();
    let use_cubin = nv.cubin.is_some() && nv.cubin_size.is_some() && !std::env::var("MICROKIMI_CUDA_PTX").is_ok();
    let arch = if use_cubin { format!("--gpu-architecture=sm_{major}{minor}") } else { format!("--gpu-architecture=compute_{major}{minor}") };
    let mut opt_strs: Vec<String> = ["-default-device", "-std=c++17", "-lineinfo", &arch].iter().map(|s| s.to_string()).collect();
    if std::env::var("MICROKIMI_CUDA_PTXAS_V").is_ok() {
        opt_strs.push("--ptxas-options=-v".to_string());
    }
    let opts: Vec<CString> = opt_strs.iter().map(|s| CString::new(s.as_str()).unwrap()).collect();
    let optp: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: NVRTC calls with valid pointers; the program is destroyed on every path.
    unsafe {
        let mut prog: *mut c_void = std::ptr::null_mut();
        let r = (nv.create)(&mut prog, csrc.as_ptr(), cname.as_ptr(), 0, std::ptr::null(), std::ptr::null());
        if r != 0 {
            return Err(format!("nvrtcCreateProgram failed ({r})"));
        }
        let r = (nv.compile)(prog, optp.len() as c_int, optp.as_ptr());
        let mut log_len: usize = 0;
        (nv.log_size)(prog, &mut log_len);
        let mut log = vec![0u8; log_len.max(1)];
        (nv.log)(prog, log.as_mut_ptr() as *mut c_char);
        let log_s = String::from_utf8_lossy(&log[..log_len.saturating_sub(1)]).into_owned();
        if r != 0 {
            (nv.destroy)(&mut prog);
            return Err(format!("kernel compilation failed:\n{log_s}"));
        }
        if std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() && log_s.trim().len() > 1 {
            println!("cuda: nvrtc log:\n{log_s}");
        }
        let mut image: Vec<u8>;
        if use_cubin {
            let mut n: usize = 0;
            (nv.cubin_size.unwrap())(prog, &mut n);
            image = vec![0u8; n];
            (nv.cubin.unwrap())(prog, image.as_mut_ptr() as *mut c_char);
        } else {
            let mut n: usize = 0;
            (nv.ptx_size)(prog, &mut n);
            image = vec![0u8; n];
            (nv.ptx)(prog, image.as_mut_ptr() as *mut c_char);
        }
        (nv.destroy)(&mut prog);
        if let (Some(dir), Some(p)) = (&cache_dir, &cache_path) {
            let _ = std::fs::create_dir_all(dir);
            let tmp = format!("{p}.tmp{}", std::process::id());
            if std::fs::write(&tmp, &image).is_ok() {
                let _ = std::fs::rename(&tmp, p);
            }
        }
        Ok(image)
    }
}

// ── device memory ──

/// A page-locked host allocation, freed on drop.
pub struct PinBuf {
    pub ptr: *mut u8,
    pub len: usize,
}
impl Drop for PinBuf {
    fn drop(&mut self) {
        if let Some(c) = ctx() {
            if !self.ptr.is_null() {
                // SAFETY: allocated by cuMemHostAlloc.
                unsafe {
                    (c.drv.mem_free_host)(self.ptr as *mut c_void);
                }
            }
        }
    }
}
impl PinBuf {
    /// The buffer as a typed slice (host memory).
    pub fn as_mut_slice<T: Copy>(&self) -> &mut [T] {
        // SAFETY: len bytes of host memory owned by this buffer.
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len / std::mem::size_of::<T>()) }
    }
}

/// A device allocation, freed on drop.
pub struct DBuf {
    pub ptr: CUdeviceptr,
    pub len: usize,
}

impl Drop for DBuf {
    fn drop(&mut self) {
        if let Some(c) = ctx() {
            if self.ptr != 0 {
                // SAFETY: allocated by cuMemAlloc in this context.
                unsafe {
                    (c.drv.mem_free)(self.ptr);
                }
            }
        }
    }
}

impl CudaCtx {
    pub fn alloc(&self, len: usize) -> Option<DBuf> {
        let mut p: CUdeviceptr = 0;
        // SAFETY: valid out pointer.
        let r = unsafe { (self.drv.mem_alloc)(&mut p, len.max(16)) };
        if r != 0 {
            if std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
                println!("cuda: alloc {len} failed: {}", err_str(&self.drv, r));
            }
            return None;
        }
        Some(DBuf { ptr: p, len })
    }
    pub fn upload<T: Copy>(&self, data: &[T]) -> Option<DBuf> {
        let bytes = std::mem::size_of_val(data);
        let b = self.alloc(bytes)?;
        if bytes > 0 {
            // SAFETY: b holds `bytes` bytes; data is readable.
            let r = unsafe { (self.drv.memcpy_htod)(b.ptr, data.as_ptr() as *const c_void, bytes) };
            if r != 0 {
                return None;
            }
        }
        Some(b)
    }
    pub fn upload_bytes(&self, data: &[u8]) -> Option<DBuf> {
        self.upload(data)
    }
    pub fn zeroed(&self, len: usize) -> Option<DBuf> {
        let b = self.alloc(len)?;
        // SAFETY: b holds len bytes.
        let r = unsafe { (self.drv.memset_d8_async)(b.ptr, 0, len.max(16), self.stream) };
        if r != 0 {
            return None;
        }
        Some(b)
    }
    pub fn write<T: Copy>(&self, b: &DBuf, off_bytes: usize, data: &[T]) -> bool {
        let bytes = std::mem::size_of_val(data);
        if off_bytes + bytes > b.len.max(16) {
            return false;
        }
        // SAFETY: bounds checked above.
        unsafe { (self.drv.memcpy_htod_async)(b.ptr + off_bytes as u64, data.as_ptr() as *const c_void, bytes, self.stream) == 0 }
    }
    pub fn read<T: Copy>(&self, b: &DBuf, off_bytes: usize, out: &mut [T]) -> bool {
        let bytes = std::mem::size_of_val(out);
        if off_bytes + bytes > b.len.max(16) {
            return false;
        }
        // SAFETY: bounds checked; synchronous copy.
        unsafe {
            (self.drv.stream_synchronize)(self.stream);
            (self.drv.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, b.ptr + off_bytes as u64, bytes) == 0
        }
    }
    pub fn copy_dtod(&self, dst: &DBuf, dst_off: usize, src: &DBuf, src_off: usize, bytes: usize) -> bool {
        if dst_off + bytes > dst.len.max(16) || src_off + bytes > src.len.max(16) {
            return false;
        }
        // SAFETY: bounds checked.
        unsafe { (self.drv.memcpy_dtod_async)(dst.ptr + dst_off as u64, src.ptr + src_off as u64, bytes, self.stream) == 0 }
    }
    pub fn sync(&self) -> bool {
        // SAFETY: valid stream.
        unsafe { (self.drv.stream_synchronize)(self.stream) == 0 }
    }
    pub fn mem_info(&self) -> (usize, usize) {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: valid out pointers.
        unsafe {
            (self.drv.mem_get_info)(&mut free, &mut total);
        }
        (free, total)
    }
    /// Launches `name` with the given grid/block, dynamic shared bytes and
    /// argument list (each entry a pointer to the argument value).
    fn launch(&self, name: &str, grid: (u32, u32, u32), block: (u32, u32, u32), shared: u32, args: &mut [*mut c_void]) -> bool {
        // MICROKIMI_CUDA_MV_ONLY=1: a timing experiment - only the matvec
        // kernels run (the answer is garbage), isolating the weight stream
        // from everything else in a token
        static MV_ONLY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *MV_ONLY.get_or_init(|| std::env::var("MICROKIMI_CUDA_MV_ONLY").map(|v| v == "1").unwrap_or(false)) && !name.starts_with("k_matvec") {
            return true;
        }
        let f = match self.funcs.get(name) {
            Some(f) => *f,
            None => return false,
        };
        if shared > 48 * 1024 {
            // SAFETY: valid function.
            unsafe {
                (self.drv.func_set_attribute)(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared as c_int);
            }
        }
        // SAFETY: args point at values that live for the duration of the call
        // (the driver copies them at launch).
        let r = unsafe {
            (self.drv.launch_kernel)(f, grid.0, grid.1, grid.2, block.0, block.1, block.2, shared, self.stream, args.as_mut_ptr(), std::ptr::null_mut())
        };
        if r != 0 && std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
            println!("cuda: launch {name} failed: {}", err_str(&self.drv, r));
        }
        r == 0
    }
    /// Page-locked host memory (async copies to and from it are truly
    /// asynchronous, and graph memcpy nodes may source it).
    pub fn pinned(&self, bytes: usize) -> Option<PinBuf> {
        let mut p: *mut c_void = std::ptr::null_mut();
        // SAFETY: valid out pointer.
        let r = unsafe { (self.drv.mem_host_alloc)(&mut p, bytes.max(16), 0) };
        if r != 0 || p.is_null() {
            return None;
        }
        Some(PinBuf { ptr: p as *mut u8, len: bytes })
    }
    /// Async host->device from pinned memory (stream-ordered).
    pub fn write_async(&self, b: &DBuf, off_bytes: usize, src: &PinBuf, bytes: usize) -> bool {
        if off_bytes + bytes > b.len.max(16) || bytes > src.len {
            return false;
        }
        // SAFETY: bounds checked; src stays allocated for the decoder's life.
        unsafe { (self.drv.memcpy_htod_async)(b.ptr + off_bytes as u64, src.ptr as *const c_void, bytes, self.stream) == 0 }
    }
    /// Async device->host into pinned memory (stream-ordered; sync before reading).
    pub fn read_async(&self, b: &DBuf, off_bytes: usize, dst: &PinBuf, bytes: usize) -> bool {
        if off_bytes + bytes > b.len.max(16) || bytes > dst.len {
            return false;
        }
        // SAFETY: bounds checked.
        unsafe { (self.drv.memcpy_dtoh_async)(dst.ptr as *mut c_void, b.ptr + off_bytes as u64, bytes, self.stream) == 0 }
    }
    /// Captures the stream work issued by `f` into an executable graph
    /// (None when capture or instantiation fails; the work is NOT run).
    pub fn capture<F: FnOnce() -> bool>(&self, f: F) -> Option<CUgraphExec> {
        // SAFETY: capture brackets on this context's stream; graph handles freed on every failure path.
        unsafe {
            if (self.drv.stream_begin_capture)(self.stream, 1 /* thread-local */) != 0 {
                return None;
            }
            let ok = f();
            let mut g: CUgraph = std::ptr::null_mut();
            let r = (self.drv.stream_end_capture)(self.stream, &mut g);
            if r != 0 || g.is_null() {
                return None;
            }
            if !ok {
                (self.drv.graph_destroy)(g);
                return None;
            }
            let mut exec: CUgraphExec = std::ptr::null_mut();
            let r = (self.drv.graph_instantiate)(&mut exec, g, 0);
            (self.drv.graph_destroy)(g);
            if r != 0 || exec.is_null() {
                if std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
                    println!("cuda: graph instantiate failed: {}", err_str(&self.drv, r));
                }
                return None;
            }
            Some(exec)
        }
    }
    pub fn graph_launch(&self, exec: CUgraphExec) -> bool {
        // SAFETY: exec instantiated on this context.
        unsafe { (self.drv.graph_launch)(exec, self.stream) == 0 }
    }
    pub fn graph_free(&self, exec: CUgraphExec) {
        // SAFETY: as above.
        unsafe {
            (self.drv.graph_exec_destroy)(exec);
        }
    }
    /// Milliseconds between two events around `f` (stream-ordered).
    pub fn timed<F: FnOnce()>(&self, f: F) -> f32 {
        let (mut e0, mut e1): (CUevent, CUevent) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: valid stream and event handles.
        unsafe {
            (self.drv.event_create)(&mut e0, 0);
            (self.drv.event_create)(&mut e1, 0);
            (self.drv.event_record)(e0, self.stream);
            f();
            (self.drv.event_record)(e1, self.stream);
            (self.drv.event_synchronize)(e1);
            let mut ms = 0f32;
            (self.drv.event_elapsed)(&mut ms, e0, e1);
            ms
        }
    }
}

macro_rules! args {
    ($($x:expr),* $(,)?) => {
        [$( (&$x) as *const _ as *mut c_void ),*]
    };
}

// ── kernel wrappers (all stream-ordered, no sync) ──

impl CudaCtx {
    /// xq[n] i8, xs[n/32] f32 <- x[n] f32 (block quantization, the CPU's
    /// rounding: half away from zero, IEEE division). `rows` vectors of
    /// `n` (row-major); one thread per block of 32.
    pub fn quantize_q8(&self, x: &DBuf, x_off: usize, xq: &DBuf, xs: &DBuf, rows: u32, n: u32) -> bool {
        let nb = rows * (n / 32);
        let xp = x.ptr + x_off as u64;
        let mut a = args!(xp, xq.ptr, xs.ptr, nb, n);
        self.launch("k_quantize_q8", (nb.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a)
    }
    /// y[rows] = W(q8 rows x cols) . x(q8). One warp per row.
    pub fn matvec_q8(&self, wq: &DBuf, ws: &DBuf, xq: &DBuf, xs: &DBuf, y: &DBuf, y_off: usize, rows: u32, cols: u32) -> bool {
        let yp = y.ptr + y_off as u64;
        let (w, rpw) = self.mv_shape(rows);
        let mut a = args!(wq.ptr, ws.ptr, xq.ptr, xs.ptr, yp, rows, cols, rpw);
        let shared = cols + cols / 32 * 4;
        self.launch("k_matvec_q8", (rows.div_ceil(w * rpw), 1, 1), (32 * w, 1, 1), shared, &mut a)
    }
    /// y[rows] = W(MXFP4 rows x cols) . x(q8). One warp per row.
    pub fn matvec_fp4(&self, wp: &DBuf, wsc: &DBuf, xq: &DBuf, xs: &DBuf, y: &DBuf, y_off: usize, rows: u32, cols: u32) -> bool {
        let yp = y.ptr + y_off as u64;
        let (w, rpw) = self.mv_shape(rows);
        let mut a = args!(wp.ptr, wsc.ptr, xq.ptr, xs.ptr, yp, rows, cols, rpw);
        let shared = cols + cols / 32 * 4;
        self.launch("k_matvec_fp4", (rows.div_ceil(w * rpw), 1, 1), (32 * w, 1, 1), shared, &mut a)
    }
    /// (warps per block, rows per warp) of a matvec: MICROKIMI_CUDA_MV_WARPS
    /// and MICROKIMI_CUDA_MV_RPW override; otherwise 8 warps and as many
    /// rows per warp as keep four blocks per SM.
    fn mv_shape(&self, rows: u32) -> (u32, u32) {
        static RPW: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        let w = self.mv_warps(rows, 0);
        if let Some(r) = *RPW.get_or_init(|| std::env::var("MICROKIMI_CUDA_MV_RPW").ok().and_then(|v| v.parse().ok())) {
            return (w, r.max(1));
        }
        let _ = rows;
        (w, 1)
    }
    /// Warps (rows) per matvec block: MICROKIMI_CUDA_MV_WARPS overrides;
    /// otherwise 8, or 4 when the grid would be under four blocks per SM.
    fn mv_warps(&self, rows: u32, _cols: u32) -> u32 {
        static FORCE: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
        if let Some(w) = *FORCE.get_or_init(|| std::env::var("MICROKIMI_CUDA_MV_WARPS").ok().and_then(|v| v.parse().ok())) {
            return w.clamp(1, 32);
        }
        if rows.div_ceil(8) < 4 * self.sm_count as u32 {
            4
        } else {
            8
        }
    }
    /// C[t][rows] (row stride `ldc`) = X(q8, t rows of cols) . W(q8 rows x cols)^T.
    pub fn gemm_q8(&self, wq: &DBuf, ws: &DBuf, xq: &DBuf, xs: &DBuf, c: &DBuf, c_off: usize, rows: u32, cols: u32, t: u32, ldc: u32) -> bool {
        if self.mma_on() {
            return self.gemm_mma_split(true, wq, ws, xq, xs, c, c_off, rows, cols, t, ldc);
        }
        let cp = c.ptr + c_off as u64;
        let mut a = args!(wq.ptr, ws.ptr, xq.ptr, xs.ptr, cp, rows, cols, t, ldc);
        self.launch("k_gemm_q8", (rows.div_ceil(64), t.div_ceil(64), 1), (256, 1, 1), 0, &mut a)
    }
    /// The gate|up GEMM with an f16 output when the pipelined kernel runs
    /// (returns true in .1 when the output is f16), the f32 form otherwise.
    pub fn gemm_fp4_gu(&self, wp: &DBuf, wsc: &DBuf, xq: &DBuf, xs: &DBuf, c: &DBuf, rows: u32, cols: u32, t: u32) -> (bool, bool) {
        static PIPE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let pipe = *PIPE.get_or_init(|| !std::env::var("MICROKIMI_CUDA_NO_PIPE").map(|v| v == "1").unwrap_or(false));
        if self.mma_on() && pipe && t >= 64 {
            let mut a = args!(wp.ptr, wsc.ptr, xq.ptr, xs.ptr, c.ptr, rows, cols, t, rows);
            let stage = 128 * 32 + 128 * 64 + 2 * 128 * 2 + 2 * 128 * 4;
            let shared = (3 * stage) as u32;
            let ok = self.launch("k_gemm_fp4_pipe_h", (rows.div_ceil(128), t.div_ceil(128), 1), (256, 1, 1), shared, &mut a);
            return (ok, true);
        }
        (self.gemm_fp4(wp, wsc, xq, xs, c, 0, rows, cols, t, rows), false)
    }
    /// h = silu(gate) * up from an f16 gate|up, quantized into xq/xs.
    pub fn silu_mul_quant_rows_h(&self, gu: &DBuf, h: &DBuf, xq: &DBuf, xs: &DBuf, rows: u32, inter: u32) -> bool {
        let mut a = args!(gu.ptr, h.ptr, xq.ptr, xs.ptr, rows, inter);
        self.launch("k_silu_mul_quant_rows_h", (inter.div_ceil(1024), rows, 1), (1024, 1, 1), 0, &mut a)
    }
    /// C[t][rows] = X(q8) . W(MXFP4)^T.
    pub fn gemm_fp4(&self, wp: &DBuf, wsc: &DBuf, xq: &DBuf, xs: &DBuf, c: &DBuf, c_off: usize, rows: u32, cols: u32, t: u32, ldc: u32) -> bool {
        if self.mma_on() {
            return self.gemm_mma_split(false, wp, wsc, xq, xs, c, c_off, rows, cols, t, ldc);
        }
        let cp = c.ptr + c_off as u64;
        let mut a = args!(wp.ptr, wsc.ptr, xq.ptr, xs.ptr, cp, rows, cols, t, ldc);
        self.launch("k_gemm_fp4", (rows.div_ceil(64), t.div_ceil(64), 1), (256, 1, 1), 0, &mut a)
    }
    /// The tensor-core GEMM over `t` tokens as one launch with the chosen
    /// tile over the whole tiles, plus one launch with a smaller tile
    /// over the remainder (a remainder of nine tokens must not stage a
    /// 256-token tile: the weight tile is re-read per token tile).
    fn gemm_mma_split(&self, q8: bool, w: &DBuf, wsc: &DBuf, xq: &DBuf, xs: &DBuf, c: &DBuf, c_off: usize, rows: u32, cols: u32, t: u32, ldc: u32) -> bool {
        let nb = cols / 32;
        static PIPE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let pipe = *PIPE.get_or_init(|| !std::env::var("MICROKIMI_CUDA_NO_PIPE").map(|v| v == "1").unwrap_or(false));
        if pipe && t >= 64 {
            let cp = c.ptr + c_off as u64;
            let mut a = args!(w.ptr, wsc.ptr, xq.ptr, xs.ptr, cp, rows, cols, t, ldc);
            let name = if q8 { "k_gemm_q8_pipe" } else { "k_gemm_fp4_pipe" };
            let wrow = if q8 { 64usize } else { 32usize };
            let stage = 128 * wrow + 128 * 64 + 2 * 128 * 2 + 2 * 128 * 4;
            let shared = (3 * stage) as u32;
            let (gx, gy) = (rows.div_ceil(128), t.div_ceil(128));
            // a grid under two blocks per SM (the 5120-row projections: 80
            // blocks on 58 SMs) splits K in two: both halves add into a zeroed
            // C - an exact, order-free sum of two terms
            let mut splits = 1u32;
            if gx * gy < (2 * self.sm_count) as u32 && ldc == rows && cols / 32 >= 8 {
                splits = 2;
                // SAFETY: c holds t rows of ldc floats from c_off (checked by the caller's sizing).
                unsafe {
                    (self.drv.memset_d8_async)(cp, 0, (t as usize) * (ldc as usize) * 4, self.stream);
                }
            }
            return self.launch(name, (gx, gy, splits), (256, 1, 1), shared, &mut a);
        }
        let one = |t0: u32, tt: u32, mt: u32, nt: u32| -> bool {
            let name = match (q8, mt, nt) {
                (true, 128, _) => "k_gemm_q8_mma128x128",
                (true, _, 64) => "k_gemm_q8_mma64",
                (true, _, 128) => "k_gemm_q8_mma128",
                (true, _, _) => "k_gemm_q8_mma256",
                (false, 128, _) => "k_gemm_fp4_mma128x128",
                (false, _, 64) => "k_gemm_fp4_mma64",
                (false, _, 128) => "k_gemm_fp4_mma128",
                (false, _, _) => "k_gemm_fp4_mma256",
            };
            let xqp = xq.ptr + (t0 as u64) * cols as u64;
            let xsp = xs.ptr + (t0 as u64) * nb as u64 * 4;
            let cp = c.ptr + c_off as u64 + (t0 as u64) * ldc as u64 * 4;
            let mut a = args!(w.ptr, wsc.ptr, xqp, xsp, cp, rows, cols, tt, ldc);
            self.launch(name, (rows.div_ceil(mt), tt.div_ceil(nt), 1), (256, 1, 1), 0, &mut a)
        };
        let (mt, nt) = self.gemm_tile(rows, t);
        let full = t / nt;
        let rem = t - full * nt;
        // a tiny tail (a few tokens over a tile boundary) gets its own
        // small-tile launch when the main launch alone still fills the
        // SMs; otherwise the last tile is simply partial (its empty n8
        // tiles skip their mma)
        let row_tiles = rows.div_ceil(mt);
        if full >= 1 && rem > 0 && rem <= 32 && row_tiles * full >= (2 * self.sm_count) as u32 {
            return one(0, full * nt, mt, nt) && one(full * nt, rem, 64, 64);
        }
        one(0, t, mt, nt)
    }
    /// Token tile of the tensor-core GEMM: the largest of 256 / 128 / 64
    /// that still puts at least two blocks per SM (the weight tile is
    /// re-read from global once per token tile, so larger is better when
    /// the grid is wide enough).
    /// (row tile, token tile) of the tensor-core GEMM: 128 x 128 when the
    /// grid still fills the SMs twice over (half the activation re-reads
    /// of the 64-row tiles), else the largest 64 x NT that does; else 64 x 64.
    /// MICROKIMI_CUDA_TILE=MTxNT forces one.
    fn gemm_tile(&self, rows: u32, t: u32) -> (u32, u32) {
        static FORCE: std::sync::OnceLock<Option<(u32, u32)>> = std::sync::OnceLock::new();
        if let Some(tile) = *FORCE.get_or_init(|| {
            std::env::var("MICROKIMI_CUDA_TILE").ok().and_then(|v| {
                let (a, b) = v.split_once('x')?;
                Some((a.parse().ok()?, b.parse().ok()?))
            })
        }) {
            return tile;
        }
        let want = (2 * self.sm_count) as u32;
        if t > 64 && rows.div_ceil(128) * t.div_ceil(128) >= want {
            return (128, 128);
        }
        let row_tiles = rows.div_ceil(64);
        for nt in [256u32, 128, 64] {
            if t <= nt / 2 && nt > 64 {
                continue; // a half-empty tile: try the smaller one
            }
            if row_tiles * t.div_ceil(nt) >= want {
                return (64, nt);
            }
        }
        (64, 64)
    }
    #[allow(dead_code)]
    fn gemm_nt(&self, rows: u32, t: u32) -> u32 {
        self.gemm_tile(rows, t).1
    }
    /// Tensor-core GEMMs: sm_80 and later, unless MICROKIMI_CUDA_NO_MMA=1.
    pub fn mma_on(&self) -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| !std::env::var("MICROKIMI_CUDA_NO_MMA").map(|v| v == "1").unwrap_or(false)) && self.cc.0 >= 8
    }
    /// x[n] += y[n]
    pub fn add(&self, x: &DBuf, x_off: usize, y: &DBuf, y_off: usize, n: u32) -> bool {
        let (xp, yp) = (x.ptr + x_off as u64, y.ptr + y_off as u64);
        let mut a = args!(xp, yp, n);
        self.launch("k_add", (n.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a)
    }
    /// out = rmsnorm(x + add?) * (1+w | w), quantized into xq/xs as well.
    pub fn add_rmsnorm_quant_rows(&self, x: &DBuf, add: Option<&DBuf>, w: &DBuf, out: Option<&DBuf>, xq: &DBuf, xs: &DBuf, rows: u32, n: u32, eps: f32, one_plus: bool) -> bool {
        let ap: CUdeviceptr = add.map(|b| b.ptr).unwrap_or(0);
        let outp: CUdeviceptr = out.map(|b| b.ptr).unwrap_or(0);
        let op: u32 = if one_plus { 1 } else { 0 };
        let mut a = args!(x.ptr, ap, w.ptr, outp, xq.ptr, xs.ptr, rows, n, eps, op);
        self.launch("k_add_rmsnorm_quant_rows", (rows, 1, 1), (1024, 1, 1), 0, &mut a)
    }
    /// h = silu(gate) * up, quantized into xq/xs as well.
    pub fn silu_mul_quant_rows(&self, gu: &DBuf, h: &DBuf, xq: &DBuf, xs: &DBuf, rows: u32, inter: u32) -> bool {
        let mut a = args!(gu.ptr, h.ptr, xq.ptr, xs.ptr, rows, inter);
        self.launch("k_silu_mul_quant_rows", (inter.div_ceil(1024), rows, 1), (1024, 1, 1), 0, &mut a)
    }
}

// ── the kernels ──

const KERNEL_SRC: &str = r#"
// microkimi CUDA kernels (compiled by NVRTC at run time; see cuda.rs)
typedef unsigned char u8;
typedef signed char i8;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// block-wide sum for blockDim.x <= 1024 (result valid in every thread)
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? red[threadIdx.x] : 0.0f;
    if (wid == 0) t = warp_sum(t);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    return red[0];
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = warp_max(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    float t = (threadIdx.x < nw) ? red[threadIdx.x] : -3.0e38f;
    if (wid == 0) t = warp_max(t);
    if (threadIdx.x == 0) red[0] = t;
    __syncthreads();
    return red[0];
}
// f16 bits -> f32 (hardware cvt; no header needed under NVRTC)
__device__ __forceinline__ float h2f(unsigned short h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }
// 2^(e-128) exactly (e8m0 scale byte, LUT2 convention: values doubled):
// the f32 bit pattern directly - biased exponent e - 1 for e >= 1, the
// one subnormal 2^-128 for e = 0; no ldexpf (a slow path with branches)
__device__ __forceinline__ float e8m0_x2(int e) {
    return __int_as_float(e >= 1 ? (e - 1) << 23 : 0x00400000);
}

// four MXFP4 nibbles (a 16-bit half-word: bytes b0 b1, low nibble first)
// -> one int of four i8 (E2M1 x 2 values), two byte permutes: the
// magnitude table 0,1,2,3,4,6,8,12 indexed by the low three bits of each
// nibble, the sign bits selecting 0xFF masks, then a bytewise negate
__device__ __forceinline__ int fp4_decode4(unsigned h) {
    unsigned sel = h & 0x7777u;                       // magnitude indices as permute selectors
    unsigned mag = __byte_perm(0x03020100u, 0x0C080604u, sel);
    unsigned ssel = (h >> 1) & 0x4444u;               // sign bit -> selector 4 (byte 0 of the second word)
    unsigned m = __byte_perm(0x00000000u, 0x000000FFu, ssel);
    return (int)__vsub4(mag ^ m, m);                  // (x ^ 0xFF) - 0xFF == -x per signed byte
}
// MXFP4 nibble decode: 8 nibbles (a 32-bit word of packed bytes, low nibble
// = even column) -> two ints of four i8 (E2M1 x 2 values), columns 0..3 and 4..7
__device__ __forceinline__ void fp4_decode8(unsigned w, int& lo4, int& hi4) {
    // magnitude table 0,1,2,3,4,6,8,12 as 4-bit fields
    const unsigned MAG = 0xC8643210u;
    unsigned v[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned nib = (w >> (4 * i)) & 0xFu;
        int mag = (MAG >> (4 * (nib & 7))) & 0xF;
        int val = (nib & 8) ? -mag : mag;
        v[i] = (unsigned)val & 0xFFu;
    }
    // byte order in memory: byte0 = nibbles 0 (lo) 1 (hi) -> columns 0,1; byte1 -> 2,3 ...
    lo4 = (int)(v[0] | (v[1] << 8) | (v[2] << 16) | (v[3] << 24));
    hi4 = (int)(v[4] | (v[5] << 8) | (v[6] << 16) | (v[7] << 24));
}

// ── quantization ──
// one thread per block of 32: rows*(n/32) blocks over a row-major [rows][n]
extern "C" __global__ void k_quantize_q8(const float* __restrict__ x, i8* __restrict__ xq, float* __restrict__ xs, unsigned nb, unsigned n) {
    unsigned b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb) return;
    const float* p = x + (size_t)b * 32;
    float m = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) m = fmaxf(m, fabsf(p[j]));
    float dx = m / 127.0f;
    xs[b] = dx;
    i8* q = xq + (size_t)b * 32;
    if (dx == 0.0f) {
        #pragma unroll
        for (int j = 0; j < 32; j++) q[j] = 0;
        return;
    }
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        float r = roundf(p[j] / dx);
        r = fminf(fmaxf(r, -127.0f), 127.0f);
        q[j] = (i8)(int)r;
    }
}

// ── matvecs: one warp per row, lane l handles blocks l, l+32, ... ──
extern "C" __global__ void k_matvec_q8(const i8* __restrict__ wq, const unsigned short* __restrict__ ws,
                                       const i8* __restrict__ xq, const float* __restrict__ xs,
                                       float* __restrict__ y, unsigned rows, unsigned cols, unsigned rpw) {
    extern __shared__ __align__(16) unsigned char smem[];
    int* sx = (int*)smem;                       // cols bytes as ints
    float* sxs = (float*)(smem + cols);         // cols/32 scales
    const unsigned nb = cols >> 5;
    for (unsigned i = threadIdx.x; i < (cols >> 2); i += blockDim.x) sx[i] = ((const int*)xq)[i];
    for (unsigned i = threadIdx.x; i < nb; i += blockDim.x) sxs[i] = xs[i];
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    // rpw rows per warp: the activation is staged once per warps*rpw rows
    for (unsigned row = blockIdx.x * warps * rpw + warp; row < min(rows, (blockIdx.x + 1) * warps * rpw); row += warps) {
    const int4* wrow = (const int4*)(wq + (size_t)row * cols);
    const unsigned short* srow = ws + (size_t)row * nb;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    unsigned g = lane;
    // four blocks per iteration, independent chains (memory-level parallelism)
    for (; g + 96 < nb; g += 128) {
        int4 a = wrow[g * 2], b = wrow[g * 2 + 1];
        int4 a2 = wrow[(g + 32) * 2], b2 = wrow[(g + 32) * 2 + 1];
        int4 a3 = wrow[(g + 64) * 2], b3 = wrow[(g + 64) * 2 + 1];
        int4 a4 = wrow[(g + 96) * 2], b4 = wrow[(g + 96) * 2 + 1];
        unsigned short s1 = srow[g], s2 = srow[g + 32], s3 = srow[g + 64], s4 = srow[g + 96];
        const int* xb = sx + g * 8;
        const int* xb2 = sx + (g + 32) * 8;
        const int* xb3 = sx + (g + 64) * 8;
        const int* xb4 = sx + (g + 96) * 8;
        int d = 0, d2 = 0, d3 = 0, d4 = 0;
        d = __dp4a(a.x, xb[0], d); d = __dp4a(a.y, xb[1], d); d = __dp4a(a.z, xb[2], d); d = __dp4a(a.w, xb[3], d);
        d = __dp4a(b.x, xb[4], d); d = __dp4a(b.y, xb[5], d); d = __dp4a(b.z, xb[6], d); d = __dp4a(b.w, xb[7], d);
        d2 = __dp4a(a2.x, xb2[0], d2); d2 = __dp4a(a2.y, xb2[1], d2); d2 = __dp4a(a2.z, xb2[2], d2); d2 = __dp4a(a2.w, xb2[3], d2);
        d2 = __dp4a(b2.x, xb2[4], d2); d2 = __dp4a(b2.y, xb2[5], d2); d2 = __dp4a(b2.z, xb2[6], d2); d2 = __dp4a(b2.w, xb2[7], d2);
        d3 = __dp4a(a3.x, xb3[0], d3); d3 = __dp4a(a3.y, xb3[1], d3); d3 = __dp4a(a3.z, xb3[2], d3); d3 = __dp4a(a3.w, xb3[3], d3);
        d3 = __dp4a(b3.x, xb3[4], d3); d3 = __dp4a(b3.y, xb3[5], d3); d3 = __dp4a(b3.z, xb3[6], d3); d3 = __dp4a(b3.w, xb3[7], d3);
        d4 = __dp4a(a4.x, xb4[0], d4); d4 = __dp4a(a4.y, xb4[1], d4); d4 = __dp4a(a4.z, xb4[2], d4); d4 = __dp4a(a4.w, xb4[3], d4);
        d4 = __dp4a(b4.x, xb4[4], d4); d4 = __dp4a(b4.y, xb4[5], d4); d4 = __dp4a(b4.z, xb4[6], d4); d4 = __dp4a(b4.w, xb4[7], d4);
        acc0 = fmaf((float)d, h2f(s1) * sxs[g], acc0);
        acc1 = fmaf((float)d2, h2f(s2) * sxs[g + 32], acc1);
        acc2 = fmaf((float)d3, h2f(s3) * sxs[g + 64], acc2);
        acc3 = fmaf((float)d4, h2f(s4) * sxs[g + 96], acc3);
    }
    for (; g + 32 < nb; g += 64) {
        int4 a = wrow[g * 2], b = wrow[g * 2 + 1];
        int4 a2 = wrow[(g + 32) * 2], b2 = wrow[(g + 32) * 2 + 1];
        unsigned short s1 = srow[g], s2 = srow[g + 32];
        const int* xb = sx + g * 8;
        const int* xb2 = sx + (g + 32) * 8;
        int d = 0, d2 = 0;
        d = __dp4a(a.x, xb[0], d); d = __dp4a(a.y, xb[1], d); d = __dp4a(a.z, xb[2], d); d = __dp4a(a.w, xb[3], d);
        d = __dp4a(b.x, xb[4], d); d = __dp4a(b.y, xb[5], d); d = __dp4a(b.z, xb[6], d); d = __dp4a(b.w, xb[7], d);
        d2 = __dp4a(a2.x, xb2[0], d2); d2 = __dp4a(a2.y, xb2[1], d2); d2 = __dp4a(a2.z, xb2[2], d2); d2 = __dp4a(a2.w, xb2[3], d2);
        d2 = __dp4a(b2.x, xb2[4], d2); d2 = __dp4a(b2.y, xb2[5], d2); d2 = __dp4a(b2.z, xb2[6], d2); d2 = __dp4a(b2.w, xb2[7], d2);
        acc0 = fmaf((float)d, h2f(s1) * sxs[g], acc0);
        acc1 = fmaf((float)d2, h2f(s2) * sxs[g + 32], acc1);
    }
    for (; g < nb; g += 32) {
        int4 a = wrow[g * 2], b = wrow[g * 2 + 1];
        const int* xb = sx + g * 8;
        int d = 0;
        d = __dp4a(a.x, xb[0], d); d = __dp4a(a.y, xb[1], d); d = __dp4a(a.z, xb[2], d); d = __dp4a(a.w, xb[3], d);
        d = __dp4a(b.x, xb[4], d); d = __dp4a(b.y, xb[5], d); d = __dp4a(b.z, xb[6], d); d = __dp4a(b.w, xb[7], d);
        acc0 = fmaf((float)d, h2f(srow[g]) * sxs[g], acc0);
    }
    float acc = warp_sum((acc0 + acc1) + (acc2 + acc3));
    if (lane == 0) y[row] = acc;
    }
}

extern "C" __global__ void k_matvec_fp4(const u8* __restrict__ wp, const u8* __restrict__ wsc,
                                        const i8* __restrict__ xq, const float* __restrict__ xs,
                                        float* __restrict__ y, unsigned rows, unsigned cols, unsigned rpw) {
    extern __shared__ __align__(16) unsigned char smem[];
    int* sx = (int*)smem;
    float* sxs = (float*)(smem + cols);
    const unsigned nb = cols >> 5;
    for (unsigned i = threadIdx.x; i < (cols >> 2); i += blockDim.x) sx[i] = ((const int*)xq)[i];
    for (unsigned i = threadIdx.x; i < nb; i += blockDim.x) sxs[i] = xs[i];
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (unsigned row = blockIdx.x * warps * rpw + warp; row < min(rows, (blockIdx.x + 1) * warps * rpw); row += warps) {
    const int4* prow = (const int4*)(wp + (size_t)row * (cols >> 1));
    const u8* srow = wsc + (size_t)row * nb;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    unsigned g = lane;
    // four blocks per iteration, independent chains, streaming loads (the
    // weights are read once per token: keep them out of the caches)
    for (; g + 96 < nb; g += 128) {
        int4 pk = __ldcs(prow + g), pk2 = __ldcs(prow + g + 32), pk3 = __ldcs(prow + g + 64), pk4 = __ldcs(prow + g + 96);
        unsigned s1 = srow[g], s2 = srow[g + 32], s3 = srow[g + 64], s4 = srow[g + 96];
        const int* xb = sx + g * 8;
        const int* xb2 = sx + (g + 32) * 8;
        const int* xb3 = sx + (g + 64) * 8;
        const int* xb4 = sx + (g + 96) * 8;
        int w0, w1, w2, w3, w4, w5, w6, w7;
        int d;
        fp4_decode8((unsigned)pk.x, w0, w1); fp4_decode8((unsigned)pk.y, w2, w3); fp4_decode8((unsigned)pk.z, w4, w5); fp4_decode8((unsigned)pk.w, w6, w7);
        d = 0; d = __dp4a(w0, xb[0], d); d = __dp4a(w1, xb[1], d); d = __dp4a(w2, xb[2], d); d = __dp4a(w3, xb[3], d);
        d = __dp4a(w4, xb[4], d); d = __dp4a(w5, xb[5], d); d = __dp4a(w6, xb[6], d); d = __dp4a(w7, xb[7], d);
        acc0 = fmaf((float)d, e8m0_x2(s1) * sxs[g], acc0);
        fp4_decode8((unsigned)pk2.x, w0, w1); fp4_decode8((unsigned)pk2.y, w2, w3); fp4_decode8((unsigned)pk2.z, w4, w5); fp4_decode8((unsigned)pk2.w, w6, w7);
        d = 0; d = __dp4a(w0, xb2[0], d); d = __dp4a(w1, xb2[1], d); d = __dp4a(w2, xb2[2], d); d = __dp4a(w3, xb2[3], d);
        d = __dp4a(w4, xb2[4], d); d = __dp4a(w5, xb2[5], d); d = __dp4a(w6, xb2[6], d); d = __dp4a(w7, xb2[7], d);
        acc1 = fmaf((float)d, e8m0_x2(s2) * sxs[g + 32], acc1);
        fp4_decode8((unsigned)pk3.x, w0, w1); fp4_decode8((unsigned)pk3.y, w2, w3); fp4_decode8((unsigned)pk3.z, w4, w5); fp4_decode8((unsigned)pk3.w, w6, w7);
        d = 0; d = __dp4a(w0, xb3[0], d); d = __dp4a(w1, xb3[1], d); d = __dp4a(w2, xb3[2], d); d = __dp4a(w3, xb3[3], d);
        d = __dp4a(w4, xb3[4], d); d = __dp4a(w5, xb3[5], d); d = __dp4a(w6, xb3[6], d); d = __dp4a(w7, xb3[7], d);
        acc2 = fmaf((float)d, e8m0_x2(s3) * sxs[g + 64], acc2);
        fp4_decode8((unsigned)pk4.x, w0, w1); fp4_decode8((unsigned)pk4.y, w2, w3); fp4_decode8((unsigned)pk4.z, w4, w5); fp4_decode8((unsigned)pk4.w, w6, w7);
        d = 0; d = __dp4a(w0, xb4[0], d); d = __dp4a(w1, xb4[1], d); d = __dp4a(w2, xb4[2], d); d = __dp4a(w3, xb4[3], d);
        d = __dp4a(w4, xb4[4], d); d = __dp4a(w5, xb4[5], d); d = __dp4a(w6, xb4[6], d); d = __dp4a(w7, xb4[7], d);
        acc3 = fmaf((float)d, e8m0_x2(s4) * sxs[g + 96], acc3);
    }
    for (; g + 32 < nb; g += 64) {
        int4 pk = __ldcs(prow + g), pk2 = __ldcs(prow + g + 32);
        unsigned s1 = srow[g], s2 = srow[g + 32];
        const int* xb = sx + g * 8;
        const int* xb2 = sx + (g + 32) * 8;
        int w0, w1, w2, w3, w4, w5, w6, w7;
        fp4_decode8((unsigned)pk.x, w0, w1); fp4_decode8((unsigned)pk.y, w2, w3);
        fp4_decode8((unsigned)pk.z, w4, w5); fp4_decode8((unsigned)pk.w, w6, w7);
        int d = 0;
        d = __dp4a(w0, xb[0], d); d = __dp4a(w1, xb[1], d); d = __dp4a(w2, xb[2], d); d = __dp4a(w3, xb[3], d);
        d = __dp4a(w4, xb[4], d); d = __dp4a(w5, xb[5], d); d = __dp4a(w6, xb[6], d); d = __dp4a(w7, xb[7], d);
        acc0 = fmaf((float)d, e8m0_x2(s1) * sxs[g], acc0);
        fp4_decode8((unsigned)pk2.x, w0, w1); fp4_decode8((unsigned)pk2.y, w2, w3);
        fp4_decode8((unsigned)pk2.z, w4, w5); fp4_decode8((unsigned)pk2.w, w6, w7);
        int d2 = 0;
        d2 = __dp4a(w0, xb2[0], d2); d2 = __dp4a(w1, xb2[1], d2); d2 = __dp4a(w2, xb2[2], d2); d2 = __dp4a(w3, xb2[3], d2);
        d2 = __dp4a(w4, xb2[4], d2); d2 = __dp4a(w5, xb2[5], d2); d2 = __dp4a(w6, xb2[6], d2); d2 = __dp4a(w7, xb2[7], d2);
        acc1 = fmaf((float)d2, e8m0_x2(s2) * sxs[g + 32], acc1);
    }
    for (; g < nb; g += 32) {
        int4 pk = __ldcs(prow + g);
        const int* xb = sx + g * 8;
        int w0, w1, w2, w3, w4, w5, w6, w7;
        fp4_decode8((unsigned)pk.x, w0, w1); fp4_decode8((unsigned)pk.y, w2, w3);
        fp4_decode8((unsigned)pk.z, w4, w5); fp4_decode8((unsigned)pk.w, w6, w7);
        int d = 0;
        d = __dp4a(w0, xb[0], d); d = __dp4a(w1, xb[1], d); d = __dp4a(w2, xb[2], d); d = __dp4a(w3, xb[3], d);
        d = __dp4a(w4, xb[4], d); d = __dp4a(w5, xb[5], d); d = __dp4a(w6, xb[6], d); d = __dp4a(w7, xb[7], d);
        acc0 = fmaf((float)d, e8m0_x2(srow[g]) * sxs[g], acc0);
    }
    float acc = warp_sum((acc0 + acc1) + (acc2 + acc3));
    if (lane == 0) y[row] = acc;
    }
}

// ── GEMM: C[t][r] = sum_g ws[r][g] xs[t][g] dot(W[r][g], X[t][g]) ──
// tiles of 64 rows x 64 tokens, k-step of one 32-column block; 256 threads,
// each a 4x4 (rows x tokens) micro-tile. W tile [64][32] i8, X tile [64][32] i8.
#define GT 64
extern "C" __global__ void k_gemm_q8(const i8* __restrict__ wq, const unsigned short* __restrict__ ws,
                                     const i8* __restrict__ xq, const float* __restrict__ xs,
                                     float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    __shared__ __align__(16) int sw[GT * 8];   // 64 rows x 8 ints
    __shared__ __align__(16) int sxt[GT * 8];  // 64 tokens x 8 ints
    __shared__ float sws[GT], sxs[GT];
    const unsigned nb = cols >> 5;
    const unsigned r0 = blockIdx.x * GT, t0 = blockIdx.y * GT;
    const unsigned tid = threadIdx.x;
    const unsigned tr = (tid & 15) * 4;   // 4 rows
    const unsigned tt = (tid >> 4) * 4;   // 4 tokens
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (unsigned g = 0; g < nb; g++) {
        // load: 64 rows x 32 bytes = 512 ints; 2 per thread; same for X
        {
            unsigned i = tid;             // 0..255 -> row i/4? no: 512 ints / 256 threads
            #pragma unroll
            for (int rep = 0; rep < 2; rep++) {
                unsigned idx = i + rep * 256;      // 0..511
                unsigned rr = idx >> 3, k = idx & 7;
                unsigned grow = r0 + rr;
                sw[idx] = (grow < rows) ? ((const int*)(wq + (size_t)grow * cols + g * 32))[k] : 0;
                unsigned gtok = t0 + rr;
                sxt[idx] = (gtok < t) ? ((const int*)(xq + (size_t)gtok * cols + g * 32))[k] : 0;
            }
            if (tid < GT) {
                unsigned grow = r0 + tid;
                sws[tid] = (grow < rows) ? h2f(ws[(size_t)grow * nb + g]) : 0.0f;
            } else if (tid < 2 * GT) {
                unsigned gtok = t0 + tid - GT;
                sxs[tid - GT] = (gtok < t) ? xs[(size_t)gtok * nb + g] : 0.0f;
            }
        }
        __syncthreads();
        int wv[4][8], xv[4][8];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int k = 0; k < 8; k++) wv[i][k] = sw[(tr + i) * 8 + k];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[j][k] = sxt[(tt + j) * 8 + k];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float wsi = sws[tr + i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int d = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) d = __dp4a(wv[i][k], xv[j][k], d);
                acc[i][j] = fmaf((float)d, wsi * sxs[tt + j], acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned gtok = t0 + tt + j;
        if (gtok >= t) continue;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned grow = r0 + tr + i;
            if (grow < rows) C[(size_t)gtok * ldc + grow] = acc[i][j];
        }
    }
}

extern "C" __global__ void k_gemm_fp4(const u8* __restrict__ wp, const u8* __restrict__ wsc,
                                      const i8* __restrict__ xq, const float* __restrict__ xs,
                                      float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    __shared__ __align__(16) int sw[GT * 8];
    __shared__ __align__(16) int sxt[GT * 8];
    __shared__ float sws[GT], sxs[GT];
    const unsigned nb = cols >> 5;
    const unsigned r0 = blockIdx.x * GT, t0 = blockIdx.y * GT;
    const unsigned tid = threadIdx.x;
    const unsigned tr = (tid & 15) * 4;
    const unsigned tt = (tid >> 4) * 4;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (unsigned g = 0; g < nb; g++) {
        {
            // W tile: 64 rows x 16 packed bytes = 256 ints of nibbles -> one per thread, decoded to 2 ints
            unsigned rr = tid >> 2, k = tid & 3;   // row, packed word
            unsigned grow = r0 + rr;
            unsigned pk = (grow < rows) ? ((const unsigned*)(wp + (size_t)grow * (cols >> 1) + g * 16))[k] : 0u;
            int lo, hi;
            fp4_decode8(pk, lo, hi);
            sw[rr * 8 + k * 2] = lo;
            sw[rr * 8 + k * 2 + 1] = hi;
            #pragma unroll
            for (int rep = 0; rep < 2; rep++) {
                unsigned idx = tid + rep * 256;
                unsigned r2 = idx >> 3, k2 = idx & 7;
                unsigned gtok = t0 + r2;
                sxt[idx] = (gtok < t) ? ((const int*)(xq + (size_t)gtok * cols + g * 32))[k2] : 0;
            }
            if (tid < GT) {
                sws[tid] = (grow < rows) ? 0.0f : 0.0f; // placeholder, set below
                unsigned gr = r0 + tid;
                sws[tid] = (gr < rows) ? e8m0_x2(wsc[(size_t)gr * nb + g]) : 0.0f;
            } else if (tid < 2 * GT) {
                unsigned gtok = t0 + tid - GT;
                sxs[tid - GT] = (gtok < t) ? xs[(size_t)gtok * nb + g] : 0.0f;
            }
        }
        __syncthreads();
        int wv[4][8], xv[4][8];
        #pragma unroll
        for (int i = 0; i < 4; i++)
            #pragma unroll
            for (int k = 0; k < 8; k++) wv[i][k] = sw[(tr + i) * 8 + k];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int k = 0; k < 8; k++) xv[j][k] = sxt[(tt + j) * 8 + k];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float wsi = sws[tr + i];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int d = 0;
                #pragma unroll
                for (int k = 0; k < 8; k++) d = __dp4a(wv[i][k], xv[j][k], d);
                acc[i][j] = fmaf((float)d, wsi * sxs[tt + j], acc[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        unsigned gtok = t0 + tt + j;
        if (gtok >= t) continue;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned grow = r0 + tr + i;
            if (grow < rows) C[(size_t)gtok * ldc + grow] = acc[i][j];
        }
    }
}


// ── GEMM on tensor cores (sm_80+): mma.sync m16n8k32 s8 x s8 -> s32; one
// instruction is one 32-column block of a 16 x 8 tile, so its s32 result IS
// the exact block dot; the two block scales apply in float per block, as in
// every other q8 kernel. Tile 64 rows x NT tokens (NT = 64, 128, 256), 8
// warps as 2 (rows) x 4 (tokens), two blocks per staging step; the weight
// tile is read from global once per NT tokens.
__device__ __forceinline__ void mma_s8_16x8x32(int& d0, int& d1, int& d2, int& d3,
                                               int a0, int a1, int a2, int a3, int b0, int b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    // sm_75 and older (T4): the tensor-core GEMMs are never launched there
    // (mma_on() is false); the symbol still has to compile
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
    d0 = d1 = d2 = d3 = 0;
#endif
}
// the same with one zero register passed in for the accumulator inputs
// (declared once by the caller; the literal form re-materializes four
// zero registers per mma)
__device__ __forceinline__ void mma_s8_16x8x32_z(int& d0, int& d1, int& d2, int& d3,
                                                 int a0, int a1, int a2, int a3, int b0, int b1, int z) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(z));
#else
    (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1; (void)z;
    d0 = d1 = d2 = d3 = 0;
#endif
}
#define SROW 9 /* shared row stride in ints per block (8 + 1 pad against bank conflicts) */
// warps: WM x WN = 8; each warp owns 32 rows and NT / WN tokens
template <int MTR> struct WMOf { static constexpr int v = MTR / 32; };
// blocks of 32 columns per staging step, by token tile (static shared memory stays under 48 KB)
template <int NT> struct KBOf { static constexpr int v = (NT >= 256) ? 2 : (NT >= 128 ? 4 : 8); };

template <int MTR, int NT>
__device__ __forceinline__ void gemm_mma_body(const int* sw, const int* sxt, const float* sws, const float* sxs,
                                              float (&acc)[2][NT / (8 / WMOf<MTR>::v) / 8][4], unsigned warp, unsigned lane, unsigned n_valid) {
    constexpr int KB = KBOf<NT>::v; constexpr int LDS = KB * SROW;
    constexpr int WM = WMOf<MTR>::v, WN = 8 / WM;
    constexpr int WT = NT / WN;  // tokens per warp
    constexpr int NW = WT / 8;   // n8 tiles per warp
    // n8 tiles past n_valid (the block's valid tokens) are skipped, so a
    // prompt a few tokens over a tile boundary does not pay a whole tile
    const unsigned wr0 = (warp % WM) * 32, wt0 = (warp / WM) * WT;
    const unsigned gid = lane >> 2, tig = lane & 3;
    if (wt0 >= n_valid) return;
    #pragma unroll
    for (int kb = 0; kb < KB; kb++) {
        int a[2][4], b[NW][2];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            unsigned r = wr0 + mi * 16 + gid;
            a[mi][0] = sw[r * LDS + kb * SROW + tig];
            a[mi][1] = sw[(r + 8) * LDS + kb * SROW + tig];
            a[mi][2] = sw[r * LDS + kb * SROW + 4 + tig];
            a[mi][3] = sw[(r + 8) * LDS + kb * SROW + 4 + tig];
        }
        #pragma unroll
        for (int ni = 0; ni < NW; ni++) {
            unsigned tt = wt0 + ni * 8 + gid;
            b[ni][0] = sxt[tt * LDS + kb * SROW + tig];
            b[ni][1] = sxt[tt * LDS + kb * SROW + 4 + tig];
        }
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            float ws0 = sws[kb * MTR + wr0 + mi * 16 + gid];
            float ws1 = sws[kb * MTR + wr0 + mi * 16 + gid + 8];
            #pragma unroll
            for (int ni = 0; ni < NW; ni++) {
                if (wt0 + ni * 8 >= n_valid) break;
                int d0, d1, d2, d3;
                mma_s8_16x8x32(d0, d1, d2, d3, a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[ni][0], b[ni][1]);
                float xs0 = sxs[kb * NT + wt0 + ni * 8 + tig * 2];
                float xs1 = sxs[kb * NT + wt0 + ni * 8 + tig * 2 + 1];
                acc[mi][ni][0] = fmaf((float)d0, ws0 * xs0, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf((float)d1, ws0 * xs1, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf((float)d2, ws1 * xs0, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf((float)d3, ws1 * xs1, acc[mi][ni][3]);
            }
        }
    }
}
template <int MTR, int NT>
__device__ __forceinline__ void gemm_mma_store(float (&acc)[2][NT / (8 / WMOf<MTR>::v) / 8][4], float* __restrict__ C, unsigned r0, unsigned t0,
                                               unsigned rows, unsigned t, unsigned ldc, unsigned warp, unsigned lane) {
    constexpr int WM = WMOf<MTR>::v, WN = 8 / WM;
    constexpr int WT = NT / WN, NW = WT / 8;
    const unsigned wr0 = (warp % WM) * 32, wt0 = (warp / WM) * WT;
    const unsigned gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < NW; ni++) {
            unsigned r = r0 + wr0 + mi * 16 + gid;
            unsigned tt = t0 + wt0 + ni * 8 + tig * 2;
            if (tt < t) {
                if (r < rows) C[(size_t)tt * ldc + r] = acc[mi][ni][0];
                if (r + 8 < rows) C[(size_t)tt * ldc + r + 8] = acc[mi][ni][2];
            }
            if (tt + 1 < t) {
                if (r < rows) C[(size_t)(tt + 1) * ldc + r] = acc[mi][ni][1];
                if (r + 8 < rows) C[(size_t)(tt + 1) * ldc + r + 8] = acc[mi][ni][3];
            }
        }
    }
}
template <int MTR, int NT>
__device__ __forceinline__ void gemm_q8_mma_t(const i8* __restrict__ wq, const unsigned short* __restrict__ ws,
                                              const i8* __restrict__ xq, const float* __restrict__ xs,
                                              float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    constexpr int KB = KBOf<NT>::v; constexpr int LDS = KB * SROW;
    __shared__ __align__(16) int sw[MTR * LDS];
    __shared__ __align__(16) int sxt[NT * LDS];
    __shared__ float sws[KB * MTR], sxs[KB * NT];
    const unsigned nb = cols >> 5;
    const unsigned r0 = blockIdx.x * MTR, t0 = blockIdx.y * NT;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float acc[2][NT / (8 / WMOf<MTR>::v) / 8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < NT / (8 / WMOf<MTR>::v) / 8; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    for (unsigned g0 = 0; g0 < nb; g0 += KB) {
        #pragma unroll
        for (int rep = 0; rep < (MTR * KB * 8) / 256; rep++) {
            unsigned idx = tid + rep * 256;
            unsigned rr = idx / (KB * 8), kk = idx % (KB * 8);
            unsigned kb = kk >> 3, k = kk & 7;
            unsigned g = g0 + kb, grow = r0 + rr;
            sw[rr * LDS + kb * SROW + k] = (g < nb && grow < rows) ? ((const int*)(wq + (size_t)grow * cols + g * 32))[k] : 0;
        }
        #pragma unroll
        for (int rep = 0; rep < (NT * KB * 8) / 256; rep++) {
            unsigned idx = tid + rep * 256;
            unsigned rr = idx / (KB * 8), kk = idx % (KB * 8);
            unsigned kb = kk >> 3, k = kk & 7;
            unsigned g = g0 + kb, gtok = t0 + rr;
            sxt[rr * LDS + kb * SROW + k] = (g < nb && gtok < t) ? ((const int*)(xq + (size_t)gtok * cols + g * 32))[k] : 0;
        }
        for (unsigned i = tid; i < KB * MTR; i += 256) {
            unsigned kb = i / MTR, rr = i % MTR;
            unsigned g = g0 + kb, grow = r0 + rr;
            sws[i] = (g < nb && grow < rows) ? h2f(ws[(size_t)grow * nb + g]) : 0.0f;
        }
        for (unsigned i = tid; i < KB * NT; i += 256) {
            unsigned kb = i / NT, rr = i % NT;
            unsigned g = g0 + kb, gtok = t0 + rr;
            sxs[i] = (g < nb && gtok < t) ? xs[(size_t)gtok * nb + g] : 0.0f;
        }
        __syncthreads();
        gemm_mma_body<MTR, NT>(sw, sxt, sws, sxs, acc, warp, lane, t - t0);
        __syncthreads();
    }
    gemm_mma_store<MTR, NT>(acc, C, r0, t0, rows, t, ldc, warp, lane);
}
template <int MTR, int NT>
__device__ __forceinline__ void gemm_fp4_mma_t(const u8* __restrict__ wp, const u8* __restrict__ wsc,
                                               const i8* __restrict__ xq, const float* __restrict__ xs,
                                               float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    constexpr int KB = KBOf<NT>::v; constexpr int LDS = KB * SROW;
    __shared__ __align__(16) int sw[MTR * LDS];
    __shared__ __align__(16) int sxt[NT * LDS];
    __shared__ float sws[KB * MTR], sxs[KB * NT];
    const unsigned nb = cols >> 5;
    const unsigned r0 = blockIdx.x * MTR, t0 = blockIdx.y * NT;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float acc[2][NT / (8 / WMOf<MTR>::v) / 8][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < NT / (8 / WMOf<MTR>::v) / 8; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    for (unsigned g0 = 0; g0 < nb; g0 += KB) {
        #pragma unroll
        for (int rep = 0; rep < (MTR * KB * 4) / 256; rep++) {
            unsigned idx = tid + rep * 256;
            unsigned rr = idx / (KB * 4), kk = idx % (KB * 4);
            unsigned kb = kk >> 2, k = kk & 3;
            unsigned g = g0 + kb, grow = r0 + rr;
            unsigned pk = (g < nb && grow < rows) ? ((const unsigned*)(wp + (size_t)grow * (cols >> 1) + g * 16))[k] : 0u;
            int lo, hi;
            fp4_decode8(pk, lo, hi);
            sw[rr * LDS + kb * SROW + k * 2] = lo;
            sw[rr * LDS + kb * SROW + k * 2 + 1] = hi;
        }
        #pragma unroll
        for (int rep = 0; rep < (NT * KB * 8) / 256; rep++) {
            unsigned idx = tid + rep * 256;
            unsigned rr = idx / (KB * 8), kk = idx % (KB * 8);
            unsigned kb = kk >> 3, k = kk & 7;
            unsigned g = g0 + kb, gtok = t0 + rr;
            sxt[rr * LDS + kb * SROW + k] = (g < nb && gtok < t) ? ((const int*)(xq + (size_t)gtok * cols + g * 32))[k] : 0;
        }
        for (unsigned i = tid; i < KB * MTR; i += 256) {
            unsigned kb = i / MTR, rr = i % MTR;
            unsigned g = g0 + kb, grow = r0 + rr;
            sws[i] = (g < nb && grow < rows) ? e8m0_x2(wsc[(size_t)grow * nb + g]) : 0.0f;
        }
        for (unsigned i = tid; i < KB * NT; i += 256) {
            unsigned kb = i / NT, rr = i % NT;
            unsigned g = g0 + kb, gtok = t0 + rr;
            sxs[i] = (g < nb && gtok < t) ? xs[(size_t)gtok * nb + g] : 0.0f;
        }
        __syncthreads();
        gemm_mma_body<MTR, NT>(sw, sxt, sws, sxs, acc, warp, lane, t - t0);
        __syncthreads();
    }
    gemm_mma_store<MTR, NT>(acc, C, r0, t0, rows, t, ldc, warp, lane);
}
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_q8_mma64(const i8* __restrict__ wq, const unsigned short* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<64, 64>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_q8_mma128(const i8* __restrict__ wq, const unsigned short* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<64, 128>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_q8_mma256(const i8* __restrict__ wq, const unsigned short* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<64, 256>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_q8_mma128x128(const i8* __restrict__ wq, const unsigned short* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<128, 128>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_fp4_mma64(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<64, 64>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_fp4_mma128(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<64, 128>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_fp4_mma256(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<64, 256>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void __launch_bounds__(256, 2) k_gemm_fp4_mma128x128(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<128, 128>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }


// ── pipelined tensor-core GEMM (sm_80+): cp.async multi-stage staging ──
// Tile 128 rows x PN tokens, 8 warps as 4 x 2 (32 rows x PN/2 tokens
// each), KBS blocks (KBS*32 columns) per stage, STAGES stages in flight:
// the loads of a later step are in the air while this step computes, so
// the DRAM latency of a weight tile is hidden instead of paid at every
// step. Rows are stored as 16-byte chunks XOR-swizzled by the row for
// the fragment loads (zero bank conflicts); the fp4 weights stay packed
// in shared memory and decode at fragment load. Same exact block dots,
// same scale application.
#define PIPE_STAGES 3
__device__ __forceinline__ void cp_async16(void* smem_dst, const void* gsrc, bool valid) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem_dst);
    int n = valid ? 16 : 0;
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(s), "l"(gsrc), "r"(n));
#else
    (void)s; (void)gsrc; (void)n;
#endif
}
__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
// shared layout of one stage (bytes): W tile [PM][WROW] | X tile [PN][XROW] | ws [KBS*PM] u16 | xs [KBS*PN] f32
template <bool Q8, int PM, int PN, int KBS>
struct PipeLayout {
    static constexpr int WROW = Q8 ? KBS * 32 : KBS * 16;   // bytes per row per stage
    static constexpr int XROW = KBS * 32;
    static constexpr int WCH = WROW / 16;                   // 16-byte chunks per W row
    static constexpr int XCH = XROW / 16;                   // ... per X row
    static constexpr int W_BYTES = PM * WROW;
    static constexpr int X_BYTES = PN * XROW;
    static constexpr int WS_BYTES = KBS * PM * 2;
    static constexpr int XS_BYTES = KBS * PN * 4;
    static constexpr int STAGE = W_BYTES + X_BYTES + WS_BYTES + XS_BYTES;
    static constexpr int WM = PM / 32;                      // warp rows
    static constexpr int WN = 8 / WM;                       // warp columns
    static constexpr int WT = PN / WN;                      // tokens per warp
    static constexpr int NW = WT / 8;                       // n8 tiles per warp
};
__device__ __forceinline__ constexpr int log2c(int n) { return n <= 1 ? 0 : 1 + log2c(n / 2); }
// swizzled chunk index of (row, chunk c) for rows of 2^log2 chunks: the chunk
// XORed with row bits chosen so that eight consecutive rows of a fragment
// load hit eight distinct bank groups
__device__ __forceinline__ unsigned swz(unsigned row, unsigned c, unsigned log2) {
    unsigned mask = (1u << log2) - 1u;
    unsigned shift = log2 >= 3 ? 0u : (3u - log2);
    return (row << log2) + ((c ^ ((row >> shift) & mask)) & mask);
}
template <bool Q8, int PM, int PN, int KBS, bool FULL>
__device__ __forceinline__ void pipe_step(const unsigned char* base, unsigned wr0, unsigned wt0, unsigned gid, unsigned tig, unsigned nw,
                                          float (&acc)[2][PipeLayout<Q8, PM, PN, KBS>::NW][4], int zero) {
    typedef PipeLayout<Q8, PM, PN, KBS> L;
    constexpr int NW = L::NW;
    constexpr int WL = log2c(L::WCH), XL = log2c(L::XCH);
    const int* sw = (const int*)base;
    const int* sx = (const int*)(base + L::W_BYTES);
    const unsigned short* sw16 = (const unsigned short*)base;
    const unsigned short* sws = (const unsigned short*)(base + L::W_BYTES + L::X_BYTES);
    const float* sxs = (const float*)(base + L::W_BYTES + L::X_BYTES + L::WS_BYTES);
    #pragma unroll
    for (int kb = 0; kb < KBS; kb++) {
        int a[2][4], b[NW][2];
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            unsigned r = wr0 + mi * 16 + gid;
            if (Q8) {
                // block kb of a q8 row = chunks kb*2 (ints k 0..3), kb*2+1 (ints 4..7)
                a[mi][0] = sw[swz(r, kb * 2, WL) * 4 + tig];
                a[mi][1] = sw[swz(r + 8, kb * 2, WL) * 4 + tig];
                a[mi][2] = sw[swz(r, kb * 2 + 1, WL) * 4 + tig];
                a[mi][3] = sw[swz(r + 8, kb * 2 + 1, WL) * 4 + tig];
            } else {
                // block kb of a packed row = chunk kb (eight u16: half-words tig, 4 + tig)
                const unsigned short* p0 = sw16 + swz(r, kb, WL) * 8;
                const unsigned short* p1 = sw16 + swz(r + 8, kb, WL) * 8;
                a[mi][0] = fp4_decode4(p0[tig]);
                a[mi][1] = fp4_decode4(p1[tig]);
                a[mi][2] = fp4_decode4(p0[4 + tig]);
                a[mi][3] = fp4_decode4(p1[4 + tig]);
            }
        }
        #pragma unroll
        for (int ni = 0; ni < NW; ni++) {
            unsigned tt = wt0 + ni * 8 + gid;
            b[ni][0] = sx[swz(tt, kb * 2, XL) * 4 + tig];
            b[ni][1] = sx[swz(tt, kb * 2 + 1, XL) * 4 + tig];
        }
        #pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            unsigned rr0 = wr0 + mi * 16 + gid;
            float ws0, ws1;
            if (Q8) { ws0 = h2f(sws[kb * PM + rr0]); ws1 = h2f(sws[kb * PM + rr0 + 8]); }
            else { ws0 = e8m0_x2(sws[kb * PM + rr0]); ws1 = e8m0_x2(sws[kb * PM + rr0 + 8]); }
            #pragma unroll
            for (int ni = 0; ni < NW; ni++) {
                if (FULL || (unsigned)ni < nw) {
                    int d0, d1, d2, d3;
                    mma_s8_16x8x32_z(d0, d1, d2, d3, a[mi][0], a[mi][1], a[mi][2], a[mi][3], b[ni][0], b[ni][1], zero);
                    float xs0 = sxs[kb * PN + wt0 + ni * 8 + tig * 2];
                    float xs1 = sxs[kb * PN + wt0 + ni * 8 + tig * 2 + 1];
                    acc[mi][ni][0] = fmaf((float)d0, ws0 * xs0, acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf((float)d1, ws0 * xs1, acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf((float)d2, ws1 * xs0, acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf((float)d3, ws1 * xs1, acc[mi][ni][3]);
                }
            }
        }
    }
}
__device__ __forceinline__ unsigned short f2h(float f) {
    unsigned short h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
template <bool Q8, int PM, int PN, int KBS, int STAGES, bool OUT_HALF>
__device__ __forceinline__ void gemm_pipe_t(const u8* __restrict__ wbytes, const void* __restrict__ wsc,
                                            const i8* __restrict__ xq, const float* __restrict__ xs,
                                            float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    typedef PipeLayout<Q8, PM, PN, KBS> L;
    constexpr int NW = L::NW;
    constexpr int WL = log2c(L::WCH), XL = log2c(L::XCH);
    extern __shared__ __align__(16) unsigned char pipe_smem[];
    const unsigned nb = cols >> 5;
    // split-K: blockIdx.z of gridDim.z splits owns steps [step_lo, step_hi);
    // partial tiles are added into a zeroed C (two splits: an exact, order-
    // free sum)
    const unsigned nsteps_all = (nb + KBS - 1) / KBS;
    const unsigned splits = gridDim.z, split = blockIdx.z;
    const unsigned step_lo = (nsteps_all * split) / splits, step_hi = (nsteps_all * (split + 1)) / splits;
    const unsigned nsteps = step_hi - step_lo;
    const unsigned r0 = blockIdx.x * PM, t0 = blockIdx.y * PN;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned wr0 = (warp % L::WM) * 32, wt0 = (warp / L::WM) * L::WT;   // WM x WN warps
    const unsigned gid = lane >> 2, tig = lane & 3;
    const unsigned n_valid = t - t0;
    const unsigned wrow_bytes = Q8 ? cols : (cols >> 1);           // global row stride
    const unsigned wblk_bytes = Q8 ? 32 : 16;                        // bytes per 32-column block
    float acc[2][NW][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < NW; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    // one stage's asynchronous part: W chunks and X chunks (16-byte
    // cp.async). Every thread owns fixed chunks whose global sources only
    // advance by one stage's columns per step and whose shared
    // destinations are fixed per stage: addresses are computed once
    // (the tile shapes instantiated below keep PM * WCH and PN * XCH multiples of 256)
    constexpr int WREPS = (PM * L::WCH) / 256;
    constexpr int XREPS = (PN * L::XCH) / 256;
    static_assert(WREPS * 256 == PM * L::WCH && XREPS * 256 == PN * L::XCH, "chunk counts must divide the block");
    const u8* wsrc[WREPS];
    unsigned wdst[WREPS];
    unsigned wblk[WREPS];       // block offset of the chunk within a stage
    bool wok[WREPS];
    #pragma unroll
    for (int rep = 0; rep < WREPS; rep++) {
        unsigned i = tid + rep * 256;
        unsigned rr = i / L::WCH, cc = i % L::WCH;
        unsigned grow = r0 + rr;
        wblk[rep] = cc * 16 / wblk_bytes;
        unsigned off = (cc * 16) % wblk_bytes;
        wok[rep] = grow < rows;
        wsrc[rep] = wbytes + (size_t)(wok[rep] ? grow : 0) * wrow_bytes + off;
        wdst[rep] = swz(rr, cc, WL) * 16;
    }
    const i8* xsrc[XREPS];
    unsigned xdst[XREPS];
    unsigned xblk[XREPS];
    bool xok[XREPS];
    #pragma unroll
    for (int rep = 0; rep < XREPS; rep++) {
        unsigned i = tid + rep * 256;
        unsigned tt = i / L::XCH, cc = i % L::XCH;
        unsigned gtok = t0 + tt;
        xblk[rep] = cc >> 1;
        xok[rep] = gtok < t;
        xsrc[rep] = xq + (size_t)(xok[rep] ? gtok : 0) * cols + (cc & 1) * 16;
        xdst[rep] = swz(tt, cc, XL) * 16;
    }
    auto issue = [&](unsigned step, unsigned stage) {
        unsigned char* base = pipe_smem + stage * L::STAGE;
        unsigned g0 = (step_lo + step) * KBS;
        #pragma unroll
        for (int rep = 0; rep < WREPS; rep++) {
            unsigned g = g0 + wblk[rep];
            bool valid = wok[rep] && g < nb;
            cp_async16(base + wdst[rep], wsrc[rep] + (size_t)g * wblk_bytes, valid);
        }
        #pragma unroll
        for (int rep = 0; rep < XREPS; rep++) {
            unsigned g = g0 + xblk[rep];
            bool valid = xok[rep] && g < nb;
            cp_async16(base + L::W_BYTES + xdst[rep], xsrc[rep] + (size_t)g * 32, valid);
        }
        cp_async_commit();
    };
    // the scales are gathered per (block, row) - not contiguous in global
    // memory - so they travel through registers one step ahead: loaded
    // during a step's compute, stored into the stage before its sync
    constexpr int WSR = (KBS * PM + 255) / 256;
    constexpr int XSR = (KBS * PN + 255) / 256;
    unsigned short wsreg[WSR];
    float xsreg[XSR];
    auto load_scales = [&](unsigned step) {
        unsigned g0 = (step_lo + step) * KBS;
        #pragma unroll
        for (int rep = 0; rep < WSR; rep++) {
            unsigned i = tid + rep * 256;
            unsigned kb = i / PM, rr = i % PM;
            unsigned g = g0 + kb, grow = r0 + rr;
            unsigned short v = 0;
            if (i < KBS * PM && g < nb && grow < rows) {
                if (Q8) v = ((const unsigned short*)wsc)[(size_t)grow * nb + g];
                else v = (unsigned short)((const u8*)wsc)[(size_t)grow * nb + g];
            }
            wsreg[rep] = v;
        }
        #pragma unroll
        for (int rep = 0; rep < XSR; rep++) {
            unsigned i = tid + rep * 256;
            unsigned kb = i / PN, tt = i % PN;
            unsigned g = g0 + kb, gtok = t0 + tt;
            xsreg[rep] = (i < KBS * PN && g < nb && gtok < t) ? xs[(size_t)gtok * nb + g] : 0.0f;
        }
    };
    auto store_scales = [&](unsigned stage) {
        unsigned char* base = pipe_smem + stage * L::STAGE;
        unsigned short* sws = (unsigned short*)(base + L::W_BYTES + L::X_BYTES);
        float* sxs = (float*)(base + L::W_BYTES + L::X_BYTES + L::WS_BYTES);
        #pragma unroll
        for (int rep = 0; rep < WSR; rep++) {
            unsigned i = tid + rep * 256;
            if (i < KBS * PM) sws[i] = wsreg[rep];
        }
        #pragma unroll
        for (int rep = 0; rep < XSR; rep++) {
            unsigned i = tid + rep * 256;
            if (i < KBS * PN) sxs[i] = xsreg[rep];
        }
    };
    const int zero = 0;
    // prologue: stages 0 .. STAGES-2 in flight, the scales of step 0 in registers
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if ((unsigned)s < nsteps) issue(s, s);
        else cp_async_commit();
    }
    load_scales(0);
    for (unsigned step = 0; step < nsteps; step++) {
        cp_async_wait<STAGES - 2>();
        store_scales(step % STAGES);
        __syncthreads();
        {
            unsigned nxt = step + STAGES - 1;
            if (nxt < nsteps) issue(nxt, nxt % STAGES);
            else cp_async_commit();
        }
        if (step + 1 < nsteps) load_scales(step + 1);
        // a warp whose tokens are all valid runs the unrolled tile; a warp on
        // the prompt's tail runs only its valid n8 tiles (uniform per warp and
        // step); X rows past the prompt are zero-filled and never stored
        const unsigned nw = (wt0 >= n_valid) ? 0u : min((unsigned)NW, (n_valid - wt0 + 7) / 8);
        if (nw > 0) {
            const unsigned char* base = pipe_smem + (step % STAGES) * L::STAGE;
            if (nw == (unsigned)NW) pipe_step<Q8, PM, PN, KBS, true>(base, wr0, wt0, gid, tig, (unsigned)NW, acc, zero);
            else pipe_step<Q8, PM, PN, KBS, false>(base, wr0, wt0, gid, tig, nw, acc, zero);
        }
    }
    cp_async_wait<0>();
    // store (or add, when the K dimension is split)
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < NW; ni++) {
            unsigned r = r0 + wr0 + mi * 16 + gid;
            unsigned tt = t0 + wt0 + ni * 8 + tig * 2;
            if (OUT_HALF) {
                // f16 output (the gate|up rows: SiLU x up reads them once and requantizes)
                unsigned short* H = (unsigned short*)C;
                if (tt < t) {
                    if (r < rows) H[(size_t)tt * ldc + r] = f2h(acc[mi][ni][0]);
                    if (r + 8 < rows) H[(size_t)tt * ldc + r + 8] = f2h(acc[mi][ni][2]);
                }
                if (tt + 1 < t) {
                    if (r < rows) H[(size_t)(tt + 1) * ldc + r] = f2h(acc[mi][ni][1]);
                    if (r + 8 < rows) H[(size_t)(tt + 1) * ldc + r + 8] = f2h(acc[mi][ni][3]);
                }
            } else if (splits == 1) {
                if (tt < t) {
                    if (r < rows) C[(size_t)tt * ldc + r] = acc[mi][ni][0];
                    if (r + 8 < rows) C[(size_t)tt * ldc + r + 8] = acc[mi][ni][2];
                }
                if (tt + 1 < t) {
                    if (r < rows) C[(size_t)(tt + 1) * ldc + r] = acc[mi][ni][1];
                    if (r + 8 < rows) C[(size_t)(tt + 1) * ldc + r + 8] = acc[mi][ni][3];
                }
            } else {
                if (tt < t) {
                    if (r < rows) atomicAdd(C + (size_t)tt * ldc + r, acc[mi][ni][0]);
                    if (r + 8 < rows) atomicAdd(C + (size_t)tt * ldc + r + 8, acc[mi][ni][2]);
                }
                if (tt + 1 < t) {
                    if (r < rows) atomicAdd(C + (size_t)(tt + 1) * ldc + r, acc[mi][ni][1]);
                    if (r + 8 < rows) atomicAdd(C + (size_t)(tt + 1) * ldc + r + 8, acc[mi][ni][3]);
                }
            }
        }
    }
}
// 128 x 128, two blocks per stage, three stages (q8 54 KB, fp4 42 KB of shared memory)
extern "C" __global__ void __launch_bounds__(256, 1) k_gemm_q8_pipe(const i8* __restrict__ wq, const unsigned short* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    gemm_pipe_t<true, 128, 128, 2, PIPE_STAGES, false>((const u8*)wq, (const void*)ws, xq, xs, C, rows, cols, t, ldc);
}
extern "C" __global__ void __launch_bounds__(256, 1) k_gemm_fp4_pipe(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    gemm_pipe_t<false, 128, 128, 2, PIPE_STAGES, false>(wp, (const void*)wsc, xq, xs, C, rows, cols, t, ldc);
}
// f16 output form (the MLP's gate|up projection)
extern "C" __global__ void __launch_bounds__(256, 1) k_gemm_fp4_pipe_h(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    gemm_pipe_t<false, 128, 128, 2, PIPE_STAGES, true>(wp, (const void*)wsc, xq, xs, C, rows, cols, t, ldc);
}

// ── elementwise ──
extern "C" __global__ void k_add(float* __restrict__ x, const float* __restrict__ y, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
extern "C" __global__ void k_scale_rows(float* __restrict__ x, unsigned n, float s) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}
// out[r] = rmsnorm(x[r] (+= add[r])) * (one_plus ? 1+w : w); block per row, blockDim 1024
extern "C" __global__ void k_add_rmsnorm_rows(float* __restrict__ x, const float* __restrict__ add, const float* __restrict__ w,
                                              float* __restrict__ out, unsigned rows, unsigned n, float eps, unsigned one_plus) {
    __shared__ float red[32];
    unsigned r = blockIdx.x;
    float* xr = x + (size_t)r * n;
    float* orow = out + (size_t)r * n;
    float ss = 0.0f;
    if (add) {
        const float* ar = add + (size_t)r * n;
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i] + ar[i]; xr[i] = v; ss += v * v; }
    } else {
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    }
    ss = block_sum(ss, red);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float ww = one_plus ? (1.0f + w[i]) : w[i];
        orow[i] = xr[i] * inv * ww;
    }
}
// out = rmsnorm(x (+= add)) * (1+w | w), then block-quantized straight into xq/xs
// (rows of n; one block of 1024 per row; n multiple of 32)
extern "C" __global__ void k_add_rmsnorm_quant_rows(float* __restrict__ x, const float* __restrict__ add, const float* __restrict__ w,
                                                    float* __restrict__ out, i8* __restrict__ xq, float* __restrict__ xs,
                                                    unsigned rows, unsigned n, float eps, unsigned one_plus) {
    __shared__ float red[32];
    unsigned r = blockIdx.x;
    float* xr = x + (size_t)r * n;
    float* orow = out ? out + (size_t)r * n : (float*)0;
    float ss = 0.0f;
    if (add) {
        const float* ar = add + (size_t)r * n;
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i] + ar[i]; xr[i] = v; ss += v * v; }
    } else {
        for (unsigned i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    }
    ss = block_sum(ss, red);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    // normalize and quantize in one pass: warp w handles blocks w, w+warps, ...
    // (32 lanes = the 32 values of a block; the max by shuffle; the values
    // stay in registers between the norm and the quantization)
    unsigned nb = n >> 5;
    unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (unsigned b = warp; b < nb; b += warps) {
        unsigned i = b * 32 + lane;
        float ww = one_plus ? (1.0f + w[i]) : w[i];
        float v = xr[i] * inv * ww;
        if (out) orow[i] = v;   // the f32 form is optional (nothing downstream reads it in the graphs)
        float m = warp_max(fabsf(v));
        float dx = m / 127.0f;
        i8 qv = 0;
        if (dx != 0.0f) {
            float rr = roundf(v / dx);
            rr = fminf(fmaxf(rr, -127.0f), 127.0f);
            qv = (i8)(int)rr;
        }
        xq[((size_t)r * nb + b) * 32 + lane] = qv;
        if (lane == 0) xs[(size_t)r * nb + b] = dx;
    }
}
// h[t][inter] = silu(gu[t][i]) * gu[t][inter+i], quantized into xq/xs (block per (row, 1024 columns))
// the same from an f16 gate|up (the pipelined GEMM's f16 output form)
extern "C" __global__ void k_silu_mul_quant_rows_h(const unsigned short* __restrict__ gu, float* __restrict__ h, i8* __restrict__ xq, float* __restrict__ xs,
                                                   unsigned rows, unsigned inter) {
    unsigned r = blockIdx.y;
    unsigned c0 = blockIdx.x * 1024;
    unsigned i = c0 + threadIdx.x;
    const unsigned short* g = gu + (size_t)r * 2 * inter;
    (void)h;
    float v = 0.0f;
    if (i < inter) v = siluf_(h2f(g[i])) * h2f(g[inter + i]);
    unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned b = (c0 >> 5) + warp;
    if (b < (inter >> 5)) {
        float m = warp_max(fabsf(v));
        float dx = m / 127.0f;
        i8 qv = 0;
        if (dx != 0.0f) {
            float rr = roundf(v / dx);
            rr = fminf(fmaxf(rr, -127.0f), 127.0f);
            qv = (i8)(int)rr;
        }
        xq[((size_t)r * (inter >> 5) + b) * 32 + lane] = qv;
        if (lane == 0) xs[(size_t)r * (inter >> 5) + b] = dx;
    }
}
extern "C" __global__ void k_silu_mul_quant_rows(const float* __restrict__ gu, float* __restrict__ h, i8* __restrict__ xq, float* __restrict__ xs,
                                                 unsigned rows, unsigned inter) {
    // grid: (inter/1024 ceil, rows); block 1024 -> 32 blocks of 32 per CTA
    unsigned r = blockIdx.y;
    unsigned c0 = blockIdx.x * 1024;
    unsigned i = c0 + threadIdx.x;
    const float* g = gu + (size_t)r * 2 * inter;
    (void)h; // only the quantized form is consumed downstream
    float v = 0.0f;
    if (i < inter) v = siluf_(g[i]) * g[inter + i];
    // warp w quantizes block w (32 values) via a warp max
    unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    unsigned b = (c0 >> 5) + warp;
    if (b < (inter >> 5)) {
        float m = warp_max(fabsf(v));
        float dx = m / 127.0f;
        i8 qv = 0;
        if (dx != 0.0f) {
            float rr = roundf(v / dx);
            rr = fminf(fmaxf(rr, -127.0f), 127.0f);
            qv = (i8)(int)rr;
        }
        xq[((size_t)r * (inter >> 5) + b) * 32 + lane] = qv;
        if (lane == 0) xs[(size_t)r * (inter >> 5) + b] = dx;
    }
}
// single-vector convenience forms
extern "C" __global__ void k_rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, unsigned n, float eps, unsigned one_plus) {
    __shared__ float red[32];
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) { float v = x[i]; ss += v * v; }
    ss = block_sum(ss, red);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * inv * (one_plus ? (1.0f + w[i]) : w[i]);
}
// rows of n (n <= 1024): warp/block per row through blockDim threads
extern "C" __global__ void k_rmsnorm_rows(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, unsigned rows, unsigned n, float eps, unsigned one_plus) {
    __shared__ float red[32];
    unsigned r = blockIdx.x;
    if (r >= rows) return;
    const float* xr = x + (size_t)r * n;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) { float v = xr[i]; ss += v * v; }
    ss = block_sum(ss, red);
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) out[(size_t)r * n + i] = xr[i] * inv * (one_plus ? (1.0f + w[i]) : w[i]);
}
extern "C" __global__ void k_silu_mul(const float* __restrict__ gu, float* __restrict__ h, unsigned inter) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < inter) h[i] = siluf_(gu[i]) * gu[inter + i];
}
extern "C" __global__ void k_silu_mul_rows(const float* __restrict__ gu, float* __restrict__ h, unsigned rows, unsigned inter) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * inter) return;
    unsigned r = i / inter, c = i - r * inter;
    const float* g = gu + (size_t)r * 2 * inter;
    h[i] = siluf_(g[c]) * g[inter + c];
}

// ── linear attention: conv + SiLU (one token), state [conv_dim][k-1] ──
extern "C" __global__ void k_conv_silu(const float* __restrict__ x, const float* __restrict__ w, unsigned k,
                                       float* __restrict__ state, float* __restrict__ out, unsigned conv_dim) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= conv_dim) return;
    float* st = state + (size_t)i * (k - 1);
    const float* wt = w + (size_t)i * k;
    float acc = 0.0f;
    for (unsigned j = 0; j < k - 1; j++) acc += st[j] * wt[j];
    acc += x[i] * wt[k - 1];
    out[i] = acc / (1.0f + expf(-acc));
    for (unsigned j = 0; j + 2 < k; j++) st[j] = st[j + 1];
    st[k - 2] = x[i];
}
// prefill: t tokens (x [t][conv_dim] -> out [t][conv_dim]), thread per channel, sequential in t
extern "C" __global__ void k_conv_prefill(const float* __restrict__ x, unsigned ldx, const float* __restrict__ w, unsigned k,
                                          float* __restrict__ state, float* __restrict__ out, unsigned ldo, unsigned conv_dim, unsigned t) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= conv_dim) return;
    float* st = state + (size_t)i * (k - 1);
    const float* wt = w + (size_t)i * k;
    // k <= 8 taps kept in registers; four tokens per iteration with their
    // loads issued together (one dependent load per token was latency-bound)
    float s[8];
    for (unsigned j = 0; j < k - 1; j++) s[j] = st[j];
    float wk[8];
    for (unsigned j = 0; j < 8; j++) wk[j] = (j < k) ? wt[j] : 0.0f;
    unsigned tok = 0;
    for (; tok + 4 <= t; tok += 4) {
        float x0 = x[(size_t)tok * ldx + i], x1 = x[(size_t)(tok + 1) * ldx + i], x2 = x[(size_t)(tok + 2) * ldx + i], x3 = x[(size_t)(tok + 3) * ldx + i];
        float xs4[4] = {x0, x1, x2, x3};
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            float acc = 0.0f;
            for (unsigned j = 0; j < k - 1; j++) acc += s[j] * wk[j];
            acc += xs4[u] * wk[k - 1];
            out[(size_t)(tok + u) * ldo + i] = acc / (1.0f + expf(-acc));
            for (unsigned j = 0; j + 2 < k; j++) s[j] = s[j + 1];
            s[k - 2] = xs4[u];
        }
    }
    for (; tok < t; tok++) {
        float xi = x[(size_t)tok * ldx + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < k - 1; j++) acc += s[j] * wk[j];
        acc += xi * wk[k - 1];
        out[(size_t)tok * ldo + i] = acc / (1.0f + expf(-acc));
        for (unsigned j = 0; j + 2 < k; j++) s[j] = s[j + 1];
        s[k - 2] = xi;
    }
    for (unsigned j = 0; j < k - 1; j++) st[j] = s[j];
}

// per (token, head): q = l2norm(conved[kh*kd..]) / sqrt(kd), k = l2norm(...), beta, decay
// conved [t][conv_dim]; b_raw/a_raw [t][heads]; qn/kn [t][heads][kd]; beta/decay [t][heads]
extern "C" __global__ void k_lin_prep(const float* __restrict__ conved, unsigned ldc, const float* __restrict__ b_raw, const float* __restrict__ a_raw, unsigned ldba,
                                      const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                      float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ beta, float* __restrict__ decay,
                                      unsigned t, unsigned heads, unsigned kv_heads, unsigned kd) {
    __shared__ float red[32];
    unsigned tok = blockIdx.x, h = blockIdx.y;
    if (tok >= t || h >= heads) return;
    unsigned rep = heads / kv_heads;
    unsigned kh = h / rep;
    unsigned kt = kv_heads * kd;
    const float* row = conved + (size_t)tok * ldc;
    const float* qsrc = row + kh * kd;
    const float* ksrc = row + kt + kh * kd;
    float sq = 0.0f, sk = 0.0f;
    for (unsigned i = threadIdx.x; i < kd; i += blockDim.x) { sq += qsrc[i] * qsrc[i]; sk += ksrc[i] * ksrc[i]; }
    sq = block_sum(sq, red);
    sk = block_sum(sk, red);
    float nq = sqrtf(sq + 1e-6f), nk = sqrtf(sk + 1e-6f);
    float scale = 1.0f / sqrtf((float)kd);
    float* qd = qn + ((size_t)tok * heads + h) * kd;
    float* kdst = kn + ((size_t)tok * heads + h) * kd;
    for (unsigned i = threadIdx.x; i < kd; i += blockDim.x) { qd[i] = (qsrc[i] / nq) * scale; kdst[i] = ksrc[i] / nk; }
    if (threadIdx.x == 0) {
        float b = b_raw[(size_t)tok * ldba + h];
        beta[(size_t)tok * heads + h] = 1.0f / (1.0f + expf(-b));
        float a = a_raw[(size_t)tok * ldba + h] + dt_bias[h];
        float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
        decay[(size_t)tok * heads + h] = expf(-expf(a_log[h]) * sp);
    }
}

// delta step over t tokens per head (block per head, thread per value column j):
//   S *= decay; pred = S^T k; S += k (x) (v - pred) beta; out = S^T q
// state [heads][kd][vd]; qn/kn [t][heads][kd]; v from conved [t][conv_dim] at 2*kt + h*vd;
// out [t][heads*vd] (token-major, gated-norm input)
extern "C" __global__ void k_delta_scan(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                        const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                        float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned kd, unsigned vd) {
    extern __shared__ float sh[]; // q[kd], k[kd]
    float* sq = sh;
    float* sk = sh + kd;
    unsigned h = blockIdx.x;
    unsigned j = threadIdx.x;
    if (h >= heads || j >= vd) return;
    unsigned kt = kv_heads * kd;
    float* S = state + (size_t)h * kd * vd;
    for (unsigned tok = 0; tok < t; tok++) {
        __syncthreads();
        for (unsigned i = threadIdx.x; i < kd; i += blockDim.x) {
            sq[i] = qn[((size_t)tok * heads + h) * kd + i];
            sk[i] = kn[((size_t)tok * heads + h) * kd + i];
        }
        __syncthreads();
        float dec = decay[(size_t)tok * heads + h];
        float bt = beta[(size_t)tok * heads + h];
        float vj = conved[(size_t)tok * ldc + 2 * kt + h * vd + j];
        float pred = 0.0f;
        for (unsigned i = 0; i < kd; i++) {
            float s = S[(size_t)i * vd + j] * dec;
            S[(size_t)i * vd + j] = s;
            pred = fmaf(sk[i], s, pred);
        }
        float delta = (vj - pred) * bt;
        float o = 0.0f;
        for (unsigned i = 0; i < kd; i++) {
            float s = S[(size_t)i * vd + j] + sk[i] * delta;
            S[(size_t)i * vd + j] = s;
            o = fmaf(sq[i], s, o);
        }
        out[((size_t)tok * heads + h) * vd + j] = o;
    }
}
// register-resident form: thread j keeps its state column S[0..KD][j] in
// registers for the whole prompt (the generic form's per-step walks over
// the state in L2 were latency-bound: 24 us per (layer, token)); same
// arithmetic per element, same order.
template <int KD>
__device__ __forceinline__ void delta_scan_reg(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                               const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                               float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned vd) {
    __shared__ float sq[KD];
    __shared__ float sk[KD];
    unsigned h = blockIdx.x;
    unsigned j = threadIdx.x;
    if (h >= heads || j >= vd) return;
    unsigned kt = kv_heads * KD;
    float* S = state + (size_t)h * KD * vd;
    float col[KD];
    #pragma unroll
    for (int i = 0; i < KD; i++) col[i] = S[(size_t)i * vd + j];
    for (unsigned tok = 0; tok < t; tok++) {
        __syncthreads();
        for (unsigned i = threadIdx.x; i < KD; i += blockDim.x) {
            sq[i] = qn[((size_t)tok * heads + h) * KD + i];
            sk[i] = kn[((size_t)tok * heads + h) * KD + i];
        }
        __syncthreads();
        float dec = decay[(size_t)tok * heads + h];
        float bt = beta[(size_t)tok * heads + h];
        float vj = conved[(size_t)tok * ldc + 2 * kt + h * vd + j];
        float pred = 0.0f;
        #pragma unroll
        for (int i = 0; i < KD; i++) {
            float s = col[i] * dec;
            col[i] = s;
            pred = fmaf(sk[i], s, pred);
        }
        float delta = (vj - pred) * bt;
        float o = 0.0f;
        #pragma unroll
        for (int i = 0; i < KD; i++) {
            float s = col[i] + sk[i] * delta;
            col[i] = s;
            o = fmaf(sq[i], s, o);
        }
        out[((size_t)tok * heads + h) * vd + j] = o;
    }
    #pragma unroll
    for (int i = 0; i < KD; i++) S[(size_t)i * vd + j] = col[i];
}
// four thread groups per head (block = 4 * vd threads): group g keeps
// rows [g*KD/4, (g+1)*KD/4) of its column in registers; the two column
// reductions of every token combine the groups through shared memory
// (four times the memory-level parallelism, chains a quarter as long)
template <int KD>
__device__ __forceinline__ void delta_scan_reg4(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                                const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                                float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned vd) {
    constexpr int G = 4;
    constexpr int KR = KD / G;
    __shared__ float sq[KD];
    __shared__ float sk[KD];
    __shared__ float part[G * 256];
    unsigned h = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned grp = tid / vd, j = tid % vd;
    if (h >= heads) return;
    unsigned kt = kv_heads * KD;
    float* S = state + (size_t)h * KD * vd;
    float col[KR];
    #pragma unroll
    for (int i = 0; i < KR; i++) col[i] = S[(size_t)(grp * KR + i) * vd + j];
    // the next token's q, k, beta, decay and v ride in registers, loaded
    // during the current token's work (their global latency off the chain)
    float nq = 0.0f, nk = 0.0f;
    if (tid < KD) { nq = qn[(size_t)h * KD + tid]; nk = kn[(size_t)h * KD + tid]; }
    float ndec = decay[h], nbt = beta[h];
    float nv = (j < vd) ? conved[2 * kt + h * vd + j] : 0.0f;
    for (unsigned tok = 0; tok < t; tok++) {
        __syncthreads();
        if (tid < KD) { sq[tid] = nq; sk[tid] = nk; }
        float dec = ndec, bt = nbt, vj = nv;
        __syncthreads();
        if (tok + 1 < t) {
            if (tid < KD) { nq = qn[((size_t)(tok + 1) * heads + h) * KD + tid]; nk = kn[((size_t)(tok + 1) * heads + h) * KD + tid]; }
            ndec = decay[(size_t)(tok + 1) * heads + h];
            nbt = beta[(size_t)(tok + 1) * heads + h];
            nv = conved[(size_t)(tok + 1) * ldc + 2 * kt + h * vd + j];
        }
        float pred = 0.0f;
        #pragma unroll
        for (int i = 0; i < KR; i++) { float s = col[i] * dec; col[i] = s; pred = fmaf(sk[grp * KR + i], s, pred); }
        part[grp * vd + j] = pred;
        __syncthreads();
        pred = ((part[j] + part[vd + j]) + part[2 * vd + j]) + part[3 * vd + j];
        float delta = (vj - pred) * bt;
        float o = 0.0f;
        #pragma unroll
        for (int i = 0; i < KR; i++) { float s = col[i] + sk[grp * KR + i] * delta; col[i] = s; o = fmaf(sq[grp * KR + i], s, o); }
        __syncthreads();
        part[grp * vd + j] = o;
        __syncthreads();
        if (grp == 0) out[((size_t)tok * heads + h) * vd + j] = ((part[j] + part[vd + j]) + part[2 * vd + j]) + part[3 * vd + j];
    }
    #pragma unroll
    for (int i = 0; i < KR; i++) S[(size_t)(grp * KR + i) * vd + j] = col[i];
}
extern "C" __global__ void __launch_bounds__(1024) k_delta_scan4x128(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                             const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                             float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned vd) {
    delta_scan_reg4<128>(state, qn, kn, conved, ldc, beta, decay, out, t, heads, kv_heads, vd);
}
extern "C" __global__ void __launch_bounds__(1024) k_delta_scan4x64(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                            const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                            float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned vd) {
    delta_scan_reg4<64>(state, qn, kn, conved, ldc, beta, decay, out, t, heads, kv_heads, vd);
}
extern "C" __global__ void k_delta_scan128(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                           const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                           float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned vd) {
    delta_scan_reg<128>(state, qn, kn, conved, ldc, beta, decay, out, t, heads, kv_heads, vd);
}
extern "C" __global__ void k_delta_scan64(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                          const float* __restrict__ conved, unsigned ldc, const float* __restrict__ beta, const float* __restrict__ decay,
                                          float* __restrict__ out, unsigned t, unsigned heads, unsigned kv_heads, unsigned vd) {
    delta_scan_reg<64>(state, qn, kn, conved, ldc, beta, decay, out, t, heads, kv_heads, vd);
}
// the whole per-head work of a linear layer's decode step in one launch
// (KD = 128 / 64, block = vd threads, vd a multiple of 32): q/k norms of
// the head's kv head, beta and decay, the delta step with the state
// column in registers, the gated norm with silu(z), then the head's vd
// outputs quantized (one warp per block of 32) into xq/xs for the out
// projection. Same arithmetic as the separate kernels.
template <int KD>
__device__ __forceinline__ void lin_head_step(const float* __restrict__ conved, const float* __restrict__ b_raw, const float* __restrict__ a_raw,
                                              const float* __restrict__ z, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                              float* __restrict__ state, const float* __restrict__ gated_w,
                                              float* __restrict__ mixed, i8* __restrict__ xq, float* __restrict__ xs,
                                              unsigned heads, unsigned kv_heads, unsigned vd, float eps) {
    // block = 4 * vd threads: group g (0..3) owns rows [g*KD/4, (g+1)*KD/4)
    // of column j = threadIdx.x % vd; the two column reductions (pred, out)
    // combine the four groups through shared memory - four times the
    // memory-level parallelism of one thread per column
    constexpr int G = 4;
    constexpr int KR = KD / G;
    __shared__ float sq[KD];
    __shared__ float sk[KD];
    __shared__ float red[32];
    __shared__ float part[G * 1024 / 4]; // [G][vd] partials (vd <= 256)
    __shared__ float sh_beta, sh_dec;
    unsigned h = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned grp = tid / vd, j = tid % vd;
    if (h >= heads) return;
    unsigned rep = heads / kv_heads;
    unsigned kh = h / rep;
    unsigned kt = kv_heads * KD;
    const float* qsrc = conved + kh * KD;
    const float* ksrc = conved + kt + kh * KD;
    float sqv = 0.0f, skv = 0.0f;
    for (unsigned i = tid; i < KD; i += blockDim.x) { sqv += qsrc[i] * qsrc[i]; skv += ksrc[i] * ksrc[i]; }
    sqv = block_sum(sqv, red);
    skv = block_sum(skv, red);
    float nq = sqrtf(sqv + 1e-6f), nk = sqrtf(skv + 1e-6f);
    float scale = 1.0f / sqrtf((float)KD);
    for (unsigned i = tid; i < KD; i += blockDim.x) { sq[i] = (qsrc[i] / nq) * scale; sk[i] = ksrc[i] / nk; }
    if (tid == 0) {
        float b = b_raw[h];
        sh_beta = 1.0f / (1.0f + expf(-b));
        float a = a_raw[h] + dt_bias[h];
        float sp = (a > 20.0f) ? a : logf(1.0f + expf(a));
        sh_dec = expf(-expf(a_log[h]) * sp);
    }
    __syncthreads();
    float* S = state + (size_t)h * KD * vd;
    float col[KR];
    #pragma unroll
    for (int i = 0; i < KR; i++) col[i] = S[(size_t)(grp * KR + i) * vd + j];
    float dec = sh_dec, bt = sh_beta;
    float pred = 0.0f;
    #pragma unroll
    for (int i = 0; i < KR; i++) { float s = col[i] * dec; col[i] = s; pred = fmaf(sk[grp * KR + i], s, pred); }
    part[grp * vd + j] = pred;
    __syncthreads();
    // pred over the four groups in group order (the same sum for every group)
    pred = ((part[j] + part[vd + j]) + part[2 * vd + j]) + part[3 * vd + j];
    float vj = conved[2 * kt + h * vd + j];
    float delta = (vj - pred) * bt;
    float o = 0.0f;
    #pragma unroll
    for (int i = 0; i < KR; i++) { float s = col[i] + sk[grp * KR + i] * delta; col[i] = s; o = fmaf(sq[grp * KR + i], s, o); }
    #pragma unroll
    for (int i = 0; i < KR; i++) S[(size_t)(grp * KR + i) * vd + j] = col[i];
    __syncthreads();
    part[grp * vd + j] = o;
    __syncthreads();
    o = ((part[j] + part[vd + j]) + part[2 * vd + j]) + part[3 * vd + j];
    // gated norm over the head's vd outputs (group 0 finishes; the block
    // reduction takes every thread)
    float ss = block_sum((grp == 0) ? o * o : 0.0f, red);
    float inv = 1.0f / sqrtf(ss / (float)vd + eps);
    if (grp == 0) {
        float g = z[h * vd + j];
        float outv = o * inv * gated_w[j] * (g / (1.0f + expf(-g)));
        mixed[(size_t)h * vd + j] = outv;
        unsigned warp = j >> 5, lane = j & 31;
        float m = warp_max(fabsf(outv));
        float dx = m / 127.0f;
        i8 qv = 0;
        if (dx != 0.0f) {
            float rr = roundf(outv / dx);
            rr = fminf(fmaxf(rr, -127.0f), 127.0f);
            qv = (i8)(int)rr;
        }
        unsigned blk = (h * vd) / 32 + warp;
        xq[(size_t)blk * 32 + lane] = qv;
        if (lane == 0) xs[blk] = dx;
    }
}
extern "C" __global__ void __launch_bounds__(512) k_lin_head_step128(const float* __restrict__ conved, const float* __restrict__ b_raw, const float* __restrict__ a_raw,
                                              const float* __restrict__ z, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                              float* __restrict__ state, const float* __restrict__ gated_w,
                                              float* __restrict__ mixed, i8* __restrict__ xq, float* __restrict__ xs,
                                              unsigned heads, unsigned kv_heads, unsigned vd, float eps) {
    lin_head_step<128>(conved, b_raw, a_raw, z, a_log, dt_bias, state, gated_w, mixed, xq, xs, heads, kv_heads, vd, eps);
}
extern "C" __global__ void __launch_bounds__(512) k_lin_head_step64(const float* __restrict__ conved, const float* __restrict__ b_raw, const float* __restrict__ a_raw,
                                             const float* __restrict__ z, const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                                             float* __restrict__ state, const float* __restrict__ gated_w,
                                             float* __restrict__ mixed, i8* __restrict__ xq, float* __restrict__ xs,
                                             unsigned heads, unsigned kv_heads, unsigned vd, float eps) {
    lin_head_step<64>(conved, b_raw, a_raw, z, a_log, dt_bias, state, gated_w, mixed, xq, xs, heads, kv_heads, vd, eps);
}
// single-token form (same math), out [heads*vd]
extern "C" __global__ void k_delta_step(float* __restrict__ state, const float* __restrict__ qn, const float* __restrict__ kn,
                                        const float* __restrict__ conved, const float* __restrict__ beta, const float* __restrict__ decay,
                                        float* __restrict__ out, unsigned heads, unsigned kv_heads, unsigned kd, unsigned vd, unsigned conv_dim) {
    extern __shared__ float sh[];
    float* sq = sh;
    float* sk = sh + kd;
    unsigned h = blockIdx.x;
    unsigned j = threadIdx.x;
    if (h >= heads || j >= vd) return;
    unsigned kt = kv_heads * kd;
    float* S = state + (size_t)h * kd * vd;
    for (unsigned i = threadIdx.x; i < kd; i += blockDim.x) { sq[i] = qn[(size_t)h * kd + i]; sk[i] = kn[(size_t)h * kd + i]; }
    __syncthreads();
    float dec = decay[h], bt = beta[h];
    float vj = conved[2 * kt + h * vd + j];
    float pred = 0.0f;
    for (unsigned i = 0; i < kd; i++) {
        float s = S[(size_t)i * vd + j] * dec;
        S[(size_t)i * vd + j] = s;
        pred = fmaf(sk[i], s, pred);
    }
    float delta = (vj - pred) * bt;
    float o = 0.0f;
    for (unsigned i = 0; i < kd; i++) {
        float s = S[(size_t)i * vd + j] + sk[i] * delta;
        S[(size_t)i * vd + j] = s;
        o = fmaf(sq[i], s, o);
    }
    out[(size_t)h * vd + j] = o;
}
// gated rms norm per (token, head): x[t][heads*vd] normalized per head, * w[vd] * silu(z[t][heads*vd])
extern "C" __global__ void k_gated_norm_rows(float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ z, unsigned ldz,
                                             unsigned t, unsigned heads, unsigned vd, float eps) {
    __shared__ float red[32];
    unsigned tok = blockIdx.x, h = blockIdx.y;
    if (tok >= t || h >= heads) return;
    float* xr = x + ((size_t)tok * heads + h) * vd;
    const float* zr = z + (size_t)tok * ldz + h * vd;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < vd; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    float inv = 1.0f / sqrtf(ss / (float)vd + eps);
    for (unsigned i = threadIdx.x; i < vd; i += blockDim.x) {
        float g = zr[i];
        xr[i] = xr[i] * inv * w[i] * (g / (1.0f + expf(-g)));
    }
}
// rows form + quantization: x[t][heads*vd] normalized per head and gated,
// then each head's vd outputs quantized (warp per block of 32) into xq/xs
// laid out as [t][heads*vd/32] - the out projection's input
extern "C" __global__ void k_gated_norm_quant_rows(float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ z, unsigned ldz,
                                                   i8* __restrict__ xq, float* __restrict__ xs,
                                                   unsigned t, unsigned heads, unsigned vd, float eps) {
    __shared__ float red[32];
    unsigned tok = blockIdx.x, h = blockIdx.y;
    if (tok >= t || h >= heads) return;
    float* xr = x + ((size_t)tok * heads + h) * vd;
    const float* zr = z + (size_t)tok * ldz + h * vd;
    unsigned i = threadIdx.x;
    float v = (i < vd) ? xr[i] : 0.0f;
    float ss = block_sum(v * v, red);
    float inv = 1.0f / sqrtf(ss / (float)vd + eps);
    if (i < vd) {
        float g = zr[i];
        float o = v * inv * w[i] * (g / (1.0f + expf(-g)));
        xr[i] = o;
        unsigned warp = i >> 5, lane = i & 31;
        float m = warp_max(fabsf(o));
        float dx = m / 127.0f;
        i8 qv = 0;
        if (dx != 0.0f) {
            float rr = roundf(o / dx);
            rr = fminf(fmaxf(rr, -127.0f), 127.0f);
            qv = (i8)(int)rr;
        }
        unsigned nbrow = heads * vd / 32;
        unsigned blk = (h * vd) / 32 + warp;
        xq[((size_t)tok * nbrow + blk) * 32 + lane] = qv;
        if (lane == 0) xs[(size_t)tok * nbrow + blk] = dx;
    }
}
extern "C" __global__ void k_gated_norm(float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ z, unsigned heads, unsigned vd, float eps) {
    __shared__ float red[32];
    unsigned h = blockIdx.x;
    if (h >= heads) return;
    float* xr = x + (size_t)h * vd;
    const float* zr = z + (size_t)h * vd;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < vd; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss, red);
    float inv = 1.0f / sqrtf(ss / (float)vd + eps);
    for (unsigned i = threadIdx.x; i < vd; i += blockDim.x) {
        float g = zr[i];
        xr[i] = xr[i] * inv * w[i] * (g / (1.0f + expf(-g)));
    }
}

// ── full attention ──
// qkv rows of one token: [q|gate interleaved per head: (q hd, gate hd) x n_heads | k kvw | v kvw]
// per token: q_norm(1+w) + partial rope on each q head; k_norm + rope on each k head; write q [n_heads*hd]
// (post), gate [n_heads*hd] (raw), K/V rows into the cache at position pos.
// grid: (t, n_heads + n_kv), block: hd threads (hd <= 1024)
extern "C" __global__ void k_qk_prep_rows(const float* __restrict__ qkv, unsigned ld_qkv, const float* __restrict__ qw, const float* __restrict__ kw,
                                          float* __restrict__ q_out, float* __restrict__ gate_out,
                                          float* __restrict__ kc, float* __restrict__ vc, unsigned kv_width,
                                          unsigned t, unsigned pos0, unsigned n_heads, unsigned n_kv, unsigned hd, unsigned rope_dim, const float* __restrict__ rope, float eps) {
    __shared__ float red[32];
    unsigned tok = blockIdx.x, hh = blockIdx.y;
    unsigned i = threadIdx.x;
    if (tok >= t || i >= hd) return;
    const float* row = qkv + (size_t)tok * ld_qkv;
    unsigned pos = pos0 + tok;
    unsigned qw_ = n_heads * hd;
    if (hh < n_heads) {
        unsigned h = hh;
        float v = row[h * hd * 2 + i];
        float g = row[h * hd * 2 + hd + i];
        float ss = block_sum(v * v, red);
        float inv = 1.0f / sqrtf(ss / (float)hd + eps);
        float qn = v * inv * (1.0f + qw[i]);
        // rope on the first rope_dim channels: pairs (i, i+half)
        unsigned half = rope_dim / 2;
        __shared__ float sq[1024];
        sq[i] = qn;
        __syncthreads();
        float outv = qn;
        const float* rp = rope + (size_t)pos * rope_dim;   // [cos(half) | sin(half)] of this position
        if (i < half) {
            float s = rp[half + i], c = rp[i];
            float a = sq[i], b = sq[i + half];
            outv = a * c - b * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            float s = rp[half + i0], c = rp[i0];
            float a = sq[i0], b = sq[i];
            outv = a * s + b * c;
        }
        q_out[(size_t)tok * qw_ + h * hd + i] = outv;
        gate_out[(size_t)tok * qw_ + h * hd + i] = g;
    } else {
        unsigned h = hh - n_heads;
        float v = row[2 * qw_ + h * hd + i];
        float vv = row[2 * qw_ + kv_width + h * hd + i];
        float ss = block_sum(v * v, red);
        float inv = 1.0f / sqrtf(ss / (float)hd + eps);
        float kn = v * inv * (1.0f + kw[i]);
        unsigned half = rope_dim / 2;
        __shared__ float sk[1024];
        sk[i] = kn;
        __syncthreads();
        float outv = kn;
        const float* rp = rope + (size_t)pos * rope_dim;
        if (i < half) {
            float s = rp[half + i], c = rp[i];
            float a = sk[i], b = sk[i + half];
            outv = a * c - b * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            float s = rp[half + i0], c = rp[i0];
            float a = sk[i0], b = sk[i];
            outv = a * s + b * c;
        }
        kc[(size_t)pos * kv_width + h * hd + i] = outv;
        vc[(size_t)pos * kv_width + h * hd + i] = vv;
    }
}
extern "C" __global__ void k_qk_prep(const float* __restrict__ qkv, const float* __restrict__ qw, const float* __restrict__ kw,
                                     float* __restrict__ q_out, float* __restrict__ gate_out,
                                     float* __restrict__ kc, float* __restrict__ vc, unsigned kv_width,
                                     const unsigned* __restrict__ posp, unsigned n_heads, unsigned n_kv, unsigned hd, unsigned rope_dim, const float* __restrict__ rope, float eps) {
    // single-token form; the position comes from device memory so that the
    // decode graph is identical from one token to the next
    __shared__ float red[32];
    unsigned hh = blockIdx.y;
    unsigned i = threadIdx.x;
    if (i >= hd) return;
    unsigned pos = *posp;
    const float* row = qkv;
    unsigned qw_ = n_heads * hd;
    if (hh < n_heads) {
        unsigned h = hh;
        float v = row[h * hd * 2 + i];
        float g = row[h * hd * 2 + hd + i];
        float ss = block_sum(v * v, red);
        float inv = 1.0f / sqrtf(ss / (float)hd + eps);
        float qn = v * inv * (1.0f + qw[i]);
        unsigned half = rope_dim / 2;
        __shared__ float sq[1024];
        sq[i] = qn;
        __syncthreads();
        float outv = qn;
        const float* rp = rope + (size_t)pos * rope_dim;
        if (i < half) {
            float s = rp[half + i], c = rp[i];
            outv = sq[i] * c - sq[i + half] * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            float s = rp[half + i0], c = rp[i0];
            outv = sq[i0] * s + sq[i] * c;
        }
        q_out[h * hd + i] = outv;
        gate_out[h * hd + i] = g;
    } else {
        unsigned h = hh - n_heads;
        float v = row[2 * qw_ + h * hd + i];
        float vv = row[2 * qw_ + kv_width + h * hd + i];
        float ss = block_sum(v * v, red);
        float inv = 1.0f / sqrtf(ss / (float)hd + eps);
        float kn = v * inv * (1.0f + kw[i]);
        unsigned half = rope_dim / 2;
        __shared__ float sk[1024];
        sk[i] = kn;
        __syncthreads();
        float outv = kn;
        const float* rp = rope + (size_t)pos * rope_dim;
        if (i < half) {
            float s = rp[half + i], c = rp[i];
            outv = sk[i] * c - sk[i + half] * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            float s = rp[half + i0], c = rp[i0];
            outv = sk[i0] * s + sk[i] * c;
        }
        kc[(size_t)pos * kv_width + h * hd + i] = outv;
        vc[(size_t)pos * kv_width + h * hd + i] = vv;
    }
}
// decode attention: block per q head, hd threads (hd <= 1024, multiple of 32); dynamic shared: (len) floats
// scores over positions [0, len): warp per position (lanes stride hd), then softmax, then V mix
// mixed[h*hd + i] = (sum_t p_t V[t][kh*hd + i]) * sigmoid(gate[h*hd + i])
extern "C" __global__ void k_attn_decode(const float* __restrict__ q, const float* __restrict__ gate, const float* __restrict__ kc, const float* __restrict__ vc,
                                         float* __restrict__ mixed, unsigned kv_width, const unsigned* __restrict__ posp, unsigned n_heads, unsigned n_kv, unsigned hd) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    unsigned len = *posp + 1;
    unsigned h = blockIdx.x;
    unsigned i = threadIdx.x;
    unsigned groups = n_heads / n_kv;
    unsigned kh = h / groups;
    const float* qh = q + (size_t)h * hd;
    float scale = 1.0f / sqrtf((float)hd);
    unsigned warp = i >> 5, lane = i & 31, nwarps = blockDim.x >> 5;
    for (unsigned tpos = warp; tpos < len; tpos += nwarps) {
        const float* kr = kc + (size_t)tpos * kv_width + kh * hd;
        float s = 0.0f;
        for (unsigned d = lane; d < hd; d += 32) s = fmaf(qh[d], kr[d], s);
        s = warp_sum(s);
        if (lane == 0) sc[tpos] = s * scale;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (unsigned tpos = i; tpos < len; tpos += blockDim.x) m = fmaxf(m, sc[tpos]);
    m = block_max(m, red);
    float ssum = 0.0f;
    for (unsigned tpos = i; tpos < len; tpos += blockDim.x) { float e = expf(sc[tpos] - m); sc[tpos] = e; ssum += e; }
    ssum = block_sum(ssum, red);
    if (i < hd) {
        float acc = 0.0f;
        for (unsigned tpos = 0; tpos < len; tpos++) acc = fmaf(sc[tpos] / ssum, vc[(size_t)tpos * kv_width + kh * hd + i], acc);
        float g = gate[(size_t)h * hd + i];
        mixed[(size_t)h * hd + i] = acc * (1.0f / (1.0f + expf(-g)));
    }
}
// prefill attention, grouped: block per (kv head, group of TQ query tokens),
// serving the `groups` query heads of that kv head at once - every key
// and value row is read once per block instead of once per (token, head).
// Scores for the (head, token) pairs of the block live in shared memory
// (len <= 512: the caller falls back to the per-(token, head) kernel
// beyond); warp per key for the scores, warp per (head, token) row for
// the softmax, thread per output element for the mix.
#define AG_TQ 4
extern "C" __global__ void k_attn_prefill_grouped(const float* __restrict__ q, const float* __restrict__ gate, const float* __restrict__ kc, const float* __restrict__ vc,
                                                  float* __restrict__ mixed, i8* __restrict__ xq, float* __restrict__ xs,
                                                  unsigned kv_width, unsigned pos0, unsigned t, unsigned n_heads, unsigned n_kv, unsigned hd) {
    extern __shared__ __align__(16) float ash[];
    unsigned kh = blockIdx.x;                    // kv head
    unsigned tq0 = blockIdx.y * AG_TQ;           // first query token of the group
    unsigned groups = n_heads / n_kv;
    unsigned npairs = groups * AG_TQ;            // (head, token) pairs served
    unsigned pp4 = (npairs + 3) & ~3u;           // pairs padded to a multiple of four (float4 rows)
    unsigned len_max = pos0 + min(tq0 + AG_TQ, t); // keys the last token of the group sees
    unsigned qw_ = n_heads * hd;
    float* sc = ash;                              // [len_max][pp4] scores, keys outer
    float* red = sc + len_max * pp4;              // [pp4] softmax sums
    unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nwarps = blockDim.x >> 5;
    float scale = 1.0f / sqrtf((float)hd);
    // scores: warp w owns pairs [w*PPW, (w+1)*PPW) with their queries in
    // registers (hd/32 lanes x PPW, straight from global memory) and walks
    // every key: one key row read per key per warp
    {
        constexpr int PPW = 4;                          // pairs per warp (npairs <= 8 * PPW)
        unsigned p0 = warp * PPW;
        float qr[PPW][8];
        #pragma unroll
        for (int pp = 0; pp < PPW; pp++) {
            unsigned p = p0 + pp;
            unsigned g = p / AG_TQ, tk = p % AG_TQ;
            unsigned tok = tq0 + tk;
            bool ok = p < npairs && tok < t;
            const float* qp = q + (size_t)(ok ? tok : 0) * qw_ + (kh * groups + (ok ? g : 0)) * hd;
            #pragma unroll
            for (int j = 0; j < 8; j++) qr[pp][j] = (ok && lane + 32 * j < hd) ? qp[lane + 32 * j] : 0.0f;
        }
        if (p0 < pp4) {
            for (unsigned key = 0; key < len_max; key++) {
                const float* kr = kc + (size_t)key * kv_width + kh * hd;
                float kv[8];
                #pragma unroll
                for (int j = 0; j < 8; j++) kv[j] = (lane + 32 * j < hd) ? kr[lane + 32 * j] : 0.0f;
                float sv[PPW];
                #pragma unroll
                for (int pp = 0; pp < PPW; pp++) {
                    float s = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) s = fmaf(qr[pp][j], kv[j], s);
                    sv[pp] = warp_sum(s) * scale;
                }
                if (lane == 0) *(float4*)(sc + key * pp4 + p0) = make_float4(sv[0], sv[1], sv[2], sv[3]);
            }
        }
    }
    __syncthreads();
    // softmax per pair over its causal window [0, pos0 + tok]; keys past
    // the window (and the pairs of tokens past the prompt) become zero
    for (unsigned p = warp; p < pp4; p += nwarps) {
        unsigned tk = p % AG_TQ;
        unsigned tok = tq0 + tk;
        bool live = p < npairs && tok < t;
        unsigned len = live ? pos0 + tok + 1 : 0;
        float m = -3.0e38f;
        for (unsigned k = lane; k < len; k += 32) m = fmaxf(m, sc[k * pp4 + p]);
        m = warp_max(m);
        float ssum = 0.0f;
        for (unsigned k = lane; k < len; k += 32) { float e = expf(sc[k * pp4 + p] - m); sc[k * pp4 + p] = e; ssum += e; }
        for (unsigned k = len + lane; k < len_max; k += 32) sc[k * pp4 + p] = 0.0f;
        ssum = warp_sum(ssum);
        if (lane == 0) red[p] = live ? ssum : 1.0f;
    }
    __syncthreads();
    // mix: thread per dimension d (hd <= 256 = blockDim), all pairs at once -
    // every value row is read once per block per key; the scores of a key
    // come from shared memory as float4 warp broadcasts (npairs <= 32)
    if (tid < hd) {
        unsigned d = tid;
        float acc[32];
        #pragma unroll
        for (int p = 0; p < 32; p++) acc[p] = 0.0f;
        for (unsigned key = 0; key < len_max; key++) {
            float v = vc[(size_t)key * kv_width + kh * hd + d];
            const float4* row4 = (const float4*)(sc + key * pp4);
            #pragma unroll
            for (int p4 = 0; p4 < 8; p4++) {
                if (p4 * 4 < (int)pp4) {
                    float4 s4 = row4[p4];
                    acc[p4 * 4 + 0] = fmaf(s4.x, v, acc[p4 * 4 + 0]);
                    acc[p4 * 4 + 1] = fmaf(s4.y, v, acc[p4 * 4 + 1]);
                    acc[p4 * 4 + 2] = fmaf(s4.z, v, acc[p4 * 4 + 2]);
                    acc[p4 * 4 + 3] = fmaf(s4.w, v, acc[p4 * 4 + 3]);
                }
            }
        }
        // gate, store, and quantize: the warp's 32 lanes are 32 consecutive
        // dims of the same (token, head) - one block of 32 (hd multiple of 32)
        unsigned warp = tid >> 5, lane = tid & 31;
        unsigned nbrow = qw_ / 32;
        for (unsigned p = 0; p < npairs; p++) {
            unsigned g = p / AG_TQ, tk = p % AG_TQ;
            unsigned tok = tq0 + tk;
            if (tok >= t) continue;
            unsigned h = kh * groups + g;
            float inv = 1.0f / red[p];
            float gt = gate[(size_t)tok * qw_ + h * hd + d];
            float o = acc[p] * inv * (1.0f / (1.0f + expf(-gt)));
            mixed[(size_t)tok * qw_ + h * hd + d] = o;
            float m = warp_max(fabsf(o));
            float dx = m / 127.0f;
            i8 qv = 0;
            if (dx != 0.0f) {
                float rr = roundf(o / dx);
                rr = fminf(fmaxf(rr, -127.0f), 127.0f);
                qv = (i8)(int)rr;
            }
            unsigned blk = (h * hd) / 32 + warp;
            xq[((size_t)tok * nbrow + blk) * 32 + lane] = qv;
            if (lane == 0) xs[(size_t)tok * nbrow + blk] = dx;
        }
    }
}
// prefill attention: block per (query token, head); keys [0, pos0 + tok]; q [t][n_heads*hd] (post), gate same
extern "C" __global__ void k_attn_prefill(const float* __restrict__ q, const float* __restrict__ gate, const float* __restrict__ kc, const float* __restrict__ vc,
                                          float* __restrict__ mixed, unsigned kv_width, unsigned pos0, unsigned t, unsigned n_heads, unsigned n_kv, unsigned hd) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    unsigned tok = blockIdx.x, h = blockIdx.y;
    unsigned i = threadIdx.x;
    unsigned len = pos0 + tok + 1;
    unsigned groups = n_heads / n_kv;
    unsigned kh = h / groups;
    unsigned qw_ = n_heads * hd;
    const float* qh = q + (size_t)tok * qw_ + h * hd;
    float scale = 1.0f / sqrtf((float)hd);
    unsigned warp = i >> 5, lane = i & 31, nwarps = blockDim.x >> 5;
    for (unsigned tpos = warp; tpos < len; tpos += nwarps) {
        const float* kr = kc + (size_t)tpos * kv_width + kh * hd;
        float s = 0.0f;
        for (unsigned d = lane; d < hd; d += 32) s = fmaf(qh[d], kr[d], s);
        s = warp_sum(s);
        if (lane == 0) sc[tpos] = s * scale;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (unsigned tpos = i; tpos < len; tpos += blockDim.x) m = fmaxf(m, sc[tpos]);
    m = block_max(m, red);
    float ssum = 0.0f;
    for (unsigned tpos = i; tpos < len; tpos += blockDim.x) { float e = expf(sc[tpos] - m); sc[tpos] = e; ssum += e; }
    ssum = block_sum(ssum, red);
    if (i < hd) {
        float acc = 0.0f;
        for (unsigned tpos = 0; tpos < len; tpos++) acc = fmaf(sc[tpos] / ssum, vc[(size_t)tpos * kv_width + kh * hd + i], acc);
        float g = gate[(size_t)tok * qw_ + h * hd + i];
        mixed[(size_t)tok * qw_ + h * hd + i] = acc * (1.0f / (1.0f + expf(-g)));
    }
}
"#;

// ── host-side helpers ──

/// Row-major q8_0 quantization of an f32 [rows, cols] tensor (the CPU
/// spine's numbers: per block of 32, scale = max|w| / 127, values
/// round-half-away, clamped). Returns (i8 rows, f32 scales).
pub fn quantize_rows_q8(w: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let nb = cols / 32;
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![0f32; rows * nb];
    for r in 0..rows {
        let mut xq = crate::quant::q8::Q8Vec::new();
        crate::quant::q8::quantize_q8_into(&w[r * cols..(r + 1) * cols], &mut xq);
        q[r * cols..(r + 1) * cols].copy_from_slice(&xq.q);
        s[r * nb..(r + 1) * nb].copy_from_slice(&xq.scales);
    }
    (q, s)
}

/// `microkimi cudabench`: probes the device, checks every kernel against
/// the CPU kernels on random data (matvec q8 / fp4, GEMM q8 / fp4,
/// quantization) and measures the matvecs' weight bandwidth out of a
/// 1 GB matrix set (the decode question).
pub fn cudabench_cmd(args: &[String]) {
    let _ = args;
    let Some(c) = ctx() else {
        println!("cuda: no usable device (set MICROKIMI_CUDA_VERBOSE=1 for the reason)");
        return;
    };
    let (free, total) = c.mem_info();
    println!("device: {} sm_{}{} {} SMs, {:.1} GB ({:.1} GB free)", c.name, c.cc.0, c.cc.1, c.sm_count, total as f64 / 1e9, free as f64 / 1e9);
    struct Rng(u64);
    impl Rng {
        fn f(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            ((self.0 >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        }
    }
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let (rows, cols) = (4096usize, 5120usize);
    let w: Vec<f32> = (0..rows * cols).map(|_| rng.f()).collect();
    let x: Vec<f32> = (0..cols).map(|_| rng.f() * 3.0).collect();
    // CPU references
    let head = crate::model::Q8Head::from_f32(&w, rows, cols);
    let mut y_cpu = vec![0f32; rows];
    head.matvec(&x, &mut y_cpu);
    let (wq, ws) = quantize_rows_q8(&w, rows, cols);
    let ws16: Vec<u16> = ws.iter().map(|&v| crate::quant::f16::f32_to_f16(v)).collect();
    let (dwq, dws) = (c.upload(&wq).unwrap(), c.upload(&ws16).unwrap());
    let dx = c.upload(&x).unwrap();
    let dxq = c.alloc(cols).unwrap();
    let dxs = c.alloc(cols / 32 * 4).unwrap();
    let dy = c.alloc(rows * 4).unwrap();
    assert!(c.quantize_q8(&dx, 0, &dxq, &dxs, 1, cols as u32));
    // quantization check vs CPU
    let mut xq_cpu = crate::quant::q8::Q8Vec::new();
    crate::quant::q8::quantize_q8_into(&x, &mut xq_cpu);
    let mut xq_gpu = vec![0i8; cols];
    let mut xs_gpu = vec![0f32; cols / 32];
    c.read(&dxq, 0, &mut xq_gpu);
    c.read(&dxs, 0, &mut xs_gpu);
    let qeq = xq_gpu == xq_cpu.q && xs_gpu.iter().zip(&xq_cpu.scales).all(|(a, b)| a.to_bits() == b.to_bits());
    println!("quantize_q8 vs CPU: {}", if qeq { "IDENTICAL" } else { "DIFFERENT" });
    assert!(c.matvec_q8(&dwq, &dws, &dxq, &dxs, &dy, 0, rows as u32, cols as u32));
    let mut y_gpu = vec![0f32; rows];
    c.read(&dy, 0, &mut y_gpu);
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let mut num = 0f64;
        let mut den = 0f64;
        for (p, q) in a.iter().zip(b) {
            num += ((p - q) as f64).powi(2);
            den += (*q as f64).powi(2);
        }
        (num / den.max(1e-30)).sqrt()
    };
    println!("matvec_q8 vs CPU q8: rel err {:.2e}  (y[0] {:.5} vs {:.5})", rel(&y_gpu, &y_cpu), y_gpu[0], y_cpu[0]);
    // fp4
    let (pk, sc) = crate::quant::mxfp4::quantize_naive(&w, rows, cols);
    let mut y4_cpu = vec![0f32; rows];
    crate::quant::mxfp4::matvec_packed_q8(&pk, &sc, rows, cols, &xq_cpu, &mut y4_cpu, 1);
    let (dpk, dsc) = (c.upload_bytes(&pk).unwrap(), c.upload_bytes(&sc).unwrap());
    assert!(c.matvec_fp4(&dpk, &dsc, &dxq, &dxs, &dy, 0, rows as u32, cols as u32));
    let mut y4_gpu = vec![0f32; rows];
    c.read(&dy, 0, &mut y4_gpu);
    println!("matvec_fp4 vs CPU fp4: rel err {:.2e}  (y[0] {:.5} vs {:.5})", rel(&y4_gpu, &y4_cpu), y4_gpu[0], y4_cpu[0]);
    // GEMM: t tokens
    let t = 137usize;
    let xs_all: Vec<f32> = (0..t * cols).map(|_| rng.f() * 3.0).collect();
    let xr: Vec<&[f32]> = xs_all.chunks(cols).collect();
    let mut outs: Vec<Vec<f32>> = vec![vec![0f32; rows]; t];
    {
        let mut om: Vec<&mut [f32]> = outs.iter_mut().map(|o| o.as_mut_slice()).collect();
        head.matvec_multi(&xr, &mut om);
    }
    let dxa = c.upload(&xs_all).unwrap();
    let dxqa = c.alloc(t * cols).unwrap();
    let dxsa = c.alloc(t * cols / 32 * 4).unwrap();
    assert!(c.quantize_q8(&dxa, 0, &dxqa, &dxsa, t as u32, cols as u32));
    let dc = c.alloc(t * rows * 4).unwrap();
    assert!(c.gemm_q8(&dwq, &dws, &dxqa, &dxsa, &dc, 0, rows as u32, cols as u32, t as u32, rows as u32));
    let mut c_gpu = vec![0f32; t * rows];
    c.read(&dc, 0, &mut c_gpu);
    let c_cpu: Vec<f32> = outs.iter().flatten().cloned().collect();
    println!("gemm_q8 vs CPU multi: rel err {:.2e}", rel(&c_gpu, &c_cpu));
    // fp4 GEMM vs CPU packed multi via per-token matvec_packed_q8
    let mut c4_cpu = vec![0f32; t * rows];
    for (ti, xt) in xr.iter().enumerate() {
        let mut xq = crate::quant::q8::Q8Vec::new();
        crate::quant::q8::quantize_q8_into(xt, &mut xq);
        crate::quant::mxfp4::matvec_packed_q8(&pk, &sc, rows, cols, &xq, &mut c4_cpu[ti * rows..(ti + 1) * rows], 1);
    }
    assert!(c.gemm_fp4(&dpk, &dsc, &dxqa, &dxsa, &dc, 0, rows as u32, cols as u32, t as u32, rows as u32));
    c.read(&dc, 0, &mut c_gpu);
    println!("gemm_fp4 vs CPU fp4: rel err {:.2e}", rel(&c_gpu, &c4_cpu));
    // --gemm ROWS COLS T: the tensor-core GEMMs on one shape, then out
    if let Some(i) = args.iter().position(|a| a == "--gemm") {
        let gr: usize = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(17408);
        let gc: usize = args.get(i + 2).and_then(|v| v.parse().ok()).unwrap_or(5120);
        let gt: usize = args.get(i + 3).and_then(|v| v.parse().ok()).unwrap_or(256);
        let gw: Vec<i8> = (0..gr * gc).map(|i| ((i * 7 + 3) % 23) as i8 - 11).collect();
        let gs: Vec<u16> = vec![crate::quant::f16::f32_to_f16(0.01); gr * gc / 32];
        let dgw = c.upload(&gw).unwrap();
        let dgs = c.upload(&gs).unwrap();
        let gx: Vec<i8> = (0..gt * gc).map(|i| ((i * 5 + 1) % 13) as i8 - 6).collect();
        let gxs: Vec<f32> = vec![0.02; gt * gc / 32];
        let dgx = c.upload(&gx).unwrap();
        let dgxs = c.upload(&gxs).unwrap();
        let dgc = c.alloc(gt * gr * 4).unwrap();
        let gp: Vec<u8> = (0..gr * gc / 2).map(|i| (i * 37 % 251) as u8).collect();
        let gsc: Vec<u8> = vec![120u8; gr * gc / 32];
        let dgp = c.upload_bytes(&gp).unwrap();
        let dgsc = c.upload_bytes(&gsc).unwrap();
        let macs = (gr * gc * gt) as f64;
        // --copies N: N distinct weight copies cycled so the weights are cold
        // in L2 as they are in a real prompt (one matrix per layer)
        let copies: usize = args.iter().position(|a| a == "--copies").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok()).unwrap_or(1);
        let mut wq_copies: Vec<(DBuf, DBuf)> = Vec::new();
        let mut wp_copies: Vec<(DBuf, DBuf)> = Vec::new();
        for _ in 1..copies {
            wq_copies.push((c.upload(&gw).unwrap(), c.upload(&gs).unwrap()));
            wp_copies.push((c.upload_bytes(&gp).unwrap(), c.upload_bytes(&gsc).unwrap()));
        }
        for round in 0..3 {
            let mut ms = 0f32;
            let mut ms4 = 0f32;
            for k in 0..copies {
                let (wq_k, ws_k) = if k == 0 { (&dgw, &dgs) } else { (&wq_copies[k - 1].0, &wq_copies[k - 1].1) };
                let (wp_k, wsc_k) = if k == 0 { (&dgp, &dgsc) } else { (&wp_copies[k - 1].0, &wp_copies[k - 1].1) };
                ms += c.timed(|| {
                    c.gemm_q8(wq_k, ws_k, &dgx, &dgxs, &dgc, 0, gr as u32, gc as u32, gt as u32, gr as u32);
                });
                ms4 += c.timed(|| {
                    c.gemm_fp4(wp_k, wsc_k, &dgx, &dgxs, &dgc, 0, gr as u32, gc as u32, gt as u32, gr as u32);
                });
            }
            ms /= copies as f32;
            ms4 /= copies as f32;
            println!("gemm {}x{} t={} round {}: q8 {:.0} GMAC/s ({:.3} ms) | fp4 {:.0} GMAC/s ({:.3} ms) | nt {}{}", gr, gc, gt, round, macs / (ms as f64 * 1e-3) / 1e9, ms, macs / (ms4 as f64 * 1e-3) / 1e9, ms4, c.gemm_nt(gr as u32, gt as u32), if copies > 1 { format!(" | {} copies (cold)", copies) } else { String::new() });
        }
        return;
    }
    // --shape ROWS COLS: bandwidth of the two matvecs on one shape (many copies), then out
    if let Some(i) = args.iter().position(|a| a == "--shape") {
        let sr: usize = args.get(i + 1).and_then(|v| v.parse().ok()).unwrap_or(5120);
        let scc: usize = args.get(i + 2).and_then(|v| v.parse().ok()).unwrap_or(17408);
        let copies = (600usize << 20) / (sr * scc / 2).max(1);
        let pk1: Vec<u8> = (0..sr * scc / 2).map(|i| (i * 37 % 251) as u8).collect();
        let sc1: Vec<u8> = vec![120u8; sr * scc / 32];
        let mut allp: Vec<u8> = Vec::with_capacity(copies * pk1.len());
        let mut alls: Vec<u8> = Vec::with_capacity(copies * sc1.len());
        for _ in 0..copies.max(1) {
            allp.extend_from_slice(&pk1);
            alls.extend_from_slice(&sc1);
        }
        let dp = c.upload_bytes(&allp).unwrap();
        let ds = c.upload_bytes(&alls).unwrap();
        let x1: Vec<f32> = (0..scc).map(|i| ((i * 5 + 1) % 13) as f32 * 0.02 - 0.1).collect();
        let dx1 = c.upload(&x1).unwrap();
        let dxq1 = c.alloc(scc).unwrap();
        let dxs1 = c.alloc(scc / 32 * 4).unwrap();
        c.quantize_q8(&dx1, 0, &dxq1, &dxs1, 1, scc as u32);
        let dy1 = c.alloc(sr * 4).unwrap();
        let bytes = (pk1.len() + sc1.len()) as f64;
        for round in 0..3 {
            let ms = c.timed(|| {
                for k in 0..copies.max(1) {
                    let dpk = DBuf { ptr: dp.ptr + (k * pk1.len()) as u64, len: pk1.len() };
                    let dsk = DBuf { ptr: ds.ptr + (k * sc1.len()) as u64, len: sc1.len() };
                    c.matvec_fp4(&dpk, &dsk, &dxq1, &dxs1, &dy1, 0, sr as u32, scc as u32);
                    std::mem::forget(dpk);
                    std::mem::forget(dsk);
                }
            });
            println!("matvec_fp4 {}x{} x{} round {}: {:.1} GB/s ({:.3} ms per matvec)", sr, scc, copies, round, bytes * copies as f64 / (ms as f64 * 1e-3) / 1e9, ms / copies as f32);
        }
        return;
    }
    // speed: matvec bandwidth on a 1 GB set (q8) and 0.5 GB (fp4)
    let big_rows = 8 * 4096usize; // 8 matrices' worth as one tall matrix: 32768 x 5120 i8 = 168 MB.. use 6 copies
    let reps = 6usize;
    let mut bigq: Vec<i8> = Vec::with_capacity(reps * big_rows * cols);
    let mut bigs: Vec<u16> = Vec::with_capacity(reps * big_rows * cols / 32);
    for _ in 0..reps {
        for _ in 0..(big_rows / rows) {
            bigq.extend_from_slice(&wq);
            bigs.extend_from_slice(&ws16);
        }
    }
    let brows = reps * big_rows;
    let dbq = c.upload(&bigq).unwrap();
    let dbs = c.upload(&bigs).unwrap();
    let dby = c.alloc(brows * 4).unwrap();
    let bytes = (bigq.len() + bigs.len() * 2) as f64;
    for round in 0..3 {
        let ms = c.timed(|| {
            c.matvec_q8(&dbq, &dbs, &dxq, &dxs, &dby, 0, brows as u32, cols as u32);
        });
        println!("matvec_q8  {:.0} MB round {}: {:.1} GB/s ({:.2} ms)", bytes / 1e6, round, bytes / (ms as f64 * 1e-3) / 1e9, ms);
    }
    drop((dbq, dbs));
    let mut bigp: Vec<u8> = Vec::with_capacity(reps * big_rows * cols / 2);
    let mut bigsc: Vec<u8> = Vec::with_capacity(reps * big_rows * cols / 32);
    for _ in 0..reps {
        for _ in 0..(big_rows / rows) {
            bigp.extend_from_slice(&pk);
            bigsc.extend_from_slice(&sc);
        }
    }
    let dbp = c.upload_bytes(&bigp).unwrap();
    let dbsc = c.upload_bytes(&bigsc).unwrap();
    let bytes4 = (bigp.len() + bigsc.len()) as f64;
    for round in 0..3 {
        let ms = c.timed(|| {
            c.matvec_fp4(&dbp, &dbsc, &dxq, &dxs, &dby, 0, brows as u32, cols as u32);
        });
        println!("matvec_fp4 {:.0} MB round {}: {:.1} GB/s ({:.2} ms)", bytes4 / 1e6, round, bytes4 / (ms as f64 * 1e-3) / 1e9, ms);
    }
    // GEMM speed on the 27B shape (17408 x 5120, 256 tokens)
    let (gr, gc, gt) = (17408usize, 5120usize, 256usize);
    let gw: Vec<i8> = (0..gr * gc).map(|i| ((i * 7 + 3) % 23) as i8 - 11).collect();
    let gs: Vec<u16> = vec![crate::quant::f16::f32_to_f16(0.01); gr * gc / 32];
    let dgw = c.upload(&gw).unwrap();
    let dgs = c.upload(&gs).unwrap();
    let gx: Vec<i8> = (0..gt * gc).map(|i| ((i * 5 + 1) % 13) as i8 - 6).collect();
    let gxs: Vec<f32> = vec![0.02; gt * gc / 32];
    let dgx = c.upload(&gx).unwrap();
    let dgxs = c.upload(&gxs).unwrap();
    let dgc = c.alloc(gt * gr * 4).unwrap();
    for round in 0..3 {
        let ms = c.timed(|| {
            c.gemm_q8(&dgw, &dgs, &dgx, &dgxs, &dgc, 0, gr as u32, gc as u32, gt as u32, gr as u32);
        });
        let macs = (gr * gc * gt) as f64;
        println!("gemm_q8 {}x{}x{} round {}: {:.0} GMAC/s ({:.2} ms)", gr, gc, gt, round, macs / (ms as f64 * 1e-3) / 1e9, ms);
    }
    let gp: Vec<u8> = (0..gr * gc / 2).map(|i| (i * 37 % 251) as u8).collect();
    let gsc: Vec<u8> = vec![120u8; gr * gc / 32];
    let dgp = c.upload_bytes(&gp).unwrap();
    let dgsc = c.upload_bytes(&gsc).unwrap();
    for round in 0..3 {
        let ms = c.timed(|| {
            c.gemm_fp4(&dgp, &dgsc, &dgx, &dgxs, &dgc, 0, gr as u32, gc as u32, gt as u32, gr as u32);
        });
        let macs = (gr * gc * gt) as f64;
        println!("gemm_fp4 {}x{}x{} round {}: {:.0} GMAC/s ({:.2} ms)", gr, gc, gt, round, macs / (ms as f64 * 1e-3) / 1e9, ms);
    }
}

// ── the resident decoder ──

use crate::model::decode_refs::{DecodeLayerRefs, DecodeModelRefs};

/// Phase nanoseconds under MICROKIMI_CUDA_PROF=1: 0 norm+quantize,
/// 1 spine GEMM/matvec, 2 conv+prep, 3 scan, 4 gated norm, 5 attention
/// (qk prep + attn), 6 MLP GEMM/matvec, 7 silu, 8 head.
pub static PROF: [std::sync::atomic::AtomicU64; 9] = [
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

/// Prints and clears the phase profile (no-op when nothing was recorded).
pub fn prof_report(label: &str) {
    let names = ["norm+quant", "spine gemm", "conv+prep", "scan", "gated norm", "attention", "mlp gemm", "silu", "head"];
    let vals: Vec<u64> = PROF.iter().map(|a| a.swap(0, std::sync::atomic::Ordering::Relaxed)).collect();
    if vals.iter().all(|&v| v == 0) {
        return;
    }
    let parts: Vec<String> = names.iter().zip(&vals).map(|(n, v)| format!("{n} {:.1} ms", *v as f64 / 1e6)).collect();
    println!("cuda prof [{label}]: {}", parts.join(" | "));
}

/// A q8 weight on the device: int8 rows and f32 block scales.
#[allow(dead_code)]
struct WQ8 {
    q: DBuf,
    s: DBuf,
    rows: usize,
    cols: usize,
}
/// An MXFP4 weight on the device: packed nibbles and e8m0 scale bytes.
#[allow(dead_code)]
struct WFp4 {
    p: DBuf,
    s: DBuf,
    rows: usize,
    cols: usize,
}

enum LayerDev {
    Linear {
        in_norm: DBuf,
        post_norm: DBuf,
        proj: WQ8, // rows [qkv (conv_dim) | z (vt) | b (heads) | a (heads)]
        out: WQ8,  // [d, vt]
        gu: WFp4,  // rows [gate (inter) | up (inter)]
        dn: WFp4,  // [d, inter]
        conv_w: DBuf,
        a_log: DBuf,
        dt_bias: DBuf,
        gated_w: DBuf,
        conv_state: DBuf, // [conv_dim][k-1] f32
        scan_state: DBuf, // [heads][kd][vd] f32
        heads: usize,
        kv_heads: usize,
        kd: usize,
        vd: usize,
        conv_k: usize,
        inter: usize,
    },
    Full {
        in_norm: DBuf,
        post_norm: DBuf,
        proj: WQ8, // rows [q|gate per head (2 qw) | k (kvw) | v (kvw)]
        o: WQ8,    // [d, qw]
        gu: WFp4,
        dn: WFp4,
        q_norm: DBuf,
        k_norm: DBuf,
        kc: DBuf, // [cap][kvw] f32
        vc: DBuf,
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        inter: usize,
    },
}

pub struct CudaDecoder {
    layers: Vec<LayerDev>,
    lm_head: WQ8,
    norm_f: DBuf,
    d: usize,
    vocab: usize,
    eps: f32,
    #[allow(dead_code)]
    kv_width: usize,
    cap: usize,
    pos: usize,
    // scratch
    hidden: DBuf,  // [d] f32
    normed: DBuf,  // [d] f32
    xq: DBuf,      // [max_cols] i8
    xs: DBuf,      // [max_cols/32] f32
    big_a: DBuf,   // [max_rows] f32 (projection outputs)
    big_b: DBuf,   // [max_rows] f32
    y_d: DBuf,     // [d] f32
    q_buf: DBuf,   // [qw or heads*kd] f32
    gate_buf: DBuf,// [qw or heads*kd] f32
    mixed: DBuf,   // [max(qw, vt)] f32
    beta: DBuf,    // [heads]
    decay: DBuf,   // [heads]
    logits: DBuf,  // [vocab] f32
    trace_buf: DBuf, // [n_layers * d] f32
    /// the position as the decode kernels read it (device u32), fed from pos_pin
    pos_dev: DBuf,
    pos_pin: PinBuf,    // host u32
    emb_pin: PinBuf,    // host [d] f32: the token's embedding row
    logits_pin: PinBuf, // host [vocab] f32
    /// the token graph, captured on the first step (None = direct launches)
    graph: Option<CUgraphExec>,
    /// cos | sin of the partial rotary per position, [cap][rope_dim] f32,
    /// computed on the host exactly as the CPU does (f64 angle, f32 result)
    rope: DBuf,
    /// prefill workspace, grown to the largest prompt seen
    ws: Option<PrefillWs>,
    max_rows: usize,
    max_cols: usize,
    max_qw: usize,
    max_mixed: usize,
    max_heads: usize,
}
unsafe impl Send for CudaDecoder {}

/// Token-major scratch of a prefill: rows of the per-token vectors.
struct PrefillWs {
    t_cap: usize,
    hidden: DBuf, // [t][d]
    xq: DBuf,     // [t][max_cols] i8
    xs: DBuf,     // [t][max_cols/32]
    big_a: DBuf,  // [t][max_rows]
    big_b: DBuf,  // [t][max_rows]
    y: DBuf,      // [t][d]
    q: DBuf,      // [t][max_qw]
    gate: DBuf,   // [t][max_qw]
    mixed: DBuf,  // [t][max_mixed]
    beta: DBuf,   // [t][heads]
    decay: DBuf,  // [t][heads]
}

impl CudaDecoder {
    pub fn pos(&self) -> usize {
        self.pos
    }
    /// KV capacity in positions.
    pub fn cap(&self) -> usize {
        self.cap
    }
}
impl Drop for CudaDecoder {
    fn drop(&mut self) {
        if let (Some(g), Some(c)) = (self.graph.take(), ctx()) {
            c.graph_free(g);
        }
    }
}

fn upload_q8(c: &CudaCtx, w: &[f32], rows: usize, cols: usize) -> Option<WQ8> {
    let nb = cols / 32;
    // parallel row quantization on the pool (the 27B spine is 7 B values)
    let mut q = vec![0i8; rows * cols];
    let mut s = vec![0f32; rows * nb];
    {
        let p = crate::model::pool::pool();
        let workers = p.workers.max(1).min(rows.max(1));
        let chunk = rows.div_ceil(workers).max(1);
        let qp = crate::model::pool::MPtr(q.as_mut_ptr() as *mut f32);
        let sp = crate::model::pool::MPtr(s.as_mut_ptr());
        let wp = crate::model::pool::SPtr(w.as_ptr());
        let mut jobs: Vec<crate::model::pool::Job> = Vec::new();
        let mut r0 = 0usize;
        while r0 < rows {
            let r1 = (r0 + chunk).min(rows);
            jobs.push(Box::new(move || {
                let (qp, sp, wp) = (qp, sp, wp);
                // SAFETY: disjoint row ranges; the pool barrier outlives the pointers.
                unsafe {
                    let w = std::slice::from_raw_parts(wp.0, rows * cols);
                    let mut xq = crate::quant::q8::Q8Vec::new();
                    for r in r0..r1 {
                        crate::quant::q8::quantize_q8_into(&w[r * cols..(r + 1) * cols], &mut xq);
                        std::ptr::copy_nonoverlapping(xq.q.as_ptr(), (qp.0 as *mut i8).add(r * cols), cols);
                        std::ptr::copy_nonoverlapping(xq.scales.as_ptr(), sp.0.add(r * nb), nb);
                    }
                }
            }));
            r0 = r1;
        }
        p.run(jobs);
    }
    // f16 block scales: half the scale bytes per token (Metal stores them
    // the same way); a scale is max|w|/127, well inside f16's range
    let s16: Vec<u16> = s.iter().map(|&v| crate::quant::f16::f32_to_f16(v)).collect();
    Some(WQ8 { q: c.upload(&q)?, s: c.upload(&s16)?, rows, cols })
}

/// [cap][rope_dim]: for each position, cos of the `rope_dim/2` angles then
/// their sin - the numbers `rope_partial` on the CPU computes (f64 angle,
/// f32 result), so the device rotation matches it.
fn rope_table(cap: usize, rope_dim: usize, theta: f64) -> Vec<f32> {
    let half = rope_dim / 2;
    let mut t = vec![0f32; cap * rope_dim.max(1)];
    if rope_dim == 0 {
        return t;
    }
    for pos in 0..cap {
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f64 / rope_dim as f64);
            let ang = pos as f64 * freq;
            t[pos * rope_dim + i] = ang.cos() as f32;
            t[pos * rope_dim + half + i] = ang.sin() as f32;
        }
    }
    t
}

/// Concatenated q8 weight: the given (f32 matrix, rows) share `cols`.
fn upload_q8_cat(c: &CudaCtx, parts: &[(&[f32], usize)], cols: usize) -> Option<WQ8> {
    let rows: usize = parts.iter().map(|p| p.1).sum();
    let mut all: Vec<f32> = Vec::with_capacity(rows * cols);
    for (w, r) in parts {
        all.extend_from_slice(&w[..r * cols]);
    }
    upload_q8(c, &all, rows, cols)
}

fn upload_fp4(c: &CudaCtx, packed: &[u8], scales: &[u8], rows: usize, cols: usize) -> Option<WFp4> {
    Some(WFp4 { p: c.upload_bytes(&packed[..rows * cols / 2])?, s: c.upload_bytes(&scales[..rows * cols / 32])?, rows, cols })
}

fn upload_fp4_cat(c: &CudaCtx, parts: &[(&[u8], &[u8], usize)], cols: usize) -> Option<WFp4> {
    let rows: usize = parts.iter().map(|p| p.2).sum();
    let mut pk: Vec<u8> = Vec::with_capacity(rows * cols / 2);
    let mut sc: Vec<u8> = Vec::with_capacity(rows * cols / 32);
    for (p, s, r) in parts {
        pk.extend_from_slice(&p[..r * cols / 2]);
        sc.extend_from_slice(&s[..r * cols / 32]);
    }
    Some(WFp4 { p: c.upload_bytes(&pk)?, s: c.upload_bytes(&sc)?, rows, cols })
}

/// Builds the resident decoder from the model view and the current CPU
/// caches (`lin_states`: per linear layer (conv, scan) f32; `full_kv`:
/// per full layer (k rows, v rows, len)). `kv_cap` positions of KV
/// cache. None when the GPU is unavailable or memory runs out.
pub fn cuda_decoder_new(
    m: &DecodeModelRefs,
    lin_states: &[(&[f32], &[f32])],
    full_kv: &[(&[f32], &[f32], usize)],
    kv_width: usize,
    kv_cap: usize,
    pos: usize,
) -> Option<CudaDecoder> {
    let c = ctx()?;
    let d = m.d;
    let t0 = std::time::Instant::now();
    let mut layers = Vec::with_capacity(m.layers.len());
    let (mut li, mut fi) = (0usize, 0usize);
    let mut max_rows = m.vocab;
    let mut max_cols = d;
    let mut max_qw = 0usize;
    let mut max_mixed = 0usize;
    let mut max_heads = 0usize;
    for l in &m.layers {
        match l {
            DecodeLayerRefs::Linear { in_norm, post_norm, w, gated_w, dm } => {
                let kt = dm.kv_heads * dm.kd;
                let vt = dm.heads * dm.vd;
                let cd = 2 * kt + vt;
                let proj = upload_q8_cat(c, &[(w.in_qkv, cd), (w.in_z, vt), (w.in_b, dm.heads), (w.in_a, dm.heads)], d)?;
                let out = upload_q8(c, w.out_proj, d, vt)?;
                let gu = upload_fp4_cat(c, &[(w.gate.0, w.gate.1, dm.inter), (w.up.0, w.up.1, dm.inter)], d)?;
                let dn = upload_fp4(c, w.down.0, w.down.1, d, dm.inter)?;
                let (cs, ss) = lin_states.get(li)?;
                li += 1;
                let conv_state = c.upload(cs)?;
                let scan_state = c.upload(ss)?;
                max_rows = max_rows.max(cd + vt + 2 * dm.heads).max(2 * dm.inter);
                max_cols = max_cols.max(vt).max(dm.inter);
                max_qw = max_qw.max(dm.heads * dm.kd);
                max_mixed = max_mixed.max(vt);
                max_heads = max_heads.max(dm.heads);
                layers.push(LayerDev::Linear {
                    in_norm: c.upload(in_norm)?,
                    post_norm: c.upload(post_norm)?,
                    proj,
                    out,
                    gu,
                    dn,
                    conv_w: c.upload(w.conv_w)?,
                    a_log: c.upload(w.a_log)?,
                    dt_bias: c.upload(w.dt_bias)?,
                    gated_w: c.upload(gated_w)?,
                    conv_state,
                    scan_state,
                    heads: dm.heads,
                    kv_heads: dm.kv_heads,
                    kd: dm.kd,
                    vd: dm.vd,
                    conv_k: dm.conv_k,
                    inter: dm.inter,
                });
            }
            DecodeLayerRefs::Full { in_norm, post_norm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, gate, up, down, n_heads, n_kv, hd, rope_dim, theta, inter } => {
                let qw = n_heads * hd;
                let kvw = n_kv * hd;
                if kvw != kv_width {
                    return None;
                }
                let proj = upload_q8_cat(c, &[(q_proj, qw * 2), (k_proj, kvw), (v_proj, kvw)], d)?;
                let o = upload_q8(c, o_proj, d, qw)?;
                let gu = upload_fp4_cat(c, &[(gate.0, gate.1, *inter), (up.0, up.1, *inter)], d)?;
                let dn = upload_fp4(c, down.0, down.1, d, *inter)?;
                let (k, v, len) = full_kv.get(fi)?;
                fi += 1;
                if *len > kv_cap {
                    return None;
                }
                let kc = c.zeroed(kv_cap * kvw * 4)?;
                let vc = c.zeroed(kv_cap * kvw * 4)?;
                if *len > 0 {
                    if !c.write(&kc, 0, &k[..len * kvw]) || !c.write(&vc, 0, &v[..len * kvw]) {
                        return None;
                    }
                }
                max_rows = max_rows.max(2 * qw + 2 * kvw).max(2 * inter);
                max_cols = max_cols.max(qw).max(*inter);
                max_qw = max_qw.max(qw);
                max_mixed = max_mixed.max(qw);
                layers.push(LayerDev::Full {
                    in_norm: c.upload(in_norm)?,
                    post_norm: c.upload(post_norm)?,
                    proj,
                    o,
                    gu,
                    dn,
                    q_norm: c.upload(q_norm)?,
                    k_norm: c.upload(k_norm)?,
                    kc,
                    vc,
                    n_heads: *n_heads,
                    n_kv: *n_kv,
                    hd: *hd,
                    rope_dim: *rope_dim,
                    theta: *theta,
                    inter: *inter,
                });
            }
        }
    }
    let lm_head = upload_q8(c, m.lm_head, m.vocab, d)?;
    let norm_f = c.upload(m.norm_f)?;
    let dec = CudaDecoder {
        layers,
        lm_head,
        norm_f,
        d,
        vocab: m.vocab,
        eps: m.eps,
        kv_width,
        cap: kv_cap,
        pos,
        hidden: c.alloc(d * 4)?,
        normed: c.alloc(d * 4)?,
        xq: c.alloc(max_cols)?,
        xs: c.alloc(max_cols / 32 * 4)?,
        big_a: c.alloc(max_rows * 4)?,
        big_b: c.alloc(max_rows * 4)?,
        y_d: c.alloc(d * 4)?,
        q_buf: c.alloc(max_qw.max(1) * 4)?,
        gate_buf: c.alloc(max_qw.max(1) * 4)?,
        mixed: c.alloc(max_mixed.max(1) * 4)?,
        beta: c.alloc(max_heads.max(1) * 4)?,
        decay: c.alloc(max_heads.max(1) * 4)?,
        logits: c.alloc(m.vocab * 4)?,
        trace_buf: c.alloc(m.layers.len().max(1) * d * 4)?,
        rope: {
            let (rd, theta) = m
                .layers
                .iter()
                .find_map(|l| match l {
                    DecodeLayerRefs::Full { rope_dim, theta, .. } => Some((*rope_dim, *theta as f64)),
                    _ => None,
                })
                .unwrap_or((0, 10000.0));
            let table = rope_table(kv_cap, rd, theta);
            c.upload(&table)?
        },
        pos_dev: c.zeroed(16)?,
        pos_pin: c.pinned(16)?,
        emb_pin: c.pinned(d * 4)?,
        logits_pin: c.pinned(m.vocab * 4)?,
        graph: None,
        ws: None,
        max_rows,
        max_cols,
        max_qw,
        max_mixed,
        max_heads,
    };
    if !c.sync() {
        return None;
    }
    if std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
        let (free, total) = c.mem_info();
        println!("cuda: decoder resident in {:.1} s ({:.1} of {:.1} GB used)", t0.elapsed().as_secs_f64(), (total - free) as f64 / 1e9, total as f64 / 1e9);
    }
    Some(dec)
}

impl CudaDecoder {
    /// One token: embeds on the host, runs every layer and the head on
    /// the stream, copies the logits back. None = refused before any
    /// state mutation (KV capacity reached).
    pub fn step(&mut self, m: &DecodeModelRefs, token: u32, logits_out: &mut [f32]) -> Option<()> {
        self.step_impl(m, token, logits_out, None)
    }

    /// `step` that also returns the hidden state after every layer.
    pub fn step_trace(&mut self, m: &DecodeModelRefs, token: u32, logits_out: &mut [f32], trace: &mut Vec<Vec<f32>>) -> Option<()> {
        self.step_impl(m, token, logits_out, Some(trace))
    }

    fn step_impl(&mut self, m: &DecodeModelRefs, token: u32, logits_out: &mut [f32], trace: Option<&mut Vec<Vec<f32>>>) -> Option<()> {
        let c = ctx()?;
        let d = self.d;
        let pos = self.pos;
        if pos >= self.cap {
            return None;
        }
        // host side of the step: the embedding row and the position into
        // pinned memory (the graph's memcpy nodes read them at launch)
        self.emb_pin.as_mut_slice::<f32>()[..d].copy_from_slice(&m.embed[token as usize * d..(token as usize + 1) * d]);
        self.pos_pin.as_mut_slice::<u32>()[0] = pos as u32;
        let use_graph = trace.is_none()
            && !std::env::var("MICROKIMI_CUDA_NO_GRAPH").map(|v| v == "1").unwrap_or(false)
            && !std::env::var("MICROKIMI_CUDA_PROF").map(|v| v == "1").unwrap_or(false);
        if use_graph {
            if self.graph.is_none() {
                let g = c.capture(|| self.encode_token(m, false));
                if g.is_none() && std::env::var("MICROKIMI_CUDA_VERBOSE").is_ok() {
                    println!("cuda: token graph capture failed - direct launches");
                }
                self.graph = g;
            }
            if let Some(g) = self.graph {
                if !c.graph_launch(g) || !c.sync() {
                    return None;
                }
                logits_out[..self.vocab].copy_from_slice(&self.logits_pin.as_mut_slice::<f32>()[..self.vocab]);
                self.pos += 1;
                return Some(());
            }
        }
        if !self.encode_token(m, trace.is_some()) || !c.sync() {
            return None;
        }
        logits_out[..self.vocab].copy_from_slice(&self.logits_pin.as_mut_slice::<f32>()[..self.vocab]);
        if let Some(tr) = trace {
            let n_layers = self.layers.len();
            let mut all = vec![0f32; n_layers * d];
            c.read(&self.trace_buf, 0, &mut all);
            tr.clear();
            for l in 0..n_layers {
                tr.push(all[l * d..(l + 1) * d].to_vec());
            }
        }
        self.pos += 1;
        Some(())
    }

    /// Issues one token's work on the stream (no sync): the memcpys from
    /// the pinned inputs, every layer, the head, the logits memcpy out.
    /// The position is whatever pos_dev holds. `trace` also copies the
    /// hidden state after each layer into trace_buf.
    fn encode_token(&self, m: &DecodeModelRefs, trace: bool) -> bool {
        let Some(c) = ctx() else { return false };
        let _ = m;
        let d = self.d;
        let eps = self.eps;
        let ok = std::cell::Cell::new(true);
        let chk = |b: bool| {
            if !b {
                ok.set(false);
            }
        };
        chk(c.write_async(&self.hidden, 0, &self.emb_pin, d * 4));
        chk(c.write_async(&self.pos_dev, 0, &self.pos_pin, 4));
        // the MLP residual of a layer folds into the next layer's input
        // norm (one kernel instead of add + norm); in trace mode it is
        // applied eagerly so the traced hidden state is the layer's output
        for (li, l) in self.layers.iter().enumerate() {
            let prev: Option<&DBuf> = if li == 0 || trace { None } else { Some(&self.y_d) };
            match l {
                LayerDev::Linear { in_norm, post_norm, proj, out, gu, dn, conv_w, a_log, dt_bias, gated_w, conv_state, scan_state, heads, kv_heads, kd, vd, conv_k, inter, .. } => {
                    let kt = kv_heads * kd;
                    let vt = heads * vd;
                    let cd = 2 * kt + vt;
                    // input norm (+ the previous layer's MLP residual)
                    self.prof(0, || {
                        chk(c.add_rmsnorm_quant_rows(&self.hidden, prev, in_norm, None, &self.xq, &self.xs, 1, d as u32, eps, true));
                    });
                    // fused projections -> big_a [qkv | z | b | a]
                    self.prof(1, || chk(c.matvec_q8(&proj.q, &proj.s, &self.xq, &self.xs, &self.big_a, 0, proj.rows as u32, d as u32)));
                    // conv + silu -> big_b[0..cd]
                    self.prof(2, || {
                        let k = *conv_k as u32;
                        let cdim = cd as u32;
                        let mut a = args!(self.big_a.ptr, conv_w.ptr, k, conv_state.ptr, self.big_b.ptr, cdim);
                        chk(c.launch("k_conv_silu", (cdim.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a));
                    });
                    let fused = (*kd == 128 || *kd == 64) && vd % 32 == 0 && *vd <= 256;
                    if fused {
                        // q/k norms, beta/decay, delta step, gated norm, quantization: one launch per layer
                        self.prof(3, || {
                            let bp = self.big_a.ptr + ((cd + vt) * 4) as u64;
                            let ap = self.big_a.ptr + ((cd + vt + heads) * 4) as u64;
                            let zp = self.big_a.ptr + (cd * 4) as u64;
                            let (hh, kvh, vdd) = (*heads as u32, *kv_heads as u32, *vd as u32);
                            let name = if *kd == 128 { "k_lin_head_step128" } else { "k_lin_head_step64" };
                            let mut a = args!(self.big_b.ptr, bp, ap, zp, a_log.ptr, dt_bias.ptr, scan_state.ptr, gated_w.ptr, self.mixed.ptr, self.xq.ptr, self.xs.ptr, hh, kvh, vdd, eps);
                            chk(c.launch(name, (*heads as u32, 1, 1), (4 * *vd as u32, 1, 1), 0, &mut a));
                        });
                    } else {
                        // q/k norms, beta, decay: qn -> q_buf, kn -> gate_buf
                        self.prof(2, || {
                            let bp = self.big_a.ptr + ((cd + vt) * 4) as u64;
                            let ap = self.big_a.ptr + ((cd + vt + heads) * 4) as u64;
                            let (t1, hh, kvh, kdd, cdim, ldba) = (1u32, *heads as u32, *kv_heads as u32, *kd as u32, cd as u32, *heads as u32);
                            let mut a = args!(self.big_b.ptr, cdim, bp, ap, ldba, a_log.ptr, dt_bias.ptr, self.q_buf.ptr, self.gate_buf.ptr, self.beta.ptr, self.decay.ptr, t1, hh, kvh, kdd);
                            chk(c.launch("k_lin_prep", (1, *heads as u32, 1), ((*kd as u32).max(32), 1, 1), 0, &mut a));
                        });
                        // delta step per head -> mixed [heads*vd]
                        self.prof(3, || {
                            let (hh, kvh, kdd, vdd, cdim) = (*heads as u32, *kv_heads as u32, *kd as u32, *vd as u32, cd as u32);
                            let mut a = args!(scan_state.ptr, self.q_buf.ptr, self.gate_buf.ptr, self.big_b.ptr, self.beta.ptr, self.decay.ptr, self.mixed.ptr, hh, kvh, kdd, vdd, cdim);
                            chk(c.launch("k_delta_step", (*heads as u32, 1, 1), (*vd as u32, 1, 1), (2 * kd * 4) as u32, &mut a));
                        });
                        // gated norm with z = big_a[cd..cd+vt]
                        self.prof(4, || {
                            let zp = self.big_a.ptr + (cd * 4) as u64;
                            let (hh, vdd) = (*heads as u32, *vd as u32);
                            let mut a = args!(self.mixed.ptr, gated_w.ptr, zp, hh, vdd, eps);
                            chk(c.launch("k_gated_norm", (*heads as u32, 1, 1), ((*vd as u32).max(32), 1, 1), 0, &mut a));
                            chk(c.quantize_q8(&self.mixed, 0, &self.xq, &self.xs, 1, vt as u32));
                        });
                    }
                    // out projection
                    self.prof(1, || chk(c.matvec_q8(&out.q, &out.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, vt as u32)));
                    // residual + post norm
                    self.prof(0, || {
                        chk(c.add_rmsnorm_quant_rows(&self.hidden, Some(&self.y_d), post_norm, None, &self.xq, &self.xs, 1, d as u32, eps, true));
                    });
                    // MLP
                    self.prof(6, || chk(c.matvec_fp4(&gu.p, &gu.s, &self.xq, &self.xs, &self.big_a, 0, (2 * inter) as u32, d as u32)));
                    self.prof(7, || chk(c.silu_mul_quant_rows(&self.big_a, &self.big_b, &self.xq, &self.xs, 1, *inter as u32)));
                    self.prof(6, || chk(c.matvec_fp4(&dn.p, &dn.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, *inter as u32)));
                    if trace {
                        chk(c.add(&self.hidden, 0, &self.y_d, 0, d as u32));
                    }
                }
                LayerDev::Full { in_norm, post_norm, proj, o, gu, dn, q_norm, k_norm, kc, vc, n_heads, n_kv, hd, rope_dim, theta, inter } => {
                    let qw = n_heads * hd;
                    let kvw = n_kv * hd;
                    self.prof(0, || {
                        chk(c.add_rmsnorm_quant_rows(&self.hidden, prev, in_norm, None, &self.xq, &self.xs, 1, d as u32, eps, true));
                    });
                    self.prof(1, || chk(c.matvec_q8(&proj.q, &proj.s, &self.xq, &self.xs, &self.big_a, 0, proj.rows as u32, d as u32)));
                    // q/k norms + rope; K/V rows into the cache at pos; attention over [0, pos]
                    self.prof(5, || {
                        let (kvw_, nh, nkv, hdd, rd) = (kvw as u32, *n_heads as u32, *n_kv as u32, *hd as u32, *rope_dim as u32);
                        let _ = theta;
                        let mut a = args!(self.big_a.ptr, q_norm.ptr, k_norm.ptr, self.q_buf.ptr, self.gate_buf.ptr, kc.ptr, vc.ptr, kvw_, self.pos_dev.ptr, nh, nkv, hdd, rd, self.rope.ptr, eps);
                        chk(c.launch("k_qk_prep", (1, (*n_heads + *n_kv) as u32, 1), (*hd as u32, 1, 1), 0, &mut a));
                        let (kvw_, nh, nkv, hdd) = (kvw as u32, *n_heads as u32, *n_kv as u32, *hd as u32);
                        let mut a = args!(self.q_buf.ptr, self.gate_buf.ptr, kc.ptr, vc.ptr, self.mixed.ptr, kvw_, self.pos_dev.ptr, nh, nkv, hdd);
                        chk(c.launch("k_attn_decode", (*n_heads as u32, 1, 1), (*hd as u32, 1, 1), (self.cap * 4) as u32, &mut a));
                        chk(c.quantize_q8(&self.mixed, 0, &self.xq, &self.xs, 1, qw as u32));
                    });
                    self.prof(1, || chk(c.matvec_q8(&o.q, &o.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, qw as u32)));
                    self.prof(0, || {
                        chk(c.add_rmsnorm_quant_rows(&self.hidden, Some(&self.y_d), post_norm, None, &self.xq, &self.xs, 1, d as u32, eps, true));
                    });
                    self.prof(6, || chk(c.matvec_fp4(&gu.p, &gu.s, &self.xq, &self.xs, &self.big_a, 0, (2 * inter) as u32, d as u32)));
                    self.prof(7, || chk(c.silu_mul_quant_rows(&self.big_a, &self.big_b, &self.xq, &self.xs, 1, *inter as u32)));
                    self.prof(6, || chk(c.matvec_fp4(&dn.p, &dn.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, *inter as u32)));
                    if trace {
                        chk(c.add(&self.hidden, 0, &self.y_d, 0, d as u32));
                    }
                }
            }
            if trace {
                chk(c.copy_dtod(&self.trace_buf, li * d * 4, &self.hidden, 0, d * 4));
            }
        }
        // final norm (+ the last MLP residual) + head, logits out
        let last: Option<&DBuf> = if trace || self.layers.is_empty() { None } else { Some(&self.y_d) };
        self.prof(8, || {
            chk(c.add_rmsnorm_quant_rows(&self.hidden, last, &self.norm_f, None, &self.xq, &self.xs, 1, d as u32, eps, true));
            chk(c.matvec_q8(&self.lm_head.q, &self.lm_head.s, &self.xq, &self.xs, &self.logits, 0, self.vocab as u32, d as u32));
            chk(c.read_async(&self.logits, 0, &self.logits_pin, self.vocab * 4));
        });
        prof_report("token");
        ok.get()
    }

    fn workspace(&mut self, t: usize) -> Option<()> {
        if self.ws.as_ref().map(|w| w.t_cap >= t).unwrap_or(false) {
            return Some(());
        }
        let c = ctx()?;
        self.ws = None;
        let cap = t.next_power_of_two().max(64);
        let d = self.d;
        let ws = PrefillWs {
            t_cap: cap,
            hidden: c.alloc(cap * d * 4)?,
            xq: c.alloc(cap * self.max_cols)?,
            xs: c.alloc(cap * self.max_cols / 32 * 4)?,
            big_a: c.alloc(cap * self.max_rows * 4)?,
            big_b: c.alloc(cap * self.max_rows * 4)?,
            y: c.alloc(cap * d * 4)?,
            q: c.alloc(cap * self.max_qw.max(1) * 4)?,
            gate: c.alloc(cap * self.max_qw.max(1) * 4)?,
            mixed: c.alloc(cap * self.max_mixed.max(1) * 4)?,
            beta: c.alloc(cap * self.max_heads.max(1) * 4)?,
            decay: c.alloc(cap * self.max_heads.max(1) * 4)?,
        };
        self.ws = Some(ws);
        Some(())
    }

    /// Under MICROKIMI_CUDA_PROF=1: synchronizes and adds the elapsed
    /// wall time of `f` to the named slot (the prompt runs slower with
    /// the syncs; the split is what it shows).
    fn prof<F: FnOnce()>(&self, slot: usize, f: F) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var("MICROKIMI_CUDA_PROF").map(|v| v == "1").unwrap_or(false));
        if !on {
            f();
            return;
        }
        let c = ctx().unwrap();
        c.sync();
        let t0 = std::time::Instant::now();
        f();
        c.sync();
        PROF[slot].fetch_add(t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// The whole prompt on the GPU: every layer over `tokens` (GEMMs,
    /// the conv and the delta scan sequential in time, causal attention
    /// with the new rows appended to the resident KV cache), the state
    /// advanced by `tokens.len()`, the last position's logits returned.
    /// None = refused before any state mutation (capacity, memory).
    pub fn prefill(&mut self, m: &DecodeModelRefs, tokens: &[u32]) -> Option<Vec<f32>> {
        let c = ctx()?;
        let t = tokens.len();
        if t == 0 {
            return None;
        }
        let d = self.d;
        let pos0 = self.pos;
        if pos0 + t > self.cap {
            return None;
        }
        self.workspace(t)?;
        let ws = self.ws.as_ref()?;
        // embeddings, gathered on the host
        let mut emb = vec![0f32; t * d];
        for (i, &tok) in tokens.iter().enumerate() {
            emb[i * d..(i + 1) * d].copy_from_slice(&m.embed[tok as usize * d..(tok as usize + 1) * d]);
        }
        if !c.write(&ws.hidden, 0, &emb) {
            return None;
        }
        let eps = self.eps;
        let tu = t as u32;
        let ok = std::cell::Cell::new(true);
        let chk = |b: bool| {
            if !b {
                ok.set(false);
            }
        };
        let mut first = true;
        for l in self.layers.iter() {
            match l {
                LayerDev::Linear { in_norm, post_norm, proj, out, gu, dn, conv_w, a_log, dt_bias, gated_w, conv_state, scan_state, heads, kv_heads, kd, vd, conv_k, inter } => {
                    let kt = kv_heads * kd;
                    let vt = heads * vd;
                    let cd = 2 * kt + vt;
                    let pr = proj.rows;
                    // (residual of the previous MLP) + input norm
                    self.prof(0, || chk(c.add_rmsnorm_quant_rows(&ws.hidden, if first { None } else { Some(&ws.y) }, in_norm, None, &ws.xq, &ws.xs, tu, d as u32, eps, true)));
                    self.prof(1, || chk(c.gemm_q8(&proj.q, &proj.s, &ws.xq, &ws.xs, &ws.big_a, 0, pr as u32, d as u32, tu, pr as u32)));
                    // conv + silu over time -> big_b [t][cd]; q/k norms, beta, decay per (token, head)
                    self.prof(2, || {
                        let (k, cdim, ldx, ldo) = (*conv_k as u32, cd as u32, pr as u32, cd as u32);
                        let mut a = args!(ws.big_a.ptr, ldx, conv_w.ptr, k, conv_state.ptr, ws.big_b.ptr, ldo, cdim, tu);
                        chk(c.launch("k_conv_prefill", (cdim.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a));
                        let bp = ws.big_a.ptr + ((cd + vt) * 4) as u64;
                        let ap = ws.big_a.ptr + ((cd + vt + heads) * 4) as u64;
                        let (hh, kvh, kdd, ldc, ldba) = (*heads as u32, *kv_heads as u32, *kd as u32, cd as u32, pr as u32);
                        let mut a = args!(ws.big_b.ptr, ldc, bp, ap, ldba, a_log.ptr, dt_bias.ptr, ws.q.ptr, ws.gate.ptr, ws.beta.ptr, ws.decay.ptr, tu, hh, kvh, kdd);
                        chk(c.launch("k_lin_prep", (tu, *heads as u32, 1), ((*kd as u32).max(32), 1, 1), 0, &mut a));
                    });
                    // delta scan per head over time -> mixed [t][heads*vd]
                    self.prof(3, || {
                        let (hh, kvh, kdd, vdd, ldc) = (*heads as u32, *kv_heads as u32, *kd as u32, *vd as u32, cd as u32);
                        if (*kd == 128 || *kd == 64) && *vd <= 256 {
                            let name = if *kd == 128 { "k_delta_scan4x128" } else { "k_delta_scan4x64" };
                            let mut a = args!(scan_state.ptr, ws.q.ptr, ws.gate.ptr, ws.big_b.ptr, ldc, ws.beta.ptr, ws.decay.ptr, ws.mixed.ptr, tu, hh, kvh, vdd);
                            chk(c.launch(name, (*heads as u32, 1, 1), (4 * *vd as u32, 1, 1), 0, &mut a));
                        } else if *kd == 128 || *kd == 64 {
                            let name = if *kd == 128 { "k_delta_scan128" } else { "k_delta_scan64" };
                            let mut a = args!(scan_state.ptr, ws.q.ptr, ws.gate.ptr, ws.big_b.ptr, ldc, ws.beta.ptr, ws.decay.ptr, ws.mixed.ptr, tu, hh, kvh, vdd);
                            chk(c.launch(name, (*heads as u32, 1, 1), (*vd as u32, 1, 1), 0, &mut a));
                        } else {
                            let mut a = args!(scan_state.ptr, ws.q.ptr, ws.gate.ptr, ws.big_b.ptr, ldc, ws.beta.ptr, ws.decay.ptr, ws.mixed.ptr, tu, hh, kvh, kdd, vdd);
                            chk(c.launch("k_delta_scan", (*heads as u32, 1, 1), (*vd as u32, 1, 1), (2 * kd * 4) as u32, &mut a));
                        }
                    });
                    // gated norm with z = big_a[.., cd..cd+vt]
                    self.prof(4, || {
                        let zp = ws.big_a.ptr + (cd * 4) as u64;
                        let (hh, vdd, ldz) = (*heads as u32, *vd as u32, pr as u32);
                        if vd % 32 == 0 && *vd <= 1024 {
                            let mut a = args!(ws.mixed.ptr, gated_w.ptr, zp, ldz, ws.xq.ptr, ws.xs.ptr, tu, hh, vdd, eps);
                            chk(c.launch("k_gated_norm_quant_rows", (tu, *heads as u32, 1), (*vd as u32, 1, 1), 0, &mut a));
                        } else {
                            let mut a = args!(ws.mixed.ptr, gated_w.ptr, zp, ldz, tu, hh, vdd, eps);
                            chk(c.launch("k_gated_norm_rows", (tu, *heads as u32, 1), ((*vd as u32).max(32), 1, 1), 0, &mut a));
                            chk(c.quantize_q8(&ws.mixed, 0, &ws.xq, &ws.xs, tu, vt as u32));
                        }
                    });
                    self.prof(1, || chk(c.gemm_q8(&out.q, &out.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, vt as u32, tu, d as u32)));
                    self.prof(0, || chk(c.add_rmsnorm_quant_rows(&ws.hidden, Some(&ws.y), post_norm, None, &ws.xq, &ws.xs, tu, d as u32, eps, true)));
                    let half = std::cell::Cell::new(false);
                    self.prof(6, || {
                        let (ok, h16) = c.gemm_fp4_gu(&gu.p, &gu.s, &ws.xq, &ws.xs, &ws.big_a, (2 * inter) as u32, d as u32, tu);
                        chk(ok);
                        half.set(h16);
                    });
                    self.prof(7, || {
                        if half.get() {
                            chk(c.silu_mul_quant_rows_h(&ws.big_a, &ws.big_b, &ws.xq, &ws.xs, tu, *inter as u32));
                        } else {
                            chk(c.silu_mul_quant_rows(&ws.big_a, &ws.big_b, &ws.xq, &ws.xs, tu, *inter as u32));
                        }
                    });
                    self.prof(6, || chk(c.gemm_fp4(&dn.p, &dn.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, *inter as u32, tu, d as u32)));
                }
                LayerDev::Full { in_norm, post_norm, proj, o, gu, dn, q_norm, k_norm, kc, vc, n_heads, n_kv, hd, rope_dim, theta, inter } => {
                    let qw = n_heads * hd;
                    let kvw = n_kv * hd;
                    let pr = proj.rows;
                    self.prof(0, || chk(c.add_rmsnorm_quant_rows(&ws.hidden, if first { None } else { Some(&ws.y) }, in_norm, None, &ws.xq, &ws.xs, tu, d as u32, eps, true)));
                    self.prof(1, || chk(c.gemm_q8(&proj.q, &proj.s, &ws.xq, &ws.xs, &ws.big_a, 0, pr as u32, d as u32, tu, pr as u32)));
                    self.prof(5, || {
                        let (ldq, kvw_, p0, nh, nkv, hdd, rd) = (pr as u32, kvw as u32, pos0 as u32, *n_heads as u32, *n_kv as u32, *hd as u32, *rope_dim as u32);
                        let _ = theta;
                        let mut a = args!(ws.big_a.ptr, ldq, q_norm.ptr, k_norm.ptr, ws.q.ptr, ws.gate.ptr, kc.ptr, vc.ptr, kvw_, tu, p0, nh, nkv, hdd, rd, self.rope.ptr, eps);
                        chk(c.launch("k_qk_prep_rows", (tu, (*n_heads + *n_kv) as u32, 1), (*hd as u32, 1, 1), 0, &mut a));
                        let (kvw_, p0, nh, nkv, hdd) = (kvw as u32, pos0 as u32, *n_heads as u32, *n_kv as u32, *hd as u32);
                        let groups = n_heads / n_kv;
                        let npairs = groups * 4;
                        let len_max = pos0 + t;
                        // grouped kernel: shared = queries + scores + sums; per-(token, head) kernel beyond 48 KB
                        let pp4 = (npairs + 3) & !3;
                        let shared_g = (len_max * pp4 * 4 + pp4 * 4) as u32;
                        if *hd <= 256 && hd % 32 == 0 && npairs <= 32 && shared_g <= 96 * 1024 {
                            // the grouped kernel quantizes its output itself
                            let mut a = args!(ws.q.ptr, ws.gate.ptr, kc.ptr, vc.ptr, ws.mixed.ptr, ws.xq.ptr, ws.xs.ptr, kvw_, p0, tu, nh, nkv, hdd);
                            chk(c.launch("k_attn_prefill_grouped", (*n_kv as u32, tu.div_ceil(4), 1), (256, 1, 1), shared_g, &mut a));
                        } else {
                            let mut a = args!(ws.q.ptr, ws.gate.ptr, kc.ptr, vc.ptr, ws.mixed.ptr, kvw_, p0, tu, nh, nkv, hdd);
                            chk(c.launch("k_attn_prefill", (tu, *n_heads as u32, 1), (*hd as u32, 1, 1), ((pos0 + t) * 4) as u32, &mut a));
                            chk(c.quantize_q8(&ws.mixed, 0, &ws.xq, &ws.xs, tu, qw as u32));
                        }
                    });
                    self.prof(1, || chk(c.gemm_q8(&o.q, &o.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, qw as u32, tu, d as u32)));
                    self.prof(0, || chk(c.add_rmsnorm_quant_rows(&ws.hidden, Some(&ws.y), post_norm, None, &ws.xq, &ws.xs, tu, d as u32, eps, true)));
                    let half = std::cell::Cell::new(false);
                    self.prof(6, || {
                        let (ok, h16) = c.gemm_fp4_gu(&gu.p, &gu.s, &ws.xq, &ws.xs, &ws.big_a, (2 * inter) as u32, d as u32, tu);
                        chk(ok);
                        half.set(h16);
                    });
                    self.prof(7, || {
                        if half.get() {
                            chk(c.silu_mul_quant_rows_h(&ws.big_a, &ws.big_b, &ws.xq, &ws.xs, tu, *inter as u32));
                        } else {
                            chk(c.silu_mul_quant_rows(&ws.big_a, &ws.big_b, &ws.xq, &ws.xs, tu, *inter as u32));
                        }
                    });
                    self.prof(6, || chk(c.gemm_fp4(&dn.p, &dn.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, *inter as u32, tu, d as u32)));
                }
            }
            first = false;
        }
        // last residual, final norm of the last position, head
        chk(c.add(&ws.hidden, (t - 1) * d * 4, &ws.y, (t - 1) * d * 4, d as u32));
        {
            let hp = ws.hidden.ptr + ((t - 1) * d * 4) as u64;
            let (n, op) = (d as u32, 1u32);
            let mut a = args!(hp, self.norm_f.ptr, self.normed.ptr, n, eps, op);
            chk(c.launch("k_rmsnorm", (1, 1, 1), (1024, 1, 1), 0, &mut a));
        }
        self.prof(8, || {
            chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
            chk(c.matvec_q8(&self.lm_head.q, &self.lm_head.s, &self.xq, &self.xs, &self.logits, 0, self.vocab as u32, d as u32));
        });
        if !ok.get() {
            return None;
        }
        let mut logits = vec![0f32; self.vocab];
        if !c.read(&self.logits, 0, &mut logits) {
            return None;
        }
        prof_report(&format!("prefill {t} tokens"));
        self.pos += t;
        Some(logits)
    }

    /// Copies the resident state back to host layouts: every linear
    /// layer's (conv, scan) state, and for every full layer the KV rows
    /// [from, pos) appended to the given vectors (`from` = the cache's
    /// current row count). Returns the position.
    pub fn export(&self, lin_states: &mut [(&mut [f32], &mut [f32])], full_kv: &mut [(&mut Vec<f32>, &mut Vec<f32>, usize)], kv_width: usize) -> usize {
        let Some(c) = ctx() else {
            return self.pos;
        };
        let (mut li, mut fi) = (0usize, 0usize);
        for l in &self.layers {
            match l {
                LayerDev::Linear { conv_state, scan_state, .. } => {
                    let (conv, state) = &mut lin_states[li];
                    li += 1;
                    c.read(conv_state, 0, conv);
                    c.read(scan_state, 0, state);
                }
                LayerDev::Full { kc, vc, .. } => {
                    let (k, v, from) = &mut full_kv[fi];
                    fi += 1;
                    let to = self.pos.min(self.cap);
                    if *from >= to {
                        continue;
                    }
                    let n = (to - *from) * kv_width;
                    let mut tmp = vec![0f32; n];
                    c.read(kc, *from * kv_width * 4, &mut tmp);
                    k.truncate(*from * kv_width);
                    k.extend_from_slice(&tmp);
                    c.read(vc, *from * kv_width * 4, &mut tmp);
                    v.truncate(*from * kv_width);
                    v.extend_from_slice(&tmp);
                }
            }
        }
        self.pos
    }
}
