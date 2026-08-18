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
}

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
    "k_gemm_fp4_mma64",
    "k_gemm_fp4_mma128",
    "k_gemm_fp4_mma256",
    "k_add",
    "k_rmsnorm",
    "k_rmsnorm_rows",
    "k_silu_mul",
    "k_conv_silu",
    "k_delta_step",
    "k_qk_prep",
    "k_attn_decode",
    "k_attn_prefill",
    "k_gated_norm",
    "k_scale_rows",
    "k_lin_prep",
    "k_conv_prefill",
    "k_delta_scan",
    "k_gated_norm_rows",
    "k_silu_mul_rows",
    "k_add_rmsnorm_rows",
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
        h.update(format!("sm{major}{minor}-nvrtc{nmaj}.{nmin}-v3").as_bytes());
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
    let opts: Vec<CString> = ["-default-device", "-std=c++17", "-lineinfo", &arch].iter().map(|s| CString::new(*s).unwrap()).collect();
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
        let mut a = args!(wq.ptr, ws.ptr, xq.ptr, xs.ptr, yp, rows, cols);
        let shared = cols + cols / 32 * 4;
        self.launch("k_matvec_q8", (rows.div_ceil(4), 1, 1), (128, 1, 1), shared, &mut a)
    }
    /// y[rows] = W(MXFP4 rows x cols) . x(q8). One warp per row.
    pub fn matvec_fp4(&self, wp: &DBuf, wsc: &DBuf, xq: &DBuf, xs: &DBuf, y: &DBuf, y_off: usize, rows: u32, cols: u32) -> bool {
        let yp = y.ptr + y_off as u64;
        let mut a = args!(wp.ptr, wsc.ptr, xq.ptr, xs.ptr, yp, rows, cols);
        let shared = cols + cols / 32 * 4;
        self.launch("k_matvec_fp4", (rows.div_ceil(4), 1, 1), (128, 1, 1), shared, &mut a)
    }
    /// C[t][rows] (row stride `ldc`) = X(q8, t rows of cols) . W(q8 rows x cols)^T.
    pub fn gemm_q8(&self, wq: &DBuf, ws: &DBuf, xq: &DBuf, xs: &DBuf, c: &DBuf, c_off: usize, rows: u32, cols: u32, t: u32, ldc: u32) -> bool {
        let cp = c.ptr + c_off as u64;
        let mut a = args!(wq.ptr, ws.ptr, xq.ptr, xs.ptr, cp, rows, cols, t, ldc);
        if self.mma_on() {
            let (name, nt) = if t <= 64 { ("k_gemm_q8_mma64", 64) } else if t <= 128 { ("k_gemm_q8_mma128", 128) } else { ("k_gemm_q8_mma256", 256) };
            return self.launch(name, (rows.div_ceil(64), t.div_ceil(nt), 1), (256, 1, 1), 0, &mut a);
        }
        self.launch("k_gemm_q8", (rows.div_ceil(64), t.div_ceil(64), 1), (256, 1, 1), 0, &mut a)
    }
    /// C[t][rows] = X(q8) . W(MXFP4)^T.
    pub fn gemm_fp4(&self, wp: &DBuf, wsc: &DBuf, xq: &DBuf, xs: &DBuf, c: &DBuf, c_off: usize, rows: u32, cols: u32, t: u32, ldc: u32) -> bool {
        let cp = c.ptr + c_off as u64;
        let mut a = args!(wp.ptr, wsc.ptr, xq.ptr, xs.ptr, cp, rows, cols, t, ldc);
        if self.mma_on() {
            let (name, nt) = if t <= 64 { ("k_gemm_fp4_mma64", 64) } else if t <= 128 { ("k_gemm_fp4_mma128", 128) } else { ("k_gemm_fp4_mma256", 256) };
            return self.launch(name, (rows.div_ceil(64), t.div_ceil(nt), 1), (256, 1, 1), 0, &mut a);
        }
        self.launch("k_gemm_fp4", (rows.div_ceil(64), t.div_ceil(64), 1), (256, 1, 1), 0, &mut a)
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
    /// out = rmsnorm(x + add?) * (one_plus ? 1 + w : w); rows of n, one block per row.
    /// `add` may be null (0) - then x alone; when given, x += add first (in place).
    pub fn add_rmsnorm_rows(&self, x: &DBuf, add: Option<&DBuf>, w: &DBuf, out: &DBuf, rows: u32, n: u32, eps: f32, one_plus: bool) -> bool {
        let ap: CUdeviceptr = add.map(|b| b.ptr).unwrap_or(0);
        let op: u32 = if one_plus { 1 } else { 0 };
        let mut a = args!(x.ptr, ap, w.ptr, out.ptr, rows, n, eps, op);
        self.launch("k_add_rmsnorm_rows", (rows, 1, 1), (1024, 1, 1), 0, &mut a)
    }
    /// h[t][inter] = silu(gu[t][i]) * gu[t][inter + i]
    pub fn silu_mul_rows(&self, gu: &DBuf, h: &DBuf, rows: u32, inter: u32) -> bool {
        let n = rows * inter;
        let mut a = args!(gu.ptr, h.ptr, rows, inter);
        self.launch("k_silu_mul_rows", (n.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a)
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
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }
// 2^(e-128) exactly (e8m0 scale byte, LUT2 convention: values doubled)
__device__ __forceinline__ float e8m0_x2(int e) { return ldexpf(1.0f, e - 128); }

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
extern "C" __global__ void k_matvec_q8(const i8* __restrict__ wq, const float* __restrict__ ws,
                                       const i8* __restrict__ xq, const float* __restrict__ xs,
                                       float* __restrict__ y, unsigned rows, unsigned cols) {
    extern __shared__ __align__(16) unsigned char smem[];
    int* sx = (int*)smem;                       // cols bytes as ints
    float* sxs = (float*)(smem + cols);         // cols/32 scales
    const unsigned nb = cols >> 5;
    for (unsigned i = threadIdx.x; i < (cols >> 2); i += blockDim.x) sx[i] = ((const int*)xq)[i];
    for (unsigned i = threadIdx.x; i < nb; i += blockDim.x) sxs[i] = xs[i];
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    const int4* wrow = (const int4*)(wq + (size_t)row * cols);
    const float* srow = ws + (size_t)row * nb;
    float acc = 0.0f;
    for (unsigned g = lane; g < nb; g += 32) {
        int4 a = wrow[g * 2], b = wrow[g * 2 + 1];
        const int* xb = sx + g * 8;
        int d = 0;
        d = __dp4a(a.x, xb[0], d); d = __dp4a(a.y, xb[1], d); d = __dp4a(a.z, xb[2], d); d = __dp4a(a.w, xb[3], d);
        d = __dp4a(b.x, xb[4], d); d = __dp4a(b.y, xb[5], d); d = __dp4a(b.z, xb[6], d); d = __dp4a(b.w, xb[7], d);
        acc = fmaf((float)d, srow[g] * sxs[g], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

extern "C" __global__ void k_matvec_fp4(const u8* __restrict__ wp, const u8* __restrict__ wsc,
                                        const i8* __restrict__ xq, const float* __restrict__ xs,
                                        float* __restrict__ y, unsigned rows, unsigned cols) {
    extern __shared__ __align__(16) unsigned char smem[];
    int* sx = (int*)smem;
    float* sxs = (float*)(smem + cols);
    const unsigned nb = cols >> 5;
    for (unsigned i = threadIdx.x; i < (cols >> 2); i += blockDim.x) sx[i] = ((const int*)xq)[i];
    for (unsigned i = threadIdx.x; i < nb; i += blockDim.x) sxs[i] = xs[i];
    __syncthreads();
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    const int4* prow = (const int4*)(wp + (size_t)row * (cols >> 1));
    const u8* srow = wsc + (size_t)row * nb;
    float acc = 0.0f;
    for (unsigned g = lane; g < nb; g += 32) {
        int4 pk = prow[g];
        const int* xb = sx + g * 8;
        int w0, w1, w2, w3, w4, w5, w6, w7;
        fp4_decode8((unsigned)pk.x, w0, w1);
        fp4_decode8((unsigned)pk.y, w2, w3);
        fp4_decode8((unsigned)pk.z, w4, w5);
        fp4_decode8((unsigned)pk.w, w6, w7);
        int d = 0;
        d = __dp4a(w0, xb[0], d); d = __dp4a(w1, xb[1], d); d = __dp4a(w2, xb[2], d); d = __dp4a(w3, xb[3], d);
        d = __dp4a(w4, xb[4], d); d = __dp4a(w5, xb[5], d); d = __dp4a(w6, xb[6], d); d = __dp4a(w7, xb[7], d);
        acc = fmaf((float)d, e8m0_x2(srow[g]) * sxs[g], acc);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[row] = acc;
}

// ── GEMM: C[t][r] = sum_g ws[r][g] xs[t][g] dot(W[r][g], X[t][g]) ──
// tiles of 64 rows x 64 tokens, k-step of one 32-column block; 256 threads,
// each a 4x4 (rows x tokens) micro-tile. W tile [64][32] i8, X tile [64][32] i8.
#define GT 64
extern "C" __global__ void k_gemm_q8(const i8* __restrict__ wq, const float* __restrict__ ws,
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
                sws[tid] = (grow < rows) ? ws[(size_t)grow * nb + g] : 0.0f;
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
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
}
#define MT 64
#define MKB 2  /* blocks per staging step */
#define SROW 9 /* shared row stride in ints per block (8 + 1 pad against bank conflicts) */
#define LDS (MKB * SROW)

template <int NT>
__device__ __forceinline__ void gemm_mma_body(const int* sw, const int* sxt, const float* sws, const float* sxs,
                                              float (&acc)[2][NT / 32][4], unsigned warp, unsigned lane) {
    // warp tile: rows wr0 = (warp & 1) * 32, tokens wt0 = (warp >> 1) * (NT / 4)
    constexpr int NW = NT / 32; // n8 tiles per warp
    const unsigned wr0 = (warp & 1) * 32, wt0 = (warp >> 1) * (NT / 4);
    const unsigned gid = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int kb = 0; kb < MKB; kb++) {
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
            float ws0 = sws[kb * MT + wr0 + mi * 16 + gid];
            float ws1 = sws[kb * MT + wr0 + mi * 16 + gid + 8];
            #pragma unroll
            for (int ni = 0; ni < NW; ni++) {
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
template <int NT>
__device__ __forceinline__ void gemm_mma_store(float (&acc)[2][NT / 32][4], float* __restrict__ C, unsigned r0, unsigned t0,
                                               unsigned rows, unsigned t, unsigned ldc, unsigned warp, unsigned lane) {
    constexpr int NW = NT / 32;
    const unsigned wr0 = (warp & 1) * 32, wt0 = (warp >> 1) * (NT / 4);
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
// staging of the X tile (NT tokens x MKB blocks) and its scales
template <int NT>
__device__ __forceinline__ void stage_x(int* sxt, float* sxs, const i8* __restrict__ xq, const float* __restrict__ xs,
                                        unsigned t0, unsigned t, unsigned cols, unsigned nb, unsigned g0, unsigned tid) {
    #pragma unroll
    for (int rep = 0; rep < (NT * MKB * 8) / 256; rep++) {
        unsigned idx = tid + rep * 256;
        unsigned rr = idx / (MKB * 8), kk = idx % (MKB * 8);
        unsigned kb = kk >> 3, k = kk & 7;
        unsigned g = g0 + kb;
        int xv = 0;
        unsigned gtok = t0 + rr;
        if (g < nb && gtok < t) xv = ((const int*)(xq + (size_t)gtok * cols + g * 32))[k];
        sxt[rr * LDS + kb * SROW + k] = xv;
    }
    for (unsigned i = tid; i < MKB * NT; i += 256) {
        unsigned kb = i / NT, rr = i % NT;
        unsigned g = g0 + kb, gtok = t0 + rr;
        sxs[i] = (g < nb && gtok < t) ? xs[(size_t)gtok * nb + g] : 0.0f;
    }
}
template <int NT>
__device__ __forceinline__ void gemm_q8_mma_t(const i8* __restrict__ wq, const float* __restrict__ ws,
                                              const i8* __restrict__ xq, const float* __restrict__ xs,
                                              float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    __shared__ __align__(16) int sw[MT * LDS];
    __shared__ __align__(16) int sxt[NT * LDS];
    __shared__ float sws[MKB * MT], sxs[MKB * NT];
    const unsigned nb = cols >> 5;
    const unsigned r0 = blockIdx.x * MT, t0 = blockIdx.y * NT;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float acc[2][NT / 32][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < NT / 32; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    for (unsigned g0 = 0; g0 < nb; g0 += MKB) {
        #pragma unroll
        for (int rep = 0; rep < (MT * MKB * 8) / 256; rep++) {
            unsigned idx = tid + rep * 256;
            unsigned rr = idx / (MKB * 8), kk = idx % (MKB * 8);
            unsigned kb = kk >> 3, k = kk & 7;
            unsigned g = g0 + kb;
            int wv = 0;
            unsigned grow = r0 + rr;
            if (g < nb && grow < rows) wv = ((const int*)(wq + (size_t)grow * cols + g * 32))[k];
            sw[rr * LDS + kb * SROW + k] = wv;
        }
        if (tid < MKB * MT) {
            unsigned kb = tid / MT, rr = tid % MT;
            unsigned g = g0 + kb, grow = r0 + rr;
            sws[tid] = (g < nb && grow < rows) ? ws[(size_t)grow * nb + g] : 0.0f;
        }
        stage_x<NT>(sxt, sxs, xq, xs, t0, t, cols, nb, g0, tid);
        __syncthreads();
        gemm_mma_body<NT>(sw, sxt, sws, sxs, acc, warp, lane);
        __syncthreads();
    }
    gemm_mma_store<NT>(acc, C, r0, t0, rows, t, ldc, warp, lane);
}
template <int NT>
__device__ __forceinline__ void gemm_fp4_mma_t(const u8* __restrict__ wp, const u8* __restrict__ wsc,
                                               const i8* __restrict__ xq, const float* __restrict__ xs,
                                               float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) {
    __shared__ __align__(16) int sw[MT * LDS];
    __shared__ __align__(16) int sxt[NT * LDS];
    __shared__ float sws[MKB * MT], sxs[MKB * NT];
    const unsigned nb = cols >> 5;
    const unsigned r0 = blockIdx.x * MT, t0 = blockIdx.y * NT;
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float acc[2][NT / 32][4];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < NT / 32; j++)
            #pragma unroll
            for (int k = 0; k < 4; k++) acc[i][j][k] = 0.0f;
    for (unsigned g0 = 0; g0 < nb; g0 += MKB) {
        #pragma unroll
        for (int rep = 0; rep < (MT * MKB * 4) / 256; rep++) {
            unsigned idx = tid + rep * 256;
            unsigned rr = idx / (MKB * 4), kk = idx % (MKB * 4);
            unsigned kb = kk >> 2, k = kk & 3;
            unsigned g = g0 + kb;
            unsigned pk = 0u;
            unsigned grow = r0 + rr;
            if (g < nb && grow < rows) pk = ((const unsigned*)(wp + (size_t)grow * (cols >> 1) + g * 16))[k];
            int lo, hi;
            fp4_decode8(pk, lo, hi);
            sw[rr * LDS + kb * SROW + k * 2] = lo;
            sw[rr * LDS + kb * SROW + k * 2 + 1] = hi;
        }
        if (tid < MKB * MT) {
            unsigned kb = tid / MT, rr = tid % MT;
            unsigned g = g0 + kb, grow = r0 + rr;
            sws[tid] = (g < nb && grow < rows) ? e8m0_x2(wsc[(size_t)grow * nb + g]) : 0.0f;
        }
        stage_x<NT>(sxt, sxs, xq, xs, t0, t, cols, nb, g0, tid);
        __syncthreads();
        gemm_mma_body<NT>(sw, sxt, sws, sxs, acc, warp, lane);
        __syncthreads();
    }
    gemm_mma_store<NT>(acc, C, r0, t0, rows, t, ldc, warp, lane);
}
extern "C" __global__ void k_gemm_q8_mma64(const i8* __restrict__ wq, const float* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<64>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void k_gemm_q8_mma128(const i8* __restrict__ wq, const float* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<128>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void k_gemm_q8_mma256(const i8* __restrict__ wq, const float* __restrict__ ws, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_q8_mma_t<256>(wq, ws, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void k_gemm_fp4_mma64(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<64>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void k_gemm_fp4_mma128(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<128>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }
extern "C" __global__ void k_gemm_fp4_mma256(const u8* __restrict__ wp, const u8* __restrict__ wsc, const i8* __restrict__ xq, const float* __restrict__ xs, float* __restrict__ C, unsigned rows, unsigned cols, unsigned t, unsigned ldc) { gemm_fp4_mma_t<256>(wp, wsc, xq, xs, C, rows, cols, t, ldc); }

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
    // k <= 8 taps kept in registers
    float s[8];
    for (unsigned j = 0; j < k - 1; j++) s[j] = st[j];
    for (unsigned tok = 0; tok < t; tok++) {
        float xi = x[(size_t)tok * ldx + i];
        float acc = 0.0f;
        for (unsigned j = 0; j < k - 1; j++) acc += s[j] * wt[j];
        acc += xi * wt[k - 1];
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
                                          unsigned t, unsigned pos0, unsigned n_heads, unsigned n_kv, unsigned hd, unsigned rope_dim, float theta, float eps) {
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
        if (i < half) {
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
            float a = sq[i], b = sq[i + half];
            outv = a * c - b * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i0 / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
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
        if (i < half) {
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
            float a = sk[i], b = sk[i + half];
            outv = a * c - b * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i0 / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
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
                                     unsigned pos, unsigned n_heads, unsigned n_kv, unsigned hd, unsigned rope_dim, float theta, float eps) {
    // single-token alias of the rows form (t = 1)
    // (kept as its own entry point so the decode graph launches by name)
    __shared__ float red[32];
    unsigned hh = blockIdx.y;
    unsigned i = threadIdx.x;
    if (i >= hd) return;
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
        if (i < half) {
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
            outv = sq[i] * c - sq[i + half] * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i0 / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
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
        if (i < half) {
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
            outv = sk[i] * c - sk[i + half] * s;
        } else if (i < rope_dim) {
            unsigned i0 = i - half;
            double freq = 1.0 / pow((double)theta, 2.0 * (double)i0 / (double)rope_dim);
            double ang = (double)pos * freq;
            float s = (float)sin(ang), c = (float)cos(ang);
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
                                         float* __restrict__ mixed, unsigned kv_width, unsigned len, unsigned n_heads, unsigned n_kv, unsigned hd) {
    extern __shared__ float sc[];
    __shared__ float red[32];
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
    let (dwq, dws) = (c.upload(&wq).unwrap(), c.upload(&ws).unwrap());
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
    // speed: matvec bandwidth on a 1 GB set (q8) and 0.5 GB (fp4)
    let big_rows = 8 * 4096usize; // 8 matrices' worth as one tall matrix: 32768 x 5120 i8 = 168 MB.. use 6 copies
    let reps = 6usize;
    let mut bigq: Vec<i8> = Vec::with_capacity(reps * big_rows * cols);
    let mut bigs: Vec<f32> = Vec::with_capacity(reps * big_rows * cols / 32);
    for _ in 0..reps {
        for _ in 0..(big_rows / rows) {
            bigq.extend_from_slice(&wq);
            bigs.extend_from_slice(&ws);
        }
    }
    let brows = reps * big_rows;
    let dbq = c.upload(&bigq).unwrap();
    let dbs = c.upload(&bigs).unwrap();
    let dby = c.alloc(brows * 4).unwrap();
    let bytes = (bigq.len() + bigs.len() * 4) as f64;
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
    let gs: Vec<f32> = vec![0.01; gr * gc / 32];
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
    normed: DBuf, // [t][d]
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
    Some(WQ8 { q: c.upload(&q)?, s: c.upload(&s)?, rows, cols })
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
        let eps = self.eps;
        // embed on the host
        let row = &m.embed[token as usize * d..(token as usize + 1) * d];
        if !c.write(&self.hidden, 0, row) {
            return None;
        }
        let ok = std::cell::Cell::new(true);
        let chk = |b: bool| {
            if !b {
                ok.set(false);
            }
        };
        let n_layers = self.layers.len();
        for (li, l) in self.layers.iter().enumerate() {
            match l {
                LayerDev::Linear { in_norm, post_norm, proj, out, gu, dn, conv_w, a_log, dt_bias, gated_w, conv_state, scan_state, heads, kv_heads, kd, vd, conv_k, inter, .. } => {
                    let kt = kv_heads * kd;
                    let vt = heads * vd;
                    let cd = 2 * kt + vt;
                    // input norm
                    chk(c.add_rmsnorm_rows(&self.hidden, None, in_norm, &self.normed, 1, d as u32, eps, true));
                    chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
                    // fused projections -> big_a [qkv | z | b | a]
                    chk(c.matvec_q8(&proj.q, &proj.s, &self.xq, &self.xs, &self.big_a, 0, proj.rows as u32, d as u32));
                    // conv + silu -> big_b[0..cd]
                    {
                        let k = *conv_k as u32;
                        let cdim = cd as u32;
                        let mut a = args!(self.big_a.ptr, conv_w.ptr, k, conv_state.ptr, self.big_b.ptr, cdim);
                        chk(c.launch("k_conv_silu", (cdim.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a));
                    }
                    // q/k norms, beta, decay (t = 1): qn -> q_buf, kn -> gate_buf
                    {
                        let bp = self.big_a.ptr + ((cd + vt) * 4) as u64;
                        let ap = self.big_a.ptr + ((cd + vt + heads) * 4) as u64;
                        let (t1, hh, kvh, kdd, cdim, ldba) = (1u32, *heads as u32, *kv_heads as u32, *kd as u32, cd as u32, *heads as u32);
                        let mut a = args!(self.big_b.ptr, cdim, bp, ap, ldba, a_log.ptr, dt_bias.ptr, self.q_buf.ptr, self.gate_buf.ptr, self.beta.ptr, self.decay.ptr, t1, hh, kvh, kdd);
                        chk(c.launch("k_lin_prep", (1, *heads as u32, 1), ((*kd as u32).max(32), 1, 1), 0, &mut a));
                    }
                    // delta step per head -> mixed [heads*vd]
                    {
                        let (hh, kvh, kdd, vdd, cdim) = (*heads as u32, *kv_heads as u32, *kd as u32, *vd as u32, cd as u32);
                        let mut a = args!(scan_state.ptr, self.q_buf.ptr, self.gate_buf.ptr, self.big_b.ptr, self.beta.ptr, self.decay.ptr, self.mixed.ptr, hh, kvh, kdd, vdd, cdim);
                        chk(c.launch("k_delta_step", (*heads as u32, 1, 1), (*vd as u32, 1, 1), (2 * kd * 4) as u32, &mut a));
                    }
                    // gated norm with z = big_a[cd..cd+vt]
                    {
                        let zp = self.big_a.ptr + (cd * 4) as u64;
                        let (hh, vdd) = (*heads as u32, *vd as u32);
                        let mut a = args!(self.mixed.ptr, gated_w.ptr, zp, hh, vdd, eps);
                        chk(c.launch("k_gated_norm", (*heads as u32, 1, 1), ((*vd as u32).max(32), 1, 1), 0, &mut a));
                    }
                    // out projection
                    chk(c.quantize_q8(&self.mixed, 0, &self.xq, &self.xs, 1, vt as u32));
                    chk(c.matvec_q8(&out.q, &out.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, vt as u32));
                    // residual + post norm
                    chk(c.add_rmsnorm_rows(&self.hidden, Some(&self.y_d), post_norm, &self.normed, 1, d as u32, eps, true));
                    // MLP
                    chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
                    chk(c.matvec_fp4(&gu.p, &gu.s, &self.xq, &self.xs, &self.big_a, 0, (2 * inter) as u32, d as u32));
                    chk(c.silu_mul_rows(&self.big_a, &self.big_b, 1, *inter as u32));
                    chk(c.quantize_q8(&self.big_b, 0, &self.xq, &self.xs, 1, *inter as u32));
                    chk(c.matvec_fp4(&dn.p, &dn.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, *inter as u32));
                    chk(c.add(&self.hidden, 0, &self.y_d, 0, d as u32));
                }
                LayerDev::Full { in_norm, post_norm, proj, o, gu, dn, q_norm, k_norm, kc, vc, n_heads, n_kv, hd, rope_dim, theta, inter } => {
                    let qw = n_heads * hd;
                    let kvw = n_kv * hd;
                    chk(c.add_rmsnorm_rows(&self.hidden, None, in_norm, &self.normed, 1, d as u32, eps, true));
                    chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
                    chk(c.matvec_q8(&proj.q, &proj.s, &self.xq, &self.xs, &self.big_a, 0, proj.rows as u32, d as u32));
                    // q/k norms + rope; K/V rows into the cache at pos
                    {
                        let (kvw_, p, nh, nkv, hdd, rd) = (kvw as u32, pos as u32, *n_heads as u32, *n_kv as u32, *hd as u32, *rope_dim as u32);
                        let mut a = args!(self.big_a.ptr, q_norm.ptr, k_norm.ptr, self.q_buf.ptr, self.gate_buf.ptr, kc.ptr, vc.ptr, kvw_, p, nh, nkv, hdd, rd, *theta, eps);
                        chk(c.launch("k_qk_prep", (1, (*n_heads + *n_kv) as u32, 1), (*hd as u32, 1, 1), 0, &mut a));
                    }
                    // attention over [0, pos]
                    {
                        let len = (pos + 1) as u32;
                        let (kvw_, nh, nkv, hdd) = (kvw as u32, *n_heads as u32, *n_kv as u32, *hd as u32);
                        let mut a = args!(self.q_buf.ptr, self.gate_buf.ptr, kc.ptr, vc.ptr, self.mixed.ptr, kvw_, len, nh, nkv, hdd);
                        chk(c.launch("k_attn_decode", (*n_heads as u32, 1, 1), (*hd as u32, 1, 1), len * 4, &mut a));
                    }
                    chk(c.quantize_q8(&self.mixed, 0, &self.xq, &self.xs, 1, qw as u32));
                    chk(c.matvec_q8(&o.q, &o.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, qw as u32));
                    chk(c.add_rmsnorm_rows(&self.hidden, Some(&self.y_d), post_norm, &self.normed, 1, d as u32, eps, true));
                    chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
                    chk(c.matvec_fp4(&gu.p, &gu.s, &self.xq, &self.xs, &self.big_a, 0, (2 * inter) as u32, d as u32));
                    chk(c.silu_mul_rows(&self.big_a, &self.big_b, 1, *inter as u32));
                    chk(c.quantize_q8(&self.big_b, 0, &self.xq, &self.xs, 1, *inter as u32));
                    chk(c.matvec_fp4(&dn.p, &dn.s, &self.xq, &self.xs, &self.y_d, 0, d as u32, *inter as u32));
                    chk(c.add(&self.hidden, 0, &self.y_d, 0, d as u32));
                }
            }
            if trace.is_some() {
                chk(c.copy_dtod(&self.trace_buf, li * d * 4, &self.hidden, 0, d * 4));
            }
        }
        // final norm + head
        chk(c.add_rmsnorm_rows(&self.hidden, None, &self.norm_f, &self.normed, 1, d as u32, eps, true));
        chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
        chk(c.matvec_q8(&self.lm_head.q, &self.lm_head.s, &self.xq, &self.xs, &self.logits, 0, self.vocab as u32, d as u32));
        if !ok.get() {
            return None;
        }
        if !c.read(&self.logits, 0, &mut logits_out[..self.vocab]) {
            return None;
        }
        if let Some(tr) = trace {
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
            normed: c.alloc(cap * d * 4)?,
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
                    chk(c.add_rmsnorm_rows(&ws.hidden, if first { None } else { Some(&ws.y) }, in_norm, &ws.normed, tu, d as u32, eps, true));
                    chk(c.quantize_q8(&ws.normed, 0, &ws.xq, &ws.xs, tu, d as u32));
                    chk(c.gemm_q8(&proj.q, &proj.s, &ws.xq, &ws.xs, &ws.big_a, 0, pr as u32, d as u32, tu, pr as u32));
                    // conv + silu over time -> big_b [t][cd]
                    {
                        let (k, cdim, ldx, ldo) = (*conv_k as u32, cd as u32, pr as u32, cd as u32);
                        let mut a = args!(ws.big_a.ptr, ldx, conv_w.ptr, k, conv_state.ptr, ws.big_b.ptr, ldo, cdim, tu);
                        chk(c.launch("k_conv_prefill", (cdim.div_ceil(256), 1, 1), (256, 1, 1), 0, &mut a));
                    }
                    // q/k norms, beta, decay per (token, head)
                    {
                        let bp = ws.big_a.ptr + ((cd + vt) * 4) as u64;
                        let ap = ws.big_a.ptr + ((cd + vt + heads) * 4) as u64;
                        let (hh, kvh, kdd, ldc, ldba) = (*heads as u32, *kv_heads as u32, *kd as u32, cd as u32, pr as u32);
                        let mut a = args!(ws.big_b.ptr, ldc, bp, ap, ldba, a_log.ptr, dt_bias.ptr, ws.q.ptr, ws.gate.ptr, ws.beta.ptr, ws.decay.ptr, tu, hh, kvh, kdd);
                        chk(c.launch("k_lin_prep", (tu, *heads as u32, 1), ((*kd as u32).max(32), 1, 1), 0, &mut a));
                    }
                    // delta scan per head over time -> mixed [t][heads*vd]
                    {
                        let (hh, kvh, kdd, vdd, ldc) = (*heads as u32, *kv_heads as u32, *kd as u32, *vd as u32, cd as u32);
                        let mut a = args!(scan_state.ptr, ws.q.ptr, ws.gate.ptr, ws.big_b.ptr, ldc, ws.beta.ptr, ws.decay.ptr, ws.mixed.ptr, tu, hh, kvh, kdd, vdd);
                        chk(c.launch("k_delta_scan", (*heads as u32, 1, 1), (*vd as u32, 1, 1), (2 * kd * 4) as u32, &mut a));
                    }
                    // gated norm with z = big_a[.., cd..cd+vt]
                    {
                        let zp = ws.big_a.ptr + (cd * 4) as u64;
                        let (hh, vdd, ldz) = (*heads as u32, *vd as u32, pr as u32);
                        let mut a = args!(ws.mixed.ptr, gated_w.ptr, zp, ldz, tu, hh, vdd, eps);
                        chk(c.launch("k_gated_norm_rows", (tu, *heads as u32, 1), ((*vd as u32).max(32), 1, 1), 0, &mut a));
                    }
                    chk(c.quantize_q8(&ws.mixed, 0, &ws.xq, &ws.xs, tu, vt as u32));
                    chk(c.gemm_q8(&out.q, &out.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, vt as u32, tu, d as u32));
                    chk(c.add_rmsnorm_rows(&ws.hidden, Some(&ws.y), post_norm, &ws.normed, tu, d as u32, eps, true));
                    chk(c.quantize_q8(&ws.normed, 0, &ws.xq, &ws.xs, tu, d as u32));
                    chk(c.gemm_fp4(&gu.p, &gu.s, &ws.xq, &ws.xs, &ws.big_a, 0, (2 * inter) as u32, d as u32, tu, (2 * inter) as u32));
                    chk(c.silu_mul_rows(&ws.big_a, &ws.big_b, tu, *inter as u32));
                    chk(c.quantize_q8(&ws.big_b, 0, &ws.xq, &ws.xs, tu, *inter as u32));
                    chk(c.gemm_fp4(&dn.p, &dn.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, *inter as u32, tu, d as u32));
                }
                LayerDev::Full { in_norm, post_norm, proj, o, gu, dn, q_norm, k_norm, kc, vc, n_heads, n_kv, hd, rope_dim, theta, inter } => {
                    let qw = n_heads * hd;
                    let kvw = n_kv * hd;
                    let pr = proj.rows;
                    chk(c.add_rmsnorm_rows(&ws.hidden, if first { None } else { Some(&ws.y) }, in_norm, &ws.normed, tu, d as u32, eps, true));
                    chk(c.quantize_q8(&ws.normed, 0, &ws.xq, &ws.xs, tu, d as u32));
                    chk(c.gemm_q8(&proj.q, &proj.s, &ws.xq, &ws.xs, &ws.big_a, 0, pr as u32, d as u32, tu, pr as u32));
                    {
                        let (ldq, kvw_, p0, nh, nkv, hdd, rd) = (pr as u32, kvw as u32, pos0 as u32, *n_heads as u32, *n_kv as u32, *hd as u32, *rope_dim as u32);
                        let mut a = args!(ws.big_a.ptr, ldq, q_norm.ptr, k_norm.ptr, ws.q.ptr, ws.gate.ptr, kc.ptr, vc.ptr, kvw_, tu, p0, nh, nkv, hdd, rd, *theta, eps);
                        chk(c.launch("k_qk_prep_rows", (tu, (*n_heads + *n_kv) as u32, 1), (*hd as u32, 1, 1), 0, &mut a));
                    }
                    {
                        let (kvw_, p0, nh, nkv, hdd) = (kvw as u32, pos0 as u32, *n_heads as u32, *n_kv as u32, *hd as u32);
                        let mut a = args!(ws.q.ptr, ws.gate.ptr, kc.ptr, vc.ptr, ws.mixed.ptr, kvw_, p0, tu, nh, nkv, hdd);
                        chk(c.launch("k_attn_prefill", (tu, *n_heads as u32, 1), (*hd as u32, 1, 1), ((pos0 + t) * 4) as u32, &mut a));
                    }
                    chk(c.quantize_q8(&ws.mixed, 0, &ws.xq, &ws.xs, tu, qw as u32));
                    chk(c.gemm_q8(&o.q, &o.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, qw as u32, tu, d as u32));
                    chk(c.add_rmsnorm_rows(&ws.hidden, Some(&ws.y), post_norm, &ws.normed, tu, d as u32, eps, true));
                    chk(c.quantize_q8(&ws.normed, 0, &ws.xq, &ws.xs, tu, d as u32));
                    chk(c.gemm_fp4(&gu.p, &gu.s, &ws.xq, &ws.xs, &ws.big_a, 0, (2 * inter) as u32, d as u32, tu, (2 * inter) as u32));
                    chk(c.silu_mul_rows(&ws.big_a, &ws.big_b, tu, *inter as u32));
                    chk(c.quantize_q8(&ws.big_b, 0, &ws.xq, &ws.xs, tu, *inter as u32));
                    chk(c.gemm_fp4(&dn.p, &dn.s, &ws.xq, &ws.xs, &ws.y, 0, d as u32, *inter as u32, tu, d as u32));
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
        chk(c.quantize_q8(&self.normed, 0, &self.xq, &self.xs, 1, d as u32));
        chk(c.matvec_q8(&self.lm_head.q, &self.lm_head.s, &self.xq, &self.xs, &self.logits, 0, self.vocab as u32, d as u32));
        if !ok.get() {
            return None;
        }
        let mut logits = vec![0f32; self.vocab];
        if !c.read(&self.logits, 0, &mut logits) {
            return None;
        }
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
