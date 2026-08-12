// microkimi .bin format (microkimi-debug.bin, nanokimi-0.2b.bin, ...):
//   magic "MKIM0001" (8 bytes)
//   u32 n_tensors
//   directory × n : u16 name_len | name (utf-8) | u8 dtype (0=f32, 1=mxfp4, 2=i32, 3=vq1, 4=mxfp4sq) |
//                   u8 n_dims | u32 dims[n_dims] | u64 offset | u64 size
//   raw data (offsets from start of file; routed expert blobs 4096-aligned so
//   mmap demand-paging and streaming preads never pull a neighbor expert's
//   pages, everything else 64-aligned; padding is zero space readers never
//   see - they only use the directory's offset/size)
// mxfp4 dtype: blob = packed (R×C/2 bytes) then scales (R×C/32), dims = logical [R,C].
// mxfp4sq dtype (4): same layout + one trailing f32 smax; the scale byte decodes
//   quadratically, s(q) = ((q+1)/256)^2 * smax (see mxfp4.rs). Storage/measurement
//   variant: the packed runtime matvec reads mxfp4 only, dequant readers take both.
// vq1 dtype: blob = R×C/16 index bytes, one u8 codebook index per vector of 16
//   consecutive row-major values; dims = logical [R,C] with C % 16 == 0. The
//   codebook is NOT in the blob: it is a global f32 tensor "vq_codebook"
//   [256,16] shared by all vq1 tensors (written by `slice --cold-vq`).
// MKIM0002: u32 config_len + JSON config right after the magic.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

pub const MAGIC: &[u8; 8] = b"MKIM0001";
pub const MAGIC_V2: &[u8; 8] = b"MKIM0002";
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_MXFP4: u8 = 1;
pub const DTYPE_I32: u8 = 2;
pub const DTYPE_VQ1: u8 = 3;
pub const DTYPE_MXFP4SQ: u8 = 4;

#[derive(Debug, Clone)]
pub struct Entry {
    pub dtype: u8,
    pub dims: Vec<u32>,
    pub offset: u64,
    pub size: u64,
}

/// Size in bytes of a data blob.
pub fn blob_size(dtype: u8, dims: &[u32]) -> u64 {
    let n: u64 = dims.iter().map(|&d| d as u64).product();
    match dtype {
        DTYPE_F32 => n * 4,
        DTYPE_I32 => n * 4,
        DTYPE_MXFP4 => {
            let (r, c) = (dims[0] as u64, dims[1] as u64);
            r * c / 2 + r * c / 32
        }
        DTYPE_VQ1 => {
            // 1 byte index per vector of VQ_DIM consecutive values (row-major);
            // the 256 x VQ_DIM f32 codebook lives in a separate global tensor
            // ("vq_codebook"), shared by every VQ1 tensor of the file.
            let (r, c) = (dims[0] as u64, dims[1] as u64);
            r * c / crate::quant::quant::VQ_DIM as u64
        }
        DTYPE_MXFP4SQ => {
            // mxfp4 layout + one trailing f32 smax (quadratic scale encoding,
            // see mxfp4.rs): packed (R*C/2) + scales (R*C/32) + 4 bytes.
            let (r, c) = (dims[0] as u64, dims[1] as u64);
            r * c / 2 + r * c / 32 + 4
        }
        _ => panic!("unknown dtype {}", dtype),
    }
}

fn dir_entry_size(name: &str, dims: &[u32]) -> u64 {
    2 + name.len() as u64 + 1 + 1 + 4 * dims.len() as u64 + 8 + 8
}

// ── writing (build) ──

pub struct BinWriter {
    pub names_order: Vec<(String, u8, Vec<u32>)>, // (name, dtype, dims) in write order
    expert_align: u64,                            // start alignment of routed expert blobs (see layout)
}

impl BinWriter {
    pub fn new() -> Self {
        BinWriter { names_order: Vec::new(), expert_align: 4096 }
    }

    /// Overrides the routed-expert blob alignment (default 4096: mmap
    /// demand-paging and single-expert preads never touch a neighbor's
    /// pages). slice --expert-order packs experts densely (64) instead: the
    /// reordered blobs are meant to be read in fused spans, where page
    /// padding would be read and discarded on every fused read.
    pub fn set_expert_align(&mut self, a: u64) {
        self.expert_align = a;
    }

    pub fn add(&mut self, name: &str, dtype: u8, dims: Vec<u32>) {
        self.names_order.push((name.to_string(), dtype, dims));
    }

    /// Computes the blob offsets in write order; `prefix` = number of bytes
    /// before the u32 tensor count (8 for MKIM0001, 8+4+config_len for
    /// MKIM0002). Alignment: routed expert blobs (the MXFP4 w1/w2/w3 the
    /// stream engine preads one routed expert at a time) start on a
    /// `expert_align` boundary - 4096 by default, so the mmap demand-paging
    /// and those preads never touch a neighbor expert's pages (a token
    /// faults only the pages of its routed experts, never fragments of the
    /// unrouted ones); 64 when the writer packs experts densely for fused
    /// span reads (set_expert_align). Everything else keeps the historical
    /// 64-byte alignment (vectorized loads, f32 slice alignment). The
    /// padding is plain zero space between blobs: readers only ever use the
    /// directory's (offset, size) and never see it.
    fn layout(&self, prefix: u64) -> Vec<u64> {
        let dir_size: u64 = self
            .names_order
            .iter()
            .map(|(n, _, d)| dir_entry_size(n, d))
            .sum();
        let data_start = prefix + 4 + dir_size;
        let mut offsets = Vec::with_capacity(self.names_order.len());
        let mut pos = data_start;
        for (name, dtype, dims) in &self.names_order {
            let align = if is_expert_tensor(name) { self.expert_align } else { 64 };
            pos = pos.div_ceil(align) * align;
            offsets.push(pos);
            pos += blob_size(*dtype, dims);
        }
        offsets
    }

    fn write_directory(&self, f: &mut std::fs::File, offsets: &[u64]) {
        f.write_all(&(self.names_order.len() as u32).to_le_bytes()).unwrap();
        for ((name, dtype, dims), &off) in self.names_order.iter().zip(offsets) {
            f.write_all(&(name.len() as u16).to_le_bytes()).unwrap();
            f.write_all(name.as_bytes()).unwrap();
            f.write_all(&[*dtype]).unwrap();
            f.write_all(&(dims.len() as u8).to_le_bytes()).unwrap();
            for d in dims {
                f.write_all(&d.to_le_bytes()).unwrap();
            }
            f.write_all(&off.to_le_bytes()).unwrap();
            f.write_all(&blob_size(*dtype, dims).to_le_bytes()).unwrap();
        }
    }

    /// Computes the directory (64-aligned offsets) and writes it with the header.
    /// Returns the offsets in the same order as names_order.
    pub fn write_header(&self, f: &mut std::fs::File) -> Vec<u64> {
        let offsets = self.layout(8);
        f.write_all(MAGIC).unwrap();
        self.write_directory(f, &offsets);
        offsets
    }

    /// MKIM0002 header: magic + u32 config_len + JSON config + directory.
    pub fn write_header_v2(&self, f: &mut std::fs::File, config_json: &str) -> Vec<u64> {
        let offsets = self.layout(8 + 4 + config_json.len() as u64);
        f.write_all(MAGIC_V2).unwrap();
        f.write_all(&(config_json.len() as u32).to_le_bytes()).unwrap();
        f.write_all(config_json.as_bytes()).unwrap();
        self.write_directory(f, &offsets);
        offsets
    }

    /// Writes a blob at its offset (with alignment padding before).
    pub fn write_blob_at(&self, f: &mut std::fs::File, offset: u64, data: &[u8]) {
        let cur = f.stream_position().unwrap();
        if cur < offset {
            f.write_all(&vec![0u8; (offset - cur) as usize]).unwrap();
        } else if cur > offset {
            f.seek(SeekFrom::Start(offset)).unwrap();
        }
        f.write_all(data).unwrap();
    }
}

// ── reading (inference) ──

// ── file backing: owned Vec or read-only mmap (RAM overcommit) ──
//
// Loading a 20-32 GB model with std::fs::read commits the whole file to
// ANONYMOUS memory, which the kernel cannot reclaim under pressure: a model
// larger than RAM OOMs (observed: a 21 GB run on a 15 GB VM). A PROT_READ /
// MAP_PRIVATE mmap keeps the bytes file-backed instead: pages fault in on
// first touch and are reclaimable at any time (re-faulted on the next token
// pass), so the kernel does the demand-paging and the resident set stays
// bounded by what the memory pressure allows. Every engine access is a
// read-only &[u8]/&[f32] slice into BinFile.data, so the backing switch is
// invisible upstream (Deref<Target = [u8]>). mmap is the default at ANY
// file size: one uniform code path, no up-front copy, instant load, and the
// page cache sharing benefits small models too; the Vec path remains as the
// MICROKIMI_NO_MMAP=1 fallback (and mmap failure fallback).

/// Private file mapping. The mapping is created PROT_READ|PROT_WRITE with
/// MAP_PRIVATE: adapter packs may patch selected fp32 pages during model
/// construction, while the file on disk never changes. After construction
/// the engine only exposes shared reads. Mutations require an exclusive
/// `&mut Backing`, so `Send` and `Sync` remain sound.
pub struct Mmap {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

impl Drop for Mmap {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        unsafe {
            mman::munmap(self.ptr as *mut std::ffi::c_void, self.len);
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = (self.ptr, self.len);
    }
}

/// What backs BinFile.data. Derefs to [u8]; nothing upstream changes.
pub enum Backing {
    Vec(Vec<u8>),
    Mmap(Mmap),
}

impl Backing {
    /// Mutable bytes for load-time copy-on-write patches. No reference into
    /// the backing may coexist with the exclusive borrow required here.
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Backing::Vec(v) => v.as_mut_slice(),
            Backing::Mmap(m) => unsafe {
                std::slice::from_raw_parts_mut(m.ptr, m.len)
            },
        }
    }
}

impl std::ops::Deref for Backing {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Vec(v) => v.as_slice(),
            // sound: `ptr..ptr+len` is a live PROT_READ mapping for the whole
            // lifetime of `self` (munmap only in Drop), u8 has no validity
            // constraints, and mutation requires an exclusive `&mut Backing`
            Backing::Mmap(m) => unsafe { std::slice::from_raw_parts(m.ptr, m.len) },
        }
    }
}

// Direct libc FFI (no crate). Constant values verified against the system
// headers: Linux glibc asm-generic/mman-common.h (PROT_READ 0x1,
// MADV_RANDOM 1, MADV_WILLNEED 3) and bits/mman-linux.h (MAP_PRIVATE 0x02);
// Apple XNU bsd/sys/mman.h (PROT_READ 0x01, MAP_PRIVATE 0x0002, MADV_RANDOM
// 1 = POSIX_MADV_RANDOM, MADV_WILLNEED 3 = POSIX_MADV_WILLNEED). Identical
// values on Linux and macOS. off_t is 64-bit on both x86_64/aarch64 Linux
// and macOS.
#[cfg(any(target_os = "linux", target_os = "macos"))]
mod mman {
    use std::ffi::c_void;
    pub const PROT_READ: i32 = 0x1;
    pub const PROT_WRITE: i32 = 0x2;
    pub const MAP_PRIVATE: i32 = 0x2;
    pub const MADV_RANDOM: i32 = 1;
    pub const MADV_WILLNEED: i32 = 3;
    unsafe extern "C" {
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }
}

/// True when mmap is disabled by MICROKIMI_NO_MMAP=1 (A/B toggle, also
/// disables the madvise hint: there is no mapping to advise).
fn no_mmap() -> bool {
    std::env::var("MICROKIMI_NO_MMAP").map(|v| v == "1").unwrap_or(false)
}

/// Whole-file private copy-on-write mapping. None when mmap is disabled, unsupported,
/// the file is empty, or the call fails (caller falls back to reading into
/// a Vec). No madvise hint is applied here: the right advice depends on the
/// access pattern, which only the caller knows (full load reads the routed
/// experts through the mapping, the streaming load never does - see
/// advise_random / advise_willneed below). Alignment: the mapping base is
/// page-aligned and the format's tensor offsets are 64-aligned, so f32
/// slices keep at least the alignment the Vec path had.
fn mmap_file(f: &std::fs::File, len: u64) -> Option<Mmap> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        use std::os::unix::io::AsRawFd;
        if no_mmap() || len == 0 {
            return None;
        }
        let p = unsafe {
            mman::mmap(
                std::ptr::null_mut(),
                len as usize,
                mman::PROT_READ | mman::PROT_WRITE,
                mman::MAP_PRIVATE,
                f.as_raw_fd(),
                0,
            )
        };
        if p.is_null() || p as isize == -1 {
            return None; // MAP_FAILED: Vec fallback
        }
        return Some(Mmap { ptr: p as *mut u8, len: len as usize });
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = (f, len);
        None
    }
}

// madvise operates on whole pages and requires a page-aligned address, and
// the mapping base is page-aligned, so a file range [off, off+len) maps to
// address range base+off .. base+off+len. The two helpers below differ in
// how they treat the partial edge pages:
// - RANDOM on expert spans rounds OUTWARD: the edge pages are almost all
//   format padding (expert blobs start 4096-aligned, their tails pad to the
//   next page), so covering the full span costs at most one shared page per
//   expert.
// - WILLNEED on spine gaps rounds INWARD: pulling an edge page would fault
//   in the tail of a neighbor expert blob the stream engine will never read
//   through the mapping, so the edge pages are left to demand paging.
#[cfg(any(target_os = "linux", target_os = "macos"))]
const PAGE: u64 = 4096;

/// Best-effort madvise over the file range [off, off+len) of the mapping.
/// `outward` rounds the range to the enclosing pages, `!outward` to the
/// covered pages only (an empty result advises nothing).
#[cfg(any(target_os = "linux", target_os = "macos"))]
fn advise(m: &Mmap, off: u64, len: u64, hint: i32, outward: bool) {
    let file_end = m.len as u64;
    let (s, e) = if outward {
        (off & !(PAGE - 1), (off + len).next_multiple_of(PAGE).min(file_end))
    } else {
        (off.next_multiple_of(PAGE), (off + len).min(file_end) & !(PAGE - 1))
    };
    if e <= s {
        return;
    }
    unsafe { mman::madvise(m.ptr.add(s as usize) as *mut std::ffi::c_void, (e - s) as usize, hint) };
}

/// Sorted, merged byte ranges of the routed expert blobs (span ends padded
/// to the 64-byte format alignment, capped at the file length). Used by the
/// streaming Vec compaction (open_spine) and by the mmap advice logic.
fn expert_spans(entries: &HashMap<String, Entry>, file_len: u64) -> Vec<(u64, u64)> {
    let mut spans: Vec<(u64, u64)> = entries
        .iter()
        .filter(|(n, _)| is_expert_tensor(n))
        .map(|(_, e)| {
            let end = (e.offset + e.size).div_ceil(64) * 64;
            (e.offset, end.min(file_len))
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
    for (s, e) in spans {
        match merged.last_mut() {
            Some(last) if s <= last.1 => last.1 = last.1.max(e),
            _ => merged.push((s, e)),
        }
    }
    merged
}

/// Available RAM in bytes, from /proc/meminfo MemAvailable (Linux). None on
/// other OSes (no portable no-crate source; the memory line just omits it).
fn mem_available() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb << 10);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn gb(n: u64) -> String {
    format!("{:.1} GB", n as f64 / (1u64 << 30) as f64)
}

/// One upfront memory line at model load: model (or spine) size vs available
/// RAM and the chosen backing. Never a hard refusal: when even the bytes
/// that must be resident exceed the available RAM, an explicit warning is
/// printed and the load proceeds.
fn mem_report(model: u64, resident: u64, mmap: bool) {
    let avail = mem_available();
    let avail_s = avail.map(gb).unwrap_or_else(|| "? GB".to_string());
    let kind = if mmap { "mmap demand-paging" } else { "in-RAM load" };
    if model == resident {
        println!("memory: model {}, RAM available {} - {}", gb(model), avail_s, kind);
    } else {
        println!("memory: spine {} of model {}, RAM available {} - {}", gb(resident), gb(model), avail_s, kind);
    }
    if let Some(a) = avail {
        if !mmap && model > a {
            println!("memory: WARNING model {} > RAM available {} with mmap off (MICROKIMI_NO_MMAP=1) - likely OOM", gb(model), gb(a));
        } else if mmap && resident > a {
            println!(
                "memory: WARNING resident part {} > RAM available {} - heavy disk thrashing expected, consider a smaller model",
                gb(resident),
                gb(a)
            );
        }
    }
}

pub struct BinFile {
    pub data: Backing,
    pub entries: HashMap<String, Entry>,
    pub config: crate::config::Config,
}

/// Reads ONLY the header (magic + config block) without loading the whole
/// file — used to pick the right tokenizer before loading the model.
pub fn read_config(path: &str) -> crate::config::Config {
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {} (run `microkimi build` first)", path, e));
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    if magic == *MAGIC {
        return crate::config::Config::microkimi();
    }
    assert_eq!(magic, *MAGIC_V2, "bad magic in {} (expected MKIM0001 or MKIM0002)", path);
    let mut clen = [0u8; 4];
    f.read_exact(&mut clen).unwrap();
    let clen = u32::from_le_bytes(clen) as usize;
    let mut cbuf = vec![0u8; clen];
    f.read_exact(&mut cbuf).unwrap();
    crate::config::Config::from_json(&crate::json::parse(&cbuf))
}

/// True for the routed MXFP4 expert matrices (layers.N.block_sparse_moe.experts.E.wI):
/// exactly the tensors `--stream` keeps out of RAM (see stream.rs).
pub fn is_expert_tensor(name: &str) -> bool {
    name.contains(".block_sparse_moe.experts.") && (name.ends_with(".w1") || name.ends_with(".w2") || name.ends_with(".w3"))
}

/// Parses the header (magic + config block) and the tensor directory, leaving
/// the file positioned anywhere: data blobs are NOT read. Shared by open
/// (full load) and open_spine (streaming load).
fn read_directory(path: &str) -> (std::fs::File, crate::config::Config, HashMap<String, Entry>) {
    let mut f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {} (run `microkimi build` first)", path, e));
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    let config = if magic == *MAGIC {
        crate::config::Config::microkimi() // MKIM0001: implicit microkimi config
    } else if magic == *MAGIC_V2 {
        // MKIM0002: u32 config_len + explicit JSON config
        let mut clen = [0u8; 4];
        f.read_exact(&mut clen).unwrap();
        let clen = u32::from_le_bytes(clen) as usize;
        let mut cbuf = vec![0u8; clen];
        f.read_exact(&mut cbuf).unwrap();
        crate::config::Config::from_json(&crate::json::parse(&cbuf))
    } else {
        panic!("bad magic in {} (expected MKIM0001 or MKIM0002)", path)
    };
    let mut nbuf = [0u8; 4];
    f.read_exact(&mut nbuf).unwrap();
    let n = u32::from_le_bytes(nbuf) as usize;
    let mut entries = HashMap::with_capacity(n);
    for _ in 0..n {
        let mut nlen = [0u8; 2];
        f.read_exact(&mut nlen).unwrap();
        let nlen = u16::from_le_bytes(nlen) as usize;
        let mut name = vec![0u8; nlen];
        f.read_exact(&mut name).unwrap();
        let mut fixed = [0u8; 2];
        f.read_exact(&mut fixed).unwrap();
        let (dtype, n_dims) = (fixed[0], fixed[1] as usize);
        let mut dims = vec![0u32; n_dims];
        for d in dims.iter_mut() {
            let mut b = [0u8; 4];
            f.read_exact(&mut b).unwrap();
            *d = u32::from_le_bytes(b);
        }
        let mut b16 = [0u8; 16];
        f.read_exact(&mut b16).unwrap();
        let offset = u64::from_le_bytes(b16[0..8].try_into().unwrap());
        let size = u64::from_le_bytes(b16[8..16].try_into().unwrap());
        entries.insert(String::from_utf8(name).unwrap(), Entry { dtype, dims, offset, size });
    }
    (f, config, entries)
}

impl BinFile {
    pub fn open(path: &str) -> Self {
        let (f, config, entries) = read_directory(path);
        let file_len = f.metadata().unwrap().len();
        if let Some(m) = mmap_file(&f, file_len) {
            // Full load: the routed experts ARE read through this mapping
            // (sparse per-token picks, so RANDOM on their spans kills the
            // useless speculative readahead), while the spine (attention,
            // embeddings, norms, lm_head) is swept sequentially once per
            // token and keeps the kernel default: sequential readahead.
            // A whole-file RANDOM here would cap the spine at single-page
            // faults (measured: a few MB/s instead of 100-200 MB/s of
            // sequential demand paging). Best-effort, advice only.
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            for &(s, e) in &expert_spans(&entries, file_len) {
                advise(&m, s, e - s, mman::MADV_RANDOM, true);
            }
            mem_report(file_len, file_len, true);
            return BinFile { data: Backing::Mmap(m), entries, config };
        }
        drop(f);
        let data = std::fs::read(path).unwrap_or_else(|e| {
            panic!(
                "cannot load {} into RAM: {} (the whole file must fit in memory on this path; mmap demand-paging, the default, has no such requirement - unset MICROKIMI_NO_MMAP)",
                path, e
            )
        });
        mem_report(file_len, file_len, false);
        BinFile { data: Backing::Vec(data), entries, config }
    }

    /// Streaming load (--stream): the MXFP4 routed-expert blobs are NOT read
    /// into RAM. With mmap (the default), the whole file is mapped read-only
    /// and NO compaction happens: spine tensors are sliced on demand through
    /// the mapping (their pages fault in), the expert regions are simply
    /// never touched through it (the stream engine preads them through its
    /// own, possibly O_DIRECT/F_NOCACHE, fd - the two access paths are
    /// independent read-only views of the same file and coexist freely), so
    /// they never become resident. With mmap off (MICROKIMI_NO_MMAP=1 or a
    /// failed mapping), the historical path loads the spine bytes compacted
    /// into a Vec: `data` holds only the spine (embeddings, attention,
    /// norms, router, dense/shared MLP, lm_head) with the offsets of those
    /// entries remapped to the compacted layout; expert entries keep their
    /// absolute file offsets so the stream engine can pread them on demand
    /// (stream.rs). Alignment is preserved: every expert span is extended to
    /// the next 64-byte boundary (format-level alignment padding), so the
    /// skipped total stays a multiple of 64 and remapped f32 tensors remain
    /// 64-aligned.
    pub fn open_spine(path: &str) -> Self {
        use std::os::unix::fs::FileExt;
        let (f, config, mut entries) = read_directory(path);
        let file_len = f.metadata().unwrap().len();
        let expert_bytes: u64 = entries.iter().filter(|(n, _)| is_expert_tensor(n)).map(|(_, e)| e.size).sum();
        let spine_bytes = file_len - expert_bytes;
        if let Some(m) = mmap_file(&f, file_len) {
            // Streaming load: the experts are pread through the stream
            // engine's own fd (O_DIRECT/F_NOCACHE when available), so this
            // mapping serves ONLY the spine, swept sequentially once per
            // token. No MADV_RANDOM here: it would cap the spine at
            // single-page faults (measured: a few MB/s instead of 100-200
            // MB/s of sequential demand paging). Instead, when the RAM
            // clearly fits the spine, MADV_WILLNEED on the spine gaps asks
            // the kernel for a background sequential read-in at load time,
            // replacing the manual warm-up pass. Best-effort, advice only.
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            {
                let warm = matches!(mem_available(), Some(a) if a > spine_bytes);
                if warm {
                    let mut cursor = 0u64;
                    let gaps = expert_spans(&entries, file_len);
                    for &(s, e) in &gaps {
                        advise(&m, cursor, s - cursor, mman::MADV_WILLNEED, false);
                        cursor = e;
                    }
                    advise(&m, cursor, file_len - cursor, mman::MADV_WILLNEED, false);
                    println!("memory: spine warm-up on (MADV_WILLNEED, background sequential read-in)");
                }
            }
            mem_report(file_len, spine_bytes, true);
            return BinFile { data: Backing::Mmap(m), entries, config };
        }
        mem_report(file_len, spine_bytes, false);
        // expert byte ranges held back from RAM (end padded to 64, see above)
        let merged = expert_spans(&entries, file_len);
        // skipped[i] = total expert bytes in merged[0..=i]
        let mut skipped: Vec<u64> = Vec::with_capacity(merged.len());
        let mut acc = 0u64;
        for &(s, e) in &merged {
            acc += e - s;
            skipped.push(acc);
        }
        let skipped_before = |off: u64| -> u64 {
            let i = merged.partition_point(|&(_, e)| e <= off);
            if i == 0 { 0 } else { skipped[i - 1] }
        };
        // spine bytes: the whole file minus the expert ranges (64 MB chunks)
        let mut data = Vec::with_capacity((file_len - acc) as usize);
        let mut cursor = 0u64;
        let read_gap = |from: u64, to: u64, data: &mut Vec<u8>| {
            let mut pos = from;
            while pos < to {
                let n = ((to - pos) as usize).min(1 << 26);
                let old = data.len();
                data.resize(old + n, 0);
                f.read_exact_at(&mut data[old..], pos).unwrap();
                pos += n as u64;
            }
        };
        for &(s, e) in &merged {
            read_gap(cursor, s, &mut data);
            cursor = e;
        }
        read_gap(cursor, file_len, &mut data);
        // remap the offsets of the in-RAM entries; experts keep absolute offsets
        for (name, e) in entries.iter_mut() {
            if !is_expert_tensor(name) {
                e.offset -= skipped_before(e.offset);
            }
        }
        BinFile { data: Backing::Vec(data), entries, config }
    }

    fn blob(&self, name: &str) -> &[u8] {
        let e = self
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor: {}", name));
        &self.data[e.offset as usize..(e.offset + e.size) as usize]
    }

    /// f32 tensor → slice over the file bytes (lazy conversion elsewhere).
    #[allow(dead_code)]
    pub fn f32_bytes(&self, name: &str) -> &[u8] {
        let e = self.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
        assert_eq!(e.dtype, DTYPE_F32, "{} is not f32", name);
        self.blob(name)
    }

    /// f32 tensor → Vec<f32> (copy).
    #[allow(dead_code)]
    pub fn f32_vec(&self, name: &str) -> Vec<f32> {
        self.f32_bytes(name)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// mxfp4 tensor → (packed, scales, rows, cols).
    #[allow(dead_code)]
    pub fn mxfp4_parts(&self, name: &str) -> (&[u8], &[u8], usize, usize) {
        let e = self.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
        assert_eq!(e.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
        let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
        let b = self.blob(name);
        let np = r * c / 2;
        (&b[..np], &b[np..], r, c)
    }

    /// i32 tensor → Vec<i32> (copy).
    #[allow(dead_code)]
    pub fn i32_vec(&self, name: &str) -> Vec<i32> {
        let e = self.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name));
        assert_eq!(e.dtype, DTYPE_I32, "{} is not i32", name);
        self.blob(name)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

pub fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
