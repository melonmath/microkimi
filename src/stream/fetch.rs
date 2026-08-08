// Physical reads for the expert cache: pread/O_DIRECT direct reads, fused
// span reads (fuse_runs merges adjacent expert blobs into one syscall),
// fake-disk latency injection for benchmarks, and the remote tier
// (RemoteSource: HTTP range fetch persisted to a per-tensor disk cache with
// a JSON manifest). Byte serving only: no dequant, no routing decisions.

use super::*;

// ── contiguous-run fusion (--stream, local source) ──
//
// The top-k experts of one MoE layer are submitted offset-sorted, but each
// used to be pread individually: k physical reads per layer per token, each
// paying the fixed latency of the disk tier (a network PD serves ~1700 4K
// IOPS, so that latency dominates). Expert blobs of a layer are stored as a
// near-contiguous run (see weights.rs BinWriter), so when the router picks
// file-adjacent experts their payloads can be served by ONE span read
// covering the first to the last, each expert extracted by its directory
// offsets. warm_batch (below) does this once per MoE layer before the
// compute jobs start; their ExpertCache::get then hits the RAM LRU. The
// bytes served are exactly those the individual preads would return, so the
// model output is bit-identical; only the number of physical reads changes.
// Measured adjacency on the nano checkpoints: 0% (their routers happen to
// pick only odd expert ids), which is why the slicer can physically reorder
// experts by routing frequency (slice --expert-order=frequency) to create
// the adjacency this fusion then exploits.
pub(super) static FUSE_READS: AtomicU64 = AtomicU64::new(0); // physical span reads issued by warm_batch
pub(super) static FUSE_EXPERTS: AtomicU64 = AtomicU64::new(0); // experts served by those span reads
pub(super) static FUSE_NS: AtomicU64 = AtomicU64::new(0); // wall time of the warm_batch read phase

const MANIFEST_FLUSH_EVERY: u64 = 64;

pub(super) struct Manifest {
    pub(super) map: HashMap<String, (u64, u64)>, // tensor name -> (size bytes, last access unix)
    dirty: u64,
}

pub(super) fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Manifest {
    /// Loads <dir>/manifest.json, tolerating a missing or corrupt file, then
    /// reconciles with the files actually on disk: a crash between a tensor
    /// write and the next manifest flush must not lose the entry.
    pub(super) fn load(dir: &std::path::Path) -> Manifest {
        let mut map: HashMap<String, (u64, u64)> = HashMap::new();
        if let Ok(bytes) = std::fs::read(dir.join("manifest.json")) {
            // the mini JSON parser panics on malformed input: catch it and
            // fall back to the on-disk rebuild below
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let parsed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::json::parse(&bytes)));
            std::panic::set_hook(prev);
            if let Ok(j) = parsed {
                if let Some(crate::json::Json::Obj(tensors)) = j.get("tensors") {
                    for (name, v) in tensors {
                        let size = v.get("size").and_then(|x| x.as_num()).unwrap_or(0.0) as u64;
                        let access = v.get("access").and_then(|x| x.as_num()).unwrap_or(0.0) as u64;
                        map.insert(name.clone(), (size, access));
                    }
                }
            }
        }
        // reconcile with the actual files: add anything the manifest missed,
        // drop entries whose file is gone
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_file() {
                    continue;
                }
                let name = e.file_name().to_string_lossy().into_owned();
                if name == "manifest.json" || name.contains(".partial-") {
                    continue; // the manifest itself and interrupted tmp writes
                }
                if !map.contains_key(&name) {
                    let md = e.metadata().ok();
                    let size = md.as_ref().map(|m| m.len()).unwrap_or(0);
                    let access = md
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        .unwrap_or(0);
                    map.insert(name, (size, access));
                }
            }
        }
        map.retain(|name, _| dir.join(sanitize(name)).is_file());
        Manifest { map, dirty: 0 }
    }

    fn record(&mut self, name: &str, size: u64) {
        self.map.insert(name.to_string(), (size, now_unix()));
        self.dirty += 1;
    }

    fn to_json(&self) -> String {
        let mut names: Vec<&String> = self.map.keys().collect();
        names.sort();
        let mut s = String::from("{\"tensors\":{");
        for (i, n) in names.iter().enumerate() {
            let (size, access) = self.map[*n];
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!("\"{}\":{{\"size\":{},\"access\":{}}}", n, size, access));
        }
        s.push_str("}}");
        s
    }

    /// Persists the manifest (tmp + rename) when `force`, or every
    /// MANIFEST_FLUSH_EVERY updates since the last flush.
    fn flush_maybe(&mut self, dir: &std::path::Path, force: bool) {
        if self.dirty == 0 || (!force && self.dirty < MANIFEST_FLUSH_EVERY) {
            return;
        }
        let tmp = dir.join(format!("manifest.json.partial-{}", std::process::id()));
        if std::fs::write(&tmp, self.to_json()).is_ok() {
            std::fs::rename(&tmp, dir.join("manifest.json")).ok();
            self.dirty = 0;
        }
    }
}

/// Routed expert tensors are the only evictable class of the disk cache:
/// canonical layers.N.block_sparse_moe.experts.E.w{1,2,3} or their raw
/// layers.N.mlp.experts.E.* equivalents. Shared experts (shared_experts.*),
/// the router, attention, norms, embeddings and lm_head are spine and are
/// NEVER evicted.
pub(super) fn is_expert_tensor(name: &str) -> bool {
    let Some(pos) = name.find(".experts.") else {
        return false;
    };
    let rest = &name[pos + ".experts.".len()..];
    let Some(dot) = rest.find('.') else {
        return false;
    };
    !rest[..dot].is_empty() && rest[..dot].bytes().all(|b| b.is_ascii_digit())
}

// ── remote source: per-tensor persistent disk cache over HTTP range fetches ──

/// Sanitized cache directory name for a remote repo URL:
/// "https://huggingface.co/moonshotai/Kimi-K3" -> "huggingface.co_moonshotai_Kimi-K3".
pub(super) fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect()
}

/// Default persistent cache root for a remote model URL.
pub fn default_cache_root(url: &str) -> std::path::PathBuf {
    let bare = url.trim_start_matches("https://").trim_start_matches("http://");
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::PathBuf::from(format!("{}/.cache/microkimi/{}", home, sanitize(bare)))
}

/// Remote safetensors model (HTTP range fetches via slice_st::StDir) with a
/// persistent per-tensor disk cache: every remote byte is fetched once, ever
/// (unless the --stream-disk rollover evicts it).
pub struct RemoteSource {
    st: crate::tools::slice_st::StDir,
    cache_dir: std::path::PathBuf,
    disk_budget: u64, // bytes, 0 = unlimited (historical behavior)
    manifest: Mutex<Manifest>,
}

impl RemoteSource {
    /// Opens a remote (or local safetensors) model; only the index, the config
    /// and the shard headers of `kept_layers` (+ global tensors) are fetched.
    /// Unlimited disk cache (historical behavior).
    pub fn open(url: &str, cache_dir: std::path::PathBuf, kept_layers: &[usize]) -> RemoteSource {
        Self::open_disk(url, cache_dir, kept_layers, 0)
    }

    /// Same as open, with a disk cache budget of `disk_mb` MB (0 = unlimited).
    pub fn open_disk(url: &str, cache_dir: std::path::PathBuf, kept_layers: &[usize], disk_mb: u64) -> RemoteSource {
        let mut st = crate::tools::slice_st::StDir::open(url, "/tmp/microkimi-stream");
        st.resolve(kept_layers);
        std::fs::create_dir_all(&cache_dir).ok();
        let manifest = Mutex::new(Manifest::load(&cache_dir));
        RemoteSource { st, cache_dir, disk_budget: disk_mb << 20, manifest }
    }

    fn cache_file(&self, name: &str) -> std::path::PathBuf {
        self.cache_dir.join(sanitize(name))
    }

    /// .bin-layout bytes of one logical tensor (packed ++ scales for MXFP4
    /// experts, converted f32 LE otherwise): disk cache first, one HTTP range
    /// fetch on a cold miss, then persisted (tmp + rename for crash safety).
    pub fn tensor_bytes(&self, name: &str) -> Vec<u8> {
        let path = self.cache_file(name);
        if let Ok(b) = std::fs::read(&path) {
            DISK_BYTES.fetch_add(b.len() as u64, Ordering::Relaxed);
            let mut m = self.manifest.lock().unwrap();
            m.record(name, b.len() as u64);
            m.flush_maybe(&self.cache_dir, false);
            return b;
        }
        let e = &self.st.entries[self.st.index[name]];
        let b = self.st.raw_blob(e);
        let tmp = path.with_extension(format!("partial-{}", std::process::id()));
        std::fs::write(&tmp, &b).unwrap_or_else(|e| panic!("cannot write {:?}: {}", tmp, e));
        std::fs::rename(&tmp, &path).ok();
        {
            let mut m = self.manifest.lock().unwrap();
            m.record(name, b.len() as u64);
            m.flush_maybe(&self.cache_dir, false);
        }
        self.enforce_budget();
        b
    }

    /// Disk LRU rollover (--stream-disk): after a cold fetch pushed the repo
    /// cache over the budget, evict least-recently-used EXPERT tensors until
    /// back under budget. Spine tensors are never evicted: if only spine
    /// remains and the budget is still exceeded, the cache stays over budget
    /// (the budget is best-effort for experts). An evicted expert is simply
    /// re-fetched over HTTP on its next miss (one range fetch per tensor), so
    /// an undersized budget turns into repeated network traffic.
    fn enforce_budget(&self) {
        if self.disk_budget == 0 {
            return;
        }
        let mut m = self.manifest.lock().unwrap();
        let mut total: u64 = m.map.values().map(|e| e.0).sum();
        if total <= self.disk_budget {
            return;
        }
        let mut experts: Vec<(String, u64, u64)> = m
            .map
            .iter()
            .filter(|(n, _)| is_expert_tensor(n))
            .map(|(n, &(sz, at))| (n.clone(), sz, at))
            .collect();
        experts.sort_by_key(|e| e.2); // least recently used first
        for (name, sz, _) in experts {
            if total <= self.disk_budget {
                break;
            }
            if std::fs::remove_file(self.cache_file(&name)).is_ok() {
                total = total.saturating_sub(sz);
                m.map.remove(&name);
                m.dirty += 1;
                println!("stream-disk: evicted {} ({})", name, mb(sz));
            }
        }
        m.flush_maybe(&self.cache_dir, true);
    }

    /// The 3 MXFP4 blobs of one routed expert, concatenated w1 ++ w2 ++ w3.
    pub fn expert_blobs(&self, layer: u32, expert: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for w in ["w1", "w2", "w3"] {
            out.extend_from_slice(&self.tensor_bytes(&format!("layers.{}.block_sparse_moe.experts.{}.{}", layer, expert, w)));
        }
        out
    }

    /// Bypasses the disk cache (slice_st's direct fetch): reference bytes for
    /// the streamtest byte-equality proof.
    pub(super) fn direct_bytes(&self, name: &str) -> Vec<u8> {
        let e = &self.st.entries[self.st.index[name]];
        self.st.raw_blob(e)
    }
}

impl Drop for RemoteSource {
    /// Final manifest flush at exit (a kill -9 skips this; the next open
    /// rebuilds whatever is missing from the files on disk).
    fn drop(&mut self) {
        self.manifest.get_mut().unwrap().flush_maybe(&self.cache_dir, true);
    }
}

// ── Markov predictor: online transition statistics over router picks ──

/// MICROKIMI_NO_OFFSORT=1 disables the offset-sorted submission of the
/// expert reads of a layer (model.rs), restoring the router-id order.
pub fn offset_sort() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OFF.get_or_init(|| env_off("MICROKIMI_NO_OFFSORT"))
}

/// MICROKIMI_NO_RUNFUSE=1 disables the contiguous-run fusion of the expert
/// reads of a layer (warm_batch becomes a no-op and every expert is pread
/// individually by its compute job, the historical behavior).
pub fn run_fuse() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OFF.get_or_init(|| env_off("MICROKIMI_NO_RUNFUSE"))
}

/// MICROKIMI_FAKE_DISK_MS=N (debug/bench only): sleep N ms per physical disk
/// read of the local tier, behind a process-wide lock so the sleeps
/// serialize the way a latency/IOPS-bound disk (network PD) serializes its
/// requests. One sleep per expert pread without fusion, one per merged span
/// read with fusion: exactly the A/B the run fusion is built for.
pub(super) fn fake_disk_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| std::env::var("MICROKIMI_FAKE_DISK_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// One serialized fake disk latency, no-op unless MICROKIMI_FAKE_DISK_MS set.
pub(super) fn fake_disk_sleep() {
    let ms = fake_disk_ms();
    if ms > 0 {
        static FAKE_DISK: Mutex<()> = Mutex::new(());
        let _io = FAKE_DISK.lock().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}

pub(super) fn env_off(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

// ── run fusion: merge file-adjacent expert reads into one span read ──

/// Maximum hole between two expert footprints still merged into one physical
/// read: covers the format alignment padding between blobs (zero on densely
/// packed .bins). The hole bytes are read and discarded; on a latency-bound
/// disk one span read still beats two separate reads.
pub(super) const FUSE_GAP_MAX: u64 = 4096;

/// An expert request is compact when its three blobs span at most their
/// payload plus per-blob alignment padding. Non-compact requests (scattered
/// w1/w2/w3) never join a run: merging them would read an unbounded hole.
pub(super) fn compact(offs: &[u64; 3], blob: usize) -> bool {
    offs[1] >= offs[0] && offs[2] >= offs[1] && offs[2] + blob as u64 - offs[0] <= 3 * blob as u64 + 2 * FUSE_GAP_MAX
}

/// Groups offset-sorted expert requests (expert, offs, blob) into maximal
/// runs of file-adjacent compact footprints (footprint = [offs[0], offs[2] +
/// blob)): two consecutive requests merge when both are compact and the gap
/// between their footprints is at most FUSE_GAP_MAX. Returns (start, end)
/// index ranges into the input slice. Pure: unit-tested directly.
pub(super) fn fuse_runs(sorted: &[(u32, [u64; 3], usize)]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0;
    for i in 1..=sorted.len() {
        let merges = i < sorted.len() && {
            let (_, po, pb) = &sorted[i - 1];
            let (_, co, cb) = &sorted[i];
            let prev_end = po[2] + *pb as u64;
            compact(po, *pb) && compact(co, *cb) && co[0] >= prev_end && co[0] - prev_end <= FUSE_GAP_MAX
        };
        if !merges {
            runs.push((start, i));
            start = i;
        }
    }
    runs
}

// ── Direct I/O (Linux O_DIRECT / macOS F_NOCACHE): cache-bypassing reads ──
//
// The streamed expert blobs are read once per cache miss and never re-read
// through the same pages (the RAM LRU above is the cache), so the page
// cache only pollutes. Linux O_DIRECT requires a 4 KiB aligned buffer,
// offset and length: read_direct widens every window to the enclosing 4 KiB
// boundaries and copies the payload out. macOS F_NOCACHE is a plain fcntl
// flag on the fd with no alignment constraints, so reads stay plain preads.
// Both serve exactly the bytes a buffered pread would.

/// Linux O_DIRECT open flag (hardcoded, no libc crate). The value is
/// arch-dependent in the kernel UAPI: aarch64/arm/m68k use 0o200000, the
/// asm-generic arches (x86_64, riscv64, ...) use 0o40000. Where the
/// constant does not match, the open simply fails and the buffered pread
/// fallback serves the reads.
#[cfg(all(target_os = "linux", any(target_arch = "aarch64", target_arch = "arm", target_arch = "m68k")))]
pub(super) const O_DIRECT: i32 = 0o200000;
#[cfg(all(target_os = "linux", not(any(target_arch = "aarch64", target_arch = "arm", target_arch = "m68k"))))]
pub(super) const O_DIRECT: i32 = 0o40000;

/// Opens the direct-I/O handle for `path`: O_DIRECT on Linux, an F_NOCACHE
/// fd on macOS. None when MICROKIMI_NO_ODIRECT=1, on other OSes, or when the
/// open/flag is rejected: the caller falls back to plain buffered pread.
pub(super) fn open_direct(path: &str) -> Option<std::fs::File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        if env_off("MICROKIMI_NO_ODIRECT") {
            return None;
        }
        return std::fs::OpenOptions::new().read(true).custom_flags(O_DIRECT).open(path).ok();
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        if env_off("MICROKIMI_NO_ODIRECT") {
            return None;
        }
        // F_NOCACHE = 48: this fd bypasses the unified buffer cache.
        // Direct FFI to the system lib, no libc crate.
        unsafe extern "C" {
            fn fcntl(fd: i32, cmd: i32, arg: i32) -> i32;
        }
        const F_NOCACHE: i32 = 48;
        let f = std::fs::File::open(path).ok()?;
        let r = unsafe { fcntl(f.as_raw_fd(), F_NOCACHE, 1) };
        return if r == 0 { Some(f) } else { None };
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = path;
        None
    }
}

/// O_DIRECT read of `out.len()` payload bytes at file offset `off`: the read
/// window is widened to the enclosing 4 KiB boundaries and served through a
/// manually aligned buffer (std::alloc, Layout align 4096), then the payload
/// is copied out. The bytes landing in `out` are exactly those a plain
/// pread(off, out.len()) would return.
pub(super) fn read_direct(f: &std::fs::File, off: u64, out: &mut [u8]) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    #[cfg(target_os = "macos")]
    {
        // F_NOCACHE fd: no alignment constraints, plain pread.
        return f.read_exact_at(out, off);
    }
    #[cfg(not(target_os = "macos"))]
    {
        const ALIGN: u64 = 4096;
        let start = off & !(ALIGN - 1);
        let end = (off + out.len() as u64).next_multiple_of(ALIGN);
        let n = (end - start) as usize;
        let layout = std::alloc::Layout::from_size_align(n, ALIGN as usize).unwrap();
        let p = unsafe { std::alloc::alloc(layout) };
        if p.is_null() {
            return Err(std::io::Error::new(std::io::ErrorKind::OutOfMemory, "aligned O_DIRECT buffer"));
        }
        let buf = unsafe { std::slice::from_raw_parts_mut(p, n) };
        let r = f.read_exact_at(buf, start);
        if r.is_ok() {
            let skip = (off - start) as usize;
            out.copy_from_slice(&buf[skip..skip + out.len()]);
        }
        unsafe { std::alloc::dealloc(p, layout) };
        r
    }
}
