// MoE expert streaming (--stream): routed MXFP4 expert weights are not held
// in RAM. A three-tier cache serves the expert bytes on demand:
//
//   RAM LRU (budget --stream-ram, MB) -> local disk / persistent cache -> HTTP range fetch
//
// - Local .bin: the spine (everything except layers.*.block_sparse_moe.experts.*)
//   is loaded compacted into RAM (weights::BinFile::open_spine); expert blobs
//   are pread from the .bin itself (the file IS the disk tier).
// - Remote safetensors (https://huggingface.co/org/repo): every tensor is
//   fetched ONCE via HTTP range requests (slice_st.rs machinery) and persisted
//   under ~/.cache/microkimi/<sanitized-repo>/<tensor-name> in .bin blob
//   layout (packed ++ scales for MXFP4, converted f32 LE otherwise); later
//   runs are served from that disk cache with zero network traffic.
//
// The LRU caches PACKED expert bytes (3 MXFP4 blobs, w1 ++ w2 ++ w3), not the
// dequantized f32: 17x smaller per entry, and the matvec dequantizes on the
// fly anyway (mxfp4::matvec_packed), so the cached form is exactly the form
// the compute consumes. Bit-exactness: the bytes served are byte-identical
// to the full-load path (same file bytes / same fetched blob), and the same
// matvec sequence runs on them.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// global fetch report counters (one model per process)
static RAM_HITS: AtomicU64 = AtomicU64::new(0);
static RAM_MISSES: AtomicU64 = AtomicU64::new(0);
static DISK_BYTES: AtomicU64 = AtomicU64::new(0); // bytes served by the disk tier

fn mb(n: u64) -> String {
    format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
}

/// One-line fetch report printed at exit when --stream is active.
pub fn report_line() -> String {
    format!(
        "stream: {} expert RAM hits, {} misses, {} read from the disk tier, {} fetched over HTTP ({} requests)",
        RAM_HITS.load(Ordering::Relaxed),
        RAM_MISSES.load(Ordering::Relaxed),
        mb(DISK_BYTES.load(Ordering::Relaxed)),
        mb(crate::http::fetched_bytes()),
        crate::http::fetched_requests(),
    )
}

// ── RAM LRU ──

struct Lru {
    map: HashMap<(u32, u32), (Arc<Vec<u8>>, u64)>, // (layer, expert) -> (w1++w2++w3, tick)
    queue: VecDeque<((u32, u32), u64)>,            // access order, stale gens skipped
    cur: usize,
    tick: u64,
    budget: usize,
}

impl Lru {
    fn get(&mut self, k: (u32, u32)) -> Option<Arc<Vec<u8>>> {
        if !self.map.contains_key(&k) {
            return None;
        }
        self.tick += 1;
        let e = self.map.get_mut(&k).unwrap();
        e.1 = self.tick;
        let v = e.0.clone();
        self.queue.push_back((k, self.tick));
        Some(v)
    }

    fn insert(&mut self, k: (u32, u32), v: Arc<Vec<u8>>) {
        let sz = v.len();
        if sz > self.budget {
            return; // a single expert exceeds the budget: serve without caching
        }
        self.tick += 1;
        if let Some(old) = self.map.insert(k, (v, self.tick)) {
            self.cur -= old.0.len();
        }
        self.cur += sz;
        self.queue.push_back((k, self.tick));
        // evict least recently used until back under budget (keep the new entry)
        while self.cur > self.budget && self.map.len() > 1 {
            match self.queue.pop_front() {
                Some((fk, fg)) => {
                    if let Some(e) = self.map.get(&fk) {
                        if e.1 == fg {
                            self.cur -= e.0.len();
                            self.map.remove(&fk);
                        }
                    }
                }
                None => break,
            }
        }
        // amortized queue compaction (every access pushes one entry)
        if self.queue.len() > 4 * self.map.len().max(16) {
            let mut live: Vec<((u32, u32), u64)> = self.map.iter().map(|(&k, &(_, g))| (k, g)).collect();
            live.sort_by_key(|&(_, g)| g);
            self.queue = live.into();
        }
    }
}

// ── remote source: per-tensor persistent disk cache over HTTP range fetches ──

/// Sanitized cache directory name for a remote repo URL:
/// "https://huggingface.co/moonshotai/Kimi-K3" -> "huggingface.co_moonshotai_Kimi-K3".
fn sanitize(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' }).collect()
}

/// Default persistent cache root for a remote model URL.
pub fn default_cache_root(url: &str) -> std::path::PathBuf {
    let bare = url.trim_start_matches("https://").trim_start_matches("http://");
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::PathBuf::from(format!("{}/.cache/microkimi/{}", home, sanitize(bare)))
}

/// Remote safetensors model (HTTP range fetches via slice_st::StDir) with a
/// persistent per-tensor disk cache: every remote byte is fetched once, ever.
pub struct RemoteSource {
    st: crate::slice_st::StDir,
    cache_dir: std::path::PathBuf,
}

impl RemoteSource {
    /// Opens a remote (or local safetensors) model; only the index, the config
    /// and the shard headers of `kept_layers` (+ global tensors) are fetched.
    pub fn open(url: &str, cache_dir: std::path::PathBuf, kept_layers: &[usize]) -> RemoteSource {
        let mut st = crate::slice_st::StDir::open(url, "/tmp/microkimi-stream");
        st.resolve(kept_layers);
        std::fs::create_dir_all(&cache_dir).ok();
        RemoteSource { st, cache_dir }
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
            return b;
        }
        let e = &self.st.entries[self.st.index[name]];
        let b = self.st.raw_blob(e);
        let tmp = path.with_extension(format!("partial-{}", std::process::id()));
        std::fs::write(&tmp, &b).unwrap_or_else(|e| panic!("cannot write {:?}: {}", tmp, e));
        std::fs::rename(&tmp, &path).ok();
        b
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
    fn direct_bytes(&self, name: &str) -> Vec<u8> {
        let e = &self.st.entries[self.st.index[name]];
        self.st.raw_blob(e)
    }
}

// ── expert cache: RAM LRU over a byte source ──

enum Src {
    /// Local .bin: expert blobs pread from the model file itself.
    Local(std::fs::File),
    /// Remote safetensors: per-tensor persistent cache (+ HTTP on cold miss).
    Remote(RemoteSource),
}

/// RAM LRU of packed expert bytes keyed by (layer, expert), budgeted by
/// --stream-ram. Thread-safe: the 16 router-selected experts of a layer are
// fetched and computed in parallel pool jobs (model.rs).
pub struct ExpertCache {
    inner: Mutex<Lru>,
    src: Src,
}

impl ExpertCache {
    /// Local streaming source: `path` is the .bin the spine was loaded from.
    pub fn local(path: &str, ram_mb: usize) -> ExpertCache {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
        ExpertCache {
            inner: Mutex::new(Lru { map: HashMap::new(), queue: VecDeque::new(), cur: 0, tick: 0, budget: ram_mb << 20 }),
            src: Src::Local(file),
        }
    }

    /// Remote streaming source (per-tensor persistent cache).
    #[allow(dead_code)] // wired end-to-end once real-dim MLA layers run (see docs)
    pub fn remote(url: &str, ram_mb: usize, kept_layers: &[usize]) -> ExpertCache {
        ExpertCache {
            inner: Mutex::new(Lru { map: HashMap::new(), queue: VecDeque::new(), cur: 0, tick: 0, budget: ram_mb << 20 }),
            src: Src::Remote(RemoteSource::open(url, default_cache_root(url), kept_layers)),
        }
    }

    /// The 3 MXFP4 blobs of expert `expert` of `layer`, concatenated
    /// w1 ++ w2 ++ w3, `blob` bytes each. `offs` = absolute file offsets of
    /// the three blobs (local source; ignored by the remote one).
    pub fn get(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize) -> Arc<Vec<u8>> {
        let k = (layer, expert);
        {
            let mut lru = self.inner.lock().unwrap();
            if let Some(v) = lru.get(k) {
                RAM_HITS.fetch_add(1, Ordering::Relaxed);
                return v;
            }
        }
        RAM_MISSES.fetch_add(1, Ordering::Relaxed);
        let bytes = match &self.src {
            Src::Local(file) => {
                use std::os::unix::fs::FileExt;
                let mut out = vec![0u8; 3 * blob];
                if offs[1] == offs[0] + blob as u64 && offs[2] == offs[1] + blob as u64 {
                    // the .bin writers emit w1, w2, w3 back to back: one pread
                    file.read_exact_at(&mut out, offs[0]).unwrap();
                } else {
                    for i in 0..3 {
                        file.read_exact_at(&mut out[i * blob..(i + 1) * blob], offs[i]).unwrap();
                    }
                }
                DISK_BYTES.fetch_add(out.len() as u64, Ordering::Relaxed);
                out
            }
            Src::Remote(r) => r.expert_blobs(layer, expert),
        };
        let v = Arc::new(bytes);
        self.inner.lock().unwrap().insert(k, v.clone());
        v
    }
}

// ── streamtest: remote per-tensor cache + LRU budget proof ──

/// `microkimi streamtest --model https://huggingface.co/org/repo [--cache-dir D]`
///
/// Bandwidth-safe proof of the remote tier against the real K3 repo:
/// 1. cold fetch of 3 real tensors (one MoE router, one expert w1, one KDA
///    q_proj) through the per-tensor persistent cache, byte-compared against
///    slice_st's direct fetch;
/// 2. warm fetch of the same tensors: served from disk, zero network bytes;
/// 3. LRU eviction respects the --stream-ram budget.
/// Only layers 0-2 are resolved (KDA in real K3: 0 dense, 1-2 MoE), so the
/// index, the config and a handful of shard headers are the only fixed cost.
pub fn streamtest(args: &[String]) {
    let url = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .unwrap_or_else(|| {
            eprintln!("error: streamtest requires --model https://huggingface.co/org/repo");
            std::process::exit(1);
        })
        .clone();
    let cache_dir = args
        .iter()
        .position(|a| a == "--cache-dir")
        .and_then(|i| args.get(i + 1))
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("/tmp/microkimi-streamtest-{}", std::process::id())));
    // cold start: this proof needs an empty disk cache
    let _ = std::fs::remove_dir_all(&cache_dir);
    println!("streamtest: {} (layers 0-2, cache {})", url, cache_dir.display());

    let src = RemoteSource::open(&url, cache_dir.clone(), &[0, 1, 2]);
    let tensors = [
        "layers.1.block_sparse_moe.gate.weight", // MoE router
        "layers.1.block_sparse_moe.experts.0.w1", // expert w1 (MXFP4)
        "layers.1.self_attn.q_proj.weight",      // KDA q_proj
    ];

    // 1) cold fetch through the persistent cache, byte-compare with slice_st
    println!("-- 1) cold fetch (network) + byte comparison with slice_st's fetch");
    for name in tensors {
        let net0 = crate::http::fetched_bytes();
        let cached = src.tensor_bytes(name);
        let net = crate::http::fetched_bytes() - net0;
        let reference = src.direct_bytes(name);
        assert_eq!(cached, reference, "{}: cached bytes differ from slice_st's fetch", name);
        println!(
            "  {:<48} {} bytes, network {}, byte-identical to slice_st: OK",
            name,
            cached.len(),
            mb(net)
        );
        assert!(net > 0, "{}: cold fetch used no network", name);
    }

    // 2) warm fetch: disk cache, zero network
    println!("-- 2) warm fetch (disk cache)");
    for name in tensors {
        let net0 = crate::http::fetched_bytes();
        let req0 = crate::http::fetched_requests();
        let b = src.tensor_bytes(name);
        let net = crate::http::fetched_bytes() - net0;
        assert_eq!(net, 0, "{}: warm fetch hit the network ({} bytes)", name, net);
        assert_eq!(crate::http::fetched_requests(), req0);
        println!("  {:<48} {} bytes, network 0 B (disk cache): OK", name, b.len());
    }

    // 3) LRU budget: 3 entries fit, the 4th evicts the least recently used
    println!("-- 3) LRU eviction under a 3-entry budget");
    let entry = 1 << 20; // 1 MB synthetic entries
    let cache = ExpertCache {
        inner: Mutex::new(Lru { map: HashMap::new(), queue: VecDeque::new(), cur: 0, tick: 0, budget: 3 * entry }),
        src: Src::Remote(src),
    };
    for e in 0..4u32 {
        cache.inner.lock().unwrap().insert((0, e), Arc::new(vec![(e + 1) as u8; entry]));
    }
    {
        let lru = cache.inner.lock().unwrap();
        assert!(lru.cur <= 3 * entry, "LRU over budget: {} > {}", lru.cur, 3 * entry);
        assert!(!lru.map.contains_key(&(0, 0)), "oldest entry was not evicted");
        assert!(lru.map.contains_key(&(0, 3)), "newest entry is missing");
        println!(
            "  4 x {} inserted under a {} budget: resident {}, oldest evicted, newest present: OK",
            mb(entry as u64),
            mb(3 * entry as u64),
            mb(lru.cur as u64)
        );
    }
    let _ = &cache; // silence unused-variable lint if asserts change
    println!("streamtest: all checks passed");
    println!("{}", report_line());
}
