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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// global fetch report counters (one model per process)
static RAM_HITS: AtomicU64 = AtomicU64::new(0);
static RAM_MISSES: AtomicU64 = AtomicU64::new(0);
static DISK_BYTES: AtomicU64 = AtomicU64::new(0); // bytes served by the disk tier

// ── predictive prefetch (--stream-predict N) ──
//
// Expert selection is temporally correlated: within a conversation the experts
// the router picks at MoE layer L correlate with the picks at the next MoE
// layer. The cache observes every actual pick (model.rs pulls the 16 selected
// experts through ExpertCache::get), so a Markov predictor can be trained
// online: per-layer marginal expert frequencies plus per-(layer, prev_expert)
// transition counts into the next layer, exponentially decayed (halflife
// DECAY_HALFLIFE token passes) so the model adapts to topic drift. When a
// layer's top-k set is complete, the predicted top-N experts of the NEXT MoE
// layer are fetched on a detached thread while the current layer computes.
// Prediction only changes WHEN bytes land in the RAM LRU, never WHICH experts
// are computed: a mispredicted prefetch is a wasted fetch, an unpredicted pick
// is fetched on demand as usual. The served bytes are always the same file
// bytes, so the output stays bit-identical.
static PREDICT_N: AtomicUsize = AtomicUsize::new(0); // predicted experts per layer, 0 = off
static TOP_K: AtomicUsize = AtomicUsize::new(16); // router top-k (batch size of one MoE layer)
static PREF_ISSUED: AtomicU64 = AtomicU64::new(0); // prefetches that fetched bytes
static PREF_CACHED: AtomicU64 = AtomicU64::new(0); // predicted experts already in RAM
static PREF_USED: AtomicU64 = AtomicU64::new(0); // prefetched entries later consumed on demand
static PRED_HIT: AtomicU64 = AtomicU64::new(0); // predicted experts the router actually picked
static PRED_TOT: AtomicU64 = AtomicU64::new(0); // total predicted experts

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
static FUSE_READS: AtomicU64 = AtomicU64::new(0); // physical span reads issued by warm_batch
static FUSE_EXPERTS: AtomicU64 = AtomicU64::new(0); // experts served by those span reads
static FUSE_NS: AtomicU64 = AtomicU64::new(0); // wall time of the warm_batch read phase

/// `--stream-predict N`: enable the Markov expert prefetcher with N predicted
/// experts per MoE layer (0 = off, the default). `top_k` is the router's
/// top-k from the model config (the expert batch size of one MoE layer).
pub fn set_predict(n: usize, top_k: usize) {
    PREDICT_N.store(n, Ordering::Relaxed);
    TOP_K.store(top_k.max(1), Ordering::Relaxed);
}

/// Number of experts the prefetcher targets per MoE layer (--stream-predict
/// N, 0 = off).
pub fn predict_n() -> usize {
    PREDICT_N.load(Ordering::Relaxed)
}

// ── router-lookahead prefetch (streaming v2) ──
//
// Next-layer expert prediction via the ROUTER ITSELF instead of history:
// model.rs runs the next MoE layer's gate GEMV on the current MoE input
// (the closest available state to what that router will see) and pushes the
// predicted ids to ExpertCache::prefetch. External measurements put this at
// ~72% recall vs ~41% for the Markov/history approach. ON by default when
// --stream-predict > 0 (it REPLACES the Markov predictor: both would
// compete for the same prefetch bandwidth); MICROKIMI_LOOKAHEAD=0 reverts
// to the Markov predictor.
pub fn lookahead_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_LOOKAHEAD").map(|v| v != "0").unwrap_or(true))
}

fn mb(n: u64) -> String {
    format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
}

// ── expert request trace (MICROKIMI_TRACE=/path/trace.bin) ──
//
// The router picks do not depend on the cache contents, so every demand
// request ExpertCache::get serves can be recorded as an ordered (layer,
// expert) stream and replayed OFFLINE under any cache policy and capacity
// (see tools_replay.rs, `microkimi cachereplay`): one traced run yields the
// whole hit-rate vs capacity curve without rerunning the model.
//
// Trace format: raw little-endian record stream, no header. One record per
// demand request, 8 bytes: u32 layer LE ++ u32 expert LE. Records are in
// call order (within one MoE layer the 16 parallel pool jobs serialize on
// the trace mutex, so the intra-layer order is the lock acquisition order;
// layers never interleave, pool barrier). Recording is demand-only:
// prefetches do not record. Unbuffered writes (one 8-byte write per record)
// so a process exit() cannot lose tail records; zero overhead when the env
// var is absent (one OnceLock read + None check per request).
static TRACE_SINK: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();

/// The trace sink, initialized once from MICROKIMI_TRACE. Logs one startup
/// line when active.
fn trace_sink() -> &'static Option<Mutex<std::fs::File>> {
    TRACE_SINK.get_or_init(|| match std::env::var("MICROKIMI_TRACE") {
        Ok(p) if !p.is_empty() => match std::fs::File::create(&p) {
            Ok(f) => {
                println!("stream: tracing expert requests to {} (MICROKIMI_TRACE, u32 layer LE ++ u32 expert LE)", p);
                Some(Mutex::new(f))
            }
            Err(e) => {
                eprintln!("warning: cannot open trace file {}: {}", p, e);
                None
            }
        },
        _ => None,
    })
}

/// Appends one (layer, expert) record to the trace (no-op when tracing is
/// off).
fn trace_record(layer: u32, expert: u32) {
    if let Some(m) = trace_sink() {
        use std::io::Write;
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&layer.to_le_bytes());
        b[4..].copy_from_slice(&expert.to_le_bytes());
        m.lock().unwrap().write_all(&b).ok();
    }
}

/// One-line fetch report printed at exit when --stream is active.
pub fn report_line() -> String {
    let mut s = format!(
        "stream: {} expert RAM hits, {} misses, {} read from the disk tier, {} fetched over HTTP ({} requests)",
        RAM_HITS.load(Ordering::Relaxed),
        RAM_MISSES.load(Ordering::Relaxed),
        mb(DISK_BYTES.load(Ordering::Relaxed)),
        mb(crate::http::fetched_bytes()),
        crate::http::fetched_requests(),
    );
    let (fe, fr) = (FUSE_EXPERTS.load(Ordering::Relaxed), FUSE_READS.load(Ordering::Relaxed));
    if fe > 0 {
        s.push_str(&format!(
            "\nstream-fuse: {} experts served by {} span reads ({:.2} experts/read, {:.2}s in the batched read phase)",
            fe,
            fr,
            fe as f64 / fr.max(1) as f64,
            FUSE_NS.load(Ordering::Relaxed) as f64 / 1e9,
        ));
    }
    if PREDICT_N.load(Ordering::Relaxed) > 0 {
        let issued = PREF_ISSUED.load(Ordering::Relaxed);
        let used = PREF_USED.load(Ordering::Relaxed);
        let wasted = if issued > 0 { 100.0 * (issued - used) as f64 / issued as f64 } else { 0.0 };
        let recall = if issued > 0 { 100.0 * used as f64 / issued as f64 } else { 0.0 };
        s.push_str(&format!(
            "\nstream-predict: {} prefetched ({} already cached), {} consumed on demand ({:.0}% recall, {:.0}% of fetched wasted)",
            issued,
            PREF_CACHED.load(Ordering::Relaxed),
            used,
            recall,
            wasted,
        ));
        // Markov-only line (the lookahead path does not train the predictor)
        let (hit, tot) = (PRED_HIT.load(Ordering::Relaxed), PRED_TOT.load(Ordering::Relaxed));
        if tot > 0 {
            let acc = 100.0 * hit as f64 / tot as f64;
            s.push_str(&format!(", markov prediction accuracy {}/{} ({:.0}%)", hit, tot, acc));
        }
    }
    s
}

// ── RAM LRU ──

/// MICROKIMI_NO_LFU=1: restore pure LRU eviction (A/B toggle). The default
/// is LFU with a recency tie-break: the eviction victim is the entry with
/// the lowest use count (1 at insert, +1 per demand hit), ties broken by
/// the oldest last access. Expert reuse is strongly bimodal (a hot working
/// set of frequently re-routed experts vs one-shot picks), so frequency is
/// a better residency signal than recency alone; the cachereplay LFU column
/// quantifies it offline. NOTE: an unused prefetch sits at count 1 like a
/// one-shot demand pick and ages out by recency - a 0-count scheme was
/// measured to self-evict fresh prefetch batches BEFORE the router could
/// consume them (0% prefetch recall), because every count-0 entry is a
/// better victim than every proven entry.
fn lfu_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_NO_LFU").map(|v| v != "1").unwrap_or(true))
}

struct Lru {
    map: HashMap<(u32, u32), (Arc<Vec<u8>>, u64, bool, u64, bool)>, // (layer, expert) -> (w1++w2++w3, tick, from_prefetch, hits, warm)
    queue: VecDeque<((u32, u32), u64)>,                             // access order, stale gens skipped (pure-LRU mode)
    cur: usize,
    tick: u64,
    budget: usize,
}

impl Lru {
    /// On a hit, also reports (and clears) the from_prefetch mark, so a
    /// prefetched entry consumed on demand is counted exactly once. A
    /// warm-marked entry (batched demand fetch, warm_batch) is served
    /// WITHOUT bumping the LFU count: the insert already carries the demand
    /// credit of the request consuming it now. Counting it again would push
    /// every fused entry to count >= 2 (even one-shot picks) and drown the
    /// one-shot/reused distinction the LFU victim choice relies on - with a
    /// full cache the next warm wave would then find no better victim than
    /// its own fresh inserts (measured: 6x the disk traffic at a tight
    /// budget).
    fn get(&mut self, k: (u32, u32)) -> Option<(Arc<Vec<u8>>, bool)> {
        if !self.map.contains_key(&k) {
            return None;
        }
        self.tick += 1;
        let e = self.map.get_mut(&k).unwrap();
        e.1 = self.tick;
        if !std::mem::take(&mut e.4) {
            e.3 += 1; // LFU: one more demand hit
        }
        let v = e.0.clone();
        let pref = std::mem::take(&mut e.2);
        self.queue.push_back((k, self.tick));
        Some((v, pref))
    }

    /// `warm` marks a batched demand fetch (warm_batch): the first demand
    /// get is served without a count bump (see get).
    fn insert(&mut self, k: (u32, u32), v: Arc<Vec<u8>>, pref: bool, warm: bool) {
        let sz = v.len();
        if sz > self.budget {
            return; // a single expert exceeds the budget: serve without caching
        }
        self.tick += 1;
        // LFU count: 1 at insert (demand or prefetch alike - see lfu_on's
        // comment for why prefetch does not start at 0), +1 per demand hit;
        // a re-insert keeps the hits earned
        if let Some(old) = self.map.insert(k, (v, self.tick, pref, 0, warm)) {
            self.cur -= old.0.len();
            self.map.get_mut(&k).unwrap().3 = old.3 + 1;
        } else {
            self.map.get_mut(&k).unwrap().3 = 1;
        }
        self.cur += sz;
        self.queue.push_back((k, self.tick));
        // evict until back under budget (keep the new entry)
        while self.cur > self.budget && self.map.len() > 1 {
            if lfu_on() {
                // LFU victim: lowest hit count, then oldest access. O(map)
                // per victim, negligible next to the disk fetch that
                // triggered the insert (and amortized: several victims may
                // leave per insert but each scan is a tight integer pass).
                let victim = self.map.iter().min_by_key(|(_, e)| (e.3, e.1)).map(|(&k, _)| k).unwrap();
                self.cur -= self.map[&victim].0.len();
                self.map.remove(&victim);
            } else {
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
        }
        // amortized queue compaction (every access pushes one entry)
        if self.queue.len() > 4 * self.map.len().max(16) {
            let mut live: Vec<((u32, u32), u64)> = self.map.iter().map(|(&k, &(_, g, _, _, _))| (k, g)).collect();
            live.sort_by_key(|&(_, g)| g);
            self.queue = live.into();
        }
    }
}

// ── disk tier manifest: per-repo access book-keeping for the disk LRU ──
//
// atime is unreliable (mount options, tmp+rename persistence), so every repo
// cache dir keeps a tiny manifest.json: tensor name -> {size, last access
// (unix seconds)}. It is rewritten (tmp + rename, crash-safe) on process exit
// and every MANIFEST_FLUSH_EVERY updates to bound the IO cost. A missing or
// corrupt manifest is rebuilt from the files on disk (size + mtime), so a
// kill -9 mid-run loses at most the unflushed access timestamps.

const MANIFEST_FLUSH_EVERY: u64 = 64;

struct Manifest {
    map: HashMap<String, (u64, u64)>, // tensor name -> (size bytes, last access unix)
    dirty: u64,
}

fn now_unix() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl Manifest {
    /// Loads <dir>/manifest.json, tolerating a missing or corrupt file, then
    /// reconciles with the files actually on disk: a crash between a tensor
    /// write and the next manifest flush must not lose the entry.
    fn load(dir: &std::path::Path) -> Manifest {
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
fn is_expert_tensor(name: &str) -> bool {
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
/// persistent per-tensor disk cache: every remote byte is fetched once, ever
/// (unless the --stream-disk rollover evicts it).
pub struct RemoteSource {
    st: crate::slice_st::StDir,
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
        let mut st = crate::slice_st::StDir::open(url, "/tmp/microkimi-stream");
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
    fn direct_bytes(&self, name: &str) -> Vec<u8> {
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

/// Halflife of the statistics decay, in token passes (one pass = one sweep
/// over the MoE layers). Counts older than this weigh half as much, so the
/// predictor follows topic drift within a conversation.
const DECAY_HALFLIFE: f32 = 256.0;
/// Per-(layer, prev_expert) transition rows are capped: when a row grows past
/// ROW_CAP entries it is pruned back to ROW_KEEP by effective count. Predicted
/// experts come from the head of the distribution anyway.
const ROW_CAP: usize = 128;
const ROW_KEEP: usize = 64;
/// Weight of the per-layer marginal frequency relative to the transition
/// counts: a backoff for experts with no observed transition history.
const MARGINAL_W: f32 = 0.25;

/// Decay factor for a count stamped `stamp` read at `epoch` (lazy decay: each
/// entry carries the epoch of its last update, no global rescale pass).
fn decay(epoch: u64, stamp: u64) -> f32 {
    (-(epoch.saturating_sub(stamp) as f32) / DECAY_HALFLIFE).exp2()
}

/// Online Markov model of the router's expert picks, fed by ExpertCache::get.
/// Batching: model.rs fetches the top-k experts of one MoE layer as one
/// parallel pool batch and layers run sequentially, so the gets of one layer
/// never interleave with the next layer's (pool barrier). A get whose layer
/// differs from the open batch closes it.
/// pub(crate): also driven offline by tools_replay (cachereplay), as-is.
pub(crate) struct Predictor {
    cur_layer: u32,            // layer of the open batch (u32::MAX = none yet)
    cur_set: Vec<u32>,         // distinct experts of the open batch
    prev_set: Vec<u32>,        // last closed batch (the previous MoE layer)
    epoch: u64,                // token pass counter (incremented on layer wrap)
    last_fired: (u64, u32),    // (epoch, layer) of the last prefetch trigger
    next_layer: HashMap<u32, u32>, // observed MoE layer sequence: layer -> next MoE layer
    marginal: HashMap<u32, HashMap<u32, (f32, u64)>>, // layer -> expert -> (count, stamp)
    trans: HashMap<(u32, u32), HashMap<u32, (f32, u64)>>, // (layer, prev_expert) -> expert -> (count, stamp)
    pred_made: HashMap<u32, Vec<u32>>, // layer -> experts predicted for it (accuracy accounting)
    blob: usize,               // bytes per MXFP4 blob (constant per model)
    offs: HashMap<(u32, u32), [u64; 3]>, // observed file offsets per (layer, expert)
    base: HashMap<u32, (u64, bool)>,     // layer -> (offs[0] - expert * 3 * blob, affine-ok)
}

/// One predicted expert to fetch in the background: (layer, expert, offsets).
pub(crate) type PrefetchJob = (u32, u32, [u64; 3]);

impl Predictor {
    pub(crate) fn new() -> Predictor {
        Predictor {
            cur_layer: u32::MAX,
            cur_set: Vec::new(),
            prev_set: Vec::new(),
            epoch: 0,
            last_fired: (u64::MAX, u32::MAX),
            next_layer: HashMap::new(),
            marginal: HashMap::new(),
            trans: HashMap::new(),
            pred_made: HashMap::new(),
            blob: 0,
            offs: HashMap::new(),
            base: HashMap::new(),
        }
    }

    /// Adds `incr` to a lazily-decayed (count, stamp) entry.
    fn bump(e: &mut (f32, u64), epoch: u64, incr: f32) {
        e.0 = e.0 * decay(epoch, e.1) + incr;
        e.1 = epoch;
    }

    /// Records one observed router pick. Returns prefetch jobs for the next
    /// MoE layer when the open batch just reached top-k (the full router set
    /// of a decode step): the current layer's experts are all in flight, so
    /// the prefetch overlaps the current layer's compute.
    pub(crate) fn observe(&mut self, layer: u32, expert: u32, offs: [u64; 3], blob: usize, top_k: usize, n: usize) -> Vec<PrefetchJob> {
        // offset book-keeping (for prefetching experts never seen before)
        self.blob = blob;
        self.offs.insert((layer, expert), offs);
        let stride_ok = offs[1] == offs[0] + blob as u64 && offs[2] == offs[1] + blob as u64;
        let b = offs[0].wrapping_sub(expert as u64 * 3 * blob as u64);
        match self.base.get_mut(&layer) {
            None => {
                self.base.insert(layer, (b, stride_ok));
            }
            Some(e) => {
                if !stride_ok || e.0 != b {
                    e.1 = false; // layout is not a plain expert-major run: recorded offsets only
                }
            }
        }
        // batching
        if layer != self.cur_layer {
            self.close_batch();
            if self.cur_layer != u32::MAX {
                self.next_layer.insert(self.cur_layer, layer);
                if layer < self.cur_layer {
                    self.epoch += 1; // layers increase within a token pass: a wrap is a new token
                }
            }
            self.prev_set = std::mem::take(&mut self.cur_set);
            self.cur_layer = layer;
        }
        if !self.cur_set.contains(&expert) {
            self.cur_set.push(expert);
        }
        // trigger: the batch just reached the router's top-k (complete set)
        if n == 0 || self.cur_set.len() != top_k || self.last_fired == (self.epoch, self.cur_layer) {
            return Vec::new();
        }
        self.last_fired = (self.epoch, self.cur_layer);
        // accuracy of the prediction made for THIS layer earlier in the pass
        if let Some(pred) = self.pred_made.remove(&self.cur_layer) {
            let hits = pred.iter().filter(|e| self.cur_set.contains(e)).count() as u64;
            PRED_HIT.fetch_add(hits, Ordering::Relaxed);
            PRED_TOT.fetch_add(pred.len() as u64, Ordering::Relaxed);
        }
        let Some(&nl) = self.next_layer.get(&self.cur_layer) else {
            return Vec::new(); // first pass: the layer sequence is still being learned
        };
        let predicted = self.top_predicted(nl, n);
        self.pred_made.insert(nl, predicted.clone());
        // resolve file offsets: affine layout when verified, else observed only
        let mut jobs = Vec::new();
        for e in predicted {
            if let Some(o) = self.offs.get(&(nl, e)) {
                jobs.push((nl, e, *o));
            } else if let Some(&(b, true)) = self.base.get(&nl) {
                let o0 = b + e as u64 * 3 * self.blob as u64;
                jobs.push((nl, e, [o0, o0 + self.blob as u64, o0 + 2 * self.blob as u64]));
            }
        }
        jobs
    }

    /// Folds the closing batch into the statistics: marginals for its own
    /// layer, transitions from the previous MoE layer's set into this one.
    fn close_batch(&mut self) {
        if self.cur_set.is_empty() || self.cur_layer == u32::MAX {
            return;
        }
        let epoch = self.epoch;
        let layer = self.cur_layer;
        let row = self.marginal.entry(layer).or_default();
        for &e in &self.cur_set {
            Self::bump(row.entry(e).or_insert((0.0, epoch)), epoch, 1.0);
        }
        for &p in &self.prev_set {
            let row = self.trans.entry((layer, p)).or_default();
            for &e in &self.cur_set {
                Self::bump(row.entry(e).or_insert((0.0, epoch)), epoch, 1.0);
            }
            if row.len() > ROW_CAP {
                // keep the ROW_KEEP strongest entries by effective count
                let mut v: Vec<(u32, f32)> = row.iter().map(|(&e, &(c, s))| (e, c * decay(epoch, s))).collect();
                v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                v.truncate(ROW_KEEP);
                *row = v.into_iter().map(|(e, c)| (e, (c, epoch))).collect();
            }
        }
    }

    /// Top-n predicted experts of `layer`: transition counts from the current
    /// layer's actual picks, plus the layer marginal as a backoff.
    fn top_predicted(&self, layer: u32, n: usize) -> Vec<u32> {
        let epoch = self.epoch;
        let mut score: HashMap<u32, f32> = HashMap::new();
        for &p in &self.cur_set {
            if let Some(row) = self.trans.get(&(layer, p)) {
                for (&e, &(c, s)) in row {
                    *score.entry(e).or_insert(0.0) += c * decay(epoch, s);
                }
            }
        }
        if let Some(row) = self.marginal.get(&layer) {
            for (&e, &(c, s)) in row {
                *score.entry(e).or_insert(0.0) += MARGINAL_W * c * decay(epoch, s);
            }
        }
        // never predict what the router is already fetching for the current
        // layer... different layer, so no overlap possible; just rank
        let mut v: Vec<(u32, f32)> = score.into_iter().collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        v.truncate(n);
        v.into_iter().map(|(e, _)| e).collect()
    }
}

// ── expert cache: RAM LRU over a byte source ──

// ── runtime A/B toggles (env vars, read once per process) ──

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
fn fake_disk_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| std::env::var("MICROKIMI_FAKE_DISK_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(0))
}

/// One serialized fake disk latency, no-op unless MICROKIMI_FAKE_DISK_MS set.
fn fake_disk_sleep() {
    let ms = fake_disk_ms();
    if ms > 0 {
        static FAKE_DISK: Mutex<()> = Mutex::new(());
        let _io = FAKE_DISK.lock().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(ms));
    }
}

fn env_off(name: &str) -> bool {
    std::env::var(name).map(|v| v == "1").unwrap_or(false)
}

// ── run fusion: merge file-adjacent expert reads into one span read ──

/// Maximum hole between two expert footprints still merged into one physical
/// read: covers the format alignment padding between blobs (zero on densely
/// packed .bins). The hole bytes are read and discarded; on a latency-bound
/// disk one span read still beats two separate reads.
const FUSE_GAP_MAX: u64 = 4096;

/// An expert request is compact when its three blobs span at most their
/// payload plus per-blob alignment padding. Non-compact requests (scattered
/// w1/w2/w3) never join a run: merging them would read an unbounded hole.
fn compact(offs: &[u64; 3], blob: usize) -> bool {
    offs[1] >= offs[0] && offs[2] >= offs[1] && offs[2] + blob as u64 - offs[0] <= 3 * blob as u64 + 2 * FUSE_GAP_MAX
}

/// Groups offset-sorted expert requests (expert, offs, blob) into maximal
/// runs of file-adjacent compact footprints (footprint = [offs[0], offs[2] +
/// blob)): two consecutive requests merge when both are compact and the gap
/// between their footprints is at most FUSE_GAP_MAX. Returns (start, end)
/// index ranges into the input slice. Pure: unit-tested directly.
fn fuse_runs(sorted: &[(u32, [u64; 3], usize)]) -> Vec<(usize, usize)> {
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
const O_DIRECT: i32 = 0o200000;
#[cfg(all(target_os = "linux", not(any(target_arch = "aarch64", target_arch = "arm", target_arch = "m68k"))))]
const O_DIRECT: i32 = 0o40000;

/// Opens the direct-I/O handle for `path`: O_DIRECT on Linux, an F_NOCACHE
/// fd on macOS. None when MICROKIMI_NO_ODIRECT=1, on other OSes, or when the
/// open/flag is rejected: the caller falls back to plain buffered pread.
fn open_direct(path: &str) -> Option<std::fs::File> {
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
fn read_direct(f: &std::fs::File, off: u64, out: &mut [u8]) -> std::io::Result<()> {
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

/// Local .bin source: expert blobs pread from the model file itself, through
/// the O_DIRECT handle when available (Linux page-cache bypass), the plain
/// buffered handle otherwise.
struct LocalSrc {
    file: std::fs::File,
    direct: Option<std::fs::File>,
}

enum Src {
    /// Local .bin: expert blobs pread from the model file itself.
    Local(LocalSrc),
    /// Remote safetensors: per-tensor persistent cache (+ HTTP on cold miss).
    Remote(RemoteSource),
}

/// Shared state behind an Arc so detached prefetch threads can outlive the
/// trigger point safely (the model - and with it the cache - may be dropped
/// while a prefetch is still reading).
struct CacheInner {
    lru: Mutex<Lru>,
    src: Src,
    pred: Mutex<Predictor>,
}

impl CacheInner {
    /// Raw bytes of one expert from the disk/HTTP tier (no RAM LRU lookup).
    fn fetch(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize) -> Vec<u8> {
        match &self.src {
            Src::Local(l) => {
                use std::os::unix::fs::FileExt;
                let mut out = vec![0u8; 3 * blob];
                // the .bin writers emit w1, w2, w3 back to back: one read
                let contiguous = offs[1] == offs[0] + blob as u64 && offs[2] == offs[1] + blob as u64;
                // O_DIRECT first (page-cache bypass), plain pread as fallback.
                // Either path serves the same file bytes.
                let served = match &l.direct {
                    Some(df) if contiguous => read_direct(df, offs[0], &mut out).is_ok(),
                    Some(df) => (0..3).all(|i| read_direct(df, offs[i], &mut out[i * blob..(i + 1) * blob]).is_ok()),
                    None => false,
                };
                if !served {
                    if contiguous {
                        l.file.read_exact_at(&mut out, offs[0]).unwrap();
                    } else {
                        for i in 0..3 {
                            l.file.read_exact_at(&mut out[i * blob..(i + 1) * blob], offs[i]).unwrap();
                        }
                    }
                }
                fake_disk_sleep(); // one physical read batch served (bench knob)
                DISK_BYTES.fetch_add(out.len() as u64, Ordering::Relaxed);
                out
            }
            Src::Remote(r) => r.expert_blobs(layer, expert),
        }
    }

    /// One fused run of file-adjacent experts (see fuse_runs): the span from
    /// the first member's w1 to the last member's w3 tail is served by a
    /// single physical read (O_DIRECT when available, buffered pread
    /// otherwise), then each member's w1 ++ w2 ++ w3 payload is extracted by
    /// its directory offsets and inserted into the RAM LRU. The inserted
    /// bytes are byte-identical to per-expert preads of the same ranges.
    /// Singleton runs go through the plain per-expert fetch.
    fn fetch_run(&self, layer: u32, members: &[(u32, [u64; 3], usize)]) {
        if members.len() == 1 {
            let (e, offs, blob) = members[0];
            let bytes = self.fetch(layer, e, offs, blob);
            self.lru.lock().unwrap().insert((layer, e), Arc::new(bytes), false, true);
            RAM_MISSES.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let Src::Local(l) = &self.src else { return };
        use std::os::unix::fs::FileExt;
        let start = members[0].1[0];
        let end = members.iter().map(|&(_, o, b)| o[2] + b as u64).max().unwrap();
        let mut span = vec![0u8; (end - start) as usize];
        let served = match &l.direct {
            Some(df) => read_direct(df, start, &mut span).is_ok(),
            None => false,
        };
        if !served {
            l.file.read_exact_at(&mut span, start).unwrap();
        }
        fake_disk_sleep(); // ONE physical read for the whole run
        FUSE_READS.fetch_add(1, Ordering::Relaxed);
        for &(e, offs, blob) in members {
            let mut bytes = vec![0u8; 3 * blob];
            for i in 0..3 {
                let lo = (offs[i] - start) as usize;
                bytes[i * blob..(i + 1) * blob].copy_from_slice(&span[lo..lo + blob]);
            }
            DISK_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            FUSE_EXPERTS.fetch_add(1, Ordering::Relaxed);
            RAM_MISSES.fetch_add(1, Ordering::Relaxed); // a demand miss served by the batch
            self.lru.lock().unwrap().insert((layer, e), Arc::new(bytes), false, true);
        }
    }

    /// Background prefetch of one predicted expert: already cached = just
    /// refresh the recency (protects it from eviction), otherwise fetch and
    /// insert with the from_prefetch mark. Never touches the router path.
    fn prefetch_one(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize) {
        let k = (layer, expert);
        {
            let mut lru = self.lru.lock().unwrap();
            if lru.get(k).is_some() {
                PREF_CACHED.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let bytes = self.fetch(layer, expert, offs, blob);
        self.lru.lock().unwrap().insert(k, Arc::new(bytes), true, false);
        PREF_ISSUED.fetch_add(1, Ordering::Relaxed);
    }
}

/// RAM LRU of packed expert bytes keyed by (layer, expert), budgeted by
/// --stream-ram. Thread-safe: the 16 router-selected experts of a layer are
// fetched and computed in parallel pool jobs (model.rs).
pub struct ExpertCache {
    inner: Arc<CacheInner>,
}

impl ExpertCache {
    /// Local streaming source: `path` is the .bin the spine was loaded from.
    pub fn local(path: &str, ram_mb: usize) -> ExpertCache {
        let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
        let direct = open_direct(path);
        // one-time startup lines: state of the runtime A/B toggles
        let dio_name = if cfg!(target_os = "macos") { "F_NOCACHE" } else { "O_DIRECT" };
        if direct.is_some() {
            println!("stream: {} on (MICROKIMI_NO_ODIRECT=1 to disable)", dio_name);
        } else if cfg!(any(target_os = "linux", target_os = "macos")) && env_off("MICROKIMI_NO_ODIRECT") {
            println!("stream: {} off (MICROKIMI_NO_ODIRECT=1)", dio_name);
        } else if cfg!(any(target_os = "linux", target_os = "macos")) {
            println!("stream: {} unavailable (open failed, buffered pread fallback)", dio_name);
        } else {
            println!("stream: direct I/O n/a (buffered pread)");
        }
        if offset_sort() {
            println!("stream: offset-sorted expert reads on (MICROKIMI_NO_OFFSORT=1 to disable)");
        } else {
            println!("stream: offset-sorted expert reads off (MICROKIMI_NO_OFFSORT=1)");
        }
        if run_fuse() {
            println!("stream: contiguous-run read fusion on (MICROKIMI_NO_RUNFUSE=1 to disable)");
        } else {
            println!("stream: contiguous-run read fusion off (MICROKIMI_NO_RUNFUSE=1)");
        }
        if fake_disk_ms() > 0 {
            println!("stream: FAKE disk latency {} ms/read, serialized (MICROKIMI_FAKE_DISK_MS, bench only)", fake_disk_ms());
        }
        // initialize the trace sink now so its startup line prints with the
        // other stream lines (no-op when MICROKIMI_TRACE is unset)
        trace_sink();
        ExpertCache {
            inner: Arc::new(CacheInner {
                lru: Mutex::new(Lru { map: HashMap::new(), queue: VecDeque::new(), cur: 0, tick: 0, budget: ram_mb << 20 }),
                src: Src::Local(LocalSrc { file, direct }),
                pred: Mutex::new(Predictor::new()),
            }),
        }
    }

    /// Remote streaming source (per-tensor persistent cache, disk budget in MB).
    #[allow(dead_code)] // wired end-to-end once real-dim MLA layers run (see docs)
    pub fn remote(url: &str, ram_mb: usize, kept_layers: &[usize], disk_mb: u64) -> ExpertCache {
        ExpertCache {
            inner: Arc::new(CacheInner {
                lru: Mutex::new(Lru { map: HashMap::new(), queue: VecDeque::new(), cur: 0, tick: 0, budget: ram_mb << 20 }),
                src: Src::Remote(RemoteSource::open_disk(url, default_cache_root(url), kept_layers, disk_mb)),
                pred: Mutex::new(Predictor::new()),
            }),
        }
    }

    /// Batched demand fetch of one MoE layer's selected experts (local
    /// source): every requested expert missing from the RAM LRU is fetched,
    /// but maximal runs of file-adjacent experts (fuse_runs over the
    /// offset-sorted requests) are served by ONE span read each instead of
    /// one pread per expert. model.rs calls this once per MoE layer before
    /// submitting the compute jobs, whose ExpertCache::get then hits the RAM
    /// LRU; the run reads themselves run in parallel pool jobs (one per
    /// run), keeping the cross-run parallelism of the historical per-expert
    /// submission. Only the number of physical reads changes: the served
    /// bytes are exactly those of the individual preads, so the model output
    /// is bit-identical. No-op for the remote source (per-tensor cache
    /// files, no shared spans), for batches smaller than 2, and with
    /// MICROKIMI_NO_RUNFUSE=1.
    ///
    /// Note on accounting: the experts fetched here count as RAM misses
    /// (they were not in the cache) and the compute jobs' gets then count as
    /// RAM hits, so the report's hits inflate by the batched amount; the
    /// stream-fuse line carries the real batched-fetch figures.
    pub fn warm_batch(&self, layer: u32, items: &[(u32, [u64; 3], usize)]) {
        if items.len() < 2 || !run_fuse() || !matches!(self.inner.src, Src::Local(_)) {
            return;
        }
        // misses only, offset-sorted (the LRU may change before the reads
        // land: a concurrent prefetch of the same expert is harmless - the
        // run insert refreshes the same file bytes)
        let mut miss: Vec<(u32, [u64; 3], usize)> = Vec::with_capacity(items.len());
        {
            let lru = self.inner.lru.lock().unwrap();
            for &(e, offs, blob) in items {
                if !lru.map.contains_key(&(layer, e)) {
                    miss.push((e, offs, blob));
                }
            }
            // a batch larger than the budget would evict its own earliest
            // inserts before the compute jobs consume them (double fetch):
            // leave those to the plain per-expert path
            let miss_bytes: usize = miss.iter().map(|&(_, _, b)| 3 * b).sum();
            if miss_bytes > lru.budget {
                return;
            }
        }
        if miss.len() < 2 {
            return; // a lone miss is fetched by its compute job as usual
        }
        miss.sort_by_key(|&(_, o, _)| o[0]);
        let runs = fuse_runs(&miss);
        let t0 = std::time::Instant::now();
        let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(runs.len());
        for (s, e) in runs {
            let members: Vec<(u32, [u64; 3], usize)> = miss[s..e].to_vec();
            let inner = Arc::clone(&self.inner);
            jobs.push(Box::new(move || inner.fetch_run(layer, &members)));
        }
        crate::pool::pool().run(jobs);
        FUSE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// The 3 MXFP4 blobs of expert `expert` of `layer`, concatenated
    /// w1 ++ w2 ++ w3, `blob` bytes each. `offs` = absolute file offsets of
    /// the three blobs (local source; ignored by the remote one).
    pub fn get(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize) -> Arc<Vec<u8>> {
        // record the demand request (MICROKIMI_TRACE; no-op when unset)
        trace_record(layer, expert);
        // feed the Markov predictor; on a completed top-k batch this may
        // return prefetch jobs for the next MoE layer, run on a detached
        // thread so they overlap the current layer's compute
        let n = PREDICT_N.load(Ordering::Relaxed);
        if n > 0 && !lookahead_on() {
            // Markov predictor path (MICROKIMI_LOOKAHEAD=0); when the
            // router-lookahead is on it replaces this entirely
            let jobs = self.inner.pred.lock().unwrap().observe(layer, expert, offs, blob, TOP_K.load(Ordering::Relaxed), n);
            if !jobs.is_empty() {
                let inner = Arc::clone(&self.inner);
                std::thread::spawn(move || {
                    for (l, e, o) in jobs {
                        inner.prefetch_one(l, e, o, blob);
                    }
                });
            }
        }
        let k = (layer, expert);
        {
            let mut lru = self.inner.lru.lock().unwrap();
            if let Some((v, pref)) = lru.get(k) {
                if pref {
                    PREF_USED.fetch_add(1, Ordering::Relaxed);
                }
                RAM_HITS.fetch_add(1, Ordering::Relaxed);
                return v;
            }
        }
        RAM_MISSES.fetch_add(1, Ordering::Relaxed);
        let bytes = self.inner.fetch(layer, expert, offs, blob);
        let v = Arc::new(bytes);
        self.inner.lru.lock().unwrap().insert(k, v.clone(), false, false);
        v
    }

    /// Router-lookahead prefetch (streaming v2): background-fetch the
    /// (layer, expert, offsets, blob) jobs predicted by the NEXT MoE layer's
    /// router (model.rs runs the gate GEMV on the current MoE input). One
    /// detached thread per batch, same semantics as the Markov prefetch
    /// (prefetch_one: cached = recency refresh, missing = fetch + insert
    /// with the from_prefetch mark; the PREF_* counters apply). Never
    /// touches the router path: the output is unaffected.
    pub fn prefetch(&self, jobs: Vec<(u32, u32, [u64; 3], usize)>) {
        if jobs.is_empty() {
            return;
        }
        let inner = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            for (l, e, o, blob) in jobs {
                inner.prefetch_one(l, e, o, blob);
            }
        });
    }
}

// ── streamtest: remote per-tensor cache + LRU budget proof ──

/// `microkimi streamtest --model https://huggingface.co/org/repo [--cache-dir D] [--stream-disk N]`
///
/// Bandwidth-safe proof of the remote tier against the real K3 repo:
/// 1. cold fetch of 3 real tensors (one MoE router, one expert w1, one KDA
///    q_proj) through the per-tensor persistent cache, byte-compared against
///    slice_st's direct fetch;
/// 2. warm fetch of the same tensors: served from disk, zero network bytes;
/// 3. LRU eviction respects the --stream-ram budget;
/// 4. with --stream-disk N (or env MICROKIMI_STREAM_DISK): disk LRU rollover,
///    expert-only eviction, spine survival and re-fetch of an evicted expert.
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
    // disk cache budget in MB (0 = unlimited, the historical behavior)
    let disk_mb: u64 = args
        .iter()
        .position(|a| a == "--stream-disk")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .or_else(|| std::env::var("MICROKIMI_STREAM_DISK").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(0);
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
        inner: Arc::new(CacheInner {
            lru: Mutex::new(Lru { map: HashMap::new(), queue: VecDeque::new(), cur: 0, tick: 0, budget: 3 * entry }),
            src: Src::Remote(src),
            pred: Mutex::new(Predictor::new()),
        }),
    };
    for e in 0..4u32 {
        cache.inner.lru.lock().unwrap().insert((0, e), Arc::new(vec![(e + 1) as u8; entry]), false, false);
    }
    {
        let lru = cache.inner.lru.lock().unwrap();
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

    // 4) disk LRU rollover (--stream-disk N): expert-only eviction proof.
    // Runs in a fresh subdir so the big spine tensors cached above (a real
    // q_proj alone is ~336 MB) do not dominate the budget arithmetic.
    if disk_mb > 0 {
        println!("-- 4) disk rollover under a {} MB budget (expert-only, spine never evicted)", disk_mb);
        let roll_dir = cache_dir.join("roll");
        let _ = std::fs::remove_dir_all(&roll_dir);
        let roll = RemoteSource::open_disk(&url, roll_dir.clone(), &[0, 1, 2], disk_mb);
        // two small real spine tensors: router bias + input layernorm
        let spine = ["layers.1.block_sparse_moe.gate.e_score_correction_bias", "layers.1.input_layernorm.weight"];
        // real expert w1 blobs (~5.6 MB packed each): 3 fetches overflow an
        // 8 MB budget twice, oldest first
        let experts = [
            "layers.1.block_sparse_moe.experts.0.w1",
            "layers.1.block_sparse_moe.experts.1.w1",
            "layers.1.block_sparse_moe.experts.2.w1",
        ];
        for name in spine {
            roll.tensor_bytes(name);
        }
        for name in experts {
            roll.tensor_bytes(name); // each fetch persists, then rollover runs
        }
        let cached = |n: &str| roll_dir.join(sanitize(n)).is_file();
        assert!(!cached(experts[0]), "{}: oldest expert was not evicted", experts[0]);
        assert!(!cached(experts[1]), "{}: second-oldest expert was not evicted", experts[1]);
        assert!(cached(experts[2]), "{}: newest expert is missing", experts[2]);
        for name in spine {
            assert!(cached(name), "spine tensor {} was evicted (must never happen)", name);
        }
        println!("  3 expert w1 fetched under {} MB: two oldest evicted, spine intact: OK", disk_mb);
        // an evicted expert is re-fetched over HTTP on its next miss
        let net0 = crate::http::fetched_bytes();
        let b = roll.tensor_bytes(experts[0]);
        let net = crate::http::fetched_bytes() - net0;
        assert!(net > 0, "{}: evicted expert re-fetch used no network", experts[0]);
        assert!(cached(experts[0]), "{}: re-fetched expert was not re-cached", experts[0]);
        println!("  re-fetch of evicted {}: {} bytes, network {}: OK", experts[0], b.len(), mb(net));
        // the manifest tracks every cached tensor and is valid JSON
        let mbytes = std::fs::read(roll_dir.join("manifest.json")).expect("manifest.json missing after rollover");
        assert!(crate::json::parse(&mbytes).get("tensors").is_some(), "manifest.json has no tensors object");
        println!("  manifest.json present and valid: OK");
        // spine alone over budget: the 25 MB MoE router (spine) exceeds the
        // budget by itself; with nothing evictable the spine is kept anyway
        let big_dir = cache_dir.join("roll-spine");
        let _ = std::fs::remove_dir_all(&big_dir);
        let big = RemoteSource::open_disk(&url, big_dir.clone(), &[0, 1, 2], disk_mb);
        big.tensor_bytes("layers.1.block_sparse_moe.gate.weight"); // 25 MB spine > 8 MB budget
        big.tensor_bytes(experts[0]); // rollover: the expert is the only evictable tensor
        assert!(big_dir.join(sanitize("layers.1.block_sparse_moe.gate.weight")).is_file(), "spine router was evicted over budget");
        assert!(!big_dir.join(sanitize(experts[0])).is_file(), "{}: expert should have been evicted", experts[0]);
        println!("  spine alone over budget (25 MB router > {} MB): spine kept, expert evicted instead: OK", disk_mb);
    }
    println!("streamtest: all checks passed");
    println!("{}", report_line());
}

// ── cache command: disk cache inspection and cleanup ──

/// unix seconds -> "YYYY-MM-DD HH:MM:SS UTC" (civil-from-days, no tables).
fn fmt_unix(t: u64) -> String {
    if t == 0 {
        return "-".to_string();
    }
    let days = (t / 86400) as i64;
    let secs = t % 86400;
    // Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", y, m, d, secs / 3600, (secs / 60) % 60, secs % 60)
}

fn dir_name(p: &std::path::Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `microkimi cache --info` / `microkimi cache --clean [--repo X]`
///
/// --info: per-repo disk usage under ~/.cache/microkimi (bytes, tensor count,
/// oldest/newest recorded access) plus a total. Access times come from each
/// repo's manifest.json, rebuilt from the files on disk when missing/corrupt.
/// --clean: deletes the cached tensors of every repo (or of --repo X only,
/// matched by its sanitized directory name or the original URL) and prints
/// the freed bytes. Never asks for confirmation.
pub fn cache_cmd(args: &[String]) {
    let info = args.iter().any(|a| a == "--info");
    let clean = args.iter().any(|a| a == "--clean");
    if info == clean {
        eprintln!("usage: microkimi cache --info");
        eprintln!("       microkimi cache --clean [--repo X]");
        std::process::exit(1);
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let root = std::path::PathBuf::from(format!("{}/.cache/microkimi", home));
    let mut repos: Vec<std::path::PathBuf> = std::fs::read_dir(&root)
        .map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect())
        .unwrap_or_default();
    repos.sort();
    if let Some(r) = args.iter().position(|a| a == "--repo").and_then(|i| args.get(i + 1)) {
        // accept the sanitized directory name or the original repo URL
        let bare = r.trim_start_matches("https://").trim_start_matches("http://");
        let want = sanitize(bare);
        repos.retain(|p| {
            let n = dir_name(p);
            n == *r || n == want
        });
        if repos.is_empty() {
            eprintln!("error: no cached repo matches '{}'", r);
            std::process::exit(1);
        }
    }
    if repos.is_empty() {
        println!("cache: no repos under {}", root.display());
        return;
    }
    if info {
        let (mut tot_b, mut tot_n) = (0u64, 0usize);
        for d in &repos {
            let m = Manifest::load(d);
            let bytes: u64 = m.map.values().map(|e| e.0).sum();
            let oldest = m.map.values().map(|e| e.1).filter(|&t| t > 0).min();
            let newest = m.map.values().map(|e| e.1).filter(|&t| t > 0).max();
            println!("{}", dir_name(d));
            println!("  tensors: {}", m.map.len());
            println!("  bytes:   {} ({} B)", mb(bytes), bytes);
            println!("  oldest access: {}", oldest.map(fmt_unix).unwrap_or_else(|| "-".to_string()));
            println!("  newest access: {}", newest.map(fmt_unix).unwrap_or_else(|| "-".to_string()));
            tot_b += bytes;
            tot_n += m.map.len();
        }
        println!("total: {} in {} tensors across {} repo(s)", mb(tot_b), tot_n, repos.len());
    } else {
        let mut tot = 0u64;
        for d in &repos {
            let (mut freed, mut n) = (0u64, 0usize);
            if let Ok(rd) = std::fs::read_dir(d) {
                for e in rd.flatten() {
                    let p = e.path();
                    if !p.is_file() {
                        continue;
                    }
                    freed += e.metadata().map(|m| m.len()).unwrap_or(0);
                    n += 1;
                    std::fs::remove_file(&p).ok();
                }
            }
            println!("{}: freed {} ({} files)", dir_name(d), mb(freed), n);
            tot += freed;
        }
        println!("cache: freed {} total", mb(tot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// request with a compact w1++w2++w3 footprint starting at `off`
    fn req(e: u32, off: u64, blob: u64) -> (u32, [u64; 3], usize) {
        (e, [off, off + blob, off + 2 * blob], blob as usize)
    }

    #[test]
    fn fuse_runs_merges_adjacent() {
        // 4 experts, 1000-byte blobs: 0-1 adjacent, hole, 2-3 adjacent
        let v = vec![req(0, 0, 1000), req(1, 3000, 1000), req(2, 100_000, 1000), req(3, 103_000, 1000)];
        assert_eq!(fuse_runs(&v), vec![(0, 2), (2, 4)]);
    }

    #[test]
    fn fuse_runs_no_adjacency() {
        let v: Vec<_> = (0..16).map(|e| req(e, e as u64 * 1_000_000, 1000)).collect();
        assert_eq!(fuse_runs(&v).len(), 16);
    }

    #[test]
    fn fuse_runs_gap_boundary() {
        // gap exactly FUSE_GAP_MAX merges, FUSE_GAP_MAX + 1 splits
        let v = vec![req(0, 0, 1000), req(1, 3000 + FUSE_GAP_MAX, 1000)];
        assert_eq!(fuse_runs(&v), vec![(0, 2)]);
        let v = vec![req(0, 0, 1000), req(1, 3000 + FUSE_GAP_MAX + 64, 1000)];
        assert_eq!(fuse_runs(&v), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn fuse_runs_overlap_never_merges() {
        // overlapping footprints (corrupt input) split instead of merging
        let v = vec![req(0, 0, 1000), req(1, 100, 1000)];
        assert_eq!(fuse_runs(&v), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn fuse_runs_scattered_blobs_stay_singleton() {
        // expert 1 has its w3 far away (non-compact): no run across it
        let mut v = vec![req(0, 0, 1000), req(1, 3000, 1000), req(2, 6000, 1000)];
        v[1].1[2] = 50_000_000;
        assert_eq!(fuse_runs(&v), vec![(0, 1), (1, 2), (2, 3)]);
    }

    #[test]
    fn fuse_runs_padded_blobs_merge() {
        // 4096-aligned 4352-byte blobs (re-sliced layout): 3840-byte padding
        // holes inside and between footprints still merge into one span
        let b = 4352u64;
        let stride = 3 * b.next_multiple_of(4096);
        let v: Vec<_> = (0..4)
            .map(|e| {
                let o = e as u64 * stride;
                (e, [o, o + 8192, o + 16384], b as usize)
            })
            .collect();
        assert_eq!(fuse_runs(&v), vec![(0, 4)]);
    }

    #[test]
    fn fuse_runs_empty_and_single() {
        assert_eq!(fuse_runs(&[]), Vec::<(usize, usize)>::new());
        assert_eq!(fuse_runs(&[req(7, 42, 1000)]), vec![(0, 1)]);
    }
}
