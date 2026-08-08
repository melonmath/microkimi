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
//
// Optional degraded mode (--stream-fallback, OFF by default): a VQ1 shadow
// of EVERY expert (shadow.rs, 0.5 bit/weight, resident in RAM) is served
// immediately on a cache miss while the full-precision blob is refilled in
// the background, so the decode never blocks on the disk tier. A
// shadow-served expert is NOT bit-identical - a latency mode, not a
// quality mode; the report line counts the degraded computations.
//
// Module map: this root keeps the cache itself (ExpertCache, CacheInner, Lru
// and the eviction policies), the fallback flags, the report counters and the
// unit tests. Submodules: prefetch (Markov/draft/trace-similarity prediction,
// route history), fetch (direct reads, fused spans, remote tier), cmd
// (streamtest and cache CLI diagnostics).

mod cmd;
mod fetch;
mod prefetch;

// The public surface stays crate::stream::*; submodules are private.
pub use cmd::{cache_cmd, streamtest};
pub use fetch::{default_cache_root, offset_sort, run_fuse, RemoteSource};
pub use prefetch::{
    draft_prefetch_on, lookahead_on, predict_n, route_hist_clear, route_lookup, route_record,
    set_predict, tracesim_on,
};
pub(crate) use prefetch::{Predictor, COLD_PASSES, TSIM_MIN_REQS, TSIM_THRESHOLD};
use fetch::{
    env_off, fake_disk_ms, fake_disk_sleep, fuse_runs, open_direct, read_direct, FUSE_EXPERTS,
    FUSE_NS, FUSE_READS,
};
#[cfg(test)]
use fetch::{compact, FUSE_GAP_MAX, O_DIRECT};
use prefetch::{
    pref_used, trace_record, trace_sink, TraceSim, DPREF_CACHED, DPREF_COOL, DPREF_ISSUED,
    DPREF_USED, DPREF_WIN_I, DPREF_WIN_U, PREDICT_N, PRED_HIT, PRED_TOT, PREF_CACHED, PREF_ISSUED,
    PREF_USED, TOP_K, TS_CACHED, TS_ISSUED, TS_MATCH, TS_RUPT, TS_USED,
};

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// global fetch report counters (one model per process)
static RAM_HITS: AtomicU64 = AtomicU64::new(0);
static RAM_MISSES: AtomicU64 = AtomicU64::new(0);
static DISK_BYTES: AtomicU64 = AtomicU64::new(0); // bytes served by the disk tier

// ── VQ1 shadow fallback (--stream-fallback, DEGRADED latency mode) ──
//
// On an expert cache miss the historical path blocks the decode on the disk
// tier (the worst case on a latency-bound network disk). With the fallback
// active, the engine instead serves IMMEDIATELY the expert's VQ1 shadow
// (shadow.rs: 0.5 bit/weight, fully resident in RAM, a few microseconds of
// LUT matvec) and refills the full-precision mxfp4 blob in the background:
// the NEXT request for the expert hits the RAM LRU as usual. The miss
// latency becomes bounded by the shadow serve, the decode never stops on
// the disk.
//
// HONESTY: a token whose experts were shadow-served is computed with
// degraded weights - this path is NOT bit-identical. It is a latency mode,
// not a quality mode, and it is OFF by default: opt in with the explicit
// --stream-fallback flag or MICROKIMI_STREAM_FALLBACK=1, and the report
// line counts exactly how many expert computations were degraded.
// MICROKIMI_FORCE_FALLBACK=1 (test knob) serves the shadow for EVERY
// expert, hit or miss, to measure the quality cost of a VQ1 expert in
// place of an mxfp4 one.
static FALLBACK_FLAG: AtomicBool = AtomicBool::new(false); // --stream-fallback
static FB_SERVED: AtomicU64 = AtomicU64::new(0); // expert computations served from a shadow
static FB_GETS: AtomicU64 = AtomicU64::new(0); // total demand gets while the fallback is active
static FB_FETCH: AtomicU64 = AtomicU64::new(0); // background full-precision refills
static FB_READS: AtomicU64 = AtomicU64::new(0); // span reads issued by the fused refills
static FB_EPT: AtomicUsize = AtomicUsize::new(0); // experts per token (MoE layers x top-k)

/// `--stream-fallback`: enable the VQ1 shadow fallback (DEGRADED latency
/// mode, see the block comment above). The shadows are loaded by
/// Model::load_streaming; without a sidecar the flag only arms the toggle.
pub fn set_fallback(on: bool) {
    FALLBACK_FLAG.store(on, Ordering::Relaxed);
}

/// Shadow fallback toggle: the --stream-fallback flag or
/// MICROKIMI_STREAM_FALLBACK=1. Default OFF (bit-identical decode).
pub fn fallback_on() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    FALLBACK_FLAG.load(Ordering::Relaxed) || *ENV.get_or_init(|| env_off("MICROKIMI_STREAM_FALLBACK"))
}

/// MICROKIMI_FORCE_FALLBACK=1 (test knob): serve the shadow for EVERY
/// expert demand, hit or miss, and skip all refills. Deterministic (the
/// shadow bytes are constant), used to measure the quality cost of a fully
/// VQ1 expert set.
fn force_fallback() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_off("MICROKIMI_FORCE_FALLBACK"))
}

/// Experts computed per token (MoE layers x router top-k), for the
/// per-token degraded share of the report line. Set at model load.
pub fn set_fallback_shape(experts_per_token: usize) {
    FB_EPT.store(experts_per_token, Ordering::Relaxed);
}

fn mb(n: u64) -> String {
    format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
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
    let (di, dc) = (DPREF_ISSUED.load(Ordering::Relaxed), DPREF_CACHED.load(Ordering::Relaxed));
    if di + dc > 0 {
        let used = DPREF_USED.load(Ordering::Relaxed);
        let recall = if di > 0 { 100.0 * used as f64 / di as f64 } else { 0.0 };
        s.push_str(&format!(
            "\nstream-draft: {} experts prefetched for the speculative verification ({} already cached), {} consumed on demand ({:.0}% recall)",
            di, dc, used, recall
        ));
    }
    if tracesim_on() {
        let issued = TS_ISSUED.load(Ordering::Relaxed);
        let used = TS_USED.load(Ordering::Relaxed);
        let recall = if issued > 0 { 100.0 * used as f64 / issued as f64 } else { 0.0 };
        s.push_str(&format!(
            "\nstream-tracesim: {} session match(es) ({} rupture re-matches), {} experts prefetched ({} already cached), {} consumed on demand ({:.0}% recall)",
            TS_MATCH.load(Ordering::Relaxed),
            TS_RUPT.load(Ordering::Relaxed),
            issued,
            TS_CACHED.load(Ordering::Relaxed),
            used,
            recall
        ));
    }
    let fb = FB_SERVED.load(Ordering::Relaxed);
    if fb > 0 {
        let tot = FB_GETS.load(Ordering::Relaxed).max(1);
        let mut line = format!(
            "\nstream-fallback: {} expert computations served from the VQ1 shadows ({:.1}% of {} demands) - DEGRADED latency mode, NOT bit-identical",
            fb,
            100.0 * fb as f64 / tot as f64,
            tot
        );
        let ept = FB_EPT.load(Ordering::Relaxed);
        if ept > 0 {
            let tokens = (tot as f64 / ept as f64).max(1.0);
            line.push_str(&format!(" (~{:.1} degraded experts/token of {})", fb as f64 / tokens, ept));
        }
        line.push_str(&format!(
            ", {} full-precision refills in the background ({} span reads)",
            FB_FETCH.load(Ordering::Relaxed),
            FB_READS.load(Ordering::Relaxed)
        ));
        s.push_str(&line);
    }
    s
}

/// Expert bytes served by ExpertCache::get.
pub enum Served {
    /// The blob from the expert cache (mxfp4, or VQ1 for a --cold-vq sliced
    /// expert): interpret with the model's own experts_vq flag and
    /// vq_codebook. Byte-identical to the full-load path.
    Full(Arc<Vec<u8>>),
    /// A VQ1 shadow served on a cache miss under --stream-fallback: always
    /// VQ1, shadow codebook, w1 ++ w2 ++ w3 index bytes at this offset of
    /// Shadows::data. DEGRADED (not bit-identical); the full-precision blob
    /// is refilled in the background for the next request.
    Shadow(Arc<crate::shadow::Shadows>, usize),
}

// ── RAM cache: eviction policies ──

/// Eviction policy of the expert RAM cache. MICROKIMI_CACHE=arc|lru|lfu
/// selects it; unset, the legacy MICROKIMI_NO_LFU=1 restores pure LRU and
/// the default is LFU with a recency tie-break: the eviction victim is the
/// entry with the lowest use count (1 at insert, +1 per demand hit), ties
/// broken by the oldest last access. Expert reuse is strongly bimodal (a
/// hot working set of frequently re-routed experts vs one-shot picks), so
/// frequency is a better residency signal than recency alone; the
/// cachereplay LFU column quantifies it offline. NOTE: an unused prefetch
/// sits at count 1 like a one-shot demand pick and ages out by recency - a
/// 0-count scheme was measured to self-evict fresh prefetch batches BEFORE
/// the router could consume them (0% prefetch recall), because every
/// count-0 entry is a better victim than every proven entry.
///
/// ARC (MICROKIMI_CACHE=arc) is the scan-resistant alternative: resident
/// entries split into T1 (referenced once) and T2 (referenced twice), and
/// the B1/B2 ghost lists keep the keys+sizes of recent evictions. A ghost
/// hit in B1 grows the adaptive T1 target p (recency matters), a ghost hit
/// in B2 shrinks it (frequency matters); a pure scan touches T1 only and
/// cannot flush the T2 working set. Byte-budgeted: the list capacities and
/// p are in bytes, entry sizes vary (MXFP4 vs VQ1 blobs). Available but
/// non-default, same status as the LRU toggle: the cachereplay ARC column
/// quantifies it offline against LRU/LFU/Belady.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Policy {
    Lfu,
    Lru,
    Arc,
}

/// The configured eviction policy (see the Policy comment).
fn policy() -> Policy {
    static P: std::sync::OnceLock<Policy> = std::sync::OnceLock::new();
    *P.get_or_init(|| match std::env::var("MICROKIMI_CACHE").map(|v| v.to_ascii_lowercase()) {
        Ok(v) if v == "arc" => Policy::Arc,
        Ok(v) if v == "lru" => Policy::Lru,
        Ok(v) if v == "lfu" => Policy::Lfu,
        Ok(v) => {
            eprintln!("warning: unknown MICROKIMI_CACHE={} (expected arc|lru|lfu), using the default", v);
            Policy::Lfu
        }
        // legacy toggle, honored when MICROKIMI_CACHE is unset
        Err(_) if std::env::var("MICROKIMI_NO_LFU").map(|v| v == "1").unwrap_or(false) => Policy::Lru,
        Err(_) => Policy::Lfu,
    })
}

/// Policy name for the startup line / report.
fn policy_name(p: Policy) -> &'static str {
    match p {
        Policy::Lfu => "lfu",
        Policy::Lru => "lru",
        Policy::Arc => "arc",
    }
}

struct Lru {
    map: HashMap<(u32, u32), (Arc<Vec<u8>>, u64, u8, u64, bool)>, // (layer, expert) -> (w1++w2++w3, tick, prefetch tag, hits, warm)
    queue: VecDeque<((u32, u32), u64)>,                             // access order, stale gens skipped (pure-LRU mode)
    cur: usize,
    tick: u64,
    budget: usize,
    pol: Policy,
    // ARC state (pol == Arc only): T2 membership of resident entries
    // (map \ t2 = T1), the per-list recency orders (stale gens skipped),
    // the B1/B2 ghost lists (key -> size at eviction) and the adaptive T1
    // target p. All capacities in bytes; t1b + t2b == cur.
    t2: std::collections::HashSet<(u32, u32)>,
    t1q: VecDeque<((u32, u32), u64)>,
    t2q: VecDeque<((u32, u32), u64)>,
    t1b: usize,
    t2b: usize,
    b1: HashMap<(u32, u32), usize>,
    b1q: VecDeque<(u32, u32)>,
    b1b: usize,
    b2: HashMap<(u32, u32), usize>,
    b2q: VecDeque<(u32, u32)>,
    b2b: usize,
    p: usize,
}

impl Lru {
    fn new(budget: usize) -> Lru {
        Lru::with_policy(budget, policy())
    }

    fn with_policy(budget: usize, pol: Policy) -> Lru {
        Lru {
            map: HashMap::new(),
            queue: VecDeque::new(),
            cur: 0,
            tick: 0,
            budget,
            pol,
            t2: std::collections::HashSet::new(),
            t1q: VecDeque::new(),
            t2q: VecDeque::new(),
            t1b: 0,
            t2b: 0,
            b1: HashMap::new(),
            b1q: VecDeque::new(),
            b1b: 0,
            b2: HashMap::new(),
            b2q: VecDeque::new(),
            b2b: 0,
            p: 0,
        }
    }
    /// On a hit, also reports (and clears) the prefetch tag (0 = demand
    /// fetch, 1 = Markov/lookahead prefetch, 2 = draft prefetch), so a
    /// prefetched entry consumed on demand is counted exactly once. A
    /// warm-marked entry (batched demand fetch, warm_batch) is served
    /// WITHOUT bumping the LFU count: the insert already carries the demand
    /// credit of the request consuming it now. Counting it again would push
    /// every fused entry to count >= 2 (even one-shot picks) and drown the
    /// one-shot/reused distinction the LFU victim choice relies on - with a
    /// full cache the next warm wave would then find no better victim than
    /// its own fresh inserts (measured: 6x the disk traffic at a tight
    /// budget).
    fn get(&mut self, k: (u32, u32)) -> Option<(Arc<Vec<u8>>, u8)> {
        if self.pol == Policy::Arc {
            return self.arc_get(k);
        }
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

    /// Pure presence check for the draft prefetch's already-cached
    /// predictions. Deliberately NO recency/LFU update: refreshing thousands
    /// of predicted entries per pass protects them from eviction at the
    /// expense of the decode loop's own working set (measured on the smoke
    /// model at --spec 8, 8 MB: +65% demand misses from the churn), and a
    /// count bump per draft inflates predicted entries into immortality
    /// under LFU. A prediction that the pass really needs will be demanded
    /// - and refreshed - by the pass itself.
    fn peek(&self, k: (u32, u32)) -> bool {
        self.map.contains_key(&k)
    }

    /// `warm` marks a batched demand fetch (warm_batch): the first demand
    /// get is served without a count bump (see get). `pref` is the prefetch
    /// tag (0 = demand, 1 = Markov/lookahead, 2 = draft).
    fn insert(&mut self, k: (u32, u32), v: Arc<Vec<u8>>, pref: u8, warm: bool) {
        if self.pol == Policy::Arc {
            self.arc_insert(k, v, pref, warm);
            return;
        }
        let sz = v.len();
        if sz > self.budget {
            return; // a single expert exceeds the budget: serve without caching
        }
        self.tick += 1;
        // LFU count: 1 at insert (demand or prefetch alike - see the
        // Policy comment for why prefetch does not start at 0), +1 per
        // demand hit; a re-insert keeps the hits earned
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
            if self.pol == Policy::Lfu {
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

    // ── ARC internals (pol == Arc; see the Policy comment) ──

    /// ARC hit: a T1 entry is promoted to T2 (second reference), a T2 entry
    /// is refreshed MRU. Same tag/warm accounting as the LRU/LFU get.
    fn arc_get(&mut self, k: (u32, u32)) -> Option<(Arc<Vec<u8>>, u8)> {
        if !self.map.contains_key(&k) {
            return None;
        }
        self.tick += 1;
        let e = self.map.get_mut(&k).unwrap();
        e.1 = self.tick;
        if !std::mem::take(&mut e.4) {
            e.3 += 1;
        }
        let v = e.0.clone();
        let pref = std::mem::take(&mut e.2);
        if !self.t2.contains(&k) {
            self.t2.insert(k);
            self.t1b -= v.len();
            self.t2b += v.len();
        }
        self.t2q.push_back((k, self.tick));
        Some((v, pref))
    }

    /// Evicts the LRU entry of T1, ghosting its key+size into B1 unless
    /// `ghost` is false (the |T1| == c miss case of the paper drops it).
    fn arc_evict_t1(&mut self, ghost: bool) {
        while let Some((k, g)) = self.t1q.pop_front() {
            let live = !self.t2.contains(&k) && self.map.get(&k).map(|e| e.1 == g).unwrap_or(false);
            if !live {
                continue;
            }
            let sz = self.map.remove(&k).unwrap().0.len();
            self.cur -= sz;
            self.t1b -= sz;
            if ghost {
                self.b1.insert(k, sz);
                self.b1q.push_back(k);
                self.b1b += sz;
            }
            return;
        }
    }

    /// Evicts the LRU entry of T2, ghosting its key+size into B2.
    fn arc_evict_t2(&mut self) {
        while let Some((k, g)) = self.t2q.pop_front() {
            let live = self.t2.contains(&k) && self.map.get(&k).map(|e| e.1 == g).unwrap_or(false);
            if !live {
                continue;
            }
            let sz = self.map.remove(&k).unwrap().0.len();
            self.t2.remove(&k);
            self.cur -= sz;
            self.t2b -= sz;
            self.b2.insert(k, sz);
            self.b2q.push_back(k);
            self.b2b += sz;
            return;
        }
    }

    /// Drops the LRU ghost of a ghost list (stale queue entries belong to
    /// keys already consumed by a ghost hit).
    fn ghost_pop(b: &mut HashMap<(u32, u32), usize>, q: &mut VecDeque<(u32, u32)>, bytes: &mut usize) {
        while let Some(k) = q.pop_front() {
            if let Some(sz) = b.remove(&k) {
                *bytes -= sz;
                return;
            }
        }
    }

    /// ARC REPLACE: evict the LRU of T1 into B1 when T1 exceeds the target
    /// p (or meets it on a B2-guided reference), else the LRU of T2 into B2.
    fn arc_replace(&mut self, in_b2: bool) {
        let over = self.t1b > self.p || (in_b2 && self.t1b == self.p);
        if over && self.t1b > 0 {
            self.arc_evict_t1(true);
        } else if self.t2b > 0 {
            self.arc_evict_t2();
        } else if self.t1b > 0 {
            self.arc_evict_t1(true);
        }
    }

    /// Enters a freshly fetched entry into T2 (ghost-guided reference).
    fn arc_fill_t2(&mut self, k: (u32, u32), v: Arc<Vec<u8>>, pref: u8, warm: bool) {
        self.t2b += v.len();
        self.cur += v.len();
        self.t2.insert(k);
        self.map.insert(k, (v, self.tick, pref, 1, warm));
        self.t2q.push_back((k, self.tick));
    }

    /// ARC reference miss path (see the Policy comment for the invariant
    /// sketch): ghost hits adapt the target p and land in T2; plain misses
    /// maintain the ghost bounds, REPLACE, and land in T1. Byte-budgeted
    /// adaptation of the entry-count paper: list sizes and p are bytes.
    fn arc_insert(&mut self, k: (u32, u32), v: Arc<Vec<u8>>, pref: u8, warm: bool) {
        let sz = v.len();
        if sz > self.budget {
            return; // a single expert exceeds the budget: serve without caching
        }
        self.tick += 1;
        // re-insert of a resident entry (concurrent duplicate fetch of the
        // same file bytes): keeps the hits earned, counts as a reference
        if self.map.contains_key(&k) {
            let old = self.map.insert(k, (v, self.tick, pref, 0, warm)).unwrap();
            self.map.get_mut(&k).unwrap().3 = old.3 + 1;
            self.cur -= old.0.len();
            self.cur += sz;
            if self.t2.contains(&k) {
                self.t2b -= old.0.len();
                self.t2b += sz;
            } else {
                self.t2.insert(k);
                self.t1b -= old.0.len();
                self.t2b += sz;
            }
            self.t2q.push_back((k, self.tick));
        } else if let Some(gsz) = self.b1.remove(&k) {
            // B1 ghost hit: recency matters more - grow the T1 target
            self.b1b -= gsz;
            let ratio = (self.b2b / self.b1b.max(1)).max(1);
            self.p = (self.p + sz * ratio).min(self.budget);
            self.arc_replace(false);
            self.arc_fill_t2(k, v, pref, warm);
        } else if let Some(gsz) = self.b2.remove(&k) {
            // B2 ghost hit: frequency matters more - shrink the T1 target
            self.b2b -= gsz;
            let ratio = (self.b1b / self.b2b.max(1)).max(1);
            self.p = self.p.saturating_sub(sz * ratio);
            self.arc_replace(true);
            self.arc_fill_t2(k, v, pref, warm);
        } else {
            // plain miss: maintain the ghost bounds, REPLACE, land in T1
            if self.t1b + self.b1b >= self.budget {
                if self.t1b < self.budget {
                    Self::ghost_pop(&mut self.b1, &mut self.b1q, &mut self.b1b);
                    self.arc_replace(false);
                } else {
                    // B1 empty, T1 alone fills the cache: drop the T1 LRU
                    // without ghosting it (the paper's |T1| == c case)
                    self.arc_evict_t1(false);
                }
            } else if self.t1b + self.b1b + self.t2b + self.b2b >= self.budget {
                if self.t1b + self.b1b + self.t2b + self.b2b >= 2 * self.budget {
                    Self::ghost_pop(&mut self.b2, &mut self.b2q, &mut self.b2b);
                }
                self.arc_replace(false);
            }
            self.t1b += sz;
            self.cur += sz;
            self.map.insert(k, (v, self.tick, pref, 1, warm));
            self.t1q.push_back((k, self.tick));
        }
        // budget safety under variable entry sizes (the paper evicts one
        // entry per reference; sizes here vary by a few %)
        while self.cur > self.budget && self.map.len() > 1 {
            let before = self.cur;
            self.arc_replace(false);
            if self.cur == before {
                break; // nothing evictable (should not happen)
            }
        }
        // amortized compaction of the recency queues (every reference
        // pushes one entry)
        if self.t1q.len() + self.t2q.len() > 4 * self.map.len().max(16) {
            let mut live: Vec<((u32, u32), u64, bool)> = self
                .map
                .iter()
                .map(|(&k, &(_, g, _, _, _))| (k, g, self.t2.contains(&k)))
                .collect();
            live.sort_by_key(|&(_, g, _)| g);
            self.t1q = live.iter().filter(|e| !e.2).map(|&(k, g, _)| (k, g)).collect();
            self.t2q = live.iter().filter(|e| e.2).map(|&(k, g, _)| (k, g)).collect();
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
    /// Keys a draft prefetch is currently reading. A demand get on one of
    /// these WAITS on the condvar for the in-flight read to land instead of
    /// fetching the same bytes a second time; warm_batch skips them for the
    /// same reason. Only the draft-aware prefetch marks keys (it launches
    /// right before the verification pass that will demand them). The shadow
    /// fallback reuses the same set for its background refills: a demand get
    /// on a marked key serves the shadow WITHOUT waiting (bounded latency is
    /// the whole point of the mode).
    inflight: (Mutex<std::collections::HashSet<(u32, u32)>>, std::sync::Condvar),
    /// VQ1 shadows of every expert (shadow.rs), resident in RAM under
    /// --stream-fallback. None in the default bit-identical mode.
    shadows: Option<Arc<crate::shadow::Shadows>>,
    /// Trace-similarity prefetcher state (MICROKIMI_TRACESIM=1). Some on the
    /// local source even with an empty store: the current session is still
    /// recorded and appended to <model>.routes at drop.
    tsim: Mutex<Option<TraceSim>>,
}

impl Drop for CacheInner {
    /// Appends the current session's routing signature to <model>.routes
    /// (TraceSim::save; no-op when tracesim is off or the session is too
    /// short to be a signature).
    fn drop(&mut self) {
        if let Some(t) = self.tsim.get_mut().unwrap().as_mut() {
            t.save();
        }
    }
}

impl CacheInner {
    /// Marks an in-flight read as landed (draft prefetch): wake the demand
    /// gets waiting on it. A remove of an unmarked key is a no-op (the
    /// Markov/lookahead prefetchers never mark).
    fn land(&self, k: (u32, u32)) {
        self.inflight.0.lock().unwrap().remove(&k);
        self.inflight.1.notify_all();
    }
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
    /// `pref` is the insert tag: 0 = batched DEMAND fetch (warm_batch: RAM
    /// miss + fuse counters), 2 = batched background draft prefetch
    /// (`issued` counter instead). Both insert with the warm mark: for a
    /// prefetched entry it makes the first demand get count-neutral (net
    /// LFU count 1, exactly what a demand fetch of the same expert would
    /// carry - without it a consumed prefetch ends at count 2, the
    /// predicted union turns unevictable and a tight cache thrashes:
    /// measured +80% demand misses at 8 MB).
    fn fetch_run(&self, layer: u32, members: &[(u32, [u64; 3], usize)], pref: u8, issued: &AtomicU64) {
        if members.len() == 1 {
            let (e, offs, blob) = members[0];
            let bytes = self.fetch(layer, e, offs, blob);
            self.lru.lock().unwrap().insert((layer, e), Arc::new(bytes), pref, true);
            if pref != 0 {
                issued.fetch_add(1, Ordering::Relaxed);
                self.land((layer, e));
            } else {
                RAM_MISSES.fetch_add(1, Ordering::Relaxed);
            }
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
            if pref != 0 {
                issued.fetch_add(1, Ordering::Relaxed);
            } else {
                FUSE_EXPERTS.fetch_add(1, Ordering::Relaxed);
                RAM_MISSES.fetch_add(1, Ordering::Relaxed); // a demand miss served by the batch
            }
            self.lru.lock().unwrap().insert((layer, e), Arc::new(bytes), pref, true);
            if pref != 0 {
                self.land((layer, e));
            }
        }
    }

    /// Background prefetch of one predicted expert: already cached = just
    /// refresh the recency (protects it from eviction), otherwise fetch and
    /// insert with the from_prefetch mark. Never touches the router path.
    /// `issued`/`cached` are the counter pair of the calling prefetcher
    /// (PREF_* for --stream-predict, DPREF_* for the draft-aware prefetch).
    /// `warm` makes the first demand get count-neutral (draft prefetch: the
    /// consumed entry nets LFU count 1, what a demand fetch would carry -
    /// see fetch_run); the Markov/lookahead prefetch keeps the historical
    /// consume-bump. `tag` is the prefetch tag stored with the entry (1 =
    /// Markov/lookahead, 2 = draft).
    fn prefetch_one(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize, issued: &AtomicU64, cached: &AtomicU64, warm: bool, tag: u8) {
        let k = (layer, expert);
        {
            let mut lru = self.lru.lock().unwrap();
            if lru.get(k).is_some() {
                cached.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
        let bytes = self.fetch(layer, expert, offs, blob);
        self.lru.lock().unwrap().insert(k, Arc::new(bytes), tag, warm);
        issued.fetch_add(1, Ordering::Relaxed);
        self.land(k);
    }

    /// Background full-precision refill of one expert (shadow fallback): the
    /// demand get already served the VQ1 shadow, so this only lands the
    /// mxfp4 bytes in the RAM LRU for the NEXT request, then unmarks the
    /// in-flight key. Insert tag 0 (demand-like LFU credit), no warm mark.
    fn fill_one(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize) {
        let bytes = self.fetch(layer, expert, offs, blob);
        self.lru.lock().unwrap().insert((layer, expert), Arc::new(bytes), 0, false);
        FB_FETCH.fetch_add(1, Ordering::Relaxed);
        self.land((layer, expert));
    }

    /// Background fused refill of one run of file-adjacent experts (shadow
    /// fallback warm_batch): the same span-read fusion as fetch_run (the
    /// served bytes are byte-identical to per-expert preads), with the
    /// fallback's own counters and every member landed. Singleton runs and
    /// the remote source go through plain per-expert refills.
    fn fill_run(&self, layer: u32, members: &[(u32, [u64; 3], usize)]) {
        if members.len() == 1 {
            let (e, offs, blob) = members[0];
            self.fill_one(layer, e, offs, blob);
            return;
        }
        let Src::Local(l) = &self.src else {
            for &(e, offs, blob) in members {
                self.fill_one(layer, e, offs, blob);
            }
            return;
        };
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
        FB_READS.fetch_add(1, Ordering::Relaxed);
        for &(e, offs, blob) in members {
            let mut bytes = vec![0u8; 3 * blob];
            for i in 0..3 {
                let lo = (offs[i] - start) as usize;
                bytes[i * blob..(i + 1) * blob].copy_from_slice(&span[lo..lo + blob]);
            }
            DISK_BYTES.fetch_add(bytes.len() as u64, Ordering::Relaxed);
            self.lru.lock().unwrap().insert((layer, e), Arc::new(bytes), 0, false);
            FB_FETCH.fetch_add(1, Ordering::Relaxed);
            self.land((layer, e));
        }
    }

    /// The VQ1 shadow of one expert, when the fallback can serve it.
    fn shadow(&self, layer: u32, expert: u32) -> Option<(Arc<crate::shadow::Shadows>, usize)> {
        self.shadows.as_ref().and_then(|s| s.offset(layer, expert).map(|off| (Arc::clone(s), off)))
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
    /// `shadows`: the resident VQ1 expert shadows (shadow.rs) when
    /// --stream-fallback is active, None in the default bit-identical mode.
    pub fn local(path: &str, ram_mb: usize, shadows: Option<crate::shadow::Shadows>) -> ExpertCache {
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
        if draft_prefetch_on() {
            println!("stream: draft-aware expert prefetch on (MICROKIMI_DRAFTPREFETCH=0 to disable; --spec/--spec-rosa only)");
        } else {
            println!("stream: draft-aware expert prefetch off (MICROKIMI_DRAFTPREFETCH=0)");
        }
        println!("stream: expert cache policy {} (MICROKIMI_CACHE=arc|lru|lfu; default lfu)", policy_name(policy()));
        let shadows = shadows.map(|s| {
            let bytes = s.data.len() + s.cb.len() * 4;
            println!(
                "memory: expert shadows {} resident in RAM ({} MoE layers x {} experts, VQ1 0.5-bit, --stream-fallback)",
                mb(bytes as u64),
                s.layers.len(),
                s.n_experts
            );
            println!("stream: fallback mode DEGRADED (VQ1 shadows served on expert cache misses, background full-precision refill) - a latency mode, NOT bit-identical");
            if force_fallback() {
                println!("stream: FORCE_FALLBACK on (MICROKIMI_FORCE_FALLBACK=1, test knob: every expert served from its shadow)");
            }
            Arc::new(s)
        });
        // initialize the trace sink now so its startup line prints with the
        // other stream lines (no-op when MICROKIMI_TRACE is unset)
        trace_sink();
        // trace-similarity prefetcher (MICROKIMI_TRACESIM=1): load the
        // stored session signatures of this model (no-op when off)
        let tsim = if tracesim_on() {
            let rp = format!("{}.routes", path);
            let store = crate::routes::RouteStore::load(&rp).unwrap_or_else(|_| crate::routes::RouteStore::empty());
            println!(
                "stream-tracesim: {} stored session(s) in {} (MICROKIMI_TRACESIM=1: cold-start/topic-rupture expert prefetch)",
                store.sessions.len(),
                rp
            );
            Some(TraceSim::new(rp, store))
        } else {
            None
        };
        ExpertCache {
            inner: Arc::new(CacheInner {
                lru: Mutex::new(Lru::new(ram_mb << 20)),
                src: Src::Local(LocalSrc { file, direct }),
                pred: Mutex::new(Predictor::new()),
                inflight: (Mutex::new(std::collections::HashSet::new()), std::sync::Condvar::new()),
                shadows,
                tsim: Mutex::new(tsim),
            }),
        }
    }

    /// Remote streaming source (per-tensor persistent cache, disk budget in MB).
    #[allow(dead_code)] // wired end-to-end once real-dim MLA layers run (see docs)
    pub fn remote(url: &str, ram_mb: usize, kept_layers: &[usize], disk_mb: u64) -> ExpertCache {
        ExpertCache {
            inner: Arc::new(CacheInner {
                lru: Mutex::new(Lru::new(ram_mb << 20)),
                src: Src::Remote(RemoteSource::open_disk(url, default_cache_root(url), kept_layers, disk_mb)),
                pred: Mutex::new(Predictor::new()),
                inflight: (Mutex::new(std::collections::HashSet::new()), std::sync::Condvar::new()),
                shadows: None, // the shadow fallback is a local-source mode
                tsim: Mutex::new(None), // the signature file derives from the local .bin path
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
        // Shadow fallback: the decode must NEVER block on the disk tier, so
        // there is no synchronous batched read phase. Every missing expert is
        // marked in flight and refilled on ONE detached thread (offset-sorted,
        // one fused span read per file-adjacent run - the same bytes the
        // synchronous path would read); the compute jobs' gets serve the VQ1
        // shadow until the refill lands. No-op under MICROKIMI_FORCE_FALLBACK
        // (test knob: shadows always served, refills useless).
        if self.inner.shadows.is_some() && fallback_on() {
            if force_fallback() {
                return;
            }
            let mut miss: Vec<(u32, [u64; 3], usize)> = Vec::new();
            {
                let lru = self.inner.lru.lock().unwrap();
                let mut inf = self.inner.inflight.0.lock().unwrap();
                for &(e, offs, blob) in items {
                    if !lru.map.contains_key(&(layer, e)) && inf.insert((layer, e)) {
                        miss.push((e, offs, blob));
                    }
                }
            }
            if miss.is_empty() {
                return;
            }
            miss.sort_by_key(|&(_, o, _)| o[0]);
            let inner = Arc::clone(&self.inner);
            std::thread::spawn(move || {
                if run_fuse() && matches!(inner.src, Src::Local(_)) {
                    for (s, e) in fuse_runs(&miss) {
                        inner.fill_run(layer, &miss[s..e]);
                    }
                } else {
                    for (e, o, b) in miss {
                        inner.fill_one(layer, e, o, b);
                    }
                }
            });
            return;
        }
        if items.len() < 2 || !run_fuse() || !matches!(self.inner.src, Src::Local(_)) {
            return;
        }
        // misses only, offset-sorted (the LRU may change before the reads
        // land: a concurrent prefetch of the same expert is harmless - the
        // run insert refreshes the same file bytes). Keys a draft prefetch
        // is already reading are skipped: the compute jobs' get waits for
        // the in-flight read instead of fetching the same bytes twice.
        let mut miss: Vec<(u32, [u64; 3], usize)> = Vec::with_capacity(items.len());
        {
            let lru = self.inner.lru.lock().unwrap();
            let inf = self.inner.inflight.0.lock().unwrap();
            for &(e, offs, blob) in items {
                if !lru.map.contains_key(&(layer, e)) && !inf.contains(&(layer, e)) {
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
            jobs.push(Box::new(move || inner.fetch_run(layer, &members, 0, &PREF_ISSUED)));
        }
        crate::pool::pool().run(jobs);
        FUSE_NS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// The 3 MXFP4 blobs of expert `expert` of `layer`, concatenated
    /// w1 ++ w2 ++ w3, `blob` bytes each. `offs` = absolute file offsets of
    /// the three blobs (local source; ignored by the remote one).
    ///
    /// Return value (Served): normally Served::Full with the cache bytes.
    /// Under --stream-fallback (shadows resident), a cache MISS returns
    /// Served::Shadow immediately - the VQ1 shadow, a degraded
    /// low-precision stand-in - and the full-precision blob is refilled in
    /// the background for the next request: the decode never blocks on the
    /// disk tier. NOT bit-identical; off by default. With
    /// MICROKIMI_FORCE_FALLBACK=1 (test knob) EVERY demand returns the
    /// shadow, hit or miss, and nothing is fetched.
    pub fn get(&self, layer: u32, expert: u32, offs: [u64; 3], blob: usize) -> Served {
        // record the demand request (MICROKIMI_TRACE; no-op when unset)
        trace_record(layer, expert);
        // trace-similarity prefetcher (MICROKIMI_TRACESIM=1): maintain the
        // session signature, fire the cold-window chained prefetch on layer
        // transitions. Demand-only observation, same contract as the Markov
        // path: only fetch timing changes, the output stays bit-identical.
        if tracesim_on() {
            let jobs = {
                let mut g = self.inner.tsim.lock().unwrap();
                match g.as_mut() {
                    Some(t) => {
                        let p = predict_n();
                        let n = if p > 0 { p } else { TOP_K.load(Ordering::Relaxed) };
                        t.observe(layer, expert, offs, blob, n)
                    }
                    None => Vec::new(),
                }
            };
            if !jobs.is_empty() {
                let inner = Arc::clone(&self.inner);
                std::thread::spawn(move || {
                    for (l, e, o) in jobs {
                        inner.prefetch_one(l, e, o, blob, &TS_ISSUED, &TS_CACHED, false, 3);
                    }
                });
            }
        }
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
                        inner.prefetch_one(l, e, o, blob, &PREF_ISSUED, &PREF_CACHED, false, 1);
                    }
                });
            }
        }
        let k = (layer, expert);
        let fb = self.inner.shadows.is_some() && fallback_on();
        if fb {
            FB_GETS.fetch_add(1, Ordering::Relaxed);
            // test knob (quality measurement): every expert degraded
            if force_fallback() {
                if let Some((s, off)) = self.inner.shadow(layer, expert) {
                    FB_SERVED.fetch_add(1, Ordering::Relaxed);
                    return Served::Shadow(s, off);
                }
            }
        }
        {
            let mut lru = self.inner.lru.lock().unwrap();
            if let Some((v, pref)) = lru.get(k) {
                pref_used(pref);
                RAM_HITS.fetch_add(1, Ordering::Relaxed);
                return Served::Full(v);
            }
        }
        // shadow fallback: serve the resident VQ1 shadow NOW (bounded miss
        // latency) and make sure a background refill of the full-precision
        // bytes is running (the key may already be in flight from
        // warm_batch or a draft prefetch - their fills land it either way).
        if fb {
            if let Some((s, off)) = self.inner.shadow(layer, expert) {
                FB_SERVED.fetch_add(1, Ordering::Relaxed);
                let spawn = {
                    let mut inf = self.inner.inflight.0.lock().unwrap();
                    inf.insert(k)
                };
                if spawn {
                    let inner = Arc::clone(&self.inner);
                    std::thread::spawn(move || inner.fill_one(layer, expert, offs, blob));
                }
                return Served::Shadow(s, off);
            }
        }
        // a draft prefetch is already reading these bytes: wait for it to
        // land instead of fetching the same expert a second time. The
        // prefetcher unmarks strictly AFTER inserting into the LRU, so a
        // key absent from the set is a key whose read completed (no missed
        // wakeup; the LRU is never locked while this lock is held, and the
        // marking side locks LRU-then-inflight, never the reverse).
        {
            let (lock, cvar) = &self.inner.inflight;
            let mut inf = lock.lock().unwrap();
            while inf.contains(&k) {
                inf = cvar.wait(inf).unwrap();
            }
        }
        // the in-flight read (if any) landed before the unmark: one cache
        // pass serves the bytes (they may also have been evicted already
        // under a full cache: fall through to the demand fetch then)
        {
            let mut lru = self.inner.lru.lock().unwrap();
            if let Some((v, pref)) = lru.get(k) {
                pref_used(pref);
                RAM_HITS.fetch_add(1, Ordering::Relaxed);
                return Served::Full(v);
            }
        }
        RAM_MISSES.fetch_add(1, Ordering::Relaxed);
        let bytes = self.inner.fetch(layer, expert, offs, blob);
        let v = Arc::new(bytes);
        self.inner.lru.lock().unwrap().insert(k, v.clone(), 0, false);
        Served::Full(v)
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
                inner.prefetch_one(l, e, o, blob, &PREF_ISSUED, &PREF_CACHED, false, 1);
            }
        });
    }

    /// Draft-aware prefetch (--spec / --spec-rosa + --stream): background-
    /// fetch the union of experts the drafted tokens are predicted to route
    /// to (model.rs replays the routing recorded at the draft's source
    /// occurrence). The missing experts are marked IN FLIGHT synchronously
    /// (no demand read is running at draft time: the previous pass ended on
    /// a pool barrier), so the verification pass's warm_batch skips them and
    /// its ExpertCache::get waits for the in-flight read instead of fetching
    /// the same bytes twice. The detached thread serves each layer's missing
    /// experts offset-sorted, one span read per file-adjacent run (same
    /// fuse_runs machinery as warm_batch, from_prefetch inserts), so the
    /// reads of the later layers overlap the compute of the earlier ones.
    /// Never touches the router path: the output is unaffected.
    pub fn prefetch_draft(&self, jobs: Vec<(u32, u32, [u64; 3], usize)>) {
        if jobs.is_empty() || !draft_prefetch_on() {
            return;
        }
        // adaptive gate: a prefetch whose entries are evicted before the
        // pass consumes them only costs disk reads and cache slots. Two
        // adaptive recall gate: a prefetch whose entries are evicted before
        // the pass consumes them only costs disk reads and cache slots.
        // Over a window of GATE_WIN issued prefetches, fewer than 3/4
        // consumed on demand suspends the prefetch for GATE_COOL drafts
        // (measured on the smoke model: 78-89% recall runs gain 6-14% of
        // the demand misses at tight budgets; below that the inserts churn
        // more than they save).
        const GATE_WIN: u64 = 128;
        const GATE_COOL: u64 = 8;
        let cool = DPREF_COOL.load(Ordering::Relaxed);
        if cool > 0 {
            DPREF_COOL.store(cool - 1, Ordering::Relaxed);
            return;
        }
        let (wi, wu) = (DPREF_WIN_I.load(Ordering::Relaxed), DPREF_WIN_U.load(Ordering::Relaxed));
        if wi >= GATE_WIN {
            DPREF_WIN_I.store(0, Ordering::Relaxed);
            DPREF_WIN_U.store(0, Ordering::Relaxed);
            if wu * 4 < wi * 3 {
                DPREF_COOL.store(GATE_COOL, Ordering::Relaxed);
                return;
            }
        }
        // synchronous split: already cached (pure presence check - no
        // recency/LFU update, see Lru::peek) vs missing (marked in flight).
        // A key already in flight from the previous draft is left to its
        // own read.
        let mut by_layer: HashMap<u32, Vec<(u32, [u64; 3], usize)>> = HashMap::new();
        for (l, e, o, b) in jobs {
            by_layer.entry(l).or_default().push((e, o, b));
        }
        let mut layers: Vec<u32> = by_layer.keys().copied().collect();
        layers.sort_unstable();
        let mut missing: Vec<(u32, u32, [u64; 3], usize)> = Vec::new();
        {
            let lru = self.inner.lru.lock().unwrap();
            let mut inf = self.inner.inflight.0.lock().unwrap();
            for l in layers {
                for (e, o, b) in by_layer.remove(&l).unwrap() {
                    let k = (l, e);
                    if lru.peek(k) {
                        DPREF_CACHED.fetch_add(1, Ordering::Relaxed);
                    } else if inf.insert(k) {
                        missing.push((l, e, o, b));
                    }
                }
            }
        }
        if missing.is_empty() {
            return;
        }
        DPREF_WIN_I.fetch_add(missing.len() as u64, Ordering::Relaxed);
        let inner = Arc::clone(&self.inner);
        std::thread::spawn(move || {
            let mut by_layer: HashMap<u32, Vec<(u32, [u64; 3], usize)>> = HashMap::new();
            for (l, e, o, b) in missing {
                by_layer.entry(l).or_default().push((e, o, b));
            }
            let mut layers: Vec<u32> = by_layer.keys().copied().collect();
            layers.sort_unstable(); // layer order: the pass needs the low layers first
            for l in layers {
                let mut items = by_layer.remove(&l).unwrap();
                // fused span reads for the local source, per-expert
                // otherwise (remote: per-tensor cache files, no shared spans)
                if items.len() >= 2 && matches!(inner.src, Src::Local(_)) {
                    items.sort_by_key(|&(_, o, _)| o[0]);
                    for (s, e) in fuse_runs(&items) {
                        inner.fetch_run(l, &items[s..e], 2, &DPREF_ISSUED);
                    }
                } else {
                    for (e, o, b) in items {
                        inner.prefetch_one(l, e, o, b, &DPREF_ISSUED, &DPREF_CACHED, true, 2);
                    }
                }
            }
        });
    }
}

// ── streamtest: remote per-tensor cache + LRU budget proof ──

#[cfg(test)]
mod tests {
    use super::*;

    /// request with a compact w1++w2++w3 footprint starting at `off`
    fn req(e: u32, off: u64, blob: u64) -> (u32, [u64; 3], usize) {
        (e, [off, off + blob, off + 2 * blob], blob as usize)
    }

    // ── VQ1 shadow fallback (--stream-fallback) ──

    /// A miss serves the resident VQ1 shadow immediately and the background
    /// refill lands the full-precision file bytes in the LRU for the next
    /// demand, which is then served Full (bit-identical bytes).
    #[test]
    fn fallback_serves_shadow_then_refills() {
        set_fallback(true);
        // two fake experts, 64-byte blobs, expert e at file offset e*192
        let blob = 64usize;
        let file_bytes: Vec<u8> = (0..2 * 3 * blob).map(|i| (i % 253) as u8).collect();
        let dir = std::env::temp_dir();
        let fpath = dir.join(format!("microkimi-fb-test-{}", std::process::id()));
        std::fs::write(&fpath, &file_bytes).unwrap();
        let sh = crate::shadow::Shadows {
            layers: vec![0],
            n_experts: 2,
            vq_blob: 8,
            cb: vec![0.25; crate::quant::VQ_K * crate::quant::VQ_DIM],
            data: (0..2 * 3 * 8).map(|i| (i % 241) as u8).collect(),
        };
        let shadow_bytes = sh.data.clone();
        let cache = ExpertCache {
            inner: Arc::new(CacheInner {
                lru: Mutex::new(Lru::new(512)),
                src: Src::Local(LocalSrc {
                    file: std::fs::File::open(&fpath).unwrap(),
                    direct: None,
                }),
                pred: Mutex::new(Predictor::new()),
                inflight: (Mutex::new(std::collections::HashSet::new()), std::sync::Condvar::new()),
                shadows: Some(Arc::new(sh)),
                tsim: Mutex::new(None),
            }),
        };
        let offs = |e: u32| [e as u64 * 192, e as u64 * 192 + 64, e as u64 * 192 + 128];
        // cold miss: shadow served at the expert's byte offset, refill spawned
        match cache.get(0, 1, offs(1), blob) {
            Served::Shadow(s, off) => {
                assert_eq!(off, 3 * 8, "expert 1 shadow offset");
                assert_eq!(&s.data[off..off + 8], &shadow_bytes[off..off + 8]);
            }
            Served::Full(_) => panic!("cold miss must serve the shadow under the fallback"),
        }
        // the background refill lands the file bytes in the LRU
        let t0 = std::time::Instant::now();
        loop {
            if cache.inner.lru.lock().unwrap().peek((0, 1)) {
                break;
            }
            assert!(t0.elapsed() < std::time::Duration::from_secs(10), "background refill never landed");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        // the next demand is a full-precision hit with the exact file bytes
        match cache.get(0, 1, offs(1), blob) {
            Served::Full(v) => assert_eq!(&v[..], &file_bytes[192..384], "refill must serve the exact file bytes"),
            Served::Shadow(..) => panic!("refilled expert must be served Full"),
        }
        // the in-flight set is clean after the refill
        assert!(cache.inner.inflight.0.lock().unwrap().is_empty());
        std::fs::remove_file(&fpath).ok();
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

    // ── ARC policy (MICROKIMI_CACHE=arc) ──

    const E: usize = 1000; // synthetic entry size

    fn arc_lru(entries: usize) -> Lru {
        Lru::with_policy(entries * E, Policy::Arc)
    }

    fn arc_insert(c: &mut Lru, e: u32) {
        c.insert((0, e), Arc::new(vec![e as u8; E]), 0, false);
    }

    #[test]
    fn arc_evicts_t1_lru_first() {
        // budget 3: insert 1,2,3 (T1), hit 1 (-> T2), insert 4: the T1 LRU
        // (2) is evicted into B1, the T2 entry (1) is protected
        let mut c = arc_lru(3);
        arc_insert(&mut c, 1);
        arc_insert(&mut c, 2);
        arc_insert(&mut c, 3);
        assert!(c.get((0, 1)).is_some());
        arc_insert(&mut c, 4);
        assert!(c.map.contains_key(&(0, 1)), "T2 entry must survive");
        assert!(!c.map.contains_key(&(0, 2)), "T1 LRU must be evicted");
        assert!(c.map.contains_key(&(0, 3)));
        assert!(c.map.contains_key(&(0, 4)));
        assert!(c.b1.contains_key(&(0, 2)), "the T1 victim ghosts into B1");
        assert_eq!(c.cur, 3 * E);
        assert_eq!(c.t1b + c.t2b, c.cur);
    }

    #[test]
    fn arc_ghost_hit_grows_p_and_lands_in_t2() {
        // continue the scenario above: referencing the B1 ghost (2) is a
        // cache miss that fetches into T2 and grows the T1 target p
        let mut c = arc_lru(3);
        arc_insert(&mut c, 1);
        arc_insert(&mut c, 2);
        arc_insert(&mut c, 3);
        c.get((0, 1));
        arc_insert(&mut c, 4); // evicts 2 into B1
        assert!(c.get((0, 2)).is_none(), "ghost: not resident");
        arc_insert(&mut c, 2); // the demand fetch re-inserts: ghost hit
        assert!(c.map.contains_key(&(0, 2)));
        assert!(c.t2.contains(&(0, 2)), "a ghost-guided reference lands in T2");
        assert_eq!(c.p, E, "p grew by one entry (B2 empty: ratio 1)");
        assert!(!c.b1.contains_key(&(0, 2)), "the ghost is consumed");
        assert!(c.cur <= 3 * E);
    }

    #[test]
    fn arc_is_scan_resistant() {
        // hot entries referenced twice sit in T2; a one-shot scan then
        // cannot flush them (plain LRU loses the whole working set)
        let mut c = arc_lru(4);
        for _ in 0..2 {
            for e in [10, 11] {
                arc_insert(&mut c, e);
                c.get((0, e));
            }
        }
        for e in 0..8 {
            arc_insert(&mut c, e); // the scan: 8 one-shot entries, budget 4
        }
        assert!(c.map.contains_key(&(0, 10)), "T2 hot entry flushed by a scan");
        assert!(c.map.contains_key(&(0, 11)), "T2 hot entry flushed by a scan");
        // the scan leaves only its own tail in T1
        assert!(c.cur <= 4 * E);
    }

    #[test]
    fn arc_budget_invariant_under_mixed_load() {
        let mut c = arc_lru(16);
        let mut x = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        for _ in 0..4000 {
            let e = (next() % 64) as u32;
            if c.get((0, e)).is_none() {
                arc_insert(&mut c, e);
            }
            assert!(c.cur <= 16 * E, "over budget: {}", c.cur);
            assert_eq!(c.t1b + c.t2b, c.cur, "T1+T2 must account for all residents");
            // every resident entry is in exactly one list
            for k in c.map.keys() {
                assert_eq!(c.t2.contains(k), c.t2.contains(k));
            }
        }
    }
}
