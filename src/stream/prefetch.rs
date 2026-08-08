// Prefetch prediction for the expert cache: given the experts the router
// just picked, guess the next ones and warm them before demand.
// - Predictor: per-layer transition table with time-decayed counts.
// - draft prefetch: speculative pulls under an adaptive hit-rate gate.
// - TraceSim: match the live route trace against saved session traces
//   (MICROKIMI_TRACESIM=1) and prefetch the matched session experts.
// Also holds the route history window and the route trace sink.
// Pure prediction: a wrong guess only warms RAM, never changes the output.

use super::*;

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
pub(super) static PREDICT_N: AtomicUsize = AtomicUsize::new(0); // predicted experts per layer, 0 = off
pub(super) static TOP_K: AtomicUsize = AtomicUsize::new(16); // router top-k (batch size of one MoE layer)
pub(super) static PREF_ISSUED: AtomicU64 = AtomicU64::new(0); // prefetches that fetched bytes
pub(super) static PREF_CACHED: AtomicU64 = AtomicU64::new(0); // predicted experts already in RAM
pub(super) static PREF_USED: AtomicU64 = AtomicU64::new(0); // prefetched entries later consumed on demand
pub(super) static PRED_HIT: AtomicU64 = AtomicU64::new(0); // predicted experts the router actually picked
pub(super) static PRED_TOT: AtomicU64 = AtomicU64::new(0); // total predicted experts

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

// ── draft-aware expert prefetch (--spec / --spec-rosa + --stream) ──
//
// The batched verification pass of a speculative round routes the drafted
// tokens for real (model.rs moe_prefill), so the experts it will pull are
// predictable BEFORE the pass starts. Both proposers draft tokens that
// already occurred in the committed context (n-gram: verbatim lift; Rosa:
// frequency chain resolved to a source occurrence by model/spec.rs), and every
// ingested position had its top-k router picks recorded below (real hidden
// states - routing the token EMBEDDINGS through the routers instead was
// measured at ~0% recall: embeddings are too far from the hidden states
// the routers actually see). The prediction for a drafted token is the
// recorded routing of its source occurrence: model/spec.rs hands the source
// positions to Model::draft_prefetch, which unions the recorded top-k sets
// and background-fetches them here while the verification pass runs, so
// its warm_batch finds the experts already in the RAM LRU. Same contract
// as the router-lookahead prefetch: only WHEN bytes land in the cache
// changes, never WHICH experts are computed - a mispredicted draft expert
// is a harmless LRU fill, so the greedy output stays bit-identical.
// MICROKIMI_DRAFTPREFETCH=0 disables.
pub(super) static DPREF_ISSUED: AtomicU64 = AtomicU64::new(0); // draft prefetches that fetched bytes
pub(super) static DPREF_CACHED: AtomicU64 = AtomicU64::new(0); // predicted draft experts already in RAM
pub(super) static DPREF_USED: AtomicU64 = AtomicU64::new(0); // draft-prefetched entries consumed on demand
pub(super) static DPREF_WIN_I: AtomicU64 = AtomicU64::new(0); // adaptive gate window: issued
pub(super) static DPREF_WIN_U: AtomicU64 = AtomicU64::new(0); // adaptive gate window: consumed
pub(super) static DPREF_COOL: AtomicU64 = AtomicU64::new(0); // drafts the gate suspends the prefetch for

/// Accounts a consumed prefetched entry to its producer's counter (prefetch
/// tag: 1 = Markov/lookahead, 2 = draft, 3 = tracesim).
pub(super) fn pref_used(tag: u8) {
    match tag {
        1 => {
            PREF_USED.fetch_add(1, Ordering::Relaxed);
        }
        2 => {
            DPREF_USED.fetch_add(1, Ordering::Relaxed);
            DPREF_WIN_U.fetch_add(1, Ordering::Relaxed);
        }
        3 => {
            TS_USED.fetch_add(1, Ordering::Relaxed);
        }
        _ => {}
    }
}

/// Draft-aware prefetch toggle (default on; only used in --spec /
/// --spec-rosa streaming runs).
pub fn draft_prefetch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_DRAFTPREFETCH").map(|v| v != "0").unwrap_or(true))
}

/// Positions kept in the routing history (a sliding window: the n-gram
/// proposer picks the MOST RECENT earlier occurrence, so old positions are
/// the least useful sources; 4096 positions x per-MoE-layer top-k sets is a
/// few MB).
const ROUTE_HIST_WINDOW: usize = 4096;

/// Per-position router picks: pos -> (layer, top-k experts) per MoE layer.
/// Written by model.rs on every MoE forward/prefill of a streaming run,
/// read by the draft-aware prefetch.
struct RouteHist {
    by_pos: HashMap<u32, Vec<(u32, Vec<u32>)>>,
    order: VecDeque<u32>, // insertion order of positions, for the window
}

static ROUTE_HIST: std::sync::LazyLock<Mutex<RouteHist>> = std::sync::LazyLock::new(|| Mutex::new(RouteHist { by_pos: HashMap::new(), order: VecDeque::new() }));

/// Records the top-k router picks of one position at one MoE layer
/// (no-op when the draft prefetch is disabled). Re-recording a (pos, layer)
/// pair REPLACES it: the optimistic verification batch records positions a
/// partial accept then rejects, and the committed re-ingestion of those
/// positions must overwrite the rejected routing.
pub fn route_record(pos: u32, layer: u32, experts: Vec<u32>) {
    if !draft_prefetch_on() {
        return;
    }
    let mut h = ROUTE_HIST.lock().unwrap();
    if !h.by_pos.contains_key(&pos) {
        h.order.push_back(pos);
        if h.order.len() > ROUTE_HIST_WINDOW {
            if let Some(old) = h.order.pop_front() {
                h.by_pos.remove(&old);
            }
        }
    }
    let layers = h.by_pos.entry(pos).or_default();
    match layers.iter_mut().find(|(l, _)| *l == layer) {
        Some(e) => e.1 = experts,
        None => layers.push((layer, experts)),
    }
}

/// The recorded router picks of one position (None when unknown: before the
/// first ingestion, window-evicted, or restored from a .mkmem snapshot).
pub fn route_lookup(pos: u32) -> Option<Vec<(u32, Vec<u32>)>> {
    if !draft_prefetch_on() {
        return None;
    }
    ROUTE_HIST.lock().unwrap().by_pos.get(&pos).cloned()
}

/// Drops the routing history (Model::reset_cache: positions restart at 0).
pub fn route_hist_clear() {
    let mut h = ROUTE_HIST.lock().unwrap();
    h.by_pos.clear();
    h.order.clear();
}

// ── expert request trace (MICROKIMI_TRACE=/path/trace.bin) ──
//
// The router picks do not depend on the cache contents, so every demand
// request ExpertCache::get serves can be recorded as an ordered (layer,
// expert) stream and replayed OFFLINE under any cache policy and capacity
// (see tools/replay.rs, `microkimi cachereplay`): one traced run yields the
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
pub(super) fn trace_sink() -> &'static Option<Mutex<std::fs::File>> {
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
pub(super) fn trace_record(layer: u32, expert: u32) {
    if let Some(m) = trace_sink() {
        use std::io::Write;
        let mut b = [0u8; 8];
        b[..4].copy_from_slice(&layer.to_le_bytes());
        b[4..].copy_from_slice(&expert.to_le_bytes());
        m.lock().unwrap().write_all(&b).ok();
    }
}

// ── trace-similarity prefetch (MICROKIMI_TRACESIM=1, OFF by default) ──
//
// Cross-session prediction for the two regimes the online predictors are
// blind in: session COLD START (the Markov has observed no transition yet)
// and a mid-session TOPIC CHANGE (its decayed statistics describe the old
// topic). A compact per-layer expert histogram of every past session is
// kept in <model>.routes (route_history.rs); the demand stream of the CURRENT
// session builds the same signature, and once it resembles a stored session
// (cosine >= TSIM_THRESHOLD), that session's per-layer top experts become
// the prefetch source for a cold window:
//
//   - match: at every layer wrap with enough context (the prompt prefill
//     routing alone is context enough), the best-cosine stored session is
//     selected; retried on every wrap while unmatched.
//   - chained prefetch: on every MoE layer transition during the cold
//     window (COLD_PASSES token passes after the match), the top-N experts
//     of the matched session for the NEXT MoE layer are background-fetched
//     (prefetch_one, tag 3), overlapping the current layer's compute. A
//     one-shot warm of all layers at match time was measured strictly worse
//     (+0.9-1.3 points of first-50-tokens hit-rate vs +8.4-8.8 chained,
//     cachereplay --tracesim): the bulk insert thrashes a tight cache
//     before the decode reaches the later layers.
//   - rupture: past the cold window, every RUPT_EVERY passes the Markov
//     rolling accuracy (PRED_HIT/PRED_TOT deltas; the tracesim recall
//     deltas in lookahead mode, where the Markov does not run) is checked;
//     below RUPT_RECALL_FLOOR the recent-window signature is re-matched and
//     a new best session restarts the cold window.
//
// The Markov/lookahead predictors stay the primary source in the
// established regime: the chained fire stops after the cold window. A
// prefetch only changes WHEN bytes land in the RAM LRU, never WHICH experts
// are computed: the greedy output stays bit-identical. Local source only
// (the signature file path derives from the .bin path).
pub(super) static TS_ISSUED: AtomicU64 = AtomicU64::new(0); // tracesim prefetches that fetched bytes
pub(super) static TS_CACHED: AtomicU64 = AtomicU64::new(0); // predicted experts already in RAM
pub(super) static TS_USED: AtomicU64 = AtomicU64::new(0); // tracesim-prefetched entries consumed on demand
pub(super) static TS_MATCH: AtomicU64 = AtomicU64::new(0); // accepted session matches
pub(super) static TS_RUPT: AtomicU64 = AtomicU64::new(0); // rupture re-matches that switched the session

/// Token passes of context before the first match attempt. The prompt
/// prefill routing lands in one layer sweep, so the first decode wrap
/// already carries the full prompt signature.
pub(crate) const TSIM_MIN_REQS: u64 = 64;
/// Cosine gate for accepting a session match (routes::cosine). Same-model
/// prefixes measured at 0.3+ on the nano chat traces, converging to ~0.99
/// with a full session of context; a mismatched or degenerate store stays
/// near 0.
pub(crate) const TSIM_THRESHOLD: f64 = 0.15;
/// Length of the cold window in token passes: the chained prefetch fires
/// while pass - matched_pass < COLD_PASSES, then yields to the Markov.
pub(crate) const COLD_PASSES: u64 = 50;
/// Rupture check cadence (token passes) past the cold window.
const RUPT_EVERY: u64 = 64;
/// Rupture gate: rolling prediction recall below this (over at least
/// RUPT_MIN_SAMPLE predictions) triggers a re-match of the recent window.
const RUPT_RECALL_FLOOR: f64 = 0.25;
const RUPT_MIN_SAMPLE: u64 = 128;
/// Demand events kept for the recent-window re-match signature.
const RECENT_WINDOW: usize = 8192;
/// Minimum demand requests of a session for it to be worth appending to
/// the store at exit.
const MIN_SAVE_REQS: u64 = 256;

/// MICROKIMI_TRACESIM=1: cross-session trace-similarity prefetch (default
/// OFF; see the block comment above).
pub fn tracesim_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env_off("MICROKIMI_TRACESIM"))
}

/// Engine state of the trace-similarity prefetcher: the stored sessions,
/// the current session signature, the match state and the offset
/// book-keeping prefetch jobs need (same affine-layout trick as Predictor).
pub(super) struct TraceSim {
    path: String,                      // <model>.routes
    store: crate::stream::route_history::RouteStore,  // past sessions
    cur: HashMap<u32, HashMap<u32, u32>>, // current session: layer -> expert -> count
    recent: VecDeque<(u32, u32)>,      // sliding window for rupture re-matches
    reqs: u64,
    pass: u64,                         // token passes (layer wraps)
    prev_layer: u32,                   // u32::MAX = none yet
    matched: Option<usize>,            // index of the matched stored session
    matched_pass: u64,
    rupt_checked: u64,                 // pass of the last rupture check
    mk_sample: (u64, u64),             // (PRED_HIT, PRED_TOT) at the last check
    ts_sample: (u64, u64),             // (TS_USED, TS_ISSUED) at the last check
    blob: usize,
    offs: HashMap<(u32, u32), [u64; 3]>,
    base: HashMap<u32, (u64, bool)>,
}

impl TraceSim {
    pub(super) fn new(path: String, store: crate::stream::route_history::RouteStore) -> TraceSim {
        TraceSim {
            path,
            store,
            cur: HashMap::new(),
            recent: VecDeque::new(),
            reqs: 0,
            pass: 0,
            prev_layer: u32::MAX,
            matched: None,
            matched_pass: 0,
            rupt_checked: 0,
            mk_sample: (0, 0),
            ts_sample: (0, 0),
            blob: 0,
            offs: HashMap::new(),
            base: HashMap::new(),
        }
    }

    /// Resolves the top-n experts of the matched session for `layer` into
    /// prefetch jobs (observed offsets, or the affine layout when verified).
    fn jobs_for(&self, layer: u32, n: usize) -> Vec<PrefetchJob> {
        let Some(m) = self.matched else { return Vec::new() };
        let mut jobs = Vec::new();
        for e in self.store.sessions[m].top_n(layer, n) {
            if let Some(o) = self.offs.get(&(layer, e)) {
                jobs.push((layer, e, *o));
            } else if let Some(&(b, true)) = self.base.get(&layer) {
                let o0 = b + e as u64 * 3 * self.blob as u64;
                jobs.push((layer, e, [o0, o0 + self.blob as u64, o0 + 2 * self.blob as u64]));
            }
        }
        jobs
    }

    /// Chained prediction fired on the transition INTO `layer`: the top-n
    /// experts of the matched session for the next MoE layer (wrapping to
    /// the first layer after the last, i.e. the next token's first layer).
    fn chained_jobs(&self, layer: u32, n: usize) -> Vec<PrefetchJob> {
        let Some(m) = self.matched else { return Vec::new() };
        if self.pass.saturating_sub(self.matched_pass) >= COLD_PASSES {
            return Vec::new(); // cold window over: the Markov is primary again
        }
        let rl = self.store.sessions[m].routed_layers();
        let Some(&target) = rl.iter().find(|&&l| l > layer).or_else(|| rl.first()) else { return Vec::new() };
        if target == layer {
            return Vec::new(); // single-MoE-layer model: nothing to chain to
        }
        self.jobs_for(target, n)
    }

    /// Match attempt of the current signature against the stored sessions.
    fn try_match(&mut self, counts: &HashMap<u32, HashMap<u32, u32>>, n: usize, layer: u32) -> Vec<PrefetchJob> {
        let cur = crate::stream::route_history::Session::from_counts(counts);
        let Some((idx, sim)) = self.store.best_match(&cur, TSIM_THRESHOLD) else { return Vec::new() };
        if self.matched == Some(idx) {
            return Vec::new();
        }
        self.matched = Some(idx);
        self.matched_pass = self.pass;
        TS_MATCH.fetch_add(1, Ordering::Relaxed);
        println!("stream-tracesim: matched stored session #{} (cosine {:.2}, {} requests of context)", idx, sim, self.reqs);
        self.chained_jobs(layer, n)
    }

    /// Rupture check (past the cold window, every RUPT_EVERY passes): the
    /// Markov rolling accuracy is the primary signal (its statistics decay
    /// with the topic; a cliff means the established regime broke). In
    /// lookahead mode the Markov does not run, so the tracesim prefetch's
    /// own consumption recall is the fallback signal. On a rupture the
    /// recent-window signature is re-matched; a different best session
    /// restarts the cold window.
    fn rupture_check(&mut self, layer: u32, n: usize) -> Vec<PrefetchJob> {
        if self.matched.is_none() || self.pass < self.rupt_checked + RUPT_EVERY {
            return Vec::new();
        }
        self.rupt_checked = self.pass;
        let (h, t) = (PRED_HIT.load(Ordering::Relaxed), PRED_TOT.load(Ordering::Relaxed));
        let (dh, dt) = (h - self.mk_sample.0, t - self.mk_sample.1);
        self.mk_sample = (h, t);
        let ruptured = if dt >= RUPT_MIN_SAMPLE {
            (dh as f64) < RUPT_RECALL_FLOOR * dt as f64
        } else {
            let (u, i) = (TS_USED.load(Ordering::Relaxed), TS_ISSUED.load(Ordering::Relaxed));
            let (du, di) = (u - self.ts_sample.0, i - self.ts_sample.1);
            self.ts_sample = (u, i);
            di >= RUPT_MIN_SAMPLE && (du as f64) < RUPT_RECALL_FLOOR * di as f64
        };
        if !ruptured {
            return Vec::new();
        }
        // re-match on the recent window only: the full-session histogram is
        // dominated by the old topic after a change
        let mut counts: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
        for &(l, e) in &self.recent {
            *counts.entry(l).or_default().entry(e).or_insert(0) += 1;
        }
        let before = self.matched;
        let jobs = self.try_match(&counts, n, layer);
        if self.matched != before {
            TS_RUPT.fetch_add(1, Ordering::Relaxed);
            println!("stream-tracesim: topic rupture (rolling recall < {}), re-matched session #{}", RUPT_RECALL_FLOOR, self.matched.unwrap());
        }
        jobs
    }

    /// Demand-stream hook (one router pick): maintains the current session
    /// signature and the offset book-keeping, detects layer transitions and
    /// token wraps, and returns the prefetch jobs to run in the background
    /// (match / chained fire / rupture re-match). `n` is the per-layer
    /// prediction count.
    pub(super) fn observe(&mut self, layer: u32, expert: u32, offs: [u64; 3], blob: usize, n: usize) -> Vec<PrefetchJob> {
        // offset book-keeping (same affine-layout trick as Predictor)
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
                    e.1 = false;
                }
            }
        }
        // session signature
        *self.cur.entry(layer).or_default().entry(expert).or_insert(0) += 1;
        self.recent.push_back((layer, expert));
        if self.recent.len() > RECENT_WINDOW {
            self.recent.pop_front();
        }
        self.reqs += 1;
        // layer transition / token wrap detection
        if layer == self.prev_layer {
            return Vec::new();
        }
        let wrap = self.prev_layer != u32::MAX && layer < self.prev_layer;
        let fired = if self.prev_layer == u32::MAX {
            Vec::new()
        } else {
            self.chained_jobs(layer, n)
        };
        self.prev_layer = layer;
        if wrap {
            self.pass += 1;
        }
        // cold-start match (retried on every wrap while unmatched: more
        // context only helps the cosine)
        if self.matched.is_none() && wrap && self.reqs >= TSIM_MIN_REQS {
            let counts = self.cur.clone();
            return self.try_match(&counts, n, layer);
        }
        // rupture check past the cold window
        if wrap {
            let jobs = self.rupture_check(layer, n);
            if !jobs.is_empty() {
                return jobs;
            }
        }
        fired
    }

    /// Appends the current session to the store (CacheInner drop). No-op
    /// below MIN_SAVE_REQS of context (a probe run is not a signature).
    pub(super) fn save(&mut self) {
        if self.reqs < MIN_SAVE_REQS {
            return;
        }
        let s = crate::stream::route_history::Session::from_counts(&self.cur);
        match crate::stream::route_history::RouteStore::append(&self.path, s) {
            Ok(n) => println!("stream-tracesim: session appended to {} ({} requests, {} sessions stored)", self.path, self.reqs, n),
            Err(e) => eprintln!("warning: cannot write {}: {}", self.path, e),
        }
    }
}

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
