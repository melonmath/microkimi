// cachereplay: offline replay of a recorded expert-request trace under
// different cache policies and capacities.
//
// The router's expert picks do not depend on the cache contents, so the
// ordered (layer, expert) demand stream recorded under MICROKIMI_TRACE (see
// stream.rs) fully determines the cache behavior of ANY policy: one traced
// run yields the whole hit-rate vs capacity curve without rerunning the
// model.
//
// Trace format: raw little-endian record stream, no header. One record per
// demand request, 8 bytes: u32 layer LE ++ u32 expert LE, in call order.
//
//   microkimi cachereplay trace.bin [--top-k K] [--predict N]
//
// Policies replayed, at each capacity (in ENTRIES: expert blobs have a
// uniform size per model, so entry counts and byte budgets are equivalent):
//   - LRU:    plain least-recently-used eviction
//   - LFU:    lowest demand-hit count evicted first, recency tie-break (the
//             engine Lru's default eviction, stream.rs)
//   - ARC:    T1/T2 resident split + B1/B2 ghost lists, adaptive T1 target
//             (the engine's MICROKIMI_CACHE=arc policy, stream.rs)
//   - Markov: LRU + the engine's Markov prefetcher (stream.rs Predictor,
//             driven as-is with N predicted experts per MoE layer)
//   - Belady: optimal OFFLINE eviction (evict the entry reused farthest in
//             the future; the full trace is known). The upper bound for
//             demand-only policies: the LRU->Belady gap quantifies what a
//             better eviction policy could earn. (Prefetch policies are NOT
//             bounded by it: they preposition entries between demands.)

use std::collections::HashMap;

/// Capacities swept, in cache entries (one entry = one (layer, expert)).
const CAPS: [usize; 7] = [64, 128, 256, 512, 1024, 2048, 4096];

type Key = (u32, u32);

/// Entry-count LRU (the engine's Lru is byte-budgeted; with uniform entry
/// sizes an entry budget is the same policy).
struct EntryLru {
    map: HashMap<Key, u64>, // key -> last access tick
    tick: u64,
    cap: usize,
}

impl EntryLru {
    fn new(cap: usize) -> EntryLru {
        EntryLru { map: HashMap::new(), tick: 0, cap }
    }

    fn get(&mut self, k: Key) -> bool {
        match self.map.get_mut(&k) {
            Some(t) => {
                self.tick += 1;
                *t = self.tick;
                true
            }
            None => false,
        }
    }

    fn insert(&mut self, k: Key) {
        if self.cap == 0 {
            return;
        }
        self.tick += 1;
        self.map.insert(k, self.tick);
        while self.map.len() > self.cap {
            // evict the least recently used (never the just-inserted entry:
            // it carries the newest tick)
            let victim = *self.map.iter().min_by_key(|p| *p.1).unwrap().0;
            self.map.remove(&victim);
        }
    }
}

/// Plain LRU hit count over the trace at capacity `cap`.
fn replay_lru(trace: &[Key], cap: usize) -> u64 {
    let mut lru = EntryLru::new(cap);
    let mut hits = 0;
    for &k in trace {
        if lru.get(k) {
            hits += 1;
        } else {
            lru.insert(k);
        }
    }
    hits
}

/// LFU (recency tie-break) hit count over the trace at capacity `cap`:
/// victim = lowest demand-hit count, then oldest last access - the eviction
/// policy of the engine's Lru (stream.rs) by default.
fn replay_lfu(trace: &[Key], cap: usize) -> u64 {
    let mut map: HashMap<Key, (u64, u64)> = HashMap::new(); // key -> (hits, last tick)
    let mut tick = 0u64;
    let mut hits = 0;
    for &k in trace {
        tick += 1;
        match map.get_mut(&k) {
            Some(e) => {
                e.0 += 1;
                e.1 = tick;
                hits += 1;
            }
            None => {
                if cap == 0 {
                    continue;
                }
                if map.len() == cap {
                    let victim = *map.iter().min_by_key(|p| p.1).unwrap().0;
                    map.remove(&victim);
                }
                map.insert(k, (1, tick));
            }
        }
    }
    hits
}

/// ARC hit count over the trace at capacity `cap` (entries): the classic
/// T1/T2 resident split + B1/B2 ghost lists with the adaptive T1 target p -
/// the same scheme the engine runs under MICROKIMI_CACHE=arc (stream.rs),
/// in entry counts rather than bytes (uniform entry size per model).
fn replay_arc(trace: &[Key], cap: usize) -> u64 {
    if cap == 0 {
        return 0;
    }
    let mut t1: HashMap<Key, u64> = HashMap::new(); // key -> recency tick
    let mut t2: HashMap<Key, u64> = HashMap::new();
    let mut b1: HashMap<Key, u64> = HashMap::new();
    let mut b2: HashMap<Key, u64> = HashMap::new();
    let mut p = 0usize; // adaptive T1 target
    let mut tick = 0u64;
    let mut hits = 0u64;
    // LRU victim of a tick-stamped list
    fn lru_of(m: &HashMap<Key, u64>) -> Key {
        *m.iter().min_by_key(|p| *p.1).unwrap().0
    }
    // REPLACE(x, in_b2): evict the T1 LRU into B1 when T1 exceeds p (or
    // meets it on a B2-guided reference), else the T2 LRU into B2
    macro_rules! replace {
        ($in_b2:expr) => {
            if !t1.is_empty() && (t1.len() > p || ($in_b2 && t1.len() == p)) {
                let v = lru_of(&t1);
                t1.remove(&v);
                b1.insert(v, tick);
            } else if !t2.is_empty() {
                let v = lru_of(&t2);
                t2.remove(&v);
                b2.insert(v, tick);
            } else if !t1.is_empty() {
                let v = lru_of(&t1);
                t1.remove(&v);
                b1.insert(v, tick);
            }
        };
    }
    for &k in trace {
        tick += 1;
        if t1.remove(&k).is_some() {
            hits += 1;
            t2.insert(k, tick); // second reference: T1 -> T2
        } else if t2.contains_key(&k) {
            hits += 1;
            t2.insert(k, tick);
        } else if b1.remove(&k).is_some() {
            p = (p + (b2.len() / b1.len().max(1)).max(1)).min(cap);
            replace!(false);
            t2.insert(k, tick);
        } else if b2.remove(&k).is_some() {
            p = p.saturating_sub((b1.len() / b2.len().max(1)).max(1));
            replace!(true);
            t2.insert(k, tick);
        } else {
            if t1.len() + b1.len() >= cap {
                if t1.len() < cap {
                    let v = lru_of(&b1);
                    b1.remove(&v);
                    replace!(false);
                } else {
                    // B1 empty, T1 alone fills the cache: drop the T1 LRU
                    // without ghosting it
                    let v = lru_of(&t1);
                    t1.remove(&v);
                }
            } else if t1.len() + b1.len() + t2.len() + b2.len() >= cap {
                if t1.len() + b1.len() + t2.len() + b2.len() >= 2 * cap {
                    let v = lru_of(&b2);
                    b2.remove(&v);
                }
                replace!(false);
            }
            t1.insert(k, tick);
        }
    }
    hits
}

/// Belady (optimal offline) hit count: on a miss at a full cache, evict the
/// resident entry whose NEXT use in the trace is farthest away (or never).
fn replay_belady(trace: &[Key], cap: usize) -> u64 {
    if cap == 0 {
        return 0;
    }
    // next[i] = index of the next occurrence of trace[i] after i (MAX = never)
    let mut next = vec![usize::MAX; trace.len()];
    let mut last: HashMap<Key, usize> = HashMap::new();
    for i in (0..trace.len()).rev() {
        if let Some(&j) = last.get(&trace[i]) {
            next[i] = j;
        }
        last.insert(trace[i], i);
    }
    let mut cache: HashMap<Key, usize> = HashMap::new(); // key -> next use
    let mut hits = 0;
    for (i, &k) in trace.iter().enumerate() {
        if cache.contains_key(&k) {
            hits += 1;
        } else {
            if cache.len() == cap {
                let victim = *cache.iter().max_by_key(|p| *p.1).unwrap().0;
                cache.remove(&victim);
            }
            cache.insert(k, usize::MAX);
        }
        cache.insert(k, next[i]);
    }
    hits
}

/// Markov-prefetch replay: the engine's own Predictor (stream.rs), driven
/// as-is, on top of the same entry LRU. Synthetic offsets mimic the uniform
/// expert-major .bin layout (expert e at [3e, 3e+3), blob 1) so the
/// predictor's affine offset extrapolation works exactly as on a real model.
/// Prefetches replicate CacheInner::prefetch_one: a cached entry is only
/// recency-refreshed, a missing one is inserted (and counted as a fetch).
///
/// Returns (demand hits, prefetches fetched). NOTE: as a PREFETCH policy the
/// demand hit-rate is NOT bounded by Belady (the optimal demand-only bound):
/// prefetches preposition entries between demands, so Markov can beat the
/// Belady column at tight capacities; the fetch columns show the bandwidth
/// spent for it (demand misses + prefetches vs demand misses only).
fn replay_markov(trace: &[Key], cap: usize, top_k: usize, n: usize) -> (u64, u64) {
    let mut pred = crate::stream::Predictor::new();
    let mut lru = EntryLru::new(cap);
    let mut hits = 0;
    let mut prefetched = 0;
    for &(layer, expert) in trace {
        let o = expert as u64 * 3;
        let jobs = pred.observe(layer, expert, [o, o + 1, o + 2], 1, top_k, n);
        for (jl, je, _) in jobs {
            if !lru.get((jl, je)) {
                lru.insert((jl, je));
                prefetched += 1;
            }
        }
        if lru.get((layer, expert)) {
            hits += 1;
        } else {
            lru.insert((layer, expert));
        }
    }
    (hits, prefetched)
}

fn flag_value(args: &[String], name: &str) -> Option<usize> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
}

// ── tracesim: cross-session cold-start prefetch, offline A/B ──
//
// `microkimi routebuild store.routes trace.bin [trace2.bin ...]` converts
// MICROKIMI_TRACE request streams into routing signature sessions appended
// to a <model>.routes store (routes.rs): the same records the engine writes
// at exit under MICROKIMI_TRACESIM=1, built offline for benchmarking.
//
// `microkimi cachereplay trace.bin --tracesim store.routes [--first N]`
// replays the trace with the engine's trace-similarity prefetch policy
// (stream.rs TraceSim, mirrored below) on top of the entry LRU and reports
// the hit-rate over the first N token passes - the cold-start window where
// the Markov predictor has no history - against plain LRU and the Markov
// prefetch. The matching metric check (cosine vs top-k set overlap) is
// printed at match time.

/// One token pass = one layer wrap in the request stream (layers strictly
/// ascend within a pass; the prompt prefill is one unioned sweep).
fn pass_of(prev_layer: u32, layer: u32) -> bool {
    prev_layer != u32::MAX && layer < prev_layer
}

/// Builds the routing signature of one trace (per-layer expert histogram).
fn trace_to_session(trace: &[Key]) -> crate::routes::Session {
    let mut counts: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
    for &(l, e) in trace {
        *counts.entry(l).or_default().entry(e).or_insert(0) += 1;
    }
    crate::routes::Session::from_counts(&counts)
}

/// Reads a MICROKIMI_TRACE record stream (u32 layer LE ++ u32 expert LE).
fn read_trace(path: &str) -> Vec<Key> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", path, e);
        std::process::exit(1);
    });
    if bytes.len() % 8 != 0 {
        eprintln!("error: {} is {} bytes, not a multiple of 8 (u32 layer LE ++ u32 expert LE records)", path, bytes.len());
        std::process::exit(1);
    }
    bytes
        .chunks_exact(8)
        .map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), u32::from_le_bytes(c[4..].try_into().unwrap())))
        .collect()
}

/// `microkimi routebuild store.routes trace.bin [trace2.bin ...]`
pub fn routebuild(args: &[String]) {
    let files: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
    if files.len() < 2 {
        eprintln!("usage: microkimi routebuild store.routes trace.bin [trace2.bin ...]");
        eprintln!("  appends the routing signature of each MICROKIMI_TRACE stream to the");
        eprintln!("  store (routes.rs format, the same records MICROKIMI_TRACESIM=1 writes)");
        std::process::exit(1);
    }
    let store_path = files[0].as_str();
    for f in &files[1..] {
        let trace = read_trace(f);
        let session = trace_to_session(&trace);
        let n = crate::routes::RouteStore::append(store_path, session).unwrap_or_else(|e| {
            eprintln!("error: cannot write {}: {}", store_path, e);
            std::process::exit(1);
        });
        println!("routebuild: {} -> {} requests appended to {} ({} sessions stored)", f, trace.len(), store_path, n);
    }
}

/// Cold-start A/B row: hit counts over the first `first_passes` token passes
/// at capacity `cap` - plain LRU, Markov prefetch (the engine's Predictor),
/// and LRU + the trace-similarity prefetch (the engine's TraceSim policy,
/// mirrored). Returns (lru, markov, tracesim, tracesim prefetches issued,
/// matched session (index, cosine), overlap-metric pick (index, overlap)).
fn cold_row(
    trace: &[Key],
    cap: usize,
    store: &crate::routes::RouteStore,
    top_k: usize,
    n_pred: usize,
    first_passes: u64,
) -> (u64, u64, u64, u64, Option<(usize, f64)>, Option<(usize, f64)>) {
    // plain LRU, windowed hits
    let mut lru_hits = 0u64;
    {
        let mut lru = EntryLru::new(cap);
        let (mut prev, mut pass) = (u32::MAX, 0u64);
        for &(l, e) in trace {
            if pass_of(prev, l) {
                pass += 1;
            }
            prev = l;
            if lru.get((l, e)) {
                if pass < first_passes {
                    lru_hits += 1;
                }
            } else {
                lru.insert((l, e));
            }
        }
    }
    // Markov prefetch (the engine's Predictor, driven as in replay_markov),
    // windowed hits
    let mut mkv_hits = 0u64;
    {
        let mut pred = crate::stream::Predictor::new();
        let mut lru = EntryLru::new(cap);
        let (mut prev, mut pass) = (u32::MAX, 0u64);
        for &(l, e) in trace {
            if pass_of(prev, l) {
                pass += 1;
            }
            prev = l;
            let o = e as u64 * 3;
            for (jl, je, _) in pred.observe(l, e, [o, o + 1, o + 2], 1, top_k, n_pred) {
                if !lru.get((jl, je)) {
                    lru.insert((jl, je));
                }
            }
            if lru.get((l, e)) {
                if pass < first_passes {
                    mkv_hits += 1;
                }
            } else {
                lru.insert((l, e));
            }
        }
    }
    // LRU + trace-similarity prefetch: mirror of stream.rs TraceSim (match at
    // the first wrap with TSIM_MIN_REQS of context, cosine >= TSIM_THRESHOLD,
    // chained top-n prefetch of the next MoE layer on every layer transition
    // during the COLD_PASSES cold window). Windowed hits.
    let mut tsim_hits = 0u64;
    let mut tsim_pref = 0u64;
    let mut matched: Option<(usize, f64)> = None;
    let mut overlap_pick: Option<(usize, f64)> = None;
    {
        let mut lru = EntryLru::new(cap);
        let mut counts: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
        let mut reqs = 0u64;
        let (mut prev, mut pass) = (u32::MAX, 0u64);
        let mut matched_pass = 0u64;
        // chained fire: top-n of the matched session for the MoE layer
        // after `l` (mirror of TraceSim::chained_jobs)
        let fire = |lru: &mut EntryLru, l: u32, pass: u64, matched_pass: u64, matched: Option<(usize, f64)>, tsim_pref: &mut u64| {
            let Some((m, _)) = matched else { return };
            if pass - matched_pass >= crate::stream::COLD_PASSES {
                return;
            }
            let rl = store.sessions[m].routed_layers();
            let Some(&target) = rl.iter().find(|&&x| x > l).or_else(|| rl.first()) else { return };
            if target == l {
                return;
            }
            for pe in store.sessions[m].top_n(target, n_pred) {
                if !lru.get((target, pe)) {
                    lru.insert((target, pe));
                    *tsim_pref += 1;
                }
            }
        };
        for &(l, e) in trace {
            *counts.entry(l).or_default().entry(e).or_insert(0) += 1;
            reqs += 1;
            let wrap = pass_of(prev, l);
            // chained fire on the transition into layer l (before the wrap
            // increments the pass, as in TraceSim::observe)
            if prev != u32::MAX && l != prev {
                fire(&mut lru, l, pass, matched_pass, matched, &mut tsim_pref);
            }
            prev = l;
            if wrap {
                pass += 1;
            }
            // cold-start match at the first wrap with enough context, then
            // an immediate chained fire (as try_match does in the engine)
            if matched.is_none() && wrap && reqs >= crate::stream::TSIM_MIN_REQS {
                let cur = crate::routes::Session::from_counts(&counts);
                if let Some((idx, sim)) = store.best_match(&cur, crate::stream::TSIM_THRESHOLD) {
                    matched = Some((idx, sim));
                    matched_pass = pass;
                    // the alternative metric, for the printed comparison
                    overlap_pick = store
                        .sessions
                        .iter()
                        .enumerate()
                        .map(|(i, s)| (i, crate::routes::top_overlap(&cur, s, top_k)))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    fire(&mut lru, l, pass, matched_pass, matched, &mut tsim_pref);
                }
            }
            if lru.get((l, e)) {
                if pass < first_passes {
                    tsim_hits += 1;
                }
            } else {
                lru.insert((l, e));
            }
        }
    }
    (lru_hits, mkv_hits, tsim_hits, tsim_pref, matched, overlap_pick)
}

/// `microkimi cachereplay trace.bin [--top-k K] [--predict N] [--tracesim store.routes] [--first N]`
pub fn run(args: &[String]) {
    let Some(path) = args.get(2).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: microkimi cachereplay trace.bin [--top-k K] [--predict N] [--tracesim store.routes] [--first N]");
        eprintln!("  replays a MICROKIMI_TRACE expert-request trace under LRU, Markov");
        eprintln!("  (prefetch, N experts/layer, default 16) and Belady (optimal offline),");
        eprintln!("  at cache capacities {:?} entries", CAPS);
        eprintln!("  --tracesim store.routes: cold-start A/B of the trace-similarity prefetch");
        eprintln!("  (stream.rs TraceSim policy mirrored offline) over the first --first N");
        eprintln!("  token passes (default 50)");
        std::process::exit(1);
    };
    let top_k = flag_value(args, "--top-k").unwrap_or(16);
    let n_pred = flag_value(args, "--predict").unwrap_or(16);
    let tsim_store: Option<String> = args
        .iter()
        .position(|a| a == "--tracesim")
        .and_then(|i| args.get(i + 1))
        .filter(|s| !s.starts_with("--"))
        .cloned();
    let first_passes = flag_value(args, "--first").unwrap_or(50) as u64;
    let trace = read_trace(path);
    if trace.is_empty() {
        eprintln!("error: {} holds no requests", path);
        std::process::exit(1);
    }
    let distinct: std::collections::HashSet<Key> = trace.iter().copied().collect();
    println!(
        "cachereplay: {} - {} requests, {} distinct experts (top-k {}, markov predict {})",
        path,
        trace.len(),
        distinct.len(),
        top_k,
        n_pred
    );
    println!();
    println!("{:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>10} {:>12} {:>12}", "capacity", "LRU", "LFU", "ARC", "Markov", "Belady", "gap B-LRU", "LRU fetches", "Markov fetches");
    let total = trace.len() as f64;
    let n_req = trace.len() as u64;
    for &cap in &CAPS {
        let lru_h = replay_lru(&trace, cap);
        let lfu_h = replay_lfu(&trace, cap);
        let arc_h = replay_arc(&trace, cap);
        let (mkv_h, mkv_pref) = replay_markov(&trace, cap, top_k, n_pred);
        let bel_h = replay_belady(&trace, cap);
        let lru = lru_h as f64 / total;
        let lfu = lfu_h as f64 / total;
        let arc = arc_h as f64 / total;
        let mkv = mkv_h as f64 / total;
        let bel = bel_h as f64 / total;
        println!(
            "{:>8} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}% {:>+9.1}% {:>12} {:>12}",
            cap,
            100.0 * lru,
            100.0 * lfu,
            100.0 * arc,
            100.0 * mkv,
            100.0 * bel,
            100.0 * (bel - lru),
            n_req - lru_h,                    // LRU fetches = demand misses
            n_req - mkv_h + mkv_pref          // Markov fetches = demand misses + prefetches
        );
    }
    println!();
    println!("hit rates over {} requests; Belady is the offline optimum for DEMAND-ONLY policies.", trace.len());
    println!("LFU = lowest demand-hit count evicted first, recency tie-break (the engine's default policy).");
    println!("ARC = T1/T2 + B1/B2 ghosts with the adaptive T1 target (MICROKIMI_CACHE=arc in the engine).");
    println!("Markov is a prefetch policy: its demand hit-rate can exceed Belady at tight capacities;");
    println!("the fetch columns show the bandwidth spent for it (demand misses + prefetches vs misses).");
    println!("capacity unit: cache entries (uniform expert size per model, so entries == byte budget).");

    // --tracesim store.routes: cold-start A/B over the first --first N token
    // passes (the window where the Markov has no history). The tracesim
    // column replays the engine's TraceSim policy (stream.rs) mirrored by
    // cold_row above.
    if let Some(store_path) = tsim_store {
        let store = crate::routes::RouteStore::load(&store_path).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        if store.sessions.is_empty() {
            eprintln!("error: {} holds no sessions (see routebuild)", store_path);
            std::process::exit(1);
        }
        // requests in the window (pass < first_passes), for the rates
        let mut window = 0u64;
        {
            let (mut prev, mut pass) = (u32::MAX, 0u64);
            for &(l, _) in &trace {
                if pass_of(prev, l) {
                    pass += 1;
                }
                prev = l;
                if pass < first_passes {
                    window += 1;
                }
            }
        }
        println!();
        println!(
            "tracesim cold-start A/B: first {} token passes ({} requests), store {} ({} sessions)",
            first_passes,
            window,
            store_path,
            store.sessions.len()
        );
        println!("{:>8} {:>8} {:>8} {:>9} {:>10} {:>12}", "capacity", "LRU", "Markov", "tracesim", "gain pts", "tsim fetches");
        for &cap in &[128usize, 256, 512, 1024] {
            let (lru_h, mkv_h, tsim_h, tsim_pref, matched, overlap_pick) = cold_row(&trace, cap, &store, top_k, n_pred, first_passes);
            println!(
                "{:>8} {:>7.1}% {:>7.1}% {:>8.1}% {:>+9.1} {:>12}",
                cap,
                100.0 * lru_h as f64 / window as f64,
                100.0 * mkv_h as f64 / window as f64,
                100.0 * tsim_h as f64 / window as f64,
                100.0 * (tsim_h as f64 - lru_h as f64) / window as f64,
                tsim_pref
            );
            if cap == 128 {
                match matched {
                    Some((idx, sim)) => println!("  matched session #{} at the first qualified wrap (cosine {:.3})", idx, sim),
                    None => println!("  no session matched (cosine below {})", crate::stream::TSIM_THRESHOLD),
                }
                if let Some((idx, ov)) = overlap_pick {
                    println!("  metric check: the top-{} set-overlap metric would have picked session #{} (overlap {:.3})", top_k, idx, ov);
                }
            }
        }
        println!("tracesim = LRU + the cross-session chained prefetch (MICROKIMI_TRACESIM=1 in the engine);");
        println!("hit rates over the first {} token passes only (cold start: the Markov column shows the same", first_passes);
        println!("window for the online predictor, which has no transition history there).");
    }
}
