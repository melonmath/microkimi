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
//   - LRU:    plain least-recently-used (the engine's --stream policy)
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

/// `microkimi cachereplay trace.bin [--top-k K] [--predict N]`
pub fn run(args: &[String]) {
    let Some(path) = args.get(2).filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: microkimi cachereplay trace.bin [--top-k K] [--predict N]");
        eprintln!("  replays a MICROKIMI_TRACE expert-request trace under LRU, Markov");
        eprintln!("  (prefetch, N experts/layer, default 16) and Belady (optimal offline),");
        eprintln!("  at cache capacities {:?} entries", CAPS);
        std::process::exit(1);
    };
    let top_k = flag_value(args, "--top-k").unwrap_or(16);
    let n_pred = flag_value(args, "--predict").unwrap_or(16);
    let bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("error: cannot read {}: {}", path, e);
        std::process::exit(1);
    });
    if bytes.len() % 8 != 0 {
        eprintln!("error: {} is {} bytes, not a multiple of 8 (u32 layer LE ++ u32 expert LE records)", path, bytes.len());
        std::process::exit(1);
    }
    let trace: Vec<Key> = bytes
        .chunks_exact(8)
        .map(|c| (u32::from_le_bytes(c[..4].try_into().unwrap()), u32::from_le_bytes(c[4..].try_into().unwrap())))
        .collect();
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
    println!("{:>8} {:>8} {:>8} {:>8} {:>10} {:>12} {:>12}", "capacity", "LRU", "Markov", "Belady", "gap B-LRU", "LRU fetches", "Markov fetches");
    let total = trace.len() as f64;
    let n_req = trace.len() as u64;
    for &cap in &CAPS {
        let lru_h = replay_lru(&trace, cap);
        let (mkv_h, mkv_pref) = replay_markov(&trace, cap, top_k, n_pred);
        let bel_h = replay_belady(&trace, cap);
        let lru = lru_h as f64 / total;
        let mkv = mkv_h as f64 / total;
        let bel = bel_h as f64 / total;
        println!(
            "{:>8} {:>7.1}% {:>7.1}% {:>7.1}% {:>+9.1}% {:>12} {:>12}",
            cap,
            100.0 * lru,
            100.0 * mkv,
            100.0 * bel,
            100.0 * (bel - lru),
            n_req - lru_h,                    // LRU fetches = demand misses
            n_req - mkv_h + mkv_pref          // Markov fetches = demand misses + prefetches
        );
    }
    println!();
    println!("hit rates over {} requests; Belady is the offline optimum for DEMAND-ONLY policies.", trace.len());
    println!("Markov is a prefetch policy: its demand hit-rate can exceed Belady at tight capacities;");
    println!("the fetch columns show the bandwidth spent for it (demand misses + prefetches vs misses).");
    println!("capacity unit: cache entries (uniform expert size per model, so entries == byte budget).");
}
