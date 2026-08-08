// routes - cross-session routing signature store for the trace-similarity
// expert prefetcher (MICROKIMI_TRACESIM, stream.rs).
//
// The online predictors of stream.rs are blind in two regimes: session COLD
// START (no observed transitions yet) and a mid-session TOPIC CHANGE (the
// decayed statistics describe the old topic). The routing histograms of PAST
// sessions are a usable stand-in there: measured on MICROKIMI_TRACE
// recordings of the nano chat checkpoints, the per-layer expert histograms
// of different prompts agree at cosine ~0.99 over a full session, and a
// 4-token prefix already scores ~0.31 against same-model sessions (the
// chained replay of tools_replay.rs, `cachereplay --tracesim`, turns that
// signal into +8.5 points of demand hit-rate over the first 50 tokens at
// tight cache capacities).
//
// One Session is a compact per-MoE-layer expert histogram of one run. The
// ordered request stream is NOT kept (MICROKIMI_TRACE already records that
// for offline replay); the signature only has to answer "which experts does
// a session like this one touch, per layer". Transitions are deliberately
// absent: the established regime belongs to the Markov/lookahead predictors.
//
// File format (<model>.routes, integers little-endian):
//   magic     : 8 bytes "MKRTS001"
//   version   : u32 (1)
//   n_layers  : u32 (max layer index + 1 over the stored sessions)
//   n_experts : u32 (max expert index + 1)
//   n_sessions: u32
//   reserved  : u32 (0)
//   then n_sessions session records:
//     requests  : u64 (routing decisions recorded)
//     n_layers  : u32 (this session)
//     n_experts : u32 (this session)
//     per layer l in 0..n_layers-1:
//       nz : u32, then nz x (expert u32, count u32), sorted by expert id
//
// Append semantics, rewrite mechanics: a save rewrites the whole file
// (tmp + rename). A session record is a few KB (sparse per-layer pairs), the
// cap of MAX_SESSIONS bounds the file, and an atomic rewrite cannot leave a
// torn tail the way an in-place append can. Oldest sessions are dropped past
// the cap: recent sessions are the most representative of the current usage.

use std::collections::HashMap;

const MAGIC: &[u8; 8] = b"MKRTS001";
const VERSION: u32 = 1;
/// Stored sessions cap (oldest dropped past it): bounds the file at a few MB
/// even on the full-size geometry (61 MoE layers x 896 experts).
pub const MAX_SESSIONS: usize = 32;

/// Per-layer expert histogram of one run: layers[layer] = (expert, count)
/// pairs sorted by expert id (empty for layers with no routing recorded).
pub struct Session {
    pub requests: u64,
    pub layers: Vec<Vec<(u32, u32)>>,
}

impl Session {
    /// Builds a session signature from per-layer expert counts.
    pub fn from_counts(counts: &HashMap<u32, HashMap<u32, u32>>) -> Session {
        let n_layers = counts.keys().max().map(|m| *m as usize + 1).unwrap_or(0);
        let mut layers = vec![Vec::new(); n_layers];
        let mut requests = 0u64;
        for (&l, row) in counts {
            let mut v: Vec<(u32, u32)> = row.iter().map(|(&e, &c)| (e, c)).collect();
            v.sort_unstable();
            requests += row.values().map(|&c| c as u64).sum::<u64>();
            layers[l as usize] = v;
        }
        Session { requests, layers }
    }

    /// Max expert index + 1 (0 when empty).
    pub fn n_experts(&self) -> u32 {
        self.layers.iter().flatten().map(|&(e, _)| e + 1).max().unwrap_or(0)
    }

    /// The n most routed experts of `layer`, count-descending, ties broken by
    /// expert id (deterministic). Empty when the layer was never routed.
    pub fn top_n(&self, layer: u32, n: usize) -> Vec<u32> {
        let Some(row) = self.layers.get(layer as usize) else { return Vec::new() };
        let mut v = row.clone();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(n);
        v.into_iter().map(|(e, _)| e).collect()
    }

    /// Layers with at least one recorded routing decision, ascending.
    pub fn routed_layers(&self) -> Vec<u32> {
        self.layers
            .iter()
            .enumerate()
            .filter(|(_, row)| !row.is_empty())
            .map(|(l, _)| l as u32)
            .collect()
    }
}

/// Cosine similarity of two session signatures over the concatenated
/// per-layer histograms (rows are expert-sorted: merge join per layer).
/// Chosen over top-k set overlap by measurement on the nano chat traces: a
/// 4-token prefix keeps a usable cosine contrast (~0.31 same-model vs ~0.27
/// for the mismatched window) while top-16 overlap collapses to 0.02-0.05
/// with no contrast at all (the prefix top sets are pure noise that early).
pub fn cosine(a: &Session, b: &Session) -> f64 {
    let nl = a.layers.len().max(b.layers.len());
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for l in 0..nl {
        let ra = a.layers.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
        let rb = b.layers.get(l).map(|v| v.as_slice()).unwrap_or(&[]);
        for &(_, c) in ra {
            na += (c as f64) * (c as f64);
        }
        for &(_, c) in rb {
            nb += (c as f64) * (c as f64);
        }
        let (mut i, mut j) = (0, 0);
        while i < ra.len() && j < rb.len() {
            if ra[i].0 == rb[j].0 {
                dot += (ra[i].1 as f64) * (rb[j].1 as f64);
                i += 1;
                j += 1;
            } else if ra[i].0 < rb[j].0 {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

/// Mean per-layer overlap of the top-k expert sets (the alternative metric
/// kept for the cachereplay metric comparison; see the cosine comment).
pub fn top_overlap(a: &Session, b: &Session, k: usize) -> f64 {
    let nl = a.layers.len().max(b.layers.len());
    let (mut sum, mut cnt) = (0.0f64, 0usize);
    for l in 0..nl {
        let ta: std::collections::HashSet<u32> = a.top_n(l as u32, k).into_iter().collect();
        let tb: std::collections::HashSet<u32> = b.top_n(l as u32, k).into_iter().collect();
        if ta.is_empty() && tb.is_empty() {
            continue;
        }
        let inter = ta.intersection(&tb).count();
        let union = ta.len().max(tb.len()).max(1);
        sum += inter as f64 / union as f64;
        cnt += 1;
    }
    if cnt == 0 {
        0.0
    } else {
        sum / cnt as f64
    }
}

/// The stored sessions of one model (<model>.routes).
pub struct RouteStore {
    pub n_layers: u32,
    pub n_experts: u32,
    pub sessions: Vec<Session>,
}

impl RouteStore {
    pub fn empty() -> RouteStore {
        RouteStore { n_layers: 0, n_experts: 0, sessions: Vec::new() }
    }

    /// Loads a store. A torn tail (crash between the record append and the
    /// header rewrite of a pre-1 writer, or a truncated copy) is tolerated:
    /// decoding stops at the first unreadable record and keeps the valid
    /// prefix. Bad magic/version/dims are hard errors.
    pub fn load(path: &str) -> Result<RouteStore, String> {
        let b = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        let get = |p: usize, n: usize| -> Result<&[u8], String> {
            b.get(p..p + n).ok_or_else(|| format!("{}: truncated routes file", path))
        };
        if get(0, 8)? != MAGIC {
            return Err(format!("{}: not a routes file (bad magic)", path));
        }
        let u32at = |p: usize| -> Result<u32, String> { Ok(u32::from_le_bytes(get(p, 4)?.try_into().unwrap())) };
        let version = u32at(8)?;
        if version != VERSION {
            return Err(format!("{}: unsupported routes version {}", path, version));
        }
        let (n_layers, n_experts, n_sessions) = (u32at(12)?, u32at(16)?, u32at(20)?);
        let mut sessions = Vec::new();
        let mut p = 28usize;
        for _ in 0..n_sessions {
            // torn-tail tolerance: an undecodable record ends the valid prefix
            let rec = (|| -> Option<Session> {
                let u32r = |p: usize| -> Option<u32> { Some(u32::from_le_bytes(b.get(p..p + 4)?.try_into().ok()?)) };
                let requests = u64::from_le_bytes(b.get(p..p + 8)?.try_into().ok()?);
                let sl = u32r(p + 8)? as usize;
                let _se = u32r(p + 12)?;
                let mut q = p + 16;
                let mut layers = Vec::with_capacity(sl);
                for _ in 0..sl {
                    let nz = u32r(q)? as usize;
                    q += 4;
                    let mut row = Vec::with_capacity(nz);
                    for _ in 0..nz {
                        let e = u32r(q)?;
                        let c = u32r(q + 4)?;
                        q += 8;
                        row.push((e, c));
                    }
                    layers.push(row);
                }
                p = q;
                Some(Session { requests, layers })
            })();
            match rec {
                Some(s) => sessions.push(s),
                None => break,
            }
        }
        Ok(RouteStore { n_layers, n_experts, sessions })
    }

    /// Serializes the store (see the format comment at the top).
    fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&self.n_layers.to_le_bytes());
        out.extend_from_slice(&self.n_experts.to_le_bytes());
        out.extend_from_slice(&(self.sessions.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved
        for s in &self.sessions {
            out.extend_from_slice(&s.requests.to_le_bytes());
            out.extend_from_slice(&(s.layers.len() as u32).to_le_bytes());
            out.extend_from_slice(&s.n_experts().to_le_bytes());
            for row in &s.layers {
                out.extend_from_slice(&(row.len() as u32).to_le_bytes());
                for &(e, c) in row {
                    out.extend_from_slice(&e.to_le_bytes());
                    out.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        out
    }

    /// Atomic whole-file save (tmp + rename; see the top comment for why the
    /// file is rewritten rather than appended in place).
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let tmp = format!("{}.partial-{}", path, std::process::id());
        std::fs::write(&tmp, self.to_bytes())?;
        std::fs::rename(&tmp, path)
    }

    /// Appends one session to the store file (load + push + cap + rewrite).
    /// A missing or unreadable existing file starts a fresh store.
    pub fn append(path: &str, session: Session) -> std::io::Result<usize> {
        let mut store = Self::load(path).unwrap_or_else(|_| RouteStore::empty());
        store.n_layers = store.n_layers.max(session.layers.len() as u32);
        store.n_experts = store.n_experts.max(session.n_experts());
        store.sessions.push(session);
        if store.sessions.len() > MAX_SESSIONS {
            store.sessions.drain(..store.sessions.len() - MAX_SESSIONS);
        }
        store.save(path)?;
        Ok(store.sessions.len())
    }

    /// Best-matching stored session for the current context signature:
    /// (session index, cosine) when the best cosine reaches `threshold`.
    pub fn best_match(&self, cur: &Session, threshold: f64) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        for (i, s) in self.sessions.iter().enumerate() {
            let sim = cosine(cur, s);
            if best.map(|(_, b)| sim > b).unwrap_or(true) {
                best = Some((i, sim));
            }
        }
        best.filter(|&(_, sim)| sim >= threshold && sim > 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        std::env::temp_dir().join(format!("microkimi_routes_test_{}_{}.bin", std::process::id(), name)).to_string_lossy().into_owned()
    }

    fn counts(v: &[(u32, u32, u32)]) -> HashMap<u32, HashMap<u32, u32>> {
        let mut m: HashMap<u32, HashMap<u32, u32>> = HashMap::new();
        for &(l, e, c) in v {
            *m.entry(l).or_default().entry(e).or_insert(0) += c;
        }
        m
    }

    #[test]
    fn save_load_roundtrip() {
        let p = tmp("roundtrip");
        let s = Session::from_counts(&counts(&[(1, 5, 3), (1, 2, 7), (3, 9, 1), (3, 9, 1)]));
        assert_eq!(s.requests, 12);
        let n = RouteStore::append(&p, s).unwrap();
        assert_eq!(n, 1);
        let back = RouteStore::load(&p).unwrap();
        assert_eq!(back.sessions.len(), 1);
        assert_eq!(back.n_layers, 4);
        assert_eq!(back.n_experts, 10);
        let s0 = &back.sessions[0];
        assert_eq!(s0.requests, 12);
        assert_eq!(s0.layers[1], vec![(2, 7), (5, 3)]);
        assert_eq!(s0.layers[3], vec![(9, 2)]);
        assert!(s0.layers[0].is_empty() && s0.layers[2].is_empty());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn append_caps_at_max_sessions() {
        let p = tmp("cap");
        for i in 0..MAX_SESSIONS + 5 {
            RouteStore::append(&p, Session::from_counts(&counts(&[(0, i as u32, 1)]))).unwrap();
        }
        let back = RouteStore::load(&p).unwrap();
        assert_eq!(back.sessions.len(), MAX_SESSIONS);
        // the 5 oldest are gone: the first stored session is expert 5
        assert_eq!(back.sessions[0].layers[0], vec![(5, 1)]);
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn load_tolerates_torn_tail() {
        let p = tmp("torn");
        RouteStore::append(&p, Session::from_counts(&counts(&[(0, 1, 1)]))).unwrap();
        RouteStore::append(&p, Session::from_counts(&counts(&[(0, 2, 2)]))).unwrap();
        let mut b = std::fs::read(&p).unwrap();
        b.truncate(b.len() - 5); // tear the second record
        std::fs::write(&p, &b).unwrap();
        let back = RouteStore::load(&p).unwrap();
        assert_eq!(back.sessions.len(), 1, "the torn record must be dropped, the valid prefix kept");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn load_rejects_garbage() {
        let p = tmp("garbage");
        std::fs::write(&p, b"not a routes file").unwrap();
        assert!(RouteStore::load(&p).is_err());
        std::fs::remove_file(&p).ok();
        assert!(RouteStore::load("/nonexistent/file.routes").is_err());
    }

    #[test]
    fn cosine_bounds_and_identity() {
        let a = Session::from_counts(&counts(&[(1, 5, 3), (1, 2, 7)]));
        let b = Session::from_counts(&counts(&[(1, 5, 3), (1, 2, 7)]));
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-9);
        let c = Session::from_counts(&counts(&[(1, 8, 4)])); // disjoint experts
        assert_eq!(cosine(&a, &c), 0.0);
        let d = Session::from_counts(&counts(&[(2, 5, 3), (2, 2, 7)])); // same experts, other layer
        assert_eq!(cosine(&a, &d), 0.0);
        let e = Session { requests: 0, layers: Vec::new() };
        assert_eq!(cosine(&a, &e), 0.0);
    }

    #[test]
    fn top_n_is_count_ordered_and_deterministic() {
        let s = Session::from_counts(&counts(&[(1, 5, 3), (1, 2, 7), (1, 9, 7), (1, 4, 1)]));
        assert_eq!(s.top_n(1, 2), vec![2, 9]); // count tie: expert id ascending
        assert_eq!(s.top_n(1, 16), vec![2, 9, 5, 4]);
        assert_eq!(s.top_n(0, 4), Vec::<u32>::new());
        assert_eq!(s.routed_layers(), vec![1]);
    }

    #[test]
    fn best_match_threshold() {
        let p = tmp("match");
        RouteStore::append(&p, Session::from_counts(&counts(&[(1, 5, 3), (1, 2, 7)]))).unwrap();
        RouteStore::append(&p, Session::from_counts(&counts(&[(1, 8, 9)]))).unwrap();
        let store = RouteStore::load(&p).unwrap();
        let cur = Session::from_counts(&counts(&[(1, 2, 4), (1, 5, 1)]));
        let (idx, sim) = store.best_match(&cur, 0.5).unwrap();
        assert_eq!(idx, 0);
        assert!(sim > 0.9);
        // above the reachable similarity: no match
        assert!(store.best_match(&cur, 1.1).is_none());
        std::fs::remove_file(&p).ok();
    }
}
