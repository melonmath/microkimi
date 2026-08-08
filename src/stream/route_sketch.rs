// cms - count-min sketch of the expert routing decisions of a run.
//
// Records every (layer, expert) pair the noaux_tc router actually selects,
// without storing the full request stream (that is what MICROKIMI_TRACE does,
// see stream.rs / tools/replay.rs). A sketch is a fixed 4 x 4096 u32 table
// (64 KB), so weeks of runs cost nothing to keep; the per-request order is
// lost, only the frequencies survive. Intended use: hot/warm expert tiering
// (which experts deserve the fast tier) fed by real traffic instead of a
// single calibration pass.
//
// The estimate for a pair is the minimum over the 4 rows: it is never below
// the true count and can overshoot when unrelated pairs collide on all 4
// rows (with 4096 columns and tens of thousands of live pairs, collisions
// are rare but possible). Top-N rankings read from a sketch are therefore
// approximate: good enough for tiering, not for exact accounting.
//
// File format (integers little-endian):
//   magic     : 8 bytes "MKCMS001"
//   version   : u32 (1)
//   rows      : u32 (4)
//   cols      : u32 (4096)
//   n_layers  : u32 (max recorded layer + 1)
//   n_experts : u32 (max recorded expert + 1)
//   total     : u64 routing decisions recorded
//   counters  : rows x cols u32, row-major

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

const MAGIC: &[u8; 8] = b"MKCMS001";
const VERSION: u32 = 1;
pub const ROWS: usize = 4;
pub const COLS: usize = 4096;

pub struct Cms {
    table: Vec<u32>, // ROWS x COLS, row-major
    total: u64,
    max_layer: u32,
    max_expert: u32,
}

/// splitmix64, one finalizer per (key, row) pair: row-independent hashes
/// from a single 64-bit key, no tables to keep in sync.
fn hash(key: u64, row: usize) -> usize {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(row as u64 + 1));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) as usize) % COLS
}

impl Cms {
    pub fn new() -> Self {
        Cms { table: vec![0; ROWS * COLS], total: 0, max_layer: 0, max_expert: 0 }
    }

    fn key(layer: u32, expert: u32) -> u64 {
        ((layer as u64) << 32) | expert as u64
    }

    pub fn add(&mut self, layer: u32, expert: u32) {
        let k = Self::key(layer, expert);
        for r in 0..ROWS {
            let c = hash(k, r);
            self.table[r * COLS + c] = self.table[r * COLS + c].saturating_add(1);
        }
        self.total += 1;
        self.max_layer = self.max_layer.max(layer);
        self.max_expert = self.max_expert.max(expert);
    }

    /// Count-min estimate: min over rows. Never below the true count.
    pub fn estimate(&self, layer: u32, expert: u32) -> u32 {
        let k = Self::key(layer, expert);
        (0..ROWS).map(|r| self.table[r * COLS + hash(k, r)]).min().unwrap()
    }

    /// Total routing decisions recorded in the sketch.
    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut out = Vec::with_capacity(8 + 6 * 4 + 8 + self.table.len() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&(ROWS as u32).to_le_bytes());
        out.extend_from_slice(&(COLS as u32).to_le_bytes());
        out.extend_from_slice(&(self.max_layer + 1).to_le_bytes());
        out.extend_from_slice(&(self.max_expert + 1).to_le_bytes());
        out.extend_from_slice(&self.total.to_le_bytes());
        for &c in &self.table {
            out.extend_from_slice(&c.to_le_bytes());
        }
        std::fs::write(path, out)
    }

    pub fn load(path: &str) -> Result<Cms, String> {
        let b = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        let get = |p: usize, n: usize| -> Result<&[u8], String> {
            b.get(p..p + n).ok_or_else(|| format!("{}: truncated sketch file", path))
        };
        if get(0, 8)? != MAGIC {
            return Err(format!("{}: not a routing sketch (bad magic)", path));
        }
        let u32at = |p: usize| -> Result<u32, String> { Ok(u32::from_le_bytes(get(p, 4)?.try_into().unwrap())) };
        let version = u32at(8)?;
        if version != VERSION {
            return Err(format!("{}: unsupported sketch version {}", path, version));
        }
        let (rows, cols) = (u32at(12)? as usize, u32at(16)? as usize);
        if rows != ROWS || cols != COLS {
            return Err(format!("{}: sketch geometry {}x{} unsupported (this build reads {}x{})", path, rows, cols, ROWS, COLS));
        }
        let n_layers = u32at(20)?;
        let n_experts = u32at(24)?;
        let total = u64::from_le_bytes(get(28, 8)?.try_into().unwrap());
        let raw = get(36, ROWS * COLS * 4)?;
        let table: Vec<u32> = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
        Ok(Cms {
            table,
            total,
            max_layer: n_layers.saturating_sub(1),
            max_expert: n_experts.saturating_sub(1),
        })
    }
}

// ── model.rs hook + run-wide lifecycle ──
//
// Same shape as the imatrix calibration hook: an atomic gate keeps the
// no-op cost at one relaxed load per MoE layer per token; when armed, the
// sketch lives behind a mutex (routing runs on the main thread, so the lock
// is uncontended in practice).

static ACTIVE: AtomicBool = AtomicBool::new(false);
static SKETCH: Mutex<Option<Cms>> = Mutex::new(None);
static SAVE_PATH: Mutex<Option<String>> = Mutex::new(None);

/// Arms the sketch and sets the file written by `finish`.
pub fn start(path: &str) {
    *SKETCH.lock().unwrap() = Some(Cms::new());
    *SAVE_PATH.lock().unwrap() = Some(path.to_string());
    ACTIVE.store(true, Ordering::Relaxed);
}

/// Env activation for run/chat/prefill/absorb: MICROKIMI_ROUTECMS=path.bin.
pub fn start_from_env() {
    if let Ok(p) = std::env::var("MICROKIMI_ROUTECMS") {
        if !p.is_empty() {
            start(&p);
            println!("routecms: recording routing decisions, sketch saved to {} on exit (MICROKIMI_ROUTECMS)", p);
        }
    }
}

/// model.rs hook: one routing decision (one selected expert of one MoE
/// layer, one token). No-op unless the sketch is armed.
pub fn record(layer: usize, expert: u32) {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    if let Some(cms) = SKETCH.lock().unwrap().as_mut() {
        cms.add(layer as u32, expert);
    }
}

/// Writes the sketch at the end of a run (no-op when not armed).
pub fn finish() {
    if !ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    ACTIVE.store(false, Ordering::Relaxed);
    let cms = SKETCH.lock().unwrap().take();
    let path = SAVE_PATH.lock().unwrap().take();
    if let (Some(cms), Some(path)) = (cms, path) {
        match cms.save(&path) {
            Ok(()) => println!("routecms: {} routing decisions -> {} ({} x {} sketch)", cms.total, path, ROWS, COLS),
            Err(e) => eprintln!("warning: cannot write routing sketch {}: {}", path, e),
        }
    }
}

/// `microkimi cmsinfo sketch.bin`: top-50 (layer, expert, count) by count-min
/// estimate, then the coverage curve: the share of all recorded requests
/// carried by the top N% of the (layer, expert) pairs. The pair universe is
/// enumerated from the header (n_layers x n_experts); pairs never routed sit
/// at estimate 0 (or catch collision noise, hence the min-over-rows).
pub fn info_cmd(args: &[String]) {
    let Some(path) = args.get(2).filter(|a| !a.starts_with("--")) else {
        eprintln!("error: cmsinfo requires a sketch file (microkimi cmsinfo sketch.bin)");
        std::process::exit(1);
    };
    let cms = match Cms::load(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };
    let (nl, ne) = (cms.max_layer as usize + 1, cms.max_expert as usize + 1);
    println!("sketch: {} x {} u32, {} routing decisions recorded, universe {} layers x {} experts", ROWS, COLS, cms.total, nl, ne);
    let mut est: Vec<(u32, u32, u32)> = Vec::with_capacity(nl * ne); // (count, layer, expert)
    for l in 0..nl {
        for e in 0..ne {
            let c = cms.estimate(l as u32, e as u32);
            if c > 0 {
                est.push((c, l as u32, e as u32));
            }
        }
    }
    est.sort_by(|a, b| b.0.cmp(&a.0));
    let sum_est: u64 = est.iter().map(|x| x.0 as u64).sum();
    println!("distinct (layer, expert) pairs touched: {} (estimates sum to {}, recorded total {}; the gap is collision overshoot)", est.len(), sum_est, cms.total);
    println!("\ntop-50 (count-min estimates, upper bounds):");
    println!("  {:>6} {:>8} {:>10}", "layer", "expert", "count");
    for &(c, l, e) in est.iter().take(50) {
        println!("  {:>6} {:>8} {:>10}", l, e, c);
    }
    if !est.is_empty() && sum_est > 0 {
        println!("\ncoverage: share of requests carried by the top N% of touched pairs:");
        let mut cum = 0u64;
        let marks = [1usize, 5, 10, 25, 50, 100];
        let mut mi = 0;
        for (i, &(c, _, _)) in est.iter().enumerate() {
            cum += c as u64;
            let pct_pairs = 100 * (i + 1) / est.len();
            while mi < marks.len() && pct_pairs >= marks[mi] {
                println!("  top {:>3}% of pairs ({:>6} pairs) cover {:>6.2}% of requests", marks[mi], i + 1, 100.0 * cum as f64 / sum_est as f64);
                mi += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        std::env::temp_dir().join(format!("microkimi_cms_test_{}_{}.bin", std::process::id(), name)).to_string_lossy().into_owned()
    }

    #[test]
    fn exact_counts_without_collisions() {
        let mut cms = Cms::new();
        for _ in 0..7 {
            cms.add(3, 42);
        }
        for _ in 0..3 {
            cms.add(3, 41);
        }
        cms.add(90, 895);
        // count-min never underestimates; with this few live keys the odds of
        // a 4-row collision on these exact keys are negligible
        assert_eq!(cms.estimate(3, 42), 7);
        assert_eq!(cms.estimate(3, 41), 3);
        assert_eq!(cms.estimate(90, 895), 1);
        assert_eq!(cms.total, 11);
        assert!(cms.estimate(0, 0) <= 1); // untouched key: 0 or collision noise
    }

    #[test]
    fn estimate_never_below_truth() {
        let mut cms = Cms::new();
        let mut truth = std::collections::HashMap::new();
        // flood the sketch with enough keys to force collisions
        for l in 0..93u32 {
            for e in 0..896u32 {
                let n = ((l * 31 + e * 7) % 5) + 1;
                for _ in 0..n {
                    cms.add(l, e);
                }
                truth.insert((l, e), n);
            }
        }
        for ((l, e), n) in truth {
            assert!(cms.estimate(l, e) >= n, "estimate below truth at ({}, {})", l, e);
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let p = tmp("roundtrip");
        let mut cms = Cms::new();
        for i in 0..100u32 {
            cms.add(i % 93, (i * 13) % 896);
        }
        cms.save(&p).unwrap();
        let back = Cms::load(&p).unwrap();
        assert_eq!(back.table, cms.table);
        assert_eq!(back.total, 100);
        assert_eq!(back.max_layer, 92);
        assert_eq!(back.max_expert, (0..100u32).map(|i| (i * 13) % 896).max().unwrap());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn load_rejects_garbage() {
        let p = tmp("garbage");
        std::fs::write(&p, b"not a sketch at all").unwrap();
        assert!(Cms::load(&p).is_err());
        std::fs::remove_file(&p).ok();
        assert!(Cms::load("/nonexistent/sketch.bin").is_err());
    }
}
