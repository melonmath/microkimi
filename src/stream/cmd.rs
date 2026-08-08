// CLI diagnostics for the streaming cache: streamtest (cold/warm remote
// fetch proof, LRU budget and disk rollover checks against a real repo) and
// cache (inspect the on-disk remote cache: manifest, sizes, dates).
// Reporting only: these commands never mutate a model or a cache entry.

use super::*;
use super::fetch::{sanitize, Manifest};

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
        let net0 = crate::stream::http::fetched_bytes();
        let cached = src.tensor_bytes(name);
        let net = crate::stream::http::fetched_bytes() - net0;
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
        let net0 = crate::stream::http::fetched_bytes();
        let req0 = crate::stream::http::fetched_requests();
        let b = src.tensor_bytes(name);
        let net = crate::stream::http::fetched_bytes() - net0;
        assert_eq!(net, 0, "{}: warm fetch hit the network ({} bytes)", name, net);
        assert_eq!(crate::stream::http::fetched_requests(), req0);
        println!("  {:<48} {} bytes, network 0 B (disk cache): OK", name, b.len());
    }

    // 3) LRU budget: 3 entries fit, the 4th evicts the least recently used
    println!("-- 3) LRU eviction under a 3-entry budget");
    let entry = 1 << 20; // 1 MB synthetic entries
    let cache = ExpertCache {
        inner: Arc::new(CacheInner {
            lru: Mutex::new(Lru::new(3 * entry)),
            src: Src::Remote(src),
            pred: Mutex::new(Predictor::new()),
            inflight: (Mutex::new(std::collections::HashSet::new()), std::sync::Condvar::new()),
            shadows: None,
            tsim: Mutex::new(None),
        }),
    };
    for e in 0..4u32 {
        cache.inner.lru.lock().unwrap().insert((0, e), Arc::new(vec![(e + 1) as u8; entry]), 0, false);
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
        let net0 = crate::stream::http::fetched_bytes();
        let b = roll.tensor_bytes(experts[0]);
        let net = crate::stream::http::fetched_bytes() - net0;
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
