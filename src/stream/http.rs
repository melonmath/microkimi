// HTTP via curl as a subprocess (no TLS in std, so curl is shelled out).
// Range requests against HuggingFace with redirect following (-L) and retries.
// Byte-exact pass-through: the response body is delivered untouched, the
// fetched byte/request counters are observability only.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

const UA: &str = "microkimi/0.1 (pure-rust; curl-shellout)";

// Global concurrency limit: HuggingFace rate-limits aggressive parallel range
// requests (HTTP 429), so every fetch funnels through this semaphore.
// Override with MICROKIMI_HTTP_CONCURRENCY.
static SEM_N: Mutex<usize> = Mutex::new(0);
static SEM_CV: Condvar = Condvar::new();

fn max_concurrent() -> usize {
    std::env::var("MICROKIMI_HTTP_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
}

fn acquire() {
    let mut n = SEM_N.lock().unwrap();
    while *n >= max_concurrent() {
        n = SEM_CV.wait(n).unwrap();
    }
    *n += 1;
}

fn release() {
    let mut n = SEM_N.lock().unwrap();
    *n = n.saturating_sub(1);
    SEM_CV.notify_one();
}

/// Total bytes / requests downloaded in this process (bandwidth accounting
/// for the remote slice path).
static FETCHED_BYTES: AtomicU64 = AtomicU64::new(0);
static FETCHED_REQUESTS: AtomicU64 = AtomicU64::new(0);

pub fn fetched_bytes() -> u64 {
    FETCHED_BYTES.load(Ordering::Relaxed)
}

pub fn fetched_requests() -> u64 {
    FETCHED_REQUESTS.load(Ordering::Relaxed)
}

/// Downloads `url` in full. None on failure after retries.
pub fn fetch(url: &str) -> Option<Vec<u8>> {
    fetch_range(url, None)
}

/// Downloads the inclusive byte range `start..=end`, or the whole file if None.
pub fn fetch_range(url: &str, range: Option<(u64, u64)>) -> Option<Vec<u8>> {
    acquire();
    let r = fetch_range_throttled(url, range);
    release();
    r
}

fn fetch_range_throttled(url: &str, range: Option<(u64, u64)>) -> Option<Vec<u8>> {
    for attempt in 0..8u32 {
        let mut cmd = Command::new("curl");
        cmd.arg("-sfL")
            .arg("--max-time")
            .arg("300")
            .arg("-H")
            .arg(format!("User-Agent: {}", UA));
        if let Some((s, e)) = range {
            cmd.arg("-r").arg(format!("{}-{}", s, e));
        }
        cmd.arg(url);
        match cmd.output() {
            Ok(out) if out.status.success() && !out.stdout.is_empty() => {
                // curl may return an error page; check the expected size
                if let Some((s, e)) = range {
                    let want = (e - s + 1) as usize;
                    if out.stdout.len() != want {
                        eprintln!(
                            "  http: unexpected size {}/{} (attempt {})",
                            out.stdout.len(),
                            want,
                            attempt + 1
                        );
                        backoff(attempt);
                        continue;
                    }
                }
                FETCHED_REQUESTS.fetch_add(1, Ordering::Relaxed);
                FETCHED_BYTES.fetch_add(out.stdout.len() as u64, Ordering::Relaxed);
                return Some(out.stdout);
            }
            Ok(out) => {
                // -f makes curl exit 22 on HTTP errors (429, 503, ...)
                eprintln!(
                    "  http: curl exit {:?} (attempt {})",
                    out.status.code(),
                    attempt + 1
                );
            }
            Err(e) => {
                eprintln!("  http: curl not found? {} (attempt {})", e, attempt + 1);
            }
        }
        backoff(attempt);
    }
    None
}

// Exponential backoff with jitter: 1s, 2s, 4s, ... capped at ~32s.
fn backoff(attempt: u32) {
    let base = 1000u64 << attempt.min(5);
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos() as u64 % 1000)
        .unwrap_or(0);
    std::thread::sleep(std::time::Duration::from_millis(base + jitter));
}
