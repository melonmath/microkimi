// HTTP via curl as a subprocess (no TLS in std, so curl is shelled out).
// Range requests against HuggingFace with redirect following (-L) and retries.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

const UA: &str = "microkimi/0.1 (pure-rust; curl-shellout)";

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
    for attempt in 0..4 {
        let mut cmd = Command::new("curl");
        cmd.arg("-sL")
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
                        std::thread::sleep(std::time::Duration::from_millis(800 * (attempt as u64 + 1)));
                        continue;
                    }
                }
                FETCHED_REQUESTS.fetch_add(1, Ordering::Relaxed);
                FETCHED_BYTES.fetch_add(out.stdout.len() as u64, Ordering::Relaxed);
                return Some(out.stdout);
            }
            Ok(out) => {
                eprintln!(
                    "  http: curl status {:?} (attempt {}): {}",
                    out.status.code(),
                    attempt + 1,
                    String::from_utf8_lossy(&out.stderr).chars().take(120).collect::<String>()
                );
            }
            Err(e) => {
                eprintln!("  http: curl not found? {} (attempt {})", e, attempt + 1);
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(800 * (attempt as u64 + 1)));
    }
    None
}
