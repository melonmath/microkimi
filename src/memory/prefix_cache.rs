// prefix_cache.rs - prefix cache: .mkmem state snapshots keyed by token prefix.
//
// A K3 state after N tokens is fully described by the fixed-size KDA states,
// the short conv windows and the MLA cache (see src/memory/memory_pack.rs), so the state
// after ANY prefix of the conversation can be cached and reused: re-running
// the same chat (or continuing one) only has to prefill the tokens that were
// not covered by a previous turn.
//
// One .pck file per cached prefix, in `<model>.pck/` next to the model
// (override: MICROKIMI_PCK_DIR):
//   magic    : 8 bytes "MKPCK001"
//   n_tokens : u32, then n_tokens x u32 token ids (the covered prefix)
//   payload  : a standard MKMEM001 image (src/memory/memory_pack.rs), model fingerprint
//              included - a snapshot taken with a different model fails the
//              fingerprint check on load and degrades to a plain miss.
// The file name is the FNV-1a hash of the prefix; the stored token list is
// always compared in full before a hit is accepted (the hash is only a name).
//
// Matching rule (strict invalidation): an entry covering k tokens can only be
// used when the first k tokens of the new prompt are identical, token for
// token - the snapshot is the state after EXACTLY those k tokens. Among the
// matching entries the longest prefix wins. The resume then restores the
// state and only prefills the suffix, exactly like the --memory path
// (positions: K3 has no positional encoding, the position counter is seeded
// from the restored cache length like run_turn_resume does).
//
// Bit-identity: resuming must produce exactly the same tokens as a full
// prefill. The state round trip is exact (KDA caches are f32; the q8 MLA
// cache dequantizes to f32 on save and requantizes on load, which lands back
// on the same q8 grid), and every prefill step is per-position except the
// chunked KDA recurrence, which reassociates per 64-token chunk - so the pck
// chat pins the sequential KDA loop (kda_chunk::force_sequential), whose
// per-position operations do not depend on how the sequence is split.
// (Caveat: MICROKIMI_KV_HADAMARD=1 rotates the q8 cache through an f32
// Hadamard on save/load; that opt-in mode is not bit-exact through a snapshot.)
//
// Only model state is ever stored: the sampler, its RNG and the speculative
// proposers live outside the snapshot, so nothing leaks between sessions.
// With --spec/--spec-rosa the pck is bypassed (the verify passes carry their
// own batched state; the historical path is kept untouched).
//
// The cache is a best-effort optimization: any I/O or parse problem degrades
// to a plain full prefill, never to an error.

use crate::model::{Model, Sampler};
use crate::tokenizer::AnyTokenizer;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"MKPCK001";

/// Keep at most this many entries per cache directory (oldest evicted).
const MAX_ENTRIES: usize = 64;

pub struct Pck {
    dir: String,
}

/// False when MICROKIMI_NO_PCK=1 (prefix cache disabled).
fn env_enabled() -> bool {
    std::env::var("MICROKIMI_NO_PCK").map(|v| v != "1").unwrap_or(true)
}

/// Cache directory for a model: MICROKIMI_PCK_DIR when set, otherwise the
/// `<model>.pck/` sidecar directory (same convention as `<model>.shadows`).
pub fn cache_dir(model_path: &str) -> String {
    if let Ok(d) = std::env::var("MICROKIMI_PCK_DIR") {
        if !d.is_empty() {
            return d;
        }
    }
    format!("{}.pck", model_path)
}

/// Opens the cache for a model, None when disabled or the directory cannot
/// be created (read-only model directory: the chat then runs uncached).
pub fn open(model_path: &str) -> Option<Pck> {
    if !env_enabled() {
        return None;
    }
    let dir = cache_dir(model_path);
    match std::fs::create_dir_all(&dir) {
        Ok(()) => Some(Pck { dir }),
        Err(e) => {
            eprintln!("pck: {} not usable ({}), prefix cache disabled", dir, e);
            None
        }
    }
}

/// FNV-1a 64 over the token ids: the entry file name (the authoritative
/// comparison is the stored token list, the hash is only a lookup name).
fn fnv64(tokens: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Writes one entry (header + mkmem payload), atomically (tmp + rename).
fn write_entry(dir: &str, tokens: &[u32], payload: &[u8]) {
    let mut body = Vec::with_capacity(12 + tokens.len() * 4 + payload.len());
    body.extend_from_slice(MAGIC);
    body.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for &t in tokens {
        body.extend_from_slice(&t.to_le_bytes());
    }
    body.extend_from_slice(payload);
    let name = format!("{}/{:016x}.pck", dir, fnv64(tokens));
    let tmp = format!("{}.tmp{}", name, std::process::id());
    if std::fs::write(&tmp, &body).is_ok() {
        std::fs::rename(&tmp, &name).ok();
    }
}

/// Reads the header of one entry: the covered prefix, None on any corruption.
fn read_header(path: &std::path::Path) -> Option<Vec<u32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut head = [0u8; 12];
    f.read_exact(&mut head).ok()?;
    if &head[..8] != MAGIC {
        return None;
    }
    let n = u32::from_le_bytes(head[8..12].try_into().unwrap()) as usize;
    if n == 0 || n > 1_000_000 {
        return None;
    }
    let mut raw = vec![0u8; n * 4];
    f.read_exact(&mut raw).ok()?;
    Some(raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
}

impl Pck {
    /// Longest cached prefix of `ids`: (covered length k, mkmem payload).
    /// An entry only matches when ALL its k tokens equal ids[..k].
    pub fn lookup(&self, ids: &[u32]) -> Option<(usize, Vec<u8>)> {
        let mut best: Option<(usize, std::path::PathBuf)> = None;
        for e in std::fs::read_dir(&self.dir).ok()?.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("pck") {
                continue;
            }
            let Some(tokens) = read_header(&p) else { continue };
            let k = tokens.len();
            if k <= ids.len() && tokens == ids[..k] && best.as_ref().map(|(bk, _)| k > *bk).unwrap_or(true) {
                best = Some((k, p));
            }
        }
        let (k, p) = best?;
        let body = std::fs::read(&p).ok()?;
        Some((k, body[12 + k * 4..].to_vec()))
    }

    /// Snapshots the current model state as the entry covering `tokens`.
    pub fn store(&self, model: &Model, tokens: &[u32], logits: &[f32]) {
        write_entry(&self.dir, tokens, &crate::memory::memory_pack::serialize(model, logits));
        self.evict();
    }

    /// Keeps the newest MAX_ENTRIES files (by modification time).
    fn evict(&self) {
        let mut files: Vec<(std::path::PathBuf, std::time::SystemTime)> = std::fs::read_dir(&self.dir)
            .map(|rd| {
                rd.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("pck"))
                    .map(|p| {
                        let t = std::fs::metadata(&p).and_then(|m| m.modified()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);
                        (p, t)
                    })
                    .collect()
            })
            .unwrap_or_default();
        if files.len() > MAX_ENTRIES {
            files.sort_by_key(|(_, t)| *t);
            for (p, _) in files.iter().take(files.len() - MAX_ENTRIES) {
                std::fs::remove_file(p).ok();
            }
        }
    }
}

/// One chat turn through the prefix cache. `pck == None` (or an active
/// speculative proposer) falls back to the historical full-prefill turn.
///
/// On a hit the restored state covers the first k tokens of `ids`, only the
/// suffix is prefilled, and the post-prefill state (covering exactly `ids`)
/// is stored back for the next turn / the next session. Timings go to stderr
/// so stdout stays strictly the conversation.
#[allow(clippy::too_many_arguments)]
pub fn run_turn_chat(pck: Option<&Pck>, ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug_routing: bool, stop_id: u32, sampler: &mut Sampler) -> String {
    let bypass = sampler.spec > 0 || sampler.spec_rosa > 0;
    let Some(pck) = pck.filter(|_| !bypass && !ids.is_empty()) else {
        return crate::model::run_turn(ids, max_new, tok, model, false, debug_routing, stop_id, sampler);
    };
    model.prof = crate::model::Prof::default();
    let t0 = Instant::now();
    model.reset_cache();
    let mut k = 0usize;
    let mut init = None;
    if let Some((nk, blob)) = pck.lookup(ids) {
        match crate::memory::memory_pack::load_slice(model, &blob, "pck entry") {
            Ok(l) => {
                k = nk;
                init = Some(l);
            }
            Err(e) => {
                // corrupt or foreign-model entry: plain full prefill
                eprintln!("pck: ignoring unusable entry ({})", e);
                model.reset_cache();
            }
        }
    }
    let suffix = &ids[k..];
    let mut pos = k;
    let logits = if suffix.is_empty() {
        // the snapshot covers the whole prompt: pure continuation
        init.expect("a full-length pck hit carries its logits")
    } else {
        let l = model.prefill(suffix, pos);
        pos += suffix.len();
        l
    };
    eprintln!(
        "pck: {}/{} prompt tokens from cache, prefilled {} in {:.1?}",
        k,
        ids.len(),
        suffix.len(),
        t0.elapsed()
    );
    // snapshot the post-prefill state: it covers exactly `ids` (skipped on an
    // exact hit, which would rewrite the entry it just read)
    if k < ids.len() {
        pck.store(model, ids, &logits);
    }
    let answer = crate::model::run_turn_core_batch(
        &[],
        max_new,
        tok,
        &mut |batch: &[u32]| {
            let l = model.prefill(batch, pos);
            pos += batch.len();
            l
        },
        false,
        debug_routing,
        stop_id,
        Some(logits),
        sampler,
    );
    model.prof.print_cfg(&model.cfg);
    answer
}

/// `microkimi pck --info` / `microkimi pck --clean [--model X.bin]`.
pub fn cmd(args: &[String]) {
    let info = args.iter().any(|a| a == "--info");
    let clean = args.iter().any(|a| a == "--clean");
    if info == clean {
        eprintln!("usage: microkimi pck --info [--model X.bin]");
        eprintln!("       microkimi pck --clean [--model X.bin]");
        std::process::exit(1);
    }
    let mp = args.iter().position(|a| a == "--model").and_then(|i| args.get(i + 1)).cloned();
    let dir = match mp {
        Some(mp) => cache_dir(&mp),
        None if std::env::var("MICROKIMI_PCK_DIR").map(|d| !d.is_empty()).unwrap_or(false) => cache_dir(""),
        None => cache_dir(&crate::bin_path()),
    };
    let mut entries: Vec<(String, usize, u64)> = Vec::new(); // (name, tokens, bytes)
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("pck") {
                continue;
            }
            let bytes = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            let ntok = read_header(&p).map(|t| t.len()).unwrap_or(0);
            entries.push((e.file_name().to_string_lossy().into_owned(), ntok, bytes));
        }
    }
    entries.sort();
    if info {
        if entries.is_empty() {
            println!("pck: no entries in {}", dir);
            return;
        }
        let total: u64 = entries.iter().map(|e| e.2).sum();
        println!("pck: {} entries in {} ({:.1} KB total)", entries.len(), dir, total as f64 / 1024.0);
        for (name, ntok, bytes) in &entries {
            println!("  {} : {} tokens, {:.1} KB", name, ntok, *bytes as f64 / 1024.0);
        }
        return;
    }
    // --clean
    let freed: u64 = entries.iter().map(|e| e.2).sum();
    let n = entries.len();
    for (name, _, _) in &entries {
        std::fs::remove_file(format!("{}/{}", dir, name)).ok();
    }
    std::fs::remove_dir(&dir).ok(); // only succeeds when empty
    println!("pck: removed {} entries from {} ({:.1} KB freed)", n, dir, freed as f64 / 1024.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("microkimi_pck_test_{}_{}", std::process::id(), name));
        let d = d.to_string_lossy().into_owned();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn entry(dir: &str, tokens: &[u32]) {
        write_entry(dir, tokens, b"PAYLOAD");
    }

    #[test]
    fn longest_matching_prefix_wins() {
        let d = tmpdir("longest");
        entry(&d, &[1, 2, 3]);
        entry(&d, &[1, 2, 3, 4, 5]);
        entry(&d, &[1, 9]);
        let p = Pck { dir: d.clone() };
        let (k, blob) = p.lookup(&[1, 2, 3, 4, 5, 6, 7]).unwrap();
        assert_eq!(k, 5);
        assert_eq!(blob, b"PAYLOAD");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn strict_invalidation_on_divergence() {
        let d = tmpdir("diverge");
        entry(&d, &[1, 2, 3, 4]);
        let p = Pck { dir: d.clone() };
        // one token differs inside the covered prefix: no partial credit
        assert!(p.lookup(&[1, 2, 9, 4, 5]).is_none());
        // a shorter prompt than the entry cannot use it either
        assert!(p.lookup(&[1, 2, 3]).is_none());
        // exact coverage matches
        assert_eq!(p.lookup(&[1, 2, 3, 4]).unwrap().0, 4);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn corrupt_entries_are_skipped() {
        let d = tmpdir("corrupt");
        entry(&d, &[7, 7]);
        std::fs::write(format!("{}/deadbeefdeadbeef.pck", d), b"garbage").unwrap();
        let p = Pck { dir: d.clone() };
        assert_eq!(p.lookup(&[7, 7, 8]).unwrap().0, 2);
        std::fs::remove_dir_all(&d).ok();
    }
}
