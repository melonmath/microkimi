// Diagnostics: parity dump recorder (PARITY thread-local, fed by the layer
// code), --dump-hidden per-layer RMS report, --logit-lens per-layer logit
// projection, routing debug counters (ROUTING). Observability only:
// nothing in here feeds back into the computed logits.

use super::*;

#[derive(Default)]
pub struct ParityDump {
    pub hiddens: std::collections::HashMap<(usize, usize), Vec<f32>>, // (pos, layer)
    pub l1_attn: std::collections::HashMap<usize, Vec<f32>>,
    pub l1_routed: std::collections::HashMap<usize, Vec<f32>>,
    pub l1_shared: std::collections::HashMap<usize, Vec<f32>>,
    pub router: std::collections::HashMap<(usize, usize), Vec<u32>>, // (pos, layer) sorted top-16
}

pub const DUMP_LAYERS: [usize; 7] = [0, 1, 3, 4, 12, 47, 92];
pub const ROUTER_LAYERS: [usize; 3] = [1, 47, 92];

thread_local! {
    pub static PARITY: std::cell::RefCell<Option<ParityDump>> = std::cell::RefCell::new(None);
}

// ── --debug-routing collection (thread-local, inactive by default) ──

#[derive(Default)]
pub struct RoutingDebug {
    pub cur: Vec<(usize, Vec<(u32, f32)>)>, // layer → top-3 (expert, renormalized weight)
    pub counts: std::collections::HashMap<(usize, u32), u32>, // (layer, expert) → times in top-16
}

thread_local! {
    pub static ROUTING: std::cell::RefCell<Option<RoutingDebug>> = std::cell::RefCell::new(None);
}

pub(super) fn parity_rec(f: impl FnOnce(&mut ParityDump)) {
    PARITY.with(|p| {
        if let Some(d) = p.borrow_mut().as_mut() {
            f(d);
        }
    });
}

// ── --dump-hidden collection (inactive by default) ──
//
// Diagnostic instrument for pruned-model collapse: the RMS of the hidden
// state after each layer, printed ONCE at the end of the first prefill (or
// the first generated token when the prefill is empty). Read-only: without
// the flag the forward paths are untouched (bit-exact).
pub static DUMP_HIDDEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static DUMP_HIDDEN_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_dump_hidden(on: bool) {
    DUMP_HIDDEN.store(on, std::sync::atomic::Ordering::Relaxed);
    DUMP_HIDDEN_DONE.store(false, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn dump_hidden_on() -> bool {
    use std::sync::atomic::Ordering;
    DUMP_HIDDEN.load(Ordering::Relaxed) && !DUMP_HIDDEN_DONE.load(Ordering::Relaxed)
}

/// RMS (sqrt of the mean of squares): the same norm rmsnorm rescales to ~1.
pub(super) fn vec_rms(v: &[f32]) -> f64 {
    (v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / v.len().max(1) as f64).sqrt()
}

/// Prints the per-layer table once, then disarms (subsequent tokens of the
/// same run are not re-dumped).
pub(super) fn dump_hidden_print(per_layer: &[(usize, &'static str, f64)], residual_rms: f64, logits: &[f32]) {
    let mean = logits.iter().map(|&x| x as f64).sum::<f64>() / logits.len().max(1) as f64;
    let std = (logits.iter().map(|&x| (x as f64 - mean) * (x as f64 - mean)).sum::<f64>() / logits.len().max(1) as f64).sqrt();
    println!("── dump-hidden: rms of the hidden state after each layer ──");
    for (l, kind, rms) in per_layer {
        println!("  layer {:>2} ({}): rms={:.4}", l, kind, rms);
    }
    println!("  final residual: rms={:.4}", residual_rms);
    println!("  logits: std={:.4} mean={:.4}", std, mean);
    DUMP_HIDDEN_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
}

// ── --logit-lens collection (inactive by default) ──
//
// Logit lens: each post-layer hidden state is projected through the FINAL
// norm + lm_head and the top-5 softmax tokens are printed, showing where in
// depth the next-token semantics emerge (or die). Read-only diagnostic: the
// captured hiddens are clones, the forward math is untouched (bit-exact).
pub static LOGIT_LENS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOGIT_LENS_ALL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LOGIT_LENS_DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// on: lens active at the last prefill position. all: also on every
/// generated token (--logit-lens-all).
pub fn set_logit_lens(on: bool, all: bool) {
    LOGIT_LENS.store(on, std::sync::atomic::Ordering::Relaxed);
    LOGIT_LENS_ALL.store(all, std::sync::atomic::Ordering::Relaxed);
    LOGIT_LENS_DONE.store(false, std::sync::atomic::Ordering::Relaxed);
}

pub(super) fn logit_lens_on() -> bool {
    use std::sync::atomic::Ordering;
    LOGIT_LENS.load(Ordering::Relaxed) && (LOGIT_LENS_ALL.load(Ordering::Relaxed) || !LOGIT_LENS_DONE.load(Ordering::Relaxed))
}

/// One probe measurement on a lens row: (token id, 1-based rank by logit,
/// softmax prob).
type ProbeStat = (u32, usize, f32);

/// One lens row: (layer index, attention kind, top-5 (token id, softmax
/// prob), per-probe stats).
type LensRows = Vec<(usize, &'static str, Vec<(usize, f32)>, Vec<ProbeStat>)>;
static LENS_PENDING: std::sync::Mutex<Option<LensRows>> = std::sync::Mutex::new(None);

/// --lens-probe token ids (main resolves the strings against the vocab at
/// startup, resolve_lens_probe). Empty = the historical top-5-only report.
static LENS_PROBES: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

/// Arms the per-row probe columns of the lens report (--lens-probe).
pub fn set_lens_probes(ids: Vec<u32>) {
    *LENS_PROBES.lock().unwrap() = ids;
}

/// The lens readout projection: post-layer hidden -> rmsnorm(norm_f) -> f32
/// lm_head matvec. Shared by the --logit-lens rows and the --exit-layer
/// truncated forward, so `--exit-layer K` greedy picks exactly the lens row K
/// top-1. Always the f32 matvec, never the q8 head: bit-matched to the lens.
pub(super) fn lens_project(cfg: &Config, lm_head: &[f32], norm_f: &[f32], h: &[f32], logits: &mut [f32]) {
    let mut xf = vec![0f32; cfg.d];
    rmsnorm(cfg, h, norm_f, &mut xf);
    matvec(lm_head, cfg.vocab, cfg.d, &xf, logits);
}

/// (1-based rank by logit, softmax prob) of token `probe` over `logits`.
/// Strict-greater count: exact logit ties share the better rank.
fn probe_rank_prob(logits: &[f32], probe: u32) -> ProbeStat {
    let lp = logits[probe as usize];
    let rank = 1 + logits.iter().filter(|&&l| l > lp).count();
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let z: f32 = logits.iter().map(|&l| (l - m).exp()).sum();
    (probe, rank, (lp - m).exp() / z)
}

/// Projects every captured post-layer hidden through the lens readout
/// (lens_project) and stashes the top-5 rows for logit_lens_print_maybe, with
/// the rank/prob of every --lens-probe token when probes are armed. The last
/// layer reuses the real logits: the output residual mix (out_res_w) sits
/// between it and the final norm, so this keeps the bottom row bit-identical
/// to the model's own candidates.
pub(super) fn logit_lens_compute(cfg: &Config, lm_head: &[f32], norm_f: &[f32], per_layer: &[(usize, &'static str, Vec<f32>)], final_logits: &[f32]) {
    let probes = LENS_PROBES.lock().unwrap().clone();
    let mut rows: LensRows = Vec::with_capacity(per_layer.len());
    let mut logits = vec![0f32; cfg.vocab];
    for (i, (l, kind, h)) in per_layer.iter().enumerate() {
        let row_logits = if i + 1 == per_layer.len() {
            final_logits
        } else {
            lens_project(cfg, lm_head, norm_f, h, &mut logits);
            &logits
        };
        let stats = probes.iter().map(|&p| probe_rank_prob(row_logits, p)).collect();
        rows.push((*l, kind, top_k_probs(row_logits, 5), stats));
    }
    LOGIT_LENS_DONE.store(true, std::sync::atomic::Ordering::Relaxed);
    *LENS_PENDING.lock().unwrap() = Some(rows);
}

/// Prints the pending lens table (called by the generation loop, which owns
/// the tokenizer, right after each forward/prefill). No-op without a dump.
pub fn logit_lens_print_maybe(tok: &AnyTokenizer, label: &str) {
    let rows = LENS_PENDING.lock().unwrap().take();
    let Some(rows) = rows else { return };
    println!("── logit lens ({}): top-5 of each layer through final norm + lm_head ──", label);
    for (l, kind, top, probes) in rows {
        let segs: Vec<String> = top.iter().map(|&(id, p)| format!("{} {:.1}%", py_repr(&tok.decode_id(id as u32)), p * 100.0)).collect();
        println!("  layer {:>2} ({}): {}", l, kind, segs.join("  "));
        for (id, rank, p) in probes {
            println!("      probe {}: p = {:.1}%  rank {}", py_repr(&tok.decode_id(id)), p * 100.0, rank);
        }
    }
}

/// Test-only drain of the pending lens rows (the rows logit_lens_print_maybe
/// would print).
#[cfg(test)]
pub(crate) fn lens_rows_take() -> Option<LensRows> {
    LENS_PENDING.lock().unwrap().take()
}

/// Resolves one --lens-probe string to a token id: the string must be exactly
/// one vocab entry as decode_id renders it (leading space included). An
/// absent entry is a hard error listing the near matches, so a missed or
/// extra leading space is easy to spot.
pub fn resolve_lens_probe(tok: &AnyTokenizer, s: &str) -> Result<u32, String> {
    resolve_probe_scan(tok.vocab_size(), |id| tok.decode_id(id), s)
}

/// Scan-based core of resolve_lens_probe: one pass over the vocab, runs once
/// at startup. Generic over the decode function for the unit tests.
fn resolve_probe_scan(vocab: usize, decode: impl Fn(u32) -> String, s: &str) -> Result<u32, String> {
    let key = s.trim_start();
    let mut near: Vec<String> = Vec::new();
    for id in 0..vocab as u32 {
        let d = decode(id);
        if d == s {
            return Ok(id);
        }
        if near.len() < 8 && !key.is_empty() && (d.contains(s) || d.contains(key)) && !near.contains(&d) {
            near.push(d);
        }
    }
    let mut msg = format!("--lens-probe {}: no vocab entry is exactly this string", py_repr(s));
    if !near.is_empty() {
        let list: Vec<String> = near.iter().map(|d| py_repr(d)).collect();
        msg.push_str(&format!(" (near matches: {})", list.join(", ")));
    }
    Err(msg)
}

// ── full forward ──

#[cfg(test)]
mod probe_tests {
    use super::{probe_rank_prob, resolve_probe_scan};

    /// Rank is the strict-greater count + 1; the prob is the softmax one.
    #[test]
    fn probe_rank_prob_basics() {
        let logits = [1.0f32, 4.0, -2.0, 4.0, 0.5];
        let (id, rank, p) = probe_rank_prob(&logits, 2);
        assert_eq!((id, rank), (2, 5)); // the four other entries are all above -2
        let (_, rank_top, p_top) = probe_rank_prob(&logits, 1);
        assert_eq!(rank_top, 1); // tie at the max shares rank 1
        let (_, rank_tie, _) = probe_rank_prob(&logits, 3);
        assert_eq!(rank_tie, 1);
        assert!(p > 0.0 && p < 1.0 && p_top > p);
        let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let z: f32 = logits.iter().map(|&l| (l - m).exp()).sum();
        assert_eq!(p_top, (4.0f32 - m).exp() / z);
    }

    /// Exact string match wins; an absent probe errors and lists the entries
    /// containing it (trimmed or not), e.g. a missed leading space.
    #[test]
    fn resolve_probe_scan_match_and_near() {
        let vocab = ["hello", " France", "France", " world", "France."];
        let dec = |id: u32| vocab[id as usize].to_string();
        assert_eq!(resolve_probe_scan(vocab.len(), dec, " France"), Ok(1));
        assert_eq!(resolve_probe_scan(vocab.len(), dec, "hello"), Ok(0));
        // "France" IS a vocab entry: exact match, not a near-match report
        assert_eq!(resolve_probe_scan(vocab.len(), dec, "France"), Ok(2));
        let err = resolve_probe_scan(vocab.len(), dec, "Fran").unwrap_err();
        assert!(err.contains("no vocab entry"), "{}", err);
        assert!(err.contains("' France'") && err.contains("'France'"), "{}", err);
        let err = resolve_probe_scan(vocab.len(), dec, "Franc").unwrap_err();
        assert!(err.contains("'France.'"), "{}", err);
        // nothing related: no near matches, still a clean error
        let err = resolve_probe_scan(vocab.len(), dec, "zzzz").unwrap_err();
        assert!(err.contains("no vocab entry") && !err.contains("near matches"), "{}", err);
    }
}
