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

/// One lens row: (layer index, attention kind, top-5 (token id, softmax prob)).
type LensRows = Vec<(usize, &'static str, Vec<(usize, f32)>)>;
static LENS_PENDING: std::sync::Mutex<Option<LensRows>> = std::sync::Mutex::new(None);

/// Projects every captured post-layer hidden through norm_f + lm_head (the
/// same rmsnorm/matvec pair as the normal path) and stashes the top-5 rows
/// for logit_lens_print_maybe. The last layer reuses the real logits: the
/// output residual mix (out_res_w) sits between it and the final norm, so
/// this keeps the bottom row bit-identical to the model's own candidates.
pub(super) fn logit_lens_compute(cfg: &Config, lm_head: &[f32], norm_f: &[f32], per_layer: &[(usize, &'static str, Vec<f32>)], final_logits: &[f32]) {
    let mut rows: LensRows = Vec::with_capacity(per_layer.len());
    let mut xf = vec![0f32; cfg.d];
    let mut logits = vec![0f32; cfg.vocab];
    for (i, (l, kind, h)) in per_layer.iter().enumerate() {
        if i + 1 == per_layer.len() {
            rows.push((*l, kind, top_k_probs(final_logits, 5)));
        } else {
            rmsnorm(cfg, h, norm_f, &mut xf);
            matvec(lm_head, cfg.vocab, cfg.d, &xf, &mut logits);
            rows.push((*l, kind, top_k_probs(&logits, 5)));
        }
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
    for (l, kind, top) in rows {
        let segs: Vec<String> = top.iter().map(|&(id, p)| format!("{} {:.1}%", py_repr(&tok.decode_id(id as u32)), p * 100.0)).collect();
        println!("  layer {:>2} ({}): {}", l, kind, segs.join("  "));
    }
}

// ── full forward ──
