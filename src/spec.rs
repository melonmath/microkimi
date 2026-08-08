// spec.rs - n-gram speculative decoding (--spec N), greedy only.
//
// Proposal (prompt lookup, no draft model): after each committed token, find
// the longest suffix n-gram (2..=8 tokens) of the full context that also
// occurs EARLIER in the context, and propose the N tokens that followed the
// most recent earlier occurrence.
//
// Verification: the pending token (emitted last round, not yet ingested) +
// the N proposals are ingested in ONE batched pass (Model::prefill_all,
// per-position logits, bit-identical to sequential forwards). A proposal is
// accepted while it matches the greedy argmax of the previous position; the
// longest valid prefix is committed plus one bonus token selected at the
// divergence position (vanilla speculative decoding with an n-gram "draft").
//
// Rollback: the optimistic batch ingests tokens that may be rejected. The
// MLA caches are append-only per position but the KDA state is recurrent
// (NOT truncatable: the recurrence S += (beta k) x delta cannot un-ingest),
// so the whole layer-cache set is cloned BEFORE the batch (a few MB of
// memcpy, cheap next to a forward pass) and restored on a partial accept;
// the accepted prefix is then re-ingested with one batched prefill, which
// rebuilds the exact same state (prefill is bit-identical to n sequential
// forwards). Full accept: no restore at all.
//
// Greedy output is BIT-IDENTICAL to the non-speculative loop: every emitted
// token is the argmax (top_k_probs, same tie-breaking) of the same
// per-position logits the sequential loop would see. Sampling (temp > 0)
// declines the flag at dispatch (exact rejection sampling is future work).

use crate::model::{Model, Sampler};
use crate::tokenizer::AnyTokenizer;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// MICROKIMI_DRAFTSTATS=1 (debug): quality of the draft-aware prefetch
// predictor. After each verification pass, compare the routing recorded at
// the COMMITTED positions (the pass's real picks, just ingested) with the
// routing replayed from their source positions (the prediction):
// per position-layer |predicted ∩ actual| / |actual|, summed.
static DPRED_HIT: AtomicU64 = AtomicU64::new(0);
static DPRED_TOT: AtomicU64 = AtomicU64::new(0);

fn draft_stats_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("MICROKIMI_DRAFTSTATS").map(|v| v == "1").unwrap_or(false))
}

/// Greedy selection, same function and tie-breaking as the
/// run_turn_core_batch greedy path (required for bit-identity). With
/// --dry, the same penalty is applied over the same emitted-token context
/// the plain loop would see at this position.
fn select(logits: &[f32], sampler: &Sampler, gen_ctx: &[u32]) -> u32 {
    if sampler.dry > 0.0 {
        let mut adj = logits.to_vec();
        crate::model::apply_dry(&mut adj, gen_ctx, sampler.dry);
        return crate::model::top_k_probs(&adj, 5)[0].0 as u32;
    }
    crate::model::top_k_probs(logits, 5)[0].0 as u32
}

/// Longest suffix n-gram (2..=8) of ctx that also occurs earlier in ctx;
/// returns up to `n` tokens following the MOST RECENT earlier occurrence,
/// plus the source position `from` the proposal was lifted from (prop[j]
/// sat at from + j; consumed by the draft-aware prefetch).
/// Plain nested scan: contexts are modest (one prompt + one answer).
fn propose(ctx: &[u32], n: usize) -> (Vec<u32>, Option<(usize, usize)>) {
    for l in (2..=8usize).rev() {
        if ctx.len() <= l {
            continue;
        }
        let suffix = &ctx[ctx.len() - l..];
        for i in (0..ctx.len() - l).rev() {
            if &ctx[i..i + l] == suffix {
                let from = i + l;
                let to = (from + n).min(ctx.len());
                return (ctx[from..to].to_vec(), Some((from, to - from)));
            }
        }
    }
    (Vec::new(), None)
}

/// Most recent occurrence of the longest prefix of `prop` in ctx:
/// (start, matched length). The Rosa chain is frequency-stitched, not
/// lifted from one position, so its source occurrence is recovered by a
/// plain backward scan (contexts are modest); used by the draft-aware
/// prefetch to replay the routing recorded at those positions.
fn source_of(ctx: &[u32], prop: &[u32]) -> Option<(usize, usize)> {
    for l in (1..=prop.len()).rev() {
        if ctx.len() < l {
            continue;
        }
        for i in (0..=ctx.len() - l).rev() {
            if ctx[i..i + l] == prop[..l] {
                return Some((i, l));
            }
        }
    }
    None
}

/// Draft proposer: the plain bounded n-gram scan above, or the incremental
/// suffix automaton (--spec-rosa, src/rosa.rs) fed with every committed
/// token. Same contract: up to `n` candidate tokens, verified by the model.
enum Proposer {
    NGram,
    Rosa(crate::rosa::SuffixAutomaton),
}

impl Proposer {
    /// Up to `n` candidate tokens, plus the source occurrence they were
    /// lifted from: (start, matched length) - prop[j] sat at start + j for
    /// j < matched. None = no usable source (the prefetch skips the draft).
    fn propose(&self, ctx: &[u32], n: usize) -> (Vec<u32>, Option<(usize, usize)>) {
        match self {
            Proposer::NGram => propose(ctx, n),
            Proposer::Rosa(a) => {
                let prop = a.propose(n);
                let src = source_of(ctx, &prop);
                (prop, src)
            }
        }
    }
    /// Keeps the automaton in sync with the committed context (no-op for
    /// the n-gram proposer, which rescans ctx on every call).
    fn feed(&mut self, toks: &[u32]) {
        if let Proposer::Rosa(a) = self {
            for &t in toks {
                a.feed(t);
            }
        }
    }
}

/// Speculative generation loop (see the header comment). Mirrors
/// run_turn_core_batch's contract: prefill, generate up to max_new tokens,
/// stop on stop_id (not emitted), print the answer + tok/s line.
pub fn run_turn_spec(
    ids: &[u32],
    max_new: usize,
    tok: &AnyTokenizer,
    model: &mut Model,
    pos0: usize,
    init_logits: Option<Vec<f32>>,
    debug: bool,
    stop_id: u32,
    sampler: &mut Sampler,
) -> String {
    let k = sampler.spec.max(sampler.spec_rosa);
    let mut proposer = if sampler.spec_rosa > 0 {
        let mut a = crate::rosa::SuffixAutomaton::new();
        for &t in ids {
            a.feed(t); // the prompt is already committed context
        }
        Proposer::Rosa(a)
    } else {
        Proposer::NGram
    };
    let t0 = Instant::now();
    let mut pos = pos0;
    let mut logits = init_logits.unwrap_or_default();
    if !ids.is_empty() {
        logits = model.prefill(ids, pos);
        pos += ids.len();
    }
    if logits.is_empty() {
        eprintln!("error: nothing to continue from (empty prompt and no logits stored in the .mkmem snapshot)");
        std::process::exit(1);
    }
    let t_prefill = t0.elapsed();
    let t_gen = Instant::now();

    let mut ctx: Vec<u32> = ids.to_vec(); // committed context (text-wise)
    let mut generated: Vec<u32> = Vec::new();
    let mut pending = false; // generated.last() is emitted but not ingested
    let mut passes = 0usize; // batched verification passes
    let mut pass_tokens = 0usize; // tokens ingested through those passes
    let mut singles = 0usize; // plain greedy steps (no proposal available)
    let mut stop_hit = false;
    // adaptive gate: a pass that commits nothing beyond the pending token is
    // pure overhead (snapshot + batch + rollback for zero gain). After
    // STRIKES consecutive fruitless passes, fall back to COOL plain greedy
    // steps, then retry (repetition often appears only later in the answer).
    const STRIKES: usize = 1;
    const COOL: usize = 48;
    let mut strikes = 0usize;
    let mut cool = 0usize;

    while generated.len() < max_new && !stop_hit {
        let (prop, src) = if cool > 0 { (Vec::new(), None) } else { proposer.propose(&ctx, k) };
        if prop.is_empty() {
            // no earlier n-gram to continue from (or cooldown): plain step.
            // With a pending token, just ingest it (it was selected by
            // `logits` last round); otherwise select + ingest + emit.
            if pending {
                logits = model.prefill(&[*generated.last().unwrap()], pos);
                pos += 1;
                pending = false;
            } else {
                let sel = select(&logits, sampler, &generated);
                if sel == stop_id {
                    break;
                }
                logits = model.prefill(&[sel], pos);
                pos += 1;
                ctx.push(sel);
                proposer.feed(&[sel]);
                generated.push(sel);
            }
            cool = cool.saturating_sub(1);
            singles += 1;
            continue;
        }
        // optimistic batch: pending token (if any) + the proposals
        let snap = model.caches.clone();
        let skip = pending as usize; // batch[0] was already emitted last round
        let mut batch: Vec<u32> = Vec::with_capacity(k + 1);
        if pending {
            batch.push(*generated.last().unwrap());
        }
        batch.extend_from_slice(&prop);
        // draft-aware prefetch: the pending token continues the source
        // occurrence (position from - 1), prop[j] sat at from + j. Replay
        // the routing recorded at those positions to background-fetch the
        // experts the verification pass will pull (--stream only,
        // MICROKIMI_DRAFTPREFETCH=0 disables; output-neutral)
        let srcs: Vec<Option<usize>> = match src {
            Some((from, matched)) => {
                let mut v: Vec<Option<usize>> = Vec::with_capacity(batch.len());
                if pending {
                    v.push(from.checked_sub(1).filter(|_| matched > 0));
                }
                for j in 0..prop.len() {
                    v.push(if j < matched { Some(from + j) } else { None });
                }
                v
            }
            None => Vec::new(),
        };
        model.draft_prefetch(&batch, &srcs);
        let pos_batch = pos; // position of batch[0] (for the draft stats)
        let g = model.prefill_all(&batch, pos);
        passes += 1;
        // verification: accept while the argmax matches the proposal
        let mut committed = 0usize;
        let mut next_sel: Option<u32> = None;
        let mut vctx: Vec<u32> = Vec::new(); // --dry context, reused per position
        for i in 0..batch.len() {
            let prev = if i == 0 { &logits } else { &g[i - 1] };
            let sel = if sampler.dry > 0.0 {
                // tokens emitted before batch[i]: generated + batch[skip..i]
                vctx.clear();
                vctx.extend_from_slice(&generated);
                vctx.extend_from_slice(&batch[skip..i]);
                select(prev, sampler, &vctx)
            } else {
                select(prev, sampler, &[])
            };
            if sel == stop_id {
                stop_hit = true;
                break;
            }
            if sel != batch[i] {
                next_sel = Some(sel);
                break;
            }
            committed = i + 1;
        }
        if !stop_hit && next_sel.is_none() {
            let sel = if sampler.dry > 0.0 {
                vctx.clear();
                vctx.extend_from_slice(&generated);
                vctx.extend_from_slice(&batch[skip..]);
                select(g.last().unwrap(), sampler, &vctx)
            } else {
                select(g.last().unwrap(), sampler, &[])
            };
            if sel == stop_id {
                stop_hit = true;
            } else {
                next_sel = Some(sel);
            }
        }
        if debug {
            println!(
                "  spec pass {}: proposed {:?}, accepted {}, next {}",
                passes,
                prop.iter().map(|&t| tok.decode_id(t)).collect::<Vec<_>>(),
                committed - pending as usize,
                next_sel.map(|t| tok.decode_id(t)).unwrap_or_else(|| "<stop>".to_string())
            );
        }
        // rollback: restore the pre-batch caches, re-ingest the accepted
        // prefix in one batched prefill (bit-identical state)
        if committed < batch.len() {
            model.caches = snap;
            if committed > 0 {
                model.prefill(&batch[..committed], pos);
            }
        }
        pos += committed;
        pass_tokens += committed;
        // draft prefetch predictor stats (MICROKIMI_DRAFTSTATS=1): the
        // committed positions were just re-ingested, so their recorded
        // routing is the pass's REAL picks; score the replayed source
        // routing against them
        if draft_stats_on() {
            for j in 0..committed {
                let Some(Some(sp)) = srcs.get(j) else { continue };
                let (Some(pred), Some(act)) = (crate::stream::route_lookup(*sp as u32), crate::stream::route_lookup((pos_batch + j) as u32)) else { continue };
                let mut hit = 0u64;
                let mut tot = 0u64;
                for (l, ae) in &act {
                    tot += ae.len() as u64;
                    if let Some((_, pe)) = pred.iter().find(|(pl, _)| pl == l) {
                        hit += ae.iter().filter(|e| pe.contains(e)).count() as u64;
                    }
                }
                DPRED_HIT.fetch_add(hit, Ordering::Relaxed);
                DPRED_TOT.fetch_add(tot, Ordering::Relaxed);
            }
        }
        // a pass is fruitful only if it accepted at least one PROPOSED token
        // (beyond the pending one it would have ingested anyway)
        if committed == pending as usize {
            strikes += 1;
            if strikes >= STRIKES {
                cool = COOL;
                strikes = 0;
            }
        } else {
            strikes = 0;
        }
        ctx.extend_from_slice(&batch[..committed]);
        proposer.feed(&batch[..committed]);
        if committed > 0 {
            logits = g[committed - 1].clone();
        }
        // emission: accepted proposals are new, plus the divergence token
        let n_acc = committed - skip;
        let mut newtoks: Vec<u32> = batch[skip..committed].to_vec();
        if let Some(s) = next_sel {
            newtoks.push(s);
        }
        let room = max_new - generated.len();
        let clamped = newtoks.len() > room;
        newtoks.truncate(room);
        ctx.extend_from_slice(&newtoks[n_acc.min(newtoks.len())..]); // the pending tail
        proposer.feed(&newtoks[n_acc.min(newtoks.len())..]);
        generated.extend_from_slice(&newtoks);
        pending = !clamped && !stop_hit && next_sel.is_some();
        if clamped {
            break;
        }
    }

    let gen_dt = t_gen.elapsed().as_secs_f64();
    let answer = tok.decode(&generated);
    if debug {
        println!();
        println!("{}", "=".repeat(64));
        println!("SUMMARY");
        println!("{}", "=".repeat(64));
        println!("answer: {}", answer);
    } else {
        println!("Bot > {}", answer);
    }
    if !generated.is_empty() {
        let moy = gen_dt / generated.len() as f64;
        if debug {
            println!(
                "prefill: {:.2} s  |  generation: {:.0} ms/token average ({:.1} tok/s)",
                t_prefill.as_secs_f64(),
                moy * 1000.0,
                1.0 / moy
            );
        } else {
            println!("  ({:.0} ms/token, {:.1} tok/s)", moy * 1000.0, 1.0 / moy);
        }
    }
    println!(
        "  spec: {} tokens in {} batched passes ({:.2} tokens/pass) + {} single steps",
        pass_tokens,
        passes,
        pass_tokens as f64 / passes.max(1) as f64,
        singles
    );
    if draft_stats_on() {
        let (hit, tot) = (DPRED_HIT.load(Ordering::Relaxed), DPRED_TOT.load(Ordering::Relaxed));
        if tot > 0 {
            println!(
                "  draft-prefetch predictor: {}/{} of the verification picks were in the replayed source routing ({:.0}%)",
                hit,
                tot,
                100.0 * hit as f64 / tot as f64
            );
        }
    }
    answer
}
