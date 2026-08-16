// Decode loop: Sampler (greedy or top-p with temperature), XorShift RNG,
// DRY repetition penalty, run_turn* drivers over Model::forward (single turn,
// resumed with initial logits, or core over an arbitrary fwd closure).
// Owns token emission, stop handling and tok/s reporting.

use super::*;

pub(crate) fn top_k_probs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut z = 0f32;
    for &l in logits {
        z += (l - m).exp();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp() / z;
        if top.len() < k {
            top.push((i, p));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        } else if p > top[k - 1].1 {
            top[k - 1] = (i, p);
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
    }
    top
}

// ── sampling: temperature + top-p nucleus, xorshift64* RNG (tools/build.rs style) ──

/// xorshift64* RNG, same generator style as build::Rng, seedable via --seed
/// for reproducible sampling (same seed + same prompt = same output).
pub struct XorShift(u64);

impl XorShift {
    pub fn new(seed: u64) -> XorShift {
        XorShift(seed | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Decoding policy: temp <= 0 keeps the exact greedy argmax path; temp > 0
/// samples from softmax(logits / temp) restricted to the top-p nucleus.
/// spec > 0 enables n-gram speculative decoding (src/model/spec.rs, greedy only);
/// spec_rosa > 0 swaps the proposer for the suffix automaton (src/model/rosa.rs).
/// dry > 0 subtracts a DRY-style anti-repetition penalty from the logits
/// (apply_dry; 0 = off, the historical bit-exact path).
pub struct Sampler {
    pub temp: f32,
    pub top_p: f32,
    pub rng: XorShift,
    pub spec: usize,
    pub spec_rosa: usize,
    /// --mtp: greedy self-speculative decoding through the converted
    /// multi-token-prediction head (Qwen dense models only).
    pub mtp: bool,
    /// --mtp-depth: chained draft length per verification pass.
    pub mtp_depth: usize,
    pub dry: f32,
}

impl Sampler {
    pub fn new(temp: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler { temp, top_p, rng: XorShift::new(seed), spec: 0, spec_rosa: 0, mtp: false, mtp_depth: 4, dry: 0.0 }
    }
    /// Default no-op decoding: the historical greedy behavior.
    pub fn greedy() -> Sampler {
        Sampler::new(0.0, 1.0, 0)
    }
}

/// Nucleus (top-p) sampling from softmax(logits / temp): sort the candidates
/// by probability desc, keep the smallest set covering `top_p` of the mass,
/// renormalize, draw with `rng`. Returns (token id, probability under the
/// truncated renormalized distribution).
fn sample_top_p(logits: &[f32], temp: f32, top_p: f32, rng: &mut XorShift) -> (u32, f32) {
    let inv_t = 1.0 / temp;
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut probs: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as u32, ((l - m) * inv_t).exp()))
        .collect();
    probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f32 = probs.iter().map(|&(_, p)| p).sum();
    // smallest prefix covering top_p of the mass
    let top_p = top_p.clamp(0.0, 1.0);
    let mut keep = probs.len();
    let mut cum = 0f32;
    for (i, &(_, p)) in probs.iter().enumerate() {
        cum += p / total;
        if cum >= top_p {
            keep = i + 1;
            break;
        }
    }
    let nucleus = &probs[..keep.max(1)];
    let nsum: f32 = nucleus.iter().map(|&(_, p)| p).sum();
    let mut r = rng.uniform() as f32 * nsum;
    for &(id, p) in nucleus {
        r -= p;
        if r <= 0.0 {
            return (id, p / nsum);
        }
    }
    let (id, p) = nucleus[nucleus.len() - 1];
    (id, p / nsum)
}

pub(crate) fn py_repr(s: &str) -> String {
    let mut out = String::from("'");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\'' => out.push_str("\\'"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

/// DRY-style anti-repetition penalty (--dry P): a token that would EXTEND an
/// n-gram (length >= 3) already present earlier in the GENERATION gets
/// P x DECAY^distance subtracted from its logit, where distance is the
/// number of tokens between the end of the earlier occurrence and the
/// current tail (DECAY = 0.9: a repetition starting far back hurts less).
/// Only the last 64 generated tokens are scanned, and the prompt is never
/// scanned: what matters is the text the model itself produced. Plain
/// quadratic scan over the 64-token window, negligible next to a forward.
/// Shared by the run_turn_core_batch loop and the --spec verification.
pub(crate) fn apply_dry(logits: &mut [f32], generated: &[u32], pen: f32) {
    const WIN: usize = 64;
    const DECAY: f32 = 0.9;
    let w = &generated[generated.len().saturating_sub(WIN)..];
    if w.len() < 3 {
        return;
    }
    for n in 3..=8usize.min(w.len()) {
        let m = n - 1; // matched suffix length (the n-gram completes with the next token)
        let suffix = &w[w.len() - m..];
        for i in 0..w.len() - m {
            // the final occurrence (the suffix itself, at i == w.len() - m) is excluded
            if w[i..i + m] == *suffix {
                let culprit = w[i + m] as usize; // token that followed the earlier occurrence
                let dist = (w.len() - m - i) as f32;
                if culprit < logits.len() {
                    logits[culprit] -= pen * DECAY.powf(dist);
                }
            }
        }
    }
}

pub fn run_turn(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, sampler: &mut Sampler) -> String {
    model.reset_cache();
    run_turn_impl(ids, max_new, tok, model, debug, debug_routing, stop_id, false, None, sampler)
}

/// Same as run_turn but keeps the current caches (restored from a .mkmem
/// snapshot via --memory): the prompt tokens are fed on top of the loaded
/// state and `init_logits` (the logits stored in the snapshot) seed the
/// decoding when the prompt is empty - a pure continuation.
pub fn run_turn_resume(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    run_turn_impl(ids, max_new, tok, model, debug, debug_routing, stop_id, true, init_logits, sampler)
}

fn run_turn_impl(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, resumed: bool, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    model.prof = Prof::default();
    let mut pos = if resumed { model.cached_tokens() } else { 0 };
    // --spec N / --spec-rosa N: speculative decoding, greedy only (rejection
    // sampling for temp > 0 is future work; the flags are ignored there)
    if sampler.spec > 0 || sampler.spec_rosa > 0 {
        if sampler.temp > 0.0 {
            eprintln!("warning: --spec/--spec-rosa are greedy-only, ignoring them with --temp > 0");
        } else {
            let answer = crate::model::spec::run_turn_spec(ids, max_new, tok, model, pos, init_logits, debug, stop_id, sampler);
            model.prof.print_cfg(&model.cfg);
            return answer;
        }
    }
    let answer = run_turn_core_batch(
        ids,
        max_new,
        tok,
        &mut |batch: &[u32]| {
            let l = model.prefill(batch, pos);
            pos += batch.len();
            l
        },
        debug,
        debug_routing,
        stop_id,
        init_logits,
        sampler,
    );
    model.prof.print_cfg(&model.cfg);
    answer
}

/// One decoding selection over `logits`, exactly like the generation
/// loop: greedy argmax of the top-5 at temperature 0, top-p nucleus
/// sampling otherwise, with the optional DRY penalty over the emitted
/// context. Shared with `microkimi serve`.
pub(crate) fn sample_next(logits: &[f32], sampler: &mut Sampler, gen_ctx: &[u32]) -> u32 {
    let mut dry_logits;
    let sel: &[f32] = if sampler.dry > 0.0 {
        dry_logits = logits.to_vec();
        apply_dry(&mut dry_logits, gen_ctx, sampler.dry);
        &dry_logits
    } else {
        logits
    };
    if sampler.temp > 0.0 {
        sample_top_p(sel, sampler.temp, sampler.top_p, &mut sampler.rng).0
    } else {
        top_k_probs(sel, 5)[0].0 as u32
    }
}

/// Generic greedy generation loop: prefill then argmax decode through the
/// `fwd` closure (one forward per token, position tracked by the caller).
/// With `sampler.temp > 0` the argmax becomes top-p nucleus sampling.
/// Shared by the K3 Model (run_turn) and the DeepSeek DsModel (ds_run_turn).
pub fn run_turn_core(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(u32) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, sampler: &mut Sampler) -> String {
    run_turn_core_resume(ids, max_new, tok, fwd, debug, debug_routing, stop_id, None, sampler)
}

/// run_turn_core + optional initial logits restored from a .mkmem snapshot:
/// with an empty prompt the decoding starts straight from them (pure
/// continuation, no token is re-ingested).
pub fn run_turn_core_resume(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(u32) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    run_turn_core_batch(
        ids,
        max_new,
        tok,
        &mut |batch: &[u32]| {
            // sequential prefill, one forward per token
            let mut l = Vec::new();
            for &id in batch {
                l = fwd(id);
            }
            l
        },
        debug,
        debug_routing,
        stop_id,
        init_logits,
        sampler,
    )
}

/// Batch variant of run_turn_core_resume: `fwd` ingests a slice of tokens
/// and returns the logits of its last token. The whole prompt is handed
/// over in ONE call (batched prefill on the K3 Model); during decoding each
/// call carries exactly one token. Bit-identical generation as long as the
/// closure's prefill matches n sequential single-token forwards.
pub fn run_turn_core_batch(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(&[u32]) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>, sampler: &mut Sampler) -> String {
    if debug_routing {
        ROUTING.with(|r| *r.borrow_mut() = Some(RoutingDebug::default()));
    }

    if debug {
        println!("{}", "=".repeat(64));
        println!("STEP 0 - TOKENIZATION  ({} tokens)", ids.len());
        println!("{}", "=".repeat(64));
        for (i, &id) in ids.iter().enumerate() {
            println!("  position {:2} : token {:6} = {}", i, id, py_repr(&tok.decode_id(id)));
        }
    }

    // ── prefill: the whole prompt in one batched call ──
    let t2 = Instant::now();
    let io0 = io_stats();
    let mut logits = init_logits.unwrap_or_default();
    if !ids.is_empty() {
        logits = fwd(ids);
        logit_lens_print_maybe(tok, "last prefill position");
    }
    if logits.is_empty() {
        eprintln!("error: nothing to continue from (empty prompt and no logits stored in the .mkmem snapshot)");
        std::process::exit(1);
    }
    let t_prefill = t2.elapsed();
    if debug {
        println!();
        println!("{}", "=".repeat(64));
        println!("STEP 1 - PREFILL  (caches filled)");
        println!("{}", "=".repeat(64));
        if ids.is_empty() {
            println!("⏱  skipped: pure continuation from the .mkmem snapshot");
        } else {
            println!("⏱  {:.2} s  for {} tokens ({:.1} ms/token)", t_prefill.as_secs_f64(), ids.len(), t_prefill.as_secs_f64() / ids.len() as f64 * 1000.0);
            if let (Some((b0, f0)), Some((b1, f1))) = (io0, io_stats()) {
                let gb = (b1 - b0) as f64 / 1e9;
                if gb > 0.01 {
                    println!("💾 {:.1} GB paged in from disk during prefill ({} major faults)", gb, f1 - f0);
                }
            }
        }
        println!();
        println!("{}", "=".repeat(64));
        if sampler.temp > 0.0 {
            println!("STEP 2 - GENERATION  (sampling: temp = {}, top-p = {}, stop = token {})", sampler.temp, sampler.top_p, stop_id);
        } else {
            println!("STEP 2 - GENERATION  (greedy: softmax → argmax, stop = token {})", stop_id);
        }
        println!("{}", "=".repeat(64));
    }

    let mut generated: Vec<u32> = Vec::new();
    let mut gen_times: Vec<f64> = Vec::new();
    if debug_routing {
        // ignore prefill in the routing display (generated tokens only)
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                d.cur.clear();
            }
        });
    }
    for i in 0..max_new {
        // --dry P: anti-repetition penalty on the tokens that would extend
        // an already-seen n-gram of the generation. Off by default: the
        // selection below is bit-identical when P == 0.
        let mut dry_logits;
        let sel_logits: &Vec<f32> = if sampler.dry > 0.0 {
            dry_logits = logits.clone();
            apply_dry(&mut dry_logits, &generated, sampler.dry);
            &dry_logits
        } else {
            &logits
        };
        let top = top_k_probs(sel_logits, 5);
        // temp <= 0: exact historical greedy path (argmax of the top-5);
        // temp > 0: top-p nucleus sampling over softmax(logits / temp).
        let (next_id, sampled_p) = if sampler.temp > 0.0 {
            sample_top_p(sel_logits, sampler.temp, sampler.top_p, &mut sampler.rng)
        } else {
            (top[0].0 as u32, top[0].1)
        };
        if debug {
            let candidats: Vec<String> = top
                .iter()
                .map(|&(tid, p)| format!("{} {:.1}%", py_repr(&tok.decode_id(tid as u32)), p * 100.0))
                .collect();
            println!();
            println!("token {:2} → {}", i + 1, py_repr(&tok.decode_id(next_id)));
            println!("  candidates: {}", candidats.join("  "));
            if sampler.temp > 0.0 {
                println!("  sampled: p = {:.1}% (temp = {}, top-p = {})", sampled_p * 100.0, sampler.temp, sampler.top_p);
            }
        }
        if next_id == stop_id {
            if debug {
                println!("  [end: stop token {}]", stop_id);
            }
            break;
        }
        let ta = Instant::now();
        logits = fwd(&[next_id]);
        logit_lens_print_maybe(tok, &format!("generated token {}", i + 1));
        let dt = ta.elapsed().as_secs_f64();
        gen_times.push(dt);
        generated.push(next_id);
        if debug_routing {
            ROUTING.with(|r| {
                if let Some(d) = r.borrow_mut().as_mut() {
                    let segs: Vec<String> = d
                        .cur
                        .iter()
                        .map(|(l, top3)| {
                            let exps: Vec<String> = top3
                                .iter()
                                .map(|(e, w)| format!("E{}({:.2})", e, w))
                                .collect();
                            format!("L{}: {}", l, exps.join(" "))
                        })
                        .collect();
                    println!("tok {} | {}", py_repr(&tok.decode_id(next_id)), segs.join(" | "));
                    d.cur.clear();
                }
            });
        }
        if debug {
            println!("  ⏱  {:.0} ms for this token", dt * 1000.0);
        }
    }

    if debug_routing {
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                let mut all: Vec<((usize, u32), u32)> = d.counts.iter().map(|(k, v)| (*k, *v)).collect();
                all.sort_by(|a, b| b.1.cmp(&a.1));
                println!();
                println!("Most-used experts of the run (top-10, top-16 appearances):");
                for ((l, e), n) in all.iter().take(10) {
                    println!("  L{} E{} : {}×", l, e, n);
                }
            }
        });
    }

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
    if !gen_times.is_empty() {
        let moy = gen_times.iter().sum::<f64>() / gen_times.len() as f64;
        if debug {
            println!("prefill: {:.2} s  |  generation: {:.0} ms/token average ({:.1} tok/s)",
                t_prefill.as_secs_f64(), moy * 1000.0, 1.0 / moy);
        } else {
            println!("  ({:.0} ms/token, {:.1} tok/s)", moy * 1000.0, 1.0 / moy);
        }
    }
    answer
}
