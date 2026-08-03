// microkimi - 100% Rust inference engine + weight builder, zero dependencies.
// Micro reimplementation of the Kimi K3 architecture (MoE): same counts
// (93 layers, 69 KDA + 24 MLA, 896 experts top-16 + 2 shared, AttnRes block 12),
// same mechanisms (KDA, MLA NoPE, latent MoE, SiTU, MXFP4, noaux_tc router),
// reduced dims. std only.

mod build;
mod build_ds;
mod deepseek;
mod dequant;
mod dstok;
mod config;
mod http;
mod json;
#[cfg(target_os = "macos")]
mod metal;
mod mkmem;
mod model;
mod mxfp4;
mod parity;
mod pool;
mod safetensors;
mod selftest;
mod slice;
mod tokenizer;
mod weights;

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let t0 = Instant::now();

    // unified arch selector: --arch k3 (default) | dsv4
    let arch = args
        .iter()
        .position(|a| a == "--arch")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("k3");

    match cmd {
        "build" => {
            if arch == "dsv4" { build_ds::run() } else { build::run() }
        }
        "build-ds" => build_ds::run(), // alias for `build --arch dsv4`
        // microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers "0-11"]
        "slice" => slice::run(&args),
        "selftest" => { selftest::run(); selftest::run_ds(); selftest::run_ds2(); selftest::run_ds3(); selftest::run_ds4(); },
        "metaltest" => metaltest_cmd(),
        "gputest" => gputest_cmd(),
        "dstest" => dstest_cmd(),
        "gpubench" => gpubench_cmd(&args),
        "paritytest" | "parity" => {
            if arch == "dsv4" { parity::run_ds() } else { parity::run(args.iter().any(|a| a == "--show")) }
        }
        "dsparity" => parity::run_ds(), // alias for `parity --arch dsv4`
        "run" => {
            // microkimi run "prompt" [--max-new N] [--model X.bin] [--vocab V.json]
            //                        [--memory mem.mkmem] [--save mem.mkmem]
            //                        [--temp T] [--top-p P] [--seed N]
            let positional: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
            let prompt = positional.first().map(|s| s.to_string()).unwrap_or_else(|| "Hello".to_string());
            let max_new = args
                .iter()
                .position(|a| a == "--max-new")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            run_inference(&prompt, max_new, true, &model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"), &value_flag(&args, "--memory"), &value_flag(&args, "--save"), &mut sampler_flag(&args));
        }
        "chat" => {
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            chat_loop(&model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"), value_flag(&args, "--memory"), value_flag(&args, "--save"), &mut sampler_flag(&args));
        }
        // microkimi prefill "text" --save mem.mkmem [--model X.bin] [--vocab V.json] [--chat]
        "prefill" => {
            let positional: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
            let text = positional.first().map(|s| s.to_string()).unwrap_or_default();
            let Some(save) = value_flag(&args, "--save") else {
                eprintln!("error: prefill requires --save mem.mkmem");
                std::process::exit(1);
            };
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            prefill_cmd(&text, &save, &model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--chat"));
        }
        // debug command (debug helper): prints the tokenization of a text
        "tok" => {
            let tok = tokenizer::Tokenizer::load(&tokenizer_path());
            let text = args.get(2).cloned().unwrap_or_default();
            let any = tokenizer::AnyTokenizer::Full(tok);
            let ids = if text.starts_with("--chat2") {
                let q = args.get(3).cloned().unwrap_or_default();
                any.encode_chat(&[("hello".to_string(), "hi".to_string())], &q)
            } else if text.starts_with("--chat") {
                let q = args.get(3).cloned().unwrap_or_default();
                any.encode_chat_user(&q)
            } else {
                match &any {
                    tokenizer::AnyTokenizer::Full(t) => t.encode(&text),
                    _ => unreachable!(),
                }
            };
            println!("{:?}", ids);
            if !text.starts_with("--") {
                for &id in &ids {
                    println!("  {:6} = {:?}", id, any.decode_id(id));
                }
            }
        }
        _ => {
            println!("microkimi - micro K3 inference engine, zero dependencies");
            println!("usage:");
            println!("  microkimi build                      builds microkimi-debug.bin (K3 fetch + generation)");
            println!("  microkimi build-ds                   builds microdeepseek-debug.bin (DeepSeek-V4 fetch + generation)");
            println!("  microkimi selftest                   compares against golden values (ref/golden.json)");
            println!("  microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers \"0-11\"]");
            println!("                                         structural pruning (channels / experts / layers)");
            println!("  microkimi run \"prompt\" [--max-new N]  greedy generation with detailed steps");
            println!("  microkimi chat                       interactive with history ('quit' to exit)");
            println!("  microkimi prefill \"text\" --save mem.mkmem  ingest text, snapshot the state (.mkmem)");
            println!("  run/chat options: --model X.bin --vocab vocab_nano.json (auto if next to the .bin)");
            println!("                    --raw (raw completion, for nanokimi)  --debug-routing  --gpu (Metal, macOS)");
            println!("                    --memory mem.mkmem (resume a state)  --save mem.mkmem (snapshot after the run)");
            println!("                    --temp T (0 = greedy, default)  --top-p P (nucleus, default 1.0)  --seed N");
            println!("  microkimi metaltest | gputest | dstest | gpubench   Metal GPU checks (macOS only)");
        }
    }
    let _ = t0;
    // Exit explicitly: with an active Metal context + GBs of cached weight
    // buffers, graceful ObjC teardown at process exit can hang on macOS.
    // The OS reclaims everything anyway.
    if model::gpu_on() {
        std::process::exit(0);
    }
}

#[cfg(target_os = "macos")]
fn metaltest_cmd() {
    metal::metaltest();
}

#[cfg(not(target_os = "macos"))]
fn metaltest_cmd() {
    println!("metaltest is only available on macOS (Metal GPU support step 1)");
}

#[cfg(target_os = "macos")]
fn gputest_cmd() {
    metal::gputest();
}

#[cfg(not(target_os = "macos"))]
fn gputest_cmd() {
    println!("gputest is only available on macOS (Metal GPU support)");
}

#[cfg(target_os = "macos")]
fn dstest_cmd() {
    metal::dstest();
}

#[cfg(not(target_os = "macos"))]
fn dstest_cmd() {
    println!("dstest is only available on macOS (Metal GPU support, DeepSeek fp4 kernel)");
}

#[cfg(target_os = "macos")]
fn gpubench_cmd(args: &[String]) {
    let tl = Instant::now();
    let mp = model_flag(args).unwrap_or_else(bin_path);
    let tok = load_any_tokenizer(&mp, vocab_flag(args), crate::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    let question = "Once upon a time";
    let ids = tok.encode_chat_user(question);
    let n = 8;
    model::set_gpu(false);
    let (p_cpu, t_cpu, _) = bench_tokens(&ids, n, &tok, &mut model);
    model::set_gpu(true);
    let (p_gpu, t_gpu, ans) = bench_tokens(&ids, n, &tok, &mut model);
    println!();
    println!("gpubench - {} decode tokens on {}", n, mp);
    println!("  CPU : prefill {:.2} s, decode {:.0} ms/token ({:.1} tok/s)", p_cpu, t_cpu, 1000.0 / t_cpu);
    println!("  GPU : prefill {:.2} s, decode {:.0} ms/token ({:.1} tok/s)", p_gpu, t_gpu, 1000.0 / t_gpu);
    if t_cpu > 0.0 && t_gpu > 0.0 {
        println!("  decode speedup: {:.2}x", t_cpu / t_gpu);
    }
    println!("  answer (gpu): {}", ans);
}

/// Times prefill and per-token decode for n greedy steps. Returns
/// (prefill_s, avg_ms_per_token, decoded_answer).
#[cfg(target_os = "macos")]
fn bench_tokens(ids: &[u32], n: usize, tok: &tokenizer::AnyTokenizer, model: &mut model::Model) -> (f64, f64, String) {
    model.reset_cache();
    let t0 = Instant::now();
    let mut logits = Vec::new();
    let mut pos = 0usize;
    for &id in ids {
        logits = model.forward(id, pos);
        pos += 1;
    }
    let prefill = t0.elapsed().as_secs_f64();
    let mut times = Vec::new();
    let mut generated = Vec::new();
    for _ in 0..n {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0 as u32;
        if next == tok.end_of_msg() {
            break;
        }
        let ta = Instant::now();
        logits = model.forward(next, pos);
        pos += 1;
        times.push(ta.elapsed().as_secs_f64());
        generated.push(next);
    }
    let avg = if times.is_empty() {
        0.0
    } else {
        times.iter().sum::<f64>() / times.len() as f64 * 1000.0
    };
    (prefill, avg, tok.decode(&generated))
}

#[cfg(not(target_os = "macos"))]
fn gpubench_cmd(_args: &[String]) {
    println!("gpubench is only available on macOS (Metal GPU support)");
}

#[cfg(target_os = "macos")]
fn gpu_status_line() {
    if model::gpu_on() {
        println!("GPU: Metal (matvecs >= {} elems)", model::GPU_MIN_ELEMS);
    } else {
        println!("GPU: off (CPU)");
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_status_line() {
    if model::gpu_on() {
        println!("GPU: requested via --gpu but Metal is macOS-only - using CPU");
    } else {
        println!("GPU: off (CPU)");
    }
}

#[cfg(target_os = "macos")]
fn gpu_prof_maybe_print() {
    if model::gpu_on() {
        metal::gpu_prof_print();
    }
}

#[cfg(not(target_os = "macos"))]
fn gpu_prof_maybe_print() {}

fn model_flag(args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn vocab_flag(args: &[String]) -> Option<String> {
    args.iter()
        .position(|a| a == "--vocab")
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Generic --flag value extractor (--memory, --save).
fn value_flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Builds the decoding policy from --temp / --top-p / --seed.
/// temp absent or 0 -> greedy (the exact historical path). With temp > 0 and
/// no --seed, the RNG is seeded from the wall clock (non reproducible).
fn sampler_flag(args: &[String]) -> model::Sampler {
    let temp: f32 = value_flag(args, "--temp").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let top_p: f32 = value_flag(args, "--top-p").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let seed: u64 = value_flag(args, "--seed").and_then(|s| s.parse().ok()).unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x9E37_79B9_7F4A_7C15)
    });
    model::Sampler::new(temp, top_p, seed)
}

/// Drops the leading BOS of a freshly encoded prompt when resuming from a
/// .mkmem snapshot: BOS belongs to the stream start, already ingested when
/// the snapshot was taken. For the nano tokenizer encode_raw("") is just
/// [BOS], so an empty prompt becomes a pure continuation.
fn strip_bos(ids: &mut Vec<u32>, tok: &tokenizer::AnyTokenizer) {
    let bos = match tok {
        tokenizer::AnyTokenizer::Full(_) => Some(tokenizer::BOS),
        tokenizer::AnyTokenizer::Nano(n) => Some(n.bos),
        _ => None,
    };
    if ids.first() == bos.as_ref() {
        ids.remove(0);
    }
}

/// Loads the tokenizer matching the model: explicit --vocab, otherwise vocab_nano.json
/// next to the .bin — but ONLY when its vocab size matches the model's (a stray
/// vocab_nano.json next to microkimi-debug.bin must NOT hijack the full tokenizer),
/// otherwise the full Kimi vocabulary.
fn load_any_tokenizer(model_path: &str, vocab: Option<String>, model_vocab: usize) -> tokenizer::AnyTokenizer {
    let full = tokenizer::Tokenizer::load(&tokenizer_path());
    let nano_path = vocab.or_else(|| {
        let dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
        let cand = dir.join("vocab_nano.json");
        if cand.exists() {
            Some(cand.to_string_lossy().into_owned())
        } else {
            None
        }
    });
    match nano_path {
        Some(p) => {
            // read the nano vocab size before committing to the nano tokenizer
            let nano_vs = std::fs::read(&p).ok().and_then(|b| {
                crate::json::parse(&b).get("vocab_size").and_then(|x| x.as_num()).map(|n| n as usize)
            });
            if nano_vs == Some(model_vocab) {
                println!("nano vocabulary (remap): {}", p);
                tokenizer::AnyTokenizer::Nano(tokenizer::NanoTokenizer::load(&p, full))
            } else {
                println!(
                    "warning: ignoring {} (nano vocab {} != model vocab {}) - using the full Kimi tokenizer",
                    p,
                    nano_vs.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                    model_vocab
                );
                tokenizer::AnyTokenizer::Full(full)
            }
        }
        None => tokenizer::AnyTokenizer::Full(full),
    }
}

/// Fails cleanly when the tokenizer vocab does not match the model vocab
/// (either direction: nano on microkimi = [UNK] everywhere; full tokenizer on
/// nanokimi = out-of-range ids).
fn check_tok_compat(tok: &tokenizer::AnyTokenizer, model: &model::Model) {
    if tok.vocab_size() != model.cfg.vocab {
        eprintln!("error: tokenizer/model mismatch - model vocab is {}, tokenizer vocab is {}.", model.cfg.vocab, tok.vocab_size());
        eprintln!("hint: place vocab_nano.json next to the model (it ships in the GitHub release), or pass --vocab vocab_nano.json");
        std::process::exit(1);
    }
}

/// Loads tokenizer + weights, runs one inference turn with detailed output.
fn run_inference(question: &str, max_new: usize, debug: bool, model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool, memory: &Option<String>, save: &Option<String>, sampler: &mut model::Sampler) -> String {
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    // DeepSeek-V4 model → dedicated tokenizer + DsModel engine
    let mp_cfg = crate::weights::read_config(&mp);
    if mp_cfg.ds.is_some() {
        let tok = load_ds_any_tokenizer(&mp, vocab, mp_cfg.vocab);
        let mut model = deepseek::DsModel::load(&mp);
        println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
        println!("cores used for matvecs: {}", model::n_threads());
        let (ids, stop) = if raw {
            (tok.encode_raw(question), tok.raw_stop())
        } else {
            (tok.encode_chat_user(question), tok.end_of_msg())
        };
        return deepseek::ds_run_turn(&ids, max_new, &tok, &mut model, debug, debug_routing, stop);
    }
    let tok = load_any_tokenizer(&mp, vocab, crate::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    println!("cores used for matvecs: {}", model::n_threads());
    gpu_status_line();

    let mut init_logits = None;
    if let Some(m) = memory {
        match crate::mkmem::load(&mut model, m) {
            Ok(l) => {
                println!("memory loaded: {}", m);
                init_logits = Some(l);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }
    let (mut ids, stop) = if raw {
        (tok.encode_raw(question), tok.raw_stop())
    } else {
        (tok.encode_chat_user(question), tok.end_of_msg())
    };
    let answer = if init_logits.is_some() {
        strip_bos(&mut ids, &tok);
        model::run_turn_resume(&ids, max_new, &tok, &mut model, debug, debug_routing, stop, init_logits, sampler)
    } else {
        model::run_turn(&ids, max_new, &tok, &mut model, debug, debug_routing, stop, sampler)
    };
    if let Some(s) = save {
        save_memory(&model, s);
    }
    gpu_prof_maybe_print();
    answer
}

/// Snapshots the current state (caches + last logits) to a .mkmem file.
fn save_memory(model: &model::Model, path: &str) {
    match crate::mkmem::save(model, &model.last_logits, path) {
        Ok(()) => println!("memory saved: {}", path),
        Err(e) => {
            eprintln!("error: cannot write {}: {}", path, e);
            std::process::exit(1);
        }
    }
}

/// `microkimi prefill "text" --save mem.mkmem`: ingests the text (raw
/// completion encoding by default, chat template with --chat) and snapshots
/// the resulting state, without generating anything.
fn prefill_cmd(text: &str, save: &str, model_path: &Option<String>, vocab: Option<String>, chat: bool) {
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    if crate::weights::read_config(&mp).ds.is_some() {
        eprintln!("error: prefill is only supported for K3 models (not DeepSeek-V4)");
        std::process::exit(1);
    }
    let tok = load_any_tokenizer(&mp, vocab, crate::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    let ids = if chat {
        tok.encode_chat_user(text)
    } else {
        tok.encode_raw(text)
    };
    let tp = Instant::now();
    if !ids.is_empty() {
        model.prefill(&ids, 0);
    }
    save_memory(&model, save);
    let size = std::fs::metadata(save).map(|m| m.len()).unwrap_or(0);
    println!("prefill: {} tokens ingested in {:.1?} - state saved to {} ({:.1} KB)", ids.len(), tp.elapsed(), save, size as f64 / 1024.0);
}

fn chat_loop(model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool, memory: Option<String>, save: Option<String>, sampler: &mut model::Sampler) {
    use std::io::Write;
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    if crate::weights::read_config(&mp).ds.is_some() {
        return chat_loop_ds(&mp, vocab, debug_routing, raw);
    }
    let tok = load_any_tokenizer(&mp, vocab, crate::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    gpu_status_line();
    // --memory: resume a .mkmem snapshot; turns are then fed incrementally on
    // top of the loaded state (the history lives in the caches, no re-prefill).
    let mut init_logits = None;
    if let Some(m) = &memory {
        match crate::mkmem::load(&mut model, m) {
            Ok(l) => {
                println!("memory loaded: {}", m);
                init_logits = Some(l);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
    }
    let resumed = memory.is_some();
    if raw {
        println!("\nRAW interactive mode - each line is a story beginning to continue (type 'quit' to exit)");
    } else {
        println!("\nInteractive mode - history kept (type 'quit' to exit)");
    }
    let stdin = std::io::stdin();
    let mut history: Vec<(String, String)> = Vec::new(); // (question, answer)
    loop {
        print!("\nYou > ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let q = line.trim();
        if q.eq_ignore_ascii_case("quit") || q.eq_ignore_ascii_case("exit") {
            break;
        }
        if q.is_empty() {
            continue;
        }
        if raw {
            // raw completion: BOS + text, stop on EOS, each turn independent
            // (with --memory: one continuous stream, BOS stripped after the snapshot)
            let mut ids = tok.encode_raw(q);
            let stop = tok.raw_stop();
            if resumed {
                strip_bos(&mut ids, &tok);
                model::run_turn_resume(&ids, 200, &tok, &mut model, false, debug_routing, stop, init_logits.take(), sampler);
            } else {
                model::run_turn(&ids, 200, &tok, &mut model, false, debug_routing, stop, sampler);
            }
        } else if resumed {
            let ids = tok.encode_chat(&[], q);
            let answer = model::run_turn_resume(&ids, 200, &tok, &mut model, false, debug_routing, tok.end_of_msg(), init_logits.take(), sampler);
            history.push((q.to_string(), answer));
        } else {
            let ids = tok.encode_chat(&history, q);
            let answer = model::run_turn(&ids, 200, &tok, &mut model, false, debug_routing, tok.end_of_msg(), sampler);
            history.push((q.to_string(), answer));
        }
    }
    if let Some(s) = &save {
        save_memory(&model, s);
    }
}

/// Interactive loop for DeepSeek-V4 models (DsTokenizer + DsModel).
fn chat_loop_ds(mp: &str, vocab: Option<String>, debug_routing: bool, raw: bool) {
    use std::io::Write;
    let tl = Instant::now();
    let tok = load_ds_any_tokenizer(mp, vocab, crate::weights::read_config(mp).vocab);
    let mut model = deepseek::DsModel::load(mp);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    if raw {
        println!("\nRAW interactive mode - each line is a story beginning to continue (type 'quit' to exit)");
    } else {
        println!("\nInteractive mode - history kept (type 'quit' to exit)");
    }
    let stdin = std::io::stdin();
    let mut history: Vec<(String, String)> = Vec::new();
    loop {
        print!("\nYou > ");
        std::io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.read_line(&mut line).unwrap() == 0 {
            break;
        }
        let q = line.trim();
        if q.eq_ignore_ascii_case("quit") || q.eq_ignore_ascii_case("exit") {
            break;
        }
        if q.is_empty() {
            continue;
        }
        if raw {
            let ids = tok.encode_raw(q);
            let stop = tok.raw_stop();
            deepseek::ds_run_turn(&ids, 200, &tok, &mut model, false, debug_routing, stop);
        } else {
            let ids = tok.encode_chat(&history, q);
            let answer = deepseek::ds_run_turn(&ids, 200, &tok, &mut model, false, debug_routing, tok.end_of_msg());
            history.push((q.to_string(), answer));
        }
    }
}

/// Loads the tokenizer for a DeepSeek-V4 model: explicit --vocab (a full
/// tokenizer.json OR a vocab_ds_nano.json remap), then vocab_ds_nano.json next
/// to the bin when its vocab size matches the model's, otherwise the full V4
/// tokenizer (tokenizer.json next to the bin / cache / HF download).
fn load_ds_any_tokenizer(mp: &str, vocab: Option<String>, model_vocab: usize) -> tokenizer::AnyTokenizer {
    let full = || dstok::DsTokenizer::load(&ds_tokenizer_path(mp, None));
    let try_nano = |p: &str| -> Option<tokenizer::AnyTokenizer> {
        let bytes = std::fs::read(p).ok()?;
        let j = crate::json::parse(&bytes);
        j.get("nano_to_ds")?;
        let vs = j.get("vocab_size").and_then(|x| x.as_num()).map(|n| n as usize);
        if vs != Some(model_vocab) {
            eprintln!(
                "warning: ignoring {} (nano vocab {} != model vocab {}) - using the full V4 tokenizer",
                p,
                vs.map(|v| v.to_string()).unwrap_or_else(|| "?".to_string()),
                model_vocab
            );
            return None;
        }
        println!("DS nano vocabulary (remap): {}", p);
        Some(tokenizer::AnyTokenizer::DsNano(tokenizer::DsNanoTokenizer::load(p, full())))
    };
    if let Some(p) = &vocab {
        // explicit --vocab: nano remap or a plain tokenizer.json
        if let Some(t) = try_nano(p) {
            return t;
        }
        return tokenizer::AnyTokenizer::Ds(dstok::DsTokenizer::load(p));
    }
    let dir = std::path::Path::new(mp).parent().unwrap_or(std::path::Path::new("."));
    let cand = dir.join("vocab_ds_nano.json");
    if cand.exists() {
        if let Some(t) = try_nano(&cand.to_string_lossy()) {
            return t;
        }
    }
    tokenizer::AnyTokenizer::Ds(full())
}

/// Locates the DeepSeek-V4 tokenizer.json: explicit --vocab, then next to the
/// model (written by `build-ds`), then the local cache, then downloaded.
pub fn ds_tokenizer_path(model_path: &str, vocab: Option<String>) -> String {
    if let Some(v) = vocab {
        return v;
    }
    let dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
    let cand = dir.join("microdeepseek.tokenizer.json");
    if cand.exists() {
        return cand.to_string_lossy().into_owned();
    }
    let cache = format!("{}/.cache/microkimi", std::env::var("HOME").unwrap_or_default());
    let dst = format!("{}/microdeepseek.tokenizer.json", cache);
    if std::path::Path::new(&dst).exists() {
        return dst;
    }
    std::fs::create_dir_all(&cache).ok();
    println!("downloading tokenizer.json from huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 …");
    let data = crate::http::fetch(
        "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main/tokenizer.json",
    )
    .expect("failed to download the V4 tokenizer.json (no local file found)");
    std::fs::write(&dst, data).unwrap();
    dst
}

pub fn bin_path() -> String {    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();
    // Default model: nanokimi-0.2b (the pretrained demo shipped in the GitHub
    // release), then microkimi-debug (the 93-layer architecture demo from
    // `build`). Legacy file names are kept as fallbacks.
    for name in ["nanokimi-0.2b.bin", "nanokimi.bin", "microkimi-debug.bin", "microkimi.bin"] {
        let candidates = [
            std::path::PathBuf::from(name),
            exe_dir.join(name),
            exe_dir.join(format!("../../{}", name)),
        ];
        for c in &candidates {
            if c.exists() {
                return c.to_string_lossy().into_owned();
            }
        }
    }
    eprintln!("error: no model found.");
    eprintln!("  download nanokimi-0.2b.bin + vocab_nano.json from the GitHub Releases page into the repo root,");
    eprintln!("  or run 'microkimi build' to assemble microkimi-debug.bin (93-layer architecture demo).");
    std::process::exit(1);
}

pub fn tokenizer_path() -> String {
    for c in ["ref/tiktoken.model", "ref/tiktoken.model"] {
        if std::path::Path::new(c).exists() {
            return c.to_string();
        }
    }
    // tiktoken.model is not vendored (Moonshot license): local cache first, then
    // download from huggingface.co/moonshotai/Kimi-K3 if missing.
    let cache = format!("{}/.cache/microkimi", std::env::var("HOME").unwrap_or_default());
    let dst = format!("{}/tiktoken.model", cache);
    if std::path::Path::new(&dst).exists() {
        return dst;
    }
    std::fs::create_dir_all(&cache).ok();
    println!("downloading tiktoken.model from huggingface.co/moonshotai/Kimi-K3 …");
    let data = crate::http::fetch(
        "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/tiktoken.model",
    )
    .expect("failed to download tiktoken.model (no local file found)");
    std::fs::write(&dst, data).unwrap();
    dst
}
