// microkimi - 100% Rust inference engine + weight builder, zero dependencies.
// Micro reimplementation of the Kimi K3 architecture (MoE): same counts
// (93 layers, 69 KDA + 24 MLA, 896 experts top-16 + 2 shared, AttnRes block 12),
// same mechanisms (KDA, MLA NoPE, latent MoE, SiTU, MXFP4, noaux_tc router),
// reduced dims. std only.

mod build;
mod deepseek;
mod dequant;
mod config;
mod http;
mod json;
#[cfg(target_os = "macos")]
mod metal;
mod model;
mod mxfp4;
mod parity;
mod pool;
mod safetensors;
mod selftest;
mod tokenizer;
mod weights;

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");
    let t0 = Instant::now();

    match cmd {
        "build" => build::run(),
        "selftest" => { selftest::run(); selftest::run_ds(); selftest::run_ds2(); selftest::run_ds3(); },
        "metaltest" => metaltest_cmd(),
        "gputest" => gputest_cmd(),
        "gpubench" => gpubench_cmd(&args),
        "paritytest" => parity::run(args.iter().any(|a| a == "--show")),
        "run" => {
            // microkimi run "prompt" [--max-new N] [--model X.bin] [--vocab V.json]
            let positional: Vec<&String> = args.iter().skip(2).filter(|a| !a.starts_with("--")).collect();
            let prompt = positional.first().map(|s| s.to_string()).unwrap_or_else(|| "Hello".to_string());
            let max_new = args
                .iter()
                .position(|a| a == "--max-new")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok())
                .unwrap_or(20);
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            run_inference(&prompt, max_new, true, &model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"));
        }
        "chat" => {
            model::set_gpu(args.iter().any(|a| a == "--gpu"));
            chat_loop(&model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"));
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
            println!("  microkimi build                      builds microkimi.bin (K3 fetch + generation)");
            println!("  microkimi selftest                   compares against golden values (ref/golden.json)");
            println!("  microkimi run \"prompt\" [--max-new N]  greedy generation with detailed steps");
            println!("  microkimi chat                       interactive with history ('quit' to exit)");
            println!("  run/chat options: --model X.bin --vocab vocab_nano.json (auto if next to the .bin)");
            println!("                    --raw (raw completion, for nanokimi)  --debug-routing  --gpu (Metal, macOS)");
            println!("  microkimi metaltest | gputest | gpubench   Metal GPU checks (macOS only)");
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

/// Loads the tokenizer matching the model: explicit --vocab, otherwise vocab_nano.json
/// next to the .bin — but ONLY when its vocab size matches the model's (a stray
/// vocab_nano.json next to microkimi.bin must NOT hijack the full tokenizer),
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
fn run_inference(question: &str, max_new: usize, debug: bool, model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool) -> String {
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    let tok = load_any_tokenizer(&mp, vocab, crate::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    println!("cores used for matvecs: {}", model::n_threads());
    gpu_status_line();

    let (ids, stop) = if raw {
        (tok.encode_raw(question), tok.raw_stop())
    } else {
        (tok.encode_chat_user(question), tok.end_of_msg())
    };
    let answer = model::run_turn(&ids, max_new, &tok, &mut model, debug, debug_routing, stop);
    gpu_prof_maybe_print();
    answer
}

fn chat_loop(model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool) {
    use std::io::Write;
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    let tok = load_any_tokenizer(&mp, vocab, crate::weights::read_config(&mp).vocab);
    let mut model = model::Model::load(&mp);
    check_tok_compat(&tok, &model);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    gpu_status_line();
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
            let ids = tok.encode_raw(q);
            let stop = tok.raw_stop();
            model::run_turn(&ids, 200, &tok, &mut model, false, debug_routing, stop);
        } else {
            let ids = tok.encode_chat(&history, q);
            let answer = model::run_turn(&ids, 200, &tok, &mut model, false, debug_routing, tok.end_of_msg());
            history.push((q.to_string(), answer));
        }
    }
}

pub fn bin_path() -> String {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_default();
    // Default model: nanokimi (the pretrained demo shipped in the GitHub
    // release), then microkimi (the 93-layer architecture demo from `build`).
    for name in ["nanokimi.bin", "microkimi.bin"] {
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
    eprintln!("  download nanokimi.bin + vocab_nano.json from the GitHub Releases page into the repo root,");
    eprintln!("  or run 'microkimi build' to assemble microkimi.bin (93-layer architecture demo).");
    std::process::exit(1);
}

pub fn tokenizer_path() -> String {
    for c in ["ref/tiktoken.model", "/workspace/microkimi/ref/tiktoken.model"] {
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
