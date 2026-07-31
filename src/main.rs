// microkimi - 100% Rust inference engine + weight builder, zero dependencies.
// Micro reimplementation of the Kimi K3 architecture (MoE): same counts
// (93 layers, 69 KDA + 24 MLA, 896 experts top-16 + 2 shared, AttnRes block 12),
// same mechanisms (KDA, MLA NoPE, latent MoE, SiTU, MXFP4, noaux_tc router),
// reduced dims. std only.

mod build;
mod config;
mod http;
mod json;
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
        "selftest" => selftest::run(),
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
            run_inference(&prompt, max_new, true, &model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw"));
        }
        "chat" => chat_loop(&model_flag(&args), vocab_flag(&args), args.iter().any(|a| a == "--debug-routing"), args.iter().any(|a| a == "--raw")),
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
            println!("                    --raw (raw completion, for nanokimi)  --debug-routing");
        }
    }
    let _ = t0;
}

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
/// next to the .bin (nanokimi), otherwise the full Kimi vocabulary (microkimi).
fn load_any_tokenizer(model_path: &str, vocab: Option<String>) -> tokenizer::AnyTokenizer {
    let nano_path = vocab.or_else(|| {
        let dir = std::path::Path::new(model_path).parent().unwrap_or(std::path::Path::new("."));
        let cand = dir.join("vocab_nano.json");
        if cand.exists() {
            Some(cand.to_string_lossy().into_owned())
        } else {
            None
        }
    });
    let full = tokenizer::Tokenizer::load(&tokenizer_path());
    match nano_path {
        Some(p) => {
            println!("nano vocabulary (remap): {}", p);
            tokenizer::AnyTokenizer::Nano(tokenizer::NanoTokenizer::load(&p, full))
        }
        None => tokenizer::AnyTokenizer::Full(full),
    }
}

/// Loads tokenizer + weights, runs one inference turn with detailed output.
fn run_inference(question: &str, max_new: usize, debug: bool, model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool) -> String {
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    let tok = load_any_tokenizer(&mp, vocab);
    let mut model = model::Model::load(&mp);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
    println!("cores used for matvecs: {}", model::n_threads());

    let (ids, stop) = if raw {
        (tok.encode_raw(question), tok.raw_stop())
    } else {
        (tok.encode_chat_user(question), tok.end_of_msg())
    };
    model::run_turn(&ids, max_new, &tok, &mut model, debug, debug_routing, stop)
}

fn chat_loop(model_path: &Option<String>, vocab: Option<String>, debug_routing: bool, raw: bool) {
    use std::io::Write;
    let tl = Instant::now();
    let mp = model_path.clone().unwrap_or_else(bin_path);
    let tok = load_any_tokenizer(&mp, vocab);
    let mut model = model::Model::load(&mp);
    println!("loading tokenizer + weights: {:.1?}", tl.elapsed());
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
    // the cargo binary is in target/release/ → go up to the project root
    let candidates = [
        std::path::PathBuf::from("microkimi.bin"),
        exe_dir.join("microkimi.bin"),
        exe_dir.join("../../microkimi.bin"),
        std::path::PathBuf::from("/workspace/microkimi/microkimi.bin"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.to_string_lossy().into_owned();
        }
    }
    "microkimi.bin".to_string()
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
