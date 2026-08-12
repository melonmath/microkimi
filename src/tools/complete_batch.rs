//! Deterministic batched completion with one model load.
//!
//! Input is JSONL with one object per line:
//! `{"id":"task","prompt":"text","max_new":128,"stop":["\nclass "]}`.
//! The output is JSONL containing the id, completion, and token counts. Caches
//! are reset between requests, while model weights and external adapter packs
//! stay loaded.

use crate::json::Json;
use crate::tokenizer::AnyTokenizer;
use std::collections::HashSet;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

struct Request {
    id: String,
    prompt: String,
    max_new: usize,
    stops: Vec<String>,
}

enum BatchModel {
    K3(crate::model::Model),
    Qwen(crate::model::qwen::QwenModel),
    DeepSeek(crate::model::deepseek::DsModel),
}

impl BatchModel {
    fn reset(&mut self) {
        match self {
            BatchModel::K3(model) => model.reset_cache(),
            BatchModel::Qwen(model) => model.reset(),
            BatchModel::DeepSeek(model) => model.reset(),
        }
    }

    fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        match self {
            BatchModel::K3(model) => model.forward(token, pos),
            BatchModel::Qwen(model) => model.forward(token),
            BatchModel::DeepSeek(model) => model.forward(token, pos),
        }
    }

    fn prefill(&mut self, ids: &[u32]) -> Vec<f32> {
        match self {
            BatchModel::K3(model) => model.prefill(ids, 0),
            BatchModel::Qwen(model) => {
                let mut logits = Vec::new();
                for &token in ids {
                    logits = model.forward(token);
                }
                logits
            }
            BatchModel::DeepSeek(model) => {
                let mut logits = Vec::new();
                for (pos, &token) in ids.iter().enumerate() {
                    logits = model.forward(token, pos);
                }
                logits
            }
        }
    }
}

fn object<'a>(value: &'a Json, line: usize) -> &'a [(String, Json)] {
    let Json::Obj(pairs) = value else {
        panic!("input line {} must be a JSON object", line);
    };
    let mut seen = HashSet::new();
    for (key, _) in pairs {
        assert!(
            seen.insert(key),
            "input line {} contains duplicate field {:?}",
            line,
            key
        );
        assert!(
            matches!(key.as_str(), "id" | "prompt" | "max_new" | "stop"),
            "input line {} contains unsupported field {:?}",
            line,
            key
        );
    }
    pairs
}

fn field<'a>(pairs: &'a [(String, Json)], key: &str, line: usize) -> &'a Json {
    pairs
        .iter()
        .find(|(name, _)| name == key)
        .map(|(_, value)| value)
        .unwrap_or_else(|| panic!("input line {} is missing {:?}", line, key))
}

fn parse_requests(path: &str, default_max_new: usize) -> Vec<Request> {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
    let mut requests = Vec::new();
    let mut ids = HashSet::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let parsed = crate::json::parse_complete(line.as_bytes());
        let pairs = object(&parsed, line_number);
        let id = field(pairs, "id", line_number)
            .as_str()
            .unwrap_or_else(|| panic!("input line {} id must be a string", line_number));
        let prompt = field(pairs, "prompt", line_number)
            .as_str()
            .unwrap_or_else(|| panic!("input line {} prompt must be a string", line_number));
        assert!(
            !id.is_empty() && !id.chars().any(char::is_control),
            "input line {} id must be non-empty without control characters",
            line_number
        );
        assert!(
            !prompt.is_empty(),
            "input line {} prompt is empty",
            line_number
        );
        assert!(ids.insert(id.to_string()), "duplicate request id {:?}", id);
        let max_new = pairs
            .iter()
            .find(|(name, _)| name == "max_new")
            .map(|(_, value)| {
                let number = value.as_num().unwrap_or_else(|| {
                    panic!("input line {} max_new must be a number", line_number)
                });
                assert!(
                    number.is_finite()
                        && number.fract() == 0.0
                        && (1.0..=1_000_000.0).contains(&number),
                    "input line {} max_new must be an integer in 1..=1000000",
                    line_number
                );
                number as usize
            })
            .unwrap_or(default_max_new);
        let stops = pairs
            .iter()
            .find(|(name, _)| name == "stop")
            .map(|(_, value)| {
                value
                    .as_arr()
                    .unwrap_or_else(|| panic!("input line {} stop must be an array", line_number))
                    .iter()
                    .map(|value| {
                        let stop = value.as_str().unwrap_or_else(|| {
                            panic!("input line {} stop entries must be strings", line_number)
                        });
                        assert!(
                            !stop.is_empty(),
                            "input line {} contains an empty stop string",
                            line_number
                        );
                        stop.to_string()
                    })
                    .collect()
            })
            .unwrap_or_default();
        requests.push(Request {
            id: id.to_string(),
            prompt: prompt.to_string(),
            max_new,
            stops,
        });
    }
    assert!(!requests.is_empty(), "{} contains no requests", path);
    requests
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
        .expect("cannot continue an empty prompt")
        .0 as u32
}

fn complete(
    model: &mut BatchModel,
    tokenizer: &AnyTokenizer,
    request: &Request,
    chat: bool,
) -> (String, usize, usize) {
    model.reset();
    let ids = if chat {
        tokenizer.encode_chat_user(&request.prompt)
    } else {
        tokenizer.encode_raw(&request.prompt)
    };
    assert!(
        !ids.is_empty(),
        "request {:?} encoded to no tokens",
        request.id
    );
    let stop = if chat {
        tokenizer.end_of_msg()
    } else {
        tokenizer.raw_stop()
    };
    let mut logits = model.prefill(&ids);
    let mut generated = Vec::new();
    for _ in 0..request.max_new {
        let token = argmax(&logits);
        if token == stop {
            break;
        }
        generated.push(token);
        if !request.stops.is_empty() {
            let text = tokenizer.decode(&generated);
            if request.stops.iter().any(|stop| text.contains(stop)) {
                break;
            }
        }
        logits = model.forward(token, ids.len() + generated.len() - 1);
    }
    (tokenizer.decode(&generated), ids.len(), generated.len())
}

fn json_string(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch < '\u{20}' => out.push_str(&format!("\\u{:04x}", ch as u32)),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn load(args: &[String], path: &str, stream_mb: Option<usize>) -> (AnyTokenizer, BatchModel) {
    let config = crate::quant::weights::read_config(path);
    let vocab = crate::vocab_flag(args);
    let adapters = crate::adapter_flags(args);
    if config.qwen.is_some() {
        assert!(
            stream_mb.is_none(),
            "--stream is not supported by Qwen batch completion"
        );
        let tokenizer = crate::load_qwen_any_tokenizer(path, vocab, config.vocab);
        let model = if adapters.is_empty() {
            crate::model::qwen::QwenModel::load(path)
        } else {
            crate::model::qwen::QwenModel::load_with_adapters(path, &adapters)
        };
        (tokenizer, BatchModel::Qwen(model))
    } else if config.ds.is_some() {
        assert!(
            adapters.is_empty(),
            "external adapter packs are not supported by DeepSeek batch completion"
        );
        assert!(
            stream_mb.is_none(),
            "--stream is not supported by DeepSeek batch completion"
        );
        (
            crate::load_ds_any_tokenizer(path, vocab, config.vocab),
            BatchModel::DeepSeek(crate::model::deepseek::DsModel::load(path)),
        )
    } else {
        let tokenizer = crate::load_any_tokenizer(path, vocab, config.vocab);
        let model = crate::load_k3_model(path, stream_mb);
        crate::check_tok_compat(&tokenizer, &model);
        (tokenizer, BatchModel::K3(model))
    }
}

pub fn run(args: &[String]) {
    let input = crate::value_flag(args, "--input")
        .unwrap_or_else(|| panic!("complete-batch requires --input REQUESTS.jsonl"));
    let output = crate::value_flag(args, "--out")
        .unwrap_or_else(|| panic!("complete-batch requires --out COMPLETIONS.jsonl"));
    let model_path = crate::model_flag(args).unwrap_or_else(crate::bin_path);
    let default_max_new = crate::value_flag(args, "--max-new")
        .map(|value| {
            value
                .parse::<usize>()
                .ok()
                .filter(|&count| count > 0)
                .expect("--max-new must be a positive integer")
        })
        .unwrap_or(128);
    assert!(!Path::new(&output).exists(), "{} already exists", output);
    let requests = parse_requests(&input, default_max_new);
    let chat = args.iter().any(|arg| arg == "--chat");
    let stream_mb = crate::stream_ram_flag(args);
    let (tokenizer, mut model) = load(args, &model_path, stream_mb);
    let partial = format!("{}.partial.{}", output, std::process::id());
    let file = std::fs::File::create(&partial).unwrap();
    let mut writer = BufWriter::new(file);
    for (index, request) in requests.iter().enumerate() {
        let (completion, prompt_tokens, generated_tokens) =
            complete(&mut model, &tokenizer, request, chat);
        writeln!(
            writer,
            "{{\"id\":{},\"completion\":{},\"prompt_tokens\":{},\"generated_tokens\":{}}}",
            json_string(&request.id),
            json_string(&completion),
            prompt_tokens,
            generated_tokens
        )
        .unwrap();
        eprintln!(
            "complete-batch: {}/{} {:?} ({} tokens)",
            index + 1,
            requests.len(),
            request.id,
            generated_tokens
        );
    }
    writer.flush().unwrap();
    writer.get_ref().sync_all().unwrap();
    drop(writer);
    std::fs::rename(&partial, &output).unwrap();
    println!("completions: {} ({} requests)", output, requests.len());
    crate::stream_report_maybe(stream_mb);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> String {
        std::env::temp_dir()
            .join(format!(
                "microkimi_complete_batch_{}_{}",
                std::process::id(),
                name
            ))
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn parses_jsonl_requests_and_escapes_outputs() {
        let path = temp("requests.jsonl");
        std::fs::write(
            &path,
            "{\"id\":\"one\",\"prompt\":\"line\\ntext\",\"max_new\":7,\"stop\":[\"\\ndef \"]}\n\n",
        )
        .unwrap();
        let requests = parse_requests(&path, 3);
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].id, "one");
        assert_eq!(requests[0].prompt, "line\ntext");
        assert_eq!(requests[0].max_new, 7);
        assert_eq!(requests[0].stops, vec!["\ndef "]);
        assert_eq!(json_string("a\n\"b"), "\"a\\n\\\"b\"");
        std::fs::remove_file(path).ok();
    }
}
