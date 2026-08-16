//! `microkimi serve`: an OpenAI-compatible HTTP endpoint over a Qwen
//! model, zero dependencies (std TcpListener, the crate's JSON parser).
//!
//!   microkimi serve --model X.bin [--host 127.0.0.1] [--port 8080]
//!       [--vocab T.json] [--adapter P.mkap ...] [--mtp] [--max-new N]
//!
//! Routes:
//!   GET  /health               -> {"status":"ok"}
//!   GET  /v1/models            -> model list (the loaded file)
//!   POST /v1/completions       -> raw completion  (prompt, max_tokens,
//!                                 temperature, top_p, seed, stream)
//!   POST /v1/chat/completions  -> chat template   (messages [+ system],
//!                                 same sampling fields, stream)
//!
//! Design points, in order of importance:
//! - one model instance, requests served strictly one at a time (the
//!   decoder is stateful); each connection is closed after its response.
//! - the chat prefix cache is active across requests: a conversation that
//!   resends its history only prefills the new suffix, bit-identically.
//! - streaming responses are SSE chunks; decoded bytes are buffered and
//!   only complete UTF-8 prefixes are flushed, so a token boundary inside
//!   a multibyte character never garbles the stream.
//! - `<think>` reasoning is split into the DeepSeek-style
//!   `reasoning_content` field; `content` carries the visible answer.
//! - the default bind is 127.0.0.1. Binding elsewhere is an explicit
//!   choice; there is no authentication layer.
//! - requests are capped at 1 MiB, sockets carry read timeouts, a
//!   malformed request gets a JSON error, and a panic inside one request
//!   is caught so the server keeps serving.
//! - greedy requests (temperature 0) use MTP speculative decoding when
//!   the server was started with --mtp and the model carries its head.

use crate::model::qwen::QwenModel;
use crate::model::Sampler;
use crate::tokenizer::AnyTokenizer;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_BODY: usize = 1 << 20;
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(0);

// ── JSON writing ──

/// Escapes a string for inclusion in a JSON document.
pub fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

// ── HTTP plumbing ──

pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub body: Vec<u8>,
}

/// Reads one HTTP/1.1 request: request line, headers (only content-length
/// is interpreted), then exactly content-length body bytes.
pub fn read_request(stream: &mut dyn Read) -> Result<HttpRequest, String> {
    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
        if buf.len() > 64 << 10 {
            return Err("headers too large".to_string());
        }
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed before headers".to_string());
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    if method.is_empty() || path.is_empty() {
        return Err("malformed request line".to_string());
    }
    let mut content_length = 0usize;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().map_err(|_| "bad content-length")?;
            }
        }
    }
    if content_length > MAX_BODY {
        return Err("body too large".to_string());
    }
    let mut body = buf[header_end + 4..].to_vec();
    while body.len() < content_length {
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("connection closed mid-body".to_string());
        }
        body.extend_from_slice(&chunk[..n]);
    }
    body.truncate(content_length);
    Ok(HttpRequest { method, path, body })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn respond(stream: &mut dyn Write, status: &str, content_type: &str, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        status,
        content_type,
        body.len(),
        body
    );
}

fn respond_error(stream: &mut dyn Write, status: &str, message: &str) {
    respond(
        stream,
        status,
        "application/json",
        &format!(
            "{{\"error\":{{\"message\":\"{}\",\"type\":\"invalid_request_error\"}}}}",
            json_escape(message)
        ),
    );
}

// ── request parsing ──

pub struct GenRequest {
    pub prompt_ids: Vec<u32>,
    /// Index where the final assistant priming begins (chat requests):
    /// the prefix before it is what a future turn re-renders verbatim.
    pub conversation_split: usize,
    /// Fork source: continue from this stored timeline state instead of
    /// ingesting the full conversation ("state_id" in the request).
    pub fork_from: Option<String>,
    pub max_new: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u64,
    pub stream: bool,
}

fn number(json: &crate::json::Json, key: &str, default: f64) -> f64 {
    json.get(key).and_then(|x| x.as_num()).unwrap_or(default)
}

fn boolean(json: &crate::json::Json, key: &str) -> bool {
    matches!(json.get(key), Some(crate::json::Json::Bool(true)))
}

/// Collapses OpenAI-style messages into (system, history pairs, question).
/// The system message must come first; roles then alternate user /
/// assistant and the last message must be a user turn.
pub fn parse_messages(
    json: &crate::json::Json,
) -> Result<(Option<String>, Vec<(String, String)>, String), String> {
    let Some(messages) = json.get("messages").and_then(|x| x.as_arr()) else {
        return Err("messages missing".to_string());
    };
    let mut system: Option<String> = None;
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut pending_user: Option<String> = None;
    for (index, message) in messages.iter().enumerate() {
        let role = message
            .get("role")
            .and_then(|x| x.as_str())
            .ok_or("message role missing")?;
        let content = message
            .get("content")
            .and_then(|x| x.as_str())
            .ok_or("message content must be a string")?
            .to_string();
        match role {
            "system" => {
                if index != 0 {
                    return Err("system message must come first".to_string());
                }
                system = Some(content);
            }
            "user" => {
                if pending_user.is_some() {
                    return Err("two user messages in a row".to_string());
                }
                pending_user = Some(content);
            }
            "assistant" => {
                let Some(user) = pending_user.take() else {
                    return Err("assistant message without a preceding user message".to_string());
                };
                turns.push((user, content));
            }
            other => return Err(format!("unsupported role {:?}", other)),
        }
    }
    let Some(question) = pending_user else {
        return Err("the last message must be a user message".to_string());
    };
    Ok((system, turns, question))
}

fn parse_request(
    body: &[u8],
    chat: bool,
    tok: &AnyTokenizer,
    default_max_new: usize,
) -> Result<GenRequest, String> {
    if body.is_empty() {
        return Err("empty body".to_string());
    }
    let json = std::panic::catch_unwind(|| crate::json::parse_complete(body))
        .map_err(|_| "malformed JSON body".to_string())?;
    let fork_from = json
        .get("state_id")
        .and_then(|x| x.as_str())
        .map(|x| x.to_string());
    let (prompt_ids, conversation_split) = if chat {
        let (system, history, question) = parse_messages(&json)?;
        let AnyTokenizer::Qwen(qtok) = tok else {
            return Err("serve requires a Qwen model".to_string());
        };
        // "enable_thinking": false renders the disabled think block, so a
        // small model spends its budget on the visible answer
        let thinking = !matches!(json.get("enable_thinking"), Some(crate::json::Json::Bool(false)));
        if fork_from.is_some() {
            // the forked state already contains the whole conversation:
            // the request carries exactly the next user turn
            if system.is_some() || !history.is_empty() {
                return Err(
                    "with state_id, send exactly one user message (the state carries the history)"
                        .to_string(),
                );
            }
            let ids = qtok.continuation_turn(&question, thinking);
            let split = ids.len();
            (ids, split)
        } else {
            qtok.encode_chat_split(system.as_deref(), &history, &question, thinking)
        }
    } else {
        let prompt = json
            .get("prompt")
            .and_then(|x| x.as_str())
            .ok_or("prompt missing")?;
        let ids = tok.encode_raw(prompt);
        let split = ids.len();
        (ids, split)
    };
    let max_new = number(&json, "max_tokens", default_max_new as f64) as usize;
    if max_new == 0 || max_new > 8192 {
        return Err("max_tokens must be within 1..=8192".to_string());
    }
    Ok(GenRequest {
        prompt_ids,
        conversation_split,
        fork_from,
        max_new,
        temperature: number(&json, "temperature", 0.0) as f32,
        top_p: number(&json, "top_p", 1.0) as f32,
        seed: number(&json, "seed", 0.0) as u64,
        stream: boolean(&json, "stream"),
    })
}

// ── incremental UTF-8 flushing ──

/// Byte buffer that yields only complete UTF-8 prefixes.
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn new() -> Utf8Stream {
        Utf8Stream { pending: Vec::new() }
    }

    /// Appends decoded bytes and returns the longest flushable prefix.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) => e.valid_up_to(),
        };
        let out = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
        self.pending.drain(..valid);
        out
    }

    /// Flushes whatever remains at end of stream (lossy on a truncated
    /// final character).
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

/// Splits a raw chat answer into (reasoning, visible) following the
/// `<think>` template block, DeepSeek-style.
pub fn split_reasoning(answer: &str) -> (Option<String>, String) {
    match answer.split_once("</think>") {
        Some((reasoning, visible)) => (
            Some(reasoning.trim_start_matches('\n').trim_end().to_string()),
            visible.trim_start_matches('\n').to_string(),
        ),
        None => (None, answer.to_string()),
    }
}

// ── generation ──

struct Generated {
    ids: Vec<u32>,
    finish: &'static str,
}

/// Plain sampling loop over an ingested prompt. `on_token` receives each
/// committed token id (SSE streaming) and returns whether to continue: a
/// disconnected client stops the generation instead of burning compute
/// into a dead socket.
fn generate(
    model: &mut QwenModel,
    mut logits: Vec<f32>,
    stop_id: u32,
    max_new: usize,
    sampler: &mut Sampler,
    mut on_token: impl FnMut(u32) -> bool,
) -> Generated {
    let mut ids = Vec::new();
    let mut finish = "length";
    for _ in 0..max_new {
        let next = crate::model::sample_next(&logits, sampler, &ids);
        if next == stop_id {
            finish = "stop";
            break;
        }
        ids.push(next);
        if !on_token(next) {
            finish = "stop";
            break;
        }
        logits = model.prefill(&[next]);
    }
    Generated { ids, finish }
}

// ── server ──

struct Server {
    model: QwenModel,
    tok: AnyTokenizer,
    pck: Option<crate::memory::prefix_cache::Pck>,
    timelines: Option<crate::memory::timeline::TimelineStore>,
    model_name: String,
    default_max_new: usize,
    mtp: bool,
}

impl Server {
    fn handle(&mut self, request: &HttpRequest, stream: &mut TcpStream) {
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/health") => respond(stream, "200 OK", "application/json", "{\"status\":\"ok\"}"),
            ("GET", "/v1/models") => {
                let body = format!(
                    "{{\"object\":\"list\",\"data\":[{{\"id\":\"{}\",\"object\":\"model\",\"owned_by\":\"microkimi\"}}]}}",
                    json_escape(&self.model_name)
                );
                respond(stream, "200 OK", "application/json", &body);
            }
            ("POST", "/v1/completions") => self.completion(request, stream, false),
            ("POST", "/v1/chat/completions") => self.completion(request, stream, true),
            ("GET", "/v1/timelines") => self.timelines_list(stream),
            ("POST", "/v1/timelines/diff") => self.timelines_diff(request, stream),
            ("POST", "/v1/timelines/merge") => self.timelines_merge(request, stream),
            _ => respond_error(stream, "404 Not Found", "unknown route"),
        }
    }

    fn completion(&mut self, request: &HttpRequest, stream: &mut TcpStream, chat: bool) {
        let parsed = match parse_request(&request.body, chat, &self.tok, self.default_max_new) {
            Ok(parsed) => parsed,
            Err(message) => return respond_error(stream, "400 Bad Request", &message),
        };
        if parsed.prompt_ids.is_empty() {
            return respond_error(stream, "400 Bad Request", "empty prompt after encoding");
        }
        let id = REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let stop_id = if chat {
            self.tok.end_of_msg()
        } else {
            self.tok.raw_stop()
        };
        let mut sampler = Sampler::new(
            parsed.temperature.max(0.0),
            parsed.top_p.clamp(0.0, 1.0),
            if parsed.seed == 0 { created ^ id } else { parsed.seed },
        );

        // fork: restore the requested state and remember its token stream
        // (the commit below records the full covering stream)
        let mut base_tokens: Vec<u32> = Vec::new();
        if let Some(state_id) = &parsed.fork_from {
            let Some(store) = &self.timelines else {
                return respond_error(stream, "400 Bad Request", "timeline store unavailable");
            };
            let node = match store.get(state_id) {
                Ok(node) => node,
                Err(e) => return respond_error(stream, "400 Bad Request", &e),
            };
            if let Err(e) =
                crate::memory::qwen_state::load_slice(&mut self.model, &node.payload, "fork")
            {
                return respond_error(stream, "400 Bad Request", &e);
            }
            base_tokens = node.tokens;
        }

        // MTP speculative decoding: greedy, non-streaming, non-fork
        // requests only (a forked state has no valid draft cache).
        if self.mtp
            && sampler.temp <= 0.0
            && !parsed.stream
            && self.model.has_mtp()
            && parsed.fork_from.is_none()
        {
            self.model.reset();
            let (ids, _passes, _accepted) = crate::model::qwen::mtp_generate(
                &mut self.model,
                &parsed.prompt_ids,
                parsed.max_new,
                stop_id,
                &sampler,
                false,
            );
            let finish = if ids.len() >= parsed.max_new { "length" } else { "stop" };
            self.store_conversation(&parsed.prompt_ids, &ids);
            let state_id = self.commit_state(&parsed, &[], &ids, chat);
            return self.respond_full(stream, &parsed, &ids, finish, id, created, chat, state_id);
        }

        // prompt ingestion through the prefix cache when available. The
        // lookup and store happen at the conversation prefix (before the
        // generation priming): that prefix is what the next turn extends,
        // so it is the entry that can hit across turns. A fork skips the
        // cache: its state is already loaded.
        let split = parsed.conversation_split.min(parsed.prompt_ids.len());
        let logits = if parsed.fork_from.is_some() {
            self.model.prefill(&parsed.prompt_ids)
        } else {
        match &self.pck {
            Some(pck) if split > 0 => {
                let prefix_logits = crate::memory::prefix_cache::qwen_cached_prefill(
                    pck,
                    &parsed.prompt_ids[..split],
                    &mut self.model,
                );
                if split < parsed.prompt_ids.len() {
                    self.model.prefill(&parsed.prompt_ids[split..])
                } else {
                    prefix_logits
                }
            }
            _ => {
                self.model.reset();
                self.model.prefill(&parsed.prompt_ids)
            }
        }
        };

        if parsed.stream {
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
            );
            let mut utf8 = Utf8Stream::new();
            let mut all_ids: Vec<u32> = Vec::new();
            let object = if chat { "chat.completion.chunk" } else { "text_completion" };
            let generated = generate(
                &mut self.model,
                logits,
                stop_id,
                parsed.max_new,
                &mut sampler,
                |token| {
                    all_ids.push(token);
                    let piece = match &self.tok {
                        AnyTokenizer::Qwen(qtok) => utf8.push(&qtok.decode_bytes(&[token])),
                        _ => self.tok.decode_id(token),
                    };
                    if piece.is_empty() {
                        return true;
                    }
                    let ok = write!(stream, "data: {}\n\n", sse_chunk(object, id, created, &self.model_name, &piece, chat, None)).is_ok();
                    ok && stream.flush().is_ok()
                },
            );
            let tail = utf8.finish();
            if !tail.is_empty() {
                let _ = write!(stream, "data: {}\n\n", sse_chunk(object, id, created, &self.model_name, &tail, chat, None));
            }
            let _ = write!(
                stream,
                "data: {}\n\ndata: [DONE]\n\n",
                sse_chunk(object, id, created, &self.model_name, "", chat, Some(generated.finish))
            );
            let _ = stream.flush();
            if chat {
                self.store_conversation(&parsed.prompt_ids, &all_ids);
                self.commit_state(&parsed, &base_tokens, &all_ids, chat);
            }
            return;
        }

        let generated = generate(&mut self.model, logits, stop_id, parsed.max_new, &mut sampler, |_| true);
        if chat {
            self.store_conversation(&parsed.prompt_ids, &generated.ids);
        }
        let state_id = self.commit_state(&parsed, &base_tokens, &generated.ids, chat);
        self.respond_full(stream, &parsed, &generated.ids, generated.finish, id, created, chat, state_id);
    }

    /// Commits the post-generation state as a timeline node (chat only).
    /// The node's parent is the forked state when the request forked, and
    /// its token stream covers everything the state has ingested.
    fn commit_state(
        &mut self,
        parsed: &GenRequest,
        base_tokens: &[u32],
        generated: &[u32],
        chat: bool,
    ) -> Option<String> {
        if !chat || generated.is_empty() {
            return None;
        }
        let store = self.timelines.as_ref()?;
        let mut tokens = base_tokens.to_vec();
        tokens.extend_from_slice(&parsed.prompt_ids);
        tokens.extend_from_slice(generated);
        let payload = match crate::memory::qwen_state::serialize(&self.model) {
            Ok(payload) => payload,
            Err(e) => {
                eprintln!("timelines: state not committed ({})", e);
                return None;
            }
        };
        match store.put(parsed.fork_from.as_deref(), &tokens, &payload) {
            Ok(id) => Some(id),
            Err(e) => {
                eprintln!("timelines: {}", e);
                None
            }
        }
    }

    fn timelines_list(&self, stream: &mut TcpStream) {
        let Some(store) = &self.timelines else {
            return respond_error(stream, "400 Bad Request", "timeline store unavailable");
        };
        let mut rows: Vec<String> = store
            .list()
            .into_iter()
            .map(|meta| {
                format!(
                    "{{\"id\":\"{}\",\"parent\":{},\"tokens\":{}}}",
                    meta.id,
                    meta.parent
                        .map(|p| format!("\"{}\"", p))
                        .unwrap_or_else(|| "null".to_string()),
                    meta.n_tokens
                )
            })
            .collect();
        rows.sort();
        respond(
            stream,
            "200 OK",
            "application/json",
            &format!("{{\"object\":\"list\",\"data\":[{}]}}", rows.join(",")),
        );
    }

    /// Runs one prompt greedily from two states and reports where the two
    /// universes diverge. Deterministic by construction (greedy over
    /// bit-exact restored states), so a zero-diff means the states are
    /// behaviorally identical on this probe.
    fn timelines_diff(&mut self, request: &HttpRequest, stream: &mut TcpStream) {
        let json = match std::panic::catch_unwind(|| crate::json::parse_complete(&request.body)) {
            Ok(json) => json,
            Err(_) => return respond_error(stream, "400 Bad Request", "malformed JSON body"),
        };
        let (Some(a), Some(b), Some(prompt)) = (
            json.get("a").and_then(|x| x.as_str()),
            json.get("b").and_then(|x| x.as_str()),
            json.get("prompt").and_then(|x| x.as_str()),
        ) else {
            return respond_error(stream, "400 Bad Request", "diff needs a, b, prompt");
        };
        let max_new = number(&json, "max_tokens", 64.0) as usize;
        let AnyTokenizer::Qwen(qtok) = &self.tok else {
            return respond_error(stream, "400 Bad Request", "serve requires a Qwen model");
        };
        let continuation = qtok.continuation_turn(prompt, false);
        let stop_id = self.tok.end_of_msg();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        for state_id in [a, b] {
            let node = match self.timelines.as_ref().unwrap_or_else(|| unreachable!()).get(state_id) {
                Ok(node) => node,
                Err(e) => return respond_error(stream, "400 Bad Request", &e),
            };
            if let Err(e) =
                crate::memory::qwen_state::load_slice(&mut self.model, &node.payload, "diff")
            {
                return respond_error(stream, "400 Bad Request", &e);
            }
            let logits = self.model.prefill(&continuation);
            let mut sampler = Sampler::greedy();
            let generated = generate(&mut self.model, logits, stop_id, max_new, &mut sampler, |_| true);
            outputs.push(generated.ids);
        }
        let divergence = outputs[0]
            .iter()
            .zip(&outputs[1])
            .position(|(x, y)| x != y)
            .map(|i| i as i64)
            .unwrap_or_else(|| {
                if outputs[0].len() == outputs[1].len() {
                    -1
                } else {
                    outputs[0].len().min(outputs[1].len()) as i64
                }
            });
        let shared: Vec<u32> = match divergence {
            -1 => outputs[0].clone(),
            n => outputs[0][..n as usize].to_vec(),
        };
        let body = format!(
            "{{\"a_text\":\"{}\",\"b_text\":\"{}\",\"divergence_token\":{},\"shared_prefix\":\"{}\"}}",
            json_escape(&self.tok.decode(&outputs[0])),
            json_escape(&self.tok.decode(&outputs[1])),
            divergence,
            json_escape(&self.tok.decode(&shared))
        );
        respond(stream, "200 OK", "application/json", &body);
    }

    /// Three-way merge of two branches through their lowest common
    /// ancestor (see src/memory/timeline.rs for the semantics and the
    /// declared approximations).
    fn timelines_merge(&mut self, request: &HttpRequest, stream: &mut TcpStream) {
        let json = match std::panic::catch_unwind(|| crate::json::parse_complete(&request.body)) {
            Ok(json) => json,
            Err(_) => return respond_error(stream, "400 Bad Request", "malformed JSON body"),
        };
        let (Some(a), Some(b)) = (
            json.get("a").and_then(|x| x.as_str()),
            json.get("b").and_then(|x| x.as_str()),
        ) else {
            return respond_error(stream, "400 Bad Request", "merge needs a, b");
        };
        let Some(store) = &self.timelines else {
            return respond_error(stream, "400 Bad Request", "timeline store unavailable");
        };
        match crate::memory::timeline::merge_nodes(store, &mut self.model, a, b) {
            Ok(id) => {
                let tokens = store.get(&id).map(|n| n.tokens.len()).unwrap_or(0);
                respond(
                    stream,
                    "200 OK",
                    "application/json",
                    &format!("{{\"state_id\":\"{}\",\"tokens\":{}}}", id, tokens),
                );
            }
            Err(e) => respond_error(stream, "400 Bad Request", &e),
        }
    }

    /// Stores the post-generation state as a prefix-cache entry covering
    /// prompt + answer. The next turn re-renders the conversation with the
    /// answer inline, so its prompt extends exactly this prefix (the
    /// generation-priming header of THIS prompt never reappears, which is
    /// why the pre-generation entry alone cannot hit across turns).
    fn store_conversation(&self, prompt_ids: &[u32], generated: &[u32]) {
        let Some(pck) = &self.pck else { return };
        if generated.is_empty() {
            return;
        }
        let mut covered = prompt_ids.to_vec();
        covered.extend_from_slice(generated);
        match crate::memory::qwen_state::serialize(&self.model) {
            Ok(payload) => pck.store_payload(&covered, &payload),
            Err(e) => eprintln!("pck: conversation state not stored ({})", e),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn respond_full(
        &self,
        stream: &mut TcpStream,
        parsed: &GenRequest,
        ids: &[u32],
        finish: &'static str,
        id: u64,
        created: u64,
        chat: bool,
        state_id: Option<String>,
    ) {
        let text = self.tok.decode(ids);
        let usage = format!(
            "{{\"prompt_tokens\":{},\"completion_tokens\":{},\"total_tokens\":{}}}",
            parsed.prompt_ids.len(),
            ids.len(),
            parsed.prompt_ids.len() + ids.len()
        );
        let body = if chat {
            let (reasoning, visible) = split_reasoning(&text);
            let reasoning_field = reasoning
                .map(|r| format!("\"reasoning_content\":\"{}\",", json_escape(&r)))
                .unwrap_or_default();
            let state_field = state_id
                .map(|sid| format!("\"state_id\":\"{}\",", sid))
                .unwrap_or_default();
            format!(
                "{{\"id\":\"chatcmpl-{}\",\"object\":\"chat.completion\",\"created\":{},\"model\":\"{}\",{}\
                 \"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",{}\"content\":\"{}\"}},\
                 \"finish_reason\":\"{}\"}}],\"usage\":{}}}",
                id,
                created,
                json_escape(&self.model_name),
                state_field,
                reasoning_field,
                json_escape(&visible),
                finish,
                usage
            )
        } else {
            format!(
                "{{\"id\":\"cmpl-{}\",\"object\":\"text_completion\",\"created\":{},\"model\":\"{}\",\
                 \"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":\"{}\"}}],\"usage\":{}}}",
                id,
                created,
                json_escape(&self.model_name),
                json_escape(&text),
                finish,
                usage
            )
        };
        respond(stream, "200 OK", "application/json", &body);
    }
}

fn sse_chunk(
    object: &str,
    id: u64,
    created: u64,
    model: &str,
    piece: &str,
    chat: bool,
    finish: Option<&str>,
) -> String {
    let finish_json = finish
        .map(|f| format!("\"{}\"", f))
        .unwrap_or_else(|| "null".to_string());
    if chat {
        let delta = if piece.is_empty() {
            "{}".to_string()
        } else {
            format!("{{\"content\":\"{}\"}}", json_escape(piece))
        };
        format!(
            "{{\"id\":\"chatcmpl-{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",\
             \"choices\":[{{\"index\":0,\"delta\":{},\"finish_reason\":{}}}]}}",
            id, object, created, json_escape(model), delta, finish_json
        )
    } else {
        format!(
            "{{\"id\":\"cmpl-{}\",\"object\":\"{}\",\"created\":{},\"model\":\"{}\",\
             \"choices\":[{{\"index\":0,\"text\":\"{}\",\"finish_reason\":{}}}]}}",
            id, object, created, json_escape(model), json_escape(piece), finish_json
        )
    }
}

/// `microkimi serve --model X.bin [--host H] [--port P] ...`
pub fn run(args: &[String]) {
    let value = |flag: &str| crate::value_flag(args, flag);
    let model_path = value("--model").unwrap_or_else(crate::bin_path);
    let host = value("--host").unwrap_or_else(|| "127.0.0.1".to_string());
    let port: u16 = value("--port").and_then(|v| v.parse().ok()).unwrap_or(8080);
    let default_max_new: usize = value("--max-new").and_then(|v| v.parse().ok()).unwrap_or(512);
    let mtp = args.iter().any(|a| a == "--mtp");

    let cfg = crate::quant::weights::read_config(&model_path);
    if cfg.qwen.is_none() {
        eprintln!("error: serve currently supports Qwen-family models only");
        std::process::exit(1);
    }
    let tok = crate::load_qwen_any_tokenizer(&model_path, value("--vocab"), cfg.vocab);
    let packs = crate::adapter_flags(args);
    let model = if packs.is_empty() {
        QwenModel::load(&model_path)
    } else {
        QwenModel::load_with_adapters(&model_path, &packs)
    };
    if mtp && !model.has_mtp() {
        eprintln!("warning: --mtp ignored, the model was converted without its MTP head");
    }
    let pck = crate::memory::prefix_cache::open(&model_path);
    let timelines = crate::memory::timeline::TimelineStore::open(&model_path);
    let model_name = std::path::Path::new(&model_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "microkimi".to_string());

    let mut server = Server {
        mtp: mtp && model.has_mtp(),
        model,
        tok,
        pck,
        timelines,
        model_name,
        default_max_new,
    };
    let listener = TcpListener::bind((host.as_str(), port))
        .unwrap_or_else(|e| panic!("cannot bind {}:{}: {}", host, port, e));
    println!(
        "serving {} on http://{}:{}  (chat: /v1/chat/completions, raw: /v1/completions{})",
        server.model_name,
        host,
        port,
        if server.mtp { ", --mtp active for greedy requests" } else { "" }
    );
    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(30)));
        let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
        let request = match read_request(&mut stream) {
            Ok(request) => request,
            Err(message) => {
                respond_error(&mut stream, "400 Bad Request", &message);
                continue;
            }
        };
        // a panic inside one request must not kill the server
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            server.handle(&request, &mut stream);
        }));
        if result.is_err() {
            respond_error(&mut stream, "500 Internal Server Error", "request failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_parser_reads_line_headers_and_body() {
        let raw = b"POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 7\r\n\r\n{\"a\":1}";
        let mut cursor = std::io::Cursor::new(raw.to_vec());
        let request = read_request(&mut cursor).unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/v1/completions");
        assert_eq!(request.body, b"{\"a\":1}");
    }

    #[test]
    fn request_parser_rejects_oversized_bodies() {
        let raw = format!(
            "POST /x HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        );
        let mut cursor = std::io::Cursor::new(raw.into_bytes());
        assert!(read_request(&mut cursor).is_err());
    }

    #[test]
    fn message_collapse_enforces_role_discipline() {
        let ok = crate::json::parse_complete(
            br#"{"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"},{"role":"assistant","content":"hello"},{"role":"user","content":"how?"}]}"#,
        );
        let (system, history, question) = parse_messages(&ok).unwrap();
        assert_eq!(system.as_deref(), Some("be brief"));
        assert_eq!(history, vec![("hi".to_string(), "hello".to_string())]);
        assert_eq!(question, "how?");

        for bad in [
            r#"{"messages":[{"role":"user","content":"a"},{"role":"user","content":"b"}]}"#,
            r#"{"messages":[{"role":"assistant","content":"a"}]}"#,
            r#"{"messages":[{"role":"user","content":"a"},{"role":"system","content":"late"}]}"#,
            r#"{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}"#,
        ] {
            assert!(parse_messages(&crate::json::parse_complete(bad.as_bytes())).is_err(), "{}", bad);
        }
    }

    #[test]
    fn utf8_stream_never_splits_a_character() {
        let text = "héllo 世界";
        let bytes = text.as_bytes();
        let mut out = String::new();
        let mut stream = Utf8Stream::new();
        for b in bytes {
            let piece = stream.push(&[*b]);
            assert!(std::str::from_utf8(piece.as_bytes()).is_ok());
            out.push_str(&piece);
        }
        out.push_str(&stream.finish());
        assert_eq!(out, text);
    }

    #[test]
    fn reasoning_splits_on_the_think_close() {
        let (reasoning, visible) = split_reasoning("first think\n</think>\nThe answer is 4.");
        assert_eq!(reasoning.as_deref(), Some("first think"));
        assert_eq!(visible, "The answer is 4.");
        let (none, plain) = split_reasoning("no reasoning here");
        assert!(none.is_none());
        assert_eq!(plain, "no reasoning here");
    }

    #[test]
    fn json_escape_covers_controls_and_quotes() {
        assert_eq!(json_escape("a\"b\\c\nd\te\r"), "a\\\"b\\\\c\\nd\\te\\r");
        assert_eq!(json_escape("\u{1}"), "\\u0001");
    }
}
