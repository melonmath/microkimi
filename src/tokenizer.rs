// Kimi tiktoken tokenizer: real BPE vocabulary (ref/tiktoken.model,
// lines `base64(token_bytes) rank`), 8-alternative pre-tokenizer reimplemented
// by hand (Unicode approximation: range tables for Han, L, N, Lu/Ll, M),
// minimal-rank BPE merge loop, simplified XTML chat template.
//
// + NanoTokenizer: remap to the nano vocab (top-8192 of the training corpus
// + contiguous specials), loaded from vocab_nano.json produced by nano/prepare.py.
// Encode: real Kimi BPE → remap (out-of-vocab → UNK); decode: nano → Kimi bytes.

use std::collections::HashMap;

pub const NUM_BASE: u32 = 163_584;
pub const BOS: u32 = 163_584;
#[allow(dead_code)]
pub const EOS: u32 = 163_585;
pub const END_OF_MSG: u32 = 163_586; // stop token
pub const OPEN: u32 = 163_587;
pub const CLOSE: u32 = 163_588;
pub const SEP: u32 = 163_589;
#[allow(dead_code)]
pub const UNK: u32 = 163_838;
#[allow(dead_code)]
pub const PAD: u32 = 163_839;

// ── base64 (standard alphabet, with '=' padding) ──
fn b64_decode(s: &str) -> Vec<u8> {
    fn val(c: u8) -> u8 {
        match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    }
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc: u32 = 0;
        for (i, &c) in chunk.iter().enumerate() {
            acc |= (val(c) as u32) << (18 - 6 * i);
        }
        let n = chunk.len();
        out.push((acc >> 16) as u8);
        if n >= 3 {
            out.push((acc >> 8) as u8);
        }
        if n >= 4 {
            out.push(acc as u8);
        }
    }
    out
}

// ── Unicode classes (range-based approximation) ──
fn is_han(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF |
        0x20000..=0x2A6DF | 0x2A700..=0x2EBEF)
}
fn is_mark(c: char) -> bool {
    matches!(c as u32, 0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}
fn is_l(c: char) -> bool {
    c.is_alphabetic() || is_mark(c)
}
fn is_n(c: char) -> bool {
    c.is_numeric()
}
// "upper" class: Lu/Lt/Lm/Lo/M outside Han ≈ non-lowercase letter (or mark) outside Han
fn is_upper_class(c: char) -> bool {
    !is_han(c) && (is_mark(c) || (c.is_alphabetic() && !c.is_lowercase()))
}
// "lower" class: Ll/Lm/Lo/M outside Han ≈ non-uppercase letter (or mark) outside Han
fn is_lower_class(c: char) -> bool {
    !is_han(c) && (is_mark(c) || (c.is_alphabetic() && !c.is_uppercase()))
}

pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    id_to_bytes: Vec<Vec<u8>>, // 163840 entries (specials = ascii name)
}

impl Tokenizer {
    pub fn load(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("tiktoken.model unreadable");
        let mut ranks = HashMap::with_capacity(NUM_BASE as usize);
        let mut id_to_bytes: Vec<Vec<u8>> = vec![Vec::new(); 163_840];
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let (Some(b64), Some(rank)) = (it.next(), it.next()) else {
                continue;
            };
            let tok = b64_decode(b64);
            let rank: u32 = rank.parse().unwrap();
            if (rank as usize) < id_to_bytes.len() {
                id_to_bytes[rank as usize] = tok.clone();
            }
            ranks.insert(tok, rank);
        }
        // names of the 256 special tokens
        for i in 0..256u32 {
            let id = NUM_BASE + i;
            let name = special_name(id);
            id_to_bytes[id as usize] = name.as_bytes().to_vec();
        }
        Tokenizer { ranks, id_to_bytes }
    }

    // ── pre-tokenizer: 8-alternative pattern from tokenization_kimi.py ──
    fn pre_tokenize(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut out = Vec::new();
        let is_ws = |c: char| c.is_whitespace();
        let mut i = 0;
        // contraction (?i:'s|'t|'re|'ve|'m|'ll|'d) at position j → length or 0
        let contraction = |j: usize| -> usize {
            if j >= n || chars[j] != '\'' || j + 1 >= n {
                return 0;
            }
            let rest: String = chars[j + 1..(j + 3).min(n)].iter().collect();
            let low = rest.to_lowercase();
            if low.starts_with("re") || low.starts_with("ve") || low.starts_with("ll") {
                3 // apostrophe included
            } else if matches!(low.chars().next(), Some('s' | 't' | 'm' | 'd')) {
                2
            } else {
                0
            }
        };
        while i < n {
            let c = chars[i];
            // 1) Han+
            if is_han(c) {
                let mut j = i;
                while j < n && is_han(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            // 2-3) words: optional head character (neither \r\n, nor L, nor N) then
            //      upper*lower+ or upper+lower*, optional contraction
            if is_l(c) || (!matches!(c, '\r' | '\n') && !is_n(c) && i + 1 < n && is_l(chars[i + 1])) {
                let lead = if is_l(c) { 0 } else { 1 };
                let s = i + lead;
                let mut j = s;
                while j < n && is_upper_class(chars[j]) {
                    j += 1;
                }
                let mut k = j;
                while k < n && is_lower_class(chars[k]) {
                    k += 1;
                }
                let (end, ok) = if k > j {
                    (k, true) // alt 2: upper* lower+
                } else if j > s {
                    let mut k2 = j;
                    while k2 < n && is_lower_class(chars[k2]) {
                        k2 += 1;
                    }
                    (k2, true) // alt 3: upper+ lower*
                } else {
                    (s, false)
                };
                if ok {
                    let end = end + contraction(end);
                    out.push(chars[i..end].iter().collect());
                    i = end;
                    continue;
                }
                // no word: the head character falls through to the following alternatives
            }
            // 4) numbers: 1 to 3 digits
            if is_n(c) {
                let mut j = i;
                while j < n && j < i + 3 && is_n(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            // 5) optional space + punctuation/symbols + any newlines
            let punct_start = if c == ' ' && i + 1 < n && {
                let d = chars[i + 1];
                !is_ws(d) && !is_l(d) && !is_n(d)
            } {
                i + 1
            } else if !is_ws(c) && !is_l(c) && !is_n(c) {
                i
            } else {
                usize::MAX
            };
            if punct_start != usize::MAX {
                let mut j = punct_start;
                while j < n && !is_ws(chars[j]) && !is_l(chars[j]) && !is_n(chars[j]) {
                    j += 1;
                }
                while j < n && matches!(chars[j], '\r' | '\n') {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
            // 6-8) whitespace: \s*[\r\n]+ | \s+(?!\S) | \s+
            if is_ws(c) {
                let mut j = i;
                while j < n && is_ws(chars[j]) {
                    j += 1;
                }
                // end with the last group of newlines if any
                let mut e = j;
                while e > i && !matches!(chars[e - 1], '\r' | '\n') {
                    e -= 1;
                }
                if e > i && chars[i..e].iter().any(|&x| matches!(x, '\r' | '\n')) {
                    out.push(chars[i..e].iter().collect());
                    i = e;
                    continue;
                }
                if j == n {
                    out.push(chars[i..j].iter().collect()); // \s+(?!\S) : trailing whitespace
                    i = j;
                } else {
                    // the last space can attach to the next chunk (word/punctuation)
                    let next = chars[j];
                    let attach = is_l(next) || (chars[j - 1] == ' ' && !is_n(next));
                    if attach && j - i > 1 {
                        out.push(chars[i..j - 1].iter().collect());
                        i = j - 1;
                    } else if attach {
                        i = j - 1; // a single space: leave it for the next chunk
                    } else {
                        out.push(chars[i..j].iter().collect());
                        i = j;
                    }
                }
                continue;
            }
            // fallback: isolated character
            out.push(c.to_string());
            i += 1;
        }
        out
    }

    // ── BPE: minimal-rank merge (like rustgpt) ──
    fn bpe(&self, chunk: &[u8]) -> Vec<u32> {
        let mut word: Vec<Vec<u8>> = chunk.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for k in 0..word.len().saturating_sub(1) {
                let mut pair = word[k].clone();
                pair.extend_from_slice(&word[k + 1]);
                if let Some(&rank) = self.ranks.get(&pair) {
                    if best.map_or(true, |(br, _)| rank < br) {
                        best = Some((rank, k));
                    }
                }
            }
            let Some((_, pos)) = best else { break };
            let mut merged_token = word[pos].clone();
            merged_token.extend_from_slice(&word[pos + 1]);
            let mut merged: Vec<Vec<u8>> = Vec::with_capacity(word.len());
            let mut k = 0;
            while k < word.len() {
                if k + 1 < word.len() && word[k] == word[pos] && word[k + 1] == word[pos + 1] {
                    merged.push(merged_token.clone());
                    k += 2;
                } else {
                    merged.push(word[k].clone());
                    k += 1;
                }
            }
            word = merged;
        }
        word.iter()
            .filter_map(|t| self.ranks.get(t).copied())
            .collect()
    }

    /// Encode ordinary text (no special tokens).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in Self::pre_tokenize(text) {
            ids.extend(self.bpe(chunk.as_bytes()));
        }
        ids
    }

    #[allow(dead_code)]
    pub fn bytes_of(&self, id: u32) -> &[u8] {
        &self.id_to_bytes[id as usize]
    }

    pub fn decode_id(&self, id: u32) -> String {
        String::from_utf8_lossy(&self.id_to_bytes[id as usize]).into_owned()
    }
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.id_to_bytes[id as usize]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub fn special_name(id: u32) -> String {
    match id {
        BOS => "[BOS]".to_string(),
        EOS => "[EOS]".to_string(),
        END_OF_MSG => "<|end_of_msg|>".to_string(),
        OPEN => "<|open|>".to_string(),
        CLOSE => "<|close|>".to_string(),
        SEP => "<|sep|>".to_string(),
        UNK => "[UNK]".to_string(),
        PAD => "[PAD]".to_string(),
        _ => format!("<|reserved_token_{}|>", id),
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Generic XTML template (shared full/nano via AnyTokenizer)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub struct Markers {
    pub open: u32,
    pub close: u32,
    pub sep: u32,
    pub end_of_msg: u32,
}

fn open_tag(ids: &mut Vec<u32>, m: Markers, encode_seg: &mut dyn FnMut(&str) -> Vec<u32>, tag: &str) {
    ids.push(m.open);
    ids.extend(encode_seg(tag));
    ids.push(m.sep);
}
fn close_tag(ids: &mut Vec<u32>, m: Markers, encode_seg: &mut dyn FnMut(&str) -> Vec<u32>, tag: &str) {
    ids.push(m.close);
    ids.extend(encode_seg(tag));
    ids.push(m.sep);
}
fn message(
    ids: &mut Vec<u32>,
    m: Markers,
    encode_seg: &mut dyn FnMut(&str) -> Vec<u32>,
    role: &str,
    content_open: &str,
    content: &str,
    content_close: &str,
) {
    open_tag(ids, m, encode_seg, &format!("message role=\"{}\"", role));
    if !content_open.is_empty() {
        open_tag(ids, m, encode_seg, content_open);
    }
    if !content.is_empty() {
        ids.extend(encode_seg(content));
    }
    if !content_close.is_empty() {
        close_tag(ids, m, encode_seg, content_close);
    }
    close_tag(ids, m, encode_seg, "message");
    ids.push(m.end_of_msg);
}

/// Builds the complete chat sequence: history + question + generation prompt.
fn build_chat(
    history: &[(String, String)],
    question: &str,
    m: Markers,
    mut encode_seg: impl FnMut(&str) -> Vec<u32>,
) -> Vec<u32> {
    let mut ids = Vec::new();
    for (q, a) in history {
        message(&mut ids, m, &mut encode_seg, "user", "", q, "");
        // completed assistant message (encoding_k3._render_assistant_segments):
        // EMPTY think channel then response channel with the answer.
        open_tag(&mut ids, m, &mut encode_seg, "message role=\"assistant\"");
        open_tag(&mut ids, m, &mut encode_seg, "think");
        close_tag(&mut ids, m, &mut encode_seg, "think");
        open_tag(&mut ids, m, &mut encode_seg, "response");
        if !a.is_empty() {
            ids.extend(encode_seg(a));
        }
        close_tag(&mut ids, m, &mut encode_seg, "response");
        close_tag(&mut ids, m, &mut encode_seg, "message");
        ids.push(m.end_of_msg);
    }
    message(&mut ids, m, &mut encode_seg, "user", "", question, "");
    // generation prompt : <|open|> message role="assistant" <|sep|> <|open|> think <|sep|>
    open_tag(&mut ids, m, &mut encode_seg, "message role=\"assistant\"");
    open_tag(&mut ids, m, &mut encode_seg, "think");
    ids
}

// ════════════════════════════════════════════════════════════════════════════
// AnyTokenizer: full Kimi (microkimi) or nano remap (nanokimi)
// ════════════════════════════════════════════════════════════════════════════

pub struct NanoTokenizer {
    pub full: Tokenizer,
    pub vocab_size: usize,
    pub nano_to_kimi: Vec<u32>,
    pub kimi_to_nano: Vec<u32>, // 163840, out-of-vocab → unk
    pub markers: Markers,
    pub bos: u32,
    pub eos: u32,
    #[allow(dead_code)]
    pub unk: u32,
    pub names: Vec<(u32, String)>, // displayable names of nano special tokens
}

impl NanoTokenizer {
    pub fn load(vocab_path: &str, full: Tokenizer) -> Self {
        let bytes = std::fs::read(vocab_path).expect("vocab_nano.json unreadable");
        let j = crate::json::parse(&bytes);
        let nano_to_kimi: Vec<u32> = j
            .get("nano_to_kimi")
            .and_then(|x| x.as_arr())
            .expect("vocab_nano.json: nano_to_kimi missing")
            .iter()
            .map(|x| x.as_num().unwrap() as u32)
            .collect();
        let vocab_size = j
            .get("vocab_size")
            .and_then(|x| x.as_num())
            .unwrap_or(nano_to_kimi.len() as f64 + 8.0) as usize;
        let sp = j.get("specials").expect("vocab_nano.json: specials missing");
        let get = |k: &str| sp.get(k).and_then(|x| x.as_num()).unwrap() as u32;
        let (bos, eos, unk) = (get("bos"), get("eos"), get("unk"));
        let markers = Markers {
            open: get("open"),
            close: get("close"),
            sep: get("sep"),
            end_of_msg: get("end_of_msg"),
        };
        let mut kimi_to_nano = vec![unk; 163_840];
        for (n, &k) in nano_to_kimi.iter().enumerate() {
            kimi_to_nano[k as usize] = n as u32;
        }
        let names = vec![
            (bos, "[BOS]".to_string()),
            (eos, "[EOS]".to_string()),
            (markers.open, "<|open|>".to_string()),
            (markers.close, "<|close|>".to_string()),
            (markers.sep, "<|sep|>".to_string()),
            (markers.end_of_msg, "<|end_of_msg|>".to_string()),
            (unk, "[UNK]".to_string()),
        ];
        NanoTokenizer { full, vocab_size, nano_to_kimi, kimi_to_nano, markers, bos, eos, unk, names }
    }

    fn encode_nano(&self, text: &str) -> Vec<u32> {
        self.full
            .encode(text)
            .iter()
            .map(|&id| self.kimi_to_nano[id as usize])
            .collect()
    }
}

pub enum AnyTokenizer {
    Full(Tokenizer),
    Nano(NanoTokenizer),
    Ds(crate::model::dstok::DsTokenizer),
    DsNano(DsNanoTokenizer),
}

/// DeepSeek-V4 nano remap (nanodeepseek): real V4 BPE → nano ids (top-8192
/// of the training corpus + contiguous specials), loaded from
/// vocab_ds_nano.json produced by nano_ds/prepare.py.
pub struct DsNanoTokenizer {
    pub full: crate::model::dstok::DsTokenizer,
    pub vocab_size: usize,
    pub nano_to_ds: Vec<u32>,
    pub ds_to_nano: Vec<u32>, // 129280, out-of-vocab → unk
    pub bos: u32,
    pub eos: u32,
    pub unk: u32,
}

impl DsNanoTokenizer {
    pub fn load(vocab_path: &str, full: crate::model::dstok::DsTokenizer) -> Self {
        let bytes = std::fs::read(vocab_path).expect("vocab_ds_nano.json unreadable");
        let j = crate::json::parse(&bytes);
        let nano_to_ds: Vec<u32> = j
            .get("nano_to_ds")
            .and_then(|x| x.as_arr())
            .expect("vocab_ds_nano.json: nano_to_ds missing")
            .iter()
            .map(|x| x.as_num().unwrap() as u32)
            .collect();
        let vocab_size = j
            .get("vocab_size")
            .and_then(|x| x.as_num())
            .unwrap_or(nano_to_ds.len() as f64 + 8.0) as usize;
        let sp = j.get("specials").expect("vocab_ds_nano.json: specials missing");
        let get = |k: &str| sp.get(k).and_then(|x| x.as_num()).unwrap() as u32;
        let (bos, eos, unk) = (get("bos"), get("eos"), get("unk"));
        let mut ds_to_nano = vec![unk; 129_280];
        for (n, &k) in nano_to_ds.iter().enumerate() {
            ds_to_nano[k as usize] = n as u32;
        }
        DsNanoTokenizer { full, vocab_size, nano_to_ds, ds_to_nano, bos, eos, unk }
    }

    fn encode_nano(&self, text: &str) -> Vec<u32> {
        self.full
            .encode(text)
            .iter()
            .map(|&id| self.ds_to_nano[id as usize])
            .collect()
    }

    /// Raw completion: nano BOS + remapped BPE (nanodeepseek is raw-only).
    fn encode_raw_nano(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![self.bos];
        ids.extend(self.encode_nano(text));
        ids
    }

    fn decode_nano_id(&self, id: u32) -> String {
        if id == self.bos {
            return "[BOS]".to_string();
        }
        if id == self.eos {
            return "[EOS]".to_string();
        }
        if id == self.unk {
            return "[UNK]".to_string();
        }
        if (id as usize) < self.nano_to_ds.len() {
            return self.full.decode_id(self.nano_to_ds[id as usize]);
        }
        format!("<|dsnano_{}|>", id)
    }

    fn decode_nano(&self, ids: &[u32]) -> String {
        let ds_ids: Vec<u32> = ids
            .iter()
            .map(|&id| {
                if (id as usize) < self.nano_to_ds.len() {
                    self.nano_to_ds[id as usize]
                } else {
                    match id {
                        x if x == self.bos => crate::model::dstok::DS_BOS,
                        x if x == self.eos => crate::model::dstok::DS_EOS,
                        _ => 2, // <｜▁pad▁｜> (unused slot)
                    }
                }
            })
            .collect();
        self.full.decode(&ds_ids)
    }
}

const FULL_MARKERS: Markers = Markers { open: OPEN, close: CLOSE, sep: SEP, end_of_msg: END_OF_MSG };

impl AnyTokenizer {
    pub fn end_of_msg(&self) -> u32 {
        match self {
            AnyTokenizer::Full(_) => END_OF_MSG,
            AnyTokenizer::Nano(n) => n.markers.end_of_msg,
            AnyTokenizer::Ds(_) => crate::model::dstok::DS_EOS,
            AnyTokenizer::DsNano(n) => n.eos,
        }
    }

    /// Upper bound of token ids this tokenizer can produce.
    pub fn vocab_size(&self) -> usize {
        match self {
            AnyTokenizer::Full(_) => 163_840,
            AnyTokenizer::Nano(n) => n.vocab_size,
            AnyTokenizer::Ds(_) => 129_280,
            AnyTokenizer::DsNano(n) => n.vocab_size,
        }
    }

    pub fn encode_chat_user(&self, question: &str) -> Vec<u32> {
        self.encode_chat(&[], question)
    }

    /// Raw completion (no XTML template): BOS + BPE text.
    /// For models trained on raw text (nanokimi).
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        match self {
            AnyTokenizer::Full(t) => t.encode(text),
            AnyTokenizer::Nano(n) => {
                let mut ids = vec![n.bos];
                ids.extend(n.encode_nano(text));
                ids
            }
            AnyTokenizer::Ds(t) => {
                let mut ids = vec![crate::model::dstok::DS_BOS];
                ids.extend(t.encode(text));
                ids
            }
            AnyTokenizer::DsNano(n) => n.encode_raw_nano(text),
        }
    }

    /// Stop token in raw completion mode: EOS for nano, end_of_msg otherwise.
    pub fn raw_stop(&self) -> u32 {
        match self {
            AnyTokenizer::Full(_) => END_OF_MSG,
            AnyTokenizer::Nano(n) => n.eos,
            AnyTokenizer::Ds(_) => crate::model::dstok::DS_EOS,
            AnyTokenizer::DsNano(n) => n.eos,
        }
    }

    pub fn encode_chat(&self, history: &[(String, String)], question: &str) -> Vec<u32> {
        match self {
            AnyTokenizer::Full(t) => build_chat(history, question, FULL_MARKERS, |s| t.encode(s)),
            AnyTokenizer::Nano(n) => build_chat(history, question, n.markers, |s| n.encode_nano(s)),
            AnyTokenizer::Ds(t) => t.encode_chat(history, question),
            // nanodeepseek is trained on raw stories: chat == raw completion
            AnyTokenizer::DsNano(n) => n.encode_raw_nano(question),
        }
    }

    pub fn decode_id(&self, id: u32) -> String {
        match self {
            AnyTokenizer::Full(t) => t.decode_id(id),
            AnyTokenizer::Nano(n) => {
                if let Some(name) = n.names.iter().find(|(i, _)| *i == id) {
                    return name.1.clone();
                }
                if (id as usize) < n.nano_to_kimi.len() {
                    return n.full.decode_id(n.nano_to_kimi[id as usize]);
                }
                format!("<|nano_{}|>", id)
            }
            AnyTokenizer::Ds(t) => t.decode_id(id),
            AnyTokenizer::DsNano(n) => n.decode_nano_id(id),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        match self {
            AnyTokenizer::Full(t) => t.decode(ids),
            AnyTokenizer::Nano(n) => {
                let kimi_ids: Vec<u32> = ids
                    .iter()
                    .map(|&id| {
                        if (id as usize) < n.nano_to_kimi.len() {
                            n.nano_to_kimi[id as usize]
                        } else {
                            match id {
                                x if x == n.markers.open => OPEN,
                                x if x == n.markers.close => CLOSE,
                                x if x == n.markers.sep => SEP,
                                x if x == n.markers.end_of_msg => END_OF_MSG,
                                x if x == n.bos => BOS,
                                x if x == n.eos => EOS,
                                _ => UNK,
                            }
                        }
                    })
                    .collect();
                n.full.decode(&kimi_ids)
            }
            AnyTokenizer::Ds(t) => t.decode(ids),
            AnyTokenizer::DsNano(n) => n.decode_nano(ids),
        }
    }
}
