// DeepSeek-V4 tokenizer: byte-level BPE parsed from the HF tokenizer.json
// (model.vocab + model.merges + added_tokens), 3-stage pre-tokenizer
// reimplemented by hand from the pre_tokenizer chain:
//   1. Split `\p{N}{1,3}` Isolated (digit runs of 1-3)
//   2. Split `[Han,Hiragana,Katakana]+` Isolated
//   3. Split on the 6-alternative GPT-2-ish pattern
//      (punct+ASCII-letters | prefix?+letters | space?+punct/symbols+newlines
//       | ws-with-newlines | trailing ws | ws)
// Alternatives are tried in order at each position (backtracking semantics);
// non-matching spans accumulate as whole gap pieces. \p{P}/\p{S} are
// approximated with range tables (exact for ASCII), like the Kimi tokenizer.
// Simplified V4 chat template (encoding_dsv4.py, chat mode, no tools):
// BOS (<｜User｜> q <｜Assistant｜> </think> a EOS)* <｜User｜> q <｜Assistant｜> </think>

use std::collections::HashMap;

pub const DS_BOS: u32 = 0; // <｜begin▁of▁sentence｜>
pub const DS_EOS: u32 = 1; // <｜end▁of▁sentence｜>
pub const DS_USER: u32 = 128_803; // <｜User｜>
pub const DS_ASSISTANT: u32 = 128_804; // <｜Assistant｜>
pub const DS_THINK_END: u32 = 128_822; // </think>

pub struct DsTokenizer {
    ids: HashMap<Vec<u8>, u32>,       // token bytes → id (byte-level decoded vocab)
    pair_rank: HashMap<Vec<u8>, u32>, // merged token bytes → merge rank
    id_to_bytes: Vec<Vec<u8>>,        // id → bytes (specials: raw UTF-8 content)
    added: Vec<(String, u32)>,        // added tokens, longest content first
}

// ── GPT-2 byte-level alphabet ──

/// byte → its byte-level char (printable bytes map to themselves, the others
/// to 0x100+n), the standard bytes_to_unicode() table.
fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(0x100 + extra).unwrap();
            extra += 1;
            c
        };
    }
    table
}

fn byte_level_decode(s: &str) -> Vec<u8> {
    let table = byte_to_char_table();
    let mut inv: HashMap<char, u8> = HashMap::with_capacity(256);
    for (b, &c) in table.iter().enumerate() {
        inv.insert(c, b as u8);
    }
    // tokens outside the byte-level alphabet (the special tokens, also present
    // in model.vocab) fall back to their raw UTF-8 bytes
    s.chars().map(|c| inv.get(&c).copied()).collect::<Option<Vec<u8>>>().unwrap_or_else(|| s.as_bytes().to_vec())
}

// ── Unicode classes (range-based approximation; ASCII exact) ──

fn is_cjk(c: char) -> bool {
    // stage-2 ranges: Han, Hiragana, Katakana (NOT the extensions)
    matches!(c as u32, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
}

fn is_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F)
}

fn is_l(c: char) -> bool {
    c.is_alphabetic()
}

fn is_lm(c: char) -> bool {
    is_l(c) || is_mark(c)
}

/// \p{P} (punctuation): exact for ASCII, range approximation beyond.
fn is_p(c: char) -> bool {
    let u = c as u32;
    match u {
        0x21..=0x23 | 0x25..=0x2F | 0x3A..=0x3B | 0x3F..=0x40 | 0x5B..=0x5D | 0x5F | 0x7B | 0x7D => true,
        0xA1 | 0xA7 | 0xAB | 0xB6..=0xB7 | 0xBB | 0xBF => true,
        0x2000..=0x206F => !matches!(u, 0x2044 | 0x2052),
        0x3000..=0x3004 | 0x3008..=0x301F | 0x3030..=0x3031 | 0x303D => true,
        // fullwidth mirrors of the ASCII punctuation
        0xFF01..=0xFF03 | 0xFF05..=0xFF0F | 0xFF1A..=0xFF1B | 0xFF1F..=0xFF20 | 0xFF3B..=0xFF3D | 0xFF3F
        | 0xFF5B | 0xFF5D..=0xFF65 => true,
        _ => false,
    }
}

/// \p{S} (symbol): exact for ASCII, range approximation beyond.
fn is_s(c: char) -> bool {
    let u = c as u32;
    match u {
        0x24 | 0x2B | 0x3C..=0x3E | 0x5E | 0x60 | 0x7C | 0x7E => true,
        0xA2..=0xA6 | 0xA8..=0xA9 | 0xAC | 0xAE..=0xB1 | 0xB4 | 0xB8 | 0xD7 | 0xF7 => true,
        0x20A0..=0x20BF | 0x2190..=0x23FF | 0x2500..=0x2BFF => true,
        // pictographs / emoticons / supplemental symbols planes
        0x1F000..=0x1FBFF => true,
        // fullwidth mirrors of the ASCII symbols + fullwidth currency
        0xFF04 | 0xFF0B | 0xFF1C..=0xFF1E | 0xFF3E | 0xFF40 | 0xFF5C | 0xFF7E | 0xFFE0..=0xFFE6 => true,
        _ => false,
    }
}

fn is_ps(c: char) -> bool {
    is_p(c) || is_s(c)
}

// ── pre-tokenizer ──

/// Stage 1+2: digit runs of 1-3 then CJK runs become isolated pieces.
fn split_isolated(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c.is_numeric() {
            let mut j = i;
            while j < n && j < i + 3 && chars[j].is_numeric() {
                j += 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
        } else {
            // gap until the next digit
            let mut j = i;
            while j < n && !chars[j].is_numeric() {
                j += 1;
            }
            // stage 2 inside the gap: CJK runs isolated
            let mut k = i;
            while k < j {
                if is_cjk(chars[k]) {
                    let mut e = k;
                    while e < j && is_cjk(chars[e]) {
                        e += 1;
                    }
                    out.push(chars[k..e].iter().collect());
                    k = e;
                } else {
                    let mut e = k;
                    while e < j && !is_cjk(chars[e]) {
                        e += 1;
                    }
                    out.push(chars[k..e].iter().collect());
                    k = e;
                }
            }
            i = j;
        }
    }
    out
}

/// Stage 3: the 6-alternative GPT-2-ish pattern. Alternatives are tried in
/// order at each position; unmatched chars accumulate as whole gap pieces.
fn stage3(piece: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = piece.chars().collect();
    let n = chars.len();
    let is_ws = |c: char| c.is_whitespace();
    let mut gap: Vec<char> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        let mut matched = 0usize; // length of the match at i, 0 = no match
        // alt 1: [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
        if i + 1 < n
            && matches!(c, '!'..='/' | ':'..='@' | '['..='`' | '{'..='~')
            && chars[i + 1].is_ascii_alphabetic()
        {
            let mut j = i + 1;
            while j < n && chars[j].is_ascii_alphabetic() {
                j += 1;
            }
            matched = j - i;
        }
        // alt 2: [^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+
        if matched == 0 {
            let prefix = if !matches!(c, '\r' | '\n') && !is_lm(c) && !is_ps(c) { 1 } else { 0 };
            let s = i + prefix;
            if s < n && is_lm(chars[s]) {
                let mut j = s;
                while j < n && is_lm(chars[j]) {
                    j += 1;
                }
                matched = j - i;
            }
        }
        // alt 3: ` ?[\p{P}\p{S}]+[\r\n]*`
        if matched == 0 {
            let s = if c == ' ' && i + 1 < n && is_ps(chars[i + 1]) { i + 1 } else { i };
            if s < n && is_ps(chars[s]) {
                let mut j = s;
                while j < n && is_ps(chars[j]) {
                    j += 1;
                }
                while j < n && matches!(chars[j], '\r' | '\n') {
                    j += 1;
                }
                matched = j - i;
            }
        }
        // alt 4-6: whitespace (\s*[\r\n]+ | \s+(?!\S) | \s+)
        if matched == 0 && is_ws(c) {
            let mut j = i;
            while j < n && is_ws(chars[j]) {
                j += 1;
            }
            // end the match at the last group of newlines if any (alt 4)
            let mut e = j;
            while e > i && !matches!(chars[e - 1], '\r' | '\n') {
                e -= 1;
            }
            if e > i && chars[i..e].iter().any(|&x| matches!(x, '\r' | '\n')) {
                matched = e - i;
            } else if j == n {
                matched = j - i; // alt 5: trailing whitespace
            } else if j - i > 1 {
                matched = j - i - 1; // alt 5: run minus its last char (followed by \S)
            } else {
                matched = 1; // alt 6
            }
        }
        if matched == 0 {
            gap.push(c);
            i += 1;
        } else {
            if !gap.is_empty() {
                out.push(gap.iter().collect());
                gap.clear();
            }
            out.push(chars[i..i + matched].iter().collect());
            i += matched;
        }
    }
    if !gap.is_empty() {
        out.push(gap.iter().collect());
    }
}

pub fn ds_pre_tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for piece in split_isolated(text) {
        stage3(&piece, &mut out);
    }
    out
}

// ── loading ──

impl DsTokenizer {
    pub fn load(path: &str) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
        let j = crate::json::parse(&bytes);
        let model = j.get("model").expect("tokenizer.json: model missing");
        let vocab = model.get("vocab").expect("tokenizer.json: model.vocab missing");
        let mut ids = HashMap::with_capacity(128_000);
        let mut id_to_bytes: Vec<Vec<u8>> = vec![Vec::new(); 129_280];
        if let crate::json::Json::Obj(pairs) = vocab {
            for (tok, id) in pairs {
                let id = id.as_num().unwrap() as u32;
                let b = byte_level_decode(tok);
                if (id as usize) < id_to_bytes.len() {
                    id_to_bytes[id as usize] = b.clone();
                }
                ids.insert(b, id);
            }
        }
        let mut pair_rank = HashMap::with_capacity(128_000);
        if let Some(crate::json::Json::Arr(merges)) = model.get("merges") {
            for (rank, m) in merges.iter().enumerate() {
                let s = m.as_str().unwrap();
                let (a, b) = s.split_once(' ').expect("tokenizer.json: bad merge");
                let mut merged = byte_level_decode(a);
                merged.extend_from_slice(&byte_level_decode(b));
                pair_rank.insert(merged, rank as u32);
            }
        }
        let mut added: Vec<(String, u32)> = Vec::new();
        if let Some(crate::json::Json::Arr(arr)) = j.get("added_tokens") {
            for t in arr {
                let id = t.get("id").and_then(|x| x.as_num()).unwrap() as u32;
                let content = t.get("content").and_then(|x| x.as_str()).unwrap();
                if (id as usize) < id_to_bytes.len() {
                    id_to_bytes[id as usize] = content.as_bytes().to_vec();
                }
                added.push((content.to_string(), id));
            }
        }
        // longest first → greedy matching (like the HF added-vocabulary trie)
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        DsTokenizer { ids, pair_rank, id_to_bytes, added }
    }

    /// BPE: minimal-rank merge (pair rank = position in the merges list).
    fn bpe(&self, chunk: &[u8]) -> Vec<u32> {
        let mut word: Vec<Vec<u8>> = chunk.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for k in 0..word.len().saturating_sub(1) {
                let mut pair = word[k].clone();
                pair.extend_from_slice(&word[k + 1]);
                if let Some(&rank) = self.pair_rank.get(&pair) {
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
        word.iter().filter_map(|t| self.ids.get(t).copied()).collect()
    }

    /// Encode ordinary text. Like the HF runtime, added (special) tokens are
    /// matched greedily (longest first) as literal substrings; the spans
    /// between them go through the pre-tokenizer + BPE pipeline.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut seg_start = 0usize;
        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < text.len() {
            // added-token match only plausible on '<' or '｜' (covers the whole V4 list)
            let c = bytes[i];
            if c == b'<' || (c == 0xEF && i + 2 < text.len() && bytes[i + 1] == 0xBD && bytes[i + 2] == 0x9C) {
                let rest = &text[i..];
                if let Some((content, id)) = self.added.iter().find(|(c, _)| rest.starts_with(c.as_str())) {
                    if seg_start < i {
                        for chunk in ds_pre_tokenize(&text[seg_start..i]) {
                            ids.extend(self.bpe(chunk.as_bytes()));
                        }
                    }
                    ids.push(*id);
                    i += content.len();
                    seg_start = i;
                    continue;
                }
            }
            i += text[i..].chars().next().unwrap().len_utf8();
        }
        if seg_start < text.len() {
            for chunk in ds_pre_tokenize(&text[seg_start..]) {
                ids.extend(self.bpe(chunk.as_bytes()));
            }
        }
        ids
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

    /// Simplified V4 chat template (encoding_dsv4.py, chat mode, no tools):
    /// BOS, then per history turn <｜User｜>q <｜Assistant｜></think> a EOS,
    /// then <｜User｜>q <｜Assistant｜></think> as the generation prompt.
    pub fn encode_chat(&self, history: &[(String, String)], question: &str) -> Vec<u32> {
        let mut ids = vec![DS_BOS];
        for (q, a) in history {
            ids.push(DS_USER);
            ids.extend(self.encode(q));
            ids.push(DS_ASSISTANT);
            ids.push(DS_THINK_END);
            ids.extend(self.encode(a));
            ids.push(DS_EOS);
        }
        ids.push(DS_USER);
        ids.extend(self.encode(question));
        ids.push(DS_ASSISTANT);
        ids.push(DS_THINK_END);
        ids
    }
}
