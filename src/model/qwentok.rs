//! Qwen3.5 byte-level BPE tokenizer and text-only chat template.
//!
//! The Qwen2-compatible pre-tokenizer selected by Transformers is
//! reimplemented without dependencies: contractions, optional-prefix letter
//! runs, one numeral per piece, punctuation/symbol runs, and
//! whitespace/newline alternatives. The tokenizer's NFC normalizer is
//! provided by the crate's generated Unicode tables, keeping decomposed and
//! composed input behavior exact without a runtime dependency.

use std::collections::{HashMap, HashSet};

pub const QWEN_ENDOFTEXT: u32 = 248_044;
pub const QWEN_IM_START: u32 = 248_045;
pub const QWEN_IM_END: u32 = 248_046;
pub const QWEN_THINK: u32 = 248_068;

pub struct QwenTokenizer {
    ids: HashMap<Vec<u8>, u32>,
    pair_rank: HashMap<(Vec<u8>, Vec<u8>), u32>,
    id_to_bytes: Vec<Vec<u8>>,
    added: Vec<(String, u32)>,
    model_vocab: usize,
}

fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut extra = 0u32;
    for b in 0..256u32 {
        let printable = matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
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

fn byte_level_decode(value: &str) -> Vec<u8> {
    let table = byte_to_char_table();
    let inverse: HashMap<char, u8> = table
        .iter()
        .enumerate()
        .map(|(byte, &c)| (c, byte as u8))
        .collect();
    value
        .chars()
        .map(|c| inverse.get(&c).copied())
        .collect::<Option<Vec<u8>>>()
        .unwrap_or_else(|| value.as_bytes().to_vec())
}

fn contraction(chars: &[char], i: usize) -> usize {
    if chars.get(i) != Some(&'\'') {
        return 0;
    }
    for suffix in ["re", "ve", "ll", "s", "t", "m", "d"] {
        let n = suffix.len();
        if i + 1 + n <= chars.len()
            && chars[i + 1..i + 1 + n]
                .iter()
                .collect::<String>()
                .eq_ignore_ascii_case(suffix)
        {
            return n + 1;
        }
    }
    0
}

/// Exact control flow of the seven-alternative Qwen2 compatibility regex,
/// using Unicode general categories `L` and `N`.
pub fn qwen_pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let c = chars[i];
        let mut matched = contraction(&chars, i);

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if matched == 0 {
            let prefix = usize::from(
                !matches!(c, '\r' | '\n')
                    && !crate::unicode_nfc::is_letter(c)
                    && !crate::unicode_nfc::is_number(c),
            );
            let start = i + prefix;
            if start < chars.len() && crate::unicode_nfc::is_letter(chars[start]) {
                let mut end = start + 1;
                while end < chars.len() && crate::unicode_nfc::is_letter(chars[end]) {
                    end += 1;
                }
                matched = end - i;
            }
        }

        // \p{N}: one numeral per piece.
        if matched == 0 && crate::unicode_nfc::is_number(c) {
            matched = 1;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        if matched == 0 {
            let special = |x: char| {
                !x.is_whitespace()
                    && !crate::unicode_nfc::is_letter(x)
                    && !crate::unicode_nfc::is_number(x)
            };
            let start = if c == ' ' && i + 1 < chars.len() && special(chars[i + 1]) {
                i + 1
            } else {
                i
            };
            if start < chars.len() && special(chars[start]) {
                let mut end = start + 1;
                while end < chars.len() && special(chars[end]) {
                    end += 1;
                }
                while end < chars.len() && matches!(chars[end], '\r' | '\n') {
                    end += 1;
                }
                matched = end - i;
            }
        }

        // \s*[\r\n]+ | \s+(?!\S) | \s+
        if matched == 0 && c.is_whitespace() {
            let mut end = i + 1;
            while end < chars.len() && chars[end].is_whitespace() {
                end += 1;
            }
            let mut through_newline = end;
            while through_newline > i && !matches!(chars[through_newline - 1], '\r' | '\n') {
                through_newline -= 1;
            }
            if through_newline > i
                && chars[i..through_newline]
                    .iter()
                    .any(|x| matches!(x, '\r' | '\n'))
            {
                matched = through_newline - i;
            } else if end == chars.len() {
                matched = end - i;
            } else if end - i > 1 {
                matched = end - i - 1;
            } else {
                matched = 1;
            }
        }

        // The regex is exhaustive. Keeping a one-character fallback makes a
        // future Unicode category expansion fail softly rather than dropping.
        if matched == 0 {
            matched = 1;
        }
        out.push(chars[i..i + matched].iter().collect());
        i += matched;
    }
    out
}

impl QwenTokenizer {
    pub fn load(path: &str, model_vocab: usize) -> QwenTokenizer {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
        let root = crate::json::parse_complete(&bytes);
        let model = root.get("model").expect("tokenizer.json: model missing");
        assert_eq!(
            model.get("type").and_then(|x| x.as_str()),
            Some("BPE"),
            "tokenizer.json: expected BPE"
        );
        let crate::json::Json::Obj(vocab) = model
            .get("vocab")
            .expect("tokenizer.json: model.vocab missing")
        else {
            panic!("tokenizer.json: model.vocab must be an object");
        };
        let mut seen_tokens = HashSet::new();
        let mut seen_ids = HashSet::new();
        let mut ids = HashMap::with_capacity(vocab.len());
        let mut id_to_bytes = vec![Vec::new(); model_vocab];
        for (token, id) in vocab {
            assert!(
                seen_tokens.insert(token),
                "tokenizer.json: duplicate vocab token"
            );
            let id = id.as_num().expect("tokenizer.json: non-numeric id") as u32;
            assert!(
                (id as usize) < model_vocab,
                "tokenizer token {} exceeds model vocab",
                id
            );
            assert!(
                seen_ids.insert(id),
                "tokenizer.json: duplicate vocab id {}",
                id
            );
            let decoded = byte_level_decode(token);
            id_to_bytes[id as usize] = decoded.clone();
            ids.insert(decoded, id);
        }

        let merges = model
            .get("merges")
            .and_then(|x| x.as_arr())
            .expect("tokenizer.json: model.merges missing");
        let mut pair_rank = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let merge = merge
                .as_str()
                .expect("tokenizer.json: merge must be a string");
            let (left, right) = merge
                .split_once(' ')
                .expect("tokenizer.json: malformed merge");
            let pair = (byte_level_decode(left), byte_level_decode(right));
            assert!(
                pair_rank.insert(pair, rank as u32).is_none(),
                "tokenizer.json: duplicate merge"
            );
        }

        let mut added = Vec::new();
        if let Some(values) = root.get("added_tokens").and_then(|x| x.as_arr()) {
            for value in values {
                let id = value
                    .get("id")
                    .and_then(|x| x.as_num())
                    .expect("tokenizer.json: added token id missing")
                    as u32;
                let content = value
                    .get("content")
                    .and_then(|x| x.as_str())
                    .expect("tokenizer.json: added token content missing");
                assert!(
                    (id as usize) < model_vocab,
                    "added token {} exceeds model vocab",
                    id
                );
                id_to_bytes[id as usize] = content.as_bytes().to_vec();
                added.push((content.to_string(), id));
            }
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
        for (id, expected) in [
            (QWEN_ENDOFTEXT, "<|endoftext|>"),
            (QWEN_IM_START, "<|im_start|>"),
            (QWEN_IM_END, "<|im_end|>"),
            (QWEN_THINK, "<think>"),
        ] {
            assert_eq!(
                &id_to_bytes[id as usize],
                expected.as_bytes(),
                "tokenizer special id {} mismatch",
                id
            );
        }
        QwenTokenizer {
            ids,
            pair_rank,
            id_to_bytes,
            added,
            model_vocab,
        }
    }

    fn bpe(&self, bytes: &[u8]) -> Vec<u32> {
        let mut word: Vec<Vec<u8>> = bytes.iter().map(|&b| vec![b]).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..word.len().saturating_sub(1) {
                if let Some(&rank) = self.pair_rank.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.map_or(true, |(old, _)| rank < old) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, selected)) = best else {
                break;
            };
            let left = word[selected].clone();
            let right = word[selected + 1].clone();
            let mut merged_token = left.clone();
            merged_token.extend_from_slice(&right);
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0usize;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
                    merged.push(merged_token.clone());
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
        }
        word.iter()
            .map(|token| {
                self.ids
                    .get(token)
                    .copied()
                    .unwrap_or_else(|| panic!("tokenizer.json lacks a byte-level token"))
            })
            .collect()
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for piece in qwen_pre_tokenize(text) {
            out.extend(self.bpe(piece.as_bytes()));
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = crate::unicode_nfc::normalize_nfc(text);
        let text = normalized.as_str();
        let mut out = Vec::new();
        let mut segment = 0usize;
        let mut i = 0usize;
        while i < text.len() {
            let rest = &text[i..];
            if let Some((content, id)) = self
                .added
                .iter()
                .find(|(content, _)| rest.starts_with(content))
            {
                if segment < i {
                    self.encode_plain(&text[segment..i], &mut out);
                }
                out.push(*id);
                i += content.len();
                segment = i;
            } else {
                i += rest.chars().next().unwrap().len_utf8();
            }
        }
        if segment < text.len() {
            self.encode_plain(&text[segment..], &mut out);
        }
        out
    }

    pub fn decode_id(&self, id: u32) -> String {
        self.id_to_bytes
            .get(id as usize)
            .filter(|bytes| !bytes.is_empty())
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_else(|| format!("<|qwen_unused_{}|>", id))
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(value) = self.id_to_bytes.get(id as usize) {
                bytes.extend_from_slice(value);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.model_vocab
    }

    /// Text-only, no-tools form of the checkpoint chat template. Historical
    /// assistant reasoning is omitted, while a fresh generation starts the
    /// default thinking block exactly like tokenizer_config.json.
    pub fn encode_chat(&self, history: &[(String, String)], question: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for (q, answer) in history {
            ids.push(QWEN_IM_START);
            ids.extend(self.encode("user\n"));
            ids.extend(self.encode(q.trim()));
            ids.push(QWEN_IM_END);
            ids.extend(self.encode("\n"));

            let trimmed = answer.trim();
            let visible = trimmed
                .rsplit_once("</think>")
                .map(|(_, rest)| rest.trim_start_matches('\n'))
                .unwrap_or(trimmed);
            ids.push(QWEN_IM_START);
            ids.extend(self.encode("assistant\n"));
            ids.extend(self.encode(visible));
            ids.push(QWEN_IM_END);
            ids.extend(self.encode("\n"));
        }
        ids.push(QWEN_IM_START);
        ids.extend(self.encode("user\n"));
        ids.extend(self.encode(question.trim()));
        ids.push(QWEN_IM_END);
        ids.extend(self.encode("\n"));
        ids.push(QWEN_IM_START);
        ids.extend(self.encode("assistant\n"));
        ids.push(QWEN_THINK);
        ids.extend(self.encode("\n"));
        ids
    }
}

/// Hidden tokenizer parity helper for comparing tokenizer.json behavior.
pub fn dump_cmd(args: &[String]) {
    let value = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let path = value("--vocab").expect("qwen-tok requires --vocab tokenizer.json");
    let text = value("--text").unwrap_or_default();
    let vocab = value("--model-vocab")
        .and_then(|value| value.parse().ok())
        .unwrap_or(248_320usize);
    let tokenizer = QwenTokenizer::load(&path, vocab);
    let ids = if args.iter().any(|arg| arg == "--chat") {
        tokenizer.encode_chat(&[], &text)
    } else {
        tokenizer.encode(&text)
    };
    println!(
        "{}",
        ids.iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_pretokenizer_follows_qwen_alternatives() {
        assert_eq!(
            qwen_pre_tokenize("Hello, world! 123\nnext"),
            vec!["Hello", ",", " world", "!", " ", "1", "2", "3", "\n", "next"]
        );
        assert_eq!(
            qwen_pre_tokenize("WE'RE can't"),
            vec!["WE", "'RE", " can", "'t"]
        );
        assert_eq!(qwen_pre_tokenize(" देवनागरी"), vec![" द", "ेवन", "ागर", "ी"]);
    }

    #[test]
    fn pretokenizer_never_drops_utf8_text() {
        for text in ["", "naïve café", "中文🙂", "a\r\n  b", "e\u{301}"] {
            assert_eq!(qwen_pre_tokenize(text).concat(), text);
        }
    }

    #[test]
    fn bpe_merge_rank_keeps_pair_boundaries() {
        let tokenizer = QwenTokenizer {
            ids: [
                (b"a".to_vec(), 0),
                (b"b".to_vec(), 1),
                (b"c".to_vec(), 2),
                (b"ab".to_vec(), 3),
                (b"abc".to_vec(), 4),
            ]
            .into_iter()
            .collect(),
            pair_rank: [
                ((b"a".to_vec(), b"b".to_vec()), 0),
                ((b"b".to_vec(), b"c".to_vec()), 1),
                ((b"a".to_vec(), b"bc".to_vec()), 2),
            ]
            .into_iter()
            .collect(),
            id_to_bytes: Vec::new(),
            added: Vec::new(),
            model_vocab: 5,
        };
        assert_eq!(tokenizer.bpe(b"abc"), vec![3, 2]);
    }
}
