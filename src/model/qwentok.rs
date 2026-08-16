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
    /// Sliced-vocabulary remap (qwen.vocabmap.json beside the model).
    remap: Option<QwenVocabRemap>,
}

/// Remap for a vocabulary-sliced model: encoding runs on the full
/// vocabulary, then every id is translated to its kept row; a dropped
/// token falls back to its single-byte tokens, which the slicer always
/// keeps (byte-level BPE covers any byte sequence with them).
pub struct QwenVocabRemap {
    pub new_to_old: Vec<u32>,
    old_to_new: Vec<u32>,
    byte_new_ids: [u32; 256],
}

impl QwenVocabRemap {
    /// Builds the remap from the sidecar's new -> old table, resolving the
    /// 256 single-byte tokens through the full tokenizer. Fails closed on
    /// out-of-range entries, duplicates, or a dropped byte token.
    pub fn build(new_to_old: Vec<u32>, full: &QwenTokenizer) -> Result<QwenVocabRemap, String> {
        let full_vocab = full.id_to_bytes.len();
        let mut old_to_new = vec![u32::MAX; full_vocab];
        for (new_id, &old_id) in new_to_old.iter().enumerate() {
            if old_id as usize >= full_vocab {
                return Err(format!("vocab map: id {} exceeds the full vocabulary", old_id));
            }
            if old_to_new[old_id as usize] != u32::MAX {
                return Err(format!("vocab map: duplicate id {}", old_id));
            }
            old_to_new[old_id as usize] = new_id as u32;
        }
        let mut byte_new_ids = [0u32; 256];
        for b in 0..256usize {
            let full_id = *full
                .ids
                .get(&vec![b as u8])
                .ok_or_else(|| format!("tokenizer has no single-byte token for byte {}", b))?;
            let new_id = old_to_new[full_id as usize];
            if new_id == u32::MAX {
                return Err(format!("vocab map drops the byte token for byte {}", b));
            }
            byte_new_ids[b] = new_id;
        }
        Ok(QwenVocabRemap { new_to_old, old_to_new, byte_new_ids })
    }
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
            remap: None,
        }
    }

    /// Attaches a sliced-vocabulary remap; ids produced and consumed by
    /// this tokenizer then live in the sliced space of `sliced_vocab`.
    pub fn attach_remap(&mut self, new_to_old: Vec<u32>, sliced_vocab: usize) -> Result<(), String> {
        if new_to_old.len() != sliced_vocab {
            return Err("vocab map length does not match the sliced vocabulary".to_string());
        }
        let remap = QwenVocabRemap::build(new_to_old, self)?;
        self.model_vocab = sliced_vocab;
        self.remap = Some(remap);
        Ok(())
    }

    /// Translates full-vocabulary ids into the sliced space, expanding a
    /// dropped token into its kept single-byte tokens. Identity without a
    /// remap.
    fn map_out(&self, ids: Vec<u32>) -> Vec<u32> {
        let Some(remap) = &self.remap else { return ids };
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            let mapped = remap.old_to_new.get(id as usize).copied().unwrap_or(u32::MAX);
            if mapped != u32::MAX {
                out.push(mapped);
            } else {
                for &b in &self.id_to_bytes[id as usize] {
                    out.push(remap.byte_new_ids[b as usize]);
                }
            }
        }
        out
    }

    /// Full-vocabulary id of the single-byte token for `b` (byte-level BPE
    /// has one for every byte). The vocabulary slicer keeps all of them so
    /// dropped tokens can fall back to bytes.
    pub fn single_byte_full_id(&self, b: u8) -> Option<u32> {
        self.ids.get(&vec![b]).copied()
    }

    /// A special token's id in the tokenizer's output space.
    fn special_out(&self, full_id: u32) -> u32 {
        match &self.remap {
            Some(remap) => remap.old_to_new[full_id as usize],
            None => full_id,
        }
    }

    /// Generation stop in chat mode (im_end), in the output space.
    pub fn stop_end_of_msg(&self) -> u32 {
        self.special_out(QWEN_IM_END)
    }

    /// Generation stop in raw completion mode, in the output space.
    pub fn stop_endoftext(&self) -> u32 {
        self.special_out(QWEN_ENDOFTEXT)
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
        let ids = self.encode_full(text);
        self.map_out(ids)
    }

    /// Full-vocabulary encoding (before any sliced remap).
    fn encode_full(&self, text: &str) -> Vec<u32> {
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
        let id = match &self.remap {
            Some(remap) => remap.new_to_old.get(id as usize).copied().unwrap_or(id),
            None => id,
        };
        self.id_to_bytes
            .get(id as usize)
            .filter(|bytes| !bytes.is_empty())
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_else(|| format!("<|qwen_unused_{}|>", id))
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Raw decoded bytes (before UTF-8 replacement): streaming callers
    /// buffer these and flush only complete UTF-8 prefixes, so a token
    /// boundary inside a multibyte character never garbles the stream.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &id in ids {
            let id = match &self.remap {
                Some(remap) => remap.new_to_old.get(id as usize).copied().unwrap_or(id),
                None => id,
            };
            if let Some(value) = self.id_to_bytes.get(id as usize) {
                bytes.extend_from_slice(value);
            }
        }
        bytes
    }

    pub fn vocab_size(&self) -> usize {
        self.model_vocab
    }

    /// Text-only, no-tools form of the checkpoint chat template. Historical
    /// assistant reasoning is omitted, while a fresh generation starts the
    /// default thinking block exactly like tokenizer_config.json.
    pub fn encode_chat(&self, history: &[(String, String)], question: &str) -> Vec<u32> {
        self.encode_chat_system(None, history, question)
    }

    /// `encode_chat` with an optional leading system block, exactly like
    /// tokenizer_config.json renders one:
    /// `<|im_start|>system\n{content}<|im_end|>\n`.
    pub fn encode_chat_system(
        &self,
        system: Option<&str>,
        history: &[(String, String)],
        question: &str,
    ) -> Vec<u32> {
        self.encode_chat_prompt(system, history, question, true)
    }

    /// Full chat prompt with the thinking toggle: `thinking == false`
    /// renders the template's disabled block `<think>\n\n</think>\n\n`
    /// after the assistant header, exactly like
    /// `enable_thinking=false` in tokenizer_config.json.
    pub fn encode_chat_prompt(
        &self,
        system: Option<&str>,
        history: &[(String, String)],
        question: &str,
        thinking: bool,
    ) -> Vec<u32> {
        self.encode_chat_split(system, history, question, thinking).0
    }

    /// `encode_chat_prompt` + the index where the final assistant priming
    /// begins. Everything before that index is the conversation prefix a
    /// future turn re-renders verbatim, so a prefix-cache entry stored at
    /// the split survives across turns. With `thinking == false`, history
    /// assistant turns replay the disabled think block exactly as it was
    /// ingested when they were generated, keeping the token stream an
    /// exact extension (the reference template strips it, which breaks
    /// stream-prefix reuse; the empty block is what the model actually
    /// saw, and it is three tokens per turn).
    pub fn encode_chat_split(
        &self,
        system: Option<&str>,
        history: &[(String, String)],
        question: &str,
        thinking: bool,
    ) -> (Vec<u32>, usize) {
        let mut ids = Vec::new();
        if let Some(content) = system {
            ids.push(QWEN_IM_START);
            ids.extend(self.encode_full("system\n"));
            ids.extend(self.encode_full(content.trim()));
            ids.push(QWEN_IM_END);
            ids.extend(self.encode_full("\n"));
        }
        for (q, answer) in history {
            ids.push(QWEN_IM_START);
            ids.extend(self.encode_full("user\n"));
            ids.extend(self.encode_full(q.trim()));
            ids.push(QWEN_IM_END);
            ids.extend(self.encode_full("\n"));

            let trimmed = answer.trim();
            let visible = trimmed
                .rsplit_once("</think>")
                .map(|(_, rest)| rest.trim_start_matches('\n'))
                .unwrap_or(trimmed);
            ids.push(QWEN_IM_START);
            ids.extend(self.encode_full("assistant\n"));
            if !thinking {
                self.push_disabled_think(&mut ids);
            }
            ids.extend(self.encode_full(visible));
            ids.push(QWEN_IM_END);
            ids.extend(self.encode_full("\n"));
        }
        ids.push(QWEN_IM_START);
        ids.extend(self.encode_full("user\n"));
        ids.extend(self.encode_full(question.trim()));
        ids.push(QWEN_IM_END);
        ids.extend(self.encode_full("\n"));
        let split = self.map_out(ids.clone()).len();
        ids.push(QWEN_IM_START);
        ids.extend(self.encode_full("assistant\n"));
        if thinking {
            ids.push(QWEN_THINK);
            ids.extend(self.encode_full("\n"));
        } else {
            self.push_disabled_think(&mut ids);
        }
        (self.map_out(ids), split)
    }

    /// The template's disabled reasoning block `<think>\n\n</think>\n\n`.
    fn push_disabled_think(&self, ids: &mut Vec<u32>) {
        ids.push(QWEN_THINK);
        ids.extend(self.encode_full("\n\n"));
        ids.extend(self.encode_full("</think>"));
        ids.extend(self.encode_full("\n\n"));
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

    /// Synthetic byte-level tokenizer: 256 single-byte tokens at their
    /// byte value, "ab" at 256, "cd" at 257.
    fn byte_level_fixture() -> QwenTokenizer {
        let mut ids = HashMap::new();
        let mut id_to_bytes: Vec<Vec<u8>> = Vec::new();
        for b in 0..=255u8 {
            ids.insert(vec![b], b as u32);
            id_to_bytes.push(vec![b]);
        }
        ids.insert(b"ab".to_vec(), 256);
        id_to_bytes.push(b"ab".to_vec());
        ids.insert(b"cd".to_vec(), 257);
        id_to_bytes.push(b"cd".to_vec());
        QwenTokenizer {
            ids,
            pair_rank: HashMap::new(),
            id_to_bytes,
            added: Vec::new(),
            model_vocab: 258,
            remap: None,
        }
    }

    #[test]
    fn sliced_remap_falls_back_to_byte_tokens() {
        let mut tok = byte_level_fixture();
        // keep every byte token plus "ab"; drop "cd"
        let new_to_old: Vec<u32> = (0..=255).chain([256]).collect();
        tok.attach_remap(new_to_old.clone(), 257).unwrap();
        assert_eq!(tok.vocab_size(), 257);
        // kept token maps to its new row; dropped token becomes bytes
        assert_eq!(tok.map_out(vec![256, 257]), vec![256, b'c' as u32, b'd' as u32]);
        // decoding round-trips through the new ids
        assert_eq!(tok.decode(&[256, b'c' as u32, b'd' as u32]), "abcd");
    }

    #[test]
    fn sliced_remap_refuses_a_dropped_byte_token() {
        let mut tok = byte_level_fixture();
        // dropping byte 7 makes the fallback unsound: fail closed
        let new_to_old: Vec<u32> = (0..=255).filter(|&b| b != 7).chain([256, 257]).collect();
        assert!(tok.attach_remap(new_to_old, 257).is_err());
    }

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
            remap: None,
        };
        assert_eq!(tokenizer.bpe(b"abc"), vec![3, 2]);
    }
}
