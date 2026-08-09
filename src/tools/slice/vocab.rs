// Vocabulary pruning (--vocab-top / --vocab-list): keep-set selection and runtime remap (moved from slice.rs).

use super::score::top_n;
use crate::config::Config;

// ── vocabulary pruning (--vocab-top / --vocab-list) ──

/// Special tokens the runtime remap must carry (NanoTokenizer::load unwraps
/// all of them but pad).
const SPECIAL_NAMES: [&str; 8] = ["bos", "eos", "open", "close", "sep", "end_of_msg", "unk", "pad"];

/// Id of a special token in the FULL Kimi vocabulary (163584+ reserved block).
pub(super) fn kimi_special_id(name: &str) -> Option<u32> {
    use crate::tokenizer as t;
    Some(match name {
        "bos" => t::BOS,
        "eos" => t::EOS,
        "open" => t::OPEN,
        "close" => t::CLOSE,
        "sep" => t::SEP,
        "end_of_msg" => t::END_OF_MSG,
        "unk" => t::UNK,
        "pad" => t::PAD,
        _ => return None,
    })
}

/// Old-vocab ids of every known special token: the source config "specials"
/// block first (nano models carry all 8), completed with the full-Kimi
/// constants when the vocab covers the reserved block [NUM_BASE, vocab).
/// Conservative by design: anything structural is kept, never ranked.
pub(super) fn known_specials(source_json: &crate::json::Json, cfg: &Config) -> Vec<(String, u32)> {
    let mut out: Vec<(String, u32)> = Vec::new();
    if let Some(crate::json::Json::Obj(pairs)) = source_json.get("specials") {
        for (k, v) in pairs {
            if let (Some(_), Some(id)) = (kimi_special_id(k), v.as_num()) {
                out.push((k.clone(), id as u32));
            }
        }
    }
    if cfg.vocab > crate::tokenizer::NUM_BASE as usize {
        for n in SPECIAL_NAMES {
            if !out.iter().any(|(k, _)| k == n) {
                out.push((n.to_string(), kimi_special_id(n).unwrap()));
            }
        }
    }
    // the config-level bos / end_of_msg are structural: never drop them
    for (n, id) in [("bos", cfg.bos_id), ("end_of_msg", cfg.eos_id)] {
        if !out.iter().any(|(k, _)| k == n) {
            out.push((n.to_string(), id));
        }
    }
    out.retain(|&(_, id)| (id as usize) < cfg.vocab);
    out
}

/// Parses a freqfile into per-id counts (len = vocab). Text format:
/// "<token_id> <count>" per line ('#' comments and blank lines ignored); a
/// JSON object {"<id>": <count>, ...} is also accepted. Ids index the model's
/// CURRENT vocabulary: an out-of-range id means the freqfile was built for
/// another model and the slice would silently corrupt the embeddings.
pub(super) fn parse_freqfile(path: &str, vocab: usize) -> Vec<u64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("freqfile {} unreadable: {}", path, e));
    let mut counts = vec![0u64; vocab];
    let mut put = |id: usize, c: u64, what: &str| {
        assert!(
            id < vocab,
            "freqfile {}: token id {} out of range (model vocab is {}) - {}",
            path, id, vocab, what
        );
        counts[id] = c;
    };
    if text.trim_start().starts_with('{') {
        if let crate::json::Json::Obj(pairs) = crate::json::parse(text.as_bytes()) {
            for (k, v) in pairs {
                let id: usize = k.parse().unwrap_or_else(|_| panic!("freqfile {}: bad object key '{}'", path, k));
                let c = v.as_num().unwrap_or_else(|| panic!("freqfile {}: bad count for id {}", path, id));
                put(id, c as u64, "the freqfile must count the ids of the model's CURRENT vocabulary");
            }
        } else {
            panic!("freqfile {}: expected a flat JSON object {{\"<id>\": <count>, ...}}", path);
        }
    } else {
        for (ln, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap().trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let (Some(id), Some(c)) = (it.next(), it.next()) else {
                panic!("freqfile {}:{}: expected '<token_id> <count>'", path, ln + 1);
            };
            let id: usize = id.parse().unwrap_or_else(|_| panic!("freqfile {}:{}: bad id '{}'", path, ln + 1, id));
            let c: u64 = c.parse().unwrap_or_else(|_| panic!("freqfile {}:{}: bad count '{}'", path, ln + 1, c));
            put(id, c, "the freqfile must count the ids of the model's CURRENT vocabulary");
        }
    }
    counts
}

/// Old row id -> kimi id table of an ALREADY remapped source vocab (e.g. the
/// nano 8200): explicit --vocab-base, else vocab_nano.json next to the source
/// model with a matching vocab_size. None = identity (full Kimi vocab).
pub(super) fn base_vocab_map(model: &str, base_flag: Option<String>, vocab: usize) -> Option<(crate::json::Json, Vec<u32>)> {
    let try_load = |p: &str| -> Option<(crate::json::Json, Vec<u32>)> {
        let bytes = std::fs::read(p).ok()?;
        let j = crate::json::parse(&bytes);
        let vs = j.get("vocab_size").and_then(|x| x.as_num()).map(|n| n as usize);
        if vs != Some(vocab) {
            return None;
        }
        let m: Vec<u32> = j.get("nano_to_kimi")?.as_arr()?.iter().map(|x| x.as_num().unwrap() as u32).collect();
        Some((j, m))
    };
    if let Some(p) = base_flag {
        return Some(
            try_load(&p).unwrap_or_else(|| panic!("--vocab-base {}: not a vocab remap matching the source vocab {}", p, vocab)),
        );
    }
    let dir = std::path::Path::new(model).parent().unwrap_or(std::path::Path::new("."));
    let cand = dir.join("vocab_nano.json");
    if let Some(bm) = cand.to_str().and_then(&try_load) {
        println!("vocab: composing through the base remap {}", cand.display());
        return Some(bm);
    }
    assert!(
        vocab > crate::tokenizer::NUM_BASE as usize,
        "vocab pruning on an already remapped vocab ({} ids) needs --vocab-base <remap.json> \
(or a vocab_nano.json with a matching vocab_size next to the source model) to map rows back to kimi ids",
        vocab
    );
    None
}

/// Result of the --vocab-top / --vocab-list selection: the kept rows plus
/// the runtime remap file (engine --vocab compatible) to write next to the .bin.
pub(super) struct VocabPlan {
    pub(super) keep: Vec<usize>,                 // old row ids kept, ascending
    pub(super) specials_new: Vec<(String, u32)>, // name -> NEW id (ascending old order)
    pub(super) remap_path: String,
    pub(super) remap_json: String,
}

/// Where the vocab keep-set comes from: the N most frequent ids of a
/// freqfile (--vocab-top), or an explicit id list file (--vocab-list, one
/// subject token id per line, '#' comments allowed - nano/vocab_cross.py
/// writes these). Either way the specials/reserved rows are force-kept on
/// top (in doubt, keep).
#[derive(Debug, PartialEq)]
pub(super) enum VocabSelect {
    TopN(usize, String), // N rows + freqfile path
    List(String),        // id list file path
}

/// Parses an id list file (--vocab-list): one token id per line, '#'
/// comments and blank lines ignored. Ids index the model's CURRENT
/// vocabulary, same contract as the freqfile.
pub(super) fn parse_id_list(path: &str, vocab: usize) -> Vec<usize> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("vocab list {} unreadable: {}", path, e));
    let mut ids = Vec::new();
    for (ln, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let id: usize = line.parse().unwrap_or_else(|_| panic!("vocab list {}:{}: expected one token id per line, got '{}'", path, ln + 1, line));
        assert!(
            id < vocab,
            "vocab list {}: token id {} out of range (model vocab is {}) - the list must index the model's CURRENT vocabulary",
            path, id, vocab
        );
        ids.push(id);
    }
    assert!(!ids.is_empty(), "vocab list {}: no token ids", path);
    ids
}

/// Top-N by frequency OR an explicit id list, + all specials/reserved ids,
/// and the remap JSON.
pub(super) fn build_vocab_plan(model: &str, out: &str, source_json: &crate::json::Json, cfg: &Config, select: &VocabSelect, base_flag: Option<String>) -> VocabPlan {
    let (mut keep, sel_desc) = match select {
        VocabSelect::TopN(n_top, freq_path) => {
            let counts = parse_freqfile(freq_path, cfg.vocab);
            let scores: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
            let keep = top_n(&scores, (*n_top).min(cfg.vocab));
            let total: u64 = counts.iter().sum();
            let covered: u64 = keep.iter().map(|&j| counts[j]).sum();
            let desc = format!("top-{} by frequency ({:.2}% of the counted token mass)", keep.len(), covered as f64 / total.max(1) as f64 * 100.0);
            (keep, desc)
        }
        VocabSelect::List(list_path) => {
            let ids = parse_id_list(list_path, cfg.vocab);
            let desc = format!("{} ids from {}", ids.len(), list_path);
            (ids, desc)
        }
    };
    keep.sort_unstable();
    keep.dedup();
    let n_sel = keep.len();
    let specials_old = known_specials(source_json, cfg);
    // specials are never ranked: force-keep them all (in doubt, keep)
    for &(_, id) in &specials_old {
        keep.push(id as usize);
    }
    // full Kimi vocab: the whole reserved block [NUM_BASE, vocab) stays
    if cfg.vocab > crate::tokenizer::NUM_BASE as usize {
        keep.extend(crate::tokenizer::NUM_BASE as usize..cfg.vocab);
    }
    keep.sort_unstable();
    keep.dedup();
    println!(
        "vocab: keeping {}/{} rows ({} + {} special/reserved)",
        keep.len(),
        cfg.vocab,
        sel_desc,
        keep.len() - n_sel,
    );

    // old row -> kimi id (through the base remap when the source is remapped)
    let base = base_vocab_map(model, base_flag, cfg.vocab);
    let kimi_of = |old: usize| -> u32 {
        match &base {
            None => old as u32,
            Some((_, m)) => {
                if old < m.len() {
                    m[old]
                } else {
                    // special row of the base vocab: back to the kimi constant
                    specials_old
                        .iter()
                        .find(|&(_, id)| *id as usize == old)
                        .and_then(|(n, _)| kimi_special_id(n))
                        .unwrap_or(crate::tokenizer::UNK)
                }
            }
        }
    };
    let specials_new: Vec<(String, u32)> = specials_old
        .iter()
        .map(|(n, id)| {
            let new = keep.binary_search(&(*id as usize)).expect("special token missing from the keep-set") as u32;
            (n.clone(), new)
        })
        .collect();
    for req in ["bos", "eos", "open", "close", "sep", "end_of_msg", "unk"] {
        assert!(
            specials_new.iter().any(|(n, _)| n == req),
            "vocab pruning: no '{}' id found (source config specials + kimi constants) - the runtime remap would be unloadable",
            req
        );
    }
    let nano_to_kimi: Vec<u32> = keep.iter().map(|&j| kimi_of(j)).collect();
    let specials_json = specials_new
        .iter()
        .map(|(n, id)| format!("\"{}\": {}", n, id))
        .collect::<Vec<_>>()
        .join(", ");
    let remap_json = format!(
        "{{\n \"format\": \"microkimi-vocab-remap-1\",\n \"source_vocab\": {},\n \"vocab_size\": {},\n \"nano_to_kimi\": [{}],\n \"specials\": {{{}}},\n \"kimi_special_ids\": {{\"open\": {}, \"close\": {}, \"sep\": {}, \"end_of_msg\": {}}}\n}}\n",
        cfg.vocab,
        keep.len(),
        nano_to_kimi.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "),
        specials_json,
        crate::tokenizer::OPEN,
        crate::tokenizer::CLOSE,
        crate::tokenizer::SEP,
        crate::tokenizer::END_OF_MSG,
    );
    let remap_path = format!("{}.vocab.json", out.strip_suffix(".bin").unwrap_or(out));
    VocabPlan { keep, specials_new, remap_path, remap_json }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Unique temp dir per test (no crates: process id + name).
    fn tmpdir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mkim-vocab-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// --vocab-list with exactly the top-N ids must produce the SAME kept set
    /// as --vocab-top N, with a contiguous new-id remap.
    #[test]
    fn vocab_list_matches_top_n() {
        let cfg = Config::microkimi(); // full Kimi vocab: 163840
        let dir = tmpdir("list-vs-topn");
        let model = dir.join("m.bin");
        let out = dir.join("o.bin");
        let source_json = crate::json::parse(b"{}");
        // 1000 ids with distinct deterministic counts
        let n_ids = 1000usize;
        let count_of = |i: usize| (i * 7919) % n_ids + 1; // 7919 coprime with 1000: a permutation
        let freq_path = dir.join("freq.txt");
        let mut freq = String::from("# synthetic freqfile\n");
        for i in 0..n_ids {
            freq.push_str(&format!("{} {}\n", i, count_of(i)));
        }
        std::fs::write(&freq_path, &freq).unwrap();
        // the top-100 ids by frequency, computed independently
        let n_top = 100usize;
        let mut ranked: Vec<usize> = (0..n_ids).collect();
        ranked.sort_by(|&a, &b| count_of(b).cmp(&count_of(a)).then(a.cmp(&b)));
        let top: Vec<usize> = ranked[..n_top].to_vec();
        // the list file: same ids, shuffled order, with comments/blanks/dups
        let list_path = dir.join("keep.txt");
        let mut list = String::from("# vocab_cross.py keep-list (synthetic)\n\n");
        for &i in top.iter().rev() {
            list.push_str(&format!("{}\n", i));
        }
        list.push_str(&format!("{} # duplicate\n{}\n", top[0], top[1]));
        std::fs::write(&list_path, &list).unwrap();

        let m = model.to_str().unwrap();
        let o = out.to_str().unwrap();
        let plan_top = build_vocab_plan(m, o, &source_json, &cfg, &VocabSelect::TopN(n_top, freq_path.to_str().unwrap().to_string()), None);
        let plan_list = build_vocab_plan(m, o, &source_json, &cfg, &VocabSelect::List(list_path.to_str().unwrap().to_string()), None);
        assert_eq!(plan_top.keep, plan_list.keep, "list of the top-N ids == --vocab-top N keep-set");

        // keep-set content: the top-100 + all specials + the reserved block
        let keep = &plan_list.keep;
        assert!(keep.windows(2).all(|w| w[0] < w[1]), "keep-set strictly ascending (contiguous remap)");
        for &i in &top {
            assert!(keep.binary_search(&i).is_ok(), "top-N id {} kept", i);
        }
        assert!(keep.len() > n_top + 200, "specials + reserved block force-kept on top: {}", keep.len());

        // remap contiguity: every special has a distinct dense new id, and the
        // remap table has exactly one entry per kept row
        let remap = crate::json::parse(plan_list.remap_json.as_bytes());
        let vocab_size = remap.get("vocab_size").and_then(|v| v.as_num()).unwrap() as usize;
        assert_eq!(vocab_size, keep.len());
        let n2k = remap.get("nano_to_kimi").and_then(|v| v.as_arr()).unwrap();
        assert_eq!(n2k.len(), keep.len(), "one remap entry per kept row");
        let mut new_ids: Vec<u32> = plan_list.specials_new.iter().map(|&(_, id)| id).collect();
        new_ids.sort_unstable();
        new_ids.dedup();
        assert_eq!(new_ids.len(), plan_list.specials_new.len(), "special new ids distinct");
        assert!(*new_ids.last().unwrap() < keep.len() as u32, "special new ids dense within the new vocab");
        for (name, old) in [("bos", crate::tokenizer::BOS), ("end_of_msg", crate::tokenizer::END_OF_MSG)] {
            let new = plan_list.specials_new.iter().find(|(n, _)| n == name).map(|&(_, id)| id).unwrap();
            assert_eq!(new as usize, keep.binary_search(&(old as usize)).unwrap(), "{} remapped to its dense position", name);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parse_id_list_basics() {
        let dir = tmpdir("idlist");
        let p = dir.join("l.txt");
        std::fs::write(&p, "# comment\n\n5\n7  \n3\n").unwrap();
        assert_eq!(parse_id_list(p.to_str().unwrap(), 10), vec![5, 7, 3]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn parse_id_list_rejects_out_of_range() {
        let dir = tmpdir("idlist-oor");
        let p = dir.join("l.txt");
        std::fs::write(&p, "5\n10\n").unwrap();
        parse_id_list(p.to_str().unwrap(), 10);
    }

    #[test]
    #[should_panic(expected = "expected one token id per line")]
    fn parse_id_list_rejects_garbage() {
        let dir = tmpdir("idlist-bad");
        let p = dir.join("l.txt");
        std::fs::write(&p, "5\nabc\n").unwrap();
        parse_id_list(p.to_str().unwrap(), 10);
    }
}
