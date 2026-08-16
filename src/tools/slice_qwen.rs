//! Vocabulary slicing for converted Qwen models (`slice-qwen-vocab`).
//!
//! The 248320-entry vocabulary carries embedding and head rows for many
//! tokens a deployment never emits. The slicer keeps a subset of rows and
//! rewrites the converted file: `embed_tokens` and `lm_head` are gathered
//! row by row, every other tensor is byte-copied, and the config records
//! the sliced vocabulary plus remapped special ids. A `qwen.vocabmap.json`
//! sidecar carries the new-to-old table; the tokenizer loads it, encodes
//! on the full vocabulary, remaps, and re-encodes any dropped token as its
//! single-byte tokens - which the keep set always contains, so every byte
//! sequence stays representable (no unknown token exists or is needed).
//!
//! The keep set is: the special block (`<|endoftext|>` upward), the 256
//! single-byte tokens, the chat-template pieces, and the top-N ids of a
//! frequency file counting the model's own vocabulary.

use crate::config::QwenConfig;
use crate::quant::weights::{BinFile, BinWriter, DTYPE_F32};
use std::collections::BTreeSet;

/// First id of the added-special block (<|endoftext|>).
const SPECIAL_BLOCK_START: u32 = crate::model::qwentok::QWEN_ENDOFTEXT;

/// Rewrites `model` into `out`, keeping exactly the rows of `new_to_old`
/// (ascending, deduplicated, all within the source vocabulary) in
/// `embed_tokens` and `lm_head`. Returns the sliced config.
pub fn slice_vocab_bin(model: &str, out: &str, new_to_old: &[u32]) -> QwenConfig {
    let bin = BinFile::open(model);
    let c = bin
        .config
        .qwen
        .clone()
        .expect("slice-qwen-vocab requires a Qwen MKIM0002 file");
    let vocab = c.vocab;
    assert!(!new_to_old.is_empty(), "empty keep set");
    assert!(
        new_to_old.windows(2).all(|w| w[0] < w[1]),
        "keep set must be ascending and unique"
    );
    assert!(
        (*new_to_old.last().unwrap() as usize) < vocab,
        "keep set exceeds the source vocabulary"
    );
    assert!(!std::path::Path::new(out).exists(), "{} already exists", out);

    let mut c2 = c.clone();
    c2.vocab = new_to_old.len();
    // remap the structural specials when they exist in the source vocab
    let map_special = |old: u32| -> u32 {
        new_to_old
            .binary_search(&old)
            .map(|new| new as u32)
            .unwrap_or(old)
    };
    let (bos_old, eom_old) = (bin.config.bos_id, bin.config.eos_id);
    if (bos_old as usize) < vocab {
        assert!(
            new_to_old.binary_search(&bos_old).is_ok(),
            "the keep set must retain the bos special"
        );
    }
    if (eom_old as usize) < vocab {
        assert!(
            new_to_old.binary_search(&eom_old).is_ok(),
            "the keep set must retain the end_of_msg special"
        );
    }

    let layout = super::convert_qwen::output_layout(&c);
    let mut writer = BinWriter::new();
    let mut out_dims: Vec<Vec<u32>> = Vec::with_capacity(layout.len());
    for (name, dtype, dims) in &layout {
        let dims = if name == "model.language_model.embed_tokens.weight"
            || name == "lm_head.weight"
        {
            vec![new_to_old.len() as u32, dims[1]]
        } else {
            dims.clone()
        };
        writer.add(name, *dtype, dims.clone());
        out_dims.push(dims);
    }
    let config = super::convert_qwen::config_json_with_specials(
        &c2,
        "qwen.tokenizer.json",
        map_special(bos_old),
        map_special(eom_old),
    );
    let partial = format!("{}.partial.{}", out, std::process::id());
    let mut file = std::fs::File::create(&partial)
        .unwrap_or_else(|e| panic!("cannot create {}: {}", partial, e));
    let offsets = writer.write_header_v2(&mut file, &config);
    for (((name, dtype, _), dims), &offset) in layout.iter().zip(&out_dims).zip(&offsets) {
        let entry = bin
            .entries
            .get(name)
            .unwrap_or_else(|| panic!("{}: missing tensor {}", model, name));
        let src = &bin.data[entry.offset as usize..(entry.offset + entry.size) as usize];
        let blob: Vec<u8> = if name == "model.language_model.embed_tokens.weight"
            || name == "lm_head.weight"
        {
            assert_eq!(*dtype, DTYPE_F32);
            let d = dims[1] as usize;
            let row_bytes = d * 4;
            let mut gathered = Vec::with_capacity(new_to_old.len() * row_bytes);
            for &old in new_to_old {
                let start = old as usize * row_bytes;
                gathered.extend_from_slice(&src[start..start + row_bytes]);
            }
            gathered
        } else {
            src.to_vec()
        };
        writer.write_blob_at(&mut file, offset, &blob);
    }
    file.sync_all().unwrap();
    drop(file);
    std::fs::rename(&partial, out).unwrap();
    c2
}

/// `microkimi slice-qwen-vocab --model X.bin --out Y.bin --top N
/// --freqfile F [--vocab tokenizer.json]`.
pub fn run(args: &[String]) {
    let value = |flag: &str| crate::value_flag(args, flag);
    let model = value("--model").unwrap_or_else(|| {
        eprintln!("error: slice-qwen-vocab requires --model MODEL.bin");
        std::process::exit(2);
    });
    let out = value("--out").unwrap_or_else(|| {
        eprintln!("error: slice-qwen-vocab requires --out SLICED.bin");
        std::process::exit(2);
    });
    let top: usize = value("--top").and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        eprintln!("error: slice-qwen-vocab requires --top N");
        std::process::exit(2);
    });
    let freqfile = value("--freqfile").unwrap_or_else(|| {
        eprintln!("error: slice-qwen-vocab requires --freqfile F (\"<token_id> <count>\" lines)");
        std::process::exit(2);
    });

    let cfg = crate::quant::weights::read_config(&model);
    let Some(qcfg) = cfg.qwen.clone() else {
        eprintln!("error: {} is not a Qwen MKIM0002 file", model);
        std::process::exit(1);
    };
    let vocab = qcfg.vocab;

    // full tokenizer: byte tokens + chat template pieces
    let tok_path = value("--vocab").unwrap_or_else(|| {
        let dir = std::path::Path::new(&model)
            .parent()
            .unwrap_or(std::path::Path::new("."));
        ["qwen.tokenizer.json", "tokenizer.json"]
            .iter()
            .map(|name| dir.join(name))
            .find(|p| p.exists())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| {
                eprintln!("error: no Qwen tokenizer found beside {}", model);
                std::process::exit(1);
            })
    });
    let tok = crate::model::qwentok::QwenTokenizer::load(&tok_path, vocab);

    let mut keep: BTreeSet<u32> = BTreeSet::new();
    for id in SPECIAL_BLOCK_START..vocab as u32 {
        keep.insert(id);
    }
    for b in 0..=255u8 {
        let id = tok.single_byte_full_id(b).unwrap_or_else(|| {
            eprintln!("error: {} has no single-byte token for byte {}", tok_path, b);
            std::process::exit(1);
        });
        keep.insert(id);
    }
    for piece in ["user\n", "assistant\n", "\n", "<think>\n", "</think>\n"] {
        for id in tok.encode(piece) {
            keep.insert(id);
        }
    }
    let structural = keep.len();

    let counts = crate::tools::slice::parse_freqfile_pub(&freqfile, vocab);
    let mut ranked: Vec<(u64, u32)> = counts
        .iter()
        .enumerate()
        .filter(|&(id, &count)| count > 0 && !keep.contains(&(id as u32)))
        .map(|(id, &count)| (count, id as u32))
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    for &(_, id) in ranked.iter().take(top) {
        keep.insert(id);
    }
    let new_to_old: Vec<u32> = keep.into_iter().collect();
    println!(
        "keep set: {} rows ({} structural + byte + template, {} ranked of {} requested)",
        new_to_old.len(),
        structural,
        new_to_old.len() - structural,
        top
    );

    let c2 = slice_vocab_bin(&model, &out, &new_to_old);
    let map_path = std::path::Path::new(&out)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("qwen.vocabmap.json");
    assert!(
        !map_path.exists(),
        "{} already exists (one sliced model per directory)",
        map_path.display()
    );
    let ids: Vec<String> = new_to_old.iter().map(|id| id.to_string()).collect();
    std::fs::write(
        &map_path,
        format!(
            "{{\"vocab_size\":{},\"full_vocab_size\":{},\"new_to_old\":[{}]}}\n",
            new_to_old.len(),
            vocab,
            ids.join(",")
        ),
    )
    .unwrap();
    let size = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
    println!(
        "sliced: {} ({:.2} GB, vocab {} -> {}) + {}",
        out,
        size as f64 / 1e9,
        vocab,
        c2.vocab,
        map_path.display()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sliced_rows_decode_identically() {
        let mut c = QwenConfig::qwen38_dense();
        c.n_layers = 4;
        c.d = 32;
        c.vocab = 64;
        c.n_heads = 2;
        c.n_kv_heads = 1;
        c.head_dim = 16;
        c.lin_k_heads = 1;
        c.lin_v_heads = 1;
        c.lin_k_dim = 32;
        c.lin_v_dim = 32;
        c.dense_inter = 64;
        let full_path = crate::model::qwen::test_fixture(&c);
        let out_path = std::env::temp_dir()
            .join(format!("microkimi_qwen_vocab_slice_{}.bin", std::process::id()))
            .to_string_lossy()
            .into_owned();
        std::fs::remove_file(&out_path).ok();

        let new_to_old: Vec<u32> = vec![0, 1, 2, 3, 5, 8, 13, 21, 34, 55];
        let c2 = slice_vocab_bin(&full_path, &out_path, &new_to_old);
        assert_eq!(c2.vocab, new_to_old.len());

        let mut full = crate::model::qwen::QwenModel::load(&full_path);
        let mut sliced = crate::model::qwen::QwenModel::load(&out_path);
        // same token stream in both spaces: kept rows only
        for (new_id, &old_id) in new_to_old.iter().enumerate() {
            let lf = full.forward(old_id);
            let ls = sliced.forward(new_id as u32);
            assert_eq!(ls.len(), new_to_old.len());
            for (j, &old_row) in new_to_old.iter().enumerate() {
                assert_eq!(
                    ls[j], lf[old_row as usize],
                    "logit mismatch at kept row {} for input {}",
                    j, old_id
                );
            }
        }
        drop((full, sliced));
        std::fs::remove_file(full_path).ok();
        std::fs::remove_file(out_path).ok();
    }
}
