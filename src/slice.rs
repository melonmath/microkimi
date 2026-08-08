// `microkimi slice`: structural pruning of a K3 .bin model (MKIM0001/0002 in,
// MKIM0002 out). The output loads through the unmodified Model::load: every
// pruned dim is recorded in the MKIM0002 JSON config (hidden, n_experts,
// top_k, n_layers, mla_layers, dense_layers).
//
//   microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers "spec"] [--cold-vq N]
//                                                [--vocab-top N <freqfile> [--vocab-base <remap.json>]]
//                                                [--imatrix imatrix.bin [--imatrix-score-only]]
//                                                [--expert-order=frequency --route-cms sketch.bin]
//
// Expert reordering (--expert-order=frequency --route-cms SKETCH): no
// pruning, ALL experts stay; per MoE layer the expert blobs are physically
// rewritten in descending routing-frequency order (the count-min sketch
// recorded with MICROKIMI_ROUTECMS, cms.rs), hottest first, densely packed
// (64-byte alignment). The router gate rows and bias are permuted with the
// same order, so expert ids are simply relabeled and the model is
// mathematically unchanged: any engine reads the reordered .bin, and old
// .bins keep working (the permutation rides in the MKIM0002 config as an
// "expert_order" index table, new_id -> old_id). The point is physical: hot
// experts become file-adjacent, so the stream engine's contiguous-run
// fusion (stream.rs warm_batch) serves a layer's top-k batch in far fewer
// physical reads on latency-bound disks. Combined with --experts N, the
// Frobenius keep-set membership is unchanged and only the file order
// follows the frequency.
//
// Vocabulary pruning (--vocab-top N freqfile): keeps the N most frequent
// token rows of embed_tokens.weight / lm_head.weight plus ALL special tokens
// (they have near-zero corpus frequency but are structural: <|open|>, <|sep|>,
// <|close|>, <|end_of_msg|>, UNK, PAD...). Detection is conservative: every id
// of the source config "specials" block is kept, and on a full Kimi vocab
// (vocab > 163584) the whole reserved block [163584, vocab) is kept. The
// freqfile ids index the model's CURRENT vocabulary: text format is
// "<token_id> <count>" per line ('#' comments, blank lines ok); a JSON object
// {"<id>": <count>, ...} is also accepted. nano/count_freq.py builds one from
// a tokenized corpus (u32/u16 binary + .meta.json sidecar). The output config
// carries the new (smaller) vocab size, and a runtime remap compatible with
// the engine's --vocab mechanism is written next to the .bin as
// <stem>.vocab.json (new_id -> kimi id via "nano_to_kimi"; dropped tokens
// encode as UNK). When the source model is itself remapped (e.g. nano vocab),
// the remap is composed through the base table: --vocab-base, else
// vocab_nano.json next to the source with a matching vocab_size, else (full
// Kimi vocab only) the identity.
//
// Precision tiering (--cold-vq N): no structural pruning, ALL experts stay
// (router untouched); per MoE layer the top-N experts by Frobenius score
// stay mxfp4 (hot) and the rest are requantized to VQ1 (cold): vectors of 16
// consecutive values mapped to the nearest of 256 entries of ONE global
// codebook (tensor "vq_codebook" [256,16] f32, 16 KB), 1 byte per vector =
// 0.5 bit/weight vs 4.25 for mxfp4. The codebook is Lloyd k-means (seeded,
// deterministic) over a reservoir sample of all cold-expert dequantized
// values, raw (unnormalized) vectors. With --experts M (M >= N): the tail
// below M is still pruned, top-N hot, ranks N..M cold VQ1.
// With --imatrix FILE (from `microkimi calibrate`): activation second
// moments weight both the k-means (distance + per-dimension means) and the
// nearest-centroid assignment, so weight columns feeding large activations
// keep more fidelity; the written file format is unchanged.
// --imatrix-score-only loads the stats only to REPORT the activation-weighted
// error of the blind codebook (A/B measurement).
//
// Ranking (v1, weight magnitude only, no activation calibration):
//   - channels (--hidden): score[c] = sum of |w| over every tensor touching
//     hidden channel c (column sums of input projections, embeddings and
//     lm_head; row sums of output projections; |w[c]| of the [d] norm
//     vectors). The top-N channels are kept, the SAME indices everywhere.
//   - experts (--experts): per MoE layer, score[e] = squared Frobenius norm
//     of the expert's w1+w2+w3 (dequantized). Top-N kept per layer, the 2
//     shared experts always stay, the router is re-indexed (rows sliced) and
//     top_k becomes min(top_k, N).
//   - layers (--layers): "0-11", "0,10,20" or mixes; tensors are renumbered
//     layers.* in keep order and the config records which kept layers are
//     MLA / dense. AttnRes choice: the block structure is NOT carried over
//     (blocks are an inference-time grouping, not weights); attn_res_block
//     keeps its value and is re-applied on the RENUMBERED layers, exactly as
//     if the pruned model had been built with that block size. The per-layer
//     res norm/proj vectors simply follow their layer.
//
// MXFP4 decision: kept experts are copied byte-for-byte in mxfp4 (their dims
// [moe_inter, routed_hidden] / [routed_hidden, moe_inter] never change: the
// latent MoE sits behind routed_hidden, so channel pruning does not touch
// them). Zero requantization loss. All sliced tensors are f32 and stay f32.

use crate::config::Config;
use crate::slice_st::{StArch, StDir};
use crate::weights::{BinWriter, DTYPE_F32, DTYPE_MXFP4, DTYPE_MXFP4SQ, DTYPE_VQ1, MAGIC, MAGIC_V2, blob_size, f32_to_bytes};
use std::io::Read;

/// Row chunk for streaming f32 processing (~256 MB of f32 per chunk: the
/// 163840x7168 embeddings never sit in RAM as a whole).
const CHUNK_VALS: usize = 1 << 26;

fn n_rows(e: &DirEntry) -> usize {
    if e.dims.len() <= 1 { 1 } else { e.dims[0] as usize }
}

fn row_width(e: &DirEntry) -> usize {
    if e.dims.len() <= 1 { e.dims[0] as usize } else { e.dims[1..].iter().map(|&d| d as usize).product() }
}

fn row_chunks(rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let per = (CHUNK_VALS / cols.max(1)).max(1);
    (0..rows).step_by(per).map(|r0| (r0, (r0 + per).min(rows))).collect()
}

pub(crate) struct DirEntry {
    pub(crate) name: String,
    pub(crate) dtype: u8,
    pub(crate) dims: Vec<u32>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// Header + directory of a .bin, without loading the weight data (the slicer
/// streams tensor blobs on demand; a 2.5 GB model must not sit in RAM twice).
struct BinDir {
    config: Config,
    source_json: crate::json::Json,
    entries: Vec<DirEntry>, // in file (directory) order: deterministic output
    index: std::collections::HashMap<String, usize>, // name -> entries position
    file: std::fs::File,
}

impl BinDir {
    fn open(path: &str) -> Self {
        let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{} unreadable: {}", path, e));
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).unwrap();
        let source_json = if magic == *MAGIC {
            crate::json::Json::Null // MKIM0001: implicit microkimi config
        } else if magic == *MAGIC_V2 {
            let mut clen = [0u8; 4];
            f.read_exact(&mut clen).unwrap();
            let clen = u32::from_le_bytes(clen) as usize;
            let mut cbuf = vec![0u8; clen];
            f.read_exact(&mut cbuf).unwrap();
            crate::json::parse(&cbuf)
        } else {
            panic!("bad magic in {} (expected MKIM0001 or MKIM0002)", path)
        };
        let config = Config::from_json(&source_json);
        assert!(config.ds.is_none(), "slice only supports K3 models (not DeepSeek-V4)");
        let mut nbuf = [0u8; 4];
        f.read_exact(&mut nbuf).unwrap();
        let n = u32::from_le_bytes(nbuf) as usize;
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            let mut nlen = [0u8; 2];
            f.read_exact(&mut nlen).unwrap();
            let nlen = u16::from_le_bytes(nlen) as usize;
            let mut name = vec![0u8; nlen];
            f.read_exact(&mut name).unwrap();
            let mut fixed = [0u8; 2];
            f.read_exact(&mut fixed).unwrap();
            let (dtype, n_dims) = (fixed[0], fixed[1] as usize);
            let mut dims = vec![0u32; n_dims];
            for d in dims.iter_mut() {
                let mut b = [0u8; 4];
                f.read_exact(&mut b).unwrap();
                *d = u32::from_le_bytes(b);
            }
            let mut b16 = [0u8; 16];
            f.read_exact(&mut b16).unwrap();
            entries.push(DirEntry {
                name: String::from_utf8(name).unwrap(),
                dtype,
                dims,
                offset: u64::from_le_bytes(b16[0..8].try_into().unwrap()),
                size: u64::from_le_bytes(b16[8..16].try_into().unwrap()),
            });
        }
        let index = entries.iter().enumerate().map(|(i, e)| (e.name.clone(), i)).collect();
        BinDir { config, source_json, entries, index, file: f }
    }

    fn entry(&self, name: &str) -> &DirEntry {
        &self.entries[*self.index.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name))]
    }

    /// Raw blob of a tensor (thread-safe: read_exact_at takes &File).
    fn blob(&self, e: &DirEntry) -> Vec<u8> {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0u8; e.size as usize];
        self.file.read_exact_at(&mut buf, e.offset).unwrap();
        buf
    }

    /// Rows r0..r1 of an f32 tensor (whole row width).
    fn f32_rows(&self, e: &DirEntry, r0: usize, r1: usize) -> Vec<f32> {
        assert_eq!(e.dtype, DTYPE_F32, "{} is not f32", e.name);
        use std::os::unix::fs::FileExt;
        let cols = row_width(e);
        let mut buf = vec![0u8; (r1 - r0) * cols * 4];
        self.file.read_exact_at(&mut buf, e.offset + (r0 * cols * 4) as u64).unwrap();
        buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Only the scale bytes of an MXFP4 expert blob (packed ++ scales layout).
    fn expert_scales(&self, e: &DirEntry) -> Vec<u8> {
        assert_eq!(e.dtype, DTYPE_MXFP4, "{} is not mxfp4", e.name);
        use std::os::unix::fs::FileExt;
        let (r, c) = (e.dims[0] as u64, e.dims[1] as u64);
        let mut buf = vec![0u8; (r * c / 32) as usize];
        self.file.read_exact_at(&mut buf, e.offset + r * c / 2).unwrap();
        buf
    }
}

/// Slice input: a .bin (MKIM0001/0002) or a safetensors source (local
/// file/dir, or a HuggingFace URL read through range requests).
enum Source {
    Bin(BinDir),
    St(StDir),
}

impl Source {
    fn open(model: &str, cache_hint: &str) -> Source {
        let is_url = model.starts_with("http://") || model.starts_with("https://");
        let is_dir = std::path::Path::new(model).is_dir();
        if is_url || is_dir {
            return Source::St(StDir::open(model, cache_hint));
        }
        let mut f = std::fs::File::open(model).unwrap_or_else(|e| panic!("{} unreadable: {}", model, e));
        let mut magic = [0u8; 8];
        f.read_exact(&mut magic).unwrap_or_else(|_| panic!("{}: too small", model));
        drop(f);
        if &magic[0..4] == b"MKIM" {
            Source::Bin(BinDir::open(model))
        } else {
            Source::St(StDir::open(model, cache_hint))
        }
    }

    fn config(&self) -> &Config {
        match self {
            Source::Bin(b) => &b.config,
            Source::St(s) => &s.config,
        }
    }

    fn entries(&self) -> &[DirEntry] {
        match self {
            Source::Bin(b) => &b.entries,
            Source::St(s) => &s.entries,
        }
    }

    fn entry(&self, name: &str) -> &DirEntry {
        match self {
            Source::Bin(b) => b.entry(name),
            Source::St(s) => &s.entries[*s.index.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name))],
        }
    }

    fn arch(&self) -> Arch {
        match self {
            // a .bin produced by an earlier dense slice re-slices as dense
            Source::Bin(b) => {
                if b.source_json.get("arch").and_then(|x| x.as_str()) == Some("dense") {
                    Arch::Dense
                } else {
                    Arch::Micro
                }
            }
            Source::St(s) => match s.arch {
                StArch::K3 => Arch::K3Real,
                StArch::Dense => Arch::Dense,
            },
        }
    }

    fn source_json(&self) -> crate::json::Json {
        match self {
            Source::Bin(b) => b.source_json.clone(),
            Source::St(_) => crate::json::Json::Null,
        }
    }

    /// MKIM0002 config "arch" marker to carry over (dense round-trips).
    fn arch_config_key(&self) -> &'static str {
        match self.arch() {
            Arch::Dense => ", \"arch\": \"dense\"",
            _ => "",
        }
    }

    fn is_remote(&self) -> bool {
        matches!(self, Source::St(s) if s.is_remote())
    }

    fn resolve(&mut self, kept_layers: &[usize]) {
        if let Source::St(s) = self {
            s.resolve(kept_layers);
        }
    }

    fn enable_caching(&self) {
        if let Source::St(s) = self {
            s.enable_caching();
        }
    }

    fn f32_rows(&self, e: &DirEntry, r0: usize, r1: usize) -> Vec<f32> {
        match self {
            Source::Bin(b) => b.f32_rows(e, r0, r1),
            Source::St(s) => s.f32_rows(e, r0, r1),
        }
    }

    fn raw_blob(&self, e: &DirEntry) -> Vec<u8> {
        match self {
            Source::Bin(b) => b.blob(e),
            Source::St(s) => s.raw_blob(e),
        }
    }

    fn expert_scales(&self, e: &DirEntry) -> Vec<u8> {
        match self {
            Source::Bin(b) => b.expert_scales(e),
            Source::St(s) => s.expert_scales(e),
        }
    }
}

/// Which role table applies. Micro: the historical .bin config where several
/// dims coincide (kda_proj == d, MLA g_proj [d,d]). K3Real: the real K3
/// safetensors dims (hidden 7168, MLA g_proj [H*v, d] -> columns only).
/// Dense: plain llama-like models (Qwen), no MoE/KDA/MLA.
#[derive(Clone, Copy, PartialEq)]
enum Arch {
    Micro,
    K3Real,
    Dense,
}

/// How a tensor relates to the hidden dimension d (for channel pruning) and
/// to the routed expert axis. Classification is NAME-based: several micro
/// dims coincide numerically (kda_proj == d == 512, mla kv_a rows == mla_qa),
/// so dims alone cannot identify which axis is the hidden one.
#[derive(Clone, Copy, PartialEq)]
enum Role {
    Copy,    // no hidden / expert axis: copied verbatim (any dtype)
    VecD,    // [d] vector (layernorms, AttnRes norm/proj)
    ColsD,   // [R, d]: slice columns (input projections, embed, lm_head)
    RowsD,   // [d, C]: slice rows (output projections)
    BothD,   // [d, d]: slice rows and columns (micro MLA g_proj only)
    RouterW, // [n_experts, d]: expert rows + hidden columns
    RouterB, // [n_experts]: e_score_correction_bias
    Expert,  // block_sparse_moe.experts.{e}.{w1,w2,w3} (mxfp4, dims untouched)
}

/// Splits "layers.{i}.{rest}" into (layer_index, rest).
pub(crate) fn split_layer(name: &str) -> Option<(usize, &str)> {
    let s = name.strip_prefix("layers.")?;
    let dot = s.find('.')?;
    Some((s[..dot].parse().ok()?, &s[dot + 1..]))
}

fn role_of(name: &str, cfg: &Config, arch: Arch) -> Role {
    if arch == Arch::Dense {
        return role_dense(name);
    }
    match name {
        "embed_tokens.weight" | "lm_head.weight" => return Role::ColsD,
        "norm.weight" | "output_attn_res_norm.weight" | "output_attn_res_proj.weight" => return Role::VecD,
        _ => {}
    }
    let Some((l, r)) = split_layer(name) else {
        panic!("slice: unknown tensor (no layers. prefix): {}", name);
    };
    match r {
        "input_layernorm.weight" | "post_attention_layernorm.weight" | "self_attention_res_norm.weight"
        | "self_attention_res_proj.weight" | "mlp_res_norm.weight" | "mlp_res_proj.weight" => Role::VecD,
        "block_sparse_moe.gate.weight" => Role::RouterW,
        "block_sparse_moe.gate.e_score_correction_bias" => Role::RouterB,
        "block_sparse_moe.routed_expert_down_proj.weight" | "block_sparse_moe.shared_experts.gate_proj.weight"
        | "block_sparse_moe.shared_experts.up_proj.weight" => Role::ColsD,
        "block_sparse_moe.routed_expert_up_proj.weight" | "block_sparse_moe.shared_experts.down_proj.weight" => Role::RowsD,
        "block_sparse_moe.routed_expert_norm.weight" => Role::Copy,
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => Role::ColsD,
        "mlp.down_proj.weight" => Role::RowsD,
        _ if r.starts_with("block_sparse_moe.experts.") => Role::Expert,
        _ if cfg.is_mla(l) => match r {
            "self_attn.q_a_proj.weight" | "self_attn.kv_a_proj_with_mqa.weight" => Role::ColsD,
            // real K3 g_proj is [H*v, d]: only the columns are the hidden axis
            // (the micro [d,d] coincidence allowed the historical BothD).
            "self_attn.g_proj.weight" if arch == Arch::K3Real => Role::ColsD,
            "self_attn.g_proj.weight" => Role::BothD,
            "self_attn.o_proj.weight" => Role::RowsD,
            "self_attn.q_a_layernorm.weight" | "self_attn.q_b_proj.weight" | "self_attn.kv_a_layernorm.weight"
            | "self_attn.kv_b_proj.weight" => Role::Copy,
            _ => panic!("slice: unknown MLA tensor: {}", name),
        },
        _ => match r {
            "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" | "self_attn.g_proj.weight"
            | "self_attn.f_a_proj.weight" | "self_attn.b_proj.weight" => Role::ColsD,
            "self_attn.o_proj.weight" => Role::RowsD,
            "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight"
            | "self_attn.f_b_proj.weight" | "self_attn.A_log" | "self_attn.dt_bias" | "self_attn.o_norm.weight" => Role::Copy,
            _ => panic!("slice: unknown KDA tensor: {}", name),
        },
    }
}

/// Role table for plain dense models (Qwen/llama-like): q/k/v/gate/up slice
/// columns, o/down slice rows, norms are [d] vectors, biases have no hidden
/// axis (they index output rows) and are copied.
fn role_dense(name: &str) -> Role {
    match name {
        "embed_tokens.weight" | "lm_head.weight" => return Role::ColsD,
        "norm.weight" => return Role::VecD,
        _ => {}
    }
    let Some((_, r)) = split_layer(name) else {
        panic!("slice: unknown tensor (no layers. prefix): {}", name);
    };
    match r {
        "input_layernorm.weight" | "post_attention_layernorm.weight" => Role::VecD,
        "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => Role::ColsD,
        "self_attn.o_proj.weight" => Role::RowsD,
        "self_attn.q_proj.bias" | "self_attn.k_proj.bias" | "self_attn.v_proj.bias" | "self_attn.o_proj.bias" => Role::Copy,
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => Role::ColsD,
        "mlp.down_proj.weight" => Role::RowsD,
        _ => panic!("slice: unknown dense tensor: {}", name),
    }
}

/// Parses a layer spec: "0-11", "0,10,20", "0-3,7,9-11" (ranges inclusive).
fn parse_layer_spec(spec: &str, n_layers: usize) -> Vec<usize> {
    let mut keep = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        assert!(!part.is_empty(), "bad --layers spec: '{}'", spec);
        if let Some((a, b)) = part.split_once('-') {
            let a: usize = a.trim().parse().unwrap_or_else(|_| panic!("bad --layers spec: '{}'", spec));
            let b: usize = b.trim().parse().unwrap_or_else(|_| panic!("bad --layers spec: '{}'", spec));
            assert!(a <= b, "bad --layers range '{}'", part);
            keep.extend(a..=b);
        } else {
            keep.push(part.parse().unwrap_or_else(|_| panic!("bad --layers spec: '{}'", spec)));
        }
    }
    keep.sort_unstable();
    keep.dedup();
    assert!(!keep.is_empty(), "--layers keeps nothing");
    assert!(keep.last().unwrap() < &n_layers, "--layers index out of range (model has {} layers)", n_layers);
    keep
}

/// Top-n indices by descending score, ties broken by lower index, returned
/// in ascending order (deterministic keep-set).
fn top_n(scores: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(n);
    idx.sort_unstable();
    idx
}

/// Channel scores: for every tensor touching d, sum |w| per channel.
/// Tensors are processed in row chunks (embeddings never sit in RAM whole;
/// remote sources stream the same chunks).
fn channel_scores(src: &Source, kept_layers: &[usize], d: usize) -> Vec<f64> {
    let cfg = src.config();
    let arch = src.arch();
    let mut s = vec![0f64; d];
    for e in src.entries() {
        let in_layers = match split_layer(&e.name) {
            None => true, // global tensor
            Some((l, _)) => kept_layers.contains(&l),
        };
        if !in_layers {
            continue;
        }
        let role = role_of(&e.name, cfg, arch);
        if matches!(role, Role::Copy | Role::RouterB | Role::Expert) {
            continue;
        }
        let (rows, cols) = (n_rows(e), row_width(e));
        match role {
            Role::VecD => {
                let w = src.f32_rows(e, 0, 1);
                assert_eq!(w.len(), d, "{}: expected [{}]", e.name, d);
                for (j, &x) in w.iter().enumerate() {
                    s[j] += x.abs() as f64;
                }
            }
            Role::ColsD | Role::RouterW => {
                assert_eq!(cols, d, "{}: expected [_, {}]", e.name, d);
                for (r0, r1) in row_chunks(rows, cols) {
                    for row in src.f32_rows(e, r0, r1).chunks_exact(cols) {
                        for (j, &x) in row.iter().enumerate() {
                            s[j] += x.abs() as f64;
                        }
                    }
                }
            }
            Role::RowsD => {
                assert_eq!(rows, d, "{}: expected [{}, _]", e.name, d);
                for (r0, r1) in row_chunks(rows, cols) {
                    for (j, row) in src.f32_rows(e, r0, r1).chunks_exact(cols).enumerate() {
                        s[r0 + j] += row.iter().map(|&x| x.abs() as f64).sum::<f64>();
                    }
                }
            }
            Role::BothD => {
                assert_eq!((rows, cols), (d, d), "{}: expected [{}, {}]", e.name, d, d);
                let w = src.f32_rows(e, 0, rows);
                // the channel appears as BOTH the output (row) and the input
                // (column) axis: row sums + column sums
                for (j, row) in w.chunks_exact(cols).enumerate() {
                    s[j] += row.iter().map(|&x| x.abs() as f64).sum::<f64>();
                }
                for row in w.chunks_exact(cols) {
                    for (j, &x) in row.iter().enumerate() {
                        s[j] += x.abs() as f64;
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    s
}

/// Squared Frobenius norm of an expert's w1+w2+w3 (mxfp4 dequantized).
fn expert_norm_sq(blob_w1: &[u8], blob_w2: &[u8], blob_w3: &[u8], rows_cols: [(usize, usize); 3]) -> f64 {
    let mut s = 0f64;
    for (blob, &(r, c)) in [blob_w1, blob_w2, blob_w3].iter().zip(rows_cols.iter()) {
        let np = r * c / 2;
        let w = crate::mxfp4::dequant(&blob[..np], &blob[np..], r, c);
        s += w.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
    }
    s
}

/// Scale-energy score: proportional to the squared Frobenius norm under the
/// e2m1 codebook. ||W||^2 = sum over groups of 2^(2*(s-127)) * sum(lut^2);
/// the per-group lut factor has the same expectation for every expert, so
/// ranking on the scale energy alone ranks the same way while reading 1/17
/// of the bytes (the weight_scale tensors only). Used for safetensors
/// sources; the .bin path keeps the exact dequantized Frobenius.
fn mxfp4_scale_energy(scale_bytes: &[u8]) -> f64 {
    scale_bytes
        .iter()
        .map(|&s| {
            let e = s as i32 - 127;
            (2f64).powi(2 * e)
        })
        .sum()
}

/// Per kept MoE layer: the n_experts_keep expert indices to keep (ascending).
/// .bin path: exact dequantized Frobenius, one thread per layer (disk reads).
/// Safetensors path: scale-energy from the weight_scale tensors only; layers
/// are processed sequentially but each layer's experts are split across 8
/// threads (remote scoring is curl-latency bound).
/// Every finished layer is logged and checkpointed immediately; layers found
/// in the checkpoint are restored instead of re-scored (both branches). Full
/// scores are persisted in the persistent ScoreCache (config-independent:
/// the keep-set is recomputed from the cached scores with the current N).
fn expert_keep_sets(src: &Source, kept_layers: &[usize], n_keep: usize, ckpt: &SliceCkpt, sc: &ScoreCache) -> std::collections::HashMap<usize, Vec<usize>> {
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let total = moe_layers.len();
    let mut restored: Vec<(usize, Vec<usize>)> = Vec::new();
    let mut todo: Vec<usize> = Vec::new();
    for &l in &moe_layers {
        match ckpt.experts.get(&l) {
            Some(set) => restored.push((l, set.clone())),
            None => todo.push(l),
        }
    }
    for (l, _) in &restored {
        println!("experts: layer {} restored from checkpoint", l);
    }
    let done = std::sync::atomic::AtomicUsize::new(restored.len());
    // shared by both branches: keep-set of one layer, from the score cache
    // when possible (scoring skipped entirely), freshly computed otherwise.
    // Returns (keep-set, Some(top score range) when freshly scored).
    let keep_of = |l: usize, compute: &dyn Fn() -> Vec<f64>| -> (Vec<usize>, Option<(f64, f64)>) {
        if let Some(cached) = sc.get(l) {
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            println!("experts: layer {} scores restored from score cache ({}/{})", l, n, total);
            return (top_n(cached, n_keep.min(cfg.n_experts)), None);
        }
        let scores = compute();
        sc.record(l, &scores);
        let keep = top_n(&scores, n_keep.min(cfg.n_experts));
        let lo = keep.iter().map(|&e| scores[e]).fold(f64::INFINITY, f64::min);
        let hi = keep.iter().map(|&e| scores[e]).fold(f64::NEG_INFINITY, f64::max);
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        println!("experts: scored layer {}/{} (layer {}, kept {}/{}, top score range {:.3} .. {:.3})", n, total, l, keep.len(), cfg.n_experts, lo, hi);
        (keep, Some((lo, hi)))
    };
    if matches!(src, Source::St(_)) {
        let mut results = restored;
        for &l in &todo {
            let (keep, _) = keep_of(l, &|| {
                let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
                let mut scores = vec![0f64; cfg.n_experts];
                let nt = 8.min(cfg.n_experts);
                let chunk = cfg.n_experts.div_ceil(nt);
                std::thread::scope(|scope| {
                    for (ti, sc) in scores.chunks_mut(chunk).enumerate() {
                        let pfx = &pfx;
                        scope.spawn(move || {
                            for (i, slot) in sc.iter_mut().enumerate() {
                                let e = ti * chunk + i;
                                // scales only: the packed nibbles are never fetched
                                *slot = ["w1", "w2", "w3"]
                                    .iter()
                                    .map(|wn| mxfp4_scale_energy(&src.expert_scales(src.entry(&format!("{}{}.{}", pfx, e, wn)))))
                                    .sum();
                            }
                        });
                    }
                });
                scores
            });
            ckpt.record_experts(l, &keep);
            results.push((l, keep));
        }
        return results.into_iter().collect();
    }
    let results: Vec<(usize, Vec<usize>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &l in &todo {
            let keep_of = &keep_of;
            handles.push(scope.spawn(move || {
                let (keep, _) = keep_of(l, &|| {
                    let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
                    let mut scores = vec![0f64; cfg.n_experts];
                    for e in 0..cfg.n_experts {
                        let mut blobs = [Vec::new(), Vec::new(), Vec::new()];
                        let mut rc = [(0usize, 0usize); 3];
                        for (i, wn) in ["w1", "w2", "w3"].iter().enumerate() {
                            let name = format!("{}{}.{}", pfx, e, wn);
                            let entry = src.entry(&name);
                            assert_eq!(entry.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
                            blobs[i] = src.raw_blob(entry);
                            rc[i] = (entry.dims[0] as usize, entry.dims[1] as usize);
                        }
                        scores[e] = expert_norm_sq(&blobs[0], &blobs[1], &blobs[2], rc);
                    }
                    scores
                });
                ckpt.record_experts(l, &keep);
                (l, keep)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().chain(restored).collect()
}

/// Per kept MoE layer: the full per-expert Frobenius scores (.bin sources
/// only; the exact dequantized w1+w2+w3 norm, one thread per layer). Used by
/// --cold-vq, which needs the complete ranking (hot top-N vs cold tail), not
/// just a keep-set.
fn expert_score_map(src: &Source, kept_layers: &[usize]) -> std::collections::HashMap<usize, Vec<f64>> {
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let results: Vec<(usize, Vec<f64>)> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for &l in &moe_layers {
            handles.push(scope.spawn(move || {
                let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
                let mut scores = vec![0f64; cfg.n_experts];
                for e in 0..cfg.n_experts {
                    let mut blobs = [Vec::new(), Vec::new(), Vec::new()];
                    let mut rc = [(0usize, 0usize); 3];
                    for (i, wn) in ["w1", "w2", "w3"].iter().enumerate() {
                        let name = format!("{}{}.{}", pfx, e, wn);
                        let entry = src.entry(&name);
                        assert_eq!(entry.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
                        blobs[i] = src.raw_blob(entry);
                        rc[i] = (entry.dims[0] as usize, entry.dims[1] as usize);
                    }
                    scores[e] = expert_norm_sq(&blobs[0], &blobs[1], &blobs[2], rc);
                }
                (l, scores)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().collect()
}

/// Reservoir sample (algorithm R, seeded splitmix64) of the raw 16-vectors
/// of every cold expert tensor, for the global VQ codebook training. Each
/// tensor is dequantized from mxfp4 (either flavor) one at a time (never the
/// whole model in RAM). `cold` = per kept MoE layer the cold expert indices
/// (ascending). With an imatrix, per-value activation weights are sampled
/// alongside the vectors (same layout) for the weighted codebook training;
/// without one the rng/sample sequence is exactly the historical one
/// (bit-identical codebook).
fn vq_reservoir(
    src: &Source,
    kept_layers: &[usize],
    cold: &std::collections::HashMap<usize, Vec<usize>>,
    cap: usize,
    seed: u64,
    im: Option<&crate::imatrix::Imatrix>,
) -> (Vec<f32>, Option<Vec<f32>>) {
    use crate::quant::{Rng, VQ_DIM};
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    let mut rng = Rng::new(seed);
    let mut res: Vec<f32> = Vec::with_capacity(cap * VQ_DIM);
    let mut wres: Option<Vec<f32>> = im.map(|_| Vec::with_capacity(cap * VQ_DIM));
    let mut seen = 0u64; // vectors offered so far
    // A tensor without usable imatrix stats falls back to flat weights (1.0),
    // so `wres` always stays aligned with `res`. `cw` holds one weight per
    // matrix COLUMN: vector `vi` of a row of `vpr` vectors covers columns
    // (vi % vpr) * VQ_DIM .. + VQ_DIM.
    let feed = |w: &[f32], cw: Option<&[f32]>, vpr: usize, res: &mut Vec<f32>, wres: &mut Option<Vec<f32>>, rng: &mut Rng, seen: &mut u64| {
        for (vi, v) in w.chunks_exact(VQ_DIM).enumerate() {
            let c0 = (vi % vpr) * VQ_DIM;
            let t = *seen;
            *seen += 1;
            if (t as usize) < cap {
                res.extend_from_slice(v);
                if let Some(wr) = wres.as_mut() {
                    match cw {
                        Some(cw) => wr.extend_from_slice(&cw[c0..c0 + VQ_DIM]),
                        None => wr.extend_from_slice(&[1.0; VQ_DIM]),
                    }
                }
            } else {
                let j = rng.below(t as usize + 1);
                if j < cap {
                    res[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(v);
                    if let Some(wr) = wres.as_mut() {
                        match cw {
                            Some(cw) => wr[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(&cw[c0..c0 + VQ_DIM]),
                            None => wr[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(&[1.0; VQ_DIM]),
                        }
                    }
                }
            }
        }
    };
    for &l in &moe_layers {
        let pfx = format!("layers.{}.block_sparse_moe.experts.", l);
        for &e in &cold[&l] {
            for wn in ["w1", "w2", "w3"] {
                let name = format!("{}{}.{}", pfx, e, wn);
                let entry = src.entry(&name);
                assert!(matches!(entry.dtype, DTYPE_MXFP4 | DTYPE_MXFP4SQ), "{} is not an mxfp4 flavor", name);
                let blob = src.raw_blob(entry);
                let (r, c) = (entry.dims[0] as usize, entry.dims[1] as usize);
                let w = crate::mxfp4::dequant_any(entry.dtype, &blob, r, c);
                let cw = im.and_then(|im| im.col_weights(l, wn));
                feed(&w, cw.as_deref(), c / VQ_DIM, &mut res, &mut wres, &mut rng, &mut seen);
            }
        }
    }
    println!("vq: reservoir sampled {}/{} cold-expert vectors (cap {})", res.len() / VQ_DIM, seen, cap);
    (res, wres)
}

/// Dequantizes one source expert tensor (either mxfp4 flavor) and VQ-quantizes
/// it with the shared codebook. With `quant_col_w` the nearest-centroid
/// assignment is the activation-weighted one (slice --imatrix); both weight
/// arguments hold ONE WEIGHT PER MATRIX COLUMN (expanded per element here).
/// Returns (index bytes, relative Frobenius error, activation-weighted
/// relative error when `score_col_w` is given).
fn vq_quantize_tensor(
    src: &Source,
    e: &DirEntry,
    codebook: &[f32],
    quant_col_w: Option<&[f32]>,
    score_col_w: Option<&[f32]>,
) -> (Vec<u8>, f64, Option<f64>) {
    assert!(matches!(e.dtype, DTYPE_MXFP4 | DTYPE_MXFP4SQ), "{} is not an mxfp4 flavor", e.name);
    let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
    assert_eq!(c % crate::quant::VQ_DIM, 0, "{}: cols {} not a multiple of {}", e.name, c, crate::quant::VQ_DIM);
    let blob = src.raw_blob(e);
    let w = crate::mxfp4::dequant_any(e.dtype, &blob, r, c);
    // per-column importance -> per-element weights (same row of c weights
    // repeated r times; expert matrices are small enough to materialize it)
    let expand = |cw: &[f32]| -> Vec<f32> {
        assert_eq!(cw.len(), c, "{}: imatrix column count {} != tensor cols {}", e.name, cw.len(), c);
        let mut v = Vec::with_capacity(r * c);
        for _ in 0..r {
            v.extend_from_slice(cw);
        }
        v
    };
    let quant_w = quant_col_w.map(expand);
    let score_w = score_col_w.map(expand);
    let idx = match &quant_w {
        Some(wv) => crate::quant::quantize_weighted(&w, wv, codebook),
        None => crate::quant::quantize(&w, codebook),
    };
    let err = crate::quant::rel_error(&w, &idx, codebook);
    let werr = score_w.map(|wv| crate::quant::rel_error_weighted(&w, &wv, &idx, codebook));
    (idx, err, werr)
}

/// One output tensor: what to write and where the data comes from.
struct Plan {
    out_name: String,
    dtype: u8,
    dims: Vec<u32>,
    src_name: String,
    role: Role,
    channels: Vec<usize>,          // hidden keep-set (identity when no --hidden)
    experts: Option<Vec<usize>>,  // expert keep-set for RouterW/RouterB rows
    vocab: Option<Vec<usize>>,    // vocab keep-set (old row ids, ascending) for embed/lm_head
}

/// Slices an f32 tensor according to its role. Returns (values, new_dims).
fn slice_f32(e: &DirEntry, w: &[f32], role: Role, ch: &[usize], experts: Option<&Vec<usize>>) -> (Vec<f32>, Vec<u32>) {
    let (r, c) = (e.dims[0] as usize, *e.dims.get(1).unwrap_or(&1) as usize);
    match role {
        Role::VecD => (ch.iter().map(|&j| w[j]).collect(), vec![ch.len() as u32]),
        Role::ColsD => {
            let mut out = Vec::with_capacity(r * ch.len());
            for row in w.chunks_exact(c) {
                out.extend(ch.iter().map(|&j| row[j]));
            }
            (out, vec![r as u32, ch.len() as u32])
        }
        Role::RowsD => {
            let mut out = Vec::with_capacity(ch.len() * c);
            for &j in ch {
                out.extend_from_slice(&w[j * c..(j + 1) * c]);
            }
            (out, vec![ch.len() as u32, c as u32])
        }
        Role::BothD => {
            let mut out = Vec::with_capacity(ch.len() * ch.len());
            for &i in ch {
                let row = &w[i * c..(i + 1) * c];
                out.extend(ch.iter().map(|&j| row[j]));
            }
            (out, vec![ch.len() as u32, ch.len() as u32])
        }
        Role::RouterW => {
            let rows = experts.cloned().unwrap_or_else(|| (0..r).collect());
            let mut out = Vec::with_capacity(rows.len() * ch.len());
            for &i in &rows {
                let row = &w[i * c..(i + 1) * c];
                out.extend(ch.iter().map(|&j| row[j]));
            }
            (out, vec![rows.len() as u32, ch.len() as u32])
        }
        Role::RouterB => {
            let rows = experts.cloned().unwrap_or_else(|| (0..r).collect());
            (rows.iter().map(|&i| w[i]).collect(), vec![rows.len() as u32])
        }
        _ => unreachable!(),
    }
}

/// Row-chunk slicing for ColsD/RowsD. `vals` holds input rows r0..r1 and the
/// returned output rows are produced in ascending order (ch is sorted), so
/// concatenating over chunks yields the full sliced tensor.
fn slice_f32_rows(role: Role, vals: &[f32], r0: usize, r1: usize, cols: usize, ch: &[usize]) -> Vec<f32> {
    match role {
        Role::ColsD => {
            let mut out = Vec::with_capacity((r1 - r0) * ch.len());
            for row in vals.chunks_exact(cols) {
                out.extend(ch.iter().map(|&j| row[j]));
            }
            out
        }
        Role::RowsD => {
            let lo = ch.partition_point(|&j| j < r0);
            let hi = ch.partition_point(|&j| j < r1);
            let mut out = Vec::with_capacity((hi - lo) * cols);
            for &j in &ch[lo..hi] {
                out.extend_from_slice(&vals[(j - r0) * cols..(j - r0 + 1) * cols]);
            }
            out
        }
        _ => unreachable!(),
    }
}

/// Row-chunk slicing for embed/lm_head under --vocab-top: emits only the
/// kept vocab rows of input chunk r0..r1 (keep is ascending), with columns
/// pruned to ch exactly like ColsD.
fn slice_vocab_rows(vals: &[f32], r0: usize, r1: usize, cols: usize, ch: &[usize], keep: &[usize]) -> Vec<f32> {
    let lo = keep.partition_point(|&j| j < r0);
    let hi = keep.partition_point(|&j| j < r1);
    let mut out = Vec::with_capacity((hi - lo) * ch.len());
    for &j in &keep[lo..hi] {
        let row = &vals[(j - r0) * cols..(j - r0 + 1) * cols];
        out.extend(ch.iter().map(|&c| row[c]));
    }
    out
}

// ── vocabulary pruning (--vocab-top) ──

/// Special tokens the runtime remap must carry (NanoTokenizer::load unwraps
/// all of them but pad).
const SPECIAL_NAMES: [&str; 8] = ["bos", "eos", "open", "close", "sep", "end_of_msg", "unk", "pad"];

/// Id of a special token in the FULL Kimi vocabulary (163584+ reserved block).
fn kimi_special_id(name: &str) -> Option<u32> {
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
fn known_specials(source_json: &crate::json::Json, cfg: &Config) -> Vec<(String, u32)> {
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
fn parse_freqfile(path: &str, vocab: usize) -> Vec<u64> {
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
fn base_vocab_map(model: &str, base_flag: Option<String>, vocab: usize) -> Option<(crate::json::Json, Vec<u32>)> {
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
        "--vocab-top on an already remapped vocab ({} ids) needs --vocab-base <remap.json> \
(or a vocab_nano.json with a matching vocab_size next to the source model) to map rows back to kimi ids",
        vocab
    );
    None
}

/// Result of the --vocab-top selection: the kept rows plus the runtime remap
/// file (engine --vocab compatible) to write next to the .bin.
struct VocabPlan {
    keep: Vec<usize>,                 // old row ids kept, ascending
    specials_new: Vec<(String, u32)>, // name -> NEW id (ascending old order)
    remap_path: String,
    remap_json: String,
}

/// Top-N by frequency + all specials/reserved ids, and the remap JSON.
fn build_vocab_plan(model: &str, out: &str, source_json: &crate::json::Json, cfg: &Config, n_top: usize, freq_path: &str, base_flag: Option<String>) -> VocabPlan {
    let counts = parse_freqfile(freq_path, cfg.vocab);
    let specials_old = known_specials(source_json, cfg);
    let scores: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let mut keep = top_n(&scores, n_top.min(cfg.vocab));
    let n_freq = keep.len();
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
    let total: u64 = counts.iter().sum();
    let covered: u64 = keep.iter().map(|&j| counts[j]).sum();
    println!(
        "vocab: keeping {}/{} rows (top-{} by frequency + {} special/reserved), {:.2}% of the counted token mass",
        keep.len(),
        cfg.vocab,
        n_freq,
        keep.len() - n_freq,
        covered as f64 / total.max(1) as f64 * 100.0
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
            "--vocab-top: no '{}' id found (source config specials + kimi constants) - the runtime remap would be unloadable",
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

fn value_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

// ── crash-safe resume checkpoint (<out>.sliceckpt) ──
//
// The scoring phases (channel |w| sums, per-layer expert scale-energy) are
// the long part of a remote slice (tens of thousands of range requests, ~1h
// of silence) and used to be lost entirely on a spot-VM preemption. Every
// finished entry is appended to the sidecar and fsynced IMMEDIATELY (it
// survives kill -9); a rerun with the same parameters resumes from it.
//
// Format (text, one entry per line):
//   config <fnv1a-64 hex>     run parameters: model | kept layers | hidden | experts
//   channels <c0,c1,...>      hidden keep-set (ascending)
//   experts <layer> <e0,e1,...>  keep-set of one scored MoE layer (ascending)
//
// A `config` mismatch (different model/layers/hidden/experts) discards the
// sidecar with a warning and starts fresh. Covered: channel keep-set and
// expert keep-sets (both the remote scale-energy branch and the local .bin
// Frobenius branch of expert_keep_sets). NOT covered: the --cold-vq score
// map (expert_score_map restarts from scratch) and the write phase itself
// (it restarts from scratch too: it is much faster than the scoring phases
// and resumability would complicate the append-only .bin layout). The file
// is deleted on successful completion.

struct SliceCkpt {
    path: std::path::PathBuf,
    file: std::sync::Mutex<std::fs::File>,
    channels: Option<Vec<usize>>,
    experts: std::collections::HashMap<usize, Vec<usize>>,
}

/// FNV-1a 64 over the run-parameter string (no hash crates, std only).
fn fnv1a(s: &str) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn parse_csv(s: &str) -> Option<Vec<usize>> {
    s.split(',').map(|t| t.parse().ok()).collect()
}

fn join_csv(v: &[usize]) -> String {
    v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",")
}

impl SliceCkpt {
    /// Loads <out>.sliceckpt when present. A matching `config` line restores
    /// the recorded entries, anything else (missing or corrupt file, parameter
    /// mismatch) starts fresh with a warning.
    fn open(out: &str, key: &str) -> SliceCkpt {
        use std::io::Write;
        let path = std::path::PathBuf::from(format!("{}.sliceckpt", out));
        let want = format!("{:016x}", fnv1a(key));
        let mut config_ok = false;
        let mut channels = None;
        let mut experts = std::collections::HashMap::new();
        let existed = path.is_file();
        if let Ok(bytes) = std::fs::read(&path) {
            // a kill -9 between the write and the fsync of the LAST entry can
            // leave a torn trailing line: only complete lines are parsed
            let text = String::from_utf8_lossy(&bytes);
            let body = match text.rsplit_once('\n') {
                Some((body, _)) => body,
                None => "",
            };
            for line in body.lines() {
                let mut it = line.split_whitespace();
                match it.next() {
                    Some("config") => config_ok = it.next() == Some(want.as_str()),
                    Some("channels") => channels = it.next().and_then(parse_csv),
                    Some("experts") => {
                        if let (Some(l), Some(set)) = (it.next().and_then(|s| s.parse().ok()), it.next().and_then(parse_csv)) {
                            experts.insert(l, set);
                        }
                    }
                    _ => {}
                }
            }
        }
        if existed && !config_ok {
            println!("sliceckpt: ignoring {} (parameters differ), starting fresh", path.display());
            std::fs::remove_file(&path).ok();
            channels = None;
            experts = std::collections::HashMap::new();
        }
        let n = experts.len() + channels.is_some() as usize;
        if config_ok && n > 0 {
            println!("sliceckpt: resumed {} ({} entries)", path.display(), n);
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
        if f.metadata().unwrap().len() == 0 {
            f.write_all(format!("config {}\n", want).as_bytes()).unwrap();
            f.sync_data().unwrap();
        }
        SliceCkpt { path, file: std::sync::Mutex::new(f), channels, experts }
    }

    /// Appends one entry and fsyncs it (every entry must survive kill -9).
    /// The whole line goes out in ONE write: a kill -9 lands either before
    /// or after it, never in the middle of a line.
    fn record(&self, line: &str) {
        use std::io::Write;
        let mut f = self.file.lock().unwrap();
        f.write_all(format!("{}\n", line).as_bytes()).unwrap();
        f.sync_data().unwrap();
    }

    fn record_channels(&self, ch: &[usize]) {
        self.record(&format!("channels {}", join_csv(ch)));
    }

    fn record_experts(&self, layer: usize, set: &[usize]) {
        self.record(&format!("experts {} {}", layer, join_csv(set)));
    }

    /// Successful completion: a finished .bin needs no checkpoint.
    fn finish(self) {
        drop(self.file);
        if std::fs::remove_file(&self.path).is_ok() {
            println!("sliceckpt: {} removed (slice complete)", self.path.display());
        }
    }
}

// ── persistent expert-score cache (<out>.slicecache/expert_scores.<fnv1a64>.bin) ──
//
// expert_keep_sets scores every kept MoE layer of the SOURCE model
// (scale-energy for remote safetensors, dequantized Frobenius for local
// .bin). That scoring depends ONLY on the source model and the layer - not
// on --layers/--hidden/--experts/--cold-vq - yet the full scores used to be
// discarded once the top-N keep-set was computed (~45k HTTP range requests
// / 2h+ lost on every remote rerun). This cache persists the full scores
// across runs AND across pruning configs, one level below the per-run
// .sliceckpt (which keeps the config-dependent keep-sets of the current
// run; its config hash does NOT cover this cache, by design). It is a pure
// optimization: deleting the file changes nothing but time.
//
// Format (all little-endian):
//   magic "MKSCORE1" (8 bytes) | u32 version (=1) | u32 n_layers | u32 n_experts
//   then one record per scored MoE layer, appended as soon as it finishes:
//     u32 layer | n_experts * f64 score
// Appends are one write + fsync (kill -9 safe); a torn trailing record is
// ignored on read. magic/version/n_experts mismatch -> the file is ignored
// and replaced (never a panic). Duplicate layer records: the last wins.

const SCORE_MAGIC: &[u8; 8] = b"MKSCORE1";
const SCORE_VERSION: u32 = 1;
const SCORE_HEADER: usize = 8 + 4 + 4 + 4;

struct ScoreCache {
    file: std::sync::Mutex<std::fs::File>,
    scores: std::collections::HashMap<usize, Vec<f64>>, // layer -> full per-expert scores
}

impl ScoreCache {
    /// Opens (or creates) the score cache for `model` under
    /// <out>.slicecache/. A valid existing file restores its layer scores,
    /// anything else (missing, bad magic/version/expert count) starts empty.
    fn open(out: &str, model: &str, n_layers: usize, n_experts: usize) -> ScoreCache {
        use std::io::Write;
        let dir = std::path::PathBuf::from(format!("{}.slicecache", out));
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join(format!("expert_scores.{:016x}.bin", fnv1a(model)));
        let mut scores = std::collections::HashMap::new();
        let mut valid = !path.is_file();
        let mut valid_len = 0usize; // end of the last complete record
        if let Ok(bytes) = std::fs::read(&path) {
            valid = false;
            if bytes.len() >= SCORE_HEADER
                && &bytes[..8] == SCORE_MAGIC
                && u32::from_le_bytes(bytes[8..12].try_into().unwrap()) == SCORE_VERSION
                && u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize == n_experts
            {
                let rec = 4 + 8 * n_experts;
                let mut pos = SCORE_HEADER;
                while pos + rec <= bytes.len() {
                    let layer = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
                    let v: Vec<f64> = (0..n_experts)
                        .map(|i| f64::from_le_bytes(bytes[pos + 4 + 8 * i..pos + 12 + 8 * i].try_into().unwrap()))
                        .collect();
                    scores.insert(layer, v); // a duplicate layer record: last wins
                    pos += rec;
                }
                // a torn trailing record (kill between write and fsync):
                // dropped below, otherwise the records appended after it
                // would be orphaned behind unparsable bytes
                valid = true;
                valid_len = pos;
                if !scores.is_empty() {
                    println!("experts: score cache {} ({} layers restored)", path.display(), scores.len());
                }
            }
        }
        if !valid {
            println!("experts: ignoring score cache {} (bad magic/version/expert count), rescoring", path.display());
            std::fs::remove_file(&path).ok();
            scores = std::collections::HashMap::new();
        }
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).unwrap();
        let len = f.metadata().unwrap().len() as usize;
        if len == 0 {
            let mut hdr = Vec::with_capacity(SCORE_HEADER);
            hdr.extend_from_slice(SCORE_MAGIC);
            hdr.extend_from_slice(&SCORE_VERSION.to_le_bytes());
            hdr.extend_from_slice(&(n_layers as u32).to_le_bytes());
            hdr.extend_from_slice(&(n_experts as u32).to_le_bytes());
            f.write_all(&hdr).unwrap();
            f.sync_data().unwrap();
        } else if valid && valid_len < len {
            f.set_len(valid_len as u64).unwrap();
            println!("experts: score cache torn tail dropped ({} bytes)", len - valid_len);
        }
        ScoreCache { file: std::sync::Mutex::new(f), scores }
    }

    fn get(&self, layer: usize) -> Option<&Vec<f64>> {
        self.scores.get(&layer)
    }

    /// Appends one layer record in ONE write + fsync (kill -9 safe).
    fn record(&self, layer: usize, scores: &[f64]) {
        use std::io::Write;
        let mut buf = Vec::with_capacity(4 + 8 * scores.len());
        buf.extend_from_slice(&(layer as u32).to_le_bytes());
        for s in scores {
            buf.extend_from_slice(&s.to_le_bytes());
        }
        let mut f = self.file.lock().unwrap();
        f.write_all(&buf).unwrap();
        f.sync_data().unwrap();
    }
}

/// (new expert id, w index) of an expert plan's output name
/// "layers.N.block_sparse_moe.experts.<id>.<w>", for the physical re-sort of
/// a reordered (--expert-order) layer's expert run.
fn expert_plan_key(out_name: &str) -> (usize, u8) {
    let tail = out_name.rsplit(".experts.").next().unwrap();
    let dot = tail.find('.').unwrap();
    let id: usize = tail[..dot].parse().unwrap();
    let w = match &tail[dot + 1..] {
        "w1" => 0,
        "w2" => 1,
        "w3" => 2,
        _ => 3,
    };
    (id, w)
}

pub fn run(args: &[String]) {
    let t0 = std::time::Instant::now();
    let Some(model) = value_flag(args, "--model") else {
        eprintln!("error: slice requires --model X.bin | model.safetensors | dir/ | https://huggingface.co/org/repo");
        std::process::exit(1);
    };
    let Some(out) = value_flag(args, "--out") else {
        eprintln!("error: slice requires --out Y.bin");
        std::process::exit(1);
    };
    let hidden: Option<usize> = value_flag(args, "--hidden").map(|s| s.parse().expect("bad --hidden"));
    let experts: Option<usize> = value_flag(args, "--experts").map(|s| s.parse().expect("bad --experts"));
    let layers_spec = value_flag(args, "--layers");
    // --cold-vq N: precision-tiered expert storage. ALL experts stay in the
    // file (the router is untouched); per MoE layer the top-N experts by
    // Frobenius score stay mxfp4 (hot), the rest are requantized to VQ1
    // (cold, ~0.5 bit/weight, one global 256x16 codebook). Combined with
    // --experts M (M >= N): only the top-M experts are kept, top-N hot.
    let cold_vq: Option<usize> = value_flag(args, "--cold-vq").map(|s| s.parse().expect("bad --cold-vq"));
    // --vocab-top N <freqfile>: vocabulary pruning (see the header comment).
    // N rows by frequency + every special token; the remap rides next to the
    // .bin as <stem>.vocab.json (engine --vocab compatible).
    let vocab_top: Option<(usize, String)> = args.iter().position(|a| a == "--vocab-top").map(|i| {
        let n: usize = args.get(i + 1).and_then(|s| s.parse().ok()).expect("--vocab-top needs N (rows to keep) and a freqfile path");
        let f = args.get(i + 2).cloned().expect("--vocab-top needs a freqfile path after N");
        (n, f)
    });
    let vocab_base = value_flag(args, "--vocab-base");
    // --expert-order=frequency --route-cms SKETCH: physical reorder of the
    // expert blobs of every MoE layer by descending routing frequency (the
    // count-min sketch recorded with MICROKIMI_ROUTECMS, cms.rs), hottest
    // expert first. The router gate rows and bias are permuted with the same
    // order, so expert ids are simply relabeled: the model is mathematically
    // unchanged, any engine reads the reordered .bin (old engines included)
    // and old .bins keep working. The point is physical: hot experts become
    // file-adjacent, so the stream engine's contiguous-run fusion (stream.rs
    // warm_batch) serves the top-k batch of a layer in far fewer physical
    // reads. The permutation rides in the MKIM0002 config as an index table
    // ("expert_order"). Combined with --experts N the keep-set membership
    // still comes from the Frobenius scores, only the file order changes.
    let kv_flag = |name: &str| {
        value_flag(args, name).or_else(|| args.iter().find_map(|a| a.strip_prefix(&format!("{}=", name)).map(|s| s.to_string())))
    };
    let expert_order_flag = kv_flag("--expert-order");
    let route_cms_path = kv_flag("--route-cms");
    if let Some(o) = &expert_order_flag {
        if o != "frequency" {
            eprintln!("error: --expert-order supports only 'frequency' (got '{}')", o);
            std::process::exit(1);
        }
        if route_cms_path.is_none() {
            eprintln!("error: --expert-order=frequency requires --route-cms SKETCH (record one with MICROKIMI_ROUTECMS=SKETCH)");
            std::process::exit(1);
        }
    }
    if hidden.is_none() && experts.is_none() && layers_spec.is_none() && cold_vq.is_none() && vocab_top.is_none() && expert_order_flag.is_none() {
        eprintln!("error: slice needs at least one of --hidden / --experts / --layers / --cold-vq / --vocab-top / --expert-order");
        std::process::exit(1);
    }
    if let (Some(m), Some(n)) = (experts, cold_vq) {
        assert!(n <= m, "--cold-vq N must be <= --experts M (hot experts are a subset of the kept ones)");
    }
    if cold_vq.is_some() {
        assert!(
            !model.starts_with("http://") && !model.starts_with("https://"),
            "--cold-vq requires a local .bin source (mxfp4 expert blobs)"
        );
    }

    let mut source = Source::open(&model, &out);
    if cold_vq.is_some() {
        assert!(matches!(source, Source::Bin(_)), "--cold-vq requires a .bin source (mxfp4 expert blobs)");
    }
    // --imatrix FILE: activation importance stats (microkimi calibrate) used
    // to weight the VQ codebook training + assignment of the cold experts.
    // --imatrix-score-only: load the same stats but only to REPORT the
    // activation-weighted error of the blind codebook (A/B measurement).
    let imatrix_score_only = args.iter().any(|a| a == "--imatrix-score-only");
    let imatrix: Option<crate::imatrix::Imatrix> = match value_flag(args, "--imatrix") {
        Some(p) => {
            assert!(cold_vq.is_some(), "--imatrix only applies to --cold-vq");
            let im = crate::imatrix::load(&p).unwrap_or_else(|e| {
                eprintln!("error: {}", e);
                std::process::exit(1);
            });
            let cfg0 = source.config();
            assert_eq!(im.routed_hidden, cfg0.routed_hidden, "imatrix routed_hidden {} != model {}", im.routed_hidden, cfg0.routed_hidden);
            assert_eq!(im.moe_inter, cfg0.moe_inter, "imatrix moe_inter {} != model {}", im.moe_inter, cfg0.moe_inter);
            println!(
                "imatrix: {} ({} tokens, {} MoE layers){}",
                p,
                im.tokens,
                im.layers.len(),
                if imatrix_score_only { " [score only: blind codebook, weighted error report]" } else { "" }
            );
            Some(im)
        }
        None => None,
    };

    // ── 1. layer selection (then resolve tensor shapes/byte sources) ──
    let kept_layers = match &layers_spec {
        Some(s) => parse_layer_spec(s, source.config().n_layers),
        None => (0..source.config().n_layers).collect(),
    };
    let new_layer_of = |old: usize| kept_layers.iter().position(|&l| l == old);
    source.resolve(&kept_layers);
    // with --hidden the non-expert tensors are read twice (scoring, writing):
    // cache the converted bytes on disk so remote bytes are fetched once
    if hidden.is_some() {
        source.enable_caching();
    }
    let arch = source.arch();
    let cfg = source.config();
    let d = cfg.d;
    println!(
        "slice: {} ({} layers, hidden {}, {} experts top-{} + {} shared, vocab {})",
        model, cfg.n_layers, d, cfg.n_experts, cfg.top_k, cfg.n_shared, cfg.vocab
    );
    println!("layers: keeping {}/{} {:?}", kept_layers.len(), cfg.n_layers, kept_layers);

    // physical expert order (--expert-order=frequency): per kept MoE layer
    // the old expert ids in write order, hottest first by count-min estimate
    // (ties and never-recorded experts id-ascending, deterministic). Computed
    // before the checkpoint key: the sketch identity is part of the key.
    let mut eorder_key = String::new();
    let expert_order: Option<std::collections::HashMap<usize, Vec<usize>>> = expert_order_flag.as_ref().map(|_| {
        let path = route_cms_path.as_deref().unwrap();
        let sketch = crate::cms::Cms::load(path).unwrap_or_else(|e| {
            eprintln!("error: {}", e);
            std::process::exit(1);
        });
        eorder_key = format!("frequency:{}:{}", path, sketch.total());
        let mut m = std::collections::HashMap::new();
        for &l in &kept_layers {
            if !cfg.is_moe(l) {
                continue;
            }
            let mut ids: Vec<usize> = (0..cfg.n_experts).collect();
            ids.sort_by_key(|&e| (std::cmp::Reverse(sketch.estimate(l as u32, e as u32)), e));
            m.insert(l, ids);
        }
        println!(
            "expert-order: frequency from {} ({} routing decisions, {} MoE layers reordered, hottest first)",
            path,
            sketch.total(),
            m.len()
        );
        m
    });

    // crash-safe resume checkpoint: model + kept layers + pruning params
    // (vocab-top is part of the key: N and the freqfile content hash)
    let vocabtop_key = vocab_top
        .as_ref()
        .map(|(n, f)| {
            let text = std::fs::read_to_string(f).unwrap_or_else(|e| panic!("freqfile {} unreadable: {}", f, e));
            format!("{}:{:016x}", n, fnv1a(&text))
        })
        .unwrap_or_default();
    let ckpt_key = format!(
        "model={}|layers={}|hidden={}|experts={}|vocabtop={}|eorder={}",
        model,
        join_csv(&kept_layers),
        hidden.map(|h| h.to_string()).unwrap_or_default(),
        experts.map(|e| e.to_string()).unwrap_or_default(),
        vocabtop_key,
        eorder_key
    );
    let ckpt = SliceCkpt::open(&out, &ckpt_key);

    // ── 2. channel selection (scored on the kept layers only) ──
    let channels: Option<Vec<usize>> = hidden.map(|h| {
        assert!(h > 0 && h <= d, "--hidden must be in 1..={}", d);
        if let Some(ch) = &ckpt.channels {
            println!("hidden: {}/{} channels restored from checkpoint", ch.len(), d);
            return ch.clone();
        }
        let n_scored = source
            .entries()
            .iter()
            .filter(|e| {
                split_layer(&e.name).map(|(l, _)| kept_layers.contains(&l)).unwrap_or(true)
                    && !matches!(role_of(&e.name, cfg, arch), Role::Copy | Role::RouterB | Role::Expert)
            })
            .count();
        println!("hidden: scoring channels over {} tensors...", n_scored);
        let scores = channel_scores(&source, &kept_layers, d);
        let keep = top_n(&scores, h);
        println!("hidden: keeping {}/{} channels (top-|w|), score range {:.3} .. {:.3}", h, d,
            keep.iter().map(|&i| scores[i]).fold(f64::INFINITY, f64::min),
            keep.iter().map(|&i| scores[i]).fold(f64::NEG_INFINITY, f64::max));
        ckpt.record_channels(&keep);
        keep
    });

    // ── 3. expert selection (per kept MoE layer) ──
    let expert_sets = experts.map(|n| {
        assert!(n > 0, "--experts must be >= 1");
        let t = std::time::Instant::now();
        // persistent full-score cache (config-independent, one level below
        // the per-run .sliceckpt): saves the whole scoring on reruns
        let score_cache = ScoreCache::open(&out, &model, cfg.n_layers, cfg.n_experts);
        let sets = expert_keep_sets(&source, &kept_layers, n, &ckpt, &score_cache);
        let how = if matches!(source, Source::Bin(_)) {
            "Frobenius of dequantized w1+w2+w3"
        } else {
            "scale-energy of w1+w2+w3 (weight_scale tensors only, 1/17 of the bytes)"
        };
        println!("experts: keeping {}/{} per MoE layer ({}), scored in {:.1?}", n, cfg.n_experts, how, t.elapsed());
        sets
    });

    // fold --expert-order into the per-layer expert lists the plan builder
    // uses: the list order IS the new expert id order (Expert rename and
    // RouterW/RouterB row gather both follow it). With --experts N the
    // Frobenius keep-set membership is preserved, only the order changes.
    let expert_sets = match &expert_order {
        None => expert_sets,
        Some(order) => Some(
            order
                .iter()
                .map(|(&l, ids)| {
                    let ids: Vec<usize> = match &expert_sets {
                        Some(sets) => {
                            let keep: std::collections::HashSet<usize> = sets[&l].iter().copied().collect();
                            ids.iter().copied().filter(|e| keep.contains(e)).collect()
                        }
                        None => ids.clone(),
                    };
                    (l, ids)
                })
                .collect(),
        ),
    };

    // ── 3b. precision tiering (--cold-vq): hot/cold split + global codebook ──
    // vq_hot: per kept MoE layer the hot (mxfp4) expert indices, ascending.
    // The codebook is trained on a seeded reservoir sample of ALL cold-expert
    // dequantized values, raw 16-vectors (no per-vector normalization: the
    // mxfp4 source already keeps per-32-group magnitudes similar enough that
    // raw VQ works; measured in the microquant report).
    let (vq_hot, vq_codebook): (Option<std::collections::HashMap<usize, Vec<usize>>>, Option<Vec<f32>>) =
        match cold_vq {
            None => (None, None),
            Some(n_hot) => {
                let t = std::time::Instant::now();
                let scores = expert_score_map(&source, &kept_layers);
                let hot: std::collections::HashMap<usize, Vec<usize>> =
                    scores.iter().map(|(&l, s)| (l, top_n(s, n_hot.min(cfg.n_experts)))).collect();
                // cold = kept experts minus the hot ones (with --experts M
                // the pruned tail never reaches the file NOR the codebook)
                let cold: std::collections::HashMap<usize, Vec<usize>> = scores
                    .iter()
                    .map(|(&l, s)| {
                        let keep: std::collections::HashSet<usize> = hot[&l].iter().copied().collect();
                        let kept: Vec<usize> = match &expert_sets {
                            Some(sets) => sets[&l].clone(),
                            None => (0..s.len()).collect(),
                        };
                        (l, kept.into_iter().filter(|e| !keep.contains(e)).collect())
                    })
                    .collect();
                let n_cold: usize = cold.values().map(|v| v.len()).sum();
                if n_cold == 0 {
                    println!("cold-vq: hot set covers all experts, nothing to quantize (no VQ tensors written)");
                    (Some(hot), None)
                } else {
                    println!(
                        "cold-vq: {} hot mxfp4 + {} cold VQ1 experts per MoE layer (ranked in {:.1?})",
                        n_hot.min(cfg.n_experts),
                        n_cold / cold.len().max(1),
                        t.elapsed()
                    );
                    let t = std::time::Instant::now();
                    let seed = 0x5EED_C0DE_B00B_1E5u64;
                    let train_im = if imatrix_score_only { None } else { imatrix.as_ref() };
                    let (samples, sample_w) = vq_reservoir(&source, &kept_layers, &cold, 300_000, seed, train_im);
                    let cb = match &sample_w {
                        Some(sw) => crate::quant::train_codebook_weighted(&samples, sw, seed),
                        None => crate::quant::train_codebook(&samples, seed),
                    };
                    println!(
                        "vq: global codebook ({}x{}) trained in {:.1?}{}",
                        crate::quant::VQ_K,
                        crate::quant::VQ_DIM,
                        t.elapsed(),
                        if sample_w.is_some() { " (activation-weighted)" } else { "" }
                    );
                    (Some(hot), Some(cb))
                }
            }
        };

    // ── 3c. vocabulary selection (--vocab-top): cheap, not checkpointed ──
    let vocab_plan: Option<VocabPlan> = vocab_top.as_ref().map(|(n, freq)| {
        assert!(*n > 0, "--vocab-top must be >= 1");
        build_vocab_plan(&model, &out, &source.source_json(), cfg, *n, freq, vocab_base.clone())
    });

    // ── 4. plan: output tensors in input directory order ──
    let mut plans: Vec<Plan> = Vec::new();
    for e in source.entries() {
        let role = role_of(&e.name, cfg, arch);
        let (out_name, experts_for_tensor, dtype_override): (String, Option<Vec<usize>>, Option<u8>) = match split_layer(&e.name) {
            None => (e.name.clone(), None, None),
            Some((l, rest)) => {
                let Some(nl) = new_layer_of(l) else { continue }; // pruned layer
                let pfx = format!("layers.{}.", nl);
                if role == Role::Expert {
                    // block_sparse_moe.experts.{e}.{w}
                    let tail = rest.strip_prefix("block_sparse_moe.experts.").unwrap();
                    let dot = tail.find('.').unwrap();
                    let oe: usize = tail[..dot].parse().unwrap();
                    let keep = expert_sets.as_ref().map(|s| &s[&l]);
                    let idx = match keep {
                        Some(k) => match k.iter().position(|&x| x == oe) {
                            Some(i) => i,
                            None => continue, // pruned expert
                        },
                        None => oe,
                    };
                    // cold experts (below the hot top-N) become VQ1
                    let cold = vq_hot.as_ref().is_some_and(|h| !h[&l].contains(&oe));
                    let dt = if cold { Some(DTYPE_VQ1) } else { None };
                    (format!("{}block_sparse_moe.experts.{}.{}", pfx, idx, &tail[dot + 1..]), None, dt)
                } else if matches!(role, Role::RouterW | Role::RouterB) {
                    (format!("{}{}", pfx, rest), expert_sets.as_ref().map(|s| s[&l].clone()), None)
                } else {
                    (format!("{}{}", pfx, rest), None, None)
                }
            }
        };
        let ch: Vec<usize> = channels.clone().unwrap_or_else(|| (0..d).collect());
        // embed/lm_head rows are the vocab axis: pruned by --vocab-top
        let vrows: Option<Vec<usize>> = match (&vocab_plan, e.name.as_str()) {
            (Some(v), "embed_tokens.weight" | "lm_head.weight") => Some(v.keep.clone()),
            _ => None,
        };
        let dims = if matches!(role, Role::Copy | Role::Expert) {
            e.dims.clone()
        } else {
            // compute the sliced dims without materializing the data
            let r = e.dims[0] as usize;
            let out_rows = vrows.as_ref().map(|k| k.len()).unwrap_or(r) as u32;
            match role {
                Role::VecD => vec![ch.len() as u32],
                Role::ColsD => vec![out_rows, ch.len() as u32],
                Role::RowsD => vec![ch.len() as u32, e.dims[1]],
                Role::BothD => vec![ch.len() as u32, ch.len() as u32],
                Role::RouterW => vec![experts_for_tensor.as_ref().map(|k| k.len()).unwrap_or(r) as u32, ch.len() as u32],
                Role::RouterB => vec![experts_for_tensor.as_ref().map(|k| k.len()).unwrap_or(r) as u32],
                _ => unreachable!(),
            }
        };
        plans.push(Plan {
            out_name,
            dtype: dtype_override.unwrap_or(e.dtype),
            dims,
            src_name: e.name.clone(),
            role,
            channels: ch,
            experts: experts_for_tensor,
            vocab: vrows,
        });
    }
    // physical write order with --expert-order: the plan list follows the
    // SOURCE directory order (old expert ids), which would scatter the
    // relabeled blobs. Re-sort each layer's expert run by new expert id
    // (then w1/w2/w3) so file-adjacent ids are byte-adjacent - the
    // precondition of the stream engine's run fusion. Everything else keeps
    // the source order.
    if expert_order.is_some() {
        let mut i = 0;
        while i < plans.len() {
            if plans[i].role != Role::Expert {
                i += 1;
                continue;
            }
            let layer = split_layer(&plans[i].out_name).map(|(l, _)| l);
            let mut j = i + 1;
            while j < plans.len() && plans[j].role == Role::Expert && split_layer(&plans[j].out_name).map(|(l, _)| l) == layer {
                j += 1;
            }
            plans[i..j].sort_by_key(|p| expert_plan_key(&p.out_name));
            i = j;
        }
    }
    // the global VQ codebook rides as one extra f32 tensor (src_name "" marks it)
    if vq_codebook.is_some() {
        plans.push(Plan {
            out_name: "vq_codebook".to_string(),
            dtype: DTYPE_F32,
            dims: vec![crate::quant::VQ_K as u32, crate::quant::VQ_DIM as u32],
            src_name: String::new(),
            role: Role::Copy,
            channels: Vec::new(),
            experts: None,
            vocab: None,
        });
    }

    // ── 5. MKIM0002 config ──
    let new_n_layers = kept_layers.len();
    let new_d = channels.as_ref().map(|c| c.len()).unwrap_or(d);
    let new_n_experts = experts.unwrap_or(cfg.n_experts);
    let new_top_k = cfg.top_k.min(new_n_experts);
    let new_vocab = vocab_plan.as_ref().map(|v| v.keep.len()).unwrap_or(cfg.vocab);
    // specials: with --vocab-top every known special is recorded at its NEW
    // id (a re-slice can then find them all again); otherwise the historical
    // bos/end_of_msg pair, unchanged.
    let specials_kv = match &vocab_plan {
        Some(v) => v
            .specials_new
            .iter()
            .map(|(n, id)| format!("\"{}\": {}", n, id))
            .collect::<Vec<_>>()
            .join(", "),
        None => format!("\"bos\": {}, \"end_of_msg\": {}", cfg.bos_id, cfg.eos_id),
    };
    let mla_layers: Vec<usize> = kept_layers.iter().enumerate().filter(|&(_, &l)| cfg.is_mla(l)).map(|(i, _)| i).collect();
    let dense_layers: Vec<usize> = kept_layers.iter().enumerate().filter(|&(_, &l)| !cfg.is_moe(l)).map(|(i, _)| i).collect();
    let tokenizer_kv = source
        .source_json()
        .get("tokenizer")
        .and_then(|t| t.as_str().map(|s| s.to_string()))
        .map(|s| format!(", \"tokenizer\": \"{}\"", s))
        .unwrap_or_default();
    let arch_kv = source.arch_config_key();
    let list = |v: &[usize]| v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
    // --expert-order audit table: new_id -> old_id per (renumbered) MoE layer
    let expert_order_kv = expert_order.as_ref().map(|m| {
        let layers: Vec<String> = kept_layers
            .iter()
            .enumerate()
            .filter_map(|(nl, &l)| m.get(&l).map(|ids| format!("\"{}\": [{}]", nl, list(ids))))
            .collect();
        format!(
            ", \"expert_order\": {{\"method\": \"frequency\", \"source\": \"{}\", \"new_to_old\": {{{}}}}}",
            route_cms_path.as_deref().unwrap(),
            layers.join(", ")
        )
    });
    let config_json = format!(
        "{{\"format\": 2{}, \"n_layers\": {}, \"hidden\": {}, \"vocab\": {}, \"n_experts\": {}, \"top_k\": {}, \"n_shared\": {}, \
\"kda_heads\": {}, \"kda_dim\": {}, \"kda_conv\": {}, \"kda_fa_rank\": {}, \"gate_lower_bound\": {}, \
\"mla_heads\": {}, \"mla_q_lora\": {}, \"mla_kv_lora\": {}, \"mla_nope\": {}, \"mla_rope\": {}, \"mla_v\": {}, \
\"routed_hidden\": {}, \"moe_inter\": {}, \"shared_inter\": {}, \"dense_inter\": {}, \
\"attn_res_block\": {}, \"first_k_dense\": {}, \"rms_eps\": {}{}, \
\"mla_layers\": [{}], \"dense_layers\": [{}], \
\"specials\": {{{}}}, \
\"pruning\": {{\"method\": \"weight-magnitude-v1\", \"hidden\": {}, \"experts\": {}, \"layers\": \"{}\"{}{}}}}}",
        arch_kv,
        new_n_layers, new_d, new_vocab, new_n_experts, new_top_k, cfg.n_shared,
        cfg.kda_heads, cfg.kda_dim, cfg.kda_conv, cfg.kda_fa, cfg.gate_lb,
        cfg.mla_heads, cfg.mla_qa, cfg.mla_kva, cfg.mla_nope, cfg.mla_rope, cfg.mla_v,
        cfg.routed_hidden, cfg.moe_inter, cfg.shared_inter, cfg.dense_inter,
        cfg.attn_res_block, cfg.first_k_dense, cfg.rms_eps, tokenizer_kv,
        list(&mla_layers), list(&dense_layers),
        specials_kv,
        new_d, new_n_experts, kept_layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(","),
        cold_vq.map(|n| format!(", \"cold_vq\": {}", n)).unwrap_or_default(),
        vocab_top.as_ref().map(|(n, _)| format!(", \"vocab_top\": {}", n)).unwrap_or_default(),
    );
    // the expert_order audit table goes in as a top-level key (inserted
    // post-hoc: one less placeholder in an already dense format string)
    let mut config_json = config_json;
    if let Some(kv) = expert_order_kv {
        config_json.insert_str(config_json.len() - 1, &kv);
    }

    // ── 6. write ──
    let mut w = BinWriter::new();
    if expert_order.is_some() {
        // dense expert packing: the reordered blobs are read in fused spans,
        // page-alignment padding would be read and discarded on every span
        w.set_expert_align(64);
    }
    for p in &plans {
        w.add(&p.out_name, p.dtype, p.dims.clone());
    }
    let mut f = std::fs::File::create(&out).unwrap();
    let offsets = w.write_header_v2(&mut f, &config_json);
    let mut done = 0usize;
    let mut last_fetch_report = 0u64;
    let mut cur_layer: Option<usize> = None;
    let mut vq_err_sum = 0f64;
    let mut vq_werr_sum = 0f64;
    let mut vq_err_n = 0u64;
    for (p, &off) in plans.iter().zip(&offsets) {
        // the codebook plan has no source tensor: its data is the trained codebook
        if p.src_name.is_empty() {
            let cb = vq_codebook.as_ref().expect("codebook plan without --cold-vq");
            w.write_blob_at(&mut f, off, &f32_to_bytes(cb));
            done += 1;
            continue;
        }
        let se = source.entry(&p.src_name);
        match p.role {
            Role::Expert if p.dtype == DTYPE_VQ1 => {
                // imatrix column weights for this expert matrix (original
                // layer numbering: src_name), w1/w3 -> hidden, w2 -> inter
                let wts = imatrix.as_ref().and_then(|im| {
                    let (l, rest) = split_layer(&p.src_name)?;
                    im.col_weights(l, rest.rsplit('.').next()?)
                });
                let quant_w = if imatrix_score_only { None } else { wts.as_deref() };
                let (idx, err, werr) = vq_quantize_tensor(&source, se, vq_codebook.as_ref().unwrap(), quant_w, wts.as_deref());
                assert_eq!(blob_size(p.dtype, &p.dims), idx.len() as u64, "{}: vq size mismatch", p.src_name);
                w.write_blob_at(&mut f, off, &idx);
                vq_err_sum += err;
                if let Some(we) = werr {
                    vq_werr_sum += we;
                }
                vq_err_n += 1;
                if vq_err_n % 500 == 0 {
                    println!(
                        "  vq: {} tensors quantized, mean rel Frobenius error {:.3}{}",
                        vq_err_n,
                        vq_err_sum / vq_err_n as f64,
                        if werr.is_some() {
                            format!(", mean activation-weighted error {:.3}", vq_werr_sum / vq_err_n as f64)
                        } else {
                            String::new()
                        }
                    );
                }
            }
            Role::Copy | Role::Expert => {
                let blob = source.raw_blob(se);
                assert_eq!(blob_size(p.dtype, &p.dims), blob.len() as u64, "{}: size mismatch on copy", p.src_name);
                w.write_blob_at(&mut f, off, &blob);
            }
            Role::ColsD | Role::RowsD => {
                let (rows, cols) = (n_rows(se), row_width(se));
                let mut written = 0u64;
                for (r0, r1) in row_chunks(rows, cols) {
                    let vals = source.f32_rows(se, r0, r1);
                    let sliced = match &p.vocab {
                        Some(keep) => slice_vocab_rows(&vals, r0, r1, cols, &p.channels, keep),
                        None => slice_f32_rows(p.role, &vals, r0, r1, cols, &p.channels),
                    };
                    let bytes = f32_to_bytes(&sliced);
                    w.write_blob_at(&mut f, off + written, &bytes);
                    written += bytes.len() as u64;
                }
                assert_eq!(written, blob_size(DTYPE_F32, &p.dims), "{}: planned dims mismatch", p.src_name);
            }
            _ => {
                let (vals, dims) = slice_f32(se, &source.f32_rows(se, 0, n_rows(se)), p.role, &p.channels, p.experts.as_ref());
                assert_eq!(dims, p.dims, "{}: planned dims mismatch", p.src_name);
                w.write_blob_at(&mut f, off, &f32_to_bytes(&vals));
            }
        }
        done += 1;
        // one progress line per layer written (plans follow the directory
        // order, so layers are contiguous)
        if let Some((nl, _)) = split_layer(&p.out_name) {
            if cur_layer != Some(nl) {
                cur_layer = Some(nl);
                println!("  write: layer {}/{} ({}% of tensors)", nl + 1, new_n_layers, 100 * done / plans.len());
            }
        }
        if done % 20000 == 0 {
            println!("  {}/{} tensors written ({:.0?})", done, plans.len(), t0.elapsed());
        }
        if source.is_remote() {
            let fb = crate::http::fetched_bytes();
            if fb - last_fetch_report >= (1 << 30) {
                last_fetch_report = fb;
                println!("  fetched {:.2} GB so far ({}/{} tensors written)", fb as f64 / 1e9, done, plans.len());
            }
        }
    }
    let out_size = std::fs::metadata(&out).unwrap().len();
    println!();
    println!("══ {} : {} tensors ══", out, plans.len());
    match std::fs::metadata(&model).ok().map(|m| m.len()) {
        Some(in_size) if !source.is_remote() => println!(
            "  size: {:.2} GB -> {:.2} GB ({:.1}%)",
            in_size as f64 / 1e9,
            out_size as f64 / 1e9,
            out_size as f64 / in_size as f64 * 100.0
        ),
        _ => println!("  input: remote safetensors via range requests (no full shard downloaded) -> {:.2} GB", out_size as f64 / 1e9),
    }
    if source.is_remote() {
        println!(
            "  bandwidth: {:.3} GB fetched in {} HTTP range requests",
            crate::http::fetched_bytes() as f64 / 1e9,
            crate::http::fetched_requests()
        );
    }
    println!("  config: {} layers (MLA {:?}, dense {:?}), hidden {}, {} experts top-{}", 
        new_n_layers, mla_layers, dense_layers, new_d, new_n_experts, new_top_k);
    println!("  AttnRes: block={} re-applied on the renumbered layers", cfg.attn_res_block);
    println!("  experts: mxfp4 blobs copied verbatim (no requantization)");
    if expert_order.is_some() {
        println!("  expert-order: frequency (route-cms) - router rows and expert blobs relabeled, hot experts file-adjacent, dense packing");
    }
    if vq_err_n > 0 {
        println!(
            "  cold-vq: {} tensors requantized to VQ1 ({} B/expert-matrix + one 16 KB global codebook), mean rel Frobenius error {:.3}",
            vq_err_n,
            cfg.moe_inter * cfg.routed_hidden / crate::quant::VQ_DIM,
            vq_err_sum / vq_err_n as f64
        );
        if imatrix.is_some() {
            println!(
                "  cold-vq: mean activation-weighted rel error {:.3} (imatrix{})",
                vq_werr_sum / vq_err_n as f64,
                if imatrix_score_only { ", score only" } else { "-weighted codebook" }
            );
        }
    }
    println!("  done in {:.0?}", t0.elapsed());
    if let Some(v) = &vocab_plan {
        std::fs::write(&v.remap_path, &v.remap_json).unwrap_or_else(|e| panic!("{} unwritable: {}", v.remap_path, e));
        println!("  vocab: {} -> {} rows kept; runtime remap: {}", cfg.vocab, v.keep.len(), v.remap_path);
        println!("         run with: microkimi run \"...\" --model {} --vocab {}", out, v.remap_path);
    }
    ckpt.finish();
}
