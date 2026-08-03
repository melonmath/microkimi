// `microkimi slice`: structural pruning of a K3 .bin model (MKIM0001/0002 in,
// MKIM0002 out). The output loads through the unmodified Model::load: every
// pruned dim is recorded in the MKIM0002 JSON config (hidden, n_experts,
// top_k, n_layers, mla_layers, dense_layers).
//
//   microkimi slice --model X.bin --out Y.bin [--hidden N] [--experts N] [--layers "spec"]
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
use crate::weights::{BinWriter, DTYPE_F32, DTYPE_MXFP4, MAGIC, MAGIC_V2, blob_size, f32_to_bytes};
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
fn expert_keep_sets(src: &Source, kept_layers: &[usize], n_keep: usize) -> std::collections::HashMap<usize, Vec<usize>> {
    let cfg = src.config();
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
    if matches!(src, Source::St(_)) {
        let mut results = Vec::new();
        for &l in &moe_layers {
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
            results.push((l, top_n(&scores, n_keep.min(cfg.n_experts))));
        }
        return results.into_iter().collect();
    }
    let results: Vec<(usize, Vec<usize>)> = std::thread::scope(|scope| {
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
                (l, top_n(&scores, n_keep.min(cfg.n_experts)))
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    results.into_iter().collect()
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

fn value_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
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
    if hidden.is_none() && experts.is_none() && layers_spec.is_none() {
        eprintln!("error: slice needs at least one of --hidden / --experts / --layers");
        std::process::exit(1);
    }

    let mut source = Source::open(&model, &out);

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

    // ── 2. channel selection (scored on the kept layers only) ──
    let channels: Option<Vec<usize>> = hidden.map(|h| {
        assert!(h > 0 && h <= d, "--hidden must be in 1..={}", d);
        let scores = channel_scores(&source, &kept_layers, d);
        let keep = top_n(&scores, h);
        println!("hidden: keeping {}/{} channels (top-|w|), score range {:.3} .. {:.3}", h, d,
            keep.iter().map(|&i| scores[i]).fold(f64::INFINITY, f64::min),
            keep.iter().map(|&i| scores[i]).fold(f64::NEG_INFINITY, f64::max));
        keep
    });

    // ── 3. expert selection (per kept MoE layer) ──
    let expert_sets = experts.map(|n| {
        assert!(n > 0, "--experts must be >= 1");
        let t = std::time::Instant::now();
        let sets = expert_keep_sets(&source, &kept_layers, n);
        let how = if matches!(source, Source::Bin(_)) {
            "Frobenius of dequantized w1+w2+w3"
        } else {
            "scale-energy of w1+w2+w3 (weight_scale tensors only, 1/17 of the bytes)"
        };
        println!("experts: keeping {}/{} per MoE layer ({}), scored in {:.1?}", n, cfg.n_experts, how, t.elapsed());
        sets
    });

    // ── 4. plan: output tensors in input directory order ──
    let mut plans: Vec<Plan> = Vec::new();
    for e in source.entries() {
        let role = role_of(&e.name, cfg, arch);
        let (out_name, experts_for_tensor): (String, Option<Vec<usize>>) = match split_layer(&e.name) {
            None => (e.name.clone(), None),
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
                    (format!("{}block_sparse_moe.experts.{}.{}", pfx, idx, &tail[dot + 1..]), None)
                } else if matches!(role, Role::RouterW | Role::RouterB) {
                    (format!("{}{}", pfx, rest), expert_sets.as_ref().map(|s| s[&l].clone()))
                } else {
                    (format!("{}{}", pfx, rest), None)
                }
            }
        };
        let ch: Vec<usize> = channels.clone().unwrap_or_else(|| (0..d).collect());
        let dims = if matches!(role, Role::Copy | Role::Expert) {
            e.dims.clone()
        } else {
            // compute the sliced dims without materializing the data
            let r = e.dims[0] as usize;
            match role {
                Role::VecD => vec![ch.len() as u32],
                Role::ColsD => vec![r as u32, ch.len() as u32],
                Role::RowsD => vec![ch.len() as u32, e.dims[1]],
                Role::BothD => vec![ch.len() as u32, ch.len() as u32],
                Role::RouterW => vec![experts_for_tensor.as_ref().map(|k| k.len()).unwrap_or(r) as u32, ch.len() as u32],
                Role::RouterB => vec![experts_for_tensor.as_ref().map(|k| k.len()).unwrap_or(r) as u32],
                _ => unreachable!(),
            }
        };
        plans.push(Plan {
            out_name,
            dtype: e.dtype,
            dims,
            src_name: e.name.clone(),
            role,
            channels: ch,
            experts: experts_for_tensor,
        });
    }

    // ── 5. MKIM0002 config ──
    let new_n_layers = kept_layers.len();
    let new_d = channels.as_ref().map(|c| c.len()).unwrap_or(d);
    let new_n_experts = experts.unwrap_or(cfg.n_experts);
    let new_top_k = cfg.top_k.min(new_n_experts);
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
    let config_json = format!(
        "{{\"format\": 2{}, \"n_layers\": {}, \"hidden\": {}, \"vocab\": {}, \"n_experts\": {}, \"top_k\": {}, \"n_shared\": {}, \
\"kda_heads\": {}, \"kda_dim\": {}, \"kda_conv\": {}, \"kda_fa_rank\": {}, \"gate_lower_bound\": {}, \
\"mla_heads\": {}, \"mla_q_lora\": {}, \"mla_kv_lora\": {}, \"mla_nope\": {}, \"mla_rope\": {}, \"mla_v\": {}, \
\"routed_hidden\": {}, \"moe_inter\": {}, \"shared_inter\": {}, \"dense_inter\": {}, \
\"attn_res_block\": {}, \"first_k_dense\": {}, \"rms_eps\": {}{}, \
\"mla_layers\": [{}], \"dense_layers\": [{}], \
\"specials\": {{\"bos\": {}, \"end_of_msg\": {}}}, \
\"pruning\": {{\"method\": \"weight-magnitude-v1\", \"hidden\": {}, \"experts\": {}, \"layers\": \"{}\"}}}}",
        arch_kv,
        new_n_layers, new_d, cfg.vocab, new_n_experts, new_top_k, cfg.n_shared,
        cfg.kda_heads, cfg.kda_dim, cfg.kda_conv, cfg.kda_fa, cfg.gate_lb,
        cfg.mla_heads, cfg.mla_qa, cfg.mla_kva, cfg.mla_nope, cfg.mla_rope, cfg.mla_v,
        cfg.routed_hidden, cfg.moe_inter, cfg.shared_inter, cfg.dense_inter,
        cfg.attn_res_block, cfg.first_k_dense, cfg.rms_eps, tokenizer_kv,
        list(&mla_layers), list(&dense_layers),
        cfg.bos_id, cfg.eos_id,
        new_d, new_n_experts, kept_layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(","),
    );

    // ── 6. write ──
    let mut w = BinWriter::new();
    for p in &plans {
        w.add(&p.out_name, p.dtype, p.dims.clone());
    }
    let mut f = std::fs::File::create(&out).unwrap();
    let offsets = w.write_header_v2(&mut f, &config_json);
    let mut done = 0usize;
    let mut last_fetch_report = 0u64;
    for (p, &off) in plans.iter().zip(&offsets) {
        let se = source.entry(&p.src_name);
        match p.role {
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
                    let bytes = f32_to_bytes(&slice_f32_rows(p.role, &vals, r0, r1, cols, &p.channels));
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
    println!("  done in {:.0?}", t0.elapsed());
}
