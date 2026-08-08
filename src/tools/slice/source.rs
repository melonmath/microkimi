// Source .bin/safetensors directory access and tensor role classification (moved from slice.rs).

use crate::config::Config;
use crate::quant::weights::{DTYPE_F32, DTYPE_MXFP4, MAGIC, MAGIC_V2};
use crate::tools::slice_st::{StArch, StDir};
use std::io::Read;

/// Row chunk for streaming f32 processing (~256 MB of f32 per chunk: the
/// 163840x7168 embeddings never sit in RAM as a whole).
const CHUNK_VALS: usize = 1 << 26;

pub(super) fn n_rows(e: &DirEntry) -> usize {
    if e.dims.len() <= 1 { 1 } else { e.dims[0] as usize }
}

pub(super) fn row_width(e: &DirEntry) -> usize {
    if e.dims.len() <= 1 { e.dims[0] as usize } else { e.dims[1..].iter().map(|&d| d as usize).product() }
}

pub(super) fn row_chunks(rows: usize, cols: usize) -> Vec<(usize, usize)> {
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
pub(super) struct BinDir {
    config: Config,
    source_json: crate::json::Json,
    entries: Vec<DirEntry>, // in file (directory) order: deterministic output
    index: std::collections::HashMap<String, usize>, // name -> entries position
    file: std::fs::File,
}

impl BinDir {
    pub(super) fn open(path: &str) -> Self {
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

    pub(super) fn entry(&self, name: &str) -> &DirEntry {
        &self.entries[*self.index.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name))]
    }

    /// Raw blob of a tensor (thread-safe: read_exact_at takes &File).
    pub(super) fn blob(&self, e: &DirEntry) -> Vec<u8> {
        use std::os::unix::fs::FileExt;
        let mut buf = vec![0u8; e.size as usize];
        self.file.read_exact_at(&mut buf, e.offset).unwrap();
        buf
    }

    /// Rows r0..r1 of an f32 tensor (whole row width).
    pub(super) fn f32_rows(&self, e: &DirEntry, r0: usize, r1: usize) -> Vec<f32> {
        assert_eq!(e.dtype, DTYPE_F32, "{} is not f32", e.name);
        use std::os::unix::fs::FileExt;
        let cols = row_width(e);
        let mut buf = vec![0u8; (r1 - r0) * cols * 4];
        self.file.read_exact_at(&mut buf, e.offset + (r0 * cols * 4) as u64).unwrap();
        buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    /// Only the scale bytes of an MXFP4 expert blob (packed ++ scales layout).
    pub(super) fn expert_scales(&self, e: &DirEntry) -> Vec<u8> {
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
pub(super) enum Source {
    Bin(BinDir),
    St(StDir),
}

impl Source {
    pub(super) fn open(model: &str, cache_hint: &str) -> Source {
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

    pub(super) fn config(&self) -> &Config {
        match self {
            Source::Bin(b) => &b.config,
            Source::St(s) => &s.config,
        }
    }

    pub(super) fn entries(&self) -> &[DirEntry] {
        match self {
            Source::Bin(b) => &b.entries,
            Source::St(s) => &s.entries,
        }
    }

    pub(super) fn entry(&self, name: &str) -> &DirEntry {
        match self {
            Source::Bin(b) => b.entry(name),
            Source::St(s) => &s.entries[*s.index.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name))],
        }
    }

    pub(super) fn arch(&self) -> Arch {
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

    pub(super) fn source_json(&self) -> crate::json::Json {
        match self {
            Source::Bin(b) => b.source_json.clone(),
            Source::St(_) => crate::json::Json::Null,
        }
    }

    /// MKIM0002 config "arch" marker to carry over (dense round-trips).
    pub(super) fn arch_config_key(&self) -> &'static str {
        match self.arch() {
            Arch::Dense => ", \"arch\": \"dense\"",
            _ => "",
        }
    }

    pub(super) fn is_remote(&self) -> bool {
        matches!(self, Source::St(s) if s.is_remote())
    }

    pub(super) fn resolve(&mut self, kept_layers: &[usize]) {
        if let Source::St(s) = self {
            s.resolve(kept_layers);
        }
    }

    pub(super) fn enable_caching(&self) {
        if let Source::St(s) = self {
            s.enable_caching();
        }
    }

    pub(super) fn f32_rows(&self, e: &DirEntry, r0: usize, r1: usize) -> Vec<f32> {
        match self {
            Source::Bin(b) => b.f32_rows(e, r0, r1),
            Source::St(s) => s.f32_rows(e, r0, r1),
        }
    }

    pub(super) fn raw_blob(&self, e: &DirEntry) -> Vec<u8> {
        match self {
            Source::Bin(b) => b.blob(e),
            Source::St(s) => s.raw_blob(e),
        }
    }

    pub(super) fn expert_scales(&self, e: &DirEntry) -> Vec<u8> {
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
pub(super) enum Arch {
    Micro,
    K3Real,
    Dense,
}

/// How a tensor relates to the hidden dimension d (for channel pruning) and
/// to the routed expert axis. Classification is NAME-based: several micro
/// dims coincide numerically (kda_proj == d == 512, mla kv_a rows == mla_qa),
/// so dims alone cannot identify which axis is the hidden one.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum Role {
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

pub(super) fn role_of(name: &str, cfg: &Config, arch: Arch) -> Role {
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
pub(super) fn role_dense(name: &str) -> Role {
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
