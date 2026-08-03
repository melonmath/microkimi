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
use crate::weights::{BinWriter, DTYPE_F32, DTYPE_MXFP4, MAGIC, MAGIC_V2, blob_size, f32_to_bytes};
use std::io::Read;

struct DirEntry {
    name: String,
    dtype: u8,
    dims: Vec<u32>,
    offset: u64,
    size: u64,
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

    fn f32s(&self, e: &DirEntry) -> Vec<f32> {
        assert_eq!(e.dtype, DTYPE_F32, "{} is not f32", e.name);
        self.blob(e)
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
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
    BothD,   // [d, d]: slice rows and columns (MLA g_proj)
    RouterW, // [n_experts, d]: expert rows + hidden columns
    RouterB, // [n_experts]: e_score_correction_bias
    Expert,  // block_sparse_moe.experts.{e}.{w1,w2,w3} (mxfp4, dims untouched)
}

/// Splits "layers.{i}.{rest}" into (layer_index, rest).
fn split_layer(name: &str) -> Option<(usize, &str)> {
    let s = name.strip_prefix("layers.")?;
    let dot = s.find('.')?;
    Some((s[..dot].parse().ok()?, &s[dot + 1..]))
}

fn role_of(name: &str, cfg: &Config) -> Role {
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
fn channel_scores(bin: &BinDir, kept_layers: &[usize], d: usize) -> Vec<f64> {
    let cfg = &bin.config;
    let mut s = vec![0f64; d];
    for e in &bin.entries {
        let in_layers = match split_layer(&e.name) {
            None => true, // global tensor
            Some((l, _)) => kept_layers.contains(&l),
        };
        if !in_layers {
            continue;
        }
        let role = role_of(&e.name, cfg);
        if matches!(role, Role::Copy | Role::RouterB | Role::Expert) {
            continue;
        }
        let w = bin.f32s(e);
        match role {
            Role::VecD => {
                assert_eq!(w.len(), d, "{}: expected [{}]", e.name, d);
                for (j, &x) in w.iter().enumerate() {
                    s[j] += x.abs() as f64;
                }
            }
            Role::ColsD | Role::RouterW => {
                let c = e.dims[1] as usize;
                assert_eq!(c, d, "{}: expected [_, {}]", e.name, d);
                for row in w.chunks_exact(c) {
                    for (j, &x) in row.iter().enumerate() {
                        s[j] += x.abs() as f64;
                    }
                }
            }
            Role::RowsD => {
                let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
                assert_eq!(r, d, "{}: expected [{}, _]", e.name, d);
                for (j, row) in w.chunks_exact(c).enumerate() {
                    s[j] += row.iter().map(|&x| x.abs() as f64).sum::<f64>();
                }
            }
            Role::BothD => {
                let (r, c) = (e.dims[0] as usize, e.dims[1] as usize);
                assert_eq!((r, c), (d, d), "{}: expected [{}, {}]", e.name, d, d);
                // the channel appears as BOTH the output (row) and the input
                // (column) axis: row sums + column sums
                for (j, row) in w.chunks_exact(c).enumerate() {
                    s[j] += row.iter().map(|&x| x.abs() as f64).sum::<f64>();
                }
                for row in w.chunks_exact(c) {
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

/// Per kept MoE layer: the n_experts_keep expert indices to keep (ascending).
/// Layers are scored in parallel threads (shared &File, read_exact_at).
fn expert_keep_sets(bin: &BinDir, kept_layers: &[usize], n_keep: usize) -> std::collections::HashMap<usize, Vec<usize>> {
    let cfg = &bin.config;
    let moe_layers: Vec<usize> = kept_layers.iter().copied().filter(|&l| cfg.is_moe(l)).collect();
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
                        let entry = bin.entry(&name);
                        assert_eq!(entry.dtype, DTYPE_MXFP4, "{} is not mxfp4", name);
                        blobs[i] = bin.blob(entry);
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

fn value_flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

pub fn run(args: &[String]) {
    let t0 = std::time::Instant::now();
    let Some(model) = value_flag(args, "--model") else {
        eprintln!("error: slice requires --model X.bin");
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

    let bin = BinDir::open(&model);
    let cfg = &bin.config;
    let d = cfg.d;
    println!(
        "slice: {} ({} layers, hidden {}, {} experts top-{} + {} shared, vocab {})",
        model, cfg.n_layers, d, cfg.n_experts, cfg.top_k, cfg.n_shared, cfg.vocab
    );

    // ── 1. layer selection ──
    let kept_layers = match &layers_spec {
        Some(s) => parse_layer_spec(s, cfg.n_layers),
        None => (0..cfg.n_layers).collect(),
    };
    let new_layer_of = |old: usize| kept_layers.iter().position(|&l| l == old);
    println!("layers: keeping {}/{} {:?}", kept_layers.len(), cfg.n_layers, kept_layers);

    // ── 2. channel selection (scored on the kept layers only) ──
    let channels: Option<Vec<usize>> = hidden.map(|h| {
        assert!(h > 0 && h <= d, "--hidden must be in 1..={}", d);
        let scores = channel_scores(&bin, &kept_layers, d);
        let keep = top_n(&scores, h);
        println!("hidden: keeping {}/{} channels (top-|w|), score range {:.3} .. {:.3}", h, d,
            keep.iter().map(|&i| scores[i]).fold(f64::INFINITY, f64::min),
            keep.iter().map(|&i| scores[i]).fold(f64::NEG_INFINITY, f64::max));
        keep
    });

    // ── 3. expert selection (per kept MoE layer, scored on dequantized w1+w2+w3) ──
    let expert_sets = experts.map(|n| {
        assert!(n > 0, "--experts must be >= 1");
        let t = std::time::Instant::now();
        let sets = expert_keep_sets(&bin, &kept_layers, n);
        println!("experts: keeping {}/{} per MoE layer (Frobenius of w1+w2+w3), scored in {:.1?}", n, cfg.n_experts, t.elapsed());
        sets
    });

    // ── 4. plan: output tensors in input directory order ──
    let mut plans: Vec<Plan> = Vec::new();
    for e in &bin.entries {
        let role = role_of(&e.name, cfg);
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
    let tokenizer_kv = bin
        .source_json
        .get("tokenizer")
        .and_then(|t| t.as_str())
        .map(|s| format!(", \"tokenizer\": \"{}\"", s))
        .unwrap_or_default();
    let list = |v: &[usize]| v.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", ");
    let config_json = format!(
        "{{\"format\": 2, \"n_layers\": {}, \"hidden\": {}, \"vocab\": {}, \"n_experts\": {}, \"top_k\": {}, \"n_shared\": {}, \
\"kda_heads\": {}, \"kda_dim\": {}, \"kda_conv\": {}, \"kda_fa_rank\": {}, \"gate_lower_bound\": {}, \
\"mla_heads\": {}, \"mla_q_lora\": {}, \"mla_kv_lora\": {}, \"mla_nope\": {}, \"mla_rope\": {}, \"mla_v\": {}, \
\"routed_hidden\": {}, \"moe_inter\": {}, \"shared_inter\": {}, \"dense_inter\": {}, \
\"attn_res_block\": {}, \"first_k_dense\": {}, \"rms_eps\": {}{}, \
\"mla_layers\": [{}], \"dense_layers\": [{}], \
\"specials\": {{\"bos\": {}, \"end_of_msg\": {}}}, \
\"pruning\": {{\"method\": \"weight-magnitude-v1\", \"hidden\": {}, \"experts\": {}, \"layers\": \"{}\"}}}}",
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
    for (p, &off) in plans.iter().zip(&offsets) {
        let src = bin.entry(&p.src_name);
        let blob = if matches!(p.role, Role::Copy | Role::Expert) {
            assert_eq!(blob_size(p.dtype, &p.dims), src.size, "{}: size mismatch on copy", p.src_name);
            bin.blob(src)
        } else {
            let (vals, dims) = slice_f32(src, &bin.f32s(src), p.role, &p.channels, p.experts.as_ref());
            assert_eq!(dims, p.dims, "{}: planned dims mismatch", p.src_name);
            f32_to_bytes(&vals)
        };
        w.write_blob_at(&mut f, off, &blob);
        done += 1;
        if done % 20000 == 0 {
            println!("  {}/{} tensors written ({:.0?})", done, plans.len(), t0.elapsed());
        }
    }
    let in_size = std::fs::metadata(&model).unwrap().len();
    let out_size = std::fs::metadata(&out).unwrap().len();
    println!();
    println!("══ {} : {} tensors ══", out, plans.len());
    println!("  size: {:.2} GB -> {:.2} GB ({:.1}%)", in_size as f64 / 1e9, out_size as f64 / 1e9, out_size as f64 / in_size as f64 * 100.0);
    println!("  config: {} layers (MLA {:?}, dense {:?}), hidden {}, {} experts top-{}", 
        new_n_layers, mla_layers, dense_layers, new_d, new_n_experts, new_top_k);
    println!("  AttnRes: block={} re-applied on the renumbered layers", cfg.attn_res_block);
    println!("  experts: mxfp4 blobs copied verbatim (no requantization)");
    println!("  done in {:.0?}", t0.elapsed());
}
