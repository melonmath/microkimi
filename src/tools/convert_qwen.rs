//! Local Qwen3.5-family text-checkpoint conversion.
//!
//! The converter reads a sharded Hugging Face safetensors directory and
//! writes an MKIM0002 model. It accepts both text decoders of the family:
//! the MoE variant (`qwen3_5_moe_text`: Qwen3.5/3.6-MoE, Qwen3.8-2.4T-A95B)
//! and the dense variant (`qwen3_5_text`: Qwen3.8-27B). Text-spine tensors
//! become f32 under their native logical names, so low-rank adapter targets
//! remain unambiguous. The fused routed-expert banks are split into one
//! gate/down/up matrix per expert and quantized to MXFP4; the dense
//! variant's per-layer MLP matrices are quantized to MXFP4 the same way.
//! Vision and multi-token-prediction tensors are outside the text decoder
//! and are not copied.
//!
//! Peak conversion memory is bounded by one source chunk, one expert
//! matrix, or one dense-MLP row block. Embeddings, the language-model head,
//! and dense MLP matrices are converted in chunks; a fused expert bank is
//! read only over the selected expert's byte range.

use crate::config::QwenConfig;
use crate::json::{self, Json};
use crate::quant::safetensors;
use crate::quant::weights::{BinWriter, DTYPE_F32, DTYPE_MXFP4};
use std::collections::{HashMap, HashSet};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const TOKENIZER_NAME: &str = "qwen.tokenizer.json";
const CHUNK_BYTES: usize = 32 << 20;

#[derive(Clone)]
struct SourceTensor {
    file: PathBuf,
    start: u64,
    elements: usize,
    dtype: String,
    shape: Vec<usize>,
}

struct StSource {
    tensors: HashMap<String, SourceTensor>,
}

fn dtype_bytes(dtype: &str) -> usize {
    match dtype {
        "BF16" | "F16" => 2,
        "F32" => 4,
        _ => panic!(
            "unsupported safetensors dtype {} (expected BF16, F16, or F32)",
            dtype
        ),
    }
}

fn read_header(path: &Path) -> (u64, HashMap<String, safetensors::TensorInfo>) {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("{} unreadable: {}", path.display(), e));
    let file_len = f.metadata().unwrap().len();
    assert!(
        file_len >= 8,
        "{}: truncated safetensors file",
        path.display()
    );
    let mut first = [0u8; 8];
    f.read_exact_at(&mut first, 0).unwrap();
    let hlen = u64::from_le_bytes(first);
    assert!(
        hlen > 0 && hlen <= 1 << 30 && 8 + hlen <= file_len,
        "{}: invalid safetensors header length {}",
        path.display(),
        hlen
    );
    let mut bytes = vec![0u8; hlen as usize];
    f.read_exact_at(&mut bytes, 8).unwrap();
    let parsed = json::parse_complete(&bytes);
    let pairs = object_pairs(&parsed, "safetensors header");
    let mut tensors = HashMap::new();
    let mut ranges = Vec::new();
    for (name, value) in pairs {
        if name == "__metadata__" {
            continue;
        }
        assert!(
            !name.is_empty() && !name.chars().any(char::is_control),
            "{}: invalid tensor name {:?}",
            path.display(),
            name
        );
        let meta = object_pairs(value, &format!("safetensors tensor {:?}", name));
        let fields: HashSet<&str> = meta.iter().map(|(key, _)| key.as_str()).collect();
        assert_eq!(
            fields,
            HashSet::from(["dtype", "shape", "data_offsets"]),
            "{}: tensor {:?} has unsupported metadata fields",
            path.display(),
            name
        );
        let get = |key: &str| {
            meta.iter()
                .find(|(field, _)| field == key)
                .map(|(_, value)| value)
                .unwrap()
        };
        let dtype = get("dtype")
            .as_str()
            .unwrap_or_else(|| panic!("{}: {:?} dtype is not a string", path.display(), name))
            .to_string();
        dtype_bytes(&dtype);
        let shape = get("shape")
            .as_arr()
            .unwrap_or_else(|| panic!("{}: {:?} shape is not an array", path.display(), name))
            .iter()
            .map(|value| json_nonnegative_integer(value, "safetensors shape") as usize)
            .collect::<Vec<_>>();
        let offsets = get("data_offsets").as_arr().unwrap_or_else(|| {
            panic!(
                "{}: {:?} data_offsets is not an array",
                path.display(),
                name
            )
        });
        assert_eq!(
            offsets.len(),
            2,
            "{}: {:?} must have two data offsets",
            path.display(),
            name
        );
        let start = json_nonnegative_integer(&offsets[0], "safetensors data offset");
        let end = json_nonnegative_integer(&offsets[1], "safetensors data offset");
        assert!(
            start <= end,
            "{}: {:?} has reversed data offsets",
            path.display(),
            name
        );
        ranges.push((start, end, name.clone()));
        assert!(
            tensors
                .insert(
                    name.clone(),
                    safetensors::TensorInfo {
                        dtype,
                        shape,
                        offsets: (start, end),
                    },
                )
                .is_none(),
            "{}: duplicate tensor {:?}",
            path.display(),
            name
        );
    }
    ranges.sort_by_key(|range| range.0);
    let mut cursor = 0u64;
    for (start, end, name) in ranges {
        assert_eq!(
            start,
            cursor,
            "{}: tensor {:?} leaves a gap or overlaps another tensor",
            path.display(),
            name
        );
        cursor = end;
    }
    assert_eq!(
        cursor,
        file_len - 8 - hlen,
        "{}: safetensors payload has trailing or missing bytes",
        path.display()
    );
    (hlen, tensors)
}

fn json_nonnegative_integer(value: &Json, where_: &str) -> u64 {
    let number = value
        .as_num()
        .unwrap_or_else(|| panic!("{} must be a number", where_));
    assert!(
        number.is_finite() && number >= 0.0 && number.fract() == 0.0 && number <= u64::MAX as f64,
        "{} must be a non-negative integer",
        where_
    );
    number as u64
}

fn object_pairs<'a>(value: &'a Json, where_: &str) -> &'a [(String, Json)] {
    let Json::Obj(pairs) = value else {
        panic!("{} must be an object", where_);
    };
    let mut seen = HashSet::new();
    for (name, _) in pairs {
        assert!(
            seen.insert(name),
            "{} contains duplicate key {:?}",
            where_,
            name
        );
    }
    pairs
}

impl StSource {
    fn open(path: &str) -> StSource {
        let input = Path::new(path);
        let (root, weight_map): (PathBuf, HashMap<String, String>) = if input.is_file() {
            let root = input.parent().unwrap_or(Path::new(".")).to_path_buf();
            let file = input.file_name().unwrap().to_string_lossy().into_owned();
            let (_, header) = read_header(input);
            (
                root,
                header
                    .keys()
                    .map(|name| (name.clone(), file.clone()))
                    .collect(),
            )
        } else {
            assert!(
                input.is_dir(),
                "{}: source must be a local directory or safetensors file",
                path
            );
            let index = input.join("model.safetensors.index.json");
            if index.exists() {
                let bytes = std::fs::read(&index).unwrap();
                let parsed = json::parse_complete(&bytes);
                let pairs = object_pairs(
                    parsed
                        .get("weight_map")
                        .expect("safetensors index: weight_map missing"),
                    "safetensors index weight_map",
                );
                let mut map = HashMap::with_capacity(pairs.len());
                for (name, value) in pairs {
                    let file = value.as_str().unwrap_or_else(|| {
                        panic!("safetensors index: {:?} shard is not a string", name)
                    });
                    map.insert(name.clone(), file.to_string());
                }
                (input.to_path_buf(), map)
            } else {
                let mut files: Vec<PathBuf> = std::fs::read_dir(input)
                    .unwrap()
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
                    .collect();
                files.sort();
                assert_eq!(
                    files.len(),
                    1,
                    "{}: expected model.safetensors.index.json or one safetensors file",
                    path
                );
                return Self::open(&files[0].to_string_lossy());
            }
        };
        assert!(!weight_map.is_empty(), "{}: empty safetensors source", path);

        let mut headers: HashMap<String, (u64, u64, HashMap<String, safetensors::TensorInfo>)> =
            HashMap::new();
        let mut shard_names: Vec<String> = weight_map.values().cloned().collect();
        shard_names.sort();
        shard_names.dedup();
        for shard in shard_names {
            assert!(
                Path::new(&shard).file_name().and_then(|x| x.to_str()) == Some(shard.as_str()),
                "safetensors index: unsafe shard path {:?}",
                shard
            );
            let file = root.join(&shard);
            let file_len = std::fs::metadata(&file)
                .unwrap_or_else(|e| panic!("{} missing: {}", file.display(), e))
                .len();
            let (hlen, map) = read_header(&file);
            headers.insert(shard, (hlen, file_len, map));
        }

        let mut header_owners = HashMap::<String, String>::new();
        for (shard, (_, _, map)) in &headers {
            for name in map.keys() {
                assert_eq!(
                    weight_map.get(name),
                    Some(shard),
                    "{}: tensor {:?} is missing from the index or assigned to another shard",
                    shard,
                    name
                );
                assert!(
                    header_owners.insert(name.clone(), shard.clone()).is_none(),
                    "tensor {:?} is present in more than one safetensors shard",
                    name
                );
            }
        }

        let mut tensors = HashMap::with_capacity(weight_map.len());
        for (name, shard) in weight_map {
            let (hlen, file_len, map) = &headers[&shard];
            let info = map.get(&name).unwrap_or_else(|| {
                panic!(
                    "{}: tensor {:?} is absent from its indexed shard",
                    shard, name
                )
            });
            let elements = info.shape.iter().product::<usize>();
            let bpe = dtype_bytes(&info.dtype);
            assert_eq!(
                (info.offsets.1 - info.offsets.0) as usize,
                elements * bpe,
                "{}: data range does not match shape {:?} and dtype {}",
                name,
                info.shape,
                info.dtype
            );
            let start = 8 + *hlen + info.offsets.0;
            assert!(
                start + elements as u64 * bpe as u64 <= *file_len,
                "{}: tensor range exceeds {}",
                name,
                shard
            );
            tensors.insert(
                name,
                SourceTensor {
                    file: root.join(&shard),
                    start,
                    elements,
                    dtype: info.dtype.clone(),
                    shape: info.shape.clone(),
                },
            );
        }
        StSource { tensors }
    }

    fn expect(&self, name: &str, shape: &[usize]) {
        let t = self
            .tensors
            .get(name)
            .unwrap_or_else(|| panic!("checkpoint is missing {}", name));
        assert_eq!(t.shape, shape, "{}: unexpected source shape", name);
        dtype_bytes(&t.dtype);
    }

    fn read_f32(&self, name: &str, element: usize, count: usize) -> Vec<f32> {
        use std::os::unix::fs::FileExt;
        let t = &self.tensors[name];
        assert!(
            element <= t.elements && count <= t.elements - element,
            "{}: read out of range",
            name
        );
        let bpe = dtype_bytes(&t.dtype);
        let mut raw = vec![0u8; count * bpe];
        std::fs::File::open(&t.file)
            .unwrap()
            .read_exact_at(&mut raw, t.start + (element * bpe) as u64)
            .unwrap();
        let values = match t.dtype.as_str() {
            "BF16" => safetensors::bf16_slice_to_f32(&raw),
            "F16" => raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            "F32" => raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            _ => unreachable!(),
        };
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{} contains a non-finite value",
            name
        );
        values
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits & 0x03ff;
    let out = match exp {
        0 if frac == 0 => sign,
        0 => {
            let mut f = frac as u32;
            let mut e = -14i32;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((f & 0x3ff) << 13)
        }
        0x1f => sign | 0x7f80_0000 | ((frac as u32) << 13),
        _ => sign | (((exp as u32) + 112) << 23) | ((frac as u32) << 13),
    };
    f32::from_bits(out)
}

#[derive(Clone)]
enum PlanSource {
    F32(String),
    /// A whole source tensor quantized to MXFP4 (dense MLP matrices).
    Packed(String),
    Gate {
        bank: String,
        expert: usize,
        up: bool,
    },
    Down {
        bank: String,
        expert: usize,
    },
}

#[derive(Clone)]
struct PlannedTensor {
    name: String,
    dtype: u8,
    dims: Vec<u32>,
    source: PlanSource,
}

fn add_f32(plan: &mut Vec<PlannedTensor>, name: String, dims: Vec<u32>) {
    plan.push(PlannedTensor {
        source: PlanSource::F32(name.clone()),
        name,
        dtype: DTYPE_F32,
        dims,
    });
}

/// Multi-token-prediction draft head of the dense variant: one trunk-style
/// full-attention decoder layer plus the fc merge and its three norms. The
/// MLP matrices are MXFP4-packed exactly like the trunk MLP.
fn mtp_plan(c: &QwenConfig, out: &mut Vec<PlannedTensor>) {
    let full = c.n_heads * c.head_dim;
    let kv = c.n_kv_heads * c.head_dim;
    add_f32(out, "mtp.fc.weight".to_string(), vec![c.d as u32, 2 * c.d as u32]);
    for name in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
    ] {
        add_f32(out, name.to_string(), vec![c.d as u32]);
    }
    add_f32(
        out,
        "mtp.layers.0.self_attn.q_proj.weight".to_string(),
        vec![(2 * full) as u32, c.d as u32],
    );
    add_f32(
        out,
        "mtp.layers.0.self_attn.k_proj.weight".to_string(),
        vec![kv as u32, c.d as u32],
    );
    add_f32(
        out,
        "mtp.layers.0.self_attn.v_proj.weight".to_string(),
        vec![kv as u32, c.d as u32],
    );
    add_f32(
        out,
        "mtp.layers.0.self_attn.o_proj.weight".to_string(),
        vec![c.d as u32, full as u32],
    );
    add_f32(
        out,
        "mtp.layers.0.self_attn.q_norm.weight".to_string(),
        vec![c.head_dim as u32],
    );
    add_f32(
        out,
        "mtp.layers.0.self_attn.k_norm.weight".to_string(),
        vec![c.head_dim as u32],
    );
    for (suffix, dims) in [
        ("gate_proj", vec![c.dense_inter as u32, c.d as u32]),
        ("up_proj", vec![c.dense_inter as u32, c.d as u32]),
        ("down_proj", vec![c.d as u32, c.dense_inter as u32]),
    ] {
        let name = format!("mtp.layers.0.mlp.{}.weight", suffix);
        out.push(PlannedTensor {
            source: PlanSource::Packed(name.clone()),
            name,
            dtype: DTYPE_MXFP4,
            dims,
        });
    }
}

fn conversion_plan(c: &QwenConfig) -> Vec<PlannedTensor> {
    let mut out = Vec::new();
    if c.mtp_layers > 0 {
        assert!(
            c.is_dense() && c.mtp_layers == 1,
            "MTP conversion supports exactly one draft layer on the dense variant"
        );
        mtp_plan(c, &mut out);
    }
    add_f32(
        &mut out,
        "model.language_model.embed_tokens.weight".to_string(),
        vec![c.vocab as u32, c.d as u32],
    );
    add_f32(
        &mut out,
        "model.language_model.norm.weight".to_string(),
        vec![c.d as u32],
    );
    if !c.tied_embeddings {
        add_f32(
            &mut out,
            "lm_head.weight".to_string(),
            vec![c.vocab as u32, c.d as u32],
        );
    }
    for l in 0..c.n_layers {
        let p = format!("model.language_model.layers.{}", l);
        add_f32(
            &mut out,
            format!("{}.input_layernorm.weight", p),
            vec![c.d as u32],
        );
        add_f32(
            &mut out,
            format!("{}.post_attention_layernorm.weight", p),
            vec![c.d as u32],
        );
        if c.is_full_attn(l) {
            let full = c.n_heads * c.head_dim;
            let kv = c.n_kv_heads * c.head_dim;
            add_f32(
                &mut out,
                format!("{}.self_attn.q_proj.weight", p),
                vec![(2 * full) as u32, c.d as u32],
            );
            add_f32(
                &mut out,
                format!("{}.self_attn.k_proj.weight", p),
                vec![kv as u32, c.d as u32],
            );
            add_f32(
                &mut out,
                format!("{}.self_attn.v_proj.weight", p),
                vec![kv as u32, c.d as u32],
            );
            add_f32(
                &mut out,
                format!("{}.self_attn.o_proj.weight", p),
                vec![c.d as u32, full as u32],
            );
            add_f32(
                &mut out,
                format!("{}.self_attn.q_norm.weight", p),
                vec![c.head_dim as u32],
            );
            add_f32(
                &mut out,
                format!("{}.self_attn.k_norm.weight", p),
                vec![c.head_dim as u32],
            );
        } else {
            let conv = c.lin_key_total() * 2 + c.lin_value_total();
            add_f32(
                &mut out,
                format!("{}.linear_attn.in_proj_qkv.weight", p),
                vec![conv as u32, c.d as u32],
            );
            add_f32(
                &mut out,
                format!("{}.linear_attn.in_proj_z.weight", p),
                vec![c.lin_value_total() as u32, c.d as u32],
            );
            for suffix in ["in_proj_b.weight", "in_proj_a.weight"] {
                add_f32(
                    &mut out,
                    format!("{}.linear_attn.{}", p, suffix),
                    vec![c.lin_v_heads as u32, c.d as u32],
                );
            }
            add_f32(
                &mut out,
                format!("{}.linear_attn.out_proj.weight", p),
                vec![c.d as u32, c.lin_value_total() as u32],
            );
            add_f32(
                &mut out,
                format!("{}.linear_attn.conv1d.weight", p),
                vec![conv as u32, 1, c.conv_kernel as u32],
            );
            for suffix in ["A_log", "dt_bias"] {
                add_f32(
                    &mut out,
                    format!("{}.linear_attn.{}", p, suffix),
                    vec![c.lin_v_heads as u32],
                );
            }
            add_f32(
                &mut out,
                format!("{}.linear_attn.norm.weight", p),
                vec![c.lin_v_dim as u32],
            );
        }
        if c.is_dense() {
            for (suffix, dims) in [
                ("gate_proj", vec![c.dense_inter as u32, c.d as u32]),
                ("up_proj", vec![c.dense_inter as u32, c.d as u32]),
                ("down_proj", vec![c.d as u32, c.dense_inter as u32]),
            ] {
                let name = format!("{}.mlp.{}.weight", p, suffix);
                out.push(PlannedTensor {
                    source: PlanSource::Packed(name.clone()),
                    name,
                    dtype: DTYPE_MXFP4,
                    dims,
                });
            }
            continue;
        }
        add_f32(
            &mut out,
            format!("{}.mlp.gate.weight", p),
            vec![c.n_experts as u32, c.d as u32],
        );
        add_f32(
            &mut out,
            format!("{}.mlp.shared_expert.gate_proj.weight", p),
            vec![c.shared_inter as u32, c.d as u32],
        );
        add_f32(
            &mut out,
            format!("{}.mlp.shared_expert.down_proj.weight", p),
            vec![c.d as u32, c.shared_inter as u32],
        );
        add_f32(
            &mut out,
            format!("{}.mlp.shared_expert.up_proj.weight", p),
            vec![c.shared_inter as u32, c.d as u32],
        );
        add_f32(
            &mut out,
            format!("{}.mlp.shared_expert_gate.weight", p),
            vec![1, c.d as u32],
        );
        let gate_bank = format!("{}.mlp.experts.gate_up_proj", p);
        let down_bank = format!("{}.mlp.experts.down_proj", p);
        for e in 0..c.n_experts {
            let ep = format!("layers.{}.block_sparse_moe.experts.{}", l, e);
            out.push(PlannedTensor {
                name: format!("{}.w1", ep),
                dtype: DTYPE_MXFP4,
                dims: vec![c.moe_inter as u32, c.d as u32],
                source: PlanSource::Gate {
                    bank: gate_bank.clone(),
                    expert: e,
                    up: false,
                },
            });
            out.push(PlannedTensor {
                name: format!("{}.w2", ep),
                dtype: DTYPE_MXFP4,
                dims: vec![c.d as u32, c.moe_inter as u32],
                source: PlanSource::Down {
                    bank: down_bank.clone(),
                    expert: e,
                },
            });
            out.push(PlannedTensor {
                name: format!("{}.w3", ep),
                dtype: DTYPE_MXFP4,
                dims: vec![c.moe_inter as u32, c.d as u32],
                source: PlanSource::Gate {
                    bank: gate_bank.clone(),
                    expert: e,
                    up: true,
                },
            });
        }
    }
    out
}

/// Names the source tensors required by one layer, including its fused banks.
#[cfg(test)]
pub fn layer_tensors(c: &QwenConfig, l: usize) -> Vec<String> {
    let p = format!("model.language_model.layers.{}", l);
    let prefix = format!("{}.", p);
    let mut names: Vec<String> = conversion_plan(c)
        .into_iter()
        .filter_map(|entry| match entry.source {
            PlanSource::F32(name) | PlanSource::Packed(name) if name.starts_with(&prefix) => {
                Some(name)
            }
            PlanSource::Gate { bank, .. } | PlanSource::Down { bank, .. }
                if bank.starts_with(&prefix) =>
            {
                Some(bank)
            }
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Layout summary: every tensor of a converted checkpoint in file order.
/// Used by fixture tests and by the vocabulary slicer, which rewrites a
/// converted file tensor by tensor.
pub fn output_layout(c: &QwenConfig) -> Vec<(String, u8, Vec<u32>)> {
    conversion_plan(c)
        .into_iter()
        .map(|p| (p.name, p.dtype, p.dims))
        .collect()
}

/// Emits the MKIM0002 config block for a converted checkpoint. Dense
/// decoders declare arch "qwen3_5" and `intermediate_size`; MoE decoders
/// declare arch "qwen3_5_moe" and the expert fields.
pub fn config_json(c: &QwenConfig, tokenizer: &str) -> String {
    config_json_with_specials(c, tokenizer, 248044, 248046)
}

/// `config_json` with explicit special ids (a vocabulary-sliced model
/// remaps bos and end_of_msg into its kept rows).
pub fn config_json_with_specials(
    c: &QwenConfig,
    tokenizer: &str,
    bos: u32,
    end_of_msg: u32,
) -> String {
    let arch = if c.is_dense() { "qwen3_5" } else { "qwen3_5_moe" };
    let mlp = if c.is_dense() {
        format!(
            "\"intermediate_size\":{},\"mtp_layers\":{}",
            c.dense_inter, c.mtp_layers
        )
    } else {
        format!(
            "\"num_experts\":{},\"num_experts_per_tok\":{},\"moe_intermediate_size\":{},\
             \"shared_expert_intermediate_size\":{}",
            c.n_experts, c.top_k, c.moe_inter, c.shared_inter
        )
    };
    format!(
        "{{\"format\":2,\"arch\":\"{}\",\"n_layers\":{},\"hidden\":{},\"vocab\":{},\
         \"tokenizer\":\"{}\",\"specials\":{{\"bos\":{},\"end_of_msg\":{}}},\
         \"qwen\":{{\"num_hidden_layers\":{},\"hidden_size\":{},\"vocab_size\":{},\
         \"num_attention_heads\":{},\"num_key_value_heads\":{},\"head_dim\":{},\
         \"partial_rotary_factor\":{},\"rope_theta\":{},\"linear_num_key_heads\":{},\
         \"linear_num_value_heads\":{},\"linear_key_head_dim\":{},\"linear_value_head_dim\":{},\
         \"linear_conv_kernel_dim\":{},\"full_attention_interval\":{},{},\
         \"tie_word_embeddings\":{},\"rms_norm_eps\":{}}}}}",
        arch,
        c.n_layers,
        c.d,
        c.vocab,
        tokenizer,
        bos,
        end_of_msg,
        c.n_layers,
        c.d,
        c.vocab,
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        c.partial_rotary,
        c.rope_theta,
        c.lin_k_heads,
        c.lin_v_heads,
        c.lin_k_dim,
        c.lin_v_dim,
        c.conv_kernel,
        c.full_attn_interval,
        mlp,
        c.tied_embeddings as u8,
        c.norm_eps
    )
}

fn json_bool(value: Option<&Json>, key: &str) -> bool {
    match value.unwrap_or_else(|| panic!("config.json: {} missing", key)) {
        Json::Bool(v) => *v,
        _ => panic!("config.json: {} must be boolean", key),
    }
}

/// Reads and validates the checkpoint's text configuration.
pub fn read_hf_config(dir: &str) -> QwenConfig {
    let root = if Path::new(dir).is_dir() {
        Path::new(dir)
    } else {
        Path::new(dir).parent().unwrap_or(Path::new("."))
    };
    let path = root.join("config.json");
    let raw =
        std::fs::read(&path).unwrap_or_else(|e| panic!("{} unreadable: {}", path.display(), e));
    let j = json::parse_complete(&raw);
    let t = j.get("text_config").unwrap_or(&j);
    let model_type = t.get("model_type").and_then(|x| x.as_str());
    assert!(
        matches!(model_type, Some("qwen3_5_moe_text") | Some("qwen3_5_text")),
        "config.json: expected qwen3_5_moe_text or qwen3_5_text, found {:?}",
        model_type
    );
    assert_eq!(t.get("hidden_act").and_then(|x| x.as_str()), Some("silu"));
    assert!(!json_bool(t.get("attention_bias"), "attention_bias"));
    let tied = json_bool(t.get("tie_word_embeddings"), "tie_word_embeddings");
    assert_eq!(
        t.get("attention_dropout").and_then(|x| x.as_num()),
        Some(0.0),
        "attention dropout is unsupported"
    );
    // Qwen3.8 names the Gated DeltaNet output gate's activation; the
    // runtime applies SiLU (swish) there, so only that spelling passes.
    if let Some(gate) = t.get("output_gate_type").and_then(|x| x.as_str()) {
        assert!(
            matches!(gate, "swish" | "silu"),
            "config.json: output_gate_type {:?} is unsupported (swish/silu only)",
            gate
        );
    }
    let mut c = QwenConfig::from_json(t);
    c.tied_embeddings = tied;
    assert_eq!(
        c.is_dense(),
        model_type == Some("qwen3_5_text"),
        "config.json: model_type and intermediate_size disagree on density"
    );
    let types = t
        .get("layer_types")
        .and_then(|x| x.as_arr())
        .expect("config.json: layer_types missing");
    assert_eq!(
        types.len(),
        c.n_layers,
        "config.json: layer_types length mismatch"
    );
    for (l, value) in types.iter().enumerate() {
        let expected = if c.is_full_attn(l) {
            "full_attention"
        } else {
            "linear_attention"
        };
        assert_eq!(
            value.as_str(),
            Some(expected),
            "config.json: unsupported layer type at {}",
            l
        );
    }
    let rope = t
        .get("rope_parameters")
        .expect("config.json: rope_parameters missing");
    assert_eq!(
        rope.get("rope_type").and_then(|x| x.as_str()),
        Some("default")
    );
    // Multimodal checkpoints declare interleaved mrope, which degenerates
    // to standard rope on the text path; text-only checkpoints (e.g.
    // Qwen3.8-2.4T-A95B) omit the key entirely.
    if rope.get("mrope_interleaved").is_some() {
        assert!(json_bool(
            rope.get("mrope_interleaved"),
            "rope_parameters.mrope_interleaved"
        ));
    }
    let packed_inter = if c.is_dense() { c.dense_inter } else { c.moe_inter };
    assert!(
        c.d % 32 == 0 && packed_inter % 32 == 0,
        "MXFP4 dimensions must be multiples of 32"
    );
    c
}

/// Writes a synthetic checkpoint of configuration `c` at `path`: the
/// converter's own output layout, deterministic patterned weights (norms
/// neutral, the causal conv an identity tap), MXFP4 MLP blocks quantized
/// from the pattern. Shape checks (batched vs sequential prefill, GPU vs
/// CPU decode) run on it; it says nothing about language.
pub fn write_synthetic_checkpoint(c: &QwenConfig, path: &str) {
    use crate::quant::weights::DTYPE_MXFP4;
    let layout = crate::tools::convert_qwen::output_layout(c);
    let mut writer = crate::quant::weights::BinWriter::new();
    for (name, dtype, dims) in &layout {
        writer.add(name, *dtype, dims.clone());
    }
    let mut file = std::fs::File::create(&path).unwrap();
    let offsets = writer.write_header_v2(
        &mut file,
        &crate::tools::convert_qwen::config_json(c, "qwen.tokenizer.json"),
    );
    for ((name, dtype, dims), offset) in layout.iter().zip(offsets) {
        let n: usize = dims.iter().map(|&d| d as usize).product();
        let blob = if *dtype == DTYPE_MXFP4 {
            let values: Vec<f32> = (0..n).map(|i| ((i % 9) as f32 - 4.0) * 0.004).collect();
            let (packed, scales) =
                crate::quant::mxfp4::quantize(&values, dims[0] as usize, dims[1] as usize);
            [packed, scales].concat()
        } else {
            let mut values = if name.contains(".linear_attn.norm.weight") {
                vec![1.0f32; n]
            } else if name.ends_with("norm.weight")
                || name.ends_with("layernorm.weight")
                || name.ends_with("A_log")
                || name.ends_with("dt_bias")
                || name.ends_with("shared_expert_gate.weight")
            {
                vec![0.0f32; n]
            } else {
                (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.002).collect()
            };
            if name.ends_with("linear_attn.conv1d.weight") {
                values.fill(0.0);
                for channel in 0..(n / c.conv_kernel) {
                    values[channel * c.conv_kernel + c.conv_kernel - 1] = 1.0;
                }
            }
            crate::quant::weights::f32_to_bytes(&values)
        };
        writer.write_blob_at(&mut file, offset, &blob);
    }
    file.sync_all().unwrap();
}

/// `microkimi qwen-fixture --out X.bin [--profile 27b|0.8b] [--layers N]
/// [--scale S]`: a synthetic checkpoint carrying a real model's shape
/// signature at toy size. `27b` is Qwen3.8-27B's: three value heads per key
/// head, six query heads per kv head, a quarter rotary, untied embeddings,
/// dense MLP; `0.8b` is Qwen3.5-0.8B's (one-to-one heads, tied). `--scale`
/// multiplies the toy widths (1 = d 32, head dim 32; 4 = d 128, head dim
/// 128), `--layers` the depth (default 4: three linear, one full).
pub fn fixture_cmd(args: &[String]) {
    let value = |flag: &str| args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned();
    let out = value("--out").unwrap_or_else(|| {
        eprintln!("error: qwen-fixture requires --out X.bin");
        std::process::exit(2);
    });
    let profile = value("--profile").unwrap_or_else(|| "27b".to_string());
    let scale: usize = value("--scale").and_then(|v| v.parse().ok()).unwrap_or(1).max(1);
    let layers: usize = value("--layers").and_then(|v| v.parse().ok()).unwrap_or(4).max(1);
    let mut c = QwenConfig::qwen35_moe();
    c.n_experts = 0;
    c.top_k = 0;
    c.moe_inter = 0;
    c.shared_inter = 0;
    c.n_layers = layers;
    c.vocab = 64;
    match profile.as_str() {
        "27b" => {
            c.d = 32 * scale;
            c.n_heads = 6;
            c.n_kv_heads = 1;
            c.head_dim = 32 * scale;
            c.partial_rotary = 0.25;
            c.lin_k_heads = 2;
            c.lin_v_heads = 6;
            c.lin_k_dim = 16 * scale;
            c.lin_v_dim = 16 * scale;
            c.dense_inter = 64 * scale;
            c.tied_embeddings = false;
        }
        "0.8b" => {
            c.d = 32 * scale;
            c.n_heads = 2;
            c.n_kv_heads = 1;
            c.head_dim = 32 * scale;
            c.partial_rotary = 0.25;
            c.lin_k_heads = 2;
            c.lin_v_heads = 2;
            c.lin_k_dim = 16 * scale;
            c.lin_v_dim = 16 * scale;
            c.dense_inter = 64 * scale;
            c.tied_embeddings = true;
        }
        other => {
            eprintln!("error: unknown profile {:?} (27b or 0.8b)", other);
            std::process::exit(2);
        }
    }
    write_synthetic_checkpoint(&c, &out);
    println!(
        "qwen-fixture: {} ({} profile) - {} layers, d {}, heads {}/{} (hd {}), linear {}v/{}k (kd {}, vd {}), rotary {}, {} embeddings",
        out,
        profile,
        c.n_layers,
        c.d,
        c.n_heads,
        c.n_kv_heads,
        c.head_dim,
        c.lin_v_heads,
        c.lin_k_heads,
        c.lin_k_dim,
        c.lin_v_dim,
        c.partial_rotary,
        if c.tied_embeddings { "tied" } else { "untied" }
    );
}

fn expected_sources(c: &QwenConfig) -> HashMap<String, Vec<usize>> {
    let mut expected = HashMap::new();
    for item in conversion_plan(c) {
        match item.source {
            PlanSource::F32(name) | PlanSource::Packed(name) => {
                expected.insert(name, item.dims.iter().map(|&d| d as usize).collect());
            }
            PlanSource::Gate { bank, .. } => {
                expected.insert(bank, vec![c.n_experts, 2 * c.moe_inter, c.d]);
            }
            PlanSource::Down { bank, .. } => {
                expected.insert(bank, vec![c.n_experts, c.d, c.moe_inter]);
            }
        }
    }
    expected
}

fn write_f32_tensor(
    source: &StSource,
    source_name: &str,
    elements: usize,
    file: &mut std::fs::File,
    offset: u64,
) {
    file.seek(SeekFrom::Start(offset)).unwrap();
    let bpe = dtype_bytes(&source.tensors[source_name].dtype);
    let per = (CHUNK_BYTES / bpe).max(1);
    let mut done = 0usize;
    while done < elements {
        let n = (elements - done).min(per);
        let values = source.read_f32(source_name, done, n);
        file.write_all(&crate::quant::weights::f32_to_bytes(&values))
            .unwrap();
        done += n;
    }
}

fn quantize_parallel(
    values: &[f32],
    rows: usize,
    cols: usize,
    imp: Option<&[f32]>,
) -> (Vec<u8>, Vec<u8>) {
    fn quantize_rows(
        values: &[f32],
        rows: usize,
        cols: usize,
        imp: Option<&[f32]>,
    ) -> (Vec<u8>, Vec<u8>) {
        match imp {
            Some(weights) => crate::quant::mxfp4::quantize_weighted(values, rows, cols, weights),
            None => crate::quant::mxfp4::quantize(values, rows, cols),
        }
    }
    let workers = crate::model::pool::pool().workers;
    let jobs_count = workers.min(rows).min((rows * cols).div_ceil(16_384)).max(1);
    if jobs_count == 1 {
        return quantize_rows(values, rows, cols, imp);
    }
    let rows_per_job = rows.div_ceil(jobs_count);
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * cols / 32];
    let vp = crate::model::pool::SPtr(values.as_ptr());
    let pp = crate::model::pool::MPtrU8(packed.as_mut_ptr());
    let sp = crate::model::pool::MPtrU8(scales.as_mut_ptr());
    let ip = imp.map(|w| crate::model::pool::SPtr(w.as_ptr()));
    let mut jobs: Vec<crate::model::pool::Job> = Vec::with_capacity(jobs_count);
    for job in 0..jobs_count {
        let row0 = job * rows_per_job;
        let row1 = ((job + 1) * rows_per_job).min(rows);
        if row0 >= row1 {
            break;
        }
        jobs.push(Box::new(move || {
            let (vp, pp, sp, ip) = (vp, pp, sp, ip);
            unsafe {
                let input = std::slice::from_raw_parts(vp.0.add(row0 * cols), (row1 - row0) * cols);
                let imp = ip.map(|w| std::slice::from_raw_parts(w.0, cols));
                let (local_packed, local_scales) = quantize_rows(input, row1 - row0, cols, imp);
                std::ptr::copy_nonoverlapping(
                    local_packed.as_ptr(),
                    pp.0.add(row0 * cols / 2),
                    local_packed.len(),
                );
                std::ptr::copy_nonoverlapping(
                    local_scales.as_ptr(),
                    sp.0.add(row0 * cols / 32),
                    local_scales.len(),
                );
            }
        }));
    }
    crate::model::pool::pool().run(jobs);
    (packed, scales)
}

/// Quantizes one whole source matrix to MXFP4 in bounded row blocks. The
/// output blob is all packed nibbles then all scales, so each block lands
/// at two computed offsets.
#[allow(clippy::too_many_arguments)]
fn write_packed_streamed(
    source: &StSource,
    source_name: &str,
    rows: usize,
    cols: usize,
    file: &mut std::fs::File,
    offset: u64,
    imp: Option<&[f32]>,
) {
    let rows_per_block = (CHUNK_BYTES / (cols * 4)).max(1);
    write_packed_blocks(source, source_name, rows, cols, file, offset, rows_per_block, imp);
}

#[allow(clippy::too_many_arguments)]
fn write_packed_blocks(
    source: &StSource,
    source_name: &str,
    rows: usize,
    cols: usize,
    file: &mut std::fs::File,
    offset: u64,
    rows_per_block: usize,
    imp: Option<&[f32]>,
) {
    let scales_base = offset + (rows * cols / 2) as u64;
    let mut row = 0usize;
    while row < rows {
        let n = (rows - row).min(rows_per_block);
        let values = source.read_f32(source_name, row * cols, n * cols);
        let (packed, scales) = quantize_parallel(&values, n, cols, imp);
        file.seek(SeekFrom::Start(offset + (row * cols / 2) as u64))
            .unwrap();
        file.write_all(&packed).unwrap();
        file.seek(SeekFrom::Start(scales_base + (row * cols / 32) as u64))
            .unwrap();
        file.write_all(&scales).unwrap();
        row += n;
    }
}

fn write_expert(
    source: &StSource,
    item: &PlannedTensor,
    c: &QwenConfig,
    file: &mut std::fs::File,
    offset: u64,
) {
    let (values, rows, cols) = match &item.source {
        PlanSource::Gate { bank, expert, up } => {
            let rows = c.moe_inter;
            let first_row = expert * 2 * rows + if *up { rows } else { 0 };
            (
                source.read_f32(bank, first_row * c.d, rows * c.d),
                rows,
                c.d,
            )
        }
        PlanSource::Down { bank, expert } => (
            source.read_f32(bank, expert * c.d * c.moe_inter, c.d * c.moe_inter),
            c.d,
            c.moe_inter,
        ),
        PlanSource::F32(_) | PlanSource::Packed(_) => unreachable!(),
    };
    let (packed, scales) = quantize_parallel(&values, rows, cols, None);
    file.seek(SeekFrom::Start(offset)).unwrap();
    file.write_all(&packed).unwrap();
    file.write_all(&scales).unwrap();
}

/// Column weights for one packed dense-MLP tensor from a calibration
/// imatrix: gate/up read the hidden stream (hidden sums), down reads the
/// SiLU-gated activation (inter sums). MTP tensors stay unweighted (their
/// activations are not part of the calibration pass).
fn imatrix_col_weights(im: &crate::quant::imatrix::Imatrix, name: &str) -> Option<Vec<f32>> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let (layer, kind) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    match kind {
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => im.col_weights(layer, "w1"),
        "mlp.down_proj.weight" => im.col_weights(layer, "w2"),
        _ => None,
    }
}

/// `microkimi convert-qwen --source DIR --out MODEL.bin [--imatrix F]`.
pub fn run(args: &[String]) {
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let source_path = value("--source").unwrap_or_else(|| {
        eprintln!("error: convert-qwen requires --source CHECKPOINT_DIR");
        std::process::exit(2);
    });
    let out_path = value("--out").unwrap_or_else(|| {
        eprintln!("error: convert-qwen requires --out MODEL.bin");
        std::process::exit(2);
    });
    assert!(
        !Path::new(&out_path).exists(),
        "{} already exists",
        out_path
    );
    let mut c = read_hf_config(&source_path);
    let source = StSource::open(&source_path);
    if c.is_dense() && source.tensors.contains_key("mtp.fc.weight") {
        c.mtp_layers = 1;
        println!("multi-token-prediction head found: converting the draft layer");
    }
    let imatrix = value("--imatrix").map(|path| {
        let im = crate::quant::imatrix::load(&path)
            .unwrap_or_else(|e| panic!("--imatrix: {}", e));
        assert!(
            c.is_dense(),
            "--imatrix weighting is supported for the dense variant only"
        );
        assert!(
            im.n_layers == c.n_layers
                && im.routed_hidden == c.d
                && im.moe_inter == c.dense_inter
                && im.tokens > 0,
            "--imatrix: calibration dimensions do not match this checkpoint"
        );
        println!(
            "importance matrix: {} ({} calibration tokens)",
            path, im.tokens
        );
        im
    });
    let expected = expected_sources(&c);
    for (name, shape) in &expected {
        source.expect(name, shape);
    }
    let unexpected: Vec<&String> = source
        .tensors
        .keys()
        .filter(|name| {
            (**name == "lm_head.weight"
                || name.starts_with("model.language_model.")
                || (c.mtp_layers > 0 && name.starts_with("mtp.")))
                && !expected.contains_key(*name)
        })
        .collect();
    assert!(
        unexpected.is_empty(),
        "unsupported text-decoder tensors in checkpoint, e.g. {:?}",
        &unexpected[..unexpected.len().min(8)]
    );
    if c.is_dense() {
        println!(
            "Qwen text checkpoint: {} layers, hidden {}, dense MLP {}",
            c.n_layers, c.d, c.dense_inter
        );
    } else {
        println!(
            "Qwen text checkpoint: {} layers, hidden {}, {} experts top-{}",
            c.n_layers, c.d, c.n_experts, c.top_k
        );
    }
    println!(
        "source audit: {} required tensors, all shapes and dtypes accepted",
        expected.len()
    );

    let plan = conversion_plan(&c);
    if args.iter().any(|arg| arg == "--audit-only") {
        let payload: u64 = plan
            .iter()
            .map(|item| crate::quant::weights::blob_size(item.dtype, &item.dims))
            .sum();
        println!(
            "conversion plan: {} output tensors, {:.2} GB payload (audit only)",
            plan.len(),
            payload as f64 / 1e9
        );
        return;
    }
    let mut writer = BinWriter::new();
    for item in &plan {
        writer.add(&item.name, item.dtype, item.dims.clone());
    }
    let partial = format!("{}.partial.{}", out_path, std::process::id());
    let mut file = std::fs::File::create(&partial)
        .unwrap_or_else(|e| panic!("cannot create {}: {}", partial, e));
    let offsets = writer.write_header_v2(&mut file, &config_json(&c, TOKENIZER_NAME));
    let mut last_layer = None;
    for (item, &offset) in plan.iter().zip(&offsets) {
        if let Some(layer) = item
            .name
            .split("layers.")
            .nth(1)
            .and_then(|s| s.split('.').next())
            .and_then(|s| s.parse::<usize>().ok())
        {
            if last_layer != Some(layer) {
                println!("  layer {}/{}", layer + 1, c.n_layers);
                last_layer = Some(layer);
            }
        }
        match &item.source {
            PlanSource::F32(source_name) => {
                let elements = item.dims.iter().map(|&d| d as usize).product();
                write_f32_tensor(&source, source_name, elements, &mut file, offset);
            }
            PlanSource::Packed(source_name) => {
                let imp = imatrix
                    .as_ref()
                    .and_then(|im| imatrix_col_weights(im, source_name));
                write_packed_streamed(
                    &source,
                    source_name,
                    item.dims[0] as usize,
                    item.dims[1] as usize,
                    &mut file,
                    offset,
                    imp.as_deref(),
                );
            }
            PlanSource::Gate { .. } | PlanSource::Down { .. } => {
                write_expert(&source, item, &c, &mut file, offset);
            }
        }
    }
    file.sync_all().unwrap();
    drop(file);
    std::fs::rename(&partial, &out_path).unwrap();

    let source_root = if Path::new(&source_path).is_dir() {
        Path::new(&source_path)
    } else {
        Path::new(&source_path).parent().unwrap_or(Path::new("."))
    };
    let tokenizer_source = source_root.join("tokenizer.json");
    if tokenizer_source.exists() {
        let tokenizer_out = Path::new(&out_path)
            .parent()
            .unwrap_or(Path::new("."))
            .join(TOKENIZER_NAME);
        if tokenizer_out.exists() {
            assert_eq!(
                std::fs::read(&tokenizer_out).unwrap(),
                std::fs::read(&tokenizer_source).unwrap(),
                "{} already exists with different contents",
                tokenizer_out.display()
            );
        } else {
            std::fs::copy(&tokenizer_source, &tokenizer_out).unwrap();
        }
        println!("tokenizer: {}", tokenizer_out.display());
    } else {
        println!("warning: tokenizer.json not found; pass its path with --vocab at inference time");
    }
    let size = std::fs::metadata(&out_path).unwrap().len();
    println!(
        "converted: {} ({:.2} GB, {} tensors)",
        out_path,
        size as f64 / 1e9,
        plan.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safetensors_file(name: &str, header: &str, payload: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "microkimi_qwen_safetensors_{}_{}",
            std::process::id(),
            name
        ));
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(payload);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn strict_safetensors_header_accepts_a_canonical_payload() {
        let path = safetensors_file(
            "canonical.st",
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            &[0; 8],
        );
        let (_, tensors) = read_header(&path);
        assert_eq!(tensors["x"].shape, vec![2]);
        assert_eq!(tensors["x"].offsets, (0, 8));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn strict_safetensors_header_rejects_a_payload_gap() {
        let path = safetensors_file(
            "gap.st",
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[1,9]}}"#,
            &[0; 9],
        );
        let result = std::panic::catch_unwind(|| read_header(&path));
        std::fs::remove_file(path).ok();
        assert!(result.is_err());
    }

    #[test]
    fn layer_tensor_sets_follow_the_attention_kind() {
        let c = QwenConfig::qwen35_moe();
        let full = layer_tensors(&c, 3);
        assert!(full.iter().any(|n| n.ends_with("self_attn.q_proj.weight")));
        assert!(!full.iter().any(|n| n.contains("linear_attn")));
        let linear = layer_tensors(&c, 0);
        assert!(linear.iter().any(|n| n.ends_with("linear_attn.A_log")));
        assert!(!linear.iter().any(|n| n.contains("self_attn")));
        for names in [&full, &linear] {
            assert!(names.iter().any(|n| n.ends_with("experts.gate_up_proj")));
            assert!(names
                .iter()
                .any(|n| n.ends_with("shared_expert.down_proj.weight")));
        }
    }

    #[test]
    fn output_layout_splits_every_routed_expert() {
        let mut c = QwenConfig::qwen35_moe();
        c.n_layers = 4;
        c.n_experts = 3;
        let layout = output_layout(&c);
        let experts: Vec<_> = layout
            .iter()
            .filter(|(name, _, _)| name.contains(".block_sparse_moe.experts."))
            .collect();
        assert_eq!(experts.len(), c.n_layers * c.n_experts * 3);
        assert!(experts.iter().all(|(_, dtype, _)| *dtype == DTYPE_MXFP4));
        assert!(layout.iter().any(|(name, dtype, dims)| {
            name == "model.language_model.layers.3.self_attn.q_proj.weight"
                && *dtype == DTYPE_F32
                && dims == &vec![(2 * c.n_heads * c.head_dim) as u32, c.d as u32]
        }));
    }

    #[test]
    fn config_block_round_trips() {
        let mut c = QwenConfig::qwen35_moe();
        c.n_layers = 12;
        c.n_experts = 64;
        c.top_k = 4;
        let s = config_json(&c, TOKENIZER_NAME);
        let j = json::parse_complete(s.as_bytes());
        assert_eq!(j.get("arch").and_then(|x| x.as_str()), Some("qwen3_5_moe"));
        let back = QwenConfig::from_json(j.get("qwen").unwrap());
        assert_eq!(back.n_layers, 12);
        assert_eq!(back.n_experts, 64);
        assert_eq!(back.top_k, 4);
        assert_eq!(back.full_attn_interval, c.full_attn_interval);
    }

    #[test]
    fn dense_layout_packs_the_mlp_and_skips_expert_tensors() {
        let mut c = QwenConfig::qwen38_dense();
        c.n_layers = 4;
        let layout = output_layout(&c);
        assert!(!layout
            .iter()
            .any(|(name, _, _)| name.contains("experts") || name.contains("shared_expert")));
        let mlp: Vec<_> = layout
            .iter()
            .filter(|(name, _, _)| name.contains(".mlp."))
            .collect();
        assert_eq!(mlp.len(), c.n_layers * 3);
        assert!(mlp.iter().all(|(_, dtype, _)| *dtype == DTYPE_MXFP4));
        assert!(layout.iter().any(|(name, dtype, dims)| {
            name == "model.language_model.layers.0.mlp.down_proj.weight"
                && *dtype == DTYPE_MXFP4
                && dims == &vec![c.d as u32, c.dense_inter as u32]
        }));
        // attention spine stays f32
        assert!(layout.iter().any(|(name, dtype, _)| {
            name == "model.language_model.layers.3.self_attn.q_proj.weight" && *dtype == DTYPE_F32
        }));
    }

    #[test]
    fn dense_config_block_round_trips() {
        let mut c = QwenConfig::qwen38_dense();
        c.n_layers = 8;
        let s = config_json(&c, TOKENIZER_NAME);
        let j = json::parse_complete(s.as_bytes());
        assert_eq!(j.get("arch").and_then(|x| x.as_str()), Some("qwen3_5"));
        let back = QwenConfig::from_json(j.get("qwen").unwrap());
        assert!(back.is_dense());
        assert_eq!(back.n_layers, 8);
        assert_eq!(back.dense_inter, c.dense_inter);
        assert_eq!(back.n_experts, 0);
        assert_eq!(back.n_heads, 24);
        assert_eq!(back.lin_v_heads, 48);
        let full = crate::config::Config::from_json(&j);
        assert!(full.qwen.is_some_and(|q| q.is_dense()));
    }

    #[test]
    fn half_conversion_handles_normals_and_subnormals() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001).to_bits(), 0x3380_0000);
    }

    #[test]
    fn dense_imatrix_roundtrip_yields_column_weights() {
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
        let bin_path = crate::model::qwen::test_fixture(&c);
        let mut model = crate::model::qwen::QwenModel::load(&bin_path);
        crate::quant::imatrix::start_dims(
            c.n_layers,
            c.d,
            c.dense_inter,
            (0..c.n_layers as u32).collect(),
        );
        for &t in &[3u32, 5, 7] {
            model.forward(t);
            crate::quant::imatrix::tick();
        }
        let im_path = std::env::temp_dir()
            .join(format!("microkimi_qwen_imatrix_{}.bin", std::process::id()))
            .to_string_lossy()
            .into_owned();
        crate::quant::imatrix::stop_and_save(&im_path).unwrap();
        let im = crate::quant::imatrix::load(&im_path).unwrap();
        assert_eq!(im.tokens, 3);
        assert_eq!((im.routed_hidden, im.moe_inter), (c.d, c.dense_inter));

        let gate = imatrix_col_weights(&im, "model.language_model.layers.0.mlp.gate_proj.weight")
            .expect("gate weights");
        assert_eq!(gate.len(), c.d);
        assert!(gate.iter().any(|&w| w > 0.0));
        let down = imatrix_col_weights(&im, "model.language_model.layers.2.mlp.down_proj.weight")
            .expect("down weights");
        assert_eq!(down.len(), c.dense_inter);
        assert!(imatrix_col_weights(&im, "mtp.layers.0.mlp.gate_proj.weight").is_none());
        assert!(
            imatrix_col_weights(&im, "model.language_model.layers.0.self_attn.q_proj.weight")
                .is_none()
        );

        drop(model);
        std::fs::remove_file(bin_path).ok();
        std::fs::remove_file(im_path).ok();
    }

    #[test]
    fn streamed_packed_write_matches_one_shot_quantization() {
        let (rows, cols) = (7usize, 64usize);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 29 + 3) % 191) as f32 - 95.0) * 0.001)
            .collect();
        let payload = crate::quant::weights::f32_to_bytes(&values);
        let header = format!(
            r#"{{"m":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,{}]}}}}"#,
            rows,
            cols,
            payload.len()
        );
        let st_path = safetensors_file("streamed.st", &header, &payload);
        let source = StSource::open(&st_path.to_string_lossy());
        let out_path = std::env::temp_dir().join(format!(
            "microkimi_qwen_streamed_out_{}",
            std::process::id()
        ));
        let mut file = std::fs::File::create(&out_path).unwrap();
        // three rows per block: exercises full blocks plus a short tail
        write_packed_blocks(&source, "m", rows, cols, &mut file, 0, 3, None);
        file.sync_all().unwrap();
        let written = std::fs::read(&out_path).unwrap();
        let (packed, scales) = crate::quant::mxfp4::quantize(&values, rows, cols);
        assert_eq!(written, [packed, scales].concat());
        std::fs::remove_file(st_path).ok();
        std::fs::remove_file(out_path).ok();
    }

    #[test]
    fn parallel_quantization_matches_serial_bytes() {
        let (rows, cols) = (64usize, 512usize);
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| (((i * 37 + 11) % 257) as f32 - 128.0) * 0.0007)
            .collect();
        let serial = crate::quant::mxfp4::quantize(&values, rows, cols);
        let parallel = quantize_parallel(&values, rows, cols, None);
        assert_eq!(parallel, serial);
    }
}
