// `microkimi build`: builds microkimi-debug.bin.
// Sources: small REAL K3 tensors via HTTP range requests (index → shard
// header → offsets → bytes), 3 real MXFP4 experts → value pools,
// Qwen2.5-0.5B (local HF cache) for embedding/lm_head + projection pools,
// reproducible xorshift64* generation (seed per tensor), MXFP4 quantization.
// Graceful fallback: if the network fails → Qwen + Gaussian σ=0.02.

use crate::stream::http;
use crate::quant::mxfp4;
use crate::quant::safetensors;
use crate::quant::weights::{self, BinWriter, DTYPE_F32, DTYPE_MXFP4};
use std::collections::HashMap;

const K3_BASE: &str = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/";
const OUT: &str = "microkimi-debug.bin";
const SEED_BASE: u64 = 0x4B49_4D49_5EED_0001; // "KIMI"^seed - reproducible
const SIGMA: f32 = 0.02;

// ── xorshift64* RNG + Box-Muller, seed derived from the tensor name ──

pub struct Rng(u64);

impl Rng {
    pub fn for_tensor(name: &str) -> Rng {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Rng((h ^ SEED_BASE) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn gauss(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
    pub fn pick(&mut self, pool: &[f32]) -> f32 {
        pool[(self.next_u64() % pool.len() as u64) as usize]
    }
}

// ── fetch K3: index + shard headers + tensors ──

struct K3 {
    weight_map: HashMap<String, String>, // tensor → shard
    headers: HashMap<String, (u64, HashMap<String, safetensors::TensorInfo>)>,
}

impl K3 {
    fn open() -> Option<K3> {
        let idx = http::fetch(&format!("{}model.safetensors.index.json", K3_BASE))?;
        let parsed = crate::json::parse(&idx);
        let wm = parsed.get("weight_map")?;
        let mut weight_map = HashMap::new();
        if let crate::json::Json::Obj(pairs) = wm {
            for (k, v) in pairs {
                if let Some(s) = v.as_str() {
                    weight_map.insert(k.clone(), s.to_string());
                }
            }
        }
        println!("  K3 index: {} tensors referenced", weight_map.len());
        Some(K3 { weight_map, headers: HashMap::new() })
    }

    fn shard_header(&mut self, shard: &str) -> Option<&HashMap<String, safetensors::TensorInfo>> {
        if !self.headers.contains_key(shard) {
            let url = format!("{}{}", K3_BASE, shard);
            let first = http::fetch_range(&url, Some((0, 7)))?;
            let hlen = u64::from_le_bytes(first[0..8].try_into().unwrap());
            let head = http::fetch_range(&url, Some((8, 8 + hlen - 1)))?;
            let full = [first, head].concat();
            let (hlen, map) = safetensors::parse_header(&full);
            self.headers.insert(shard.to_string(), (hlen, map));
        }
        self.headers.get(shard).map(|(_, m)| m)
    }

    /// Downloads the raw bytes of a tensor (dtype preserved).
    fn tensor_bytes(&mut self, name: &str) -> Option<(Vec<u8>, safetensors::TensorInfo)> {
        let shard = self.weight_map.get(name)?.clone();
        let (hlen, info) = {
            let map = self.shard_header(&shard)?;
            let info = map.get(name)?.clone();
            (self.headers[&shard].0, info)
        };
        let start = 8 + hlen + info.offsets.0;
        let end = 8 + hlen + info.offsets.1 - 1;
        let bytes = http::fetch_range(&format!("{}{}", K3_BASE, shard), Some((start, end)))?;
        Some((bytes, info))
    }
}

// ── source collection ──

struct Sources {
    real: HashMap<String, Vec<f32>>, // real values by microkimi logical name
    pool_w1: Vec<f32>,
    pool_w2: Vec<f32>,
    pool_w3: Vec<f32>,
    qwen_attn: Vec<f32>,
    qwen_mlp: Vec<f32>,
    qwen_embed: Vec<(u32, Vec<f32>)>, // reserved (embed handled separately)
    report: Vec<String>,
}

fn note(report: &mut Vec<String>, what: &str, src: &str) {
    report.push(format!("    {:<58} {}", what, src));
}

fn bytes_to_f32(bytes: &[u8], dtype: &str, context: &str) -> Vec<f32> {
    match dtype {
        "BF16" => safetensors::bf16_slice_to_f32(bytes),
        "F32" => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        dt => panic!("{}: unhandled dtype {}", context, dt),
    }
}

/// Fetches a small real tensor (bf16 or f32), truncated to the first `n` values.
fn fetch_bf16_first(k3: &mut K3, full_name: &str, n: usize, logical: &str, src: &mut Sources) -> Option<()> {
    let (bytes, info) = k3.tensor_bytes(full_name)?;
    let bpe = if info.dtype == "BF16" { 2 } else { 4 };
    let vals = bytes_to_f32(&bytes[..n * bpe], &info.dtype, full_name);
    note(&mut src.report, logical, &format!("REAL K3 ({}→f32)", info.dtype));
    src.real.insert(logical.to_string(), vals);
    Some(())
}

fn gather_sources() -> Sources {
    let mut src = Sources {
        real: HashMap::new(),
        pool_w1: Vec::new(),
        pool_w2: Vec::new(),
        pool_w3: Vec::new(),
        qwen_attn: Vec::new(),
        qwen_mlp: Vec::new(),
        qwen_embed: Vec::new(),
        report: Vec::new(),
    };

    // ── 1) small real K3 tensors ──
    println!("── fetch K3 (range requests) ──");
    match K3::open() {
        Some(mut k3) => {
            let p = "language_model.model.";
            let jobs: &[(&str, usize, &str)] = &[
                ("layers.1.self_attn.A_log", 128, "A_log"),
                ("layers.1.self_attn.dt_bias", 512, "dt_bias"),
                ("layers.1.self_attn.o_norm.weight", 128, "o_norm"),
                ("layers.1.input_layernorm.weight", 512, "input_layernorm"),
                ("layers.1.post_attention_layernorm.weight", 512, "post_attention_layernorm"),
                ("layers.1.self_attention_res_norm.weight", 512, "self_attention_res_norm"),
                ("layers.1.self_attention_res_proj.weight", 512, "self_attention_res_proj"),
                ("layers.1.mlp_res_norm.weight", 512, "mlp_res_norm"),
                ("layers.1.mlp_res_proj.weight", 512, "mlp_res_proj"),
                ("output_attn_res_norm.weight", 512, "output_attn_res_norm"),
                ("output_attn_res_proj.weight", 512, "output_attn_res_proj"),
                ("norm.weight", 512, "norm_final"),
                ("layers.3.self_attn.q_a_layernorm.weight", 128, "q_a_layernorm"),
                ("layers.3.self_attn.kv_a_layernorm.weight", 64, "kv_a_layernorm"),
                ("layers.1.block_sparse_moe.routed_expert_norm.weight", 128, "routed_expert_norm"),
                (
                    "layers.1.block_sparse_moe.gate.e_score_correction_bias",
                    896,
                    "gate_bias",
                ),
            ];
            for (name, n, logical) in jobs {
                let full = format!("{}{}", p, name);
                if fetch_bf16_first(&mut k3, &full, *n, logical, &mut src).is_none() {
                    note(&mut src.report, logical, "FAIL fetch → synthetic fallback");
                }
            }
            // conv1d [12288,4] → first 512 rows
            for w in ["q", "k", "v"] {
                let full = format!("{}layers.1.self_attn.{}_conv1d.weight", p, w);
                let logical = format!("{}_conv1d", w);
                if let Some((bytes, info)) = k3.tensor_bytes(&full) {
                    let bpe = if info.dtype == "BF16" { 2 } else { 4 };
                    let vals = bytes_to_f32(&bytes[..512 * 4 * bpe], &info.dtype, &full);
                    note(&mut src.report, &logical, &format!("REAL K3 (512 rows, {})", info.dtype));
                    src.real.insert(logical, vals);
                } else {
                    note(&mut src.report, &logical, "FAIL fetch → Gaussian fallback");
                }
            }
            // gate.weight [896,7168] → columns 0..512 (single range ~12.8 MB)
            let full = format!("{}layers.1.block_sparse_moe.gate.weight", p);
            if let Some((bytes, info)) = k3.tensor_bytes(&full) {
                let (r, c) = (info.shape[0], info.shape[1]);
                let bpe = if info.dtype == "BF16" { 2 } else { 4 };
                let mut vals = vec![0f32; r * 512];
                for row in 0..r {
                    let off = row * c * bpe;
                    vals[row * 512..(row + 1) * 512]
                        .copy_from_slice(&bytes_to_f32(&bytes[off..off + 512 * bpe], &info.dtype, &full));
                }
                note(&mut src.report, "gate.weight (cols 0..512)", &format!("REAL K3 (~13 MB, {})", info.dtype));
                src.real.insert("gate_weight".to_string(), vals);
            } else {
                note(&mut src.report, "gate.weight", "FAIL fetch → Qwen pool fallback");
            }
            // ── 2) three real MXFP4 experts → pools ──
            for (layer, eid) in [(1, 0), (45, 0), (91, 0)] {
                for w in ["w1", "w2", "w3"] {
                    let pfx = format!("{}layers.{}.block_sparse_moe.experts.{}.{}", p, layer, eid, w);
                    let pk = k3.tensor_bytes(&format!("{}.weight_packed", pfx));
                    let sc = k3.tensor_bytes(&format!("{}.weight_scale", pfx));
                    if let (Some((pb, pi)), Some((sb, _))) = (pk, sc) {
                        // weight_packed: [R, C/2] bytes → LOGICAL dims [R, C]
                        let (r, c) = (pi.shape[0], pi.shape[1] * 2);
                        let vals = mxfp4::dequant(&pb, &sb, r, c);
                        let pool = match w {
                            "w1" => &mut src.pool_w1,
                            "w2" => &mut src.pool_w2,
                            _ => &mut src.pool_w3,
                        };
                        pool.extend_from_slice(&vals);
                        note(
                            &mut src.report,
                            &format!("expert L{}-E{} {} [{}×{}]", layer, eid, w, r, c),
                            "REAL K3 MXFP4 → pool",
                        );
                    }
                }
            }
        }
        None => {
            note(&mut src.report, "K3 network", "INACCESSIBLE → full Qwen + Gaussian fallback");
        }
    }
    if src.pool_w1.is_empty() {
        note(&mut src.report, "expert pools", "EMPTY → Gaussian fallback σ=0.02");
    }

    // ── 3) local Qwen2.5-0.5B: embed + projection pools ──
    println!("── Qwen2.5-0.5B (local HF cache) ──");
    let qwen = qwen_path();
    let data = std::fs::read(&qwen).expect("Qwen model.safetensors unreadable");
    let (hlen, map) = safetensors::parse_header(&data);
    let dstart = 8 + hlen as usize;
    let get_bf16 = |name: &str| -> Vec<f32> {
        let info = &map[name];
        let (s, e) = (dstart + info.offsets.0 as usize, dstart + info.offsets.1 as usize);
        safetensors::bf16_slice_to_f32(&data[s..e])
    };
    for w in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        src.qwen_attn
            .extend_from_slice(&get_bf16(&format!("model.layers.0.self_attn.{}.weight", w)));
    }
    for w in ["gate_proj", "up_proj", "down_proj"] {
        src.qwen_mlp
            .extend_from_slice(&get_bf16(&format!("model.layers.0.mlp.{}.weight", w)));
    }
    note(&mut src.report, "pool projections attn (q/k/v/o L0)", "REAL Qwen");
    note(&mut src.report, "pool projections mlp (gate/up/down L0)", "REAL Qwen");
    // embed: [151936,896] → columns 0..512
    {
        let info = &map["model.embed_tokens.weight"];
        let (r, c) = (info.shape[0], info.shape[1]);
        let (s, _) = (dstart + info.offsets.0 as usize, dstart + info.offsets.1 as usize);
        let mut vals = vec![0f32; r * 512];
        for row in 0..r {
            let off = s + row * c * 2;
            vals[row * 512..(row + 1) * 512]
                .copy_from_slice(&safetensors::bf16_slice_to_f32(&data[off..off + 512 * 2]));
        }
        note(&mut src.report, "embed_tokens (151936 rows × cols 0..512)", "REAL Qwen");
        src.qwen_embed = vec![(0, vals)]; // stored as-is, size 151936×512
    }
    src
}

fn qwen_path() -> String {
    let cache_dir = format!(
        "{}/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots",
        std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
    );
    if let Ok(entries) = std::fs::read_dir(&cache_dir) {
        for e in entries.flatten() {
            let p = e.path().join("model.safetensors");
            if p.exists() {
                return p.to_string_lossy().into_owned();
            }
        }
    }
    // no local HF cache: download the Qwen2.5-0.5B-Instruct checkpoint
    // (Apache 2.0, ~943 MB) into ~/.cache/microkimi/
    let cache = format!("{}/.cache/microkimi", std::env::var("HOME").unwrap_or_default());
    std::fs::create_dir_all(&cache).ok();
    let dst = format!("{}/qwen2.5-0.5b-instruct.safetensors", cache);
    if !std::path::Path::new(&dst).exists() {
        println!("downloading Qwen2.5-0.5B-Instruct model.safetensors (~943 MB) …");
        let data = crate::stream::http::fetch(
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors",
        )
        .expect("failed to download the Qwen checkpoint (no local HF cache, no network)");
        std::fs::write(&dst, data).unwrap();
    }
    dst
}

// ── generation helpers ──

fn fill_pool_or_gauss(name: &str, n: usize, pool: &[f32]) -> Vec<f32> {
    let mut rng = Rng::for_tensor(name);
    (0..n)
        .map(|_| {
            if pool.is_empty() {
                rng.gauss() * SIGMA
            } else {
                rng.pick(pool)
            }
        })
        .collect()
}

fn fill_gauss(name: &str, n: usize, sigma: f32) -> Vec<f32> {
    let mut rng = Rng::for_tensor(name);
    (0..n).map(|_| rng.gauss() * sigma).collect()
}

/// Returns the real value if available, otherwise the fallback.
fn real_or(src: &Sources, logical: &str, fallback: Vec<f32>) -> Vec<f32> {
    src.real.get(logical).cloned().unwrap_or(fallback)
}

// ── main build ──

pub fn run() {
    let t0 = std::time::Instant::now();
    println!("microkimi build - building {}", OUT);
    let src = gather_sources();

    // ── registration of all tensors (order = write order) ──
    let mut w = BinWriter::new();
    w.add("embed_tokens.weight", DTYPE_F32, vec![163_840, 512]);
    w.add("lm_head.weight", DTYPE_F32, vec![163_840, 512]);
    w.add("norm.weight", DTYPE_F32, vec![512]);
    w.add("output_attn_res_norm.weight", DTYPE_F32, vec![512]);
    w.add("output_attn_res_proj.weight", DTYPE_F32, vec![512]);
    for l in 0..93 {
        let p = format!("layers.{}.", l);
        w.add(&format!("{}input_layernorm.weight", p), DTYPE_F32, vec![512]);
        w.add(&format!("{}post_attention_layernorm.weight", p), DTYPE_F32, vec![512]);
        w.add(&format!("{}self_attention_res_norm.weight", p), DTYPE_F32, vec![512]);
        w.add(&format!("{}self_attention_res_proj.weight", p), DTYPE_F32, vec![512]);
        w.add(&format!("{}mlp_res_norm.weight", p), DTYPE_F32, vec![512]);
        w.add(&format!("{}mlp_res_proj.weight", p), DTYPE_F32, vec![512]);
        if crate::model::is_mla(l) {
            w.add(&format!("{}self_attn.q_a_proj.weight", p), DTYPE_F32, vec![128, 512]);
            w.add(&format!("{}self_attn.q_a_layernorm.weight", p), DTYPE_F32, vec![128]);
            w.add(&format!("{}self_attn.q_b_proj.weight", p), DTYPE_F32, vec![768, 128]);
            w.add(&format!("{}self_attn.kv_a_proj_with_mqa.weight", p), DTYPE_F32, vec![128, 512]);
            w.add(&format!("{}self_attn.kv_a_layernorm.weight", p), DTYPE_F32, vec![64]);
            w.add(&format!("{}self_attn.kv_b_proj.weight", p), DTYPE_F32, vec![1024, 64]);
            w.add(&format!("{}self_attn.g_proj.weight", p), DTYPE_F32, vec![512, 512]);
            // NB: the flattened attention output is 4×128=512
            // (Moonshot: Linear(H·v_dim, hidden)), so o_proj is [512,512]
            // (verified 1:1 against modeling_kimi_linear.py).
            w.add(&format!("{}self_attn.o_proj.weight", p), DTYPE_F32, vec![512, 512]);
        } else {
            for x in ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"] {
                w.add(&format!("{}self_attn.{}.weight", p, x), DTYPE_F32, vec![512, 512]);
            }
            for x in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                w.add(&format!("{}self_attn.{}.weight", p, x), DTYPE_F32, vec![512, 4]);
            }
            w.add(&format!("{}self_attn.f_a_proj.weight", p), DTYPE_F32, vec![128, 512]);
            w.add(&format!("{}self_attn.f_b_proj.weight", p), DTYPE_F32, vec![512, 128]);
            w.add(&format!("{}self_attn.A_log", p), DTYPE_F32, vec![128]);
            w.add(&format!("{}self_attn.dt_bias", p), DTYPE_F32, vec![512]);
            w.add(&format!("{}self_attn.b_proj.weight", p), DTYPE_F32, vec![4, 512]);
            w.add(&format!("{}self_attn.o_norm.weight", p), DTYPE_F32, vec![128]);
        }
        if crate::model::is_moe(l) {
            w.add(&format!("{}block_sparse_moe.gate.weight", p), DTYPE_F32, vec![896, 512]);
            w.add(&format!("{}block_sparse_moe.gate.e_score_correction_bias", p), DTYPE_F32, vec![896]);
            w.add(&format!("{}block_sparse_moe.routed_expert_down_proj.weight", p), DTYPE_F32, vec![128, 512]);
            w.add(&format!("{}block_sparse_moe.routed_expert_up_proj.weight", p), DTYPE_F32, vec![512, 128]);
            w.add(&format!("{}block_sparse_moe.routed_expert_norm.weight", p), DTYPE_F32, vec![128]);
            w.add(&format!("{}block_sparse_moe.shared_experts.gate_proj.weight", p), DTYPE_F32, vec![128, 512]);
            w.add(&format!("{}block_sparse_moe.shared_experts.up_proj.weight", p), DTYPE_F32, vec![128, 512]);
            w.add(&format!("{}block_sparse_moe.shared_experts.down_proj.weight", p), DTYPE_F32, vec![512, 128]);
        } else {
            w.add(&format!("{}mlp.gate_proj.weight", p), DTYPE_F32, vec![2048, 512]);
            w.add(&format!("{}mlp.up_proj.weight", p), DTYPE_F32, vec![2048, 512]);
            w.add(&format!("{}mlp.down_proj.weight", p), DTYPE_F32, vec![512, 2048]);
        }
    }
    // MXFP4 experts (92 MoE layers × 896 experts × 3 matrices)
    for l in 1..93 {
        for e in 0..896 {
            let p = format!("layers.{}.block_sparse_moe.experts.{}.", l, e);
            w.add(&format!("{}w1", p), DTYPE_MXFP4, vec![64, 128]);
            w.add(&format!("{}w2", p), DTYPE_MXFP4, vec![128, 64]);
            w.add(&format!("{}w3", p), DTYPE_MXFP4, vec![64, 128]);
        }
    }
    println!("{} tensors in the directory", w.names_order.len());

    let mut f = std::fs::File::create(OUT).unwrap();
    let offsets = w.write_header(&mut f);
    let off_map: HashMap<&str, u64> = w
        .names_order
        .iter()
        .zip(&offsets)
        .map(|((n, _, _), &o)| (n.as_str(), o))
        .collect();

    // ── writing the f32 tensors ──
    println!("── writing f32 tensors (spine + embeddings) ──");
    let put = |f: &mut std::fs::File, w: &BinWriter, name: &str, vals: &[f32]| {
        w.write_blob_at(f, off_map[name], &weights::f32_to_bytes(vals));
    };

    // embed: rows 0..151935 = Qwen (cols 0..512), 151936..163839 = Gaussian
    {
        let mut embed = src.qwen_embed[0].1.clone();
        let tail = fill_gauss("embed_tail", (163_840 - 151_936) * 512, SIGMA);
        embed.extend_from_slice(&tail);
        put(&mut f, &w, "embed_tokens.weight", &embed);
        // lm_head = independent copy (same Qwen values, own Gaussian draw)
        let mut lm = src.qwen_embed[0].1.clone();
        let tail2 = fill_gauss("lm_head_tail", (163_840 - 151_936) * 512, SIGMA);
        lm.extend_from_slice(&tail2);
        put(&mut f, &w, "lm_head.weight", &lm);
        note_print("embed_tokens / lm_head: 151936 REAL Qwen rows + Gaussian tail");
    }
    put(&mut f, &w, "norm.weight", &real_or(&src, "norm_final", vec![1.0; 512]));
    put(&mut f, &w, "output_attn_res_norm.weight", &real_or(&src, "output_attn_res_norm", vec![1.0; 512]));
    put(
        &mut f,
        &w,
        "output_attn_res_proj.weight",
        &real_or(&src, "output_attn_res_proj", fill_gauss("oar_proj_fb", 512, SIGMA)),
    );

    for l in 0..93 {
        let p = format!("layers.{}.", l);
        put(&mut f, &w, &format!("{}input_layernorm.weight", p), &real_or(&src, "input_layernorm", vec![1.0; 512]));
        put(&mut f, &w, &format!("{}post_attention_layernorm.weight", p), &real_or(&src, "post_attention_layernorm", vec![1.0; 512]));
        put(&mut f, &w, &format!("{}self_attention_res_norm.weight", p), &real_or(&src, "self_attention_res_norm", vec![1.0; 512]));
        put(&mut f, &w, &format!("{}self_attention_res_proj.weight", p), &real_or(&src, "self_attention_res_proj", fill_gauss(&format!("sar{}", l), 512, SIGMA)));
        put(&mut f, &w, &format!("{}mlp_res_norm.weight", p), &real_or(&src, "mlp_res_norm", vec![1.0; 512]));
        put(&mut f, &w, &format!("{}mlp_res_proj.weight", p), &real_or(&src, "mlp_res_proj", fill_gauss(&format!("mlpres{}", l), 512, SIGMA)));

        if crate::model::is_mla(l) {
            let attn_pool = &src.qwen_attn;
            put(&mut f, &w, &format!("{}self_attn.q_a_proj.weight", p), &fill_pool_or_gauss(&format!("{}qa", p), 128 * 512, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.q_a_layernorm.weight", p), &real_or(&src, "q_a_layernorm", vec![1.0; 128]));
            put(&mut f, &w, &format!("{}self_attn.q_b_proj.weight", p), &fill_pool_or_gauss(&format!("{}qb", p), 768 * 128, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.kv_a_proj_with_mqa.weight", p), &fill_pool_or_gauss(&format!("{}kva", p), 128 * 512, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.kv_a_layernorm.weight", p), &real_or(&src, "kv_a_layernorm", vec![1.0; 64]));
            put(&mut f, &w, &format!("{}self_attn.kv_b_proj.weight", p), &fill_pool_or_gauss(&format!("{}kvb", p), 1024 * 64, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.g_proj.weight", p), &fill_pool_or_gauss(&format!("{}g", p), 512 * 512, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.o_proj.weight", p), &fill_pool_or_gauss(&format!("{}o", p), 512 * 512, attn_pool));
        } else {
            let attn_pool = &src.qwen_attn;
            for x in ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"] {
                put(&mut f, &w, &format!("{}self_attn.{}.weight", p, x), &fill_pool_or_gauss(&format!("{}{}", p, x), 512 * 512, attn_pool));
            }
            for x in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                put(
                    &mut f,
                    &w,
                    &format!("{}self_attn.{}.weight", p, x),
                    &real_or(&src, x, fill_gauss(&format!("{}{}", p, x), 512 * 4, SIGMA)),
                );
            }
            put(&mut f, &w, &format!("{}self_attn.f_a_proj.weight", p), &fill_pool_or_gauss(&format!("{}fa", p), 128 * 512, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.f_b_proj.weight", p), &fill_pool_or_gauss(&format!("{}fb", p), 512 * 128, attn_pool));
            put(
                &mut f,
                &w,
                &format!("{}self_attn.A_log", p),
                &real_or(&src, "A_log", {
                    // realistic init: log(uniform(1,16))
                    let mut rng = Rng::for_tensor(&format!("{}alog", p));
                    (0..128).map(|_| (1.0 + rng.uniform() * 15.0).ln() as f32).collect()
                }),
            );
            put(&mut f, &w, &format!("{}self_attn.dt_bias", p), &real_or(&src, "dt_bias", fill_gauss(&format!("{}dt", p), 512, SIGMA)));
            put(&mut f, &w, &format!("{}self_attn.b_proj.weight", p), &fill_pool_or_gauss(&format!("{}b", p), 4 * 512, attn_pool));
            put(&mut f, &w, &format!("{}self_attn.o_norm.weight", p), &real_or(&src, "o_norm", vec![1.0; 128]));
        }
        if crate::model::is_moe(l) {
            let mlp_pool = &src.qwen_mlp;
            put(
                &mut f,
                &w,
                &format!("{}block_sparse_moe.gate.weight", p),
                &real_or(&src, "gate_weight", fill_pool_or_gauss(&format!("{}gate", p), 896 * 512, &src.qwen_attn)),
            );
            put(
                &mut f,
                &w,
                &format!("{}block_sparse_moe.gate.e_score_correction_bias", p),
                &real_or(&src, "gate_bias", vec![0.0; 896]),
            );
            put(&mut f, &w, &format!("{}block_sparse_moe.routed_expert_down_proj.weight", p), &fill_pool_or_gauss(&format!("{}rd", p), 128 * 512, mlp_pool));
            put(&mut f, &w, &format!("{}block_sparse_moe.routed_expert_up_proj.weight", p), &fill_pool_or_gauss(&format!("{}ru", p), 512 * 128, mlp_pool));
            put(&mut f, &w, &format!("{}block_sparse_moe.routed_expert_norm.weight", p), &real_or(&src, "routed_expert_norm", vec![1.0; 128]));
            put(&mut f, &w, &format!("{}block_sparse_moe.shared_experts.gate_proj.weight", p), &fill_pool_or_gauss(&format!("{}sg", p), 128 * 512, mlp_pool));
            put(&mut f, &w, &format!("{}block_sparse_moe.shared_experts.up_proj.weight", p), &fill_pool_or_gauss(&format!("{}su", p), 128 * 512, mlp_pool));
            put(&mut f, &w, &format!("{}block_sparse_moe.shared_experts.down_proj.weight", p), &fill_pool_or_gauss(&format!("{}sd", p), 512 * 128, mlp_pool));
        } else {
            let mlp_pool = &src.qwen_mlp;
            put(&mut f, &w, &format!("{}mlp.gate_proj.weight", p), &fill_pool_or_gauss(&format!("{}dg", p), 2048 * 512, mlp_pool));
            put(&mut f, &w, &format!("{}mlp.up_proj.weight", p), &fill_pool_or_gauss(&format!("{}du", p), 2048 * 512, mlp_pool));
            put(&mut f, &w, &format!("{}mlp.down_proj.weight", p), &fill_pool_or_gauss(&format!("{}dd", p), 512 * 2048, mlp_pool));
        }
        if l % 20 == 0 {
            println!("  spine layer {}/93 written ({:.0?})", l + 1, t0.elapsed());
        }
    }

    // ── MXFP4 experts: parallel generation per layer (seeds per tensor) ──
    println!("── MXFP4 experts: 92 layers × 896 experts × 3 matrices ──");
    let pool_w1 = std::sync::Arc::new(src.pool_w1.clone());
    let pool_w2 = std::sync::Arc::new(src.pool_w2.clone());
    let pool_w3 = std::sync::Arc::new(src.pool_w3.clone());
    let off_map = std::sync::Arc::new(
        w.names_order
            .iter()
            .zip(&offsets)
            .map(|((n, _, _), &o)| (n.clone(), o))
            .collect::<HashMap<String, u64>>(),
    );
    let nt = crate::model::n_threads();
    let layers: Vec<usize> = (1..93).collect();
    let chunk = layers.len().div_ceil(nt);
    let te = std::time::Instant::now();
    std::thread::scope(|s| {
        for layer_chunk in layers.chunks(chunk) {
            let (p1, p2, p3, om) = (pool_w1.clone(), pool_w2.clone(), pool_w3.clone(), off_map.clone());
            let layer_chunk = layer_chunk.to_vec();
            s.spawn(move || {
                let mut f = std::fs::OpenOptions::new().write(true).open(OUT).unwrap();
                use std::io::{Seek, SeekFrom, Write};
                for l in layer_chunk {
                    for e in 0..896 {
                        for (wn, dims, pool) in [
                            ("w1", (64usize, 128usize), &p1),
                            ("w2", (128, 64), &p2),
                            ("w3", (64, 128), &p3),
                        ] {
                            let name = format!("layers.{}.block_sparse_moe.experts.{}.{}", l, e, wn);
                            let vals = fill_pool_or_gauss(&name, dims.0 * dims.1, pool);
                            let (packed, scales) = mxfp4::quantize(&vals, dims.0, dims.1);
                            let mut blob = packed;
                            blob.extend_from_slice(&scales);
                            f.seek(SeekFrom::Start(om[&name])).unwrap();
                            f.write_all(&blob).unwrap();
                        }
                    }
                }
            });
        }
    });
    println!("  experts written in {:.0?}", te.elapsed());

    let size = std::fs::metadata(OUT).unwrap().len();
    println!();
    println!("══ microkimi-debug.bin: {:.2} GB in {:.0?} ══", size as f64 / 1e9, t0.elapsed());
    println!("weight provenance:");
    for line in &src.report {
        println!("{}", line);
    }
    println!("    projections (q/k/v/g/o/f/b, MLA, MoE shared/routed, dense)   REAL Qwen pools (i.i.d.) or Gaussian fallback");
    println!("    82432 MXFP4 experts                                          sampled i.i.d. from the real K3 pools (or Gaussian)");
    println!("    all layers reuse the same small real values ");
}

fn note_print(s: &str) {
    println!("    {}", s);
}
