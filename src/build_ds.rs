// `microkimi build-ds`: builds microdeepseek-debug.bin (MKIM0002, arch "deepseek_v4").
// Same philosophy as build.rs (K3): small REAL DeepSeek-V4 fragments fetched by
// HTTP range requests (leading rows of row-major tensors are contiguous ranges),
// a few real fp4 experts → value pools, reproducible xorshift64* generation
// (seed per tensor) for everything else, fp4 quantization via mxfp4.
// Graceful fallback: if the network fails → Gaussian σ=0.02 everywhere
// (norms → 1.0, biases → 0.0), hash tables → seeded pseudo-random.

use crate::config::DsConfig;
use crate::http;
use crate::mxfp4;
use crate::safetensors;
use crate::weights::{self, BinWriter, DTYPE_F32, DTYPE_I32, DTYPE_MXFP4};
use std::collections::HashMap;

const DS_BASE: &str = "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/main/";
const OUT: &str = "microdeepseek-debug.bin";
const SIGMA: f32 = 0.02;
const REAL_ROWS: usize = 16384; // embed/head: this many real leading rows

// Reuse the K3 builder RNG (same xorshift64* + name-seeded derivation, own base).
use crate::build::Rng;

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

// ── fetch: index + shard headers + tensor slices ──

struct Ds {
    weight_map: HashMap<String, String>,
    headers: HashMap<String, (u64, HashMap<String, safetensors::TensorInfo>)>,
}

impl Ds {
    fn open() -> Option<Ds> {
        let idx = http::fetch(&format!("{}model.safetensors.index.json", DS_BASE))?;
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
        println!("  V4 index: {} tensors referenced", weight_map.len());
        Some(Ds { weight_map, headers: HashMap::new() })
    }

    fn shard_of(&mut self, name: &str) -> Option<(u64, String, safetensors::TensorInfo)> {
        let shard = self.weight_map.get(name)?.clone();
        if !self.headers.contains_key(&shard) {
            let url = format!("{}{}", DS_BASE, shard);
            let first = http::fetch_range(&url, Some((0, 7)))?;
            let hlen = u64::from_le_bytes(first[0..8].try_into().unwrap());
            let head = http::fetch_range(&url, Some((8, 8 + hlen - 1)))?;
            let full = [first, head].concat();
            let (hlen, map) = safetensors::parse_header(&full);
            self.headers.insert(shard.clone(), (hlen, map));
        }
        let (hlen, map) = self.headers.get(&shard)?;
        let info = map.get(name)?.clone();
        Some((*hlen, shard, info))
    }

    /// Raw bytes of the inclusive range [off0, off1) INSIDE the tensor.
    fn tensor_range(&mut self, name: &str, off0: u64, off1: u64) -> Option<(Vec<u8>, safetensors::TensorInfo)> {
        let (hlen, shard, info) = self.shard_of(name)?;
        let start = 8 + hlen + info.offsets.0 + off0;
        let end = 8 + hlen + info.offsets.0 + off1 - 1;
        let bytes = http::fetch_range(&format!("{}{}", DS_BASE, shard), Some((start, end)))?;
        Some((bytes, info))
    }

    /// Whole tensor.
    fn tensor_bytes(&mut self, name: &str) -> Option<(Vec<u8>, safetensors::TensorInfo)> {
        let (hlen, shard, info) = self.shard_of(name)?;
        let start = 8 + hlen + info.offsets.0;
        let end = 8 + hlen + info.offsets.1 - 1;
        let bytes = http::fetch_range(&format!("{}{}", DS_BASE, shard), Some((start, end)))?;
        Some((bytes, info))
    }
}

fn bytes_to_f32(bytes: &[u8], dtype: &str, context: &str) -> Vec<f32> {
    match dtype {
        "BF16" => safetensors::bf16_slice_to_f32(bytes),
        "F32" => bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        dt => panic!("{}: unhandled dtype {}", context, dt),
    }
}

// ── collected sources ──

struct Src {
    real: HashMap<String, Vec<f32>>, // real values by micro logical name
    tid2eid: [Option<Vec<i32>>; 3],  // real hash tables (layers 0/1/2)
    pool_w1: Vec<f32>,
    pool_w2: Vec<f32>,
    pool_w3: Vec<f32>,
    pool_attn: Vec<f32>, // real attn projection values (layer 2)
    pool_embed: Vec<f32>,
    pool_head: Vec<f32>,
    report: Vec<String>,
}

fn note(report: &mut Vec<String>, what: &str, src: &str) {
    report.push(format!("    {:<56} {}", what, src));
}

/// Small real vector → first n values as f32.
fn fetch_first(ds: &mut Ds, full: &str, n: usize, logical: &str, src: &mut Src) -> Option<()> {
    let (bytes, info) = ds.tensor_bytes(full)?;
    let bpe = if info.dtype == "BF16" { 2 } else { 4 };
    let vals = bytes_to_f32(&bytes[..n * bpe], &info.dtype, full);
    note(&mut src.report, logical, &format!("REAL V4 ({}→f32)", info.dtype));
    src.real.insert(logical.to_string(), vals);
    Some(())
}

/// Real matrix, leading rows (contiguous range) → micro slice [r0, c0] f32.
/// The full fetched rows are dequantized, then the leading [r0, c0] corner is
/// taken; the rest of the fetched values feeds the matching pool.
fn fetch_fp8_corner(
    ds: &mut Ds,
    full: &str,
    fetch_rows: usize,
    cols: usize,
    r0: usize,
    c0: usize,
    logical: &str,
    pool: &mut Vec<f32>,
    src: &mut Src,
) -> Option<()> {
    let (_, _, info) = ds.shard_of(full)?;
    assert_eq!(info.dtype, "F8_E4M3", "{}: expected fp8", full);
    let w = ds.tensor_range(full, 0, (fetch_rows * cols) as u64)?.0;
    let (sc, _) = ds.tensor_bytes(&format!("{}.scale", full.strip_suffix(".weight")?))?;
    let srows = info.shape[0].div_ceil(128);
    let scols = info.shape[1].div_ceil(128);
    assert_eq!(sc.len(), srows * scols, "{}: scale size", full);
    // dequant the fetched [fetch_rows, cols] with its scale blocks
    let mut vals = vec![0f32; fetch_rows * cols];
    for r in 0..fetch_rows {
        for c in 0..cols {
            let s = mxfp4::exp2_i(sc[(r / 128) * scols + c / 128] as i32 - 127);
            vals[r * cols + c] = crate::dequant::e4m3_to_f32(w[r * cols + c]) * s;
        }
    }
    let mut corner = vec![0f32; r0 * c0];
    for r in 0..r0 {
        corner[r * c0..(r + 1) * c0].copy_from_slice(&vals[r * cols..r * cols + c0]);
    }
    note(&mut src.report, logical, "REAL V4 fp8 (leading rows)");
    src.real.insert(logical.to_string(), corner);
    pool.extend_from_slice(&vals);
    Some(())
}

/// bf16 matrix, leading rows → micro corner [r0, c0].
fn fetch_bf16_corner(
    ds: &mut Ds,
    full: &str,
    fetch_rows: usize,
    cols: usize,
    r0: usize,
    c0: usize,
    logical: &str,
    src: &mut Src,
) -> Option<()> {
    let (raw, info) = ds.tensor_range(full, 0, (fetch_rows * cols * 2) as u64)?;
    assert_eq!(info.dtype, "BF16", "{}: expected bf16", full);
    let vals = safetensors::bf16_slice_to_f32(&raw);
    let mut corner = vec![0f32; r0 * c0];
    for r in 0..r0 {
        corner[r * c0..(r + 1) * c0].copy_from_slice(&vals[r * cols..r * cols + c0]);
    }
    note(&mut src.report, logical, "REAL V4 bf16 (leading rows)");
    src.real.insert(logical.to_string(), corner);
    Some(())
}

fn gather_sources(cfg: &DsConfig) -> Src {
    let mut src = Src {
        real: HashMap::new(),
        tid2eid: [None, None, None],
        pool_w1: Vec::new(),
        pool_w2: Vec::new(),
        pool_w3: Vec::new(),
        pool_attn: Vec::new(),
        pool_embed: Vec::new(),
        pool_head: Vec::new(),
        report: Vec::new(),
    };
    let (d, hd, ql, il) = (cfg.d, cfg.head_dim, cfg.q_lora_rank, cfg.index_head_dim);

    println!("── fetch DeepSeek-V4 (range requests) ──");
    match Ds::open() {
        Some(mut ds) => {
            // norms (layer 2 = overlap layer; layer 3 = dense layer), reused everywhere
            for (full, n, logical) in [
                ("norm.weight", d, "norm_final"),
                ("layers.2.attn_norm.weight", d, "attn_norm"),
                ("layers.2.ffn_norm.weight", d, "ffn_norm"),
                ("layers.2.attn.q_norm.weight", ql, "q_norm"),
                ("layers.2.attn.kv_norm.weight", hd, "kv_norm"),
                ("layers.2.attn.compressor.norm.weight", hd, "comp_norm_ov"),
                ("layers.3.attn.compressor.norm.weight", hd, "comp_norm_de"),
                ("layers.2.attn.indexer.compressor.norm.weight", il, "idx_comp_norm"),
            ] {
                if fetch_first(&mut ds, full, n, logical, &mut src).is_none() {
                    note(&mut src.report, logical, "FAIL → 1.0 fallback");
                }
            }
            // attn_sink (f32 [64] → first n_heads), hc head params
            if fetch_first(&mut ds, "layers.2.attn.attn_sink", cfg.n_heads, "attn_sink", &mut src).is_none() {
                note(&mut src.report, "attn_sink", "FAIL → 0.0 fallback");
            }
            // hc_head_fn [4, 16384] → [4, hc*d] corner; bases/scales
            {
                let hc = cfg.hc_mult;
                if let Some((raw, _)) = ds.tensor_bytes("hc_head_fn") {
                    let vals = bytes_to_f32(&raw, "F32", "hc_head_fn");
                    let mut corner = vec![0f32; hc * hc * d];
                    for r in 0..hc {
                        corner[r * hc * d..(r + 1) * hc * d].copy_from_slice(&vals[r * 16384..r * 16384 + hc * d]);
                    }
                    src.real.insert("hc_head_fn".to_string(), corner);
                    note(&mut src.report, "hc_head_fn", "REAL V4 f32 (corner)");
                } else {
                    note(&mut src.report, "hc_head_fn", "FAIL → Gaussian fallback");
                }
                if fetch_first(&mut ds, "hc_head_base", hc, "hc_head_base", &mut src).is_none() {
                    note(&mut src.report, "hc_head_base", "FAIL → 0.0 fallback");
                }
                if fetch_first(&mut ds, "hc_head_scale", 1, "hc_head_scale", &mut src).is_none() {
                    note(&mut src.report, "hc_head_scale", "FAIL → 1.0 fallback");
                }
            }
            // per-layer hc params from layer 2, reused everywhere: [24, 16384] → [24, hc*d]
            for kind in ["attn", "ffn"] {
                let name = format!("layers.2.hc_{}_fn", kind);
                let logical = format!("hc_{}_fn", kind);
                if let Some((raw, _)) = ds.tensor_bytes(&name) {
                    let vals = bytes_to_f32(&raw, "F32", &name);
                    let mix = 24usize;
                    let mut corner = vec![0f32; mix * cfg.hc_mult * d];
                    for r in 0..mix {
                        corner[r * cfg.hc_mult * d..(r + 1) * cfg.hc_mult * d]
                            .copy_from_slice(&vals[r * 16384..r * 16384 + cfg.hc_mult * d]);
                    }
                    src.real.insert(logical.clone(), corner);
                    note(&mut src.report, &logical, "REAL V4 f32 (corner, layer 2)");
                } else {
                    note(&mut src.report, &logical, "FAIL → Gaussian fallback");
                }
                if fetch_first(&mut ds, &format!("layers.2.hc_{}_base", kind), 24, &format!("hc_{}_base", kind), &mut src).is_none() {
                    note(&mut src.report, &format!("hc_{}_base", kind), "FAIL → 0.0 fallback");
                }
                if fetch_first(&mut ds, &format!("layers.2.hc_{}_scale", kind), 3, &format!("hc_{}_scale", kind), &mut src).is_none() {
                    note(&mut src.report, &format!("hc_{}_scale", kind), "FAIL → 1.0 fallback");
                }
            }
            // gate.weight layer 2 [256, 4096] → cols 0..d (whole rows); bias layer 3
            {
                if fetch_bf16_corner(&mut ds, "layers.2.ffn.gate.weight", 256, 4096, 256, d, "gate_weight", &mut src).is_none() {
                    note(&mut src.report, "gate.weight", "FAIL → Gaussian fallback");
                }
                if fetch_first(&mut ds, "layers.3.ffn.gate.bias", cfg.n_routed_experts, "gate_bias", &mut src).is_none() {
                    note(&mut src.report, "gate.bias", "FAIL → 0.0 fallback");
                }
            }
            // hash tables (i64 [129280, 6] → i32), one per hash layer
            for l in 0..3 {
                let name = format!("layers.{}.ffn.gate.tid2eid", l);
                match ds.tensor_bytes(&name) {
                    Some((raw, _)) => {
                        let t: Vec<i32> = raw
                            .chunks_exact(8)
                            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as i32)
                            .collect();
                        note(&mut src.report, &format!("tid2eid layer {}", l), "REAL V4 i64→i32");
                        src.tid2eid[l] = Some(t);
                    }
                    None => note(&mut src.report, &format!("tid2eid layer {}", l), "FAIL → seeded fallback"),
                }
            }
            // compressor (overlap, layer 2): wkv/wgate [1024, 4096] → [2*hd, d]
            {
                for w in ["wkv", "wgate"] {
                    let logical = format!("comp_ov_{}", w);
                    if fetch_bf16_corner(&mut ds, &format!("layers.2.attn.compressor.{}.weight", w), 2 * hd, 4096, 2 * hd, d, &logical, &mut src).is_none() {
                        note(&mut src.report, &logical, "FAIL → Gaussian fallback");
                    }
                }
                // ape f32 [4, 1024] → [4, 2*hd] corner
                if let Some((raw, _)) = ds.tensor_bytes("layers.2.attn.compressor.ape") {
                    let vals = bytes_to_f32(&raw, "F32", "ape");
                    let mut corner = vec![0f32; 4 * 2 * hd];
                    for r in 0..4 {
                        corner[r * 2 * hd..(r + 1) * 2 * hd].copy_from_slice(&vals[r * 1024..r * 1024 + 2 * hd]);
                    }
                    src.real.insert("comp_ov_ape".to_string(), corner);
                    note(&mut src.report, "comp ape (overlap)", "REAL V4 f32 (corner)");
                } else {
                    note(&mut src.report, "comp ape (overlap)", "FAIL → Gaussian fallback");
                }
            }
            // compressor (dense, layer 3): wkv/wgate [512, 4096] → [hd, d]
            {
                for w in ["wkv", "wgate"] {
                    let logical = format!("comp_de_{}", w);
                    if fetch_bf16_corner(&mut ds, &format!("layers.3.attn.compressor.{}.weight", w), hd, 4096, hd, d, &logical, &mut src).is_none() {
                        note(&mut src.report, &logical, "FAIL → Gaussian fallback");
                    }
                }
                // ape f32 [128, 512] → [128, hd] corner
                if let Some((raw, _)) = ds.tensor_bytes("layers.3.attn.compressor.ape") {
                    let vals = bytes_to_f32(&raw, "F32", "ape");
                    let mut corner = vec![0f32; 128 * hd];
                    for r in 0..128 {
                        corner[r * hd..(r + 1) * hd].copy_from_slice(&vals[r * 512..r * 512 + hd]);
                    }
                    src.real.insert("comp_de_ape".to_string(), corner);
                    note(&mut src.report, "comp ape (dense)", "REAL V4 f32 (corner)");
                } else {
                    note(&mut src.report, "comp ape (dense)", "FAIL → Gaussian fallback");
                }
            }
            // indexer (layer 2): wq_b fp8 [8192, 1024] → [idx_nh*il, ql] corner;
            // weights_proj bf16 [64, 4096] → [idx_nh, d]; compressor bf16 → [2*il, d]
            {
                let mut pool = Vec::new();
                if fetch_fp8_corner(&mut ds, "layers.2.attn.indexer.wq_b.weight", cfg.index_n_heads * il, 1024, cfg.index_n_heads * il, ql, "idx_wq_b", &mut pool, &mut src).is_none() {
                    note(&mut src.report, "idx_wq_b", "FAIL → Gaussian fallback");
                }
                if fetch_bf16_corner(&mut ds, "layers.2.attn.indexer.weights_proj.weight", cfg.index_n_heads, 4096, cfg.index_n_heads, d, "idx_weights_proj", &mut src).is_none() {
                    note(&mut src.report, "idx_weights_proj", "FAIL → Gaussian fallback");
                }
                for w in ["wkv", "wgate"] {
                    let logical = format!("idx_comp_{}", w);
                    if fetch_bf16_corner(&mut ds, &format!("layers.2.attn.indexer.compressor.{}.weight", w), 2 * il, 4096, 2 * il, d, &logical, &mut src).is_none() {
                        note(&mut src.report, &logical, "FAIL → Gaussian fallback");
                    }
                }
                // indexer ape f32 [4, 256] = exact micro shape
                if let Some((raw, _)) = ds.tensor_bytes("layers.2.attn.indexer.compressor.ape") {
                    let vals = bytes_to_f32(&raw, "F32", "idx ape");
                    src.real.insert("idx_comp_ape".to_string(), vals[..4 * 2 * il].to_vec());
                    note(&mut src.report, "idx comp ape", "REAL V4 f32 (full)");
                } else {
                    note(&mut src.report, "idx comp ape", "FAIL → Gaussian fallback");
                }
            }
            // attn projections (layer 2, fp8): leading rows, real corners + pool
            {
                let mut pool = Vec::new();
                let jobs: [(&str, usize, usize, usize, usize, &str); 5] = [
                    // (name, fetch_rows, cols, r0, c0, logical)
                    ("layers.2.attn.wq_a.weight", ql, 4096, ql, d, "wq_a"),
                    ("layers.2.attn.wq_b.weight", cfg.n_heads * hd, 1024, cfg.n_heads * hd, ql, "wq_b"),
                    ("layers.2.attn.wkv.weight", hd, 4096, hd, d, "wkv"),
                    ("layers.2.attn.wo_a.weight", cfg.o_groups * cfg.o_lora_rank, 4096, cfg.o_groups * cfg.o_lora_rank, hd, "wo_a"),
                    ("layers.2.attn.wo_b.weight", d, 8192, d, cfg.o_groups * cfg.o_lora_rank, "wo_b"),
                ];
                for (name, fr, cols, r0, c0, logical) in jobs {
                    if fetch_fp8_corner(&mut ds, name, fr, cols, r0, c0, logical, &mut pool, &mut src).is_none() {
                        note(&mut src.report, logical, "FAIL → Gaussian/pool fallback");
                    }
                }
                src.pool_attn = pool;
            }
            // real fp4 experts → pools (layer 2 expert 0, layer 11 expert 0, layer 33 expert 0)
            for l in [2usize, 11, 33] {
                for w in ["w1", "w2", "w3"] {
                    let pfx = format!("layers.{}.ffn.experts.0.{}", l, w);
                    let pk = ds.tensor_range(&format!("{}.weight", pfx), 0, (if w == "w2" { 512 * 1024 } else { 128 * 2048 }) as u64);
                    let sc = ds.tensor_range(&format!("{}.scale", pfx), 0, (if w == "w2" { 512 * 64 } else { 128 * 128 }) as u64);
                    if let (Some((pb, _)), Some((sb, _))) = (pk, sc) {
                        let (r, c) = if w == "w2" { (512, 2048) } else { (128, 4096) };
                        let vals = mxfp4::dequant(&pb, &sb, r, c);
                        let pool = match w {
                            "w1" => &mut src.pool_w1,
                            "w2" => &mut src.pool_w2,
                            _ => &mut src.pool_w3,
                        };
                        pool.extend_from_slice(&vals);
                        note(&mut src.report, &format!("expert L{}-E0 {} [{}×{}]", l, w, r, c), "REAL V4 fp4 → pool");
                    } else {
                        note(&mut src.report, &format!("expert L{}-E0 {}", l, w), "FAIL");
                    }
                }
            }
            // embed / head: REAL_ROWS leading rows (contiguous range), cols 0..d
            for (name, logical) in [("embed.weight", "embed"), ("head.weight", "head")] {
                match ds.tensor_range(name, 0, (REAL_ROWS * 4096 * 2) as u64) {
                    Some((raw, _)) => {
                        let vals = safetensors::bf16_slice_to_f32(&raw);
                        let mut corner = vec![0f32; REAL_ROWS * d];
                        for r in 0..REAL_ROWS {
                            corner[r * d..(r + 1) * d].copy_from_slice(&vals[r * 4096..r * 4096 + d]);
                        }
                        src.real.insert(logical.to_string(), corner);
                        let pool = if logical == "embed" { &mut src.pool_embed } else { &mut src.pool_head };
                        pool.extend_from_slice(&vals);
                        note(&mut src.report, &format!("{} ({} real rows × cols 0..{})", name, REAL_ROWS, d), "REAL V4 bf16");
                    }
                    None => note(&mut src.report, name, "FAIL → Gaussian fallback"),
                }
            }
            // tokenizer (saved next to the bin for `run`/`chat`)
            match http::fetch(&format!("{}tokenizer.json", DS_BASE)) {
                Some(t) => {
                    std::fs::write("microdeepseek.tokenizer.json", &t).ok();
                    note(&mut src.report, "tokenizer.json", "REAL V4 (saved microdeepseek.tokenizer.json)");
                }
                None => note(&mut src.report, "tokenizer.json", "FAIL (run/chat will need --vocab)"),
            }
        }
        None => {
            note(&mut src.report, "V4 network", "INACCESSIBLE → full Gaussian fallback");
        }
    }
    if src.pool_w1.is_empty() {
        note(&mut src.report, "expert pools", "EMPTY → Gaussian fallback σ=0.02");
    }
    src
}

fn real_or(src: &Src, logical: &str, fallback: Vec<f32>) -> Vec<f32> {
    src.real.get(logical).cloned().unwrap_or(fallback)
}

// ── main build ──

pub fn run() {
    let t0 = std::time::Instant::now();
    let cfg = DsConfig::microdeepseek();
    let (d, hd, ql, il) = (cfg.d, cfg.head_dim, cfg.q_lora_rank, cfg.index_head_dim);
    let (nh, og, ol) = (cfg.n_heads, cfg.o_groups, cfg.o_lora_rank);
    let hc = cfg.hc_mult;
    let inter = cfg.moe_inter_dim;
    println!("microkimi build-ds - building {} (DeepSeek-V4 micro: d={}, {} layers)", OUT, d, cfg.n_layers);
    let src = gather_sources(&cfg);

    // ── MKIM0002 config block ──
    let ratios: Vec<String> = cfg.compress_ratios.iter().map(|r| r.to_string()).collect();
    let config_json = format!(
        "{{\"arch\":\"deepseek_v4\",\"vocab\":{},\"n_layers\":{},\"specials\":{{\"bos\":0,\"end_of_msg\":1}},\
\"ds\":{{\"n_layers\":{},\"hidden\":{},\"vocab\":{},\"n_heads\":{},\"head_dim\":{},\"qk_rope_head_dim\":{},\
\"q_lora_rank\":{},\"o_lora_rank\":{},\"o_groups\":{},\"sliding_window\":{},\"compress_ratios\":[{}],\
\"rope_theta\":{},\"compress_rope_theta\":{},\"index_n_heads\":{},\"index_head_dim\":{},\"index_topk\":{},\
\"n_routed_experts\":{},\"num_experts_per_tok\":{},\"moe_intermediate_size\":{},\"num_hash_layers\":{},\
\"routed_scaling_factor\":{},\"swiglu_limit\":{},\"rms_norm_eps\":{}}}}}",
        cfg.vocab, cfg.n_layers, cfg.n_layers, d, cfg.vocab, nh, hd, cfg.rope_head_dim,
        ql, ol, og, cfg.window_size, ratios.join(","),
        cfg.rope_theta, cfg.compress_rope_theta, cfg.index_n_heads, il, cfg.index_topk,
        cfg.n_routed_experts, cfg.n_activated_experts, inter, cfg.n_hash_layers,
        cfg.route_scale, cfg.swiglu_limit, cfg.norm_eps,
    );

    // ── tensor registration (order = write order) ──
    let mut w = BinWriter::new();
    w.add("embed.weight", DTYPE_F32, vec![cfg.vocab as u32, d as u32]);
    w.add("head.weight", DTYPE_F32, vec![cfg.vocab as u32, d as u32]);
    w.add("norm.weight", DTYPE_F32, vec![d as u32]);
    w.add("hc_head_fn", DTYPE_F32, vec![hc as u32, (hc * d) as u32]);
    w.add("hc_head_base", DTYPE_F32, vec![hc as u32]);
    w.add("hc_head_scale", DTYPE_F32, vec![1]);
    for l in 0..cfg.n_layers {
        let p = format!("layers.{}.", l);
        let ratio = cfg.compress_ratio(l);
        let coff = if ratio == 4 { 2usize } else { 1 };
        w.add(&format!("{}attn_norm.weight", p), DTYPE_F32, vec![d as u32]);
        w.add(&format!("{}ffn_norm.weight", p), DTYPE_F32, vec![d as u32]);
        for kind in ["attn", "ffn"] {
            w.add(&format!("{}hc_{}_fn", p, kind), DTYPE_F32, vec![24, (hc * d) as u32]);
            w.add(&format!("{}hc_{}_base", p, kind), DTYPE_F32, vec![24]);
            w.add(&format!("{}hc_{}_scale", p, kind), DTYPE_F32, vec![3]);
        }
        w.add(&format!("{}attn.wq_a.weight", p), DTYPE_F32, vec![ql as u32, d as u32]);
        w.add(&format!("{}attn.q_norm.weight", p), DTYPE_F32, vec![ql as u32]);
        w.add(&format!("{}attn.wq_b.weight", p), DTYPE_F32, vec![(nh * hd) as u32, ql as u32]);
        w.add(&format!("{}attn.wkv.weight", p), DTYPE_F32, vec![hd as u32, d as u32]);
        w.add(&format!("{}attn.kv_norm.weight", p), DTYPE_F32, vec![hd as u32]);
        w.add(&format!("{}attn.wo_a.weight", p), DTYPE_F32, vec![(og * ol) as u32, (nh * hd / og) as u32]);
        w.add(&format!("{}attn.wo_b.weight", p), DTYPE_F32, vec![d as u32, (og * ol) as u32]);
        w.add(&format!("{}attn.attn_sink", p), DTYPE_F32, vec![nh as u32]);
        if ratio > 0 {
            w.add(&format!("{}attn.compressor.wkv.weight", p), DTYPE_F32, vec![(coff * hd) as u32, d as u32]);
            w.add(&format!("{}attn.compressor.wgate.weight", p), DTYPE_F32, vec![(coff * hd) as u32, d as u32]);
            w.add(&format!("{}attn.compressor.ape", p), DTYPE_F32, vec![ratio as u32, (coff * hd) as u32]);
            w.add(&format!("{}attn.compressor.norm.weight", p), DTYPE_F32, vec![hd as u32]);
        }
        if ratio == 4 {
            w.add(&format!("{}attn.indexer.wq_b.weight", p), DTYPE_F32, vec![(cfg.index_n_heads * il) as u32, ql as u32]);
            w.add(&format!("{}attn.indexer.weights_proj.weight", p), DTYPE_F32, vec![cfg.index_n_heads as u32, d as u32]);
            w.add(&format!("{}attn.indexer.compressor.wkv.weight", p), DTYPE_F32, vec![(2 * il) as u32, d as u32]);
            w.add(&format!("{}attn.indexer.compressor.wgate.weight", p), DTYPE_F32, vec![(2 * il) as u32, d as u32]);
            w.add(&format!("{}attn.indexer.compressor.ape", p), DTYPE_F32, vec![4, (2 * il) as u32]);
            w.add(&format!("{}attn.indexer.compressor.norm.weight", p), DTYPE_F32, vec![il as u32]);
        }
        w.add(&format!("{}ffn.gate.weight", p), DTYPE_F32, vec![cfg.n_routed_experts as u32, d as u32]);
        if l < cfg.n_hash_layers {
            w.add(&format!("{}ffn.gate.tid2eid", p), DTYPE_I32, vec![cfg.vocab as u32, cfg.n_activated_experts as u32]);
        } else {
            w.add(&format!("{}ffn.gate.bias", p), DTYPE_F32, vec![cfg.n_routed_experts as u32]);
        }
        for (wn, r, c) in [("w1", inter, d), ("w2", d, inter), ("w3", inter, d)] {
            w.add(&format!("{}ffn.shared_experts.{}.weight", p, wn), DTYPE_F32, vec![r as u32, c as u32]);
        }
    }
    for l in 0..cfg.n_layers {
        for e in 0..cfg.n_routed_experts {
            let p = format!("layers.{}.ffn.experts.{}.", l, e);
            w.add(&format!("{}w1", p), DTYPE_MXFP4, vec![inter as u32, d as u32]);
            w.add(&format!("{}w2", p), DTYPE_MXFP4, vec![d as u32, inter as u32]);
            w.add(&format!("{}w3", p), DTYPE_MXFP4, vec![inter as u32, d as u32]);
        }
    }
    println!("{} tensors in the directory", w.names_order.len());

    let mut f = std::fs::File::create(OUT).unwrap();
    let offsets = w.write_header_v2(&mut f, &config_json);
    let off_map: HashMap<&str, u64> = w
        .names_order
        .iter()
        .zip(&offsets)
        .map(|((n, _, _), &o)| (n.as_str(), o))
        .collect();

    // ── f32 tensors ──
    println!("── writing f32 tensors (spine + embeddings) ──");
    let put = |f: &mut std::fs::File, w: &BinWriter, name: &str, vals: &[f32]| {
        w.write_blob_at(f, off_map[name], &weights::f32_to_bytes(vals));
    };

    // embed/head: real leading rows + pool-sampled tail
    for (name, logical, pool) in [("embed.weight", "embed", &src.pool_embed), ("head.weight", "head", &src.pool_head)] {
        let mut vals = match src.real.get(logical) {
            Some(v) => v.clone(),
            None => fill_gauss(&format!("{}_real", logical), REAL_ROWS * d, SIGMA),
        };
        let mut tail = fill_pool_or_gauss(&format!("{}_tail", logical), (cfg.vocab - REAL_ROWS) * d, pool);
        vals.append(&mut tail);
        put(&mut f, &w, name, &vals);
        note_print(&format!("{}: {} REAL rows + pool tail", name, REAL_ROWS.min(cfg.vocab)));
    }
    put(&mut f, &w, "norm.weight", &real_or(&src, "norm_final", vec![1.0; d]));
    put(&mut f, &w, "hc_head_fn", &real_or(&src, "hc_head_fn", fill_gauss("hc_head_fn_fb", hc * hc * d, SIGMA)));
    put(&mut f, &w, "hc_head_base", &real_or(&src, "hc_head_base", vec![0.0; hc]));
    put(&mut f, &w, "hc_head_scale", &real_or(&src, "hc_head_scale", vec![1.0]));

    for l in 0..cfg.n_layers {
        let p = format!("layers.{}.", l);
        let ratio = cfg.compress_ratio(l);
        let ov = ratio == 4;
        put(&mut f, &w, &format!("{}attn_norm.weight", p), &real_or(&src, "attn_norm", vec![1.0; d]));
        put(&mut f, &w, &format!("{}ffn_norm.weight", p), &real_or(&src, "ffn_norm", vec![1.0; d]));
        for kind in ["attn", "ffn"] {
            put(&mut f, &w, &format!("{}hc_{}_fn", p, kind), &real_or(&src, &format!("hc_{}_fn", kind), fill_gauss(&format!("hcfn{}{}", kind, l), 24 * hc * d, SIGMA)));
            put(&mut f, &w, &format!("{}hc_{}_base", p, kind), &real_or(&src, &format!("hc_{}_base", kind), vec![0.0; 24]));
            put(&mut f, &w, &format!("{}hc_{}_scale", p, kind), &real_or(&src, &format!("hc_{}_scale", kind), vec![1.0; 3]));
        }
        let apool = &src.pool_attn;
        put(&mut f, &w, &format!("{}attn.wq_a.weight", p), &real_or(&src, "wq_a", fill_pool_or_gauss(&format!("{}wqa", p), ql * d, apool)));
        put(&mut f, &w, &format!("{}attn.q_norm.weight", p), &real_or(&src, "q_norm", vec![1.0; ql]));
        put(&mut f, &w, &format!("{}attn.wq_b.weight", p), &real_or(&src, "wq_b", fill_pool_or_gauss(&format!("{}wqb", p), nh * hd * ql, apool)));
        put(&mut f, &w, &format!("{}attn.wkv.weight", p), &real_or(&src, "wkv", fill_pool_or_gauss(&format!("{}wkv", p), hd * d, apool)));
        put(&mut f, &w, &format!("{}attn.kv_norm.weight", p), &real_or(&src, "kv_norm", vec![1.0; hd]));
        put(&mut f, &w, &format!("{}attn.wo_a.weight", p), &real_or(&src, "wo_a", fill_pool_or_gauss(&format!("{}woa", p), og * ol * (nh * hd / og), apool)));
        put(&mut f, &w, &format!("{}attn.wo_b.weight", p), &real_or(&src, "wo_b", fill_pool_or_gauss(&format!("{}wob", p), d * og * ol, apool)));
        put(&mut f, &w, &format!("{}attn.attn_sink", p), &real_or(&src, "attn_sink", vec![0.0; nh]));
        if ratio > 0 {
            let tag = if ov { "comp_ov" } else { "comp_de" };
            let ntag = if ov { "comp_norm_ov" } else { "comp_norm_de" };
            let cd = coff_dim(ratio) * hd;
            put(&mut f, &w, &format!("{}attn.compressor.wkv.weight", p), &real_or(&src, &format!("{}_wkv", tag), fill_pool_or_gauss(&format!("{}cwkv", p), cd * d, apool)));
            put(&mut f, &w, &format!("{}attn.compressor.wgate.weight", p), &real_or(&src, &format!("{}_wgate", tag), fill_pool_or_gauss(&format!("{}cwg", p), cd * d, apool)));
            put(&mut f, &w, &format!("{}attn.compressor.ape", p), &real_or(&src, &format!("{}_ape", tag), fill_gauss(&format!("{}cape", p), ratio as usize * cd, SIGMA)));
            put(&mut f, &w, &format!("{}attn.compressor.norm.weight", p), &real_or(&src, ntag, vec![1.0; hd]));
        }
        if ov {
            put(&mut f, &w, &format!("{}attn.indexer.wq_b.weight", p), &real_or(&src, "idx_wq_b", fill_pool_or_gauss(&format!("{}iwqb", p), cfg.index_n_heads * il * ql, apool)));
            put(&mut f, &w, &format!("{}attn.indexer.weights_proj.weight", p), &real_or(&src, "idx_weights_proj", fill_pool_or_gauss(&format!("{}iwp", p), cfg.index_n_heads * d, apool)));
            put(&mut f, &w, &format!("{}attn.indexer.compressor.wkv.weight", p), &real_or(&src, "idx_comp_wkv", fill_pool_or_gauss(&format!("{}ickv", p), 2 * il * d, apool)));
            put(&mut f, &w, &format!("{}attn.indexer.compressor.wgate.weight", p), &real_or(&src, "idx_comp_wgate", fill_pool_or_gauss(&format!("{}icwg", p), 2 * il * d, apool)));
            put(&mut f, &w, &format!("{}attn.indexer.compressor.ape", p), &real_or(&src, "idx_comp_ape", fill_gauss(&format!("{}icape", p), 4 * 2 * il, SIGMA)));
            put(&mut f, &w, &format!("{}attn.indexer.compressor.norm.weight", p), &real_or(&src, "idx_comp_norm", vec![1.0; il]));
        }
        put(&mut f, &w, &format!("{}ffn.gate.weight", p), &real_or(&src, "gate_weight", fill_pool_or_gauss(&format!("{}gate", p), cfg.n_routed_experts * d, &src.pool_w1)));
        if l < cfg.n_hash_layers {
            let table = match &src.tid2eid[l] {
                Some(t) => t.clone(),
                None => {
                    let mut rng = Rng::for_tensor(&format!("tid2eid_fb{}", l));
                    (0..cfg.vocab * cfg.n_activated_experts)
                        .map(|_| (rng.next_u64() % cfg.n_routed_experts as u64) as i32)
                        .collect()
                }
            };
            let bytes: Vec<u8> = table.iter().flat_map(|v| v.to_le_bytes()).collect();
            w.write_blob_at(&mut f, off_map[format!("{}ffn.gate.tid2eid", p).as_str()], &bytes);
        } else {
            put(&mut f, &w, &format!("{}ffn.gate.bias", p), &real_or(&src, "gate_bias", vec![0.0; cfg.n_routed_experts]));
        }
        // shared expert (f32, sampled from the real expert pools)
        for (wn, r, c, pool) in [
            ("w1", inter, d, &src.pool_w1),
            ("w2", d, inter, &src.pool_w2),
            ("w3", inter, d, &src.pool_w3),
        ] {
            put(&mut f, &w, &format!("{}ffn.shared_experts.{}.weight", p, wn), &fill_pool_or_gauss(&format!("{}sh{}", p, wn), r * c, pool));
        }
        if l % 10 == 0 {
            println!("  spine layer {}/{} written ({:.0?})", l + 1, cfg.n_layers, t0.elapsed());
        }
    }

    // ── fp4 routed experts: parallel generation per layer (seeds per tensor) ──
    println!("── fp4 experts: {} layers × {} experts × 3 matrices ──", cfg.n_layers, cfg.n_routed_experts);
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
    let layers: Vec<usize> = (0..cfg.n_layers).collect();
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
                    for e in 0..cfg.n_routed_experts {
                        for (wn, dims, pool) in [
                            ("w1", (inter, d), &p1),
                            ("w2", (d, inter), &p2),
                            ("w3", (inter, d), &p3),
                        ] {
                            let name = format!("layers.{}.ffn.experts.{}.{}", l, e, wn);
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
    println!("══ microdeepseek-debug.bin: {:.2} GB in {:.0?} ══", size as f64 / 1e9, t0.elapsed());
    println!("weight provenance:");
    for line in &src.report {
        println!("{}", line);
    }
    println!("    {} fp4 experts                                          sampled i.i.d. from the real V4 pools (or Gaussian)", cfg.n_layers * cfg.n_routed_experts * 3);
    println!("    all layers reuse the same small real values (norms, hc, gate, compressors, attn projections)");
}

fn coff_dim(ratio: i32) -> usize {
    if ratio == 4 {
        2
    } else {
        1
    }
}

fn note_print(s: &str) {
    println!("    {}", s);
}
