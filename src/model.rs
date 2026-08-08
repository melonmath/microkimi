// microkimi forward pass: 93 layers, AttnRes block 12, KDA (69),
// MLA NoPE (24), latent MoE 896 experts top-16 + 2 shared (layers 1..92),
// dense MLP layer 0, SiTU everywhere, MXFP4 experts dequantized on the fly.
// All in f32, zero-copy: f32 tensors are read in slices directly from the
// mmap-like file (Vec<u8> + align_to), experts stay packed.
//
// Module map: this root keeps the weight structs (T, LayerW and friends), the
// cache types, the Model type (load, forward/prefill dispatch) and the unit
// tests. Domain code lives in submodules: ops (math kernels), seam (slice
// adapter), kda (linear recurrence), attention (MLA), moe (expert FFN),
// lens (diagnostics), generate (sampler + decode loop).

mod attention;
mod generate;
mod kda;
mod lens;
mod moe;
mod ops;
mod seam;
pub(crate) mod kda_chunk;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod pool;
pub mod rosa;
pub mod spec;

use crate::config::Config;
use crate::tokenizer::AnyTokenizer;
use crate::quant::weights::{BinFile, Entry};
use std::time::Instant;

// The public surface stays crate::model::*; submodules are private.
pub use ops::{attn_res, dot, gemm_batch, matvec, rmsnorm, situ, Q8Head};
#[allow(unused_imports)] // only metal.rs (macOS) calls it, through crate::model::
pub use ops::matvec_cpu;
pub(crate) use ops::matvec_packed_nt;
use ops::{attn_res_refs, q8head_enabled, sigmoid, silu};
#[cfg(test)]
use ops::{dot8t, dot_scalar};
pub(crate) use attention::{
    hadamard64, mla_attn_flash, mla_attn_flash_mqa, mla_attn_flash_q8, mla_attn_flash_q8_mqa,
    mla_attn_ref, mla_attn_ref_q8,
};
use attention::{mla_forward, mla_prefill};
#[cfg(test)]
pub(crate) use kda::kda_recur_step_pub;
use kda::{kda_forward, kda_prefill};
use moe::{dense_forward, dense_prefill, moe_forward, moe_lookahead, moe_prefill};
use seam::{seam_apply, seam_load, SeamW};
pub use lens::{
    logit_lens_print_maybe, set_dump_hidden, set_logit_lens, ParityDump, RoutingDebug, DUMP_LAYERS,
    PARITY, ROUTER_LAYERS, ROUTING,
};
use lens::{dump_hidden_on, dump_hidden_print, logit_lens_compute, logit_lens_on, parity_rec, vec_rms};
pub use generate::{run_turn, run_turn_core, run_turn_core_batch, run_turn_resume, Sampler};
pub(crate) use generate::{apply_dry, py_repr, top_k_probs};

// ── default microkimi dims - used ONLY by build.rs (micro builder)
// and tests (selftest/parity are micro-specific). The inference engine
// is entirely driven by Config (config.rs, MKIM0002 block or microkimi default).
pub const D: usize = 512;

pub const fn is_mla(l: usize) -> bool {
    l % 4 == 3 || l == 92
}
pub const fn is_moe(l: usize) -> bool {
    l >= 1
}

pub fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

// ── zero-copy bytes → f32 conversion (64-byte alignment guaranteed by the format) ──
fn as_f32(bytes: &[u8]) -> &[f32] {
    let (pre, mid, post) = unsafe { bytes.align_to::<f32>() };
    assert!(pre.is_empty() && post.is_empty(), "unexpected f32 alignment");
    mid
}

/// (bytes actually read from disk, major page faults) from /proc, Linux only.
/// Used to show the mmap demand-paging cost of a prefill.
fn io_stats() -> Option<(u64, u64)> {
    let io = std::fs::read_to_string("/proc/self/io").ok()?;
    let read_bytes = io.lines().find_map(|l| l.strip_prefix("read_bytes:"))?.trim().parse().ok()?;
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    // after the ") " of comm, fields are numbered from 3: majflt is field 12
    let majflt = stat.rsplit_once(") ")?.1.split_whitespace().nth(9)?.parse().ok()?;
    Some((read_bytes, majflt))
}

// ── math kernels ──
//
// Bit-exactness contract of dot(): every path (scalar fallback, NEON, AVX2)
// computes the SAME IEEE operations in the SAME order:
//   - 8 parallel accumulators; element j of each 8-wide chunk goes to acc[j]
//     (mul, then add - NEVER a fused multiply-add: FMA skips the intermediate
//     rounding and would drift from the scalar path);
//   - fixed reduction: pairs p01=(a0+a1), p23=(a2+a3), p45=(a4+a5),
//     p67=(a6+a7), then ((p01 + p23) + p45) + p67, left-associative;
//   - the remainder (< 8 elements) is accumulated sequentially into s.
// The SIMD kernels keep the accumulators in vector lanes and replay this
// exact reduction, so they are bit-identical to dot_scalar BY CONSTRUCTION.

// ── --gpu flag: routes large matvecs to Metal on macOS ──
//
// GPU_ENABLED is set once from main.rs when --gpu is passed. matvec() is the
// single entry point used by every projection in the engine (KDA/MLA/MoE/
// dense/router/lm_head); when the flag is on and the matvec is large enough,
// it is dispatched to Metal (see metal.rs). Without the flag, behavior is
// bit-identical to the pure-CPU path.
//
// Threshold rationale (64k MACs): encoding + dispatching a Metal command
// buffer costs ~50-100 µs of latency; below ~64k multiply-accumulates the
// CPU finishes faster than the round trip, so small matvecs stay on the CPU.
// Tune later with real measurements on device.
pub static GPU_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(target_os = "macos")]
// GPU threshold measured on Apple M5: each Metal dispatch costs ~0.25 ms of
// sync latency, and the micro model runs ~1 200 matvecs per token - only
// genuinely large matvecs (lm_head: 163840×512 = 84M MACs) are a net win on
// GPU. Smaller ones stay on the CPU thread pool.
pub const GPU_MIN_ELEMS: usize = 2 * 1024 * 1024;

pub fn set_gpu(on: bool) {
    GPU_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn gpu_on() -> bool {
    GPU_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Copy)]
struct T {
    off: usize,
    len: usize, // in f32
}

impl T {
    fn from(e: &Entry) -> T {
        let len: usize = e.dims.iter().map(|&d| d as usize).product();
        T { off: e.offset as usize, len }
    }
}

// ── seam adapter (embedded by nano/apply_lora_bin.py --write-seam) ──
//
// Exact deployment of the trained residual-stream correction h' = h + B A h
// (it cannot be folded into the existing weights exactly - see the
// apply_lora_bin.py docstring). The .bin carries the fp32 tensors "seam.A"
// [rank, d] and "seam.B" [d, rank] plus the config key "seam_after" (0-based
// layer index); the engine applies h += (h @ A^T) @ B^T right after that
// layer, in prefill and decode alike. A and B are tiny (2 * rank * d floats)
// and are read through the spine mapping like any other plain tensor.

struct KdaW {
    q_proj: T,
    k_proj: T,
    v_proj: T,
    q_conv: T,
    k_conv: T,
    v_conv: T,
    f_a: T,
    f_b: T,
    a_log: T,
    dt_bias: T,
    b_proj: T,
    g_proj: T,
    o_norm: T,
    o_proj: T,
}

struct MlaW {
    q_a: T,
    q_a_ln: T,
    q_b: T,
    kv_a: T,
    kv_a_ln: T,
    kv_b: T,
    g_proj: T,
    o_proj: T,
}

enum AttnW {
    Kda(KdaW),
    Mla(MlaW),
}

struct MoeW {
    gate_w: T,
    gate_b: T,
    routed_down: T,
    routed_up: T,
    routed_norm: T,
    shared_gate: T,
    shared_up: T,
    shared_down: T,
    experts: Vec<[u64; 3]>, // offsets of the mxfp4/vq1 blobs [w1, w2, w3] per expert
    experts_vq: Vec<bool>,  // true = cold expert stored as VQ1 (DTYPE_VQ1 indices)
    vq_cb: Vec<f32>,        // global VQ codebook [256*16] (empty when no VQ1 tensors)
}

struct DenseW {
    gate: T,
    up: T,
    down: T,
}

enum FfnW {
    Dense(DenseW),
    Moe(MoeW),
}

struct LayerW {
    input_ln: T,
    post_ln: T,
    sa_res_w: Vec<f32>,  // pre-combined norm·proj [512]
    mlp_res_w: Vec<f32>, // pre-combined norm·proj [512]
    attn: AttnW,
    ffn: FfnW,
}

// ── caches ──

// pub(crate): read/rewritten by mkmem.rs (.mkmem state snapshots)
#[derive(Clone)]
pub(crate) struct KdaCache {
    pub conv_q: Vec<f32>, // 3 × 512 (raw pre-conv)
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    pub s: Vec<f32>, // 4 × 128 × 128
}

#[derive(Clone)]
pub(crate) struct MlaCache {
    /// f32 layout (MICROKIMI_NO_KVQ8=1): k = pos × H×(nope+rope), v = pos × H×v
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    // q8_0 layout (default): the latent (nope) part of K and all of V are
    // q8_0 (i8 + one f32 scale per 32); the rope part of K stays f32
    // (position-sensitive) and is stored ONCE per position - the f32 layout
    // duplicates it per head. With MICROKIMI_KV_HADAMARD=1 the latent K and
    // V rows are kept in the 64-point Hadamard domain (see hadamard64).
    pub kq: Vec<i8>,  // pos × H × nope (Hadamard-rotated when `had`)
    pub ks: Vec<f32>, // pos × H × nope/32 scales
    pub kr: Vec<f32>, // pos × rope, never quantized
    pub vq: Vec<i8>,  // pos × H × v (Hadamard-rotated when `had`)
    pub vs: Vec<f32>, // pos × H × v/32 scales
    pub q8: bool,
    pub had: bool,
}

#[derive(Clone)]
pub(crate) enum Cache {
    Kda(KdaCache),
    Mla(MlaCache),
}

// ── profiler ──

#[derive(Default)]
pub struct Prof {
    pub t_norm_res: f64,
    pub t_kda_proj: f64,
    pub t_kda_conv: f64,
    pub t_kda_recur: f64,
    pub t_mla: f64,
    pub t_router: f64,
    pub t_experts: f64,
    pub t_lm_head: f64,
}

impl Prof {
    #[allow(dead_code)]
    pub fn print(&self) {
        self.print_cfg(&Config::microkimi());
    }

    pub fn print_cfg(&self, cfg: &Config) {
        let tot = self.t_norm_res + self.t_kda_proj + self.t_kda_conv + self.t_kda_recur + self.t_mla + self.t_router + self.t_experts + self.t_lm_head;
        if tot == 0.0 {
            return;
        }
        let lm_label = format!("lm_head ({} x {})", cfg.d, cfg.vocab);
        let router_label = format!("MoE router ({})", cfg.n_experts);
        let mla_label = format!(
            "MLA attention (H={}, {}/head, d={})",
            cfg.mla_heads, cfg.mla_v, cfg.d
        );
        let rows = [
            ("RMSNorm + AttnRes".to_string(), self.t_norm_res),
            ("KDA projections (qkv/f/g/o)".to_string(), self.t_kda_proj),
            ("causal KDA conv1d".to_string(), self.t_kda_conv),
            ("KDA recurrence (state S)".to_string(), self.t_kda_recur),
            (mla_label, self.t_mla),
            (router_label, self.t_router),
            ("MoE experts + shared + dense".to_string(), self.t_experts),
            (lm_label, self.t_lm_head),
        ];
        println!("Compute time breakdown (cumulative over processed tokens):");
        for (name, t) in rows {
            println!("  {:<32} {:6.1}%  ({:.2} s)", name, t / tot * 100.0, t);
        }
    }
}

// ── model ──

pub struct Model {
    pub cfg: Config,
    bin: BinFile,
    embed: T,
    lm_head: T,
    /// q8_0 runtime copy of lm_head (see the Q8Head section above): None when
    /// MICROKIMI_Q8HEAD=0, when --gpu serves the large matvecs, or when the
    /// dims do not divide into 32-wide blocks.
    lm_head_q8: Option<Q8Head>,
    norm_f: T,
    out_res_w: Vec<f32>,
    layers: Vec<LayerW>,
    pub(crate) caches: Vec<Cache>, // pub(crate): saved/restored by mkmem.rs
    pub last_logits: Vec<f32>,     // logits of the last forward (source for mkmem --save)
    pub prof: Prof,
    /// Embedded seam adapter (seam.A / seam.B + config seam_after), applied
    /// to the residual stream right after layer seam.after (seam_load).
    seam: Option<SeamW>,
    /// --stream: RAM LRU of packed expert bytes over the disk/HTTP tiers
    /// (stream.rs). None = historical full-load path, byte-identical behavior.
    stream: Option<crate::stream::ExpertCache>,
}

impl Model {
    fn t<'a>(data: &'a [u8], t: &T) -> &'a [f32] {
        as_f32(&data[t.off..t.off + t.len * 4])
    }

    /// Final logits projection: the q8_0 copy of lm_head when it was built at
    /// load (default), the exact f32 matvec otherwise (MICROKIMI_Q8HEAD=0).
    /// The q8 path is not bit-identical to f32 (q8 rounding, see q8.rs).
    fn logits_project(data: &[u8], lm_head: &T, q8: Option<&Q8Head>, cfg: &Config, x: &[f32], out: &mut [f32]) {
        match q8 {
            Some(h) => h.matvec(x, out),
            None => matvec(Self::t(data, lm_head), cfg.vocab, cfg.d, x, out),
        }
    }

    pub fn load(path: &str) -> Self {
        Self::from_bin(BinFile::open(path), None)
    }

    /// Streaming load (--stream): the spine is loaded compacted (expert MXFP4
    /// blobs excluded), experts are served on demand by the three-tier cache
    /// (RAM LRU of `ram_mb` MB -> the .bin on disk). Same weights, same
    /// dequant, same matvec: the output is bit-identical to `load`.
    /// `fallback` (--stream-fallback): the VQ1 expert shadows sidecar
    /// (<path>.shadows, shadow.rs) is loaded resident in RAM and the stream
    /// engine serves it on expert cache misses while refilling full
    /// precision in the background - a DEGRADED latency mode, NOT
    /// bit-identical, off by default.
    pub fn load_streaming(path: &str, ram_mb: usize, fallback: bool) -> Self {
        let bin = BinFile::open_spine(path);
        let shadows = if fallback {
            let cfg = &bin.config;
            let moe_layers: Vec<usize> = (0..cfg.n_layers).filter(|&l| cfg.is_moe(l)).collect();
            assert!(!moe_layers.is_empty(), "--stream-fallback on a MoE-less model is meaningless");
            crate::stream::set_fallback_shape(moe_layers.len() * cfg.top_k);
            Some(crate::stream::shadow::Shadows::load(
                &crate::stream::shadow::sidecar_path(path),
                &moe_layers,
                cfg.n_experts,
                cfg.routed_hidden * cfg.moe_inter / crate::quant::quant::VQ_DIM,
            ))
        } else {
            None
        };
        Self::from_bin(bin, Some(crate::stream::ExpertCache::local(path, ram_mb, shadows)))
    }

    fn from_bin(bin: BinFile, stream: Option<crate::stream::ExpertCache>) -> Self {
        let cfg = bin.config.clone();
        let get = |name: &str| -> T {
            T::from(bin.entries.get(name).unwrap_or_else(|| panic!("missing tensor: {}", name)))
        };
        let combine = |norm: &T, proj: &T| -> Vec<f32> {
            let n = Self::t(&bin.data, norm);
            let p = Self::t(&bin.data, proj);
            (0..cfg.d).map(|i| n[i] * p[i]).collect()
        };
        let embed = get("embed_tokens.weight");
        let lm_head = get("lm_head.weight");
        // q8_0 runtime copy of lm_head (default on): the .bin format is
        // unchanged, this is a load-time requantization of the f32 tensor.
        // Skipped under --gpu (Metal already serves the large matvecs).
        let lm_head_q8 = if q8head_enabled() && !gpu_on() && cfg.d % 32 == 0 && lm_head.len == cfg.vocab * cfg.d {
            Some(Q8Head::from_f32(Self::t(&bin.data, &lm_head), cfg.vocab, cfg.d))
        } else {
            None
        };
        let norm_f = get("norm.weight");
        let out_res_w = combine(&get("output_attn_res_norm.weight"), &get("output_attn_res_proj.weight"));
        let mut layers = Vec::with_capacity(cfg.n_layers);
        let mut caches = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = format!("layers.{}.", l);
            let input_ln = get(&format!("{}input_layernorm.weight", p));
            let post_ln = get(&format!("{}post_attention_layernorm.weight", p));
            let sa_res_w = combine(
                &get(&format!("{}self_attention_res_norm.weight", p)),
                &get(&format!("{}self_attention_res_proj.weight", p)),
            );
            let mlp_res_w = combine(
                &get(&format!("{}mlp_res_norm.weight", p)),
                &get(&format!("{}mlp_res_proj.weight", p)),
            );
            let attn = if cfg.is_mla(l) {
                caches.push(Cache::Mla(MlaCache::new()));
                AttnW::Mla(MlaW {
                    q_a: get(&format!("{}self_attn.q_a_proj.weight", p)),
                    q_a_ln: get(&format!("{}self_attn.q_a_layernorm.weight", p)),
                    q_b: get(&format!("{}self_attn.q_b_proj.weight", p)),
                    kv_a: get(&format!("{}self_attn.kv_a_proj_with_mqa.weight", p)),
                    kv_a_ln: get(&format!("{}self_attn.kv_a_layernorm.weight", p)),
                    kv_b: get(&format!("{}self_attn.kv_b_proj.weight", p)),
                    g_proj: get(&format!("{}self_attn.g_proj.weight", p)),
                    o_proj: get(&format!("{}self_attn.o_proj.weight", p)),
                })
            } else {
                caches.push(Cache::Kda(KdaCache {
                    conv_q: vec![0.0; 3 * cfg.kda_proj()],
                    conv_k: vec![0.0; 3 * cfg.kda_proj()],
                    conv_v: vec![0.0; 3 * cfg.kda_proj()],
                    s: vec![0.0; cfg.kda_heads * cfg.kda_dim * cfg.kda_dim],
                }));
                AttnW::Kda(KdaW {
                    q_proj: get(&format!("{}self_attn.q_proj.weight", p)),
                    k_proj: get(&format!("{}self_attn.k_proj.weight", p)),
                    v_proj: get(&format!("{}self_attn.v_proj.weight", p)),
                    q_conv: get(&format!("{}self_attn.q_conv1d.weight", p)),
                    k_conv: get(&format!("{}self_attn.k_conv1d.weight", p)),
                    v_conv: get(&format!("{}self_attn.v_conv1d.weight", p)),
                    f_a: get(&format!("{}self_attn.f_a_proj.weight", p)),
                    f_b: get(&format!("{}self_attn.f_b_proj.weight", p)),
                    a_log: get(&format!("{}self_attn.A_log", p)),
                    dt_bias: get(&format!("{}self_attn.dt_bias", p)),
                    b_proj: get(&format!("{}self_attn.b_proj.weight", p)),
                    g_proj: get(&format!("{}self_attn.g_proj.weight", p)),
                    o_norm: get(&format!("{}self_attn.o_norm.weight", p)),
                    o_proj: get(&format!("{}self_attn.o_proj.weight", p)),
                })
            };
            let ffn = if cfg.is_moe(l) {
                let pfx = format!("{}block_sparse_moe.experts.", p);
                let mut experts_vq = Vec::with_capacity(cfg.n_experts);
                let experts: Vec<[u64; 3]> = (0..cfg.n_experts)
                    .map(|e| {
                        ["w1", "w2", "w3"].map(|wn| {
                            let entry = bin
                                .entries
                                .get(&format!("{}{}.{}", pfx, e, wn))
                                .unwrap_or_else(|| panic!("missing expert: {}{}.{}", pfx, e, wn));
                            if wn == "w1" {
                                experts_vq.push(entry.dtype == crate::quant::weights::DTYPE_VQ1);
                            }
                            entry.offset
                        })
                    })
                    .collect();
                // one global codebook shared by every VQ1 tensor (16 KB, L1-resident)
                let vq_cb = if experts_vq.iter().any(|&v| v) {
                    let cb = bin.f32_vec("vq_codebook");
                    assert_eq!(cb.len(), crate::quant::quant::VQ_K * crate::quant::quant::VQ_DIM, "vq_codebook: bad dims");
                    cb
                } else {
                    Vec::new()
                };
                FfnW::Moe(MoeW {
                    gate_w: get(&format!("{}block_sparse_moe.gate.weight", p)),
                    gate_b: get(&format!("{}block_sparse_moe.gate.e_score_correction_bias", p)),
                    routed_down: get(&format!("{}block_sparse_moe.routed_expert_down_proj.weight", p)),
                    routed_up: get(&format!("{}block_sparse_moe.routed_expert_up_proj.weight", p)),
                    routed_norm: get(&format!("{}block_sparse_moe.routed_expert_norm.weight", p)),
                    shared_gate: get(&format!("{}block_sparse_moe.shared_experts.gate_proj.weight", p)),
                    shared_up: get(&format!("{}block_sparse_moe.shared_experts.up_proj.weight", p)),
                    shared_down: get(&format!("{}block_sparse_moe.shared_experts.down_proj.weight", p)),
                    experts,
                    experts_vq,
                    vq_cb,
                })
            } else {
                FfnW::Dense(DenseW {
                    gate: get(&format!("{}mlp.gate_proj.weight", p)),
                    up: get(&format!("{}mlp.up_proj.weight", p)),
                    down: get(&format!("{}mlp.down_proj.weight", p)),
                })
            };
            layers.push(LayerW { input_ln, post_ln, sa_res_w, mlp_res_w, attn, ffn });
        }
        let seam = seam_load(&cfg, &bin.entries);
        if let Some(s) = &seam {
            println!("seam: adapter rank {} after layer {}", s.rank, s.after);
        }
        Model { cfg, bin, embed, lm_head, lm_head_q8, norm_f, out_res_w, layers, caches, last_logits: Vec::new(), prof: Prof::default(), seam, stream }
    }

    /// Number of tokens already represented in the caches (from the first MLA
    /// layer; 0 for a KDA-only model). Only seeds the debug position counter:
    /// K3 has no positional encoding, so the math does not depend on it.
    pub fn cached_tokens(&self) -> usize {
        for c in &self.caches {
            if let Cache::Mla(m) = c {
                return m.positions(&self.cfg);
            }
        }
        0
    }

    pub fn reset_cache(&mut self) {
        for c in &mut self.caches {
            match c {
                Cache::Kda(k) => {
                    k.conv_q.iter_mut().for_each(|x| *x = 0.0);
                    k.conv_k.iter_mut().for_each(|x| *x = 0.0);
                    k.conv_v.iter_mut().for_each(|x| *x = 0.0);
                    k.s.iter_mut().for_each(|x| *x = 0.0);
                }
                Cache::Mla(m) => {
                    m.k.clear();
                    m.v.clear();
                    m.kq.clear();
                    m.ks.clear();
                    m.kr.clear();
                    m.vq.clear();
                    m.vs.clear();
                }
            }
        }
        // positions restart at 0: the routing history of the previous turn
        // (draft-aware prefetch) would alias the new positions
        crate::stream::route_hist_clear();
    }
}

// ── KDA ──

impl Model {
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // destructuring: independent per-field borrows (data immutable,
        // caches/prof mutable) - no raw pointers.
        let Self { cfg, bin, embed, lm_head, lm_head_q8, norm_f, out_res_w, layers, caches, last_logits, prof, seam, stream } = self;
        let cfg = &*cfg;
        let data = &bin.data[..];
        let embed = Self::t(data, embed);
        let mut hidden = embed[token as usize * cfg.d..(token as usize + 1) * cfg.d].to_vec();
        let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(8);
        let mut buf_res = vec![0f32; cfg.d];
        let mut x = vec![0f32; cfg.d];
        let hd_on = dump_hidden_on();
        let mut hd: Vec<(usize, &'static str, f64)> = Vec::new();
        let lens_on = logit_lens_on();
        let mut lens: Vec<(usize, &'static str, Vec<f32>)> = Vec::new();

        for l in 0..cfg.n_layers {
            let prefix: Option<Vec<f32>> = Some(hidden.clone());
            let layer = &layers[l];
            let tm = Instant::now();
            if !blocks.is_empty() {
                attn_res(cfg, prefix.as_ref().unwrap(), &blocks, &layer.sa_res_w, &mut buf_res);
                hidden.copy_from_slice(&buf_res);
            }
            let prefix: Option<Vec<f32>> = if l % cfg.attn_res_block == 0 {
                blocks.push(prefix.unwrap());
                None
            } else {
                prefix
            };
            rmsnorm(cfg, &hidden, Self::t(data, &layer.input_ln), &mut x);
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let attn_out = match (&layer.attn, &mut caches[l]) {
                (AttnW::Kda(w), Cache::Kda(c)) => {
                    let mut p = Prof::default();
                    let out = kda_forward(cfg, data, w, c, &x, &mut p);
                    prof.t_kda_proj += p.t_kda_proj;
                    prof.t_kda_conv += p.t_kda_conv;
                    prof.t_kda_recur += p.t_kda_recur;
                    out
                }
                (AttnW::Mla(w), Cache::Mla(c)) => {
                    let mut p = Prof::default();
                    let out = mla_forward(cfg, data, w, c, &x, &mut p);
                    prof.t_mla += p.t_mla;
                    out
                }
                _ => unreachable!(),
            };
            if l == 1 {
                let a = attn_out.clone();
                parity_rec(|d| {
                    d.l1_attn.insert(pos, a);
                });
            }
            let prefix2: Vec<f32> = match prefix {
                Some(p) => {
                    let mut p = p;
                    for j in 0..cfg.d {
                        p[j] += attn_out[j];
                    }
                    p
                }
                None => attn_out,
            };

            let tm = Instant::now();
            attn_res(cfg, &prefix2, &blocks, &layer.mlp_res_w, &mut buf_res);
            hidden.copy_from_slice(&buf_res);
            rmsnorm(cfg, &hidden, Self::t(data, &layer.post_ln), &mut x);
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let mlp_out = match &layer.ffn {
                FfnW::Dense(w) => {
                    let mut p = Prof::default();
                    let out = dense_forward(cfg, data, w, &x, &mut p);
                    prof.t_experts += p.t_experts;
                    out
                }
                FfnW::Moe(w) => {
                    let mut p = Prof::default();
                    // router-lookahead: while this layer's experts compute,
                    // predict the NEXT MoE layer's experts with its own
                    // router on the current MoE input x and prefetch them
                    if let Some(cache) = stream.as_ref() {
                        let n = crate::stream::predict_n();
                        if n > 0 && crate::stream::lookahead_on() {
                            let next = (l + 1..cfg.n_layers).find_map(|l2| match &layers[l2].ffn {
                                FfnW::Moe(w2) => Some((l2, w2)),
                                _ => None,
                            });
                            if let Some((l2, w2)) = next {
                                let tml = Instant::now();
                                moe_lookahead(cfg, data, w2, l2, &x, n, cache);
                                p.t_router += tml.elapsed().as_secs_f64();
                            }
                        }
                    }
                    let out = moe_forward(cfg, data, w, &x, &mut p, l, pos, stream.as_ref());
                    prof.t_router += p.t_router;
                    prof.t_experts += p.t_experts;
                    out
                }
            };
            for j in 0..cfg.d {
                hidden[j] = prefix2[j] + mlp_out[j];
            }
            // seam adapter: h += (h @ A^T) @ B^T right after layer seam.after
            // (the residual stream the next layer, the attn_res blocks and the
            // final norm all read)
            if let Some(s) = seam {
                if l == s.after {
                    seam_apply(data, s, cfg.d, &mut hidden);
                }
            }
            if hd_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                hd.push((l, kind, vec_rms(&hidden)));
            }
            if lens_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                lens.push((l, kind, hidden.clone()));
            }
            if DUMP_LAYERS.contains(&l) {
                let h = hidden.clone();
                parity_rec(|d| {
                    d.hiddens.insert((pos, l), h);
                });
            }
        }

        let tm = Instant::now();
        attn_res(cfg, &hidden, &blocks, &out_res_w, &mut buf_res);
        hidden.copy_from_slice(&buf_res);
        let mut xf = vec![0f32; cfg.d];
        rmsnorm(cfg, &hidden, Self::t(data, &norm_f), &mut xf);
        prof.t_norm_res += tm.elapsed().as_secs_f64();
        let tm = Instant::now();
        let mut logits = vec![0f32; cfg.vocab];
        Self::logits_project(data, &lm_head, lm_head_q8.as_ref(), cfg, &xf, &mut logits);
        prof.t_lm_head += tm.elapsed().as_secs_f64();
        if hd_on {
            dump_hidden_print(&hd, vec_rms(&hidden), &logits);
        }
        if lens_on {
            logit_lens_compute(cfg, Self::t(data, &lm_head), Self::t(data, &norm_f), &lens, &logits);
        }
        *last_logits = logits.clone();
        logits
    }

    /// Batched prefill: ingests `ids` (absolute positions pos0..pos0+n) in
    /// one pass and returns the logits of the last position. Every layer is
    /// applied to all positions at once: projections/router/experts run as
    /// gemm_batch over the [n * d] hidden buffer, the KDA conv + recurrence
    /// stay sequential over positions (cheap elementwise work), MLA appends
    /// its n latent K/V rows in order and attends causally per query
    /// position. lm_head runs only on the last position (the only logits the
    /// generation loop consumes). Caches end in exactly the same state as n
    /// sequential forward calls, and every per-position computation keeps
    /// the same accumulation order, so the result is bit-identical.
    pub fn prefill(&mut self, ids: &[u32], pos0: usize) -> Vec<f32> {
        self.prefill_impl(ids, pos0, false).pop().unwrap()
    }

    /// Batched prefill returning the logits of EVERY position, not just the
    /// last (consumed by the --spec verification pass). Same pass as prefill
    /// with lm_head applied per position: each logits vector is bit-identical
    /// to what a sequential forward of that prefix would produce.
    pub fn prefill_all(&mut self, ids: &[u32], pos0: usize) -> Vec<Vec<f32>> {
        self.prefill_impl(ids, pos0, true)
    }

    /// Draft-aware expert prefetch (--spec / --spec-rosa with --stream;
    /// MICROKIMI_DRAFTPREFETCH=0 disables): the batched verification pass
    /// that ingests `toks` (pending token + drafted proposals) will route
    /// them for real, so the experts it will pull are predictable before
    /// the pass starts. Both proposers draft tokens that ALREADY occurred
    /// in the committed context, and every ingested position had its router
    /// picks recorded (stream::route_record, real hidden states, not an
    /// embedding proxy - measured ~0% recall), so the routing of the source
    /// occurrence, `srcs[t]` = the context position toks[t] was lifted
    /// from, is the prediction: union of the recorded top-k sets over the
    /// source positions, background-fetched through the stream cache so the
    /// pass finds its experts in the RAM LRU. Same contract as the
    /// router-lookahead prefetch: only WHEN bytes land in the cache
    /// changes, never WHICH experts the model computes - mispredictions are
    /// harmless LRU fills and the greedy output stays bit-identical.
    /// No-op without --stream.
    pub fn draft_prefetch(&self, toks: &[u32], srcs: &[Option<usize>]) {
        let Some(cache) = &self.stream else { return };
        if toks.is_empty() || !crate::stream::draft_prefetch_on() {
            return;
        }
        let cfg = &self.cfg;
        let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
        let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
        let expert_vq_blob = cfg.routed_hidden * cfg.moe_inter / crate::quant::quant::VQ_DIM;
        let mut seen: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
        let mut jobs: Vec<(u32, u32, [u64; 3], usize)> = Vec::new();
        for src in srcs.iter().flatten() {
            let Some(layers) = crate::stream::route_lookup(*src as u32) else { continue };
            for (layer, experts) in layers {
                let FfnW::Moe(w) = &self.layers[layer as usize].ffn else { continue };
                for e in experts {
                    if seen.insert((layer, e)) {
                        let eblob = if w.experts_vq[e as usize] { expert_vq_blob } else { expert_blob };
                        jobs.push((layer, e, w.experts[e as usize], eblob));
                    }
                }
            }
        }
        cache.prefetch_draft(jobs);
    }

    fn prefill_impl(&mut self, ids: &[u32], pos0: usize, all_logits: bool) -> Vec<Vec<f32>> {
        if ids.len() == 1 {
            return vec![self.forward(ids[0], pos0)];
        }
        let Self { cfg, bin, embed, lm_head, lm_head_q8, norm_f, out_res_w, layers, caches, last_logits, prof, seam, stream } = self;
        let cfg = &*cfg;
        let data = &bin.data[..];
        let n = ids.len();
        let d = cfg.d;
        let embed = Self::t(data, embed);
        let mut hidden = vec![0f32; n * d];
        for (t, &id) in ids.iter().enumerate() {
            hidden[t * d..(t + 1) * d].copy_from_slice(&embed[id as usize * d..(id as usize + 1) * d]);
        }
        let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(8); // each [n * d]
        let mut buf_res = vec![0f32; n * d];
        let mut x = vec![0f32; n * d];
        let hd_on = dump_hidden_on();
        let mut hd: Vec<(usize, &'static str, f64)> = Vec::new();
        let lens_on = logit_lens_on();
        let mut lens: Vec<(usize, &'static str, Vec<f32>)> = Vec::new();

        for l in 0..cfg.n_layers {
            let layer = &layers[l];
            let tm = Instant::now();
            let mut prefix: Option<Vec<f32>> = Some(hidden.clone());
            if !blocks.is_empty() {
                for t in 0..n {
                    let brefs: Vec<&[f32]> = blocks.iter().map(|b| &b[t * d..(t + 1) * d]).collect();
                    attn_res_refs(cfg, &prefix.as_ref().unwrap()[t * d..(t + 1) * d], &brefs, &layer.sa_res_w, &mut buf_res[t * d..(t + 1) * d]);
                }
                hidden.copy_from_slice(&buf_res);
            }
            let prefix: Option<Vec<f32>> = if l % cfg.attn_res_block == 0 {
                blocks.push(prefix.take().unwrap());
                None
            } else {
                prefix
            };
            for t in 0..n {
                rmsnorm(cfg, &hidden[t * d..(t + 1) * d], Self::t(data, &layer.input_ln), &mut x[t * d..(t + 1) * d]);
            }
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let attn_out = match (&layer.attn, &mut caches[l]) {
                (AttnW::Kda(w), Cache::Kda(c)) => {
                    let mut p = Prof::default();
                    let out = kda_prefill(cfg, data, w, c, &x, n, &mut p);
                    prof.t_kda_proj += p.t_kda_proj;
                    prof.t_kda_conv += p.t_kda_conv;
                    prof.t_kda_recur += p.t_kda_recur;
                    out
                }
                (AttnW::Mla(w), Cache::Mla(c)) => {
                    let mut p = Prof::default();
                    let out = mla_prefill(cfg, data, w, c, &x, n, &mut p);
                    prof.t_mla += p.t_mla;
                    out
                }
                _ => unreachable!(),
            };
            if l == 1 {
                for t in 0..n {
                    let a = attn_out[t * d..(t + 1) * d].to_vec();
                    parity_rec(|d| {
                        d.l1_attn.insert(pos0 + t, a);
                    });
                }
            }
            let prefix2: Vec<f32> = match prefix {
                Some(mut p) => {
                    for j in 0..n * d {
                        p[j] += attn_out[j];
                    }
                    p
                }
                None => attn_out,
            };

            let tm = Instant::now();
            for t in 0..n {
                let brefs: Vec<&[f32]> = blocks.iter().map(|b| &b[t * d..(t + 1) * d]).collect();
                attn_res_refs(cfg, &prefix2[t * d..(t + 1) * d], &brefs, &layer.mlp_res_w, &mut buf_res[t * d..(t + 1) * d]);
            }
            hidden.copy_from_slice(&buf_res);
            for t in 0..n {
                rmsnorm(cfg, &hidden[t * d..(t + 1) * d], Self::t(data, &layer.post_ln), &mut x[t * d..(t + 1) * d]);
            }
            prof.t_norm_res += tm.elapsed().as_secs_f64();

            let mlp_out = match &layer.ffn {
                FfnW::Dense(w) => {
                    let mut p = Prof::default();
                    let out = dense_prefill(cfg, data, w, &x, n, &mut p);
                    prof.t_experts += p.t_experts;
                    out
                }
                FfnW::Moe(w) => {
                    let mut p = Prof::default();
                    let out = moe_prefill(cfg, data, w, &x, n, &mut p, l, pos0, stream.as_ref());
                    prof.t_router += p.t_router;
                    prof.t_experts += p.t_experts;
                    out
                }
            };
            for j in 0..n * d {
                hidden[j] = prefix2[j] + mlp_out[j];
            }
            // seam adapter, applied per position: the same row-wise matvecs
            // as the single-token forward, so prefill stays bit-identical to
            // n sequential forwards
            if let Some(s) = seam {
                if l == s.after {
                    for t in 0..n {
                        seam_apply(data, s, d, &mut hidden[t * d..(t + 1) * d]);
                    }
                }
            }
            if hd_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                hd.push((l, kind, vec_rms(&hidden[(n - 1) * d..n * d])));
            }
            if lens_on {
                let kind = if matches!(layer.attn, AttnW::Kda(_)) { "KDA" } else { "MLA" };
                lens.push((l, kind, hidden[(n - 1) * d..n * d].to_vec()));
            }
            if DUMP_LAYERS.contains(&l) {
                for t in 0..n {
                    let h = hidden[t * d..(t + 1) * d].to_vec();
                    parity_rec(|d| {
                        d.hiddens.insert((pos0 + t, l), h);
                    });
                }
            }
        }

        let tm = Instant::now();
        for t in 0..n {
            let brefs: Vec<&[f32]> = blocks.iter().map(|b| &b[t * d..(t + 1) * d]).collect();
            attn_res_refs(cfg, &hidden[t * d..(t + 1) * d], &brefs, &out_res_w, &mut buf_res[t * d..(t + 1) * d]);
        }
        hidden.copy_from_slice(&buf_res);
        if all_logits {
            // --spec verification: rmsnorm + lm_head on EVERY position (the
            // same matvec as the single-token forward, so per-position
            // logits are bit-identical to a sequential run)
            let tm = Instant::now();
            let mut out = Vec::with_capacity(n);
            for t in 0..n {
                let mut xf = vec![0f32; d];
                rmsnorm(cfg, &hidden[t * d..(t + 1) * d], Self::t(data, &norm_f), &mut xf);
                let mut logits = vec![0f32; cfg.vocab];
                Self::logits_project(data, &lm_head, lm_head_q8.as_ref(), cfg, &xf, &mut logits);
                out.push(logits);
            }
            prof.t_norm_res += tm.elapsed().as_secs_f64();
            if lens_on {
                logit_lens_compute(cfg, Self::t(data, &lm_head), Self::t(data, &norm_f), &lens, out.last().unwrap());
            }
            *last_logits = out.last().unwrap().clone();
            return out;
        }
        let mut xf = vec![0f32; d];
        rmsnorm(cfg, &hidden[(n - 1) * d..n * d], Self::t(data, &norm_f), &mut xf);
        prof.t_norm_res += tm.elapsed().as_secs_f64();
        let tm = Instant::now();
        let mut logits = vec![0f32; cfg.vocab];
        Self::logits_project(data, &lm_head, lm_head_q8.as_ref(), cfg, &xf, &mut logits);
        prof.t_lm_head += tm.elapsed().as_secs_f64();
        if hd_on {
            // per-layer rms was taken on the LAST position (the one the
            // logits are computed from)
            dump_hidden_print(&hd, vec_rms(&hidden[(n - 1) * d..n * d]), &logits);
        }
        if lens_on {
            logit_lens_compute(cfg, Self::t(data, &lm_head), Self::t(data, &norm_f), &lens, &logits);
        }
        *last_logits = logits.clone();
        vec![logits]
    }
}

// ── greedy generation + display (rustgpt style) ──

#[cfg(test)]
mod dot_simd_tests {
    use super::{dot, dot_scalar};

    /// deterministic filler (splitmix64), no rand crate
    struct Rng(u64);
    impl Rng {
        fn f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64 - 0.5) as f32
        }
    }

    /// The dispatched dot() must be BIT-IDENTICAL to the scalar reference on
    /// every length (8-chunks, awkward remainders, degenerate cases).
    #[test]
    fn dot_simd_bit_exact() {
        let mut rng = Rng(0x1234567890ABCDEF);
        for n in [0usize, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 63, 64, 65, 100, 127, 128, 129, 1024, 1025, 4096, 16384, 16387] {
            let a: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let b: Vec<f32> = (0..n).map(|_| rng.f32()).collect();
            let (want, got) = (dot_scalar(&a, &b), dot(&a, &b));
            assert_eq!(want.to_bits(), got.to_bits(), "bit mismatch at n={}", n);
        }
        // a few pathological values too (infinities, subnormals, zeros)
        for n in [8usize, 9, 64, 1000] {
            let a: Vec<f32> = (0..n)
                .map(|i| match i % 5 {
                    0 => 0.0,
                    1 => f32::MIN_POSITIVE,
                    2 => -f32::MIN_POSITIVE,
                    3 => 1e30,
                    _ => -1e-30,
                })
                .collect();
            let b: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 1e-10).collect();
            let (want, got) = (dot_scalar(&a, &b), dot(&a, &b));
            assert_eq!(want.to_bits(), got.to_bits(), "bit mismatch (pathological) at n={}", n);
        }
    }

    /// gemm_batch (the batched-prefill GEMM) must be BIT-IDENTICAL to n
    /// separate matvec calls: every (row, position) dot keeps the same
    /// accumulation order in every kernel (dot, dot8t tiles, tail dots,
    /// pooled row split).
    #[test]
    fn gemm_matches_matvec_bit_exact() {
        use super::{dot, gemm_batch};
        let mut rng = Rng(0x0F0F0F0F0F0F0F0F);
        for (rows, cols, n) in [
            (5usize, 3usize, 1usize),
            (7, 8, 3),
            (16, 64, 4),
            (33, 96, 7),
            (64, 128, 8),
            (64, 130, 13), // awkward cols remainder
            (128, 512, 32), // exercises the pooled row split
        ] {
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
            let x: Vec<f32> = (0..n * cols).map(|_| rng.f32()).collect();
            let mut out = vec![0f32; n * rows];
            gemm_batch(&w, rows, cols, &x, n, &mut out);
            for t in 0..n {
                for r in 0..rows {
                    let want = dot(&w[r * cols..(r + 1) * cols], &x[t * cols..(t + 1) * cols]);
                    assert_eq!(
                        want.to_bits(),
                        out[t * rows + r].to_bits(),
                        "bit mismatch at (t={}, r={}) for {}x{}x{}",
                        t,
                        r,
                        rows,
                        cols,
                        n
                    );
                }
            }
        }
    }

    /// Integration: kda_prefill over n = 200 positions (chunked recurrence,
    /// n >= kda_chunk::MIN_LEN) vs n single-token kda_forward calls (the
    /// sequential step) on one synthetic KDA layer. The conv caches must be
    /// bit-identical (same sequential code path); the recurrence state and
    /// the layer outputs must match within the chunk transform tolerance.
    #[test]
    fn kda_prefill_chunked_matches_forward() {
        use super::{kda_forward, kda_prefill, KdaCache, KdaW, Prof, T};
        let cfg = crate::config::Config::microkimi();
        let (d, kp, fa, hn, kd) = (cfg.d, cfg.kda_proj(), cfg.kda_fa, cfg.kda_heads, cfg.kda_dim);
        let mut rng = Rng(0xDA7A_DA7A_DA7A_DA7A);
        // f32 weights laid out back to back in one byte buffer, KdaW field order
        let lens = [
            kp * d, // q_proj
            kp * d, // k_proj
            kp * d, // v_proj
            kp * cfg.kda_conv, // q_conv
            kp * cfg.kda_conv, // k_conv
            kp * cfg.kda_conv, // v_conv
            fa * d, // f_a
            kp * fa, // f_b
            kp,     // a_log
            kp,     // dt_bias
            hn * d, // b_proj
            kp * d, // g_proj
            kp,     // o_norm
            d * kp, // o_proj
        ];
        let mut buf: Vec<f32> = Vec::new();
        let mut offs: Vec<usize> = Vec::new();
        for &len in &lens {
            offs.push(buf.len() * 4);
            buf.extend((0..len).map(|_| rng.f32() * 0.1));
        }
        let data: Vec<u8> = buf.iter().flat_map(|f| f.to_le_bytes()).collect();
        let t = |i: usize| T { off: offs[i], len: lens[i] };
        let w = KdaW {
            q_proj: t(0),
            k_proj: t(1),
            v_proj: t(2),
            q_conv: t(3),
            k_conv: t(4),
            v_conv: t(5),
            f_a: t(6),
            f_b: t(7),
            a_log: t(8),
            dt_bias: t(9),
            b_proj: t(10),
            g_proj: t(11),
            o_norm: t(12),
            o_proj: t(13),
        };
        let new_cache = || KdaCache {
            conv_q: vec![0.0; 3 * kp],
            conv_k: vec![0.0; 3 * kp],
            conv_v: vec![0.0; 3 * kp],
            s: vec![0.0; hn * kd * kd],
        };
        let n = 200usize; // > kda_chunk::MIN_LEN, spans 4 chunks
        assert!(n >= crate::model::kda_chunk::MIN_LEN);
        let x: Vec<f32> = (0..n * d).map(|_| rng.f32()).collect();
        let mut prof = Prof::default();
        let (mut c_chk, mut c_seq) = (new_cache(), new_cache());
        let out_chk = kda_prefill(&cfg, &data, &w, &mut c_chk, &x, n, &mut prof);
        let mut out_seq = vec![0f32; n * d];
        for t in 0..n {
            let o = kda_forward(&cfg, &data, &w, &mut c_seq, &x[t * d..(t + 1) * d], &mut prof);
            out_seq[t * d..(t + 1) * d].copy_from_slice(&o);
        }
        // conv caches: identical sequential code in both paths
        for (a, b) in c_chk.conv_q.iter().zip(c_seq.conv_q.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_q not bit-identical");
        }
        for (a, b) in c_chk.conv_k.iter().zip(c_seq.conv_k.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_k not bit-identical");
        }
        for (a, b) in c_chk.conv_v.iter().zip(c_seq.conv_v.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "conv_v not bit-identical");
        }
        let max_o = out_chk.iter().zip(out_seq.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let max_s = c_chk.s.iter().zip(c_seq.s.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("kda_prefill chunked vs forward: max|dO|={:.3e}  max|dS|={:.3e}", max_o, max_s);
        assert!(max_o < 1e-4, "layer output deviation {}", max_o);
        assert!(max_s < 1e-4, "recurrence state deviation {}", max_s);
    }
}

#[cfg(test)]
mod q8head_tests {
    use super::Q8Head;

    struct Rng(u64);
    impl Rng {
        fn f32(&mut self) -> f32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64 - 0.5) as f32
        }
    }

    /// The q8 lm_head projection must track the f32 matvec: max error small
    /// relative to the logit span, and identical argmax (the greedy token).
    #[test]
    fn q8head_matches_f32() {
        let (rows, cols) = (512usize, 1024usize); // multi-job split (>60k MACs)
        let mut rng = Rng(0xC0FFEE);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.f32() * 0.02).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.f32()).collect();
        let h = Q8Head::from_f32(&w, rows, cols);
        let mut got = vec![0f32; rows];
        let mut want = vec![0f32; rows];
        h.matvec(&x, &mut got);
        super::matvec_cpu(&w, rows, cols, &x, &mut want);
        let span = want.iter().map(|v| v.abs()).fold(0f32, f32::max) as f64;
        let max_err = got.iter().zip(&want).map(|(&a, &b)| (a as f64 - b as f64).abs()).fold(0f64, f64::max);
        assert!(max_err / span < 1e-2, "q8 head max err {} vs span {}", max_err, span);
        let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(am(&got), am(&want), "q8 head argmax differs from f32");
    }
}

#[cfg(test)]
mod seam_tests {
    use super::{seam_apply, seam_load, SeamW};
    use crate::config::Config;
    use crate::quant::weights::{Entry, DTYPE_F32};
    use std::collections::HashMap;

    fn entry(dims: &[u32]) -> Entry {
        Entry { dtype: DTYPE_F32, dims: dims.to_vec(), offset: 0, size: dims.iter().map(|&d| d as u64).product::<u64>() * 4 }
    }

    fn cfg(d: usize, n_layers: usize, seam_after: Option<usize>) -> Config {
        let mut c = Config::microkimi();
        c.d = d;
        c.n_layers = n_layers;
        c.seam_after = seam_after;
        c
    }

    fn seam_entries(rank: u32, d: u32) -> HashMap<String, Entry> {
        HashMap::from([("seam.A".to_string(), entry(&[rank, d])), ("seam.B".to_string(), entry(&[d, rank]))])
    }

    #[test]
    fn seam_load_valid() {
        let e = seam_entries(64, 512);
        let s = seam_load(&cfg(512, 8, Some(3)), &e).expect("valid seam must load");
        assert_eq!((s.rank, s.after), (64, 3));
        // after = n_layers - 2 is the last legal position (a layer N+1 remains)
        let s = seam_load(&cfg(512, 8, Some(6)), &e).expect("after = n_layers - 2 must load");
        assert_eq!(s.after, 6);
        // no tensors, no config key: no adapter
        assert!(seam_load(&cfg(512, 8, None), &HashMap::new()).is_none());
    }

    #[test]
    #[should_panic(expected = "seam_after 7 out of range [0, 6] for a 8-layer model")]
    fn seam_after_last_layer_rejected() {
        let e = seam_entries(64, 512);
        seam_load(&cfg(512, 8, Some(7)), &e);
    }

    #[test]
    #[should_panic(expected = "seam_after 8 out of range [0, 6] for a 8-layer model")]
    fn seam_after_beyond_layers_rejected() {
        let e = seam_entries(64, 512);
        seam_load(&cfg(512, 8, Some(8)), &e);
    }

    #[test]
    #[should_panic(expected = "no seam_after key")]
    fn seam_tensors_without_config_key_rejected() {
        let e = seam_entries(64, 512);
        seam_load(&cfg(512, 8, None), &e);
    }

    #[test]
    #[should_panic(expected = "no seam.A/seam.B tensors")]
    fn seam_config_key_without_tensors_rejected() {
        seam_load(&cfg(512, 8, Some(3)), &HashMap::new());
    }

    #[test]
    #[should_panic(expected = "seam.B is missing")]
    fn seam_unpaired_tensors_rejected() {
        let e = HashMap::from([("seam.A".to_string(), entry(&[64, 512]))]);
        seam_load(&cfg(512, 8, Some(3)), &e);
    }

    #[test]
    #[should_panic(expected = "seam.B: dims")]
    fn seam_bad_shape_rejected() {
        let e = HashMap::from([("seam.A".to_string(), entry(&[64, 512])), ("seam.B".to_string(), entry(&[64, 512]))]);
        seam_load(&cfg(512, 8, Some(3)), &e);
    }

    /// seam_apply must compute h + (h @ A^T) @ B^T: A [rank, d] row-major
    /// (tmp[r] = A[r, :] . h), then B [d, rank] row-major (delta[i] = B[i, :] . tmp).
    #[test]
    fn seam_apply_matches_reference() {
        let (rank, d) = (3usize, 10usize);
        let a: Vec<f32> = (0..rank * d).map(|i| (i as f32 - 7.0) * 0.01).collect();
        let b: Vec<f32> = (0..d * rank).map(|i| (i as f32 - 11.0) * 0.02).collect();
        let mut data = crate::quant::weights::f32_to_bytes(&a);
        data.extend_from_slice(&crate::quant::weights::f32_to_bytes(&b));
        let w = SeamW {
            a: super::T { off: 0, len: rank * d },
            b: super::T { off: rank * d * 4, len: d * rank },
            rank,
            after: 0,
        };
        let h0: Vec<f32> = (0..d).map(|i| (i as f32 - 5.0) * 0.1).collect();
        let mut h = h0.clone();
        seam_apply(&data, &w, d, &mut h);
        // f64 reference of the Python formula (tolerance: the engine's dot()
        // uses 8 parallel accumulators, the reference is sequential)
        let tmp: Vec<f64> = (0..rank)
            .map(|r| (0..d).map(|c| a[r * d + c] as f64 * h0[c] as f64).sum())
            .collect();
        let want: Vec<f64> = (0..d)
            .map(|i| h0[i] as f64 + (0..rank).map(|r| tmp[r] * b[i * rank + r] as f64).sum::<f64>())
            .collect();
        for j in 0..d {
            let err = (h[j] as f64 - want[j]).abs();
            assert!(err < 1e-6, "h[{}]: got {} want {}", j, h[j], want[j]);
        }
    }
}
