// microkimi forward pass: 93 layers, AttnRes block 12, KDA (69),
// MLA NoPE (24), latent MoE 896 experts top-16 + 2 shared (layers 1..92),
// dense MLP layer 0, SiTU everywhere, MXFP4 experts dequantized on the fly.
// All in f32, zero-copy: f32 tensors are read in slices directly from the
// mmap-like file (Vec<u8> + align_to), experts stay packed.

use crate::config::Config;
use crate::tokenizer::AnyTokenizer;
use crate::weights::{BinFile, Entry};
use std::time::Instant;

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

// ── math kernels ──

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    loop {
        match (ca.next(), cb.next()) {
            (Some(av), Some(bv)) => {
                for j in 0..8 {
                    acc[j] += av[j] * bv[j];
                }
            }
            _ => break,
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7]);
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x * y;
    }
    s
}

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

/// f32 matrix × vector. Entry point for the whole engine.
pub fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_os = "macos")]
    {
        if gpu_on() && rows * cols >= GPU_MIN_ELEMS && crate::metal::gpu_available() {
            crate::metal::gpu_matvec(w, rows, cols, x, out);
            return;
        }
    }
    matvec_cpu(w, rows, cols, x, out);
}

/// f32 matrix × vector on the persistent pool (std::thread). Adaptive job
/// count (~200k MACs/job): small matvecs stay inline, large ones are split
/// into rows. The pool barrier guarantees the validity of the raw pointers
/// captured by the jobs.
pub fn matvec_cpu(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    let p = crate::pool::pool();
    let njobs = (rows * cols / 60_000).clamp(1, p.workers).min(rows);
    if njobs <= 1 {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot(&w[r * cols..(r + 1) * cols], x);
        }
        return;
    }
    let chunk = rows.div_ceil(njobs);
    let wp = crate::pool::SPtr(w.as_ptr());
    let xp = crate::pool::SPtr(x.as_ptr());
    let op = crate::pool::MPtr(out.as_mut_ptr());
    let mut jobs: Vec<crate::pool::Job> = Vec::new();
    for j in 0..njobs {
        let (r0, r1) = (j * chunk, ((j + 1) * chunk).min(rows));
        if r0 >= r1 {
            break;
        }
        jobs.push(Box::new(move || {
            // rebind → capture whole structs (Send), not fields
            let (wp, xp, op) = (wp, xp, op);
            unsafe {
                let w = std::slice::from_raw_parts(wp.0, rows * cols);
                let x = std::slice::from_raw_parts(xp.0, cols);
                let out = std::slice::from_raw_parts_mut(op.0, rows);
                for r in r0..r1 {
                    out[r] = dot(&w[r * cols..(r + 1) * cols], x);
                }
            }
        }));
    }
    p.run(jobs);
}

pub fn rmsnorm(cfg: &Config, x: &[f32], w: &[f32], out: &mut [f32]) {
    let ss = dot(x, x) / x.len() as f32;
    let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// SiTU: a = 4·tanh(g/4)·sigmoid(g) ; u = 25·tanh(u/25) ; out = a·u
#[inline]
pub fn situ(g: f32, u: f32) -> f32 {
    4.0 * (g / 4.0).tanh() * sigmoid(g) * (25.0 * (u / 25.0).tanh())
}

/// AttnRes: softmax over the RMS-normed scores of blocks + prefix,
/// RAW values as output. `w` = norm.weight · proj.weight (pre-combined).
pub fn attn_res(cfg: &Config, prefix: &[f32], blocks: &[Vec<f32>], w: &[f32], out: &mut [f32]) {
    let b = blocks.len();
    let mut scores = vec![0f32; b + 1];
    let mut kbuf = vec![0f32; cfg.d];
    let mut score_of = |v: &[f32]| {
        let ss = dot(v, v) / cfg.d as f32;
        let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
        for j in 0..cfg.d {
            kbuf[j] = v[j] * inv;
        }
        dot(&kbuf, w)
    };
    for (i, v) in blocks.iter().enumerate() {
        scores[i] = score_of(v);
    }
    scores[b] = score_of(prefix);
    let m = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
    let mut z = 0f32;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        z += *s;
    }
    for s in scores.iter_mut() {
        *s /= z;
    }
    for j in 0..cfg.d {
        out[j] = 0.0;
    }
    for (i, v) in blocks.iter().enumerate() {
        for j in 0..cfg.d {
            out[j] += scores[i] * v[j];
        }
    }
    let p = scores[b];
    for j in 0..cfg.d {
        out[j] += p * prefix[j];
    }
}

// ── weights: (offset, dims) descriptors into the file ──

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
    experts: Vec<[u64; 3]>, // offsets of the mxfp4 blobs [w1, w2, w3] per expert
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
pub(crate) struct KdaCache {
    pub conv_q: Vec<f32>, // 3 × 512 (raw pre-conv)
    pub conv_k: Vec<f32>,
    pub conv_v: Vec<f32>,
    pub s: Vec<f32>, // 4 × 128 × 128
}

pub(crate) struct MlaCache {
    pub k: Vec<f32>, // pos × (4×192)
    pub v: Vec<f32>, // pos × (4×128)
}

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
        self.print_cfg(512, 163_840, 896);
    }

    pub fn print_cfg(&self, d: usize, vocab: usize, n_experts: usize) {
        let tot = self.t_norm_res + self.t_kda_proj + self.t_kda_conv + self.t_kda_recur + self.t_mla + self.t_router + self.t_experts + self.t_lm_head;
        if tot == 0.0 {
            return;
        }
        let lm_label = format!("lm_head ({} x {})", d, vocab);
        let router_label = format!("MoE router ({})", n_experts);
        let rows = [
            ("RMSNorm + AttnRes".to_string(), self.t_norm_res),
            ("KDA projections (qkv/f/g/o)".to_string(), self.t_kda_proj),
            ("causal KDA conv1d".to_string(), self.t_kda_conv),
            ("KDA recurrence (state S)".to_string(), self.t_kda_recur),
            ("MLA attention".to_string(), self.t_mla),
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
    norm_f: T,
    out_res_w: Vec<f32>,
    layers: Vec<LayerW>,
    pub(crate) caches: Vec<Cache>, // pub(crate): saved/restored by mkmem.rs
    pub last_logits: Vec<f32>,     // logits of the last forward (source for mkmem --save)
    pub prof: Prof,
}

impl Model {
    fn t<'a>(data: &'a [u8], t: &T) -> &'a [f32] {
        as_f32(&data[t.off..t.off + t.len * 4])
    }

    pub fn load(path: &str) -> Self {
        let bin = BinFile::open(path);
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
                caches.push(Cache::Mla(MlaCache { k: Vec::new(), v: Vec::new() }));
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
                let experts: Vec<[u64; 3]> = (0..cfg.n_experts)
                    .map(|e| {
                        ["w1", "w2", "w3"].map(|wn| {
                            bin.entries
                                .get(&format!("{}{}.{}", pfx, e, wn))
                                .unwrap_or_else(|| panic!("missing expert: {}{}.{}", pfx, e, wn))
                                .offset
                        })
                    })
                    .collect();
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
        Model { cfg, bin, embed, lm_head, norm_f, out_res_w, layers, caches, last_logits: Vec::new(), prof: Prof::default() }
    }

    /// Number of tokens already represented in the caches (from the first MLA
    /// layer; 0 for a KDA-only model). Only seeds the debug position counter:
    /// K3 has no positional encoding, so the math does not depend on it.
    fn cached_tokens(&self) -> usize {
        for c in &self.caches {
            if let Cache::Mla(m) = c {
                return m.k.len() / (self.cfg.mla_heads * self.cfg.mla_qh());
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
                }
            }
        }
    }
}

// ── KDA ──

#[allow(clippy::too_many_arguments)]
fn kda_forward(
    cfg: &Config,
    data: &[u8],
    w: &KdaW,
    cache: &mut KdaCache,
    x: &[f32],
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    let mut q = vec![0f32; cfg.kda_proj()];
    let mut k = vec![0f32; cfg.kda_proj()];
    let mut v = vec![0f32; cfg.kda_proj()];
    matvec(Model::t(data, &w.q_proj), cfg.kda_proj(), cfg.d, x, &mut q);
    matvec(Model::t(data, &w.k_proj), cfg.kda_proj(), cfg.d, x, &mut k);
    matvec(Model::t(data, &w.v_proj), cfg.kda_proj(), cfg.d, x, &mut v);
    let mut g_low = vec![0f32; cfg.kda_proj()];
    {
        let mut fa = vec![0f32; cfg.kda_fa];
        matvec(Model::t(data, &w.f_a), cfg.kda_fa, cfg.d, x, &mut fa);
        matvec(Model::t(data, &w.f_b), cfg.kda_proj(), cfg.kda_fa, &fa, &mut g_low);
    }
    let mut beta = vec![0f32; cfg.kda_heads];
    matvec(Model::t(data, &w.b_proj), cfg.kda_heads, cfg.d, x, &mut beta);
    for b in beta.iter_mut() {
        *b = sigmoid(*b);
    }
    let mut g2 = vec![0f32; cfg.kda_proj()];
    matvec(Model::t(data, &w.g_proj), cfg.kda_proj(), cfg.d, x, &mut g2);
    prof.t_kda_proj += tm.elapsed().as_secs_f64();

    // depthwise causal conv kernel 4 + SiLU; cache = last 3 raw inputs
    let tm = Instant::now();
    let do_conv = |vec: &mut [f32], conv_w: &T, cache_raw: &mut Vec<f32>| {
        let w_conv = Model::t(data, conv_w);
        // window = 3 previous raw inputs + current input (weight j=0 → oldest)
        let mut out = vec![0f32; cfg.kda_proj()];
        for c in 0..cfg.kda_proj() {
            let mut acc = 0f32;
            for j in 0..3 {
                acc += w_conv[c * cfg.kda_conv + j] * cache_raw[j * cfg.kda_proj() + c];
            }
            acc += w_conv[c * cfg.kda_conv + 3] * vec[c];
            out[c] = silu(acc);
        }
        // cache update: left shift, push the current raw input
        cache_raw.copy_within(cfg.kda_proj()..3 * cfg.kda_proj(), 0);
        cache_raw[2 * cfg.kda_proj()..3 * cfg.kda_proj()].copy_from_slice(vec);
        vec.copy_from_slice(&out);
    };
    do_conv(&mut q, &w.q_conv, &mut cache.conv_q);
    do_conv(&mut k, &w.k_conv, &mut cache.conv_k);
    do_conv(&mut v, &w.v_conv, &mut cache.conv_v);
    prof.t_kda_conv += tm.elapsed().as_secs_f64();

    let tm = Instant::now();
    // per-head L2-norm over 128, q × 128^-0.5
    let norm_head = |vec: &mut [f32], scale: f32| {
        for h in 0..cfg.kda_heads {
            let head = &mut vec[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
            let n2 = dot(head, head);
            let inv = scale / n2.sqrt().max(1e-12);
            for x in head.iter_mut() {
                *x *= inv;
            }
        }
    };
    norm_head(&mut q, (cfg.kda_dim as f32).powf(-0.5));
    norm_head(&mut k, 1.0);
    // log-decay gate: g = -5 · sigmoid(exp(A_log) · (g_low + dt_bias))
    let a_log = Model::t(data, &w.a_log);
    let dt_bias = Model::t(data, &w.dt_bias);
    let mut g = vec![0f32; cfg.kda_proj()];
    for h in 0..cfg.kda_heads {
        for c in 0..cfg.kda_dim {
            let i = h * cfg.kda_dim + c;
            g[i] = cfg.gate_lb * sigmoid(a_log[c].exp() * (g_low[i] + dt_bias[i]));
        }
    }
    // recurrence: persistent S[4,128,128]
    let s = &mut cache.s;
    let mut o = vec![0f32; cfg.kda_proj()];
    for h in 0..cfg.kda_heads {
        let sh = &mut s[h * cfg.kda_dim * cfg.kda_dim..(h + 1) * cfg.kda_dim * cfg.kda_dim];
        let gh = &g[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let kh = &k[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let vh = &v[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let qh = &q[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        // decay along K
        for i in 0..cfg.kda_dim {
            let decay = gh[i].exp();
            let row = &mut sh[i * cfg.kda_dim..(i + 1) * cfg.kda_dim];
            for x in row.iter_mut() {
                *x *= decay;
            }
        }
        // δ = v - kᵀ S
        let mut delta = vh.to_vec();
        for i in 0..cfg.kda_dim {
            let row = &sh[i * cfg.kda_dim..(i + 1) * cfg.kda_dim];
            let ki = kh[i];
            for j in 0..cfg.kda_dim {
                delta[j] -= ki * row[j];
            }
        }
        // S += (β k) ⊗ δ ; o = qᵀ S  →  o[j] = Σ_i q[i]·S[i][j]
        let bh = beta[h];
        let oh = &mut o[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        for j in 0..cfg.kda_dim {
            oh[j] = 0.0;
        }
        for i in 0..cfg.kda_dim {
            let row = &mut sh[i * cfg.kda_dim..(i + 1) * cfg.kda_dim];
            let bk = bh * kh[i];
            for j in 0..cfg.kda_dim {
                row[j] += bk * delta[j];
            }
            let qi = qh[i];
            for j in 0..cfg.kda_dim {
                oh[j] += qi * row[j];
            }
        }
    }
    prof.t_kda_recur += tm.elapsed().as_secs_f64();

    let tm = Instant::now();
    // per-head gated rmsnorm: y = o·rsqrt(mean(o²)+eps)·o_norm ; o = y·sigmoid(g2)
    let o_norm = Model::t(data, &w.o_norm);
    for h in 0..cfg.kda_heads {
        let oh = &mut o[h * cfg.kda_dim..(h + 1) * cfg.kda_dim];
        let ss = dot(oh, oh) / cfg.kda_dim as f32;
        let inv = 1.0 / (ss + cfg.rms_eps).sqrt();
        for c in 0..cfg.kda_dim {
            oh[c] = oh[c] * inv * o_norm[c] * sigmoid(g2[h * cfg.kda_dim + c]);
        }
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.o_proj), cfg.d, cfg.kda_proj(), &o, &mut out);
    prof.t_kda_proj += tm.elapsed().as_secs_f64();
    out
}

// ── MLA: full NoPE ──

fn mla_forward(
    cfg: &Config,
    data: &[u8],
    w: &MlaW,
    cache: &mut MlaCache,
    x: &[f32],
    prof: &mut Prof,
) -> Vec<f32> {
    let tm = Instant::now();
    // q = q_b(rmsnorm(q_a(x))) [768]
    let mut qa = vec![0f32; cfg.mla_qa];
    matvec(Model::t(data, &w.q_a), cfg.mla_qa, cfg.d, x, &mut qa);
    let mut qa_n = vec![0f32; cfg.mla_qa];
    rmsnorm(cfg, &qa, Model::t(data, &w.q_a_ln), &mut qa_n);
    let mut q = vec![0f32; cfg.mla_qb()];
    matvec(Model::t(data, &w.q_b), cfg.mla_qb(), cfg.mla_qa, &qa_n, &mut q);
    // c = kv_a(x) [128] ; k_pass [64] ; k_rot [64] (shared across heads)
    let mut c = vec![0f32; cfg.mla_qa];
    matvec(Model::t(data, &w.kv_a), cfg.mla_qa, cfg.d, x, &mut c);
    let k_rot: Vec<f32> = c[cfg.mla_kva..cfg.mla_kva + cfg.mla_rope].to_vec();
    let mut kp_n = vec![0f32; cfg.mla_kva];
    rmsnorm(cfg, &c[..cfg.mla_kva], Model::t(data, &w.kv_a_ln), &mut kp_n);
    let mut kb = vec![0f32; cfg.mla_kvb()];
    matvec(Model::t(data, &w.kv_b), cfg.mla_kvb(), cfg.mla_kva, &kp_n, &mut kb);
    // K[h] = kb[h][..128] ++ k_rot ; V[h] = kb[h][128..256]
    let mut k_new = vec![0f32; cfg.mla_heads * cfg.mla_qh()];
    let mut v_new = vec![0f32; cfg.mla_heads * cfg.mla_v];
    for h in 0..cfg.mla_heads {
        k_new[h * cfg.mla_qh()..h * cfg.mla_qh() + cfg.mla_nope]
            .copy_from_slice(&kb[h * (cfg.mla_nope + cfg.mla_v)..h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope]);
        k_new[h * cfg.mla_qh() + cfg.mla_nope..(h + 1) * cfg.mla_qh()].copy_from_slice(&k_rot);
        v_new[h * cfg.mla_v..(h + 1) * cfg.mla_v].copy_from_slice(
            &kb[h * (cfg.mla_nope + cfg.mla_v) + cfg.mla_nope..(h + 1) * (cfg.mla_nope + cfg.mla_v)],
        );
    }
    cache.k.extend_from_slice(&k_new);
    cache.v.extend_from_slice(&v_new);
    let pos = cache.k.len() / (cfg.mla_heads * cfg.mla_qh()) - 1;
    // causal attention, scale 192^-0.5
    let scale = (cfg.mla_qh() as f32).powf(-0.5);
    let mut attn = vec![0f32; cfg.mla_heads * cfg.mla_v];
    for h in 0..cfg.mla_heads {
        let qh = &q[h * cfg.mla_qh()..(h + 1) * cfg.mla_qh()];
        let mut scores = vec![0f32; pos + 1];
        for j in 0..=pos {
            let kj = &cache.k[(j * cfg.mla_heads + h) * cfg.mla_qh()..(j * cfg.mla_heads + h + 1) * cfg.mla_qh()];
            scores[j] = dot(qh, kj) * scale;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
        let mut z = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            z += *s;
        }
        for s in scores.iter_mut() {
            *s /= z;
        }
        let oh = &mut attn[h * cfg.mla_v..(h + 1) * cfg.mla_v];
        for j in 0..=pos {
            let vj = &cache.v[(j * cfg.mla_heads + h) * cfg.mla_v..(j * cfg.mla_heads + h + 1) * cfg.mla_v];
            let p = scores[j];
            for d in 0..cfg.mla_v {
                oh[d] += p * vj[d];
            }
        }
    }
    // output gate + o_proj
    let mut g = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.g_proj), cfg.d, cfg.d, x, &mut g);
    for i in 0..cfg.d {
        attn[i] *= sigmoid(g[i]);
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.o_proj), cfg.d, cfg.mla_heads * cfg.mla_v, &attn, &mut out);
    prof.t_mla += tm.elapsed().as_secs_f64();
    out
}

// ── MoE ──

fn moe_forward(cfg: &Config, data: &[u8], w: &MoeW, x: &[f32], prof: &mut Prof, layer: usize, pos: usize) -> Vec<f32> {
    // noaux_tc router: sigmoid, +bias for selection, weights without bias
    let tm = Instant::now();
    let gate_w = Model::t(data, &w.gate_w);
    let gate_b = Model::t(data, &w.gate_b);
    let mut logits = vec![0f32; cfg.n_experts];
    matvec(gate_w, cfg.n_experts, cfg.d, x, &mut logits);
    let mut sel: Vec<(u32, f32, f32)> = Vec::with_capacity(cfg.top_k); // (expert, score, key)
    for (i, &l) in logits.iter().enumerate() {
        let sc = sigmoid(l);
        let key = sc + gate_b[i];
        let item = (i as u32, sc, key);
        if sel.len() < cfg.top_k {
            sel.push(item);
            if sel.len() == cfg.top_k {
                sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        } else if key > sel[cfg.top_k - 1].2 {
            sel[cfg.top_k - 1] = item;
            sel.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        }
    }
    let sumw: f32 = sel.iter().map(|s| s.1).sum::<f32>() + 1e-20;
    let weights: Vec<f32> = sel.iter().map(|s| s.1 / sumw).collect();
    if ROUTER_LAYERS.contains(&layer) {
        let mut ids: Vec<u32> = sel.iter().map(|s| s.0).collect();
        ids.sort();
        parity_rec(|d| {
            d.router.insert((pos, layer), ids);
        });
    }
    // --debug-routing: top-3 by renormalized weight + count of top-16 appearances
    ROUTING.with(|r| {
        if let Some(d) = r.borrow_mut().as_mut() {
            let mut top3: Vec<(u32, f32)> = sel.iter().map(|s| (s.0, s.1 / sumw)).collect();
            top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            top3.truncate(3);
            d.cur.push((layer, top3));
            for s in &sel {
                *d.counts.entry((layer, s.0)).or_insert(0) += 1;
            }
        }
    });
    let mut h = vec![0f32; cfg.routed_hidden];
    matvec(Model::t(data, &w.routed_down), cfg.routed_hidden, cfg.d, x, &mut h);
    prof.t_router += tm.elapsed().as_secs_f64();

    // MXFP4 experts (dequantized on the fly): SiTU(cat(w1 h, w3 h)) then w2.
    // The 16 experts are independent → one pool job per expert (offsets
    // precomputed at load time, zero lookup). Combination in fixed order after
    // the barrier → deterministic.
    let tm = Instant::now();
    let expert_packed = cfg.routed_hidden * cfg.moe_inter / 2;
    let expert_blob = expert_packed + cfg.routed_hidden * cfg.moe_inter / 32;
    let (erh, emi) = (cfg.routed_hidden, cfg.moe_inter); // copies for the 'static closures
    let mut outs = vec![0f32; cfg.top_k * cfg.routed_hidden];
    {
        let dp = crate::pool::SPtrU8(data.as_ptr());
        let dlen = data.len();
        let hp = crate::pool::SPtr(h.as_ptr());
        let op = crate::pool::MPtr(outs.as_mut_ptr());
        let mut jobs: Vec<crate::pool::Job> = Vec::with_capacity(cfg.top_k);
        for (ei, _) in weights.iter().enumerate() {
            let offs = w.experts[sel[ei].0 as usize];
            jobs.push(Box::new(move || {
                let (dp, hp, op) = (dp, hp, op);
                unsafe {
                    let data = std::slice::from_raw_parts(dp.0, dlen);
                    let h = std::slice::from_raw_parts(hp.0, erh);
                    let blob = |i: usize| &data[offs[i] as usize..offs[i] as usize + expert_blob];
                    let mut a = vec![0f32; emi];
                    let mut u = vec![0f32; emi];
                    crate::mxfp4::matvec_packed(&blob(0)[..expert_packed], &blob(0)[expert_packed..], emi, erh, h, &mut a, 1);
                    crate::mxfp4::matvec_packed(&blob(2)[..expert_packed], &blob(2)[expert_packed..], emi, erh, h, &mut u, 1);
                    let mut act = vec![0f32; emi];
                    for j in 0..emi {
                        act[j] = situ(a[j], u[j]);
                    }
                    let o = std::slice::from_raw_parts_mut(op.0.add(ei * erh), erh);
                    crate::mxfp4::matvec_packed(&blob(1)[..expert_packed], &blob(1)[expert_packed..], erh, emi, &act, o, 1);
                }
            }));
        }
        crate::pool::pool().run(jobs);
    }
    let mut y = vec![0f32; cfg.routed_hidden];
    for (ei, &wi) in weights.iter().enumerate() {
        for j in 0..cfg.routed_hidden {
            y[j] += wi * outs[ei * cfg.routed_hidden + j];
        }
    }
    // norm BEFORE up-proj
    let mut yn = vec![0f32; cfg.routed_hidden];
    rmsnorm(cfg, &y, Model::t(data, &w.routed_norm), &mut yn);
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.routed_up), cfg.d, cfg.routed_hidden, &yn, &mut out);
    // shared experts (2): SiTU MLP on the pre-down input
    let mut sa = vec![0f32; cfg.shared_inter];
    let mut su = vec![0f32; cfg.shared_inter];
    matvec(Model::t(data, &w.shared_gate), cfg.shared_inter, cfg.d, x, &mut sa);
    matvec(Model::t(data, &w.shared_up), cfg.shared_inter, cfg.d, x, &mut su);
    let mut sact = vec![0f32; cfg.shared_inter];
    for j in 0..cfg.shared_inter {
        sact[j] = situ(sa[j], su[j]);
    }
    let mut sout = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.shared_down), cfg.d, cfg.shared_inter, &sact, &mut sout);
    if layer == 1 {
        let (routed, shared) = (out.clone(), sout.clone());
        parity_rec(|d| {
            d.l1_routed.insert(pos, routed);
            d.l1_shared.insert(pos, shared);
        });
    }
    for j in 0..cfg.d {
        out[j] += sout[j];
    }
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

fn dense_forward(cfg: &Config, data: &[u8], w: &DenseW, x: &[f32], prof: &mut Prof) -> Vec<f32> {
    let tm = Instant::now();
    let mut a = vec![0f32; cfg.dense_inter];
    let mut u = vec![0f32; cfg.dense_inter];
    matvec(Model::t(data, &w.gate), cfg.dense_inter, cfg.d, x, &mut a);
    matvec(Model::t(data, &w.up), cfg.dense_inter, cfg.d, x, &mut u);
    let mut act = vec![0f32; cfg.dense_inter];
    for j in 0..cfg.dense_inter {
        act[j] = situ(a[j], u[j]);
    }
    let mut out = vec![0f32; cfg.d];
    matvec(Model::t(data, &w.down), cfg.d, cfg.dense_inter, &act, &mut out);
    prof.t_experts += tm.elapsed().as_secs_f64();
    out
}

// ── parity dumps (thread-local, inactive during normal inference) ──

#[derive(Default)]
pub struct ParityDump {
    pub hiddens: std::collections::HashMap<(usize, usize), Vec<f32>>, // (pos, layer)
    pub l1_attn: std::collections::HashMap<usize, Vec<f32>>,
    pub l1_routed: std::collections::HashMap<usize, Vec<f32>>,
    pub l1_shared: std::collections::HashMap<usize, Vec<f32>>,
    pub router: std::collections::HashMap<(usize, usize), Vec<u32>>, // (pos, layer) sorted top-16
}

pub const DUMP_LAYERS: [usize; 7] = [0, 1, 3, 4, 12, 47, 92];
pub const ROUTER_LAYERS: [usize; 3] = [1, 47, 92];

thread_local! {
    pub static PARITY: std::cell::RefCell<Option<ParityDump>> = std::cell::RefCell::new(None);
}

// ── --debug-routing collection (thread-local, inactive by default) ──

#[derive(Default)]
pub struct RoutingDebug {
    pub cur: Vec<(usize, Vec<(u32, f32)>)>, // layer → top-3 (expert, renormalized weight)
    pub counts: std::collections::HashMap<(usize, u32), u32>, // (layer, expert) → times in top-16
}

thread_local! {
    pub static ROUTING: std::cell::RefCell<Option<RoutingDebug>> = std::cell::RefCell::new(None);
}

fn parity_rec(f: impl FnOnce(&mut ParityDump)) {
    PARITY.with(|p| {
        if let Some(d) = p.borrow_mut().as_mut() {
            f(d);
        }
    });
}

// ── full forward ──

impl Model {
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // destructuring: independent per-field borrows (data immutable,
        // caches/prof mutable) - no raw pointers.
        let Self { cfg, bin, embed, lm_head, norm_f, out_res_w, layers, caches, last_logits, prof } = self;
        let cfg = &*cfg;
        let data = &bin.data[..];
        let embed = Self::t(data, embed);
        let mut hidden = embed[token as usize * cfg.d..(token as usize + 1) * cfg.d].to_vec();
        let mut blocks: Vec<Vec<f32>> = Vec::with_capacity(8);
        let mut buf_res = vec![0f32; cfg.d];
        let mut x = vec![0f32; cfg.d];

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
                    let out = moe_forward(cfg, data, w, &x, &mut p, l, pos);
                    prof.t_router += p.t_router;
                    prof.t_experts += p.t_experts;
                    out
                }
            };
            for j in 0..cfg.d {
                hidden[j] = prefix2[j] + mlp_out[j];
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
        matvec(Self::t(data, &lm_head), cfg.vocab, cfg.d, &xf, &mut logits);
        prof.t_lm_head += tm.elapsed().as_secs_f64();
        *last_logits = logits.clone();
        logits
    }
}

// ── greedy generation + display (rustgpt style) ──

fn top_k_probs(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let m = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut z = 0f32;
    for &l in logits {
        z += (l - m).exp();
    }
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        let p = (l - m).exp() / z;
        if top.len() < k {
            top.push((i, p));
            if top.len() == k {
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
        } else if p > top[k - 1].1 {
            top[k - 1] = (i, p);
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }
    }
    top
}

fn py_repr(s: &str) -> String {
    let mut out = String::from("'");
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\'' => out.push_str("\\'"),
            c => out.push(c),
        }
    }
    out.push('\'');
    out
}

pub fn run_turn(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32) -> String {
    model.reset_cache();
    run_turn_impl(ids, max_new, tok, model, debug, debug_routing, stop_id, false, None)
}

/// Same as run_turn but keeps the current caches (restored from a .mkmem
/// snapshot via --memory): the prompt tokens are fed on top of the loaded
/// state and `init_logits` (the logits stored in the snapshot) seed the
/// decoding when the prompt is empty - a pure continuation.
pub fn run_turn_resume(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>) -> String {
    run_turn_impl(ids, max_new, tok, model, debug, debug_routing, stop_id, true, init_logits)
}

fn run_turn_impl(ids: &[u32], max_new: usize, tok: &AnyTokenizer, model: &mut Model, debug: bool, debug_routing: bool, stop_id: u32, resumed: bool, init_logits: Option<Vec<f32>>) -> String {
    model.prof = Prof::default();
    let mut pos = if resumed { model.cached_tokens() } else { 0 };
    let answer = run_turn_core_resume(
        ids,
        max_new,
        tok,
        &mut |id| {
            let l = model.forward(id, pos);
            pos += 1;
            l
        },
        debug,
        debug_routing,
        stop_id,
        init_logits,
    );
    model.prof.print_cfg(model.cfg.d, model.cfg.vocab, model.cfg.n_experts);
    answer
}

/// Generic greedy generation loop: prefill then argmax decode through the
/// `fwd` closure (one forward per token, position tracked by the caller).
/// Shared by the K3 Model (run_turn) and the DeepSeek DsModel (ds_run_turn).
pub fn run_turn_core(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(u32) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32) -> String {
    run_turn_core_resume(ids, max_new, tok, fwd, debug, debug_routing, stop_id, None)
}

/// run_turn_core + optional initial logits restored from a .mkmem snapshot:
/// with an empty prompt the decoding starts straight from them (pure
/// continuation, no token is re-ingested).
pub fn run_turn_core_resume(ids: &[u32], max_new: usize, tok: &AnyTokenizer, fwd: &mut dyn FnMut(u32) -> Vec<f32>, debug: bool, debug_routing: bool, stop_id: u32, init_logits: Option<Vec<f32>>) -> String {
    if debug_routing {
        ROUTING.with(|r| *r.borrow_mut() = Some(RoutingDebug::default()));
    }

    if debug {
        println!("{}", "=".repeat(64));
        println!("STEP 0 - TOKENIZATION  ({} tokens)", ids.len());
        println!("{}", "=".repeat(64));
        for (i, &id) in ids.iter().enumerate() {
            println!("  position {:2} : token {:6} = {}", i, id, py_repr(&tok.decode_id(id)));
        }
    }

    // ── sequential prefill (simple) ──
    let t2 = Instant::now();
    let mut logits = init_logits.unwrap_or_default();
    for &id in ids {
        logits = fwd(id);
    }
    if logits.is_empty() {
        eprintln!("error: nothing to continue from (empty prompt and no logits stored in the .mkmem snapshot)");
        std::process::exit(1);
    }
    let t_prefill = t2.elapsed();
    if debug {
        println!();
        println!("{}", "=".repeat(64));
        println!("STEP 1 - sequential PREFILL  (caches filled)");
        println!("{}", "=".repeat(64));
        if ids.is_empty() {
            println!("⏱  skipped: pure continuation from the .mkmem snapshot");
        } else {
            println!("⏱  {:.2} s  for {} tokens ({:.1} ms/token)", t_prefill.as_secs_f64(), ids.len(), t_prefill.as_secs_f64() / ids.len() as f64 * 1000.0);
        }
        println!();
        println!("{}", "=".repeat(64));
        println!("STEP 2 - GENERATION  (greedy: softmax → argmax, stop = token {})", stop_id);
        println!("{}", "=".repeat(64));
    }

    let mut generated: Vec<u32> = Vec::new();
    let mut gen_times: Vec<f64> = Vec::new();
    if debug_routing {
        // ignore prefill in the routing display (generated tokens only)
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                d.cur.clear();
            }
        });
    }
    for i in 0..max_new {
        let top = top_k_probs(&logits, 5);
        let next_id = top[0].0 as u32;
        if debug {
            let candidats: Vec<String> = top
                .iter()
                .map(|&(tid, p)| format!("{} {:.1}%", py_repr(&tok.decode_id(tid as u32)), p * 100.0))
                .collect();
            println!();
            println!("token {:2} → {}", i + 1, py_repr(&tok.decode_id(next_id)));
            println!("  candidates: {}", candidats.join("  "));
        }
        if next_id == stop_id {
            if debug {
                println!("  [end: stop token {}]", stop_id);
            }
            break;
        }
        let ta = Instant::now();
        logits = fwd(next_id);
        let dt = ta.elapsed().as_secs_f64();
        gen_times.push(dt);
        generated.push(next_id);
        if debug_routing {
            ROUTING.with(|r| {
                if let Some(d) = r.borrow_mut().as_mut() {
                    let segs: Vec<String> = d
                        .cur
                        .iter()
                        .map(|(l, top3)| {
                            let exps: Vec<String> = top3
                                .iter()
                                .map(|(e, w)| format!("E{}({:.2})", e, w))
                                .collect();
                            format!("L{}: {}", l, exps.join(" "))
                        })
                        .collect();
                    println!("tok {} | {}", py_repr(&tok.decode_id(next_id)), segs.join(" | "));
                    d.cur.clear();
                }
            });
        }
        if debug {
            println!("  ⏱  {:.0} ms for this token", dt * 1000.0);
        }
    }

    if debug_routing {
        ROUTING.with(|r| {
            if let Some(d) = r.borrow_mut().as_mut() {
                let mut all: Vec<((usize, u32), u32)> = d.counts.iter().map(|(k, v)| (*k, *v)).collect();
                all.sort_by(|a, b| b.1.cmp(&a.1));
                println!();
                println!("Most-used experts of the run (top-10, top-16 appearances):");
                for ((l, e), n) in all.iter().take(10) {
                    println!("  L{} E{} : {}×", l, e, n);
                }
            }
        });
    }

    let answer = tok.decode(&generated);
    if debug {
        println!();
        println!("{}", "=".repeat(64));
        println!("SUMMARY");
        println!("{}", "=".repeat(64));
        println!("answer: {}", answer);
    } else {
        println!("Bot > {}", answer);
    }
    if !gen_times.is_empty() {
        let moy = gen_times.iter().sum::<f64>() / gen_times.len() as f64;
        if debug {
            println!("prefill: {:.2} s  |  generation: {:.0} ms/token average ({:.1} tok/s)",
                t_prefill.as_secs_f64(), moy * 1000.0, 1.0 / moy);
        } else {
            println!("  ({:.0} ms/token, {:.1} tok/s)", moy * 1000.0, 1.0 / moy);
        }
    }
    answer
}
