// Explicit architecture configuration (MKIM0002) + microkimi defaults (MKIM0001).
// Everything that was hardcoded in model.rs (93 layers, block 12, micro dims)
// becomes fields - the same engine runs both microkimi and nanokimi.

use crate::json::Json;

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layers: usize,
    pub d: usize,               // hidden_size
    pub vocab: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub n_shared: usize,
    pub kda_heads: usize,
    pub kda_dim: usize,
    pub kda_conv: usize,
    pub kda_fa: usize,          // f_a rank
    pub gate_lb: f32,           // gate_lower_bound (-5.0)
    pub mla_heads: usize,
    pub mla_qa: usize,          // q_lora_rank
    pub mla_kva: usize,         // kv_lora_rank
    pub mla_nope: usize,
    pub mla_rope: usize,
    pub mla_v: usize,
    pub routed_hidden: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub dense_inter: usize,
    pub attn_res_block: usize,
    pub first_k_dense: usize,
    pub rms_eps: f32,
    pub eos_id: u32,            // end_of_msg (generation stop)
    pub bos_id: u32,
    /// Explicit per-layer types, written by `microkimi slice` after layer
    /// pruning (the l%4==3 / first_k_dense patterns no longer hold on the
    /// renumbered layers). None = derive from the historical patterns.
    pub mla_layers: Option<Vec<usize>>,
    pub dense_layers: Option<Vec<usize>>,
    /// Layer index after which the embedded seam adapter (fp32 tensors
    /// seam.A / seam.B, written by nano/apply_lora_bin.py --write-seam) is
    /// applied to the residual stream. None: no seam adapter.
    pub seam_after: Option<usize>,
    /// Present when the MKIM0002 config JSON declares arch "qwen3_5_moe"
    /// (MoE text decoders: Qwen3.5/3.6-MoE, Qwen3.8-2.4T-A95B) or
    /// "qwen3_5" (dense text decoders: Qwen3.8-27B).
    pub qwen: Option<QwenConfig>,
    /// Present when the MKIM0002 config JSON declares arch "deepseek_v4"
    /// (parsed from the "ds" object). None for K3 (microkimi/nanokimi).
    pub ds: Option<DsConfig>,
}

impl Config {
    /// Historical microkimi architecture (MKIM0001, historical micro config).
    pub fn microkimi() -> Config {
        Config {
            n_layers: 93,
            d: 512,
            vocab: 163_840,
            n_experts: 896,
            top_k: 16,
            n_shared: 2,
            kda_heads: 4,
            kda_dim: 128,
            kda_conv: 4,
            kda_fa: 128,
            gate_lb: -5.0,
            mla_heads: 4,
            mla_qa: 128,
            mla_kva: 64,
            mla_nope: 128,
            mla_rope: 64,
            mla_v: 128,
            routed_hidden: 128,
            moe_inter: 64,
            shared_inter: 128,
            dense_inter: 2048,
            attn_res_block: 12,
            first_k_dense: 1,
            rms_eps: 1e-5,
            eos_id: 163_586,
            bos_id: 163_584,
            ds: None,
            qwen: None,
            mla_layers: None,
            dense_layers: None,
            seam_after: None,
        }
    }

    fn num(j: &Json, key: &str, default: f64) -> f64 {
        j.get(key).and_then(|x| x.as_num()).unwrap_or(default)
    }

    /// Parses the MKIM0002 JSON config block (missing keys take the micro defaults).
    pub fn from_json(j: &Json) -> Config {
        let mut c = Config::microkimi();
        c.n_layers = Self::num(j, "n_layers", c.n_layers as f64) as usize;
        c.d = Self::num(j, "hidden", c.d as f64) as usize;
        c.vocab = Self::num(j, "vocab", c.vocab as f64) as usize;
        c.n_experts = Self::num(j, "n_experts", c.n_experts as f64) as usize;
        c.top_k = Self::num(j, "top_k", c.top_k as f64) as usize;
        c.n_shared = Self::num(j, "n_shared", c.n_shared as f64) as usize;
        c.kda_heads = Self::num(j, "kda_heads", c.kda_heads as f64) as usize;
        c.kda_dim = Self::num(j, "kda_dim", c.kda_dim as f64) as usize;
        c.kda_conv = Self::num(j, "kda_conv", c.kda_conv as f64) as usize;
        c.kda_fa = Self::num(j, "kda_fa_rank", c.kda_fa as f64) as usize;
        c.gate_lb = Self::num(j, "gate_lower_bound", c.gate_lb as f64) as f32;
        c.mla_heads = Self::num(j, "mla_heads", c.mla_heads as f64) as usize;
        c.mla_qa = Self::num(j, "mla_q_lora", c.mla_qa as f64) as usize;
        c.mla_kva = Self::num(j, "mla_kv_lora", c.mla_kva as f64) as usize;
        c.mla_nope = Self::num(j, "mla_nope", c.mla_nope as f64) as usize;
        c.mla_rope = Self::num(j, "mla_rope", c.mla_rope as f64) as usize;
        c.mla_v = Self::num(j, "mla_v", c.mla_v as f64) as usize;
        c.routed_hidden = Self::num(j, "routed_hidden", c.routed_hidden as f64) as usize;
        c.moe_inter = Self::num(j, "moe_inter", c.moe_inter as f64) as usize;
        c.shared_inter = Self::num(j, "shared_inter", c.shared_inter as f64) as usize;
        c.dense_inter = Self::num(j, "dense_inter", c.dense_inter as f64) as usize;
        c.attn_res_block = Self::num(j, "attn_res_block", c.attn_res_block as f64) as usize;
        c.first_k_dense = Self::num(j, "first_k_dense", c.first_k_dense as f64) as usize;
        c.rms_eps = Self::num(j, "rms_eps", c.rms_eps as f64) as f32;
        if let Some(sp) = j.get("specials") {
            c.eos_id = Self::num(sp, "end_of_msg", c.eos_id as f64) as u32;
            c.bos_id = Self::num(sp, "bos", c.bos_id as f64) as u32;
        }
        if j.get("arch").and_then(|x| x.as_str()) == Some("deepseek_v4") {
            c.ds = Some(DsConfig::from_json(j));
        }
        if matches!(
            j.get("arch").and_then(|x| x.as_str()),
            Some("qwen3_5_moe") | Some("qwen3_5")
        ) {
            c.qwen = Some(QwenConfig::from_json(j));
        }
        let ids = |key: &str| -> Option<Vec<usize>> {
            j.get(key).and_then(|x| x.as_arr()).map(|a| a.iter().filter_map(|v| v.as_num().map(|n| n as usize)).collect())
        };
        c.mla_layers = ids("mla_layers");
        c.dense_layers = ids("dense_layers");
        c.seam_after = j.get("seam_after").and_then(|x| x.as_num()).map(|n| n as usize);
        c
    }

    pub fn is_mla(&self, l: usize) -> bool {
        match &self.mla_layers {
            Some(v) => v.contains(&l),
            None => l % 4 == 3 || l == self.n_layers - 1,
        }
    }
    pub fn is_moe(&self, l: usize) -> bool {
        match &self.dense_layers {
            Some(v) => !v.contains(&l),
            None => l >= self.first_k_dense,
        }
    }
    pub fn kda_proj(&self) -> usize {
        self.kda_heads * self.kda_dim
    }
    pub fn mla_qh(&self) -> usize {
        self.mla_nope + self.mla_rope
    }
    pub fn mla_qb(&self) -> usize {
        self.mla_heads * self.mla_qh()
    }
    pub fn mla_kvb(&self) -> usize {
        self.mla_heads * (self.mla_nope + self.mla_v)
    }
    /// kv_a_proj_with_mqa output width: latent KV + the shared rope key.
    /// Coincides with mla_q_lora in the micro config (64+64 == 128) but not
    /// in real K3 (512+64 == 576 vs q_lora 1536).
    pub fn mla_c_dim(&self) -> usize {
        self.mla_kva + self.mla_rope
    }
    /// Flat attention output width H*v (g_proj rows / o_proj cols).
    /// Coincides with d in the micro config (4*128 == 512) but not in real
    /// K3 (96*128 == 12288 vs hidden 7168).
    pub fn mla_hv(&self) -> usize {
        self.mla_heads * self.mla_v
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DeepSeek-V4 (microdeepseek) — micro dims, exact architecture
// ════════════════════════════════════════════════════════════════════════════

/// DeepSeek-V4 configuration (micro scale: same layer count, same expert
/// counts, same attention structure; reduced widths). Parsed from the
/// MKIM0002 config JSON ("arch": "deepseek_v4", fields under "ds").
#[derive(Debug, Clone)]
pub struct DsConfig {
    pub n_layers: usize,       // 43 (kept)
    pub d: usize,              // hidden 512 (real: 4096)
    pub vocab: usize,          // 129280 (real vocab kept)
    pub n_heads: usize,        // 8 (real: 64)
    pub head_dim: usize,       // 128 (real: 512)
    pub rope_head_dim: usize,  // 64 (kept)
    pub q_lora_rank: usize,    // 128 (real: 1024)
    pub o_lora_rank: usize,    // 128 (real: 1024)
    pub o_groups: usize,       // 8 (kept)
    pub window_size: usize,    // 128 (kept)
    pub compress_ratios: Vec<i32>, // per layer: 0,0,4,128,4,128,... (kept)
    pub rope_theta: f64,           // 10000 (window layers)
    pub compress_rope_theta: f64,  // 160000 (compressed layers)
    pub yarn_factor: f64,          // 16
    pub yarn_beta_fast: i32,       // 32
    pub yarn_beta_slow: i32,       // 1
    pub yarn_orig_seq_len: i32,    // 65536
    pub index_n_heads: usize,  // 8 (real: 64)
    pub index_head_dim: usize, // 128 (kept)
    pub index_topk: usize,     // 64 (real: 512)
    pub n_routed_experts: usize, // 256 (kept)
    pub n_activated_experts: usize, // 6 (kept)
    pub moe_inter_dim: usize,     // 128 (real: 2048)
    pub n_hash_layers: usize,     // 3 (kept)
    pub route_scale: f64,         // 1.5
    pub swiglu_limit: f64,        // 10.0
    pub hc_mult: usize,           // 4 (kept)
    pub hc_sinkhorn_iters: usize, // 20 (kept)
    pub hc_eps: f64,              // 1e-6
    pub norm_eps: f64,            // 1e-6
    pub max_seq_len: usize,       // 4096 (micro)
}

impl DsConfig {
    pub fn microdeepseek() -> DsConfig {
        let mut compress_ratios = vec![0i32, 0];
        for i in 2..43 {
            compress_ratios.push(if i % 2 == 0 { 4 } else { 128 });
        }
        compress_ratios.extend_from_slice(&[0, 0, 0]);
        DsConfig {
            n_layers: 43,
            d: 512,
            vocab: 129_280,
            n_heads: 8,
            head_dim: 128,
            rope_head_dim: 64,
            q_lora_rank: 128,
            o_lora_rank: 128,
            o_groups: 8,
            window_size: 128,
            compress_ratios,
            rope_theta: 10000.0,
            compress_rope_theta: 160000.0,
            yarn_factor: 16.0,
            yarn_beta_fast: 32,
            yarn_beta_slow: 1,
            yarn_orig_seq_len: 65536,
            index_n_heads: 8,
            index_head_dim: 128,
            index_topk: 64,
            n_routed_experts: 256,
            n_activated_experts: 6,
            moe_inter_dim: 128,
            n_hash_layers: 3,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            norm_eps: 1e-6,
            max_seq_len: 4096,
        }
    }

    fn num(j: &Json, key: &str, default: f64) -> f64 {
        j.get(key).and_then(|x| x.as_num()).unwrap_or(default)
    }

    pub fn from_json(j: &Json) -> DsConfig {
        let mut c = DsConfig::microdeepseek();
        let d = j.get("ds").unwrap_or(j);
        c.n_layers = Self::num(d, "n_layers", c.n_layers as f64) as usize;
        c.d = Self::num(d, "hidden", c.d as f64) as usize;
        c.vocab = Self::num(d, "vocab", c.vocab as f64) as usize;
        c.n_heads = Self::num(d, "n_heads", c.n_heads as f64) as usize;
        c.head_dim = Self::num(d, "head_dim", c.head_dim as f64) as usize;
        c.rope_head_dim = Self::num(d, "qk_rope_head_dim", c.rope_head_dim as f64) as usize;
        c.q_lora_rank = Self::num(d, "q_lora_rank", c.q_lora_rank as f64) as usize;
        c.o_lora_rank = Self::num(d, "o_lora_rank", c.o_lora_rank as f64) as usize;
        c.o_groups = Self::num(d, "o_groups", c.o_groups as f64) as usize;
        c.window_size = Self::num(d, "sliding_window", c.window_size as f64) as usize;
        if let Some(Json::Arr(a)) = d.get("compress_ratios") {
            c.compress_ratios = a.iter().filter_map(|x| x.as_num().map(|n| n as i32)).collect();
        }
        c.rope_theta = Self::num(d, "rope_theta", c.rope_theta);
        c.compress_rope_theta = Self::num(d, "compress_rope_theta", c.compress_rope_theta);
        c.index_n_heads = Self::num(d, "index_n_heads", c.index_n_heads as f64) as usize;
        c.index_head_dim = Self::num(d, "index_head_dim", c.index_head_dim as f64) as usize;
        c.index_topk = Self::num(d, "index_topk", c.index_topk as f64) as usize;
        c.n_routed_experts = Self::num(d, "n_routed_experts", c.n_routed_experts as f64) as usize;
        c.n_activated_experts = Self::num(d, "num_experts_per_tok", c.n_activated_experts as f64) as usize;
        c.moe_inter_dim = Self::num(d, "moe_intermediate_size", c.moe_inter_dim as f64) as usize;
        c.n_hash_layers = Self::num(d, "num_hash_layers", c.n_hash_layers as f64) as usize;
        c.route_scale = Self::num(d, "routed_scaling_factor", c.route_scale);
        c.swiglu_limit = Self::num(d, "swiglu_limit", c.swiglu_limit);
        c.norm_eps = Self::num(d, "rms_norm_eps", c.norm_eps);
        c
    }

    pub fn compress_ratio(&self, layer: usize) -> i32 {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }
}


/// Qwen3.5-family text decoder. Layers alternate a gated delta-rule
/// linear attention with a full-attention layer every
/// `full_attn_interval`. The MoE variant (qwen3_5_moe_text) carries a
/// softmax-routed expert bank plus one always-on shared expert on every
/// layer; the dense variant (qwen3_5_text, e.g. Qwen3.8-27B) carries a
/// single SiLU-gated MLP of width `dense_inter` instead.
#[derive(Clone, Debug)]
pub struct QwenConfig {
    pub n_layers: usize,
    pub d: usize,
    pub vocab: usize,
    /// full attention: heads, kv heads, head dim, partial rope fraction
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub partial_rotary: f64,
    pub rope_theta: f64,
    /// linear attention: key/value heads and dims, depthwise conv width
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    /// one full-attention layer every `full_attn_interval` layers
    pub full_attn_interval: usize,
    pub n_experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    /// Dense MLP width (HF `intermediate_size`). Zero for MoE decoders;
    /// nonzero marks the dense variant and voids the MoE fields above.
    pub dense_inter: usize,
    /// Converted multi-token-prediction depth (0 = the mtp tensors were
    /// absent or not converted; 1 = one draft layer usable for greedy
    /// self-speculative decoding). Set by the converter, never read from
    /// Hugging Face configs (their key is mtp_num_hidden_layers).
    pub mtp_layers: usize,
    /// The checkpoint ties lm_head to the input embedding (small dense
    /// models, e.g. Qwen3.5-0.8B): the converted file stores the matrix
    /// once and the runtime reads logits through the embedding rows.
    pub tied_embeddings: bool,
    pub norm_eps: f64,
}

impl QwenConfig {
    pub fn qwen35_moe() -> QwenConfig {
        QwenConfig {
            n_layers: 40,
            d: 2048,
            vocab: 248320,
            n_heads: 16,
            n_kv_heads: 2,
            head_dim: 256,
            partial_rotary: 0.25,
            rope_theta: 10_000_000.0,
            lin_k_heads: 16,
            lin_v_heads: 32,
            lin_k_dim: 128,
            lin_v_dim: 128,
            conv_kernel: 4,
            full_attn_interval: 4,
            n_experts: 256,
            top_k: 8,
            moe_inter: 512,
            shared_inter: 512,
            dense_inter: 0,
            mtp_layers: 0,
            tied_embeddings: false,
            norm_eps: 1e-6,
        }
    }

    /// Qwen3.8-27B: the dense multimodal checkpoint's text decoder.
    #[cfg(test)]
    pub fn qwen38_dense() -> QwenConfig {
        QwenConfig {
            n_layers: 64,
            d: 5120,
            vocab: 248320,
            n_heads: 24,
            n_kv_heads: 4,
            head_dim: 256,
            partial_rotary: 0.25,
            rope_theta: 10_000_000.0,
            lin_k_heads: 16,
            lin_v_heads: 48,
            lin_k_dim: 128,
            lin_v_dim: 128,
            conv_kernel: 4,
            full_attn_interval: 4,
            n_experts: 0,
            top_k: 0,
            moe_inter: 0,
            shared_inter: 0,
            dense_inter: 17408,
            mtp_layers: 0,
            tied_embeddings: false,
            norm_eps: 1e-6,
        }
    }

    fn num(j: &Json, key: &str, default: f64) -> f64 {
        j.get(key).and_then(|x| x.as_num()).unwrap_or(default)
    }

    pub fn from_json(j: &Json) -> QwenConfig {
        let mut c = QwenConfig::qwen35_moe();
        let d = j.get("qwen").unwrap_or(j);
        c.n_layers = Self::num(d, "num_hidden_layers", c.n_layers as f64) as usize;
        c.d = Self::num(d, "hidden_size", c.d as f64) as usize;
        c.vocab = Self::num(d, "vocab_size", c.vocab as f64) as usize;
        c.n_heads = Self::num(d, "num_attention_heads", c.n_heads as f64) as usize;
        c.n_kv_heads = Self::num(d, "num_key_value_heads", c.n_kv_heads as f64) as usize;
        c.head_dim = Self::num(d, "head_dim", c.head_dim as f64) as usize;
        // HF checkpoints keep these under rope_parameters; converted
        // MKIM0002 files store the same values flat in the qwen object.
        let rope = d.get("rope_parameters").unwrap_or(d);
        c.partial_rotary = Self::num(rope, "partial_rotary_factor", c.partial_rotary);
        c.rope_theta = Self::num(rope, "rope_theta", c.rope_theta);
        c.lin_k_heads = Self::num(d, "linear_num_key_heads", c.lin_k_heads as f64) as usize;
        c.lin_v_heads = Self::num(d, "linear_num_value_heads", c.lin_v_heads as f64) as usize;
        c.lin_k_dim = Self::num(d, "linear_key_head_dim", c.lin_k_dim as f64) as usize;
        c.lin_v_dim = Self::num(d, "linear_value_head_dim", c.lin_v_dim as f64) as usize;
        c.conv_kernel = Self::num(d, "linear_conv_kernel_dim", c.conv_kernel as f64) as usize;
        c.full_attn_interval = Self::num(d, "full_attention_interval", c.full_attn_interval as f64) as usize;
        c.n_experts = Self::num(d, "num_experts", c.n_experts as f64) as usize;
        c.top_k = Self::num(d, "num_experts_per_tok", c.top_k as f64) as usize;
        c.moe_inter = Self::num(d, "moe_intermediate_size", c.moe_inter as f64) as usize;
        c.shared_inter = Self::num(d, "shared_expert_intermediate_size", c.shared_inter as f64) as usize;
        c.dense_inter = Self::num(d, "intermediate_size", 0.0) as usize;
        c.mtp_layers = Self::num(d, "mtp_layers", 0.0) as usize;
        c.tied_embeddings = Self::num(d, "tie_word_embeddings", 0.0) != 0.0;
        if c.dense_inter > 0 {
            // The dense variant has no router, expert bank, or shared
            // expert; zero the MoE fields so their defaults cannot leak.
            c.n_experts = 0;
            c.top_k = 0;
            c.moe_inter = 0;
            c.shared_inter = 0;
        }
        c.norm_eps = Self::num(d, "rms_norm_eps", c.norm_eps);
        c
    }

    /// Dense decoder (qwen3_5_text): one SiLU MLP per layer, no experts.
    pub fn is_dense(&self) -> bool {
        self.dense_inter > 0
    }

    /// Layers are linear-attention unless their index sits on the full
    /// attention stride (the reference uses index+1 % interval == 0).
    pub fn is_full_attn(&self, layer: usize) -> bool {
        (layer + 1) % self.full_attn_interval == 0
    }

    pub fn lin_key_total(&self) -> usize { self.lin_k_heads * self.lin_k_dim }
    pub fn lin_value_total(&self) -> usize { self.lin_v_heads * self.lin_v_dim }
    pub fn rope_dim(&self) -> usize {
        ((self.head_dim as f64 * self.partial_rotary) as usize) / 2 * 2
    }
}
