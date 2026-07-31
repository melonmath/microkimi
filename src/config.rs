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
        c
    }

    pub fn is_mla(&self, l: usize) -> bool {
        l % 4 == 3 || l == self.n_layers - 1
    }
    pub fn is_moe(&self, l: usize) -> bool {
        l >= self.first_k_dense
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
}
