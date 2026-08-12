//! Converts a qwen3_5_moe checkpoint (safetensors, text path) into the
//! engine's MKIM0002 container.
//!
//! Reads the reference layout - fused expert banks `gate_up_proj`
//! [E, 2*inter, d] and `down_proj` [E, d, inter], a shared expert, a
//! router, and per-layer attention weights that differ between linear
//! and full attention layers - and writes one tensor per logical piece,
//! experts quantized to MXFP4 like every other expert bank in the
//! format. Vision tensors and the multi-token-prediction head are
//! skipped: this is the text decoder only.
//!
//! This module holds the conversion PLAN: which tensor each layer needs
//! given its attention kind, and the config block the engine reads back.
//! The byte-level write reuses the existing BinWriter two-phase layout
//! (header then blobs) and the MXFP4 quantizer, as build_ds does.

use crate::config::QwenConfig;
#[allow(unused_imports)]
use crate::json::Json;

/// Names the converter reads per layer, by attention kind.
pub fn layer_tensors(c: &QwenConfig, l: usize) -> Vec<String> {
    let p = format!("model.language_model.layers.{}", l);
    let mut v = vec![
        format!("{}.input_layernorm.weight", p),
        format!("{}.post_attention_layernorm.weight", p),
        format!("{}.mlp.gate.weight", p),
        format!("{}.mlp.experts.gate_up_proj", p),
        format!("{}.mlp.experts.down_proj", p),
        format!("{}.mlp.shared_expert.gate_proj.weight", p),
        format!("{}.mlp.shared_expert.up_proj.weight", p),
        format!("{}.mlp.shared_expert.down_proj.weight", p),
        format!("{}.mlp.shared_expert_gate.weight", p),
    ];
    if c.is_full_attn(l) {
        for n in ["q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"] {
            v.push(format!("{}.self_attn.{}.weight", p, n));
        }
    } else {
        for n in ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "conv1d", "norm"] {
            v.push(format!("{}.linear_attn.{}.weight", p, n));
        }
        v.push(format!("{}.linear_attn.A_log", p));
        v.push(format!("{}.linear_attn.dt_bias", p));
    }
    v
}

/// Emits the MKIM0002 config block for a converted checkpoint.
pub fn config_json(c: &QwenConfig, tokenizer: &str) -> String {
    format!(
        "{{\"format\":2,\"arch\":\"qwen3_5_moe\",\"n_layers\":{},\"hidden\":{},\"vocab\":{},\
         \"tokenizer\":\"{}\",\"qwen\":{{\"num_hidden_layers\":{},\"hidden_size\":{},\
         \"vocab_size\":{},\"num_attention_heads\":{},\"num_key_value_heads\":{},\
         \"head_dim\":{},\"partial_rotary_factor\":{},\"rope_theta\":{},\
         \"linear_num_key_heads\":{},\"linear_num_value_heads\":{},\"linear_key_head_dim\":{},\
         \"linear_value_head_dim\":{},\"linear_conv_kernel_dim\":{},\"full_attention_interval\":{},\
         \"num_experts\":{},\"num_experts_per_tok\":{},\"moe_intermediate_size\":{},\
         \"shared_expert_intermediate_size\":{},\"rms_norm_eps\":{}}}}}",
        c.n_layers, c.d, c.vocab, tokenizer,
        c.n_layers, c.d, c.vocab, c.n_heads, c.n_kv_heads, c.head_dim,
        c.partial_rotary, c.rope_theta, c.lin_k_heads, c.lin_v_heads,
        c.lin_k_dim, c.lin_v_dim, c.conv_kernel, c.full_attn_interval,
        c.n_experts, c.top_k, c.moe_inter, c.shared_inter, c.norm_eps
    )
}

/// Reads the checkpoint's own config.json into a QwenConfig.
pub fn read_hf_config(dir: &str) -> QwenConfig {
    let path = format!("{}/config.json", dir.trim_end_matches('/'));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("{}: cannot read", path));
    let j = crate::json::parse(raw.as_bytes());
    let t = j.get("text_config").unwrap_or(&j);
    QwenConfig::from_json(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_tensor_sets_follow_the_attention_kind() {
        let c = QwenConfig::qwen35_moe();
        // interval 4: layer 3 is full attention, layer 0 is linear
        assert!(c.is_full_attn(3) && !c.is_full_attn(0));
        let full = layer_tensors(&c, 3);
        assert!(full.iter().any(|n| n.ends_with("self_attn.q_proj.weight")));
        assert!(!full.iter().any(|n| n.contains("linear_attn")));
        let lin = layer_tensors(&c, 0);
        assert!(lin.iter().any(|n| n.ends_with("linear_attn.A_log")));
        assert!(!lin.iter().any(|n| n.contains("self_attn")));
        // both carry the expert bank and the shared expert
        for set in [&full, &lin] {
            assert!(set.iter().any(|n| n.ends_with("experts.gate_up_proj")));
            assert!(set.iter().any(|n| n.ends_with("shared_expert.down_proj.weight")));
        }
    }

    #[test]
    fn config_block_round_trips() {
        let mut c = QwenConfig::qwen35_moe();
        c.n_layers = 12;
        c.n_experts = 64;
        c.top_k = 4;
        let s = config_json(&c, "qwen");
        let j = crate::json::parse(s.as_bytes());
        assert_eq!(j.get("arch").and_then(|x| x.as_str()), Some("qwen3_5_moe"));
        let back = QwenConfig::from_json(j.get("qwen").unwrap());
        assert_eq!(back.n_layers, 12);
        assert_eq!(back.n_experts, 64);
        assert_eq!(back.top_k, 4);
        assert_eq!(back.full_attn_interval, c.full_attn_interval);
        assert_eq!(back.lin_v_heads, c.lin_v_heads);
    }
}
