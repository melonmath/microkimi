//! By-reference views of a Qwen model that the GPU decoders consume
//! (Metal on macOS, CUDA on Linux): plain data, no platform code.

/// Weight references of one linear-attention layer (f32 spine tensors as
/// stored, MXFP4 MLP as (packed nibbles, e8m0 scales)).
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub struct LinLayerRefs<'a> {
    pub in_qkv: &'a [f32],
    pub in_z: &'a [f32],
    pub in_b: &'a [f32],
    pub in_a: &'a [f32],
    pub conv_w: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    pub norm_w: &'a [f32],
    pub out_proj: &'a [f32],
    pub post_norm_w: &'a [f32],
    pub gate: (&'a [u8], &'a [u8]),
    pub up: (&'a [u8], &'a [u8]),
    pub down: (&'a [u8], &'a [u8]),
}

/// Dimensions of a linear layer.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub struct LinDims {
    pub d: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub kd: usize,
    pub vd: usize,
    pub conv_k: usize,
    pub inter: usize,
    pub eps: f32,
}

/// What the decoder needs from the model, by reference.
pub struct DecodeModelRefs<'a> {
    pub layers: Vec<DecodeLayerRefs<'a>>,
    pub embed: &'a [f32],     // [vocab, d]
    pub norm_f: &'a [f32],    // [d]
    pub lm_head: &'a [f32],   // [vocab, d]
    pub d: usize,
    pub vocab: usize,
    pub eps: f32,
}

pub enum DecodeLayerRefs<'a> {
    Linear {
        in_norm: &'a [f32],
        post_norm: &'a [f32],
        w: LinLayerRefs<'a>,
        gated_w: &'a [f32],
        dm: LinDims,
    },
    Full {
        in_norm: &'a [f32],
        post_norm: &'a [f32],
        q_proj: &'a [f32],
        k_proj: &'a [f32],
        v_proj: &'a [f32],
        o_proj: &'a [f32],
        q_norm: &'a [f32],
        k_norm: &'a [f32],
        gate: (&'a [u8], &'a [u8]),
        up: (&'a [u8], &'a [u8]),
        down: (&'a [u8], &'a [u8]),
        n_heads: usize,
        n_kv: usize,
        hd: usize,
        rope_dim: usize,
        theta: f32,
        inter: usize,
    },
}

