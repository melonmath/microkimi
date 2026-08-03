#!/usr/bin/env python3
"""nanokimi - export of a trained checkpoint to the extended microkimi.bin
format (MKIM0002: magic + explicit JSON config block + directory + data).

Experts are MXFP4-quantized (e2m1 + e8m0 scale per 32, same rule as
src/mxfp4.rs) to stay compatible with the MoE path of the Rust engine.

usage: python3 export.py --ckpt out_dev/ckpt/ckpt_0000200.pt --out nanokimi.bin
"""
import argparse
import json
import os

import numpy as np
import torch

MAGIC = b"MKIM0002"
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def quantize_mxfp4(w):
    """w f32 [R, C] (C % 32 == 0) → (packed u8 [R, C/2], scales u8 [R, C/32]).
    Same rule as src/mxfp4.rs: scale_exp = max(-127, ceil(log2(maxabs/6))),
    nearest e2m1 level to v/2^e (midpoint cutoffs)."""
    r, c = w.shape
    g = w.reshape(r, c // 32, 32)
    maxabs = np.abs(g).max(axis=-1)
    with np.errstate(divide="ignore"):
        e = np.maximum(np.ceil(np.log2(maxabs / 6.0)), -127)
    e = np.where(maxabs == 0.0, -127.0, e).clip(-127, 128)
    scales = (e + 127).clip(0, 255).astype(np.uint8)
    inv = np.exp2(-e)[..., None]
    q = (g * inv).clip(-6.0, 6.0)
    mag = np.abs(q)
    idx = (mag[..., None] >= BOUNDS).sum(axis=-1).astype(np.uint8)  # 0..7
    idx = np.where(np.signbit(q), idx + 8, idx)
    packed = idx[..., 0::2] | (idx[..., 1::2] << 4)
    return packed.reshape(r, c // 2), scales


def write_bin(path, config, tensors):
    """tensors: ordered list of (name, dtype, dims, blob_bytes)."""
    dir_size = sum(2 + len(n.encode()) + 1 + 1 + 4 * len(d) + 8 + 8 for n, _, d, _ in tensors)
    cfg_bytes = json.dumps(config).encode()
    data_start = 8 + 4 + len(cfg_bytes) + 4 + dir_size
    pos = data_start
    offsets = []
    for _, dt, dims, blob in tensors:
        pos = (pos + 63) // 64 * 64
        offsets.append(pos)
        pos += len(blob)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(len(cfg_bytes).to_bytes(4, "little"))
        f.write(cfg_bytes)
        f.write(len(tensors).to_bytes(4, "little"))
        for (name, dt, dims, blob), off in zip(tensors, offsets):
            nb = name.encode()
            f.write(len(nb).to_bytes(2, "little"))
            f.write(nb)
            f.write(bytes([dt, len(dims)]))
            for d in dims:
                f.write(int(d).to_bytes(4, "little"))
            f.write(off.to_bytes(8, "little"))
            f.write(len(blob).to_bytes(8, "little"))
        cur = f.tell()
        for (_, _, _, blob), off in zip(tensors, offsets):
            if cur < off:
                f.write(b"\0" * (off - cur))
                cur = off
            f.write(blob)
            cur += len(blob)
    return pos


def f32_blob(t):
    return np.ascontiguousarray(t.detach().float().numpy()).tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    c = ck["cfg"]
    n_layers, D, V = c["n_layers"], c["hidden"], c["vocab"]

    config = {
        "format": 2, "n_layers": n_layers, "hidden": D, "vocab": V,
        "n_experts": c["n_experts"], "top_k": c["top_k"], "n_shared": c["n_shared"],
        "kda_heads": c["kda_heads"], "kda_dim": c["kda_dim"], "kda_conv": c["kda_conv"],
        "kda_fa_rank": c["kda_fa_rank"], "gate_lower_bound": c["gate_lower_bound"],
        "mla_heads": c["mla_heads"], "mla_q_lora": c["mla_q_lora"], "mla_kv_lora": c["mla_kv_lora"],
        "mla_nope": c["mla_nope"], "mla_rope": c["mla_rope"], "mla_v": c["mla_v"],
        "routed_hidden": c["routed_hidden"], "moe_inter": c["moe_inter"],
        "shared_inter": c["shared_inter"], "dense_inter": c["dense_inter"],
        "attn_res_block": c["attn_res_block"], "first_k_dense": c["first_k_dense"],
        "rms_eps": c["rms_eps"],
        "tokenizer": "kimi-remap-nano",
        "specials": {"bos": 8192, "eos": 8193, "open": 8194, "close": 8195,
                     "sep": 8196, "end_of_msg": 8197, "unk": 8198, "pad": 8199},
    }
    # explicit layer-type lists (self-describing config, same convention as
    # the Rust slicer; bin2pt and the Rust engine read them back)
    mla_set = set(c["mla_layers"]) if "mla_layers" in c else {l for l in range(n_layers) if l % 4 == 3 or l == n_layers - 1}
    dense_set = set(c["dense_layers"]) if "dense_layers" in c else set(range(c["first_k_dense"]))
    config["mla_layers"] = sorted(mla_set)
    config["dense_layers"] = sorted(dense_set)

    tensors = []

    def add(name, tensor, dtype=0, dims=None):
        if dtype == 0:
            t = tensor.detach().float().numpy()
            tensors.append((name, 0, list(t.shape), np.ascontiguousarray(t).tobytes()))
        else:
            w = tensor.detach().float().numpy()
            packed, scales = quantize_mxfp4(w)
            tensors.append((name, 1, list(w.shape), packed.tobytes() + scales.tobytes()))

    def get(name):
        return sd[name]

    add("embed_tokens.weight", get("embed_tokens.weight"))
    add("lm_head.weight", get("lm_head.weight"))
    add("norm.weight", get("norm.weight"))
    add("output_attn_res_norm.weight", get("output_attn_res_norm.weight"))
    add("output_attn_res_proj.weight", get("output_attn_res_proj.weight"))

    is_mla = lambda l: l in mla_set
    for l in range(n_layers):
        p = f"layers.{l}."
        add(p + "input_layernorm.weight", get(p + "input_layernorm.weight"))
        add(p + "post_attention_layernorm.weight", get(p + "post_attention_layernorm.weight"))
        add(p + "self_attention_res_norm.weight", get(p + "self_attention_res_norm.weight"))
        add(p + "self_attention_res_proj.weight", get(p + "self_attention_res_proj.weight"))
        add(p + "mlp_res_norm.weight", get(p + "mlp_res_norm.weight"))
        add(p + "mlp_res_proj.weight", get(p + "mlp_res_proj.weight"))
        a = p + "self_attn."
        if is_mla(l):
            add(a + "q_a_proj.weight", get(a + "q_a_proj.weight"))
            add(a + "q_a_layernorm.weight", get(a + "q_a_layernorm.weight"))
            add(a + "q_b_proj.weight", get(a + "q_b_proj.weight"))
            add(a + "kv_a_proj_with_mqa.weight", get(a + "kv_a_proj_with_mqa.weight"))
            add(a + "kv_a_layernorm.weight", get(a + "kv_a_layernorm.weight"))
            add(a + "kv_b_proj.weight", get(a + "kv_b_proj.weight"))
            add(a + "g_proj.weight", get(a + "g_proj.weight"))
            add(a + "o_proj.weight", get(a + "o_proj.weight"))
        else:
            for x in ["q_proj", "k_proj", "v_proj", "g_proj", "o_proj"]:
                add(a + x + ".weight", get(a + x + ".weight"))
            for x in ["q_conv1d", "k_conv1d", "v_conv1d"]:
                # nn.Conv1d [C,1,K] → [C,K]
                add(a + x + ".weight", get(a + x + ".weight").reshape(D, -1))
            add(a + "f_a_proj.weight", get(a + "f_a_proj.weight"))
            add(a + "f_b_proj.weight", get(a + "f_b_proj.weight"))
            add(a + "A_log", get(a + "A_log"))
            add(a + "dt_bias", get(a + "dt_bias"))
            add(a + "b_proj.weight", get(a + "b_proj.weight"))
            add(a + "o_norm.weight", get(a + "o_norm.weight"))
        if l not in dense_set:
            m = p + "block_sparse_moe."
            add(m + "gate.weight", get(m + "gate.weight"))
            add(m + "gate.e_score_correction_bias", get(m + "gate.e_score_correction_bias"))
            add(m + "routed_expert_down_proj.weight", get(m + "routed_expert_down_proj.weight"))
            add(m + "routed_expert_up_proj.weight", get(m + "routed_expert_up_proj.weight"))
            add(m + "routed_expert_norm.weight", get(m + "routed_expert_norm.weight"))
            add(m + "shared_experts.gate_proj.weight", get(m + "shared_experts.gate_proj.weight"))
            add(m + "shared_experts.up_proj.weight", get(m + "shared_experts.up_proj.weight"))
            add(m + "shared_experts.down_proj.weight", get(m + "shared_experts.down_proj.weight"))
            for e in range(c["n_experts"]):
                ep = m + f"experts.{e}."
                add(ep + "w1", get(ep + "w1.weight"), dtype=1)
                add(ep + "w2", get(ep + "w2.weight"), dtype=1)
                add(ep + "w3", get(ep + "w3.weight"), dtype=1)
        else:
            add(p + "mlp.gate_proj.weight", get(p + "mlp.gate_proj.weight"))
            add(p + "mlp.up_proj.weight", get(p + "mlp.up_proj.weight"))
            add(p + "mlp.down_proj.weight", get(p + "mlp.down_proj.weight"))
        if l % 4 == 0:
            print(f"  layer {l + 1}/{n_layers} exported", flush=True)

    size = write_bin(args.out, config, tensors)
    print(f"→ {args.out} : {size / 1e6:.0f} MB, {len(tensors)} tensors, step {ck['step']}", flush=True)


if __name__ == "__main__":
    main()
