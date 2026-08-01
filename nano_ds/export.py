#!/usr/bin/env python3
"""nanodeepseek - export of a trained checkpoint to MKIM0002 V4 (the format
the Rust engine reads: magic + JSON config block {"arch":"deepseek_v4"} +
directory + data).

Experts are fp4-quantized (e2m1 + ue8m0 scale per 32, same rule as
src/mxfp4.rs - identical to nano/export.py's MXFP4 path). Tensor names match
DsModel::load in src/deepseek.rs exactly.

usage: python3 export.py --ckpt out/ckpt/ckpt_0000200.pt --out nanodeepseek.bin
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


def dequant_mxfp4(packed, scale):
    """packed u8 [R, C/2], scale u8 [R, C/32] -> float32 [R, C] (test helper)."""
    r, half = packed.shape
    cols = half * 2
    idx = np.empty((r, cols), dtype=np.uint8)
    idx[:, 0::2] = packed & 0x0F
    idx[:, 1::2] = packed >> 4
    vals = E2M1[idx]
    sc = np.exp2(scale.astype(np.int32) - 127)
    vals *= np.repeat(sc, 32, axis=1)
    return vals


def write_bin(path, config, tensors):
    """tensors: ordered list of (name, dtype, dims, blob_bytes).
    dtype: 0=f32, 1=mxfp4, 2=i32."""
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


def ds_config_json(c):
    """MKIM0002 config block for arch deepseek_v4 (read by DsConfig::from_json)."""
    return {
        "arch": "deepseek_v4",
        "vocab": c["vocab"],
        "n_layers": c["n_layers"],
        "specials": {"bos": 8192, "end_of_msg": 8193},
        "tokenizer": "deepseek-v4-remap-nano",
        "ds": {
            "n_layers": c["n_layers"], "hidden": c["hidden"], "vocab": c["vocab"],
            "n_heads": c["n_heads"], "head_dim": c["head_dim"],
            "qk_rope_head_dim": c["rope_head_dim"],
            "q_lora_rank": c["q_lora_rank"], "o_lora_rank": c["o_lora_rank"],
            "o_groups": c["o_groups"], "sliding_window": c["window_size"],
            "compress_ratios": c["compress_ratios"],
            "rope_theta": c["rope_theta"], "compress_rope_theta": c["compress_rope_theta"],
            "index_n_heads": c["index_n_heads"], "index_head_dim": c["index_head_dim"],
            "index_topk": c["index_topk"],
            "n_routed_experts": c["n_routed_experts"],
            "num_experts_per_tok": c["top_k"],
            "moe_intermediate_size": c["moe_inter"],
            "num_hash_layers": c["n_hash_layers"],
            "routed_scaling_factor": c["route_scale"],
            "swiglu_limit": c["swiglu_limit"],
            "rms_norm_eps": 1e-6,
        },
    }


def export_model(sd, c, out_path, verbose=True):
    """sd: model state_dict (torch tensors), c: the NANO_DS config dict."""
    n_layers = c["n_layers"]
    config = ds_config_json(c)
    tensors = []

    def add(name, tensor):
        t = tensor.detach().float().cpu().numpy()
        tensors.append((name, 0, list(t.shape), np.ascontiguousarray(t).tobytes()))

    def add_fp4(name, tensor):
        w = tensor.detach().float().cpu().numpy()
        packed, scales = quantize_mxfp4(w)
        tensors.append((name, 1, list(w.shape), packed.tobytes() + scales.tobytes()))

    def add_i32(name, tensor):
        t = tensor.detach().cpu().numpy().astype(np.int32)
        tensors.append((name, 2, list(t.shape), np.ascontiguousarray(t).tobytes()))

    add("embed.weight", sd["embed.weight"])
    add("head.weight", sd["head.weight"])
    add("norm.weight", sd["norm_w"])
    add("hc_head_fn", sd["hc_head_fn"])
    add("hc_head_base", sd["hc_head_base"])
    add("hc_head_scale", sd["hc_head_scale"])

    for l in range(n_layers):
        p = f"layers.{l}."
        a = p + "attn."
        ratio = c["compress_ratios"][l]
        add(p + "attn_norm.weight", sd[p + "attn_norm"])
        add(p + "ffn_norm.weight", sd[p + "ffn_norm"])
        for kind in ("attn", "ffn"):
            add(p + f"hc_{kind}_fn", sd[p + f"hc_{kind}_fn"])
            add(p + f"hc_{kind}_base", sd[p + f"hc_{kind}_base"])
            add(p + f"hc_{kind}_scale", sd[p + f"hc_{kind}_scale"])
        add(a + "wq_a.weight", sd[a + "wq_a"])
        add(a + "q_norm.weight", sd[a + "q_norm"])
        add(a + "wq_b.weight", sd[a + "wq_b"])
        add(a + "wkv.weight", sd[a + "wkv"])
        add(a + "kv_norm.weight", sd[a + "kv_norm"])
        add(a + "wo_a.weight", sd[a + "wo_a"])
        add(a + "wo_b.weight", sd[a + "wo_b"])
        add(a + "attn_sink", sd[a + "attn_sink"])
        if ratio > 0:
            add(a + "compressor.wkv.weight", sd[a + "comp_wkv"])
            add(a + "compressor.wgate.weight", sd[a + "comp_wgate"])
            add(a + "compressor.ape", sd[a + "comp_ape"])
            add(a + "compressor.norm.weight", sd[a + "comp_norm"])
        if ratio == 4:
            add(a + "indexer.wq_b.weight", sd[a + "idx_wq_b"])
            add(a + "indexer.weights_proj.weight", sd[a + "idx_weights_proj"])
            add(a + "indexer.compressor.wkv.weight", sd[a + "idx_comp_wkv"])
            add(a + "indexer.compressor.wgate.weight", sd[a + "idx_comp_wgate"])
            add(a + "indexer.compressor.ape", sd[a + "idx_comp_ape"])
            add(a + "indexer.compressor.norm.weight", sd[a + "idx_comp_norm"])
        m = p + "moe."
        add(p + "ffn.gate.weight", sd[m + "gate_w"])
        if l < c["n_hash_layers"]:
            add_i32(p + "ffn.gate.tid2eid", sd[m + "tid2eid"])
        else:
            add(p + "ffn.gate.bias", sd[m + "gate_bias"])
        add(p + "ffn.shared_experts.w1.weight", sd[m + "sh1"])
        add(p + "ffn.shared_experts.w2.weight", sd[m + "sh2"])
        add(p + "ffn.shared_experts.w3.weight", sd[m + "sh3"])
        w1, w2, w3 = sd[m + "w1"], sd[m + "w2"], sd[m + "w3"]
        for e in range(c["n_routed_experts"]):
            add_fp4(p + f"ffn.experts.{e}.w1", w1[e])
            add_fp4(p + f"ffn.experts.{e}.w2", w2[e])
            add_fp4(p + f"ffn.experts.{e}.w3", w3[e])
        if verbose and l % 2 == 0:
            print(f"  layer {l + 1}/{n_layers} exported", flush=True)

    size = write_bin(out_path, config, tensors)
    if verbose:
        print(f"→ {out_path} : {size / 1e6:.1f} MB, {len(tensors)} tensors", flush=True)
    return size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    export_model(ck["model"], ck["cfg"], args.out)
    print(f"  (from step {ck['step']})", flush=True)


if __name__ == "__main__":
    main()
