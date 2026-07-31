#!/usr/bin/env python3
"""End-to-end microkimi parity (TEST tool, not a product dependency).

Drives the REAL Moonshot code (modeling_kimi_linear.py - Kimi K3, Moonshot
AI / DeepSeek-AI / HuggingFace, see the file header; downloaded at runtime
into ~/.cache/microkimi/moonshot, never vendored) layer by layer with the
micro dims of SPEC §1 and the weights of microkimi.bin, in strict fp32.

The manual driving (embed → KimiDecoderLayer × 93 → output attn_res → norm →
lm_head) mirrors KimiLinearModel.forward exactly, but keeps only ONE
dequantized expert layer in RAM at a time (~90 MB) - the full state_dict
would be ~9 GB (OOM). All the mathematics are done by the real Moonshot
classes: KimiRMSNorm, KimiDeltaAttention, KimiMLAAttention,
KimiSparseMoeBlock, _apply_attn_res, F.linear.

Prerequisites (test venv): torch, numpy, transformers==4.56.2, einops
+ fla shim vendored in nano/vendor/fla (MIT).
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "nano", "vendor"))  # shim fla (MIT, vendored)
sys.path.insert(0, _HERE)

import numpy as np
import torch
import torch.nn.functional as F

from fetch_moonshot import ensure_moonshot
from mxfp4_numpy import dequant_mxfp4

# Moonshot reference files: downloaded at runtime, never vendored (Moonshot license)
_MS_CACHE = ensure_moonshot()
sys.path.insert(0, _MS_CACHE)
from moonshot.configuration_kimi_k3 import KimiLinearConfig  # noqa: E402
from moonshot.modeling_kimi_linear import (  # noqa: E402
    KimiDecoderLayer,
    KimiRMSNorm,
    _apply_attn_res,
)

BIN = os.environ.get("MICROKIMI_BIN", os.path.join(_HERE, "..", "microkimi.bin"))
OUT = os.path.join(_HERE, "parity_golden.json")
IDS = [163584, 100, 200, 300, 400]  # [BOS] + 4 arbitrary ids
DUMP_LAYERS = [0, 1, 3, 4, 12, 47, 92]
ROUTER_LAYERS = [1, 47, 92]

# ── microkimi.bin parser (SPEC §4), memmap to avoid the RAM copy ──
def load_bin(path):
    data = np.memmap(path, dtype=np.uint8, mode="r")
    assert data[:8].tobytes() == b"MKIM0001", "bad magic"
    n = int.from_bytes(data[8:12], "little")
    off = 12
    tensors = {}
    for _ in range(n):
        nl = int.from_bytes(data[off:off + 2], "little"); off += 2
        name = data[off:off + nl].tobytes().decode(); off += nl
        dt, nd = int(data[off]), int(data[off + 1]); off += 2
        dims = [int.from_bytes(data[off + 4 * i:off + 4 * i + 4], "little") for i in range(nd)]
        off += 4 * nd
        o = int.from_bytes(data[off:off + 8], "little")
        s = int.from_bytes(data[off + 8:off + 16], "little"); off += 16
        tensors[name] = (dt, dims, o, s)
    return data, tensors

def tensor_f32(data, dt, dims, o, s):
    if dt == 0:  # f32
        cnt = 1
        for d in dims:
            cnt *= d
        arr = np.frombuffer(data, dtype=np.float32, count=cnt, offset=o)
        return np.ascontiguousarray(arr.reshape(dims))
    # mxfp4: packed (R×C/2) then scales (R×C/32)
    r, c = dims
    packed = np.frombuffer(data, dtype=np.uint8, count=r * c // 2, offset=o).reshape(r, c // 2)
    scales = np.frombuffer(data, dtype=np.uint8, count=r * c // 32, offset=o + r * c // 2).reshape(r, c // 32)
    return dequant_mxfp4(packed, scales)

def is_mla(l):
    return l % 4 == 3 or l == 92

# ── micro config (SPEC §1) ──
def micro_config():
    full_attn = list(range(4, 94, 4)) + [93]        # 1-based {4,8,…,92,93}
    kda_layers = [i for i in range(1, 94) if i not in full_attn]
    return KimiLinearConfig(
        vocab_size=163840, hidden_size=512, intermediate_size=2048,
        num_hidden_layers=93, num_attention_heads=4, num_key_value_heads=4,
        hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
        rms_norm_eps=1e-5, tie_word_embeddings=False,
        max_position_embeddings=1048576,
        q_lora_rank=128, kv_lora_rank=64,
        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
        mla_use_nope=True, mla_use_output_gate=True,
        num_experts=896, num_experts_per_token=16, num_shared_experts=2,
        routed_expert_hidden_size=128, moe_intermediate_size=64,
        latent_moe_use_norm=True,
        moe_renormalize=True, moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0, first_k_dense_replace=1, moe_layer_freq=1,
        use_grouped_topk=True, num_expert_group=1, topk_group=1,
        linear_attn_config={
            "full_attn_layers": full_attn, "kda_layers": kda_layers,
            "head_dim": 128, "num_heads": 4, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0, "use_full_rank_gate": True,
        },
        attn_res_block_size=12,
        _attn_implementation="eager",
    )

def load_layer(config, data, tensors, l):
    """Instantiate a real KimiDecoderLayer and load its weights (strict)."""
    layer = KimiDecoderLayer(config, l).float().eval()
    pfx = f"layers.{l}."
    sd = {}
    for name, (dt, dims, o, s) in tensors.items():
        if not name.startswith(pfx):
            continue
        key = name[len(pfx):]
        if ".experts." in key:  # the bin does not have the .weight suffix on experts
            key += ".weight"
        t = torch.from_numpy(tensor_f32(data, dt, dims, o, s))
        if "conv1d" in name:  # nn.Conv1d : [channels, 1, kernel]
            t = t.reshape(t.shape[0], 1, t.shape[1])
        elif "res_proj" in name:  # nn.Linear(D, 1) : [1, D]
            t = t.reshape(1, t.shape[0])
        sd[key] = t
    # A_log: the K3 checkpoint is per-CHANNEL [128] (the fla shim accepts [H] or [K]);
    # we replace the module's [H] parameter with the [128] version.
    if "self_attn.A_log" in sd:
        layer.self_attn.A_log = torch.nn.Parameter(sd.pop("self_attn.A_log"))
    missing, unexpected = layer.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "self_attn.A_log"]  # assigned directly above
    assert not missing and not unexpected, f"layer {l}: missing={missing[:5]} unexpected={unexpected[:5]}"
    return layer

def main():
    torch.set_grad_enabled(False)
    print("loading microkimi.bin …")
    data, tensors = load_bin(BIN)
    print(f"  {len(tensors)} tensors")
    config = micro_config()
    T = len(IDS)

    # embedding (no scale)
    embed = torch.from_numpy(tensor_f32(data, *tensors["embed_tokens.weight"]))
    ids = torch.tensor([IDS], dtype=torch.long)
    hidden = embed[ids]  # [1, T, 512]

    # additive causal mask for the MLA layers (eager attention)
    causal = torch.zeros(1, 1, T, T)
    causal.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))

    # final norm / output attn_res / lm_head (real modules)
    norm_f = KimiRMSNorm(512, eps=1e-5)
    norm_f.weight.data = torch.from_numpy(tensor_f32(data, *tensors["norm.weight"]))
    out_norm = KimiRMSNorm(512, eps=1e-5)
    out_norm.weight.data = torch.from_numpy(tensor_f32(data, *tensors["output_attn_res_norm.weight"]))
    out_proj = torch.nn.Linear(512, 1, bias=False)
    out_proj.weight.data = torch.from_numpy(tensor_f32(data, *tensors["output_attn_res_proj.weight"])).reshape(1, 512)
    lm_head_w = torch.from_numpy(tensor_f32(data, *tensors["lm_head.weight"]))

    blocks = hidden.new_zeros(T, 0, 512)  # block_residual (num_tokens, 0, D)
    dumps = {"hiddens": {}, "router": {}, "l1_attn": None, "l1_routed": None, "l1_shared": None}

    for l in range(93):
        layer = load_layer(config, data, tensors, l)
        if l == 1:
            layer.self_attn.register_forward_hook(
                lambda mod, args, out: dumps.__setitem__("l1_attn", out[0].detach().float().numpy()))
            layer.block_sparse_moe.routed_expert_up_proj.register_forward_hook(
                lambda mod, args, out: dumps.__setitem__("l1_routed", out.detach().float().numpy()))
            layer.block_sparse_moe.shared_experts.register_forward_hook(
                lambda mod, args, out: dumps.__setitem__("l1_shared", out.detach().float().numpy()))
        if l in ROUTER_LAYERS:
            def make_ghook(layer_idx):
                def hook(mod, args, out):
                    idx, _w = out
                    dumps["router"][layer_idx] = torch.sort(idx, dim=-1).values.numpy()  # [T, 16] sorted
                return hook
            layer.block_sparse_moe.gate.register_forward_hook(make_ghook(l))

        mask = causal if is_mla(l) else None
        hidden, blocks = layer._forward_attn_residual(
            hidden, attention_mask=mask, block_residual=blocks)
        if l in DUMP_LAYERS:
            dumps["hiddens"][l] = hidden[0].detach().float().numpy()  # [T, D]
        if l % 20 == 0:
            print(f"  layer {l}/93 …")
        del layer

    # output attn_res then final norm then lm_head (real functions/modules)
    hidden = _apply_attn_res(hidden.view(-1, 512), blocks, out_proj, out_norm).view(1, T, 512)
    hidden = norm_f(hidden)
    logits = F.linear(hidden, lm_head_w)[0].numpy()  # [T, vocab]

    golden = {
        "ids": IDS,
        "logits_last": logits[-1].tolist(),
        "logits_top": {
            str(p): sorted(enumerate(logits[p].tolist()), key=lambda kv: -kv[1])[:16]
            for p in range(T)
        },
        "hiddens": {str(l): dumps["hiddens"][l].flatten().tolist() for l in DUMP_LAYERS},
        "l1_attn": dumps["l1_attn"].flatten().tolist(),
        "l1_routed": dumps["l1_routed"].flatten().tolist(),
        "l1_shared": dumps["l1_shared"].flatten().tolist(),
        "router": {str(l): dumps["router"][l].tolist() for l in ROUTER_LAYERS},
    }
    with open(OUT, "w") as f:
        json.dump(golden, f)
    print(f"→ {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
