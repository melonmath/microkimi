#!/usr/bin/env python3
"""Build a tiny deterministic Qwen3.5-family checkpoint and reference logits.

Default: the MoE text decoder (qwen3_5_moe_text). With ``--dense``: the dense
text decoder (qwen3_5_text) used by Qwen3.8-27B.

This is a development parity fixture for ``microkimi convert-qwen`` and the
Rust decoder. It requires PyTorch, Transformers, and safetensors, but none of
those packages are runtime dependencies of microkimi.
"""

import argparse
import json
import math
import pathlib
import struct

import torch
from safetensors.torch import save_file
from transformers import (
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
)


TOKENS = [3, 5, 7, 11]
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def config():
    rope = {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.5,
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
    }
    return Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_parameters=rope,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=2,
        use_cache=True,
    )


def dense_config():
    rope = {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 0.5,
        "mrope_interleaved": True,
        "mrope_section": [2, 1, 1],
    }
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_parameters=rope,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        use_cache=True,
    )


def exact_mxfp4_values(shape, generator=None):
    numel = 1
    for size in shape:
        numel *= size
    values = E2M1.repeat((numel + 15) // 16)[:numel]
    return (values * (2.0 ** -5)).reshape(shape)


def make_mtp_tensors(cfg):
    """Synthetic multi-token-prediction tensors for the dense fixture.

    Transformers does not implement the MTP module (its keys are ignored on
    load), so the reference logits for these tensors are computed by
    ``mtp_reference`` below, which mirrors the deployed proposer semantics:
    slot i merges the embedding of token i+1 with the trunk's final-norm
    hidden of position i at rotary position i.
    """
    generator = torch.Generator().manual_seed(0x51478)
    d = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv = cfg.num_key_value_heads
    hd = cfg.head_dim
    inter = cfg.intermediate_size

    def randn(*shape, scale=0.05):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale

    return {
        "mtp.fc.weight": randn(d, 2 * d),
        "mtp.pre_fc_norm_embedding.weight": randn(d, scale=0.1),
        "mtp.pre_fc_norm_hidden.weight": randn(d, scale=0.1),
        "mtp.norm.weight": randn(d, scale=0.1),
        "mtp.layers.0.input_layernorm.weight": randn(d, scale=0.1),
        "mtp.layers.0.post_attention_layernorm.weight": randn(d, scale=0.1),
        "mtp.layers.0.self_attn.q_proj.weight": randn(2 * heads * hd, d),
        "mtp.layers.0.self_attn.k_proj.weight": randn(kv * hd, d),
        "mtp.layers.0.self_attn.v_proj.weight": randn(kv * hd, d),
        "mtp.layers.0.self_attn.o_proj.weight": randn(d, heads * hd),
        "mtp.layers.0.self_attn.q_norm.weight": randn(hd, scale=0.1),
        "mtp.layers.0.self_attn.k_norm.weight": randn(hd, scale=0.1),
        "mtp.layers.0.mlp.gate_proj.weight": exact_mxfp4_values((inter, d)),
        "mtp.layers.0.mlp.up_proj.weight": exact_mxfp4_values((inter, d)),
        "mtp.layers.0.mlp.down_proj.weight": exact_mxfp4_values((d, inter)),
    }


def rmsnorm1p(x, weight, eps=1e-6):
    inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return x.float() * inv * (1.0 + weight.float())


def rope_partial(vec, pos, rope_dim, theta):
    out = vec.clone()
    half = rope_dim // 2
    for i in range(half):
        freq = 1.0 / (theta ** (2.0 * i / rope_dim))
        ang = pos * freq
        s, c = math.sin(ang), math.cos(ang)
        a, b = vec[i].item(), vec[i + half].item()
        out[i] = a * c - b * s
        out[i + half] = a * s + b * c
    return out


def mtp_reference(model, cfg, mtp, tokens, hidden_norm):
    """Draft logits per prompt pair, mirroring the runtime math exactly."""
    d = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    hd = cfg.head_dim
    theta = cfg.rope_parameters["rope_theta"]
    rope_dim = int(hd * cfg.rope_parameters["partial_rotary_factor"]) // 2 * 2
    groups = heads // kv_heads
    embed = model.get_input_embeddings().weight.detach().float()

    cache_k, cache_v = [], []
    out_logits = []
    for i in range(len(tokens) - 1):
        e = rmsnorm1p(embed[tokens[i + 1]], mtp["mtp.pre_fc_norm_embedding.weight"])
        h = rmsnorm1p(hidden_norm[i], mtp["mtp.pre_fc_norm_hidden.weight"])
        x = mtp["mtp.fc.weight"].float() @ torch.cat([e, h])

        normed = rmsnorm1p(x, mtp["mtp.layers.0.input_layernorm.weight"])
        qg = (mtp["mtp.layers.0.self_attn.q_proj.weight"].float() @ normed).view(heads, 2, hd)
        k_raw = (mtp["mtp.layers.0.self_attn.k_proj.weight"].float() @ normed).view(kv_heads, hd)
        v = (mtp["mtp.layers.0.self_attn.v_proj.weight"].float() @ normed).view(kv_heads, hd)
        q = torch.stack(
            [
                rope_partial(
                    rmsnorm1p(qg[h_i, 0], mtp["mtp.layers.0.self_attn.q_norm.weight"]),
                    i,
                    rope_dim,
                    theta,
                )
                for h_i in range(heads)
            ]
        )
        gate = qg[:, 1, :].reshape(-1)
        k = torch.stack(
            [
                rope_partial(
                    rmsnorm1p(k_raw[h_i], mtp["mtp.layers.0.self_attn.k_norm.weight"]),
                    i,
                    rope_dim,
                    theta,
                )
                for h_i in range(kv_heads)
            ]
        )
        cache_k.append(k)
        cache_v.append(v)

        mixed = torch.zeros(heads, hd)
        for h_i in range(heads):
            kh = h_i // groups
            scores = torch.tensor(
                [float(q[h_i] @ cache_k[t][kh]) / math.sqrt(hd) for t in range(len(cache_k))]
            )
            attn = torch.softmax(scores, dim=-1)
            for t in range(len(cache_v)):
                mixed[h_i] += attn[t] * cache_v[t][kh]
        mixed = mixed.reshape(-1) * torch.sigmoid(gate)
        x = x + mtp["mtp.layers.0.self_attn.o_proj.weight"].float() @ mixed

        normed = rmsnorm1p(x, mtp["mtp.layers.0.post_attention_layernorm.weight"])
        gate_h = mtp["mtp.layers.0.mlp.gate_proj.weight"].float() @ normed
        up_h = mtp["mtp.layers.0.mlp.up_proj.weight"].float() @ normed
        act = torch.nn.functional.silu(gate_h) * up_h
        x = x + mtp["mtp.layers.0.mlp.down_proj.weight"].float() @ act

        final = rmsnorm1p(x, mtp["mtp.norm.weight"])
        out_logits.append(model.lm_head.weight.detach().float() @ final)
    return torch.stack(out_logits)


def make_exact_mxfp4(model, dense):
    """Use values exactly representable by microkimi's MXFP4 encoding on
    every matrix the converter quantizes: routed experts for the MoE
    variant, the per-layer MLP for the dense variant."""
    marker = ".mlp." if dense else ".mlp.experts."
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if marker not in name:
                continue
            values = E2M1.repeat((parameter.numel() + 15) // 16)[: parameter.numel()]
            parameter.copy_((values * (2.0 ** -5)).reshape_as(parameter))


def renamed_state(model):
    state = {}
    for name, tensor in model.state_dict().items():
        if name.startswith("model."):
            name = "model.language_model." + name[len("model."):]
        state[name] = tensor.detach().contiguous()
    return state


def write_logits(path, logits):
    logits = logits.detach().cpu().to(torch.float32).contiguous()
    with path.open("wb") as handle:
        handle.write(b"QWLOGIT1")
        handle.write(struct.pack("<II", logits.shape[0], logits.shape[1]))
        handle.write(logits.numpy().tobytes())


def read_logits(path):
    data = path.read_bytes()
    if data[:8] != b"QWLOGIT1" or len(data) < 16:
        raise ValueError(f"{path}: invalid QWLOGIT1 file")
    rows, cols = struct.unpack_from("<II", data, 8)
    expected = 16 + rows * cols * 4
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, found {len(data)}")
    return torch.frombuffer(bytearray(data[16:]), dtype=torch.float32).reshape(rows, cols)


def compare_logits(reference_path, candidate_path):
    reference = read_logits(reference_path)
    candidate = read_logits(candidate_path)
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    delta = candidate - reference
    max_abs = delta.abs().max().item()
    rmse = delta.square().mean().sqrt().item()
    top_match = int((reference.argmax(-1) == candidate.argmax(-1)).sum().item())
    print(
        f"max_abs={max_abs:.9g} rmse={rmse:.9g} "
        f"top1={top_match}/{reference.shape[0]}"
    )
    if max_abs > 1e-5 or top_match != reference.shape[0]:
        raise SystemExit("Qwen parity FAILED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--compare", type=pathlib.Path)
    parser.add_argument(
        "--dense",
        action="store_true",
        help="build the dense qwen3_5_text variant instead of the MoE one",
    )
    parser.add_argument(
        "--compare-mtp",
        type=pathlib.Path,
        help="compare a qwen-dump --mtp output against the MTP reference",
    )
    args = parser.parse_args()
    if args.compare_mtp is not None:
        compare_logits(args.out / "hf_mtp_logits.bin", args.compare_mtp)
        return
    if args.compare is not None:
        compare_logits(args.out / "hf_logits.bin", args.compare)
        return
    args.out.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(0x5147)
    if args.dense:
        cfg = dense_config()
        model = Qwen3_5ForCausalLM(cfg).to(dtype=torch.float32).eval()
    else:
        cfg = config()
        model = Qwen3_5MoeForCausalLM(cfg).to(dtype=torch.float32).eval()
    make_exact_mxfp4(model, args.dense)
    ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        result = model(ids, use_cache=False, output_hidden_states=True)

    text_config = cfg.to_dict()
    text_config["full_attention_interval"] = 4
    root_config = {"text_config": text_config}
    (args.out / "config.json").write_text(
        json.dumps(root_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    state = renamed_state(model)
    if args.dense:
        mtp = make_mtp_tensors(cfg)
        state.update({name: tensor.contiguous() for name, tensor in mtp.items()})
        final_norm_weight = model.model.norm.weight.detach()
        hidden_norm = rmsnorm1p(result.hidden_states[-1][0].detach(), final_norm_weight)
        write_logits(
            args.out / "hf_mtp_logits.bin",
            mtp_reference(model, cfg, mtp, TOKENS, hidden_norm),
        )
    save_file(state, args.out / "model.safetensors")
    write_logits(args.out / "hf_logits.bin", result.logits[0])
    torch.save(
        {
            "tokens": TOKENS,
            "hidden_states": [value[0].cpu() for value in result.hidden_states],
        },
        args.out / "hf_hidden.pt",
    )
    print(f"wrote {args.out} ({sum(p.numel() for p in model.parameters())} parameters)")


if __name__ == "__main__":
    main()
