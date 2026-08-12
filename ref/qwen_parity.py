#!/usr/bin/env python3
"""Build a tiny deterministic Qwen3.5-MoE checkpoint and reference logits.

This is a development parity fixture for ``microkimi convert-qwen`` and the
Rust decoder. It requires PyTorch, Transformers, and safetensors, but none of
those packages are runtime dependencies of microkimi.
"""

import argparse
import json
import pathlib
import struct

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig


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


def make_exact_mxfp4_experts(model):
    """Use values exactly representable by microkimi's MXFP4 encoding."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".mlp.experts." not in name:
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
    args = parser.parse_args()
    if args.compare is not None:
        compare_logits(args.out / "hf_logits.bin", args.compare)
        return
    args.out.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(0x5147)
    cfg = config()
    model = Qwen3_5MoeForCausalLM(cfg).to(dtype=torch.float32).eval()
    make_exact_mxfp4_experts(model)
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
    save_file(renamed_state(model), args.out / "model.safetensors")
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
