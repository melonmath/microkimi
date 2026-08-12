#!/usr/bin/env python3
"""Batch greedy completions from a local Hugging Face checkpoint.

Reads prompts from a JSONL file (one object per line with "id", "prompt",
and optionally "max_new"), generates a greedy continuation for each prompt,
and writes a JSONL file of {"id", "completion"} objects. The completion
excludes the prompt text.

With --lora the tool first folds an elementary PEFT LoRA adapter into the
loaded weights:

    W <- W + (lora_alpha / r) * B @ A

Only plain LoRA is accepted (no DoRA, rank patterns, alpha patterns, bias
tensors, or modules_to_save); anything else is rejected instead of being
approximated. The fold happens in the loaded copy at model precision; files
on disk are never modified.

Examples:
  python3 nano/hf_complete.py --model /models/checkpoint \
      --prompts prompts.jsonl --out completions.jsonl \
      --stop $'\ndef ' --stop $'\nclass '
  python3 nano/hf_complete.py --model /models/checkpoint --lora /adapters/x \
      --prompts prompts.jsonl --out completions.jsonl
  python3 nano/hf_complete.py --selftest
"""

import argparse
import json
import sys
import time
from pathlib import Path

LORA_KEY_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
WRAPPER_PREFIX = "base_model.model."


def load_lora_pairs(adapter_dir):
    """Return (scale, {module: {"A": tensor, "B": tensor}}) or raise."""

    from safetensors.torch import load_file

    config_path = Path(adapter_dir) / "adapter_config.json"
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("peft_type") not in (None, "LORA"):
        raise ValueError(f"unsupported peft_type {config.get('peft_type')!r}")
    for field in ("rank_pattern", "alpha_pattern"):
        if config.get(field):
            raise ValueError(f"{field} is not supported by the elementary fold")
    if config.get("use_dora"):
        raise ValueError("DoRA adapters are not supported by the elementary fold")
    if config.get("bias") not in (None, "none"):
        raise ValueError("LoRA bias tensors are not supported")
    if config.get("modules_to_save"):
        raise ValueError("modules_to_save is not supported by the elementary fold")
    rank = config["r"]
    alpha = config.get("lora_alpha", rank)
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("adapter rank must be a positive integer")
    scale = alpha / rank

    tensors = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    pairs = {}
    for key, tensor in tensors.items():
        for suffix in LORA_KEY_SUFFIXES:
            if key.endswith(suffix):
                module = key[: -len(suffix)]
                factor = "A" if suffix == ".lora_A.weight" else "B"
                break
        else:
            raise ValueError(f"unsupported adapter tensor {key!r}")
        if module.startswith(WRAPPER_PREFIX):
            module = module[len(WRAPPER_PREFIX):]
        pairs.setdefault(module, {})[factor] = tensor
    for module, factors in pairs.items():
        if set(factors) != {"A", "B"}:
            raise ValueError(f"unpaired LoRA factors for {module}")
        if factors["A"].shape[0] != rank or factors["B"].shape[1] != rank:
            raise ValueError(f"{module}: factor shapes do not match the declared rank")
    if not pairs:
        raise ValueError("adapter contains no LoRA factor pairs")
    return scale, pairs


def fold_lora(model, adapter_dir):
    """Fold the adapter into the loaded weights; returns the target count."""

    import torch

    scale, pairs = load_lora_pairs(adapter_dir)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name in sorted(pairs):
            module = modules.get(name)
            if module is None or not hasattr(module, "weight"):
                raise ValueError(f"model has no weighted module named {name!r}")
            weight = module.weight
            a = pairs[name]["A"].to(torch.float32)
            b = pairs[name]["B"].to(torch.float32)
            if weight.shape != (b.shape[0], a.shape[1]):
                raise ValueError(
                    f"{name}: weight is {tuple(weight.shape)}, adapter implies "
                    f"{(b.shape[0], a.shape[1])}"
                )
            delta = (scale * (b @ a)).to(device=weight.device, dtype=weight.dtype)
            weight += delta
    return len(pairs)


def read_prompts(path):
    prompts = []
    seen = set()
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "prompt" not in row:
                raise ValueError(f"{path}:{line_number}: rows need id and prompt")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            prompts.append(row)
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def generate_all(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = read_prompts(args.prompts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    print(f"model loaded in {time.time() - started:.0f}s", flush=True)
    if args.lora:
        folded = fold_lora(model, args.lora)
        print(f"folded {folded} LoRA targets from {args.lora}", flush=True)

    rows = []
    for index, row in enumerate(prompts):
        prompt = row["prompt"]
        max_new = int(row.get("max_new", args.max_new))
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        generate_args = {
            "max_new_tokens": max_new,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if args.stop:
            generate_args["stop_strings"] = list(args.stop)
            generate_args["tokenizer"] = tokenizer
        with torch.no_grad():
            output = model.generate(**inputs, **generate_args)
        generated = output[0][inputs["input_ids"].shape[1]:]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        rows.append({"id": row["id"], "completion": completion})
        print(
            f"[{index + 1}/{len(prompts)}] {row['id']}: {len(generated)} tokens",
            flush=True,
        )

    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"wrote {len(rows)} completions -> {args.out}", flush=True)


def selftest():
    """Exercise the pure logic (pairing, strictness, prompt IO) without torch."""

    import tempfile

    import numpy as np

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array, dtype=np.float32)
            self.shape = self.array.shape

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "bias": "none"}),
            encoding="utf-8",
        )
        prompts = root / "prompts.jsonl"
        prompts.write_text(
            '{"id": "a", "prompt": "x"}\n{"id": "b", "prompt": "y", "max_new": 7}\n',
            encoding="utf-8",
        )
        rows = read_prompts(prompts)
        assert [row["id"] for row in rows] == ["a", "b"]
        try:
            read_prompts(root / "missing.jsonl")
        except OSError:
            pass
        else:
            raise AssertionError("missing prompt file was accepted")
        duplicated = root / "dup.jsonl"
        duplicated.write_text(
            '{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n', encoding="utf-8"
        )
        try:
            read_prompts(duplicated)
        except ValueError as error:
            assert "duplicate id" in str(error)
        else:
            raise AssertionError("duplicate ids were accepted")

        import types

        fake_module = types.SimpleNamespace()

        def fake_load_file(path):
            return {
                "base_model.model.model.layers.0.proj.lora_A.weight": FakeTensor(
                    np.ones((2, 3))
                ),
                "base_model.model.model.layers.0.proj.lora_B.weight": FakeTensor(
                    np.ones((4, 2))
                ),
            }

        fake_module.load_file = fake_load_file
        fake_package = types.SimpleNamespace(torch=fake_module)
        sys.modules.setdefault("safetensors", fake_package)
        sys.modules.setdefault("safetensors.torch", fake_module)
        (root / "adapter_model.safetensors").write_bytes(b"")
        scale, pairs = load_lora_pairs(root)
        assert scale == 2.0
        assert list(pairs) == ["model.layers.0.proj"]
        assert set(pairs["model.layers.0.proj"]) == {"A", "B"}

        (root / "adapter_config.json").write_text(
            json.dumps({"peft_type": "LORA", "r": 2, "lora_alpha": 4, "use_dora": True}),
            encoding="utf-8",
        )
        try:
            load_lora_pairs(root)
        except ValueError as error:
            assert "DoRA" in str(error)
        else:
            raise AssertionError("a DoRA adapter was accepted")
    print("hf_complete selftest OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--model", help="local checkpoint directory")
    parser.add_argument("--lora", help="fold this adapter directory before generating")
    parser.add_argument("--prompts", help="input JSONL with id/prompt/max_new")
    parser.add_argument("--out", help="output JSONL with id/completion")
    parser.add_argument("--max-new", type=int, default=256, help="default token budget")
    parser.add_argument(
        "--stop", action="append", help="stop string, repeatable, kept in the output"
    )
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not (args.model and args.prompts and args.out):
        parser.error("--model, --prompts, and --out are required")
    generate_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
