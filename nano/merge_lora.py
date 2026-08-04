#!/usr/bin/env python3
"""nanokimi - merge_lora: fold a LoRA checkpoint's adapters back into the
base weights, producing a PLAIN checkpoint (no LoRA modules) that
export.py can turn into a .bin for the Rust engine (which knows nothing
about LoRA).

Pure state-dict surgery: for every "<mod>.lora_A" / "<mod>.lora_B" pair,
merged weight = base.weight + (B @ A) * alpha / rank. Everything else is
copied through, and the "lora" entry is dropped from the payload.

usage: python3 merge_lora.py --ckpt healed.pt --out merged.pt
"""
import argparse

import torch


def merge(ck):
    sd = ck["model"]
    info = ck.get("lora") or {}
    rank = info.get("rank")
    alpha = info.get("alpha")
    out = {}
    consumed = set()
    merged = 0
    for name, tensor in sd.items():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            consumed.add(name)
            continue
        if name.endswith(".base.weight"):
            mod = name[: -len(".base.weight")]
            a_key, b_key = mod + ".lora_A", mod + ".lora_B"
            if a_key in sd and b_key in sd:
                r = rank or sd[a_key].shape[0]
                a = alpha if alpha is not None else r
                tensor = tensor + (sd[b_key] @ sd[a_key]) * (a / r)
                consumed.update((a_key, b_key))
                merged += 1
            out[mod + ".weight"] = tensor
            continue
        out[name] = tensor
    leftover = [k for k in sd if (k.endswith(".lora_A") or k.endswith(".lora_B")) and k not in consumed]
    if leftover:
        raise SystemExit(f"merge_lora: unpaired LoRA tensors: {leftover[:4]}")
    payload = {k: v for k, v in ck.items() if k != "lora"}
    payload["model"] = out
    payload["lora_merged"] = merged
    return payload, merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    payload, merged = merge(ck)
    torch.save(payload, args.out)
    print(f"-> {args.out} : {merged} LoRA adapters merged into their base weights, "
          f"{len(payload['model'])} tensors, step {payload.get('step')}", flush=True)


if __name__ == "__main__":
    main()
