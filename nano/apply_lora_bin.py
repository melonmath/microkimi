#!/usr/bin/env python3
"""nanokimi - apply_lora_bin: fold a heal_stream LoRA checkpoint directly into
a COPY of the original .bin (MKIM0002), producing a healed model the Rust
engine can load - without ever materializing the dequantized model.

merge_lora.py cannot be used at full scale: it operates on a .pt state dict,
which means the 130+ GB bin2pt conversion this pipeline exists to avoid. The
observation that makes the direct merge possible: LoRA healing only touches
the attention projections, which are PLAIN fp32 tensors in the .bin, stored
under the same names the torch model uses, and a merged weight has exactly the
same shape and size as the original. So the merge is:

  copy the .bin, then for every "<mod>.lora_A"/"<mod>.lora_B" pair patch the
  fp32 tensor "<mod>.weight" IN PLACE at its directory offset:
      W <- W + (B @ A) * alpha / rank

Packed expert blobs and every untouched tensor are copied byte for byte.

usage: python3 apply_lora_bin.py --ckpt healed.pt --bin model.bin --out healed.bin
"""
import argparse
import os
import shutil

import numpy as np
import torch

from bin2pt import read_bin, DTYPE_F32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="heal_stream checkpoint (LoRA tensors only)")
    ap.add_argument("--bin", required=True, help="original .bin the checkpoint was trained from")
    ap.add_argument("--out", required=True, help="healed .bin (copy of --bin, patched in place)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    info = ck.get("lora") or {}
    src = ck.get("bin_source")
    if src and os.path.abspath(src) != os.path.abspath(args.bin):
        print(f"warning: checkpoint was trained from {src}, patching {args.bin}", flush=True)

    pairs = {}
    for name, t in sd.items():
        if name.endswith(".lora_A"):
            mod = name[: -len(".lora_A")]
            pairs.setdefault(mod, {})["A"] = t
        elif name.endswith(".lora_B"):
            mod = name[: -len(".lora_B")]
            pairs.setdefault(mod, {})["B"] = t
    for mod, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            raise SystemExit(f"{mod}: unpaired LoRA tensors")
    if not pairs:
        raise SystemExit(f"{args.ckpt}: no LoRA tensors found")

    print(f"copying {args.bin} -> {args.out} ...", flush=True)
    shutil.copyfile(args.bin, args.out)

    _, entries, f = read_bin(args.out)  # read/write mode below; this f is rb only
    f.close()
    index = {n: (dt, d, o, s) for n, dt, d, o, s in entries}
    rank = info.get("rank")
    alpha = info.get("alpha")
    patched = 0
    with open(args.out, "r+b") as out:
        for mod, pair in sorted(pairs.items()):
            bin_name = mod + ".weight"
            if bin_name not in index:
                raise SystemExit(f"{bin_name}: no such tensor in {args.out}")
            dt, dims, off, size = index[bin_name]
            if dt != DTYPE_F32:
                raise SystemExit(f"{bin_name}: dtype {dt}, only fp32 tensors can be patched")
            w = np.memmap(args.out, dtype=np.float32, mode="r+", offset=off, shape=tuple(dims))
            a = pair["A"].float()
            b = pair["B"].float()
            r = rank or a.shape[0]
            al = alpha if alpha is not None else r
            if tuple(w.shape) != (b.shape[0], a.shape[1]):
                raise SystemExit(f"{bin_name}: shape {tuple(w.shape)} vs adapter "
                                 f"[{b.shape[0]}x{a.shape[1]}]")
            delta = (b @ a).numpy() * (al / r)
            w += delta  # fused in-place add over the mmap, no second full copy
            w.flush()
            patched += 1
            if patched % 20 == 0:
                print(f"  {patched}/{len(pairs)} tensors patched", flush=True)
    print(f"-> {args.out} : {patched} attention tensors patched in place "
          f"(rank {rank}, alpha {alpha}, step {ck.get('step')}), "
          f"{os.path.getsize(args.out) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
