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

SEAM ADAPTER fold (--seam-adapter checkpoints, model_nano.SeamAdapter):
the adapter is h' = (I + B A) h on the residual stream right after layer N.
It CANNOT be folded exactly into existing weights: the stream is consumed
through RMSNorm (non-linear) and also passes through unchanged into the
residual add of layer N+1, the block_residual buffers of the attn_res
mechanism and every later layer - none of those paths has a weight that
could absorb (I + B A). What CAN be folded are the DIRECT input projections
of layer N+1's attention (q/k/v/gate/..., the consumers of
input_layernorm(stream)): for a linear read, W h' = W (I + B A) h, so
    W <- W + W B A          (computed in float64, cast back to fp32)
Same shapes, same sizes: the .bin stays standard and needs no reader
support.

Measured on the smoke model (nano/test_seam_adapter.py): the fold is exact
at zero-init (B = 0, the merged .bin is byte-identical) but the forward of
the merged model diverges from the adapted one linearly with |B A| -
max|logit diff| 5.7e-6 at max|B A| 1e-8, 9.2e-5 at 1e-5, and already 0.21
(rel 1.3e-2) at the modestly trained max|B A| 4.4e-4, because the residual
pass-through part of the correction is lost. That is far above the 1e-3
relative budget, so a TRAINED seam adapter is REFUSED by default: exact
deployment needs reader-side support for the two small adapter tensors (a
sidecar next to the .bin), which is out of scope here. A zero-init adapter
folds to a no-op and is accepted. --force-seam-fold produces the approximate
merge anyway (experimentation only, NOT a deployment artifact).

usage: python3 apply_lora_bin.py --ckpt healed.pt --bin model.bin --out healed.bin
"""
import argparse
import os
import shutil

import numpy as np
import torch

from bin2pt import read_bin, DTYPE_F32

# leaf names of the attention Linears that read the (normed) residual stream
# directly: KDA layers (q/k/v, the f gate input, the beta head, the full-rank
# output gate) and MLA layers (q or its a-side, the kv a-side, the output
# gate). Second-stage projections (q_b_proj, kv_b_proj, f_b_proj, g_b_proj,
# o_proj) do NOT read the stream and are excluded.
_SEAM_CONSUMER_LEAVES = (
    "q_proj", "q_a_proj", "k_proj", "v_proj",
    "kv_a_proj_with_mqa", "f_a_proj", "b_proj", "g_proj",
)


def fold_seam(out_path, index, seam, sd):
    """Fold the seam adapter h' = (I + B A) h into the input projections of
    layer seam["after"] + 1, in place in the .bin at out_path. See the module
    docstring for why this is an approximation after training."""
    after = seam["after"]
    layer = after + 1
    a = sd.get("seam_adapter.A")
    b = sd.get("seam_adapter.B")
    if a is None or b is None:
        raise SystemExit(f"checkpoint declares a seam adapter {seam} but "
                         "seam_adapter.A/B are missing from its tensors")
    prefix = f"layers.{layer}.self_attn."
    targets = [prefix + leaf + ".weight" for leaf in _SEAM_CONSUMER_LEAVES
               if prefix + leaf + ".weight" in index]
    if not targets:
        raise SystemExit(f"seam fold: no input projection of layer {layer} found in "
                         f"{out_path} (--seam-after {after} needs a layer {layer})")
    hidden = a.shape[1]
    ba = (b.double() @ a.double()).numpy()  # [hidden, hidden] float64
    ba_max = float(np.abs(ba).max())
    patched = 0
    for name in sorted(targets):
        dt, dims, off, size = index[name]
        if dt != DTYPE_F32:
            raise SystemExit(f"{name}: dtype {dt}, only fp32 tensors can be patched")
        if len(dims) != 2 or dims[1] != hidden:
            raise SystemExit(f"{name}: shape {tuple(dims)} incompatible with the seam "
                             f"adapter (hidden {hidden})")
        w = np.memmap(out_path, dtype=np.float32, mode="r+", offset=off, shape=tuple(dims))
        w64 = np.asarray(w, dtype=np.float64)
        w[...] = (w64 + w64 @ ba).astype(np.float32)  # W (I + B A)
        w.flush()
        patched += 1
    return patched, layer, ba_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="heal_stream checkpoint (LoRA tensors only)")
    ap.add_argument("--bin", required=True, help="original .bin the checkpoint was trained from")
    ap.add_argument("--out", required=True, help="healed .bin (copy of --bin, patched in place)")
    ap.add_argument("--force-seam-fold", action="store_true",
                    help="fold a TRAINED seam adapter anyway (approximate merge: the "
                         "residual pass-through part of the correction is lost, see the "
                         "module docstring; experimentation only, not a deployment artifact)")
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
    seam = ck.get("seam")
    if not pairs and not seam:
        raise SystemExit(f"{args.ckpt}: no LoRA or seam tensors found")
    if seam:
        # decide BEFORE any file write: a trained seam adapter cannot be
        # folded exactly (module docstring), refuse unless forced
        a, b = sd.get("seam_adapter.A"), sd.get("seam_adapter.B")
        if a is None or b is None:
            raise SystemExit(f"checkpoint declares a seam adapter {seam} but "
                             "seam_adapter.A/B are missing from its tensors")
        ba_max = float((b.double() @ a.double()).abs().max())
        if ba_max > 0 and not args.force_seam_fold:
            raise SystemExit(
                f"refusing to fold a TRAINED seam adapter (max|B A| = {ba_max:.3e}): "
                f"the fold W <- W + W B A into layer {seam['after'] + 1} is exact only "
                "for the linear attention-input read; the residual pass-through part "
                "of the correction cannot be folded into any existing weight and the "
                "merged model would NOT match the adapted one (measured on the smoke "
                "model: max|logit diff| 0.21, rel 1.3e-2, already at max|B A| 4.4e-4; "
                "see the module docstring). Exact deployment needs reader-side support "
                "for the two adapter tensors (sidecar), out of scope here. Rerun with "
                "--force-seam-fold to produce the APPROXIMATE merge anyway.")

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
        if seam:
            # LoRA first, seam second: the fold applies to the ALREADY
            # LoRA-merged projections of layer N+1, matching the training
            # composition (LoRALinear output read through the seam adapter)
            n_fold, layer, ba_max = fold_seam(args.out, index, seam, sd)
            print(f"  seam adapter folded into {n_fold} input projections of layer "
                  f"{layer} (rank {seam['rank']}, max|B A| {ba_max:.3e})", flush=True)
            if ba_max > 0:
                print("  WARNING (--force-seam-fold): APPROXIMATE merge - the residual "
                      "pass-through part of the seam correction is lost (measured: "
                      "rel logit error 1.3e-2 at max|B A| 4.4e-4, growing linearly). "
                      "Not a deployment artifact; verify against the adapted model.",
                      flush=True)
    print(f"-> {args.out} : {patched} attention tensors patched in place "
          f"(rank {rank}, alpha {alpha}, step {ck.get('step')}), "
          f"{os.path.getsize(args.out) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
