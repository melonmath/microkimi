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
relative budget, so a TRAINED seam adapter is REFUSED by default. A zero-init
adapter folds to a no-op and is accepted. --force-seam-fold produces the
approximate merge anyway (experimentation only, NOT a deployment artifact).

EXACT deployment: --write-seam. Instead of folding, the adapter itself is
embedded in the output .bin (still plain MKIM0002):
  - the config JSON gains "seam_after": N (u32, 0-based layer index);
  - the tensor directory gains two fp32 tensors, appended after the existing
    blobs: "seam.A" [rank, hidden] and "seam.B" [hidden, rank] (row-major,
    same layout as the torch parameters).
The Rust engine reads them at load and applies h += (h @ A^T) @ B^T to the
residual stream right after layer N, in prefill and decode alike - the same
two matvecs the Python SeamAdapter computes, so the deployed model matches
the adapted one to float noise.

Compatibility: an old .bin without seam tensors loads exactly as before (the
reader only looks up the names it knows). A .bin WITH a seam read by an OLD
engine (before reader support) also loads fine - the unknown config key and
the two unknown directory entries are simply never accessed - but it then
generates WITHOUT the adapter (silently unadapted logits). The engine prints
"seam: adapter rank R after layer N" at load when the adapter is applied;
absence of that line on an old binary is the tell.

usage: python3 apply_lora_bin.py --ckpt healed.pt --bin model.bin --out healed.bin
       python3 apply_lora_bin.py --ckpt healed.pt --bin model.bin --out healed.bin --write-seam
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


def write_seam_bin(src, dst, seam, sd):
    """Exact seam deployment (--write-seam): rewrites src into dst with the
    adapter embedded - config key "seam_after" plus the fp32 tensors "seam.A"
    [rank, hidden] and "seam.B" [hidden, rank] appended after the existing
    blobs. Every other blob is copied byte for byte (the LoRA fold then
    patches the attention projections in place in dst, as usual). The header
    grows (config key + 2 directory entries), so the blobs move: offsets are
    recomputed with the format's alignment rule (src/weights.rs layout:
    expert blobs 4096-aligned, everything else 64), detected from the source
    (dense-packed slices use 64 for the experts too)."""
    import json  # local: the fold-only path never needs it

    config, entries, f = read_bin(src)
    a = sd["seam_adapter.A"].float().numpy()  # [rank, hidden], row-major
    b = sd["seam_adapter.B"].float().numpy()  # [hidden, rank]
    rank, hidden = a.shape
    if b.shape != (hidden, rank):
        raise SystemExit(f"seam_adapter.B: shape {tuple(b.shape)}, expected "
                         f"({hidden}, {rank})")
    names = [n for n, _, _, _, _ in entries]
    if "seam.A" in names or "seam.B" in names:
        raise SystemExit(f"{src} already embeds a seam adapter - start from "
                         "the original .bin")
    cfg = dict(config)
    cfg["seam_after"] = seam["after"]
    cfg_bytes = json.dumps(cfg).encode()

    def is_expert(name):
        return (".block_sparse_moe.experts." in name
                and name.rsplit(".", 1)[-1] in ("w1", "w2", "w3"))

    experts = [o for n, _, _, o, _ in entries if is_expert(n)]
    expert_align = 4096 if experts and all(o % 4096 == 0 for o in experts) else 64
    a_bytes = np.ascontiguousarray(a, dtype=np.float32).tobytes()
    b_bytes = np.ascontiguousarray(b, dtype=np.float32).tobytes()
    # (name, dtype, dims, size, source): source is the old offset, or the
    # blob bytes for the two new tensors
    tensors = [(n, dt, d, s, o) for n, dt, d, o, s in entries]
    tensors.append(("seam.A", DTYPE_F32, [rank, hidden], len(a_bytes), a_bytes))
    tensors.append(("seam.B", DTYPE_F32, [hidden, rank], len(b_bytes), b_bytes))
    dir_size = sum(2 + len(n.encode()) + 1 + 1 + 4 * len(d) + 8 + 8
                   for n, _, d, _, _ in tensors)
    pos = 8 + 4 + len(cfg_bytes) + 4 + dir_size
    offsets = []
    for n, _, _, s, _ in tensors:
        align = expert_align if is_expert(n) else 64
        pos = (pos + align - 1) // align * align
        offsets.append(pos)
        pos += s
    with open(dst, "wb") as out:
        out.write(b"MKIM0002")
        out.write(len(cfg_bytes).to_bytes(4, "little"))
        out.write(cfg_bytes)
        out.write(len(tensors).to_bytes(4, "little"))
        for (n, dt, d, s, _), off in zip(tensors, offsets):
            nb = n.encode()
            out.write(len(nb).to_bytes(2, "little"))
            out.write(nb)
            out.write(bytes([dt, len(d)]))
            for dim in d:
                out.write(int(dim).to_bytes(4, "little"))
            out.write(off.to_bytes(8, "little"))
            out.write(s.to_bytes(8, "little"))
        for (n, _, _, s, source), off in zip(tensors, offsets):
            if out.tell() < off:
                out.write(b"\0" * (off - out.tell()))
            if isinstance(source, bytes):
                out.write(source)
            else:
                f.seek(source)
                left = s
                while left:
                    chunk = f.read(min(left, 1 << 26))
                    out.write(chunk)
                    left -= len(chunk)
    f.close()
    return rank, hidden


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
    ap.add_argument("--write-seam", action="store_true",
                    help="EXACT seam deployment: embed the adapter in the output .bin "
                         "(config key seam_after + fp32 tensors seam.A/seam.B) instead of "
                         "folding it. The Rust engine applies it at load (see the module "
                         "docstring for the format and the old-reader behavior)")
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
    if args.write_seam and args.force_seam_fold:
        raise SystemExit("--write-seam and --force-seam-fold are mutually exclusive "
                         "(exact embedding vs approximate fold)")
    seam = ck.get("seam")
    if not pairs and not seam:
        raise SystemExit(f"{args.ckpt}: no LoRA or seam tensors found")
    if args.write_seam and not seam:
        raise SystemExit("--write-seam but the checkpoint carries no seam adapter")
    if seam:
        # decide BEFORE any file write: a trained seam adapter cannot be
        # folded exactly (module docstring); --write-seam deploys it exactly,
        # the fold is refused unless forced
        a, b = sd.get("seam_adapter.A"), sd.get("seam_adapter.B")
        if a is None or b is None:
            raise SystemExit(f"checkpoint declares a seam adapter {seam} but "
                             "seam_adapter.A/B are missing from its tensors")
        ba_max = float((b.double() @ a.double()).abs().max())
        if ba_max > 0 and not args.force_seam_fold and not args.write_seam:
            raise SystemExit(
                f"refusing to fold a TRAINED seam adapter (max|B A| = {ba_max:.3e}): "
                f"the fold W <- W + W B A into layer {seam['after'] + 1} is exact only "
                "for the linear attention-input read; the residual pass-through part "
                "of the correction cannot be folded into any existing weight and the "
                "merged model would NOT match the adapted one (measured on the smoke "
                "model: max|logit diff| 0.21, rel 1.3e-2, already at max|B A| 4.4e-4; "
                "see the module docstring). Rerun with --write-seam for the EXACT "
                "deployment (the adapter is embedded in the .bin and applied by the "
                "engine), or --force-seam-fold to produce the APPROXIMATE merge anyway.")

    if seam and args.write_seam:
        print(f"rewriting {args.bin} -> {args.out} with the seam adapter embedded ...",
              flush=True)
        wrank, whidden = write_seam_bin(args.bin, args.out, seam, sd)
        print(f"  seam.A [{wrank}x{whidden}] + seam.B [{whidden}x{wrank}] appended, "
              f"seam_after={seam['after']} in the config", flush=True)
    else:
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
        if seam and not args.write_seam:
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
    seam_note = (f", seam adapter embedded (rank {seam['rank']}, after layer "
                 f"{seam['after']})" if seam and args.write_seam else "")
    print(f"-> {args.out} : {patched} attention tensors patched in place "
          f"(rank {rank}, alpha {alpha}, step {ck.get('step')}){seam_note}, "
          f"{os.path.getsize(args.out) / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
