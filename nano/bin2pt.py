#!/usr/bin/env python3
"""nanokimi - bin2pt: convert a microkimi .bin (MKIM0002) into a torch
checkpoint loadable by train.py --resume (the inverse of export.py).

Every .bin tensor becomes a state_dict entry under the SAME name the torch
model uses (the correspondence is export.py's, inverted):
  - f32 tensors keep their .bin name (torch and .bin names already match,
    e.g. "layers.0.self_attn.q_proj.weight", "norm.weight", "A_log",
    "gate.e_score_correction_bias");
  - KDA conv1d weights are stored [C, K] in the .bin and expanded back to
    nn.Conv1d's [C, 1, K];
  - MXFP4 expert blobs (dtype 1: packed nibbles ++ e8m0 scales) are
    DEQUANTIZED to fp32 and get the torch ".weight" suffix
    ("layers.L.block_sparse_moe.experts.E.w1" -> "...w1.weight");
  - VQ1 blobs (dtype 3, --cold-vq slices) are dequantized through the
    "vq_codebook" tensor of the same file, also to fp32 ".weight".

The embedded MKIM0002 JSON config becomes the checkpoint "cfg" (same keys
as model_nano.NANO plus the explicit mla_layers / dense_layers lists a
sliced model carries). There is no optimizer state: train.py --resume
treats that as a fresh-opt start (weights only).

usage: python3 bin2pt.py model.bin out.pt
"""
import json
import struct
import sys

import numpy as np
import torch

MAGIC_V1 = b"MKIM0001"
MAGIC_V2 = b"MKIM0002"
DTYPE_F32, DTYPE_MXFP4, DTYPE_I32, DTYPE_VQ1 = 0, 1, 2, 3

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)

# the nano cfg keys a .bin config provides (model_nano.NANO has the same keys)
CFG_KEYS = [
    "n_layers", "hidden", "vocab", "n_experts", "top_k", "n_shared",
    "kda_heads", "kda_dim", "kda_conv", "kda_fa_rank", "gate_lower_bound",
    "mla_heads", "mla_q_lora", "mla_kv_lora", "mla_nope", "mla_rope", "mla_v",
    "routed_hidden", "moe_inter", "shared_inter", "dense_inter",
    "attn_res_block", "first_k_dense", "rms_eps",
    # explicit layer-type lists (sliced models; absent in nano exports, the
    # legacy L%4==3-or-last pattern applies then - see model_nano.layer_types)
    "mla_layers", "dense_layers",
]


def read_bin(path):
    """Parses the .bin header + directory (see src/weights.rs::read_directory).
    Returns (config dict, [(name, dtype, dims, offset, size)], file object)."""
    f = open(path, "rb")
    magic = f.read(8)
    if magic == MAGIC_V1:
        f.close()
        raise SystemExit(f"{path}: MKIM0001 has no embedded config block - cannot derive a model cfg")
    if magic != MAGIC_V2:
        f.close()
        raise SystemExit(f"{path}: bad magic (expected MKIM0002, got {magic!r})")
    (clen,) = struct.unpack("<I", f.read(4))
    config = json.loads(f.read(clen))
    (n,) = struct.unpack("<I", f.read(4))
    entries = []
    for _ in range(n):
        (nlen,) = struct.unpack("<H", f.read(2))
        name = f.read(nlen).decode()
        dtype, n_dims = struct.unpack("BB", f.read(2))
        dims = list(struct.unpack(f"<{n_dims}I", f.read(4 * n_dims)))
        offset, size = struct.unpack("<QQ", f.read(16))
        entries.append((name, dtype, dims, offset, size))
    return config, entries, f


def dequant_mxfp4(blob, dims):
    """MXFP4 blob (packed [R,C/2] ++ scales [R,C/32]) -> f32 [R,C].
    Exact inverse of export.py's quantize_mxfp4 (same rule as src/mxfp4.rs):
    value = e2m1[nibble] * 2^(scale-127), groups of 32 along the row."""
    r, c = dims
    np_ = r * c // 2
    packed = np.frombuffer(blob[:np_], dtype=np.uint8).reshape(r, c // 2)
    scales = np.frombuffer(blob[np_:], dtype=np.uint8).reshape(r, c // 32)
    idx = np.empty((r, c), dtype=np.uint8)
    idx[:, 0::2] = packed & 0x0F
    idx[:, 1::2] = packed >> 4
    vals = E2M1[idx].reshape(r, c // 32, 32)
    gain = np.exp2(scales.astype(np.int32) - 127).reshape(r, c // 32, 1)
    return (vals * gain).reshape(r, c).astype(np.float32)


def dequant_vq1(blob, dims, codebook):
    """VQ1 blob (one u8 codebook index per vector of VQ_DIM=16 values) ->
    f32 [R,C]. codebook: f32 [256,16] from the "vq_codebook" tensor."""
    r, c = dims
    idx = np.frombuffer(blob, dtype=np.uint8)
    return codebook[idx].reshape(r, c).astype(np.float32)


def convert(path):
    """Reads a .bin, returns (state_dict, cfg)."""
    config, entries, f = read_bin(path)
    raw = {}
    codebook = None
    sd = {}
    for name, dtype, dims, offset, size in entries:
        f.seek(offset)
        blob = f.read(size)
        if dtype == DTYPE_F32:
            t = np.frombuffer(blob, dtype=np.float32).reshape(dims).copy()
            if name == "vq_codebook":
                codebook = t  # kept for the VQ1 experts, not a torch tensor
                continue
        elif dtype == DTYPE_I32:
            t = np.frombuffer(blob, dtype=np.int32).reshape(dims).copy()
        elif dtype == DTYPE_MXFP4:
            t = dequant_mxfp4(blob, dims)
            name = name + ".weight"  # torch expert params are w{1,2,3}.weight
        elif dtype == DTYPE_VQ1:
            raw[name] = (blob, dims)  # dequantized below (codebook may come later)
            continue
        else:
            raise SystemExit(f"{name}: unknown dtype {dtype}")
        # KDA conv1d: stored [C, K] in the .bin, nn.Conv1d wants [C, 1, K]
        if name.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")) and len(dims) == 2:
            t = t.reshape(dims[0], 1, dims[1])
        sd[name] = torch.from_numpy(np.ascontiguousarray(t))
    for name, (blob, dims) in raw.items():
        if codebook is None:
            raise SystemExit(f"{name}: VQ1 tensor but no vq_codebook in the file")
        sd[name + ".weight"] = torch.from_numpy(np.ascontiguousarray(dequant_vq1(blob, dims, codebook)))
    f.close()
    cfg = {k: config[k] for k in CFG_KEYS if k in config}
    return sd, cfg


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: python3 bin2pt.py model.bin out.pt")
    src, dst = sys.argv[1], sys.argv[2]
    sd, cfg = convert(src)
    payload = {
        "model": sd,
        "cfg": cfg,
        "step": 0,  # no optimizer/rng: train.py --resume starts fresh-opt
        "bin_source": src,
    }
    torch.save(payload, dst)
    n_experts = sum(1 for k in sd if ".experts." in k)
    print(f"-> {dst} : {len(sd)} tensors ({n_experts} dequantized expert matrices), "
          f"cfg {cfg['n_layers']} layers, hidden {cfg['hidden']}, vocab {cfg['vocab']}", flush=True)


if __name__ == "__main__":
    main()
