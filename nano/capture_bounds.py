#!/usr/bin/env python3
"""nanokimi - capture_bounds: residual-stream capture at every layer boundary
of the FULL K3 model, read directly from a sharded safetensors root
(model.safetensors.index.json + shards), with no .bin materialization.

Why: Block Influence style layer diagnostics (and slice candidate scoring)
need the residual stream at each layer boundary on a calibration sample. The
full model is far larger than host RAM (~1.5 TB of shards), so weights are
streamed exactly the way heal_stream.py streams a .bin:

  - the safetensors root is opened through the same layout rules as the
    slicer's safetensors input (src/tools/slice_st.rs, mirrored here in
    Python): index.json weight_map, lazy per-shard JSON headers, logical
    .bin-style names ("language_model.model." prefix stripped, expert
    weight_packed + weight_scale merged into one MXFP4 entry), dims derived
    from tensor shapes and scalars from config.json (shapes win);
  - the model is instantiated on the meta device (model_nano.NanoModel, zero
    allocation); every frozen parameter is resolved to a (shard, byte range)
    reference and only materialized (bf16 -> f32) inside its layer's forward
    window, then released (F32 tensors are used as zero-copy memmap views);
  - routed experts stay PACKED (U8 e2m1 nibbles + e8m0 scales): per layer,
    only the router-selected experts are read and dequantized on the compute
    device, exactly ONCE per layer, each expert applied to all of its tokens
    gathered across the whole residual plane (the capacity-bounded grouped
    GEMM of model_nano);
  - the capture is LAYERS-OUTER / CHUNKS-INNER (forward-only, chunks are
    independent sequences): a current-residual plane [n_tokens, hidden] f32
    (~1.4 GB at 50k tokens) lives in host RAM, every layer's weights are read
    from the shards exactly once, and boundary l+1 streams into the fp16
    memmap when layer l completes;

Outputs (--out PREFIX):
  PREFIX.bounds.npy : fp16 memmap, shape (n_tokens, n_layers+1, hidden),
      C-order, token-major. Boundary 0 is the embedding output, boundary b is
      the residual stream returned by layer b-1 (boundary n_layers is the
      stream entering the trunk merge + final norm, which is NOT applied).
  PREFIX.meta.json  : shapes, layer types, seq, source root, and the Block
      Influence report (per-layer cosine between input and output boundary),
      which doubles as slice --suggest data.

--ignore-unk/--unk-id follow the train.py convention: with --ignore-unk the
UNK positions are still captured (the causal stream needs them) but are
excluded from the Block Influence statistics.

Deterministic (seeded), torch.no_grad everywhere, float32 compute with fp16
storage.

examples:
  python3 capture_bounds.py --st-root /mnt/k3/K3 --data tokens.bin \
      --out calib/k3 --seq 512 --max-tokens 50000 --device cuda \
      --ignore-unk --unk-id 163585
  python3 capture_bounds.py --selftest
"""
import argparse
import json
import math
import os
import re
import struct
import tempfile
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from bin2pt import E2M1
from model_nano import NanoModel, TrainableSparseMoe, layer_types
from train import load_tokens, rss_gb

# memmap tensors are read-only; torch.from_numpy would warn once per tensor
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# experts are dequantized group by group on the compute device (RAM/VRAM bound)
MOE_CHUNK = int(os.environ.get("NANO_CAPTURE_MOE_CHUNK", "8"))
# rows per Block Influence statistics block (bounds memmap is read back once)
BI_ROWS = int(os.environ.get("NANO_CAPTURE_BI_ROWS", "256"))


def bf16_u16_to_f32(u16):
    """numpy uint16 view of little-endian bf16 -> f32 (f32 bits = u16 << 16)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


def squeeze(shape):
    """Drops size-1 dims (the .bin convention): [12288,1,4] -> [12288,4]."""
    v = [d for d in shape if d != 1]
    return v if v else [1]


class StRoot:
    """Sharded safetensors root, opened lazily: index.json + per-shard JSON
    headers give every logical tensor a (shard, byte range) location; tensor
    bytes are memmap views of the shard files, never bulk-loaded."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self._headers = {}  # shard file -> (data_start, {name: (dtype, shape, off0, off1)})
        self._mms = {}      # shard file -> np.memmap uint8
        self.bytes_read = 0  # tensor bytes fetched (progress/ETA accounting)

        # weight_map (original tensor name -> shard file)
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                weight_map = dict(json.load(f)["weight_map"])
        else:
            shards = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
            if len(shards) != 1:
                raise SystemExit(
                    f"{root}: no model.safetensors.index.json and {len(shards)} .safetensors files")
            _, hdr = self._header(shards[0])
            weight_map = {n: shards[0] for n in hdr}

        # name translation to logical .bin-style names (mirror of the
        # slicer's safetensors input): expert weight_packed + weight_scale
        # pairs merge into one MXFP4 entry keyed by the bare expert name
        self._wmap = weight_map  # original tensor name -> shard file
        self.plain = {}        # logical -> original full name (f32/bf16 tensors)
        self.expert_base = {}  # logical -> original name without suffix (MXFP4)
        skipped = 0
        for name in weight_map:
            if name.startswith("language_model.model."):
                logical = name[len("language_model.model."):]
            elif name == "language_model.lm_head.weight":
                logical = "lm_head.weight"
            elif name.startswith("model."):
                rest = name[len("model."):]
                if rest.endswith("rotary_emb.inv_freq"):
                    skipped += 1
                    continue  # precomputed rope tables are not weights
                logical = rest
            elif name.startswith(("vision_tower.", "mm_projector.")):
                skipped += 1
                continue
            elif "block_sparse_moe" in name or "self_attn.A_log" in name:
                logical = name  # bare K3 names
            else:
                skipped += 1
                continue
            if logical.endswith(".weight_scale") and ".experts." in logical:
                continue  # folded into the weight_packed entry
            if logical.endswith(".weight_packed"):
                base = logical[: -len(".weight_packed")]
                # the weight_scale half is checked at fetch time
                self.expert_base[base] = name[: -len(".weight_packed")]
            else:
                self.plain[logical] = name
        n_tensors = len(self.plain) + len(self.expert_base)
        print(f"safetensors source: local ({n_tensors} logical tensors mapped, "
              f"{skipped} skipped: vision/projector/rope)", flush=True)

        # per-layer structure from names (no shard headers needed)
        has = set(self.plain) | set(self.expert_base)
        max_layer = -1
        for n in has:
            m = re.match(r"layers\.(\d+)\.", n)
            if m:
                max_layer = max(max_layer, int(m.group(1)))
        self.n_layers = max_layer + 1
        if self.n_layers == 0:
            raise SystemExit(f"{root}: no layers.* tensors found")
        self.layer_attn = []  # "kda" | "mla" per layer
        self.layer_moe = []
        for l in range(self.n_layers):
            p = f"layers.{l}."
            if p + "self_attn.q_a_proj.weight" in has:
                self.layer_attn.append("mla")
            elif p + "self_attn.q_proj.weight" in has:
                self.layer_attn.append("kda")
            else:
                raise SystemExit(f"{root}: layer {l} has neither q_a_proj nor q_proj")
            self.layer_moe.append(p + "block_sparse_moe.gate.weight" in has)
        if not any(self.layer_moe):
            raise SystemExit(f"{root}: no MoE layer found (not a K3 root)")
        # expert count: distinct expert ids on the first MoE layer
        ml = self.layer_moe.index(True)
        pfx = f"layers.{ml}.block_sparse_moe.experts."
        ids = set()
        for n in self.expert_base:
            if n.startswith(pfx):
                rest = n[len(pfx):]
                ids.add(int(rest.split(".", 1)[0]))
        self.n_experts = len(ids)

        # config.json scalars (text_config when the file wraps one)
        cfg_path = os.path.join(root, "config.json")
        self.hf_config = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                j = json.load(f)
            self.hf_config = j.get("text_config", j)

        self.cfg = self._derive_cfg()

    # ── shard headers / memmaps ──

    def _header(self, shard_file):
        if shard_file not in self._headers:
            path = os.path.join(self.root, shard_file)
            with open(path, "rb") as f:
                (hlen,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(hlen))
            info = {}
            for name, meta in hdr.items():
                if name == "__metadata__":
                    continue
                info[name] = (meta["dtype"], list(meta["shape"]),
                              int(meta["data_offsets"][0]), int(meta["data_offsets"][1]))
            self._headers[shard_file] = (8 + hlen, info)
        return self._headers[shard_file]

    def _mm(self, shard_file):
        if shard_file not in self._mms:
            self._mms[shard_file] = np.memmap(
                os.path.join(self.root, shard_file), dtype=np.uint8, mode="r")
        return self._mms[shard_file]

    def _shard_of(self, orig_name):
        return self._wmap[orig_name]

    # ── public API ──

    def has(self, logical):
        return logical in self.plain or logical in self.expert_base

    def shape_of(self, logical):
        """Squeezed dims of a plain tensor (reads its shard header once)."""
        data_start, info = self._header(self._shard_of(self.plain[logical]))
        return squeeze(info[self.plain[logical]][1])

    def fetch_f32(self, logical):
        """Plain tensor as an f32 numpy array. F32 storage is a zero-copy
        read-only memmap view; BF16 is converted (one transient copy)."""
        orig = self.plain[logical]
        shard_file = self._shard_of(orig)
        data_start, info = self._header(shard_file)
        dt, shape, o0, o1 = info[orig]
        self.bytes_read += o1 - o0
        raw = self._mm(shard_file)[data_start + o0: data_start + o1]
        if dt == "F32":
            return np.frombuffer(raw, dtype=np.float32).reshape(shape)
        if dt == "BF16":
            return bf16_u16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
        raise SystemExit(f"{orig}: unhandled dtype {dt} (expected F32 or BF16)")

    def gather_rows(self, logical, rows):
        """Rows `rows` of a 2-D plain tensor as f32 (embedding gather)."""
        orig = self.plain[logical]
        shard_file = self._shard_of(orig)
        data_start, info = self._header(shard_file)
        dt, shape, o0, o1 = info[orig]
        n_cols = int(np.prod(shape[1:])) if len(shape) > 1 else 1
        self.bytes_read += len(rows) * n_cols * (4 if dt == "F32" else 2)
        raw = self._mm(shard_file)[data_start + o0: data_start + o1]
        if dt == "F32":
            view = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            return np.array(view[rows])
        if dt == "BF16":
            view = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            return bf16_u16_to_f32(np.array(view[rows]))
        raise SystemExit(f"{orig}: unhandled dtype {dt}")

    def expert_parts(self, logical):
        """(packed u8 view [R, C/2] flat, scales u8 view [R, C/32] flat,
        (R, C)) of an MXFP4 expert matrix. Packed and scales may live in
        different shards."""
        base = self.expert_base[logical]
        refs = []
        for suffix in (".weight_packed", ".weight_scale"):
            orig = base + suffix
            if orig not in self._wmap:
                raise SystemExit(f"{orig}: missing from the safetensors index")
            shard_file = self._wmap[orig]
            data_start, info = self._header(shard_file)
            dt, shape, o0, o1 = info[orig]
            if dt != "U8":
                raise SystemExit(f"{orig}: dtype {dt} (expected U8)")
            refs.append((shard_file, data_start + o0, o1 - o0, shape))
        psh, poff, pnb, pshape = refs[0]
        ssh, soff, snb, sshape = refs[1]
        self.bytes_read += pnb + snb
        r, c = pshape[0], pshape[1] * 2
        if pnb != r * c // 2 or snb != r * c // 32 or list(sshape) != [r, c // 32]:
            raise SystemExit(f"{base}: packed/scale shapes inconsistent: {pshape} {sshape}")
        packed = self._mm(psh)[poff: poff + pnb]
        scales = self._mm(ssh)[soff: soff + snb]
        return packed, scales, (r, c)

    # ── internals ──

    def _derive_cfg(self):
        """NANO-keyed cfg: dims from the actual tensor shapes (ground truth),
        scalars from config.json when present (mirror of the slicer's
        safetensors derive_config)."""
        c = {}
        j = self.hf_config or {}
        la = j.get("linear_attn_config") or {}

        def num(k, dflt):
            v = j.get(k)
            return type(dflt)(v) if v is not None else dflt

        embed = self.shape_of("embed_tokens.weight")
        c["vocab"], c["hidden"] = int(embed[0]), int(embed[1])
        c["n_layers"] = self.n_layers
        c["n_experts"] = self.n_experts
        c["top_k"] = num("num_experts_per_token", 16)
        c["n_shared"] = num("num_shared_experts", 2)
        dense_layers = [l for l in range(self.n_layers) if not self.layer_moe[l]]
        c["dense_layers"] = dense_layers
        c["first_k_dense"] = num("first_k_dense_replace", len(dense_layers))
        c["attn_res_block"] = num("attn_res_block_size", 4)
        c["rms_eps"] = float(j.get("rms_norm_eps", 1e-5))
        glb = la.get("gate_lower_bound", j.get("gate_lower_bound", -5.0))
        c["gate_lower_bound"] = float(glb)
        c["mla_layers"] = [l for l in range(self.n_layers) if self.layer_attn[l] == "mla"]

        def lshape(l, tail):
            return self.shape_of(f"layers.{l}.{tail}")

        kda_l = next((l for l in range(self.n_layers) if self.layer_attn[l] == "kda"), None)
        if kda_l is not None:
            c["kda_heads"] = int(lshape(kda_l, "self_attn.b_proj.weight")[0])
            c["kda_dim"] = int(np.prod(lshape(kda_l, "self_attn.o_norm.weight")))
            q_proj = lshape(kda_l, "self_attn.q_proj.weight")
            assert c["kda_heads"] * c["kda_dim"] == q_proj[0], "kda dims inconsistent"
            c["kda_conv"] = int(lshape(kda_l, "self_attn.q_conv1d.weight")[-1])
            c["kda_fa_rank"] = int(lshape(kda_l, "self_attn.f_a_proj.weight")[0])
        mla_l = next((l for l in range(self.n_layers) if self.layer_attn[l] == "mla"), None)
        if mla_l is not None:
            c["mla_q_lora"] = int(lshape(mla_l, "self_attn.q_a_proj.weight")[0])
            c["mla_kv_lora"] = int(np.prod(lshape(mla_l, "self_attn.kv_a_layernorm.weight")))
            kv_a_rows = int(lshape(mla_l, "self_attn.kv_a_proj_with_mqa.weight")[0])
            c["mla_rope"] = kv_a_rows - c["mla_kv_lora"]
            qb_rows = int(lshape(mla_l, "self_attn.q_b_proj.weight")[0])
            kvb_rows = int(lshape(mla_l, "self_attn.kv_b_proj.weight")[0])
            if all(k in j for k in ("num_attention_heads", "qk_nope_head_dim", "v_head_dim")):
                c["mla_heads"] = int(j["num_attention_heads"])
                c["mla_nope"] = int(j["qk_nope_head_dim"])
                c["mla_v"] = int(j["v_head_dim"])
            else:
                # underdetermined shapes: assume v == kda_dim (slicer rule)
                c["mla_v"] = c["kda_dim"]
                assert (kvb_rows - qb_rows) % (c["mla_v"] - c["mla_rope"]) == 0, (
                    "MLA dims underdetermined without config.json")
                c["mla_heads"] = (kvb_rows - qb_rows) // (c["mla_v"] - c["mla_rope"])
                assert qb_rows % (c["mla_v"] + c["mla_rope"]) == 0, "MLA q_b rows inconsistent"
                c["mla_nope"] = qb_rows // c["mla_heads"] - c["mla_rope"]
            assert c["mla_heads"] * (c["mla_nope"] + c["mla_rope"]) == qb_rows, "MLA q_b rows mismatch"
            assert c["mla_heads"] * (c["mla_nope"] + c["mla_v"]) == kvb_rows, "MLA kv_b rows mismatch"
        moe_l = next((l for l in range(self.n_layers) if self.layer_moe[l]), None)
        if moe_l is not None:
            p = f"layers.{moe_l}.block_sparse_moe."
            gate_rows = int(self.shape_of(p + "gate.weight")[0])
            assert gate_rows == self.n_experts, "router rows != expert count"
            c["routed_hidden"] = int(np.prod(self.shape_of(p + "routed_expert_norm.weight")))
            c["shared_inter"] = int(self.shape_of(p + "shared_experts.gate_proj.weight")[0])
            _, _, (w1r, w1c) = self.expert_parts(p + "experts.0.w1")
            c["moe_inter"] = w1r
            assert w1c == c["routed_hidden"], "expert w1 cols != routed_hidden"
            _, _, w2 = self.expert_parts(p + "experts.0.w2")
            assert w2 == (c["routed_hidden"], c["moe_inter"]), "expert w2 shape mismatch"
        if dense_layers:
            c["dense_inter"] = int(lshape(dense_layers[0], "mlp.gate_proj.weight")[0])
        return c


_E2M1_TABLES = {}


def _e2m1(dev):
    t = _E2M1_TABLES.get(str(dev))
    if t is None:
        t = torch.tensor(E2M1, dtype=torch.float32, device=dev)
        _E2M1_TABLES[str(dev)] = t
    return t


def dequant_mxfp4_dev(packed, scales, dims, dev):
    """Batched MXFP4 dequant on `dev` - torch mirror of bin2pt.dequant_mxfp4
    (same rule: e2m1 nibble * 2^(scale-127), groups of 32 along the row).
    packed uint8 [G, R*C/2], scales uint8 [G, R*C/32] -> f32 [G, R, C]."""
    r, c = dims
    p = torch.from_numpy(packed).to(dev).view(-1, r, c // 2)
    s = torch.from_numpy(scales).to(dev).view(-1, r, c // 32)
    idx = torch.stack((p & 0x0F, p >> 4), dim=-1).view(p.shape[0], r, c)
    vals = _e2m1(dev)[idx.long()]
    gain = torch.exp2(s.to(torch.float32) - 127.0).unsqueeze(-1)
    return (vals.view(p.shape[0], r, c // 32, 32) * gain).view(p.shape[0], r, c)


class StExpertStore:
    """Packed MXFP4 blobs of the routed experts of ONE MoE layer: memmap
    slices of the safetensors shards, dequantized on the compute device group
    by group (only the router-selected experts are ever read)."""

    def __init__(self, reader, layer):
        self.reader = reader
        self.pfx = f"layers.{layer}.block_sparse_moe.experts."
        self.n_experts = reader.n_experts

    def chunk(self, eids, dev):
        """Dequantized (w1, w3, w2) stacks [len(eids), ...] f32 on `dev`."""
        out = []
        for kind in ("w1", "w3", "w2"):
            packs, scales, dims = [], [], None
            for e in eids:
                p, s, dims = self.reader.expert_parts(self.pfx + f"{e}.{kind}")
                packs.append(p)
                scales.append(s)
            out.append(dequant_mxfp4_dev(np.stack(packs), np.stack(scales), dims, dev))
        return out


# rows of a grouped expert assignment processed per device batch (bounds the
# gathered-token working set on the compute device)
MOE_TOKEN_BLOCK = int(os.environ.get("NANO_CAPTURE_MOE_TOKENS", str(1 << 18)))


class StStreamedMoe(TrainableSparseMoe):
    """TrainableSparseMoe whose routed experts stay packed in the safetensors
    shards. Driven in two phases by CaptureModel (layers-outer capture):

      1. per chunk, while the layer weights are resident: gate, latent
         down-proj and shared experts run per row block (phase_a);
      2. once per layer: each router-SELECTED expert group is fetched and
         dequantized exactly once and applied to ALL captured tokens routed
         to it, gathered across the whole residual plane (dispatch_plane),
         then latent norm/up-proj and the shared residual are combined per
         row block (phase_c).

    Same grouping, capacity rule and combine as moe_train_bmm_fast, fed from
    the full plane instead of one chunk. forward() is never used (CaptureModel
    swaps a recorder in during the per-chunk attention pass)."""

    def __init__(self, config, store):
        super().__init__(config)
        self.experts = torch.nn.ModuleList()  # experts live in the store, not as params
        self.store = store

    def phase_a(self, hb):
        """hb [m, D] on the compute device (post-attention-layernorm input of
        the MoE, one row block of the plane) -> (topk_ids, topk_weight,
        latent_x, shared) cpu tensors. Mirrors the pre-dispatch part of
        TrainableSparseMoe.forward."""
        ids, w = self.gate(hb.unsqueeze(0))  # [m, k] each
        x = self.routed_expert_down_proj(hb) if self.use_latent_moe else hb
        shared = None
        if self.config.num_shared_experts is not None:
            shared = self.shared_experts(hb).cpu()
        return ids.cpu(), w.cpu(), x.cpu(), shared

    def dispatch_plane(self, x_plane, ids_all, w_all, dev):
        """x_plane [n, H] cpu (latent MoE input of every captured token),
        ids_all [n, k], w_all [n, k] -> combined expert output [n, H] cpu.
        Each selected expert group is dequantized once and applied to all of
        its tokens across the whole plane."""
        n, k = ids_all.shape
        e_max = self.store.n_experts
        h_dim = x_plane.shape[-1]
        flat_ids = ids_all.reshape(-1)
        counts = torch.bincount(flat_ids, minlength=e_max)
        starts = counts.cumsum(0) - counts
        order = flat_ids.argsort()
        sorted_eids = flat_ids[order]
        counts_np = counts.numpy()
        starts_np = starts.numpy()
        cap_budget = max(64, (3 * n * k + e_max - 1) // e_max)
        selected = np.nonzero(counts_np)[0]  # only router-selected experts
        y_flat = torch.empty(n * k, h_dim, dtype=torch.float32)
        rows_of = order // k  # plane row of each sorted assignment
        starts_dev = starts.to(dev)
        for i in range(0, len(selected), MOE_CHUNK):
            group = selected[i: i + MOE_CHUNK]
            e_lo, e_hi = int(group[0]), int(group[-1])
            # tokens of the group are one contiguous slice of the sorted order
            a = int(starts_np[e_lo])
            b = int(starts_np[e_hi] + counts_np[e_hi])
            gt = torch.from_numpy(group).to(dev)
            cap_max = int(counts_np[group].max())
            cap = min(cap_max, cap_budget)
            # the group is fetched and dequantized ONCE for the whole plane
            w1, w3, w2 = self.store.chunk([int(e) for e in group], dev)
            for s0 in range(a, b, MOE_TOKEN_BLOCK):
                s1 = min(s0 + MOE_TOKEN_BLOCK, b)
                toks = x_plane[rows_of[s0:s1]].to(dev)
                eids_dev = sorted_eids[s0:s1].to(dev)
                eids_l = torch.searchsorted(gt, eids_dev)
                slot = torch.arange(s0, s1, device=dev) - starts_dev[eids_dev]
                out = self._expert_chunk(toks, eids_l, slot, w1, w3, w2,
                                         cap, len(group), cap_max)
                y_flat[order[s0:s1]] = out.cpu()
        return (y_flat * w_all.reshape(-1, 1)).view(n, k, h_dim).sum(1)

    def phase_c(self, y, shared, dev):
        """y [m, H] cpu combined expert output of one row block -> [m, D] cpu:
        latent norm + up-proj, plus the shared-experts residual (the
        post-dispatch part of TrainableSparseMoe.forward)."""
        yb = y.to(dev)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                yb = self.routed_expert_norm(yb)
            yb = self.routed_expert_up_proj(yb)
        if shared is not None:
            yb = yb + shared.to(dev)
        return yb.cpu()


_EMPTY = torch.empty(0)

# rows per embedding gather / boundary write / phase-A block (bounds the
# transient host and device working sets)
ROW_BLOCK = int(os.environ.get("NANO_CAPTURE_ROW_BLOCK", "8192"))


class _MoeRecorder(torch.nn.Module):
    """Stand-in for the MoE block during the per-chunk attention pass of a
    MoE layer: records the post-attention-layernorm input and returns zeros.
    The layer output is then base + 0 (exact: adding a zero matrix is the
    identity), and the real MoE output is added onto the plane after the
    plane-wide expert dispatch (base + moe_out == the vendor combine)."""

    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, hidden_states):
        self.inputs.append(hidden_states.detach()[0].cpu())
        return torch.zeros_like(hidden_states)


class CaptureModel:
    """The safetensors-backed model: meta-device NanoModel whose frozen params
    are (shard, byte range) references, materialized layer by layer during the
    forward and released right after (the heal_stream.py streaming pattern,
    sourced from safetensors instead of a .bin).

    The capture itself is LAYERS-OUTER / CHUNKS-INNER (capture is
    forward-only and chunks are independent sequences, so each layer's
    weights, including the packed experts, are read from the shards exactly
    once for the whole run): a current-residual plane [n_tokens, hidden] f32
    is kept in host RAM, every layer transforms all chunk slices of it in
    turn, and boundary l+1 is appended to the fp16 memmap when layer l is
    done. The attn_res block residuals of every chunk are kept on the host
    across layers (they accumulate one block every attn_res_block layers)."""

    def __init__(self, st_root, dev):
        t0 = time.time()
        self.reader = StRoot(st_root)
        cfg = self.reader.cfg
        self.cfg = cfg
        self.dev = dev
        with torch.device("meta"):
            model = NanoModel(cfg)  # zero allocation: every param is meta
        model.eval()  # required by the MoE gate assert; no dropout/BN in the arch

        # swap every MoE block for the packed-experts version (meta device
        # too: its spine submodules resolve to safetensors ranges below, and
        # its expert Linears must never be allocated)
        n_moe = 0
        with torch.device("meta"):
            for l, layer in enumerate(model.layers):
                if hasattr(layer, "block_sparse_moe"):
                    layer.block_sparse_moe = StStreamedMoe(
                        model.config, StExpertStore(self.reader, l))
                    n_moe += 1
        for p in model.parameters():
            p.requires_grad = False
        model.eval()  # gate assert + swapped MoE blocks default to training=True

        # resolve every frozen param to its safetensors tensor; per-layer
        # params are loaded inside the layer's forward window, the trunk
        # (final norm, output attn_res, lm_head) is verified present but never
        # loaded (boundary n_layers is the pre-trunk stream), and the
        # embedding is gathered row by row straight from the shard
        missing = []
        self.layer_params = [[] for _ in range(cfg["n_layers"])]
        swaps = []  # (module, leaf, empty Parameter): meta .data assignment is
        # refused, so every layer param object is swapped for a real (empty)
        # CPU Parameter once, and only its .data flips per layer window
        for name, p in model.named_parameters():
            if name == "embed_tokens.weight":
                if not self.reader.has(name):
                    missing.append(name)
                continue
            if not self.reader.has(name):
                missing.append(name)
                continue
            m = re.match(r"layers\.(\d+)\.", name)
            if m:
                shape = tuple(p.shape)
                new_p = torch.nn.Parameter(torch.empty(0), requires_grad=False)
                swaps.append((name, new_p))
                self.layer_params[int(m.group(1))].append((new_p, name, shape))
            # else: trunk tensor, audited but unused by the capture
        for name, new_p in swaps:
            mod = model.get_submodule(name.rpartition(".")[0])
            mod._parameters[name.rpartition(".")[2]] = new_p
        if missing:
            raise SystemExit(f"params with no safetensors tensor: {missing[:6]} "
                             f"({len(missing)} total)")
        self.model = model
        self.n_moe = n_moe
        self._mla, _ = layer_types(cfg)
        n_kda = self.reader.layer_attn.count("kda")
        print(f"capture_bounds: {st_root} - {cfg['n_layers']} layers "
              f"({n_kda} KDA + {cfg['n_layers'] - n_kda} MLA), "
              f"{n_moe} MoE + {cfg['n_layers'] - n_moe} dense, hidden {cfg['hidden']}, "
              f"vocab {cfg['vocab']}, {cfg['n_experts']} experts top-{cfg['top_k']} "
              f"+ {cfg['n_shared']} shared ({time.time() - t0:.1f} s)", flush=True)

    def _layer_to_dev(self, l):
        for p, name, shape in self.layer_params[l]:
            t = torch.from_numpy(self.reader.fetch_f32(name))
            if tuple(t.shape) != shape:
                # e.g. conv1d stored [C, K] vs [C, 1, K], or a flattened [1, D] proj
                assert t.numel() == math.prod(shape), f"{name}: {tuple(t.shape)} != {shape}"
                t = t.reshape(shape)
            p.data = t.to(self.dev)

    def _layer_release(self, l):
        for p, _, _ in self.layer_params[l]:
            p.data = _EMPTY

    def _causal(self, T, dev, dtype):
        if T not in self._masks:
            m = torch.zeros(1, 1, T, T, device=dev, dtype=dtype)
            m.masked_fill_(
                torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1),
                float("-inf"),
            )
            self._masks[T] = m
        return self._masks[T]

    @torch.no_grad()
    def capture_all(self, tokens, n_tokens, seq, bounds_mm, t_start):
        """Layers-outer capture: fills bounds_mm[:, 0..n_layers] for the first
        n_tokens of `tokens`, processed as chunks of seq (last one short)."""
        dev = self.dev
        D = self.cfg["hidden"]
        n_layers = self.cfg["n_layers"]
        self._masks = {}
        chunks = [(c * seq, min((c + 1) * seq, n_tokens))
                  for c in range(math.ceil(n_tokens / seq))]

        # residual plane + boundary 0 (embedding output)
        plane = torch.empty(n_tokens, D, dtype=torch.float32)
        for r0 in range(0, n_tokens, ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, n_tokens)
            plane[r0:r1] = torch.from_numpy(
                self.reader.gather_rows("embed_tokens.weight",
                                        np.asarray(tokens[r0:r1])))
        self._write_boundary(bounds_mm, 0, plane)

        # per-chunk attn_res block residuals, kept on the host across layers
        blocks = [torch.zeros(t1 - t0, 0, D) for t0, t1 in chunks]

        for l, layer in enumerate(self.model.layers):
            t_l = time.time()
            self._layer_to_dev(l)
            if hasattr(layer, "block_sparse_moe"):
                self._moe_layer_forward(l, layer, plane, blocks, chunks)
            else:
                for c, (t0, t1) in enumerate(chunks):
                    h = plane[t0:t1].to(dev).unsqueeze(0)
                    mask = self._causal(t1 - t0, dev, h.dtype) if l in self._mla else None
                    h, b = layer._forward_attn_residual(
                        h, attention_mask=mask, block_residual=blocks[c].to(dev))
                    plane[t0:t1] = h[0].cpu()
                    blocks[c] = b.cpu()
            self._layer_release(l)
            self._write_boundary(bounds_mm, l + 1, plane)
            bounds_mm.flush()  # boundaries stream to disk as each layer completes
            dt = time.time() - t_start
            eta = dt / (l + 1) * (n_layers - l - 1)
            desc = f"{self.reader.layer_attn[l]}+{'moe' if self.reader.layer_moe[l] else 'dense'}"
            print(f"  layer {l + 1}/{n_layers} ({desc}): {dt / 60:.1f} min elapsed, "
                  f"{self.reader.bytes_read / 1e9:.1f} GB read so far, "
                  f"eta {eta / 60:.1f} min (rss {rss_gb():.1f} GB)", flush=True)
        return plane

    def _moe_layer_forward(self, l, layer, plane, blocks, chunks):
        """One MoE layer over the whole plane: per-chunk attention pass with a
        recorder stand-in, then ONE plane-wide expert dispatch, then the
        residual combine."""
        dev = self.dev
        D = self.cfg["hidden"]
        moe = layer.block_sparse_moe
        rec = _MoeRecorder()
        layer.block_sparse_moe = rec
        n_tokens = plane.shape[0]
        h2 = torch.empty(n_tokens, D, dtype=torch.float32)
        for c, (t0, t1) in enumerate(chunks):
            h = plane[t0:t1].to(dev).unsqueeze(0)
            mask = self._causal(t1 - t0, dev, h.dtype) if l in self._mla else None
            h, b = layer._forward_attn_residual(
                h, attention_mask=mask, block_residual=blocks[c].to(dev))
            plane[t0:t1] = h[0].cpu()  # base (MoE contribution still zero)
            h2[t0:t1] = rec.inputs[-1]
            blocks[c] = b.cpu()
        layer.block_sparse_moe = moe

        # phase A per row block: gate, latent down-proj, shared experts
        k = self.cfg["top_k"]
        rh = self.cfg["routed_hidden"] if moe.use_latent_moe else D
        ids_all = torch.empty(n_tokens, k, dtype=torch.int64)
        w_all = torch.empty(n_tokens, k, dtype=torch.float32)
        x_plane = torch.empty(n_tokens, rh, dtype=torch.float32)
        shared = None
        if moe.config.num_shared_experts is not None:
            shared = torch.empty(n_tokens, D, dtype=torch.float32)
        for r0 in range(0, n_tokens, ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, n_tokens)
            ids, w, x, sh = moe.phase_a(h2[r0:r1].to(dev))
            ids_all[r0:r1], w_all[r0:r1], x_plane[r0:r1] = ids, w, x
            if shared is not None:
                shared[r0:r1] = sh
        del h2

        # one plane-wide dispatch: every selected expert is fetched and
        # dequantized exactly once for all of its tokens
        y = moe.dispatch_plane(x_plane, ids_all, w_all, dev)
        del x_plane, ids_all, w_all

        # phase C + residual combine onto the plane
        for r0 in range(0, n_tokens, ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, n_tokens)
            plane[r0:r1] += moe.phase_c(
                y[r0:r1], shared[r0:r1] if shared is not None else None, dev)

    @staticmethod
    def _write_boundary(bounds_mm, b, plane):
        """bounds_mm[:, b, :] = plane, in row blocks (strided fp16 writes into
        the token-major memmap, gentle on the page cache)."""
        for r0 in range(0, plane.shape[0], ROW_BLOCK):
            r1 = min(r0 + ROW_BLOCK, plane.shape[0])
            bounds_mm[r0:r1, b, :] = plane[r0:r1].numpy().astype(np.float16)


def block_influence(mm, layer_desc, row_mask=None):
    """Per-layer cosine similarity between boundary_in (bounds[:, l]) and
    boundary_out (bounds[:, l+1]) over the captured tokens, streamed over row
    blocks of the memmap. Returns a list of per-layer stat dicts."""
    n_tokens, n_bounds, D = mm.shape
    L = n_bounds - 1
    sums = np.zeros(L, dtype=np.float64)
    mins = np.full(L, np.inf)
    maxs = np.full(L, -np.inf)
    cnt = 0
    for r0 in range(0, n_tokens, BI_ROWS):
        r1 = min(r0 + BI_ROWS, n_tokens)
        block = torch.from_numpy(np.array(mm[r0:r1])).float()
        if row_mask is not None:
            block = block[torch.from_numpy(np.asarray(row_mask[r0:r1]))]
        if block.shape[0] == 0:
            continue
        cos = F.cosine_similarity(block[:, :-1, :], block[:, 1:, :], dim=-1)
        cos = cos.double().numpy()  # [rows, L]
        sums += cos.sum(0)
        mins = np.minimum(mins, cos.min(0))
        maxs = np.maximum(maxs, cos.max(0))
        cnt += block.shape[0]
    means = sums / max(1, cnt)
    stats = [
        {"layer": l, "type": layer_desc[l],
         "cos_mean": float(means[l]), "cos_min": float(mins[l]), "cos_max": float(maxs[l])}
        for l in range(L)
    ]
    return stats, cnt, means


def print_bi_report(stats, cnt):
    print(f"Block Influence (boundary_in vs boundary_out cosine, {cnt} tokens):", flush=True)
    print("  layer  type          cos_mean   cos_min    cos_max", flush=True)
    for s in stats:
        # genuinely near-identity layers (mean cos >= 0.99) are the slice candidates
        flag = "  <- near-identity" if s["cos_mean"] >= 0.99 else ""
        print(f"  {s['layer']:5d}  {s['type']:<12s} {s['cos_mean']:.6f}   "
              f"{s['cos_min']:.6f}   {s['cos_max']:.6f}{flag}", flush=True)
    top = sorted(stats, key=lambda s: -s["cos_mean"])[: min(5, len(stats))]
    listing = ", ".join(f"layer {s['layer']} ({s['cos_mean']:.6f})" for s in top)
    print(f"  most near-identity layers: {listing}", flush=True)


def run_capture(args):
    dev = args.device
    torch.manual_seed(args.seed)
    tokens = load_tokens(args.data)
    n_want = min(args.max_tokens, len(tokens))
    if n_want <= 0:
        raise SystemExit(f"{args.data}: no tokens to capture")
    n_chunks = math.ceil(n_want / args.seq)
    n_tokens = n_want  # the last chunk may be short

    cm = CaptureModel(args.st_root, dev)
    cfg = cm.cfg
    n_layers, D = cfg["n_layers"], cfg["hidden"]
    n_bounds = n_layers + 1
    layer_desc = [
        f"{cm.reader.layer_attn[l]}+{'moe' if cm.reader.layer_moe[l] else 'dense'}"
        for l in range(n_layers)
    ]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    bounds_path = args.out + ".bounds.npy"
    meta_path = args.out + ".meta.json"
    bounds_mm = np.lib.format.open_memmap(
        bounds_path, mode="w+", dtype=np.float16, shape=(n_tokens, n_bounds, D))
    print(f"capture: {n_tokens} tokens in {n_chunks} chunks of {args.seq}, "
          f"{n_bounds} bounds x hidden {D} -> {bounds_path} "
          f"({n_tokens * n_bounds * D * 2 / 1e9:.2f} GB fp16)", flush=True)

    t_start = time.time()
    cm.capture_all(tokens, n_tokens, args.seq, bounds_mm, t_start)
    del bounds_mm

    row_mask = None
    if args.ignore_unk:
        row_mask = np.asarray(tokens[:n_tokens]) != args.unk_id
    mm = np.load(bounds_path, mmap_mode="r")
    stats, bi_cnt, _means = block_influence(mm, layer_desc, row_mask)
    print_bi_report(stats, bi_cnt)

    meta = {
        "n_tokens": n_tokens,
        "n_bounds": n_bounds,
        "hidden": D,
        "n_layers": n_layers,
        "layer_types": layer_desc,
        "mla_layers": sorted(cm._mla),
        "dense_layers": cfg["dense_layers"],
        "seq": args.seq,
        "max_tokens": args.max_tokens,
        "st_root": os.path.abspath(args.st_root),
        "data": os.path.abspath(args.data),
        "dtype": "float16",
        "layout": "(n_tokens, n_bounds, hidden) C-order token-major; "
                  "boundary 0 = embedding output, boundary b = output of layer b-1",
        "ignore_unk": bool(args.ignore_unk),
        "unk_id": args.unk_id if args.ignore_unk else None,
        "bi_tokens": bi_cnt,
        "seed": args.seed,
        "block_influence": stats,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"-> {bounds_path}\n-> {meta_path}", flush=True)
    return meta


# ── selftest: tiny synthetic safetensors root, full pipeline ──

TINY = dict(
    n_layers=4, hidden=64, vocab=120,
    n_experts=8, top_k=2, n_shared=1,
    kda_heads=2, kda_dim=16, kda_conv=4, kda_fa_rank=16, gate_lower_bound=-5.0,
    mla_heads=2, mla_q_lora=16, mla_kv_lora=16, mla_nope=16, mla_rope=8, mla_v=16,
    routed_hidden=32, moe_inter=32, shared_inter=32,
    dense_inter=64, attn_res_block=2, first_k_dense=1, rms_eps=1e-5,
    mla_layers=[3], dense_layers=[0],
)


def _write_shard(path, tensors):
    """tensors: [(name, dtype, shape, data_bytes)] -> one safetensors file."""
    header = {}
    off = 0
    for name, dt, shape, data in tensors:
        header[name] = {"dtype": dt, "shape": list(shape),
                        "data_offsets": [off, off + len(data)]}
        off += len(data)
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for _, _, _, data in tensors:
            f.write(data)


def _to_bf16_bytes(arr):
    f32 = np.ascontiguousarray(arr, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()


def build_tiny_root(root, seed):
    """Builds a tiny K3-layout safetensors root (same naming/layout rules as
    the real one: language_model.model.* names, language_model.lm_head, MXFP4
    experts as weight_packed + weight_scale, index.json, config.json) from a
    freshly initialized tiny NanoModel. Returns (state_dict numpy, tokens)."""
    from export import quantize_mxfp4

    torch.manual_seed(seed)
    model = NanoModel(TINY).eval()
    with torch.no_grad():
        for mod in model.modules():
            # torch.empty-initialized router bias: pin it for determinism
            if hasattr(mod, "e_score_correction_bias"):
                mod.e_score_correction_bias.zero_()
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in model.state_dict().items()}

    shard1, shard2 = [], []
    for name, arr in sorted(sd.items()):
        st = "language_model.lm_head.weight" if name == "lm_head.weight" \
            else "language_model.model." + name
        if ".experts." in name and name.endswith(".weight"):
            base = st[: -len(".weight")]
            packed, scales = quantize_mxfp4(np.ascontiguousarray(arr))
            shard1.append((base + ".weight_packed", "U8", packed.shape, packed.tobytes()))
            # scales in the OTHER shard: exercises the cross-shard expert read
            shard2.append((base + ".weight_scale", "U8", scales.shape, scales.tobytes()))
        elif name in ("norm.weight", "output_attn_res_norm.weight"):
            shard1.append((st, "F32", arr.shape, arr.tobytes()))
        else:
            shard1.append((st, "BF16", arr.shape, _to_bf16_bytes(arr)))
    _write_shard(os.path.join(root, "model-00001-of-00002.safetensors"), shard1)
    _write_shard(os.path.join(root, "model-00002-of-00002.safetensors"), shard2)
    weight_map = {}
    for name, _, _, _ in shard1:
        weight_map[name] = "model-00001-of-00002.safetensors"
    for name, _, _, _ in shard2:
        weight_map[name] = "model-00002-of-00002.safetensors"
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, f)
    config = {
        "hidden_size": TINY["hidden"], "vocab_size": TINY["vocab"],
        "num_hidden_layers": TINY["n_layers"],
        "intermediate_size": TINY["dense_inter"],
        "num_experts": TINY["n_experts"],
        "num_experts_per_token": TINY["top_k"],
        "num_shared_experts": TINY["n_shared"],
        "first_k_dense_replace": TINY["first_k_dense"],
        "attn_res_block_size": TINY["attn_res_block"],
        "rms_norm_eps": TINY["rms_eps"],
        "moe_intermediate_size": TINY["moe_inter"],
        "routed_expert_hidden_size": TINY["routed_hidden"],
        "num_attention_heads": TINY["mla_heads"],
        "q_lora_rank": TINY["mla_q_lora"], "kv_lora_rank": TINY["mla_kv_lora"],
        "qk_nope_head_dim": TINY["mla_nope"], "qk_rope_head_dim": TINY["mla_rope"],
        "v_head_dim": TINY["mla_v"],
        "linear_attn_config": {
            "num_heads": TINY["kda_heads"], "head_dim": TINY["kda_dim"],
            "short_conv_kernel_size": TINY["kda_conv"],
            "gate_lower_bound": TINY["gate_lower_bound"],
        },
    }
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(config, f)
    return sd


def selftest(args):
    """Tiny synthetic safetensors root -> full capture pipeline -> checks:
    output shapes/dtype, meta contents, boundary 0 == embeddings, the
    --ignore-unk exclusion from the Block Influence statistics, and a full
    per-boundary parity check against a directly loaded reference NanoModel
    (its experts fed with the dequantized MXFP4 blobs, so the only allowed
    deviation is the fp16 storage)."""
    from bin2pt import dequant_mxfp4

    seed = args.seed
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as tmp:
        sd = build_tiny_root(tmp, seed)
        n_tok = 3 * 32 + 16  # exercises a short final chunk
        tokens = rng.integers(0, TINY["vocab"], n_tok).astype(np.uint16)
        unk_id = 7
        tokens[5::17] = unk_id  # planted UNKs for the --ignore-unk path
        data_path = os.path.join(tmp, "tokens.bin")
        tokens.tofile(data_path)
        out_prefix = os.path.join(tmp, "cap")

        cap_args = argparse.Namespace(
            st_root=tmp, data=data_path, out=out_prefix, seq=32,
            max_tokens=n_tok, device="cpu", ignore_unk=True, unk_id=unk_id,
            seed=seed,
        )
        # count expert fetches: layers-outer capture must fetch/dequantize
        # every router-selected expert exactly ONCE per layer (not per chunk)
        fetch_calls = []
        orig_chunk = StExpertStore.chunk

        def counting_chunk(store, eids, dev):
            fetch_calls.append((store.pfx, tuple(eids)))
            return orig_chunk(store, eids, dev)

        StExpertStore.chunk = counting_chunk
        try:
            meta = run_capture(cap_args)
        finally:
            StExpertStore.chunk = orig_chunk
        per_layer = {}
        for pfx, eids in fetch_calls:
            per_layer.setdefault(pfx, []).extend(eids)
        assert len(per_layer) == 3, f"expected 3 MoE layers, got {sorted(per_layer)}"
        for pfx, ids in per_layer.items():
            assert len(ids) == len(set(ids)), (
                f"{pfx}: an expert was fetched twice - the layers-outer "
                "single-read property is broken")

        bounds = np.load(out_prefix + ".bounds.npy", mmap_mode="r")
        n_bounds = TINY["n_layers"] + 1
        assert bounds.shape == (n_tok, n_bounds, TINY["hidden"]), (
            f"bounds shape {bounds.shape} != {(n_tok, n_bounds, TINY['hidden'])}")
        assert bounds.dtype == np.float16, f"bounds dtype {bounds.dtype}"
        assert np.isfinite(np.asarray(bounds)).all(), "non-finite values in bounds"

        with open(out_prefix + ".meta.json") as f:
            mj = json.load(f)
        assert mj["n_tokens"] == n_tok and mj["n_bounds"] == n_bounds
        assert mj["hidden"] == TINY["hidden"] and mj["n_layers"] == TINY["n_layers"]
        assert mj["seq"] == 32 and mj["st_root"] == os.path.abspath(tmp)
        assert mj["layer_types"] == ["kda+dense", "kda+moe", "kda+moe", "mla+moe"], (
            f"layer_types {mj['layer_types']}")
        assert len(mj["block_influence"]) == TINY["n_layers"]
        n_unk = int((tokens == unk_id).sum())
        assert mj["bi_tokens"] == n_tok - n_unk, (
            f"bi_tokens {mj['bi_tokens']} != {n_tok - n_unk}")

        # boundary 0 is the embedding output of the captured tokens
        emb = sd["embed_tokens.weight"]
        got = np.asarray(bounds[:, 0, :]).astype(np.float32)
        want = emb[tokens]
        err = np.abs(got - want).max()
        assert np.allclose(got, want, atol=2e-3), f"boundary 0 != embeddings (max err {err})"

        # spot-check the reader against the generated weights (bf16 storage
        # rounding only: 7 mantissa bits, rel err <= 2^-7)
        q = np.asarray(StRoot(tmp).fetch_f32("layers.0.self_attn.q_proj.weight"))
        assert np.allclose(q, sd["layers.0.self_attn.q_proj.weight"], rtol=1e-2, atol=1e-4), (
            "reader bf16 fetch diverges from the source weights")

        # full parity: reference NanoModel loaded through the reader (experts
        # dequantized from the packed blobs, so both sides share the MXFP4
        # rounding) vs the captured bounds at EVERY boundary
        reader = StRoot(tmp)
        ref = NanoModel(TINY).eval()
        ref_sd = {}
        for name, p in ref.state_dict().items():
            if ".experts." in name and name.endswith(".weight"):
                packed, scales, dims = reader.expert_parts(name[: -len(".weight")])
                arr = dequant_mxfp4(bytes(packed) + bytes(scales), dims)
            else:
                arr = np.array(reader.fetch_f32(name))
            ref_sd[name] = torch.from_numpy(np.ascontiguousarray(arr)).reshape(p.shape)
        ref.load_state_dict(ref_sd)
        D = TINY["hidden"]
        max_abs = 0.0
        with torch.no_grad():
            for c in range(math.ceil(n_tok / 32)):
                t0, t1 = c * 32, min((c + 1) * 32, n_tok)
                ids = torch.from_numpy(tokens[t0:t1].astype(np.int64)).unsqueeze(0)
                h = ref.embed_tokens(ids)
                T = t1 - t0
                causal = torch.zeros(1, 1, T, T)
                causal.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
                                    float("-inf"))
                blocks = h.new_zeros(T, 0, D)
                hs = [h[0]]
                for l, layer in enumerate(ref.layers):
                    mask = causal if l in ref._mla else None
                    h, blocks = layer._forward_attn_residual(
                        h, attention_mask=mask, block_residual=blocks)
                    hs.append(h[0])
                ref_b = torch.stack(hs).numpy()  # [n_bounds, T, D]
                got_b = np.asarray(bounds[t0:t1]).astype(np.float32).transpose(1, 0, 2)
                max_abs = max(max_abs, float(np.abs(got_b - ref_b).max()))
                # same math on both sides (verified bitwise for the MoE
                # dispatch): only fp16 storage noise is allowed
                assert np.allclose(got_b, ref_b, atol=5e-4, rtol=1e-2), (
                    f"chunk {c}: captured bounds diverge from the reference model")
        print(f"selftest: bounds {bounds.shape} fp16, meta OK, "
              f"boundary0 == embeddings (max err {err:.2e}), "
              f"all-boundary parity vs reference (max abs diff {max_abs:.2e}), "
              f"bi over {mj['bi_tokens']} tokens ({n_unk} UNK excluded)", flush=True)
    print("capture_bounds selftest OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st-root", help="safetensors root (model.safetensors.index.json + shards)")
    ap.add_argument("--data", help="tokens.bin (uint16, or uint32 with a .meta.json sibling)")
    ap.add_argument("--out", help="output prefix for PREFIX.bounds.npy / PREFIX.meta.json")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=50000)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--ignore-unk", action="store_true",
                    help="exclude UNK positions from the Block Influence statistics "
                         "(see train.py; the positions are still captured)")
    ap.add_argument("--unk-id", type=int, default=8198,
                    help="UNK token id for --ignore-unk (8200-vocab nano default; "
                         "check the tokenizer of the target model)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--selftest", action="store_true",
                    help="tiny synthetic safetensors root, full pipeline, then exit")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")
    print(f"device: {args.device}", flush=True)

    if args.selftest:
        selftest(args)
        return
    for req in ("st_root", "data", "out"):
        if getattr(args, req) is None:
            raise SystemExit(f"--{req.replace('_', '-')} is required (or use --selftest)")
    run_capture(args)


if __name__ == "__main__":
    main()
