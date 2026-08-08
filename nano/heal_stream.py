#!/usr/bin/env python3
"""nanokimi - heal_stream: LoRA healing of a FULL-SIZE microkimi .bin without
the bin2pt conversion (no giant dequantized .pt, the model is never fully
materialized in host RAM or VRAM).

Why: bin2pt.py dequantizes every MXFP4 expert to fp32, which at v3 scale
(96 GB .bin) means a 130+ GB checkpoint that fits neither on disk nor in RAM,
and a full fp32 forward+backward that fits in no GPU. This trainer instead:

  - reads the .bin directory directly (bin2pt.read_bin) and keeps every base
    tensor where it is: spine (attention/embeddings/norms, fp32) as mmap views,
    routed experts as PACKED mxfp4 blobs (never dequantized in host memory);
  - instantiates the model on the meta device (NanoModel(cfg) under
    torch.device("meta"), zero allocation) and points each frozen parameter at
    its mmap-backed CPU tensor;
  - streams ONE decoder layer at a time to the GPU during forward, wrapped in
    a custom autograd Function that re-streams the same layer during backward
    (gradient checkpointing at layer granularity: only layer boundaries are
    retained); base weights are frozen, so no optimizer states exist for them;
  - dequantizes routed experts chunk by chunk (NANO_HEAL_MOE_CHUNK, default 8)
    on the compute device, INSIDE the checkpointed expert block, so a chunk is
    re-dequantized during backward instead of being retained;
  - trains only LoRA adapters on the attention projections (model_nano's
    apply_lora / LoRALinear, rank 8 by default), which stay resident on the
    GPU with their AdamW states (a few hundred MB).

Checkpoints hold ONLY the trainable tensors (adapters, optionally norms) plus
optimizer/step/rng/cfg - a few hundred MB. The merge back into a loadable
model is apply_lora_bin.py, which patches the fp32 attention tensors of a COPY
of the .bin in place (no dequantization, no requantization, byte-identical
everywhere else).

examples:
  # smoke on a small .bin (CPU works):
  python3 heal_stream.py --model smoke.bin --data tokens.bin --out out_heal \
      --lora 8 --seq 128 --steps 50 --device cpu --ignore-unk
  # numerical selftest: streamed forward vs the bin2pt reference model:
  python3 heal_stream.py --model smoke.bin --data tokens.bin --out /tmp/x --selftest
  # v3 run on the L4 box (see report for the measured step time):
  python3 heal_stream.py --model /mnt/k3/microkimi-v3.bin \
      --data ~/fineweb_k3/tokens.bin --out ~/heal_v3 \
      --lora 8 --ignore-unk --unk-id 163585 --seq 512 --steps 3000 --device cuda
"""
import argparse
import gc
import json
import math
import os
import re
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from bin2pt import read_bin, CFG_KEYS, E2M1, DTYPE_F32, DTYPE_MXFP4, DTYPE_I32, DTYPE_VQ1
from model_nano import NanoModel, TrainableSparseMoe, apply_lora, layer_types
from vendor.moonshot.modeling_kimi_linear import _apply_attn_res
from train import load_tokens, make_batch, lr_at, rss_gb

# mmap tensors are read-only; torch.from_numpy would warn once per tensor
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

# experts are dequantized chunk by chunk on the compute device (VRAM bound)
MOE_CHUNK = int(os.environ.get("NANO_HEAL_MOE_CHUNK", "8"))

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


class ExpertStore:
    """Packed MXFP4 blobs of the routed experts of ONE MoE layer: mmap slices
    of the .bin, dequantized on the compute device chunk by chunk."""

    def __init__(self, mm, blobs):
        self.mm = mm  # whole-file uint8 memmap
        self.blobs = blobs  # {"w1"|"w2"|"w3": [(offset, dims, size), ...]}
        self.n_experts = len(blobs["w1"])

    def chunk(self, e0, e1, dev):
        """Dequantized (w1, w3, w2) stacks [e1-e0, ...] f32 on `dev`."""
        out = []
        for kind in ("w1", "w3", "w2"):
            packs, scales, dims = [], [], None
            for e in range(e0, e1):
                off, dims, size = self.blobs[kind][e]
                r, c = dims
                npb = r * c // 2
                raw = self.mm[off: off + size]
                packs.append(raw[:npb])
                scales.append(raw[npb:])
            out.append(dequant_mxfp4_dev(np.stack(packs), np.stack(scales), dims, dev))
        return out


class StreamedSparseMoe(TrainableSparseMoe):
    """TrainableSparseMoe whose routed experts stay packed in the .bin: each
    chunk of MOE_CHUNK experts is dequantized on the compute device INSIDE the
    checkpointed expert block, so backward re-dequantizes the chunk instead of
    retaining all dequantized experts (the memory cliff at full scale)."""

    def __init__(self, config, store):
        super().__init__(config)
        self.experts = torch.nn.ModuleList()  # experts live in the store, not as params
        self.store = store

    # all three dispatch paths of TrainableSparseMoe.forward land here
    def moe_train(self, x, topk_ids, topk_weight):
        return self._moe_streamed(x, topk_ids, topk_weight)

    def moe_train_bmm(self, x, topk_ids, topk_weight):
        return self._moe_streamed(x, topk_ids, topk_weight)

    def moe_train_bmm_fast(self, x, topk_ids, topk_weight):
        return self._moe_streamed(x, topk_ids, topk_weight)

    def _dequanted_chunk(self, toks, eids_l, slot_l, e0, gsz, cap, cap_max):
        w1, w3, w2 = self.store.chunk(e0, e0 + gsz, toks.device)
        return self._expert_chunk(toks, eids_l, slot_l, w1, w3, w2, cap, gsz, cap_max)

    def _moe_streamed(self, x, topk_ids, topk_weight):
        """Same grouping, capacity rule and combine as moe_train_bmm_fast;
        only the expert weights come from the packed store instead of a stack."""
        n, k = topk_ids.shape
        e_max = self.store.n_experts
        h_dim = x.shape[-1]
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weight.reshape(-1, 1)
        rep = x.unsqueeze(1).expand(n, k, h_dim).reshape(n * k, h_dim)
        order = flat_ids.argsort()
        sorted_tokens = rep[order]
        counts = torch.bincount(flat_ids, minlength=e_max)
        starts = counts.cumsum(0) - counts
        sorted_eids = flat_ids[order]
        slot = torch.arange(n * k, device=x.device) - starts[sorted_eids]
        counts_np = counts.cpu().numpy()  # one host sync per call (chunk bounds)
        cap_budget = max(64, (3 * n * k + e_max - 1) // e_max)
        out_sorted = torch.empty_like(sorted_tokens)
        for e0 in range(0, e_max, MOE_CHUNK):
            e1 = min(e0 + MOE_CHUNK, e_max)
            m = int(counts_np[e0:e1].sum())
            if m == 0:
                continue
            a = int(counts_np[:e0].sum())  # tokens of a chunk are contiguous (sorted)
            cap_max = int(counts_np[e0:e1].max())
            cap = min(cap_max, cap_budget)
            out_sorted[a: a + m] = torch.utils.checkpoint.checkpoint(
                self._dequanted_chunk,
                sorted_tokens[a: a + m],
                sorted_eids[a: a + m] - e0,
                slot[a: a + m],
                e0, e1 - e0, cap, cap_max,
                use_reentrant=False,
            )
        unsorted = torch.empty_like(out_sorted)
        unsorted[order] = out_sorted
        return (unsorted * flat_w).view(n, k, -1).sum(1)


class _Swap:
    """Context manager: move a module's mmap-backed frozen params to the
    compute device, point them back at the CPU views on exit (the VRAM copy is
    released between layers). Trainable params have no _cpu_data and stay
    resident on the device."""

    def __init__(self, module, dev):
        self.dev = dev
        self.pairs = [
            (p, p._cpu_data) for p in module.parameters()
            if not p.requires_grad and hasattr(p, "_cpu_data")
        ]

    def __enter__(self):
        for p, cpu in self.pairs:
            p.data = cpu.to(self.dev)
        return self

    def __exit__(self, *exc):
        for p, cpu in self.pairs:
            p.data = cpu
        return False


class _StreamedLayer(torch.autograd.Function):
    """One decoder layer with streamed frozen weights: forward runs under
    no_grad (only the boundary tensors are saved); backward re-streams the
    layer, recomputes its forward with autograd on, and backprops - gradient
    checkpointing at layer granularity, with the weights themselves streamed."""

    @staticmethod
    def forward(ctx, hidden, blocks, layer, mask, dev):
        with _Swap(layer, dev), torch.no_grad():
            h, b = layer._forward_attn_residual(
                hidden, attention_mask=mask, block_residual=blocks
            )
        ctx.save_for_backward(hidden, blocks)
        ctx.layer, ctx.mask, ctx.dev = layer, mask, dev
        return h, b

    @staticmethod
    def backward(ctx, dh, db):
        hidden, blocks = ctx.saved_tensors
        need_h, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        with _Swap(ctx.layer, ctx.dev):
            hin = hidden.detach().requires_grad_(need_h)
            bin_ = blocks.detach().requires_grad_(need_b)
            with torch.enable_grad():
                h, b = ctx.layer._forward_attn_residual(
                    hin, attention_mask=ctx.mask, block_residual=bin_
                )
            torch.autograd.backward((h, b), (dh, db))
        return hin.grad, bin_.grad if need_b else None, None, None, None


class _StreamedLinear(torch.autograd.Function):
    """y = x @ W.T with a frozen mmap-backed W streamed per call (lm_head:
    several GB at full scale, needed again in backward for grad w.r.t. x)."""

    @staticmethod
    def forward(ctx, x, cpu_w, dev):
        with torch.no_grad():
            y = F.linear(x, cpu_w.to(dev))
        ctx.save_for_backward(x)
        ctx.cpu_w, ctx.dev = cpu_w, dev
        return y

    @staticmethod
    def backward(ctx, gy):
        (x,) = ctx.saved_tensors
        gx = gy @ ctx.cpu_w.to(ctx.dev) if ctx.needs_input_grad[0] else None
        return gx, None, None


class StreamedHealModel:
    """The .bin-backed model: meta-device NanoModel whose frozen params point
    at mmap views, streamed to `dev` layer by layer during forward/backward."""

    def __init__(self, bin_path, dev, lora_cfg=None):
        self.path = bin_path
        self.dev = dev
        config, entries, f = read_bin(bin_path)
        f.close()
        cfg = {k: config[k] for k in CFG_KEYS if k in config}
        self.cfg = cfg
        index = {n: (dt, d, o, s) for n, dt, d, o, s in entries}
        mm = np.memmap(bin_path, dtype=np.uint8, mode="r")
        # every step re-reads the whole file through the page cache: ask the
        # kernel to stream it in (the fault-driven 4k default readahead is
        # hopeless on a network-attached disk)
        if hasattr(os, "posix_fadvise"):
            try:
                with open(bin_path, "rb") as fw:
                    os.posix_fadvise(fw.fileno(), 0, 0, os.POSIX_FADV_WILLNEED)
            except OSError:
                pass

        with torch.device("meta"):
            model = NanoModel(cfg)  # zero allocation: every param is meta
        model.eval()  # required by the MoE gate assert; no dropout/BN in the arch

        # swap every MoE block for the packed-experts version (built on the
        # meta device too: its spine submodules are assigned from the .bin
        # below, and its expert Linears must never be allocated)
        n_moe = 0
        with torch.device("meta"):
            for l, layer in enumerate(model.layers):
                if hasattr(layer, "block_sparse_moe"):
                    blobs = {}
                    for kind in ("w1", "w2", "w3"):
                        entries_l = [
                            index[f"layers.{l}.block_sparse_moe.experts.{e}.{kind}"]
                            for e in range(cfg["n_experts"])
                        ]
                        for dt, _, _, _ in entries_l:
                            if dt != DTYPE_MXFP4:
                                raise SystemExit(f"layers.{l} experts {kind}: dtype {dt} "
                                                 "(only MXFP4 experts are supported)")
                        blobs[kind] = [(o, d, s) for _, d, o, s in entries_l]
                    layer.block_sparse_moe = StreamedSparseMoe(model.config, ExpertStore(mm, blobs))
                    n_moe += 1

        # LoRA before the spine assignment: the wrapped base weights resolve to
        # their original .bin name via the ".base.weight" -> ".weight" rewrite
        if lora_cfg:
            want_norms = lora_cfg.get("norms", False) or bool(lora_cfg.get("final_norm"))
            apply_lora(
                model, lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["targets"],
                want_norms,
            )
            # optional layer scope: unwrap the adapters outside the targeted
            # layers (the base Linear takes its place and gets the mmap weight)
            if lora_cfg.get("layers") is not None:
                keep = set(lora_cfg["layers"])
                wrapped = []
                for w in model.lora_info["wrapped"]:
                    m_l = re.match(r"layers\.(\d+)\.", w)
                    if m_l and int(m_l.group(1)) in keep:
                        wrapped.append(w)
                    else:
                        parent = model.get_submodule(w.rsplit(".", 1)[0])
                        leaf = w.rsplit(".", 1)[1]
                        setattr(parent, leaf, getattr(parent, leaf).base)
                assert wrapped, f"lora layers {sorted(keep)}: no adapter left"
                model.lora_info["wrapped"] = wrapped
                model.lora_info["layers"] = sorted(keep)
            # optional norm scope: keep only the trunk norms trainable
            if lora_cfg.get("final_norm"):
                for name, p in model.named_parameters():
                    if p.requires_grad and "norm" in name and not name.startswith(
                        ("norm.", "output_attn_res_norm.")
                    ):
                        p.requires_grad = False
                model.lora_info["final_norm"] = True
            self.n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            # no adapters (selftest): the base is frozen all the same, so the
            # assignment below takes the mmap branch for every parameter
            for p in model.parameters():
                p.requires_grad = False
            self.n_train = 0

        # replace every meta param by its .bin tensor (frozen: mmap view flipped
        # per layer; trainable: real copy resident on the compute device). The
        # Parameter object itself is swapped (meta .data assignment is refused).
        missing = []
        n_assigned = 0
        for name, p in list(model.named_parameters()):
            if not p.is_meta:
                continue  # fresh LoRA adapter params (real, trainable)
            bin_name = name.replace(".base.weight", ".weight")
            if bin_name not in index:
                missing.append(name)
                continue
            dt, dims, off, size = index[bin_name]
            if dt == DTYPE_F32:
                arr = np.memmap(bin_path, dtype=np.float32, mode="r", offset=off, shape=tuple(dims))
            elif dt == DTYPE_I32:
                arr = np.memmap(bin_path, dtype=np.int32, mode="r", offset=off, shape=tuple(dims))
            elif dt in (DTYPE_MXFP4, DTYPE_VQ1):
                continue  # packed experts: owned by the ExpertStore, no torch param
            else:
                raise SystemExit(f"{bin_name}: unknown dtype {dt}")
            t = torch.from_numpy(arr)
            if bin_name.endswith(("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight")) and len(dims) == 2:
                t = t.view(dims[0], 1, dims[1])
            if tuple(t.shape) != tuple(p.shape):
                # slicer-produced .bin may flatten a [1, D] proj weight to [D]
                assert t.numel() == p.numel(), f"{bin_name}: {tuple(t.shape)} != {tuple(p.shape)}"
                t = t.view(p.shape)
            if p.requires_grad:
                new_p = torch.nn.Parameter(t.clone().to(dev), requires_grad=True)
            else:
                new_p = torch.nn.Parameter(t, requires_grad=False)
                new_p._cpu_data = t
            mod = model.get_submodule(name.rpartition(".")[0])
            mod._parameters[name.rpartition(".")[2]] = new_p
            n_assigned += 1
        if missing:
            raise SystemExit(f"params with no .bin tensor: {missing[:6]} ({len(missing)} total)")
        self.n_assigned = n_assigned

        # trainable params (LoRA adapters) are CPU-real after apply_lora: move
        # them onto the compute device once - they stay resident
        for p in model.parameters():
            if p.requires_grad and p.device != torch.device(dev):
                p.data = p.data.to(dev)

        model.eval()  # gate assert + swapped MoE blocks default to training=True
        # trunk norms / projections are used outside the per-layer swap: the
        # frozen ones become device-resident (a few KB), embed stays on the CPU
        # (gather only) and lm_head is streamed by _StreamedLinear
        for trunk in (model.norm, model.output_attn_res_norm, model.output_attn_res_proj):
            for p in trunk.parameters():
                if not p.requires_grad and hasattr(p, "_cpu_data"):
                    p.data = p._cpu_data.to(dev)
                    del p._cpu_data

        self.model = model
        self.n_moe = n_moe
        self._mla, _ = layer_types(cfg)

    def trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_state(self):
        # named_parameters only: state_dict() would also walk the frozen spine
        return {k: v.detach().cpu() for k, v in self.model.named_parameters() if v.requires_grad}

    def load_trainable(self, sd):
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:4]}"
        left = [k for k in missing if k.endswith((".lora_A", ".lora_B"))]
        assert not left, f"missing LoRA keys: {left[:4]}"

    def forward(self, ids):
        """ids [B, T] on the compute device -> logits [B, T, V]. Same flow as
        NanoModel.forward, with per-layer weight streaming."""
        m, dev = self.model, self.dev
        B, T = ids.shape
        D = self.cfg["hidden"]
        with torch.no_grad():
            h = m.embed_tokens.weight._cpu_data[ids.cpu()].to(dev)  # frozen: gather only
        h = h.requires_grad_(True)  # graph root (its .grad is discarded)
        causal = torch.zeros(1, 1, T, T, device=dev, dtype=h.dtype)
        causal.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool, device=dev), 1),
            float("-inf"),
        )
        blocks = h.new_zeros(B * T, 0, D)
        for l, layer in enumerate(m.layers):
            mask = causal if l in self._mla else None
            h, blocks = _StreamedLayer.apply(h, blocks, layer, mask, dev)
        h = _apply_attn_res(
            h.view(-1, D), blocks, m.output_attn_res_proj, m.output_attn_res_norm
        ).view(B, T, D)
        h = m.norm(h)
        return _StreamedLinear.apply(h, m.lm_head.weight._cpu_data, dev)


def save_ckpt(sm, opt, step, rng, args, lora_info, path):
    payload = {
        "model": sm.trainable_state(),  # adapters (+trainable norms) only
        "opt": opt.state_dict(),
        "step": step,
        "rng_np": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "cfg": sm.cfg,
        "args": vars(args),
        "lora": lora_info,
        "bin_source": sm.path,
        "streamed": True,
    }
    tmp = path + f".tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem
    torch.save(payload, path + ".latest.tmp")
    os.replace(path + ".latest.tmp", os.path.join(os.path.dirname(path), "ckpt_latest.pt"))


def selftest(args):
    """Streamed forward vs the bin2pt reference model on the same tokens:
    the logits must match to float noise (same math, same dequant rule)."""
    from bin2pt import convert as bin2pt_convert
    dev = args.device
    sd, cfg = bin2pt_convert(args.model)
    ref = NanoModel(cfg).float().eval()
    ref.load_state_dict(sd)
    del sd
    # adapters included (identity at init: B is zero, so parity stays exact)
    lora_cfg = {"rank": args.lora, "alpha": args.lora_alpha or args.lora,
                "targets": ["q", "k", "v", "o"], "norms": False}
    sm = StreamedHealModel(args.model, dev, lora_cfg)
    tokens = load_tokens(args.data)
    ids = torch.from_numpy(tokens[:64].astype(np.int64)).unsqueeze(0)
    with torch.no_grad():
        out_ref = ref(ids)
        out_new = sm.forward(ids.to(dev)).cpu()
    diff = (out_ref - out_new).abs()
    rel = diff.max() / out_ref.abs().max()
    fp = f"mean {out_ref.mean():.6e} std {out_ref.std():.6e} head16 {out_ref[0, 0, :16].sum():.6e}"
    print(f"selftest: logits {tuple(out_new.shape)}, max|diff| {diff.max():.3e}, "
          f"rel {rel:.3e} | ref fingerprint: {fp}", flush=True)
    assert rel < 1e-3, f"streamed forward diverges from the reference: rel {rel}"
    print("selftest OK", flush=True)


def parse_layers(spec):
    """\"19-21\" / \"19,20,21\" -> sorted list of layer indices."""
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="microkimi .bin (MKIM0002), read via mmap")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--lora-targets", default="attn")
    ap.add_argument("--lora-norms", action="store_true")
    ap.add_argument("--lora-layers", default=None,
                    help="restrict the adapters to these layers, e.g. \"19-21\" or "
                         "\"19,20,21\" (default: every layer). For a wound localized by "
                         "the logit lens - much faster to read a signal from")
    ap.add_argument("--lora-final-norm", action="store_true",
                    help="train ONLY the trunk norms (final norm + output attn_res norm), "
                         "not the per-layer norms")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--ckpt-secs", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--ignore-unk", action="store_true",
                    help="exclude UNK targets from the loss (see train.py; the fineweb "
                         "corpus spamms [UNK] otherwise)")
    ap.add_argument("--unk-id", type=int, default=8198,
                    help="UNK token id for --ignore-unk (8200-vocab nano default; "
                         "check the tokenizer of the target model)")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--bench", type=int, default=None, help="run only N steps and stop")
    ap.add_argument("--selftest", action="store_true",
                    help="logits parity vs the bin2pt reference model, then exit")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda but torch.cuda.is_available() is False")
    print(f"device: {dev}", flush=True)

    if args.selftest:
        selftest(args)
        return

    # LoRA config: from the checkpoint being resumed when it has one, else flags
    ckpt_latest = os.path.join(args.out, "ckpt_latest.pt")
    pre_ck = None
    if args.resume and os.path.exists(ckpt_latest):
        pre_ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
    lora_cfg = (pre_ck or {}).get("lora")
    if lora_cfg is None:
        lora_cfg = {
            "rank": args.lora,
            "alpha": args.lora_alpha if args.lora_alpha is not None else args.lora,
            "targets": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
            "norms": args.lora_norms,
            "layers": parse_layers(args.lora_layers) if args.lora_layers else None,
            "final_norm": args.lora_final_norm,
        }

    t_load = time.time()
    sm = StreamedHealModel(args.model, dev, lora_cfg)
    m = sm.model
    scope = f", layers {lora_cfg['layers']}" if lora_cfg.get("layers") is not None else ""
    if lora_cfg.get("final_norm"):
        scope += ", final-norm only"
    print(f"heal_stream: {args.model} - {sm.cfg['n_layers']} layers, hidden {sm.cfg['hidden']}, "
          f"vocab {sm.cfg['vocab']}, {sm.n_moe} moe layers, {sm.n_assigned} mmap-backed tensors, "
          f"{sm.n_train / 1e6:.2f} M trainable params{scope} ({time.time() - t_load:.1f} s)", flush=True)

    trainable = sm.trainable_params()
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens = load_tokens(args.data)
    print(f"corpus: {len(tokens) / 1e6:.1f} M tokens", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    step0 = 0
    if pre_ck is not None:
        print(f"resuming from {ckpt_latest} ...", flush=True)
        sm.load_trainable(pre_ck["model"])
        opt.load_state_dict(pre_ck["opt"])
        if dev != "cpu":
            for st in opt.state.values():
                for key, val in st.items():
                    if torch.is_tensor(val):
                        st[key] = val.to(dev)
        step0 = pre_ck["step"]
        rng.bit_generator.state = pre_ck["rng_np"]
        torch.set_rng_state(pre_ck["rng_torch"])
        print(f"  -> step {step0}", flush=True)

    if args.bench is not None:
        args.steps = step0 + args.bench
        print(f"bench mode: {args.bench} steps", flush=True)

    t0 = time.time()
    t_ckpt = t0
    loss_ema = None
    toks = 0
    stop_reason = "steps reached"
    step = step0
    for step in range(step0, args.steps):
        if args.max_hours is not None and (time.time() - t0) > args.max_hours * 3600:
            stop_reason = f"time-cap {args.max_hours} h reached"
            print(f"[{stop_reason} - clean stop at step {step}]", flush=True)
            break
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            x, y = make_batch(tokens, rng, args.batch, args.seq)
            x = x.to(dev)
            y = y.to(dev)
            logits = sm.forward(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
                ignore_index=args.unk_id if args.ignore_unk else -100,
            ) / args.accum
            loss.backward()
            step_loss += loss.item()
            del x, y, logits, loss
        gn = torch.nn.utils.clip_grad_norm_(trainable, args.clip)
        opt.step()
        toks += args.accum * args.batch * args.seq
        loss_ema = step_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * step_loss

        if (step + 1) % args.log_every == 0 or step == step0:
            dt = time.time() - t0
            eta = dt / max(1, step + 1 - step0) * (args.steps - step - 1)
            mem = f"cuda {torch.cuda.memory_allocated() / 1e9:.2f} GB" if dev == "cuda" else ""
            print(
                f"step {step + 1:6d}/{args.steps} | loss {step_loss:.4f} "
                f"(ema {loss_ema:.4f}) | lr {lr:.2e} | gn {gn:.2e} | {toks / dt:.0f} tok/s | "
                f"rss {rss_gb():.1f} GB {mem} | {dt / 60:.1f} min, eta {eta / 60:.1f} min",
                flush=True,
            )
        now = time.time()
        if (step + 1) % args.ckpt_every == 0 or (now - t_ckpt) > args.ckpt_secs or (step + 1) == args.steps:
            save_ckpt(sm, opt, step + 1, rng, args, m.lora_info, os.path.join(args.out, f"ckpt_{step + 1:07d}.pt"))
            t_ckpt = now
            print(f"  [ckpt step {step + 1} written]", flush=True)

    if stop_reason != "steps reached":
        save_ckpt(sm, opt, step, rng, args, m.lora_info, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
        print(f"  [final ckpt step {step} written]", flush=True)
    print(f"done ({stop_reason}): {toks / 1e6:.2f} M tokens in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
