#!/usr/bin/env python3
"""nanokimi - PyTorch training model (model_nano.py)

Drives the REAL Moonshot classes (vendor/moonshot/modeling_kimi_linear.py -
Kimi K3, Moonshot AI / DeepSeek-AI / HuggingFace) with the nano config:
  12 layers (MLA if L%4==3 → 3,7,11 ; KDA otherwise), MoE layers 1..11, dense MLP
  layer 0, AttnRes block_size=4, hidden 512, vocab 8200 (nano remap).
The driving logic (embed → layers → output attn_res → norm → lm_head) mirrors
exactly KimiLinearModel.forward, in batched B×T mode for training.

The Moonshot files (separate license) are NOT vendored: they are
downloaded on first use from huggingface.co/moonshotai/Kimi-K3
(see vendor/README.md).

Usable as an import (train.py) or standalone for a verification forward pass.
"""
import os
import sys
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "vendor"))       # shim fla
sys.path.insert(0, os.path.join(_HERE, "vendor", ".."))  # package moonshot


def _ensure_moonshot_vendor():
    """Download the Moonshot files into vendor/moonshot/ if missing
    (downloaded at runtime, never vendored - separate Moonshot license)."""
    pkg = os.path.join(_HERE, "vendor", "moonshot")
    base = "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/"
    for name in ("modeling_kimi_linear.py", "configuration_kimi_k3.py"):
        dst = os.path.join(pkg, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        os.makedirs(pkg, exist_ok=True)
        print(f"[model_nano] download {base}{name} → {dst}")
        with urllib.request.urlopen(base + name, timeout=120) as r:
            data = r.read()
        tmp = dst + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
    init = os.path.join(pkg, "__init__.py")
    if not os.path.exists(init):
        open(init, "w").close()


_ensure_moonshot_vendor()

import torch
import torch.nn as nn

from vendor.moonshot.configuration_kimi_k3 import KimiLinearConfig
from vendor.moonshot.modeling_kimi_linear import (
    KimiDecoderLayer,
    KimiRMSNorm,
    KimiSparseMoeBlock,
    _apply_attn_res,
)


# MOE_BMM: True = vectorized grouped-GEMM (validated 1:1 against the loop), False = loop
MOE_BMM = os.environ.get("NANO_MOE_BMM", "1") == "1"
# CUDA fast path for the MoE: experts are processed in bounded chunks of
# NANO_MOE_CHUNK experts (memory-bounded, each chunk wrapped in gradient
# checkpointing) over cached stacked weights (one copy per step instead of one
# torch.stack per call). SAME mathematics as moe_train_bmm (validated 1:1);
# only enabled on the device types listed in NANO_MOE_FAST_DEVICES.
MOE_FAST_DEVICES = tuple(
    d for d in os.environ.get("NANO_MOE_FAST_DEVICES", "cuda").split(",") if d
)
MOE_CHUNK = int(os.environ.get("NANO_MOE_CHUNK", "128"))


class _CachedStack(torch.autograd.Function):
    """Identity on a pre-stacked weight cache with the gradient routing of
    torch.stack(params): backward hands each source parameter its slice of
    the incoming gradient. The cache is refreshed in place under no_grad
    inside forward, so the stacked copy never appears in the autograd graph
    (no per-call allocation, no StackBackward scatter)."""

    @staticmethod
    def forward(ctx, cache, *params):
        with torch.no_grad():
            torch.stack(params, out=cache)
        ctx.n = len(params)
        return cache

    @staticmethod
    def backward(ctx, grad_out):
        return (None,) + tuple(grad_out[i] for i in range(ctx.n))


class TrainableSparseMoe(KimiSparseMoeBlock):
    """Differentiable version of KimiSparseMoeBlock.

    The Moonshot class is inference-only: gate.forward asserts not training and
    moe_infer is @torch.no_grad(). Here we replicate EXACTLY the same
    mathematics (hard top-16, bias-free sigmoid weights, 1e-20 renorm, combine
    Σ wᵢ·expertᵢ(x)) but differentiably: the selection is discrete (no
    gradient, standard MoE), the weights wᵢ propagate the gradient to the router,
    and the group-by-expert (argsort + cat + index_put) is differentiable.
    To be used with the module in eval() mode (the gate's assert requires it) -
    the architecture has neither dropout nor batchnorm, so eval() changes nothing else.
    """

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden = hidden_states.view(-1, hidden_states.shape[-1])
        if self.use_latent_moe:
            hidden = self.routed_expert_down_proj(hidden)
        if MOE_BMM:
            if hidden.device.type in MOE_FAST_DEVICES:
                y = self.moe_train_bmm_fast(hidden, topk_idx, topk_weight)
            else:
                y = self.moe_train_bmm(hidden, topk_idx, topk_weight)
        else:
            y = self.moe_train(hidden, topk_idx, topk_weight)
        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_train(self, x, topk_ids, topk_weight):
        """same combine as moe_infer: sum_k w_k · expert_k(x), differentiable."""
        n, k = topk_ids.shape
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weight.reshape(-1, 1)
        rep = x.unsqueeze(1).expand(n, k, x.shape[-1]).reshape(n * k, -1)
        order = flat_ids.argsort()
        sorted_tokens = rep[order]
        counts = torch.bincount(flat_ids, minlength=len(self.experts)).cpu().numpy()
        outs = []
        start = 0
        for e, cnt in enumerate(counts):
            if cnt == 0:
                continue
            outs.append(self.experts[e](sorted_tokens[start: start + cnt]))
            start += cnt
        outs = torch.cat(outs, dim=0)
        unsorted = torch.empty_like(outs)
        unsorted[order] = outs
        return (unsorted * flat_w).view(n, k, -1).sum(1)

    def moe_train_bmm(self, x, topk_ids, topk_weight):
        """Vectorized grouped-GEMM (capacity-padded bmm) - SAME mathematics
        as moe_train: SiTU(cat(w1·h, w3·h)) then w2, combine Σ wᵢ·expertᵢ.
        Replaces the Python loop of 896 calls with 3 BLAS bmm per round.
        The capacity is CAPPED (~3× the average): if the router collapses
        onto a few experts during training, we process in multiple
        rounds (identical result) instead of allocating a giant dense tensor
        (that was the cause of the OOM-kills: cap = counts.max() exploded
        after warmup → dense [896, counts.max, 128] of tens of GB).
        """
        n, k = topk_ids.shape
        e_max = len(self.experts)
        h_dim = x.shape[-1]
        i_dim = self.experts[0].w1.weight.shape[0]
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weight.reshape(-1, 1)
        rep = x.unsqueeze(1).expand(n, k, h_dim).reshape(n * k, h_dim)
        order = flat_ids.argsort()
        sorted_tokens = rep[order]
        counts = torch.bincount(flat_ids, minlength=e_max)
        starts = counts.cumsum(0) - counts
        sorted_eids = flat_ids[order]
        slot = torch.arange(n * k, device=x.device) - starts[sorted_eids]
        # bounded capacity budget + rounds (1 round in the normal regime)
        cap_budget = max(64, (3 * n * k + e_max - 1) // e_max)
        w1 = torch.stack([e.w1.weight for e in self.experts])  # [E, I, H]
        w3 = torch.stack([e.w3.weight for e in self.experts])
        w2 = torch.stack([e.w2.weight for e in self.experts])  # [E, H, I]
        out_sorted = torch.empty_like(sorted_tokens)
        round_of = slot // cap_budget
        n_rounds = int(round_of.max().item()) + 1
        for r in range(n_rounds):
            mask = round_of == r
            eids_r = sorted_eids[mask]
            slot_r = slot[mask] - r * cap_budget
            dense = x.new_zeros(e_max, cap_budget, h_dim)
            dense[eids_r, slot_r] = sorted_tokens[mask]
            g = torch.bmm(dense, w1.transpose(1, 2))  # [E, cap, I]
            u = torch.bmm(dense, w3.transpose(1, 2))
            # SiTU identical to SituAndMul(4, 25): a=4·tanh(g/4)·sigmoid(g) ; u=25·tanh(u/25)
            act = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))
            y = torch.bmm(act, w2.transpose(1, 2))  # [E, cap, H]
            out_sorted[mask] = y[eids_r, slot_r]
        unsorted = torch.empty_like(out_sorted)
        unsorted[order] = out_sorted
        return (unsorted * flat_w).view(n, k, -1).sum(1)

    def _cached_expert_stacks(self):
        """Stacked expert weights [E, I, H], [E, I, H], [E, H, I], cached on the
        module and refreshed in place on every call (see _CachedStack)."""
        e0 = self.experts[0]
        ref = e0.w1.weight
        cache = getattr(self, "_wstack_cache", None)
        if cache is None or cache[0].device != ref.device or cache[0].dtype != ref.dtype:
            cache = (
                torch.empty(len(self.experts), *ref.shape, device=ref.device, dtype=ref.dtype),
                torch.empty(len(self.experts), *e0.w3.weight.shape, device=ref.device, dtype=ref.dtype),
                torch.empty(len(self.experts), *e0.w2.weight.shape, device=ref.device, dtype=ref.dtype),
            )
            self._wstack_cache = cache
        w1c, w3c, w2c = cache
        w1 = _CachedStack.apply(w1c, *[e.w1.weight for e in self.experts])
        w3 = _CachedStack.apply(w3c, *[e.w3.weight for e in self.experts])
        w2 = _CachedStack.apply(w2c, *[e.w2.weight for e in self.experts])
        return w1, w3, w2

    def _expert_chunk(self, toks, eids_l, slot_l, w1g, w3g, w2g, cap, gsz, cap_max):
        """SiTU MLP for one chunk of gsz experts over the tokens routed to them
        (toks [m, h], eids_l/slot_l local expert id / per-expert slot).
        Identical math to the per-round body of moe_train_bmm; the capacity is
        bounded by cap and overflow is handled in rounds (identical result)."""
        h_dim = toks.shape[-1]
        out = torch.empty_like(toks)
        n_rounds = (cap_max + cap - 1) // cap
        for r in range(n_rounds):
            mask = (slot_l >= r * cap) & (slot_l < (r + 1) * cap)
            eids_r = eids_l[mask]
            slot_r = slot_l[mask] - r * cap
            dense = toks.new_zeros(gsz, cap, h_dim)
            dense[eids_r, slot_r] = toks[mask]
            g = torch.bmm(dense, w1g.transpose(1, 2))  # [gsz, cap, I]
            u = torch.bmm(dense, w3g.transpose(1, 2))
            # SiTU identical to SituAndMul(4, 25): a=4·tanh(g/4)·sigmoid(g) ; u=25·tanh(u/25)
            act = (4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))
            y = torch.bmm(act, w2g.transpose(1, 2))  # [gsz, cap, H]
            out[mask] = y[eids_r, slot_r]
        return out

    def moe_train_bmm_fast(self, x, topk_ids, topk_weight):
        """CUDA fast path: SAME mathematics as moe_train_bmm (validated 1:1),
        but memory-bounded: experts are processed in chunks of MOE_CHUNK
        instead of all 896 at once, each chunk wrapped in gradient
        checkpointing (the dense [G, cap, h] activations and their SiTU
        intermediates are recomputed chunk-by-chunk during backward instead
        of being retained for all 896 experts), and the expert weights are
        stacked once per step into a cache instead of 3 torch.stack per call.
        This removes the memory cliff (dense [896, cap_budget, 512] + SiTU
        saves) that OOMed batch 32 on a 23 GB card."""
        n, k = topk_ids.shape
        e_max = len(self.experts)
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
        # bounded capacity budget, same formula as moe_train_bmm
        cap_budget = max(64, (3 * n * k + e_max - 1) // e_max)
        w1, w3, w2 = self._cached_expert_stacks()
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
                self._expert_chunk,
                sorted_tokens[a: a + m],
                sorted_eids[a: a + m] - e0,
                slot[a: a + m],
                w1[e0:e1], w3[e0:e1], w2[e0:e1],
                cap, e1 - e0, cap_max,
                use_reentrant=False,
            )
        unsorted = torch.empty_like(out_sorted)
        unsorted[order] = out_sorted
        return (unsorted * flat_w).view(n, k, -1).sum(1)

# ── nano config (see mission SPEC) ──
NANO = dict(
    n_layers=12, hidden=512, vocab=8200,
    n_experts=896, top_k=16, n_shared=2,
    kda_heads=4, kda_dim=128, kda_conv=4, kda_fa_rank=128, gate_lower_bound=-5.0,
    mla_heads=4, mla_q_lora=128, mla_kv_lora=64, mla_nope=128, mla_rope=64, mla_v=128,
    routed_hidden=128, moe_inter=64, shared_inter=128,
    dense_inter=2048, attn_res_block=4, first_k_dense=1, rms_eps=1e-5,
)


def is_mla(l, n_layers):
    return l % 4 == 3 or l == n_layers - 1


def layer_types(c):
    """(mla, dense) as 0-based layer-index sets.

    Explicit "mla_layers" / "dense_layers" lists (carried by sliced .bin
    configs, e.g. a pruned K3 where the pattern no longer holds) win;
    otherwise the legacy nano pattern (MLA if L%4==3 or last, dense prefix
    of first_k_dense layers) applies - old checkpoints are unaffected."""
    n = c["n_layers"]
    mla = set(c["mla_layers"]) if "mla_layers" in c else {l for l in range(n) if is_mla(l, n)}
    dense = set(c["dense_layers"]) if "dense_layers" in c else set(range(c["first_k_dense"]))
    assert not (mla & dense), f"layer type overlap: {sorted(mla & dense)}"
    return mla, dense


def nano_config(c):
    """KimiLinearConfig for the Moonshot classes, built from the NANO dict (or override)."""
    mla, dense = layer_types(c)
    full_attn = sorted(l + 1 for l in mla)  # 1-based
    kda_layers = sorted(l + 1 for l in range(c["n_layers"]) if l not in mla)
    assert dense == set(range(c["first_k_dense"])), (
        f"dense layers {sorted(dense)} are not the prefix first_k_dense={c['first_k_dense']} "
        "the Moonshot config can express")
    return KimiLinearConfig(
        vocab_size=c["vocab"], hidden_size=c["hidden"],
        intermediate_size=c["dense_inter"],
        num_hidden_layers=c["n_layers"],
        num_attention_heads=c["mla_heads"], num_key_value_heads=c["mla_heads"],
        hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
        rms_norm_eps=c["rms_eps"], tie_word_embeddings=False,
        max_position_embeddings=1048576,
        q_lora_rank=c["mla_q_lora"], kv_lora_rank=c["mla_kv_lora"],
        qk_nope_head_dim=c["mla_nope"], qk_rope_head_dim=c["mla_rope"], v_head_dim=c["mla_v"],
        mla_use_nope=True, mla_use_output_gate=True,
        num_experts=c["n_experts"], num_experts_per_token=c["top_k"],
        num_shared_experts=c["n_shared"],
        routed_expert_hidden_size=c["routed_hidden"], moe_intermediate_size=c["moe_inter"],
        latent_moe_use_norm=True,
        moe_renormalize=True, moe_router_activation_func="sigmoid",
        routed_scaling_factor=1.0, first_k_dense_replace=c["first_k_dense"], moe_layer_freq=1,
        use_grouped_topk=True, num_expert_group=1, topk_group=1,
        linear_attn_config={
            "full_attn_layers": full_attn, "kda_layers": kda_layers,
            "head_dim": c["kda_dim"], "num_heads": c["kda_heads"],
            "short_conv_kernel_size": c["kda_conv"],
            "gate_lower_bound": c["gate_lower_bound"], "use_full_rank_gate": True,
        },
        attn_res_block_size=c["attn_res_block"],
        _attn_implementation="eager",
    )


class NanoModel(nn.Module):
    """Replicates KimiLinearModel.forward with the real Moonshot classes."""

    def __init__(self, cfg=None, grad_ckpt=False):
        super().__init__()
        c = dict(NANO) if cfg is None else {**NANO, **cfg}
        self.c = c
        self.grad_ckpt = grad_ckpt
        self.config = nano_config(c)
        self._mla, _ = layer_types(c)  # explicit list or legacy pattern
        D, V = c["hidden"], c["vocab"]
        self.embed_tokens = nn.Embedding(V, D)
        self.layers = nn.ModuleList(
            [KimiDecoderLayer(self.config, l) for l in range(c["n_layers"])]
        )
        # replace the inference-only MoE blocks with the differentiable version
        for layer in self.layers:
            if hasattr(layer, "block_sparse_moe"):
                layer.block_sparse_moe = TrainableSparseMoe(self.config)
        self.norm = KimiRMSNorm(D, eps=c["rms_eps"])
        self.output_attn_res_norm = KimiRMSNorm(D, eps=c["rms_eps"])
        self.output_attn_res_proj = nn.Linear(D, 1, bias=False)
        self.lm_head = nn.Linear(D, V, bias=False)
        self.apply(self._init_weights)
        # A_log: the classes create it per head [H]; the microkimi format is
        # per channel [128] (real K3 checkpoint). We align on [128].
        for layer in self.layers:
            if layer.is_linear_attn:
                layer.self_attn.A_log = nn.Parameter(
                    torch.log(torch.empty(c["kda_dim"]).uniform_(1.0, 16.0))
                )
                layer.self_attn.dt_bias.data.zero_()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, ids):
        """ids [B, T] → logits [B, T, V]. Same flow as KimiLinearModel.forward."""
        c = self.c
        B, T = ids.shape
        D = c["hidden"]
        hidden = self.embed_tokens(ids)  # no scale
        causal = torch.zeros(1, 1, T, T, device=ids.device, dtype=hidden.dtype)
        causal.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool, device=ids.device), 1),
            float("-inf"),
        )
        blocks = hidden.new_zeros(B * T, 0, D)
        seam = getattr(self, "seam_adapter", None)
        seam_after = getattr(self, "seam_after", -1)
        for l, layer in enumerate(self.layers):
            mask = causal if l in self._mla else None
            if self.grad_ckpt and torch.is_grad_enabled():
                # gradient checkpointing: the graph keeps only the layer
                # boundaries; each layer's forward is recomputed during
                # backward (the shim's KDA recurrence keeps 2×[B,H,128,128] per
                # TIME STEP - without this, batch 32×512 ≈ 77 GB of graph).
                hidden, blocks = torch.utils.checkpoint.checkpoint(
                    lambda h, b, layer=layer, mask=mask: layer._forward_attn_residual(
                        h, attention_mask=mask, block_residual=b
                    ),
                    hidden,
                    blocks,
                    use_reentrant=False,
                )
            else:
                hidden, blocks = layer._forward_attn_residual(
                    hidden, attention_mask=mask, block_residual=blocks
                )
            if l == seam_after:
                hidden = seam(hidden)
        hidden = _apply_attn_res(
            hidden.view(-1, D), blocks, self.output_attn_res_proj, self.output_attn_res_norm
        ).view(B, T, D)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    experts = sum(
        p.numel()
        for name, p in model.named_parameters()
        if ".experts." in name
    )
    return total, experts


# ── LoRA (healing adapters) ──
#
# Freeze the base weights and train only small low-rank adapters on the
# attention projections: y = W x + (B A x) * alpha / rank, with A [r, in]
# kaiming-initialized and B [out, r] zero-initialized, so the adapter is the
# identity at step 0. Experts are never wrapped (they are the bulk of the
# params and not what LoRA healing targets); the MoE router gate is a custom
# module (not an nn.Linear) and stays untouched as well.

# NANO_PRETRANSPOSE: opt-in cache of a contiguous transposed copy W_t of the
# frozen LoRA-targeted base weights (the only Linears whose matmuls run in
# every micro-batch of both the forward and the backward recompute). The
# streamed trainer streams W_t alongside W (pinned host copy, see
# heal_stream.py) and LoRALinear then runs forward as x @ W_t and backward as
# gy @ W through _WTLinear, so neither direction feeds a transposed VIEW of a
# freshly streamed weight to the BLAS. Same math as F.linear. Default OFF.
PRETRANSPOSE = os.environ.get("NANO_PRETRANSPOSE", "0") == "1"


class _WTLinear(torch.autograd.Function):
    """y = x @ W^T + b for a frozen W, driven by the pre-transposed contiguous
    copy W_t in the forward and by W itself in the backward (grad w.r.t. x is
    gy @ W). W is frozen: no weight gradient is produced. Both device copies
    live only for the layer window (see heal_stream._Swap), and the graph
    nodes they feed are created and consumed inside the same window, so
    saving W for backward holds no VRAM beyond it."""

    @staticmethod
    def forward(ctx, x, w, w_t, bias):
        with torch.no_grad():
            y = torch.matmul(x, w_t)
            if bias is not None:
                y = y + bias
        ctx.save_for_backward(w)
        return y

    @staticmethod
    def backward(ctx, gy):
        (w,) = ctx.saved_tensors
        gx = torch.matmul(gy, w) if ctx.needs_input_grad[0] else None
        return gx, None, None, None


class LoRALinear(nn.Module):
    """Frozen nn.Linear + trainable low-rank adapter (A @ x then B @ ., scaled)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        w = self.base.weight
        # pre-transposed path (NANO_PRETRANSPOSE=1): _w_t is the streamed
        # contiguous W^T copy, present only inside the layer's swap window
        w_t = getattr(w, "_w_t", None)
        if (PRETRANSPOSE and w_t is not None and w_t.device == x.device
                and w_t.shape == (w.shape[1], w.shape[0])):
            y = _WTLinear.apply(x, w, w_t, self.base.bias)
        else:
            y = self.base(x)
        return y + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def merged_weight(self):
        """W + B A * scaling (for the merge-and-export path)."""
        return self.base.weight.data + (self.lora_B @ self.lora_A) * self.scaling


# target name -> projection suffixes it wraps (matched on the module's leaf name)
_LORA_TARGET_SUFFIXES = {
    "q": ("q_proj", "q_a_proj", "q_b_proj"),
    "k": ("k_proj", "kv_a_proj_with_mqa", "kv_b_proj"),
    "v": ("v_proj", "kv_a_proj_with_mqa", "kv_b_proj"),
    "o": ("o_proj",),
}
_LORA_TARGET_SUFFIXES["attn"] = tuple(
    dict.fromkeys(s for v in ("q", "k", "v", "o") for s in _LORA_TARGET_SUFFIXES[v])
)


def apply_lora(model, rank, alpha, targets, lora_norms=False):
    """Wrap the targeted attention Linears of `model` in LoRALinear and freeze
    everything else (except the norm gains when lora_norms). `targets` is a
    list from {q, k, v, o, attn}. Records model.lora_info for the checkpoint
    and returns (n_trainable, n_total)."""
    suffixes = tuple(dict.fromkeys(s for t in targets for s in _LORA_TARGET_SUFFIXES.get(t, ())))
    if not suffixes:
        raise SystemExit(f"apply_lora: no valid targets in {targets}")
    for p in model.parameters():
        p.requires_grad = False
    wrapped = []
    for mod_name, module in model.named_modules():
        leaf = mod_name.rsplit(".", 1)[-1]
        if isinstance(module, nn.Linear) and leaf in suffixes:
            parent = model.get_submodule(mod_name.rsplit(".", 1)[0])
            setattr(parent, leaf, LoRALinear(module, rank, alpha))
            wrapped.append(mod_name)
    assert wrapped, "apply_lora: no Linear matched - check the targets"
    if lora_norms:
        for name, p in model.named_parameters():
            if "norm" in name:
                p.requires_grad = True
    model.lora_info = {
        "rank": rank,
        "alpha": alpha,
        "targets": list(targets),
        "norms": bool(lora_norms),
        "wrapped": wrapped,
    }
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


# ── seam adapter (healing across a layer boundary) ──
#
# A sliced model (e.g. K3 layers "0-11,83-92" renumbered 0-21) stitches two
# non-contiguous ranges: layer N+1 was trained to consume the residual
# distribution of its ORIGINAL predecessor (layer 82), not of layer N (11).
# The seam adapter is a low-rank correction applied to the residual stream
# right after layer N:
#     h' = h + B @ (A @ h)      A [rank, hidden], B [hidden, rank]
# B is zero-initialized, so the adapter is an exact identity at step 0 (h + 0,
# bit-identical: a matmul against the zero matrix is an exact zero). No alpha
# scaling: the rank already bounds the correction. Only A and B carry
# gradients; they are tiny (2 * rank * hidden floats) and stay resident on the
# compute device, so they are saved and restored with the LoRA tensors in the
# checkpoint.


class SeamAdapter(nn.Module):
    """Low-rank residual-stream correction h' = h + B A h, identity at init."""

    def __init__(self, hidden: int, rank: int):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.empty(rank, hidden))
        self.B = nn.Parameter(torch.zeros(hidden, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

    def forward(self, h):
        return h + (h @ self.A.T) @ self.B.T


def apply_seam(model, rank, after):
    """Attach a SeamAdapter on the residual stream right after layer `after`
    (0-based). Records model.seam_info for the checkpoint. The adapter params
    are created trainable; the base weights are left untouched (apply_lora
    already froze them when LoRA is combined, and the no-LoRA path freezes
    them as well). `after` must leave a layer N+1: the merge folds the adapter
    into the input projections of that layer (exactly at zero-init, only
    approximately once trained - refused by default, see apply_lora_bin.py)."""
    n = model.c["n_layers"]
    if not 0 <= after < n - 1:
        raise SystemExit(f"apply_seam: --seam-after {after} out of range [0, {n - 2}] "
                         f"for a {n}-layer model")
    model.seam_adapter = SeamAdapter(model.c["hidden"], rank)
    model.seam_after = after
    model.seam_info = {"rank": rank, "after": after}
    return model.seam_adapter


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    m = NanoModel().float().eval()
    total, experts = count_params(m)
    print(f"params : {total / 1e6:.1f} M (of which experts {experts / 1e6:.1f} M)")
    ids = torch.randint(0, 8200, (2, 16))
    out = m(ids)
    print("logits :", tuple(out.shape), "mean", out.mean().item(), "std", out.std().item())
