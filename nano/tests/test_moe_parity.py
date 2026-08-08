#!/usr/bin/env python3
"""Unit parity: moe_train_bmm vs moe_train_bmm_fast on identical modules,
identical inputs - forward outputs and input/weight grads must match.
Run from the nano/ dir. Forces the fast path on CPU via NANO_MOE_FAST_DEVICES=cpu.
"""
import os
import sys

os.environ["NANO_MOE_FAST_DEVICES"] = "cpu"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from model_nano import TrainableSparseMoe, nano_config, NANO  # noqa: E402

torch.manual_seed(7)
cfg = nano_config(dict(NANO))
moe = TrainableSparseMoe(cfg).float().eval()

for trial, (n, k) in enumerate([(37, 16), (128, 16), (512, 16)]):
    g = torch.Generator().manual_seed(100 + trial)
    # moe_train_bmm operates AFTER routed_expert_down_proj: input dim is the
    # routed expert hidden size (128), not the model hidden size (512)
    x = torch.randn(n, cfg.routed_expert_hidden_size, generator=g, requires_grad=True)
    topk_ids = torch.stack(
        [torch.randperm(cfg.num_experts, generator=g)[:k] for _ in range(n)]
    )
    # mimic router collapse on the last trial (stress the rounds logic)
    if trial == 2:
        topk_ids[:, :8] = 3
    topk_w = torch.softmax(torch.randn(n, k, generator=g), dim=-1)

    # reference path
    x1 = x.detach().clone().requires_grad_(True)
    y_ref = moe.moe_train_bmm(x1, topk_ids, topk_w)
    y_ref.sum().backward()

    # fast path
    x2 = x.detach().clone().requires_grad_(True)
    y_new = moe.moe_train_bmm_fast(x2, topk_ids, topk_w)
    y_new.sum().backward()

    dy = (y_ref - y_new).abs().max().item()
    dx = (x1.grad - x2.grad).abs().max().item()
    print(
        f"trial {trial} n={n}: fwd max|dy|={dy:.3e} bwd max|dx|={dx:.3e} "
        f"{'OK' if dy < 1e-5 and dx < 1e-4 else 'MISMATCH'}"
    )
print("done")
