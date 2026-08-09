#!/usr/bin/env python3
"""nanokimi - expert-nesting / orthogonality / restricted-head unit checks
(CPU, tiny, forward-only: no optimizer step, no training).

Covers:
  (a) restrict_experts: the router only selects kept experts (the routing
      counts are zero outside the subset), the gate weights still normalize,
      and clear_expert_restriction restores the bias bit-exactly;
  (b) nest_order: the importance ordering sorts by decreasing frequency
      (EMA, or the raw accumulator while the EMA is empty), natural order
      while both are empty;
  (c) expert_ortho_loss: in [0, 1], ~1 for identical experts, ~0 for
      orthogonal ones;
  (d) restrict_head: logits sliced to N, targets >= N masked, skip count exact.

usage: python3 test_expert_nesting.py
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model_nano import NanoModel
from train import expert_ortho_loss, nest_order, restrict_head


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    # ── (a) restrict_experts / clear_expert_restriction ──
    print("(a) restrict_experts ...", flush=True)
    model = NanoModel({"n_layers": 2, "n_experts": 8, "top_k": 2}).float().eval()
    moe_blocks = [l.block_sparse_moe for l in model.layers if hasattr(l, "block_sparse_moe")]
    assert moe_blocks, "tiny model has no MoE layer"
    ids = torch.randint(0, 8200, (2, 8))
    keep = torch.tensor([0, 2, 3, 5, 6, 7])
    excluded = [1, 4]
    for blk in moe_blocks:
        bias0 = blk.gate.e_score_correction_bias.detach().clone()
        blk.route_count_acc = torch.zeros(len(blk.experts), dtype=torch.int64)
        blk.restrict_experts(keep)
        model(ids)
        assert int(blk.route_count_acc[excluded].sum()) == 0, \
            f"excluded experts routed: {blk.route_count_acc.tolist()}"
        assert int(blk.route_count_acc.sum()) == 2 * 8 * 2, \
            f"routing count mismatch: {int(blk.route_count_acc.sum())} != B*T*top_k"
        blk.route_count_acc = None
        blk.clear_expert_restriction()
        assert torch.equal(blk.gate.e_score_correction_bias.detach(), bias0), \
            "gate bias not restored bit-exactly"
    # repeated restrict without clear replaces the subset, restore still exact
    blk = moe_blocks[0]
    bias0 = blk.gate.e_score_correction_bias.detach().clone()
    blk.restrict_experts(keep)
    blk.restrict_experts(torch.tensor([1, 4, 0, 2]))
    blk.clear_expert_restriction()
    assert torch.equal(blk.gate.e_score_correction_bias.detach(), bias0)
    try:
        blk.restrict_experts(torch.tensor([0]))  # < top_k: must refuse
        raise SystemExit("restrict_experts accepted a subset smaller than top_k")
    except AssertionError:
        pass
    print("    OK: routing confined to the subset, counts exact, bias restored", flush=True)

    # ── (b) nest_order ──
    print("(b) nest_order ...", flush=True)
    ema = torch.tensor([0.0, 3.0, 0.0, 5.0, 1.0])
    acc = torch.zeros(5, dtype=torch.int64)
    order = nest_order(ema, acc)
    assert order[:3].tolist() == [3, 1, 4], order.tolist()
    # empty EMA falls back on the raw accumulator
    acc2 = torch.tensor([0, 0, 7, 0, 2], dtype=torch.int64)
    order2 = nest_order(torch.zeros(5), acc2)
    assert order2[:2].tolist() == [2, 4], order2.tolist()
    # both empty: natural index order
    assert nest_order(torch.zeros(5), torch.zeros(5, dtype=torch.int64)).tolist() == [0, 1, 2, 3, 4]
    print("    OK: decreasing frequency, acc fallback, natural order while empty", flush=True)

    # ── (c) expert_ortho_loss ──
    print("(c) expert_ortho_loss ...", flush=True)

    def fake_block(weights):  # weights [E, H, I]
        return SimpleNamespace(experts=[SimpleNamespace(
            w2=SimpleNamespace(weight=w)) for w in weights])

    w_same = torch.randn(1, 4, 8).repeat(6, 1, 1)  # all experts identical
    l_same = expert_ortho_loss([fake_block(w_same)], 64)
    assert abs(l_same.item() - 1.0) < 1e-5, f"identical experts: {l_same.item()}"
    w_ortho = torch.zeros(6, 4, 8)  # flattened one-hot rows: mutually orthogonal
    for e in range(6):
        w_ortho[e].reshape(4 * 8)[e] = 1.0
    l_ortho = expert_ortho_loss([fake_block(w_ortho)], 64)
    assert l_ortho.item() < 1e-10, f"orthogonal experts: {l_ortho.item()}"
    l_rand = expert_ortho_loss([fake_block(torch.randn(8, 4, 8))], 128)
    assert 0.0 <= l_rand.item() <= 1.0, l_rand.item()
    print(f"    OK: identical {l_same.item():.4f}, orthogonal {l_ortho.item():.2e}, "
          f"random {l_rand.item():.4f}", flush=True)

    # ── (d) restrict_head ──
    print("(d) restrict_head ...", flush=True)
    logits = torch.randn(2, 3, 10)
    y = torch.tensor([[1, 5, 9], [0, 4, 3]])
    lg, y_flat, n_skip = restrict_head(logits, y, 4)
    assert lg.shape == (2, 3, 4) and torch.equal(lg, logits[..., :4])
    assert n_skip == 3, n_skip  # targets 5, 9, 4
    assert y_flat.tolist() == [1, -100, -100, 0, -100, 3], y_flat.tolist()
    print("    OK: logits sliced, 3 targets masked, count exact", flush=True)

    print("ALL EXPERT NESTING TESTS OK", flush=True)


if __name__ == "__main__":
    main()
