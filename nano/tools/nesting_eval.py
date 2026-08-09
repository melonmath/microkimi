#!/usr/bin/env python3
"""nesting measurement - held-out loss across depth cutoffs and expert-count caps
(nesting_eval.py)

Given a training checkpoint (.pt), evaluates the held-out per-token loss of
every (depth cutoff, expert-count cap) combination and prints one table:

  - depth cutoff d: only the first d decoder layers run and the logits are
    read off the prefix with the logit-lens path (final norm + lm_head), the
    exit that --stochastic-depth trains;
  - expert-count cap k: every MoE router is restricted to the top-k experts
    of its importance ordering (routing frequency measured on the eval data
    itself during a full-model pass) and the gate renormalizes over the kept
    experts, the mechanism that --nest-experts trains
    (TrainableSparseMoe.restrict_experts).

The full cell (all layers, all experts) is the plain model. If the checkpoint
was trained with --head-vocab, the same logit slicing and target skipping
apply here and the skipped fraction is reported.

usage:
  python3 tools/nesting_eval.py --ckpt run/ckpt/ckpt_latest.pt --data heldout.bin \
      [--seq 256] [--batch 8] [--max-windows 64] [--threads 4]
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # nano/ (model_nano.py, train.py)

from model_nano import NanoModel  # noqa: E402
from train import load_tokens, nest_order  # noqa: E402


def ce_loss(model, tokens, depth, seq, batch, n_win, head_vocab, ignore_unk):
    """Mean per-token CE over the first n_win sequential non-overlapping
    windows, at the given depth cutoff (full model when depth == n_layers).
    Returns (nats/token, n_scored, n_skipped_by_head)."""
    full = depth == model.c["n_layers"]
    total_nats, n_scored, n_skipped = 0.0, 0, 0
    for b0 in range(0, n_win, batch):
        idx = range(b0, min(b0 + batch, n_win))
        x = np.stack([tokens[i * seq:(i + 1) * seq] for i in idx]).astype(np.int64)
        y = np.stack([tokens[i * seq + 1:(i + 1) * seq + 1] for i in idx]).astype(np.int64)
        ids = torch.from_numpy(x)
        logits = model(ids) if full else model.forward_prefix(ids, depth)
        y_t = torch.from_numpy(y).reshape(-1)
        if head_vocab > 0:
            logits = logits[..., :head_vocab]
            skip = y_t >= head_vocab
            n_skipped += int(skip.sum())
            y_t = y_t.masked_fill(skip, -100)
        if ignore_unk:
            y_t = y_t.masked_fill(y_t == 8198, -100)
        per_tok = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y_t,
            reduction="none", ignore_index=-100)
        total_nats += per_tok.sum().item()
        n_scored += int((y_t != -100).sum())
    return total_nats / max(1, n_scored), n_scored, n_skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="held-out tokens.bin")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-windows", type=int, default=64,
                    help="cap on the number of eval windows (0 = all, default 64)")
    ap.add_argument("--depths", default="",
                    help="comma list of depth cutoffs (default: quartiles of n_layers)")
    ap.add_argument("--expert-caps", default="",
                    help="comma list of expert-count caps (default: top_k, 2x, 4x, all)")
    ap.add_argument("--ignore-unk", action="store_true",
                    help="exclude UNK (8198) targets from the loss (same as train.py)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = NanoModel(ck.get("cfg"), grad_ckpt=False).float().eval()
    model.load_state_dict(ck["model"])
    c = model.c
    n_layers, n_exp, top_k = c["n_layers"], c["n_experts"], c["top_k"]
    moe_blocks = [l.block_sparse_moe for l in model.layers if hasattr(l, "block_sparse_moe")]
    assert moe_blocks, "the model has no MoE layers"

    saved = ck.get("args") or {}
    head_vocab = int(saved.get("head_vocab", 0) or 0)
    if head_vocab:
        print(f"checkpoint trained with --head-vocab {head_vocab}: same logit slicing "
              f"and target skipping applied", flush=True)

    if args.depths:
        depths = sorted(set(int(d) for d in args.depths.split(",")))
    else:
        depths = sorted({max(1, n_layers // 4), max(1, n_layers // 2),
                         max(1, (3 * n_layers) // 4), n_layers})
    assert all(1 <= d <= n_layers for d in depths), f"depths out of range: {depths}"
    if args.expert_caps:
        caps = sorted(set(int(k) for k in args.expert_caps.split(",")))
    else:
        caps = sorted({k for k in (top_k, 2 * top_k, 4 * top_k, n_exp) if k <= n_exp})
    for k in caps:
        if k < top_k:
            raise SystemExit(f"expert cap {k} < top_k {top_k}: the router picks "
                             f"{top_k} experts per token, the pool must be at least that")
        if k > n_exp:
            raise SystemExit(f"expert cap {k} > n_experts {n_exp}")

    tokens = load_tokens(args.data)
    n_win = (len(tokens) - 1) // args.seq
    if args.max_windows > 0:
        n_win = min(n_win, args.max_windows)
    print(f"eval: {n_win} windows of {args.seq} tokens | depths {depths} | "
          f"expert caps {caps}", flush=True)

    # pass 1: full model, accumulate the routing frequency of every MoE layer
    # (the importance ordering of the caps comes from the eval data itself)
    for blk in moe_blocks:
        blk.route_count_acc = torch.zeros(len(blk.experts), dtype=torch.int64)
    base, n_scored, n_skipped = ce_loss(model, tokens, n_layers, args.seq, args.batch,
                                        n_win, head_vocab, args.ignore_unk)
    orders = [nest_order(torch.zeros(len(blk.experts)), blk.route_count_acc)
              for blk in moe_blocks]
    for blk in moe_blocks:
        blk.route_count_acc = None

    results = {}
    for k in caps:
        restricted = k < n_exp
        if restricted:
            for blk, order in zip(moe_blocks, orders):
                blk.restrict_experts(order[:k])
        for d in depths:
            if k == n_exp and d == n_layers:
                results[(d, k)] = base
            else:
                results[(d, k)] = ce_loss(model, tokens, d, args.seq, args.batch,
                                          n_win, head_vocab, args.ignore_unk)[0]
            print(f"  depth {d:3d} | experts {k:4d} -> {results[(d, k)]:.4f}   ",
                  end="\r", flush=True)
        if restricted:
            for blk in moe_blocks:
                blk.clear_expert_restriction()

    print("\nheld-out loss (nats/token) - rows: depth cutoff, cols: expert cap")
    header = "depth \\ experts" + "".join(f"{k:>9d}" for k in caps)
    print(header)
    print("-" * len(header))
    for d in depths:
        print(f"{d:>15d}" + "".join(f"{results[(d, k)]:>9.4f}" for k in caps))
    print(f"\nfull model (depth {n_layers}, {n_exp} experts): {base:.4f} nats/token "
          f"over {n_scored} scored targets")
    if head_vocab:
        n_tot = n_scored + n_skipped
        print(f"targets >= {head_vocab} skipped: {n_skipped}/{n_tot} "
              f"({100 * n_skipped / max(1, n_tot):.1f}%)")


if __name__ == "__main__":
    main()
