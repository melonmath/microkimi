#!/usr/bin/env python3
"""nanokimi - training loop (train.py)

AdamW + cosine (warmup), CE loss, random windows over tokens.bin (uint16).
ATOMIC checkpoints (tmp + rename) every N steps AND every T seconds
(preemptible box: --resume restarts from the last healthy state: model+opt+step+rng).

examples:
  # local smoke (reduced config):
  python3 train.py --data out_dev/tokens.bin --out out_dev/ckpt \
      --layers 4 --batch 4 --seq 128 --steps 200 --threads 10
  # real run (32 vCPU box):
  python3 train.py --data ~/nano_out/tokens.bin --out ~/nano_out/ckpt \
      --batch 32 --seq 512 --steps 8000 --threads 32 --resume
"""
import argparse
import ctypes
import gc
import json
import math
import os
import time

import numpy as np
import torch

from model_nano import NanoModel, count_params

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def rss_gb():
    """Process RSS (GB) via /proc/self/status - pure stdlib."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return -1.0


def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def make_batch(tokens, rng, batch, seq):
    starts = rng.integers(0, len(tokens) - seq - 1, size=batch)
    x = np.stack([tokens[s: s + seq] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1: s + seq + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    floor = 0.1 * args.lr
    return floor + 0.5 * (args.lr - floor) * (1.0 + math.cos(math.pi * t))


def save_ckpt(model, opt, step, rng, args, path):
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "rng_np": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "cfg": model.c,
        "args": vars(args),
    }
    tmp = path + f".tmp{os.getpid()}"
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on the same filesystem
    torch.save(payload, path + ".latest.tmp")
    os.replace(path + ".latest.tmp", os.path.join(os.path.dirname(path), "ckpt_latest.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="override n_layers (smoke)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--ckpt-secs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--trim-every", type=int, default=0,
                    help="if >0: gc.collect() + malloc_trim(0) every N steps (glibc RSS anti-creep)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="clean stop after N hours (final checkpoint then exit)")
    ap.add_argument("--rss-cap", type=float, default=0.0,
                    help="if >0: clean stop (checkpoint) when RSS exceeds N GB - anti-OOM-kill")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    cfg = {"n_layers": args.layers} if args.layers else None
    model = NanoModel(cfg, grad_ckpt=True).float()
    model.eval()  # required by the MoE gate assert; no dropout/BN in the arch
    total, experts = count_params(model)
    print(f"nanokimi : {total / 1e6:.1f} M params (experts {experts / 1e6:.1f} M), "
          f"{model.c['n_layers']} layers, batch {args.batch}×{args.seq}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens = load_tokens(args.data)
    print(f"corpus : {len(tokens) / 1e6:.1f} M tokens", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    step0 = 0
    ckpt_latest = os.path.join(args.out, "ckpt_latest.pt")
    if args.resume and os.path.exists(ckpt_latest):
        print(f"resuming from {ckpt_latest} …", flush=True)
        ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        rng.bit_generator.state = ck["rng_np"]
        torch.set_rng_state(ck["rng_torch"])
        print(f"  → step {step0}", flush=True)

    t0 = time.time()
    t_ckpt = t0
    loss_ema = None
    toks = 0
    stop_reason = "steps reached"
    for step in range(step0, args.steps):
        if args.max_hours is not None and (time.time() - t0) > args.max_hours * 3600:
            stop_reason = f"time-cap {args.max_hours} h reached"
            print(f"[{stop_reason} - clean stop at step {step}]", flush=True)
            break
        if args.rss_cap > 0 and rss_gb() > args.rss_cap:
            stop_reason = f"rss-cap {args.rss_cap} GB exceeded ({rss_gb():.1f} GB)"
            print(f"[{stop_reason} - clean stop at step {step}]", flush=True)
            break
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = make_batch(tokens, rng, args.batch, args.seq)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        toks += args.batch * args.seq
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        # explicitly frees the step's tensors (reduces malloc churn)
        del x, y, logits, loss

        if (step + 1) % args.log_every == 0 or step == step0:
            dt = time.time() - t0
            eta = dt / max(1, step + 1 - step0) * (args.steps - step - 1)
            print(
                f"step {step + 1:6d}/{args.steps} | loss {loss_ema:.4f} "
                f"(ema {loss_ema:.4f}) | lr {lr:.2e} | {toks / dt:.0f} tok/s | "
                f"rss {rss_gb():.1f} GB | {dt / 60:.1f} min, eta {eta / 60:.1f} min",
                flush=True,
            )
        if args.trim_every > 0 and (step + 1) % args.trim_every == 0:
            gc.collect()
            _LIBC.malloc_trim(0)
        now = time.time()
        if (step + 1) % args.ckpt_every == 0 or (now - t_ckpt) > args.ckpt_secs or (step + 1) == args.steps:
            save_ckpt(model, opt, step + 1, rng, args, os.path.join(args.out, f"ckpt_{step + 1:07d}.pt"))
            t_ckpt = now
            print(f"  [ckpt step {step + 1} written]", flush=True)

    # systematic final checkpoint (in particular after a time-cap: `step` is
    # the unexecuted index = total number of steps performed → correct --resume)
    if stop_reason != "steps reached":
        save_ckpt(model, opt, step, rng, args, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
        print(f"  [final ckpt step {step} written]", flush=True)
    print(f"done ({stop_reason}): {toks / 1e6:.1f} M tokens in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
