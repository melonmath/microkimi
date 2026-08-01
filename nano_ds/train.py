#!/usr/bin/env python3
"""nanodeepseek - training loop (train.py)

AdamW + cosine (warmup), CE loss, random windows over tokens.bin (uint16).
ATOMIC checkpoints (tmp + rename) every N steps AND every T seconds
(preemptible box: --resume restarts from the last healthy state:
model+opt+step+rng). Timestamped checkpoints are pruned to --keep most
recent (they are ~1 GB each for the full 8-layer model).

examples:
  # local smoke (reduced config):
  python3 train.py --data /tmp/ds_out_smoke/tokens.bin --out /tmp/ds_ckpt \
      --layers 4 --batch 4 --seq 128 --steps 200 --threads 10
  # night run (32 vCPU box):
  python3 train.py --data ~/nanods_out/tokens.bin --out ~/nanods_out/ckpt \
      --batch 32 --seq 256 --steps 700 --threads 32 --resume --keep 3
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

from model_ds import NanoDsModel, count_params, NANO_DS

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


def prune_ckpts(out_dir, keep):
    """Keeps only the `keep` most recent ckpt_*.pt (ckpt_latest.pt untouched)."""
    if keep <= 0:
        return
    ckpts = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith("ckpt_") and f.endswith(".pt") and f != "ckpt_latest.pt"
    )
    for f in ckpts[:-keep]:
        os.remove(os.path.join(out_dir, f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=None, help="override n_layers (smoke)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=256, help="window size (multiple of 128 recommended)")
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--ckpt-secs", type=int, default=1800)
    ap.add_argument("--keep", type=int, default=3, help="timestamped checkpoints kept (0 = all)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", choices=["cpu", "mps", "auto"], default="cpu",
                    help="torch device: cpu (default) | mps (Apple Silicon GPU) | auto")
    ap.add_argument("--bench", type=int, default=None,
                    help="bench mode: run only N steps and print tok/s at the end")
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
    dev = "cpu"
    if args.device == "mps":
        dev = "mps"
    elif args.device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        if dev == "cpu":
            print("device=auto: MPS unavailable -> cpu", flush=True)
    print(f"device: {dev}", flush=True)
    cfg = None
    if args.layers:
        cfg = {"n_layers": args.layers, "compress_ratios": NANO_DS["compress_ratios"][: args.layers]}
    model = NanoDsModel(cfg, grad_ckpt=True).float().to(dev)
    model.eval()  # no dropout/BN in the architecture; eval changes nothing
    total, experts = count_params(model)
    print(f"nanodeepseek: {total / 1e6:.1f} M params (routed experts {experts / 1e6:.1f} M), "
          f"{model.c['n_layers']} layers, batch {args.batch}x{args.seq}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens = load_tokens(args.data)
    print(f"corpus: {len(tokens) / 1e6:.1f} M tokens", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.bench is not None:
        args.steps = args.bench
        print(f"bench mode: {args.bench} steps", flush=True)
    step0 = 0
    ckpt_latest = os.path.join(args.out, "ckpt_latest.pt")
    if args.resume and os.path.exists(ckpt_latest):
        print(f"resuming from {ckpt_latest} ...", flush=True)
        ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if dev != "cpu":
            for st in opt.state.values():
                for key, val in st.items():
                    if torch.is_tensor(val):
                        st[key] = val.to(dev)
        step0 = ck["step"]
        rng.bit_generator.state = ck["rng_np"]
        torch.set_rng_state(ck["rng_torch"])
        print(f"  -> step {step0}", flush=True)

    t0 = time.time()
    t_ckpt = t0
    loss_ema = None
    toks = 0
    stop_reason = "steps done"
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
        x = x.to(dev)
        y = y.to(dev)
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
        del x, y, logits, loss

        if (step + 1) % args.log_every == 0 or step == step0:
            dt = time.time() - t0
            eta = dt / max(1, step + 1 - step0) * (args.steps - step - 1)
            print(
                f"step {step + 1:6d}/{args.steps} | loss {loss_ema:.4f} "
                f"| lr {lr:.2e} | {toks / dt:.0f} tok/s | "
                f"rss {rss_gb():.1f} GB | {dt / 60:.1f} min, eta {eta / 60:.1f} min",
                flush=True,
            )
        if args.trim_every > 0 and (step + 1) % args.trim_every == 0:
            gc.collect()
            _LIBC.malloc_trim(0)
        now = time.time()
        if (step + 1) % args.ckpt_every == 0 or (now - t_ckpt) > args.ckpt_secs or (step + 1) == args.steps:
            save_ckpt(model, opt, step + 1, rng, args, os.path.join(args.out, f"ckpt_{step + 1:07d}.pt"))
            prune_ckpts(args.out, args.keep)
            t_ckpt = now
            print(f"  [ckpt step {step + 1} written]", flush=True)

    if stop_reason != "steps done":
        save_ckpt(model, opt, step, rng, args, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
        prune_ckpts(args.out, args.keep)
        print(f"  [final ckpt step {step} written]", flush=True)
    print(f"done ({stop_reason}): {toks / 1e6:.1f} M tokens in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.bench is not None:
        dt = time.time() - t0
        print(
            f"BENCH device={dev} | {args.bench} steps | {toks / dt:.0f} tok/s | "
            f"rss {rss_gb():.1f} GB | loss(ema) {loss_ema if loss_ema is not None else float('nan'):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
