#!/usr/bin/env python3
"""nanokimi - training loop (train.py)

AdamW + cosine (warmup), CE loss, random windows over tokens.bin (uint16).
ATOMIC checkpoints (tmp + rename) every N steps AND every T seconds
(preemptible box: --resume restarts from the last sane state: model+opt+step+rng).

examples:
  # local smoke (reduced config):
  python3 train.py --data out_dev/tokens.bin --out out_dev/ckpt \
      --layers 4 --batch 4 --seq 128 --steps 200 --threads 10
  # real run (32 vCPU box):
  python3 train.py --data ~/nano_out/tokens.bin --out ~/nano_out/ckpt \
      --batch 32 --seq 512 --steps 8000 --threads 32 --resume

GPU (Apple Silicon, torch MPS backend):
  python3 train.py --data tokens.bin --out ckpt --device mps --bench 20
GPU (NVIDIA, torch CUDA backend):
  python3 train.py --data tokens.bin --out ckpt --device cuda --bench 20
  --device {cpu,mps,cuda,auto} : cpu is the default and is UNCHANGED from before;
  auto picks cuda, then mps, else cpu.
  Known GPU bottleneck: the KDA recurrence is a per-token Python loop (one
  kernel launch per token per layer - much worse on GPU than on CPU). The
  exact fix is the chunked WY representation of the delta rule, intentionally
  NOT implemented here because it changes the float summation order (the
  engine is validated 1:1 against the per-token recurrence).
  On cuda, two math-preserving fast paths are active (env-tunable, see
  model_nano.py / vendor/fla/ops/kda/__init__.py): chunked MoE with cached
  expert stacks (NANO_MOE_CHUNK, NANO_MOE_FAST_DEVICES) and time-segment
  gradient checkpointing of the KDA recurrence (NANO_KDA_SEG,
  NANO_KDA_SEG_DEVICES). The cpu/mps paths are unchanged.
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


def _tokens_dtype(path):
    """Picks the memmap dtype for a tokens.bin corpus.

    Default is uint16 (nano vocab 8200). A sibling meta file switches to
    uint32 when it declares a u32 dtype or a vocab larger than 65535
    (real K3 vocab is 163840). Meta is looked up as <data>.meta.json first,
    then as <stem>.meta.json (the convention used by prepare.py). Without a
    meta file the behavior is exactly the legacy one: uint16.
    """
    metas = [path + ".meta.json"]
    stem, ext = os.path.splitext(path)
    if ext:
        metas.append(stem + ".meta.json")
    for meta in metas:
        if not os.path.exists(meta):
            continue
        try:
            with open(meta, "r", encoding="utf-8") as f:
                info = json.load(f)
        except (OSError, ValueError):
            continue
        dtype = str(info.get("dtype", "")).lower()
        vocab = info.get("vocab", info.get("vocab_size", 0)) or 0
        if dtype.replace("le", "") in ("uint32", "u32") or vocab > 65535:
            return np.uint32
        return np.uint16  # explicit meta wins; keep the declared legacy dtype
    return np.uint16


def load_tokens(path):
    return np.memmap(path, dtype=_tokens_dtype(path), mode="r")


def dev_mem_str(dev):
    """Device memory for the bench line: MPS/CUDA allocated memory when available,
    process RSS otherwise."""
    if dev == "mps":
        try:
            return f"mps alloc {torch.mps.current_allocated_memory() / 1e9:.2f} GB"
        except Exception:
            return "mps alloc n/a"
    if dev == "cuda":
        return f"cuda alloc {torch.cuda.memory_allocated() / 1e9:.2f} GB"
    return f"rss {rss_gb():.1f} GB"


def make_batch(tokens, rng, batch, seq):
    starts = rng.integers(0, len(tokens) - seq - 1, size=batch)
    x = np.stack([tokens[s: s + seq] for s in starts]).astype(np.int64)
    y = np.stack([tokens[s + 1: s + seq + 1] for s in starts]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def depth_bounds(n_layers, min_frac):
    """Sampled-depth interval: [ceil(min_frac * n_layers), n_layers] (min 1)."""
    lo = min(n_layers, max(1, int(math.ceil(min_frac * n_layers))))
    return lo, n_layers


def sample_depth(rng, n_layers, min_frac, full_p):
    """Stochastic-depth sampler (nested-model training): the FULL depth with
    probability `full_p`, else a uniform depth over the shallower prefix range.
    Draws from the training rng (the same stream as the batches), so --seed
    and --resume fully determine the sampled sequence."""
    lo, hi = depth_bounds(n_layers, min_frac)
    if lo >= hi or rng.random() < full_p:
        return hi
    return int(rng.integers(lo, hi))


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
        "lora": getattr(model, "lora_info", None),
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
    ap.add_argument("--experts", type=int, default=None, help="override n_experts (smoke)")
    ap.add_argument("--vocab", type=int, default=None,
                    help="override vocabulary size; token bins must match")
    ap.add_argument("--hidden", type=int, default=None,
                    help="override hidden size")
    ap.add_argument("--kda-heads", type=int, default=None,
                    help="override KDA head count (keep kda_heads*kda_dim ~ hidden)")
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
    ap.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"], default="cpu",
                    help="torch device: cpu (default, unchanged) | mps (Apple Silicon GPU) | cuda (NVIDIA GPU) | auto (cuda, else mps, else cpu)")
    ap.add_argument("--bench", type=int, default=None,
                    help="bench mode: run only N steps and print tok/s + device memory at the end")
    ap.add_argument("--trim-every", type=int, default=0,
                    help="if >0: gc.collect() + malloc_trim(0) every N steps (glibc RSS anti-creep)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="clean stop after N hours (final checkpoint then exit)")
    ap.add_argument("--rss-cap", type=float, default=0.0,
                    help="if >0: clean stop (checkpoint) when RSS exceeds N GB - anti-OOM-kill")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh-opt", action="store_true",
                    help="with --resume: keep ONLY the checkpoint weights "
                         "(SFT from a base model) - optimizer, step and rng restart from scratch")
    ap.add_argument("--ignore-unk", action="store_true",
                    help="exclude UNK (8198) targets from the loss (ignore_index). For chat SFT on "
                         "the nano vocab: structural XTML tags and out-of-vocab words map to UNK, "
                         "so without this the model learns to EMIT [UNK] and greedy decoding collapses")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast for the forward pass (faster on GPU, small numeric drift - "
                         "opt-in, loss is still computed in fp32)")
    ap.add_argument("--lora", type=int, default=0,
                    help="LoRA rank on the attention projections (0 = full training, default). "
                         "Base weights are frozen, only the adapters train (healing mode)")
    ap.add_argument("--lora-alpha", type=float, default=None,
                    help="LoRA alpha (default: same as the rank, i.e. scaling 1.0)")
    ap.add_argument("--lora-targets", default="attn",
                    help="comma list from q,k,v,o,attn (default attn: q/k/v/o projections)")
    ap.add_argument("--lora-norms", action="store_true",
                    help="also train the norm gains (everything else stays frozen)")
    ap.add_argument("--stochastic-depth", action="store_true",
                    help="nested-model training: each optimizer step samples a depth d in "
                         "[--stochastic-depth-min x n_layers, n_layers], forwards only that "
                         "layer prefix and reads the logits off it with the logit-lens path "
                         "(final norm + lm_head). One run yields a model whose every prefix "
                         "is a valid smaller model. Default OFF (pure opt-in)")
    ap.add_argument("--stochastic-depth-min", type=float, default=0.25,
                    help="smallest sampled depth, as a fraction of n_layers (default 0.25)")
    ap.add_argument("--stochastic-depth-full-p", type=float, default=0.5,
                    help="probability of sampling the FULL depth on a step (default 0.5; "
                         "the remaining mass is uniform over the shallower depths)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(args.threads)
    # device resolution: cpu default is untouched; auto picks cuda, then mps, then cpu
    dev = "cpu"
    if args.device in ("mps", "cuda"):
        dev = args.device
    elif args.device == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        if dev == "cpu":
            print("device=auto: no GPU available -> cpu", flush=True)
    print(f"device: {dev}", flush=True)
    if args.amp and dev != "cuda":
        print("--amp requested but device is not cuda: amp is a no-op (bf16 autocast is cuda-only here)", flush=True)
    cfg = {k: v for k, v in (("n_layers", args.layers), ("n_experts", args.experts),
                             ("vocab", args.vocab), ("hidden", args.hidden),
                             ("kda_heads", args.kda_heads)) if v}
    cfg = cfg or None
    # --resume: the model config comes from the checkpoint itself (a ckpt
    # converted from a .bin by bin2pt.py carries the .bin config, possibly
    # with explicit mla_layers/dense_layers lists); peek before building.
    ckpt_latest = os.path.join(args.out, "ckpt_latest.pt")
    pre_ck = None
    if args.resume and os.path.exists(ckpt_latest):
        pre_ck = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
        cfg = pre_ck.get("cfg", cfg)
        # a resumed run keeps the stochastic-depth behavior it was started
        # with (the flag state lives in the saved run config)
        saved_args = pre_ck.get("args") or {}
        if saved_args.get("stochastic_depth"):
            args.stochastic_depth = True
            args.stochastic_depth_min = saved_args.get("stochastic_depth_min", args.stochastic_depth_min)
            args.stochastic_depth_full_p = saved_args.get("stochastic_depth_full_p", args.stochastic_depth_full_p)
            print("stochastic depth: behavior inherited from the checkpoint args", flush=True)
    # seed BEFORE model construction: the init draws from the torch global
    # generator, which otherwise starts from per-process entropy and --seed
    # would not reproduce a run. The manual_seed after the build (below) keeps
    # the post-init stream exactly as it was; a --resume overrides both with
    # the checkpoint states.
    torch.manual_seed(args.seed)
    model = NanoModel(cfg, grad_ckpt=True).float().to(dev)
    model.eval()  # required by the MoE gate assert; no dropout/BN in the arch
    total, experts = count_params(model)
    print(f"nanokimi: {total / 1e6:.1f} M params (experts {experts / 1e6:.1f} M), "
          f"{model.c['n_layers']} layers, batch {args.batch}×{args.seq}", flush=True)

    # LoRA (healing): the adapter config comes from the checkpoint being
    # resumed when it has one, else from the flags. Base weights frozen,
    # only the adapters (+ norms with --lora-norms) receive gradients.
    from model_nano import apply_lora
    lora_cfg = (pre_ck or {}).get("lora")
    if lora_cfg is None and args.lora > 0:
        lora_cfg = {
            "rank": args.lora,
            "alpha": args.lora_alpha if args.lora_alpha is not None else args.lora,
            "targets": [t.strip() for t in args.lora_targets.split(",") if t.strip()],
            "norms": args.lora_norms,
        }
    if lora_cfg:
        n_train, n_total = apply_lora(
            model, lora_cfg["rank"], lora_cfg["alpha"], lora_cfg["targets"], lora_cfg.get("norms", False)
        )
        print(f"lora: rank {lora_cfg['rank']} alpha {lora_cfg['alpha']} targets {lora_cfg['targets']} "
              f"on {len(model.lora_info['wrapped'])} linears - {n_train / 1e6:.2f} M trainable / "
              f"{n_total / 1e6:.1f} M total ({100 * n_train / n_total:.2f}%)", flush=True)

    # only the trainable params reach the optimizer (LoRA: ~1% of the model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    tokens = load_tokens(args.data)
    print(f"corpus: {len(tokens) / 1e6:.1f} M tokens", flush=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    n_layers = model.c["n_layers"]
    if args.stochastic_depth:
        lo, _ = depth_bounds(n_layers, args.stochastic_depth_min)
        print(f"stochastic depth: sampling d in [{lo}, {n_layers}] per step, "
              f"P(full)={args.stochastic_depth_full_p} "
              f"(prefix readout = final norm + lm_head)", flush=True)

    if args.bench is not None:
        # bench mode: cap the run at N steps, summary printed at the end
        args.steps = args.bench
        print(f"bench mode: {args.bench} steps", flush=True)
    step0 = 0
    if pre_ck is not None:
        print(f"resuming from {ckpt_latest} ...", flush=True)
        ck = pre_ck
        sd = ck["model"]
        if lora_cfg and not any(k.endswith(".lora_A") for k in sd):
            # plain (non-LoRA) checkpoint loaded into a LoRA-wrapped model:
            # route the base weights to the frozen .base of each wrapper;
            # the adapters keep their fresh init (A kaiming, B zero)
            sd = dict(sd)
            for w in model.lora_info["wrapped"]:
                key = w + ".weight"
                if key in sd:
                    sd[w + ".base.weight"] = sd.pop(key)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            assert not unexpected, f"unexpected keys in checkpoint: {unexpected[:4]}"
            assert all(k.endswith((".lora_A", ".lora_B")) for k in missing), f"missing non-LoRA keys: {missing[:4]}"
        else:
            model.load_state_dict(sd)
        # optimizer resume: skipped when LoRA-healing from a PLAIN checkpoint
        # (the full-training param groups do not map onto the adapters)
        healing_fresh = lora_cfg is not None and ck.get("lora") is None and "opt" in ck
        if "opt" in ck and not healing_fresh:
            opt.load_state_dict(ck["opt"])
            # optimizer states load as CPU tensors - move them onto the run device
            # (AdamW requires state and params on the same device)
            if dev != "cpu":
                for st in opt.state.values():
                    for key, val in st.items():
                        if torch.is_tensor(val):
                            st[key] = val.to(dev)
            step0 = ck["step"]
            rng.bit_generator.state = ck["rng_np"]
            torch.set_rng_state(ck["rng_torch"])
            print(f"  -> step {step0}", flush=True)
        else:
            # weights-only checkpoint (e.g. bin2pt conversion) or LoRA
            # healing start: optimizer, step and rng stay fresh
            if healing_fresh:
                print("  lora healing from a plain checkpoint: fresh optimizer/step/rng", flush=True)
            else:
                print("  weights-only checkpoint: optimizer/step/rng start fresh", flush=True)
        if args.fresh_opt:
            # SFT from a base model: keep only the weights; optimizer
            # (AdamW moments from pretraining), step counter and rng restart
            # from scratch so the LR schedule is fresh (full cosine over --steps).
            opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                    lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
            step0 = 0
            rng = np.random.default_rng(args.seed)
            torch.manual_seed(args.seed)
            print("  --fresh-opt: optimizer/step/rng reset (weights kept)", flush=True)

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
        x = x.to(dev)
        y = y.to(dev)
        depth = n_layers
        if args.stochastic_depth:
            depth = sample_depth(rng, n_layers, args.stochastic_depth_min,
                                 args.stochastic_depth_full_p)
        if args.amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                logits = model(x) if depth == n_layers else model.forward_prefix(x, depth)
        else:
            logits = model(x) if depth == n_layers else model.forward_prefix(x, depth)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
            ignore_index=8198 if args.ignore_unk else -100,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        toks += args.batch * args.seq
        loss_ema = loss.item() if loss_ema is None else 0.98 * loss_ema + 0.02 * loss.item()
        # explicitly release the step tensors (reduces malloc churn)
        del x, y, logits, loss

        if (step + 1) % args.log_every == 0 or step == step0:
            dt = time.time() - t0
            eta = dt / max(1, step + 1 - step0) * (args.steps - step - 1)
            depth_str = f" | depth {depth}/{n_layers}" if args.stochastic_depth else ""
            print(
                f"step {step + 1:6d}/{args.steps} | loss {loss_ema:.4f} "
                f"(ema {loss_ema:.4f}){depth_str} | lr {lr:.2e} | {toks / dt:.0f} tok/s | "
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

    # systematic final checkpoint (notably after a time-cap: `step` is the
    # unexecuted index = total number of steps done -> correct --resume)
    if stop_reason != "steps reached":
        save_ckpt(model, opt, step, rng, args, os.path.join(args.out, f"ckpt_{step:07d}.pt"))
        print(f"  [final ckpt step {step} written]", flush=True)
    print(f"done ({stop_reason}): {toks / 1e6:.1f} M tokens in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.bench is not None:
        dt = time.time() - t0
        print(
            f"BENCH device={dev} | {args.bench} steps | {toks / dt:.0f} tok/s | "
            f"{dev_mem_str(dev)} | loss(ema) {loss_ema if loss_ema is not None else float('nan'):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
