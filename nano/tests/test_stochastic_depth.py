#!/usr/bin/env python3
"""nanokimi - stochastic-depth (nested model) training verification (CPU, tiny).

Covers:
  (a) CLI run with --stochastic-depth: finite decreasing loss on a learnable
      corpus, the log shows varying sampled depths, the checkpoint records the
      flag and loads back into a fresh NanoModel;
  (b) resume: a resumed run keeps the stochastic-depth behavior from the
      checkpoint args, without the flag on the command line;
  (c) determinism: same seed -> same sampled depth sequence AND same losses
      (end-to-end CLI, and unit-level on the sampler);
  (d) distribution: over many sampled steps the full-depth fraction matches
      --stochastic-depth-full-p within tolerance, and the shallower depths
      cover the whole [min, n_layers - 1] range;
  (e) flag OFF: two same-seed runs are bit-identical (same losses, no depth
      logging) - the no-regression check on the default path;
  (f) forward_prefix unit check: right logit shape, matches a manually
      truncated reference (embed + first d layers + final norm + lm_head),
      and backward touches only the used prefix + final norm + lm_head.

usage: python3 test_stochastic_depth.py
"""
import math
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model_nano import NanoModel
from train import depth_bounds, sample_depth

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS, EXPERTS = 8, 16  # tiny smoke model (nano vocab kept: the corpus ids stay < 8198)


def make_corpus(path, n=65536):
    """Low-entropy learnable corpus: a short deterministic cycle (each token
    has exactly one successor), so a few steps visibly reduce the loss."""
    t = (100 + 7 * np.arange(n)) % 97 + 100
    t.astype(np.uint16).tofile(path)
    return path


def run_train(data, out, steps, extra, seed=42):
    cmd = [
        sys.executable, os.path.join(HERE, "..", "train.py"),
        "--data", data, "--out", out,
        "--layers", str(LAYERS), "--experts", str(EXPERTS),
        "--batch", "2", "--seq", "32", "--steps", str(steps),
        "--warmup", "2", "--lr", "1e-3", "--device", "cpu",
        "--log-every", "1", "--ckpt-every", "10000", "--ckpt-secs", "36000",
        "--threads", "10", "--seed", str(seed),
    ] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"train.py failed: {' '.join(cmd)}")
    return r.stdout


def parse_losses(log):
    return [float(m) for m in re.findall(r"^step\s+\d+/\d+ \| loss ([\d.]+)", log, re.M)]


def parse_depths(log):
    return [int(m) for m in re.findall(r"\| depth (\d+)/\d+ \|", log)]


def main():
    tmp = tempfile.mkdtemp(prefix="stochastic_depth_test_")
    print(f"workdir: {tmp}", flush=True)
    data = make_corpus(os.path.join(tmp, "tokens.bin"))

    # ── (a) CLI run with --stochastic-depth ──
    print("(a) CLI run with the flag ON ...", flush=True)
    out_a = os.path.join(tmp, "run_a")
    log_a = run_train(data, out_a, 24, ["--stochastic-depth"])
    assert "stochastic depth: sampling d in [2, 8] per step, P(full)=0.5" in log_a, log_a[-2000:]
    depths_a = parse_depths(log_a)
    losses_a = parse_losses(log_a)
    assert len(depths_a) == 24 and len(losses_a) == 24, (len(depths_a), len(losses_a))
    assert all(2 <= d <= LAYERS for d in depths_a), depths_a
    assert len(set(depths_a)) >= 2, f"depths did not vary: {depths_a}"
    assert all(math.isfinite(l) for l in losses_a), losses_a
    assert losses_a[-1] < 0.9 * losses_a[0], f"loss did not decrease: {losses_a[0]} -> {losses_a[-1]}"
    ck = torch.load(os.path.join(out_a, "ckpt_latest.pt"), map_location="cpu", weights_only=False)
    assert ck["step"] == 24 and ck["cfg"]["n_layers"] == LAYERS
    assert ck["args"]["stochastic_depth"] is True
    assert ck["args"]["stochastic_depth_min"] == 0.25 and ck["args"]["stochastic_depth_full_p"] == 0.5
    fresh = NanoModel(ck["cfg"]).float().eval()
    fresh.load_state_dict(ck["model"])  # strict: the full model, no format change
    print(f"    OK: loss {losses_a[0]:.4f} -> {losses_a[-1]:.4f}, depths {depths_a}, ckpt loads", flush=True)

    # ── (b) resume keeps the behavior without re-passing the flag ──
    print("(b) resume inheritance ...", flush=True)
    log_b = run_train(data, out_a, 30, ["--resume"])
    assert "inherited from the checkpoint args" in log_b, log_b[-2000:]
    depths_b = parse_depths(log_b)
    assert len(depths_b) == 6 and all(2 <= d <= LAYERS for d in depths_b), depths_b
    print(f"    OK: resumed run samples depths {depths_b} without the CLI flag", flush=True)

    # ── (c) determinism: same seed -> same depths, same losses ──
    print("(c) determinism ...", flush=True)
    log_c1 = run_train(data, os.path.join(tmp, "run_c1"), 8, ["--stochastic-depth"])
    log_c2 = run_train(data, os.path.join(tmp, "run_c2"), 8, ["--stochastic-depth"])
    assert parse_depths(log_c1) == parse_depths(log_c2), (parse_depths(log_c1), parse_depths(log_c2))
    assert parse_losses(log_c1) == parse_losses(log_c2)
    g1, g2 = np.random.default_rng(0), np.random.default_rng(0)
    seq1 = [sample_depth(g1, LAYERS, 0.25, 0.5) for _ in range(100)]
    seq2 = [sample_depth(g2, LAYERS, 0.25, 0.5) for _ in range(100)]
    assert seq1 == seq2
    print(f"    OK: two CLI runs sampled identical depths {parse_depths(log_c1)} and losses", flush=True)

    # ── (d) distribution sanity ──
    print("(d) distribution sanity ...", flush=True)
    assert depth_bounds(8, 0.25) == (2, 8)
    assert depth_bounds(8, 0.0) == (1, 8)
    assert depth_bounds(4, 0.5) == (2, 4)
    g = np.random.default_rng(3)
    draws = [sample_depth(g, LAYERS, 0.25, 0.5) for _ in range(4000)]
    full_frac = sum(d == LAYERS for d in draws) / len(draws)
    assert abs(full_frac - 0.5) < 0.03, f"full-depth fraction {full_frac:.3f} != 0.5"
    shallow = [d for d in draws if d != LAYERS]
    assert shallow and min(shallow) >= 2 and max(shallow) <= LAYERS - 1
    assert set(shallow) == set(range(2, LAYERS)), f"shallow range not covered: {sorted(set(shallow))}"
    print(f"    OK: full-depth fraction {full_frac:.3f} (target 0.5), shallow depths {sorted(set(shallow))}", flush=True)

    # ── (e) flag OFF: same seed -> bit-identical losses, no depth logging ──
    print("(e) flag OFF no-regression ...", flush=True)
    log_e1 = run_train(data, os.path.join(tmp, "run_e1"), 8, [])
    log_e2 = run_train(data, os.path.join(tmp, "run_e2"), 8, [])
    assert "| depth " not in log_e1
    losses_e1, losses_e2 = parse_losses(log_e1), parse_losses(log_e2)
    assert len(losses_e1) == 8 and losses_e1 == losses_e2, (losses_e1, losses_e2)
    ck_e = torch.load(os.path.join(tmp, "run_e1", "ckpt_latest.pt"),
                      map_location="cpu", weights_only=False)
    assert ck_e["args"]["stochastic_depth"] is False
    print(f"    OK: two OFF runs identical (losses {losses_e1[0]:.4f} -> {losses_e1[-1]:.4f}), "
          "no depth logging", flush=True)

    # ── (f) forward_prefix unit check ──
    print("(f) forward_prefix unit check ...", flush=True)
    torch.manual_seed(0)
    model = NanoModel({"n_layers": 4, "n_experts": 4, "top_k": 2}).float().eval()
    ids = torch.randint(0, 8200, (2, 12))
    d = 2
    with torch.no_grad():
        logits = model.forward_prefix(ids, d)
        assert logits.shape == (2, 12, 8200), logits.shape
        # manually truncated reference: embed + first d layers + final norm + lm_head
        B, T = ids.shape
        D = model.c["hidden"]
        h = model.embed_tokens(ids)
        causal = torch.zeros(1, 1, T, T, dtype=h.dtype)
        causal.masked_fill_(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        blocks = h.new_zeros(B * T, 0, D)
        for l in range(d):
            mask = causal if l in model._mla else None
            h, blocks = model.layers[l]._forward_attn_residual(
                h, attention_mask=mask, block_residual=blocks)
        ref = model.lm_head(model.norm(h))
    assert torch.equal(logits, ref), \
        f"prefix forward != manual truncation (max diff {(logits - ref).abs().max()})"
    # backward touches only the used prefix + final norm + lm_head
    model.zero_grad(set_to_none=True)
    y = torch.randint(0, 8200, (2, 12))
    loss = torch.nn.functional.cross_entropy(
        model.forward_prefix(ids, d).reshape(-1, 8200), y.reshape(-1))
    loss.backward()
    for l in range(d, model.c["n_layers"]):
        assert all(p.grad is None for p in model.layers[l].parameters()), \
            f"layer {l} received gradients through a depth-{d} prefix"
    assert all(p.grad is None for p in
               list(model.output_attn_res_proj.parameters()) +
               list(model.output_attn_res_norm.parameters())), \
        "output attn-res trained by a prefix step"
    assert model.norm.weight.grad is not None and model.lm_head.weight.grad is not None
    assert model.embed_tokens.weight.grad is not None
    assert any(p.grad is not None for p in model.layers[0].parameters())
    assert any(p.grad is not None for p in model.layers[1].parameters())
    print("    OK: logits match the manual truncation; backward stops at the prefix", flush=True)

    print("ALL STOCHASTIC DEPTH TESTS OK", flush=True)


if __name__ == "__main__":
    main()
