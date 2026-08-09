#!/usr/bin/env python3
"""nanokimi - heal_stream --init-seam / --lora-auto verification (CPU, smoke model).

Covers, against a small microkimi .bin (e.g. chat_smoke/nanokimi_chat_smoke.bin):
  (a) --init-seam: a fabricated stitch.pt with a known nonzero B initializes
      the seam adapter - the log prints the nonzero |B|, the checkpoint holds
      seam_adapter.A/B, and training moves B AWAY from the init;
  (b) resume precedence: a checkpoint WITHOUT a seam resumed with
      --seam-adapter + --init-seam starts the adapter from the INIT values
      (not zero); a checkpoint WITH a seam resumes unchanged (init ignored);
      --init-seam without --seam-adapter is refused;
  (c) --lora-auto: the lens-log heuristic picks the documented layer range
      (deepest contiguous collapsed tail, one layer above it through its
      end), tolerates junk lines, falls back to every layer when no collapse
      parses, and explicit --lora-layers always wins.

usage: python3 test_heal_init.py [--model smoke.bin] [--data tokens.bin]
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from heal_stream import lens_top1, lora_layers_from_lens

HERE = os.path.dirname(os.path.abspath(__file__))
RANK, AFTER = 16, 5  # smoke model: hidden 512, 8 layers (layer 6 is KDA)
HIDDEN = 512

# 22-layer synthetic lens log: layer 0 sits below 10% before any confident
# layer (NOT a collapse - no signal existed upstream yet), layer 10 collapses
# and layer 11 recovers (an early run the deepest-tail rule must skip),
# layers 19-21 are the deepest contiguous collapsed tail -> expect 18..21.
LENS_LOG_22 = """── logit lens (prefill): top-5 of each layer through final norm + lm_head ──
  layer  0 (KDA): ' the' 5.5%  ' a' 4.1%  ' of' 3.0%  ' to' 2.2%  ' and' 1.9%
  layer  1 (MLA): ' the' 22.4%  ' a' 9.3%  ' of' 7.7%  ' to' 5.1%  ' and' 4.0%
  layer  2 (KDA): ' the' 25.1%  ' of' 8.8%  ' a' 7.2%  ' to' 4.4%  ' and' 3.3%
  layer  3 (KDA): ' France' 31.0%  ' the' 12.5%  ' of' 6.6%  ' a' 4.9%  ' to' 3.1%
some unrelated log line that must be skipped
  layer  4 (MLA): ' France' 35.2%  ' Paris' 9.1%  ' the' 6.0%  ' of' 4.4%  ' a' 3.0%
  layer  5 (KDA): ' France' 38.8%  ' Paris' 10.2%  ' the' 5.5%  ' of' 4.0%  ' a' 2.8%
  layer  6 (KDA): rms=12.3456
  layer  6 (KDA): ' France' 40.1%  ' Paris' 11.0%  ' the' 5.1%  ' of' 3.8%  ' a' 2.5%
  layer  7 (MLA): ' France' 41.7%  ' Paris' 11.4%  ' the' 4.8%  ' of' 3.5%  ' a' 2.2%
  layer  8 (KDA): ' France' 43.0%  ' Paris' 12.0%  ' the' 4.4%  ' of' 3.1%  ' a' 2.0%
  layer  9 (KDA): ' France' 44.5%  ' Paris' 12.3%  ' the' 4.1%  ' of' 2.9%  ' a' 1.8%
  layer 10 (MLA): ' of' 3.0%  ' the' 2.8%  ' a' 2.5%  ' to' 2.1%  ' and' 1.9%
  layer 11 (KDA): ' France' 31.6%  ' Paris' 9.4%  ' the' 5.6%  ' of' 3.7%  ' a' 2.4%
  layer 12 (KDA): ' France' 36.0%  ' Paris' 10.1%  ' the' 5.0%  ' of' 3.3%  ' a' 2.1%
  layer 13 (MLA): ' France' 39.4%  ' Paris' 10.9%  ' the' 4.6%  ' of' 3.0%  ' a' 1.9%
  layer 14 (KDA): ' France' 42.2%  ' Paris' 11.6%  ' the' 4.2%  ' of' 2.7%  ' a' 1.7%
  layer 15 (KDA): ' France' 45.8%  ' Paris' 12.4%  ' the' 3.9%  ' of' 2.5%  ' a' 1.5%
  layer 16 (MLA): ' France' 46.6%  ' of' 5.8%  ' Paris' 12.9%  ' the' 3.6%  ' a' 1.4%
  layer 17 (KDA): ' France' 47.1%  ' Paris' 13.2%  ' the' 3.3%  ' of' 2.2%  ' a' 1.3%
  layer 18 (KDA): ' France' 47.9%  ' Paris' 13.5%  ' the' 3.1%  ' of' 2.0%  ' a' 1.2%
  layer 19 (MLA): ' of' 4.2%  ' the' 3.9%  ' a' 3.0%  ' to' 2.6%  ' and' 2.0%
  layer 20 (MLA): " of" 2.1%  " the" 1.9%  " a" 1.7%  " to" 1.4%  " and" 1.1%
  layer 21 (KDA): ' the' 1.3%  ' of' 1.2%  ' a' 1.0%  ' to' 0.9%  ' and' 0.8%
"""

# 8-layer log for the end-to-end CLI run on the smoke model: collapse in the
# last two layers -> expect adapters on layers 5..7.
LENS_LOG_8 = "\n".join(
    [f"  layer {l} ({'KDA' if l % 3 else 'MLA'}): ' France' {40.0 + l:.1f}%  ' of' 5.0%"
     for l in range(6)]
    + ["  layer  6 (KDA): ' of' 3.3%  ' the' 2.9%  ' a' 2.2%",
       "  layer  7 (KDA): ' the' 1.1%  ' of' 1.0%  ' a' 0.9%"])


def run_heal(model, data, out, extra, steps):
    cmd = [
        sys.executable, os.path.join(HERE, "..", "heal_stream.py"),
        "--model", model, "--data", data, "--out", out,
        "--lora", "8", "--seq", "64", "--batch", "1", "--steps", str(steps),
        "--warmup", "1", "--lr", "1e-3", "--device", "cpu", "--ignore-unk",
        "--log-every", "1", "--ckpt-every", str(steps), "--ckpt-secs", "36000",
        "--threads", "8",
    ] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"heal_stream failed: {' '.join(cmd)}")
    return r.stdout


def make_stitch(path, seed, b_scale=0.05):
    """A stitch.pt as stitch_solve.py writes it, with a known nonzero B."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(RANK, HIDDEN, generator=g) * 0.02
    b = torch.randn(HIDDEN, RANK, generator=g) * b_scale
    torch.save({"seam_adapter.A": a, "seam_adapter.B": b,
                "seam_after": AFTER, "rank": RANK}, path)
    return a, b


def load_ckpt(out):
    return torch.load(os.path.join(out, "ckpt_latest.pt"), map_location="cpu",
                      weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/references/chat_smoke/nanokimi_chat_smoke.bin")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "nano_chat",
                                                   "out_smoke", "tokens_chat.bin"))
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="heal_init_test_")
    print(f"workdir: {tmp}", flush=True)

    # ── (a) --init-seam: nonzero init visible in the log, training continues ──
    print("(a) --init-seam from a fabricated stitch ...", flush=True)
    stitch = os.path.join(tmp, "stitch.pt")
    a_init, b_init = make_stitch(stitch, seed=11)
    out_a = os.path.join(tmp, "run_a")
    log = run_heal(args.model, args.data, out_a,
                   ["--seam-adapter", str(RANK), "--seam-after", str(AFTER),
                    "--init-seam", stitch], 2)
    assert "init-seam" in log, log[-2000:]
    assert f"{b_init.norm():.4e}" in log, "loaded |B| norm not printed"
    ck = load_ckpt(out_a)
    assert {"seam_adapter.A", "seam_adapter.B"} <= set(ck["model"])
    ck_b = ck["model"]["seam_adapter.B"]
    assert not torch.equal(ck_b, b_init), "seam B did not move from the init"
    # ... but it started FROM the init, not from zero: still much closer to it
    assert (ck_b - b_init).norm() < 0.5 * b_init.norm()
    assert ck["seam"] == {"rank": RANK, "after": AFTER}
    print(f"    OK: log shows |B| {b_init.norm():.4e}, ckpt holds the adapter, "
          f"training moved it ({(ck_b - b_init).norm():.2e} away from init)", flush=True)

    # refusal: --init-seam without --seam-adapter
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "..", "heal_stream.py"),
         "--model", args.model, "--data", args.data,
         "--out", os.path.join(tmp, "run_refused"), "--init-seam", stitch,
         "--device", "cpu"],
        capture_output=True, text=True)
    assert r.returncode != 0 and "--init-seam requires --seam-adapter" in (
        r.stdout + r.stderr), (r.returncode, r.stdout, r.stderr)
    print("    OK: --init-seam without --seam-adapter refused", flush=True)

    # ── (b) resume precedence: init applies only when the ckpt has no seam ──
    print("(b) resume precedence ...", flush=True)
    out_b = os.path.join(tmp, "run_b")
    run_heal(args.model, args.data, out_b, [], 2)  # no seam
    ck1 = load_ckpt(out_b)
    assert ck1.get("seam") is None
    log = run_heal(args.model, args.data, out_b,
                   ["--resume", "--seam-adapter", str(RANK), "--seam-after", str(AFTER),
                    "--init-seam", stitch], 4)
    assert "fresh zero-init" in log and "init-seam" in log, log[-2000:]
    ck2 = load_ckpt(out_b)
    assert ck2["step"] == 4 and ck2["seam"] == {"rank": RANK, "after": AFTER}
    b2 = ck2["model"]["seam_adapter.B"]
    assert (b2 - b_init).norm() < 0.5 * b_init.norm(), \
        "adapter did not start from the init values"
    print("    OK: no-seam ckpt + --init-seam: adapter started from the init", flush=True)
    # resume the ckpt that now HAS a seam, with a DIFFERENT stitch: ignored
    stitch_far = os.path.join(tmp, "stitch_far.pt")
    _, b_far = make_stitch(stitch_far, seed=99, b_scale=0.5)
    log = run_heal(args.model, args.data, out_b,
                   ["--resume", "--seam-adapter", str(RANK), "--seam-after", str(AFTER),
                    "--init-seam", stitch_far], 6)
    assert "init ignored" in log, log[-2000:]
    ck3 = load_ckpt(out_b)
    b3 = ck3["model"]["seam_adapter.B"]
    assert (b3 - b2).norm() < (b3 - b_far).norm(), \
        "checkpoint seam state lost to the init"
    print("    OK: ckpt with a seam resumes unchanged (init ignored)", flush=True)

    # ── (c) --lora-auto: heuristic, fallback, explicit override ──
    print("(c) --lora-auto ...", flush=True)
    lens22 = os.path.join(tmp, "lens22.log")
    with open(lens22, "w") as f:
        f.write(LENS_LOG_22)
    top1 = lens_top1(lens22)
    assert top1[0] == 5.5 and top1[19] == 4.2 and top1[20] == 2.1 and top1[21] == 1.3
    assert 6 in top1 and top1[6] == 40.1  # the rms line skipped, the lens row kept
    layers = lora_layers_from_lens(lens22, n_layers=22)
    assert layers == [18, 19, 20, 21], layers
    print(f"    OK: 22-layer log (collapse 19..21) -> layers {layers}", flush=True)
    # no parsable collapse -> None (the CLI falls back to every layer)
    flat = os.path.join(tmp, "lens_flat.log")
    with open(flat, "w") as f:
        f.write("\n".join(f"  layer {l} (KDA): ' the' {30.0 + l:.1f}%" for l in range(8)))
    assert lora_layers_from_lens(flat, n_layers=8) is None
    junk = os.path.join(tmp, "lens_junk.log")
    with open(junk, "w") as f:
        f.write("no lens rows here\njust noise\n")
    assert lora_layers_from_lens(junk) is None
    print("    OK: no collapse / no lens rows -> fallback (None)", flush=True)
    # end to end on the smoke model: collapse at layers 6..7 -> layers 5..7
    lens8 = os.path.join(tmp, "lens8.log")
    with open(lens8, "w") as f:
        f.write(LENS_LOG_8)
    out_c = os.path.join(tmp, "run_c")
    log = run_heal(args.model, args.data, out_c, ["--lora-auto", lens8], 1)
    assert "lora-auto" in log and "layers [5, 6, 7]" in log, log[-2000:]
    print("    OK (CLI): --lora-auto picked layers [5, 6, 7] on the smoke model", flush=True)
    # explicit --lora-layers wins over --lora-auto
    out_d = os.path.join(tmp, "run_d")
    log = run_heal(args.model, args.data, out_d,
                   ["--lora-layers", "2-3", "--lora-auto", lens8], 1)
    assert "layers [2, 3]" in log and "lora-auto" not in log, log[-2000:]
    print("    OK (CLI): explicit --lora-layers 2-3 wins over --lora-auto", flush=True)

    print("ALL HEAL INIT TESTS OK", flush=True)


if __name__ == "__main__":
    main()
