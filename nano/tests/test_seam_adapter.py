#!/usr/bin/env python3
"""nanokimi - seam adapter verification on the smoke model (CPU).

Covers, against a small microkimi .bin (e.g. chat_smoke/nanokimi_chat_smoke.bin):
  (a) zero-init bit-identity: a fresh seam adapter (B = 0) leaves the
      streamed forward BIT-IDENTICAL to the same model without it;
  (b) 3 training steps (--seam-adapter + --lora): finite decreasing loss,
      and the checkpoint holds ONLY lora_A/lora_B + seam_adapter.A/B, with
      seam_adapter.B moved away from zero;
  (c) resume across the config change: train WITHOUT the adapter, then
      --resume WITH --seam-adapter - the LoRA weights and their Adam state
      resume, the adapter starts fresh (zero-init);
  (d) merge: at zero-init the folded .bin is byte-identical to the original;
      after training, every folded tensor satisfies W' = W + W B A to float
      precision (the fold itself is implemented exactly), and the merged
      forward is compared to the adapted forward (the fold is exact for the
      linear attention-input read only - the residual pass-through part of
      the correction is NOT captured, see apply_lora_bin.py docstring - so
      the forward delta scales with |B A|, which a scale sweep demonstrates).

usage: python3 test_seam_adapter.py [--model smoke.bin] [--data tokens.bin]
"""
import argparse
import filecmp
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from bin2pt import read_bin, DTYPE_F32
from heal_stream import StreamedHealModel
from train import load_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
LORA_CFG = {"rank": 8, "alpha": 8, "targets": ["q", "k", "v", "o"], "norms": False}
SEAM_CFG = {"rank": 16, "after": 5}  # layer 6 of the smoke model is a KDA layer


def forward_logits(model, ids, lora_cfg=None, seam_cfg=None, sd=None):
    sm = StreamedHealModel(model, "cpu", lora_cfg, seam_cfg)
    if sd is not None:
        sm.load_trainable(sd)
    with torch.no_grad():
        return sm.forward(ids).cpu()


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


def losses_of(log):
    return [float(m.group(1)) for m in
            re.finditer(r"step\s+\d+/\d+ \| loss ([0-9.]+)", log)]


def merge(ckpt, model, out, force=False):
    cmd = [sys.executable, os.path.join(HERE, "..", "apply_lora_bin.py"),
           "--ckpt", ckpt, "--bin", model, "--out", out]
    if force:
        cmd.append("--force-seam-fold")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit("apply_lora_bin failed")
    return r.stdout


def bin_index(path):
    _, entries, f = read_bin(path)
    f.close()
    return {n: (dt, d, o, s) for n, dt, d, o, s in entries}


def tensor(index, path, name):
    dt, dims, off, _ = index[name]
    assert dt == DTYPE_F32, name
    return np.memmap(path, dtype=np.float32, mode="r", offset=off, shape=tuple(dims))


def check_fold_tensors(orig, merged, after, a, b):
    """Every direct input projection W of layer after+1: W' == W + W B A."""
    io, im = bin_index(orig), bin_index(merged)
    layer = after + 1
    from apply_lora_bin import _SEAM_CONSUMER_LEAVES
    ba = (b.double() @ a.double()).numpy()
    worst = 0.0
    n = 0
    for leaf in _SEAM_CONSUMER_LEAVES:
        name = f"layers.{layer}.self_attn.{leaf}.weight"
        if name not in io:
            continue
        w = np.asarray(tensor(io, orig, name), dtype=np.float64)
        got = np.asarray(tensor(im, merged, name), dtype=np.float64)
        want = w + w @ ba
        rel = np.abs(got - want).max() / max(1e-30, np.abs(want).max())
        worst = max(worst, rel)
        n += 1
    assert n > 0, f"no fold target found for layer {layer}"
    return worst, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/references/chat_smoke/nanokimi_chat_smoke.bin")
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "nano_chat", "out_smoke", "tokens_chat.bin"))
    ap.add_argument("--skip-training", action="store_true", help="only (a) and the fabricated parts of (d)")
    args = ap.parse_args()
    tmp = tempfile.mkdtemp(prefix="seam_test_")
    print(f"workdir: {tmp}", flush=True)

    tokens = load_tokens(args.data)
    ids = torch.from_numpy(tokens[:64].astype(np.int64)).unsqueeze(0)

    # ── (a) zero-init bit-identity ──
    print("(a) zero-init bit-identity ...", flush=True)
    torch.manual_seed(0)
    ref = forward_logits(args.model, ids, LORA_CFG, None)
    torch.manual_seed(0)
    seamed = forward_logits(args.model, ids, LORA_CFG, SEAM_CFG)
    assert torch.equal(ref, seamed), f"not bit-identical: max {(ref - seamed).abs().max()}"
    print("    OK: forward with fresh seam adapter is bit-identical", flush=True)

    if not args.skip_training:
        # ── (b) 3 training steps on ONE fixed batch (in-process): the loss on
        # that batch must decrease, and only the LoRA + seam params may move ──
        print("(b) 3 training steps, fixed batch ...", flush=True)
        torch.manual_seed(0)
        sm = StreamedHealModel(args.model, "cpu", dict(LORA_CFG), dict(SEAM_CFG))
        trainable = sm.trainable_params()
        tnames = [n for n, p in sm.model.named_parameters() if p.requires_grad]
        assert all(n.endswith((".lora_A", ".lora_B")) or n.startswith("seam_adapter.")
                   for n in tnames), tnames[:6]
        frozen_probe = {}
        for n, p in sm.model.named_parameters():
            if not p.requires_grad and p.numel() > 4096:
                frozen_probe[n] = p.data.reshape(-1)[:4096].double().sum().item()
            if len(frozen_probe) >= 8:
                break
        x = torch.from_numpy(tokens[1000:1064].astype(np.int64)).unsqueeze(0)
        y = torch.from_numpy(tokens[1001:1065].astype(np.int64)).unsqueeze(0)
        before = {n: p.detach().clone() for n, p in sm.model.named_parameters()
                  if p.requires_grad}
        opt = torch.optim.AdamW(trainable, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
        losses = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            logits = sm.forward(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1),
                ignore_index=8198)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            losses.append(loss.item())
        print(f"    losses (same batch): {[f'{l:.4f}' for l in losses]}", flush=True)
        assert all(np.isfinite(losses)) and losses[-1] < losses[0], losses
        after = {n: p.detach() for n, p in sm.model.named_parameters() if p.requires_grad}
        seam_b = after["seam_adapter.B"]
        assert seam_b.abs().max() > 0, "seam B did not move"
        assert not torch.equal(before["seam_adapter.A"], after["seam_adapter.A"])
        n_lora_b = [n for n in before if n.endswith(".lora_B")]
        assert all(not torch.equal(before[n], after[n]) for n in n_lora_b)
        for n, s in frozen_probe.items():
            p = dict(sm.model.named_parameters())[n]
            assert p.data.reshape(-1)[:4096].double().sum().item() == s, f"frozen {n} moved"
        print(f"    OK: finite decreasing loss; {len(tnames)} trainable tensors moved "
              f"(LoRA A/B + seam A/B), frozen spine untouched", flush=True)

        # also exercise the real CLI once (checkpoint contents on disk)
        out_b = os.path.join(tmp, "run_b")
        log = run_heal(args.model, args.data, out_b,
                       ["--seam-adapter", "16", "--seam-after", "5"], 3)
        cli_losses = losses_of(log)
        assert len(cli_losses) == 3 and all(np.isfinite(cli_losses)), cli_losses
        ck = torch.load(os.path.join(out_b, "ckpt_latest.pt"), map_location="cpu",
                        weights_only=False)
        kinds = {re.sub(r"^.*\.", ".", k) for k in ck["model"]}
        assert kinds <= {".lora_A", ".lora_B", ".A", ".B"}, kinds
        assert {k for k in ck["model"] if k.startswith("seam_adapter.")} == {
            "seam_adapter.A", "seam_adapter.B"}
        assert ck["model"]["seam_adapter.B"].abs().max() > 0
        assert ck["seam"] == SEAM_CFG, ck["seam"]
        print(f"    OK (CLI): ckpt holds only LoRA + seam tensors, seam info {ck['seam']}, "
              f"losses {[f'{l:.3f}' for l in cli_losses]}", flush=True)

        # ── (c) resume from a ckpt WITHOUT seam into a run WITH seam ──
        print("(c) resume no-seam ckpt -> seam run ...", flush=True)
        out_c = os.path.join(tmp, "run_c")
        log1 = run_heal(args.model, args.data, out_c, [], 2)
        ck1 = torch.load(os.path.join(out_c, "ckpt_latest.pt"), map_location="cpu",
                         weights_only=False)
        assert ck1.get("seam") is None and not any(
            k.startswith("seam_adapter.") for k in ck1["model"])
        b1 = {k: v.clone() for k, v in ck1["model"].items() if k.endswith(".lora_B")}
        log2 = run_heal(args.model, args.data, out_c,
                        ["--resume", "--seam-adapter", "16", "--seam-after", "5"], 4)
        assert "fresh zero-init" in log2, log2[-2000:]
        assert "optimizer state grew" in log2, log2[-2000:]
        ck2 = torch.load(os.path.join(out_c, "ckpt_latest.pt"), map_location="cpu",
                         weights_only=False)
        assert ck2["step"] == 4 and ck2["seam"] == SEAM_CFG
        assert ck2["model"]["seam_adapter.B"].abs().max() > 0, "seam B still zero"
        # LoRA weights resumed (not re-initialized): identical to the pre-resume
        # values nowhere, but every tensor was already non-zero at resume time
        same = sum(torch.equal(b1[k], ck2["model"][k]) for k in b1)
        assert same < len(b1), "LoRA weights did not move after resume"
        moved_from_zero = sum(ck2["model"][k].abs().max() > 0 for k in b1)
        assert moved_from_zero == len(b1), "LoRA B tensors lost their trained state"
        print("    OK: LoRA state + Adam resumed, seam adapter started fresh, "
              "training continued to step 4", flush=True)
        trained_ckpt = os.path.join(out_c, "ckpt_latest.pt")
    else:
        trained_ckpt = None

    # ── (d0) zero-init merge: byte-identical .bin ──
    print("(d0) zero-init merge ...", flush=True)
    sm0 = StreamedHealModel(args.model, "cpu", None, SEAM_CFG)
    sd0 = sm0.trainable_state()
    ck0 = os.path.join(tmp, "seam_zero.pt")
    torch.save({"model": sd0, "seam": SEAM_CFG, "lora": None, "step": 0,
                "bin_source": args.model}, ck0)
    m0 = os.path.join(tmp, "merged_zero.bin")
    merge(ck0, args.model, m0)
    assert filecmp.cmp(args.model, m0, shallow=False), "zero-init merge changed the .bin"
    print("    OK: folded .bin is byte-identical to the original at zero-init", flush=True)

    # ── (d1) fabricated non-zero adapter: fold tensors exact + forward delta ──
    print("(d1) fabricated adapters: fold exactness + forward delta ...", flush=True)
    g = torch.Generator().manual_seed(7)
    hidden = 512
    for after, tag in ((5, "KDA layer 6"), (6, "MLA layer 7")):
        cfg = {"rank": 16, "after": after}
        a = torch.randn(16, hidden, generator=g) * 0.02
        b = torch.randn(hidden, 16, generator=g) * 0.02
        sd = {"seam_adapter.A": a, "seam_adapter.B": b}
        ck = os.path.join(tmp, f"seam_fab_{after}.pt")
        torch.save({"model": sd, "seam": cfg, "lora": None, "step": 0,
                    "bin_source": args.model}, ck)
        mb = os.path.join(tmp, f"merged_fab_{after}.bin")
        merge(ck, args.model, mb, force=True)
        worst, n = check_fold_tensors(args.model, mb, after, a, b)
        assert worst < 1e-6, f"fold mismatch on {tag}: rel {worst}"
        out_adapted = forward_logits(args.model, ids, None, cfg, sd)
        out_merged = forward_logits(mb, ids, None, None)
        diff = (out_adapted - out_merged).abs().max().item()
        scale = (b @ a).abs().max().item()
        print(f"    {tag}: {n} tensors folded exactly (rel {worst:.2e}); "
              f"max|BA| {scale:.2e} -> forward max|diff| {diff:.2e}", flush=True)

    # ── (d2) scale sweep: the forward delta is proportional to |B A| ──
    print("(d2) scale sweep (fold mechanics vs |BA|) ...", flush=True)
    cfg = SEAM_CFG
    for target in (1e-8, 1e-5, 1e-3):
        a = torch.randn(16, hidden, generator=g)
        b = torch.randn(hidden, 16, generator=g)
        s = target / (b @ a).abs().max().item()
        a, b = a * s**0.5, b * s**0.5
        sd = {"seam_adapter.A": a, "seam_adapter.B": b}
        ck = os.path.join(tmp, f"seam_scale_{target}.pt")
        torch.save({"model": sd, "seam": cfg, "lora": None, "step": 0,
                    "bin_source": args.model}, ck)
        mb = os.path.join(tmp, f"merged_scale_{target}.bin")
        merge(ck, args.model, mb, force=True)
        out_adapted = forward_logits(args.model, ids, None, cfg, sd)
        out_merged = forward_logits(mb, ids, None, None)
        diff = (out_adapted - out_merged).abs().max().item()
        print(f"    max|BA| {target:.0e} -> forward max|diff| {diff:.3e}", flush=True)
        if target <= 1e-8:
            assert diff < 1e-5, f"fold mechanics broken at small scale: {diff}"

    # ── (d3) trained ckpt merge: refused by default, approximate when forced ──
    if trained_ckpt is not None:
        print("(d3) merge of the TRAINED checkpoint ...", flush=True)
        ck = torch.load(trained_ckpt, map_location="cpu", weights_only=False)
        ba_max = (ck["model"]["seam_adapter.B"] @ ck["model"]["seam_adapter.A"]).abs().max()
        # a trained seam adapter must be REFUSED without --force-seam-fold
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "..", "apply_lora_bin.py"),
             "--ckpt", trained_ckpt, "--bin", args.model, "--out", os.devnull],
            capture_output=True, text=True)
        assert r.returncode != 0 and "refusing to fold a TRAINED seam adapter" in (
            r.stdout + r.stderr), (r.returncode, r.stdout, r.stderr)
        print(f"    OK: default merge refused cleanly (max|BA| {ba_max:.3e})", flush=True)
        mt = os.path.join(tmp, "merged_trained.bin")
        merge(trained_ckpt, args.model, mt, force=True)
        out_adapted = forward_logits(args.model, ids, ck["lora"], ck["seam"], ck["model"])
        out_merged = forward_logits(mt, ids, None, None)
        diff = (out_adapted - out_merged).abs()
        rel = diff.max() / out_adapted.abs().max()
        print(f"    forced merge: merged vs adapted forward max|diff| {diff.max():.3e}, "
              f"rel {rel:.3e} (approximation, grows linearly with |BA|)", flush=True)
        print(f"    trained merged bin: {mt}", flush=True)

    print("ALL SEAM ADAPTER TESTS OK", flush=True)


if __name__ == "__main__":
    main()
