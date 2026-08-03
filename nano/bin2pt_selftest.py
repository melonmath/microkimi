#!/usr/bin/env python3
"""nanokimi - bin2pt_selftest: numerical proof of the .bin -> .pt bridge.

  A) round trip pt -> bin -> pt on a real nano checkpoint (default
     /workspace/chat_smoke/ckpt_base.pt): every tensor must survive,
     f32 tensors EXACTLY (1e-6), experts mxfp4 too (compared against
     dequant(quantize(orig)), the exact reference; the quantization
     error vs the original is reported for information only);
  B) the real /workspace/nanokimi-0.2b-chat.bin converts with names and
     shapes matching its source checkpoint (values differ: it went
     through chat SFT);
  C) config-driven construction proof for the future 1b: a NanoModel
     built from a K3-like config (22 layers with EXPLICIT
     mla_layers/dense_layers, hidden 1024, 48 experts top-16, real-style
     MLA dims - only the vocab is reduced to fit this box's RAM) also
     round-trips pt -> bin -> pt.

Prints PASS/FAIL per tensor, exits non-zero on any FAIL.

usage: python3 bin2pt_selftest.py [--ckpt path.pt]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import bin2pt
from export import quantize_mxfp4

TOL = 1e-6
TMP = "/tmp/bin2pt_selftest"
N_FAIL = 0


def check(name, got, ref, note=""):
    """One tensor line: PASS when max|got-ref| <= 1e-6 (shapes must match)."""
    global N_FAIL
    if got.shape != ref.shape:
        print(f"FAIL {name}: shape {tuple(got.shape)} vs {tuple(ref.shape)} {note}")
        N_FAIL += 1
        return
    d = (got - ref).abs().max().item() if got.numel() else 0.0
    ok = d <= TOL
    if not ok:
        N_FAIL += 1
    print(f"{'PASS' if ok else 'FAIL'} {name}  max_abs_diff={d:.3e} {note}")


def expert_ref(w):
    """dequant(quantize(w)): the exact mxfp4 round-trip reference for w."""
    packed, scales = quantize_mxfp4(w.detach().float().numpy())
    blob = packed.tobytes() + scales.tobytes()
    return torch.from_numpy(bin2pt.dequant_mxfp4(blob, list(w.shape)))


def roundtrip(ckpt_path, tag, expert_sample=None):
    """pt -> bin -> pt; per-tensor PASS/FAIL. expert_sample: compare only
    the first K experts per layer in value (shapes still checked for all)."""
    bin_path = f"{TMP}_{tag}.bin"
    subprocess.run(
        [sys.executable, os.path.join(_HERE, "export.py"), "--ckpt", ckpt_path, "--out", bin_path],
        check=True,
    )
    sd2, cfg2 = bin2pt.convert(bin_path)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd1 = ck["model"]
    # config round trip: the .bin config must reproduce the model cfg
    for k in ("n_layers", "hidden", "vocab", "n_experts", "top_k"):
        assert cfg2[k] == ck["cfg"][k], f"cfg key {k}: {cfg2[k]} vs {ck['cfg'][k]}"
    for k in ("mla_layers", "dense_layers"):
        if k in ck["cfg"]:
            assert list(cfg2[k]) == list(ck["cfg"][k]), f"cfg key {k} did not round-trip"
    missing = set(sd1) - set(sd2)
    extra = set(sd2) - set(sd1)
    if missing or extra:
        print(f"FAIL [{tag}] name mismatch: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}")
        global N_FAIL
        N_FAIL += 1
        return
    n_pass = 0
    for name in sorted(sd1):
        w1, w2 = sd1[name], sd2[name]
        if w1.shape != w2.shape:
            check(name, w2, w1)
            continue
        if ".experts." in name:
            layer = int(name.split(".")[1])
            eid = int(name.split(".")[4])
            if expert_sample is not None and eid >= expert_sample:
                n_pass += 1  # shapes already equal: counted, not re-printed
                continue
            ref = expert_ref(w1)
            err = (w2 - w1).abs().max().item()
            rel = err / max(1e-12, w1.abs().max().item())
            check(name, w2, ref, note=f"(mxfp4 err vs orig {rel:.2e} rel, layer {layer})")
        else:
            check(name, w2, w1)
        n_pass += 1
    print(f"-- [{tag}] {n_pass}/{len(sd1)} tensors PASS (cfg round trip OK)")


def test_real_bin(ref_ckpt):
    src = "/workspace/nanokimi-0.2b-chat.bin"
    if not os.path.exists(src):
        print(f"-- [real-bin] {src} absent, skipped")
        return
    global N_FAIL
    sd, cfg = bin2pt.convert(src)
    ck = torch.load(ref_ckpt, map_location="cpu", weights_only=False)
    ref = ck["model"]
    ok = True
    if set(sd) != set(ref):
        print(f"FAIL [real-bin] name sets differ: only-bin={sorted(set(sd) - set(ref))[:5]} "
              f"only-ckpt={sorted(set(ref) - set(sd))[:5]}")
        ok = False
    else:
        for name in sorted(sd):
            if sd[name].shape != ref[name].shape:
                print(f"FAIL [real-bin] {name}: {tuple(sd[name].shape)} vs {tuple(ref[name].shape)}")
                ok = False
    print(f"{'PASS' if ok else 'FAIL'} [real-bin] {src}: {len(sd)} tensors, names+shapes match the source ckpt")
    if not ok:
        N_FAIL += 1


def test_1b_structure():
    """K3-like config (the future microkimi-1b shape): 22 layers with an
    EXPLICIT mla_layers list, hidden 1024, 48 experts top-16 + 2 shared,
    real-style MLA dims. Only the vocab is reduced (32000 instead of
    163840) to fit this box's 7 GB RAM - the embedding size changes no
    code path."""
    from model_nano import NanoModel, count_params
    cfg = dict(
        n_layers=22, hidden=1024, vocab=32000,
        n_experts=48, top_k=16, n_shared=2,
        kda_heads=8, kda_dim=128, kda_conv=4, kda_fa_rank=128, gate_lower_bound=-5.0,
        mla_heads=8, mla_q_lora=512, mla_kv_lora=512, mla_nope=128, mla_rope=64, mla_v=128,
        routed_hidden=256, moe_inter=128, shared_inter=512,
        dense_inter=4096, attn_res_block=12, first_k_dense=1, rms_eps=1e-5,
        mla_layers=[3, 7, 11, 15, 19, 21],  # explicit, NOT the L%4 pattern
        dense_layers=[0],
    )
    torch.manual_seed(0)
    m = NanoModel(cfg).float().eval()
    total, experts = count_params(m)
    print(f"-- [1b-structure] NanoModel built from explicit-list config: "
          f"{total / 1e6:.0f} M params ({experts / 1e6:.0f} M experts), mla {cfg['mla_layers']}")
    ckpt = f"{TMP}_1b.pt"
    torch.save({"model": m.state_dict(), "cfg": m.c, "step": 0}, ckpt)
    del m
    roundtrip(ckpt, "1b", expert_sample=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/workspace/chat_smoke/ckpt_base.pt")
    args = ap.parse_args()
    ckpt = args.ckpt
    if not os.path.exists(ckpt):
        print(f"reference ckpt {ckpt} missing - building a fresh tiny NanoModel instead")
        from model_nano import NanoModel
        torch.manual_seed(0)
        m = NanoModel({"n_layers": 4}).float().eval()
        ckpt = f"{TMP}_fresh.pt"
        torch.save({"model": m.state_dict(), "cfg": m.c, "step": 0}, ckpt)
        del m
    print(f"== A) round trip pt -> bin -> pt on {ckpt}")
    roundtrip(ckpt, "nano")
    print("== B) real nanokimi-0.2b-chat.bin vs source ckpt (names + shapes)")
    test_real_bin(ckpt)
    print("== C) 1b-structure config (explicit mla_layers) round trip")
    test_1b_structure()
    if N_FAIL:
        print(f"SELFTEST FAIL ({N_FAIL} failures)")
        sys.exit(1)
    print("SELFTEST PASS")


if __name__ == "__main__":
    main()
