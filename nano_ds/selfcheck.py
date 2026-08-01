#!/usr/bin/env python3
"""nanodeepseek - selfcheck: proves the BATCHED training forward
(model_ds.py) computes the same math as the SEQUENTIAL reference replica
(ref/make_ds_parity.py, itself validated 1:1 against the Rust engine by
dsparity).

Flow: build a NanoDsModel (seeded), export its weights to a temp MKIM0002
bin (tests export.py), fp4-round-trip the expert stacks inside the model
(so its experts equal the bin's exactly), run the sequential replica on the
bin, and compare logits (last position + top-16 ids per position).
Tolerance: QAT-aware (2e-3 + 1e-3 * scale) - fp8/fp4 activation round-trip
boundaries amplify f32 summation-order noise, exactly like dsparity.

Run: /home/node/venv/bin/python3 nano_ds/selfcheck.py
"""
import json
import os
import sys
import tempfile

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "ref"))

import export as ds_export
import make_ds_parity as mdp
from model_ds import NanoDsModel

CFG = {"n_layers": 4, "compress_ratios": [0, 0, 4, 128]}


def fp4_roundtrip_experts(sd, c):
    """Returns a state_dict whose expert stacks are fp4-quantize→dequantize
    round-tripped (== the values the exported bin contains)."""
    sd = {k: v.clone() for k, v in sd.items()}
    for l in range(c["n_layers"]):
        for w in ("w1", "w2", "w3"):
            key = f"layers.{l}.moe.{w}"
            stack = sd[key].detach().float().cpu().numpy()
            out = np.empty_like(stack)
            for e in range(stack.shape[0]):
                p, s = ds_export.quantize_mxfp4(stack[e])
                out[e] = ds_export.dequant_mxfp4(p, s)
            sd[key] = torch.from_numpy(out)
    return sd


def main():
    torch.manual_seed(20260801)
    model = NanoDsModel(CFG).float().eval()
    c = model.c
    sd = model.state_dict()

    with tempfile.TemporaryDirectory() as tmp:
        bin_path = os.path.join(tmp, "selfcheck.bin")
        ds_export.export_model(sd, c, bin_path, verbose=False)
        # model with bin-identical (fp4-round-tripped) experts
        model.load_state_dict(fp4_roundtrip_experts(sd, c))
        model.eval()

        mdp.BIN = bin_path
        golden_path = os.path.join(tmp, "golden.json")
        mdp.OUT = golden_path
        os.environ["DS_PARITY_HI"] = "8000"  # ids must fit the nano vocab (8200)
        mdp.main()
        golden = json.load(open(golden_path))

    ids = golden["ids"]
    T = len(ids)
    with torch.no_grad():
        logits = model(torch.tensor(ids, dtype=torch.long).unsqueeze(0))[0]  # [T, V]

    scale = max(abs(min(golden["logits_last"])), abs(max(golden["logits_last"])))
    last = logits[-1].tolist()
    diffs = [abs(a - b) for a, b in zip(last, golden["logits_last"])]
    max_abs = max(diffs)
    bad = sum(1 for d in diffs if d > 2e-3 + 1e-3 * scale)
    print(f"logits_last: max_abs={max_abs:.3e} scale={scale:.3e} bad={bad} (tol 2e-3+1e-3·scale)")

    ids_ok = True
    for pos, g in enumerate(golden["logits_top"]):
        mine = logits[pos].topk(16).indices.tolist()
        if sorted(mine) != sorted(g["ids"]):
            ids_ok = False
            print(f"  pos {pos}: top-16 DIFFER\n    torch  {sorted(mine)}\n    golden {sorted(g['ids'])}")
    print(f"top-16 ids per position ({T} pos): {'exact' if ids_ok else 'DIFFER'}")

    if bad == 0 and ids_ok:
        print("SELFCHECK OK - batched training forward ≡ sequential replica (QAT-aware)")
    else:
        print("SELFCHECK FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
