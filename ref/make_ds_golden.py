#!/usr/bin/env python3
"""Generate ref/ds_golden.json: reference values for DeepSeek-V4 quantization
round-trips (fp8 e4m3 + ue8m0/128x128, fp4 e2m1 + ue8m0/32) computed with torch.

Test tool only. Run: /home/node/venv/bin/python3 ref/make_ds_golden.py
"""
import json
import torch

OUT = "/workspace/microkimi-oss/ref/ds_golden.json"
g = torch.Generator().manual_seed(20260731)

# ── fp8 e4m3 + ue8m0, blocks 128x128 (kernel.py: act/fp8 gemm semantics) ──
R, C = 256, 256
w = torch.randn(R, C, generator=g) * 3.0
qw = torch.empty(R, C, dtype=torch.uint8)
scales = torch.empty(R // 128, C // 128, dtype=torch.uint8)
for br in range(R // 128):
    for bc in range(C // 128):
        blk = w[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128]
        amax = blk.abs().max()
        e = max(-127, torch.ceil(torch.log2(amax / 448.0)).int().item())
        e = min(e, 8)
        scales[br, bc] = e + 127
        q = torch.clamp(blk / (2.0 ** e), -448.0, 448.0)
        qw[br * 128:(br + 1) * 128, bc * 128:(bc + 1) * 128] = q.to(torch.float8_e4m3fn).view(torch.uint8)
# NOTE: .view() reinterprets bytes as e4m3 bits; .to() would cast the integer VALUE.
deq = qw.view(torch.float8_e4m3fn).float() * (2.0 ** (scales.int() - 127)).repeat_interleave(128, 0).repeat_interleave(128, 1)

golden = {
    "fp8": {
        "rows": R, "cols": C,
        "w_packed": qw.flatten().tolist(),
        "scales": scales.flatten().tolist(),
        "dequant": deq.flatten().tolist(),
        "w_orig": w.flatten().tolist(),
    }
}

# ── fp4 e2m1 low-nibble-first + ue8m0/32 (same layout as MXFP4) ──
R2, C2 = 8, 64
w2 = torch.randn(R2, C2, generator=g) * 2.0
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
packed = torch.empty(R2, C2 // 2, dtype=torch.uint8)
scales2 = torch.empty(R2, C2 // 32, dtype=torch.uint8)
for r in range(R2):
    for gg in range(C2 // 32):
        grp = w2[r, gg * 32:(gg + 1) * 32]
        amax = grp.abs().max()
        e = max(-127, torch.ceil(torch.log2(amax / 6.0)).int().item()) if amax > 0 else -127
        scales2[r, gg] = e + 127
        q = torch.clamp(grp / (2.0 ** e), -6.0, 6.0)
        # nearest e2m1 level
        idx = (q.abs().unsqueeze(-1) - E2M1.abs()).abs().argmin(-1)
        idx = torch.where(q < 0, idx + 8, idx)
        packed[r, gg * 16:(gg + 1) * 16] = (idx[0::2] | (idx[1::2] << 4))
nib = torch.empty(R2, C2, dtype=torch.uint8)
nib[:, 0::2] = packed & 0x0F
nib[:, 1::2] = packed >> 4
deq2 = E2M1[nib.long()] * (2.0 ** (scales2.int() - 127)).repeat_interleave(32, 1)
golden["fp4"] = {
    "rows": R2, "cols": C2,
    "packed": packed.flatten().tolist(),
    "scales": scales2.flatten().tolist(),
    "dequant": deq2.flatten().tolist(),
}

with open(OUT, "w") as f:
    json.dump(golden, f)
print(f"written {OUT}")
