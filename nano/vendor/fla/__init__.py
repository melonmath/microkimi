"""Pure-PyTorch shim for fla-core, implementing exactly the names modeling_kimi_linear.py
imports, with semantics ported from fla-core 0.5.1 source (fla/ops/kda/naive.py,
fla/ops/kda/gate.py, fla/modules/conv/short_conv.py, fla/modules/fused_norm_gate.py).
Correctness-first.

Ported from flash-linear-attention (fla-core):
    https://github.com/fla-org/flash-linear-attention
    Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
    MIT License (see the LICENSE file's third-party notices)
"""
from . import modules, ops, utils  # noqa: F401
