"""Pure-PyTorch KDA (Kimi Delta Attention) kernels, ported from fla-core 0.5.1:
- recurrence: fla/ops/kda/naive.py (naive_recurrent_kda)
- gate:       fla/ops/kda/gate.py  (naive_kda_lowerbound_gate / naive_kda_gate)
chunk_kda == fused_recurrent_kda mathematically (chunking is exact); both run the
same per-token recurrence here. Handles per-head A_log [H] and per-channel A_log [K]
(the K3 checkpoint ships [128] = per-channel; broadcast across heads).
State layout [B, H, K, V] internally; square (128x128) here, and the state only
round-trips through this module, so transpose_state_layout is a no-op for us."""
import os

import torch
import torch.nn.functional as F

# K3_KDA_RECUR: where the T<=4 decode recurrence runs. "device" keeps it on
# whichever backend owns q (MPS, CUDA, or CPU); "cpu" preserves the historical
# MPS-to-CPU hop. The old value "mps" remains a compatibility alias for
# "device", because it never moved CUDA or CPU tensors to an MPS device.
def normalize_recur_mode(value):
    mode = str(value).strip().lower()
    if mode == "mps":
        return "device"
    if mode not in ("device", "cpu"):
        raise ValueError("K3_KDA_RECUR must be device or cpu (mps is an alias)")
    return mode


RECUR = normalize_recur_mode(os.environ.get("K3_KDA_RECUR", "device"))


def _kda_gate(g, A_log, dt_bias, lower_bound):
    # g: [B, T, H, K] raw; dt_bias: [H*K]; A_log: [H] or [K]
    B, T, H, K = g.shape
    g = g.to(torch.float32)
    if dt_bias is not None:
        g = g + dt_bias.to(torch.float32).view(H, K)
    if A_log.numel() == H:
        a = A_log.to(torch.float32).exp().view(H, 1)
    elif A_log.numel() == K:
        a = A_log.to(torch.float32).exp().view(1, K)
    else:
        raise ValueError(f"A_log shape {tuple(A_log.shape)} not [H]/[K]")
    if lower_bound is not None:
        # gate.py: g = lower_bound * sigmoid(exp(A_log) * g)
        return lower_bound * torch.sigmoid(a * g)
    # gate.py: g = -exp(A_log) * softplus(g)
    return -a * F.softplus(g)


def _kda_core(q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
              use_qk_l2norm_in_kernel, use_gate_in_kernel,
              use_beta_sigmoid_in_kernel, lower_bound):
    # shapes: q,k [B,T,H,K]; v [B,T,H,V]; g [B,T,H,K]; beta [B,T,H]
    B, T, H, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)
    beta = beta.to(torch.float32)
    if use_qk_l2norm_in_kernel:
        q = F.normalize(q, p=2.0, dim=-1)
        k = F.normalize(k, p=2.0, dim=-1)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta)
    if use_gate_in_kernel:
        g = _kda_gate(g, A_log, dt_bias, lower_bound)
    else:
        g = g.to(torch.float32)
    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.to(torch.float32)
    q = q * scale
    o = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)
    for t in range(T):
        q_t, k_t, v_t, g_t, b_t = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        S = S * g_t.exp().unsqueeze(-1)                                   # decay along K
        delta = v_t - (k_t.unsqueeze(-1) * S).sum(-2)                     # [B,H,V]
        S = S + torch.einsum('bhk,bhv->bhkv', b_t.unsqueeze(-1) * k_t, delta)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', q_t, S)
    return o, S


def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
                        initial_state=None, output_final_state=False,
                        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                        use_beta_sigmoid_in_kernel=False, lower_bound=None,
                        cu_seqlens=None, transpose_state_layout=False, **kw):
    assert cu_seqlens is None, "shim: cu_seqlens unsupported (batch=1 only)"
    # Decode (T small): the optional historical CPU mode avoids ~25 MPS op
    # launches by moving 240KB in / 48KB out per layer and retaining the
    # recurrent state on CPU. It applies only when q is actually on MPS.
    #
    # K3_KDA_RECUR=device keeps the operation on q.device. Re-measured on MPS at
    # T=1 with the rest of the pipeline as it stands today (median of 15, each
    # variant serialized with torch.mps.synchronize):
    #     CPU hop, state CPU-resident   3.12 ms      <- this branch
    #     all-MPS                       1.17 ms
    #     CPU math alone, no transfers  1.29 ms
    # i.e. the hop itself is ~1.9 ms/layer = 135 ms/token over 69 KDA layers,
    # and the D2H is also a hard GPU barrier that drains the queue (the layer's
    # int8 spine dequant included) 69 times per token.
    dev = q.device
    if dev.type == "mps" and q.shape[1] <= 4 and RECUR == "cpu":
        cpu = torch.device("cpu")
        o, S = _kda_core(q.to(cpu), k.to(cpu), v.to(cpu), g.to(cpu), beta.to(cpu),
                         None if A_log is None else A_log.to(cpu),
                         None if dt_bias is None else dt_bias.to(cpu),
                         scale,
                         None if initial_state is None else initial_state.to(cpu),
                         use_qk_l2norm_in_kernel, use_gate_in_kernel,
                         use_beta_sigmoid_in_kernel, lower_bound)
        return o.to(dev, v.dtype), (S if output_final_state else None)
    o, S = _kda_core(q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
                     use_qk_l2norm_in_kernel, use_gate_in_kernel,
                     use_beta_sigmoid_in_kernel, lower_bound)
    return o.to(v.dtype), (S if output_final_state else None)


def chunk_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None,
              initial_state=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, safe_gate=False, lower_bound=None,
              cu_seqlens=None, transpose_state_layout=False, **kw):
    assert cu_seqlens is None, "shim: cu_seqlens unsupported (batch=1 only)"
    if not safe_gate:
        lower_bound = None
    o, S = _kda_core(q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
                     use_qk_l2norm_in_kernel, use_gate_in_kernel,
                     use_beta_sigmoid_in_kernel, lower_bound)
    return o.to(v.dtype), (S if output_final_state else None)
