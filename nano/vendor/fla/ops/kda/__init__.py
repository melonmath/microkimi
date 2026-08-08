"""Pure-PyTorch KDA (Kimi Delta Attention) kernels, ported from fla-core 0.5.1:
- recurrence: fla/ops/kda/naive.py (naive_recurrent_kda)
- gate:       fla/ops/kda/gate.py  (naive_kda_lowerbound_gate / naive_kda_gate)
chunk_kda == fused_recurrent_kda mathematically (chunking is exact); both run the
same per-token recurrence here, unless NANO_KDA_CHUNKED=1 switches chunk_kda to
the chunkwise UT-transform form (_kda_recur_chunked, training-only speed path,
not bit-identical - see below). Handles per-head A_log [H] and per-channel A_log [K]
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

# NANO_KDA_SEG: time-segment length for gradient-checkpointing the training
# recurrence (0 = off). The recurrence itself is UNCHANGED - the exact same
# per-token loop runs, only wrapped in torch.utils.checkpoint every SEG tokens
# so autograd retains O(SEG) per-token states instead of O(T) (recompute is
# deterministic, so results are bit-identical). Enabled only on the device
# types listed in NANO_KDA_SEG_DEVICES (default "cuda": the CPU/MPS paths are
# byte-for-byte the old code).
KDA_SEG = int(os.environ.get("NANO_KDA_SEG", "64"))
KDA_SEG_DEVICES = tuple(
    d for d in os.environ.get("NANO_KDA_SEG_DEVICES", "cuda").split(",") if d
)

# NANO_KDA_CHUNKED: opt-in chunkwise (UT-transform) form of the delta rule for
# TRAINING (chunk_kda only; the decode recurrence is untouched). Same
# mathematics as the per-token loop, blocked into NANO_KDA_CHUNK-token chunks
# so the work runs as a handful of batched matmuls per chunk instead of one
# kernel launch per token per layer, and autograd flows through the same
# chunked graph (chunked backward). Pure PyTorch: no Triton, no new
# dependency, runs on CPU and CUDA alike. NOT bit-identical to the reference
# loop (decays are applied as exp(cumsum) per chunk instead of per-token
# products, and the triangular inverse uses a different operation order) -
# the deviation is float-noise level (measured <= ~1e-5 relative on outputs
# and gradients, see nano/test_kda_chunked.py), acceptable for training but
# never use it where parity with the Rust engine is required. Default OFF.
# If the chunked path ever raises OR returns a non-finite output, it disables
# itself and falls back to the reference recurrence for the rest of the
# process.
KDA_CHUNKED = os.environ.get("NANO_KDA_CHUNKED", "0") == "1"
KDA_CHUNK = int(os.environ.get("NANO_KDA_CHUNK", "64"))
_chunked_available = True


def _kda_recur(q, k, v, g, beta, S):
    """Per-token delta-rule recurrence over a (slice of) time steps.
    q,k,g: [B,t,H,K]; v: [B,t,H,V]; beta: [B,t,H]; S: [B,H,K,V] -> (o, S)."""
    B, t, H, K = q.shape
    V = v.shape[-1]
    o = torch.empty(B, t, H, V, dtype=torch.float32, device=q.device)
    for i in range(t):
        q_t, k_t, v_t, g_t, b_t = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_t.exp().unsqueeze(-1)                                   # decay along K
        delta = v_t - (k_t.unsqueeze(-1) * S).sum(-2)                     # [B,H,V]
        # NOTE: einsum replaced by broadcast+sum — identical math, but these
        # two forms only need mul/sum, which are universally supported on the
        # MPS backend (einsum is the riskiest op there).
        S = S + (b_t.unsqueeze(-1) * k_t).unsqueeze(-1) * delta.unsqueeze(-2)
        o[:, i] = (q_t.unsqueeze(-1) * S).sum(-2)
    return o, S


def _kda_recur_chunked(q, k, v, g, beta, S, chunk_size):
    """Chunkwise (UT-transform) form of _kda_recur - same math, blocked over
    time so the cost is O(T/C) + O(C) groups of batched ops instead of T
    Python steps of small kernels, with autograd flowing through the chunked
    graph.

    Within a chunk of C tokens, write G for the inclusive in-chunk cumsum of
    the log-decay g and E[i,j] = exp(G_i - G_j) (kept for j <= i only, and
    masked BEFORE the exp so nothing ever overflows - every pairwise decay is
    <= 0, the classic cumsum trap). Then:
        L[i,j] = beta_i * (k_i . k_j) * E[i,j]         (strictly lower)
        A      = (I + L)^-1 @ diag(beta)
        w      = A @ (exp(G) * k),   u = A @ v
    and per chunk, given the incoming state S [B,H,K,V]:
        v'  = u - w @ S
        o   = (q * exp(G)) @ S + ((q k^T) . E) @ v'
        S  <- S * exp(G_last) + (exp(G_last - G) * k)^T @ v'
    The triangular inverse MUST be forward substitution (backward stable: the
    entries of (I + L)^-1 stay O(1)). A closed-form nilpotent factorization
    (I - L)(I + L^2)(I - L^4)... is exact on paper but catastrophically
    unstable in fp32 in the real training regime (gates near 0, correlated
    keys, beta near 1 make L approach the strictly-lower all-ones matrix, the
    intermediate powers reach ~1e17 and the cancellation NaNs out - the
    original bug). The substitution runs once, batched over B, H and all
    chunks; the big [C, C, K] decay tensor never outlives its own chunk.
    T is zero-padded to a multiple of C; padded rows have k = v = beta = 0 and
    raw g = 0, so they neither update the state nor the valid outputs (their
    own outputs are sliced away).
    q,k,g: [B,T,H,K]; v: [B,T,H,V]; beta: [B,T,H]; S: [B,H,K,V] -> (o, S)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = (T + C - 1) // C
    pad = NT * C - T

    def chunked(x):  # [B,T,H,*] -> [B,H,NT,C,*], zero-padded on T
        if pad:
            shape = (B, pad) + tuple(x.shape[2:])
            x = torch.cat([x, x.new_zeros(shape)], dim=1)
        return x.reshape(B, NT, C, H, *x.shape[3:]).permute(
            0, 3, 1, 2, *range(4, x.dim() + 1))

    q, k, v, g, beta = (chunked(x) for x in (q, k, v, g, beta))
    gc = g.cumsum(dim=-2)  # inclusive in-chunk cumulative log-decay [B,H,NT,C,K]
    upper = torch.triu(torch.ones(C, C, dtype=torch.bool, device=q.device), 1)
    eye = torch.eye(C, dtype=torch.float32, device=q.device)
    # pass 1, state-independent: L (coupling) and QK (intra-chunk readout)
    Ls, QKs = [], []
    for n in range(NT):
        qn, kn, gn, bn = q[:, :, n], k[:, :, n], gc[:, :, n], beta[:, :, n]
        diff = gn.unsqueeze(3) - gn.unsqueeze(2)  # [B,H,C(i),C(j),K]
        E = diff.masked_fill(upper.unsqueeze(-1), float("-inf")).exp()
        KK = torch.einsum("bhik,bhjk,bhijk->bhij", kn, kn, E)
        Ls.append((KK * bn.unsqueeze(-1)).tril(-1))
        QKs.append(torch.einsum("bhik,bhjk,bhijk->bhij", qn, kn, E))
    L = torch.stack(Ls, dim=2)    # [B,H,NT,C,C]
    QK = torch.stack(QKs, dim=2)  # [B,H,NT,C,C]
    # (I + L)^-1 by forward substitution, batched over B, H, NT (the clones
    # keep every value saved for backward private to its op, so the in-place
    # row writes never corrupt the graph)
    A = -L
    for i in range(1, C):
        A[..., i, :i] = A[..., i, :i].clone() + (
            A[..., i, :, None].clone() * A[..., :, :i].clone()
        ).sum(-2)
    A = (A + eye) * beta.unsqueeze(-2)  # (I + L)^-1 @ diag(beta)
    # pass 2, stateful scan over the chunks
    o = q.new_empty(B, H, NT, C, V)
    for n in range(NT):
        qn, kn, vn, gn = q[:, :, n], k[:, :, n], v[:, :, n], gc[:, :, n]
        An = A[:, :, n]
        w = An @ (gn.exp() * kn)
        u = An @ vn
        v2 = u - w @ S
        o[:, :, n] = (qn * gn.exp()) @ S + QK[:, :, n] @ v2
        gl = gn[:, :, -1]  # total in-chunk log-decay [B,H,K]
        kd = (gl.unsqueeze(2) - gn).exp() * kn  # [B,H,C,K]
        S = S * gl.exp().unsqueeze(-1) + kd.transpose(-1, -2) @ v2
    o = o.permute(0, 2, 3, 1, 4).reshape(B, NT * C, H, V)
    return (o[:, :T] if pad else o), S


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
              use_beta_sigmoid_in_kernel, lower_bound, chunked=False):
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
    if chunked:
        # NANO_KDA_CHUNKED=1: chunkwise form, no per-token Python loop. The
        # per-token intermediate states are never materialized, so the SEG
        # checkpointing above is unnecessary on this path.
        o, S = _kda_recur_chunked(q, k, v, g, beta, S, KDA_CHUNK)
    elif KDA_SEG > 0 and T > KDA_SEG and q.device.type in KDA_SEG_DEVICES:
        # training on GPU: checkpoint the recurrence every KDA_SEG tokens.
        # Identical math (same loop, deterministic recompute); autograd only
        # retains the segment-boundary states plus one segment's internals
        # instead of ~4 state-sized tensors per token per layer.
        import torch.utils.checkpoint as _ckpt
        o = torch.empty(B, T, H, V, dtype=torch.float32, device=q.device)
        for t0 in range(0, T, KDA_SEG):
            t1 = min(t0 + KDA_SEG, T)
            o_seg, S = _ckpt.checkpoint(
                _kda_recur,
                q[:, t0:t1], k[:, t0:t1], v[:, t0:t1], g[:, t0:t1], beta[:, t0:t1], S,
                use_reentrant=False,
            )
            o[:, t0:t1] = o_seg
    else:
        o, S = _kda_recur(q, k, v, g, beta, S)
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
    global _chunked_available
    if KDA_CHUNKED and _chunked_available:
        try:
            o, S = _kda_core(q, k, v, g, beta, A_log, dt_bias, scale,
                             initial_state, use_qk_l2norm_in_kernel,
                             use_gate_in_kernel, use_beta_sigmoid_in_kernel,
                             lower_bound, chunked=True)
            # output guard: a NaN/Inf anywhere in the chunked result must
            # never reach the training step - disable the path and redo the
            # call with the reference recurrence
            if not (torch.isfinite(o).all() and torch.isfinite(S).all()):
                raise FloatingPointError("non-finite chunked output")
        except Exception as exc:  # never break a run: fall back for good
            _chunked_available = False
            import sys as _sys
            print(f"[kda] NANO_KDA_CHUNKED path disabled after {exc!r}; "
                  "falling back to the reference recurrence", file=_sys.stderr)
            o, S = _kda_core(q, k, v, g, beta, A_log, dt_bias, scale,
                             initial_state, use_qk_l2norm_in_kernel,
                             use_gate_in_kernel, use_beta_sigmoid_in_kernel,
                             lower_bound)
    else:
        o, S = _kda_core(q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
                         use_qk_l2norm_in_kernel, use_gate_in_kernel,
                         use_beta_sigmoid_in_kernel, lower_bound)
    return o.to(v.dtype), (S if output_final_state else None)
