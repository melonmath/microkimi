import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# K3_SHORTCONV=mulsum swaps the tiny-width depthwise conv for explicit
# sliding-window multiply+sum. It is the established one-token decode path on
# every supported backend and the measured T<=9 CPU path; "auto" retains
# conv1d elsewhere as a comparison/fallback until a backend gate passes.
SHORTCONV = os.environ.get("K3_SHORTCONV", "auto").strip().lower()
_SHORTCONV_AUTO_MAX_T = os.environ.get("K3_SHORTCONV_AUTO_MAX_T")


def resolve_shortconv_mode(device, requested=None):
    """Resolve auto against the tensor's real backend; overrides stay forceful."""
    mode = SHORTCONV if requested is None else str(requested).strip().lower()
    device_type = getattr(device, "type", str(device)).lower()
    if mode == "auto":
        return (
            "mulsum"
            if device_type in ("mps", "cuda", "cpu")
            else "conv1d"
        )
    if mode not in ("mulsum", "conv1d"):
        raise ValueError("K3_SHORTCONV must be auto, mulsum, or conv1d")
    return mode


def shortconv_auto_max_t(device):
    """Return the evidence-bounded automatic sliding-window threshold."""
    if _SHORTCONV_AUTO_MAX_T is not None:
        return max(1, int(_SHORTCONV_AUTO_MAX_T))
    device_type = getattr(device, "type", str(device)).lower()
    # The real K3 CPU shape wins by hundreds of times through T=9. Accelerator
    # T>1 crossovers are close and backend/runtime dependent, so preserve the
    # established T=1 fast path until their active-path gate passes.
    return 9 if device_type == "cpu" else 1


class ShortConvolution(nn.Conv1d):
    """Depthwise causal conv1d with rolling [N, D, W] cache of the last W raw inputs.
    Semantics ported from fla/modules/conv/short_conv.py: step = roll cache left,
    insert x_t, y = act(sum(cache * weight)); prefill = causal conv with cache[..., 1:]
    as left history; final state = last W raw inputs."""

    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu', **kw):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size,
                         bias=bias, padding=kernel_size - 1)
        self.hidden_size = hidden_size
        self.act = activation

    def forward(self, x, residual=None, mask=None, cache=None,
                output_final_state=False, cu_seqlens=None, **kw):
        # x: [B, T, D]
        B, T, D = x.shape
        W = self.kernel_size[0]
        mode = resolve_shortconv_mode(x.device)
        use_mulsum = (
            mode == "mulsum"
            and cache is not None
            and (SHORTCONV != "auto" or T <= shortconv_auto_max_t(x.device))
        )
        if use_mulsum:
            # Decode and speculative verification.  The causal source is the
            # last W-1 cached inputs followed by the T new inputs.  Unfolding
            # that source produces exactly the same W-wide windows as T
            # consecutive one-token calls, while retaining an explicit
            # last-dimension reduction. Backends may still vary that reduction
            # by a few ulps with outer shape, so the cache is exact while output
            # is sequence-gated rather than advertised as byte-identical.
            #
            # 12,288 depthwise groups over width four are a grouped convolution
            # only in name; the portable tensor path also avoids an especially
            # poor CPU kernel choice.
            xt = x.transpose(1, 2).to(torch.float32)       # [B,D,T]
            source = torch.cat(
                [cache.to(torch.float32)[:, :, 1:], xt], dim=-1
            )                                             # [B,D,W-1+T]
            windows = source.unfold(-1, W, 1)             # [B,D,T,W]
            weight = self.weight.view(1, D, 1, W).to(torch.float32)
            y = (windows * weight).sum(-1)                # [B,D,T]
            if self.act in ('silu', 'swish'):
                y = F.silu(y)
            y = y.transpose(1, 2).to(x.dtype)             # [B,T,D]
            if residual is not None:
                y = y + residual
            new_cache = None
            if output_final_state:
                if torch.is_grad_enabled():
                    # Preserve the historical autograd contract: callers may
                    # mutate the returned cache before backward without
                    # modifying ``source``, which this branch saves for the
                    # output gradient. Production inference is functional and
                    # can safely retain the source-tail alias below.
                    full = torch.cat([cache.to(torch.float32), xt], dim=-1)
                    new_cache = full[:, :, -W:].contiguous()
                else:
                    # ``source`` is already ``cache[..., 1:] + xt``. For every
                    # non-empty call its last W positions are therefore exactly
                    # the last W positions of ``cache + xt``. Reuse it instead
                    # of allocating and copying the same values a second time.
                    new_cache = source[:, :, -W:].contiguous()
            return y, new_cache
        w = self.weight.view(D, W).to(torch.float32)  # [D, W]
        xt = x.transpose(1, 2).to(torch.float32)      # [B, D, T]
        if cache is None:
            hist = xt.new_zeros(B, D, W - 1)
        else:
            hist = cache.to(torch.float32)[:, :, 1:]  # last W-1 raw inputs
        z = torch.cat([hist, xt], dim=-1)             # [B, D, W-1+T]
        y = F.conv1d(z, w.unsqueeze(1), groups=D)     # causal: [B, D, T]
        if self.act in ('silu', 'swish'):
            y = F.silu(y)
        y = y.transpose(1, 2).to(x.dtype)
        if residual is not None:
            y = y + residual
        new_cache = None
        if output_final_state:
            if torch.is_grad_enabled():
                # Keep returned training state independent from ``z`` because
                # conv1d backward retains ``z`` and checks its version.
                full = torch.cat(
                    [cache.to(torch.float32) if cache is not None
                     else xt.new_zeros(B, D, W), xt],
                    dim=-1,
                )
                new_cache = full[:, :, -W:].contiguous()
            else:
                # ``z`` and the historical ``full`` construction have
                # identical W-wide tails for every T >= 1, including an empty
                # initial cache.
                new_cache = z[:, :, -W:].contiguous()
        return y, new_cache


class FusedRMSNormGated(nn.Module):
    """out = RMSNorm(x) * weight * sigmoid(g)  (fp32 internal), per fla fused_norm_gate.py."""

    def __init__(self, hidden_size, eps=1e-5, activation='sigmoid', **kw):
        super().__init__()
        assert activation == 'sigmoid'
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x, g):
        xf, gf = x.to(torch.float32), g.to(torch.float32)
        var = xf.pow(2).mean(-1, keepdim=True)
        y = xf * torch.rsqrt(var + self.variance_epsilon) * self.weight.to(torch.float32)
        return (y * torch.sigmoid(gf)).to(x.dtype)
