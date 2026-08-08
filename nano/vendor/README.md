# nano/vendor - third-party code, vendored vs downloaded

## Vendored here: `fla/` (pure-PyTorch shim of fla-core)

Semantics ported from **flash-linear-attention (fla-core)**,
<https://github.com/fla-org/flash-linear-attention>
Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li - **MIT License**.

The shim is a pure-PyTorch port of fla-core semantics (MIT) that implements
exactly the names Moonshot's `modeling_kimi_linear.py` imports
(`ShortConvolution`, `FusedRMSNormGated`, `chunk_kda`, `fused_recurrent_kda`,
index helpers, `tensor_cache`), correctness-first. Each file carries the
upstream copyright notice.

## Vendored here: `transformers/` and `einops.py` (compatibility shims)

The downloaded Moonshot files import a handful of names from the `transformers`
and `einops` packages at module level. These shims provide exactly that surface
(config base class, activation dict, typing helpers, no-op decorators, and the
three `rearrange` patterns the model code uses) so the training stack runs on
hosts with only torch installed. When the real packages ARE installed they are
loaded and re-exported instead, so a provisioned host sees zero behavior change
(`NANO_VENDOR_FORCE_SHIM=1` forces the shim, used to validate it: the
heal_stream --selftest logits fingerprint is identical in both modes).

## Downloaded at runtime, never vendored: `moonshot/`

Moonshot AI's reference files (`modeling_kimi_linear.py`,
`configuration_kimi_k3.py`) are **not** part of this repository - they are
distributed under Moonshot's own license by
<https://huggingface.co/moonshotai/Kimi-K3>. `nano/model_nano.py` downloads
them automatically on first use (into this directory); you can also fetch them
by hand:

```bash
mkdir -p nano/vendor/moonshot
for f in modeling_kimi_linear.py configuration_kimi_k3.py; do
  curl -sL "https://huggingface.co/moonshotai/Kimi-K3/resolve/main/$f" -o "nano/vendor/moonshot/$f"
done
touch nano/vendor/moonshot/__init__.py
```
