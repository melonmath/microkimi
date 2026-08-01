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
