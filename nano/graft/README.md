# nano/graft - measurement and transfer instruments

Tools for measuring how two language models relate to each other, and
for moving feed-forward capacity between them. Every tool is standalone,
runs `--selftest`, and reports what it measures rather than asserting it.

## Measurement

**`expert_solve.py --map scan`** - compatibility scan. For each host
layer, ranks donor layers by the ridge R2 of predicting the donor's FFN
delta from the host's own stream. Closed form, no training, minutes: it
answers "can anything transfer between these two models, and where"
before any expensive work. Directional predictability is the criterion;
symmetric similarity scores (CKA, reported alongside) over-rank layers
whose subspaces overlap without the delta being predictable.

**`qwen_graft.py` utility rows** - per-expert utility. Forces one expert
per pass and diffs per-position loss against a baseline pass, giving the
measured contribution of an individual expert at every position of a
corpus. Applies to any fused-expert checkpoint.

**`qwen_graft.py measure_norm_gain`** - per-layer block output/input RMS
ratio. Anything injected at a block's output lives at that scale, which
differs from the post-norm scale by 20-70x in practice and varies by
layer.

**`compat_map.py`** - compatibility map. Runs the scan over every
ordered pair of captured models and every layer pair, and reports the
matrix, the best layer pair per model pair, and the directional
asymmetry. The score is directional: predicting A's delta from B's
stream is a different question from the reverse, and measurements show
the asymmetry can exceed 0.4 - predictability is largely a property of
the target model, not of the pair.

**`eval_compare.py`** - paired per-document comparison of two .bin
models: corpus cross-entropy, the paired delta with a bootstrap
confidence interval (documents are the resampling unit), and a one-sided
p-value. `--doc-range` keeps evaluation sets disjoint from calibration.

## Capture

**`capture_donor.py`** - per-layer FFN activations of a transformers
checkpoint, byte-anchored to raw text so captures of models with
different tokenizers can be paired position by position. Architecture
styles: llama/qwen/mistral, gemma3, qwen3_5_moe (routed and shared
expert), deepseek_v4. Multi-GPU and 4-bit loading; decoder root is
probed, not assumed.

**`capture_host.py`** - per-MoE-layer router-input and latent streams of
a microkimi .bin, same anchoring.

**`tokens_to_text.py`** - recovers a raw-text jsonl corpus from a
tokenized stream, so several tokenizers can read the same text.

## Transfer

**`expert_solve.py`** - closed-form host-shaped experts from two
captures and the donor weights: input fold, importance slice, and a
re-solve of the down matrix through the host activation. `--act`,
`--host-planes`, `--target` select the activation, the port convention
and whether the target is the donor delta or the part of it the host
does not already produce.

**`inject_experts.py`** - splices experts, router rows and selection
biases into a .bin. A strongly negative selection bias is exact silence
under top-k, so a freshly grafted model is bit-identical to its host.

**`qwen_graft.py`** - the same for transformers checkpoints: bank
extension with a pre-softmax selection bias, or gated side branches that
attach to any feed-forward block, MoE or dense, without touching
routing. Calibration sweeps always include silence, so a calibrated
result is never worse than its input on the calibration holdout.

**`graft_heal.py`, `grad_refit.py`, `als_refit.py`** - optional
integration passes: scoped SGD (`--train all|w2|gate`), a one-pass
Gauss-Newton update with a trust-region line search and no optimizer,
and a weighted re-solve restricted to the routed slice.

## Supported architectures

Hosts and donors: microkimi .bin (K3 family), qwen3_5_moe and
qwen3_5_moe_text (routed and shared experts), qwen3_5_text and other
dense llama-style decoders, deepseek_v4, gemma3.

```sh
python3 ../tests/test_graft.py   # runs every selftest
```
