# Kimi K3 (microkimi)

**microkimi is the Kimi K3 architecture, unchanged** - only tensor dimensions are scaled down to fit in RAM. Layer counts, expert counts, mechanisms, tokenizer: identical.

| component                       | real K3              | microkimi                      | nanokimi                   |
| ------------------------------- | -------------------- | ------------------------------ | -------------------------- |
| layers                          | 93 (69 KDA + 24 MLA) | **93 (same)**                  | 8-12                       |
| hidden                          | 7168                 | **512**                        | 512                        |
| vocab                           | 163 840              | **163 840 (real tokenizer)**   | 8 192 + 8 specials (remap) |
| KDA heads × dim                 | 96 × 128             | **4 × 128**                    | 4 × 128                    |
| MLA (NoPE) heads                | 96                   | **4** (q_lora 128, kv_lora 64) | 4                          |
| experts routed / top-k / shared | 896 / 16 / 2         | **896 / 16 / 2 (same)**        | 896 / 16 / 2               |
| expert hidden / inter           | 3584 / 3072          | **128 / 64**                   | 128 / 64                   |
| AttnRes block size              | 12                   | **12**                         | 4                          |
| expert storage                  | MXFP4                | **MXFP4 (dequant on the fly)** | MXFP4                      |

Mechanisms, implemented exactly:

- **KDA (Kimi Delta Attention)** - linear-attention recurrence, per-channel learned decay, delta-rule update; fixed-size state, no growing KV cache (million-token contexts).
- **MLA (Multi-head Latent Attention)** - keys/values compressed into a small shared latent; no positional encoding at all.
- **Fine-grained MoE** - 896 routed experts per layer (16 active + 2 shared), sigmoid router with score-correction bias.
- **AttnRes** - residual blocks re-mixed by attention every 12 layers.
- **SiTU** activations; **MXFP4** expert storage, dequantized on the fly.

What is scaled down: tensor dims (for RAM) and the training budget - not the architecture. Given the real dims, data and GPU hours, this same code path is built to run the same computation.

## Verified 1:1 against Moonshot's code

`paritytest` drives Moonshot's own `KimiDecoderLayer` / `KimiDeltaAttention` / `KimiMLAAttention` / `KimiSparseMoeBlock` classes (downloaded at runtime, fp32) with the same weights, layer by layer, and diffs against the Rust forward:

| check                                                    | result                                     |
| -------------------------------------------------------- | ------------------------------------------ |
| router top-16 indices (layers 1, 47, 92, all positions)  | **exact match**                            |
| final logits                                             | max_abs 5.8e-5 (f32 summation-order noise) |
| per-layer hidden states + KDA/MoE sub-blocks             | 1e-9 … 3e-5                                |
| mechanism goldens (KDA recurrence, SiTU, MXFP4, AttnRes) | pass at 1e-4                               |

`paritytest --show` prints the concrete side-by-side values.

```bash
./target/release/microkimi build       # assemble microkimi.bin (~2.5 GB, K3 fetch + Qwen pools)
./target/release/microkimi selftest    # mechanism self-tests (torch once: ref/make_golden.py)
python3 ref/parity_ref.py              # regenerates ref/parity_golden.json
./target/release/microkimi paritytest  # the 1:1 proof above
```

## nanokimi: from noise to stories

Same engine, same greedy decoding, same prompt - only the weights change:

| model          | weights                      | `"Once upon a time, there was a little girl named Lily."`                                                                                                                                 |
| -------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| microkimi      | synthetic, untrained         | `增进食欲蚕食蚕食蚕食蚕食…`                                                                                                                                                               |
| nanokimi smoke | 200 steps (~0.1 M tokens)    | `". She wanted to play with the toy, but the park was very happy."` - grammatical, but the _same sentence for every prompt_                                                               |
| **nanokimi**   | **560 steps (4.3 M tokens)** | `" She loved to play with her toys and her favorite toy was a big, red ball. One day, Lily's mom asked her to help clean the park. Lily was so happy and excited to play with the ball."` |

More prompts (raw, unedited): `"Tom was a small boy who loved to play outside."` → `" One day, he went to the park to play with his friends."` · `"One day, a cat named Whiskers found a shiny red ball."` → `" He wanted to play with it, but he was too small… He asked his friend, a little bird, to help him."` The router learned too: expert usage is differentiated without collapse (top expert ≈ 8 % of calls), visible live with `--debug-routing`.

### The training run

![training loss](docs/training_curve.png)

|          |                                                                                                                |
| -------- | -------------------------------------------------------------------------------------------------------------- |
| VM       | GCP e2-highcpu-32 spot (32 vCPU, 31 GB RAM, **no GPU**, ~$0.5/h)                                               |
| Corpus   | TinyStories V2, real Kimi tiktoken remapped to top 8 192 tokens (99.95 % coverage), 150 M tokens prepared      |
| Model    | 181.6 M params - 8 layers (6 KDA + 2 MLA), 896 experts top-16 + 2 shared, AttnRes (~25 M active/token)         |
| Recipe   | AdamW lr 3e-4 cosine, batch 32 × 256, fp32, gradient checkpointing, grouped-GEMM MoE (bit-exact vs naive loop) |
| Result   | 560 steps, 4.3 M tokens, 501 min (~140 tok/s), **loss 9.13 → 2.5586** (perplexity ~8200 → ~13)                 |
| Survived | one OOM (found and fixed), one GCP preemption (watchdog + atomic checkpoints + `--resume`)                     |

Only 3 % of the corpus was seen and the loss was still descending at the end - the full corpus is ~12 days on the same box, or ~1 day on 12 with data parallelism. The pipeline is `nano/`: `prepare.py`, `model_nano.py`, `train.py`, `export.py`, `ops/` (preemptible-proof watchdog).

Training also runs on Apple Silicon GPUs: `train.py --device mps` (torch MPS backend, plus `--device cpu|auto` and `--bench N` to compare). Honest caveat: the KDA recurrence is a sequential Python loop, so MPS does not help there - measure with `--bench` before choosing.

## Benchmarks

Greedy decode.

| model | workload | hardware | ms/token | tok/s |
|---|---|---|---|---|
| microkimi | decode (93 layers, 2.5 GB f32+MXFP4) | 10-core ARM64 | 59 | ~17 |
| microkimi | decode (93 layers) | Apple M5, 16 GB | 34 | ~29 |
| nanokimi | decode (8 layers, 113 MB) | 10-core ARM64 | 7.5 | ~134 |
| nanokimi | training (batch 32×256) | 32 vCPU x86-64 | - | ~130-290 |
