# Kimi K3 (via microkimi)

**microkimi implements the Kimi K3 architecture, unchanged** - only tensor dimensions are scaled down to fit in RAM. Layer counts, expert counts, mechanisms, tokenizer: identical.

| component                       | nanokimi-0.2b              | microkimi-debug                | real K3              |
| ------------------------------- | -------------------------- | ------------------------------ | -------------------- |
| layers                          | 8-12                       | **93 (same)**                  | 93 (69 KDA + 24 MLA) |
| hidden                          | 512                        | **512**                        | 7168                 |
| vocab                           | 8 192 + 8 specials (remap) | **163 840 (real tokenizer)**   | 163 840              |
| KDA heads × dim                 | 4 × 128                    | **4 × 128**                    | 96 × 128             |
| MLA (NoPE) heads                | 4                          | **4** (q_lora 128, kv_lora 64) | 96                   |
| experts routed / top-k / shared | 896 / 16 / 2               | **896 / 16 / 2 (same)**        | 896 / 16 / 2         |
| expert hidden / inter           | 128 / 64                   | **128 / 64**                   | 3584 / 3072          |
| AttnRes block size              | 4                          | **12**                         | 12                   |
| expert storage                  | MXFP4                      | **MXFP4 (dequant on the fly)** | MXFP4                |

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
./target/release/microkimi build       # assemble microkimi-debug.bin (~2.5 GB, K3 fetch + Qwen pools)
./target/release/microkimi selftest    # mechanism self-tests (torch once: ref/make_golden.py)
python3 ref/parity_ref.py              # regenerates ref/parity_golden.json
./target/release/microkimi paritytest  # the 1:1 proof above
```

## nanokimi-0.2b: from noise to stories

Same engine, same greedy decoding, same prompt - only the weights change:

| model          | weights                      | `"Once upon a time, there was a little girl named Lily."`                                                                                                                                 |
| -------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| microkimi-debug | synthetic, untrained        | `增进食欲蚕食蚕食蚕食蚕食…`                                                                                                                                                               |
| nanokimi smoke | 200 steps (~0.1 M tokens)    | `". She wanted to play with the toy, but the park was very happy."` - grammatical, but the _same sentence for every prompt_                                                               |
| **nanokimi-0.2b** | **560 steps (4.3 M tokens)** | `" She loved to play with her toys and her favorite toy was a big, red ball. One day, Lily's mom asked her to help clean the park. Lily was so happy and excited to play with the ball."` |

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

## Memory packs: save states for a neural network

A video-game save state, but for the model's mind. Because KDA layers carry a **fixed-size recurrent state** (unlike a Transformer KV cache, which grows forever), the entire working state of a conversation can be snapshotted to a small portable file - an emulator save-state for an LLM.

```bash
# snapshot at any point (during a run):
microkimi run "One morning, a little girl found a key." --model nanokimi-0.2b.bin --max-new 10 --save fork.mkmem
# resume from the snapshot - the continuation is bit-exact:
microkimi run "" --model nanokimi-0.2b.bin --memory fork.mkmem --max-new 10
# absorb a document into the state, no context window involved:
microkimi absorb notes.txt --out pack.mkmem --model nanokimi-0.2b.bin
microkimi run "What did my notes say?" --model nanokimi-0.2b.bin --memory pack.mkmem
```

What a pack contains: the KDA recurrent states + conv states (**fixed size, never grows** - on nanokimi: ~1.5 MB whatever the absorbed length), the MLA k/v caches (these do grow, ~10 KB/token on nanokimi - the KDA part is the constant one), and the last logits for exact continuation.

Measured behavior (honest status):

- **Fork/resume is bit-exact** - a resumed run reproduces the uninterrupted run token for token. Save-scumming conversations works today.
- **Absorb injects measurable information** - with a pack loaded, 30-50% of greedy token choices change vs the same prompt/seed without it. At 0.2B, no exploitable facts are retrievable yet (the toy model cannot re-read its own compressed state); retesting on microkimi-1b is planned.
- **Merge is destructive at 0.2B** (`mkmem-div` measured it) - two independently trained states live in differently-rotated spaces; naive averaging cancels them. Alignment (re-basing) is the open follow-up.

Why this is unique: it requires a fixed-size state (KDA) and an engine that exposes it. Standard Transformers cannot do this - their "state" is an ever-growing KV cache.

## Benchmarks

Greedy decode.

| model | workload | hardware | ms/token | tok/s |
|---|---|---|---|---|
| nanokimi-0.2b | decode (8 layers, 113 MB) | 10-core ARM64 | 7.5 | ~134 |
| nanokimi-0.2b | training (batch 32×256) | 32 vCPU x86-64 | - | ~130-290 |
| microkimi-debug | decode (93 layers, 2.5 GB f32+MXFP4) | 10-core ARM64 | 59 | ~17 |
| microkimi-debug | decode (93 layers) | Apple M5, 16 GB | 34 | ~29 |
