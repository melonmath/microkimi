# microkimi

> [!NOTE]
> **This project started as an experiment: can you distribute LLM inference over a peer-to-peer network?**
> (MoE experts are the natural unit for that.)
>
> **Almost all of the code was written by Kimi K3 itself**, with human guidance and review.
> So the model effectively gave birth to its own reimplementation - a strangely poetic loop: an architecture explaining itself well enough to be rebuilt from first principles.

**Run and train your own miniature Kimi K3: a zero-dependency Rust inference engine, reimplemented from scratch - and nanokimi, a small model trained from scratch on CPU by the included training pipeline. Talks straight to the Metal API for GPU on macOS - CPU mode stays the fastest at small model sizes.**

> *The moon loved the sun, but they could never meet,*
> *so every night, they would go to the park to play.*
> *Every night, the moon danced with the stars,*
> *and they were very happy, playing together every day.*
> *The end.*
>
> - nanokimi, running on microkimi

| | |
|---|---|
| **microkimi** | Zero-dependency Rust inference engine for the Kimi K3 architecture (KDA, MLA NoPE, latent MoE 896 experts top-16, SiTU, AttnRes, MXFP4). Verified 1:1 against Moonshot's official code. |
| **nanokimi** | A small K3-architecture model trained from scratch (random weights → English stories) by the included `nano/` pipeline, overnight on CPU. |
| **Dependencies** | None. Pure `std` Rust: no crates, no BLAS, no CUDA. Threads via `std::thread`, HTTP via `curl` shellout. |
| **Scope** | Independent project, no affiliation with Moonshot AI. No weights in the repo (assembled by `microkimi build`; Moonshot files downloaded at runtime). Output of the big model is deterministic gibberish by design - the point is the engine. |

## Run it in 30 seconds

You need nothing but a Rust toolchain (`rustup`).

```bash
cargo build --release
# download nanokimi.bin + vocab_nano.json from the GitHub Releases page
# into the repo root, then:
./target/release/microkimi run "Once upon a time, a kind dragon lived in a cave. Every morning, he" \
  --model nanokimi.bin --max-new 12 --raw
# answer:  would go to the park and play with his friends.

./target/release/microkimi chat --model nanokimi.bin --raw      # interactive stories
./target/release/microkimi run "One day, a cat found a ball." \
  --model nanokimi.bin --max-new 10 --raw --debug-routing       # watch the MoE router pick experts

# on macOS: Metal GPU for the large matvecs
./target/release/microkimi metaltest                            # GPU sanity check
./target/release/microkimi gputest                              # GPU vs CPU on real model matvecs
./target/release/microkimi run "Once upon a time" --model nanokimi.bin --max-new 12 --raw --gpu
```

## Architecture

**microkimi is the Kimi K3 architecture, unchanged** - only tensor dimensions are scaled down to fit in RAM. Layer counts, expert counts, mechanisms, tokenizer: identical.

| component | real K3 | microkimi | nanokimi |
|---|---|---|---|
| layers | 93 (69 KDA + 24 MLA) | **93 (same)** | 8-12 |
| hidden | 7168 | **512** | 512 |
| vocab | 163 840 | **163 840 (real tokenizer)** | 8 192 + 8 specials (remap) |
| KDA heads × dim | 96 × 128 | **4 × 128** | 4 × 128 |
| MLA (NoPE) heads | 96 | **4** (q_lora 128, kv_lora 64) | 4 |
| experts routed / top-k / shared | 896 / 16 / 2 | **896 / 16 / 2 (same)** | 896 / 16 / 2 |
| expert hidden / inter | 3584 / 3072 | **128 / 64** | 128 / 64 |
| AttnRes block size | 12 | **12** | 4 |
| expert storage | MXFP4 | **MXFP4 (dequant on the fly)** | MXFP4 |

Mechanisms, implemented exactly:

- **KDA (Kimi Delta Attention)** - linear-attention recurrence, per-channel learned decay, delta-rule update; fixed-size state, no growing KV cache (million-token contexts).
- **MLA (Multi-head Latent Attention)** - keys/values compressed into a small shared latent; no positional encoding at all.
- **Fine-grained MoE** - 896 routed experts per layer (16 active + 2 shared), sigmoid router with score-correction bias.
- **AttnRes** - residual blocks re-mixed by attention every 12 layers.
- **SiTU** activations; **MXFP4** expert storage, dequantized on the fly.

What is scaled down: tensor dims (for RAM) and the training budget - not the architecture. Given the real dims, data and GPU hours, this same code path is built to run the same computation.

## Verified 1:1 against Moonshot's code

`paritytest` drives Moonshot's own `KimiDecoderLayer` / `KimiDeltaAttention` / `KimiMLAAttention` / `KimiSparseMoeBlock` classes (downloaded at runtime, fp32) with the same weights, layer by layer, and diffs against the Rust forward:

| check | result |
|---|---|
| router top-16 indices (layers 1, 47, 92, all positions) | **exact match** |
| final logits | max_abs 5.8e-5 (f32 summation-order noise) |
| per-layer hidden states + KDA/MoE sub-blocks | 1e-9 … 3e-5 |
| mechanism goldens (KDA recurrence, SiTU, MXFP4, AttnRes) | pass at 1e-4 |

`paritytest --show` prints the concrete side-by-side values.

## nanokimi: from noise to stories

Same engine, same greedy decoding, same prompt - only the weights change:

| model | weights | `"Once upon a time, there was a little girl named Lily."` |
|---|---|---|
| microkimi | synthetic, untrained | `增进食欲蚕食蚕食蚕食蚕食…` |
| nanokimi smoke | 200 steps (~0.1 M tokens) | `". She wanted to play with the toy, but the park was very happy."` - grammatical, but the *same sentence for every prompt* |
| **nanokimi** | **560 steps (4.3 M tokens)** | `" She loved to play with her toys and her favorite toy was a big, red ball. One day, Lily's mom asked her to help clean the park. Lily was so happy and excited to play with the ball."` |

More prompts (raw, unedited): `"Tom was a small boy who loved to play outside."` → `" One day, he went to the park to play with his friends."` · `"One day, a cat named Whiskers found a shiny red ball."` → `" He wanted to play with it, but he was too small… He asked his friend, a little bird, to help him."` The router learned too: expert usage is differentiated without collapse (top expert ≈ 8 % of calls), visible live with `--debug-routing`.

### The training run

![training loss](docs/training_curve.png)

| | |
|---|---|
| VM | GCP e2-highcpu-32 spot (32 vCPU, 31 GB RAM, **no GPU**, ~$0.5/h) |
| Corpus | TinyStories V2, real Kimi tiktoken remapped to top 8 192 tokens (99.95 % coverage), 150 M tokens prepared |
| Model | 181.6 M params - 8 layers (6 KDA + 2 MLA), 896 experts top-16 + 2 shared, AttnRes (~25 M active/token) |
| Recipe | AdamW lr 3e-4 cosine, batch 32 × 256, fp32, gradient checkpointing, grouped-GEMM MoE (bit-exact vs naive loop) |
| Result | 560 steps, 4.3 M tokens, 501 min (~140 tok/s), **loss 9.13 → 2.5586** (perplexity ~8200 → ~13) |
| Survived | one OOM (found and fixed), one GCP preemption (watchdog + atomic checkpoints + `--resume`) |

Only 3 % of the corpus was seen and the loss was still descending at the end - the full corpus is ~12 days on the same box, or ~1 day on 12 with data parallelism. The pipeline is `nano/`: `prepare.py`, `model_nano.py`, `train.py`, `export.py`, `ops/` (preemptible-proof watchdog).

Training also runs on Apple Silicon GPUs: `train.py --device mps` (torch MPS backend, plus `--device cpu|auto` and `--bench N` to compare). Honest caveat: the KDA recurrence is a sequential Python loop, so MPS does not help there - measure with `--bench` before choosing.

## Measured performance

| workload | hardware | number |
|---|---|---|
| microkimi decode (93 layers, 2.5 GB f32+MXFP4) | 10-core ARM64 | **~59 ms/token** (~17 tok/s) |
| microkimi decode (93 layers) | Apple M5, 16 GB | **~34 ms/token** (~29 tok/s) |
| nanokimi decode (8 layers, 113 MB) | 10-core ARM64 | **~134 tok/s** |
| nanokimi training (batch 32×256) | 32 vCPU x86-64 | ~130-290 tok/s (shared CPU) |
| `microkimi build` (fetch + quant + write 2.5 GB) | 10-core ARM64 | ~65 s |

GPU note (macOS/Metal): `--gpu` offloads the large matvecs to the GPU with weights cached on device. At micro dims the model runs ~1,200 small matvecs per token and per-dispatch sync dominates, so the GPU only takes matvecs ≥ 2M elements (lm_head) - the rest stays faster on the CPU thread pool. At real K3 dims (88M-MAC matvecs) the balance flips in the GPU's favor.

## Full pipeline from source

```bash
cargo build --release
./target/release/microkimi build       # assemble microkimi.bin (~2.5 GB, K3 fetch + Qwen pools)
./target/release/microkimi selftest    # mechanism self-tests (torch once: ref/make_golden.py)
python3 ref/parity_ref.py              # regenerates ref/parity_golden.json
./target/release/microkimi paritytest  # the 1:1 proof above
```

## Repository layout

```
src/            the Rust engine (13 modules, zero dependencies)
nano/           training pipeline (prepare / model / train / export / ops / eval)
nano/vendor/fla pure-PyTorch fla-core shim (MIT, © fla-org - see vendor/README.md)
ref/            test tooling: make_golden.py, parity_ref.py, fetch_moonshot.py
docs/           training curve
```

## License & acknowledgments

MIT (see `LICENSE`). Kimi K3 architecture and reference code: **Moonshot AI** (downloaded at runtime, never vendored). `nano/vendor/fla`: **flash-linear-attention**, MIT, © Songlin Yang, Yu Zhang, Zhiyuan Li. Weight pools for `microkimi build`: **Qwen2.5-0.5B-Instruct** (Apache 2.0). Training data: **TinyStories** (Ronen Eldan, Microsoft Research).

