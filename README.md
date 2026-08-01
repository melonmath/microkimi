# microkimi

> [!NOTE]
> **This project started as an experiment: can you distribute LLM inference over a peer-to-peer network?**
> (MoE experts are the natural unit for that.)
>
> **Almost all of the code was written by Kimi K3 itself**, with human guidance and review.
> So the model effectively gave birth to its own reimplementation - a strangely poetic loop: an architecture explaining itself well enough to be rebuilt from first principles.

**Run and train miniature frontier MoE architectures - Kimi K3 and DeepSeek-V4-Flash-0731 - in pure Rust, zero dependencies, both verified 1:1 against the official reference code. Includes nanokimi, a small model trained from scratch overnight on CPU. Talks straight to the Metal API for GPU on macOS - CPU mode stays the fastest at small model sizes.**

> _The moon loved the sun, but they could never meet,_
> _so every night, they would go to the park to play._
> _Every night, the moon danced with the stars,_
> _and they were very happy, playing together every day._
> _The end._
>
> - nanokimi, running on microkimi

|                   |                                                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Engine**        | Zero-dependency Rust inference engine for frontier MoE architectures, verified 1:1 against the official reference code. Pure `std` Rust: no crates, no BLAS, no CUDA. See [KIMI.md](KIMI.md) and [DEEPSEEK.md](DEEPSEEK.md) for the per-model details. |
| **microkimi**     | The **Kimi K3** architecture at micro dims (93 layers kept, 896 experts top-16 kept; widths reduced) - verified 1:1 against Moonshot's code. → [KIMI.md](KIMI.md) |
| **microdeepseek** | The **DeepSeek-V4-Flash-0731** architecture at micro dims (43 layers kept, 256 experts top-6 kept; widths reduced) - verified 1:1 against DeepSeek's code. → [DEEPSEEK.md](DEEPSEEK.md) |
| **nanokimi**      | A small K3-architecture model trained from scratch (random weights → English stories) by the included `nano/` pipeline, overnight on CPU. **nanodeepseek** (the V4 counterpart) is being trained. |
| **Scope**         | Independent project, no affiliation with Moonshot AI or DeepSeek. No weights in the repo (assembled by `microkimi build`; reference files downloaded at runtime). Outputs of the big models are deterministic gibberish by design - the point is the engine. |

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

## Commands (unified for both architectures)

| task | Kimi K3 | DeepSeek-V4-Flash-0731 |
|---|---|---|
| assemble weights | `microkimi build` | `microkimi build --arch dsv4` |
| verify 1:1 vs official code | `microkimi paritytest` | `microkimi parity --arch dsv4` |
| all mechanism self-tests | `microkimi selftest` (covers both) | `microkimi selftest` (covers both) |
| generate | `microkimi run "..." --model microkimi.bin` | `microkimi run "..." --model microdeepseek.bin` |
| interactive | `microkimi chat --model nanokimi.bin --raw` | `microkimi chat --model microdeepseek.bin` |
| GPU checks (macOS) | `metaltest`, `gputest` | `dstest` |

`build-ds` and `dsparity` remain as aliases of `build --arch dsv4` / `parity --arch dsv4`.

## Measured performance

| workload                                            | hardware        | ms/token | tok/s    |
| --------------------------------------------------- | --------------- | -------- | -------- |
| microkimi decode (93 layers, 2.5 GB f32+MXFP4)      | 10-core ARM64   | 59       | ~17      |
| microkimi decode (93 layers)                        | Apple M5, 16 GB | 34       | ~29      |
| nanokimi decode (8 layers, 113 MB)                  | 10-core ARM64   | 7.5      | ~134     |
| microdeepseek decode (43 layers, 2.0 GB f32+FP4)    | 10-core ARM64   | 39       | ~26      |
| nanokimi training (batch 32×256)                    | 32 vCPU x86-64  | -        | ~130-290 |
| `microkimi build` (fetch + quant + write 2.5 GB)    | 10-core ARM64   | -        | ~65 s    |
| `microkimi build-ds` (fetch + quant + write 2.0 GB) | 10-core ARM64   | -        | ~86 s    |

GPU note (macOS/Metal): `--gpu` offloads the large matvecs to the GPU with weights cached on device. At micro dims the model runs ~1,200 small matvecs per token and per-dispatch sync dominates, so the GPU only takes matvecs ≥ 2M elements (lm_head) - the rest stays faster on the CPU thread pool. At real K3 dims (88M-MAC matvecs) the balance flips in the GPU's favor.

## Repository layout

```
src/            the Rust engine (K3 + DeepSeek-V4, zero dependencies)
nano/           training pipeline (prepare / model / train / export / ops / eval)
nano/vendor/fla pure-PyTorch fla-core shim (MIT, © fla-org - see vendor/README.md)
ref/            test tooling: make_golden.py, parity_ref.py, make_ds_parity.py, fetch_moonshot.py
docs/           training curve
KIMI.md         Kimi K3 details (architecture, parity proof, nanokimi training)
DEEPSEEK.md     DeepSeek-V4-Flash-0731 details (architecture, parity proof, what's here / not)
```

## License & acknowledgments

MIT (see `LICENSE`). Kimi K3 architecture and reference code: **Moonshot AI** (downloaded at runtime, never vendored). DeepSeek-V4 architecture, reference code and tokenizer: **DeepSeek AI** (MIT, downloaded at runtime, never vendored). `nano/vendor/fla`: **flash-linear-attention**, MIT, © Songlin Yang, Yu Zhang, Zhiyuan Li. Weight pools for `microkimi build`: **Qwen2.5-0.5B-Instruct** (Apache 2.0). Training data: **TinyStories** (Ronen Eldan, Microsoft Research).
