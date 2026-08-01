# microkimi

A zero-dependency Rust engine for two frontier MoE models - **Kimi K3** and **DeepSeek-V4-Flash-0731** - both verified 1:1 against the official reference code. Runs on a plain laptop CPU (no CUDA, no BLAS, no crates); uses the GPU via Metal on macOS. Also included: **nanokimi**, a small K3 model trained from scratch overnight on CPU, and **nanodeepseek**, its DeepSeek counterpart (being trained).

> Almost all of the code was written by Kimi K3 itself, with human guidance and review.

> _The moon loved the sun, but they could never meet,_
> _so every night, they would go to the park to play._
> _Every night, the moon danced with the stars,_
> _and they were very happy, playing together every day._
> _The end._
>
> - nanokimi, running on microkimi

## Quickstart

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

## Supported models

| model | architecture | verified | details |
|---|---|---|---|
| **microkimi** | Kimi K3 (micro dims) | 1:1 vs Moonshot's code | [KIMI.md](KIMI.md) |
| **microdeepseek** | DeepSeek-V4-Flash-0731 (micro dims) | 1:1 vs DeepSeek's code | [DEEPSEEK.md](DEEPSEEK.md) |
| **nanokimi** | Kimi K3, trained from scratch | - | [KIMI.md](KIMI.md) |
| **nanodeepseek** | DeepSeek-V4, trained from scratch | - | being trained |

Independent project, no affiliation with Moonshot AI or DeepSeek. No weights in the repo (assembled by `microkimi build`; reference files downloaded at runtime). Outputs of the big models are deterministic gibberish by design - the point is the engine.

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

GPU (macOS/Metal): `--gpu` sends the large matvecs (≥ 2M elements) to the GPU; the rest stays on CPU, which is faster at micro dims. See the per-model benchmarks in [KIMI.md](KIMI.md) and [DEEPSEEK.md](DEEPSEEK.md).

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

## License

MIT - see [LICENSE](LICENSE).

## Acknowledgments

| | |
|---|---|
| Kimi K3 architecture & reference code | **Moonshot AI** (downloaded at runtime, never vendored) |
| DeepSeek-V4 architecture, reference code & tokenizer | **DeepSeek AI** (MIT, downloaded at runtime, never vendored) |
| `nano/vendor/fla` shim | **flash-linear-attention** (MIT, © Songlin Yang, Yu Zhang, Zhiyuan Li) |
| Weight pools for `microkimi build` | **Qwen2.5-0.5B-Instruct** (Apache 2.0) |
| Training data | **TinyStories** (Ronen Eldan, Microsoft Research) |
