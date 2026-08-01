# microkimi

A zero-dependency Rust engine for two frontier MoE architectures - **Kimi K3** and **DeepSeek-V4-Flash-0731** - verified 1:1 against the official reference code. Runs on a plain laptop CPU; uses the GPU via Metal on macOS.

**This is a framework for developers, including an end-to-end engine and a test model - not a model for end-users.**

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
./target/release/microkimi run "Once upon a time" --model nanokimi.bin --max-new 12 --raw --gpu
```

## Models

| model             | architecture                        | details                    |
| ----------------- | ----------------------------------- | -------------------------- |
| **microkimi**     | Kimi K3 (micro dims)                | [KIMI.md](KIMI.md)         |
| **nanokimi**      | Kimi K3, trained from scratch       | [KIMI.md](KIMI.md)         |
| **microdeepseek** | DeepSeek-V4-Flash-0731 (micro dims) | [DEEPSEEK.md](DEEPSEEK.md) |
| **nanodeepseek**  | DeepSeek-V4, trained from scratch   | being trained              |

## Commands

| task                        | Kimi K3                                     | DeepSeek-V4-Flash-0731                          |
| --------------------------- | ------------------------------------------- | ----------------------------------------------- |
| assemble weights            | `microkimi build`                           | `microkimi build --arch dsv4`                   |
| verify 1:1 vs official code | `microkimi paritytest`                      | `microkimi parity --arch dsv4`                  |
| all mechanism self-tests    | `microkimi selftest` (covers both)          | `microkimi selftest` (covers both)              |
| generate                    | `microkimi run "..." --model microkimi.bin` | `microkimi run "..." --model microdeepseek.bin` |
| interactive                 | `microkimi chat --model nanokimi.bin --raw` | `microkimi chat --model microdeepseek.bin`      |
| GPU checks (macOS)          | `metaltest`, `gputest`                      | `dstest`                                        |

`build-ds` and `dsparity` remain as aliases of `build --arch dsv4` / `parity --arch dsv4`.

## Repository layout

```
src/            the Rust engine (K3 + DeepSeek-V4, zero dependencies)
nano/           training pipeline
ref/            test tooling
docs/           training curve
KIMI.md         Kimi K3 details (architecture, parity proof, benchmarks)
DEEPSEEK.md     DeepSeek-V4 details (architecture, parity proof, benchmarks)
```

## License

MIT - see [LICENSE](LICENSE). Credits in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
