# microkimi

A zero-dependency Rust engine for two frontier MoE architectures - **Kimi K3** and **DeepSeek-V4-Flash-0731** - verified 1:1 against the official reference code.

**This is a framework for developers, including an end-to-end engine and a test model - not a model for end-users.**

> _Almost all of the code was written by Kimi K3 itself - the quiet beauty of an LLM giving birth to another._

> _The moon loved the sun, but they could never meet,_
> _so every night, they would go to the park to play._
> _Every night, the moon danced with the stars,_
> _and they were very happy, playing together every day._
> _The end._
>
> - nanokimi-0.2b, running on microkimi

## Quickstart

You need nothing but a Rust toolchain (`rustup`). The GitHub Releases ship only the trained **nano** models (`nanokimi-0.2b.bin` + `vocab_nano.json`) - the debug models are generated locally with `microkimi build` (they are architecture demos, not shipped).

```bash
cargo build --release
# download nanokimi-0.2b.bin + vocab_nano.json from the GitHub Releases page
# into the repo root, then:
./target/release/microkimi run "Once upon a time, a kind dragon lived in a cave. Every morning, he" \
  --model nanokimi-0.2b.bin --max-new 12 --raw
# answer:  would go to the park and play with his friends.

./target/release/microkimi chat --model nanokimi-0.2b.bin --raw      # interactive stories
./target/release/microkimi run "One day, a cat found a ball." \
  --model nanokimi-0.2b.bin --max-new 10 --raw --debug-routing       # watch the MoE router pick experts
```

## Models

Naming rule: **nano** models are trained from scratch here; **micro** models are pruned from the real weights; **-debug** files are synthetic fixtures for parity tests and engine tracing.

| model | what it is | details |
|---|---|---|
| **nanokimi-0.2b** | small Kimi K3 model, trained from scratch (in Releases) | [KIMI.md](KIMI.md) |
| **nanokimi-0.2b-chat** | chat-tuned nanokimi-0.2b | being trained |
| **nanodeepseek-0.2b** | small DeepSeek-V4 model, trained from scratch | paused |
| **microkimi-0.2b / 1b** | pruned from real K3 weights | planned |
| **microdeepseek-*** | pruned from real DeepSeek-V4 weights | planned |
| **microkimi-debug** | Kimi K3 architecture demo (synthetic weights, `microkimi build`) | [KIMI.md](KIMI.md) |
| **microdeepseek-debug** | DeepSeek-V4 architecture demo (synthetic weights, `build --arch dsv4`) | [DEEPSEEK.md](DEEPSEEK.md) |

## Commands

| task | Kimi K3 | DeepSeek-V4-Flash-0731 |
|---|---|---|
| assemble weights | `microkimi build` | `microkimi build --arch dsv4` |
| verify 1:1 vs official code | `microkimi paritytest` | `microkimi parity --arch dsv4` |
| all mechanism self-tests | `microkimi selftest` (covers both) | `microkimi selftest` (covers both) |
| generate | `microkimi run "..." --model microkimi-debug.bin` | `microkimi run "..." --model microdeepseek-debug.bin` |
| interactive | `microkimi chat --model nanokimi-0.2b.bin --raw` | `microkimi chat --model microdeepseek-debug.bin` |

`build-ds` and `dsparity` remain as aliases of `build --arch dsv4` / `parity --arch dsv4`.

## License

MIT - see [LICENSE](LICENSE). Credits in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
