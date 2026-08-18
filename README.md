# microkimi

**A large-model engine in pure Rust. Zero dependencies. Measured, not promised.**

One small binary runs Kimi K3, DeepSeek-V4, and the Qwen3.5 family (Qwen3.8-27B is its default model when converted). No Python. No crates. No SDK to install: the GPU paths open Metal and the CUDA driver at run time. `cargo build`, point it at a model, go.

- **Fast.** Qwen3.8-27B on one NVIDIA L4: 12 tokens per second, the whole model resident on a single 24 GB GPU - where llama.cpp's Q8_0 needs four. On an Apple M5, prompts read at over 1,000 tokens per second, ahead of llama.cpp on CPU. [The numbers.](RESULTS.md)
- **Exact.** The default path is bit-exact f32. Every speed mode is measured against it and quoted honestly.
- **Time travel.** Conversations are version-controlled. Fork any past state. Diff two answers. Merge two branches back into one mind.
- **Serve it.** An OpenAI-compatible API lives in the same binary: `microkimi serve`.

A Python toolkit under `nano/graft/` handles model surgery - compatibility scans, expert utility measurement, feed-forward transfer - across Qwen, DeepSeek, Gemma and llama-style checkpoints. See `nano/graft/README.md`.

**This is a framework for developers - an engine and a test model, not a product for end-users.**

> _Almost all of the code was written by Kimi K3 itself - the quiet beauty of an LLM giving birth to another._

> _The moon loved the sun, but they could never meet,_
> _so every night, they would go to the park to play._
> _Every night, the moon danced with the stars,_
> _and they were very happy, playing together every day._
> _The end._
>
> - nanokimi-0.2b, running on microkimi

## Get started

All you need is Rust (`rustup`). Download `nanokimi-0.2b.bin` + `vocab_nano.json` from [Releases](https://github.com/microkimi/microkimi/releases) into the repo root, then:

```bash
cargo build --release
./target/release/microkimi run "Once upon a time, a kind dragon lived in a cave. Every morning, he" \
  --model nanokimi-0.2b.bin --max-new 12 --raw
# answer:  would go to the park and play with his friends.

./target/release/microkimi chat --model nanokimi-0.2b.bin --raw
```

To run a real Qwen checkpoint:

```bash
./target/release/microkimi convert-qwen --source /path/to/Qwen3.5-0.8B --out qwen.bin
./target/release/microkimi chat --model qwen.bin
```

## Models

| model | what it is | guide |
|---|---|---|
| **nanokimi-0.2b** | 0.2B Kimi K3, trained from scratch | [KIMI.md](KIMI.md) |
| **converted Qwen3.5 / 3.6 / 3.8** | any Qwen3.5-family text checkpoint, MoE or dense | [QWEN.md](QWEN.md) |
| **microkimi-debug / microdeepseek-debug** | synthetic architecture demos for parity tests | [KIMI.md](KIMI.md), [DEEPSEEK.md](DEEPSEEK.md) |

## Go deeper

- [RESULTS.md](RESULTS.md) - benchmarks vs llama.cpp, same machine, same model.
- [BENCH.md](BENCH.md) - how to reproduce every number.
- [QWEN.md](QWEN.md) - the Qwen runtime: conversion, Metal and CUDA, speculative decoding, serving, conversation version control.
- [KIMI.md](KIMI.md) - the K3 engine: expert streaming, memory packs, slicing, adapters.
- [DEEPSEEK.md](DEEPSEEK.md) - the DeepSeek-V4 runtime.

## License

MIT - see [LICENSE](LICENSE). Credits in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
