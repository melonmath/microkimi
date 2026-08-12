# microkimi

A zero-dependency Rust engine for three modern MoE architectures - **Kimi K3**, **DeepSeek-V4-Flash-0731**, and the **Qwen3.5/Qwen3.6-MoE text decoder** - verified against their reference implementations.

The Python side under `nano/graft/` adds architecture-agnostic instruments: compatibility scans between two models, per-expert utility measurement, paired evaluation with bootstrap intervals, and closed-form feed-forward transfer. Those read and write **Qwen3.5-MoE** (`qwen3_5_moe`, routed and shared experts), **Qwen3.5 dense**, **DeepSeek-V4**, **Gemma 3** and llama-style checkpoints in addition to the engine's own format - see `nano/graft/README.md`.

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

| model | what it is | how to get it |
|---|---|---|
| **nanokimi-0.2b** | 0.2B Kimi K3 model, trained from scratch | [Releases](https://github.com/microkimi/microkimi/releases) - see [KIMI.md](KIMI.md) |
| **microkimi-debug** | full 93-layer K3 skeleton, synthetic weights | `microkimi build` - see [KIMI.md](KIMI.md) |
| **microdeepseek-debug** | DeepSeek-V4 skeleton, synthetic weights | `microkimi build --arch dsv4` - see [DEEPSEEK.md](DEEPSEEK.md) |
| **converted Qwen3.5/Qwen3.6-MoE** | text checkpoint converted to f32 spine + MXFP4 routed experts | `microkimi convert-qwen` - see [QWEN.md](QWEN.md) |

## Commands

| task | Kimi K3 | DeepSeek-V4-Flash-0731 | Qwen3.5/Qwen3.6-MoE |
|---|---|---|---|
| assemble or convert weights | `microkimi build` | `microkimi build --arch dsv4` | `microkimi convert-qwen --source DIR --out qwen.bin` |
| verify vs reference code | `microkimi paritytest` | `microkimi parity --arch dsv4` | `python3 ref/qwen_parity.py --out DIR` + `qwen-dump` |
| mechanism self-tests | `microkimi selftest` | `microkimi selftest` | `cargo test qwen` |
| generate | `microkimi run "..." --model microkimi-debug.bin` | `microkimi run "..." --model microdeepseek-debug.bin` | `microkimi run "..." --model qwen.bin` |
| interactive | `microkimi chat --model nanokimi-0.2b.bin --raw` | `microkimi chat --model microdeepseek-debug.bin` | `microkimi chat --model qwen.bin` |

`build-ds` and `dsparity` remain as aliases of `build --arch dsv4` / `parity --arch dsv4`.

## Engine features

| feature | what it does (measured, not promised) |
|---|---|
| MoE expert streaming | `--stream` keeps expert blobs on disk and fetches on demand (LRU + rollover in RAM), offset-sorted reads, direct I/O auto-detected: O_DIRECT on Linux, F_NOCACHE on macOS (`MICROKIMI_NO_ODIRECT=1` to A/B) |
| Markov prefetch | `--stream-predict N` pre-fetches the experts the router is likely to pick next; `microkimi cachereplay <trace>` replays a recorded request trace offline under LRU / LFU / ARC / Belady / Markov policies (record with `MICROKIMI_TRACE=trace.bin`); the live eviction policy is selected with `MICROKIMI_CACHE=arc|lru|lfu` (default lfu) |
| shadow fallback | `--stream-fallback` (default OFF, DEGRADED latency mode): on an expert cache miss, serve the resident 0.5-bit VQ1 shadow of the expert immediately (`microkimi shadow --model X.bin` builds the `<model>.shadows` sidecar) and refill full precision in the background - the decode never blocks on the disk, but shadow-served tokens are not bit-identical; the stream report counts them |
| mmap demand-paging | models are mapped, not loaded: the kernel pages weights on demand, so a model larger than RAM still runs (`MICROKIMI_NO_MMAP=1` for the old full-load path) |
| microquant | `microkimi slice --cold-vq N` keeps all experts but requantizes the coldest to 0.5-bit VQ - measured better than deleting them (30.6% vs 19.1% top-1 parity with the full model) |
| structural slicing | `microkimi slice` prunes layers / hidden channels / experts (`--layers --hidden --experts`) and vocabulary (`--vocab-top`) from a .bin or straight from remote safetensors; crash-safe resume (`.sliceckpt`) and a persistent expert-score cache |
| evaluation | `microkimi eval --model X.bin` - deterministic scorecard: 40 factual QA probes (2 phrasings) + perplexity, `--json` for archiving |
| batch completion | `microkimi complete-batch` evaluates JSONL prompts with one model and adapter load, resetting caches between prompts and atomically writing JSONL results |
| model adapter packs | K3 and Qwen f32 spine tensors: `nano/adapter_pack.py` turns a standard exact-same-base PEFT LoRA into a hash-bound `.mkap`, optionally with an explicit scale multiplier; repeat `--adapter skill.mkap` to compose packs without changing the base `.bin` |
| memory packs | K3 only: `microkimi absorb doc.txt --out pack.mkmem` snapshots the fixed-size KDA state; `run --memory pack.mkmem` resumes it. A video-game save state - details in [KIMI.md](KIMI.md#memory-packs-save-states-for-a-neural-network) |

## License

MIT - see [LICENSE](LICENSE). Credits in [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
