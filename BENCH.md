# Benchmark protocol (bare metal)

Numbers live in [RESULTS.md](RESULTS.md). This file is how to produce
them.

## microkimi

```bash
cargo build --release
./target/release/microkimi convert-qwen --source /path/to/Qwen3.5-0.8B --out q08.bin
./target/release/microkimi qwenbench --model q08.bin
```

The battery is paired and reports per-round values; quote medians.
Arms: decode with the f32/q8/fp4 spines, SDOT A/B, all-cores A/B,
batched vs sequential prefill, GPU prefill vs CPU prefill (in-process,
with the logits disagreement), lane batching, and MTP when the model
has a draft head. `microkimi qwengpubench --model q08.bin` runs the
GPU prefill arm alone.

## llama.cpp, same machine

```bash
# in a llama.cpp checkout (cmake -B build && cmake --build build -t llama-bench)
curl -fLO https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
./build/bin/llama-bench -m Qwen3.5-0.8B-Q8_0.gguf -p 1024 -n 64          # Metal GPU (macOS default)
./build/bin/llama-bench -m Qwen3.5-0.8B-Q8_0.gguf -p 1024 -n 64 -ngl 0  # CPU-only
```

Trap: on macOS llama-bench uses the Metal GPU **by default** (backend
column `MTL`). The CPU row needs `-ngl 0`. Read `pp1024` against the
prefill lines and `tg64` against the decode lines.

## Paired duel (the fair protocol on a throttling host)

On battery the host throttles between runs, so serial comparisons lie.
`scripts/cpu-duel.sh` runs N interleaved rounds - microkimi decode,
llama-bench tg32, microkimi prefill, llama-bench pp1024, each engine
at its best thread count - so every comparison lives inside one
thermal window. Quote medians over rounds:

```bash
WORK=/path/with/q08.bin+q08.gguf+llama.cpp bash scripts/cpu-duel.sh 7
```

The macOS runner script also brackets its qwenbench battery with two
llama.cpp CPU rows for the same reason.

## Honesty notes

- microkimi's MLP is MXFP4 (4-bit) while Q8_0 is 8-bit everywhere, so
  microkimi reads less weight per token than llama.cpp at Q8_0.
- the q8 spine and the GPU prefill are opt-in; the default engine path
  is bit-exact f32. Quote the arm that matches the property you need,
  with the NLL deltas from [QWEN.md](QWEN.md) next to any speed claim.
- absolute numbers want a plugged-in host; on battery, use the paired
  duel or the brackets and quote ratios.
