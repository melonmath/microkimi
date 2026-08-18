# Benchmark protocol (bare metal)

Numbers live in [RESULTS.md](RESULTS.md). This file is how to produce
them.

## Qwen3.8-27B, the reference model

```bash
cargo build --release
./target/release/microkimi convert-qwen --source /path/to/Qwen3.8-27B --out qwen3.8-27b.bin --audit-only   # 866 tensors, 49 GB payload
./target/release/microkimi convert-qwen --source /path/to/Qwen3.8-27B --out qwen3.8-27b.bin
./target/release/microkimi qwenbench --model qwen3.8-27b.bin --light --rounds 2 --steps 16
```

`--light` is the battery a model this size can afford: the q8 spine
decode and the q8 batched prefill of a 256-token prompt (the A/B arms
stay off). The checkpoint is 56 GB (18 safetensors shards) and the
converted model 49 GB (f32 attention spine and embeddings, MXFP4 MLP);
under about 64 GB of RAM the model pages from disk and the numbers
measure the disk. `bench-27b.sh` at the workspace root runs the whole
thing (download, audit, conversion, the light battery, the llama.cpp
Q8_0 head-to-head with the ggml-org GGUF) on macOS or Linux, with disk
guards. `qwen3.8-27b.bin` in the repo root or `models/` is the default
model of `run`, `chat` and `serve` when it is there.

## Qwen3.5-0.8B, the small model

```bash
./target/release/microkimi convert-qwen --source /path/to/Qwen3.5-0.8B --out q08.bin
./target/release/microkimi qwenbench --model q08.bin
```

The battery is paired and reports per-round values; quote medians.
Arms: decode with the f32/q8/fp4 spines, SDOT A/B, all-cores A/B,
batched vs sequential prefill, GPU prefill vs CPU prefill (in-process,
with the logits disagreement), lane batching, and MTP when the model
has a draft head, plus the GPU decode arm on macOS. `microkimi
qwengpubench --model q08.bin` runs the GPU prefill arm alone;
`microkimi gpudecodebench --model q08.bin --steps 64` runs the GPU
decode arm alone: kernel checks, then N tokens decoded on the GPU and
the CPU from the same state, ms/token for both (first step apart) and
the greedy agreement. Read `tg64` against its median.

GPU rows on macOS: `microkimi qwengpubench --model q08.bin --gpu-only
--rounds 8` (prompt reading, GPU rounds back to back, like
`llama-bench`'s repetitions) and `microkimi gpudecodebench --model
q08.bin --steps 64` (generation), both under `MICROKIMI_QWEN_GPU=1`
semantics. The paired protocol of `qwengpubench` (a CPU round between
GPU rounds) is the fair one on a throttling host but punishes the GPU
row on a host short of memory: the weight copies a prompt touches once
get paged out between rounds and paged back inside the next command
buffer (`MICROKIMI_GPU_LAYER_PROF=2` shows it as kernel span before
GPU start). Quote which protocol you ran.

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

## Shape checks without the weights

`microkimi qwen-fixture --out X.bin --profile 27b --scale 8 --layers 8`
writes a synthetic checkpoint with Qwen3.8-27B's shape signature (three
value heads per key head, six query heads per kv head, head dim 256,
quarter rotary, untied embeddings) and `gpudecodebench --model X.bin
--trace` / `qwengpubench --model X.bin` then read the GPU graphs against
the CPU on that geometry (per-layer error and last-position logits; the
greedy line is meaningless on synthetic ties). It says nothing about
speed or language.

## Honesty notes

- microkimi's MLP is MXFP4 (4-bit) while Q8_0 is 8-bit everywhere, so
  microkimi reads less weight per token than llama.cpp at Q8_0.
- the q8 spine and the GPU paths are opt-in; the default engine path
  is bit-exact f32. Quote the arm that matches the property you need,
  with the NLL deltas from [QWEN.md](QWEN.md) next to any speed claim.
- absolute numbers want a plugged-in host; on battery, use the paired
  duel or the brackets and quote ratios.
