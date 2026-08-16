# Benchmark protocol (bare metal)

One command runs the full paired battery on a converted dense Qwen
model and prints a report:

```bash
cargo build --release
./target/release/microkimi qwenbench --model qwen.bin
```

The battery: single-stream decode with the f32, q8, and fp4 spines;
the SDOT kernel A/B; batched-versus-sequential prefill on a ~1k-token
prompt; on macOS the MPS-GEMM GPU prefill against the CPU batched
prefill (in-process paired rounds after a discarded upload warm-up,
plus the last-position logits disagreement); in-process lane-batched
aggregate A/B at 4 and 8 lanes; and the chained-MTP speculative A/B
when the model carries its draft head. Every arm shows its per-round
values so host variance is visible, and medians are the numbers to
quote. `microkimi qwengpubench --model X.bin` runs the GPU prefill
arm alone.

## macOS (Apple silicon), outside any container

Bandwidth is the decode wall, and containers sit far below the silicon:
the shared aarch64 container these features were developed in delivers
about 19 GB/s effective; an M-series package delivers 100-400 GB/s.
Expect roughly proportional single-stream decode gains on bare metal.

```bash
# on the Mac, from a clean checkout
cargo build --release
./target/release/microkimi convert-qwen --source /path/to/Qwen3.5-0.8B --out q08.bin
./target/release/microkimi qwenbench --model q08.bin
```

The SDOT kernels engage automatically (every Apple M-series reports
dotprod); `MICROKIMI_NO_SDOT=1` is the A/B arm. The `--gpu` Metal path
currently accelerates the K3 engine only; the Qwen runtime is CPU, which
is the honest comparison to llama.cpp's CPU backend.

## Comparing with llama.cpp on the same machine

Same checkpoint, closest quantization, same machine, no other load.
No conversion needed: ggml-org publishes ready-made GGUFs.

```bash
# in a llama.cpp checkout (cmake -B build && cmake --build build -t llama-bench)
curl -fLO https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
./build/bin/llama-bench -m Qwen3.5-0.8B-Q8_0.gguf -p 1024 -n 64          # Metal GPU (macOS default)
./build/bin/llama-bench -m Qwen3.5-0.8B-Q8_0.gguf -p 1024 -n 64 -ngl 0  # CPU-only row
```

One trap: on macOS llama-bench runs the **Metal GPU backend by
default** (backend column says `MTL`). The fair CPU-versus-CPU row
needs `-ngl 0`. Read `pp1024` against the qwenbench prefill line and
`tg64` against the q8-spine decode line.

Measured on an Apple M5 (4P/6E, 16 GB, AC power, 2026-08-16),
Qwen3.5-0.8B:

| arm | prefill | decode |
|---|---|---|
| microkimi q8 spine (CPU, 4 P-cores) | 64-73 tok/s | 47.6-58.8 tok/s |
| microkimi + MPS GEMM prefill offload | 275 tok/s | (decode stays CPU) |
| llama.cpp Q8_0 CPU-only (-ngl 0) | 664 tok/s | 86.3 tok/s |
| llama.cpp Q8_0 (Metal GPU) | 4548-4743 tok/s | 109.8-110.8 tok/s |
| llama.cpp Q4_0 (Metal GPU) | 4604-4715 tok/s | 139.7-140.1 tok/s |

Reading, from two runs on the same machine: the fair CPU-versus-CPU
decode gap is 1.47x (58.8 vs 86.3 tok/s), and since microkimi reads
LESS weight per token (MXFP4 MLP vs Q8_0 everywhere), that gap is
kernel bandwidth-extraction, not traffic - llama.cpp's CPU build (with
Accelerate) pulls roughly 70 GB/s from four threads against our ~40.
The MPS GEMM prefill offload measured 3.74x over the CPU batched
prefill on its first outing (13.6 to 3.64 ms/token, last-position
logits within 1.4e-2); llama.cpp's 664 tok/s CPU prefill row shows
what AMX-backed GEMMs do on this silicon, and their full-Metal graph
at ~4700 tok/s is the end state - reaching it means moving the
remaining CPU tissue (delta scan, attention, norms) onto the GPU, and
qwengpubench prints the exact GEMM-versus-tissue split to size that
work. Their Q4_0 gains only 27% over Q8_0 on the GPU (dispatch-bound
at 0.8B, consistent with our fp4-spine rejection).

Two honesty notes for any quote: microkimi's MLP is MXFP4 (4-bit)
while Q8_0 is 8-bit everywhere, so microkimi reads less weight traffic
per token than a pure Q8_0 build; and microkimi's q8 spine is off by
default because the default path is bit-exact f32 - quote whichever arm
matches the property you need, and quote the NLL deltas published in
QWEN.md next to any speed claim.

## Container ceiling (why bare metal matters)

On the development container, q8-spine decode measured 34 ms/token
against a ~26 ms/token bandwidth floor (about 0.5 GB of weight traffic
per token at ~19 GB/s): the engine runs at roughly three quarters of
that host's physical ceiling, and further container-side kernel work
cannot move much. The same binary on higher-bandwidth silicon is the
meaningful next measurement.
