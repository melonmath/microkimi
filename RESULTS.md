# The numbers

One machine. One model. Two engines.

Apple M5 (16 GB, plugged in, bare metal). Qwen3.5-0.8B. microkimi
against llama.cpp (build 4df29be, Q8_0). Higher is better.

## On the CPU

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | **90.9 tok/s** | 87.6 tok/s |
| prompt reading (1k tokens) | 115 tok/s | 661 tok/s |

## On the GPU (Metal)

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | 90.9 tok/s * | 103.3 tok/s |
| prompt reading (1k tokens) | **952 tok/s** | 4571 tok/s |

\* microkimi generates on the CPU in both columns: its GPU path
accelerates prompt reading only (for now).

## The fine print

- Quantization differs: microkimi runs a 4-bit MLP with an 8-bit
  attention spine; llama.cpp runs 8-bit everywhere. microkimi reads
  less memory per token.
- microkimi's GPU prompt reading stays within 1.4e-2 of its CPU
  output. The default engine path is bit-exact f32.
- Every number is the median of paired rounds on the same day.
- 8 concurrent streams reach 3.7x the single-stream aggregate
  (lane-batched decoding), about 280 tok/s served from the CPU.

Reproduce everything: [BENCH.md](BENCH.md).
