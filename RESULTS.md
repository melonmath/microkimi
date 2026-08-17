# The numbers

One machine. One model. Two engines.

Apple M5 (16 GB, plugged in, bare metal). Qwen3.5-0.8B. microkimi
against llama.cpp (build 4df29be, Q8_0). Higher is better.

## On the CPU

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | 62.5 tok/s | 88.8 tok/s |
| prompt reading (1k tokens) | 96 tok/s | 689 tok/s |

## On the GPU (Metal)

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | 62.5 tok/s * | 114.2 tok/s |
| prompt reading (1k tokens) | **980 tok/s** | 4771 tok/s |

\* microkimi generates on the CPU in both columns: its GPU path
accelerates prompt reading only (for now).

## The fine print

- Quantization differs: microkimi runs a 4-bit MLP with an 8-bit
  attention spine; llama.cpp runs 8-bit everywhere. microkimi reads
  less memory per token.
- microkimi's GPU prompt reading stays within 1.4e-2 of its CPU
  output. The default engine path is bit-exact f32.
- Every number is the median of paired rounds on the same day.

Reproduce everything: [BENCH.md](BENCH.md).
