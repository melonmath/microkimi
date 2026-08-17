# The numbers

One machine. One model. Two engines.

Apple M5 (16 GB, plugged in, bare metal). Qwen3.5-0.8B. microkimi
against llama.cpp (build 4df29be, Q8_0). Higher is better.

## On the CPU

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | **90.9 tok/s** | 87.6 tok/s |
| prompt reading (1k tokens) | 357 tok/s cold / **~650-670 warm** (see below) | 661 tok/s |

## On the GPU (Metal)

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | 90.9 tok/s * | 110-115 tok/s |
| prompt reading (1k tokens) | **~1000 tok/s** | ~4700 tok/s |

(The GPU rows barely move on battery - Apple throttles the CPU much
harder than the GPU - so these hold across power states.)

\* microkimi generates on the CPU in both columns: its GPU path
accelerates prompt reading only (for now).

## On battery (ratios, not absolutes)

macOS throttles hard and unevenly on battery, so these read as
microkimi-to-llama.cpp ratios inside the same thermal window
(llama.cpp rows bracketing our battery, plus interleaved duels):

| | ratio range | reading of the day |
|---|---:|---|
| generation | 0.7x - 1.1x | wins most windows; interleaved duel medians +11% to +31% for microkimi |
| prompt reading | **1.01x** (best-of-8) / 0.98x (medians) | at equal harness (both engines warm, 8 repetitions, interleaved - `scripts/cpu-duel-warm.sh`) on a calm host: microkimi 673 vs llama-bench 666 tok/s at best-of-8 (5 of 8 rounds won), 651 vs 666 on per-round medians - the two engines within 3% either way; the earlier 0.4-0.7x figures compared microkimi's single cold prefill against llama-bench's warm repeats |

The deeper the throttle, the better microkimi holds relative to
llama.cpp on generation (spinning job board + dynamic row
scheduling); prompt reading stays behind in every window pending the
chunked-scan work.

## The fine print

- Quantization differs: microkimi runs a 4-bit MLP with an 8-bit
  attention spine; llama.cpp runs 8-bit everywhere. microkimi reads
  less memory per token.
- microkimi's GPU prompt reading stays within 1.4e-2 of its CPU
  output. The default engine path is bit-exact f32.
- Every number is the median of paired rounds on the same day,
  measured plugged in; battery runs are read as within-window ratios
  only (the bench script brackets for that).
- Under sustained throttle the gap WIDENS in microkimi's favor on
  generation: interleaved duels on a throttled host measured medians
  of 62.0 vs 55.8 and 53.0 vs 40.6 tok/s (`scripts/cpu-duel.sh`).
- 8 concurrent streams reach 3.7x the single-stream aggregate
  (lane-batched decoding), about 280 tok/s served from the CPU.

Reproduce everything: [BENCH.md](BENCH.md).
