# The numbers

One machine. One model. Two engines.

Apple M5 (16 GB, plugged in, bare metal). Qwen3.5-0.8B. microkimi
against llama.cpp (build 4df29be, Q8_0). Higher is better.

## On the CPU

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | **100 tok/s** | 86 tok/s |
| prompt reading (1k tokens) | **742 tok/s** | 668 tok/s |

Both rows from one bracketed run on the plugged-in M5 (2026-08-18):
microkimi q8 spine, all cores; llama.cpp `-ngl 0`, brackets 651-682
tok/s pp1024 and 85.6-86.9 tg64 before and after. Same warm,
repeated protocol on both sides (`prefillbench` / `llama-bench -r`).

## On the GPU (Metal)

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | 100 tok/s * | 112 tok/s |
| prompt reading (1k tokens) | **~1230 tok/s** | ~4650 tok/s |

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
| generation | 0.7x - 1.2x | wins most windows; interleaved duel medians +11% to +31% for microkimi |
| prompt reading | 0.98x - 1.11x | at equal harness (both engines warm and repeating, interleaved - `scripts/cpu-duel-warm.sh`): 673 vs 666 tok/s at best-of-8 in the container, 742 vs 651-682 on the plugged M5; the earlier 0.4-0.7x figures compared microkimi's single cold prefill against llama-bench's warm repeats |

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
