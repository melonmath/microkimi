# The numbers

Two engines, same machine, same model. Two models: Qwen3.8-27B, the
reference model, measured on a 24-core cloud host with four L4s;
Qwen3.5-0.8B, the small model, measured on an Apple M5 (16 GB, plugged
in, bare metal) against llama.cpp build 4df29be, Q8_0. Higher is
better.

## Qwen3.8-27B (the reference model)

Measured on a cloud host: Intel Cascade Lake, 24 cores (48 threads),
188 GB, and four NVIDIA L4 (23 GB each) - microkimi at 6ee4236,
llama.cpp at 169e4a7 with CUDA. Paired, alternating, three rounds,
medians. The model is 49 GB converted; every number below is with the
model in RAM.

### On the CPU (24 cores)

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | **3.4 tok/s** | 2.9 tok/s |
| prompt reading (241 tokens) | 41 tok/s | 43 tok/s |

microkimi: q8 spine, fp4 MLP read as stored, one thread per physical
core; llama.cpp: Q8_0, `-ngl 0 -t 24`. Generation 298 vs 350 ms/token;
prompt reading 24.3 vs 23.4 ms/token.

### On the GPU (CUDA)

| | microkimi, one L4 | llama.cpp Q4_K_M, one L4 | llama.cpp Q8_0, four L4s |
|---|---:|---:|---:|
| generation | **12.0 tok/s** | 12.7 tok/s | 8.8 tok/s |
| prompt reading | 430 tok/s | 679 tok/s | 781 tok/s |

microkimi (`MICROKIMI_QWEN_CUDA=1`) holds the whole model on ONE L4:
the MLP as the file's MXFP4 bytes, the attention spine and the head as
q8_0 rows, 18.4 GB resident; the CPU keeps nothing. llama.cpp's Q8_0
(27 GB) does not fit one L4 and runs split across four; its Q4_K_M
(19 GB) fits one, and is the fair single-GPU row. Generation 83 vs 79
vs 113 ms/token; prompt reading of a 265-token prompt at 2.3 vs 1.5 vs
1.3 ms/token (a 241-token prompt reads at 483 tok/s: the GEMM's
128-token tile shows just past a multiple of it). 48/48 greedy tokens
of the GPU decode agree with the CPU forward after a GPU prompt.

## Qwen3.5-0.8B (the small model)

### On the CPU

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | **100 tok/s** | 86 tok/s |
| prompt reading (1k tokens) | **742 tok/s** | 668 tok/s |

Both rows from one bracketed run on the plugged-in M5 (2026-08-18):
microkimi q8 spine, all cores; llama.cpp `-ngl 0`, brackets 651-682
tok/s pp1024 and 85.6-86.9 tg64 before and after. Same warm,
repeated protocol on both sides (`prefillbench` / `llama-bench -r`).
The same battery on the same day under a foreign CPU load (a trading
terminal at one core) read 83 vs 86-87 on generation and 742 vs
657-675 on prompt reading: the loaded host takes the generation win
back to level and leaves the prompt row.

### On the GPU (Metal)

| | microkimi | llama.cpp |
|---|---:|---:|
| generation | **143 tok/s** | 114 tok/s |
| prompt reading (1k tokens) | **5100 tok/s** | 4700 tok/s |

Same M5, same window, paired (2026-08-18): microkimi decodes the whole
token in one command buffer against resident q8_0 rows and the MXFP4
MLP as stored, and reads the whole prompt in one command buffer too
(`MICROKIMI_QWEN_GPU=1`), 64/64 greedy tokens in agreement with the CPU
forward (141-146 tok/s over the pairs), 4980-5270 tok/s over the prompt
pairs; llama.cpp Metal, Q8_0, `tg64` 110-115 and `pp1024` 4670-4740
across the brackets. Prompt reading is the warm,
back-to-back protocol on both sides (`qwengpubench --gpu-only`,
`llama-bench`); under the paired protocol on a host 10 GB into swap the
GPU prompt row reads 3200-3300 tok/s - the difference is the paging of
weight copies between rounds, not the GPU (BENCH.md).

(The GPU rows barely move on battery - Apple throttles the CPU much
harder than the GPU - so these hold across power states.)

### On battery (ratios, not absolutes)

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
  attention spine; llama.cpp's Q8_0 is 8-bit everywhere (microkimi
  reads less memory per token) and its Q4_K_M is 4-bit everywhere
  (about the same bytes per token on the 27B: 18.4 GB against 19.0).
- The CUDA rows are one GPU against one GPU where the model fits;
  microkimi has no multi-GPU split, llama.cpp does.
- microkimi's GPU paths stay within 1.4e-2 of the CPU logits (prompt
  reading) and agree on every greedy token over the measured runs
  (decode); neither is bit-exact. The default engine path is bit-exact
  f32.
- Every number is the median of paired rounds on the same day,
  measured plugged in; battery runs are read as within-window ratios
  only (the bench script brackets for that).
- Under sustained throttle the gap WIDENS in microkimi's favor on
  generation: interleaved duels on a throttled host measured medians
  of 62.0 vs 55.8 and 53.0 vs 40.6 tok/s (`scripts/cpu-duel.sh`).
- 8 concurrent streams reach 3.7x the single-stream aggregate
  (lane-batched decoding), about 280 tok/s served from the CPU.

Reproduce everything: [BENCH.md](BENCH.md).
