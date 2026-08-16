# Benchmark protocol (bare metal)

One command runs the full paired battery on a converted dense Qwen
model and prints a report:

```bash
cargo build --release
./target/release/microkimi qwenbench --model qwen.bin
```

The battery: single-stream decode with the f32, q8, and fp4 spines;
the SDOT kernel A/B; batched-versus-sequential prefill on a ~1k-token
prompt; in-process lane-batched aggregate A/B at 4 and 8 lanes; and the
chained-MTP speculative A/B when the model carries its draft head.
Every arm shows its per-round values so host variance is visible, and
medians are the numbers to quote.

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

Same checkpoint, closest quantization, same machine, no other load:

```bash
# in a llama.cpp checkout
python3 convert_hf_to_gguf.py /path/to/Qwen3.5-0.8B --outfile q08-f16.gguf
./build/bin/llama-quantize q08-f16.gguf q08-q8_0.gguf Q8_0
./build/bin/llama-bench -m q08-q8_0.gguf -p 1024 -n 64
```

Read llama-bench's `pp1024` row against the qwenbench prefill line and
its `tg64` row against the q8-spine decode line. Two honesty notes for
the comparison: microkimi's MLP is MXFP4 (4-bit) while Q8_0 is 8-bit
everywhere, so microkimi reads less weight traffic per token than a
pure Q8_0 build of the same model; and microkimi's q8 spine is off by
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
