# Qwen3.5-MoE text runtime

microkimi can convert and run the text decoder used by Qwen3.5-MoE and
Qwen3.6-MoE checkpoints. The implementation covers the alternating 3:1
Gated DeltaNet/full-attention backbone, softmax top-k routed experts, the
always-on sigmoid-gated shared expert, partial rotary embeddings, grouped
query attention, and the checkpoint's byte-level BPE chat tokenizer.

Qwen3.6 uses the same `qwen3_5_moe_text` architecture and Transformers model
classes as Qwen3.5. See the upstream
[architecture documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5_moe.md)
and
[reference implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).

## Convert and run

The source must be a local Hugging Face checkpoint directory containing
`config.json`, a safetensors index or one safetensors file, and optionally
`tokenizer.json`:

```bash
cargo build --release

# Header-only validation and output-size estimate. Writes nothing.
./target/release/microkimi convert-qwen \
  --source /path/to/checkpoint --out qwen.bin --audit-only

# Full conversion. qwen.tokenizer.json is copied beside qwen.bin when present.
./target/release/microkimi convert-qwen \
  --source /path/to/checkpoint --out qwen.bin

./target/release/microkimi run "Explain why the sky is blue." \
  --model qwen.bin --max-new 128
./target/release/microkimi chat --model qwen.bin
```

`--vocab /path/to/tokenizer.json` overrides the tokenizer copied beside the
model. `--raw` bypasses the chat template and performs a plain completion.

For independent prompts that should share one model and adapter load,
`complete-batch` accepts JSONL requests and writes JSONL results atomically:

```json
{"id":"example","prompt":"def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n    ","max_new":64}
```

```bash
./target/release/microkimi complete-batch --model qwen.bin \
  --input requests.jsonl --out completions.jsonl
```

Each request resets the recurrent and KV caches. Add `--chat` to apply the
text chat template instead of raw completion encoding. An optional `stop`
array ends a request once any decoded stop string appears.

The converter fails closed on an unsupported model type, attention bias,
dropout, tied embeddings, rope mode, layer pattern, missing tensor, unexpected
text-decoder tensor, dtype, shape, or non-finite value. Vision and
multi-token-prediction tensors are separate from the text decoder and are not
copied.

## Storage and execution

Float spine tensors keep their native logical names and are converted to f32.
This includes embeddings, attention, norms, routing, shared experts, and the
language-model head. Each fused routed-expert bank is split into independent
gate/down/up matrices and quantized to MXFP4. Conversion streams large float
tensors in bounded chunks, reads only one expert matrix at a time, and
parallelizes quantization across the available CPU cores.

At inference time the model is a private, demand-paged mapping. Float tensors
are zero-copy slices of that mapping. Only the selected routed experts are
read and evaluated; their packed matrix-vector products run in parallel. The
linear-attention layers retain fixed recurrent and convolution states, while
every fourth layer keeps an ordinary growing KV cache.

The regular Qwen RMSNorm weights are checkpoint offsets from one. The gated
DeltaNet output norm uses direct multipliers. The runtime preserves this
distinction, Qwen's per-head query/gate interleave, grouped-query cache layout,
and the exact full-softmax then top-k renormalization order.

## Adapter packs

The generic `MKADAPT1` format works on Qwen float spine matrices. Convert an
elementary, exact-same-base PEFT LoRA with `nano/adapter_pack.py`, then repeat
`--adapter` to compose packs:

```bash
python3 nano/adapter_pack.py create \
  --base qwen.bin --adapter /path/to/adapter --name coding --out coding.mkap

./target/release/microkimi run "Write a parser." \
  --model qwen.bin --adapter coding.mkap
```

The complete base file and every factor are SHA-256 verified before private
copy-on-write pages are patched. Several packs compose additively in pack
digest order. Routed experts are packed rather than f32, so expert targets are
rejected instead of being silently requantized.

## Deterministic evaluation

`microkimi eval` supports converted Qwen models and repeated `--adapter`
arguments. `--skip-qa` runs only next-token NLL, and
`--ppl-max-tokens N` caps the encoded evaluation window for a preregistered
compute budget. Both choices are recorded in the JSON scorecard:

```bash
./target/release/microkimi eval --model qwen.bin --adapter coding.mkap \
  --ppl-file held-out.txt --skip-qa --ppl-max-tokens 1024 \
  --json scorecard.json
```

## Independent parity

`ref/qwen_parity.py` constructs a deterministic four-layer Transformers
checkpoint with three Gated DeltaNet layers, one full-attention layer, routed
and shared experts, and values exactly representable by the MXFP4 converter.
It writes reference logits that can be compared with the hidden `qwen-dump`
development command. Disable the optional q8 language-model-head cache for
this exact f32 comparison:

The reference-only Python step requires PyTorch, Transformers, and
safetensors; the converter and runtime remain dependency-free Rust.

```bash
python3 ref/qwen_parity.py --out /tmp/qwen-parity
./target/release/microkimi convert-qwen \
  --source /tmp/qwen-parity --out /tmp/qwen-parity/rust.bin
MICROKIMI_Q8HEAD=0 ./target/release/microkimi qwen-dump \
  --model /tmp/qwen-parity/rust.bin --tokens 3,5,7,11 \
  --out /tmp/qwen-parity/rust_logits.bin
python3 ref/qwen_parity.py --out /tmp/qwen-parity \
  --compare /tmp/qwen-parity/rust_logits.bin
```

Against Transformers 5.14.1 on the four-token fixture:

| check | result |
|---|---|
| maximum absolute logit difference | `2.38e-7` |
| overall logit RMSE | `4.99e-8` |
| top-1 tokens | 4 of 4 exact |
| per-token cosine similarity | effectively `1.0` |

The tokenizer is independently compared with the checkpoint tokenizer on
ordinary text, source code, multilingual text, decomposed Unicode, and the
default thinking chat template. Those cases match token for token. NFC is
implemented with generated Unicode 15.1 canonical tables and algorithmic
Hangul composition, without adding a crate.

Current limits: `--stream`, K3 memory snapshots, prefix-cache snapshots,
early exit, and speculative decoding are not wired to the Qwen runtime. The
default mmap load is still demand-paged and does not copy the model file into
anonymous RAM.
