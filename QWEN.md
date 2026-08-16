# Qwen3.5-family text runtime

microkimi can convert and run both text decoders of the Qwen3.5
architecture family:

- the MoE variant (`qwen3_5_moe_text`): Qwen3.5-MoE, Qwen3.6-MoE, and the
  Qwen3.8-2.4T-A95B open-weights release;
- the dense variant (`qwen3_5_text`): Qwen3.8-27B, the dense multimodal
  checkpoint whose text decoder shares the same backbone.

The implementation covers the alternating 3:1 Gated DeltaNet/full-attention
backbone, partial rotary embeddings, grouped query attention, the sigmoid
attention output gate, and the checkpoint's byte-level BPE chat tokenizer.
On the MoE variant every layer runs softmax top-k routed experts plus the
always-on sigmoid-gated shared expert; on the dense variant every layer runs
a single SiLU-gated MLP whose matrices are MXFP4-packed exactly like routed
experts.

Qwen3.6 and Qwen3.8-2.4T-A95B use the same `qwen3_5_moe_text` architecture
and Transformers model classes as Qwen3.5; Qwen3.8-27B uses the sibling
`qwen3_5_text` classes. See the upstream
[architecture documentation](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5_moe.md)
and the
[MoE](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
/
[dense](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
reference implementations.

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

The prompt is ingested by a batched layers-outer prefill: each weight
region is traversed once per chunk instead of once per token, and
token-independent work fans out across the CPU cores. The result is
bit-identical to sequential single-token forwards.

## Multi-token prediction (dense variant)

Dense checkpoints that ship their `mtp.*` draft tensors (Qwen3.8-27B does)
are converted with them automatically. `--mtp` on `run` and `chat` then
enables greedy self-speculative decoding: the one-layer draft head proposes
the next token, the trunk verifies the pending token and the draft in one
two-token batched pass, and a rejected draft is rolled back exactly
(linear states restored, key/value caches truncated). Every emitted token
is the greedy argmax of the same logits the plain loop would produce, so
the output is bit-identical; the tok/s line reports passes and acceptance.

Transformers does not implement this module (its checkpoints keys are
ignored on load), so the draft semantics follow the deployed reference
proposers: draft slot `i` merges the embedding of committed token `i+1`
with the trunk's final-norm hidden of position `i` at rotary position `i`.
`ref/qwen_parity.py --dense` builds synthetic `mtp.*` tensors, computes
reference draft logits with an independent PyTorch mirror of that math,
and `microkimi qwen-dump --mtp` compares against it (`--compare-mtp`):
measured `7.45e-7` maximum absolute difference, 3 of 3 top-1 exact.

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
copied. The variant is taken from the checkpoint's `model_type` and must agree
with the presence of `intermediate_size`.

## Storage and execution

Float spine tensors keep their native logical names and are converted to f32.
This includes embeddings, attention, norms, routing, shared experts, and the
language-model head. Each fused routed-expert bank is split into independent
gate/down/up matrices and quantized to MXFP4; on the dense variant the
per-layer `mlp.gate_proj` / `up_proj` / `down_proj` matrices are quantized to
MXFP4 the same way. Conversion streams large float tensors and dense MLP
matrices in bounded chunks, reads only one expert matrix at a time, and
parallelizes quantization across the available CPU cores.

At inference time the model is a private, demand-paged mapping. Float tensors
are zero-copy slices of that mapping. On the MoE variant only the selected
routed experts are read and evaluated; their packed matrix-vector products run
in parallel. On the dense variant the three MLP matvecs run row-parallel
across the worker pool. The linear-attention layers retain fixed recurrent and
convolution states, while every fourth layer keeps an ordinary growing KV
cache.

The regular Qwen RMSNorm weights are checkpoint offsets from one. The gated
DeltaNet output norm uses direct multipliers. The runtime preserves this
distinction, Qwen's per-head query/gate interleave, grouped-query cache layout,
and the exact full-softmax then top-k renormalization order.

## Calibrated quantization (dense variant)

The dense MLP is traversed by every token of every layer, so its MXFP4
error costs more than a routed expert's. `microkimi calibrate` runs a
corpus through a converted dense model and accumulates the mean squared
input of each packed matrix per column (the hidden stream for
`gate_proj`/`up_proj`, the SiLU-gated activation for `down_proj`);
`convert-qwen --imatrix` then makes the per-group scale search minimize
the importance-weighted error instead of the plain one:

```bash
./target/release/microkimi calibrate --model qwen.bin \
  --text calibration.txt --out qwen.imx --max-tokens 8192
./target/release/microkimi convert-qwen \
  --source /path/to/checkpoint --out qwen-imx.bin --imatrix qwen.imx
```

Nibble assignment stays nearest-level and the byte layout is unchanged,
so the weighted file runs everywhere an unweighted one does. Uniform
importances reproduce the unweighted bytes exactly, and the weighted
search never loses on its own metric (both unit-tested). The calibration
runs on an already-converted model, so the statistics see MXFP4 rather
than bf16 activations upstream; this is the standard imatrix compromise.
MTP tensors stay unweighted (their activations are not in the pass).

## Vocabulary slicing

Most deployments never emit most of the 248320-entry vocabulary; its
embedding and head rows dominate the f32 spine. `slice-qwen-vocab` keeps a
subset of rows and rewrites the converted file:

```bash
./target/release/microkimi slice-qwen-vocab --model qwen.bin \
  --out sliced/qwen-small.bin --top 32768 --freqfile corpus_ids.txt
```

The keep set is the whole added-special block, the 256 single-byte tokens,
the chat-template pieces, and the top-N ids of a frequency file counting
the model's own token ids. A `qwen.vocabmap.json` sidecar written beside
the output carries the new-to-old table: the tokenizer loads it, encodes
on the full vocabulary, remaps, and re-encodes any dropped token as its
single-byte tokens - byte-level BPE keeps every byte sequence
representable, so no unknown token exists or is needed (a slice that
would drop a byte token is refused). Kept rows are bit-identical to the
source model's, and the sliced config remaps the special ids.

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
digest order. Routed experts and the dense variant's MLP matrices are packed
rather than f32, so those targets are rejected instead of being silently
requantized.

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

## Serving

`microkimi serve` exposes the model as an OpenAI-compatible endpoint,
still without dependencies:

```bash
./target/release/microkimi serve --model qwen.bin --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
  "max_tokens": 64, "enable_thinking": false}'
```

`/v1/completions` (raw) and `/v1/chat/completions` accept `max_tokens`,
`temperature`, `top_p`, `seed`, `stream` (SSE), and `enable_thinking`
(false renders the template's disabled think block, so a small model
spends its budget on the visible answer). Chat answers split `<think>`
reasoning into `reasoning_content`. Multi-turn conversations hit the
chat prefix cache across requests: entries are stored at the
conversation prefix and after generation, and disabled-think history is
replayed exactly as it was ingested, so the next turn extends the exact
token stream. Requests are served one at a time against the single
stateful decoder; the default bind is 127.0.0.1 and there is no
authentication layer - put a reverse proxy in front for anything else.

## Q8 spine

`MICROKIMI_Q8_SPINE=1` quantizes every large attention matrix (the
linear layers' in_qkv / in_z / out_proj, the full layers' q / k / v / o)
to q8 at load, the same trade llama.cpp makes everywhere: the f32 spine
dominates per-token weight traffic, and q8 cuts it about 4x on the
covered matrices. Norms, the convolution, the small b/a projections, and
the MTP head stay f32; the packed MLP was already 4-bit. All three
execution paths route through the same q8 copies (the single-token
forward delegates to the batched prefill), and their agreement is
tested; the mode is NOT bit-identical to the f32 spine and is off by
default.

Measured on the real 0.8B: decode 36 vs 57-73 ms/token in paired runs
(about 1.6x, 27.8 tok/s single-stream on this shared container), and
held-out NLL 2.7952 vs 2.7961 - a -0.0009 delta, indistinguishable from
the exact spine on this corpus. Combined with lane batching, eight q8
lanes reached 84.1 aggregate tok/s (2.98x median over the q8
single-stream in in-process A/B rounds). Session baseline was 9 tok/s:
the q8 spine plus the batched prefill closed most of the gap to
integer-kernel engines.

On aarch64 CPUs that report `dotprod`, the int8 kernels emit SDOT
through stable inline asm (the intrinsic is nightly-only; the silicon is
not): one instruction per 16 lanes instead of the widening
multiply-accumulate pair. Integer sums are exact whatever the
accumulation shape, so the kernel swap is bit-identical -
`MICROKIMI_NO_SDOT=1` is the A/B toggle, and paired runs measured about
5% end-to-end on this memory-bound decode (34 vs 36 ms/token median);
`MICROKIMI_FP4_SPINE=1` applies the engine's own MXFP4 machinery to the
same attention matrices - and the measurement rejected it at this
scale, on both axes: paired decode ran SLOWER than q8 (45 vs 34
ms/token median: the nibble decode costs more compute than the halved
traffic saves on a partly compute-bound host) and held-out NLL degraded
by +0.0695 against q8's -0.0009. The q8 spine is the measured sweet
spot on the 0.8B; the fp4 mode stays available because its traffic
argument returns on bandwidth-starved hosts (a 27B paging from disk),
where it must be re-measured before use.

On bare metal (Apple M5, 4P/6E, 16 GB, AC power) the same binary
measured q8 decode at 52.6 tok/s single-stream (19 ms/token, medians
over 5 rounds) and batched prefill at 14.5 ms/token, and the llama.cpp
head-to-head on the same machine and checkpoint (ggml-org Q8_0 GGUF,
llama-bench -p 1024 -n 64) put their **Metal GPU** backend at 109.8
tok/s tg64 and 4548 tok/s pp1024 - CPU decode within 2.1x of the GPU,
prefill ~66x behind it. The full protocol, the fair `-ngl 0`
CPU-versus-CPU row, and the reading are in [BENCH.md](BENCH.md); the
structural conclusion is that parity on Apple silicon runs through a
Metal backend for this runtime, not through more CPU kernels.

## GPU prefill offload (macOS)

`MICROKIMI_QWEN_GPU=1` routes every batched-prefill matmul through one
MPSMatrixMultiplication GEMM per weight matrix (still zero crates: the
Metal Performance Shaders framework is reached over the same
objc_msgSend FFI as the K3 Metal path). The regime argument decides
what moves: a Metal dispatch costs ~0.25 ms of sync latency, so decode
(~100 matvecs per token) stays on the CPU kernels where it belongs,
while prefill pays one dispatch per weight matrix for the WHOLE prompt
- the latency amortizes to ~0.03 ms/token at 1k tokens and the
arithmetic runs on the GPU. The nonlinear tissue between matmuls
(norms, the conv, the delta scan, softmax attention, activations)
stays on the CPU, with activations crossing through unified memory.
MXFP4 MLP weights are dequantized to f32 once at first use and cached
on device (~1 GB for the 0.8B stack); f32 attention matrices upload
as-is and are cached too. The offload is NOT bit-exact against the
CPU path (no q8 activation quantization, GPU reassociation) - the
qwengpubench arm prints the measured last-position logits disagreement
next to the timing, and every failure path falls back to the CPU
kernels rather than erroring. Gated off by default; lane-batched
decode and small batches never take this route (16-lane minimum,
1M-element minimum per matrix).

First bare-metal measurement (Apple M5, AC power): 3.64 vs 13.61
ms/token against the CPU batched prefill in paired in-process rounds -
3.74x, with the offloaded GEMMs matching a CPU recompute to 1.9e-7 and
the last-position logits within 1.4e-2 of the CPU path (the expected
dequant-versus-q8-activation delta). qwengpubench also reports the
GEMM-versus-CPU-tissue split per token, which is the sizing datum for
porting the remaining ops in later phases; llama.cpp's full-Metal
graph runs the same prefill at ~4700 tok/s on this machine, so the
ceiling is known and far.

Phase 2 moved the full-attention layers onto the same machinery:
scores = scale·Q·Kᵀ and mix = P·V run as BATCHED GEMMs (every head in
one encode each, the GQA kv-heads expanded across their group), with
the causal softmax on the CPU between the two and the masked tail
zeroed so the P·V product runs over the full cache width.
MICROKIMI_QWEN_GPU_NOATTN=1 is the kill switch that keeps the
projection GEMMs offloaded while returning attention to the CPU loop;
prompts whose scores stack would exceed the 256 MB staging ceiling
fall back automatically. The CPU tissue itself also lost its serial
spots in the same pass (measured on the container at ~27% of the
batched prefill): the SiLU merge and the causal convolution now fan
out over workers, attention token ranges are cut on causal cost
rather than uniformly, and every pooled row kernel pulls fine chunks
from a shared counter (MICROKIMI_NO_DYNROWS=1 for the A/B) so an
E-core straggler delays one chunk instead of a quarter of the matrix.
The delta-scan recurrence is the one block still CPU-bound by design:
its GPU port is sized by the qwengpubench split line before being
written. Not offloaded on the arithmetic: f16 GEMMs and paired
command buffers (both under 0.05 ms/token at 1k tokens - the GEMM
slice is ~0.2-0.3 ms/token; the tissue is where the time lives).

## Chained drafting and lane-batched decoding

Two throughput mechanisms, both bit-identical to plain decoding and both
regime-dependent - the regime is stated with the numbers.

**Chained MTP drafting.** `--mtp-depth N` (default 4) chains the draft
head: each proposal's own final-normed hidden feeds the next step (the
reference proposer's multi-step contract), and one batched trunk pass
verifies the pending token plus the whole chain, with the standard
speculative bonus token at the mismatch position. Draft argmax runs
through a frequency-sliced head (`MICROKIMI_MTP_MINIHEAD` rows, default
32768, 0 = full): BPE ids are roughly frequency-ordered, so the first
rows plus the special block agree with the full argmax on most steps,
and the full-head verification corrects the rest - the head choice moves
the acceptance rate, never the output. Speculation pays where a
verification batch is cheaper than sequential steps, i.e. the
memory-bound regime (weights streaming against RAM or disk, loaded
machines, large models). Measured honestly on the 0.8B: 1.40x under
memory pressure, but SLOWER than plain decoding on an idle in-RAM run -
the 27B, which pages from disk, is the intended target.

**Lane-batched decoding.** `DecodeLane` gives every conversation its own
caches while `forward_lanes` steps all lanes through the layers
together: the multi-lane kernels (f32, q8 head, packed MXFP4) read every
weight row ONCE and dot it against each lane's input, so n lanes cost
close to one in weight traffic. Per-lane results are bit-identical to
single-stream decoding (tested on both variants). `microkimi lanebench
--lanes N [--ab]` measures aggregate throughput; `--ab` alternates
single-lane and N-lane phases in the same process, which removes reload
noise and is the number to trust. On the real 0.8B in this shared
container: median 1.95x aggregate at 4 lanes (never below 1.64x over
six rounds) and median 2.68x at 8 lanes, peaking at 47.6 aggregate
tok/s against 17.5 single-stream in the same window. Scaling is
sublinear here because parts of the step are compute-bound on this
host; the weight-traffic sharing grows with model size. Wiring lanes
into serve and complete-batch is the designated next step.

## Certified error-budget decoding

`MICROKIMI_MLP_BUDGET=e` turns the dense MLP into an exactness dial with
a mathematical guarantee. At load, the engine scans the packed matrices
once and stores, per 32-channel block, the L2 norms of the `up` rows and
the sup of the `down` block: after the gate matvec of each token, the
possible contribution of every block is bounded by
`sum |silu(g_i)| * |up_i|_2 * |x|_2 * sup|down_b|`, and blocks PROVEN to
fit inside the budget skip both their up rows and their down columns
through block-sparse kernels (SIMD granularity preserved). The
certificate is per-layer MLP output sup-norm; budget 0 - the default -
skips nothing and stays bit-exact, and the bound's dominance over the
true deviation is unit-tested block by block.

Measured curve on the real 0.8B (held-out NLL, 2600 scored tokens):

| budget | blocks skipped | NLL | note |
|---|---|---|---|
| 0 | 0% | 2.7961 | exact baseline |
| 2 | 0.7% | 2.7984 | within noise |
| 10 | 4.7% | 2.8278 | small drift |
| 50 | 20.7% | 3.0426 | real degradation |

The honest conclusion is written into the design: certified bounds are
conservative (Cauchy-Schwarz over the row norms), so at 0.8B the
quality-neutral budget only buys sub-percent skips - at this scale the
feature is a correctness dial, not a speed lever. Larger models carry
more redundant channels and their curve is the interesting one; the
mechanism, the kernels, and the measurement harness are ready for it.

## Timelines: version control for conversations

A conversation state is a commit. `serve` content-addresses every
post-answer state (SHA-256, verified on read) into `<model>.timelines/`
and returns its id in the chat response; the DAG this builds supports
the operations version control taught us to want, on model minds:

- **fork**: pass `"state_id"` with a single user message to continue
  from ANY past state. The engine is bit-exact, so a fork is a real
  checkout, not an approximation - a state diffed against itself
  generates byte-identical text (measured: divergence `-1`).
- **diff** (`POST /v1/timelines/diff {a, b, prompt}`): run one prompt
  greedily from two states and get both answers plus the first token
  where the universes diverge.
- **merge** (`POST /v1/timelines/merge {a, b}`): three-way merge through
  the lowest common ancestor. This is only possible on this
  architecture: 18 of 24 layers hold gated delta-rule states, which are
  sums of decayed outer products - linear objects - so the merge is
  literal arithmetic, `S = S_a + S_b - S_ancestor`, the
  inclusion-exclusion that counts shared history once. The six
  full-attention layers keep the ancestor prefix and append both branch
  suffixes (B's keys stay at their original rotary positions: a declared
  approximation), and the MTP cache is cleared.

Live on the real Qwen3.5-0.8B, first attempt: a root state memorized
"project codename Falcon"; branch A additionally learned "database
password mango42", branch B "deploy day Thursday". Asked for both facts,
branch A knew only its own (it invented a date), branch B only its own
(it guessed "Falcon" as the password). The merged state answered
`Database password: mango42 / Deploy day: Thursday` - both facts, each
learned in a different branch - and still recalled the ancestor's
"Falcon" once. Diffing the two branches on "list every fact you
remember" put the divergence exactly at the token where their universes
differ.

Limits, measured rather than hidden: merging two branches that
contradict each other (region "eu-west" vs "us-east") resolved silently
to branch A's value - the merge has an ordering bias (A's convolution
window and key/value precedence) and no conflict detection; the
positional overlap of B's appended keys is unprincipled for long
suffixes; and all of this is demonstrated at 0.8B scale on short
factual probes, nothing more is claimed.

## Measured on Qwen3.5-0.8B (real weights)

All of the above is fixture-verified; the numbers below are measured on
the real converted Qwen3.5-0.8B (dense, tied embeddings, MTP head;
2.14 GB converted) on a shared aarch64 container - treat them as
relative, not absolute:

| measurement | result |
|---|---|
| batched prefill vs sequential, 961-token prompt | `12.2` vs `109.0` ms/token (`8.9x`) |
| MTP draft acceptance (counting / prose / code prompts) | `96%` / `73%` / `98%`, outputs bit-identical to plain greedy |
| MTP net speed on the counting prompt | `746` vs `1048` ms/token (`1.40x`) |
| imatrix conversion vs plain, 4095-token held-out NLL | `2.7510` vs `2.7453` nats/token: **worse, rejected** (8192-token calibration) |
| vocab slice, 5k rows from a 120 KB frequency corpus | `+128%` tokens, `+323%` total nats on held-out text: **rejected for general text** |
| vocab slice, 11k rows covering the target domain | `1.17` vs `2.14` GB (`-45%`), `+0.23%` tokens, total nats within noise of full |
| serve, turn two of a cached conversation | `38/49` prompt tokens restored, `11` prefilled |

Two of those are negative results, kept on purpose. The importance-
weighted quantization did not pay on this model and corpus budget - the
plain conversion stays the default, and any larger model decides with
the same A/B before adopting it. Vocabulary slicing collapses on
out-of-coverage text because byte-fallback tokens are rare in
pretraining: it is a domain-lock tool whose keep-set must be built from
a corpus that covers the deployment distribution (and validated with
exactly this total-nats measurement), not a general compressor.

## Independent parity

`ref/qwen_parity.py` constructs a deterministic four-layer Transformers
checkpoint with three Gated DeltaNet layers, one full-attention layer, routed
and shared experts, and values exactly representable by the MXFP4 converter.
`--dense` builds the dense `qwen3_5_text` sibling (per-layer MLP instead of
experts) through the same steps. It writes reference logits that can be
compared with the hidden `qwen-dump` development command. Disable the
optional q8 language-model-head cache for this exact f32 comparison:

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

Against Transformers 5.14.1 on the four-token fixtures:

| check | MoE fixture | dense fixture (`--dense`) |
|---|---|---|
| maximum absolute logit difference | `8.34e-7` | `8.18e-7` |
| overall logit RMSE | `1.70e-7` | `1.85e-7` |
| top-1 tokens | 4 of 4 exact | 4 of 4 exact |

The tokenizer is independently compared with the checkpoint tokenizer on
ordinary text, source code, multilingual text, decomposed Unicode, and the
default thinking chat template. Those cases match token for token, including
against the published Qwen3.8-27B `tokenizer.json` (same 248320-entry
vocabulary and special tokens as Qwen3.5/3.6, and the same
`<|im_start|>assistant` + `<think>` generation prompt). NFC is
implemented with generated Unicode 15.1 canonical tables and algorithmic
Hangul composition, without adding a crate.

## State snapshots

`run --save state.mkmem` snapshots the conversational state after the turn:
recurrent and convolution states, key/value histories (including the MTP
draft cache), the absolute position, and the logits after the last ingested
token. `run --memory state.mkmem` restores it and continues on top -
resuming is bit-identical to never having stopped. The `MKMEMQW1`
fingerprint binds the snapshot to the model architecture and to the
composed adapter-pack set (SHA-256), so a snapshot taken with packs only
loads under the same packs.

The chat prefix cache also works on Qwen models: the state after each
turn's prompt is snapshotted in `<model>.pck/` (MKMEMQW1 images keyed by
token prefix) and a turn whose prompt extends a cached prefix resumes
from the snapshot, bit-identically - across turns and across sessions.
`MICROKIMI_NO_PCK=1` disables it; `--mtp` bypasses it (the draft pairing
starts at the prompt).

Current limits: `--stream`, early exit, and the n-gram speculative
decoder are not wired to the Qwen runtime (`--mtp` is the Qwen
speculative path; `--memory` cannot combine with `--mtp`). The default
mmap load is still demand-paged and does not copy the model file into
anonymous RAM.
