# Kimi K3 (via microkimi)

**microkimi implements the Kimi K3 architecture, unchanged** - only tensor dimensions are scaled down to fit in RAM. Layer counts, expert counts, mechanisms, tokenizer: identical.

| component                       | nanokimi-0.2b              | microkimi-debug                | real K3              |
| ------------------------------- | -------------------------- | ------------------------------ | -------------------- |
| layers                          | 8-12                       | **93 (same)**                  | 93 (69 KDA + 24 MLA) |
| hidden                          | 512                        | **512**                        | 7168                 |
| vocab                           | 8 192 + 8 specials (remap) | **163 840 (real tokenizer)**   | 163 840              |
| KDA heads × dim                 | 4 × 128                    | **4 × 128**                    | 96 × 128             |
| MLA (NoPE) heads                | 4                          | **4** (q_lora 128, kv_lora 64) | 96                   |
| experts routed / top-k / shared | 896 / 16 / 2               | **896 / 16 / 2 (same)**        | 896 / 16 / 2         |
| expert hidden / inter           | 128 / 64                   | **128 / 64**                   | 3584 / 3072          |
| AttnRes block size              | 4                          | **12**                         | 12                   |
| expert storage                  | MXFP4                      | **MXFP4 (dequant on the fly)** | MXFP4                |

Mechanisms, implemented exactly:

- **KDA (Kimi Delta Attention)** - linear-attention recurrence, per-channel learned decay, delta-rule update; fixed-size state, no growing KV cache (million-token contexts).
- **MLA (Multi-head Latent Attention)** - keys/values compressed into a small shared latent; no positional encoding at all.
- **Fine-grained MoE** - 896 routed experts per layer (16 active + 2 shared), sigmoid router with score-correction bias.
- **AttnRes** - residual blocks re-mixed by attention every 12 layers.
- **SiTU** activations; **MXFP4** expert storage, dequantized on the fly.

What is scaled down: tensor dims (for RAM) and the training budget - not the architecture. Given the real dims, data and GPU hours, this same code path is built to run the same computation.

## Verified 1:1 against Moonshot's code

`paritytest` drives Moonshot's own `KimiDecoderLayer` / `KimiDeltaAttention` / `KimiMLAAttention` / `KimiSparseMoeBlock` classes (downloaded at runtime, fp32) with the same weights, layer by layer, and diffs against the Rust forward:

| check                                                    | result                                     |
| -------------------------------------------------------- | ------------------------------------------ |
| router top-16 indices (layers 1, 47, 92, all positions)  | **exact match**                            |
| final logits                                             | max_abs 5.8e-5 (f32 summation-order noise) |
| per-layer hidden states + KDA/MoE sub-blocks             | 1e-9 … 3e-5                                |
| mechanism goldens (KDA recurrence, SiTU, MXFP4, AttnRes) | pass at 1e-4                               |

`paritytest --show` prints the concrete side-by-side values.

```bash
./target/release/microkimi build       # assemble microkimi-debug.bin (~2.5 GB, K3 fetch + Qwen pools)
./target/release/microkimi selftest    # mechanism self-tests (torch once: ref/make_golden.py)
python3 ref/parity_ref.py              # regenerates ref/parity_golden.json
./target/release/microkimi paritytest  # the 1:1 proof above
```

## nanokimi-0.2b: from noise to stories

Same engine, same greedy decoding, same prompt - only the weights change:

| model          | weights                      | `"Once upon a time, there was a little girl named Lily."`                                                                                                                                 |
| -------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| microkimi-debug | synthetic, untrained        | `增进食欲蚕食蚕食蚕食蚕食…`                                                                                                                                                               |
| nanokimi smoke | 200 steps (~0.1 M tokens)    | `". She wanted to play with the toy, but the park was very happy."` - grammatical, but the _same sentence for every prompt_                                                               |
| **nanokimi-0.2b** | **560 steps (4.3 M tokens)** | `" She loved to play with her toys and her favorite toy was a big, red ball. One day, Lily's mom asked her to help clean the park. Lily was so happy and excited to play with the ball."` |

More prompts (raw, unedited): `"Tom was a small boy who loved to play outside."` → `" One day, he went to the park to play with his friends."` · `"One day, a cat named Whiskers found a shiny red ball."` → `" He wanted to play with it, but he was too small… He asked his friend, a little bird, to help him."` The router learned too: expert usage is differentiated without collapse (top expert ≈ 8 % of calls), visible live with `--debug-routing`.

### The training run

|          |                                                                                                                |
| -------- | -------------------------------------------------------------------------------------------------------------- |
| VM       | GCP e2-highcpu-32 spot (32 vCPU, 31 GB RAM, **no GPU**, ~$0.5/h)                                               |
| Corpus   | TinyStories V2, real Kimi tiktoken remapped to top 8 192 tokens (99.95 % coverage), 150 M tokens prepared      |
| Model    | 181.6 M params - 8 layers (6 KDA + 2 MLA), 896 experts top-16 + 2 shared, AttnRes (~25 M active/token)         |
| Recipe   | AdamW lr 3e-4 cosine, batch 32 × 256, fp32, gradient checkpointing, grouped-GEMM MoE (bit-exact vs naive loop) |
| Result   | 560 steps, 4.3 M tokens, 501 min (~140 tok/s), **loss 9.13 → 2.5586** (perplexity ~8200 → ~13)                 |
| Survived | one OOM (found and fixed), one GCP preemption (watchdog + atomic checkpoints + `--resume`)                     |

Only 3 % of the corpus was seen and the loss was still descending at the end - the full corpus is ~12 days on the same box, or ~1 day on 12 with data parallelism. The pipeline is `nano/`: `prepare.py`, `model_nano.py`, `train.py`, `export.py`, `ops/` (preemptible-proof watchdog).

Training also runs on Apple Silicon GPUs: `train.py --device mps` (torch MPS backend, plus `--device cpu|auto` and `--bench N` to compare). Honest caveat: the KDA recurrence is a sequential Python loop, so MPS does not help there - measure with `--bench` before choosing.

## What's new in 0.4.0

### Suffix-automaton speculative decoding (`--spec-rosa N`)

An online suffix automaton built over the committed token stream: unlike `--spec N` (longest 2..=8 suffix, most recent occurrence), the context length is unbounded and continuations are frequency-ordered, with an exponential decay (halflife 256 fed tokens) so recent repetition wins and cold history fades without structural pruning. Greedy only; acceptance is exact rejection sampling, so the output is bit-identical to plain greedy. Combined with `--stream`, the drafted tokens drive a draft-aware expert prefetch (`MICROKIMI_DRAFTPREFETCH=0` disables): the recorded routing of each drafted token's source occurrence predicts 85% of the verification-pass picks on the smoke model, cutting demand misses at tight cache budgets (measured with a simulated 2 ms disk: 5667 to 4881 misses at a 4 MB cache, 2512 to 2233 at 8 MB with `--spec 8`; parity at the default 512 MB, where everything is cached anyway). `MICROKIMI_DRAFTSTATS=1` reports the predictor's per-pass replay accuracy.

### Chunked KDA prefill (invisible, no flag)

The prefill recurrence moved to a chunked WY/UT form: 64-token chunks computed as small dense GEMM-shaped passes, with only a mini-scan over chunk boundaries left sequential (T/64 steps instead of T). Decode keeps the sequential single-token step; short prompts keep the old loop. Not bit-identical by design (running-product decays reassociate the multiplications): deviation vs the sequential recurrence measured under 1e-4 in the unit test. `MICROKIMI_NO_KDACHUNK=1` forces the sequential path everywhere. Note the interaction with the chat prefix cache below: pck pins the sequential loop to keep resumed turns bit-identical.

### Spine warm-up and per-region madvise (invisible, no flag)

The mmap no longer gets a whole-file `MADV_RANDOM` - that hint killed kernel readahead on the spine, which is swept sequentially once per token (demand paging dropped to a few MB/s instead of 100-200 MB/s sequential). In stream mode the mapping serves only the spine (experts go through the stream engine's own direct-I/O fd): no `RANDOM` at all, and when RAM fits the spine, `MADV_WILLNEED` on the spine gaps pages it in the background at load - a `memory: spine warm-up on` line reports it. In full-load mode, `RANDOM` is advised on the merged expert spans only.

### LUT GEMV for codebook weights (invisible unless disabled)

The matvec over codebook-quantized weights runs as table lookups and additions, no multiplies: per activation group, every product the codebook can produce is precomputed into an L1-resident table, then the main loop is one byte load, one table load, one add per group. Accumulation order is unchanged, so results are bit-identical to the legacy gather-dot loop (unit-tested at expert dims); it is the default path for VQ1 cold experts and shadow serving. `MICROKIMI_LUTGEMV=0` restores the legacy loop. Bench (aarch64, 7168x2048): VQ1 7.06x vs gather+dot, 3.7-6.2x at 512..2048 x 1408 expert dims.

### q8 MLA KV cache + all-heads kernel

The MLA KV cache is the only state that grows with context. It is now stored q8_0 by default: the latent part of K and all of V are quantized per 32 (int8 + f32 scale) at append, the rope part of K stays f32 and is stored once per position instead of per head - 1408 B vs 5120 B per position per layer at micro dims (/3.6; -12 MB VmHWM measured on a 1283-token prompt). The latent Q.K dot runs in integer (exact int32). The all-heads (MQA-style) kernel streams the cache exactly once per token instead of once per head; it is bit-identical to the per-head loop by construction (same int32 dots, same f32 ops in the same order per head, verified `to_bits` in the selftest). Bench (real K3 head count, synthetic cache): 16k positions 351 to 79.7 ms (4.4x), 72% faster than the f32 kernel; at 64k the f32 cache (8 GB) does not fit in RAM at all. At short context the f32 kernel still wins - the byte savings only pay once both paths are bandwidth-bound. Toggles: `MICROKIMI_NO_KVQ8=1` restores the f32 cache, `MICROKIMI_NO_MQA=1` the per-head loop, `MICROKIMI_KV_HADAMARD=1` adds a Hadamard rotation before quantization (measured NOT to pay on these distributions, so opt-in and off). Full greedy runs produce bit-identical answers f32 vs q8. mkmem snapshots stay f32.

### q8 lm_head (`MICROKIMI_Q8HEAD=0`)

The final logits projection (vocab x d) is the largest single matvec of the engine and re-read the whole f32 lm_head every token. At load time the tensor is requantized row-wise to q8_0 and the projection runs on the integer block dot: ~3.5x fewer bytes streamed per token. The .bin format is unchanged. Not bit-identical to the f32 matvec (q8 rounding, max error ~5e-3 of the logit span), but greedy parity is verified on nanokimi (30 tokens x 3 prompts, identical with and without). Bench (10 threads): 65536x7168 15.3 to 5.1 ms/call (3.0x). `MICROKIMI_Q8HEAD=0` keeps the exact f32 path; skipped under `--gpu`.

### Routing count-min sketch (`routestats` / `cmsinfo` / `MICROKIMI_ROUTECMS`)

Every (layer, expert) pair the router actually selects is recorded into a fixed 4 x 4096 u32 count-min sketch (64 KB) instead of the full request stream `MICROKIMI_TRACE` logs; the estimate is the min over rows, never below the true count. `microkimi routestats "prompt" [--model X.bin] [--max-new N] [--out routecms.bin]` runs one turn armed and saves the sketch; `MICROKIMI_ROUTECMS=sketch.bin` arms it on run/chat/prefill/absorb; `microkimi cmsinfo sketch.bin` prints the top-50 (layer, expert, count) plus the coverage curve. Intended use: hot/cold expert tiering from real traffic - it is the frequency source for `slice --expert-order=frequency` below.

### Logit lens (`--logit-lens` / `--logit-lens-all`)

Each post-layer hidden state is projected through the final rmsnorm + lm_head and the top-5 softmax tokens are printed one line per layer, showing where in depth the next-token semantics emerge or die. `--logit-lens` fires once at the last prefill position, `--logit-lens-all` on every generated token. Read-only (captured hiddens are clones, the forward math is untouched); the bottom row reuses the model's own logits, so it is bit-identical to the normal candidates. Compatible with `--stream` and `--dump-hidden`. This is also the diagnostic used to localize slice damage before healing (see below).

### Frequency-ordered experts (`slice --expert-order=frequency --route-cms SKETCH`)

Two levels. Level 1 (runtime, no flag): the offset-sorted expert misses of a MoE layer are grouped into maximal runs of file-adjacent blobs and fetched as ONE span read per run instead of one pread per expert - on a latency-bound disk (~1700 4K IOPS) each pread pays the fixed request latency. `MICROKIMI_NO_RUNFUSE=1` disables. Level 2: natural adjacency is 0% on the nano checkpoints (their routers only pick odd expert ids), so `microkimi slice --expert-order=frequency --route-cms SKETCH` physically rewrites the expert blobs hottest-first per MoE layer, frequencies taken from a routing sketch. Router gate rows and bias are permuted with the same order, so ids are simply relabeled: the model is unchanged, any engine reads the reordered .bin, old .bins keep working. Measured (smoke model, simulated 2 ms disk, reordered .bin): prefill 0.33 s vs 4.22 s (12.8x); decode 250 vs 376 ms/token at a 1 MB cache (1.5x, 3.26 experts per read). Same traffic as the per-expert path at every budget.

### Shadow fallback (`--stream-fallback`, DEGRADED mode, default OFF)

On an expert cache miss the decode used to block on the disk tier. With the fallback active, the engine immediately serves the expert's resident 0.5-bit VQ1 shadow (LUT GEMV, microseconds) and refills the full-precision mxfp4 blob in the background: the miss latency becomes bounded, the decode never stops on the disk. `microkimi shadow --model X.bin [--out X.shadows]` builds the `<model>.shadows` sidecar (one global 256x16 codebook + the VQ1 index bytes of every expert; 9.2 MB vs 78.1 MB of mxfp4 expert bytes on the smoke model, 8.5x smaller). Honest status: this is a latency mode, not a quality mode - a shadow-served token is NOT bit-identical, which is why it is opt-in only (`--stream-fallback` or `MICROKIMI_STREAM_FALLBACK=1`) and the stream report counts the degraded expert computations. Measured (smoke model, simulated 5 ms disk, 4 MB cache, cold): 376 ms/token without vs 3 ms/token with, prefill 3.90 s to 0.02 s. Quality cost of a fully-shadowed expert set (forced 100% shadows, 53-token perplexity): NLL 5.0685 vs 5.1704 reference on this barely-trained checkpoint - no measurable quality loss here. `eval` honors `--stream-fallback`, so the cost is measurable on the scorecard of any checkpoint.

### Chat prefix cache (`microkimi pck`, `MICROKIMI_NO_PCK`)

The K3 state after N tokens (fixed-size KDA states + conv windows + MLA cache) fully determines the continuation, so the state after each chat turn's prompt is snapshotted into `<model>.pck/` (`MICROKIMI_PCK_DIR` overrides). A turn whose prompt extends a cached prefix resumes from the snapshot and only prefills the new tokens; a repeated chat (new process, same history) prefills nothing at all. Strict invalidation: the full token list is stored and compared, an entry is only used on an exact match, longest prefix wins. Bit-identical to a full prefill (the pck chat pins the sequential KDA loop, because the chunked recurrence reassociates per 64-token chunk and a resume moves the chunk boundaries - verified: cold-cache and warm-cache runs produce byte-identical states and answers). Best effort: any I/O or parse problem degrades to a full prefill. Bypassed in raw mode, with `--memory`, and with `--spec`/`--spec-rosa`. `microkimi pck --info` / `--clean [--model X.bin]` lists / purges entries. Measured (smoke model, repeated 2-turn chat): total prefill work ~490 ms to ~4 ms, whole-run wall 1.34-1.55 s vs 2.18-2.24 s. Default on; `MICROKIMI_NO_PCK=1` disables.

### Cross-session trace-similarity prefetch (`MICROKIMI_TRACESIM=1`, default OFF)

The Markov/lookahead predictors are blind at session cold start (no transitions observed) and after a mid-session topic change (the decayed statistics describe the old topic). With tracesim, each session's routing signature (one compact per-MoE-layer expert histogram) is appended to `<model>.routes` (capped at 32 sessions); a running session is matched by cosine similarity (gate 0.15, the prompt prefill routing suffices) and the matched session's top experts for the NEXT layer are prefetched during a 50-pass cold window - chained per layer, because a one-shot warm of all layers was measured strictly worse (it thrashes a tight cache before the decode reaches the later layers). A rupture check restarts the window on a topic change. Output-preserving: only WHEN bytes land in the cache changes, greedy output verified bit-identical with/without. Measured (4 real traces, leave-one-out stores, hit rate over the first 50 token passes, LRU baseline): cap 128 entries 24-25% to 37.5-39.1% (+13.6 to +13.9 pts), cap 256 +9.5 to +9.6 pts, cap 512 +1.4 to +1.6 pts, cap 1024 ~0. Caveat: the smoke checkpoint's routing is highly stereotyped across prompts (full-session cosine 0.99), so cross-session transfer is near-ideal here; the threshold gate is what protects mismatched stores on less collapsed models. Offline A/B: `microkimi routebuild store.routes trace.bin` builds a store from traces, `microkimi cachereplay trace.bin --tracesim store.routes` replays the exact engine policy.

### ARC eviction (`MICROKIMI_CACHE=arc`)

A scan-resistant alternative to the default LFU for the expert cache: resident entries split into T1 (referenced once) and T2 (referenced twice), ghost lists keep the keys of recent evictions, and the T1 target self-adjusts on ghost hits - a pure scan touches T1 only and cannot flush the T2 working set. Byte-budgeted (expert blobs vary: MXFP4 vs VQ1). Selected with `MICROKIMI_CACHE=arc|lru|lfu`. Honest status: ARC is NOT better than the default in our measurements. On a real 29726-request trace of the smoke model it beats plain LRU where it should (+10 to +14 points at 128-256 entries, the scan-resistant regime) but never beats LFU at any capacity: expert reuse is bimodal and frequency is the stronger signal on this workload. So the default stays lfu; arc is available, same status as the lru toggle. Bit-identical output across policies.

### Memory pack decay and merge (`microkimi decay` / `microkimi merge`)

`microkimi decay mem.mkmem --half-life H --out mem2.mkmem [--units U]` implements exp2 partial forgetting: every KDA recurrent state is scaled by 2^(-U/H), so after H units of age a pack keeps half of its state magnitude. Conv windows, MLA caches and the stored logits are untouched (the conv windows expire on their own as new tokens stream in; one global age has no meaning for per-position MLA latents). `microkimi merge a.mkmem b.mkmem --alpha A --out m.mkmem` is an experimental linear blend alpha*A + (1-alpha)*B over the KDA states AND the conv windows (MLA caches and logits come from file A). It is NOT a semantic merge: the KDA recurrence is not linear in its inputs, so alpha 0 and 1 are exact and anything in between is an off-distribution experiment - consistent with the merge-is-destructive measurement in the memory packs section below.

### LoRA healing at full scale (`nano/heal_stream.py` + `nano/apply_lora_bin.py`)

Healing a sliced model at full scale could not go through `bin2pt.py`: dequantizing every MXFP4 expert to fp32 at v3 scale (96 GB .bin) means a 130+ GB checkpoint that fits nowhere. `heal_stream.py` reads the .bin directly: spine tensors as mmap views (fp32, frozen), routed experts kept PACKED in mxfp4 and dequantized chunk by chunk on the compute device inside the checkpointed expert block; the model is built on the meta device and ONE decoder layer at a time is streamed to the GPU (re-streamed and recomputed in backward). Neither the giant checkpoint nor a full fp32 model is ever materialized; only the LoRA adapters (rank 8 on the attention projections by default) and their AdamW states live in VRAM. `apply_lora_bin.py` merges the checkpoint into a COPY of the .bin by patching the fp32 attention tensors in place (W += B A alpha/rank): no dequantization, no requantization, byte-identical everywhere else. Smoke (80 steps, lr 1e-3): loss ema 5.0725 to 4.9861 vs 4.9850 for the classic bin2pt + train.py path on the same data; the patched .bin loads and generates in the Rust engine. Two scoping flags, for wounds localized by the logit lens: `--lora-layers 19-21` restricts adapters to a layer range, `--lora-final-norm` trains the trunk norms only.

### Seam adapter (`--seam-adapter RANK`, `--seam-after N`)

For sliced models whose layer N+1 was trained against a different predecessor (the v3 slice "0-11,83-92" seams original layer 11 to 83, renumbered 11 to 12): a low-rank correction h' = h + B A h applied to the residual stream right after layer N, with B zero-initialized (exact identity at step 0). The adapter is resident on the compute device and rides the checkpoint. At merge time, `apply_lora_bin.py` folds W <- W + W B A (computed in float64) into the direct input projections of layer N+1 (KDA: q/k/v/f_a/b/g, MLA: q_a/kv_a/g). Exact at zero-init (byte-identical .bin), but only approximate once trained: the residual pass-through part of the correction has no weight to fold into (RMSNorm non-linearity, stream carried to deeper layers and AttnRes blocks). Measured on the smoke model: forward max|logit diff| 5.7e-6 at max|BA| 1e-8, but already 0.12-0.21 (rel 7e-3-1.3e-2) at max|BA| 4.4e-4, growing linearly - above the 1e-3 relative budget, so a TRAINED seam adapter is refused by default with a clear message; `--force-seam-fold` overrides, experimentation only, not a deployment artifact.

Exact deployment: `apply_lora_bin.py --write-seam` embeds the adapter in the .bin instead of folding it - still plain MKIM0002: the config JSON gains `seam_after` (u32) and the directory two fp32 tensors, `seam.A` [rank, hidden] and `seam.B` [hidden, rank], appended after the existing blobs (every other blob copied byte for byte, LoRA then patched in place as usual). The Rust engine loads the pair (a few hundred KB) and applies h += (h @ A^T) @ B^T right after layer `seam_after`, through the same bit-exact dot() as every other matvec, in the batched prefill and in the decode (and so in the --spec verification and --stream paths too); the load line shows `seam: adapter rank R after layer N`. Compatibility: a .bin without seam is unchanged; a seam .bin read by an OLD engine still loads (unknown config keys and directory entries are never accessed) but generates WITHOUT the adapter. Verified on the smoke model (rank 64 after layer 3, 3 training steps): a zero-init adapter yields a bit-identical engine state (caches + logits, prefill and decode); a trained adapter adds no measurable drift over the pre-existing Rust/Python engine gap on this model (layer-3 hidden rel diff 2.2e-3 with the seam vs 2.6e-3 without; final logits 1.6e-2 vs 1.0e-2 on the LoRA-merged baseline - that baseline gap starts at the first MoE layer and exists with no seam involved). Inconsistent files fail at load with a clear message (missing tensor or config key, seam_after out of range, bad shape/dtype: `seam_tests` in src/model.rs).

### Training opts (`NANO_KDA_CHUNKED` / `NANO_ACT_OFFLOAD` / `NANO_PRETRANSPOSE`)

Three opt-in accelerations of the nano training stack, all default OFF, safe to combine:

- `NANO_KDA_CHUNKED=1`: chunkwise (UT-transform) form of the training recurrence - the sequence is blocked into `NANO_KDA_CHUNK`-token chunks (default 64) and computed as batched matmuls, backward included, pure PyTorch, CPU and CUDA alike. Not bit-identical by design (per-chunk exp(cumsum) decays instead of per-token products): measured deviation vs the reference loop <= 1.2e-5 relative over H = 4..64, T = 1..600 (tolerance 1e-4); CPU fwd+bwd at T=512 ~1.6-1.9x, larger on GPU where the reference loop is launch-bound. On any error or non-finite output the path warns, disables itself and falls back to the reference recurrence for the rest of the process.
- `NANO_ACT_OFFLOAD=1`: the KDA time-segment checkpointing (`NANO_KDA_SEG`) stashes each segment's inputs in a ring of pinned host buffers (async D2H on a side CUDA stream during forward, H2D back before the recompute) instead of retaining them in VRAM. Same math: outputs and grads bit-identical (nano/tests/test_kda_offload_parity.py). Same error guard as NANO_KDA_CHUNKED.
- `NANO_PRETRANSPOSE=1`: the frozen LoRA-targeted base weights get a cached contiguous W^T copy in (pinned) host RAM, streamed alongside W for the layer window only, so neither the forward nor the backward matmuls feed a transposed view of a freshly streamed weight to the BLAS. Smoke cross-check (CPU, 3 steps, both flags ON vs the reference path): losses match to ~1e-5 relative.

## Memory packs: save states for a neural network

A video-game save state, but for the model's mind. Because KDA layers carry a **fixed-size recurrent state** (unlike a Transformer KV cache, which grows forever), the entire working state of a conversation can be snapshotted to a small portable file - an emulator save-state for an LLM.

```bash
# snapshot at any point (during a run):
microkimi run "One morning, a little girl found a key." --model nanokimi-0.2b.bin --max-new 10 --save fork.mkmem
# resume from the snapshot - the continuation is bit-exact:
microkimi run "" --model nanokimi-0.2b.bin --memory fork.mkmem --max-new 10
# absorb a document into the state, no context window involved:
microkimi absorb notes.txt --out pack.mkmem --model nanokimi-0.2b.bin
microkimi run "What did my notes say?" --model nanokimi-0.2b.bin --memory pack.mkmem
```

What a pack contains: the KDA recurrent states + conv states (**fixed size, never grows** - on nanokimi: ~1.5 MB whatever the absorbed length), the MLA k/v caches (these do grow, ~10 KB/token on nanokimi - the KDA part is the constant one), and the last logits for exact continuation.

Measured behavior (honest status):

- **Fork/resume is bit-exact** - a resumed run reproduces the uninterrupted run token for token. Save-scumming conversations works today.
- **Absorb injects measurable information** - with a pack loaded, 30-50% of greedy token choices change vs the same prompt/seed without it. At 0.2B, no exploitable facts are retrievable yet (the toy model cannot re-read its own compressed state); retesting on microkimi-1b is planned.
- **Merge is destructive at 0.2B** (`mkmem-div` measured it) - two independently trained states live in differently-rotated spaces; naive averaging cancels them. Alignment (re-basing) is the open follow-up.

Why this is unique: it requires a fixed-size state (KDA) and an engine that exposes it. Standard Transformers cannot do this - their "state" is an ever-growing KV cache.

## Benchmarks

Greedy decode.

| model | workload | hardware | ms/token | tok/s |
|---|---|---|---|---|
| nanokimi-0.2b | decode (8 layers, 113 MB) | 10-core ARM64 | 7.5 | ~134 |
| nanokimi-0.2b | training (batch 32×256) | 32 vCPU x86-64 | - | ~130-290 |
| microkimi-debug | decode (93 layers, 2.5 GB f32+MXFP4) | 10-core ARM64 | 59 | ~17 |
| microkimi-debug | decode (93 layers) | Apple M5, 16 GB | 34 | ~29 |
