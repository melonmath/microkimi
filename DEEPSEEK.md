# DeepSeek-V4-Flash-0731 (microdeepseek-debug)

The repo assembles **microdeepseek-debug.bin** - the DeepSeek-V4-Flash architecture at micro dims (43 layers kept, 256 experts top-6 + 1 shared kept, real 129,280-token vocab; widths reduced), same zero-dependency Rust engine as microkimi.

Mechanisms, implemented exactly:

- **Hyper-Connections** (hc_mult 4, Sinkhorn 20 iters) - the residual state is 4 copies, re-mixed per layer.
- **Sparse attention** - ring window 128, overlap/dense KV compressors, lightning indexer with per-head Hadamard + FP4 QAT, attentional sink.
- **sqrtsoftplus router** with noaux_tc bias, top-6 of 256, and hash-routed first 3 layers (real `tid2eid` tables).
- **SwiGLU clamp ±10** experts stored **FP4 e2m1** (MXFP4 layout), FP8 activation QAT round-trips.
- **RoPE/YaRN** - theta 10000 (window layers) vs 160000 + YaRN factor 16 (compressed layers).

Verified 1:1 against a plain-torch replica of DeepSeek's reference `model.py` driven by the very weights of microdeepseek-debug.bin (`dsparity`): per-layer HC hidden states match at ~1e-6 over 132 positions, router selections and top-16 logit ids are **exact**. The V4 tokenizer (byte-level BPE from the official `tokenizer.json`, hand-reimplemented 3-stage pre-tokenizer) matches the HF `tokenizers` runtime **exactly** on a 74-string battery (`selftest`).

## What is here / not here

|                          |                                                                                                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **here**                 | Full DeepSeek-V4 architecture (43 layers, Hyper-Connections, sparse attention, 256 experts top-6, sqrtsoftplus router, fp4 experts, V4 tokenizer), verified 1:1; fragments of real weights (norms, router, `tid2eid` tables, compressors, 3 real fp4 experts); the builder to assemble it. |
| **not here (yet)**       | The real DeepSeek-V4 weights (284B params - they don't fit any machine here); a _trained_ DeepSeek model. **nanodeepseek-0.2b** - the trained-from-scratch V4 model, like nanokimi-0.2b for K3 - is planned and will be trained once a training server is available. |
| **not here (by design)** | The DSpark speculative-decoding module (the forward stops at the 43 layers + head). |

```bash
./target/release/microkimi build --arch dsv4  # assemble microdeepseek-debug.bin (~2 GB, V4 range-request fetch + seeded pools)
python3 ref/make_ds_parity.py                 # regenerates ref/ds_parity_golden.json (torch replica)
./target/release/microkimi parity --arch dsv4 # the 1:1 proof above
./target/release/microkimi run "The capital of France is" --model microdeepseek-debug.bin
```

Same caveat as microkimi-debug: output is deterministic gibberish by design (untrained synthetic weights) - the point is the engine.

## 0.4.0 engine features: what applies here

Most of the 0.4.0 batch (see [KIMI.md](KIMI.md)) is K3-only and does NOT apply to the DeepSeek path:

- `--stream` refuses DSV4 models - and with it everything built on the stream engine: shadow fallback (`--stream-fallback`), tracesim prefetch (`MICROKIMI_TRACESIM`), contiguous-run fusion, and the cache policies (`MICROKIMI_CACHE=arc|lru|lfu`).
- `--spec` / `--spec-rosa` are ignored on DeepSeek (a warning says so): speculative decoding is K3-only for now.
- The routing count-min sketch (`routestats`, `cmsinfo`, `MICROKIMI_ROUTECMS`) hooks the K3 noaux_tc router; DeepSeek's sqrtsoftplus router is not instrumented, and `routestats` refuses DSV4 models.
- Memory packs (absorb/decay/merge), the chat prefix cache (`microkimi pck`) and the logit lens are K3-only: DeepSeek has no KDA state to snapshot and its chat loop takes the DsModel path.
- The q8 MLA KV cache, q8 lm_head, chunked KDA prefill and LUT GEMV live in the K3 model path: DeepSeek has no KDA layers, its sparse-attention KV layout differs, and `slice` refuses DSV4 models.

What does apply: the shared infrastructure only - `build --arch dsv4`, `parity --arch dsv4`, and mmap demand-paging with the per-region madvise (RANDOM on expert spans, sequential readahead on the spine). Even `eval` is K3-only today.

## Benchmarks

Greedy decode.

| model | workload | hardware | ms/token | tok/s |
|---|---|---|---|---|
| microdeepseek-debug | decode (43 layers, 2.0 GB f32+FP4) | 10-core ARM64 | 39 | ~26 |
