# DeepSeek-V4-Flash-0731 (microdeepseek)

The repo assembles **microdeepseek.bin** - the DeepSeek-V4-Flash architecture at micro dims (43 layers kept, 256 experts top-6 + 1 shared kept, real 129,280-token vocab; widths reduced), same zero-dependency Rust engine as microkimi.

Mechanisms, implemented exactly:

- **Hyper-Connections** (hc_mult 4, Sinkhorn 20 iters) - the residual state is 4 copies, re-mixed per layer.
- **Sparse attention** - ring window 128, overlap/dense KV compressors, lightning indexer with per-head Hadamard + FP4 QAT, attentional sink.
- **sqrtsoftplus router** with noaux_tc bias, top-6 of 256, and hash-routed first 3 layers (real `tid2eid` tables).
- **SwiGLU clamp ±10** experts stored **FP4 e2m1** (MXFP4 layout), FP8 activation QAT round-trips.
- **RoPE/YaRN** - theta 10000 (window layers) vs 160000 + YaRN factor 16 (compressed layers).

Verified 1:1 against a plain-torch replica of DeepSeek's reference `model.py` driven by the very weights of microdeepseek.bin (`dsparity`): per-layer HC hidden states match at ~1e-6 over 132 positions, router selections and top-16 logit ids are **exact**. The V4 tokenizer (byte-level BPE from the official `tokenizer.json`, hand-reimplemented 3-stage pre-tokenizer) matches the HF `tokenizers` runtime **exactly** on a 74-string battery (`selftest`).

## What is here / not here

|                          |                                                                                                                                                                                                          |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **here**                 | Full DeepSeek-V4 architecture (43 layers, Hyper-Connections, sparse attention, 256 experts top-6, sqrtsoftplus router, fp4 experts, V4 tokenizer), verified 1:1; fragments of real weights (norms, router, `tid2eid` tables, compressors, 3 real fp4 experts); the builder to assemble it. |
| **not here (yet)**       | The real DeepSeek-V4 weights (284B params - they don't fit any machine here); a _trained_ DeepSeek model. **nanodeepseek** - the trained-from-scratch V4 model, like nanokimi for K3 - is planned and will be trained once a training server is available. |
| **not here (by design)** | The DSpark speculative-decoding module (the forward stops at the 43 layers + head). |

```bash
./target/release/microkimi build --arch dsv4  # assemble microdeepseek.bin (~2 GB, V4 range-request fetch + seeded pools)
python3 ref/make_ds_parity.py                 # regenerates ref/ds_parity_golden.json (torch replica)
./target/release/microkimi parity --arch dsv4 # the 1:1 proof above
./target/release/microkimi run "The capital of France is" --model microdeepseek.bin
```

Same caveat as microkimi: output is deterministic gibberish by design (untrained synthetic weights) - the point is the engine.
