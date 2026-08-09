#!/usr/bin/env python3
"""nanokimi - eval_compare: paired statistical comparison of two .bin
models on held-out text.

Runs both models over the same documents (jsonl with a "text" field),
accumulates per-document token counts and negative log-likelihood
(UNK/BOS targets excluded), and reports:

  - corpus cross-entropy (nats/token) and perplexity for each model;
  - the paired difference with a bootstrap confidence interval, where
    the DOCUMENT is the resampling unit (tokens within a document are
    correlated; documents are the exchangeable unit);
  - a one-sided bootstrap p-value for "candidate is better than base".

The two models must share the tokenizer. Documents are selected with
--doc-range start:end (0-based line numbers in the jsonl), which makes
it easy to keep evaluation sets disjoint from any calibration range.

usage:
  python3 eval_compare.py --base a.bin --cand b.bin --text corpus.jsonl \
      --doc-range 2000:2400 --device cuda [--tiktoken p] [--vocab-nano v] \
      [--seq 512] [--batch 8] [--boot 10000] [--seed 7]
  python3 eval_compare.py --selftest
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_NANO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _NANO)


def doc_nll(model, ids, valid, unk, vocab, seq, device, batch=8):
    """(sum nll, n tokens) of one document, windows of `seq`, targets at
    invalid/UNK positions excluded."""
    import torch
    xs, ms = [], []
    for w0 in range(0, len(ids) - 1, seq):
        w1 = min(w0 + seq, len(ids) - 1)
        if w1 - w0 < 2:
            continue
        pad = seq - (w1 - w0)
        x = np.pad(ids[w0:w1], (0, pad))
        y = np.pad(ids[w0 + 1:w1 + 1], (0, pad))
        m = np.pad(valid[w0 + 1:w1 + 1], (0, pad))
        if unk is not None:
            m = m & (y != unk)
        xs.append(np.stack([x, y]))
        ms.append(m)
    if not xs:
        return 0.0, 0
    xs = np.stack(xs)  # [w, 2, seq]
    ms = np.stack(ms)
    nll, cnt = 0.0, 0
    with torch.no_grad():
        for b0 in range(0, len(xs), batch):
            x = torch.tensor(xs[b0:b0 + batch, 0], device=device)
            y = torch.tensor(xs[b0:b0 + batch, 1], device=device)
            m = torch.tensor(ms[b0:b0 + batch], device=device)
            logits = model(x).view(x.shape[0], x.shape[1], vocab)
            ce = torch.nn.functional.cross_entropy(
                logits[m], y[m], reduction="sum")
            nll += float(ce)
            cnt += int(m.sum())
    return nll, cnt


def paired_bootstrap(nll_a, nll_b, counts, boot=10000, seed=7):
    """Documents as resampling unit. Returns (delta, lo, hi, p_better):
    delta = CE_b - CE_a over the corpus, [lo, hi] the 95% interval of the
    resampled delta, p_better the one-sided p-value for delta < 0."""
    nll_a = np.asarray(nll_a, np.float64)
    nll_b = np.asarray(nll_b, np.float64)
    counts = np.asarray(counts, np.float64)
    keep = counts > 0
    nll_a, nll_b, counts = nll_a[keep], nll_b[keep], counts[keep]
    n = len(counts)
    delta = nll_b.sum() / counts.sum() - nll_a.sum() / counts.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(boot, n))
    ca = nll_a[idx].sum(axis=1)
    cb = nll_b[idx].sum(axis=1)
    ct = counts[idx].sum(axis=1)
    deltas = (cb - ca) / ct
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_better = float((deltas >= 0).mean())
    return float(delta), float(lo), float(hi), p_better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--doc-range", required=True, help="start:end lines")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tiktoken", default=None)
    ap.add_argument("--vocab-nano", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from bin2pt import convert
    from model_nano import NanoModel
    from capture_host import make_host_encoder

    encode, bos = make_host_encoder(args.tiktoken, args.vocab_nano)
    unk = None
    if args.vocab_nano:
        with open(args.vocab_nano) as fv:
            unk = json.load(fv)["specials"]["unk"]

    models = {}
    vocab = None
    for tag, path in (("base", args.base), ("cand", args.cand)):
        sd, cfg = convert(path)
        m = NanoModel(cfg)
        m.load_state_dict(sd)
        m.to(args.device).eval()
        models[tag] = m
        vocab = cfg["vocab"]

    d0, d1 = (int(x) for x in args.doc_range.split(":"))
    nll = {"base": [], "cand": []}
    counts = []
    n_docs = 0
    with open(args.text) as f:
        for i, line in enumerate(f):
            if i >= d1:
                break
            if i < d0:
                continue
            text = json.loads(line).get("text")
            if not text:
                continue
            ids, _ends, valid = encode(text)
            if bos is not None:
                ids = np.concatenate([[bos], ids])
                valid = np.concatenate([[False], valid])
            c = None
            for tag in ("base", "cand"):
                s, k = doc_nll(models[tag], ids, valid, unk, vocab,
                               args.seq, args.device, args.batch)
                nll[tag].append(s)
                c = k
            counts.append(c)
            n_docs += 1
            if n_docs % 50 == 0:
                print(f"  {n_docs} docs...", flush=True)

    tot = sum(counts)
    ce_a = sum(nll["base"]) / tot
    ce_b = sum(nll["cand"]) / tot
    delta, lo, hi, p = paired_bootstrap(nll["base"], nll["cand"], counts,
                                        args.boot, args.seed)
    print(f"docs {d0}:{d1} -> {n_docs} documents, {tot} scored tokens")
    print(f"base: CE {ce_a:.4f} nats, ppl {np.exp(ce_a):.2f}")
    print(f"cand: CE {ce_b:.4f} nats, ppl {np.exp(ce_b):.2f}")
    print(f"delta (cand - base): {delta:+.4f} nats, "
          f"95% CI [{lo:+.4f}, {hi:+.4f}], "
          f"one-sided p(cand not better) = {p:.4f}")
    verdict = "BETTER" if hi < 0 else ("worse" if lo > 0 else "inconclusive")
    print(f"verdict at 95%: {verdict}")


# ------------------------------------------------------------------ selftest

def selftest():
    rng = np.random.default_rng(2)
    n = 200
    counts = rng.integers(200, 1500, n)
    base_ce = 5.0 + 0.3 * rng.normal(size=n)
    nll_a = base_ce * counts

    # candidate genuinely better by 0.05 nats/token (with noise)
    nll_b = (base_ce - 0.05 + 0.02 * rng.normal(size=n)) * counts
    d, lo, hi, p = paired_bootstrap(nll_a, nll_b, counts)
    assert hi < 0 and p < 0.01, (d, lo, hi, p)
    print(f"claim 1: a real -0.05 nats effect is detected "
          f"(delta {d:+.4f}, CI [{lo:+.4f}, {hi:+.4f}], p {p:.4f})")

    # no effect: interval straddles zero on pure noise
    nll_c = (base_ce + 0.002 * rng.normal(size=n)) * counts
    d2, lo2, hi2, p2 = paired_bootstrap(nll_a, nll_c, counts)
    assert lo2 < 0 < hi2, (lo2, hi2)
    print(f"claim 2: a null effect stays inconclusive "
          f"(CI [{lo2:+.4f}, {hi2:+.4f}])")

    # determinism with a fixed seed
    d3 = paired_bootstrap(nll_a, nll_b, counts)
    d4 = paired_bootstrap(nll_a, nll_b, counts)
    assert d3 == d4
    print("claim 3: bootstrap is deterministic under a fixed seed")

    print("eval_compare selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
