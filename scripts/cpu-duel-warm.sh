#!/bin/bash
# Warm-vs-warm paired prompt-reading duel: microkimi prefillbench (best-of-5,
# snapshot-restored) against llama-bench pp1024 (-r 5), interleaved. The
# like-for-like protocol: both harnesses warm up and repeat.
# Usage: WORK=/path/with/q08.bin+q08.gguf+llama.cpp bash scripts/cpu-duel-warm.sh 5
S="${WORK:-.}"
MK="$(dirname "$0")/../target/release/microkimi"
LB=$S/llama.cpp/build/bin/llama-bench
echo "round,mk_best_toks,mk_med_toks,lc_pp1024_toks"
for i in $(seq 1 $1); do
  A=$(MICROKIMI_Q8_SPINE=1 $MK prefillbench --model "$S/q08.bin" --reps 5 2>/dev/null | grep -oE "best [0-9.]+ ms/token \([0-9]+ tok/s\) \| median [0-9.]+ ms/token \([0-9]+" | grep -oE "\([0-9]+" | tr -d '(' | tr '\n' ',' | sed 's/,$//')
  D=$($LB -m "$S/q08.gguf" -p 1024 -n 0 -r 5 -t 10 2>/dev/null | grep pp1024 | grep -oE "[0-9]+\.[0-9]+ ±" | head -1 | cut -d' ' -f1)
  echo "$i,$A,$D"
done
