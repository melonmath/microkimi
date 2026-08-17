#!/bin/bash
# Paired CPU duel: microkimi vs llama.cpp, N interleaved rounds.
# Every comparison lives inside one thermal window; quote medians.
# Usage: WORK=/path/with/models bash scripts/cpu-duel.sh 7
#   expects $WORK/q08.bin (converted), $WORK/q08.gguf (ggml-org Q8_0),
#   $WORK/llama.cpp/build/bin/llama-bench, and microkimi built in-tree.
set -e
N="${1:-7}"
WORK="${WORK:-.}"
MK="$(dirname "$0")/../target/release/microkimi"
LB="$WORK/llama.cpp/build/bin/llama-bench"
P=$(python3 -c "print('The history of computing spans mechanical calculators, vacuum tubes, transistors, integrated circuits, and modern accelerators. '*40)")
echo "round,mk_decode_toks,lc_decode_toks,mk_prefill_ms,lc_prefill_toks"
for i in $(seq 1 "$N"); do
  A=$(MICROKIMI_THREADS=6 MICROKIMI_Q8_SPINE=1 "$MK" run "The industrial revolution transformed European cities because" --model "$WORK/q08.bin" --raw --max-new 32 2>/dev/null | grep -o "[0-9.]* tok/s" | tail -1 | cut -d' ' -f1)
  B=$("$LB" -m "$WORK/q08.gguf" -n 32 -p 0 -r 1 -t 4 2>/dev/null | grep tg32 | grep -oE "[0-9]+\.[0-9]+ ±" | head -1 | cut -d' ' -f1)
  C=$(MICROKIMI_THREADS=10 MICROKIMI_Q8_SPINE=1 "$MK" run "$P" --model "$WORK/q08.bin" --raw --max-new 2 --debug 2>/dev/null | grep -o "([0-9.]* ms/token)" | head -1 | tr -d '(' | cut -d' ' -f1)
  D=$("$LB" -m "$WORK/q08.gguf" -p 1024 -n 0 -r 1 -t 10 2>/dev/null | grep pp1024 | grep -oE "[0-9]+\.[0-9]+ ±" | head -1 | cut -d' ' -f1)
  echo "$i,$A,$B,$C,$D"
done
