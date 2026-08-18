#!/bin/bash
# Paired GPU duel on one host: microkimi CUDA (one GPU) against llama.cpp
# CUDA (Q4_K_M on one GPU; Q8_0 across all GPUs), alternating rounds.
# Env: MK (microkimi binary), LB (llama-bench), M (qwen3.8-27b.bin),
# G4 (Q4_K_M GGUF), G8 (Q8_0 GGUF), ROUNDS (3), GPU (0).
set -u
MK=${MK:-./target/release/microkimi}
LB=${LB:-llama-bench}
M=${M:-qwen3.8-27b.bin}
G4=${G4:-Qwen3.8-27B-Q4_K_M.gguf}
G8=${G8:-Qwen3.8-27B-Q8_0.gguf}
ROUNDS=${ROUNDS:-3}
GPU=${GPU:-0}
P=$(printf "The history of computing spans mechanical calculators, vacuum tubes, transistors, integrated circuits, and modern accelerators. %.0s" $(seq 1 11))
echo "host: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l), driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
for r in $(seq 1 "$ROUNDS"); do
  echo "== round $r: microkimi CUDA, one GPU"
  CUDA_VISIBLE_DEVICES=$GPU MICROKIMI_QWEN_CUDA=1 "$MK" run "$P" --model "$M" --raw --max-new 32 --debug 2>&1 | grep -aE "tokens \(|generation:"
  echo "== round $r: llama.cpp CUDA, Q4_K_M, one GPU"
  CUDA_VISIBLE_DEVICES=$GPU "$LB" -m "$G4" -p 256 -n 32 -ngl 99 -r 1 2>&1 | grep -E "pp256|tg32"
  if [ -f "$G8" ]; then
    echo "== round $r: llama.cpp CUDA, Q8_0, all GPUs"
    "$LB" -m "$G8" -p 256 -n 32 -ngl 99 -r 1 2>&1 | grep -E "pp256|tg32"
  fi
done
