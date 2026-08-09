#!/bin/sh
# Download third-party reference tokenizers into vocabs/ (data, gitignored).
# Usage: sh nano/fetch_vocabs.sh
set -e
cd "$(dirname "$0")/.."

fetch() {
  out="$1"; shift
  if [ -s "vocabs/$out" ]; then
    echo "vocabs/$out: already present, skipping"
    return
  fi
  for url in "$@"; do
    echo "vocabs/$out <- $url"
    if curl -fL --retry 3 -o "vocabs/$out" "$url"; then
      return
    fi
    echo "  failed, trying next mirror"
  done
  echo "error: no mirror worked for $out" >&2
  rm -f "vocabs/$out"
  exit 1
}

fetch qwen3.tokenizer.json \
  "https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json"

fetch gemma3.tokenizer.json \
  "https://huggingface.co/unsloth/gemma-3-1b-it/resolve/main/tokenizer.json" \
  "https://huggingface.co/google/gemma-3-1b-it/resolve/main/tokenizer.json"

fetch llama32.tokenizer.json \
  "https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json" \
  "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/tokenizer.json"

fetch deepseek_v3.tokenizer.json \
  "https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json"

fetch glm4.tokenizer.json \
  "https://huggingface.co/zai-org/GLM-4-9B-0414/resolve/main/tokenizer.json" \
  "https://huggingface.co/unsloth/GLM-4-9B-0414/resolve/main/tokenizer.json"

fetch mistral_v3.tokenizer.json \
  "https://huggingface.co/mistralai/Mistral-7B-v0.3/resolve/main/tokenizer.json"

# OpenAI GPT-4o tokenizer, tiktoken rank-file format (not HF tokenizer.json)
fetch o200k_base.tiktoken \
  "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"

echo "done. verify with: python3 nano/vocab_cross.py --selftest"
