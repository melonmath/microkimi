# vocabs

Vocabulary data lives here. Conventions:

- third-party tokenizer files (`tokenizer.json`, `vocab.json`+`merges.txt`,
  SentencePiece models) are DATA: gitignored, downloaded on demand with
  `sh nano/fetch_vocabs.sh` (qwen3, gemma3, llama32, deepseek_v3);
  `nano/vocab_cross.py` points at them by path;
- our own small vocabs that ship in releases (e.g. `vocab_nano.json`) stay
  with their model artifacts and remain gitignored as well;
- analysis outputs (keep-lists, rarity reports) go here as plain text and
  may be committed when small and meant to be reproduced.
