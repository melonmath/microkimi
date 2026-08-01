#!/usr/bin/env python3
"""Generate ref/ds_tok_golden.json: reference token ids for a battery of test
strings, produced by the official HF `tokenizers` runtime on the real
DeepSeek-V4 tokenizer.json. The Rust selftest (run_ds4) must reproduce these
ids EXACTLY.

Run: /home/node/venv/bin/python3 ref/make_ds_tok_golden.py
"""
import json
import os
import tokenizers

_HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(_HERE, "..", "microdeepseek.tokenizer.json")
if not os.path.exists(TOK):
    TOK = "/tmp/dsv4_hf/tokenizer.json"
OUT = os.path.join(_HERE, "ds_tok_golden.json")

STRINGS = [
    "Hello world!",
    "The quick brown fox jumps over the lazy dog.",
    "abc12345de",
    "3.14159 and 1000000 dollars",
    "it's a test we're doing they've done I'll go you'd",
    "#include <stdio.h>\nint main() { return 0; }",
    "  spaces  and\nnewlines\r\nwindows",
    "trailing spaces   ",
    "中文测试1234混合english",
    "日本語のテキストと한국어",
    "mixed 中文 and english 99 bottles",
    "price: $3.50 ≈ €3.20 ± 0.30 (100%)",
    "email@example.com https://example.com/path?q=1",
    "a\tb\vc\fd",
    "emoji 😀🎉 test",
    " once upon a time",
    "once  upon   a    time",
    "\n\n\n",
    "word\n\n\nword",
    "MiXeD CaSe WoRdS",
    "don't stop believin'",
    "1 12 123 1234 12345 123456",
    "...and...more...dots!!!",
    "special <｜User｜> lookalike text",
]

tk = tokenizers.Tokenizer.from_file(TOK)
golden = []
for s in STRINGS:
    enc = tk.encode(s, add_special_tokens=False)
    golden.append({"text": s, "ids": enc.ids})

# seeded random unicode soup: letters, digits, CJK, punctuation, symbols,
# whitespace/newlines, emoji - exercises the stage transitions hard
import random
rng = random.Random(20260801)
ALPH = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    + list(" \t\n\r")
    + list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    + list("中文测试日本語 한국어 Émilie naïve")
    + ["😀", "🎉", "→", "±", "€", "¥", "©", "®", "«»", "、。", "①②③"]
)
for _ in range(50):
    s = "".join(rng.choice(ALPH) for _ in range(rng.randint(3, 40)))
    enc = tk.encode(s, add_special_tokens=False)
    golden.append({"text": s, "ids": enc.ids})

with open(OUT, "w") as f:
    json.dump(golden, f, ensure_ascii=False)
print(f"written {OUT} ({len(golden)} strings)")
