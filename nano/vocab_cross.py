#!/usr/bin/env python3
"""nanokimi - cross-model vocabulary rarity analysis (vocab_cross.py)

Compares one SUBJECT vocabulary (the K3 vocab, 163840 tokens) against N
REFERENCE vocabularies to find which subject tokens are cross-model rare,
and produces keep-lists for `microkimi slice --vocab-list FILE`.

usage:
  python3 vocab_cross.py --subject /path/to/tiktoken.model \
      --refs qwen3/tokenizer.json llama32/tokenizer.json gemma3/tokenizer.json \
      [--report] [--keep-list OUT] [--freq freq.txt] [--freq-top F] \
      [--min-refs 2] [--max-keep N] [--reserved A-B] [--kimi-vocab PATH] \
      [--recompose]   # report the re-segmentation cost of the cut tokens
  python3 vocab_cross.py --selftest        # offline synthetic check

Supported input formats (auto-detected per file):
  - tiktoken .model: lines "base64(token_bytes) rank" (the K3 vocab format;
    the token bytes ARE the surface form). A 163584-entry file is the K3
    base vocab: ids [163584, 163840) are the reserved special block and are
    auto-declared structural.
  - HF tokenizer.json: model.vocab is either a dict token->id (BPE) or a
    list of [token, score] (Unigram); added_tokens are the special tokens.
  - vocab.json + merges.txt: plain GPT-style BPE dict (byte-level unicode).
  - vocab_nano.json: our nano remap (nano_to_kimi). Has no strings of its
    own: nano ids are resolved through the K3 tiktoken vocab (the subject
    itself when it is a tiktoken file, else --kimi-vocab PATH).

Canonical surface form (the heart of the tool): every token maps to the RAW
BYTES it stands for, so " France" is the same key in every vocabulary.
Mapping rules (conservative, well-established equivalences only):
  - GPT byte-level convention: token strings live in the byte-to-unicode
    alphabet (printable bytes map to themselves, space -> U+0120, newline
    -> U+010A, the remaining bytes -> U+0100..). Inverted with the exact
    standard table; any char outside the table falls back to literal UTF-8.
  - SentencePiece convention: U+2581 (▁) -> 0x20, then literal UTF-8.
    Convention is detected per vocab: BPE dicts with more U+0120 than
    U+2581 markers are byte-level, more U+2581 than U+0120 are
    SentencePiece; Unigram defaults to SentencePiece, markerless BPE to
    byte-level (identical to literal on printable ASCII anyway).
  - WordPiece: a leading "##" continuation marker is stripped (the piece
    "##ing" surfaces as "ing").
  - Byte-fallback tokens written as the whole-token string "<0xNN>" map to
    the single byte 0xNN (SentencePiece byte_fallback).
  - tiktoken tokens need no mapping: the file stores the raw bytes.
NOT normalized (deliberately): case, Unicode normalization forms, spelling
variants. Two tokens that differ in bytes are different keys.

Structural tokens (never dropped, never ranked): the subject's special
tokens, the declared --reserved range, and the 256 byte-fallback ids
(detected by content: the ids whose keys are exactly the 256 single bytes).
"""
import argparse
import base64
import json
import os
import re
import sys
import tempfile

BYTE_TOK_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
K3_BASE = 163_584  # a tiktoken vocab of exactly this size is the K3 base vocab


# ── canonical surface form ──

def _gpt_byte_to_char():
    """The standard byte-level BPE table: byte -> unicode char."""
    bs = list(range(0x21, 0x7E + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


_CHAR_TO_BYTE = {chr(c): b for b, c in _gpt_byte_to_char().items()}


def normalize(raw, convention):
    """Token string -> canonical surface bytes (see the module docstring)."""
    m = BYTE_TOK_RE.match(raw)
    if m:
        return bytes([int(m.group(1), 16)])
    if raw.startswith("##"):
        raw = raw[2:]
    if convention == "gpt":
        try:
            return bytes(_CHAR_TO_BYTE[c] for c in raw)
        except KeyError:
            pass  # char outside the byte-level alphabet: keep it literal
    elif convention == "sp":
        raw = raw.replace("▁", " ")
    return raw.encode("utf-8")


def render(key):
    """Printable-safe rendering of a canonical key (UTF-8 chars kept, the
    rest escaped: control chars via unicode_escape, raw bytes as \\xNN)."""
    out, i = [], 0
    while i < len(key):
        for n in (1, 2, 3, 4):
            try:
                ch = key[i : i + n].decode("utf-8")
            except UnicodeDecodeError:
                continue
            out.append(ch if ch.isprintable() else ch.encode("unicode_escape").decode("ascii"))
            i += n
            break
        else:
            out.append(f"\\x{key[i]:02x}")
            i += 1
    return "".join(out)


# ── vocab loading ──

class Vocab:
    def __init__(self, name):
        self.name = name
        self.keys = {}       # id -> canonical surface bytes (matchable tokens)
        self.specials = {}   # name -> id (structural special tokens)
        self.reserved = set()  # extra structural ids (--reserved / K3 block)
        self.convention = "?"

    @property
    def size(self):
        hi = max(list(self.keys) + list(self.specials.values()) + list(self.reserved))
        return hi + 1

    def structural_ids(self):
        """Specials + reserved + the 256 byte fallbacks (content-detected)."""
        out = set(self.specials.values()) | self.reserved
        single = {i for i, k in self.keys.items() if len(k) == 1}
        if {k[0] for k in (self.keys[i] for i in single)} == set(range(256)) and len(single) >= 256:
            out |= single
        return out


def _detect_convention(tokens, model_type):
    """'gpt' (byte-level unicode) vs 'sp' (SentencePiece) per vocab."""
    n_g = sum(1 for t in tokens if "Ġ" in t)
    n_s = sum(1 for t in tokens if "▁" in t)
    if n_g > n_s:
        return "gpt"
    if n_s > 0:
        return "sp"
    return "sp" if model_type == "Unigram" else "gpt"


def load_tiktoken(path):
    v = Vocab(path)
    with open(path, "r", encoding="ascii") as f:
        for ln, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                sys.exit(f"error: {path}:{ln + 1}: expected 'base64 rank'")
            tok, rank = parts
            try:
                v.keys[int(rank)] = base64.b64decode(tok)
            except Exception as e:
                sys.exit(f"error: {path}:{ln + 1}: bad tiktoken line: {e}")
    v.convention = "bytes"
    if len(v.keys) == K3_BASE:
        # K3 base vocab: the 256-id reserved special block follows
        v.reserved = set(range(K3_BASE, K3_BASE + 256))
    return v


def load_hf_tokenizer_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    model = d.get("model")
    if not isinstance(model, dict) or "vocab" not in model:
        sys.exit(f"error: {path}: no model.vocab (not an HF tokenizer.json?)")
    raw, mtype = model["vocab"], model.get("type", "?")
    if isinstance(raw, dict):
        pairs = [(tok, int(i)) for tok, i in raw.items()]
    elif isinstance(raw, list):  # Unigram: [[token, score], ...]
        pairs = [(tok, i) for i, (tok, _score) in enumerate(raw)]
    else:
        sys.exit(f"error: {path}: model.vocab is neither a dict nor a list")
    v = Vocab(path)
    v.convention = _detect_convention([t for t, _ in pairs], mtype)
    for tok, i in pairs:
        v.keys[i] = normalize(tok, v.convention)
    for at in d.get("added_tokens") or []:
        v.specials[at.get("content", f"added:{at['id']}")] = int(at["id"])
    return v


def load_gpt_vocab_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict) or not all(isinstance(i, int) for i in d.values()):
        sys.exit(f"error: {path}: not a flat token->id vocab.json")
    v = Vocab(path)
    v.convention = "gpt"  # GPT-style vocab.json + merges.txt is byte-level
    for tok, i in d.items():
        v.keys[i] = normalize(tok, "gpt")
    return v


def load_nano(path, kimi):
    if kimi is None:
        sys.exit(f"error: {path}: vocab_nano.json needs the K3 tiktoken vocab to resolve ids "
                 "(pass a tiktoken subject or --kimi-vocab PATH)")
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    table = d.get("nano_to_kimi")
    if not isinstance(table, list):
        sys.exit(f"error: {path}: nano_to_kimi missing")
    v = Vocab(path)
    v.convention = f"nano over {os.path.basename(kimi.name)}"
    unresolved = 0
    for nano_id, kimi_id in enumerate(table):
        key = kimi.keys.get(int(kimi_id))
        if key is None:
            unresolved += 1
        else:
            v.keys[nano_id] = key
    if unresolved:
        print(f"warning: {path}: {unresolved} nano ids resolve to reserved/missing kimi ids", file=sys.stderr)
    for name, i in (d.get("specials") or {}).items():
        v.specials[name] = int(i)
    return v


def load_vocab(path, kimi=None):
    """Dispatch on content: tiktoken lines / tokenizer.json / vocab.json / vocab_nano.json."""
    with open(path, "rb") as f:
        head = f.read(4096).lstrip()
    if not head.startswith(b"{"):
        return load_tiktoken(path)
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict) and "nano_to_kimi" in d:
        return load_nano(path, kimi)
    if isinstance(d, dict) and isinstance(d.get("model"), dict):
        return load_hf_tokenizer_json(path)
    return load_gpt_vocab_json(path)


# ── freq file ("<id> <count>" lines, '#' comments) ──

def load_freq(path):
    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                sys.exit(f"error: {path}:{ln + 1}: expected '<token_id> <count>'")
            try:
                counts[int(parts[0])] = int(parts[1])
            except ValueError:
                sys.exit(f"error: {path}:{ln + 1}: bad '<token_id> <count>'")
    return counts


# ── analysis ──

def analyze(subject, refs):
    """Per subject id: in how many reference vocabs its surface appears."""
    ref_sets = [set(r.keys.values()) for r in refs]
    presence = {}
    for i, key in subject.keys.items():
        presence[i] = sum(1 for s in ref_sets if key in s)
    return presence, ref_sets


def report(subject, refs, presence, ref_sets):
    struct = subject.structural_ids()
    n = len(refs)
    print("== vocabularies ==")
    all_v = [subject] + refs
    for v in all_v:
        print(f"  {v.name}")
        print(f"    {len(v.keys)} matchable tokens, {len(v.specials)} specials, "
              f"{len(v.structural_ids())} structural, convention: {v.convention}")
    print("\n== pairwise overlap (shared surface forms) ==")
    key_sets = [set(v.keys.values()) for v in all_v]
    names = ["SUBJECT"] + [f"ref{i}" for i in range(n)]
    print("        " + "".join(f"{nm:>9}" for nm in names))
    for i, ks in enumerate(key_sets):
        print(f"{names[i]:>8}" + "".join(f"{len(ks & os):>9}" for os in key_sets))
    sub_keys = key_sets[0]
    print(f"\nsubject surface forms: {len(sub_keys)} unique over {len(subject.keys)} ids")

    print("\n== presence-count distribution (non-structural subject tokens) ==")
    hist = [0] * (n + 1)
    for i in subject.keys:
        if i not in struct:
            hist[presence[i]] += 1
    tot = sum(hist) or 1
    for k in range(n + 1):
        print(f"  in {k}/{n} refs: {hist[k]:>8}  ({hist[k] / tot * 100:5.1f}%)")

    rare = [i for i in subject.keys if i not in struct and presence[i] == 0]
    print(f"\n== rare tail: {len(rare)} subject tokens in ZERO reference vocabs (up to 30) ==")
    for i in rare[:30]:
        print(f"  id {i:>7}: {render(subject.keys[i])}")

    print(f"\n== structural ==")
    print(f"  {len(struct)}/{subject.size} subject ids structurally required "
          f"({len(struct) / subject.size * 100:.2f}%): specials + reserved + 256 byte fallbacks")


# ── keep-list ──

def build_keep_list(subject, presence, counts, freq_top, min_refs, max_keep):
    struct = subject.structural_ids()
    candidates = [i for i in subject.keys if i not in struct]
    by_freq, by_refs = set(), set()
    if freq_top > 0:
        ranked = sorted(candidates, key=lambda i: (-counts.get(i, 0), i))
        by_freq = set(ranked[:freq_top])
    for i in candidates:
        if presence.get(i, 0) >= min_refs:
            by_refs.add(i)
    selected = by_freq | by_refs
    # priority under --max-keep: frequency, then refs, then id
    ordered = sorted(selected, key=lambda i: (-counts.get(i, 0), -presence.get(i, 0), i))
    keep, dropped = set(struct), 0
    slots = None if max_keep is None else max(0, max_keep - len(struct))
    if max_keep is not None and len(struct) > max_keep:
        print(f"warning: {len(struct)} structural tokens exceed --max-keep {max_keep}; "
              f"structural tokens are kept anyway", file=sys.stderr)
    for i in ordered:
        if slots is None or len(keep) - len(struct) < slots:
            keep.add(i)
        else:
            dropped += 1
    both = len(by_freq & by_refs)
    print(f"keep-list: {len(keep)} ids = {len(struct)} structural "
          f"+ {len(by_freq - by_refs)} freq-only + {len(by_refs - by_freq)} refs-only + {both} both"
          + (f" ({dropped} selected but dropped by --max-keep {max_keep})" if dropped else ""))
    return sorted(keep)


# ── recomposition cost ──

def recompose_costs(subject, keep):
    """For every CUT subject token: into how many KEPT tokens does its byte
    surface re-segment? Greedy longest-match over the kept surfaces (the 256
    byte fallbacks are structural, so this always terminates). Greedy is an
    approximation of a real BPE re-tokenization, but it is the same order of
    cost. Returns {subject_id: n_pieces}; cost per occurrence is n_pieces-1.
    """
    keep_set = set(keep)
    kept_surfaces = {subject.keys[i] for i in keep_set if i in subject.keys}
    maxlen = max((len(s) for s in kept_surfaces), default=1)
    costs = {}
    for i, key in subject.keys.items():
        if i in keep_set:
            continue
        n, p = 0, 0
        while p < len(key):
            for ln in range(min(maxlen, len(key) - p), 0, -1):
                if key[p:p + ln] in kept_surfaces:
                    p += ln
                    n += 1
                    break
            else:  # no kept surface matches, not even a single byte
                p += 1
                n += 1
        costs[i] = n
    return costs


def recompose_report(subject, keep, costs, counts):
    cut = len(costs)
    tot_extra = sum(n - 1 for n in costs.values())
    print(f"\n== recomposition cost of the {cut} cut tokens ==")
    if cut == 0:
        print("  nothing cut")
        return
    hist = {}
    for n in costs.values():
        hist[n] = hist.get(n, 0) + 1
    print("  pieces per cut token: " + ", ".join(
        f"{n} pcs x {c}" for n, c in sorted(hist.items())[:10]))
    print(f"  mean extra tokens per occurrence: {tot_extra / cut:.2f}")
    if counts:
        weighted = sum((costs[i] - 1) * counts.get(i, 0) for i in costs)
        total = sum(counts.values()) or 1
        print(f"  corpus inflation if cut: {weighted} extra tokens over {total} observed "
              f"({weighted / total * 100:.3f}%)")
        worst = sorted(costs, key=lambda i: (-(costs[i] - 1) * counts.get(i, 0), i))[:20]
        print("  most expensive cuts (freq x extra):")
        for i in worst:
            if counts.get(i, 0) == 0:
                break
            print(f"  id {i:>7}: {render(subject.keys[i])}  freq={counts[i]} "
                  f"-> {costs[i]} pcs (+{(costs[i] - 1) * counts[i]})")
    else:
        worst = sorted(costs, key=lambda i: (-costs[i], i))[:20]
        print("  hardest to recompose (no --freq: unweighted):")
        for i in worst:
            print(f"  id {i:>7}: {render(subject.keys[i])}  -> {costs[i]} pcs")


# ── selftest (offline, synthetic vocabs in a temp dir) ──

def selftest():
    tmp = tempfile.mkdtemp(prefix="vocab_cross_selftest_")
    # subject: tiny tiktoken vocab, K3-style layout: 256 single bytes, then merges
    merges = [b" the", b" France", b"ing", b" banana", "中".encode(), b"zzz"]
    lines = [f"{base64.b64encode(bytes([b])).decode()} {b}" for b in range(256)]
    lines += [f"{base64.b64encode(t).decode()} {256 + j}" for j, t in enumerate(merges)]
    subj_path = os.path.join(tmp, "subject.model")
    with open(subj_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    # ref1: byte-level BPE tokenizer.json (" the" written with U+0120) + a special
    ref1 = os.path.join(tmp, "ref1_tokenizer.json")
    with open(ref1, "w") as f:
        json.dump({"model": {"type": "BPE", "vocab": {"!": 0, "Ġthe": 1, "ĠFrance": 2, "ing": 3, "ab": 4}},
                   "added_tokens": [{"id": 5, "content": "<s>", "special": True}]}, f)
    # ref2: SentencePiece-convention tokenizer.json (" the" written with U+2581, <0xNN> bytes)
    ref2 = os.path.join(tmp, "ref2_tokenizer.json")
    with open(ref2, "w") as f:
        json.dump({"model": {"type": "BPE", "vocab": {"▁the": 0, "<0x41>": 1, "中": 2, "##ing": 3}}}, f)
    # ref3: plain vocab.json + merges.txt (byte-level)
    ref3 = os.path.join(tmp, "vocab.json")
    with open(ref3, "w") as f:
        json.dump({"Ġthe": 0, "ing": 1, "zz": 2}, f)
    with open(os.path.join(tmp, "merges.txt"), "w") as f:
        f.write("#version: 0.2\nz z\n")
    # ref4: our own nano remap format, resolved through the subject tiktoken vocab
    ref4 = os.path.join(tmp, "vocab_nano.json")
    with open(ref4, "w") as f:
        json.dump({"nano_to_kimi": [0, 1, 256, 257], "specials": {"bos": 4, "eos": 5}, "vocab_size": 6}, f)

    # normalization: the same surface token written three ways matches up
    assert normalize("Ġthe", "gpt") == b" the"
    assert normalize("▁the", "sp") == b" the"
    assert normalize("##ing", "gpt") == b"ing"
    assert normalize("<0x41>", "sp") == b"A"
    assert normalize("ĠFrance", "gpt") == normalize("▁France", "sp") == b" France"

    subject = load_vocab(subj_path)
    refs = [load_vocab(p, kimi=subject) for p in (ref1, ref2, ref3, ref4)]
    presence, _ = analyze(subject, refs)
    assert presence[256] == 4, f"b' the' in all 4 refs (incl. nano), got {presence[256]}"
    assert presence[257] == 2, f"b' France' in ref1 + nano, got {presence[257]}"
    assert presence[258] == 3, f"b'ing' in ref1 + ref2(##) + ref3, got {presence[258]}"
    assert presence[259] == 0, f"b' banana' nowhere, got {presence[259]}"
    assert presence[260] == 1, f"CJK in ref2 only, got {presence[260]}"
    assert presence[261] == 0, f"b'zzz' is not b'zz', got {presence[261]}"

    # structural: 256 byte fallbacks + declared reserved ids
    counts = {256: 10, 258: 5, 259: 100, 261: 50}
    keep = build_keep_list(subject, presence, counts, freq_top=2, min_refs=2, max_keep=None)
    struct = subject.structural_ids()
    assert struct == set(range(256)), f"byte fallbacks detected by content, got {len(struct)}"
    assert set(range(256)) <= set(keep), "byte fallbacks always kept"
    subject.reserved = {300, 301}
    keep = build_keep_list(subject, presence, counts, freq_top=2, min_refs=2, max_keep=None)
    assert {300, 301} <= set(keep), "reserved ids always kept"
    # rules: min-refs=2 keeps 256 (4 refs) + 257 (2) + 258 (3); freq-top 2 adds 259 + 261
    want = set(range(256)) | {256, 257, 258, 259, 261, 300, 301}
    assert set(keep) == want, f"keep mismatch: {sorted(set(keep) ^ want)}"
    # min-refs=3 drops 258 (3 refs is >= 3: stays), min-refs=4 drops it
    keep4 = build_keep_list(subject, presence, counts, freq_top=0, min_refs=4, max_keep=None)
    assert 256 in keep4 and 258 not in keep4 and 261 not in keep4, "min-refs=4 rule"
    # cap: 260 slots = 258 structural + 2 ranked by frequency (banana 100, zzz 50)
    capped = build_keep_list(subject, presence, counts, freq_top=2, min_refs=2, max_keep=260)
    assert len(capped) == 260 and 259 in capped and 261 in capped, "cap keeps freq priority"
    assert 256 not in capped and 258 not in capped, "cap drops lower-priority selected"
    assert set(range(256)) | {300, 301} <= set(capped), "cap never drops structural"

    # recomposition: cut tokens re-segment greedily into kept surfaces
    subject.reserved = set()
    costs = recompose_costs(subject, set(range(256)) | {258})
    assert 258 not in costs and 0 not in costs, "kept tokens have no recomposition cost"
    assert costs[256] == 4, f"b' the' -> 4 single bytes, got {costs[256]}"
    assert costs[259] == 7, f"b' banana' -> 7 single bytes, got {costs[259]}"
    assert costs[261] == 3, f"b'zzz' -> 3 single bytes, got {costs[261]}"
    costs2 = recompose_costs(subject, set(range(256)) | {256, 258})
    assert costs2[261] == 3 and 256 not in costs2, "keeping b' the' removes its cost"
    print(f"selftest OK (synthetic vocabs in {tmp})")


# ── main ──

def parse_range(spec):
    a, _, b = spec.partition("-")
    lo, hi = int(a), int(b) if b else int(a)
    if hi < lo:
        sys.exit(f"error: bad --reserved range '{spec}'")
    return set(range(lo, hi + 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subject", help="subject vocab (K3): tiktoken .model / tokenizer.json / vocab.json / vocab_nano.json")
    ap.add_argument("--refs", nargs="*", default=[], help="reference vocab files (any supported format)")
    ap.add_argument("--kimi-vocab", help="K3 tiktoken .model used to resolve vocab_nano.json inputs")
    ap.add_argument("--reserved", action="append", default=[], help="extra structural subject id range, e.g. 163584-163839")
    ap.add_argument("--report", action="store_true", help="print the rarity report")
    ap.add_argument("--keep-list", metavar="OUT", help="write subject token ids to keep, one per line")
    ap.add_argument("--freq", help='"<id> <count>" frequency file of the subject vocab (nano/count_freq.py)')
    ap.add_argument("--freq-top", type=int, default=0, help="keep the top-F most frequent tokens (needs --freq)")
    ap.add_argument("--min-refs", type=int, default=2, help="keep tokens present in >= this many reference vocabs")
    ap.add_argument("--max-keep", type=int, default=None, help="cap the keep-list (structural tokens never dropped)")
    ap.add_argument("--recompose", action="store_true",
                    help="with --keep-list: report the re-segmentation cost of every cut token")
    ap.add_argument("--selftest", action="store_true", help="offline synthetic check of all formats and rules")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.subject:
        ap.error("--subject is required (or --selftest)")
    if args.freq_top > 0 and not args.freq:
        ap.error("--freq-top needs --freq")

    subject = load_vocab(args.subject)
    kimi = subject if subject.convention == "bytes" else (
        load_vocab(args.kimi_vocab) if args.kimi_vocab else None)
    for spec in args.reserved:
        subject.reserved |= parse_range(spec)
    refs = [load_vocab(p, kimi=kimi) for p in args.refs]
    presence, ref_sets = analyze(subject, refs)

    if args.report or not args.keep_list:
        report(subject, refs, presence, ref_sets)

    if args.keep_list:
        counts = load_freq(args.freq) if args.freq else {}
        for i in counts:
            if i >= subject.size:
                sys.exit(f"error: {args.freq}: token id {i} out of range (subject vocab is {subject.size})")
        keep = build_keep_list(subject, presence, counts, args.freq_top, args.min_refs, args.max_keep)
        if args.recompose:
            recompose_report(subject, set(keep), recompose_costs(subject, set(keep)), counts)
        with open(args.keep_list, "w", encoding="utf-8") as f:
            f.write(f"# vocab_cross.py keep-list: subject={args.subject} min_refs={args.min_refs} "
                    f"freq_top={args.freq_top} max_keep={args.max_keep}\n")
            for i in keep:
                f.write(f"{i}\n")
        print(f"wrote {args.keep_list}: {len(keep)} subject token ids")


if __name__ == "__main__":
    main()
