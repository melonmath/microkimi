#!/usr/bin/env python3
"""Align causal trace rows across tokenizer-independent UTF-8 byte spans.

Each input is a compact NumPy ``.npz`` index with parallel one-dimensional
arrays:

``sequence_id``
    Integer, Unicode, or byte-string identifier for the input sequence.
``row``
    Non-negative row in the corresponding activation array.
``byte_start`` and ``byte_end``
    Half-open offsets in the original sequence encoded as UTF-8.
``content_sha256``
    Lowercase SHA-256 hex digest of the original UTF-8 sequence bytes.
``context_fingerprint``
    Non-empty producer-defined string identifying the causal context under
    which the state was captured.
``special`` (optional)
    Boolean marker for rows which do not consume ordinary input bytes.

A causal state describes the prefix consumed through ``byte_end``. Two rows
are therefore paired when their sequence identifiers and byte-prefix ends are
identical, even when their token starts differ. The content digest and context
fingerprint must also agree between the two traces. Both proof fields must be
constant within a sequence. Special rows must have zero-width spans and never
pair. If either index contains multiple ordinary rows at a shared prefix end,
that end is ambiguous and is reported but not paired.

The output archive contains ``source_row``, ``target_row``, ``sequence_id``,
``byte_end``, both proof fields, and ``is_holdout`` arrays.
``train_pair_index`` and ``holdout_pair_index`` index those pair arrays
directly. The split selects an exact, deterministic number of sequences which
have at least one output pair by a stable hash. Every endpoint from one
``sequence_id`` stays in the same split, independently of Python's randomized
hash and archive order.

Examples:

  python3 span_align.py --source-index source.trace.npz \
      --target-index target.trace.npz --out aligned.npz
  python3 span_align.py --source-index source.trace.npz \
      --target-index target.trace.npz --out aligned.npz \
      --holdout-fraction 0.2 --split-seed 17
  python3 span_align.py --selftest
"""

import argparse
import hashlib
import math
import os
import struct
import tempfile

import numpy as np


REQUIRED_FIELDS = (
    "sequence_id",
    "row",
    "byte_start",
    "byte_end",
    "content_sha256",
    "context_fingerprint",
)


class TraceIndex:
    """Validated in-memory view of a compact causal trace index."""

    def __init__(self, sequence_id, row, byte_start, byte_end,
                 content_sha256, context_fingerprint, special,
                 label="trace index"):
        self.sequence_id = sequence_id
        self.row = row
        self.byte_start = byte_start
        self.byte_end = byte_end
        self.content_sha256 = content_sha256
        self.context_fingerprint = context_fingerprint
        self.special = special
        self.label = label

    def __len__(self):
        return self.row.size


def _sequence_key(value):
    """Return a type-tagged, hashable scalar sequence identifier."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        raise ValueError("boolean sequence identifiers are not supported")
    if isinstance(value, int):
        return ("integer", int(value))
    if isinstance(value, str):
        return ("unicode", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    raise ValueError(
        "sequence_id must contain integers, Unicode strings, or byte strings"
    )


def _sequence_key_bytes(key):
    """Serialize a normalized sequence key without platform dependencies."""
    kind, value = key
    if kind == "integer":
        payload = str(value).encode("ascii")
        tag = b"i"
    elif kind == "unicode":
        payload = value.encode("utf-8")
        tag = b"u"
    else:
        payload = value
        tag = b"b"
    return tag + struct.pack("<Q", len(payload)) + payload


def _load_array(archive, field, path):
    try:
        value = archive[field]
    except ValueError as exc:
        raise ValueError(f"{path}: cannot read {field}: {exc}") from exc
    if not isinstance(value, np.ndarray) or value.ndim != 1:
        raise ValueError(f"{path}: {field} must be a one-dimensional array")
    return np.array(value, copy=True)


def _as_nonnegative_int64(array, field, path):
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{path}: {field} must have an integer dtype")
    limit = np.iinfo(np.int64).max
    values = []
    for value in array:
        integer = int(value)
        if integer < 0 or integer > limit:
            raise ValueError(
                f"{path}: {field} values must be in [0, {limit}]"
            )
        values.append(integer)
    return np.asarray(values, dtype=np.int64)


def _as_special(array, path):
    if array.dtype.kind == "b":
        return array.astype(np.bool_, copy=False)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{path}: special must have a boolean or integer dtype")
    if any(int(value) not in (0, 1) for value in array):
        raise ValueError(f"{path}: integer special values must be zero or one")
    return array.astype(np.bool_)


def _validate_sequence_dtype(array, path):
    if array.dtype.kind not in "iuUS" or array.dtype.kind == "b":
        raise ValueError(
            f"{path}: sequence_id must have an integer, Unicode, or byte-string dtype"
        )
    return [_sequence_key(value) for value in array]


def _string_values(array, field, path):
    if array.dtype.kind not in "US":
        raise ValueError(
            f"{path}: {field} must have a Unicode or byte-string dtype"
        )
    values = []
    for entry, raw_value in enumerate(array):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{path}: {field} entry {entry} is not valid UTF-8"
                ) from exc
        if not value:
            raise ValueError(f"{path}: {field} entry {entry} must be non-empty")
        values.append(value)
    return values


def _validate_context_proofs(index, sequence_keys, content_values,
                             fingerprint_values):
    sequence_context = {}
    sequence_first_entry = {}
    hex_chars = frozenset("0123456789abcdef")
    for entry, (key, content_hash, fingerprint) in enumerate(zip(
            sequence_keys, content_values, fingerprint_values)):
        if len(content_hash) != 64 or any(
                character not in hex_chars for character in content_hash):
            raise ValueError(
                f"{index.label}: content_sha256 entry {entry} must contain "
                "exactly 64 lowercase hexadecimal characters"
            )
        proof = (content_hash, fingerprint)
        if key not in sequence_context:
            sequence_context[key] = proof
            sequence_first_entry[key] = entry
        elif sequence_context[key] != proof:
            old_hash, old_fingerprint = sequence_context[key]
            changed = []
            if content_hash != old_hash:
                changed.append("content_sha256")
            if fingerprint != old_fingerprint:
                changed.append("context_fingerprint")
            raise ValueError(
                f"{index.label}: {', '.join(changed)} must be constant for "
                f"sequence_id {key[1]!r}"
            )
    index.content_values = content_values
    index.fingerprint_values = fingerprint_values
    index.sequence_context = sequence_context
    index.sequence_first_entry = sequence_first_entry


def _validate_monotonic_spans(index, sequence_keys):
    previous_end = {}
    for entry, (key, start, end) in enumerate(
            zip(sequence_keys, index.byte_start, index.byte_end)):
        start = int(start)
        end = int(end)
        if end < start:
            raise ValueError(
                f"{index.label}: entry {entry} has byte_end {end} before "
                f"byte_start {start}"
            )
        if index.special[entry] and start != end:
            raise ValueError(
                f"{index.label}: special entry {entry} must have a "
                f"zero-width span, got [{start}, {end})"
            )
        if key in previous_end and start < previous_end[key]:
            raise ValueError(
                f"{index.label}: entry {entry} overlaps or precedes the "
                f"previous span for sequence_id {key[1]!r}: byte_start "
                f"{start} < previous byte_end {previous_end[key]}"
            )
        previous_end[key] = end


def load_trace_index(path, label=None):
    """Load and validate one compact ``.npz`` trace index.

    Span order is checked independently for every sequence in archive order.
    Gaps and repeated zero-width endpoints are allowed, but spans may not
    overlap or move backwards. Repeated endpoints are handled as ambiguity by
    :func:`align_trace_indexes`.
    """
    label = label or str(path)
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: cannot load trace index: {exc}") from exc
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(f"{path}: trace index must be a NumPy .npz archive")

    try:
        missing = [field for field in REQUIRED_FIELDS
                   if field not in loaded.files]
        if missing:
            raise ValueError(
                f"{path}: missing required field(s): {', '.join(missing)}"
            )
        sequence_id = _load_array(loaded, "sequence_id", path)
        row = _load_array(loaded, "row", path)
        byte_start = _load_array(loaded, "byte_start", path)
        byte_end = _load_array(loaded, "byte_end", path)
        content_sha256 = _load_array(loaded, "content_sha256", path)
        context_fingerprint = _load_array(
            loaded, "context_fingerprint", path
        )
        if "special" in loaded.files:
            special = _load_array(loaded, "special", path)
        else:
            special = np.zeros(row.size, dtype=np.bool_)
    finally:
        loaded.close()

    lengths = {
        "sequence_id": sequence_id.size,
        "row": row.size,
        "byte_start": byte_start.size,
        "byte_end": byte_end.size,
        "content_sha256": content_sha256.size,
        "context_fingerprint": context_fingerprint.size,
        "special": special.size,
    }
    if len(set(lengths.values())) != 1:
        detail = ", ".join(f"{name}={size}" for name, size in lengths.items())
        raise ValueError(f"{path}: parallel array lengths differ ({detail})")

    sequence_keys = _validate_sequence_dtype(sequence_id, path)
    row = _as_nonnegative_int64(row, "row", path)
    byte_start = _as_nonnegative_int64(byte_start, "byte_start", path)
    byte_end = _as_nonnegative_int64(byte_end, "byte_end", path)
    content_values = _string_values(content_sha256, "content_sha256", path)
    fingerprint_values = _string_values(
        context_fingerprint, "context_fingerprint", path
    )
    special = _as_special(special, path)
    if np.unique(row).size != row.size:
        raise ValueError(f"{path}: row values must be unique")

    index = TraceIndex(
        sequence_id, row, byte_start, byte_end, content_sha256,
        context_fingerprint, special, label=label
    )
    _validate_context_proofs(
        index, sequence_keys, content_values, fingerprint_values
    )
    _validate_monotonic_spans(index, sequence_keys)
    index.sequence_keys = sequence_keys
    return index


def _endpoint_groups(index):
    groups = {}
    for entry in range(len(index)):
        if index.special[entry]:
            continue
        key = (index.sequence_keys[entry], int(index.byte_end[entry]))
        groups.setdefault(key, []).append(entry)
    return groups


def _coverage(numerator, denominator):
    return float(numerator) / denominator if denominator else 0.0


def deterministic_holdout(sequence_keys, fraction, seed=0):
    """Return a stable mask after assigning whole sequences to holdout."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("holdout fraction must be finite and in [0, 1]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
            seed, (int, np.integer)):
        raise ValueError("split seed must be an integer")
    if not -(1 << 63) <= int(seed) < (1 << 63):
        raise ValueError("split seed must fit in a signed 64-bit integer")
    unique_sequence_keys = sorted(set(sequence_keys))
    holdout_count = int(math.floor(
        len(unique_sequence_keys) * fraction + 0.5
    ))
    ranked = []
    for sequence_key in unique_sequence_keys:
        digest = hashlib.blake2b(
            digest_size=16, person=b"mkm-span-split"
        )
        digest.update(struct.pack("<q", int(seed)))
        digest.update(_sequence_key_bytes(sequence_key))
        ranked.append((digest.digest(), sequence_key))
    ranked.sort()
    holdout_sequences = {
        sequence_key for _, sequence_key in ranked[:holdout_count]
    }
    return np.asarray(
        [key in holdout_sequences for key in sequence_keys], dtype=np.bool_
    )


def align_trace_indexes(source, target, holdout_fraction=0.1, split_seed=0):
    """Pair context-proven non-special causal rows at identical prefix ends."""
    source_groups = _endpoint_groups(source)
    target_groups = _endpoint_groups(target)
    source_sequences = set(source.sequence_context)
    target_sequences = set(target.sequence_context)
    shared_sequences = source_sequences.intersection(target_sequences)
    content_mismatches = {
        key for key in shared_sequences
        if source.sequence_context[key][0] != target.sequence_context[key][0]
    }
    fingerprint_mismatches = {
        key for key in shared_sequences
        if source.sequence_context[key][1] != target.sequence_context[key][1]
    }
    context_mismatches = content_mismatches.union(fingerprint_mismatches)
    compatible_sequences = shared_sequences.difference(context_mismatches)

    raw_common = set(source_groups).intersection(target_groups)
    common = {
        key for key in raw_common if key[0] in compatible_sequences
    }
    context_rejected_common = raw_common.difference(common)

    pairs = []
    ambiguous = []
    for key in common:
        source_entries = source_groups[key]
        target_entries = target_groups[key]
        if len(source_entries) == 1 and len(target_entries) == 1:
            pairs.append((key, source_entries[0], target_entries[0]))
        else:
            ambiguous.append((key, source_entries, target_entries))

    def pair_sort_key(item):
        (sequence_key, byte_end), source_entry, target_entry = item
        return (
            sequence_key,
            byte_end,
            int(source.row[source_entry]),
            int(target.row[target_entry]),
        )

    pairs.sort(key=pair_sort_key)
    ambiguous.sort(key=lambda item: (item[0][0], item[0][1]))

    source_entries = np.asarray([item[1] for item in pairs], dtype=np.int64)
    target_entries = np.asarray([item[2] for item in pairs], dtype=np.int64)
    pair_sequence_keys = [item[0][0] for item in pairs]
    pair_byte_ends = np.asarray(
        [item[0][1] for item in pairs], dtype=np.int64
    )
    is_holdout = deterministic_holdout(
        pair_sequence_keys, holdout_fraction, split_seed
    )

    if source_entries.size:
        pair_sequence_ids = source.sequence_id[source_entries]
    else:
        pair_sequence_ids = source.sequence_id[:0]

    paired_sequence_keys = sorted(set(pair_sequence_keys))
    train_sequence_keys = sorted({
        key for key, holdout in zip(pair_sequence_keys, is_holdout)
        if not holdout
    })
    holdout_sequence_keys = sorted({
        key for key, holdout in zip(pair_sequence_keys, is_holdout)
        if holdout
    })

    def sequence_id_array(index, keys):
        if not keys:
            return index.sequence_id[:0]
        entries = np.asarray(
            [index.sequence_first_entry[key] for key in keys], dtype=np.int64
        )
        return index.sequence_id[entries]

    ambiguous_source_entries = [entries[0] for _, entries, _ in ambiguous]
    if ambiguous_source_entries:
        ambiguous_sequence_ids = source.sequence_id[
            np.asarray(ambiguous_source_entries, dtype=np.int64)
        ]
    else:
        ambiguous_sequence_ids = source.sequence_id[:0]

    mismatch_keys = sorted(context_mismatches)
    source_mismatch_entries = np.asarray(
        [source.sequence_first_entry[key] for key in mismatch_keys],
        dtype=np.int64,
    )
    target_mismatch_entries = np.asarray(
        [target.sequence_first_entry[key] for key in mismatch_keys],
        dtype=np.int64,
    )
    source_context_compatible_rows = sum(
        not source.special[entry]
        and source.sequence_keys[entry] in compatible_sequences
        for entry in range(len(source))
    )
    target_context_compatible_rows = sum(
        not target.special[entry]
        and target.sequence_keys[entry] in compatible_sequences
        for entry in range(len(target))
    )
    source_context_rejected_rows = sum(
        not source.special[entry]
        and source.sequence_keys[entry] in context_mismatches
        for entry in range(len(source))
    )
    target_context_rejected_rows = sum(
        not target.special[entry]
        and target.sequence_keys[entry] in context_mismatches
        for entry in range(len(target))
    )

    source_eligible = int((~source.special).sum())
    target_eligible = int((~target.special).sum())
    pair_count = len(pairs)
    source_duplicate = sum(
        len(entries) > 1 for entries in source_groups.values()
    )
    target_duplicate = sum(
        len(entries) > 1 for entries in target_groups.values()
    )
    report = {
        "source_rows": len(source),
        "target_rows": len(target),
        "source_special_rows": int(source.special.sum()),
        "target_special_rows": int(target.special.sum()),
        "source_eligible_rows": source_eligible,
        "target_eligible_rows": target_eligible,
        "source_sequences": len(source_sequences),
        "target_sequences": len(target_sequences),
        "shared_sequences": len(shared_sequences),
        "context_compatible_sequences": len(compatible_sequences),
        "context_mismatched_sequences": len(context_mismatches),
        "content_mismatched_sequences": len(content_mismatches),
        "fingerprint_mismatched_sequences": len(fingerprint_mismatches),
        "source_context_compatible_rows": source_context_compatible_rows,
        "target_context_compatible_rows": target_context_compatible_rows,
        "source_context_rejected_rows": source_context_rejected_rows,
        "target_context_rejected_rows": target_context_rejected_rows,
        "source_endpoints": len(source_groups),
        "target_endpoints": len(target_groups),
        "raw_shared_endpoints": len(raw_common),
        "context_rejected_shared_endpoints": len(context_rejected_common),
        "shared_endpoints": len(common),
        "paired_rows": pair_count,
        "paired_sequences": len(paired_sequence_keys),
        "ambiguous_shared_endpoints": len(ambiguous),
        "ambiguous_source_rows": sum(
            len(source_entries_) for _, source_entries_, _ in ambiguous
        ),
        "ambiguous_target_rows": sum(
            len(target_entries_) for _, _, target_entries_ in ambiguous
        ),
        "source_duplicate_endpoints": source_duplicate,
        "target_duplicate_endpoints": target_duplicate,
        "source_coverage": _coverage(pair_count, source_eligible),
        "target_coverage": _coverage(pair_count, target_eligible),
        "shared_endpoint_coverage": _coverage(pair_count, len(common)),
        "train_pairs": int((~is_holdout).sum()),
        "holdout_pairs": int(is_holdout.sum()),
        "train_sequences": len(train_sequence_keys),
        "holdout_sequences": len(holdout_sequence_keys),
        "actual_holdout_pair_fraction": _coverage(
            int(is_holdout.sum()), pair_count
        ),
        "actual_holdout_sequence_fraction": _coverage(
            len(holdout_sequence_keys), len(paired_sequence_keys)
        ),
    }

    output = {
        "format_version": np.int64(2),
        "source_row": source.row[source_entries],
        "target_row": target.row[target_entries],
        "sequence_id": pair_sequence_ids,
        "byte_end": pair_byte_ends,
        "content_sha256": np.asarray(
            [source.sequence_context[key][0] for key in pair_sequence_keys],
            dtype="U64",
        ),
        "context_fingerprint": np.asarray(
            [source.sequence_context[key][1] for key in pair_sequence_keys],
            dtype="U",
        ),
        "is_holdout": is_holdout,
        "train_pair_index": np.flatnonzero(~is_holdout).astype(np.int64),
        "holdout_pair_index": np.flatnonzero(is_holdout).astype(np.int64),
        "train_sequence_id": sequence_id_array(source, train_sequence_keys),
        "holdout_sequence_id": sequence_id_array(
            source, holdout_sequence_keys
        ),
        "split_unit": np.asarray("sequence"),
        "holdout_fraction": np.float64(holdout_fraction),
        "split_seed": np.int64(split_seed),
        "ambiguous_sequence_id": ambiguous_sequence_ids,
        "ambiguous_byte_end": np.asarray(
            [item[0][1] for item in ambiguous], dtype=np.int64
        ),
        "ambiguous_source_count": np.asarray(
            [len(item[1]) for item in ambiguous], dtype=np.int64
        ),
        "ambiguous_target_count": np.asarray(
            [len(item[2]) for item in ambiguous], dtype=np.int64
        ),
        "ambiguous_content_sha256": np.asarray(
            [source.sequence_context[item[0][0]][0] for item in ambiguous],
            dtype="U64",
        ),
        "ambiguous_context_fingerprint": np.asarray(
            [source.sequence_context[item[0][0]][1] for item in ambiguous],
            dtype="U",
        ),
        "context_mismatch_sequence_id": sequence_id_array(
            source, mismatch_keys
        ),
        "source_mismatch_content_sha256": (
            source.content_sha256[source_mismatch_entries]
        ),
        "target_mismatch_content_sha256": (
            target.content_sha256[target_mismatch_entries]
        ),
        "source_mismatch_context_fingerprint": (
            source.context_fingerprint[source_mismatch_entries]
        ),
        "target_mismatch_context_fingerprint": (
            target.context_fingerprint[target_mismatch_entries]
        ),
    }
    for name, value in report.items():
        output[name] = (
            np.float64(value) if isinstance(value, float) else np.int64(value)
        )
    return output, report


def save_alignment(path, output):
    """Write one compressed pair archive and return its actual path."""
    if not str(path).endswith(".npz"):
        path = str(path) + ".npz"
    np.savez_compressed(path, **output)
    return path


def _append_tokenized_sequence(columns, sequence_id, text, pieces, row_start,
                               add_special=True, trailing_special=False,
                               content_sha256=None,
                               context_fingerprint="synthetic-causal-v1"):
    """Build a synthetic trace index from exact Unicode pieces."""
    if "".join(pieces) != text:
        raise AssertionError("synthetic token pieces do not reconstruct text")
    if content_sha256 is None:
        content_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def append_proof():
        columns["content_sha256"].append(content_sha256)
        columns["context_fingerprint"].append(context_fingerprint)

    row = row_start
    byte_cursor = 0
    if add_special:
        columns["sequence_id"].append(sequence_id)
        columns["row"].append(row)
        columns["byte_start"].append(0)
        columns["byte_end"].append(0)
        append_proof()
        columns["special"].append(True)
        row += 1
    for piece in pieces:
        piece_bytes = piece.encode("utf-8")
        columns["sequence_id"].append(sequence_id)
        columns["row"].append(row)
        columns["byte_start"].append(byte_cursor)
        byte_cursor += len(piece_bytes)
        columns["byte_end"].append(byte_cursor)
        append_proof()
        columns["special"].append(False)
        row += 1
    if trailing_special:
        columns["sequence_id"].append(sequence_id)
        columns["row"].append(row)
        columns["byte_start"].append(byte_cursor)
        columns["byte_end"].append(byte_cursor)
        append_proof()
        columns["special"].append(True)
        row += 1
    assert byte_cursor == len(text.encode("utf-8"))
    return row


def _synthetic_columns():
    return {name: [] for name in REQUIRED_FIELDS + ("special",)}


def _save_synthetic(path, columns):
    np.savez(
        path,
        sequence_id=np.asarray(columns["sequence_id"], dtype="U"),
        row=np.asarray(columns["row"], dtype=np.int64),
        byte_start=np.asarray(columns["byte_start"], dtype=np.int64),
        byte_end=np.asarray(columns["byte_end"], dtype=np.int64),
        content_sha256=np.asarray(columns["content_sha256"], dtype="U64"),
        context_fingerprint=np.asarray(
            columns["context_fingerprint"], dtype="U"
        ),
        special=np.asarray(columns["special"], dtype=np.bool_),
    )


def selftest():
    """Exercise Unicode spans, context proofs, ambiguity, and grouped split."""
    source_columns = _synthetic_columns()
    target_columns = _synthetic_columns()

    source_row = 100
    source_row = _append_tokenized_sequence(
        source_columns, "accent", "café ☕ 東京",
        ["ca", "fé", " ", "☕", " ", "東京"], source_row,
        trailing_special=True,
    )
    source_row = _append_tokenized_sequence(
        source_columns, "emoji", "A👩🏽‍💻Z",
        ["A", "👩🏽‍💻", "Z"], source_row,
    )
    source_row = _append_tokenized_sequence(
        source_columns, "cjk", "naïve\n東京",
        ["na", "ï", "ve\n", "東京"], source_row,
    )
    source_row = _append_tokenized_sequence(
        source_columns, "ambiguous", "xy", ["x", "y"], source_row,
    )
    source_columns["sequence_id"].append("ambiguous")
    source_columns["row"].append(source_row)
    source_columns["byte_start"].append(2)
    source_columns["byte_end"].append(2)
    source_columns["content_sha256"].append(
        hashlib.sha256(b"xy").hexdigest()
    )
    source_columns["context_fingerprint"].append("synthetic-causal-v1")
    source_columns["special"].append(False)
    source_row += 1
    source_row = _append_tokenized_sequence(
        source_columns, "content-mismatch", "q", ["q"], source_row,
        content_sha256="a" * 64,
    )
    _append_tokenized_sequence(
        source_columns, "context-mismatch", "z", ["z"], source_row,
        context_fingerprint="synthetic-context-a",
    )

    target_row = 900
    target_row = _append_tokenized_sequence(
        target_columns, "cjk", "naïve\n東京",
        ["n", "aïve", "\n東", "京"], target_row,
    )
    target_row = _append_tokenized_sequence(
        target_columns, "accent", "café ☕ 東京",
        ["c", "afé ", "☕ ", "東", "京"], target_row,
        trailing_special=True,
    )
    target_row = _append_tokenized_sequence(
        target_columns, "ambiguous", "xy", ["xy"], target_row,
    )
    target_row = _append_tokenized_sequence(
        target_columns, "content-mismatch", "q", ["q"], target_row,
        content_sha256="b" * 64,
    )
    target_row = _append_tokenized_sequence(
        target_columns, "context-mismatch", "z", ["z"], target_row,
        context_fingerprint="synthetic-context-b",
    )
    _append_tokenized_sequence(
        target_columns, "emoji", "A👩🏽‍💻Z",
        ["A👩", "🏽‍", "💻Z"], target_row,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "source.npz")
        target_path = os.path.join(temp_dir, "target.npz")
        output_path = os.path.join(temp_dir, "aligned.npz")
        _save_synthetic(source_path, source_columns)
        _save_synthetic(target_path, target_columns)

        source = load_trace_index(source_path, "synthetic source")
        target = load_trace_index(target_path, "synthetic target")
        output, report = align_trace_indexes(
            source, target, holdout_fraction=0.4, split_seed=20260809
        )
        repeated, repeated_report = align_trace_indexes(
            source, target, holdout_fraction=0.4, split_seed=20260809
        )

        assert output["sequence_id"].tolist() == [
            "accent", "accent", "accent", "cjk", "emoji"
        ]
        assert output["byte_end"].tolist() == [6, 10, 16, 13, 17]
        assert report["paired_rows"] == 5
        assert report["ambiguous_shared_endpoints"] == 1
        assert report["ambiguous_source_rows"] == 2
        assert report["ambiguous_target_rows"] == 1
        assert report["source_special_rows"] == 7
        assert report["target_special_rows"] == 7
        assert report["context_mismatched_sequences"] == 2
        assert report["content_mismatched_sequences"] == 1
        assert report["fingerprint_mismatched_sequences"] == 1
        assert report["context_rejected_shared_endpoints"] == 2
        assert report["paired_sequences"] == 3
        assert report["holdout_sequences"] == 1
        assert report["train_sequences"] == 2
        assert report["train_pairs"] + report["holdout_pairs"] == 5
        assert report == repeated_report
        assert np.array_equal(output["source_row"], repeated["source_row"])
        assert np.array_equal(output["is_holdout"], repeated["is_holdout"])
        assert not np.any(output["byte_end"] == 0)
        for sequence_id in np.unique(output["sequence_id"]):
            sequence_split = output["is_holdout"][
                output["sequence_id"] == sequence_id
            ]
            assert np.unique(sequence_split).size == 1
        assert output["context_mismatch_sequence_id"].tolist() == [
            "content-mismatch", "context-mismatch"
        ]
        assert np.all(
            output["context_fingerprint"] == "synthetic-causal-v1"
        )

        save_alignment(output_path, output)
        with np.load(output_path, allow_pickle=False) as archived:
            assert int(archived["format_version"]) == 2
            assert str(archived["split_unit"]) == "sequence"
            assert archived["source_row"].shape == (5,)
            assert archived["target_row"].shape == (5,)
            assert (archived["train_pair_index"].size
                    + archived["holdout_pair_index"].size == 5)
            assert archived["ambiguous_byte_end"].tolist() == [2]
            assert archived["content_sha256"].shape == (5,)
            assert archived["context_fingerprint"].shape == (5,)

        invalid_path = os.path.join(temp_dir, "invalid.npz")
        invalid_hash = hashlib.sha256(b"invalid").hexdigest()
        np.savez(
            invalid_path,
            sequence_id=np.asarray([7, 7], dtype=np.int64),
            row=np.asarray([0, 1], dtype=np.int64),
            byte_start=np.asarray([0, 1], dtype=np.int64),
            byte_end=np.asarray([2, 3], dtype=np.int64),
            content_sha256=np.asarray([invalid_hash, invalid_hash]),
            context_fingerprint=np.asarray(["ctx", "ctx"]),
        )
        try:
            load_trace_index(invalid_path)
        except ValueError as exc:
            assert "overlaps or precedes" in str(exc)
        else:
            raise AssertionError("non-monotonic spans were accepted")

        invalid_special_path = os.path.join(temp_dir, "invalid-special.npz")
        np.savez(
            invalid_special_path,
            sequence_id=np.asarray(["special"]),
            row=np.asarray([0], dtype=np.int64),
            byte_start=np.asarray([0], dtype=np.int64),
            byte_end=np.asarray([1], dtype=np.int64),
            content_sha256=np.asarray(["c" * 64]),
            context_fingerprint=np.asarray(["ctx"]),
            special=np.asarray([True]),
        )
        try:
            load_trace_index(invalid_special_path)
        except ValueError as exc:
            assert "zero-width" in str(exc)
        else:
            raise AssertionError("non-zero-width special span was accepted")

    print("span_align selftest OK")


def _print_report(report):
    print(
        "eligible rows         : "
        f"source={report['source_eligible_rows']}  "
        f"target={report['target_eligible_rows']}"
    )
    print(
        "shared / paired      : "
        f"{report['shared_endpoints']} / {report['paired_rows']}"
    )
    print(
        "context proof        : "
        f"compatible_sequences={report['context_compatible_sequences']}  "
        f"mismatched_sequences={report['context_mismatched_sequences']}  "
        f"content={report['content_mismatched_sequences']}  "
        f"fingerprint={report['fingerprint_mismatched_sequences']}  "
        f"rejected_endpoints={report['context_rejected_shared_endpoints']}"
    )
    print(
        "coverage             : "
        f"source={report['source_coverage']:.2%}  "
        f"target={report['target_coverage']:.2%}  "
        f"shared={report['shared_endpoint_coverage']:.2%}"
    )
    print(
        "ambiguous endpoints  : "
        f"shared={report['ambiguous_shared_endpoints']}  "
        f"source_duplicates={report['source_duplicate_endpoints']}  "
        f"target_duplicates={report['target_duplicate_endpoints']}"
    )
    print(
        "train / holdout      : "
        f"pairs={report['train_pairs']} / {report['holdout_pairs']}  "
        f"sequences={report['train_sequences']} / "
        f"{report['holdout_sequences']}  "
        f"actual_sequence_fraction="
        f"{report['actual_holdout_sequence_fraction']:.2%}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-index", metavar="SOURCE.npz",
        help="source trace index",
    )
    parser.add_argument(
        "--target-index", metavar="TARGET.npz",
        help="target trace index",
    )
    parser.add_argument(
        "--out", metavar="PAIRS.npz",
        help="output pair archive",
    )
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.1,
        help="fraction of represented sequences assigned to holdout (default: 0.1)",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0,
        help="stable split seed (default: 0)",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    missing = [name for name, value in (
        ("--source-index", args.source_index),
        ("--target-index", args.target_index),
        ("--out", args.out),
    ) if value is None]
    if missing:
        parser.error(f"required argument(s): {', '.join(missing)}")

    try:
        source = load_trace_index(args.source_index, "source index")
        target = load_trace_index(args.target_index, "target index")
        output, report = align_trace_indexes(
            source,
            target,
            holdout_fraction=args.holdout_fraction,
            split_seed=args.split_seed,
        )
        output_path = save_alignment(args.out, output)
    except ValueError as exc:
        parser.error(str(exc))

    _print_report(report)
    print(f"written              : {output_path}")


if __name__ == "__main__":
    main()
