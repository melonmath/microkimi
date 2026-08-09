#!/usr/bin/env python3
"""Capture token activation traces from a local Hugging Face causal model.

The model and tokenizer are loaded with ``local_files_only=True``. PyTorch and
Transformers are optional: they are imported only when a real capture is
requested, so metadata helpers and ``--selftest`` need only NumPy.

Input is JSONL. Each non-empty line is either a JSON string or an object with a
string ``text`` field and an optional ``sequence_id`` (``id`` is accepted as an
alias). Sequence identifiers must be all integers or all strings. Missing
identifiers use the zero-based record number.

The activation output is a float32 ``.npy`` matrix with one row per active
token. The companion trace is a format-v2 ``.npz`` archive containing the
parallel fields consumed by ``span_align.py``: ``sequence_id``, ``row``,
``byte_start``, ``byte_end``, ``content_sha256``,
``context_fingerprint``, and ``special``. Tokenizer character offsets are
converted to monotonic offsets in the original UTF-8 bytes. Repeated fallback
offsets become zero-width ordinary spans. Special tokens receive a zero-width
span at the current causal boundary, and padding rows are omitted.

Examples:

  python3 hf_trace_capture.py --model ./local-model \
      --module model.layers.3 --capture output --texts texts.jsonl \
      --out activations.npy --trace-out activations.trace.npz
  python3 hf_trace_capture.py --model ./local-model \
      --module model.layers.3 --capture both --texts texts.jsonl \
      --out-input layer-input.npy --out-output layer-output.npy \
      --trace-out layer.trace.npz
  python3 hf_trace_capture.py --selftest

The selected module must run exactly once per input sequence and its selected
input or output tensor must have shape ``[1, tokens, width]`` or
``[tokens, width]``. When a hook value is a tuple, list, or mapping, the first
tensor-like value is selected recursively.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping

import numpy as np


TRACE_FORMAT_VERSION = 2
MODEL_DTYPE_CHOICES = ("auto", "float32", "float16", "bfloat16")
CONTEXT_DESCRIPTOR = (
    b"microkimi-hf-trace-context-v1\0"
    b"causal-prefix\0add-special-tokens\0utf8-byte-offsets\0no-padding-rows"
)
CONTEXT_FINGERPRINT = (
    "sha256:" + hashlib.sha256(CONTEXT_DESCRIPTOR).hexdigest()
)
_AUXILIARY_TOKENIZER_FIELDS = frozenset({
    "offset_mapping",
    "special_tokens_mask",
    "overflow_to_sample_mapping",
    "num_truncated_tokens",
    "length",
})


class TextRecord:
    """One uniquely identified JSONL text."""

    __slots__ = ("sequence_id", "text")

    def __init__(self, sequence_id, text):
        self.sequence_id = sequence_id
        self.text = text


class EncodedText:
    """Tokenizer output plus active-token UTF-8 span metadata."""

    __slots__ = (
        "model_inputs",
        "token_count",
        "active_positions",
        "byte_start",
        "byte_end",
        "special",
    )

    def __init__(self, model_inputs, token_count, active_positions,
                 byte_start, byte_end, special):
        self.model_inputs = model_inputs
        self.token_count = token_count
        self.active_positions = active_positions
        self.byte_start = byte_start
        self.byte_end = byte_end
        self.special = special


def _sequence_id_kind(value, location):
    if isinstance(value, bool):
        raise ValueError(f"{location}: sequence_id cannot be boolean")
    if isinstance(value, int):
        limit = np.iinfo(np.int64)
        if value < limit.min or value > limit.max:
            raise ValueError(
                f"{location}: integer sequence_id is outside int64 range"
            )
        return "integer"
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(
                f"{location}: string sequence_id is not valid UTF-8 Unicode"
            ) from exc
        return "string"
    raise ValueError(f"{location}: sequence_id must be an integer or string")


def read_text_jsonl(path):
    """Read JSONL text records and validate stable, homogeneous identifiers."""
    records = []
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot open JSONL input {path}: {exc}") from exc

    try:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc

            record_number = len(records)
            if isinstance(value, str):
                text = value
                sequence_id = record_number
            elif isinstance(value, dict):
                if "text" not in value or not isinstance(value["text"], str):
                    raise ValueError(
                        f"{path}:{line_number}: object field 'text' "
                        "must be a string"
                    )
                text = value["text"]
                has_sequence_id = "sequence_id" in value
                has_id = "id" in value
                if (has_sequence_id and has_id
                        and value["sequence_id"] != value["id"]):
                    raise ValueError(
                        f"{path}:{line_number}: 'sequence_id' and 'id' differ"
                    )
                if has_sequence_id:
                    sequence_id = value["sequence_id"]
                elif has_id:
                    sequence_id = value["id"]
                else:
                    sequence_id = record_number
            else:
                raise ValueError(
                    f"{path}:{line_number}: each record must be a JSON "
                    "string or object"
                )

            try:
                text.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: text is not valid UTF-8 Unicode"
                ) from exc
            _sequence_id_kind(
                sequence_id, f"{path}:{line_number}"
            )
            records.append(TextRecord(sequence_id, text))
    finally:
        handle.close()

    if not records:
        raise ValueError(f"{path}: JSONL input contains no text records")

    kinds = {
        _sequence_id_kind(record.sequence_id, str(path))
        for record in records
    }
    if len(kinds) != 1:
        raise ValueError(
            f"{path}: sequence_id values must be all integers or all strings"
        )
    keys = [(next(iter(kinds)), record.sequence_id) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: sequence_id values must be unique")
    return records


def _sequence_id_array(values):
    """Build a non-object NumPy identifier array accepted by span_align."""
    if not values:
        return np.asarray([], dtype="U")
    kinds = {_sequence_id_kind(value, "trace") for value in values}
    if len(kinds) != 1:
        raise ValueError(
            "trace sequence_id values must be all integers or all strings"
        )
    if "integer" in kinds:
        return np.asarray(values, dtype=np.int64)
    return np.asarray(values, dtype="U")


def utf8_token_spans(text, offsets, special_mask, active_mask=None):
    """Convert fast-tokenizer character offsets to monotonic UTF-8 spans.

    ``active_mask`` is normally the tokenizer attention mask. Entries with a
    false value are excluded entirely. Special entries ignore their tokenizer
    offset, which is commonly ``(0, 0)``, and receive a zero-width span at the
    end of the preceding ordinary token. Some byte-fallback tokenizers report
    overlapping character offsets for consecutive ordinary tokens. An
    overlapping start is therefore advanced to the preceding causal endpoint,
    while a genuine gap is preserved. Ends never move backwards. A repeated
    end stays ordinary and becomes zero-width; only the tokenizer's special
    mask marks a row as special.
    """
    if len(offsets) != len(special_mask):
        raise ValueError("offset_mapping and special_tokens_mask differ in length")
    if active_mask is None:
        active_mask = [True] * len(offsets)
    if len(offsets) != len(active_mask):
        raise ValueError("offset_mapping and attention_mask differ in length")

    char_to_byte = [0]
    byte_cursor = 0
    for character in text:
        byte_cursor += len(character.encode("utf-8"))
        char_to_byte.append(byte_cursor)

    starts = []
    ends = []
    specials = []
    active_positions = []
    causal_byte_end = 0
    for token_index, (offset, is_special, is_active) in enumerate(zip(
            offsets, special_mask, active_mask)):
        if isinstance(is_active, bool):
            active = is_active
        elif isinstance(is_active, (int, np.integer)) and is_active in (0, 1):
            active = bool(is_active)
        else:
            raise ValueError(
                f"attention_mask entry {token_index} must be zero or one"
            )
        if not active:
            continue

        if isinstance(is_special, bool):
            special = is_special
        elif (isinstance(is_special, (int, np.integer))
              and is_special in (0, 1)):
            special = bool(is_special)
        else:
            raise ValueError(
                f"special_tokens_mask entry {token_index} must be zero or one"
            )

        if special:
            byte_start = causal_byte_end
            byte_end = causal_byte_end
        else:
            if (not isinstance(offset, (list, tuple))
                    or len(offset) != 2):
                raise ValueError(
                    f"offset_mapping entry {token_index} must be a pair"
                )
            char_start, char_end = offset
            if (isinstance(char_start, bool) or isinstance(char_end, bool)
                    or not isinstance(char_start, (int, np.integer))
                    or not isinstance(char_end, (int, np.integer))):
                raise ValueError(
                    f"offset_mapping entry {token_index} must contain integers"
                )
            char_start = int(char_start)
            char_end = int(char_end)
            if not (0 <= char_start <= char_end <= len(text)):
                raise ValueError(
                    f"offset_mapping entry {token_index} is outside the text: "
                    f"[{char_start}, {char_end})"
                )
            # Offset starts can overlap during byte fallback. Canonicalize an
            # overlap to the previous endpoint, preserve genuine gaps, and
            # never let an end move backwards.
            byte_start = max(causal_byte_end, char_to_byte[char_start])
            byte_end = max(byte_start, char_to_byte[char_end])
            causal_byte_end = byte_end

        active_positions.append(token_index)
        starts.append(byte_start)
        ends.append(byte_end)
        specials.append(special)

    return (
        np.asarray(active_positions, dtype=np.int64),
        np.asarray(starts, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
        np.asarray(specials, dtype=np.bool_),
    )


def _tolist(value, field):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"tokenizer field {field} must be array-like")
    return value


def _unbatch_vector(value, field):
    value = _tolist(value, field)
    if (len(value) == 1 and isinstance(value[0], (list, tuple))):
        value = value[0]
    return list(value)


def _unbatch_offsets(value):
    value = _tolist(value, "offset_mapping")
    if (len(value) == 1 and isinstance(value[0], (list, tuple))
            and (not value[0]
                 or isinstance(value[0][0], (list, tuple)))):
        value = value[0]
    return list(value)


def _token_count(input_ids):
    shape = getattr(input_ids, "shape", None)
    if shape is not None:
        shape = tuple(int(size) for size in shape)
        if len(shape) != 2 or shape[0] != 1:
            raise ValueError(
                "tokenizer input_ids must have shape [1, tokens]"
            )
        return shape[1]
    values = _tolist(input_ids, "input_ids")
    if len(values) != 1 or not isinstance(values[0], (list, tuple)):
        raise ValueError("tokenizer input_ids must have one batch dimension")
    return len(values[0])


def encode_text(tokenizer, text):
    """Tokenize one text with exact offsets and return model-ready tensors."""
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "the tokenizer must be a fast tokenizer with offset_mapping support"
        )
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
        return_offsets_mapping=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    if not isinstance(encoded, Mapping):
        raise ValueError("tokenizer output must be a mapping")
    missing = [
        field for field in ("input_ids", "offset_mapping", "special_tokens_mask")
        if field not in encoded
    ]
    if missing:
        raise ValueError(
            "tokenizer output is missing field(s): " + ", ".join(missing)
        )

    token_count = _token_count(encoded["input_ids"])
    offsets = _unbatch_offsets(encoded["offset_mapping"])
    special_mask = _unbatch_vector(
        encoded["special_tokens_mask"], "special_tokens_mask"
    )
    if "attention_mask" in encoded:
        attention_mask = _unbatch_vector(
            encoded["attention_mask"], "attention_mask"
        )
    else:
        attention_mask = [1] * token_count
    if (len(offsets) != token_count or len(special_mask) != token_count
            or len(attention_mask) != token_count):
        raise ValueError(
            "tokenizer offset, special, attention, and input token counts differ"
        )

    active_positions, byte_start, byte_end, special = utf8_token_spans(
        text, offsets, special_mask, attention_mask
    )
    if active_positions.size == 0:
        raise ValueError("tokenizer produced no active tokens for a text")
    model_inputs = {
        name: value for name, value in encoded.items()
        if name not in _AUXILIARY_TOKENIZER_FIELDS
    }
    return EncodedText(
        model_inputs, token_count, active_positions,
        byte_start, byte_end, special
    )


def resolve_module(model, module_path):
    """Resolve a dotted attribute or numeric child path from a model."""
    if module_path in ("", "."):
        return model
    if not isinstance(module_path, str) or any(
            not part for part in module_path.split(".")):
        raise ValueError("module path must be a non-empty dotted path")

    getter = getattr(model, "get_submodule", None)
    if callable(getter):
        try:
            return getter(module_path)
        except (AttributeError, IndexError, KeyError):
            pass

    current = model
    traversed = []
    for part in module_path.split("."):
        traversed.append(part)
        if part.isdecimal():
            try:
                current = current[int(part)]
            except (IndexError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"cannot resolve module path at {'.'.join(traversed)!r}"
                ) from exc
        else:
            try:
                current = getattr(current, part)
            except AttributeError as exc:
                raise ValueError(
                    f"cannot resolve module path at {'.'.join(traversed)!r}"
                ) from exc
    return current


def _first_tensor_like(value, seen=None):
    """Select the first tensor-like value in a deterministic traversal."""
    if isinstance(value, np.ndarray):
        return value
    if (hasattr(value, "shape") and hasattr(value, "detach")
            and not isinstance(value, (str, bytes))):
        return value
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return None
    seen.add(identity)
    if isinstance(value, Mapping):
        iterable = value.values()
    elif isinstance(value, (list, tuple)):
        iterable = value
    else:
        return None
    for child in iterable:
        found = _first_tensor_like(child, seen)
        if found is not None:
            return found
    return None


def _activation_array(payload, hook_name):
    tensor = _first_tensor_like(payload)
    if tensor is None:
        raise ValueError(f"{hook_name} hook did not contain a tensor-like value")
    if isinstance(tensor, np.ndarray):
        array = tensor
    else:
        array = tensor.detach()
        if hasattr(array, "float"):
            array = array.float()
        if hasattr(array, "cpu"):
            array = array.cpu()
        if not hasattr(array, "numpy"):
            raise ValueError(
                f"{hook_name} hook tensor cannot be converted to NumPy"
            )
        array = array.numpy()
    try:
        return np.array(array, dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{hook_name} hook tensor must have a real numeric dtype"
        ) from exc


def _as_token_matrix(array, token_count, active_positions, hook_name):
    if array.ndim == 3 and array.shape[0] == 1:
        matrix = array[0]
    elif array.ndim == 2:
        matrix = array
    else:
        raise ValueError(
            f"{hook_name} hook tensor must have shape [1, tokens, width] "
            "or [tokens, width]"
        )
    if matrix.shape[0] != token_count:
        raise ValueError(
            f"{hook_name} hook token count {matrix.shape[0]} does not match "
            f"tokenizer count {token_count}"
        )
    if matrix.shape[1] <= 0:
        raise ValueError(f"{hook_name} hook tensor width must be positive")
    return np.ascontiguousarray(matrix[active_positions], dtype=np.float32)


def _register_hooks(module, capture, events):
    handles = []
    if capture in ("input", "both"):
        def capture_input(_module, inputs, kwargs):
            # Positional inputs take precedence, then keyword-only inputs.
            events["input"].append(
                _activation_array((inputs, kwargs), "input")
            )

        try:
            handle = module.register_forward_pre_hook(
                capture_input, with_kwargs=True
            )
        except TypeError:
            # This fallback keeps small module facades usable. Supported
            # PyTorch releases take the with_kwargs form above.
            def capture_positional_input(_module, inputs):
                events["input"].append(_activation_array(inputs, "input"))

            handle = module.register_forward_pre_hook(
                capture_positional_input
            )
        handles.append(handle)
    if capture in ("output", "both"):
        def capture_output(_module, _inputs, output):
            events["output"].append(_activation_array(output, "output"))

        handles.append(module.register_forward_hook(capture_output))
    return handles


def _move_inputs(model_inputs, device):
    moved = {}
    for name, value in model_inputs.items():
        if not hasattr(value, "to"):
            raise ValueError(
                f"tokenizer model field {name} is not a tensor with .to()"
            )
        moved[name] = value.to(device)
    return moved


def capture_records(model, tokenizer, records, module_path, capture="output",
                    device="cpu"):
    """Capture one or both token matrices and construct their shared index."""
    if capture not in ("input", "output", "both"):
        raise ValueError("capture must be 'input', 'output', or 'both'")
    if not records:
        raise ValueError("at least one text record is required")
    identifiers = [record.sequence_id for record in records]
    _sequence_id_array(identifiers)
    if len(set((type(value), value) for value in identifiers)) != len(records):
        raise ValueError("sequence_id values must be unique")

    module = resolve_module(model, module_path)
    wanted = ("input", "output") if capture == "both" else (capture,)
    events = {name: [] for name in wanted}
    chunks = {name: [] for name in wanted}
    widths = {}
    trace_sequence_id = []
    trace_byte_start = []
    trace_byte_end = []
    trace_content_sha256 = []
    trace_context_fingerprint = []
    trace_special = []
    handles = _register_hooks(module, capture, events)
    try:
        for record in records:
            encoded = encode_text(tokenizer, record.text)
            for name in wanted:
                events[name].clear()
            model_inputs = _move_inputs(encoded.model_inputs, device)
            model(**model_inputs, use_cache=False)

            for name in wanted:
                if len(events[name]) != 1:
                    raise ValueError(
                        f"selected module ran {len(events[name])} times for "
                        f"sequence_id {record.sequence_id!r}; exactly one "
                        "hook event is required"
                    )
                matrix = _as_token_matrix(
                    events[name][0], encoded.token_count,
                    encoded.active_positions, name
                )
                if name in widths and widths[name] != matrix.shape[1]:
                    raise ValueError(
                        f"{name} hook width changed from {widths[name]} to "
                        f"{matrix.shape[1]}"
                    )
                widths[name] = matrix.shape[1]
                chunks[name].append(matrix)

            row_count = encoded.active_positions.size
            content_hash = hashlib.sha256(
                record.text.encode("utf-8")
            ).hexdigest()
            trace_sequence_id.extend([record.sequence_id] * row_count)
            trace_byte_start.extend(encoded.byte_start.tolist())
            trace_byte_end.extend(encoded.byte_end.tolist())
            trace_content_sha256.extend([content_hash] * row_count)
            trace_context_fingerprint.extend(
                [CONTEXT_FINGERPRINT] * row_count
            )
            trace_special.extend(encoded.special.tolist())
    finally:
        for handle in handles:
            handle.remove()

    activations = {
        name: np.ascontiguousarray(np.concatenate(chunks[name], axis=0))
        for name in wanted
    }
    row_count = len(trace_sequence_id)
    for name, matrix in activations.items():
        if matrix.ndim != 2 or matrix.shape[0] != row_count:
            raise AssertionError(f"internal {name} activation row mismatch")
    trace = {
        "format_version": np.int64(TRACE_FORMAT_VERSION),
        "sequence_id": _sequence_id_array(trace_sequence_id),
        "row": np.arange(row_count, dtype=np.int64),
        "byte_start": np.asarray(trace_byte_start, dtype=np.int64),
        "byte_end": np.asarray(trace_byte_end, dtype=np.int64),
        "content_sha256": np.asarray(trace_content_sha256, dtype="U64"),
        "context_fingerprint": np.asarray(
            trace_context_fingerprint, dtype="U71"
        ),
        "special": np.asarray(trace_special, dtype=np.bool_),
        "offset_unit": np.asarray("utf8-byte"),
        "context_scheme": np.asarray("causal-prefix-v1"),
    }
    return activations, trace


def _npy_path(path):
    return str(path) if str(path).endswith(".npy") else str(path) + ".npy"


def _npz_path(path):
    return str(path) if str(path).endswith(".npz") else str(path) + ".npz"


def save_capture_files(activations, trace, output_paths, trace_path):
    """Write activation matrices and their one shared compressed trace."""
    if set(activations) != set(output_paths):
        raise ValueError("activation names and output path names differ")
    normalized_paths = {
        name: _npy_path(path) for name, path in output_paths.items()
    }
    absolute_paths = [
        os.path.abspath(path) for path in normalized_paths.values()
    ]
    if len(set(absolute_paths)) != len(absolute_paths):
        raise ValueError("activation output paths must resolve to distinct files")
    actual_paths = {}
    for name, matrix in activations.items():
        path = normalized_paths[name]
        np.save(path, matrix, allow_pickle=False)
        actual_paths[name] = path
    actual_trace_path = _npz_path(trace_path)
    np.savez_compressed(actual_trace_path, **trace)
    return actual_paths, actual_trace_path


class _FakeTensor:
    """Small tensor facade used only by the dependency-free selftest."""

    def __init__(self, array):
        self.array = np.asarray(array)

    @property
    def shape(self):
        return self.array.shape

    def to(self, _device):
        return self

    def detach(self):
        return self

    def float(self):
        return _FakeTensor(self.array.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def tolist(self):
        return self.array.tolist()


class _FakeHandle:
    def __init__(self, hooks, hook):
        self.hooks = hooks
        self.hook = hook

    def remove(self):
        if self.hook in self.hooks:
            self.hooks.remove(self.hook)


class _FakeModule:
    def __init__(self):
        self.pre_hooks = []
        self.post_hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        entry = (hook, with_kwargs)
        self.pre_hooks.append(entry)
        return _FakeHandle(self.pre_hooks, entry)

    def register_forward_hook(self, hook):
        self.post_hooks.append(hook)
        return _FakeHandle(self.post_hooks, hook)

    def forward(self, *, hidden_states):
        for hook, with_kwargs in tuple(self.pre_hooks):
            if with_kwargs:
                hook(self, (), {"hidden_states": hidden_states})
            else:
                hook(self, ())
        output = _FakeTensor(hidden_states.array + np.float32(100.0))
        for hook in tuple(self.post_hooks):
            hook(self, (), (output,))
        return output


class _FakeModel:
    def __init__(self):
        self.block = _FakeModule()

    def get_submodule(self, path):
        if path != "block":
            raise AttributeError(path)
        return self.block

    def __call__(self, input_ids, attention_mask=None, use_cache=False,
                 **_unused):
        if use_cache:
            raise AssertionError("selftest capture unexpectedly enabled cache")
        ids = input_ids.array.astype(np.float32)
        hidden = np.stack((ids, ids + 1.0, ids + 2.0), axis=-1)
        return self.block.forward(hidden_states=_FakeTensor(hidden))


class _FakeTokenizer:
    is_fast = True

    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, _text, **options):
        required = {
            "add_special_tokens": True,
            "padding": False,
            "truncation": False,
            "return_attention_mask": True,
            "return_offsets_mapping": True,
            "return_special_tokens_mask": True,
            "return_tensors": "pt",
        }
        if options != required:
            raise AssertionError(f"unexpected tokenizer options: {options!r}")
        token_count = len(self.offsets)
        # The last entry is padding and is deliberately absent from the trace.
        attention = [1] * (token_count - 1) + [0]
        special = [1] + [0] * (token_count - 4) + [1, 1, 1]
        return {
            "input_ids": _FakeTensor([np.arange(token_count)]),
            "attention_mask": _FakeTensor([attention]),
            "offset_mapping": _FakeTensor([self.offsets]),
            "special_tokens_mask": _FakeTensor([special]),
        }


def selftest():
    """Exercise Unicode offsets, padding removal, both hooks, and v2 output."""
    class FakeDtypes:
        float32 = object()
        float16 = object()
        bfloat16 = object()

    auto_options = _model_load_options(FakeDtypes, "auto", False)
    native_options = _model_load_options(FakeDtypes, "bfloat16", True)
    assert auto_options == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": "auto",
    }
    assert native_options["dtype"] is FakeDtypes.bfloat16
    assert native_options["trust_remote_code"] is True

    text = "Aé 東👩🏽‍💻!"
    offsets_a = [
        (0, 0), (0, 2), (2, 4), (4, 8), (8, 9),
        (0, 0), (0, 0), (0, 0),
    ]
    offsets_b = [
        (0, 0), (0, 1), (1, 4), (4, 6), (6, 9),
        (0, 0), (0, 0), (0, 0),
    ]
    records = [TextRecord("unicode", text)]
    activations, trace = capture_records(
        _FakeModel(), _FakeTokenizer(offsets_a), records,
        "block", capture="both"
    )
    _, trace_b = capture_records(
        _FakeModel(), _FakeTokenizer(offsets_b), records,
        "block", capture="output"
    )

    assert activations["input"].shape == (7, 3)
    assert activations["output"].shape == (7, 3)
    assert np.array_equal(
        activations["output"], activations["input"] + np.float32(100.0)
    )
    assert trace["row"].tolist() == list(range(7))
    assert trace["special"].tolist() == [True, False, False, False,
                                         False, True, True]
    assert trace["byte_start"].tolist() == [0, 0, 3, 7, 22, 23, 23]
    assert trace["byte_end"].tolist() == [0, 3, 7, 22, 23, 23, 23]
    assert np.all(trace["byte_start"][trace["special"]]
                  == trace["byte_end"][trace["special"]])
    assert trace["content_sha256"][0] == trace_b["content_sha256"][0]
    assert (trace["context_fingerprint"][0]
            == trace_b["context_fingerprint"][0])
    ordinary_a = set(trace["byte_end"][~trace["special"]].tolist())
    ordinary_b = set(trace_b["byte_end"][~trace_b["special"]].tolist())
    assert ordinary_a & ordinary_b == {7, 23}

    with tempfile.TemporaryDirectory() as temp_dir:
        output_paths, trace_path = save_capture_files(
            activations, trace,
            {
                "input": os.path.join(temp_dir, "input"),
                "output": os.path.join(temp_dir, "output"),
            },
            os.path.join(temp_dir, "trace"),
        )
        for name in ("input", "output"):
            stored = np.load(output_paths[name], allow_pickle=False)
            assert stored.ndim == 2 and stored.shape[0] == 7
        with np.load(trace_path, allow_pickle=False) as stored_trace:
            assert int(stored_trace["format_version"]) == 2
            assert stored_trace["row"].shape == (7,)
            assert stored_trace["content_sha256"].dtype.kind == "U"
            assert stored_trace["context_fingerprint"].dtype.kind == "U"
            assert stored_trace["special"].dtype == np.bool_
    print("hf_trace_capture selftest OK")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true",
                        help="run a lightweight capture without HF dependencies")
    parser.add_argument("--model", "--model-path", dest="model_path",
                        help="local Hugging Face model directory")
    parser.add_argument("--module", "--module-path", dest="module_path",
                        help="dotted module path below the loaded model")
    parser.add_argument("--capture", "--hook", choices=("input", "output", "both"),
                        default="output", help="module hook side to capture")
    parser.add_argument("--texts", "--texts-jsonl", dest="texts_path",
                        help="UTF-8 JSONL input path")
    parser.add_argument("--out", "--activations-out", dest="single_out",
                        help="activation .npy path for a single hook side")
    parser.add_argument("--out-input",
                        help="input activation .npy path for --capture both")
    parser.add_argument("--out-output",
                        help="output activation .npy path for --capture both")
    parser.add_argument("--trace-out", "--index-out", dest="trace_out",
                        help="format-v2 trace .npz path")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device for model and inputs (default: cpu)")
    parser.add_argument(
        "--dtype", choices=MODEL_DTYPE_CHOICES, default="auto",
        help="model load dtype (default: auto, preserve checkpoint dtype)",
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="allow Python code shipped inside the local model directory",
    )
    return parser


def _capture_output_paths(args, parser):
    if args.capture == "both":
        if args.single_out:
            parser.error("--out cannot be used with --capture both")
        if not args.out_input or not args.out_output:
            parser.error(
                "--capture both requires --out-input and --out-output"
            )
        input_path = os.path.abspath(_npy_path(args.out_input))
        output_path = os.path.abspath(_npy_path(args.out_output))
        if input_path == output_path:
            parser.error("--out-input and --out-output must differ")
        return {"input": args.out_input, "output": args.out_output}
    if args.out_input or args.out_output:
        parser.error("--out-input/--out-output require --capture both")
    if not args.single_out:
        parser.error("single-side capture requires --out")
    return {args.capture: args.single_out}


def _model_load_options(torch_module, dtype_name, trust_remote_code):
    """Return closed, testable options for local pretrained model loading."""
    dtype_map = {
        "auto": "auto",
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    try:
        dtype = dtype_map[dtype_name]
    except KeyError as exc:
        raise ValueError(
            f"dtype must be one of: {', '.join(MODEL_DTYPE_CHOICES)}"
        ) from exc
    return {
        "local_files_only": True,
        "trust_remote_code": bool(trust_remote_code),
        "dtype": dtype,
    }


def _effective_model_dtype(model):
    """Describe the first floating parameter dtype of the loaded model."""
    try:
        parameters = model.parameters()
    except (AttributeError, TypeError):
        return "unknown"
    first_dtype = None
    for parameter in parameters:
        dtype = getattr(parameter, "dtype", None)
        if first_dtype is None and dtype is not None:
            first_dtype = dtype
        is_floating = getattr(parameter, "is_floating_point", None)
        if callable(is_floating) and is_floating():
            return str(dtype).removeprefix("torch.")
    if first_dtype is None:
        return "unknown"
    return str(first_dtype).removeprefix("torch.")


def _load_hf_runtime(model_path, device, trust_remote_code, dtype_name):
    """Import optional dependencies and load only local model assets."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for capture; install it in the runtime environment"
        ) from exc
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Transformers is required for capture; install it in the "
            "runtime environment"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=trust_remote_code,
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "the local model does not provide a fast tokenizer with offsets"
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **_model_load_options(torch, dtype_name, trust_remote_code),
    )
    model.to(device)
    model.eval()
    return torch, model, tokenizer, _effective_model_dtype(model)


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        selftest()
        return 0

    missing = [
        option for option, value in (
            ("--model", args.model_path),
            ("--module", args.module_path),
            ("--texts", args.texts_path),
            ("--trace-out", args.trace_out),
        ) if not value
    ]
    if missing:
        parser.error("capture requires " + ", ".join(missing))
    if not os.path.isdir(args.model_path):
        parser.error("--model must name an existing local directory")
    output_paths = _capture_output_paths(args, parser)

    try:
        records = read_text_jsonl(args.texts_path)
        torch, model, tokenizer, effective_dtype = _load_hf_runtime(
            args.model_path, args.device, args.trust_remote_code, args.dtype
        )
        with torch.inference_mode():
            activations, trace = capture_records(
                model, tokenizer, records, args.module_path,
                capture=args.capture, device=args.device
            )
        actual_outputs, actual_trace = save_capture_files(
            activations, trace, output_paths, args.trace_out
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"hf_trace_capture: {exc}", file=sys.stderr)
        return 2

    rows = int(trace["row"].size)
    widths = ", ".join(
        f"{name}={matrix.shape[1]}" for name, matrix in activations.items()
    )
    print(
        f"captured {rows} token rows ({widths}) from {len(records)} texts; "
        f"model dtype={effective_dtype}"
    )
    for name, path in actual_outputs.items():
        print(f"{name} activations: {path}")
    print(f"trace index: {actual_trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
