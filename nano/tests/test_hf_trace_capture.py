#!/usr/bin/env python3
"""Focused tests for dependency-light Hugging Face activation capture."""

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
NANO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, NANO)

import hf_trace_capture as capture  # noqa: E402
import span_align  # noqa: E402


def test_module_import_does_not_require_hf_dependencies():
    """A fresh import must not touch torch or transformers."""
    code = f"""
import builtins
import sys

real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in ('torch', 'transformers'):
        raise AssertionError('optional HF dependency imported at module load')
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
sys.path.insert(0, {NANO!r})
import hf_trace_capture
assert hf_trace_capture.TRACE_FORMAT_VERSION == 2
assert hf_trace_capture.main(['--selftest']) == 0
"""
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        raise AssertionError("dependency-free import failed")


def test_utf8_offsets_specials_and_padding():
    """Character offsets become bytes and special rows stay zero-width."""
    text = "é東👩🏽‍💻!"
    # BOS, accent, CJK, emoji cluster, punctuation, EOS, padding.
    offsets = [
        (0, 0), (0, 1), (1, 2), (2, 6), (6, 7), (0, 0), (0, 0)
    ]
    active, starts, ends, special = capture.utf8_token_spans(
        text,
        offsets,
        [1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 0],
    )
    assert active.tolist() == [0, 1, 2, 3, 4, 5]
    assert starts.tolist() == [0, 0, 2, 5, 20, 21]
    assert ends.tolist() == [0, 2, 5, 20, 21, 21]
    assert special.tolist() == [True, False, False, False, False, True]
    assert np.all(starts[special] == ends[special])
    assert ends[-1] == len(text.encode("utf-8"))


def test_overlapping_fallback_offsets_become_ordinary_zero_width_rows():
    """Repeated byte-fallback endpoints stay ordinary and monotonic."""
    text = "naïve café 漢字"
    # The middle fallback entries overlap and repeat one character endpoint.
    offsets = [(0, 5), (5, 10), (10, 12), (11, 12), (11, 12), (12, 13)]
    active, starts, ends, special = capture.utf8_token_spans(
        text, offsets, [0] * len(offsets)
    )
    assert active.tolist() == list(range(len(offsets)))
    assert starts.tolist() == [0, 6, 12, 16, 16, 16]
    assert ends.tolist() == [6, 12, 16, 16, 16, 19]
    assert special.tolist() == [False] * len(offsets)
    assert np.all(starts[1:] >= ends[:-1])
    assert np.count_nonzero(starts == ends) == 2

    _, gap_starts, gap_ends, _ = capture.utf8_token_spans(
        " x", [(1, 2)], [0]
    )
    assert gap_starts.tolist() == [1]
    assert gap_ends.tolist() == [2]


def test_jsonl_records_are_stable_and_typed():
    """Both accepted record forms preserve deterministic sequence IDs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        integer_path = os.path.join(temp_dir, "integer.jsonl")
        with open(integer_path, "w", encoding="utf-8") as handle:
            json.dump("first", handle, ensure_ascii=False)
            handle.write("\n\n")
            json.dump({"text": "東", "sequence_id": 1}, handle,
                      ensure_ascii=False)
            handle.write("\n")
        records = capture.read_text_jsonl(integer_path)
        assert [record.sequence_id for record in records] == [0, 1]
        assert [record.text for record in records] == ["first", "東"]

        duplicate_path = os.path.join(temp_dir, "duplicate.jsonl")
        with open(duplicate_path, "w", encoding="utf-8") as handle:
            handle.write('{"sequence_id":"x","text":"a"}\n')
            handle.write('{"id":"x","text":"b"}\n')
        try:
            capture.read_text_jsonl(duplicate_path)
        except ValueError as exc:
            assert "must be unique" in str(exc)
        else:
            raise AssertionError("duplicate sequence identifiers were accepted")

        invalid_id_path = os.path.join(temp_dir, "invalid-id.jsonl")
        with open(invalid_id_path, "w", encoding="utf-8") as handle:
            handle.write('{"sequence_id":"\\ud800","text":"a"}\n')
        try:
            capture.read_text_jsonl(invalid_id_path)
        except ValueError as exc:
            assert "sequence_id is not valid UTF-8" in str(exc)
        else:
            raise AssertionError("invalid Unicode sequence_id was accepted")


def _capture_tokenizations():
    text = "Aé 東👩🏽‍💻!"
    offsets_a = [
        (0, 0), (0, 2), (2, 4), (4, 8), (8, 9),
        (0, 0), (0, 0), (0, 0),
    ]
    offsets_b = [
        (0, 0), (0, 1), (1, 4), (4, 6), (6, 9),
        (0, 0), (0, 0), (0, 0),
    ]
    records = [capture.TextRecord("same", text)]
    activations, trace_a = capture.capture_records(
        capture._FakeModel(), capture._FakeTokenizer(offsets_a), records,
        "block", capture="both"
    )
    _, trace_b = capture.capture_records(
        capture._FakeModel(), capture._FakeTokenizer(offsets_b), records,
        "block", capture="output"
    )
    return text, activations, trace_a, trace_b


def test_both_hooks_share_span_align_v2_rows():
    """Input and output matrices share one validated row index."""
    text, activations, trace_a, trace_b = _capture_tokenizations()
    assert activations["input"].shape == activations["output"].shape == (7, 3)
    assert np.array_equal(
        activations["output"], activations["input"] + np.float32(100.0)
    )
    expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert np.all(trace_a["content_sha256"] == expected_hash)
    assert np.all(
        trace_a["context_fingerprint"] == capture.CONTEXT_FINGERPRINT
    )
    assert (trace_a["content_sha256"][0]
            == trace_b["content_sha256"][0])
    assert (trace_a["context_fingerprint"][0]
            == trace_b["context_fingerprint"][0])

    with tempfile.TemporaryDirectory() as temp_dir:
        output_paths, trace_a_path = capture.save_capture_files(
            activations,
            trace_a,
            {
                "input": os.path.join(temp_dir, "input.npy"),
                "output": os.path.join(temp_dir, "output.npy"),
            },
            os.path.join(temp_dir, "a.trace.npz"),
        )
        _, trace_b_path = capture.save_capture_files(
            {"output": np.zeros((7, 2), dtype=np.float32)},
            trace_b,
            {"output": os.path.join(temp_dir, "other.npy")},
            os.path.join(temp_dir, "b.trace.npz"),
        )
        source = span_align.load_trace_index(trace_a_path, "captured source")
        target = span_align.load_trace_index(trace_b_path, "captured target")
        aligned, report = span_align.align_trace_indexes(
            source, target, holdout_fraction=0.0
        )
        assert aligned["byte_end"].tolist() == [7, 23]
        assert report["paired_rows"] == 2
        assert report["context_mismatched_sequences"] == 0
        assert source.row.tolist() == list(range(7))
        for path in output_paths.values():
            stored = np.load(path, allow_pickle=False)
            assert stored.dtype == np.float32
            assert stored.shape[0] == source.row.size


def test_slow_tokenizer_is_rejected_before_model_execution():
    class SlowTokenizer:
        is_fast = False

    try:
        capture.encode_text(SlowTokenizer(), "text")
    except ValueError as exc:
        assert "fast tokenizer" in str(exc)
    else:
        raise AssertionError("slow tokenizer was accepted")


def test_dtype_options_and_normalized_output_collision():
    """Dtype is relayed explicitly and suffix aliases cannot overwrite."""
    class FakeDtypes:
        float32 = object()
        float16 = object()
        bfloat16 = object()

    options = capture._model_load_options(FakeDtypes, "float16", False)
    assert options == {
        "local_files_only": True,
        "trust_remote_code": False,
        "dtype": FakeDtypes.float16,
    }
    assert capture._model_load_options(
        FakeDtypes, "auto", False
    )["dtype"] == "auto"

    parser = capture._build_parser()
    args = parser.parse_args([
        "--capture", "both",
        "--out-input", "same",
        "--out-output", "same.npy",
    ])
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            capture._capture_output_paths(args, parser)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError("suffix-equivalent outputs were accepted")


def main():
    test_module_import_does_not_require_hf_dependencies()
    test_utf8_offsets_specials_and_padding()
    test_overlapping_fallback_offsets_become_ordinary_zero_width_rows()
    test_jsonl_records_are_stable_and_typed()
    test_both_hooks_share_span_align_v2_rows()
    test_slow_tokenizer_is_rejected_before_model_execution()
    test_dtype_options_and_normalized_output_collision()
    result = subprocess.run(
        [
            sys.executable,
            "-W", "error",
            os.path.join(NANO, "hf_trace_capture.py"),
            "--selftest",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if (result.returncode != 0
            or "hf_trace_capture selftest OK" not in result.stdout):
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit("test_hf_trace_capture FAILED")
    print("test_hf_trace_capture OK")


if __name__ == "__main__":
    main()
