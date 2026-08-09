#!/usr/bin/env python3
"""Focused verification for tokenizer-independent causal span alignment."""

import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
NANO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, NANO)

import span_align


def _write(path, sequence_id, row, starts, ends, content_sha256,
           context_fingerprint, special=None):
    fields = {
        "sequence_id": np.asarray(sequence_id),
        "row": np.asarray(row, dtype=np.int64),
        "byte_start": np.asarray(starts, dtype=np.int64),
        "byte_end": np.asarray(ends, dtype=np.int64),
        "content_sha256": np.asarray(content_sha256),
        "context_fingerprint": np.asarray(context_fingerprint),
    }
    if special is not None:
        fields["special"] = np.asarray(special, dtype=np.bool_)
    np.savez(path, **fields)


def focused_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "source.npz")
        target_path = os.path.join(temp_dir, "target.npz")
        output_path = os.path.join(temp_dir, "pairs.npz")
        unicode_hash = hashlib.sha256("é東!".encode("utf-8")).hexdigest()
        duplicate_hash = hashlib.sha256(b"d").hexdigest()
        mismatch_hash = hashlib.sha256(b"m").hexdigest()
        ascii_hash = hashlib.sha256(b"ab").hexdigest()

        # "é東!" is 6 UTF-8 bytes. The ordinary causal endpoints shared by
        # these two tokenizations are byte 5 and byte 6, not token positions.
        # Sequence "dup" deliberately has two ordinary states at byte 1.
        # "mismatch" has the same content digest but a different context.
        _write(
            source_path,
            ["unicode", "unicode", "unicode", "unicode", "dup", "dup",
             "mismatch", "ascii", "ascii"],
            [9, 10, 11, 12, 30, 31, 40, 50, 51],
            [0, 0, 2, 5, 0, 1, 0, 0, 1],
            [0, 2, 5, 6, 1, 1, 1, 1, 2],
            [unicode_hash] * 4 + [duplicate_hash] * 2
            + [mismatch_hash] + [ascii_hash] * 2,
            ["ctx-v1"] * 6 + ["context-a"] + ["ctx-v1"] * 2,
            [True, False, False, False, False, False, False, False, False],
        )
        _write(
            target_path,
            ["dup", "unicode", "unicode", "unicode", "mismatch", "ascii"],
            [80, 60, 61, 62, 90, 70],
            [0, 0, 0, 5, 0, 0],
            [1, 0, 5, 6, 1, 2],
            [duplicate_hash] + [unicode_hash] * 3
            + [mismatch_hash, ascii_hash],
            ["ctx-v1"] * 4 + ["context-b", "ctx-v1"],
            [False, True, False, False, False, False],
        )

        source = span_align.load_trace_index(source_path, "test source")
        target = span_align.load_trace_index(target_path, "test target")
        output, report = span_align.align_trace_indexes(
            source, target, holdout_fraction=0.5, split_seed=41
        )
        repeated, _ = span_align.align_trace_indexes(
            source, target, holdout_fraction=0.5, split_seed=41
        )

        assert output["source_row"].tolist() == [51, 11, 12]
        assert output["target_row"].tolist() == [70, 61, 62]
        assert output["byte_end"].tolist() == [2, 5, 6]
        assert report["paired_rows"] == 3
        assert report["ambiguous_shared_endpoints"] == 1
        assert report["context_mismatched_sequences"] == 1
        assert report["fingerprint_mismatched_sequences"] == 1
        assert report["context_rejected_shared_endpoints"] == 1
        assert report["source_coverage"] == 3 / 8
        assert report["target_coverage"] == 3 / 5
        assert report["train_sequences"] == 1
        assert report["holdout_sequences"] == 1
        assert report["train_pairs"] + report["holdout_pairs"] == 3
        assert np.array_equal(output["is_holdout"], repeated["is_holdout"])
        for sequence_id in ("ascii", "unicode"):
            split = output["is_holdout"][output["sequence_id"] == sequence_id]
            assert np.unique(split).size == 1
        assert output["is_holdout"][0] != output["is_holdout"][1]
        assert output["context_mismatch_sequence_id"].tolist() == ["mismatch"]
        assert output["content_sha256"].tolist() == [
            ascii_hash, unicode_hash, unicode_hash
        ]

        span_align.save_alignment(output_path, output)
        with np.load(output_path, allow_pickle=False) as stored:
            assert int(stored["format_version"]) == 2
            assert str(stored["split_unit"]) == "sequence"
            assert (stored["train_pair_index"].size
                    + stored["holdout_pair_index"].size == 3)
            assert stored["ambiguous_byte_end"].tolist() == [1]

        cli_output_path = os.path.join(temp_dir, "cli-pairs.npz")
        cli = subprocess.run(
            [
                sys.executable,
                "-W", "error",
                os.path.join(NANO, "span_align.py"),
                "--source-index", source_path,
                "--target-index", target_path,
                "--out", cli_output_path,
                "--holdout-fraction", "0.5",
                "--split-seed", "41",
            ],
            capture_output=True,
            text=True,
        )
        if cli.returncode != 0:
            print(cli.stdout, end="")
            print(cli.stderr, end="", file=sys.stderr)
            raise AssertionError("span_align CLI failed")
        assert "source=37.50%" in cli.stdout
        assert "shared=1" in cli.stdout
        with np.load(cli_output_path, allow_pickle=False) as cli_stored:
            assert cli_stored["source_row"].tolist() == [51, 11, 12]
            assert np.array_equal(
                cli_stored["is_holdout"], output["is_holdout"]
            )

        invalid_path = os.path.join(temp_dir, "overlap.npz")
        invalid_hash = hashlib.sha256(b"invalid").hexdigest()
        _write(
            invalid_path,
            [1, 1], [0, 1], [0, 1], [2, 3],
            [invalid_hash, invalid_hash], ["ctx", "ctx"],
        )
        try:
            span_align.load_trace_index(invalid_path)
        except ValueError as exc:
            assert "overlaps or precedes" in str(exc)
        else:
            raise AssertionError("overlapping spans were not rejected")

        inconsistent_path = os.path.join(temp_dir, "inconsistent.npz")
        _write(
            inconsistent_path,
            ["same", "same"], [0, 1], [0, 1], [1, 2],
            [invalid_hash, invalid_hash], ["context-a", "context-b"],
        )
        try:
            span_align.load_trace_index(inconsistent_path)
        except ValueError as exc:
            assert "context_fingerprint must be constant" in str(exc)
        else:
            raise AssertionError("inconsistent sequence context was accepted")


def main():
    focused_test()
    result = subprocess.run(
        [
            sys.executable, "-W", "error",
            os.path.join(NANO, "span_align.py"), "--selftest",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if result.returncode != 0 or "span_align selftest OK" not in result.stdout:
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit("test_span_align FAILED")
    print("test_span_align OK")


if __name__ == "__main__":
    main()
