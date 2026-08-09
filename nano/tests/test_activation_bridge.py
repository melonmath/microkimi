#!/usr/bin/env python3
"""Run the deterministic activation_bridge synthetic verification."""

import os
import subprocess
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
NANO_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, NANO_DIR)

from activation_bridge import GramAccumulator, relative_residual  # noqa: E402


def test_relative_residual_matches_rows():
    """The sufficient-statistics formula must match an explicit residual."""
    rng = np.random.default_rng(17)
    source = rng.normal(size=(19, 4))
    target = rng.normal(size=(19, 7))
    weight = rng.normal(size=(7, 4))
    bias = rng.normal(size=7)
    stats = GramAccumulator(4, 7)
    stats.add(source, target)

    prediction = source @ weight.T + bias
    expected = np.sum((prediction - target) ** 2) / np.sum(target ** 2)
    actual = relative_residual(stats, weight, bias)
    assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_zero_target_and_wide_output():
    """Zero targets are defined and wide outputs need no square temporary."""
    output_width = 50_000
    source = np.array([[1.0, -2.0], [-0.5, 3.0]], dtype=np.float64)
    target = np.zeros((source.shape[0], output_width), dtype=np.float64)
    stats = GramAccumulator(source.shape[1], output_width)
    stats.add(source, target)
    weight = np.zeros((output_width, source.shape[1]), dtype=np.float64)
    bias = np.zeros(output_width, dtype=np.float64)

    assert relative_residual(stats, weight, bias) == 0.0
    weight[0, 0] = 1.0
    assert np.isposinf(relative_residual(stats, weight, bias))


def main():
    test_relative_residual_matches_rows()
    test_zero_target_and_wide_output()
    result = subprocess.run(
        [
            sys.executable,
            os.path.join(HERE, "..", "activation_bridge.py"),
            "--selftest",
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout, end="")
    if (result.returncode != 0
            or "activation_bridge selftest OK" not in result.stdout):
        print(result.stderr, end="", file=sys.stderr)
        raise SystemExit("test_activation_bridge FAILED")
    print("test_activation_bridge OK")


if __name__ == "__main__":
    main()
