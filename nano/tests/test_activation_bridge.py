#!/usr/bin/env python3
"""Run the deterministic activation_bridge synthetic verification."""

import os
import subprocess
import sys

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
NANO_DIR = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, NANO_DIR)

from activation_bridge import (  # noqa: E402
    GramAccumulator,
    relative_residual,
    solve_bridge,
)


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


def _stored_weight(solution):
    return (
        solution["output_weight"].astype(np.float64)
        @ solution["input_weight"].astype(np.float64)
    )


def _ridge_objective(source, target, solution):
    weight = _stored_weight(solution)
    bias = solution["bias"].astype(np.float64)
    error = source @ weight.T + bias - target
    return (
        float(np.sum(error * error))
        + float(solution["ridge_lambda"]) * float(np.sum(weight * weight))
    )


def test_covariance_aware_rank_minimizes_ridge_geometry():
    """Whitened truncation must beat coefficient SVD on its own objective."""
    rng = np.random.default_rng(31)
    rows, input_width, output_width = 600, 8, 6
    scales = np.geomspace(0.015, 30.0, input_width)
    source = rng.normal(size=(rows, input_width)) * scales
    target = source @ rng.normal(size=(output_width, input_width)).T
    target += 0.2 * rng.normal(size=(rows, output_width))
    stats = GramAccumulator(input_width, output_width)
    stats.add(source, target)

    ridge_rank = solve_bridge(
        stats, rank=2, rel_lambda=0.03, rank_metric="ridge"
    )
    coefficient_rank = solve_bridge(
        stats, rank=2, rel_lambda=0.03, rank_metric="coefficient"
    )
    ridge_objective = _ridge_objective(source, target, ridge_rank)
    coefficient_objective = _ridge_objective(
        source, target, coefficient_rank
    )
    assert ridge_objective <= coefficient_objective * (1.0 + 2e-7)
    assert str(ridge_rank["rank_metric"]) == "ridge"
    assert str(coefficient_rank["rank_metric"]) == "coefficient"


def test_covariance_rank_handles_singular_unregularized_source():
    source = np.asarray(
        [[1.0, 2.0, 2.0], [2.0, 4.0, 4.0]], dtype=np.float64
    )
    target = np.asarray([[3.0, -1.0], [6.0, -2.0]], dtype=np.float64)
    stats = GramAccumulator(3, 2)
    stats.add(source, target)
    solution = solve_bridge(
        stats, rank=1, rel_lambda=0.0, rank_metric="ridge"
    )
    assert np.isfinite(solution["input_weight"]).all()
    assert np.isfinite(solution["output_weight"]).all()
    assert float(solution["rank_relative_residual"]) < 1e-12


def test_zero_ridge_never_accepts_a_fake_cholesky_pivot():
    """A numerically accepted singular Gram must still use its pseudoinverse."""
    source = np.asarray(
        [[4.0, -3.0, -4.0], [-2.0, -1.0, 3.0]], dtype=np.float64
    )
    target = np.asarray([[2.0, -1.0], [5.0, 3.0]], dtype=np.float64)
    stats = GramAccumulator(3, 2)
    stats.add(source, target)
    solution = solve_bridge(
        stats, rank=2, rel_lambda=0.0, rank_metric="ridge"
    )
    weight = _stored_weight(solution)
    expected = np.linalg.lstsq(source, target, rcond=None)[0].T
    assert np.allclose(weight, expected, rtol=2e-6, atol=2e-6)
    assert np.linalg.norm(solution["input_weight"]) < 10.0
    assert int(solution["format_version"]) == 2


def main():
    test_relative_residual_matches_rows()
    test_zero_target_and_wide_output()
    test_covariance_aware_rank_minimizes_ridge_geometry()
    test_covariance_rank_handles_singular_unregularized_source()
    test_zero_ridge_never_accepts_a_fake_cholesky_pivot()
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
