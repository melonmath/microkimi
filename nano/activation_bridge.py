#!/usr/bin/env python3
"""Fit a low-rank linear bridge between paired activation arrays.

Each source file and target file is a NumPy ``.npy`` array. Rows are paired
observations, while the source and target widths may differ. The fitted map is

    target = source @ weight.T + bias

The input files are memory-mapped and consumed in row chunks. Only the
float64 sufficient statistics X.T X, X.T Y, the first moments, and the target
squared norm remain resident. Ridge strength is relative to the mean diagonal
of X.T X, so it follows both sample count and activation scale.

The output is a NumPy ``.npz`` file containing a balanced low-rank split:

    hidden = torch.nn.functional.linear(source, input_weight)
    target = torch.nn.functional.linear(hidden, output_weight, bias)

``input_weight`` has shape ``[rank, source_width]`` and ``output_weight`` has
shape ``[target_width, rank]``. These are already in PyTorch Linear weight
orientation. With ``--affine``, the bias is fitted without regularization and
is recomputed after rank truncation so the factorized map preserves the target
mean as closely as possible.

By default, rank reduction is performed after whitening with the regularized
source Gram. This minimizes the same ridge objective as the dense solve under
the requested rank constraint. ``--rank-metric coefficient`` retains ordinary
Euclidean SVD truncation of the dense coefficient matrix when that is the
desired notion of approximation.

Examples:

  python3 activation_bridge.py --source source.npy --target target.npy \
      --rank 64 --affine --out bridge.npz
  python3 activation_bridge.py --source shard0.x.npy --target shard0.y.npy \
      --source shard1.x.npy --target shard1.y.npy --out bridge.npz
  python3 activation_bridge.py --selftest
"""

import argparse
import os
import tempfile

import numpy as np


class GramAccumulator:
    """Streaming sufficient statistics for a rectangular linear map."""

    def __init__(self, input_width, output_width):
        if input_width <= 0 or output_width <= 0:
            raise ValueError("input and output widths must be positive")
        self.input_width = int(input_width)
        self.output_width = int(output_width)
        self.gxx = np.zeros((input_width, input_width), dtype=np.float64)
        self.gxy = np.zeros((input_width, output_width), dtype=np.float64)
        self.sum_x = np.zeros(input_width, dtype=np.float64)
        self.sum_y = np.zeros(output_width, dtype=np.float64)
        self.sum_sq_y = 0.0
        self.n = 0

    def add(self, source, target):
        """Accumulate one 2D row-paired block using float64 arithmetic."""
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("source and target blocks must be 2D")
        if source.shape[0] != target.shape[0]:
            raise ValueError("source and target blocks must have equal rows")
        if source.shape[1] != self.input_width:
            raise ValueError(
                f"source width {source.shape[1]} does not match "
                f"{self.input_width}"
            )
        if target.shape[1] != self.output_width:
            raise ValueError(
                f"target width {target.shape[1]} does not match "
                f"{self.output_width}"
            )
        if not np.isfinite(source).all() or not np.isfinite(target).all():
            raise ValueError("source and target blocks must contain finite values")
        if source.shape[0] == 0:
            return
        self.gxx += source.T @ source
        self.gxy += source.T @ target
        self.sum_x += source.sum(axis=0)
        self.sum_y += target.sum(axis=0)
        self.sum_sq_y += float(np.sum(target * target))
        self.n += source.shape[0]


def _check_array(path, role):
    """Open a numeric 2D .npy file without reading its payload eagerly."""
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load {role} array {path}: {exc}") from exc
    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise ValueError(f"{role} array {path} must be a 2D .npy array")
    if not (np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.integer)):
        raise ValueError(f"{role} array {path} must have a real numeric dtype")
    return array


def accumulate_npy_pairs(source_paths, target_paths, chunk_rows=4096):
    """Stream one or more corresponding .npy file pairs into Gram statistics."""
    if len(source_paths) != len(target_paths):
        raise ValueError("the number of --source and --target files must match")
    if not source_paths:
        raise ValueError("at least one source and target pair is required")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    accumulator = None
    for source_path, target_path in zip(source_paths, target_paths):
        source = _check_array(source_path, "source")
        target = _check_array(target_path, "target")
        if source.shape[0] != target.shape[0]:
            raise ValueError(
                f"row mismatch for {source_path} and {target_path}: "
                f"{source.shape[0]} != {target.shape[0]}"
            )
        if accumulator is None:
            accumulator = GramAccumulator(source.shape[1], target.shape[1])
        elif (source.shape[1] != accumulator.input_width
              or target.shape[1] != accumulator.output_width):
            raise ValueError(
                "all file pairs must use the same source and target widths"
            )
        for row0 in range(0, source.shape[0], chunk_rows):
            row1 = min(row0 + chunk_rows, source.shape[0])
            accumulator.add(source[row0:row1], target[row0:row1])

    if accumulator.n == 0:
        raise ValueError("the input arrays contain no paired rows")
    return accumulator


def centered_grams(stats, affine):
    """Return solve Grams, centering first when an unregularized bias is used."""
    if stats.n <= 0:
        raise ValueError("cannot solve without paired rows")
    if not affine:
        return stats.gxx, stats.gxy
    mean_x = stats.sum_x / stats.n
    mean_y = stats.sum_y / stats.n
    gxx = stats.gxx - stats.n * np.outer(mean_x, mean_x)
    gxy = stats.gxy - stats.n * np.outer(mean_x, mean_y)
    # Symmetrizing removes tiny subtraction asymmetry before the solve.
    return 0.5 * (gxx + gxx.T), gxy


def ridge_solve(stats, rel_lambda=1e-4, affine=False):
    """Return dense PyTorch-oriented weight, bias, and absolute ridge value."""
    if rel_lambda < 0 or not np.isfinite(rel_lambda):
        raise ValueError("rel_lambda must be finite and non-negative")
    gxx, gxy = centered_grams(stats, affine)
    scale = float(np.trace(gxx) / stats.input_width)
    # A constant source has a zero centered Gram. A unit fallback keeps a
    # positive relative ridge well-defined and yields the zero weight map.
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("source Gram has an invalid diagonal scale")
    ridge_lambda = rel_lambda * (scale if scale > 0 else 1.0)
    system = gxx + ridge_lambda * np.eye(stats.input_width, dtype=np.float64)
    if ridge_lambda == 0:
        # An exactly singular Gram can still pass a numerical solve or
        # Cholesky with a tiny artificial pivot. Use the minimum-norm path
        # deliberately for unregularized least squares.
        weight_t = np.linalg.lstsq(system, gxy, rcond=None)[0]
    else:
        try:
            weight_t = np.linalg.solve(system, gxy)
        except np.linalg.LinAlgError:
            weight_t = np.linalg.lstsq(system, gxy, rcond=None)[0]
    weight = weight_t.T
    if affine:
        bias = stats.sum_y / stats.n - weight @ (stats.sum_x / stats.n)
    else:
        bias = np.zeros(stats.output_width, dtype=np.float64)
    return weight, bias, ridge_lambda


def svd_factors(weight, rank):
    """Balanced W = output_weight @ input_weight truncated to ``rank``."""
    weight = np.asarray(weight, dtype=np.float64)
    if weight.ndim != 2:
        raise ValueError("weight must be 2D")
    if rank <= 0:
        raise ValueError("rank must be positive")
    u, singular_values, vt = np.linalg.svd(weight, full_matrices=False)
    effective_rank = min(int(rank), singular_values.size)
    roots = np.sqrt(singular_values[:effective_rank])
    input_weight = roots[:, None] * vt[:effective_rank, :]
    output_weight = u[:, :effective_rank] * roots[None, :]
    return input_weight, output_weight, singular_values


def ridge_metric_factors(stats, rank, ridge_lambda, affine):
    """Return reduced-rank factors optimal in regularized source geometry.

    Let ``B`` be the source-by-target coefficient and
    ``G = X.T X + ridge_lambda I`` (using centered X for an affine fit). The
    ridge objective around its dense optimum is the Frobenius distance between
    ``G**0.5 B`` and ``G**0.5 B_dense``. Truncating in that whitened space and
    mapping back through ``G**-0.5`` solves the rank-constrained objective.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    gxx, gxy = centered_grams(stats, affine)
    system = gxx + ridge_lambda * np.eye(
        stats.input_width, dtype=np.float64
    )
    lower = None
    if ridge_lambda > 0:
        try:
            lower = np.linalg.cholesky(system)
        except np.linalg.LinAlgError:
            pass
    if lower is None:
        # rel_lambda=0 intentionally permits a singular ordinary least-squares
        # problem. The symmetric pseudoinverse square root keeps only the
        # source-supported subspace instead of inventing null-space weights.
        eigenvalues, eigenvectors = np.linalg.eigh(
            0.5 * (system + system.T)
        )
        largest = max(float(eigenvalues[-1]), 0.0)
        tolerance = (
            stats.input_width * np.finfo(np.float64).eps * largest
        )
        keep = eigenvalues > tolerance
        if not keep.any():
            effective_rank = min(
                int(rank), stats.input_width, stats.output_width
            )
            return (
                np.zeros((effective_rank, stats.input_width), dtype=np.float64),
                np.zeros((stats.output_width, effective_rank), dtype=np.float64),
                np.zeros(min(stats.input_width, stats.output_width),
                         dtype=np.float64),
            )
        inverse_sqrt = (
            (eigenvectors[:, keep] / np.sqrt(eigenvalues[keep]))
            @ eigenvectors[:, keep].T
        )
        whitened = inverse_sqrt @ gxy
        u, singular_values, vt = np.linalg.svd(
            whitened, full_matrices=False
        )
        effective_rank = min(int(rank), singular_values.size)
        roots = np.sqrt(singular_values[:effective_rank])
        source_columns = (
            inverse_sqrt @ u[:, :effective_rank]
        ) * roots[None, :]
    else:
        # If G = L L.T, then L.T B_dense = solve(L, X.T Y).
        whitened = np.linalg.solve(lower, gxy)
        u, singular_values, vt = np.linalg.svd(
            whitened, full_matrices=False
        )
        effective_rank = min(int(rank), singular_values.size)
        roots = np.sqrt(singular_values[:effective_rank])
        source_columns = np.linalg.solve(
            lower.T, u[:, :effective_rank] * roots[None, :]
        )
    input_weight = source_columns.T
    output_weight = vt[:effective_rank, :].T * roots[None, :]
    return input_weight, output_weight, singular_values


def relative_residual(stats, weight, bias):
    """Compute ||X W.T + b - Y||^2 / ||Y||^2 from the statistics."""
    weight = np.asarray(weight, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if weight.shape != (stats.output_width, stats.input_width):
        raise ValueError("weight has the wrong shape for these statistics")
    if bias.shape != (stats.output_width,):
        raise ValueError("bias has the wrong shape for these statistics")
    # Forming either trace expression as a matrix product would allocate an
    # output_width by output_width temporary. Contract the same scalar terms
    # directly, keeping the largest temporary in weight shape instead.
    weight_gram = weight @ stats.gxx
    quadratic = float(np.einsum(
        "oi,oi->", weight_gram, weight, optimize=False
    ))
    cross = -2.0 * float(np.einsum(
        "oi,io->", weight, stats.gxy, optimize=False
    ))
    target_norm = float(stats.sum_sq_y)
    bias_linear = 2.0 * float(
        bias @ (weight @ stats.sum_x - stats.sum_y)
    )
    bias_quadratic = stats.n * float(bias @ bias)
    terms = (quadratic, cross, target_norm, bias_linear, bias_quadratic)
    squared_error = sum(terms)
    cancellation_scale = sum(abs(term) for term in terms)
    zero_tolerance = (64.0 * np.finfo(np.float64).eps
                      * cancellation_scale)
    if stats.sum_sq_y == 0:
        if abs(squared_error) <= zero_tolerance:
            return 0.0
        return float("inf")
    # A fitted residual can be a few ulps below zero after cancellation.
    return max(squared_error, 0.0) / stats.sum_sq_y


def solve_bridge(stats, rank=64, rel_lambda=1e-4, affine=False,
                 rank_metric="ridge"):
    """Solve, truncate, and return arrays and scalar metadata for an .npz."""
    if rank_metric not in ("ridge", "coefficient"):
        raise ValueError("rank_metric must be 'ridge' or 'coefficient'")
    dense_weight, dense_bias, ridge_lambda = ridge_solve(
        stats, rel_lambda=rel_lambda, affine=affine
    )
    if rank_metric == "ridge":
        input_weight, output_weight, singular_values = ridge_metric_factors(
            stats, rank, ridge_lambda, affine
        )
    else:
        input_weight, output_weight, singular_values = svd_factors(
            dense_weight, rank
        )
    # Residual metadata describes the stored float32 factors, not an
    # unattainable float64 intermediate.
    input_weight = input_weight.astype(np.float32)
    output_weight = output_weight.astype(np.float32)
    rank_weight = (output_weight.astype(np.float64)
                   @ input_weight.astype(np.float64))
    if affine:
        bias = stats.sum_y / stats.n - rank_weight @ (stats.sum_x / stats.n)
    else:
        bias = np.zeros(stats.output_width, dtype=np.float64)
    bias = bias.astype(np.float32)
    return {
        "input_weight": input_weight,
        "output_weight": output_weight,
        "bias": bias,
        "singular_values": singular_values.astype(np.float64),
        "format_version": np.int64(2),
        "input_width": np.int64(stats.input_width),
        "output_width": np.int64(stats.output_width),
        "rank": np.int64(input_weight.shape[0]),
        "n": np.int64(stats.n),
        "affine": np.bool_(affine),
        "rank_metric": np.asarray(rank_metric),
        "rel_lambda": np.float64(rel_lambda),
        "ridge_lambda": np.float64(ridge_lambda),
        "dense_relative_residual": np.float64(
            relative_residual(stats, dense_weight, dense_bias)
        ),
        "rank_relative_residual": np.float64(
            relative_residual(stats, rank_weight, bias.astype(np.float64))
        ),
    }


def save_bridge(path, solution):
    """Write a bridge archive and return the actual .npz path."""
    if not path.endswith(".npz"):
        path += ".npz"
    np.savez(path, **solution)
    return path


def selftest():
    """Deterministic rectangular, streaming, affine, and archive checks."""
    rng = np.random.default_rng(20260809)
    rows, input_width, output_width, true_rank = 1536, 13, 9, 4
    source = rng.normal(size=(rows, input_width))
    left = rng.normal(size=(output_width, true_rank))
    right = rng.normal(size=(true_rank, input_width))
    true_weight = left @ right
    true_bias = rng.normal(size=output_width)
    target = source @ true_weight.T + true_bias
    target += 1e-5 * rng.normal(size=target.shape)

    direct = GramAccumulator(input_width, output_width)
    direct.add(source, target)
    streamed = GramAccumulator(input_width, output_width)
    for row0 in range(0, rows, 97):
        streamed.add(source[row0:row0 + 97], target[row0:row0 + 97])
    assert streamed.n == rows
    assert np.allclose(streamed.gxx, direct.gxx, rtol=1e-12, atol=1e-10)
    assert np.allclose(streamed.gxy, direct.gxy, rtol=1e-12, atol=1e-10)

    linear = solve_bridge(streamed, rank=true_rank, rel_lambda=1e-10,
                          affine=False)
    affine = solve_bridge(streamed, rank=true_rank, rel_lambda=1e-10,
                          affine=True)
    assert float(affine["rank_relative_residual"]) < 1e-10
    assert (float(affine["rank_relative_residual"])
            < float(linear["rank_relative_residual"]) * 1e-6)

    input_weight = affine["input_weight"].astype(np.float64)
    output_weight = affine["output_weight"].astype(np.float64)
    bias = affine["bias"].astype(np.float64)
    prediction = (source @ input_weight.T) @ output_weight.T + bias
    relative_error = np.sum((prediction - target) ** 2) / np.sum(target ** 2)
    assert relative_error < 1e-10

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "source.npy")
        target_path = os.path.join(temp_dir, "target.npy")
        output_path = os.path.join(temp_dir, "bridge.npz")
        np.save(source_path, source.astype(np.float32))
        np.save(target_path, target.astype(np.float32))
        from_files = accumulate_npy_pairs(
            [source_path], [target_path], chunk_rows=101
        )
        archived = solve_bridge(
            from_files, rank=true_rank, rel_lambda=1e-8, affine=True
        )
        save_bridge(output_path, archived)
        with np.load(output_path, allow_pickle=False) as loaded:
            assert int(loaded["format_version"]) == 2
            assert loaded["input_weight"].shape == (true_rank, input_width)
            assert loaded["output_weight"].shape == (output_width, true_rank)
            assert loaded["bias"].shape == (output_width,)
            assert int(loaded["n"]) == rows
            assert bool(loaded["affine"])
            assert str(loaded["rank_metric"]) == "ridge"

    print("activation_bridge selftest OK")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", action="append", default=[], metavar="SOURCE.npy",
        help="2D source array (repeat with a corresponding --target)",
    )
    parser.add_argument(
        "--target", action="append", default=[], metavar="TARGET.npy",
        help="2D target array (repeat with a corresponding --source)",
    )
    parser.add_argument(
        "--out", default=None, metavar="BRIDGE.npz",
        help="output archive containing Linear-oriented low-rank factors",
    )
    parser.add_argument(
        "--rank", type=int, default=64,
        help="maximum SVD rank (default: 64)",
    )
    parser.add_argument(
        "--rel-lambda", type=float, default=1e-4,
        help="ridge relative to the mean source-Gram diagonal (default: 1e-4)",
    )
    parser.add_argument(
        "--affine", action="store_true",
        help="fit an unregularized output bias",
    )
    parser.add_argument(
        "--rank-metric", choices=("ridge", "coefficient"), default="ridge",
        help=("rank reduction geometry: covariance-aware ridge objective "
              "(default) or Euclidean coefficient SVD"),
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=4096,
        help="rows loaded from each input array per block (default: 4096)",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if args.out is None:
        parser.error("--out is required")
    try:
        stats = accumulate_npy_pairs(
            args.source, args.target, chunk_rows=args.chunk_rows
        )
        solution = solve_bridge(
            stats, rank=args.rank, rel_lambda=args.rel_lambda,
            affine=args.affine, rank_metric=args.rank_metric,
        )
        output_path = save_bridge(args.out, solution)
    except ValueError as exc:
        parser.error(str(exc))

    print(f"paired rows          : {stats.n}")
    print(f"source -> target     : {stats.input_width} -> {stats.output_width}")
    print(f"effective rank       : {int(solution['rank'])}")
    print(f"rank metric          : {str(solution['rank_metric'])}")
    print(f"absolute ridge       : {float(solution['ridge_lambda']):.6g}")
    print(
        "relative residual   : "
        f"dense={float(solution['dense_relative_residual']):.6g}  "
        f"rank={float(solution['rank_relative_residual']):.6g}"
    )
    print(f"written              : {output_path}")


if __name__ == "__main__":
    main()
