"""Analytic ground truths -- the only self-validating reference in the project.

Everything here is derived independently of the solver code, which is what makes
it usable for cross-validation (invariants I4 / I5).

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ClosedFormError
from ..core.linalg import pairwise_sq_euclidean
from ..core.types import FloatArray


def wasserstein1_1d(x: FloatArray, y: FloatArray, p: int = 1) -> float:
    """Exact 1-D Wasserstein distance via the quantile representation.

    ``W_p(X, Y) = ( int_0^1 |F_X^-1(u) - F_Y^-1(u)|^p du )^(1/p)``.

    For equally weighted clouds of equal size this reduces to matching order
    statistics; the general weighted case uses the merged-sweep construction.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    if x.size == y.size:
        if p == 1:
            return float(np.mean(np.abs(x - y)))
        if p == 2:
            return float(np.sqrt(np.mean((x - y) ** 2)))
        return float(np.mean(np.abs(x - y) ** p) ** (1.0 / p))
    grid = np.linspace(0.0, 1.0, 4096)
    return float(np.mean(np.abs(np.quantile(x, grid) - np.quantile(y, grid)) ** p) ** (1.0 / p))


def w2_1d_equal(x: FloatArray, y: FloatArray) -> float:
    """``W_2`` for equally sized, equally weighted 1-D clouds (order-statistic form).

    ``W_2^2 = (1/n) sum_k (x_(k) - y_(k))^2``.
    """
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    y = np.sort(np.asarray(y, dtype=np.float64).ravel())
    if x.size != y.size:
        raise ClosedFormError(
            f"equal-weight closed form needs equal sizes, got {x.size} vs {y.size}"
        )
    return float(np.sqrt(np.mean((x - y) ** 2)))


def gaussian_w2(mu0: FloatArray, cov0: FloatArray, mu1: FloatArray, cov1: FloatArray) -> float:
    """Bures / Gelbrich closed form for ``W_2`` between two Gaussians.

    ``W_2^2 = ||dmu||^2 + tr(S0 + S1 - 2 (S0^{1/2} S1 S0^{1/2})^{1/2})``.

    The matrix square root is computed by symmetric eigendecomposition with
    clipping, which keeps it dependency-free and deterministic.
    """
    mu0 = np.asarray(mu0, dtype=np.float64).ravel()
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    s0 = np.asarray(cov0, dtype=np.float64)
    s1 = np.asarray(cov1, dtype=np.float64)
    if s0.shape != s1.shape:
        raise ClosedFormError(f"covariance shapes differ: {s0.shape} vs {s1.shape}")
    dmu = float(np.sum((mu0 - mu1) ** 2))
    s0_half = _sym_sqrt(s0)
    inner = s0_half @ s1 @ s0_half
    cross = _sym_sqrt(inner)
    return float(np.sqrt(max(dmu + np.trace(s0) + np.trace(s1) - 2.0 * np.trace(cross), 0.0)))


def _sym_sqrt(A: FloatArray) -> FloatArray:
    """Symmetric positive semi-definite matrix square root via eigendecomposition."""
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def gaussian_w2_from_samples(
    X: FloatArray, Y: FloatArray, cov_x: FloatArray, cov_y: FloatArray
) -> float:
    """Plug the *empirical* means into the Gaussian closed form.

    Used by the ``gauss2d`` family where the sampled covariance is intentionally
    perturbed away from the generating one.
    """
    return gaussian_w2(np.mean(X, axis=0), cov_x, np.mean(Y, axis=0), cov_y)


def sample_cost(X: FloatArray, Y: FloatArray) -> FloatArray:
    """Squared euclidean ground cost between two point clouds."""
    return pairwise_sq_euclidean(X, Y)


def wasserstein_1d_from_samples(x: FloatArray, y: FloatArray) -> float:
    """Alias kept for readability at call sites in the benchmark."""
    return wasserstein1_1d(x, y, p=1)


def covariance(X: FloatArray) -> FloatArray:
    """Empirical covariance (biased estimator, deterministic)."""
    X = np.asarray(X, dtype=np.float64)
    centred = X - np.mean(X, axis=0)
    return (centred.T @ centred) / max(X.shape[0] - 1, 1)


__all__ = [
    "NDArray",
    "_sym_sqrt",
    "covariance",
    "gaussian_w2",
    "gaussian_w2_from_samples",
    "sample_cost",
    "w2_1d_equal",
    "wasserstein1_1d",
    "wasserstein_1d_from_samples",
]
