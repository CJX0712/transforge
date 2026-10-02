"""Stable numerics shared by every layer: log-sum-exp and safe normalisation.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def logsumexp(x: FloatArray, axis: int = -1, keepdims: bool = False) -> FloatArray:
    """Numerically stable ``log(sum(exp(x)))``.

    The shift ``m`` is taken **per slice along ``axis``**, never globally: a
    global shift collapses a slice whose entries are all around -1e6 into a
    constant.  ``x - m`` has maximum exactly 0, so ``sum(exp(x - m)) <= n`` and
    the logarithm cannot overflow.

    This is the ONLY correct spelling; never compute ``log(sum(exp(x)))``.
    """
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))
    if keepdims:
        out = np.expand_dims(out, axis=axis)
    return out


def safe_normalize(w: FloatArray, floor: float = 0.0) -> FloatArray:
    """Normalise non-negative weights to sum 1, dropping entries below ``floor``.

    Raises
    ------
    ValueError
        If the weight vector contains a negative entry or sums to zero.
    """
    w = np.asarray(w, dtype=np.float64)
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    if floor > 0.0:
        w = np.where(w < floor, 0.0, w)
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights sum to zero")
    return w / total


def pairwise_sq_euclidean(X: FloatArray, Y: FloatArray) -> FloatArray:
    """Squared euclidean cost ``C_ij = ||x_i - y_j||^2`` without a 3-D temporary.

    Uses the ``(a-b)^2 = a^2 - 2ab + b^2`` expansion, clipped at 0 to remove
    cancellation noise that would otherwise yield tiny negative entries.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    x2 = np.sum(X * X, axis=1)[:, None]
    y2 = np.sum(Y * Y, axis=1)[None, :]
    C = x2 - 2.0 * (X @ Y.T) + y2
    np.maximum(C, 0.0, out=C)
    return C


def relative_residual(actual: FloatArray, target: FloatArray) -> float:
    """Scale-free L-inf residual, normalised by ``max(‖target‖inf, tiny)``."""
    denom = max(float(np.max(np.abs(target))), 1e-300)
    return float(np.max(np.abs(actual - target))) / denom


def plan_entropy_proxy(plan: FloatArray, a: FloatArray, b: FloatArray) -> float:
    """Normalised plan sharpness proxy in ``[0, 1]``: 0 = fully spread, 1 = collapsed.

    Defined as the normalised KL divergence of the plan from the independent
    coupling::

        proxy(pi) = KL(pi | a (x) b) / log(n*m)
                  = sum_ij pi_ij log(pi_ij / (a_i b_j)) / log(n*m)

    ``pi = a (x) b`` (no structure at all, i.e. epsilon far too large) gives 0.
    A permutation-like (block-diagonal) plan puts all its mass on n cells with
    ratio ``n`` each, so the KL is exactly ``log(n)`` and the proxy is
    ``log(n)/log(n*m) = 0.5`` for a square problem.  The observed range on this
    family is therefore [0, ~0.5]; the proxy is used only as an ORDERING signal,
    never as an absolute one.

    NOTE ON A CORRECTED FORMULA.  An earlier draft used
    ``-sum(rho log rho) / log(n*m)`` with ``rho = pi/(a (x) b)``, on the reasoning
    that it equals 1 for the independent plan.  It does not: for ``pi = a (x) b``
    we get ``rho = 1`` and therefore ``-sum(rho log rho) = 0``, while a sharp plan
    drives that quantity strongly *negative*.  The expression is an unnormalised
    cross-entropy, not an entropy, and it runs backwards.  The KL form above is
    the corrected version; ``tests/test_invariants.py`` pins both endpoints so the
    orientation cannot silently flip again.

    Used as the third stopping criterion to catch *false convergence*: a small
    marginal residual alone does not certify that the plan is meaningful.
    """
    ab = np.outer(a, b)
    mask = plan > 0.0
    if not np.any(mask):
        return 0.0
    ratio = plan[mask] / np.maximum(ab[mask], 1e-300)
    kl = float(np.sum(plan[mask] * np.log(np.maximum(ratio, 1e-300))))
    return float(min(max(kl / np.log(max(plan.size, 2)), 0.0), 1.0))


def effective_support(plan: FloatArray, a: FloatArray, b: FloatArray) -> float:
    """Participation ratio ``N_eff = exp(H(w))`` in ``[1, n*m]`` over plan mass ``w``.

    ``w = plan / sum(plan)``.  For a permutation-like (block-diagonal) plan with
    ``n`` equally weighted cells this equals ``n`` (it recovers the support
    size); for the fully spread independent plan it equals ``n*m``.

    CORRECTION.  The formula originally shipped here was
    ``exp(-sum rho log rho)`` with ``rho = plan/(a (x) b)``.  ``rho`` sums to
    ``n*m``, not 1, so that expression is **not** an entropy: for a permutation
    plan it returns ``exp(-n^2 log n) ~ 1e-28`` instead of ``n``, i.e. it falls
    off the bottom of its own stated range ``[1, n*m]``.  Normalising to plan mass
    first is what makes it a genuine participation ratio.  Pinned by
    ``tests/test_core.py::test_effective_support_range``.
    """
    total = float(np.sum(plan))
    if total <= 0.0:
        return 1.0
    w = plan / total
    mask = w > 0.0
    if not np.any(mask):
        return 1.0
    return float(np.exp(-np.sum(w[mask] * np.log(w[mask]))))


def plan_change_rate(prev: FloatArray, cur: FloatArray) -> float:
    """Relative Frobenius change ``||pi_t - pi_{t-1}||_F / ||pi_t||_F``."""
    denom = max(float(np.linalg.norm(cur)), 1e-300)
    return float(np.linalg.norm(cur - prev) / denom)


def dual_objective(
    a: FloatArray, b: FloatArray, f: FloatArray, g: FloatArray, C: FloatArray, eps: float
) -> float:
    """Dual objective ``Phi(f,g) = <a,f> + <b,g> - eps * sum(exp((f+g-C)/eps))``.

    Block coordinate ascent on a jointly concave objective, therefore
    monotonically non-decreasing along Sinkhorn iterations (invariant I1).
    Evaluated via log-sum-exp so it stays finite for small ``eps``.
    """
    n, m = C.shape
    total = float(np.sum(np.exp(logsumexp((f[:, None] + g[None, :] - C) / eps, axis=None))))
    return float(np.dot(a, f) + np.dot(b, g) - eps * total)


def ot_lower_bound(C: FloatArray, a: FloatArray, b: FloatArray) -> float:
    """Free certified lower bound ``max{sum_i a_i min_j C_ij, sum_j b_j min_i C_ij}``.

    Provenance: ``f_i = min_j C_ij, g = 0`` is a feasible Kantorovich dual pair,
    hence any correct OT value must be >= this.  Costs O(nm) on a matrix we
    already hold.  This is the cheapest available guard against metric fantasy:
    an OT value below this bound is a bug, not an improvement.
    """
    row = float(np.dot(a, np.min(C, axis=1)))
    col = float(np.dot(b, np.min(C, axis=0)))
    return max(row, col)


__all__ = [
    "dual_objective",
    "effective_support",
    "logsumexp",
    "ot_lower_bound",
    "pairwise_sq_euclidean",
    "plan_change_rate",
    "plan_entropy_proxy",
    "relative_residual",
    "safe_normalize",
]
