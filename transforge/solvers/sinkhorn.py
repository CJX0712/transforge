"""Log-domain and naive-domain Sinkhorn.

Two independent implementations of the same mathematical object.  They exist so
that invariant I6 can cross-validate them against each other, and so that the
project keeps working (degraded, never broken) when POT is unavailable.

THE TRAP THAT THIS MODULE IS BUILT AROUND
-----------------------------------------
In the log-domain iteration

    f <- eps*log(a) - eps*LSE_j((g - C)/eps)
    g <- eps*log(b) - eps*LSE_i((f - C)/eps)

the row marginal is *exact by construction* right after the f-update, and the
column marginal is *exact by construction* right after the g-update.  Measuring
the residual on the side that was just written therefore returns 0 no matter how
wrong the iterate is, and a naive implementation exits after ONE iteration having
produced garbage.  The informative measurement is taken against the **stale**
side: after updating f, evaluate the column residual using the previous g.

Concretely (see ``_column_residual``)::

    r_j = g_j + eps*LSE_i((f_i - C_ij)/eps) - eps*log(b_j)
    res = max_j |b_j * (exp(r_j/eps) - 1)| / max(b)

The ``expm1`` form is essential: ``exp(x) - 1`` loses all precision for small
``x``, and small ``x`` is exactly the converged regime.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import NumericError
from ..core.linalg import (
    logsumexp,
    plan_change_rate,
    plan_entropy_proxy,
    relative_residual,
)
from ..core.types import FloatArray, OTResult, TimingRecord


@dataclass
class StopTrace:
    """Per-iteration diagnostics, kept for invariant I1 and for gate reporting."""

    dual_objective: list[float] = field(default_factory=list)
    residual: list[float] = field(default_factory=list)
    entropy_proxy: list[float] = field(default_factory=list)
    change_rate: list[float] = field(default_factory=list)

    def dual_monotone_violation(self, tol: float) -> float:
        """Largest drop in the dual objective along the trajectory (0.0 if monotone).

        ``Phi`` is the dual of a convex problem and the iteration is exact block
        coordinate ascent on it, so it must be non-decreasing.
        """
        if len(self.dual_objective) < 2:
            return 0.0
        arr = np.asarray(self.dual_objective, dtype=np.float64)
        # A violation is a DECREASE of Phi, i.e. a negative diff.  Using
        # np.max(diff) here would measure the increase and report the (large,
        # legitimate) monotone rise as a violation.
        return float(max(0.0, -np.min(np.diff(arr)) + tol))


def _column_residual(
    a: FloatArray, b: FloatArray, C: FloatArray, f: FloatArray, g: FloatArray, eps: float
) -> float:
    """Relative column-marginal residual measured against the STALE g.

    Must be called immediately after the f-update and before the g-update.
    """
    lb = np.log(b)
    r = g + eps * logsumexp((f[:, None] - C) / eps, axis=0) - eps * lb
    return float(np.max(b * np.abs(np.expm1(r / eps))) / max(float(np.max(b)), 1e-300))


def _row_residual(
    a: FloatArray, b: FloatArray, C: FloatArray, f: FloatArray, g: FloatArray, eps: float
) -> float:
    """Relative row-marginal residual measured against the STALE f."""
    la = np.log(a)
    r = f + eps * logsumexp((g[None, :] - C) / eps, axis=1) - eps * la
    return float(np.max(a * np.abs(np.expm1(r / eps))) / max(float(np.max(a)), 1e-300))


def sinkhorn_log(
    a: FloatArray,
    b: FloatArray,
    C: FloatArray,
    eps: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 3000,
    f0: FloatArray | None = None,
    g0: FloatArray | None = None,
    trace: bool = False,
    collapse_margin: float = 1.0,
    change_tol: float = 0.0,
    check_every: int = 25,
) -> OTResult:
    """Log-domain Sinkhorn (numerically safe for small ``eps``).

    Implements the three-criterion stopping rule mandated by the lead:

    1. both marginal residuals below ``tol`` (necessary condition);
    2. relative plan change below ``change_tol`` between checkpoints;
    3. sharpness proxy below ``collapse_margin`` (reject block-diagonal plans).

    Criterion 1 alone is NOT sufficient.  A small epsilon drives the plan towards
    a block-diagonal / permutation-like coupling whose marginals can be perfectly
    satisfied while the coupling itself is a wrong over-sharpened matching.  That
    is false convergence, and criteria 2 and 3 exist to catch it.

    Returns an :class:`OTResult` with ``cost = <C, pi>`` (transport term only, so
    it stays comparable with exact EMD) and ``meta['kl']`` holding the entropy
    term separately.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if eps <= 0:
        raise NumericError(f"eps must be positive, got {eps}")
    if np.any(a <= 0) or np.any(b <= 0):
        raise NumericError("log-domain Sinkhorn requires strictly positive weights")

    n, m = C.shape
    la, lb = np.log(a), np.log(b)
    f = np.zeros(n, dtype=np.float64) if f0 is None else np.array(f0, dtype=np.float64)
    g = np.zeros(m, dtype=np.float64) if g0 is None else np.array(g0, dtype=np.float64)
    tr = StopTrace() if trace else None
    prev_plan: FloatArray | None = None
    it = 0
    res = float("inf")
    ent = 1.0
    change = 0.0
    change_measurable = False
    converged = False

    for it in range(1, max_iter + 1):
        f = eps * la - eps * logsumexp((g[None, :] - C) / eps, axis=1)
        if not np.all(np.isfinite(f)):
            raise NumericError(f"non-finite dual potential f at iter {it}, eps={eps:g}")
        # informative residual: column side, against the stale g
        res_col = _column_residual(a, b, C, f, g, eps)
        g = eps * lb - eps * logsumexp((f[:, None] - C) / eps, axis=0)
        if not np.all(np.isfinite(g)):
            raise NumericError(f"non-finite dual potential g at iter {it}, eps={eps:g}")
        res = max(res_col, _row_residual(a, b, C, f, g, eps))

        # Checkpoint cadence.  A rung that converges in fewer than ``check_every``
        # iterations produces only ONE checkpoint, so no change rate can be formed;
        # we must not let an unmeasurable criterion silently veto convergence.
        # The first checkpoint therefore always counts as "still moving" (rate 1.0)
        # only when a later checkpoint will actually follow -- otherwise the rung is
        # accepted on residual + entropy alone and ``change_rate`` is reported as
        # ``nan`` so callers can see the criterion was not exercised.
        due = it % max(check_every, 1) == 0 or it == 1 or it == max_iter
        if due:
            plan = np.exp((f[:, None] + g[None, :] - C) / eps)
            ent = plan_entropy_proxy(plan, a, b)
            if prev_plan is None:
                change = 0.0 if res < tol else 1.0
                change_measurable = res < tol
            else:
                change = plan_change_rate(prev_plan, plan)
                change_measurable = True
            prev_plan = plan
            if tr is not None:
                tr.residual.append(res)
                tr.entropy_proxy.append(ent)
                tr.change_rate.append(change)
                tr.dual_objective.append(_dual_value(a, b, f, g, C, eps))

        marginal_ok = res < tol
        entropy_ok = ent < collapse_margin
        # criterion 2 is enforced only when it is measurable (>= 2 checkpoints)
        change_ok = (change < change_tol) or not change_measurable
        if marginal_ok and entropy_ok and change_ok:
            converged = True
            break

    plan = np.exp((f[:, None] + g[None, :] - C) / eps)
    np.maximum(plan, 0.0, out=plan)
    kl = float(np.sum(plan * np.log(np.maximum(plan, 1e-300) / np.outer(a, b))))
    meta: dict[str, Any] = {
        "kl": kl,
        "solver": "sinkhorn_log",
        "dual_objective": _dual_value(a, b, f, g, C, eps),
        "entropy_proxy": ent,
        "change_rate": change,
        "change_rate_measurable": change_measurable,
        "f": f,
        "g": g,
    }
    if tr is not None:
        meta["trace"] = tr
    return OTResult(
        plan=plan,
        cost=float(np.sum(C * plan)),
        marginal_a=plan.sum(axis=1),
        marginal_b=plan.sum(axis=0),
        n_iter=it,
        converged=converged,
        epsilon=float(eps),
        residual=float(res),
        meta=meta,
    )


def _dual_value(
    a: FloatArray, b: FloatArray, f: FloatArray, g: FloatArray, C: FloatArray, eps: float
) -> float:
    """Dual objective ``Phi``, evaluated stably.  Must be non-decreasing (I1)."""
    total = float(np.sum(np.exp(logsumexp((f[:, None] + g[None, :] - C) / eps, axis=None))))
    return float(np.dot(a, f) + np.dot(b, g) - eps * total)


def sinkhorn_naive(
    a: FloatArray,
    b: FloatArray,
    C: FloatArray,
    eps: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 3000,
) -> OTResult:
    """Naive-domain Sinkhorn -- the *second* implementation for invariant I6.

    Uses an explicit Gibbs kernel ``K = exp(-C/eps)`` and ordinary scaling
    vectors.  It underflows catastrophically once ``(Cmax - Cmin)/eps`` exceeds
    roughly 745 (the float64 exponent range), which is exactly why the log-domain
    version exists.  Kept deliberately, not as dead code: invariant I6 needs an
    independent path, and ``tests/test_invariants.py`` asserts that the naive
    path *must fail* on the sparse-weight family.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        logK = -C / eps
        K = np.exp(logK)
    if not np.all(np.isfinite(K)) or float(np.max(K)) == 0.0:
        raise NumericError(
            f"naive kernel underflowed/overflowed at eps={eps:g}: "
            f"(Cmax-Cmin)/eps={float(np.ptp(C) / eps):.3g} exceeds the float64 range"
        )
    n, m = C.shape
    u = np.ones(n, dtype=np.float64) / n
    v = np.ones(m, dtype=np.float64) / m
    res = float("inf")
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        u = a / np.maximum(K @ v, 1e-300)
        v = b / np.maximum(K.T @ u, 1e-300)
        if not np.all(np.isfinite(u)) or not np.all(np.isfinite(v)):
            raise NumericError(f"naive Sinkhorn diverged at iter {it}, eps={eps:g}")
        plan = u[:, None] * K * v[None, :]
        res = relative_residual(plan.sum(axis=1), a)
        res = max(res, relative_residual(plan.sum(axis=0), b))
        if res < tol:
            converged = True
            break
    plan = u[:, None] * K * v[None, :]
    return OTResult(
        plan=plan,
        cost=float(np.sum(C * plan)),
        marginal_a=plan.sum(axis=1),
        marginal_b=plan.sum(axis=0),
        n_iter=it,
        converged=converged,
        epsilon=float(eps),
        residual=float(res),
        meta={"solver": "sinkhorn_naive"},
    )


def sinkhorn_divergence(
    a: FloatArray,
    b: FloatArray,
    C_ab: FloatArray,
    eps: float,
    *,
    C_aa: FloatArray | None = None,
    C_bb: FloatArray | None = None,
    tol: float = 1e-9,
    max_iter: int = 3000,
) -> OTResult:
    """Debiased Sinkhorn divergence ``S_eps = OT(a,b) - (OT(a,a) + OT(b,b)) / 2``.

    Removing the ``eps * H(a (x) b)`` drift is what makes values comparable
    *across* epsilon, which is the precondition for any automatic selection rule.
    Note the debiased quantity is a divergence, not a transport cost: it can be
    slightly negative under finite precision, and Pooladian et al. (2022) show it
    can be statistically *harmful* when epsilon is large or samples are few.
    """
    from ..core.linalg import pairwise_sq_euclidean

    X_a = _infer_points(a, C_ab)
    X_b = _infer_points(b, C_ab)
    if C_aa is None:
        C_aa = pairwise_sq_euclidean(X_a, X_a)
    if C_bb is None:
        C_bb = pairwise_sq_euclidean(X_b, X_b)
    ab = sinkhorn_log(a, b, C_ab, eps, tol=tol, max_iter=max_iter)
    aa = sinkhorn_log(a, a, C_aa, eps, tol=tol, max_iter=max_iter)
    bb = sinkhorn_log(b, b, C_bb, eps, tol=tol, max_iter=max_iter)
    div = ab.cost - 0.5 * aa.cost - 0.5 * bb.cost
    return OTResult(
        plan=ab.plan,
        cost=float(div),
        marginal_a=ab.marginal_a,
        marginal_b=ab.marginal_b,
        n_iter=ab.n_iter + aa.n_iter + bb.n_iter,
        converged=ab.converged and aa.converged and bb.converged,
        epsilon=float(eps),
        residual=float(max(ab.residual, aa.residual, bb.residual)),
        meta={
            "solver": "sinkhorn_divergence",
            "transport_cost": ab.cost,
            "self_cost_a": aa.cost,
            "self_cost_b": bb.cost,
        },
    )


def _infer_points(weights: FloatArray, C: FloatArray) -> FloatArray:
    """Recover point coordinates from a squared-euclidean cost matrix.

    Uses the double-centring identity ``<x_i, x_j> = (G_ii + G_jj)/2 - G_ij/2``
    where ``G_ij = C_ii + C_jj - 2 C_ij``.  Only needed by the divergence helper,
    which must rebuild self-costs; callers that already have coordinates should
    pass ``C_aa``/``C_bb`` explicitly.
    """
    diag = np.diag(C)
    n = C.shape[0]
    gram = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            gram[i, j] = 0.5 * (diag[i] + diag[j] - C[i, j])
    vals, vecs = np.linalg.eigh(gram)
    vals = np.clip(vals, 0.0, None)
    return vecs * np.sqrt(vals)


def timed(result: OTResult, started: float, threads: int = 1) -> TimingRecord:
    """Attach a timing record to a result produced after ``started``."""
    return TimingRecord(
        wall_s=time.perf_counter() - started,
        n_iter=result.n_iter,
        warmup_excluded=True,
        threads=threads,
    )


__all__ = [
    "StopTrace",
    "sinkhorn_divergence",
    "sinkhorn_log",
    "sinkhorn_naive",
    "timed",
]
