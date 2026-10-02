"""Metrics.  Depends on ``core`` only -- never on ``solvers``.

Keeping the metric layer independent of the solver layer is deliberate: the
referee must not play for either team.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import ot_lower_bound
from ..core.types import EmpiricalMeasure, FloatArray, OTResult


def relative_error(value: float, truth: float) -> float:
    """``|value - truth| / |truth|``.  Scale free; undefined truth raises."""
    if abs(truth) < 1e-300:
        raise ValueError("relative error is undefined for a zero reference")
    return abs(float(value) - float(truth)) / abs(float(truth))


def cost_relative_error(result: OTResult, truth: float) -> float:
    """Relative error of the transport cost against an exact reference."""
    return relative_error(result.cost, truth)


def marginal_residual(result: OTResult, a: FloatArray, b: FloatArray) -> float:
    """Max relative L-inf marginal error."""
    return result.rel_marginal_residual(a, b)


def optimality_gap(result: OTResult, a: FloatArray, b: FloatArray, C: FloatArray) -> float:
    """``<C, pi> - LB(C, a, b)``.

    Must be non-negative for any correct balanced plan: ``LB`` comes from a
    feasible Kantorovich dual pair, so a negative value means a bug.  This is the
    cheapest available guard against reporting an "improvement" that is really a
    violated bound.
    """
    return float(result.cost - ot_lower_bound(C, a, b))


def plan_consistency(result: OTResult) -> float:
    """Total mass transported.  ``<= 1`` under a KL marginal relaxation."""
    return result.mass


def transport_agreement(plan_a: FloatArray, plan_b: FloatArray) -> float:
    """L1 distance between two plans -- how much two solvers disagree."""
    return float(np.sum(np.abs(np.asarray(plan_a) - np.asarray(plan_b))))


def wasserstein_1d(x: FloatArray, y: FloatArray) -> float:
    """Exact 1-D Wasserstein-1 between two point clouds (quantile form)."""
    from ..data.closed_form import wasserstein1_1d  # noqa: PLC0415

    return wasserstein1_1d(x, y)


def measure_gap(m: EmpiricalMeasure, other: EmpiricalMeasure) -> float:
    """Squared-euclidean cost between two measures at their sample points."""
    from ..core.linalg import pairwise_sq_euclidean  # noqa: PLC0415

    n = min(m.n, other.n)
    return float(np.sum(pairwise_sq_euclidean(m.X[:n], other.X[:n])))


def bias_upper_bound(eps: float, a: FloatArray, b: FloatArray) -> float:
    """The rigorous regularisation-bias bound ``eps * log(1/(min a * min b))``."""
    return float(eps * np.log(1.0 / max(float(np.min(a)) * float(np.min(b)), 1e-300)))


def bias_ratio(eps: float, a: FloatArray, b: FloatArray, reference: float) -> float:
    """Bias bound as a fraction of a reference transport cost.

    This is the quantity the certificate gate compares against 1.
    """
    return bias_upper_bound(eps, a, b) / max(abs(float(reference)), 1e-300)


__all__ = [
    "bias_ratio",
    "bias_upper_bound",
    "cost_relative_error",
    "marginal_residual",
    "measure_gap",
    "optimality_gap",
    "plan_consistency",
    "relative_error",
    "transport_agreement",
    "wasserstein_1d",
]
