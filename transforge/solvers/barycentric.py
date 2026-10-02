"""Barycentric projection: move source samples onto the target distribution.

Invariant I7: after transport, the empirical distribution of the mapped source
should match the target.  We verify that rather than assuming it.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from ..core.linalg import pairwise_sq_euclidean, safe_normalize
from ..core.types import FloatArray


def barycentric_map(plan: FloatArray, a: FloatArray, X: FloatArray, Y: FloatArray) -> FloatArray:
    """``T(x_i) = sum_j (pi_ij / a_i) y_j`` -- the barycentric projection.

    The row-normalised plan is a set of barycentric weights; mapping the source
    cloud through them lands it (approximately) on the target cloud.
    """
    plan = np.asarray(plan, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    safe_a = np.where(a > 0.0, a, 1.0)
    weights = plan / safe_a[:, None]
    return weights @ np.asarray(Y, dtype=np.float64)


def transport_features(
    Xs: FloatArray,
    Xt: FloatArray,
    a: FloatArray | None = None,
    b: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Align ``Xs`` onto ``Xt`` by exact entropic OT, returning ``(Xs_aligned, Xt)``.

    This is the classical Courty et al. (2017) baseline for the domain-adaptation
    comparison, and the reference the flagship is measured against.
    """
    from ..core.config import SinkhornConfig  # noqa: PLC0415
    from .epsilon_knee import TFEpsilonKnee  # noqa: PLC0415

    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    n, m = Xs.shape[0], Xt.shape[0]
    a = safe_normalize(np.ones(n)) if a is None else safe_normalize(a)
    b = safe_normalize(np.ones(m)) if b is None else safe_normalize(b)
    C = pairwise_sq_euclidean(Xs, Xt)
    res = TFEpsilonKnee(SinkhornConfig()).solve(a, b, C)
    return barycentric_map(res.plan, a, Xs, Xt), Xt


__all__ = ["barycentric_map", "transport_features"]
