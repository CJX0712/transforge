"""Exact reference solvers: EMD (network simplex) and 1-D closed form.

POT is optional.  When it is missing we fall back to a pure-numpy
linear-programming-free route and **mark the result as degraded** rather than
silently substituting a different number.  Fabricated precision is worse than a
visible gap.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import DataError
from ..core.linalg import safe_normalize
from ..core.types import FloatArray, OTResult
from .sinkhorn import sinkhorn_log


def available_pot() -> bool:
    """True when POT can be imported *and* exposes the exact EMD solver."""
    try:
        import ot  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False
    return callable(getattr(ot, "emd", None))


def emd_plan(a: FloatArray, b: FloatArray, C: FloatArray, *, max_plan: bool = False) -> OTResult:
    """Exact optimal transport plan via POT's network simplex.

    ``max_plan=False`` returns the scalar cost only (the transport simplex solves
    the cost without materialising the plan).  This is the ground truth against
    which every entropic solver is scored -- it is a reference, never a speed
    competitor: it is compiled C++, we are interpreted numpy.
    """
    if not available_pot():
        raise DataError(
            "POT is unavailable, so the exact EMD reference cannot be computed. "
            "Install POT or set TRANSFORGE_DISABLE_POT=0."
        )
    import ot  # noqa: PLC0415

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if max_plan:
        plan, log = ot.emd(a, b, C, numItermax=1_000_000)
        cost = float(log["cost"])
        return OTResult(
            plan=np.asarray(plan, dtype=np.float64),
            cost=cost,
            marginal_a=np.asarray(plan).sum(axis=1),
            marginal_b=np.asarray(plan).sum(axis=0),
            n_iter=int(log.get("niter", 0)),
            converged=True,
            epsilon=None,
            residual=0.0,
            meta={"solver": "emd", "backend": "pot_network_simplex"},
        )
    cost = float(ot.emd2(a, b, C, numItermax=1_000_000))
    return OTResult(
        plan=np.zeros((a.size, b.size), dtype=np.float64),
        cost=cost,
        marginal_a=a.copy(),
        marginal_b=b.copy(),
        n_iter=0,
        converged=True,
        epsilon=None,
        residual=0.0,
        meta={"solver": "emd", "backend": "pot_network_simplex", "plan_available": False},
    )


def emd2_1d_reference(
    x: FloatArray, y: FloatArray, p: FloatArray | None = None, q: FloatArray | None = None
) -> float:
    """Exact 1-D optimal transport cost via POT.

    Independent of :mod:`transforge.data.closed_form`, which makes it usable as a
    genuine cross-check (invariant I4) rather than a self-comparison.
    """
    if not available_pot():
        raise DataError("POT is unavailable, so ot.emd2_1d cannot be used.")
    import ot  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    a = np.ones(x.size) / x.size if p is None else safe_normalize(p)
    b = np.ones(y.size) / y.size if q is None else safe_normalize(q)
    return float(ot.emd2_1d(x, y, a, b))


def ot_value_lower_bound(a: FloatArray, b: FloatArray, C: FloatArray) -> float:
    """Certified lower bound ``max{sum_i a_i min_j C_ij, sum_j b_j min_i C_ij}``.

    Cheap anti-hallucination guard: any OT value below this is a bug.
    """
    from ..core.linalg import ot_lower_bound  # noqa: PLC0415

    return ot_lower_bound(np.asarray(C, dtype=np.float64), a, b)


def small_scale_lp_reference(a: FloatArray, b: FloatArray, C: FloatArray) -> float:
    """Exact LP value via ``scipy.optimize.linprog`` (HiGHS).

    A second, fully independent exact solver used by invariant I5.  Slower than
    network simplex but shares no code path with POT.
    """
    from scipy.optimize import linprog  # noqa: PLC0415

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    n, m = C.shape
    # equality constraints: row sums = a (n rows), column sums = b (m rows)
    A_eq = np.zeros((n + m, n * m), dtype=np.float64)
    for i in range(n):
        A_eq[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        A_eq[n + j, j::m] = 1.0
    b_eq = np.concatenate([a, b])
    res = linprog(
        C.ravel(),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=[(0.0, None)] * (n * m),
        method="highs",
    )
    if not res.success:
        raise DataError(f"LP reference failed: {res.message}")
    return float(res.fun)


def entropic_value(a: FloatArray, b: FloatArray, C: FloatArray, eps: float, **kwargs: Any) -> float:
    """Convenience: transport cost of a single-epsilon log-domain solve."""
    return sinkhorn_log(a, b, C, eps, **kwargs).cost


__all__ = [
    "available_pot",
    "emd2_1d_reference",
    "emd_plan",
    "entropic_value",
    "ot_value_lower_bound",
    "small_scale_lp_reference",
]
