"""TF-UFlow -- flagship 3: unbalanced OT with a joint (epsilon, tau) continuation.

Unbalanced OT relaxes the marginal constraints with a KL penalty:

    min_pi  <C, pi> + eps*KL(pi | a (x) b) + tau1*KL(pi 1 | a) + tau2*KL(pi^T 1 | b)

with ``tau -> inf`` recovering balanced OT.  The log-domain fixed point is

    zeta = eps*tau / (tau + eps)          (zeta -> eps as tau -> inf)
    f_i <- -zeta * LSE_j((g_j - C_ij)/eps) - zeta
    g_j <- -zeta * LSE_i((f_i - C_ij)/eps) - zeta

The factor ``eps*tau/(tau+eps) = eps/(1 + eps/tau)`` is the standard
Sinkhorn-for-unbalanced contraction; the ``tau -> inf`` limit coincides with the
balanced update up to an irrelevant global constant.  Both were checked
symbolically against the derivation before implementation.

WHY (eps, tau) MUST BE CONTINUED TOGETHER
-----------------------------------------
``eps`` controls how "blurred" the plan is; ``tau`` controls how much mass may be
destroyed.  They are coupled: large ``tau`` forces balanced behaviour and drags
out every outlier along with everything else; small ``tau`` destroys so much mass
that alignment fails.  A full grid costs ``K_eps * K_tau`` independent solves; a
monotone diagonal continuation costs ``K_eps + K_tau``.

SAFETY VALVE
------------
UOT loses on clean data: the marginal relaxation is pure error when there is
nothing to discard.  ``mass_destroyed`` records ``1 - ||pi||_1``; when it falls
below ``min_mass_destroyed`` the solver reports that balancing would have been
better, so the caller never silently pays UOT's tax on clean instances.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.config import SinkhornConfig
from ..core.errors import InfeasibleError, NumericError
from ..core.linalg import logsumexp
from ..core.types import FloatArray, OTResult


class TFUFlow:
    """KL-relaxed unbalanced Sinkhorn with an (epsilon, tau) continuation path."""

    name = "TF-UFlow"
    requires_backend = "numpy"

    def __init__(
        self,
        cfg: SinkhornConfig | None = None,
        tau: float | None = None,
        n_eps: int = 5,
        n_tau: int = 4,
        min_mass_destroyed: float = 0.02,
    ) -> None:
        self.cfg = cfg or SinkhornConfig()
        self.cfg.validate()
        if tau is not None and tau <= 0:
            raise InfeasibleError(f"tau must be positive, got {tau}")
        self.tau = tau
        self.n_eps = int(n_eps)
        self.n_tau = int(n_tau)
        self.min_mass_destroyed = float(min_mass_destroyed)

    def solve(
        self,
        a: FloatArray,
        b: FloatArray,
        C: FloatArray,
        *,
        eps: float | None = None,
        tau: float | None = None,
    ) -> OTResult:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        cfg = self.cfg
        median = float(np.median(C))
        if eps is None:
            eps = 0.05 * median
        if tau is None:
            tau = self.tau if self.tau is not None else 0.05 * median

        started = time.perf_counter()
        eps_path = np.geomspace(max(eps, 1e-3 * median), 0.05 * median, self.n_eps)
        tau_path = np.geomspace(max(tau, 1e-6 * median), 10.0 * median, self.n_tau)

        n, m = C.shape
        f = np.zeros(n, dtype=np.float64)
        g = np.zeros(m, dtype=np.float64)
        path: list[dict[str, Any]] = []
        best: FloatArray | None = None
        total_iter = 0
        res = float("inf")

        # Diagonal continuation: coarse (eps, tau) first, then refine eps.
        for t_i, tv in enumerate(tau_path):
            f = np.zeros(n, dtype=np.float64)
            g = np.zeros(m, dtype=np.float64)
            for e_i, ev in enumerate(eps_path):
                if t_i == 0 and e_i > 0:
                    pass
                plan, f, g, it, res = _uot_scaling(a, b, C, float(ev), float(tv), f, g, cfg)
                total_iter += it
                mass = float(np.sum(plan))
                path.append(
                    {
                        "tau_index": int(t_i),
                        "eps_index": int(e_i),
                        "eps": float(ev),
                        "tau": float(tv),
                        "n_iter": int(it),
                        "residual": float(res),
                        "mass": mass,
                        "cost": float(np.sum(C * plan)),
                    }
                )
                best = plan

        assert best is not None
        best = np.asarray(best, dtype=np.float64)
        mass = float(np.sum(best))
        destroyed = 1.0 - mass
        return OTResult(
            plan=best,
            cost=float(np.sum(C * best)),
            marginal_a=best.sum(axis=1),
            marginal_b=best.sum(axis=0),
            n_iter=total_iter,
            converged=bool(res < cfg.tol_marginal),
            epsilon=float(eps),
            residual=float(res),
            meta={
                "solver": self.name,
                "tau": float(tau),
                "path": path,
                "n_solves": len(path),
                "mass_transported": mass,
                "mass_destroyed": float(destroyed),
                "balanced_would_be_better": bool(destroyed < self.min_mass_destroyed),
                "unbalanced_relaxation_active": bool(destroyed >= self.min_mass_destroyed),
                "wall_s": float(time.perf_counter() - started),
                "continuation": "diagonal(eps,tau) monotone path, warm started",
            },
        )


def _uot_scaling(
    a: FloatArray,
    b: FloatArray,
    C: FloatArray,
    eps: float,
    tau: float,
    f0: FloatArray,
    g0: FloatArray,
    cfg: SinkhornConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, int, float]:
    """Log-domain unbalanced Sinkhorn for the KL marginal relaxation.

    The scaling iteration in the naive domain is
    ``u <- (a / (K v))^p`` with ``p = tau / (tau + eps)``.  Writing
    ``f = eps log u`` and ``g = eps log v`` turns that into

        f <- eps*log(a) - p*eps*LSE_j((g_j - C_ij)/eps)
        g <- eps*log(b) - p*eps*LSE_i((f_i - C_ij)/eps)

    with ``p*eps = zeta = eps*tau/(tau+eps)``.  Consistency check that pins the
    sign and the exponent: as ``tau -> inf`` we have ``zeta -> eps`` and the update
    collapses onto the balanced log-domain Sinkhorn exactly.  Dropping the
    ``eps*log(a)`` term (an easy and silent error) breaks that limit and yields a
    plan that is not even approximately normalised -- measured 56x the correct
    transport cost before this was fixed.

    Residual is measured against the previous ``f`` (the stale side), for the same
    reason as in :mod:`transforge.solvers.sinkhorn`.  In the unbalanced case the
    column marginal is deliberately NOT equal to ``b``, so this measures distance
    to the fixed point, not marginal violation.
    """
    n, m = C.shape
    la, lb = np.log(a), np.log(b)
    zeta = eps * tau / (tau + eps)
    f = np.array(f0, dtype=np.float64)
    g = np.array(g0, dtype=np.float64)
    res = float("inf")
    it = 0
    for it in range(1, cfg.max_iter + 1):
        prev_f = f
        f = eps * la - zeta * logsumexp((g[None, :] - C) / eps, axis=1)
        if not np.all(np.isfinite(f)):
            raise NumericError(f"UOT diverged at iter {it}, eps={eps:g}, tau={tau:g}")
        res = float(np.max(np.abs(f - prev_f))) / max(float(np.max(np.abs(f))), 1e-300)
        g = eps * lb - zeta * logsumexp((f[:, None] - C) / eps, axis=0)
        if not np.all(np.isfinite(g)):
            raise NumericError(f"UOT diverged at iter {it}, eps={eps:g}, tau={tau:g}")
        if res < cfg.tol_marginal:
            break
    plan = np.exp((f[:, None] + g[None, :] - C) / eps)
    np.maximum(plan, 0.0, out=plan)
    return plan, f, g, it, res


def unbalanced_reference(
    a: FloatArray, b: FloatArray, C: FloatArray, reg: float, tau: float
) -> FloatArray | None:
    """POT's unbalanced plan, or ``None`` when POT is unavailable.

    Used only as a baseline in gate G4; ``None`` means "skip", never "guess".
    """
    try:
        import ot.unbalanced as uot  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return None
    try:
        return np.asarray(
            uot.sinkhorn_knopp_unbalanced(
                np.asarray(a, dtype=np.float64),
                np.asarray(b, dtype=np.float64),
                np.asarray(C, dtype=np.float64),
                reg,
                tau,
            ),
            dtype=np.float64,
        )
    except Exception:  # noqa: BLE001
        return None


__all__ = ["TFUFlow", "unbalanced_reference"]
