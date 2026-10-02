"""TF-SPR -- flagship 2: self-paired Richardson fusion of ladder plans.

Three corrections are baked in here, each because the obvious version is wrong:

1. **No ICQ "lower bound" weighting.**  Entropic / quadratic regularisation
   (including ICQ) yields an UPPER bound on the OT value, not a lower bound: for
   any ``R >= 0``, ``min_pi (cost + R) >= min_pi cost``.  Using it as a
   "guaranteed-not-to-underestimate" criterion inverts the direction and the
   metric collapses.  ICQ is therefore used only as a *diagnostic floor*.
2. **Fusion happens at the PLAN level, never the map level.**  A convex
   combination of plans preserves the marginals exactly and stays inside the
   transport polytope; a convex combination of barycentric maps preserves
   nothing.
3. **Honest expectation.**  A convex blend cannot beat the best single rung
   (it destroys each rung's KL optimality).  The win available here is variance
   reduction and robustness, not beating the best rung.  We report it that way.

Weight source: the debiased Sinkhorn divergence of each rung (no ground truth,
no target labels -- invariant I10), which is the only quantity comparable across
epsilon.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..core.config import SinkhornConfig
from ..core.errors import LadderError
from ..core.linalg import pairwise_sq_euclidean
from ..core.types import FloatArray, OTResult
from .epsilon_knee import TFEpsilonKnee
from .sinkhorn import sinkhorn_log


class TFSelfPairedRichardson:
    """Blend the plans of several ladder rungs with divergence-based weights.

    The Richardson-style extrapolation ``2*S_j - S_{j+1}`` over a geometric ladder
    cancels the leading ``O(eps)`` bias term at zero extra solve cost, because both
    rungs are already on the ladder.
    """

    name = "TF-SPR"
    requires_backend = "numpy"

    def __init__(self, cfg: SinkhornConfig | None = None, n_fuse: int = 3) -> None:
        self.cfg = cfg or SinkhornConfig()
        self.cfg.validate()
        if n_fuse < 1:
            raise LadderError(f"n_fuse must be >= 1, got {n_fuse}")
        self.n_fuse = int(n_fuse)

    def solve(
        self,
        a: FloatArray,
        b: FloatArray,
        C: FloatArray,
        *,
        X: FloatArray | None = None,
        Y: FloatArray | None = None,
        lower_bound: float | None = None,
    ) -> OTResult:
        """Fuse the deepest ``n_fuse`` rungs around the flagship's chosen epsilon.

        ``X``/``Y`` are the underlying point clouds; they are needed only to build
        the self-costs ``C_aa``/``C_bb`` required for debiasing.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        if X is None or Y is None:
            raise LadderError(
                "TF-SPR needs the point clouds X and Y to build the self-costs "
                "required for debiasing; pass X=..., Y=..."
            )
        started = time.perf_counter()
        base = TFEpsilonKnee(self.cfg).solve(a, b, C, lower_bound=lower_bound)
        rungs: list[dict[str, Any]] = base.meta["levels"]

        idx = int(base.meta["levels_total"] - 1)
        picks = [max(idx - k, 0) for k in range(self.n_fuse)][::-1]
        chosen_levels = [rungs[i] for i in picks]

        C_aa = pairwise_sq_euclidean(X, X)
        C_bb = pairwise_sq_euclidean(Y, Y)
        self_costs: dict[float, tuple[float, float]] = {}
        divergences: list[float] = []
        plans: list[FloatArray] = []
        eps_list: list[float] = []
        total_iter = 0

        for lvl in chosen_levels:
            eps = float(lvl["eps"])
            if eps not in self_costs:
                aa = sinkhorn_log(
                    a, a, C_aa, eps, tol=self.cfg.tol_marginal, max_iter=self.cfg.max_iter
                )
                bb = sinkhorn_log(
                    b, b, C_bb, eps, tol=self.cfg.tol_marginal, max_iter=self.cfg.max_iter
                )
                self_costs[eps] = (aa.cost, bb.cost)
                total_iter += aa.n_iter + bb.n_iter
            aa_cost, bb_cost = self_costs[eps]
            div = sinkhorn_log(a, b, C, eps, tol=self.cfg.tol_marginal, max_iter=self.cfg.max_iter)
            total_iter += div.n_iter
            divergences.append(div.cost - 0.5 * aa_cost - 0.5 * bb_cost)
            plans.append(div.plan)
            eps_list.append(eps)

        divergences_arr = np.asarray(divergences, dtype=np.float64)
        weights = self._weights(divergences_arr)
        blended = np.tensordot(weights, np.stack(plans), axes=(0, 0))
        np.maximum(blended, 0.0, out=blended)

        richardson = None
        if len(eps_list) >= 2:
            # Zero-extra-cost leading-order bias cancellation on the geometric ladder.
            richardson = 2.0 * divergences[0] - divergences[1]

        cost = float(np.sum(C * blended))
        from ..core.linalg import ot_lower_bound  # noqa: PLC0415

        lb = ot_lower_bound(C, a, b) if lower_bound is None else float(lower_bound)
        return OTResult(
            plan=blended,
            cost=cost,
            marginal_a=blended.sum(axis=1),
            marginal_b=blended.sum(axis=0),
            n_iter=total_iter,
            converged=bool(base.converged),
            epsilon=float(base.epsilon) if base.epsilon is not None else None,
            residual=float(base.residual),
            meta={
                "solver": self.name,
                "base": base,
                "eps_rungs": [float(e) for e in eps_list],
                "divergences": [float(d) for d in divergences],
                "weights": [float(w) for w in weights],
                "weight_source": "debiased_sinkhorn_divergence",
                "richardson_estimate": None if richardson is None else float(richardson),
                "lower_bound": float(lb),
                "wall_s": float(time.perf_counter() - started),
                "icq_note": (
                    "ICQ/entropy regularisation provides an UPPER bound on the OT "
                    "value, never a lower bound; it is excluded from the weights by "
                    "design and used only as a diagnostic floor."
                ),
                "honest_expectation": (
                    "plan-level convex fusion cannot beat the best single rung; the "
                    "gain is variance reduction and robustness, not peak quality"
                ),
            },
        )

    @staticmethod
    def _weights(divergences: FloatArray) -> FloatArray:
        """Softmax weights with a data-derived temperature (no manual knob).

        ``gamma = median(divergences)``.  Using the data's own median removes a
        hyper-parameter a human could tune to manufacture a win.
        """
        if divergences.size == 1:
            return np.ones(1, dtype=np.float64)
        gamma = float(np.median(divergences))
        if not np.isfinite(gamma) or gamma <= 0.0:
            gamma = float(np.mean(np.abs(divergences))) or 1.0
        logits = -divergences / gamma
        logits -= np.max(logits)
        w = np.exp(logits)
        total = float(np.sum(w))
        if total <= 0.0:
            return np.full(divergences.size, 1.0 / divergences.size)
        return w / total


__all__ = ["TFSelfPairedRichardson"]
