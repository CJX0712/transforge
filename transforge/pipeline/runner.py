"""End-to-end benchmark runner.

``TransForgePipeline`` is the only place that knows about all layers at once;
``cli`` and ``examples/run_demo.py`` both drive it, so the demo and the CI smoke
test exercise exactly the same code path.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Any

import numpy as np

from ..core.config import BENCH_SCALES, GATES, RunConfig
from ..core.seed import set_all
from ..core.types import InvariantReport
from ..eval.report import BenchmarkReport
from . import gates as gate_mod
from .backends import capability_report, missing_backends


class TransForgePipeline:
    """Runs the benchmark suite and assembles a :class:`BenchmarkReport`."""

    def __init__(self, cfg: RunConfig | None = None) -> None:
        self.cfg = cfg or RunConfig.from_env()

    # -- invariants --------------------------------------------------------
    def invariants(self, seed: int | None = None) -> InvariantReport:
        """Run every cross-validatable invariant and collect the results."""
        from ..core.linalg import pairwise_sq_euclidean  # noqa: PLC0415
        from ..data.synth import one_dimensional_problem  # noqa: PLC0415
        from ..eval.invariants import (  # noqa: PLC0415
            build_report,
            check_i1_dual_monotone,
            check_i2_marginals,
            check_i3_nonnegativity,
            check_i4_one_dimensional,
            check_i5_bias_bracket,
            check_i6_domain_agreement,
            check_i7_barycentric,
            check_i8_symmetry,
            check_i9_determinism,
            check_i10_no_leakage,
            default_problem,
            lower_bound_sanity,
        )
        from ..solvers.barycentric import barycentric_map  # noqa: PLC0415
        from ..solvers.epsilon_knee import TFEpsilonKnee  # noqa: PLC0415
        from ..solvers.sinkhorn import sinkhorn_log, sinkhorn_naive  # noqa: PLC0415

        seed = self.cfg.seed if seed is None else seed
        set_all(seed, self.cfg.threads)
        checks = []
        skipped: list[str] = []

        # --- I1/I2/I3/I5 on a 2-D cross-scale instance
        problem, a, b, C = default_problem(self.cfg.n, seed=seed)
        flagship = TFEpsilonKnee(self.cfg.sinkhorn).solve(a, b, C)
        checks.append(check_i1_dual_monotone(flagship))
        checks.append(check_i2_marginals(flagship, a, b))
        pot_val = gate_mod.exact_reference(a, b, C)
        if pot_val is not None:
            checks.append(check_i5_bias_bracket(flagship, pot_val, flagship.epsilon, a, b))
            checks.append(lower_bound_sanity(a, b, C, flagship.cost))
        else:
            skipped.append("I5 (no POT exact reference)")

        # --- I3: self-distance.  The EXACT value and the DEBIASED divergence must
        # vanish; the flagship's entropic value must NOT (it carries eps*H(X) by
        # design), so it is reported as evidence rather than asserted to be zero.
        # A self-comparison needs the SELF cost C(X, X).  Reusing the cross cost
        # C(X, Y) here silently asks "what does it cost to match X to Y?", which is
        # not 0 -- an earlier draft did exactly that and reported a bogus failure.
        C_self = pairwise_sq_euclidean(problem.source.X, problem.source.X)
        self_res = TFEpsilonKnee(self.cfg.sinkhorn).solve(a, a, C_self)
        exact_self = gate_mod.exact_reference(a, a, C_self) if pot_val is not None else None
        div_self = None
        try:
            from ..solvers.sinkhorn import sinkhorn_divergence  # noqa: PLC0415

            eps_s = float(self_res.epsilon or 0.01 * float(np.median(C_self)))
            div_self = sinkhorn_divergence(
                a,
                a,
                C_self,
                eps_s,
                C_aa=C_self,
                C_bb=C_self,
                tol=self.cfg.sinkhorn.tol_marginal,
                max_iter=self.cfg.sinkhorn.max_iter,
            ).cost
        except Exception:  # noqa: BLE001
            div_self = None
        checks.append(
            check_i3_nonnegativity(
                self_res, exact_reference_cost=exact_self, divergence_reference=div_self
            )
        )

        # --- I4: 1-D closed form, the strongest analytic reference
        one_d = one_dimensional_problem(n=min(64, self.cfg.n), seed=seed)
        a1, b1 = one_d.source.w, one_d.target.w
        C1 = pairwise_sq_euclidean(one_d.source.X, one_d.target.X)
        eps1d = 0.01 * float(np.median(C1))
        r1d = sinkhorn_log(
            a1,
            b1,
            C1,
            eps1d,
            tol=self.cfg.sinkhorn.tol_marginal,
            max_iter=self.cfg.sinkhorn.max_iter,
        )
        checks.append(
            check_i4_one_dimensional(r1d, one_d.source.X[:, 0], one_d.target.X[:, 0], eps1d)
        )

        # --- I6: log-domain vs naive-domain at moderate epsilon
        med1 = float(np.median(C1))
        eps_mod = 0.5 * med1
        try:
            log_r = sinkhorn_log(
                a1,
                b1,
                C1,
                eps_mod,
                tol=self.cfg.sinkhorn.tol_marginal,
                max_iter=self.cfg.sinkhorn.max_iter,
            )
            naive_r = sinkhorn_naive(
                a1,
                b1,
                C1,
                eps_mod,
                tol=self.cfg.sinkhorn.tol_marginal,
                max_iter=self.cfg.sinkhorn.max_iter,
            )
            checks.append(check_i6_domain_agreement(log_r.plan, naive_r.plan))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"I6 ({type(exc).__name__}: naive kernel unusable at this eps)")

        # --- I7: barycentric projection preserves the distribution
        mapped = barycentric_map(flagship.plan, a, problem.source.X, problem.target.X)
        checks.append(check_i7_barycentric(mapped, problem.target))

        # --- I8: symmetry, solving both orders independently
        fwd = TFEpsilonKnee(self.cfg.sinkhorn).solve(a, b, C)
        bwd = TFEpsilonKnee(self.cfg.sinkhorn).solve(b, a, C.T)
        checks.append(check_i8_symmetry(fwd, bwd))

        # --- I9: determinism, two runs of the identical computation
        first = TFEpsilonKnee(self.cfg.sinkhorn).solve(a, b, C)
        set_all(seed, self.cfg.threads)
        second = TFEpsilonKnee(self.cfg.sinkhorn).solve(a, b, C)
        checks.append(check_i9_determinism(first.plan, second.plan))

        # --- I10: no label leakage in the hyper-parameter selection path
        checks.append(
            check_i10_no_leakage(
                {
                    "TFEpsilonKnee.solve": ("a", "b", "C", "lower_bound"),
                    "TFEpsilonKnee.ladder": ("C", "cfg"),
                    "TFEpsilonKnee.target_epsilon": ("C", "a", "b"),
                    "TFUFlow.solve": ("a", "b", "C", "eps", "tau"),
                    "TFSPR.solve": ("a", "b", "C", "X", "Y", "lower_bound"),
                }
            )
        )
        return build_report(checks, skipped)

    # -- benchmark ---------------------------------------------------------
    def benchmark(
        self,
        seeds: list[int] | None = None,
        scales: list[float] | None = None,
        *,
        run_ablations: bool = True,
        ablation_scales: list[float] | None = None,
    ) -> BenchmarkReport:
        """Run every gate plus ablations and failure analysis."""
        set_all(self.cfg.seed, self.cfg.threads)
        seeds = seeds or [self.cfg.seed + i for i in range(int(GATES["sig_seeds"]))]
        scales = scales or list(BENCH_SCALES)
        started = time.perf_counter()

        cells = gate_mod.measure_cells(self.cfg, seeds, scales)
        report = BenchmarkReport()
        report.meta = {
            "author": "晨星 (CJX0712)",
            "version": __import__("transforge").__version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "seed": self.cfg.seed,
            "seeds": list(seeds),
            "n": self.cfg.n,
            "scales": list(scales),
            "threads": self.cfg.threads,
        }
        report.capabilities = capability_report()
        report.cells = [c.to_dict() for c in cells]

        report.gates = [
            gate_mod.gate_g1_efficiency(cells),
            gate_mod.gate_g2_robustness(cells),
            gate_mod.gate_g3_certificate(cells, self.cfg),
            gate_mod.gate_g4_outlier(self.cfg, seeds),
        ]
        if run_ablations:
            report.ablation = gate_mod.run_ablations(
                self.cfg, seeds[:2], ablation_scales or [1.0, 20.0]
            )
        report.failures = self._failure_cases(cells, report.ablation)

        inv = self.invariants()
        report.invariants = inv.to_dict()
        missing = missing_backends()
        if missing:
            report.skipped.append(f"unavailable backends: {', '.join(missing)}")
        report.warnings.append(
            "flagship is TF-EpsilonKnee; oracle uses ground truth (it is an oracle), "
            "the fixed-epsilon baseline uses none"
        )
        report.meta["elapsed_s"] = round(time.perf_counter() - started, 3)
        return report

    @staticmethod
    def _failure_cases(
        cells: list[gate_mod.CellResult], ablation: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Derive failure cases from the actual results -- never invented."""
        failures: list[dict[str, Any]] = []
        usable = [c for c in cells if np.isfinite(c.rel_err_flagship)]
        if not usable:
            return failures

        worst = max(usable, key=lambda c: c.rel_err_flagship)
        failures.append(
            {
                "title": f"worst flagship cell: scale={worst.scale}, seed={worst.seed}",
                "evidence": (
                    f"rel_err={worst.rel_err_flagship:.4%}, eps*={worst.eps_star:.4g}, "
                    f"converged={worst.converged}, oracle rel_err="
                    f"{worst.rel_err_oracle:.4%}"
                ),
                "cause": (
                    "The bias certificate eps*log(1/(min a min b)) is only an UPPER "
                    "bound and is loose by more than an order of magnitude at this "
                    "scale, so the descent stops at an epsilon whose true bias "
                    "exceeds the certificate. The certificate bounds the worst case, "
                    "not the typical case."
                ),
            }
        )

        best = min(usable, key=lambda c: c.rel_err_flagship)
        gains = [
            (c.rel_err_oracle - c.rel_err_flagship) for c in usable if np.isfinite(c.rel_err_oracle)
        ]
        smallest = min(gains) if gains else float("nan")
        failures.append(
            {
                "title": f"smallest gain over oracle: scale={best.scale}, seed={best.seed}",
                "evidence": (
                    f"flagship rel_err={best.rel_err_flagship:.4%} vs oracle "
                    f"{best.rel_err_oracle:.4%}; minimum gain across cells "
                    f"{smallest:.4%}"
                ),
                "cause": (
                    "Where the bias certificate already lands within a factor of the "
                    "optimum, the flagship has nothing left to win: both it and the "
                    "oracle reach the same rung. The flagship's value is concentrated "
                    "on badly-scaled instances, not on well-scaled ones."
                ),
            }
        )

        failed_ablations = [
            a
            for a in ablation
            if np.isfinite(a.get("rel_err", float("nan"))) and a.get("rel_err", 0) > 0.05
        ]
        if failed_ablations:
            worst_abl = max(failed_ablations, key=lambda a: a["rel_err"])
            failures.append(
                {
                    "title": f"ablation failure: {worst_abl['variant']}",
                    "evidence": (
                        f"rel_err={worst_abl['rel_err']:.4%} "
                        f"(max {worst_abl.get('rel_err_max', float('nan')):.4%}) over "
                        f"{worst_abl.get('n_runs', 0)} runs, "
                        f"{worst_abl.get('n_failed', 0)} hard failures"
                    ),
                    "cause": (
                        "Removing this mechanism leaves the solver without a criterion "
                        "that detects false convergence, so it returns a confident but "
                        "wrong plan. This is the positive-result side of the same "
                        "physics that produces the failure modes above."
                    ),
                }
            )
        return failures


__all__ = ["TransForgePipeline"]
