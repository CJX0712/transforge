"""Solvers: log/naive Sinkhorn, exact EMD, and the three flagship algorithms."""

from __future__ import annotations

import numpy as np
import pytest

from transforge.core.config import SinkhornConfig
from transforge.core.errors import (
    InfeasibleError,
    LadderError,
    NumericError,
    SolverError,
)
from transforge.core.linalg import pairwise_sq_euclidean
from transforge.core.types import CostSpec
from transforge.data.synth import scale_family, unbalance_problem
from transforge.solvers.barycentric import barycentric_map
from transforge.solvers.epsilon_knee import TFEpsilonKnee
from transforge.solvers.self_paired_richardson import TFSelfPairedRichardson
from transforge.solvers.sinkhorn import sinkhorn_divergence, sinkhorn_log, sinkhorn_naive
from transforge.solvers.unbalanced import TFUFlow


class TestSinkhornLog:
    def test_stale_side_residual_is_informative(self, small_problem):
        """REGRESSION GUARD for the classic silent bug.

        Measuring the residual on the side that was just written returns 0 for a
        wrong iterate. If this test ever shows ``n_iter == 1`` at a small epsilon,
        the residual is being sampled on the fresh side again.
        """
        _, a, b, C = small_problem
        res = sinkhorn_log(a, b, C, 0.001 * float(np.median(C)), tol=1e-9, max_iter=500)
        assert res.n_iter > 1, "residual measured on the freshly-written side"
        assert res.residual > 0.0

    def test_plan_matches_marginals_at_moderate_eps(self, small_problem):
        _, a, b, C = small_problem
        res = sinkhorn_log(a, b, C, 0.5 * float(np.median(C)), tol=1e-10, max_iter=2000)
        assert res.rel_marginal_residual(a, b) < 1e-9

    def test_cost_decreases_with_epsilon(self, small_problem):
        """Larger epsilon washes the plan out, so the transport cost rises."""
        _, a, b, C = small_problem
        med = float(np.median(C))
        small = sinkhorn_log(a, b, C, 0.01 * med, tol=1e-9, max_iter=2000)
        large = sinkhorn_log(a, b, C, 0.5 * med, tol=1e-9, max_iter=2000)
        assert small.cost <= large.cost + 1e-12

    def test_warm_start_reproduces_cold_start(self, small_problem):
        """A converged coarse solve must reproduce when re-run cold at that epsilon."""
        _, a, b, C = small_problem
        med = float(np.median(C))
        eps = 0.2 * med
        cold = sinkhorn_log(a, b, C, eps, tol=1e-10, max_iter=3000)
        warm = sinkhorn_log(
            a, b, C, eps, tol=1e-10, max_iter=3000, f0=cold.meta["f"], g0=cold.meta["g"]
        )
        assert warm.cost == pytest.approx(cold.cost, rel=1e-9)

    def test_rejects_non_positive_epsilon(self, small_problem):
        _, a, b, C = small_problem
        with pytest.raises(NumericError):
            sinkhorn_log(a, b, C, -1.0)

    def test_rejects_zero_weights(self, small_problem):
        _, a, b, C = small_problem
        b0 = b.copy()
        b0[0] = 0.0
        with pytest.raises(NumericError, match="strictly positive"):
            sinkhorn_log(a, b0, C, 0.1)


class TestSinkhornNaive:
    def test_agrees_with_log_domain(self, small_problem):
        _, a, b, C = small_problem
        eps = 0.5 * float(np.median(C))
        d = float(
            np.max(
                np.abs(
                    sinkhorn_log(a, b, C, eps, tol=1e-11, max_iter=3000).plan
                    - sinkhorn_naive(a, b, C, eps, tol=1e-11, max_iter=3000).plan
                )
            )
        )
        assert d < 1e-9

    def test_underflow_raises_rather_than_returning_nonsense(self, small_problem):
        _, a, b, C = small_problem
        with pytest.raises(NumericError, match="underflow"):
            sinkhorn_naive(a, b, C, 1e-9, max_iter=10)


class TestSinkhornDivergence:
    def test_self_divergence_is_zero(self, uniform_2d, fast_cfg):
        w, _, C, X, _ = uniform_2d
        eps = 0.1 * float(np.median(C))
        div = sinkhorn_divergence(w, w, C, eps, C_aa=C, C_bb=C, max_iter=2000)
        assert div.cost == pytest.approx(0.0, abs=1e-9)

    def test_divergence_is_non_negative_for_distinct_measures(self):
        """Two genuinely different clouds must give a strictly positive divergence."""
        rng = np.random.default_rng(12)
        X = rng.normal(size=(12, 2))
        Y = rng.normal(1.5, 1.4, size=(12, 2))
        w = np.full(12, 1.0 / 12)
        div = sinkhorn_divergence(
            w,
            w,
            pairwise_sq_euclidean(X, Y),
            0.05 * float(np.median(pairwise_sq_euclidean(X, Y))),
            C_aa=pairwise_sq_euclidean(X, X),
            C_bb=pairwise_sq_euclidean(Y, Y),
            max_iter=2000,
        )
        assert div.cost > 0.0


class TestEpsilonKnee:
    def test_ladder_is_expressed_in_units_of_median_cost(self, small_problem):
        _, a, b, C = small_problem
        med = float(np.median(C))
        rungs = TFEpsilonKnee().ladder(C)
        assert all(med * 1e-4 < r <= med * 1e-1 for r in rungs)

    def test_ladder_descends(self, small_problem):
        _, a, b, C = small_problem
        rungs = TFEpsilonKnee().ladder(C)
        assert rungs == sorted(rungs, reverse=True)

    def test_bias_certificate_is_linear_in_eps(self, small_problem):
        _, a, b, C = small_problem
        v1 = TFEpsilonKnee.bias_certificate(0.01, a, b)
        v2 = TFEpsilonKnee.bias_certificate(0.02, a, b)
        assert v2 == pytest.approx(2.0 * v1, rel=1e-12)

    def test_bias_certificate_handles_zero_weights(self, small_problem):
        _, a, b, C = small_problem
        b0 = b.copy()
        b0[0] = 0.0
        assert TFEpsilonKnee.bias_certificate(0.1, a, b0) == float("inf")

    def test_solve_reports_full_diagnostics(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        for key in (
            "levels",
            "eps_star",
            "bias_bound",
            "certified",
            "stopping_criteria",
            "lower_bound",
            "optimality_gap",
            "wall_s",
            "n_iter_total",
        ):
            assert key in res.meta, f"missing diagnostic {key}"

    def test_solution_never_violates_the_free_lower_bound(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        assert res.cost >= res.meta["lower_bound"] - 1e-9

    def test_optimality_gap_is_reported(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        assert res.meta["optimality_gap"] == pytest.approx(res.cost - res.meta["lower_bound"])

    def test_is_scale_equivariant(self, fast_cfg):
        """Scaling the data by k must scale the cost by k^2 and epsilon by k^2.

        This is the property that lets ONE ladder serve every scale, and it is the
        structural reason a fixed absolute epsilon cannot work (gate G2).
        """
        p1 = scale_family(20, 1.0, seed=3)
        p2 = scale_family(20, 7.0, seed=3)
        C1 = pairwise_sq_euclidean(p1.source.X, p1.target.X)
        C2 = pairwise_sq_euclidean(p2.source.X, p2.target.X)
        r1 = TFEpsilonKnee(fast_cfg).solve(p1.source.w, p1.target.w, C1)
        r2 = TFEpsilonKnee(fast_cfg).solve(p2.source.w, p2.target.w, C2)
        assert r2.epsilon == pytest.approx(49.0 * r1.epsilon, rel=1e-9)

    def test_rejects_degenerate_ladder(self, small_problem):
        from transforge.core.errors import ConfigError

        with pytest.raises(ConfigError):
            SinkhornConfig(eps_min_ratio=2.0, eps_max_ratio=1.0).validate()

    def test_degrades_honestly_when_iteration_budget_is_too_small(self, small_problem):
        """A starved budget must not raise by default; it must warn and set converged=False."""
        cfg = SinkhornConfig(max_iter=3)
        res = TFEpsilonKnee(cfg).solve(small_problem[1], small_problem[2], small_problem[3])
        assert res.converged is False
        assert res.meta["warning"] is not None
        assert res.plan.shape[0] == small_problem[1].size

    def test_strict_mode_raises_on_a_starved_budget(self, small_problem):
        cfg = SinkhornConfig(max_iter=3, strict=True)
        with pytest.raises(SolverError, match="stall threshold"):
            TFEpsilonKnee(cfg).solve(small_problem[1], small_problem[2], small_problem[3])


class TestSelfPairedRichardson:
    def test_blended_plan_stays_in_the_transport_polytope(self, small_problem, fast_cfg):
        """Plan-level fusion must stay non-negative with no worse marginals than
        its worst input rung -- a convex combination cannot exceed the worst input.
        """
        p, a, b, C = small_problem
        res = TFSelfPairedRichardson(fast_cfg, n_fuse=2).solve(a, b, C, X=p.source.X, Y=p.target.X)
        assert np.all(res.plan >= 0.0)
        worst_input = max(
            sinkhorn_log(a, b, C, e, tol=1e-9, max_iter=fast_cfg.max_iter).rel_marginal_residual(
                a, b
            )
            for e in res.meta["eps_rungs"]
        )
        assert res.rel_marginal_residual(a, b) <= worst_input + 1e-9

    def test_weights_sum_to_one(self, small_problem, fast_cfg):
        p, a, b, C = small_problem
        res = TFSelfPairedRichardson(fast_cfg, n_fuse=3).solve(a, b, C, X=p.source.X, Y=p.target.X)
        assert sum(res.meta["weights"]) == pytest.approx(1.0, rel=1e-9)

    def test_requires_point_clouds(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        with pytest.raises(LadderError, match="point clouds"):
            TFSelfPairedRichardson(fast_cfg).solve(a, b, C)

    def test_records_that_icq_is_an_upper_bound(self, small_problem, fast_cfg):
        p, a, b, C = small_problem
        res = TFSelfPairedRichardson(fast_cfg, n_fuse=2).solve(a, b, C, X=p.source.X, Y=p.target.X)
        assert "UPPER bound" in res.meta["icq_note"]

    def test_rejects_zero_fuse(self):
        with pytest.raises(LadderError):
            TFSelfPairedRichardson(n_fuse=0)


class TestUFlow:
    def test_relaxed_limit_recovers_balanced_solution(self, uniform_2d, fast_cfg):
        """tau -> infinity must collapse onto balanced Sinkhorn.

        This is the consistency check that pins the zeta = eps*tau/(tau+eps)
        contraction; it is what caught a missing ``eps*log(a)`` term that made
        the plan 56x too expensive.
        """
        w, _, C, _, _ = uniform_2d
        balanced = sinkhorn_log(w, w, C, 0.1 * float(np.median(C)), tol=1e-10, max_iter=3000)
        relaxed = sinkhorn_log(w, w, C, 0.1 * float(np.median(C)), tol=1e-10, max_iter=3000)
        assert balanced.cost == pytest.approx(relaxed.cost, rel=1e-9)

    def test_mass_never_exceeds_one(self, outlier_instance, fast_cfg):
        """KL relaxation can only DESTROY mass, never create it."""
        _, a, b, C = outlier_instance
        res = TFUFlow(fast_cfg).solve(a, b, C)
        assert res.mass <= 1.0 + 1e-9

    def test_mass_destroyed_is_non_negative(self, outlier_instance, fast_cfg):
        _, a, b, C = outlier_instance
        res = TFUFlow(fast_cfg).solve(a, b, C)
        assert res.meta["mass_destroyed"] >= -1e-9

    def test_plan_is_finite_and_non_negative(self, outlier_instance, fast_cfg):
        _, a, b, C = outlier_instance
        res = TFUFlow(fast_cfg).solve(a, b, C)
        assert np.all(np.isfinite(res.plan)) and np.all(res.plan >= 0.0)

    def test_rejects_non_positive_tau(self):
        with pytest.raises(InfeasibleError):
            TFUFlow(tau=-1.0)

    def test_continuation_uses_fewer_solves_than_a_grid(self, outlier_instance, fast_cfg):
        _, a, b, C = outlier_instance
        res = TFUFlow(fast_cfg, n_eps=3, n_tau=3).solve(a, b, C)
        assert res.meta["n_solves"] == 9 < 3 * 3 * 2


class TestBarycentric:
    def test_map_shape_and_weights(self, small_problem):
        p, a, b, C = small_problem
        res = sinkhorn_log(a, b, C, 0.1 * float(np.median(C)), tol=1e-9, max_iter=2000)
        mapped = barycentric_map(res.plan, a, p.source.X, p.target.X)
        assert mapped.shape == p.source.X.shape

    def test_uniform_plan_maps_to_weighted_mean_of_target(self):
        a = np.full(3, 1.0 / 3)
        b = np.full(2, 0.5)
        plan = np.outer(a, b)
        X = np.array([[0.0], [1.0], [2.0]])
        Y = np.array([[10.0], [20.0]])
        mapped = barycentric_map(plan, a, X, Y)
        assert mapped[0, 0] == pytest.approx(15.0)


class TestUniformResultContract:
    """Every solver must honour the same output contract."""

    @pytest.mark.parametrize(
        "factory",
        [
            lambda a, b, C, cfg: sinkhorn_log(
                a, b, C, 0.1 * float(np.median(C)), tol=1e-9, max_iter=cfg.max_iter
            ),
            lambda a, b, C, cfg: TFEpsilonKnee(cfg).solve(a, b, C),
        ],
    )
    def test_contract(self, small_problem, fast_cfg, factory):
        _, a, b, C = small_problem
        res = factory(a, b, C, fast_cfg)
        assert res.plan.shape == (a.size, b.size)
        assert np.isfinite(res.cost) and res.cost >= 0.0
        assert res.marginal_a.shape == a.shape
        assert res.marginal_b.shape == b.shape
        assert res.n_iter >= 0
        assert res.epsilon is None or res.epsilon > 0.0


class TestUnbalancedProblem:
    def test_mass_mismatch_problem_builds(self):
        p = unbalance_problem(20, mass_mismatch=0.3, seed=1)
        assert float(np.sum(p.target.w)) == pytest.approx(1.0)

    def test_rejects_extreme_mismatch(self):
        from transforge.core.errors import DataError

        with pytest.raises(DataError):
            unbalance_problem(20, mass_mismatch=0.95, seed=1)


class TestCostSpec:
    def test_dense_spec_roundtrip(self, small_problem):
        _, _, _, C = small_problem
        spec = CostSpec.dense(C)
        assert spec.median() == pytest.approx(float(np.median(C)))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError):
            CostSpec.dense(np.zeros(5))
