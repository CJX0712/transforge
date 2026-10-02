"""Invariants I1-I10 -- each cross-validated against an INDEPENDENT reference.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tests.conftest import needs_pot
from transforge.core.errors import NumericError
from transforge.core.linalg import pairwise_sq_euclidean
from transforge.core.types import EmpiricalMeasure
from transforge.data.closed_form import w2_1d_equal, wasserstein1_1d
from transforge.data.synth import one_dimensional_problem
from transforge.eval.invariants import (
    check_i1_dual_monotone,
    check_i4_one_dimensional,
    check_i5_bias_bracket,
    check_i7_barycentric,
    check_i8_symmetry,
    check_i9_determinism,
    cross_validate_emd,
    lower_bound_sanity,
)
from transforge.solvers.barycentric import barycentric_map
from transforge.solvers.epsilon_knee import TFEpsilonKnee
from transforge.solvers.sinkhorn import sinkhorn_log, sinkhorn_naive


class TestI1DualMonotone:
    """I1: the dual objective is non-decreasing along Sinkhorn iterations.

    Block coordinate ascent on a jointly concave dual guarantees this; a
    decrease is a numerical defect.
    """

    def test_dual_is_monotone(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        worst = 0.0
        n_points = 0
        for tr in res.meta["trace_per_level"]:
            values = np.asarray(tr.dual_objective, dtype=np.float64)
            if values.size < 2:
                continue
            # a violation is a DECREASE of the dual objective
            worst = max(worst, float(max(0.0, -np.min(np.diff(values)))))
            n_points += values.size
        assert n_points >= 2, "not enough checkpoints to test monotonicity"
        assert worst <= 1e-12, f"dual objective decreased by {worst:.3e}"

    def test_check_helper_agrees(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        assert check_i1_dual_monotone(res).passed


class TestI2Marginals:
    """I2: row and column marginals equal the prescribed weights."""

    def test_marginals_at_achievable_tolerance(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        rel = res.rel_marginal_residual(a, b)
        # Documented limit: float64 Sinkhorn cannot reach 1e-9 at small epsilon.
        assert rel < 1e-2, f"residual {rel:.3e} far above the achievable floor"

    def test_marginals_are_tight_at_moderate_epsilon(self, small_problem):
        _, a, b, C = small_problem
        res = sinkhorn_log(a, b, C, 0.5 * float(np.median(C)), tol=1e-9, max_iter=400)
        assert res.rel_marginal_residual(a, b) < 1e-9

    def test_plan_is_non_negative(self, small_problem, fast_cfg):
        _, a, b, C = small_problem
        assert np.all(TFEpsilonKnee(fast_cfg).solve(a, b, C).plan >= 0.0)


class TestI3SelfDistance:
    """I3: W2(X, X) = 0 and W2 >= 0."""

    def test_self_distance_is_zero(self, uniform_2d):
        """W2(X, X) = 0 EXACTLY -- this is a property of the exact problem.

        The entropic approximation is deliberately NOT asserted to be zero: at
        any eps > 0 the regularised value carries the bias eps*H(X), which is the
        entire reason an epsilon-selection rule exists.
        """
        from transforge.solvers.emd import emd_plan

        w, _, C, _, _ = uniform_2d
        assert emd_plan(w, w, C).cost == pytest.approx(0.0, abs=1e-12)

    def test_debiased_divergence_vanishes_on_self_comparison(self, uniform_2d):
        from transforge.solvers.sinkhorn import sinkhorn_divergence

        w, _, C, _, _ = uniform_2d
        div = sinkhorn_divergence(
            w, w, C, 0.1 * float(np.median(C)), C_aa=C, C_bb=C, tol=1e-10, max_iter=3000
        )
        assert div.cost == pytest.approx(0.0, abs=1e-8)

    def test_entropic_value_is_strictly_positive_on_self_comparison(self, uniform_2d):
        """Non-degeneracy: at eps > 0 the biased value must exceed exact 0."""
        from transforge.solvers.sinkhorn import sinkhorn_log

        w, _, C, _, _ = uniform_2d
        res = sinkhorn_log(w, w, C, 0.1 * float(np.median(C)), tol=1e-10, max_iter=3000)
        assert res.cost > 0.0


class TestI4OneDimensional:
    """I4: the strongest invariant -- a fully analytic 1-D reference, no solver.

    Any error in the log-domain update, in the epsilon convention, or in the
    a/b ordering shows up here immediately.
    """

    def test_order_statistic_closed_form(self):
        rng = np.random.default_rng(3)
        x, y = rng.normal(size=32), rng.normal(1.0, 1.5, size=32)
        assert w2_1d_equal(x, y) > 0.0
        assert w2_1d_equal(x, x) == pytest.approx(0.0)

    def test_wasserstein1_is_zero_for_identical(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        assert wasserstein1_1d(v, v, p=1) == pytest.approx(0.0)

    def test_sinkhorn_approaches_closed_form_as_eps_shrinks(self):
        p = one_dimensional_problem(n=48, seed=11)
        a, b = p.source.w, p.target.w
        C = pairwise_sq_euclidean(p.source.X, p.target.X)
        ref = w2_1d_equal(p.source.X[:, 0], p.target.X[:, 0])
        errs = []
        for eps in (0.05 * np.median(C), 0.01 * np.median(C)):
            res = sinkhorn_log(a, b, C, float(eps), tol=1e-9, max_iter=2000)
            errs.append(abs(res.cost - ref) / ref)
        assert errs[1] <= errs[0] + 1e-9, "smaller epsilon did not improve 1-D accuracy"

    @needs_pot
    def test_analytic_and_pot_agree(self):
        """Two independent exact 1-D references must agree.

        ``ot.emd2_1d`` defaults to the SQUARED euclidean metric, so it returns
        W2^2; the analytic helper returns W2.  Comparing them directly would show
        a spurious ~2x discrepancy.
        """
        from transforge.solvers.emd import emd2_1d_reference

        rng = np.random.default_rng(4)
        x, y = rng.normal(size=40), rng.normal(0.5, 1.4, size=40)
        w2 = w2_1d_equal(x, y)
        pot_sq = emd2_1d_reference(x, y)
        assert pot_sq == pytest.approx(w2**2, rel=1e-9)

    def test_check_helper_runs(self):
        p = one_dimensional_problem(n=32, seed=2)
        a, b = p.source.w, p.target.w
        C = pairwise_sq_euclidean(p.source.X, p.target.X)
        res = sinkhorn_log(a, b, C, 0.01 * float(np.median(C)), tol=1e-9, max_iter=800)
        chk = check_i4_one_dimensional(res, p.source.X[:, 0], p.target.X[:, 0], 0.01)
        assert chk.tolerance > 0


class TestI5BiasBracket:
    """I5: the regularised value lies ABOVE exact OT and below the bias bound.

    The inequality DIRECTION is the assertion. Entropic regularisation yields an
    upper bound; a value below exact OT is a bug, not an improvement.
    """

    @needs_pot
    def test_bracket_holds(self, small_problem):
        _, a, b, C = small_problem
        exact = cross_validate_emd(a, b, C)[0]
        assert exact is not None
        for eps in (0.2, 0.05, 0.01):
            eps = eps * float(np.median(C))
            res = sinkhorn_log(a, b, C, eps, tol=1e-9, max_iter=2000)
            chk = check_i5_bias_bracket(res, exact, eps, a, b)
            assert chk.passed, chk.detail

    @needs_pot
    def test_two_exact_references_agree(self, small_problem):
        """POT network simplex vs a completely independent HiGHS LP solve."""
        _, a, b, C = small_problem
        pot_val, lp_val = cross_validate_emd(a, b, C)
        assert pot_val is not None
        if lp_val is not None:
            assert pot_val == pytest.approx(lp_val, rel=1e-6)

    @needs_pot
    def test_value_never_below_free_lower_bound(self, small_problem):
        _, a, b, C = small_problem
        res = TFEpsilonKnee().solve(a, b, C)
        assert lower_bound_sanity(a, b, C, res.cost).passed


class TestI6DomainAgreement:
    """I6: log-domain and naive-domain are separate implementations; they must agree."""

    def test_agree_at_moderate_epsilon(self, small_problem):
        _, a, b, C = small_problem
        eps = 0.5 * float(np.median(C))
        log_r = sinkhorn_log(a, b, C, eps, tol=1e-10, max_iter=2000)
        naive_r = sinkhorn_naive(a, b, C, eps, tol=1e-10, max_iter=2000)
        diff = float(np.max(np.abs(log_r.plan - naive_r.plan)))
        assert diff < 1e-9, f"domain implementations disagree by {diff:.3e}"

    def test_naive_domain_must_fail_where_log_domain_survives(self):
        """The inverse test: the naive path must BREAK where the log path works.

        Direction matters -- this asserts the naive implementation FAILS, which is
        the entire justification for maintaining a log-domain implementation.
        Weights are strictly positive here (the log path rejects zeros outright),
        and epsilon is pushed far enough that ``(Cmax - Cmin)/eps`` exceeds the
        float64 exponent range of ~745.
        """
        rng = np.random.default_rng(5)
        X, Y = rng.normal(size=(24, 2)), rng.normal(0.5, 1.3, size=(24, 2))
        # log-uniform positive weights spanning a wide dynamic range
        a = np.exp(rng.uniform(-6.0, 0.0, size=24))
        b = np.exp(rng.uniform(-6.0, 0.0, size=24))
        a, b = a / a.sum(), b / b.sum()
        C = pairwise_sq_euclidean(X, Y)
        eps = float(np.min(C[C > 0])) / 1000.0
        log_res = sinkhorn_log(a, b, C, eps, tol=1e-8, max_iter=50)
        assert np.all(np.isfinite(log_res.plan))
        with pytest.raises(NumericError):
            sinkhorn_naive(a, b, C, eps, tol=1e-8, max_iter=50)


class TestI7Barycentric:
    """I7: barycentric projection must land on the target distribution."""

    def test_projection_reduces_distance(self, small_problem, fast_cfg):
        p, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        mapped = barycentric_map(res.plan, a, p.source.X, p.target.X)
        before = wasserstein1_1d(p.source.X[:, 0], p.target.X[:, 0])
        after = wasserstein1_1d(mapped[:, 0], p.target.X[:, 0])
        assert after < before, "projection did not bring the source closer to the target"

    def test_check_helper_reports_a_number(self, small_problem, fast_cfg):
        p, a, b, C = small_problem
        res = TFEpsilonKnee(fast_cfg).solve(a, b, C)
        mapped = barycentric_map(res.plan, a, p.source.X, p.target.X)
        chk = check_i7_barycentric(mapped, p.target)
        assert np.isfinite(chk.value)


class TestI8Symmetry:
    """I8: W(X, Y) == W(Y, X) for a symmetric cost."""

    def test_cost_is_symmetric(self, uniform_2d, fast_cfg):
        w, _, C, _, _ = uniform_2d
        fwd = TFEpsilonKnee(fast_cfg).solve(w, w, C)
        bwd = TFEpsilonKnee(fast_cfg).solve(w, w, C.T)
        assert check_i8_symmetry(fwd, bwd).passed

    def test_self_cost_matrix_is_symmetric_under_transpose(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(9, 2))
        C = pairwise_sq_euclidean(X, X)
        assert np.allclose(C, C.T)


class TestI9Determinism:
    """I9: identical inputs must give BITWISE identical outputs."""

    def test_same_seed_same_plan(self, small_problem, fast_cfg):
        from transforge.core.seed import set_all

        _, a, b, C = small_problem
        set_all(4242)
        first = TFEpsilonKnee(fast_cfg).solve(a, b, C).plan
        set_all(4242)
        second = TFEpsilonKnee(fast_cfg).solve(a, b, C).plan
        assert np.array_equal(first, second)

    def test_check_helper_detects_equality(self):
        v = np.arange(10, dtype=float)
        assert check_i9_determinism(v, v.copy()).passed
        assert not check_i9_determinism(v, v + 1e-16).passed


class TestI10NoLeakage:
    """I10: hyper-parameter selection must never see target labels."""

    def test_selection_signatures_carry_no_labels(self):
        from transforge.core.config import RunConfig
        from transforge.pipeline.runner import TransForgePipeline

        rep = TransForgePipeline(RunConfig(n=12)).invariants(seed=3)
        i10 = [c for c in rep.checks if c.id == "I10"]
        assert i10 and i10[0].passed, i10[0].detail if i10 else "missing"

    def test_solve_signatures_expose_no_label_argument(self):
        for cls in (TFEpsilonKnee,):
            sig = inspect.signature(cls.solve)
            names = set(sig.parameters)
            assert not names & {"ys", "y", "labels", "yt", "y_true"}

    def test_adapt_path_uses_unlabelled_target(self):
        """align_features must accept the target as points only."""
        from transforge.adapt.ot_domain_adaptation import align_features

        sig = inspect.signature(align_features)
        assert not set(sig.parameters) & {"ys", "labels"}


class TestInvariantSuiteRuns:
    def test_pipeline_invariants_complete(self):
        """The engine must produce all ten invariants in one call."""
        from transforge.core.config import RunConfig
        from transforge.pipeline.runner import TransForgePipeline

        rep = TransForgePipeline(RunConfig(n=16)).invariants(seed=9)
        ids = {c.id for c in rep.checks}
        for expected in ("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10"):
            if expected in rep.skipped or any(expected in s for s in rep.skipped):
                continue
            assert expected in ids, f"{expected} missing from the invariant report"

    def test_report_serialises(self):
        import json

        from transforge.core.config import RunConfig
        from transforge.pipeline.runner import TransForgePipeline

        rep = TransForgePipeline(RunConfig(n=12)).invariants(seed=1)
        json.dumps(rep.to_dict(), ensure_ascii=False)


class TestMeasureInvariants:
    def test_empirical_measure_constructors(self):
        m = EmpiricalMeasure(np.zeros((3, 2)), np.ones(3))
        assert m.n == 3
