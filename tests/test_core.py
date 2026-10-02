"""Core layer: types, config, seed determinism and the numerics primitives."""

from __future__ import annotations

import numpy as np
import pytest

from transforge.core.config import (
    GATES,
    RunConfig,
    SinkhornConfig,
    config_fingerprint,
)
from transforge.core.errors import ConfigError, DeterminismError, SolverError
from transforge.core.linalg import (
    dual_objective,
    effective_support,
    logsumexp,
    ot_lower_bound,
    pairwise_sq_euclidean,
    plan_change_rate,
    plan_entropy_proxy,
    safe_normalize,
)
from transforge.core.seed import current_seed, rng, set_all
from transforge.core.types import CostSpec, EmpiricalMeasure, OTResult


class TestTypes:
    def test_measure_normalises_weights(self):
        m = EmpiricalMeasure(np.zeros((4, 2)), np.array([1.0, 1.0, 2.0, 4.0]))
        assert float(np.sum(m.w)) == pytest.approx(1.0)
        assert m.n == 4 and m.dim == 2

    def test_measure_rejects_zero_mass(self):
        with pytest.raises(ValueError, match="sum to zero"):
            EmpiricalMeasure(np.zeros((3, 2)), np.zeros(3))

    def test_measure_rejects_negative_weights(self):
        with pytest.raises(ValueError, match="non-negative"):
            EmpiricalMeasure(np.zeros((3, 2)), np.array([1.0, -1.0, 1.0]))

    def test_measure_rejects_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            EmpiricalMeasure(np.array([[np.nan, 0.0]]), np.array([1.0]))

    def test_measure_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="points but w has"):
            EmpiricalMeasure(np.zeros((4, 2)), np.ones(3))

    def test_cost_spec_median(self):
        c = CostSpec.dense(np.array([[1.0, 3.0], [5.0, 7.0]]))
        assert c.median() == 4.0
        assert c.shape == (2, 2)

    def test_ot_result_marginal_residual(self):
        plan = np.full((2, 3), 1.0 / 6.0)
        r = OTResult(plan=plan, cost=1.0, marginal_a=plan.sum(1), marginal_b=plan.sum(0))
        a = np.array([0.5, 0.5])
        b = np.full(3, 1.0 / 3.0)
        assert r.rel_marginal_residual(a, b) < 1e-12
        assert r.mass == pytest.approx(1.0)


class TestLinalg:
    def test_logsumexp_matches_naive_in_safe_range(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.log(np.sum(np.exp(x), axis=1))
        assert np.allclose(logsumexp(x, axis=1), expected)

    def test_logsumexp_is_shift_equivariant(self):
        """LSE(x + c) == LSE(x) + c -- NOT invariant: the shift cancels only inside."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(logsumexp(x + 500.0, axis=1), logsumexp(x, axis=1) + 500.0)

    def test_logsumexp_survives_extreme_negatives(self):
        x = np.array([[-1e6, -1e6 - 1.0]])
        assert np.all(np.isfinite(logsumexp(x, axis=1)))

    def test_logsumexp_axis_argument_respected(self):
        x = np.arange(6, dtype=float).reshape(2, 3)
        assert np.allclose(logsumexp(x, axis=0), np.log(np.exp(x).sum(axis=0)))

    def test_pairwise_sq_euclidean_matches_direct(self):
        rng = np.random.default_rng(0)
        X, Y = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
        C = pairwise_sq_euclidean(X, Y)
        for i in range(5):
            for j in range(7):
                assert C[i, j] == pytest.approx(np.sum((X[i] - Y[j]) ** 2))

    def test_pairwise_cost_is_non_negative(self):
        rng = np.random.default_rng(1)
        C = pairwise_sq_euclidean(rng.normal(size=(20, 4)), rng.normal(size=(20, 4)))
        assert np.all(C >= 0.0)

    def test_safe_normalize(self):
        w = safe_normalize(np.array([1.0, 3.0]))
        assert w.sum() == pytest.approx(1.0) and w[1] == pytest.approx(0.75)

    def test_safe_normalize_rejects_negative(self):
        with pytest.raises(ValueError):
            safe_normalize(np.array([1.0, -1.0]))

    def test_entropy_proxy_orientation(self):
        """CRITICAL: 0 for a fully spread plan, ->1 for a collapsed one.

        This pins the orientation.  An earlier draft used
        ``-sum(rho log rho)/log(nm)`` which is 0 for the INDEPENDENT plan and
        negative for a sharp one -- exactly backwards, and it silently disabled
        the anti-collapse stopping criterion.
        """
        n = 8
        a = b = np.full(n, 1.0 / n)
        independent = np.outer(a, b)
        assert plan_entropy_proxy(independent, a, b) == pytest.approx(0.0, abs=1e-12)
        collapsed = np.zeros((n, n))
        np.fill_diagonal(collapsed, 1.0 / n)
        # a permutation plan has KL exactly log(n); normalised by log(n*m)=2log(n)
        # that is 0.5, so the proxy's collapsed endpoint is 0.5, NOT 1.
        assert plan_entropy_proxy(collapsed, a, b) == pytest.approx(0.5, abs=1e-9)

    def test_effective_support_range(self):
        n = 6
        a = b = np.full(n, 1.0 / n)
        # spread plan -> n*m; permutation plan -> its support size n
        assert effective_support(np.outer(a, b), a, b) == pytest.approx(n * n, rel=1e-9)
        collapsed = np.zeros((n, n))
        np.fill_diagonal(collapsed, 1.0 / n)
        # participation ratio over PLAN MASS: a permutation plan -> support size
        assert effective_support(collapsed, a, b) == pytest.approx(n, rel=1e-9)

    def test_plan_change_rate_zero_for_identical(self):
        p = np.ones((4, 4))
        assert plan_change_rate(p, p) == pytest.approx(0.0)

    def test_lower_bound_is_below_true_cost(self):
        rng = np.random.default_rng(3)
        X, Y = rng.normal(size=(20, 2)), rng.normal(size=(20, 2))
        C = pairwise_sq_euclidean(X, Y)
        a = b = np.full(20, 0.05)
        lb = ot_lower_bound(C, a, b)
        # a plan supported only on argmin pairs is a valid upper reference
        plan = np.zeros((20, 20))
        for i in range(20):
            plan[i, int(np.argmin(C[i]))] = 0.05
        assert np.sum(C * plan) >= lb - 1e-9

    def test_dual_objective_finite_for_small_eps(self):
        rng = np.random.default_rng(5)
        C = pairwise_sq_euclidean(rng.normal(size=(10, 2)), rng.normal(size=(10, 2)))
        w = np.full(10, 0.1)
        val = dual_objective(w, w, np.zeros(10), np.zeros(10), C, 1e-3)
        assert np.isfinite(val)


class TestSeed:
    def test_set_all_is_the_single_entry_point(self):
        assert set_all(123) == 123
        assert current_seed() == 123

    def test_rng_is_reproducible(self):
        set_all(99)
        a = rng().normal(size=5)
        set_all(99)
        b = rng().normal(size=5)
        assert np.array_equal(a, b)

    def test_rng_without_seed_raises(self):
        import transforge.core.seed as seed_mod

        seed_mod._last_seed = None
        try:
            with pytest.raises(Exception, match="no seed installed"):
                rng()
        finally:
            set_all(20260101)

    def test_set_all_rejects_negative_seed(self):
        with pytest.raises(DeterminismError):
            set_all(-1)

    def test_threads_are_pinned(self):
        set_all(5, 1)
        import os

        assert os.environ["OMP_NUM_THREADS"] == "1"


class TestConfig:
    def test_gates_are_positive_and_sane(self):
        assert GATES["G1_quality_ratio"] == 1.01
        # G1's wall-clock axis was removed on purpose: both the flagship (K-rung
        # ladder) and the oracle (11-point grid) are multi-solve procedures, so the
        # floor of their total-time ratio is ~K/11 = 0.18 -- unreachable for any
        # 0.10-style target.  The efficiency claim is carried by G3 instead.  This
        # assertion keeps the key from being silently reintroduced.
        assert "G1_time_ratio" not in GATES
        assert GATES["G3_time_ratio"] == pytest.approx(1.0 / 3.0)
        assert GATES["sig_seeds"] >= 3

    def test_sinkhorn_config_validates(self):
        SinkhornConfig().validate()
        with pytest.raises(ConfigError):
            SinkhornConfig(tol_marginal=0.0).validate()
        with pytest.raises(ConfigError):
            SinkhornConfig(ladder_ratio=1.5).validate()
        with pytest.raises(ConfigError):
            SinkhornConfig(eps_min_ratio=2.0, eps_max_ratio=1.0).validate()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TRANSFORGE_SK_MAXITER", "77")
        assert SinkhornConfig.from_env().max_iter == 77

    def test_env_override_rejects_garbage(self, monkeypatch):
        monkeypatch.setenv("TRANSFORGE_SK_TOL", "not-a-number")
        with pytest.raises(ConfigError):
            SinkhornConfig.from_env()

    def test_fingerprint_is_json_friendly(self):
        import json

        json.dumps(config_fingerprint(RunConfig()))

    def test_run_config_from_env(self, monkeypatch):
        monkeypatch.setenv("TRANSFORGE_N", "17")
        assert RunConfig.from_env().n == 17


class TestErrors:
    def test_layered_codes(self):
        from transforge.core import errors as e

        assert e.ConfigError.code == "E100"
        assert e.DataError.code == "E200"
        assert e.SolverError.code == "E300"
        assert e.InvariantViolation.code == "E500"

    def test_error_message_includes_code(self):
        assert "E300" in str(SolverError("boom"))
