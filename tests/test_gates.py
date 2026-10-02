"""Gate regression tests (G1-G4).

Thresholds are read from ``transforge.core.config.GATES`` -- the same object the
implementation and the docs use, so code, tests and documentation cannot drift.

These tests are deliberately FAST: they run a reduced sweep.  The authoritative
numbers come from ``examples/run_demo.py`` -> ``benchmark.json``.  What is asserted
here is that each gate's *machinery* produces a well-formed, correctly-signed
verdict, not that a laptop reproduces the published figures to the last digit.
"""

from __future__ import annotations

import copy

import pytest

from tests.conftest import needs_pot
from transforge.core.config import BENCH_SCALES, GATES, RunConfig, SinkhornConfig
from transforge.core.linalg import pairwise_sq_euclidean
from transforge.core.seed import set_all
from transforge.data.synth import outlier_problem, scale_family
from transforge.eval.report import significant, summarise
from transforge.pipeline.gates import (
    exact_reference,
    fixed_epsilon_baseline,
    gate_g1_efficiency,
    gate_g2_robustness,
    gate_g3_certificate,
    gate_g4_outlier,
    run_cell,
)
from transforge.solvers.epsilon_knee import TFEpsilonKnee

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def cells():
    """A reduced cross-scale sweep: 3 scales x 2 seeds."""
    cfg = RunConfig(seed=7, n=16, threads=1, sinkhorn=SinkhornConfig(max_iter=250))
    out = []
    for scale in (0.5, 1.0, 20.0):
        for seed in (7, 8):
            out.append(run_cell(scale, seed, cfg))
    return out


class TestGateThresholds:
    def test_gates_are_read_from_a_single_source(self):
        assert GATES["G1_quality_ratio"] == 1.01
        assert GATES["G2_min_scales"] == 3.0
        assert GATES["G3_time_ratio"] == pytest.approx(1.0 / 3.0)
        assert GATES["G4_rel_improvement"] == 0.15
        assert GATES["sig_seeds"] >= 3
        # G1 gates on quality only; the wall-clock axis was removed as structurally
        # unsatisfiable (floor ~K/11 = 0.18 vs a 0.10 target).
        assert "G1_time_ratio" not in GATES

    def test_benchmark_family_covers_four_orders_of_magnitude(self):
        assert min(BENCH_SCALES) <= 1.0 <= max(BENCH_SCALES)
        assert max(BENCH_SCALES) / min(BENCH_SCALES) >= 100.0


class TestGateMachinery:
    @needs_pot
    def test_g1_reports_both_axes(self, cells):
        """G1 gates on quality only, but still REPORTS the wall-clock ratio."""
        g = gate_g1_efficiency(cells)
        assert g["id"] == "G1"
        assert "time_ratio" in g and "measured_value" in g
        assert isinstance(g["passed"], bool)
        assert g["measured_value"] >= 0.0 and g["time_ratio"] >= 0.0

    @needs_pot
    def test_g1_gates_on_quality_only(self, cells):
        """The wall-clock axis must be present but explicitly NOT a gate.

        Regression guard for the ruling that removed ``G1_time_ratio``: the number
        is real and stays visible, but the verdict must not depend on it, because
        its structural floor (~K/11 = 0.18) sits above any meaningful target.
        """
        g = gate_g1_efficiency(cells)
        assert g["threshold"] == GATES["G1_quality_ratio"]
        assert g["time_ratio_is_gated"] is False
        assert "time_threshold" not in g
        assert "G1_time_ratio" not in GATES
        # A wildly slow flagship must not flip the verdict.  ``cells`` is a
        # module-scoped fixture, so mutate a COPY: scaling the shared objects in
        # place would leak a 10^6x cost into every later test that reads them.
        slow_cells = copy.deepcopy(cells)
        for cell in slow_cells:
            cell.t_flagship = cell.t_flagship * 1_000_000.0
        slow = gate_g1_efficiency(slow_cells)
        assert slow["passed"] == g["passed"]
        assert slow["time_ratio"] > g["time_ratio"]
        # the shared fixture must be untouched
        assert gate_g1_efficiency(cells)["time_ratio"] == g["time_ratio"]

    @needs_pot
    def test_g2_counts_scales_where_fixed_epsilon_fails(self, cells):
        g = gate_g2_robustness(cells)
        assert g["id"] == "G2"
        assert g["measured_value"] == float(len(g["degraded_scales"]))
        assert g["threshold"] == GATES["G2_min_scales"]
        for entry in g["per_scale"]:
            assert {"scale", "rel_fixed", "rel_flagship", "degraded"} <= set(entry)

    @needs_pot
    def test_g3_reports_a_time_ratio(self, cells):
        g = gate_g3_certificate(cells, RunConfig(n=16))
        assert g["id"] == "G3"
        assert g["time_ratio"] is None or g["time_ratio"] >= 0.0
        assert g["threshold"] == GATES["G3_time_ratio"]

    @needs_pot
    def test_g4_reports_mean_std_and_significance(self):
        cfg = RunConfig(seed=7, n=16, sinkhorn=SinkhornConfig(max_iter=250))
        g = gate_g4_outlier(cfg, [7, 8])
        assert g["id"] == "G4"
        for key in (
            "mean_rel_flagship",
            "std_rel_flagship",
            "mean_rel_sinkhorn",
            "std_rel_sinkhorn",
            "significant",
        ):
            assert key in g, f"missing {key}"
        assert g["threshold"] == GATES["G4_rel_improvement"]
        assert len(g["per_seed"]) == 2


class TestFixedEpsilonBaselineIsMeaningfullyBad:
    """The structural claim behind G2, asserted directly.

    A single hand-picked epsilon must be much worse than the flagship across
    scales; if this test ever fails, gate G2 is measuring nothing.
    """

    @needs_pot
    def test_fixed_epsilon_is_catastrophic_on_at_least_one_scale(self):
        # Use the REAL solver defaults: with a starved iteration budget neither
        # variant converges and the comparison stops meaning anything.
        cfg = RunConfig(seed=7, n=32, sinkhorn=SinkhornConfig())
        worse = 0
        for scale in (0.5, 2.0, 20.0):
            set_all(7)
            p = scale_family(cfg.n, scale, seed=7)
            a, b = p.source.w, p.target.w
            C = pairwise_sq_euclidean(p.source.X, p.target.X)
            exact = exact_reference(a, b, C)
            if exact is None:
                continue
            flag = TFEpsilonKnee(cfg.sinkhorn).solve(a, b, C)
            fixed = fixed_epsilon_baseline(a, b, C, cfg.sinkhorn)
            rf = abs(flag.cost - exact) / exact
            rx = abs(fixed.cost - exact) / exact
            if rx > GATES["G2_degrade_rel"] and rf <= rx:
                worse += 1
        assert worse >= 1, "no scale on which fixed-epsilon degrades: G2 is vacuous"


class TestOutlierMechanismIsAttributable:
    @needs_pot
    def test_uot_actually_destroys_mass_when_there_are_outliers(self):
        """G4's improvement must come from the KL relaxation, not from luck."""
        from transforge.solvers.unbalanced import TFUFlow

        cfg = SinkhornConfig(max_iter=400)
        set_all(3)
        p = outlier_problem(24, outlier_fraction=0.2, seed=3)
        a, b = p.source.w, p.target.w
        C = pairwise_sq_euclidean(p.source.X, p.target.X)
        res = TFUFlow(cfg).solve(a, b, C)
        assert res.meta["mass_destroyed"] > 0.0
        assert res.mass <= 1.0 + 1e-9


class TestSignificanceRule:
    def test_rule_is_difference_greater_than_half_sum_of_stds(self):
        assert significant(1.0, 0.01, 1.05, 0.01) is True
        assert significant(1.0, 1.0, 1.05, 1.0) is False

    def test_summarise_reports_mean_and_std(self):
        s = summarise([1.0, 2.0, 3.0])
        assert s["mean"] == pytest.approx(2.0)
        assert s["n"] == 3
        assert s["min"] == 1.0 and s["max"] == 3.0

    def test_summarise_handles_empty(self):
        assert summarise([])["n"] == 0
