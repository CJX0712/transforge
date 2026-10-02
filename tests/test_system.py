"""Data layer, adapt layer, architecture (no import cycles) and determinism."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from transforge.core.config import RunConfig, SinkhornConfig
from transforge.core.seed import set_all
from transforge.data.closed_form import gaussian_w2, w2_1d_equal
from transforge.data.loaders import to_measure
from transforge.data.synth import (
    build,
    gauss1d,
    gauss2d,
    one_dimensional_problem,
    outlier_problem,
    scale_family,
    sparse_weight_problem,
)
from transforge.pipeline.backends import capability_report, missing_backends
from transforge.pipeline.runner import TransForgePipeline

PKG = Path(__file__).resolve().parents[1] / "transforge"


class TestSynth:
    def test_scale_family_shapes_and_truth(self):
        p = scale_family(20, 3.0, seed=1)
        assert p.source.n == 20 and p.source.dim == 2
        assert p.w2_exact is not None and p.w2_exact > 0

    def test_scale_family_is_scale_equivariant(self):
        """Quadrupling the spread multiplies the SQUARED cost by 16, hence W2 by 4."""
        p1 = scale_family(20, 1.0, seed=2)
        p4 = scale_family(20, 4.0, seed=2)
        assert p4.w2_exact == pytest.approx(4.0 * p1.w2_exact, rel=1e-9)

    def test_gauss1d_truth_matches_closed_form(self):
        p = gauss1d(64, "mid", seed=4)
        assert p.w2_exact == pytest.approx(np.sqrt(1.5**2 + 0.5**2), rel=1e-9)

    def test_gauss2d_bures_truth_is_positive(self):
        p = gauss2d(32, "hard", seed=5)
        assert p.w2_exact > 0

    def test_gaussian_w2_of_identical_is_zero(self):
        """Eigendecomposition round-off limits this to ~1e-7 relative, not 1e-9."""
        S = np.array([[2.0, 0.3], [0.3, 1.0]])
        assert gaussian_w2(np.zeros(2), S, np.zeros(2), S) == pytest.approx(0.0, abs=1e-6)

    def test_gaussian_w2_translation_only(self):
        S = np.eye(2)
        d = gaussian_w2(np.zeros(2), S, np.array([3.0, 4.0]), S)
        assert d == pytest.approx(5.0, rel=1e-9)

    def test_w2_1d_rejects_size_mismatch(self):
        from transforge.core.errors import ClosedFormError

        with pytest.raises(ClosedFormError):
            w2_1d_equal(np.zeros(4), np.zeros(5))

    def test_one_dimensional_problem(self):
        p = one_dimensional_problem(n=32, seed=6)
        assert p.source.X.shape == (32, 1)
        assert p.w2_exact > 0 and p.w1_exact > 0

    def test_outlier_problem_counts_outliers(self):
        p = outlier_problem(40, outlier_fraction=0.2, seed=7)
        assert p.meta["n_outliers"] == 8

    def test_outlier_rejects_extreme_fraction(self):
        from transforge.core.errors import DataError

        with pytest.raises(DataError):
            outlier_problem(40, outlier_fraction=0.6, seed=1)

    def test_outliers_are_far_away(self):
        p = outlier_problem(40, outlier_fraction=0.25, outlier_radius=8.0, seed=8)
        radii = np.linalg.norm(p.target.X[:10], axis=1)
        assert np.all(radii > 5.0)

    def test_sparse_weights_span_a_wide_range(self):
        p = sparse_weight_problem(48, "hard", seed=9)
        assert np.min(p.source.w[p.source.w > 0]) < 1e-4

    def test_build_dispatches(self):
        assert build("gauss2d", 16, seed=1).family == "gauss2d"

    def test_build_rejects_unknown_family(self):
        from transforge.core.errors import DataError

        with pytest.raises(DataError):
            build("nope", 16, seed=1)

    def test_generation_is_reproducible(self):
        a = scale_family(16, 2.0, seed=77)
        b = scale_family(16, 2.0, seed=77)
        assert np.array_equal(a.source.X, b.source.X)

    def test_different_seeds_differ(self):
        a = scale_family(16, 2.0, seed=1)
        b = scale_family(16, 2.0, seed=2)
        assert not np.array_equal(a.source.X, b.source.X)


class TestLoaders:
    def test_to_measure_normalises(self):
        m = to_measure(np.zeros((3, 2)), np.array([1.0, 1.0, 2.0]))
        assert float(np.sum(m.w)) == pytest.approx(1.0)

    def test_missing_file_raises(self, tmp_path):
        from transforge.core.errors import DataError
        from transforge.data.loaders import load_npz

        with pytest.raises(DataError, match="not found"):
            load_npz(tmp_path / "absent.npz")


class TestAdapt:
    def test_align_features_preserves_shape(self):
        from transforge.adapt.ot_domain_adaptation import align_features

        set_all(5)
        Xs = np.random.default_rng(1).normal(size=(16, 2))
        Xt = np.random.default_rng(2).normal(size=(16, 2)) + 1.0
        aligned, target = align_features(Xs, Xt, cfg=SinkhornConfig(max_iter=200))
        assert aligned.shape == Xs.shape and target.shape == Xt.shape

    def test_source_only_baseline_is_identity(self):
        from transforge.adapt.ot_domain_adaptation import source_only_baseline

        Xs = np.ones((3, 2))
        Xt = np.zeros((3, 2))
        aligned, _ = source_only_baseline(Xs, Xt)
        assert np.array_equal(aligned, Xs)

    def test_coloured_mnist_lite_shapes(self):
        from transforge.adapt.ot_domain_adaptation import colored_mnist_lite

        data = colored_mnist_lite(n_per_class=40, seed=3)
        assert data["Xs"].shape == data["Xt"].shape
        assert set(np.unique(data["ys"])) <= {0, 1}

    def test_downstream_accuracy_is_a_probability(self):
        from transforge.adapt.ot_domain_adaptation import downstream_accuracy

        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        y = (X[:, 0] > 0).astype(int)
        acc = downstream_accuracy(X[:60], y[:60], X[60:], y[60:])
        assert 0.0 <= acc <= 1.0


class TestArchitecture:
    """The layering contract: core imports nothing from the project."""

    LAYERS = {"core", "data", "solvers", "eval", "adapt", "pipeline"}
    FORBIDDEN = {
        "core": set(),
        "data": {"solvers", "eval", "adapt", "pipeline"},
        "solvers": {"adapt", "eval", "pipeline"},
        "eval": {"solvers", "adapt", "pipeline"},
        "adapt": {"eval", "pipeline"},
        "pipeline": set(),
    }

    def _project_imports(self, path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("transforge")
            ):
                parts = node.module.split(".")
                if len(parts) > 1:
                    found.add(parts[1])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("transforge"):
                        parts = alias.name.split(".")
                        if len(parts) > 1:
                            found.add(parts[1])
        return found & self.LAYERS

    def test_no_layer_violations(self):
        violations: list[str] = []
        for layer, forbidden in self.FORBIDDEN.items():
            for py in (PKG / layer).rglob("*.py"):
                for imported in self._project_imports(py):
                    if imported in forbidden:
                        rel = py.relative_to(PKG.parent)
                        violations.append(f"{rel} imports {imported}")
        assert not violations, "layering violations:\n  " + "\n  ".join(violations)

    def test_core_is_import_clean(self):
        """core must import numpy and stdlib only -- no other project module."""
        for py in (PKG / "core").rglob("*.py"):
            assert not self._project_imports(py), f"{py} imports an upper layer"

    def test_every_module_imports(self):
        """Guards against a file that exists but fails at import time."""
        import importlib

        for layer in sorted(self.LAYERS):
            for py in sorted((PKG / layer).glob("*.py")):
                if py.stem == "__init__":
                    mod = f"transforge.{layer}"
                else:
                    mod = f"transforge.{layer}.{py.stem}"
                importlib.import_module(mod)

    def test_public_api_importable(self):
        import transforge
        from transforge import core, data, solvers  # noqa: F401

        assert transforge.__version__ == "0.1.0"


class TestBackends:
    def test_capability_report_is_json_serialisable(self):
        json.dumps(capability_report(), ensure_ascii=False)

    def test_numpy_is_always_available(self):
        assert capability_report()["numpy"]["available"] is True

    def test_missing_backends_is_a_list(self):
        assert isinstance(missing_backends(), list)

    def test_disable_pot_is_honoured(self, monkeypatch):
        from transforge.pipeline.backends import available_pot

        monkeypatch.setenv("TRANSFORGE_DISABLE_POT", "1")
        assert available_pot().available is False


class TestPipelineAndDeterminism:
    def test_benchmark_produces_a_report(self):
        cfg = RunConfig(seed=11, n=16, threads=1, sinkhorn=SinkhornConfig(max_iter=150))
        rep = TransForgePipeline(cfg).benchmark(seeds=[11], scales=[1.0], run_ablations=False)
        d = rep.to_dict()
        assert {"meta", "gates", "cells", "failures"} <= set(d)
        assert len(d["gates"]) == 4

    def test_report_roundtrips_through_json(self, tmp_path):
        cfg = RunConfig(seed=11, n=12, threads=1, sinkhorn=SinkhornConfig(max_iter=120))
        rep = TransForgePipeline(cfg).benchmark(seeds=[11], scales=[1.0], run_ablations=False)
        path = rep.save(tmp_path / "b.json")
        loaded = BenchmarkReportLoad(path)
        assert loaded["meta"]["seed"] == 11

    def test_benchmark_core_metrics_are_bitwise_reproducible(self):
        """Invariant I9: two independent runs must agree BITWISE on core metrics.

        Timings are excluded by construction (see ``core_metrics``); every
        relative error, selected epsilon and pass/fail flag must match exactly.
        """
        from transforge.eval.report import core_metrics

        cfg = RunConfig(seed=13, n=12, threads=1, sinkhorn=SinkhornConfig(max_iter=120))
        payloads = []
        for _ in range(2):
            set_all(13, 1)
            rep = TransForgePipeline(cfg).benchmark(seeds=[13], scales=[1.0], run_ablations=False)
            payloads.append(
                json.dumps(core_metrics(rep.to_dict()), sort_keys=True, ensure_ascii=False)
            )
        assert payloads[0] == payloads[1], "benchmark core metrics are not reproducible"

    def test_core_metrics_strips_every_timing_field(self):
        from transforge.eval.report import TIMING_FIELDS, core_metrics

        payload = {
            "a": 1,
            "wall_s": 2.0,
            "b": {"t_oracle": 3.0, "rel": 4.0},
            "c": [{"elapsed_s": 5.0, "rel_err": 6.0}],
            # Prefixed spellings must be stripped too: `meta` carries
            # `demo_elapsed_s`, and leaving it in made invariant I9 fail on
            # machine load alone rather than on a real reproducibility bug.
            "meta": {"demo_elapsed_s": 7.0, "demo_within_budget": True, "n": 24},
        }
        out = core_metrics(payload)
        flat = json.dumps(out)
        for field in TIMING_FIELDS:
            # Match the key as a JSON member, not a substring: `demo_elapsed_s`
            # contains `elapsed_s` but is a different key.  Substring matching
            # would pass for the wrong reason once prefixed keys are stripped.
            assert f'"{field}":' not in flat, f"{field} survived core_metrics()"
        assert '"demo_elapsed_s":' not in flat, "prefixed timing key survived core_metrics()"
        assert out["a"] == 1 and out["b"]["rel"] == 4.0 and out["c"][0]["rel_err"] == 6.0
        # non-timing siblings of a stripped key must survive
        assert out["meta"] == {"demo_within_budget": True, "n": 24}

    def test_core_metrics_is_stable_across_differing_timings(self):
        """Regression: two payloads differing ONLY in wall-clock must compare equal."""
        from transforge.eval.report import core_metrics

        base = {"gates": [{"id": "G1", "passed": True, "detail": "x"}], "meta": {"n": 24}}
        slow = {"gates": [{"id": "G1", "passed": True, "detail": "x"}], "meta": {"n": 24}}
        slow["meta"]["demo_elapsed_s"] = 999.0
        assert json.dumps(core_metrics(base), sort_keys=True) == json.dumps(
            core_metrics(slow), sort_keys=True
        )

    def test_gates_expose_numbers_not_booleans_only(self):
        cfg = RunConfig(seed=11, n=12, threads=1, sinkhorn=SinkhornConfig(max_iter=120))
        rep = TransForgePipeline(cfg).benchmark(seeds=[11], scales=[1.0], run_ablations=False)
        for g in rep.gates:
            assert g["id"] in {"G1", "G2", "G3", "G4"}
            assert "passed" in g and "threshold" in g


def BenchmarkReportLoad(path: Path) -> dict:
    from transforge.eval.report import BenchmarkReport

    return BenchmarkReport.load(path)


class TestCli:
    def test_info_command(self, capsys):
        from transforge.cli import main

        assert main(["info"]) == 0
        assert "capability matrix" in capsys.readouterr().out

    def test_invariant_command_non_strict(self, capsys):
        from transforge.cli import main

        code = main(["--n", "12", "invariant"])
        assert code in (0, 1)
        assert "invariant report" in capsys.readouterr().out.lower()

    def test_solve_command(self, capsys):
        from transforge.cli import main

        assert main(["--n", "16", "solve"]) == 0
        out = capsys.readouterr().out
        assert "chosen epsilon" in out

    def test_unknown_command_is_an_error(self):
        from transforge.cli import main

        with pytest.raises(SystemExit):
            main(["definitely-not-a-command"])
