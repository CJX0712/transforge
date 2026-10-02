"""Shared fixtures.  Seeds are pinned here so every test is reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from transforge.core.config import RunConfig, SinkhornConfig
from transforge.core.linalg import pairwise_sq_euclidean
from transforge.core.seed import set_all
from transforge.data.synth import outlier_problem, scale_family

TEST_SEED = 20260101


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    """Every test starts from a known seed with BLAS pinned to one thread."""
    set_all(TEST_SEED, 1)


@pytest.fixture
def small_problem():
    """A 24-point cross-scale instance: cheap enough for unit tests."""
    p = scale_family(24, 1.0, seed=TEST_SEED)
    a, b = p.source.w, p.target.w
    C = pairwise_sq_euclidean(p.source.X, p.target.X)
    return p, a, b, C


@pytest.fixture
def uniform_2d():
    """A 2-D instance used by the symmetry / self-distance invariants.

    ``C`` is the SELF cost matrix of ``X`` so that ``W(X, X) = 0`` is meaningful;
    for the transpose-symmetry test the caller transposes ``C`` explicitly.
    """
    rng = np.random.default_rng(7)
    X = rng.normal(size=(16, 2))
    w = np.full(16, 1.0 / 16)
    return w, w, pairwise_sq_euclidean(X, X), X, X.copy()


@pytest.fixture
def fast_cfg() -> SinkhornConfig:
    """A deliberately small budget so the suite stays quick."""
    return SinkhornConfig(max_iter=300, ladder_levels=2, ladder_ratio=0.316)


@pytest.fixture
def run_cfg() -> RunConfig:
    return RunConfig(seed=TEST_SEED, n=24, threads=1)


@pytest.fixture
def outlier_instance():
    p = outlier_problem(24, outlier_fraction=0.20, seed=TEST_SEED)
    a, b = p.source.w, p.target.w
    C = pairwise_sq_euclidean(p.source.X, p.target.X)
    return p, a, b, C


def has_pot() -> bool:
    from transforge.solvers.emd import available_pot

    return available_pot()


needs_pot = pytest.mark.skipif(not has_pot(), reason="POT (exact EMD reference) unavailable")
