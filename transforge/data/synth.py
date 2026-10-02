"""Synthetic data-generating processes (DGP).

Every family returns ``(source, target)`` :class:`EmpiricalMeasure` pairs plus the
analytic ground truth where one exists.  All randomness comes from
``np.random.default_rng`` seeded through :func:`transforge.core.seed.rng`, so a
given ``(family, difficulty, seed)`` always yields bit-identical data.

Difficulty knobs follow MATH_KERNEL section 7: each family exposes three
settings and the knob values live here, in version control, not in prose.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ..core.errors import DataError
from ..core.linalg import safe_normalize
from ..core.types import EmpiricalMeasure, FloatArray
from .closed_form import gaussian_w2, w2_1d_equal, wasserstein1_1d

Difficulty = Literal["easy", "mid", "hard"]


@dataclass(frozen=True)
class Problem:
    """A benchmark instance: two measures plus the analytic truth if available."""

    source: EmpiricalMeasure
    target: EmpiricalMeasure
    name: str
    family: str
    difficulty: str
    w2_exact: float | None = None
    w1_exact: float | None = None
    meta: dict[str, Any] | None = None

    @property
    def has_truth(self) -> bool:
        return self.w2_exact is not None or self.w1_exact is not None


# --------------------------------------------------------------------------
# knob tables (MATH_KERNEL 7)
# --------------------------------------------------------------------------
_GAUSS1D_KNOBS: dict[str, dict[str, float]] = {
    "easy": {"mean_shift": 0.5, "std_ratio": 1.0},
    "mid": {"mean_shift": 1.5, "std_ratio": 1.5},
    "hard": {"mean_shift": 3.0, "std_ratio": 3.0},
}

_GAUSS2D_KNOBS: dict[str, dict[str, float]] = {
    "easy": {"theta": 0.2617993877991494, "aniso": 1.2, "scale": 1.1},
    "mid": {"theta": 0.5235987755982988, "aniso": 2.0, "scale": 1.4},
    "hard": {"theta": 1.5707963267948966, "aniso": 4.0, "scale": 2.0},
}

_OUTLIER_KNOBS: dict[str, dict[str, float]] = {
    "easy": {"outlier_fraction": 0.01, "outlier_radius": 3.0},
    "mid": {"outlier_fraction": 0.10, "outlier_radius": 5.0},
    "hard": {"outlier_fraction": 0.30, "outlier_radius": 10.0},
}

_SPARSE_KNOBS: dict[str, dict[str, float]] = {
    "easy": {"dyn": 3.0, "zero_frac": 0.0},
    "mid": {"dyn": 8.0, "zero_frac": 0.1},
    "hard": {"dyn": 16.0, "zero_frac": 0.3},
}


def _uniform(n: int) -> FloatArray:
    return np.full(n, 1.0 / n, dtype=np.float64)


def gauss1d(
    n: int, difficulty: Difficulty = "mid", seed: int | None = None, scale: float = 1.0
) -> Problem:
    """Family 1: 1-D Gaussians with mean/variance drift.

    ``scale`` multiplies the whole cloud, which is exactly what makes it the
    cross-scale benchmark family used by gates G1/G2/G3: the optimal regularisation
    strength moves with the data scale but a fixed epsilon cannot follow.
    """
    generator = _generator(seed)
    knobs = _GAUSS1D_KNOBS[difficulty]
    sigma_x = 1.0
    sigma_y = knobs["std_ratio"]
    mu_x = 0.0
    mu_y = knobs["mean_shift"] * sigma_x
    x = generator.normal(mu_x, sigma_x, size=n) * scale
    y = generator.normal(mu_y, sigma_y, size=n) * scale
    w2 = float(np.sqrt((mu_y - mu_x) ** 2 + (sigma_y - sigma_x) ** 2) * scale)
    return Problem(
        source=EmpiricalMeasure(x.reshape(-1, 1), _uniform(n), "X"),
        target=EmpiricalMeasure(y.reshape(-1, 1), _uniform(n), "Y"),
        name=f"gauss1d-{difficulty}-s{scale:g}",
        family="gauss1d",
        difficulty=difficulty,
        w2_exact=w2,
        w1_exact=abs(mu_y - mu_x) * scale,
        meta={"sigma_x": sigma_x, "sigma_y": sigma_y, "scale": scale},
    )


def gauss2d(
    n: int, difficulty: Difficulty = "mid", seed: int | None = None, scale: float = 1.0
) -> Problem:
    """Family 2: 2-D Gaussians with covariance rotation and anisotropic scaling.

    The Gaussian-to-Gaussian Brenier map is affine, so the Bures closed form is an
    exact ground truth.
    """
    generator = _generator(seed)
    knobs = _GAUSS2D_KNOBS[difficulty]
    theta = knobs["theta"]
    aniso = knobs["aniso"]
    s = knobs["scale"] * scale
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    cov_x = np.diag([1.0, 1.0 / aniso])
    cov_y = rot @ np.diag([s, s / aniso]) @ rot.T
    x = generator.multivariate_normal(np.zeros(2), cov_x, size=n)
    y = generator.multivariate_normal(np.zeros(2), cov_y, size=n)
    return Problem(
        source=EmpiricalMeasure(x, _uniform(n), "X"),
        target=EmpiricalMeasure(y, _uniform(n), "Y"),
        name=f"gauss2d-{difficulty}-s{scale:g}",
        family="gauss2d",
        difficulty=difficulty,
        w2_exact=gaussian_w2(np.zeros(2), cov_x, np.zeros(2), cov_y),
        meta={"cov_x": cov_x.tolist(), "cov_y": cov_y.tolist(), "scale": scale},
    )


def scale_family(n: int, scale: float, seed: int | None = None, family: str = "gauss2d") -> Problem:
    """The cross-scale benchmark family driving gates G1/G2/G3.

    ``family='gauss2d'`` gives a rotated anisotropic covariance; the analytic
    truth is the Bures distance between the two generating covariances.
    """
    generator = _generator(seed)
    if family == "gauss2d":
        theta = 0.5235987755982988
        aniso = 2.0
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        cov_x = np.diag([1.0, 1.0 / aniso]) * scale**2
        cov_y = rot @ np.diag([1.4, 1.4 / aniso]) @ rot.T * scale**2
        x = generator.multivariate_normal(np.zeros(2), cov_x, size=n)
        y = generator.multivariate_normal(np.zeros(2), cov_y, size=n)
        return Problem(
            source=EmpiricalMeasure(x, _uniform(n), "X"),
            target=EmpiricalMeasure(y, _uniform(n), "Y"),
            name=f"scale-{family}-{scale:g}",
            family=family,
            difficulty="scale",
            w2_exact=gaussian_w2(np.zeros(2), cov_x, np.zeros(2), cov_y),
            meta={"scale": scale, "family": family},
        )
    if family == "gauss1d":
        cov_x = np.array([[scale**2]])
        cov_y = np.array([[(1.5 * scale) ** 2]])
        x = generator.normal(0.0, scale, size=n).reshape(-1, 1)
        y = generator.normal(0.5 * scale, 1.5 * scale, size=n).reshape(-1, 1)
        return Problem(
            source=EmpiricalMeasure(x, _uniform(n), "X"),
            target=EmpiricalMeasure(y, _uniform(n), "Y"),
            name=f"scale-{family}-{scale:g}",
            family=family,
            difficulty="scale",
            w2_exact=gaussian_w2(np.zeros(1), cov_x, np.zeros(1), cov_y),
            meta={"scale": scale, "family": family},
        )
    raise DataError(f"unknown scale family {family!r}")


def outlier_problem(
    n: int,
    outlier_fraction: float = 0.20,
    outlier_radius: float = 5.0,
    seed: int | None = None,
    scale: float = 1.0,
) -> Problem:
    """Family 4: outlier contamination -- the battleground for gate G4.

    ``outlier_fraction`` of the *target* points are pushed to radius
    ``outlier_radius`` (in units of sigma).  Balanced OT must move that mass no
    matter how expensive the trip; a KL-relaxed plan can destroy it instead.
    """
    generator = _generator(seed)
    if not 0.0 <= outlier_fraction < 0.5:
        raise DataError(f"outlier_fraction must lie in [0, 0.5), got {outlier_fraction}")
    x = generator.normal(0.0, 1.0, size=(n, 2)) * scale
    y = generator.normal(0.0, 1.0, size=(n, 2)) * scale
    k = int(round(outlier_fraction * n))
    if k > 0:
        angles = generator.uniform(0.0, 2.0 * np.pi, size=k)
        radius = outlier_radius * scale
        y[:k] = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)
    return Problem(
        source=EmpiricalMeasure(x, _uniform(n), "X"),
        target=EmpiricalMeasure(y, _uniform(n), "Y"),
        name=f"outlier-{outlier_fraction:g}",
        family="outlier",
        difficulty="outlier",
        w2_exact=None,  # contaminated cloud has no clean closed form
        meta={
            "outlier_fraction": outlier_fraction,
            "outlier_radius": outlier_radius,
            "scale": scale,
            "n_outliers": k,
        },
    )


def sparse_weight_problem(
    n: int, difficulty: Difficulty = "mid", seed: int | None = None
) -> Problem:
    """Family 5: log-uniform sparse weights.

    ``dyn = log(max/min)`` widens the weight dynamic range, which inflates the
    bias certificate ``eps * log(1/(min a * min b))`` and forces small epsilon --
    the family where a naive-domain implementation is mathematically doomed.
    """
    generator = _generator(seed)
    knobs = _SPARSE_KNOBS[difficulty]
    x = generator.normal(0.0, 1.0, size=(n, 2))
    y = generator.normal(0.8, 1.3, size=(n, 2))
    a = _log_uniform_weights(generator, n, knobs["dyn"], knobs["zero_frac"])
    b = _log_uniform_weights(generator, n, knobs["dyn"], knobs["zero_frac"])
    return Problem(
        source=EmpiricalMeasure(x, a, "X"),
        target=EmpiricalMeasure(y, b, "Y"),
        name=f"sparse-{difficulty}",
        family="sparse",
        difficulty=difficulty,
        w2_exact=None,
        meta={"dyn": knobs["dyn"], "zero_frac": knobs["zero_frac"]},
    )


def unbalance_problem(n: int, mass_mismatch: float = 0.3, seed: int | None = None) -> Problem:
    """Family for UOT: target mass deliberately under-supplied.

    Balanced OT has no choice but to inflate the cost of manufacturing mass;
    an unbalanced plan simply drops it.
    """
    generator = _generator(seed)
    if not 0.0 <= mass_mismatch < 0.9:
        raise DataError(f"mass_mismatch must lie in [0, 0.9), got {mass_mismatch}")
    x = generator.normal(0.0, 1.0, size=(n, 2))
    y = generator.normal(0.4, 1.0, size=(n, 2))
    b = safe_normalize(_uniform(n) * (1.0 - mass_mismatch * generator.uniform(0.5, 1.0, n)))
    return Problem(
        source=EmpiricalMeasure(x, _uniform(n), "X"),
        target=EmpiricalMeasure(y, b, "Y"),
        name=f"unbalance-{mass_mismatch:g}",
        family="unbalance",
        difficulty="unbalance",
        w2_exact=None,
        meta={"mass_mismatch": mass_mismatch},
    )


def _log_uniform_weights(
    generator: NDArray[np.float64], n: int, dyn: float, zero_frac: float
) -> FloatArray:
    """Weights spanning a log-uniform dynamic range ``dyn``, with some exact zeros."""
    raw = np.exp(generator.uniform(0.0, dyn, size=n))
    if zero_frac > 0.0:
        mask = generator.uniform(size=n) < zero_frac
        raw[mask] = 0.0
    if not np.any(raw > 0.0):
        raw[0] = 1.0
    return safe_normalize(raw)


def _generator(seed: int | None):
    from ..core.seed import rng

    return rng(seed)


FAMILIES: dict[str, Any] = {
    "gauss1d": gauss1d,
    "gauss2d": gauss2d,
    "scale": scale_family,
    "outlier": outlier_problem,
    "sparse": sparse_weight_problem,
    "unbalance": unbalance_problem,
}


def build(family: str, n: int, seed: int | None = None, **kwargs: Any) -> Problem:
    """Factory dispatching on family name."""
    if family not in FAMILIES:
        raise DataError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[family](n=n, seed=seed, **kwargs)


def one_dimensional_problem(n: int = 64, seed: int | None = None) -> Problem:
    """A 1-D instance used by invariant I4 (exact closed-form cross-check)."""
    generator = _generator(seed)
    x = generator.normal(0.0, 1.0, size=n)
    y = generator.normal(1.0, 1.6, size=n)
    return Problem(
        source=EmpiricalMeasure(x.reshape(-1, 1), _uniform(n), "X"),
        target=EmpiricalMeasure(y.reshape(-1, 1), _uniform(n), "Y"),
        name="oned-1d",
        family="gauss1d",
        difficulty="easy",
        w2_exact=w2_1d_equal(x, y),
        w1_exact=wasserstein1_1d(x, y, p=1),
        meta={},
    )


__all__ = [
    "FAMILIES",
    "Difficulty",
    "Problem",
    "build",
    "gauss1d",
    "gauss2d",
    "one_dimensional_problem",
    "outlier_problem",
    "scale_family",
    "sparse_weight_problem",
    "unbalance_problem",
]
