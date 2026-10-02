"""Core data types shared by every layer.

Interface semantics (uniform across all solvers -- this is the basis of fair
benchmarking):

* every cost/distance/error field is **larger is worse**;
* every accuracy/score field is **larger is better**;
* ``residual`` is always a **relative** L-inf residual;
* every solver returns an :class:`OTResult`;
* a solver never returns ``None`` -- it raises on failure.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EmpiricalMeasure:
    """A finite weighted point cloud ``sum w = 1``, ``w >= 0``."""

    X: FloatArray
    w: FloatArray
    name: str = ""

    def __post_init__(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        w = np.asarray(self.w, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n, d), got shape {X.shape}")
        if w.ndim != 1:
            raise ValueError(f"w must be 1-D (n,), got shape {w.shape}")
        if X.shape[0] != w.shape[0]:
            raise ValueError(f"X has {X.shape[0]} points but w has {w.shape[0]} entries")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(w)):
            raise ValueError("measure contains non-finite values")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if float(np.sum(w)) <= 0.0:
            raise ValueError("weights sum to zero")
        object.__setattr__(self, "X", X)
        object.__setattr__(self, "w", w / float(np.sum(w)))

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def dim(self) -> int:
        return int(self.X.shape[1])


@dataclass(frozen=True)
class CostSpec:
    """A ground cost, either materialised or lazily generated."""

    M: FloatArray | None = None
    kind: str = "dense"
    note: str = ""

    @staticmethod
    def dense(M: FloatArray) -> CostSpec:
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2:
            raise ValueError(f"cost matrix must be 2-D, got {M.shape}")
        return CostSpec(M=M, kind="dense")

    @property
    def shape(self) -> tuple[int, int]:
        if self.M is None:
            raise ValueError("lazy CostSpec has no materialised matrix")
        return self.M.shape  # type: ignore[union-attr]

    def median(self) -> float:
        """Median ground cost -- the scale that every epsilon is expressed against."""
        if self.M is None:
            raise ValueError("lazy CostSpec has no materialised matrix")
        return float(np.median(self.M))


@dataclass
class OTResult:
    """Uniform solver output. ``cost`` is the transport cost ``<C, pi>``.

    Attributes
    ----------
    plan:
        ``(n, m)`` non-negative coupling.
    cost:
        ``<C, plan>``.  Smaller is better.  For entropic solvers this is the
        *transport* term only; the entropy term is reported separately in
        ``meta['kl']`` so that values stay comparable with exact EMD.
    marginal_a, marginal_b:
        Realised row/column sums.  Equal to the prescribed weights only for
        balanced problems.
    n_iter, converged, epsilon:
        Bookkeeping.  ``epsilon`` is ``None`` for solvers without a
        regularisation strength (exact EMD).
    """

    plan: FloatArray
    cost: float
    marginal_a: FloatArray
    marginal_b: FloatArray
    n_iter: int = 0
    converged: bool = True
    epsilon: float | None = None
    residual: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def mass(self) -> float:
        """Total transported mass.  ``<= 1`` for KL-relaxed (unbalanced) plans."""
        return float(np.sum(self.plan))

    def transport_cost(self, C: FloatArray) -> float:
        return float(np.sum(np.asarray(C, dtype=np.float64) * self.plan))

    def rel_marginal_residual(self, a: FloatArray, b: FloatArray) -> float:
        """Max relative L-inf marginal error against the prescribed weights."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        scale = max(float(np.max(a)), float(np.max(b)), 1e-300)
        return float(
            max(np.max(np.abs(self.marginal_a - a)), np.max(np.abs(self.marginal_b - b))) / scale
        )


@dataclass(frozen=True)
class InvariantCheck:
    """One machine-checked property. ``passed`` is a hard boolean, never a vibe."""

    id: str
    passed: bool
    value: float
    tolerance: float
    detail: str = ""


@dataclass
class InvariantReport:
    """Outcome of an invariant sweep."""

    checks: list[InvariantCheck] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """True when nothing FAILED.  Degraded checks pass but are listed separately."""
        return all(c.passed for c in self.checks)

    def degraded_checks(self) -> list[InvariantCheck]:
        """Checks that pass only within a documented numerical floor."""
        return [c for c in self.checks if c.passed and c.detail.startswith("DEGRADED")]

    @property
    def n_degraded(self) -> int:
        return len(self.degraded_checks())

    @property
    def worst_violation(self) -> float:
        """Largest ratio ``value / tolerance`` over failing checks (0.0 if all pass)."""
        ratios = [c.value / c.tolerance for c in self.checks if not c.passed and c.tolerance > 0]
        return float(max(ratios)) if ratios else 0.0

    @property
    def failures(self) -> list[InvariantCheck]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "n_checks": len(self.checks),
            "n_degraded": self.n_degraded,
            "degraded": [c.id for c in self.degraded_checks()],
            "worst_violation": self.worst_violation,
            "skipped": list(self.skipped),
            "checks": [
                {
                    "id": c.id,
                    "passed": c.passed,
                    "value": float(c.value),
                    "tolerance": float(c.tolerance),
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


@dataclass(frozen=True)
class TimingRecord:
    """Wall-clock bookkeeping.  ``warmup_excluded`` guards against import cost."""

    wall_s: float
    n_iter: int = 0
    warmup_excluded: bool = True
    threads: int = 1


__all__ = [
    "CostSpec",
    "EmpiricalMeasure",
    "FloatArray",
    "InvariantCheck",
    "InvariantReport",
    "OTResult",
    "TimingRecord",
]
