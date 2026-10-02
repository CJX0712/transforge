"""Structural protocols.  These document the contract; ``core`` imports nothing
from upper layers, so they use ``typing.Protocol`` rather than ABCs.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from .types import CostSpec, EmpiricalMeasure, FloatArray, OTResult

Direction = Literal["minimize", "maximize"]


@runtime_checkable
class Solver(Protocol):
    """Anything that turns a cost matrix plus two marginals into an :class:`OTResult`."""

    name: str
    requires_backend: str

    def solve(
        self,
        a: FloatArray,
        b: FloatArray,
        C: CostSpec,
        **kwargs: Any,
    ) -> OTResult:
        """Solve the transport problem.

        Must return a feasible-result object; on failure it raises (never returns
        ``None`` and never silently degrades).
        """
        ...


@runtime_checkable
class Mapper(Protocol):
    """A fitted transport map."""

    def fit(
        self,
        Xs: FloatArray,
        Xt: FloatArray,
        a: FloatArray | None = None,
        b: FloatArray | None = None,
    ) -> Mapper: ...

    def transform(self, X: FloatArray) -> FloatArray: ...


@runtime_checkable
class Adapter(Protocol):
    """A downstream domain-adaptation strategy."""

    name: str

    def align(self, Xs: FloatArray, Xt: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return ``(Xs_aligned, Xt)``, both moved into a common space."""
        ...


@runtime_checkable
class Metric(Protocol):
    """A scalar quality measure.  ``direction`` fixes which way is better."""

    name: str
    direction: Direction

    def __call__(self, result: OTResult, truth: EmpiricalMeasure | None = None) -> float: ...


@runtime_checkable
class BackendProbe(Protocol):
    """Availability probe for an optional dependency."""

    name: str

    def available(self) -> bool: ...

    def version(self) -> str | None: ...


def as_float_array(x: Any) -> FloatArray:
    """Coerce to a C-contiguous float64 array (cheap, no copy when possible)."""
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


__all__ = [
    "Adapter",
    "BackendProbe",
    "CostSpec",
    "Direction",
    "Mapper",
    "Metric",
    "Solver",
    "as_float_array",
]
