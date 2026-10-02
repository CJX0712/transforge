"""Zero-dependency core layer: types, errors, config, interfaces, seed, linalg.

``core`` must never import another project module.  Enforced by
``tests/test_architecture.py``.
"""

from __future__ import annotations

from .config import GATES, RunConfig, SinkhornConfig
from .errors import (
    AdapterError,
    ClosedFormError,
    ConfigError,
    DataError,
    DeterminismError,
    InfeasibleError,
    InvariantViolation,
    LadderError,
    NumericError,
    SolverError,
    TransForgeError,
)
from .seed import rng, set_all
from .types import CostSpec, EmpiricalMeasure, InvariantCheck, InvariantReport, OTResult

__all__ = [
    "GATES",
    "AdapterError",
    "ClosedFormError",
    "ConfigError",
    "CostSpec",
    "DataError",
    "DeterminismError",
    "EmpiricalMeasure",
    "InfeasibleError",
    "InvariantCheck",
    "InvariantReport",
    "InvariantViolation",
    "LadderError",
    "NumericError",
    "OTResult",
    "RunConfig",
    "SinkhornConfig",
    "SolverError",
    "TransForgeError",
    "rng",
    "set_all",
]
