"""Layered error codes E1xx..E5xx.

E500 is the signature error of this project: it does not mean "the program
crashed", it means "a conclusion was falsified".
"""

from __future__ import annotations


class TransForgeError(Exception):
    """Base class for every error raised by TransForge."""

    code = "E000"

    def __init__(self, message: str, code: str | None = None) -> None:
        self.code = code or self.code
        super().__init__(f"[{self.code}] {message}")


class ConfigError(TransForgeError):
    """E100: invalid configuration or environment override."""

    code = "E100"


class DeterminismError(TransForgeError):
    """E110: conflicting determinism settings (e.g. seed with thread races)."""

    code = "E110"


class DataError(TransForgeError):
    """E200: malformed empirical measure (weights, shapes, non-finite values)."""

    code = "E200"


class ClosedFormError(TransForgeError):
    """E210: a closed-form ground truth does not apply to this distribution."""

    code = "E210"


class SolverError(TransForgeError):
    """E300: solver did not converge / residual above tolerance."""

    code = "E300"


class NumericError(TransForgeError):
    """E310: numerical breakdown (overflow / NaN in the log domain)."""

    code = "E310"


class LadderError(TransForgeError):
    """E320: the epsilon ladder collapsed or degenerated."""

    code = "E320"


class InfeasibleError(TransForgeError):
    """E330: mathematically infeasible parameter combination."""

    code = "E330"


class AdapterError(TransForgeError):
    """E400: domain-adaptation input mismatch."""

    code = "E400"


class InvariantViolation(TransForgeError):
    """E500: a verifiable invariant was falsified.

    This is the intended "loud failure" of TransForge: the CLI prints the full
    invariant table and exits non-zero rather than reporting a plausible number.
    """

    code = "E500"

    def __init__(self, message: str, report: object | None = None) -> None:
        super().__init__(message)
        self.report = report


__all__ = [
    "AdapterError",
    "ClosedFormError",
    "ConfigError",
    "DataError",
    "DeterminismError",
    "InfeasibleError",
    "InvariantViolation",
    "LadderError",
    "NumericError",
    "SolverError",
    "TransForgeError",
]
