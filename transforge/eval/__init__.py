"""Metrics, invariants and reporting.  Depends on ``core`` only."""

from __future__ import annotations

from .invariants import build_report, check_i2_marginals, default_problem
from .ot_metrics import (
    bias_ratio,
    bias_upper_bound,
    cost_relative_error,
    marginal_residual,
    optimality_gap,
    relative_error,
)
from .report import BenchmarkReport, significant, summarise

__all__ = [
    "BenchmarkReport",
    "bias_ratio",
    "bias_upper_bound",
    "build_report",
    "check_i2_marginals",
    "cost_relative_error",
    "default_problem",
    "marginal_residual",
    "optimality_gap",
    "relative_error",
    "significant",
    "summarise",
]
