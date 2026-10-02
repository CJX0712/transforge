"""Synthetic data + analytic ground truths.

Depends on ``core`` only.  Never imports ``solvers``/``eval``/``pipeline``.
"""

from __future__ import annotations

from .closed_form import gaussian_w2, w2_1d_equal, wasserstein1_1d
from .synth import FAMILIES, Problem, build, gauss1d, gauss2d, outlier_problem, scale_family

__all__ = [
    "FAMILIES",
    "Problem",
    "build",
    "gauss1d",
    "gauss2d",
    "gaussian_w2",
    "outlier_problem",
    "scale_family",
    "w2_1d_equal",
    "wasserstein1_1d",
]
