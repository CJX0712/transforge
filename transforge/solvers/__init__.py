"""Transport solvers: exact reference, entropic solvers, three flagship algorithms.

Depends on ``core`` and ``data`` only.
"""

from __future__ import annotations

from .barycentric import barycentric_map, transport_features
from .emd import available_pot, emd2_1d_reference, emd_plan, small_scale_lp_reference
from .epsilon_knee import TFEpsilonKnee
from .self_paired_richardson import TFSelfPairedRichardson
from .sinkhorn import sinkhorn_divergence, sinkhorn_log, sinkhorn_naive
from .unbalanced import TFUFlow, unbalanced_reference

__all__ = [
    "TFEpsilonKnee",
    "TFSelfPairedRichardson",
    "TFUFlow",
    "available_pot",
    "barycentric_map",
    "emd2_1d_reference",
    "emd_plan",
    "sinkhorn_divergence",
    "sinkhorn_log",
    "sinkhorn_naive",
    "small_scale_lp_reference",
    "transport_features",
    "unbalanced_reference",
]
