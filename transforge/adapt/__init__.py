"""Domain adaptation / distribution alignment layer."""

from __future__ import annotations

from .ot_domain_adaptation import (
    align_features,
    colored_mnist_lite,
    downstream_accuracy,
    source_only_baseline,
    whitened_align,
)

__all__ = [
    "align_features",
    "colored_mnist_lite",
    "downstream_accuracy",
    "source_only_baseline",
    "whitened_align",
]
