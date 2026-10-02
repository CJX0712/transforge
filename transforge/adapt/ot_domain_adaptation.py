"""Domain adaptation / distribution alignment.

The flagship's contribution here is honest and narrow: it supplies the coupling.
Everything else here is the classical Courty et al. (2017) recipe, kept as the
baseline the flagship is measured against.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import numpy as np

from ..core.config import SinkhornConfig
from ..core.linalg import pairwise_sq_euclidean, safe_normalize
from ..core.types import FloatArray
from ..solvers.barycentric import barycentric_map
from ..solvers.epsilon_knee import TFEpsilonKnee


def align_features(
    Xs: FloatArray,
    Xt: FloatArray,
    *,
    a: FloatArray | None = None,
    b: FloatArray | None = None,
    cfg: SinkhornConfig | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Transport the source features onto the target distribution.

    Returns ``(Xs_aligned, Xt)``.  No labels are used anywhere on this path
    (invariant I10): the target enters only as an unlabelled point cloud.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    n, m = Xs.shape[0], Xt.shape[0]
    a = safe_normalize(np.ones(n)) if a is None else safe_normalize(a)
    b = safe_normalize(np.ones(m)) if b is None else safe_normalize(b)
    C = pairwise_sq_euclidean(Xs, Xt)
    res = TFEpsilonKnee(cfg or SinkhornConfig()).solve(a, b, C)
    return barycentric_map(res.plan, a, Xs, Xt), Xt


def source_only_baseline(Xs: FloatArray, Xt: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Do nothing (identity).  The floor every alignment must beat."""
    return np.asarray(Xs, dtype=np.float64), np.asarray(Xt, dtype=np.float64)


def whitened_align(Xs: FloatArray, Xt: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Mean/centre-align without transport -- a cheap, strong baseline.

    Included because it is the honest adversary: any transport method that cannot
    beat plain centring has not earned its complexity.
    """
    Xs = np.asarray(Xs, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    return Xs - np.mean(Xs, axis=0) + np.mean(Xt, axis=0), Xt


def colored_mnist_lite(
    n_per_class: int = 200,
    n_colors: int = 2,
    label_noise: float = 0.25,
    flip: float = 0.2,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """A small Colored-MNIST-style domain-shift problem (2 classes x 2 colours).

    Implements the protocol of Arjovsky et al. (2020) / Bao et al. (2022) at small
    scale: colour is a spurious feature whose correlation with the label differs
    between source and target.  Nothing is downloaded; everything is generated.
    """
    from ..core.seed import rng  # noqa: PLC0415

    generator = rng(seed)
    total = n_per_class * n_colors

    def build(spurious: float) -> tuple[np.ndarray, np.ndarray]:
        ys = np.repeat(np.arange(n_colors), n_per_class * n_colors // n_colors)
        ys = (
            ys[:total]
            if ys.size >= total
            else np.pad(ys, (0, total - ys.size), constant_values=ys[-1])
        )
        xs = generator.normal(0.0, 1.0, size=(total, 1 + n_colors))
        xs[:, 0] = ys + generator.normal(0.0, 0.3, size=total)
        for c in range(n_colors):
            colour = (generator.uniform(size=total) < spurious).astype(np.float64)
            xs[:, 1 + c] = colour * 2.5 + generator.normal(0.0, 0.25, size=total)
        y_noisy = ys.copy()
        flip_mask = generator.uniform(size=total) < label_noise
        y_noisy[flip_mask] = 1 - y_noisy[flip_mask]
        return xs, y_noisy

    Xs, ys = build(flip)
    Xt, yt_hidden = build(1.0 - flip)
    return {"Xs": Xs, "ys": ys, "Xt": Xt, "yt": yt_hidden}


def downstream_accuracy(
    X_train: FloatArray,
    y_train: FloatArray,
    X_test: FloatArray,
    y_test: FloatArray,
) -> float:
    """Logistic-regression accuracy with a pinned solver.

    ``solver='lbfgs'`` is required: scikit-learn 1.9 removed the ``multi_class``
    argument entirely, so passing it raises ``TypeError``.
    """
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    # scikit-learn >= 1.5 REMOVED the `multi_class` argument entirely, so passing
    # it raises TypeError; `solver="lbfgs"` is the only requirement for multiclass.
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(np.asarray(X_train, dtype=np.float64), np.asarray(y_train))
    return float(np.mean(clf.predict(np.asarray(X_test, dtype=np.float64)) == y_test))


__all__ = [
    "align_features",
    "colored_mnist_lite",
    "downstream_accuracy",
    "source_only_baseline",
    "whitened_align",
]
