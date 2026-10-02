"""Local file loaders.  No network access, ever.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.errors import DataError
from ..core.linalg import safe_normalize
from ..core.types import EmpiricalMeasure


def load_npz(path: str | Path, x_key: str = "X", w_key: str = "w") -> tuple:
    """Load ``(X, w)`` from a ``.npz`` archive holding ``X`` and optional ``w``."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"file not found: {p}")
    with np.load(p, allow_pickle=False) as data:
        if x_key not in data:
            raise DataError(f"{p} has no array named {x_key!r}; found {list(data)}")
        X = np.asarray(data[x_key], dtype=np.float64)
        w = np.asarray(data[w_key], dtype=np.float64) if w_key in data else np.ones(X.shape[0])
    return X, safe_normalize(w)


def load_csv(path: str | Path, delimiter: str = ",") -> tuple:
    """Load ``(X, w)`` from a delimited text file; the last column may be the weight."""
    p = Path(path)
    if not p.exists():
        raise DataError(f"file not found: {p}")
    arr = np.loadtxt(p, delimiter=delimiter, ndmin=2)
    if arr.shape[1] < 2:
        raise DataError(f"{p} needs at least 2 columns (features, [weight])")
    return arr[:, :-1].astype(np.float64), safe_normalize(arr[:, -1].astype(np.float64))


def to_measure(X, w=None, name: str = "") -> EmpiricalMeasure:
    """Coerce raw arrays into a validated :class:`EmpiricalMeasure`."""
    X = np.asarray(X, dtype=np.float64)
    if w is None:
        w = np.ones(X.shape[0], dtype=np.float64)
    return EmpiricalMeasure(X, np.asarray(w, dtype=np.float64), name)


__all__ = ["load_csv", "load_npz", "to_measure"]
