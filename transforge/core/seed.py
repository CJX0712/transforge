"""The single global determinism entry point.

``set_all(seed)`` is the ONLY supported way to seed TransForge.  Everything else
in the codebase obtains randomness from ``np.random.default_rng(seed)``.

Two hazards are handled here:

1. numpy's legacy global state is seeded as well, so third-party code (POT,
   scikit-learn) that still calls ``np.random.seed`` becomes reproducible.
2. BLAS thread counts cause non-deterministic floating-point reduction order.
   The environment variables must be set **before** numpy is imported to take
   effect; we set them anyway (helps for subprocesses) and additionally pin the
   thread count at runtime where the pool allows it.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import os
import random

import numpy as np

from .errors import DeterminismError

#: Environment variables that pin BLAS/OpenMP reduction order.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_last_seed: int | None = None


def pin_threads(n_threads: int = 1) -> None:
    """Pin every BLAS backend to ``n_threads``.

    Must be called before importing numpy to be fully effective; calling it later
    still helps for libraries that read the variable lazily.
    """
    if n_threads < 1:
        raise DeterminismError(f"n_threads must be >= 1, got {n_threads}")
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(n_threads)


def set_all(seed: int, n_threads: int = 1) -> int:
    """Seed every randomness source TransForge can reach.  Returns ``seed``.

    Parameters
    ----------
    seed:
        Non-negative integer seed.
    n_threads:
        BLAS thread count.  ``1`` (the default) is required for bit-level
        reproducibility across runs.
    """
    global _last_seed
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise DeterminismError(f"seed must be a non-negative integer, got {seed!r}")
    seed = int(seed)
    pin_threads(n_threads)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # optional: silences BLAS thread pools without hard-depending on it
        from threadpoolctl import threadpool_limits

        threadpool_limits(n_threads)
    except Exception:  # noqa: BLE001 - optional dependency, never fatal
        pass
    _last_seed = seed
    return seed


def current_seed() -> int | None:
    """The seed most recently installed by :func:`set_all`, or ``None``."""
    return _last_seed


def rng(seed: int | None = None) -> np.random.Generator:
    """The ONE sanctioned randomness factory: ``np.random.default_rng``.

    Passing ``None`` reuses the seed installed by :func:`set_all`.  The legacy
    ``np.random.RandomState`` is deliberately never used, because mixing the two
    produces irreproducible streams.
    """
    if seed is None:
        seed = _last_seed
        if seed is None:
            raise DeterminismError("no seed installed; call set_all(seed) first")
    return np.random.default_rng(int(seed))


__all__ = ["THREAD_ENV_VARS", "current_seed", "pin_threads", "rng", "set_all"]
