"""Optional-backend probes.

Every probe answers without importing eagerly where possible, and a missing
backend yields ``available() is False`` -- never an exception and never a
fabricated result.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


def _pot_disabled() -> bool:
    return os.environ.get("TRANSFORGE_DISABLE_POT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Probe:
    """A capability probe result."""

    name: str
    available: bool
    version: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "available": self.available,
            "version": self.version,
            "detail": self.detail,
        }


def available_numpy() -> Probe:
    try:
        import numpy  # noqa: PLC0415

        return Probe("numpy", True, numpy.__version__)
    except Exception as exc:  # noqa: BLE001
        return Probe("numpy", False, None, str(exc))


def available_scipy() -> Probe:
    if importlib.util.find_spec("scipy") is None:
        return Probe("scipy", False, None, "not installed")
    try:
        return Probe("scipy", True, version("scipy"))
    except PackageNotFoundError:
        return Probe("scipy", True, "unknown")


def available_sklearn() -> Probe:
    if importlib.util.find_spec("sklearn") is None:
        return Probe("scikit-learn", False, None, "not installed")
    try:
        return Probe("scikit-learn", True, version("scikit-learn"))
    except PackageNotFoundError:
        return Probe("scikit-learn", True, "unknown")


def available_pot() -> Probe:
    """POT probe: importable AND exposing the exact EMD solver.

    ``ot.unbalanced.sinkhorn_reg_scaling`` is known in POT 0.9.7.post1 to warn and
    fall back to classic Sinkhorn-Knopp, i.e. epsilon-scaling is NOT implemented
    upstream.  That gap is recorded here because it is the justification for
    building our own scheduled solver.
    """
    if _pot_disabled():
        return Probe("POT", False, None, "disabled via TRANSFORGE_DISABLE_POT")
    if importlib.util.find_spec("ot") is None:
        return Probe("POT", False, None, "not installed")
    try:
        import ot  # noqa: PLC0415

        has_emd = callable(getattr(ot, "emd", None))
        return Probe(
            "POT",
            has_emd,
            getattr(ot, "__version__", None),
            "network simplex available" if has_emd else "importable but ot.emd missing",
        )
    except Exception as exc:  # noqa: BLE001
        return Probe("POT", False, None, f"import failed: {exc}")


def available_pot_unbalanced() -> Probe:
    if not available_pot().available:
        return Probe("POT.unbalanced", False, None, "POT unavailable")
    try:
        import ot.unbalanced as uot  # noqa: PLC0415

        return Probe(
            "POT.unbalanced",
            hasattr(uot, "sinkhorn_knopp_unbalanced"),
            None,
            "reg_scaling is NOT implemented upstream (warns and falls back)",
        )
    except Exception as exc:  # noqa: BLE001
        return Probe("POT.unbalanced", False, None, str(exc))


def available_pot_gromov() -> Probe:
    if not available_pot().available:
        return Probe("POT.gromov", False, None, "POT unavailable")
    try:
        import ot.gromov  # noqa: PLC0415

        return Probe("POT.gromov", hasattr(ot.gromov, "gromov_wasserstein"))
    except Exception as exc:  # noqa: BLE001
        return Probe("POT.gromov", False, None, str(exc))


def available_scipy_lp() -> Probe:
    if available_scipy().available is False:
        return Probe("scipy.optimize.linprog", False, None, "scipy unavailable")
    try:
        from scipy.optimize import linprog  # noqa: PLC0415

        return Probe("scipy.optimize.linprog", callable(linprog))
    except Exception as exc:  # noqa: BLE001
        return Probe("scipy.optimize.linprog", False, None, str(exc))


def available_blas_threads() -> Probe:
    from ..core.seed import THREAD_ENV_VARS  # noqa: PLC0415

    raw = {v: os.environ.get(v) for v in THREAD_ENV_VARS}
    pinned = raw.get("OMP_NUM_THREADS")
    return Probe(
        "blas_threads",
        True,
        pinned,
        f"cpu_count={os.cpu_count()}, pinned={pinned}",
    )


def capability_report() -> dict[str, dict[str, object]]:
    """The full capability matrix, printed by ``transforge info``."""
    probes = [
        available_numpy(),
        available_scipy(),
        available_sklearn(),
        available_pot(),
        available_pot_unbalanced(),
        available_pot_gromov(),
        available_scipy_lp(),
        available_blas_threads(),
    ]
    return {p.name: p.to_dict() for p in probes}


def missing_backends() -> list[str]:
    """Names of unavailable optional backends (numpy is mandatory)."""
    return [name for name, info in capability_report().items() if not info["available"]]


__all__ = [
    "Probe",
    "available_blas_threads",
    "available_numpy",
    "available_pot",
    "available_pot_gromov",
    "available_pot_unbalanced",
    "available_scipy",
    "available_scipy_lp",
    "available_sklearn",
    "capability_report",
    "missing_backends",
]
