"""Benchmark report: serialisation and fixed-width text tables.

Report attributes that could shadow a method carry a ``_report`` suffix, so
``report.gates_report`` is a field while ``report.gates()`` stays a method.

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkReport:
    """The full, machine-readable outcome of a benchmark run."""

    meta: dict[str, Any] = field(default_factory=dict)
    gates: list[dict[str, Any]] = field(default_factory=list)
    cells: list[dict[str, Any]] = field(default_factory=list)
    ablation: list[dict[str, Any]] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    invariants: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, Any] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- accessors (methods; the fields above keep the ``_report`` discipline) --
    def gates_passed(self) -> int:
        return int(sum(1 for g in self.gates if g.get("passed")))

    def gates_total(self) -> int:
        return len(self.gates)

    def all_gates_passed(self) -> bool:
        return self.gates_passed() == self.gates_total()

    def failing_gates(self) -> list[dict[str, Any]]:
        return [g for g in self.gates if not g.get("passed")]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready payload with every numpy scalar coerced to a Python type."""
        return {
            "meta": _clean(self.meta),
            "gates": _clean(self.gates),
            "cells": _clean(self.cells),
            "ablation": _clean(self.ablation),
            "failures": _clean(self.failures),
            "invariants": _clean(self.invariants),
            "capabilities": _clean(self.capabilities),
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
        }

    def save(self, path: str | Path) -> Path:
        """Write JSON with ``ensure_ascii=False`` so UTF-8 survives round-trip."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False, sort_keys=False)
            fh.write("\n")
        return p

    @staticmethod
    def load(path: str | Path) -> dict[str, Any]:
        p = Path(path)
        with p.open(encoding="utf-8") as fh:
            return json.load(fh)

    # -- rendering ---------------------------------------------------------
    def render_text(self) -> str:
        """Fixed-width tables.  Widths are literal so columns never shift."""
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("TransForge benchmark report")
        lines.append("=" * 78)
        for key, value in sorted(self.meta.items()):
            lines.append(f"{key:<24}: {value}")
        lines.append("")

        lines.append("-" * 78)
        lines.append("GATES")
        lines.append("-" * 78)
        header = f"{'gate':<6}{'passed':<8}{'measured':<16}{'threshold':<14}{'detail'}"
        lines.append(header)
        for g in self.gates:
            flag = "PASS" if g.get("passed") else "FAIL"
            value = g.get("measured_value")
            if value is None:
                value = g.get("time_ratio")  # G3's headline number IS a time ratio
            measured = "n/a" if value is None else f"{float(value):.4g}"
            thr = f"{g.get('threshold', float('nan')):.4g}"
            detail = str(g.get("detail", ""))[:28]
            lines.append(f"{g.get('id', '?'):<6}{flag:<8}{measured:<16}{thr:<14}{detail}")
        lines.append("")
        lines.append(f"gates passed: {self.gates_passed()}/{self.gates_total()}")
        lines.append("")

        if self.cells:
            lines.append("-" * 78)
            lines.append("CELLS (cross-scale family)")
            lines.append("-" * 78)
            lines.append(
                f"{'scale':<8}{'seed':<6}{'rel_flag':<12}{'rel_oracle':<12}"
                f"{'rel_fixed':<12}{'t_flag':<10}{'t_oracle':<10}"
            )
            for c in self.cells:
                lines.append(
                    f"{c.get('scale', '?'):<8}{c.get('seed', '?'):<6}"
                    f"{c.get('rel_err_flagship', float('nan')):<12.4%}"
                    f"{c.get('rel_err_oracle', float('nan')):<12.4%}"
                    f"{c.get('rel_err_fixed', float('nan')):<12.4%}"
                    f"{c.get('t_flagship', float('nan')):<10.3f}"
                    f"{c.get('t_oracle', float('nan')):<10.3f}"
                )
            lines.append("")

        if self.ablation:
            lines.append("-" * 78)
            lines.append("ABLATIONS (negative results retained)")
            lines.append("-" * 78)
            for a in self.ablation:
                lines.append(
                    f"{a.get('variant', '?'):<28}{a.get('rel_err', float('nan')):<14.4%}"
                    f"  {a.get('conclusion', '')}"
                )
            lines.append("")

        if self.failures:
            lines.append("-" * 78)
            lines.append("FAILURE CASES (derived from results)")
            lines.append("-" * 78)
            for f in self.failures:
                lines.append(f"* {f.get('title', '')}")
                lines.append(f"    evidence: {f.get('evidence', '')}")
                lines.append(f"    cause   : {f.get('cause', '')}")
            lines.append("")

        if self.skipped:
            lines.append(f"SKIPPED: {', '.join(self.skipped)}")
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


#: Field-name fragments that denote WALL-CLOCK and therefore cannot be required to
#: be bit-identical across runs.  Invariant I9 compares everything else verbatim.
TIMING_FIELDS: frozenset[str] = frozenset(
    {
        "wall_s",
        "elapsed_s",
        "t_flag",
        "t_fixed",
        "t_oracle",
        "t_flagship",
        "t_fixed_baseline_s",
        "t_oracle_s",
        "t_flagship_s",
        "time_ratio",
        "median_flagship_s",
        "slowest_baseline_s",
        "fastest_flagship_s",
        "fastest_baseline_s",
        "mean_wall_s",
        # human-readable summaries embed wall-clock ratios, so they are excluded too;
        # every number they quote is present verbatim in the structured fields
        "detail",
    }
)


def core_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip every wall-clock field so two runs can be compared BITWISE.

    Invariant I9 requires the benchmark's *core* metrics to be identical to the
    last bit between runs.  Timings are excluded by construction -- they depend on
    machine load, not on the algorithm.  Everything else (relative errors,
    selected epsilons, pass/fail flags, invariant verdicts) is retained.
    """

    def strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: strip(v)
                for k, v in obj.items()
                if k not in TIMING_FIELDS
                and not k.endswith("_wall_s")
                # Wall-clock keys are not always verbatim members of TIMING_FIELDS:
                # `meta` carries `demo_elapsed_s`, a prefixed spelling of
                # `elapsed_s`.  Strip any leading `*_` prefix before testing, so a
                # prefixed timing key cannot leak into the bitwise comparison and
                # make invariant I9 fail on machine load alone.  The prefix is
                # everything before the FIRST underscore (`demo_elapsed_s` ->
                # `elapsed_s`); splitting on the last one would leave `"s"`.
                and k.split("_", 1)[-1] not in TIMING_FIELDS
            }
        if isinstance(obj, list):
            return [strip(v) for v in obj]
        return obj

    return strip(payload)


def _clean(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays into JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def summarise(values: list[float]) -> dict[str, float]:
    """mean/std/min/max of a sample; ``std`` is the population std (ddof=0)."""
    if not values:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def significant(mean_a: float, std_a: float, mean_b: float, std_b: float) -> bool:
    """Forge-series significance rule: ``|delta| > (std_a + std_b) / 2``.

    Deliberately not a pooled t-test: with 3-5 seeds a t-test's p-values are
    noise.  This rule is conservative and was fixed in advance.
    """
    return abs(mean_a - mean_b) > 0.5 * (std_a + std_b)


__all__ = ["BenchmarkReport", "TIMING_FIELDS", "core_metrics", "significant", "summarise"]
