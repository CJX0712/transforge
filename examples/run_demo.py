"""End-to-end demonstration: produces ``benchmark.json``.

Every number in the output comes from an actual computation performed by this
script.  Nothing is hard-coded, and when an optional backend is missing the
affected gate reports ``skipped`` rather than a fabricated value.

Usage::

    python examples/run_demo.py                 # writes ./benchmark.json
    python examples/run_demo.py --out bench.json
    python examples/run_demo.py --quick         # smoke variant for CI

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

# Make the demo runnable straight from a clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transforge.core.config import DEMO_SCALES, GATES, RunConfig, SinkhornConfig  # noqa: E402
from transforge.core.seed import set_all  # noqa: E402
from transforge.eval.report import BenchmarkReport, core_metrics  # noqa: E402
from transforge.pipeline.backends import capability_report  # noqa: E402
from transforge.pipeline.runner import TransForgePipeline  # noqa: E402


def _utf8_stdout() -> None:
    with contextlib.suppress(AttributeError, ValueError):
        sys.stdout.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = argparse.ArgumentParser(description="TransForge end-to-end demonstration")
    parser.add_argument("--out", default="benchmark.json", help="output path")
    parser.add_argument("--n", type=int, default=24, help="support points per measure")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds per scale")
    parser.add_argument(
        "--quick", action="store_true", help="smoke variant: fewer scales, no ablations"
    )
    args = parser.parse_args(argv)

    scales = (1.0, 20.0) if args.quick else list(DEMO_SCALES)
    n_seeds = 2 if args.quick else args.seeds
    cfg = RunConfig(
        seed=20260101,
        n=args.n,
        dim=2,
        threads=1,
        sinkhorn=SinkhornConfig(),
    )
    seeds = [cfg.seed + i for i in range(n_seeds)]

    print("TransForge demonstration")
    print("=" * 70)
    print("author          : 晨星 (CJX0712)")
    print(f"n per measure   : {cfg.n}")
    print(f"scales          : {scales}")
    print(f"seeds           : {seeds}")
    print(f"threads         : {cfg.threads} (pinned for bitwise reproducibility)")
    caps = capability_report()
    print(
        "backends        : "
        + ", ".join(f"{k}={'yes' if v['available'] else 'NO'}" for k, v in sorted(caps.items()))
    )
    print()

    started = time.perf_counter()
    set_all(cfg.seed, cfg.threads)
    report = TransForgePipeline(cfg).benchmark(
        seeds=seeds,
        scales=scales,
        run_ablations=not args.quick,
        ablation_scales=[20.0],
    )
    elapsed = time.perf_counter() - started
    report.meta["demo_elapsed_s"] = round(elapsed, 3)
    report.meta["demo_budget_s"] = GATES["demo_budget_s"]
    report.meta["demo_within_budget"] = bool(elapsed <= GATES["demo_budget_s"])

    path = report.save(args.out)
    print(report.render_text())
    print()
    print("-" * 70)
    print(
        f"end-to-end elapsed : {elapsed:.2f}s "
        f"(budget {GATES['demo_budget_s']:.0f}s -> "
        f"{'WITHIN' if elapsed <= GATES['demo_budget_s'] else 'OVER'})"
    )
    print(f"benchmark written  : {path}")

    # Read the file back and re-verify, so a truncated or mis-encoded write cannot
    # pass silently.
    reloaded = BenchmarkReport.load(path)
    assert reloaded["meta"]["seed"] == cfg.seed, "benchmark.json failed read-back"
    core = core_metrics(reloaded)
    print(
        f"read-back verified : {len(core['cells'])} cells, "
        f"{len(core['gates'])} gates, "
        f"{sum(1 for g in core['gates'] if g['passed'])} passing"
    )

    inv = reloaded["invariants"]
    print(f"invariants         : {inv['n_checks']} checked, all_passed={inv['all_passed']}")
    if reloaded["skipped"]:
        print(f"skipped            : {', '.join(reloaded['skipped'])}")

    print()
    print(f"GATES PASSED: {report.gates_passed()}/{report.gates_total()}")
    for g in report.gates:
        state = {True: "PASS", False: "FAIL", None: "SKIP"}[g["passed"]]
        print(f"  {g['id']} {state}  {g.get('detail', '')[:96]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
