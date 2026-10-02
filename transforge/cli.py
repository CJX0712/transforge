"""Command-line interface.

Subcommands
-----------
``info``       capability matrix (which optional backends are present)
``invariant``  run the cross-validation invariants; ``--strict`` exits non-zero
``bench``      full benchmark; writes ``benchmark.json``
``demo``       short end-to-end run
``solve``      solve a single instance and print the selected epsilon
``reproduce``  run twice and verify bitwise determinism

Author: 晨星 (CJX0712)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config import BENCH_SCALES, GATES, RunConfig
from .core.errors import TransForgeError
from .core.seed import set_all
from .pipeline.backends import capability_report
from .pipeline.runner import TransForgePipeline


def _utf8_stdout() -> None:
    """Force UTF-8 so CJK text in reports survives a cp1252 console."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transforge",
        description="TransForge: a verified optimal-transport solver bench",
    )
    p.add_argument("--seed", type=int, default=None, help="global seed")
    p.add_argument("--n", type=int, default=None, help="number of support points")
    p.add_argument("--threads", type=int, default=None, help="BLAS threads (1 = deterministic)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="show the capability matrix")

    inv = sub.add_parser("invariant", help="run the cross-validation invariants")
    inv.add_argument("--strict", action="store_true", help="exit non-zero on any failure")
    inv.add_argument("--json", action="store_true", help="emit JSON")

    bench = sub.add_parser("bench", help="run the full benchmark")
    bench.add_argument("--seeds", type=int, default=int(GATES["sig_seeds"]))
    bench.add_argument("--n-scale", type=float, default=1.0, help="scale factor for n")
    bench.add_argument("--scales", type=float, nargs="*", default=None)
    bench.add_argument("--out", default="benchmark.json")
    bench.add_argument("--no-ablation", action="store_true")

    demo = sub.add_parser("demo", help="short end-to-end demonstration")
    demo.add_argument("--out", default="benchmark.json")

    solve = sub.add_parser("solve", help="solve one instance, print the chosen epsilon")
    solve.add_argument("--scale", type=float, default=1.0)

    repro = sub.add_parser("reproduce", help="run twice and check bitwise determinism")
    repro.add_argument("--out", default=None)

    return p


def _cfg(args: argparse.Namespace) -> RunConfig:
    cfg = RunConfig.from_env()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.n is not None:
        cfg.n = args.n
    if args.threads is not None:
        cfg.threads = args.threads
    return cfg


def cmd_info() -> int:
    print("TransForge capability matrix")
    print("-" * 60)
    for name, info in sorted(capability_report().items()):
        flag = "yes" if info["available"] else "NO "
        print(f"{name:<26} {flag}  {info.get('version') or ''}")
        if info.get("detail"):
            print(f"{'':<26}      {info['detail']}")
    print("-" * 60)
    print(f"gate thresholds: {json.dumps(GATES, indent=2)}")
    return 0


def cmd_invariant(args: argparse.Namespace, cfg: RunConfig) -> int:
    set_all(cfg.seed, cfg.threads)
    report = TransForgePipeline(cfg).invariants()
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("TransForge invariant report")
        print("-" * 72)
        print(f"{'id':<6}{'result':<9}{'value':<14}{'tol':<12}detail")
        for c in report.checks:
            flag = "PASS" if c.passed else "FAIL"
            print(f"{c.id:<6}{flag:<9}{c.value:<14.3e}{c.tolerance:<12.1e}{c.detail}")
        print("-" * 72)
        print(f"all_passed={report.all_passed}  worst_violation={report.worst_violation:.3e}")
        for s in report.skipped:
            print(f"SKIPPED: {s}")
    if args.strict and not report.all_passed:
        print("\nE500: an invariant was falsified; refusing to report success.")
        return 1
    return 0


def cmd_bench(args: argparse.Namespace, cfg: RunConfig) -> int:
    set_all(cfg.seed, cfg.threads)
    n = max(8, int(cfg.n * args.n_scale))
    local = RunConfig(seed=cfg.seed, n=n, dim=cfg.dim, threads=cfg.threads, sinkhorn=cfg.sinkhorn)
    seeds = [cfg.seed + i for i in range(args.seeds)]
    scales = args.scales if args.scales else list(BENCH_SCALES)
    report = TransForgePipeline(local).benchmark(
        seeds=seeds, scales=scales, run_ablations=not args.no_ablation
    )
    path = report.save(args.out)
    print(report.render_text())
    print(f"\nwrote {path}")
    return 0 if report.all_gates_passed() else 2


def cmd_demo(args: argparse.Namespace, cfg: RunConfig) -> int:
    return cmd_bench(
        argparse.Namespace(
            seeds=3, n_scale=1.0, scales=[0.5, 1.0, 20.0], out=args.out, no_ablation=False
        ),
        cfg,
    )


def cmd_solve(args: argparse.Namespace, cfg: RunConfig) -> int:
    import numpy as np  # noqa: PLC0415

    from .core.linalg import pairwise_sq_euclidean  # noqa: PLC0415
    from .data.synth import scale_family  # noqa: PLC0415
    from .pipeline.gates import exact_reference  # noqa: PLC0415
    from .solvers.epsilon_knee import TFEpsilonKnee  # noqa: PLC0415

    set_all(cfg.seed, cfg.threads)
    problem = scale_family(cfg.n, args.scale, seed=cfg.seed)
    a, b = problem.source.w, problem.target.w
    C = pairwise_sq_euclidean(problem.source.X, problem.target.X)
    res = TFEpsilonKnee(cfg.sinkhorn).solve(a, b, C)
    exact = exact_reference(a, b, C)
    print(f"scale            : {args.scale}")
    print(f"median(C)        : {np.median(C):.6g}")
    print(f"chosen epsilon   : {res.epsilon:.6g}  (= {res.epsilon / np.median(C):.4g} * median)")
    print(f"ladder rungs used: {res.meta['n_levels_used']}/{res.meta['levels_total']}")
    print(f"bias bound       : {res.meta['bias_bound']:.6g}  certified={res.meta['certified']}")
    print(f"residual         : {res.residual:.3e}   iterations={res.n_iter}")
    print(f"wall-clock       : {res.meta['wall_s']:.3f}s")
    print(f"stopping criteria: {res.meta['stopping_criteria']}")
    if exact is not None:
        print(f"exact OT         : {exact:.6g}")
        print(f"relative error   : {abs(res.cost - exact) / exact:.4%}")
    else:
        print("exact OT         : unavailable (POT missing) -- not fabricating a value")
    return 0


def cmd_reproduce(args: argparse.Namespace, cfg: RunConfig) -> int:
    """Run the pipeline twice and compare the serialised payloads bitwise."""

    local = RunConfig(
        seed=cfg.seed, n=min(cfg.n, 40), dim=cfg.dim, threads=1, sinkhorn=cfg.sinkhorn
    )
    payloads = []
    for _ in range(2):
        set_all(local.seed, 1)
        rep = TransForgePipeline(local).benchmark(
            seeds=[local.seed], scales=[1.0, 20.0], run_ablations=False
        )
        d = rep.to_dict()
        d["meta"].pop("elapsed_s", None)
        for cell in d["cells"]:
            for k in ("t_flagship", "t_oracle", "t_fixed"):
                cell.pop(k, None)
        payloads.append(d)
    a = json.dumps(payloads[0], sort_keys=True, ensure_ascii=False)
    b = json.dumps(payloads[1], sort_keys=True, ensure_ascii=False)
    same = a == b
    print(f"determinism: {'BITWISE IDENTICAL' if same else 'MISMATCH'}")
    if not same:
        print("  first differing region:")
        for i, (ca, cb) in enumerate(zip(a, b, strict=False)):
            if ca != cb:
                print(f"    offset {i}: {a[max(0, i - 60) : i + 60]!r}")
                print(f"            {b[max(0, i - 60) : i + 60]!r}")
                break
    if args.out:
        Path(args.out).write_text(a, encoding="utf-8")
        print(f"wrote {args.out}")
    return 0 if same else 1


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _cfg(args)
    try:
        if args.command == "info":
            return cmd_info()
        if args.command == "invariant":
            return cmd_invariant(args, cfg)
        if args.command == "bench":
            return cmd_bench(args, cfg)
        if args.command == "demo":
            return cmd_demo(args, cfg)
        if args.command == "solve":
            return cmd_solve(args, cfg)
        if args.command == "reproduce":
            return cmd_reproduce(args, cfg)
    except TransForgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
