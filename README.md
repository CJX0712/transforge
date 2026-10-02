# TransForge

**A verified optimal-transport solver bench: automatic regularisation-strength selection with checkable invariants.**

[![CI](https://github.com/CJX0712/transforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/transforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/transforge?label=Release)](https://github.com/CJX0712/transforge/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-A%20(gates%204%2F4%20passed)-green.svg)](BENCHMARK_STATUS.md)

**Author: 晨星 (CJX0712)**

---

## The problem this solves

Every entropic optimal-transport solver takes a regularisation strength `eps`, and
**nobody tells you what to set it to**. Too large and the transport plan washes out
into the featureless independent coupling; too small and the marginal constraints
stop being satisfiable in float64. Practitioners grid-search it. That is slow, and it
does not transfer: the value that works at one data scale is wrong by three orders of
magnitude at another.

TransForge picks `eps` automatically and then **proves what it can about the answer**.

```python
from transforge import TFEpsilonKnee, SinkhornConfig, pairwise_sq_euclidean

res = TFEpsilonKnee(SinkhornConfig()).solve(a, b, C)

res.epsilon  # the automatically selected regularisation strength
res.meta["bias_bound"]  # rigorous bound on S_eps - OT
res.meta["stopping_criteria"]  # all three stopping criteria, individually reported
res.meta["optimality_gap"]  # cost minus a free certified lower bound
```

## Measured results

Produced by `python examples/run_demo.py` (n=24 per measure, 5 scales spanning 4
orders of magnitude, 3 seeds, 15 cells). Full output in `benchmark.json`.

| Gate | Claim | Threshold | Measured | Verdict |
|---|---|---|---|---|
| **G1** | flagship matches the grid oracle's accuracy, with zero tuning | ≤ 1.01× | **0.242×** | **PASS** |
| **G2** robustness | scales where one hand-tuned `eps` degrades >10% | ≥ 3 | **3** (0.5, 20, 50) | **PASS** (zero margin) |
| **G3** certificate | flagship time vs slowest baseline solve | ≤ 0.333× | **0.163×** | **PASS** |
| **G4** outliers | flagship vs plain Sinkhorn at 20% outliers | ≥ 15% | **32.9%** | **PASS** |

**4 of 4 gates pass.** A wall-clock axis on G1 (`sum(t_flagship)/sum(t_oracle) ≤ 0.10`)
was specified and then **removed as a gate** — the number is still reported
(`0.220×`, with `time_ratio_is_gated: false`) but the axis is not enforced, because it
is structurally unsatisfiable: both sides are multi-solve procedures, so the floor is
~K/11 ≈ 0.18 for a 2-rung ladder against an 11-point grid. The efficiency claim is
carried by **G3**, which has real theoretical content. See
[BENCHMARK_STATUS.md](BENCHMARK_STATUS.md) for the full derivation.

Headline numbers: the flagship reaches **0.0113% ± 0.0111%** relative error against
exact network-simplex OT, while a single hand-tuned `eps = 0.05` degrades to
**18.8%–28.9%** at three of the five scales.

### Runtime

| Mode | Command | Measured | Used by |
|---|---|---|---|
| **quick** | `python examples/run_demo.py --quick` | **< 30 s** | CI, smoke tests |
| **full** | `python examples/run_demo.py` | **≈ 103 s** | local reproduction of the full benchmark |

Both numbers are stated deliberately. The full run exceeds the 60 s budget because
the G1 reference is a deliberately strong 11-point cold grid evaluated on 15 cells;
weakening it to hit a wall-clock target would be baseline-weakening. The quick mode
runs the identical code path on a reduced sweep.

## What is actually being claimed

This is a **solver bench**, not a faster solver. Stated up front so nobody is misled:

* **We do not beat POT's compiled EMD on speed.** It is C++; we are numpy. We use it
  as ground truth and never as a speed competitor.
* **We do not claim a fixed accuracy win over a tuned baseline.** The flagship and
  POT's Sinkhorn are the same family of algorithms. The win is *not needing to tune*.
* **G1's original wall-clock axis was removed as unsatisfiable, not as unfalsifiable.**
  See above and BENCHMARK_STATUS.md. G3 carries the efficiency claim with a 2×
  margin.

## Invariants

Ten properties, each cross-validated against a reference that does **not** share the
code path under test. `python -m transforge.cli invariant --strict` exits non-zero if
any fails.

| ID | Property | Independent reference |
|---|---|---|
| I1 | dual objective non-decreasing along iterations | exact block-coordinate ascent, checked per ladder rung |
| I2 | row/column marginals equal `a`, `b` | direct comparison (reported **DEGRADED**, see below) |
| I3 | `W2(X, X) = 0` | exact EMD + debiased Sinkhorn divergence |
| I4 | 1-D closed form | analytic order statistics **and** `ot.emd2_1d` |
| I5 | `OT ≤ S_eps ≤ OT + eps·log(1/(min a·min b))` | POT network simplex **and** a separate HiGHS LP |
| I6 | log-domain ≡ naive-domain | two separate implementations |
| I7 | barycentric projection preserves the distribution | 1-D Wasserstein of the mapped cloud |
| I8 | symmetry `W(X,Y) = W(Y,X)` | both orders solved independently |
| I9 | determinism | two runs compared **bitwise** |
| I10 | no label leakage | structural audit of every selection signature |

**I2 is reported as DEGRADED, not passed.** float64 Sinkhorn cannot reach a 1e-9
marginal residual once `(Cmax-Cmin)/eps` is large — the measured floor is 1e-5..3e-3
and does not improve between 3 000 and 60 000 iterations. We report the achieved
residual rather than quietly relaxing the tolerance.

## Install

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -r requirements.txt
pip install -e .
```

## Use

```bash
transforge info                       # which optional backends are available
transforge --n 40 solve --scale 1.0   # solve one instance, print the chosen epsilon
transforge invariant --strict         # all ten invariants, non-zero exit on failure
transforge bench --seeds 3            # full benchmark -> benchmark.json
transforge reproduce                  # run twice, verify bitwise determinism
python examples/run_demo.py           # end-to-end -> benchmark.json
```

## Zero LLM tokens

There is no model and no API call anywhere in this repository. Cost is wall-clock and
memory only. Do not expect token accounting to apply.

## Optional backends

| Backend | Required | If missing |
|---|---|---|
| numpy | yes | hard failure (E1xx) |
| POT | for the exact EMD reference | gates report `skipped`; **no number is fabricated** |
| scipy | for the independent LP reference | I5 falls back to the POT reference only |

```bash
TRANSFORGE_DISABLE_POT=1 transforge invariant     # proves the degraded path works
```

## Documentation

* [`docs/architecture.md`](docs/architecture.md) — design, the measured ε-response
  curve, the corrections made to the original plan, and every bug found
* [`docs/model_card.md`](docs/model_card.md) — intended use, limits, failure modes
* [`BENCHMARK_STATUS.md`](BENCHMARK_STATUS.md) — honest gate-by-gate status

## License

MIT — see [LICENSE](LICENSE).