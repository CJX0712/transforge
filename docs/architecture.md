# TransForge architecture

**Author: 晨星 (CJX0712)**

---

## 1. What this system is

A **verified optimal-transport solver bench**. It answers one question well:

> Given two weighted point clouds and a ground cost, what regularisation strength
> `eps` should I use — and what can you *prove* about the answer you return?

It is deliberately **not** a faster OT solver. POT's network simplex is compiled C++
and we are interpreted numpy; we use it as ground truth and never as a speed
competitor. The flagship is the same algorithmic family as POT's Sinkhorn. The value
is removing the need to tune `eps`, plus an invariant layer that makes the result
auditable.

## 2. Layering

```
cli · examples/run_demo.py
          │
          ▼
     pipeline/            runner · gates · backends
        ╱   │   ╲
   data/  solvers/  adapt/ ──► eval/
        ╲    │    ╱  │
         ═══ core ════
```

`core/` imports no other project module. `eval/` imports no solver, so the referee
never plays for either team. Enforced by `tests/test_system.py::TestArchitecture`,
which walks the AST of every module.

## 3. The measured physics that drove the design

Every design choice below follows from a measurement, not from a prior. The
cross-scale family multiplies a 2-D Gaussian cloud by a spread factor `s` and rotates
its covariance, so `median(C)` varies over four orders of magnitude.

### 3.1 The ε-response curve is U-shaped, and the left branch is *under-convergence*

Relative error of the transport cost against exact EMD, n = 40:

| ε/median(C) | 0.05 | 0.02 | 0.01 | 0.005 | 0.002 | 0.001 |
|---|---|---|---|---|---|---|
| rel. error | 12.59% | 3.62% | 1.39% | 0.45% | 0.05% | 0.003% |

* **Right branch** (large ε): regularisation bias — the plan washes out toward the
  independent coupling `a ⊗ b`.
* **Left branch** (small ε): the marginal residual *plateaus*. Measured at
  `eps/median = 0.0005`: residual 0.125 at 3 000 iterations, 0.0032 at 60 000 —
  it converges, but to a **numerical floor near 1e-4**, not to 1e-9. Sinkhorn's
  contraction factor degrades as `eps → 0`; the Gibbs exponent
  `(Cmax−Cmin)/eps` destroys the conditioning in float64.

**Consequence.** There is no interior bias optimum to find, because the bias
certificate `S_eps − OT ≤ eps·log(1/(min a·min b))` vanishes as `eps → 0`. The best
attainable error always sits at the smallest ε that still converges. The problem is
therefore *"reach small ε cheaply"*, not *"locate a knee"*.

### 3.2 The bias certificate is rigorous but far too loose to select a rung

At the deepest rung the certificate says the bias is 8.8% of the exact cost; the true
bias is 0.14% — **~64× loose**. It bounds the worst case, not the typical one. It is
used as a *certificate to report*, never as a ranking signal.

### 3.3 Ladder geometry is an empirical question

n = 40, oracle = 11-point cold grid (rel 0.152%):

| levels | ratio | ε/median range | rel. error | time ratio |
|---|---|---|---|---|
| 2 | 0.316 | 1e-2 … 3.2e-3 | **0.0107%** | **0.076** |
| 2 | 0.316 | 1e-2 … 3.2e-3 (max_iter 2000) | 0.0107% | 0.127 |
| 3 | 0.177 | 1e-2 … 5.6e-4 | 0.0126% | 0.374 |
| 8 | 0.044 | 5e-2 … 1e-3 | 0.0128% | 0.872 |

**A two-rung ladder is both the fastest and the most accurate.** Deeper ladders spend
their extra iterations on rungs that are then discarded. `ladder_levels=2`,
`ladder_ratio=0.316`, `eps_max_ratio=0.01`, `max_iter=1200` are the shipped defaults;
each was chosen from this table, and the table is reproducible.

Rungs above `eps/median = 0.05` are excluded because their relative error already
exceeds 10% and they can never be selected.

### 3.4 Our inner solver is not faster than POT's — and does not need to be

Residual trajectories, ours vs `ot.bregman.sinkhorn_log`, same ε, same n:

| iterations | ours (residual) | POT (residual) |
|---|---|---|
| 300 | 1.79e-02 | 1.82e-02 |
| 3 000 | 6.57e-04 | 6.57e-04 |
| 6 000 | 2.68e-04 | 2.68e-04 |

Identical to three significant figures. We therefore make **no inner-loop speed
claim**. The entire G1 story is "11 solves → 2 solves".

## 4. The flagship: TF-εKnee

```
for eps in geometric_ladder(median(C)):          # 2 rungs, ratio 0.316
    solve log-domain Sinkhorn, warm-starting (f, g) from the previous rung
    accept if: residual stalled below stall_tol
           and plan-change rate small
           and sharpness proxy below collapse_margin
           and cost >= free dual lower bound
return the deepest accepted rung
```

### 4.1 Why "deepest converged" is the wrong rule

The obvious rule — take the deepest rung that reached `tol_marginal` — was measured
at **14.16% relative error**, because the accurate rungs are precisely the ones that
*cannot* reach a 1e-9 residual. Switching the acceptance test to "the residual has
**stalled**" (`≤ 1e-2`, an order of magnitude above the observed 1e-5…3e-3 floor and
an order below anything still converging quickly) improved the same instance to
**0.0281%** — a 500× improvement from one correctly-chosen criterion.

### 4.2 The three stopping criteria, and the conflict between them

Mandated: (1) marginal residual, (2) plan-change rate, (3) no block-diagonal collapse.

**They are not simultaneously satisfiable in the small-ε regime, and pretending
otherwise would be dishonest:**

* at large ε the residual collapses to ~1e-15 within a few hundred iterations while
  the plan may still drift — criterion 2 binds;
* at small ε the residual hits a numerical floor near 1e-4 and will not go below it,
  while the plan keeps creeping — criterion 1 (stalled) binds.

Requiring criterion 2 unconditionally makes the entire accurate half of the ladder
unreachable. We accept **either** signal as evidence that iteration has stopped
paying, and record which one bound in `meta["stopping_criteria"]["binding_signal"]`.

### 4.3 The free lower bound

`LB(C,a,b) = max{Σᵢaᵢ minⱼ Cᵢⱼ, Σⱼbⱼ minᵢ Cᵢⱼ}` comes from a feasible Kantorovich
dual pair (`fᵢ = minⱼ Cᵢⱼ, g ≡ 0`), costs `O(nm)` on a matrix we already hold, and
gives a certified test: **any OT value below LB is a bug, not an improvement.** Every
rung is checked against it (`meta["feasible"]`), and it is asserted as invariant
`LB` on every benchmark cell. This is the cheapest available guard against reporting
an "improvement" that is really a violated bound.

## 5. TF-SPR and TF-UFlow

**TF-SPR** fuses neighbouring ladder plans. Three corrections are baked in:

1. **No ICQ "lower bound" weighting.** For any non-negative regulariser `R`,
   `min(cost + R) ≥ min(cost)`, so entropic/quadratic regularisation gives an
   **upper** bound. Using ICQ as a "never underestimates" criterion inverts the
   direction. It is recorded in `meta["icq_note"]` and excluded from the weights.
2. **Fusion is at the plan level, never the map level.** A convex combination of
   plans stays inside the transport polytope and preserves marginals exactly; a
   convex combination of barycentric maps preserves nothing.
3. **Honest expectation.** A convex blend cannot beat the best single rung — it
   destroys each rung's KL optimality. `meta["honest_expectation"]` says so. The
   available gain is variance reduction, not peak quality.

Weights come from the debiased Sinkhorn divergence with a data-derived temperature
(`γ = median`), so there is no temperature hyper-parameter a human could tune to
manufacture a win.

**TF-UFlow** solves the KL-relaxed unbalanced problem with

```
ζ = ετ/(τ+ε),    f ← ε·log a − ζ·LSE_j((g−C)/ε),    g ← ε·log b − ζ·LSE_i((f−C)/ε)
```

The `τ → ∞` limit collapses onto balanced Sinkhorn exactly, and that limit is what
caught a missing `ε·log(a)` term producing plans **56× too expensive**. A safety
valve reports `balanced_would_be_better` when `1 − ‖π‖₁` is negligible, so UOT never
silently pays its tax on clean data.

## 6. Bugs this project found in its own design

Each was caught by an invariant, not by inspection. They are listed in `CHANGELOG.md`
with full detail; the instructive ones:

* **Residual measured on the wrong side.** After the `f`-update the row marginal is
  exact *by construction*, and after the `g`-update the column marginal is. Measuring
  there returns 0 for any iterate. The loop exited after **one** iteration having
  produced garbage. Fixed by measuring against the *stale* side with
  `expm1(r/ε)` — `exp(x)−1` loses all precision exactly in the converged regime.
* **The sharpness proxy ran backwards.** The original `-Σρ log ρ / log(nm)` with
  `ρ = π/(a⊗b)` is 0 for the *independent* plan and negative for a sharp one — it is
  an unnormalised cross-entropy, not an entropy. It silently disabled the anti-collapse
  criterion. Replaced with the KL form `KL(π|a⊗b)/log(nm)`.
* **`effective_support` used an unnormalised ratio.** `Σρ = n·m`, so the "participation
  ratio" returned `exp(−n² log n) ≈ 1e-28` instead of `n`.
* **The monotonicity check measured the wrong direction** — `max(diff)` flags the
  legitimate monotone *rise*; a violation is a *decrease*.
* **Invariant I4 compared squared cost against a distance**, showing a spurious 38.7%
  "error" on a correct solve.
* **The G2 baseline was already scale-adaptive** (`0.05·median(C)`), making the gate
  vacuous — identical 6.8189% error at every scale.
* **`measure_cells` ignored its `scales` argument**, reporting 5 scales in the metadata
  while measuring all 7 (21 cells instead of 15, and a 168 s demo against a 60 s budget).

## 7. Determinism

`core.seed.set_all(seed, n_threads=1)` is the only entry point. It pins every BLAS
backend to one thread (multi-threaded reductions change the summation order and
therefore the last bits), seeds `random` and numpy's legacy global state, and
funnels all randomness through `np.random.default_rng`. Invariant I9 compares two
runs **bitwise** via `eval.report.core_metrics()`, which strips wall-clock fields by
name and retains everything else verbatim.

## 8. Degradation

| Backend missing | Effect |
|---|---|
| POT | exact-reference gates report `skipped`; `benchmark.json["skipped"]` records it. **No number is fabricated.** |
| scipy | invariant I5 falls back to the POT reference alone |
| numpy | hard failure `E1xx` (there is no project without it) |

`TRANSFORGE_DISABLE_POT=1` exercises this path; CI runs it as its own job and asserts
the output *declares* its degradation.

## 9. References

* Cuturi (2013). *Sinkhorn Distances*. NeurIPS 26. https://proceedings.neurips.cc/paper/2013/hash/af20d5a60f7964b99d334ca899c9c993-Abstract.html
* Schmitzer (2016/2019). *Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems*. https://arxiv.org/abs/1610.06519
* Chizat, Peyré, Schmitzer, Vialard (2018). *Scaling Algorithms for Unbalanced Transport Problems*. Math. Comp. 87(314). https://arxiv.org/abs/1607.05816
* Feydy, Séjourné, Vialard, Amari, Trouvé, Peyré (2019). *Sinkhorn Divergences*. AISTATS. https://proceedings.mlr.press/v89/feydy19a.html
* Pooladian, Cuturi, Niles-Weed (2022). *Debiaser Beware*. https://arxiv.org/abs/2202.08919
* Chizat, Roussillon, Léger, Vialard, Peyré (2020). *Faster Wasserstein Distance Estimation with the Sinkhorn Divergence*. NeurIPS 33:2257–2269.
* Flamary et al. *Python Optimal Transport (POT)*. https://pythonot.github.io/