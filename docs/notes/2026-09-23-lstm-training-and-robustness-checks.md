# Note to Silvio & AnnaLisa — LSTM training loop + robustness checks

**From:** Nilay
**Date:** 2026-09-23
**Branch:** `feature/lstm-e2e-training-and-robustness-checks` (pushed to origin, not yet merged — PR link below)
**PR:** https://github.com/NilayGanvit/capstone-regime-portfolio/pull/new/feature/lstm-e2e-training-and-robustness-checks

This covers the two items from the README's "Next steps." One is routine; the other changes what we can currently claim about the regime effect, so please read that part before we write anything into M3/M4 about it.

---

**UPDATE — 2026-09-26 (Nilay).** Section 2 below is superseded — flagged by Silvio as predating the current CSVs. Between then and now: the return-compounding bug got fixed (log returns were being summed instead of simple returns), the initial-fit window changed from an arbitrary 252 days to the actual M2-calendar-derived window (~1,946 days, through 2014-12-31), and reporting is now final-test-only (2019-01-01–2026-08-31) instead of pooled across the whole 2007–2025 sample. The section below is kept as-is for the record; **use the table right here, not the one further down**, if citing this note.

Current numbers (`outputs/robustness_regime_effect_by_spec.csv`, final test only):

| specification | regime effect (erc_regime − erc_baseline, Sharpe) | reliability effect (erc_blend − erc_regime) |
|---|---|---|
| primary (3-state, full universe) | +0.011 (90% bootstrap CI: [-0.055, 0.066] — not distinguishable from zero) | +0.004 |
| 2-state | **+0.042** (was -0.188 — sign flip) | -0.005 |
| no-VNQ | -0.003 (was -0.073 — much smaller) | +0.003 |
| no-HYG | -0.043 (was -0.116 — smaller, same sign) | -0.001 |

What changed and what didn't:
- **The bottom-line conclusion is unchanged**: the regime effect is still small and not distinguishable from zero, and it's still `equal_weight` that leads on Sharpe across every spec, not any ERC variant.
- **The supporting evidence changed enough that the old table would mislead if cited as-is.** 2-state no longer "flips the sign" — it's now positive, same direction as the primary spec. Only no-VNQ/no-HYG still go negative, and by less than before.
- **The BIC argument reverses.** Old note said BIC (on the 252-day window) favored *fewer* states. On the correct ~1,946-day window it's the opposite: `n_states=2: BIC=-140902.3`, `n_states=3: BIC=-142113.2`, `n_states=4: BIC=-142171.0` — more negative is better, so BIC now favors *more* states, which no longer supports the "don't bother with 3 over 2" argument the old note made.
- **The reliability-effect story evaporates.** Old note's "blend consistently pulls back toward baseline" (+0.096, +0.058, +0.055) doesn't hold up — new reliability effects are all near zero (-0.005, +0.003, -0.001). That was a real secondary finding in the old numbers; it isn't one anymore.

Same caveat as before: this is still the initial-window-fit-plus-scheduled-refit, ERC-only pass (see `outputs/trial_log.csv` for the still-open refit-cadence question), gross of transaction costs (see `outputs/real_data_cost_sensitivity.csv` for the primary spec's net-of-cost numbers — the robustness script doesn't yet apply the cost ledger to the alternative specs).

---

## 1. LSTM end-to-end training loop — implemented

`allocation_lstm.py` now has the differentiable optimization layer we'd flagged as an open decision, plus the actual training loop.

- Went with `cvxpylayers`, using the convex log-barrier reformulation of risk budgeting (Spinu 2013; Bai, Scheinberg & Tutuncu 2016) — the same construction Uysal, Li & Mulvey (2021) build their own differentiable layer on. Its first-order condition guarantees the risk-contribution shares match the target budgets exactly, which is why it won out over a hand-rolled differentiable proxy.
- Two things worth knowing if either of you touches this layer: (1) it deliberately has **no** `sum(w)==1` equality constraint and **no** 0.60 per-asset cap inside the convex problem — both were tried, and both made CLARABEL/SCS converge to spurious (verifiably non-optimal) points. Scale is instead pinned by the log barrier itself and normalized after solving. (2) The 0.60 cap and turnover limit still apply exactly as before, but only downstream, at backtest time, via the existing exact SLSQP solve.
- Byproduct finding, not something this work introduced: `allocation_erc.solve_risk_budget`'s SLSQP objective is non-convex and **does land in poor local optima on the real covariance** — e.g. plain equal-budget ERC pins three assets at exactly zero weight despite equal target budgets. This is pre-existing (it's what our current ERC walk-forward numbers are already built on), not a regression from this branch. Flagged in code (`budgets_to_weights_batch`'s docstring) and in the README. Worth a look at some point, but out of scope for this pass.
- New functions: `differentiable_risk_budget_layer`, `budgets_to_weights_differentiable`, `build_lstm_training_windows`, `train_lstm_allocator`. 8 new tests, full suite is 65 passed / 2 skipped (skips are just torch/cvxpylayers-absence checks, which don't apply once those packages are installed — added `cvxpylayers` to requirements.txt).
- **Not done yet:** the trained LSTM isn't wired into `run_walk_forward` as `lstm_baseline`/`lstm_regime`/`lstm_blend` configs. That's the next real step if we want the full six-configuration M2 table.

## 2. Robustness checks — run, and they don't hold up

Ran `scripts/run_robustness_checks.py` (new) on the real ten-ETF universe, 2007–2025, 4640 trading days: 2-state HMM, no-VNQ, no-HYG, alongside the primary 3-state/full-universe spec, all through the same ERC walk-forward harness.

**The primary specification's regime effect is not robust.**

| specification | regime effect (erc_regime − erc_baseline, Sharpe) |
|---|---|
| primary (3-state, full universe) | +0.013 (90% bootstrap CI: [-0.091, 0.133] — not distinguishable from zero) |
| 2-state | **-0.188** |
| no-VNQ | **-0.073** |
| no-HYG | **-0.116** |

Every alternative flips the sign, and even the primary spec's own confidence interval straddles zero. On top of that, the BIC comparison across candidate state counts on the real data actually favors **fewer** states, not more (BIC: -17648 at k=2, -17396 at k=3, -17161 at k=4 — lower is better), which cuts against picking 3 states over 2 independent of the Sharpe result.

Reliability blending's effect is more consistent in direction (it pulls back toward baseline in every non-primary spec: +0.096, +0.058, +0.055), which is at least a coherent story — the blend is doing what it's supposed to when the raw regime signal is unreliable — but that's a smaller, secondary point next to the regime-effect result above.

Full numbers: `outputs/robustness_{primary_3state,2state,no_vnq,no_hyg}_summary.csv` and `outputs/robustness_regime_effect_by_spec.csv`.

**What I think this means for M3/M4:** we probably can't write "regime information improves risk-adjusted performance" as a clean positive finding on the ERC side. The honest framing is closer to "the regime effect is small, sign-unstable across specifications, and not distinguishable from zero" — which is itself a legitimate answer to research question 1, just not the one we might have hoped for. Might be worth checking whether the LSTM allocator (once wired in) tells a different story, per research question 2, before we settle on how to write this up.

Happy to talk through this on our next call — didn't want to just quietly change the framing without flagging it first.
