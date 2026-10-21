# Solver and portfolio-accounting checks

Scope: weights, turnover and the NAV ledger. The frozen reference run is not
changed by anything here. Reproduce with `scripts/verification/audit_erc_solver.py`,
`scripts/verification/erc_exact_solver_sensitivity.py`, and
`pytest tests/test_solver_agreement.py tests/test_accounting_reconciliation.py`.

## 1. Accounting reconciliation (`tests/test_accounting_reconciliation.py`)

An independent ledger is rebuilt from the decision-day target weights, the
rebalance dates and the raw asset returns only; it never calls the walk-forward's
accounting code. For every configuration, in both execution modes (same-close and
next-open split-leg), it must reproduce:

- the daily gross return (tolerance 1e-12), including the overnight leg on the
  old holdings and the intraday leg on the new target in next-open mode;
- the holdings at the end of every day (1e-10);
- the recorded turnover of every trade, and its execution date.

A dollar-position ledger (positions grown by their own asset returns, trades to
target at the drifted NAV, fee charged on dollars bought plus sold) must give the
same NAV path as the cumulative product of `net_of_cost_returns`. Two accounting
errors planted on purpose (drifted weights not renormalised; overnight and
intraday legs swapped) are both caught.

Not tested: in next-open mode the fee is charged on the turnover measured at the
decision-day close, not on the trade measured at the next open. This is the
documented convention (turnover is defined on the decision), not an arithmetic
error, but it is an approximation of true at-open trade value.

## 2. Solver agreement (`tests/test_solver_agreement.py`)

Three routes give risk-budgeting weights:

| Route | Where used | Property |
|---|---|---|
| `solve_risk_budget` (SLSQP, squared risk-share error) | ERC and LSTM backtest weights | non-convex; can stop at a local optimum |
| `solve_risk_budget_convex` (log-barrier, NumPy Newton) | new; exact reference | strictly convex, unique, strictly positive; risk shares equal budgets to 1e-10 |
| `differentiable_risk_budget_layer` (cvxpylayers) | LSTM training | the same log-barrier problem; agrees with the exact solver to 1e-4 on the real covariances |

Training therefore optimises through the convex solution, while backtests use
SLSQP. On the three real covariances below they disagree: SLSQP's worst
risk-share error is 0.100, the training layer's is about 1e-4.

## 3. Findings of the independent ERC audit, confirmed on the real walk-forward

Inputs were captured from an ERC-only walk-forward (the same call
`run_real_data.py` makes), 92 final-period decision dates per configuration.

**Finding 1, the raw solver.** Confirmed. Over the final period `erc_regime` has
assets at zero weight on exactly 2020-03-31 (2 assets), 2022-04-29 (2) and
2026-01-30 (1). `erc_baseline` has none on any date. Elsewhere SLSQP is accurate
(median risk-share error 1.6e-7). The independent package covers every rebalance
date, not only the final period, and shows one more failure, in validation:
2015-01-30 (see section 5).
The exact convex solution has worst error 3e-11 on every date. The same solver
produces the LSTM backtest weights, so the LSTM configurations are exposed to it
too; this audit covers ERC only.

**Finding 2, the projection.** It separates into two causes
(`outputs/erc_solver_audit.csv`, worst risk-share deviation from 10%):

| erc_regime date | frozen final | exact target, same projection | best feasible found | exact target feasible? |
|---|---|---|---|---|
| 2020-03-31 | 0.0913 | 0.0000 | 0.0000 | yes, 10.66% turnover |
| 2022-04-29 | 0.1126 | 0.0046 | 0.0030 | no (30.8%) |
| 2022-05-31 | 0.0748 | 0.0748 | 0.0609 | no (45.8%) |
| 2026-01-30 | 0.1025 | 0.0456 | 0.0393 | no (39.8%) |
| 2026-02-27 | 0.0087 | 0.0087 | 0.0075 | no (33.0%) |

- March 2020 is entirely the solver: the exact target needs only 10.66% turnover,
  the projection leaves it unchanged, and parity is exact. The frozen allocation
  traded the full 30% because its raw target had two assets at zero.
- April 2022 and January 2026 have both causes: the exact target still exceeds the
  limit, but exact-plus-projection is far closer to parity than the frozen result.
- The remaining gap on 2022-05-31 and 2026-02-27 comes from the 30% turnover limit
  itself, not from a defect.
- A risk-parity-aware projection (minimising risk-share error inside the feasible
  set instead of distance to the target) improves parity a little further on four
  of the five dates (the "best feasible found" column; a multistart bound, since the
  objective is non-convex). It changes what the constraint step optimises, so it is
  a design change, not a bug fix.

## 4. Does it change the conclusions? (`erc_exact_solver_sensitivity.py`)

ERC with the exact solver, everything else unchanged. Net of 5 bps, excess over the
risk-free rate, 2019-2026:

| | frozen (SLSQP) | exact solver |
|---|---|---|
| erc_baseline Sharpe | 0.4552 | 0.4552 |
| erc_regime Sharpe | 0.4616 | 0.4546 |
| erc_regime max drawdown | -21.29% | -21.01% |
| ERC regime effect (RQ1) | +0.0064 [-0.076, +0.085] | -0.0006 [-0.083, +0.078] |
| RQ2 difference-in-differences | +0.0229 [-0.207, +0.261] | +0.0298 [-0.203, +0.265] |
| RQ3 ERC reliability effect | +0.0034 | +0.0035 |

No conclusion changes: the ERC regime effect is indistinguishable from zero under
either solver. The LSTM rows of the bootstrap are the frozen run's and are not
re-run; the LSTM configurations would need the same solver swap at inference with
their trained models.

## 5. Cross-check against the independent diagnostic package

The independent audit saved the frozen run's `erc_regime` inputs for all 141
rebalance dates (validation and final test): regularised mixture covariances,
SLSQP raw weights, drifted holdings and projected weights. Compared with the
inputs captured by `audit_erc_solver.py` from a separate ERC-only walk-forward:

- Covariances agree to 1e-18, drifted holdings to 1e-10 or better, SLSQP raw
  weights to 5e-10, on every date the two overlap. The two captures come from
  different machines (the frozen Colab run and a fresh Mac run); the ERC path is
  environment-invariant, as expected.
- The saved projected weights equal `project_onto_constraints(raw, drifted)` to
  3e-16, so they come from exactly the projection code tested here.
- The exact solver is accurate to 4e-11 on all 141 covariances.
- Over all 141 dates SLSQP leaves assets at zero weight on four: **2015-01-30**
  (validation, the first rebalance), 2020-03-31, 2022-04-29 and 2026-01-30. The
  2015 date is not in the final-period audit. It is a single date out of 48 in
  the validation period; whether it moves the validation-period comparisons
  (for example the refit-cadence selection) has not been checked. It is included
  as a regression case in `tests/test_solver_agreement.py`.
- Constrained risk-parity search on the five turnover-binding dates: the two
  searches agree on four dates to 4 decimals (frozen deviation 0.0913, 0.1126,
  0.0748, 0.1025; best found 0.0000, 0.0030, 0.0609, 0.0393). On 2026-02-27 the
  independent search found no improvement (0.0087) and this one found a marginal
  0.0075; the objective is non-convex, so neither is a certificate of the optimum.
  "Better parity exists than the frozen allocation" holds on four dates in both,
  and is marginal on the fifth.
