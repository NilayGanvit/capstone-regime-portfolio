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

## 6. Does the solver failure move the validation-period refit-cadence selection?

Yes. `scripts/verification/refit_cadence_solver_check.py` repeats the three-arm
validation comparison (2015-2018, returns cut at the validation end) under either
solver. With SLSQP it reproduces the saved comparison exactly. `erc_regime`
validation Sharpe, gross with a zero risk-free rate (the basis of the original
selection):

| arm | SLSQP (as frozen) | exact solver | SLSQP dates with an asset at zero weight |
|---|---|---|---|
| no refit | 0.4051 | 0.3864 | 2015-01-30, 2015-08-31, 2016-02-29 |
| monthly | 0.3860 | 0.3860 | none |
| quarterly | 0.4091 | 0.3825 | 2015-01-30 |

- The rule (retain an arm only if it matches or beats no refit) kept quarterly on a
  margin of +0.0041. With the exact solver the margin is -0.0039 and **no arm is
  retained**: the three cadences are indistinguishable (0.386, 0.386, 0.383).
- The failures moved the arms by 0.019 (no refit) and 0.027 (quarterly), five to
  seven times the margin the selection rested on. Monthly, which had no failures,
  is unchanged to four decimals, which is the expected control.
- The margin was never distinguishable from noise: the paired stationary-bootstrap
  interval for quarterly minus no refit is [-0.050, +0.066] with SLSQP and
  [-0.038, +0.032] with the exact solver.
- Excess-return Sharpe tells the same story (SLSQP 0.249 / 0.231 / 0.254; exact
  0.230 / 0.231 / 0.227).
- The failure is not confined to 2015-01-30: the no-refit arm also fails on
  2015-08-31 and 2016-02-29.

What this does and does not change. The frozen reference uses quarterly refitting
as its specification, and that is unaffected. What changes is the justification:
validation evidence does not favour quarterly over monthly or no refit, so the
choice is a free design choice, not an empirical finding. The Sharpes recovered
for trial 6 in `outputs/trial_candidate_sharpes.csv` (0.405, 0.386, 0.409) are the
SLSQP values.

The LSTM floor comparison is checked in section 7.

## 7. The LSTM budget-floor comparison (2.5% vs 5%) under the exact solver

`scripts/verification/lstm_floor_solver_check.py` replays the saved stage-2 raw
budget vectors (every validation rebalance, both floors, seeds 0-4, both variants)
through the walk-forward in place of a model, with the budget-to-weight step done
by SLSQP or by the exact solution. Training is not repeated, so only the solver
changes. The SLSQP replay reproduces the saved validation Sharpe to 6e-7 on all 30
(floor, seed, configuration) cells, which is the faithfulness check.

**SLSQP leaves assets at zero weight on most LSTM weight vectors.** Every learned
budget is at least the floor, so the exact weights are strictly positive; a zero
means the weight vector does not realise the budgets the model learned. Out of 240
validation weight vectors per cell (5 seeds x 48 dates):

| floor | baseline LSTM | regime LSTM |
|---|---|---|
| 2.5% | 205 (85%) | 151 (63%) |
| 5% | 191 (80%) | 116 (48%) |

For comparison, SLSQP failed on 3 of 141 `erc_regime` dates. The LSTM budgets are
concentrated, and this is the regime in which SLSQP fails. This is consistent
with the weight-bound-binding counts already printed by earlier real-data runs
(for example `lstm_baseline`: 87 of 92 final-period rebalances with a weight bound
binding), which were read at the time as constraints binding.

**The floor comparison reverses.** Validation Sharpe, gross with a zero risk-free
rate, mean over seeds 0-4; difference is 5% minus 2.5%:

| | SLSQP (as run) | exact solver |
|---|---|---|
| baseline LSTM, 2.5% / 5% | 0.585 / 0.481 | 0.558 / 0.475 |
| regime LSTM, 2.5% / 5% | 0.514 / 0.531 | 0.526 / 0.456 |
| baseline, 5% - 2.5% | -0.104 (t -2.0) | -0.084 (t -2.5) |
| regime, 5% - 2.5% | +0.016 (t +0.5; 5% better in 3/5 seeds) | **-0.070 (t -7.3; 5% better in 0/5 seeds)** |

With SLSQP the regime LSTM looked roughly tied across floors; with the exact weights
the 2.5% floor is better in every seed. The saved comparison selected 5% for its
lower cross-initialisation budget dispersion (budget dispersion does not depend on
the solver and still favours 5%), not on Sharpe, so the choice is not contradicted;
but Sharpe is described as the primary measure, and on it 5% does not win.

**Scope.** This is five seeds and two floors, on the validation period. The LSTM
Sharpe levels themselves move with the solver (regime LSTM at 5%: 0.531 with SLSQP,
0.456 exact), so the LSTM rows of every table built on SLSQP weights, including the
frozen reference, are exposed to this. Section 8 measures it on the final-test period.

## 8. The frozen LSTM results under the exact solver (final test)

`scripts/verification/lstm_checkpoint_solver_replay.py` loads the ten-seed experiment's
20 archived checkpoints (seeds 0-9, 5% floor), runs each through the full walk-forward,
and converts the same learned budgets to weights by SLSQP or by the exact solution.
Nothing is retrained and the budgets do not change. The SLSQP replay reproduces the
archived daily net returns to 2.3e-7 and the archived Sharpe ratios to 3.9e-7, so the
comparison isolates the solver. 2019-2026, net of 5 bps, in excess of the archived
risk-free series; mean (sd) over seeds 0-9:

| | SLSQP (as frozen) | exact solver | exact minus SLSQP, paired |
|---|---|---|---|
| baseline LSTM Sharpe | +0.516 (0.131) | +0.455 (0.027) | -0.061 (t -1.6) |
| regime LSTM Sharpe | +0.614 (0.056) | +0.509 (0.056) | **-0.106 (t -7.9; lower in 10/10 seeds)** |
| blend LSTM Sharpe | +0.607 (0.063) | +0.505 (0.062) | -0.101 (t -7.2) |
| regime effect | +0.098 (0.159) | +0.053 (0.074) | -0.045 (t -1.0) |
| baseline max drawdown | -23.8% | -22.5% | +1.3 pp |
| regime max drawdown | -20.3% | -22.0% | **-1.7 pp (t -7.0; worse in 10/10)** |
| drawdown improvement (regime vs baseline) | +3.42 pp, 10/10 seeds | **+0.50 pp, 6/10 seeds** | -2.9 pp (t -7.3) |
| regime effect positive in | 7/10 seeds | 6/10 seeds | |

How often SLSQP's weight vector has an asset at zero weight although every budget is at
least 5%, over final-test rebalances: baseline LSTM 56.5% (17-91% across seeds), regime
LSTM 25.8% (16-48%).

Readings, with the limits that apply:

- The headline LSTM Sharpe falls by about 0.1 when the learned budgets are implemented
  exactly. The LSTM was trained through the exact (convex) layer, so the exact weights
  are the ones its training objective assumed; the SLSQP weights are not.
- The regime effect roughly halves and stays statistically indistinguishable from zero
  across seeds; the paired change is not itself significant (t -1.0).
- The drawdown improvement that held in 10 of 10 seeds under SLSQP (3.4 pp) is
  0.5 pp and holds in 6 of 10 under the exact solver. The consistency of that finding
  depended on the solver. The 45-seed Mac drawdown result reported earlier also used
  SLSQP and is exposed in the same way; it has not been repeated.
- Regime LSTM Sharpe under the exact solver (0.509) is below the equal-weight
  benchmark's 0.608 (frozen cost-sensitivity table, 5 bps, excess), where under SLSQP
  it was level with it (0.614). It remains above ERC (0.46).
- Ten seeds, one market history, one checkpoint family. Exact-solver weights are strictly
  positive, so these are more diversified portfolios, not the same portfolios with a
  different number. Not covered: the seed-123 frozen run (no checkpoint is archived for
  it), the ERC-versus-LSTM bootstrap contrasts and DSR on exact-solver returns, and the
  45-seed Mac set.
