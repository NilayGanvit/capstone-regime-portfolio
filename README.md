# capstone-regime-portfolio

**Reliability-Gated, Regime-Aware Dynamic Risk Budgeting**
MScFE 690 Capstone — Track 7 (Medium/Long-Term Trading Models) + Track 8.2 (ML/DL Portfolio Allocation)

Silvio Mazzaro · AnnaLisa Aaron · Nilay Jayantibhai Ganvit

This repository implements the architecture specified in the M2 problem
statement and extended in the M3 literature review: a Hidden Markov
regime model, two allocators (ERC risk parity and an LSTM dynamic
risk-budgeting network) each with baseline/regime-aware variants, a
Dynamic-Model-Averaging reliability layer that blends them, shared
constraints, and a walk-forward evaluation harness.

## Status

**Working end-to-end on both synthetic and the real ten-ETF universe**,
with a full pytest suite passing (the only skip is the pre-existing
torch/cvxpylayers-absence check, which does not apply once those
packages are installed, per requirements.txt). The walk-forward
harness -- with M2's calendar boundaries (`data.M2Calendar`),
next-session-open execution, a transaction-cost ledger (5/10/25 bps
sensitivity), the Deflated Sharpe Ratio, risk-contribution diagnostics,
and scheduled model refit all wired in -- now produces all six M2
evaluation-table configs (ERC and LSTM, each baseline/regime/blend) plus
the equal-weight benchmark from one run (`scripts/run_real_data.py`,
which trains the LSTM on the initial window and passes it into
`run_walk_forward`). The pre-specified 2-state/no-VNQ/no-HYG robustness
checks, and the two secondary robustness checks M2's own scope note
deferred (HRP, per-regime ν), have all been run on real data (see "Next
steps" below for what's still open: the refit-cadence question and
feeding the robustness numbers into the M3 lit-review writeup).

```
python scripts/run_smoke_test.py       # synthetic 2-regime data, plumbing check
python scripts/run_real_data.py        # real ten-ETF universe, all 6 M2 configs + benchmark
python scripts/run_robustness_checks.py  # 2-state HMM, no-VNQ, no-HYG
python scripts/run_hrp_and_nu_robustness.py  # HRP vs ERC, shared vs per-regime nu
```
`run_smoke_test.py` runs the whole pipeline on a synthetic dataset (see
`src/data.py::make_synthetic_universe`) and writes a results table to
`outputs/smoke_test_summary.csv`; read the header comment in that script
before citing any number from it — it exists to prove the plumbing
works, not to say anything about the real research questions. The other
scripts run on the real data in `data/raw/` and write to
`outputs/real_data_summary.csv` (gross), `real_data_cost_sensitivity.csv`
(net of costs), and `outputs/robustness_*.csv`, among others.

## Module map and ownership

Matches the division of labor from M2/M3 group discussion, so each
module's docstring also states who owns it and which section of the M2
pseudocode / M3 lit review it implements.

| Module | Owner | Implements |
|---|---|---|
| `src/data.py` | Nilay | Price loading, returns, point-in-time (`as_of`) access, synthetic-data generator |
| `src/features.py` | Silvio | Candidate HMM feature classes (returns, vol, trend, correlation, bond dynamics) |
| `src/regime.py` | Silvio | Gaussian HMM (EM/Baum-Welch), filtered/predicted/smoothed probabilities, BIC state-count selection |
| `src/densities.py` | Silvio | M0 (pooled Student-t) / M1 (regime-mixture Student-t) predictive densities, shared ν |
| `src/reliability.py` | Silvio | DMA π_t recursive update, weight blending, forgetting-factor selection |
| `src/allocation_erc.py` | AnnaLisa & Nilay | ERC risk parity (baseline + regime-aware), shared risk-budgeting solver |
| `src/allocation_hrp.py` | AnnaLisa & Nilay | Hierarchical Risk Parity (baseline + regime-aware) -- secondary robustness check on ERC's construction |
| `src/allocation_lstm.py` | AnnaLisa & Nilay | LSTM → softmax risk budgets → shared risk-budgeting layer; Sharpe+turnover loss |
| `src/constraints.py` | Nilay | Joint projection onto long-only/weight/turnover limits |
| `src/evaluation.py` | Nilay | Sharpe/CAGR/MDD/Sortino/Calmar, block bootstrap, Deflated Sharpe Ratio |
| `src/walkforward.py` | Nilay | Orchestrates all of the above per the M2 pseudocode's day-by-day ordering |

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
python scripts/run_smoke_test.py
```

`torch` and `cvxpylayers` are required only for `allocation_lstm.py`'s
neural-network and end-to-end-training pieces (the latter pulls in
`cvxpy`/`diffcp` as its own dependencies). Both are installed and
exercised in this environment; if either is missing in yours, the
corresponding tests skip cleanly rather than failing (see
`allocation_lstm.require_torch` / `require_cvxpylayers`). Everything
else — HMM, densities, reliability, ERC, constraints, evaluation, and
the ERC-only walk-forward harness — does not depend on either package.

## Known scope limitations (intentional, not oversights)

- **Scheduled model refit is implemented**, expanding window through F(t),
  with state alignment via Bhattacharyya distance + Hungarian assignment
  for label continuity. Each refit's EM is warm-started from the previous
  fit rather than a fresh random init (`refit_warm_start`, default True):
  on the real ten-ETF universe, independent random inits landed in a
  different-but-plausible local optimum most months, causing
  `erc_regime`'s target weights/drawdown to whipsaw and making scheduled
  refit underperform the frozen-parameter baseline. Warm-starting closed
  most but not all of that gap; the rest came from refitting quarterly
  rather than monthly (`refit_every_n_rebalances`, default 3) -- fewer
  refits means fewer chances for the HMM to drift, and each refit sees
  3x more new data. Two other mitigations were tried and measured
  **not** to help on top of warm-starting, and are kept only as opt-in
  toggles for reproducibility: EWMA-smoothing `state_covs` across refits
  (`refit_state_cov_ewma`) made `erc_regime`'s Sharpe/drawdown worse, not
  better -- diluting a state's covariance toward its own stale history
  understates current risk right when regime identification is supposed
  to matter; aligning every refit to a single fixed reference instead of
  the previous fit (`refit_align_to_fixed_reference`) was a no-op --
  once warm-starting keeps EM in the same basin, the alignment
  permutation search never actually swaps states either way.
- **Real price data.** `data.py::PriceDataset.from_csv_dir` loads real ETF
  CSVs from `data/raw/` (one file per ticker, `Date`/`AdjClose` columns,
  plus an optional `AdjOpen` column -- adjusted the same way as `AdjClose`
  -- that enables next-session-open execution timing; without it the
  harness falls back to same-close execution). `make_synthetic_universe`
  still exists for building/testing the pipeline without real data.
- **The LSTM allocator's end-to-end training loop is now implemented**
  (`allocation_lstm.train_lstm_allocator`), through a genuinely
  differentiable risk-budgeting layer
  (`allocation_lstm.differentiable_risk_budget_layer`, via
  `cvxpylayers`). It is the convex log-barrier reformulation of risk
  budgeting (Spinu, 2013; Bai, Scheinberg & Tutuncu, 2016) that Uysal,
  Li & Mulvey (2021) build their own differentiable layer on, chosen
  over an ad hoc differentiable proxy specifically because its
  first-order condition guarantees the (normalized) output's risk
  contribution shares equal the input budgets exactly. That layer is
  training-time only, deliberately with no `sum(w)==1` equality and no
  per-asset upper-bound constraint inside the convex problem itself
  (both were tried and measured to make the solver -- CLARABEL and SCS
  alike -- converge to spurious boundary points rather than the true
  optimum; see that function's docstring for the specific numbers). The
  0.60 per-asset cap and turnover limit continue to apply only at
  evaluation/backtest time, downstream of the *exact* (non-differentiable
  SLSQP) solve in `allocation_erc.solve_risk_budget`, exactly as before.
  While validating this, that exact solver was found to converge to a
  local optimum on the real covariance for at least one plain
  equal-budget ERC case (some assets pinned at zero weight despite a
  positive target budget) -- a pre-existing property of its non-convex
  squared-error objective, not something this change introduced, and
  out of scope to fix here (see `budgets_to_weights_batch`'s docstring).
  Training data assembly (`allocation_lstm.build_lstm_training_windows`)
  uses one shared pooled covariance and one compounded holding-period
  return per monthly rebalance, not a re-implementation of
  `run_walk_forward`'s day-by-day scheduled-refit/regime-mixture-
  covariance machinery.
- **The trained LSTM is now wired into `run_walk_forward`** as
  `lstm_baseline`/`lstm_regime`/`lstm_blend` configs, via the opt-in
  `lstm_models`/`lstm_feature_matrix` parameters (`allocation_lstm.
  predict_budgets` does inference only against the same covariances ERC
  uses each rebalance date; training itself stays outside the harness --
  see `walkforward.py`'s module docstring). `scripts/run_real_data.py`
  trains both variants on the initial training window only and passes
  them in, so one harness run now produces all six M2 evaluation-table
  configs plus the equal-weight benchmark. The LSTM models are never
  retrained mid-walk (unlike the HMM/M0/M1, which refit quarterly) --
  re-deriving an LSTM retrain schedule remains out of scope.
- **HRP and per-regime ν are now implemented as secondary robustness
  checks**, per M2's own scope note, via two more opt-in
  `run_walk_forward` parameters: `include_hrp` (adds `hrp_baseline`/
  `hrp_regime`/`hrp_blend` configs, `src/allocation_hrp.py`'s
  quasi-diagonalization + recursive-bisection construction, against the
  identical covariances ERC uses) and `per_regime_nu` (one M1 mixture
  degrees-of-freedom per HMM state, `densities.fit_per_regime_nu`,
  instead of one nu shared across every state). Neither is a new primary
  allocator/density -- both are off by default and evaluated only via
  `scripts/run_hrp_and_nu_robustness.py`, which compares each against
  ERC's own regime/reliability effect on the real universe's final test.

## Next steps

Real ETF data, scheduled refit, the LSTM's differentiable training loop,
the 2-state/no-VNQ/no-HYG robustness checks, the LSTM's wiring into
`run_walk_forward`, and the HRP/per-regime-ν secondary robustness checks
are all implemented and have been run on the real universe. What's left:

1. Merge this branch (and `feature/scheduled-refit`,
   `feature/lstm-real-data-training`,
   `feature/wire-lstm-and-hrp-nu-checks`) back into `main` once the
   group has settled the open item below.
2. **Settle the refit-cadence question before citing any final-test
   number**: the quarterly default was selected by comparing Sharpe over
   a sample that includes the 2019-2026-08 final-test window, not
   validation-only evidence (see `outputs/trial_log.csv`, trial 6, and
   the M3 draft's comments). Re-derive it on development/validation data
   alone, revert to M2's monthly default, or explicitly reclassify
   current results as exploratory -- pick one before M4.
3. ~~Wire the trained LSTM allocator into `run_walk_forward`~~ -- done
   (`lstm_baseline`/`lstm_regime`/`lstm_blend`, see "Known scope
   limitations" above); `scripts/run_real_data.py` now produces all six
   M2 evaluation-table configs plus the equal-weight benchmark from one
   run.
4. Feed the M3 lit review's regime-effect/reliability-effect/ML-
   contribution analysis (see `docs/problem_statement.md`) with the
   robustness-check numbers in `outputs/robustness_*.csv` (2-state HMM,
   no-VNQ, no-HYG, HRP, per-regime ν, alongside the primary
   3-state/full-universe run) — i.e. is the primary specification's
   regime effect an artifact of the state count, one dominant ETF, the
   ERC construction, or the shared-nu simplification, or does it hold up
   across all specifications?
5. ~~HRP and per-regime ν~~ -- done, as secondary robustness checks
   (`include_hrp`/`per_regime_nu` on `run_walk_forward`, evaluated by
   `scripts/run_hrp_and_nu_robustness.py`); see "Known scope
   limitations" above.
