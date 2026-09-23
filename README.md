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
torch-absence check for `allocation_lstm.py`, which isn't wired into
this branch's harness). The ERC baseline/regime/reliability-blend
walk-forward harness has been run on real data with M2's calendar
boundaries (`data.M2Calendar`), next-session-open execution, a
transaction-cost ledger (5/10/25 bps sensitivity), the Deflated Sharpe
Ratio, and risk-contribution diagnostics all wired in, **plus scheduled
model refit** (see below) -- see "Next steps" below for what's still not
on this branch (LSTM training/integration and robustness checks live on
the LSTM feature branches).

```
python scripts/run_smoke_test.py       # synthetic 2-regime data, plumbing check
python scripts/run_real_data.py        # real ten-ETF universe, primary 3-state spec
```
`run_smoke_test.py` runs the whole pipeline on a synthetic dataset (see
`src/data.py::make_synthetic_universe`) and writes a results table to
`outputs/smoke_test_summary.csv`; read the header comment in that script
before citing any number from it — it exists to prove the plumbing
works, not to say anything about the real research questions. The other
script runs on the real data in `data/raw/` and writes to
`outputs/real_data_summary.csv` (gross) and
`outputs/real_data_cost_sensitivity.csv` (net of costs), among others.

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

`torch` is required only for `allocation_lstm.py`'s neural-network
pieces, and `cvxpylayers` (plus `cvxpy`/`diffcp`) for its end-to-end
training loop; both are installed in this environment (see the LSTM
feature branches for that work) but not used by anything on this
branch's own scripts. If either is missing in your environment, the
corresponding tests skip cleanly rather than failing.

## Known scope limitations (intentional, not oversights)

- **Scheduled model refit is implemented**, expanding window through F(t)
  with state alignment via Bhattacharyya distance + Hungarian assignment
  for label continuity. The EM iteration budget is capped at 50
  iterations per refit; warm-start, incremental filtering, and other
  performance mitigations are deferred pending real-data runtime
  assessment.
- **The LSTM allocator isn't wired into this branch's walk-forward
  harness.** Its model, loss function, and end-to-end differentiable
  training loop (via `cvxpylayers`) are implemented and have been run on
  the real universe -- see `feature/lstm-e2e-training-and-robustness-checks`
  and `feature/lstm-real-data-training` -- but training a model isn't
  the same as backtesting it: it is not yet plugged into
  `run_walk_forward` as additional configs.
- **Real price data.** `data.py::PriceDataset.from_csv_dir` loads real ETF
  CSVs from `data/raw/` (one file per ticker, `Date`/`AdjClose` columns,
  plus an optional `AdjOpen` column -- adjusted the same way as `AdjClose`
  -- that enables next-session-open execution timing; without it the
  harness falls back to same-close execution). `make_synthetic_universe`
  still exists for building/testing the pipeline without real data.
- **HRP and per-regime ν are out of scope for this pass**, per M2's
  scope note — secondary robustness checks only if time allows.

## Next steps

Real ETF data, scheduled refit, the LSTM's differentiable training loop,
and the 2-state/no-VNQ/no-HYG robustness checks are all implemented and
have been run on the real universe -- see `feature/scheduled-refit` and
the LSTM feature branches. What's left:

1. Merge the scheduled-refit and LSTM branches back into `main` once the
   group has settled the open item below.
2. **Settle the refit-cadence question before citing any final-test
   number**: the quarterly default was selected by comparing Sharpe over
   a sample that includes the 2019-2026-08 final-test window, not
   validation-only evidence (see `outputs/trial_log.csv`, trial 6, and
   the M3 draft's comments). Re-derive it on development/validation data
   alone, revert to M2's monthly default, or explicitly reclassify
   current results as exploratory -- pick one before M4.
3. Wire the trained LSTM into `run_walk_forward` as `lstm_baseline`/
   `lstm_regime`/`lstm_blend` configs alongside the ERC ones.
4. HRP and per-regime ν remain out of scope per M2's own scope note --
   secondary robustness checks only if time allows.
