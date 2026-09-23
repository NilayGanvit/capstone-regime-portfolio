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

## Status (as of this scaffold)

**Working end-to-end on synthetic data**, with a full pytest suite (43
tests passing). **Not yet run on the real ten-ETF universe** — this
development environment has no network access, so real price data has
not been pulled in. See "Next steps" below.

```
python scripts/run_smoke_test.py
```
runs the whole pipeline on a synthetic 2-regime dataset (see
`src/data.py::make_synthetic_universe`) and writes a results table to
`outputs/smoke_test_summary.csv`. Read the header comment in that script
before citing any number from it — it exists to prove the plumbing
works, not to say anything about the real research questions yet.

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
pieces. It is **not installed in the sandbox this scaffold was built in**
(no network access there), so the two LSTM-specific tests
(`test_allocation_lstm.py::test_lstm_forward_pass_shape` and
`::test_sharpe_turnover_loss_runs`) will show as skipped until you run
the suite in your own environment with `torch` installed. Everything
else — HMM, densities, reliability, ERC, constraints, evaluation, and
the full ERC-only walk-forward harness — has been run and verified in
this sandbox and does not depend on torch.

## Known scope limitations (intentional, not oversights)

- **Scheduled model refit is not yet implemented.** The walk-forward
  harness currently fits the HMM/M0/M1 once on the initial window and
  holds parameters fixed through the loop. The M2 pseudocode calls for
  periodic re-fitting using only `F(t)`. This is the most important
  next addition to `walkforward.py`.
- **The LSTM allocator's training loop is not implemented**, only the
  model, the loss function, and the (non-differentiable) bridge from
  learned budgets to weights via the shared risk-budgeting solver.
  End-to-end training through the optimization layer, the way Uysal,
  Li & Mulvey (2021) do it, needs a differentiable convex-optimization
  layer (e.g. `cvxpylayers`), which is not installed here and is flagged
  as its own follow-on task rather than approximated silently.
- **Real price data.** `data.py::PriceDataset.from_csv_dir` loads real ETF
  CSVs from `data/raw/` (one file per ticker, `Date`/`AdjClose` columns,
  plus an optional `AdjOpen` column -- adjusted the same way as `AdjClose`
  -- that enables next-session-open execution timing; without it the
  harness falls back to same-close execution). `make_synthetic_universe`
  still exists for building/testing the pipeline without real data.
- **HRP and per-regime ν are out of scope for this pass**, per M2's
  scope note — secondary robustness checks only if time allows.

## Next steps

1. Pull real ETF price histories (April 2007 onward per M2) into
   `data/raw/`, one CSV per ticker.
2. Add the scheduled-refit step to `walkforward.py`.
3. Decide on the differentiable-optimization-layer approach for LSTM
   training (`cvxpylayers`, or a documented simpler proxy) and implement
   the actual training loop.
4. Re-run `scripts/run_smoke_test.py` against real data and replace the
   "synthetic" framing everywhere it appears once that happens.
5. Wire in the 2-state HMM and no-VNQ / no-HYG robustness checks (M3
   "Scope and feasibility").
