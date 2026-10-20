# Frozen 5% bounded-Softmax reference run (Colab) — manifest

## Status

The Colab reference run for the proposed specification: bounded Softmax with a
5% minimum risk-budget floor, T-bill excess-return Sharpe ratios, and the
bootstrap on the same net-return series as the headline table. It is the
common reporting reference proposed for Tables 5 and 6. The earlier
`m4_reference_run_baf2f0e` (Mac, ordinary Softmax, zero risk-free rate) and
`colab_reference_run_9c17150` (Colab, ordinary Softmax, zero risk-free rate)
archives are kept as provenance and are superseded for LSTM results.

The 2019-01-02 through 2026-08-31 period (1,926 trading days) is an
**exploratory historical walk-forward period**, not an independent holdout,
because it was inspected during model development.

## Contents

| Path | What it is |
|---|---|
| `csv/` | Seed-123 run: summary, cost sensitivity, bootstrap, net and excess return series for six strategies, binding constraints, transaction logs, `frozen_reference_table5_net_metrics.csv`, and the run log |
| `ten_seed/` | Seeds 0-9: per-seed metrics, daily net returns and logs, the 20 trained checkpoints (`seed_NN_baseline.pt`, `seed_NN_regime.pt`), and the aggregated results and summary |
| `code_snapshot/` | The four source files the run used (`allocation_lstm.py`, `evaluation.py`, `walkforward.py`, `scripts/run_real_data.py`) |
| `data/risk_free_DGS3MO_daily.csv` | The risk-free series the run used |
| `manifest.json` | The run's own manifest (commit, specification, file hashes) |
| `requirements-reference-lock.txt` | The pinned environment lock |

## Provenance and a gap to know about

- Recorded commit: `bb46b63f4e8883f365049fcfbb227f13ce691bf0` on
  `feature/lstm-bounded-softmax-stability`.
- **The code actually run is not that commit.** `allocation_lstm.py` and
  `walkforward.py` are byte-identical to `bb46b63`, but `evaluation.py` and
  `scripts/run_real_data.py` were modified and the modifications were never
  committed. `code_snapshot/` is therefore the authoritative source for this run.
  - `evaluation.py` adds `excess_return_sharpe_ratio`, mathematically the same
    as `sharpe_ratio(..., risk_free=...)` on this branch.
  - `run_real_data.py` loads the risk-free file with strict date alignment,
    uses excess returns in the summary, cost table, bootstrap and DSR, hardcodes
    `random_state=123` and `budget_floor=0.05`, adds `sharpe_zero_rf` columns,
    and saves `real_data_rq_excess_returns_5bps.csv`.
  - Its DSR is the older single-trial-count form from before the sensitivity
    grid, so the sensitivity grid is not among this run's outputs. It is
    recomputed from this archive's returns by `scripts/compute_frozen_dsr_grid.py`
    into `outputs/frozen_reference_dsr_sensitivity_grid.csv` (the equal-weight
    series, absent from the archived return files, is regenerated and
    cross-checked against the archived ERC series and Sharpe).
- The manifest's hashes of the six files it lists were verified against the
  files in this directory (all match, 2026-10-20).

## Specification (from the run's manifest)

LSTM floor 0.05, seed 123, 200 epochs; 5 bps headline cost; risk-free rate FRED
DGS3MO, lagged one observation and compounded to a daily rate. The manifest
calls the conversion "252-trading-day approximation"; checking the file
shows `rf_daily = (1 + y/100) ** (1/252) - 1` (maximum error 5e-13 against the
file), i.e. compounded, not `y/100/252`.

## Environment

The run targets the pinned Colab environment in
`requirements-reference-lock.txt` (CPU runtime, x86_64). The run's manifest does
not record the interpreter or package versions it actually executed under, so
that is **not independently verified** from this archive.

## Reproducibility notes

- Seed-123 results reproduce exactly from the saved checkpoints in Colab
  (all 1,926 daily net returns for three LSTM variants), per the verification
  report in `docs/verification/`. Retraining in another environment does not
  reproduce them: on the same seed, Mac and Colab training give uncorrelated
  LSTM outcomes (Spearman about 0 across seeds 0-9), although the distributions
  across seeds agree.
- ERC and equal-weight rows do not depend on the LSTM and reproduce exactly on a
  Mac when `scripts/run_real_data.py` is run with this archive's risk-free file.
- Table 5's net-metrics file (`csv/frozen_reference_table5_net_metrics.csv`)
  lists the six strategies only. The equal-weight benchmark is in
  `csv/real_data_summary.csv` and `csv/real_data_cost_sensitivity.csv`
  (5 bps net excess Sharpe 0.608).
- The daily rate file differs from one rebuilt today by
  `scripts/fetch_risk_free.py` on 4 of 4,892 rows (FRED revisions; at most
  0.04 percentage points in the yield).
