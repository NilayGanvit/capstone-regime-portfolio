# M6 reference run (Colab, commit 6a6b12d) — manifest

## Status

The reference run for the M6 draft: bounded Softmax with a 5% floor, the exact
risk-budget solver in training and evaluation, and T-bill excess-return Sharpe ratios.
It is the source of Table 5, Table 6, Figure 4 and the DSR analysis in the M6 draft.
The earlier `frozen_reference_5pct_bb46b63` (SLSQP evaluation), `colab_reference_run_9c17150`
and `m4_reference_run_baf2f0e` (ordinary Softmax, zero risk-free rate) archives are kept as
provenance and are superseded for LSTM results.

The top-level `outputs/real_data_*.csv` files at commit 6a6b12d are **not** this run's
outputs: the cost-sensitivity and bootstrap files there are the `colab_reference_run_9c17150`
results. Cite the files in this directory.

The 2019-01-02 through 2026-08-31 period (1,926 trading days) is an **exploratory
historical walk-forward period**, not an independent holdout, because it was inspected
during model development.

## Code and environment

- Commit `6a6b12de19b567ba7c1feda10bd5423cd72b5e23` on `m6/final-pipeline`, no uncommitted
  files at run time (`run_manifest.json`).
- Command: `python scripts/run_real_data.py`, per `docs/colab_reference_run.md`.
- Google Colab CPU runtime, x86_64, run 2026-10-22. Python 3.13.16, torch 2.11.0+cpu,
  cvxpy 1.9.3, cvxpylayers 1.2.0, numpy 2.1.3, pandas 2.2.3, scipy 1.16.3,
  scikit-learn 1.6.1.

## Specification

Seed 123, 200 epochs, 5% budget floor, exact solver, 3-state HMM refit quarterly, LSTM
trained through 2014-12-31 and frozen. 30% maximum ETF weight and 30% maximum gross
turnover, next-session-open execution, 5 bps headline cost with 10 and 25 bps
sensitivities. Risk-free rate FRED DGS3MO, lagged one observation and compounded over 252
periods. The run used the frozen archive's risk-free file
(`frozen_reference_5pct_bb46b63/data/risk_free_DGS3MO_daily.csv`); the hash in
`run_manifest.json` matches it.

## Contents

| Path | What it is |
|---|---|
| `csv/` | The run's 21 output CSVs: summary (gross), cost sensitivity (net), stationary bootstrap, net and excess return series at 5 bps, reliability path, DSR sensitivity grid, binding constraints and transaction logs for the seven configurations |
| `lstm_checkpoints/` | The two trained networks, `seed123_baseline.pt` and `seed123_regime.pt` |
| `run_manifest.json` | The run's own manifest: commit, environment, settings, SHA-256 of inputs and outputs |
| `reference_run.log` | The run's console output, including the validation-period table |
| `M6_verification_record.md` | The team's check of the archived outputs after the run |

## Checks

- The SHA-256 hashes in `run_manifest.json` match all 21 CSVs and both checkpoints in this
  directory (checked 2026-10-28).
- Replaying the two checkpoints with the original SLSQP solver reproduces the frozen
  archive's seed-123 daily net returns (worst difference 8.7e-09), so the two runs share
  one pair of trained networks and their LSTM rows differ by the solver alone. Replaying
  with the exact solver reproduces this run's own returns (worst difference 7.0e-09).
- The ERC and equal-weight rows match the values expected before the run
  (`docs/colab_reference_run.md`, check 3).

## 5 bps net headline results (excess of the T-bill rate)

| Configuration | Net Sharpe | Net CAGR |
|---|---:|---:|
| erc_baseline | 0.455 | 0.0632 |
| erc_regime | 0.455 | 0.0629 |
| erc_blend | 0.458 | 0.0632 |
| equal_weight | 0.608 | 0.0879 |
| lstm_baseline | 0.540 | 0.0741 |
| lstm_regime | 0.439 | 0.0645 |
| lstm_blend | 0.448 | 0.0652 |

## RQ stationary-bootstrap inference

2,000 replications, contrast-specific automatic block length, 5 bps net excess returns.

| Contrast | Estimate | 95% CI low | 95% CI high | Expected block length |
|---|---:|---:|---:|---:|
| rq1_erc_regime_effect | -0.000565 | -0.082586 | 0.078071 | 4.11 |
| rq1_lstm_regime_effect | -0.100785 | -0.215793 | 0.013763 | 5.81 |
| rq2_difference_in_differences | -0.100221 | -0.193246 | -0.011498 | 19.36 |
| rq3_erc_reliability_effect | 0.003457 | -0.000105 | 0.010166 | 55.33 |
| rq3_lstm_reliability_effect | 0.008245 | -0.001604 | 0.027201 | 63.83 |
