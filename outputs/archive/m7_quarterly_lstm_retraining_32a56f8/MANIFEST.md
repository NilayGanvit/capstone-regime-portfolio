# Fixed versus quarterly-retrained LSTM (Colab, commit 32a56f8) — manifest

## Status

The source of Appendix A.1 (quarterly LSTM retraining) in the M7 report. It is a secondary
robustness check, pre-registered in
`docs/verification/quarterly_lstm_retraining_preregistration.md`; it selects nothing and the
fixed-parameter LSTM remains the primary specification. The October file
`lstm_cadence_validation_diagnostics_PR3_8515355.csv` (ordinary Softmax, SLSQP, zero risk-free
rate, validation only) is superseded by this run.

## Code and environment

- Commit `32a56f84780b9dee7d225486fb7f7cb1b3185613` on `verification/quarterly-lstm-retraining`,
  whose `src/`, `scripts/run_real_data.py` and lock file are identical to the M6 reference
  commit `6a6b12d`. No uncommitted files at run time (`record.json`).
- Command: `python scripts/verification/run_quarterly_lstm_retraining.py`, run by the
  `Capstone_QuarterlyRetrain_Run.ipynb` Colab notebook.
- Google Colab CPU runtime, x86_64, run 2026-11-02. Python 3.13.18 (the M6 run used 3.13.16),
  torch 2.11.0+cpu, cvxpy 1.9.3, cvxpylayers 1.2.0, diffcp 1.1.9, numpy 2.1.3, pandas 2.2.3,
  scipy 1.16.3, scikit-learn 1.6.1. The packages equal the M6 reference run's.

## Specification

Seed 123 trained once with every M6 setting, then two walk-forwards from separate copies of the
same networks:

- **fixed:** the M6 reference specification (no LSTM retraining);
- **quarterly:** `lstm_retrain_every_n_rebalances=3`, warm-started on the expanding window,
  50 epochs per retrain. 47 retrains: 16 in validation, 30 in the final test, and one on the
  last data day (2026-09-21), after the evaluation period.

Everything else is `run_primary_walk_forward`'s: exact solver, 30% caps, next-open execution,
quarterly HMM refits.

## Check (passed)

- The seed-123 checkpoints have the M6 reference hashes (`aa328280...`, `757819e8...`).
- The fixed arm reproduces the M6 archive's final-test 5 bps net returns for all seven
  configurations; the worst daily difference is 2.4e-16.

## Contents

| Path | What it is |
|---|---|
| `lstm_fixed_vs_quarterly_performance.csv` | Per arm, period and LSTM configuration: net excess Sharpe at 5/10/25 bps, CAGR, max drawdown, mean turnover per rebalance decision, turnover-binding count, retrains. Final-test rows are Table A1 |
| `lstm_quarterly_minus_fixed_bootstrap.csv` | Paired stationary-bootstrap 95% intervals for quarterly minus fixed Sharpe |
| `rq_contrasts_by_arm.csv` | Final-test RQ1-RQ3 contrasts per arm |
| `fixed_net_returns_5bps.csv`, `quarterly_net_returns_5bps.csv` | Daily 5 bps net returns of every configuration in each arm |
| `quarterly_retrain_dates.json` | The 47 retrain dates |
| `seed123_baseline.pt`, `seed123_regime.pt` | The initial networks (identical to the M6 reference checkpoints) |
| `record.json` | Check, timing, git state and environment |
| `run.log`, `tests.log`, `install_*.log` | Console output, the pre-run test log and package installation logs |
| `environment.json`, `reference_manifest.json`, `execution_status.json` | Written by the notebook: the session's environment, the M6 manifest it checked against, and the run status |

Turnover is the mean over the 92 rebalancing decisions dated 2019-01-31 to 2026-08-31, the same
definition `scripts/run_real_data.py` now uses for Table 5.
