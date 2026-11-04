# Pre-registration: fixed versus quarterly-retrained LSTM on the final pipeline

Written and committed before the runner was written or anything was run. Nothing below may be
changed after results are seen; any deviation is reported as a deviation.

## Why

M6 reports that quarterly LSTM retraining improved the baseline LSTM but not the regime-aware
or blended LSTMs, and that it raised turnover. That statement rests on
`lstm_cadence_validation_diagnostics_PR3_8515355.csv` (Drive, Code/runs). The file comes from
commit `8515355` (2 October), which differs from the final pipeline in three ways:
- it uses ordinary Softmax, with no budget floor;
- it uses SLSQP portfolio weights;
- it reports Sharpe with a zero risk-free rate.

It also covers 2015-2018 only. The M7 feedback asks for these numbers in an appendix. This
check recomputes them on the final pipeline.

## Design (fixed)

- **Code:** `m6/final-pipeline`, whose `src/`, `scripts/run_real_data.py` and lock file are
  identical to the M6 reference commit `6a6b12d`. The runner is
  `scripts/verification/run_quarterly_lstm_retraining.py`.
- **Seed:** 123, the M6 reference seed, trained once with `run_real_data.py`'s settings: 200
  epochs, 5% floor, 60-day windows, 252-day covariance window, three states.
- **Two arms**, each starting from its own copy of the same trained networks:
  - **Fixed:** the M6 reference specification, `lstm_retrain_every_n_rebalances=None`.
  - **Quarterly:** `lstm_retrain_every_n_rebalances=3`. At every third month-end decision,
    both networks are retrained, warm-started from their current weights, on the expanding
    window through that date. The walk-forward's own retraining defaults apply: 50 epochs,
    turnover penalty 0.1, the same covariance window and shrinkage.
- **Everything else is `run_primary_walk_forward`'s:** exact solver, 30% caps, next-open
  execution, quarterly HMM refits. The runner passes the same arguments, which a test checks.
- **Environment:** the M6 reference Colab CPU environment, set up by a copy of the team's Colab
  notebook (pinned packages, checked data, frozen risk-free file). A Mac run does not count.

## Check (before any result is read)

The fixed arm must reproduce the M6 reference archive's final-test 5 bps net returns for all
six configurations: within 1e-8 a day for ERC and within 1e-7 for the LSTMs. The seed-123
checkpoints must also have the M6 reference's SHA-256 hashes. If this fails, the session's
environment differs from the reference, and the results are reported with that label.

## What is reported

For the three LSTM configurations, in each arm, for validation (2015-2018) and the final test
(2019-2026):
- net Sharpe in excess of the risk-free rate, at 5, 10 and 25 bps, and CAGR;
- max drawdown;
- mean turnover per rebalance, and how often the turnover limit bound.

Also reported:
- the number of retrains;
- the final-test RQ1-RQ3 bootstrap contrasts for each arm, using the pipeline's
  `stationary_bootstrap_rq_contrasts`;
- paired stationary-bootstrap 95% intervals for quarterly minus fixed, per configuration, on
  the final test.

## How to read it

- **It selects nothing.** Fixed parameters remain the primary specification, as M6 states. This
  is a secondary robustness check and adds no trial.
- **It is one seed.** The ten-seed work shows how much that limits any LSTM conclusion.

## Files

The runner writes only into a new folder, `outputs/verification/quarterly_lstm_retraining/<run id>/`.
It refuses to start if that folder exists, and it never writes a file that already exists.
