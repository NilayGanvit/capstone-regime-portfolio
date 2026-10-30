# Pre-registration: regime LSTM with per-asset regime inputs (trial 20)

Written and committed before the method was implemented or run. Nothing below may be changed
after results are seen; any deviation is reported as a deviation.

## Why

Trial 18 gave each asset its own 2-state regime model and used the forecasts in ERC. The
per-asset regimes did differ from one joint regime: each asset's more likely state matched
SPY's at only 66-90% of month-ends. But ERC's weights barely moved, because ERC uses the
regimes only through the covariance.

The regime LSTM can use regimes more directly: they are among its inputs. It currently
receives the joint 3-state HMM's three state probabilities. This trial gives it each asset's
own high-variance probability instead, ten inputs, and retrains it.

## Method (fixed)

- **Per-asset regime models** exactly as in trial 18: a 2-state univariate Gaussian HMM per
  asset on daily log returns x100, fitted on the initial window, refitted on the pipeline's
  refit dates with warm starts.
- **Inputs.** For asset i on day t: P(S_i,t+1 = high-variance state | F_t), where the
  high-variance state is the state with the larger variance in the current fit. Over the
  training window these come from the initial fits, as the joint probabilities do for the
  regime LSTM now. During the walk-forward they are recorded day by day from the fit in use
  that day, a fixed causal record, as the joint probabilities are.
- **Network `lstm_asset_regime`.** The regime LSTM's recipe unchanged
  (`lstm_training.train_lstm_variants`: bounded Softmax with a 5% floor, 200 epochs, 60-day
  input window, 252-day training covariance, turnover penalty 0.1, shrinkage 0.10), trained
  on data through 2014-12-31 only, never retrained. Its inputs are the baseline features
  plus the ten per-asset probabilities, in place of the three joint probabilities.
- **Budgets to weights.** Against trial 18's per-asset regime covariance (pooled
  correlations, per-asset forecast variances, regularised as for ERC), with the exact
  solver, as `lstm_regime` uses the joint regime covariance. Then the pipeline's constraints,
  next-session-open execution and costs.
- **Seeds.** Seeds 0-9, the project's ten-seed design. For each seed, all three networks
  (`baseline`, `regime`, `asset_regime`) are trained in the same run on the same machine, so
  every comparison is within one environment. LSTM training results depend on the machine
  (see `AGENTS.md`); these are Mac results, not Colab-canonical numbers.

Run settings: the primary walk-forward (`scripts/run_real_data.py:run_primary_walk_forward`,
exact solver) with `asset_regime_states=2`. Risk-free:
`outputs/archive/frozen_reference_5pct_bb46b63/data/risk_free_DGS3MO_daily.csv`.

## Checks (before any result is read)

- A unit test shows that the day-by-day per-asset probabilities equal those computed from
  the full history at every day.
- Adding the new network changes nothing else: for every seed, the ERC and 1/N returns must
  match the M6 reference archive's final-test 5 bps net returns within 1e-8 a day, and
  `lstm_baseline`/`lstm_regime` must be identical to a run of the same networks without the
  new one (checked on one seed). If either fails, the trial stops and the failure is
  reported.

## Gates (validation 2015-01-01 to 2018-12-31, net of 5 bps)

Sharpe is annualised and in excess of the risk-free series; each statistic is the mean over
the ten seeds. The trial passes only if all hold:

1. Mean Sharpe(`lstm_asset_regime`) - Sharpe(`lstm_baseline`) >= 0: per-asset regime inputs
   add to the LSTM without regime inputs.
2. Mean Sharpe(`lstm_asset_regime`) - Sharpe(`lstm_regime`) >= 0: they do at least as well as
   the joint regime inputs.
3. Mean max drawdown of `lstm_asset_regime` no more than 2 pp worse than `lstm_baseline`'s.

Validation has been used by earlier trials; this is not a fresh sample.

Power, stated in advance: the LSTM's regime effect varied with a standard deviation of about
0.12 across seeds in the project's 15-seed check, so a ten-seed mean carries about 0.04 of
seed noise. All seeds share the same four validation years, whose own sampling noise is
larger (a single seed's paired interval was about +/-0.3). Only a large gain could be
distinguished from noise; passing is weak evidence and failing does not rule out a small
gain.

## Reporting (whatever the gates say)

For validation and, labelled exploratory, the final test (2019-01-01 to 2026-08-31):

- For `lstm_asset_regime`, `lstm_baseline`, `lstm_regime` and, for reference, `erc_asset_regime`
  and 1/N: Sharpe at 5, 10 and 25 bps, CAGR, max drawdown, turnover per rebalance; mean and
  standard error across seeds.
- Each gate difference per seed, its mean and standard error, and how many of the ten seeds
  are at least 0.
- Daily net returns at 5 bps per seed, and the trained networks, saved for replay.

## Trial accounting

Logged as trial 20 in `outputs/trial_log.csv` with `counts_toward_dsr_trials=True`.
Retraining `lstm_baseline` and `lstm_regime` for the same seeds is the comparison, not a
separate trial.
