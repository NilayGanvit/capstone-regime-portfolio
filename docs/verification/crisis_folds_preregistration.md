# Pre-registration: do the M6 choices hold in a crisis? Three validation folds on a proxy-extended history

Written and committed before any fold was run. It fixes all three folds and both stages, although
fold A is run first, so nothing later can be adjusted after fold A is seen. Nothing below may be
changed after results are seen; any deviation is reported as a deviation. A first draft, written
on a branch off `experiments/asset-regime-lstm-ensemble`, was replaced before anything ran, so
that the M6 choices are checked on `m6/final-pipeline` first; its comparisons are unchanged here.

## Why

Every M6 choice was made on the 2015-2018 validation period, the calmest in the project's data:

| | SPY volatility | SPY max drawdown | 1/N max drawdown | days with SPY 21-day volatility above 30% | SPY-TLT monthly correlation |
|---|---|---|---|---|---|
| training 2007-2014 | 22% | -55% | -34% | 11% | -0.32 |
| validation 2015-2018 | 14% | -19% | -13% | 1% | -0.26 |
| final test 2019-2026 | 19% | -34% | -20% | 5% | +0.23 |

The final test, with COVID and 2022, has already been inspected, so it cannot show whether the
choices survive stress. This check asks whether they hold in periods where selection never
looked, with the 2008-2009 crisis as fold A. It is a robustness check: it selects nothing, so it
is not logged as a trial, and it does not change the M6 specification or numbers.

## Two stages

1. **Stage 1, on `m6/final-pipeline`:** the M6 choices, with the M6 configurations only.
2. **Stage 2, ported onto `experiments/asset-regime-lstm-ensemble`:** the same folds with the
   per-asset regime configurations added (`erc_asset_regime`, `lstm_asset_regime` and its seed
   ensembles). Stage 1's results must reproduce there first.

## Data

- **History extended to 2000.** Before 2007-04-12, the ten ETFs are replaced by the long-history
  proxies of `docs/experiments/trend_blend_proxy_preregistration.md` (on
  `experiments/trend-blend-proxy`), with the futures treatment that rule selected (both futures
  as is). The proxies' daily log returns on their common trading days run from the first common
  day on or after 2000-08-30 through 2007-04-11. From 2007-04-12 onward, the project's own ETF
  returns are used. Returns are joined, not prices, so no return spans the two sources.
- **Proxies only feed model fitting.** Every fold's validation period starts in 2008 or later, so
  every trade is in the ETFs, at the next day's open, as in the pipeline. The proxies enter only
  the HMM fits, covariances, densities and LSTM training windows.
- **Known weakness.** Mutual-fund prices can be stale, which lowers the measured volatility of the
  bond funds. This affects the fitted models, not any measured return.
- **Risk-free.** The M6 reference run's file,
  `outputs/archive/frozen_reference_5pct_bb46b63/data/risk_free_DGS3MO_daily.csv`. It starts
  2007-04-12 and is needed only for the validation periods.

## Folds

Each fold fits everything on the extended history through its training end, then runs the
walk-forward through its validation end. The data are cut at the validation end; the
walk-forward is causal, so the cut changes nothing before it.

| Fold | Training | Validation | Contains |
|---|---|---|---|
| A | 2000-08 to 2007-12-31 | 2008-01-01 to 2009-12-31 | the financial crisis |
| B | 2000-08 to 2009-12-31 | 2010-01-01 to 2014-12-31 | the 2011 euro crisis, the 2013 taper tantrum |
| C | 2000-08 to 2014-12-31 | 2015-01-01 to 2018-12-31 | the M6 validation period, with the longer training history |

Pipeline: `scripts/run_real_data.py:run_primary_walk_forward`. That means a 3-state joint HMM
refitted quarterly with warm start, alpha 0.97, the exact solver, 0-30% per ETF, 30% gross
turnover and next-open execution. Costs are 5, 10 and 25 bps, with 5 bps the headline.
Sharpe is annualised, net, in excess of the risk-free rate.

LSTMs are trained on each fold's training window only, for seeds 0-9, with the pipeline's recipe
(200 epochs, 60-day windows, 252-day covariance window, turnover penalty 0.1), on a Mac, so their
numbers are not Colab-canonical. ERC and 1/N do not depend on the machine.

- Stage 1: the baseline and regime networks at the 5% floor and at a 2.5% floor.
- Stage 2: the baseline, regime and per-asset regime networks at the 5% floor
  (`train_lstm_variants(asset_regime_states=2)`), and their budget-averaged ensembles of seeds
  0-4, 5-9 and 0-9 (`network_ensemble.MeanBudgetEnsemble`).

## Checks before any fold result is read

Stage 1:

1. The eleven proxy files' SHA-256 hashes match `outputs/experiments/proxy_data_manifest.json`.
2. The spliced returns equal the ETF returns exactly from 2007-04-12, and equal the proxy loader's
   returns before that.
3. Wiring: the fold runner, given the M6 calendar, the ETF data alone and the M6 reference run's
   seed-123 checkpoints, reproduces the M6 archive's final-test 5 bps net returns, within 1e-8 a
   day for `erc_baseline`, `erc_regime`, `erc_blend` and `equal_weight` and within 1e-7 for the
   three LSTM configurations. The same run supplies the 2015-2018 reference values below, in the
   same Sharpe convention.

Stage 2, in addition:

4. Check 3 passes again on the port (with `asset_regime_states=2`).
5. The fold's `erc_baseline`, `erc_regime`, `erc_blend` and `equal_weight` returns are identical to
   stage 1's. The baseline and regime networks retrained for each seed are identical to stage 1's,
   and their walk-forward returns match stage 1's within 1e-7 a day.

If any check fails, the run stops and the failure is reported.

## What is checked in each fold

| Stage | M6 choice | How M6 chose it | Comparison in the fold's validation period | The choice holds if |
|---|---|---|---|---|
| 1 | Quarterly HMM refit | highest `erc_regime` validation Sharpe against monthly refits and no refit (`outputs/refit_cadence_validation_comparison.csv`) | `erc_regime` Sharpe, quarterly against monthly (`refit_every_n_rebalances=1`) and no refit (`refit_at_rebalance=False`) | quarterly is at least both |
| 1 | 5% budget floor | lower budget dispersion across seeds than a 2.5% floor (`outputs/bounded_softmax_stage1_stability_summary.csv`; `docs/verification/solver_and_accounting_checks.md`, section 7) | mean pairwise L1 distance between the ten seeds' budgets, averaged over the validation rebalance dates, 5% against 2.5%, for the baseline and regime networks; seed-paired Sharpe difference reported | 5% has the lower dispersion for both networks |
| 2 | Ten-network ensemble of the per-asset regime LSTM | best LSTM configuration found (`docs/experiments/asset_regime_lstm_ensemble_results.md`) | the all-ten ensemble's `lstm_asset_regime` against the all-ten ensembles' `lstm_baseline` and `lstm_regime`; each half's ensemble (seeds 0-4, 5-9) against the mean of its members run singly | all four differences are at least 0 |

Reported in every fold, not as choices:

- the regime effect: `erc_regime` minus `erc_baseline`, and `lstm_regime` minus `lstm_baseline`
  (10-seed mean);
- in stage 2, `erc_asset_regime` against `erc_baseline` and `erc_regime`, and `lstm_asset_regime`
  (10-seed mean) against the other two LSTMs;
- every configuration against 1/N;
- max drawdown and turnover per rebalance for every configuration;
- the SPY-TLT monthly correlation of the validation period, to show which folds contain a
  positive stock-bond correlation (none is expected to, so a 2022-style year stays untested).

For each comparison the results give its sign in the fold beside its value on 2015-2018. For
LSTM rows in stage 1, that reference is seed 123 alone, the only network with 2015-2018 returns
on `m6/final-pipeline`. Stage 2 adds the ten-seed Mac reference of trial 20. Single-run and
ensemble comparisons get a paired stationary-bootstrap 95% interval
(`evaluation.block_bootstrap_diff_ci`, automatic block length, 2,000 replications). Seed-level
comparisons get the seed mean, its standard error and the count of seeds at least 0.

## How to read it

Power, stated in advance: fold A's validation period is two years and one crisis.

- **Between ERC variants:** a paired interval of about +/-0.16 (the ERC variants' +/-0.11 over four
  years, scaled).
- **Against 1/N:** about +/-0.6.

A choice that flips inside its interval is reported as flipping, not as refuted. A choice that
holds in the crisis fold, as well as in 2015-2018, is supported with more confidence than it is
now. The findings are reported as a robustness section; changing any M6 choice would be a
separate, pre-registered decision.

## Order

Fold A, stage 1, then fold A, stage 2. Folds B and C follow, unchanged, if the team wants them.
