# Pre-registration: asset-specific regime detection for ERC (trial 18)

Written and committed before the method was implemented or run. Nothing below may be changed
after results are seen; any deviation is reported as a deviation.

## Why

The pipeline's regime model is one 3-state Gaussian HMM fitted jointly to all ten assets'
daily returns. Every asset is assumed to be in the same regime on the same day. Volatility
regimes do not have to coincide: Treasuries and gold often turn volatile when equities do
not, and the reverse. `erc_regime` has added nothing over `erc_baseline` (validation
excess Sharpe 0.227 against 0.228 in the M6 reference run; final-test 5 bps net 0.455 and
0.455). One explanation is that a single joint state cannot describe ten different
volatility paths.

This trial gives each asset its own regime model. The request allowed a per-asset LSTM
instead. A per-asset HMM was chosen because:

- the project's LSTMs train on about 80 monthly decisions through 2014 and were already
  seed-unstable with one network; ten networks, one per asset, would each have the same
  80 samples;
- a per-asset HMM is unsupervised, uses the 1,946 daily returns in the initial window,
  and needs no new training targets;
- two-state univariate HMMs fitted separately to each index are an established design for
  regime-based allocation (Bulla et al., 2011, Journal of Asset Management 12(5)).

## Method (fixed)

For each of the ten assets i, separately:

- **Model.** A 2-state univariate Gaussian HMM (`regime.GaussianHMM`, `n_states=2`,
  `random_state=0`) on the asset's daily log return, multiplied by 100 for the fit (the
  HMM adds a 1e-6 variance floor, which is large relative to a Treasury ETF's daily
  variance in decimal units). Two states, as in Bulla et al.; not tuned.
- **Fit and refit.** Fitted on the initial window (the same rows as the joint HMM), then
  refitted on the pipeline's refit dates (every third month-end rebalance) on the
  expanding window through close(t), warm-started from the previous fit with the
  pipeline's `refit_n_iter=50`.
- **Forecast.** At each month-end rebalance t, the one-step-ahead state probabilities
  p_k = P(S_t+1 = k | F_t) from data through close(t), the same horizon `erc_regime` uses.
  The forecast variance is the two-state mixture variance
  v_i = sum_k p_k (sigma_k^2 + mu_k^2) - (sum_k p_k mu_k)^2, rescaled to decimal units.
  It does not depend on how the states are labelled, so no state alignment is needed.

Portfolio, a new configuration `erc_asset_regime`:

- Covariance Sigma = D R D, where R is the correlation matrix of `pooled_cov` (the
  expanding-window sample covariance at the last refit, the input `erc_baseline` uses) and
  D = diag(sqrt(v_1), ..., sqrt(v_10)). A per-asset model has no joint state, so the
  correlations come from the pooled estimate.
- Weights: `erc_baseline_weights(Sigma, shrinkage=0.10, solver="exact")`, the same
  regularisation and solver as `erc_baseline`. Then the pipeline's own constraints (0-30%
  per ETF, 30% gross turnover), next-session-open execution and costs.

So `erc_asset_regime` differs from `erc_baseline` only on the diagonal: forecast regime
variances in place of sample variances.

Run settings: the primary walk-forward (`scripts/run_real_data.py:run_primary_walk_forward`:
3-state joint HMM, alpha 0.97, exact solver) on `data/raw`, without LSTM models.
Risk-free: the M6 reference run's file,
`outputs/archive/frozen_reference_5pct_bb46b63/data/risk_free_DGS3MO_daily.csv`.

## Reproduction check (before any result is read)

Adding the configuration must not change the others. The run's `erc_baseline`,
`erc_regime`, `erc_blend` and `equal_weight` returns must equal a run without the new
option exactly, and their final-test 5 bps net returns must match the M6 reference
archive (`outputs/archive/m6_reference_run_6a6b12d/csv/real_data_rq_net_returns_5bps.csv`)
to within 1e-8 a day. If either fails, the trial stops and the failure is reported.

## Gates (validation 2015-01-01 to 2018-12-31, net of 5 bps)

Sharpe is annualised and in excess of the risk-free series. The trial passes only if all
hold:

1. Sharpe(`erc_asset_regime`) - Sharpe(`erc_baseline`) >= 0: per-asset regimes add to
   ERC without regimes.
2. Sharpe(`erc_asset_regime`) - Sharpe(`erc_regime`) >= 0: per-asset regimes do at least
   as well as the joint HMM.
3. Max drawdown no more than 2 pp worse than `erc_baseline`'s.

Validation has been used by earlier trials; this is not a fresh sample.

Power, stated in advance: the two ERC variants are highly correlated, so the paired
interval is much narrower than for a comparison with 1/N. The M6 reference run's
`erc_regime` minus `erc_baseline` interval was about +/-0.08 over 7.7 years, so roughly
+/-0.11 over 4 years. Only a gain of about 0.1 or more could be significant.

Prior, stated in advance: the forecast variances move ERC toward time-varying inverse
volatility. ERC's structural underweight of equities (the main source of its gap to 1/N
in the attribution analysis) remains, so this is not expected to close the gap to 1/N.
Turnover is expected to be higher than `erc_baseline`'s.

## Reporting (whatever the gates say)

For validation and, labelled exploratory, the final test (2019-01-01 to 2026-08-31):

- Sharpe at 5, 10 and 25 bps; CAGR; max drawdown; volatility; for `erc_asset_regime`,
  `erc_baseline`, `erc_regime` and 1/N.
- Paired stationary-bootstrap 95% intervals (`evaluation.block_bootstrap_diff_ci`,
  automatic block length, 2,000 replications) for the Sharpe difference against
  `erc_baseline` and against `erc_regime`.
- Turnover per rebalance and how often the turnover limit bound.
- Average weight by asset.
- For each asset, the share of rebalances at which its higher-variance state is more
  likely, and how often the assets' high-variance states coincide with SPY's, to show
  whether the per-asset regimes in fact differ from a single joint regime.

## Trial accounting

Logged as trial 18 in `outputs/trial_log.csv` with `counts_toward_dsr_trials=True`.
Trials 12-17 are logged on other experiment branches; 18 is the next unused number.
