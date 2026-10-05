# Colab reference run — manifest

## Status

This archive is the canonical Colab reference-run snapshot generated
after the group selected Colab as the shared reference environment.
It supersedes the specific numerical LSTM results in the earlier
Mac-generated `m4_reference_run_baf2f0e` archive while preserving that
older archive as provenance.

The 2019-01-01 through 2026-08-31 evaluation period is treated as an
**exploratory historical walk-forward period**, not an independent
confirmatory holdout, because results from that period had already
been inspected during model development.

## Code

- Repository: `capstone-regime-portfolio`
- Branch: `feature/reference-run-update`
- Archive-associated commit: `9c171508e80e78e9a87897809caab305a3f1b75f`
- Short commit: `9c17150`
- Command used for the reference pipeline: `python scripts/run_real_data.py`
- The pipeline was executed from the working-tree code state that was
  subsequently committed as `9c17150`; the portfolio outputs were generated
  before the commit was created, while the archived bootstrap outputs were
  generated from the same implementation state.

## Environment

- Platform: Google Colab CPU runtime (`x86_64`)
- Python: `3.13.15`
- torch: `2.11.0+cpu`
- cvxpy: `1.9.3`
- cvxpylayers: `1.2.0`
- numpy: `2.1.3`
- pandas: `2.2.3`
- scipy: `1.16.3`
- scikit-learn: `1.6.1`

The repository's `requirements-reference-lock.txt` defines the pinned
shared reference environment.

## Primary run settings

- Universe: SPY, EFA, EEM, IEF, TLT, LQD, HYG, GLD, DBC, VNQ
- LSTM initialization seed: 123
- LSTM primary specification: trained once on the initial window
  through 2014-12-31 and frozen during the walk-forward
- HMM: 3 states, random state 0
- HMM / density refit cadence: quarterly
  (`refit_every_n_rebalances=3`)
- Constraints: long-only, fully invested, 30% maximum ETF weight,
  30% maximum gross turnover per rebalance
- Execution: next-session open
- Transaction-cost headline: 5 bps on gross turnover
- Cost sensitivities: 5, 10, and 25 bps
- Evaluation period: 2019-01-01 through 2026-08-31
- Evaluation status: exploratory historical walk-forward

## Data reference

SHA-256 hashes of the ten input CSVs:

- `91f41cc75f5a5c8386c80604a19b14606c5f969fadb0547fc385a5fb3086f8c6` — `data/raw/DBC.csv`
- `4e510c770f81a88ddac883815a5b8ceb2e320eb8df94a6cf19ce02bd45896437` — `data/raw/EEM.csv`
- `a83134bb793997bf947f3ccd9114c2c55f92607d6de23e2f4e67d9988b8c26ba` — `data/raw/EFA.csv`
- `219a18c912cfce61223a8de76d0ab9578b2ce7ab89f56c37008522d77746d220` — `data/raw/GLD.csv`
- `4971335eacd0f31118cf845e453859e8dc2445e3eb2ab2891035168f4eb53806` — `data/raw/HYG.csv`
- `d3dd1b0441d6b25075a4e40b373c7339250254abe0315dea13002117de6068ae` — `data/raw/IEF.csv`
- `5346d5545ea7f59f0b6566fedaf45880ca2eebd3d2e2098a2eb566b84bc8e07f` — `data/raw/LQD.csv`
- `8a2763a4fd9efa36143c64377b7dd1546cc5f2d00b24b523e593905532824c4d` — `data/raw/SPY.csv`
- `39b8f220ca191e0aaee32cc652e28a411ebe8ddf29e064d9433f74a3b46ea0c8` — `data/raw/TLT.csv`
- `99bb70759ef2ec9917347077c228390e55b965223cc1585efa2936424788f490` — `data/raw/VNQ.csv`

## Gross headline results

| Configuration | Sharpe | CAGR | Max drawdown |
|---|---:|---:|---:|
| erc_baseline | 0.799 | 0.0634 | -0.2022 |
| erc_regime | 0.810 | 0.0643 | -0.2122 |
| erc_blend | 0.814 | 0.0646 | -0.2122 |
| equal_weight | 0.883 | 0.0881 | -0.1995 |
| lstm_baseline | 0.537 | 0.0559 | -0.2512 |
| lstm_regime | 0.803 | 0.0992 | -0.2729 |
| lstm_blend | 0.798 | 0.0974 | -0.2729 |

## 5 bps net headline results

| Configuration | Net Sharpe | Net CAGR |
|---|---:|---:|
| erc_baseline | 0.797 | 0.0632 |
| erc_regime | 0.804 | 0.0638 |
| erc_blend | 0.807 | 0.0641 |
| equal_weight | 0.881 | 0.0879 |
| lstm_baseline | 0.534 | 0.0555 |
| lstm_regime | 0.797 | 0.0983 |
| lstm_blend | 0.791 | 0.0965 |

## RQ stationary-bootstrap inference

Primary inference uses daily net returns at 5 bps. The stationary
bootstrap uses 2,000 replications. The Politis-White automatic block
length selector with the Patton correction is applied separately to
each pre-specified daily return-contrast series. Within each contrast,
all constituent portfolio return series use the same bootstrap index
path, preserving contemporaneous dependence, and the full nonlinear
Sharpe contrast is recomputed in each replication.

| Contrast | Estimate | 95% CI low | 95% CI high | Expected block length |
|---|---:|---:|---:|---:|
| rq1_erc_regime_effect | 0.006527 | -0.076508 | 0.085225 | 3.849772 |
| rq1_lstm_regime_effect | 0.263071 | 0.030548 | 0.543710 | 19.015252 |
| rq2_difference_in_differences | 0.256543 | 0.025338 | 0.525074 | 23.815884 |
| rq3_erc_reliability_effect | 0.003361 | -0.000094 | 0.009815 | 55.329171 |
| rq3_lstm_reliability_effect | -0.005755 | -0.029514 | 0.018927 | 132.000000 |

The RQ3 LSTM reliability contrast selected the implementation's
finite-sample upper bound of 132 observations. This value is retained
rather than modified after observing the result and is flagged for a
dependence/correlogram and literature diagnostic.

## Archive contents

This archive contains 19 CSV files:

- 17 core portfolio/reference-run output files matching the structure
  of the earlier Mac reference archive
- `real_data_rq_net_returns_5bps.csv`, containing the daily net return
  series used for RQ inference
- `real_data_rq_stationary_bootstrap.csv`, containing the five
  pre-specified Sharpe contrasts, confidence intervals, block lengths,
  and bootstrap metadata

The earlier Mac-generated archive remains in the repository for
historical provenance and should not be deleted.
