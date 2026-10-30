# Results: asset-specific regime detection for ERC (trial 18)

Run exactly as pre-registered in `asset_specific_regimes_preregistration.md` (committed
first, as `a9fc6e0`; implementation committed before the run, `ec2a2e9`), by
`scripts/experiments/run_asset_specific_regimes.py`. No deviations. Outputs:
`outputs/experiments/asset_specific_regimes_results.csv`, `..._regime_occupancy.csv`,
`..._mean_weights.csv`, `..._gates.json`. Net of costs, Sharpe in excess of the M6
reference run's risk-free series.

**Reproduction check: passed.** With the option on, `erc_baseline`, `erc_regime`,
`erc_blend` and `equal_weight` are identical to a run without it, and their final-test
5 bps net returns match the M6 reference archive to 2.6e-16 a day.

## Verdict: passes its validation gates, by margins well inside the noise

| Validation gate (2015-2018, 5 bps) | Result | Pass |
|---|---|---|
| G1: Sharpe at least `erc_baseline`'s | 0.240 vs 0.225 (difference +0.015, 95% interval [-0.115, +0.152]) | yes |
| G2: Sharpe at least `erc_regime`'s | 0.240 vs 0.221 (difference +0.019, 95% interval [-0.132, +0.173]) | yes |
| G3: drawdown within 2 pp of `erc_baseline`'s | -9.6% vs -8.6% | yes |

The differences are about a tenth of their interval widths. Passing is what the
pre-registration called weak evidence: it shows no harm on validation, not a gain.

## Validation and final test

| | `erc_asset_regime` | `erc_baseline` | `erc_regime` | 1/N |
|---|---|---|---|---|
| **Validation 2015-2018** | | | | |
| Sharpe, 5 bps | 0.240 | 0.225 | 0.221 | 0.233 |
| `erc_asset_regime` minus this column, 10 / 25 bps | | +0.009 / -0.012 | +0.015 / +0.004 | -0.001 / -0.025 |
| CAGR | 1.95% | 1.86% | 1.86% | 2.21% |
| Max drawdown | -9.6% | -8.6% | -9.5% | -13.3% |
| Volatility | 5.2% | 5.2% | 5.3% | 6.9% |
| Turnover per rebalance; limit binding | 8.6%; 1 of 48 | 2.6%; 1 of 48 | 5.3%; 1 of 48 | 2.1%; 0 of 48 |
| **Final test 2019-2026 (exploratory, does not count)** | | | | |
| Sharpe, 5 bps | 0.437 | 0.455 | 0.455 | 0.608 |
| `erc_asset_regime` minus this column, 5 bps, 95% interval | | -0.018 [-0.107, +0.069] | -0.017 [-0.106, +0.081] | -0.171 [-0.430, +0.075] |
| `erc_asset_regime` minus this column, 10 / 25 bps | | -0.026 / -0.050 | -0.021 / -0.033 | -0.179 / -0.204 |
| CAGR | 6.11% | 6.32% | 6.29% | 8.79% |
| Max drawdown | -20.6% | -20.2% | -21.0% | -20.0% |
| Turnover per rebalance; limit binding | 12.7%; 3 of 92 | 2.0%; 0 of 92 | 7.9%; 3 of 92 | 2.3%; 0 of 92 |

## What happened

- **The per-asset regimes do differ from one joint regime.** On validation, each asset's
  more-likely state (high or low variance) matched SPY's at 73-90% of rebalances; on the
  final test at 66-87% (gold lowest). Treasuries were in their high-variance state more
  often than equities in 2019-2026 (IEF 36%, TLT 33%, SPY 26% of rebalances).
- **The weights moved less than the regimes.** Average weights stayed close to
  `erc_baseline`'s: equities and REITs 22.5% against 19.8% on validation, 23.4% against
  20.3% on the final test; Treasuries 36.0% against 39.6% on validation. The equity
  underweight that separates ERC from 1/N remains, as the prior said.
- **Turnover rose three- to sixfold.** 8.6% per rebalance against 2.6% on validation,
  12.7% against 2.0% on the final test. At 25 bps this cost turns the validation
  advantage over `erc_baseline` negative (-0.012).
- **The final test does not confirm it.** It trailed both ERC variants by about 0.02 at
  5 bps and 0.03-0.05 at 25 bps, again well inside the intervals. The final test was
  inspected during development and does not count, but it gives no support either.

## Conclusion

Per-asset regime detection passes its pre-registered gates, but the gains are about 0.02
Sharpe against intervals of about +/-0.13, they reverse at 25 bps, and they do not appear on
the final test. It does no harm and does not close the gap to 1/N. The evidence supports
reporting it as a regime specification that performs like `erc_regime`, not replacing
`erc_regime` with it.
