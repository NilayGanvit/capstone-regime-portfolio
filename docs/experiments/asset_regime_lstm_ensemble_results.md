# Results: seed ensembles of the per-asset regime LSTM

Trial 20 (`asset_regime_lstm_results.md`) found that the regime LSTM with each asset's own
regime probability as inputs (`lstm_asset_regime`) beat the baseline and joint-regime LSTMs
on average over ten seeds, but unevenly: a few seeds carried the mean. This averages the
networks' budgets across seeds, the plain-average ensemble of
`experiment/shrunk-ensemble-budgets`, to remove that seed noise. Trial 20's thirty trained
networks (Mac) are loaded for inference only; nothing is retrained.

Code: `src/network_ensemble.py` (`MeanBudgetEnsemble`) and
`scripts/experiments/run_asset_regime_lstm_ensemble.py`. Outputs:
`outputs/experiments/asset_regime_lstm_ensemble_{summary,metrics}.csv`, `..._gates.json`,
and each run's returns and records in `outputs/experiments/asset_regime_lstm_ensemble/`.
Net of costs, Sharpe in excess of the M6 reference run's risk-free series.

The runs were made on 2026-10-30 with a walk-forward that also contained other opt-in
options, not on this branch; they were switched off, which leaves the walk-forward's
arithmetic unchanged.

**Gate, fixed before the run** (validation 2015-2018, 5 bps, `lstm_asset_regime`): in each
disjoint half of the seeds (0-4 and 5-9), the ensemble's Sharpe is at least the mean Sharpe of
the same five networks run singly in trial 20.

**Check: passed.** For every seed, replaying its three networks reproduces trial 20's 5 bps
net returns to 7e-8 a day, and ERC and 1/N match the M6 reference archive to 2.6e-16.

## Verdict: passes

Seeds 0-4: 0.392 against its members' mean of 0.385. Seeds 5-9: 0.478 against 0.438.

## Ensemble against its members

Sharpe at 5 bps, the ensemble against the mean of the same networks run singly:

| | validation: ensemble | members' mean | final test: ensemble | members' mean |
|---|---|---|---|---|
| `lstm_asset_regime`, seeds 0-4 | 0.392 | 0.385 | 0.533 | 0.517 |
| `lstm_asset_regime`, seeds 5-9 | 0.478 | 0.438 | 0.535 | 0.523 |
| `lstm_asset_regime`, all ten | 0.441 | 0.412 | 0.534 | 0.520 |
| `lstm_baseline`, all ten | 0.377 | 0.368 | 0.459 | 0.488 |
| `lstm_regime`, all ten | 0.377 | 0.361 | 0.478 | 0.481 |

The final test (2019-2026) is exploratory and does not count.

## The per-asset advantage with seed noise removed

`lstm_asset_regime` minus the other LSTMs within each ensemble, 5 bps, with paired
stationary-bootstrap 95% intervals:

| Ensemble | validation, minus `lstm_regime` | minus `lstm_baseline` | final test, minus `lstm_regime` | minus `lstm_baseline` |
|---|---|---|---|---|
| seeds 0-4 | +0.021 [-0.217, +0.261] | +0.004 [-0.179, +0.189] | +0.042 [-0.096, +0.187] | +0.071 [-0.066, +0.212] |
| seeds 5-9 | +0.098 [-0.159, +0.383] | +0.126 [-0.128, +0.374] | +0.085 [-0.075, +0.243] | +0.065 [-0.092, +0.213] |
| all ten | +0.063 [-0.179, +0.314] | +0.064 [-0.140, +0.287] | +0.056 [-0.089, +0.198] | +0.075 [-0.060, +0.203] |

All-ten ensembles, validation / final test:

- **Max drawdown:** -8.9% / -20.7%, against -8.2% / -22.9% for the baseline ensemble and
  -9.5% / -22.1% for the regime ensemble.
- **Turnover per rebalance:** 13.6% / 18.3%, against 15.4% / 20.0% for the same networks run
  singly.
- **Against 1/N:** the final-test Sharpe is still below 1/N's (0.534 against 0.608).

## What it shows

- **Averaging helped the per-asset LSTM more than the others.** Its ensembles beat their
  members' mean in both halves and both periods (+0.006 to +0.040), and they trade less than
  single networks. The baseline ensemble fell below its members on the final test (-0.029).
- **The per-asset advantage became consistent.** With seed noise removed, it is positive in
  all twelve comparisons above, against both other LSTMs, in both halves and both periods. In
  trial 20 the median seed had gained nothing over the baseline LSTM on the final test.
- **It is still not significant.** Every interval includes zero. The remaining uncertainty is
  the market history itself, which more seeds cannot remove.

## Conclusion

The ten-network ensemble of the per-asset regime LSTM is the strongest LSTM configuration
found. It is consistently ahead of the baseline and joint-regime LSTMs, but not
significantly, and it still trails 1/N on the final test. It should be rerun in the Colab
reference environment, since these are Mac-trained networks, before it is reported.
