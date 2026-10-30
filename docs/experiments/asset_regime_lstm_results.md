# Results: regime LSTM with per-asset regime inputs (trial 20)

Run exactly as pre-registered in `asset_regime_lstm_preregistration.md` (committed first, as
`5d66247`; implementation committed before the run, `ed2ee7c`), by
`scripts/experiments/run_asset_regime_lstm.py`, seeds 0-9 trained and run in parallel on a
Mac. No deviations. Outputs: `outputs/experiments/asset_regime_lstm_per_seed.csv`,
`..._summary.csv`, `..._differences.csv`, `..._gates.json`; per seed, the three trained
networks, daily 5 bps net returns, metrics and checks in `outputs/experiments/asset_regime_lstm/`.
Net of costs, Sharpe in excess of the M6 reference run's risk-free series. These are Mac
results; LSTM training depends on the machine, so they are not Colab-canonical numbers.

**Checks: passed.** For every seed, the ERC and 1/N returns match the M6 reference archive
to 2.6e-16 a day. On seed 0, every other configuration's returns, including `lstm_baseline`
and `lstm_regime`, are identical to a run without the new network.

## Verdict: passes its validation gates

Means over the ten seeds, validation 2015-2018, 5 bps:

| Gate | Result | Pass |
|---|---|---|
| G1: Sharpe at least `lstm_baseline`'s | 0.412 vs 0.368 (difference +0.043, seed standard error 0.046; 7 of 10 seeds at least 0) | yes |
| G2: Sharpe at least `lstm_regime`'s | 0.412 vs 0.361 (difference +0.051, seed standard error 0.027; 5 of 10 seeds at least 0) | yes |
| G3: drawdown within 2 pp of `lstm_baseline`'s | -8.8% vs -8.2% | yes |

The standard errors measure only how much the result changes with the training seed. All
seeds are evaluated on the same days, so the sampling noise of the period itself, which is
larger (about +/-0.3 for one seed's paired interval), is not in them.

## Validation and final test

Means over the ten seeds (standard errors across seeds in brackets):

| | `lstm_asset_regime` | `lstm_baseline` | `lstm_regime` | `erc_asset_regime` | 1/N |
|---|---|---|---|---|---|
| **Validation 2015-2018** | | | | | |
| Sharpe, 5 bps | 0.412 (0.030) | 0.368 (0.028) | 0.361 (0.014) | 0.240 | 0.233 |
| Sharpe, 25 bps | 0.348 | 0.325 | 0.308 | 0.201 | 0.226 |
| Max drawdown | -8.8% | -8.2% | -9.7% | -9.6% | -13.3% |
| Turnover per rebalance | 15.4% | 10.4% | 13.3% | 8.6% | 2.1% |
| **Final test 2019-2026 (exploratory, does not count)** | | | | | |
| Sharpe, 5 bps | 0.520 (0.022) | 0.488 (0.015) | 0.481 (0.016) | 0.437 | 0.608 |
| Sharpe, 25 bps | 0.464 | 0.462 | 0.432 | 0.399 | 0.603 |
| Max drawdown | -20.8% | -22.3% | -21.9% | -20.6% | -20.0% |
| Turnover per rebalance | 20.0% | 9.6% | 18.2% | 12.7% | 2.3% |

`lstm_asset_regime` minus the comparison, 5 bps, by seed:

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | mean | median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| validation, vs `lstm_baseline` | +0.011 | -0.287 | +0.074 | -0.024 | +0.251 | -0.006 | +0.089 | +0.091 | +0.207 | +0.028 | +0.043 | +0.051 |
| validation, vs `lstm_regime` | +0.040 | -0.003 | -0.013 | -0.004 | +0.123 | -0.031 | +0.198 | +0.097 | +0.144 | -0.042 | +0.051 | +0.019 |
| final test, vs `lstm_baseline` | -0.066 | -0.002 | +0.064 | -0.030 | +0.199 | -0.054 | +0.051 | +0.163 | -0.001 | -0.003 | +0.032 | -0.001 |
| final test, vs `lstm_regime` | -0.113 | +0.030 | +0.083 | +0.035 | +0.047 | -0.017 | +0.042 | +0.226 | +0.010 | +0.047 | +0.039 | +0.039 |

## What happened

- **Per-asset inputs did better than joint regime inputs in both periods.** Against
  `lstm_regime` the mean gain was +0.051 on validation and +0.039 on the final test, with 8 of
  10 seeds at least 0 on the final test. It held at 25 bps (+0.040 and +0.032). It is the
  most consistent result of the regime trials.
- **The joint regime inputs add nothing on these seeds; the per-asset ones add a little.**
  `lstm_regime` minus `lstm_baseline` averaged -0.008 on validation and -0.007 on the final
  test. `lstm_asset_regime` minus `lstm_baseline` averaged +0.043 and +0.032.
- **The gain over the baseline LSTM is uneven.** A few seeds carry the mean (seeds 4 and 7
  gain more than 0.09 in both periods), and on the final test the median seed gained
  nothing (-0.001). At 25 bps
  the final-test gain over `lstm_baseline` is +0.002.
- **It trades more.** Turnover per rebalance was 15-20%, against 10% for `lstm_baseline` and
  13-18% for `lstm_regime`.
- **It does not change the comparison with 1/N.** All three LSTMs beat 1/N on validation and
  trail it on the final test (0.520 against 0.608).

## Conclusion

Giving the regime LSTM each asset's own regime probability passes its pre-registered gates:
on average it beat both the baseline LSTM and the joint-regime LSTM on validation, and the
final test points the same way against the joint-regime LSTM. The gains are about 0.04-0.05
Sharpe, uneven across seeds, partly offset by higher turnover, and small against the
period's own sampling noise. It is the strongest of the regime variants tested, not evidence
that regimes beat 1/N. Before it is reported as a result, it should be rerun in the
Colab reference environment, since these are Mac-trained networks.
