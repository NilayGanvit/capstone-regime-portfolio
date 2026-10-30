# Pre-registration addendum: a selection rule against 1/N across the three crisis folds

An addendum to `crisis_folds_preregistration.md`. It was written after fold A had been run and
read in both stages, and committed before folds B and C were run. Fold A is therefore not blind
to this rule; folds B and C are. To limit how much fold A could shape the rule, it has no
threshold or weight to choose. Folds B and C are run exactly as the main pre-registration fixes
them.

## Why

The main pre-registration checks each M6 choice against its alternative. It never asks which
configuration a robust rule would pick across folds. All three external reviews of the
validation design recommended judging configurations against 1/N in every fold rather than on
raw Sharpe, with seed treated as noise. In a crisis fold every Sharpe is low, and a raw
worst-fold rule would simply favour whatever held the most bonds.

## Candidates

The benchmark 1/N is itself a candidate, with a difference of 0 in every fold.

| Stage | Candidates, in order of simplicity |
|---|---|
| 1 (`verification/crisis-folds`) | 1/N; `erc_baseline`; `erc_regime`; `erc_blend`; `lstm_baseline`, seed-averaged; `lstm_regime`, seed-averaged |
| 2 (`verification/crisis-folds-asset-regime`) | the stage-1 candidates, then `erc_asset_regime`; `lstm_asset_regime`, seed-averaged; the ten-network ensembles of `lstm_baseline`, `lstm_regime` and `lstm_asset_regime` |

- **Configurations:** every candidate is the M6 specification (quarterly refits, 5% floor).
- **LSTM candidates:** an LSTM configuration is the algorithm, not one seed. Its seed-averaged
  return is the daily mean of its ten seeds' 5 bps net returns, the return of holding a tenth in
  each seed's portfolio.
- **Not candidates:** the LSTM blends, because the fold runs do not record their returns.

## Statistic

For candidate c in fold f:

- **D_f(c):** Sharpe(c) minus Sharpe(1/N) over the fold's validation period, net of 5 bps,
  annualised and in excess of the risk-free rate.
- **Interval:** a paired stationary-bootstrap 95% interval for D_f(c)
  (`evaluation.block_bootstrap_diff_ci`, automatic block length, 2,000 replications,
  `random_state=0`).

## Rule

1. **Eligibility:** a candidate is out if, in any fold, the upper end of its interval is below 0,
   that is, it is significantly worse than 1/N there. 1/N is always eligible.
2. **Choice:** among eligible candidates, take the one with the highest worst-fold difference,
   min over f of D_f(c).
3. **Ties:** equal to three decimals goes to the candidate earlier in the order above.

The rule is applied to folds A, B and C together, separately in each stage.

## Reporting

- For every candidate: D_f(c) and its interval in each fold, its worst fold and whether it is
  eligible.
- The choice.
- For the LSTM candidates, the median and interquartile range across the ten seeds of each
  seed's own D_f.

## How to read it

The intervals against 1/N are wide (fold A: about +/-0.4 to +/-0.6), so eligibility will rarely
exclude anything. The rule's force is the worst-fold comparison with 1/N among the candidates. It
picks a model over 1/N only if that model beat 1/N in its worst fold.

It is reported as a robustness result. It does not change the M6 specification. Like the folds,
it is not logged as a trial.
