# Pre-registration addendum: fold D, 2019-2026, for the candidate chosen on folds A-C

An addendum to `crisis_folds_preregistration.md` and `crisis_folds_selection_rule_preregistration.md`.
Written and committed after folds A, B and C and the selection rule had been run and read, and before
fold D was implemented or run. Nothing below may be changed after fold D's results are seen; any
deviation is reported as a deviation.

## Status of the period

Fold D (2019-26) is out-of-sample for both model estimation and strategy selection, every
architectural gate across M6 and post-M6 trials was evaluated exclusively on 2015-18 or Folds A-C.
General macroeconomic awareness of the 2022 rate regime motivated testing certain post-M6
hypotheses.

The trial log also records the following:

- **Trial 6:** the M6 refit cadence was first compared on a sample that included 2019-26, then
  re-derived on 2015-18 alone before M6.
- **Trials 2, 10 and 11:** the state-count check and two robustness checks were computed on samples
  including 2019-26 and left the specification unchanged.

## Fold D

| Fold | Training | Evaluation |
|---|---|---|
| D | 2000-08-31 to 2018-12-31 (proxies before 2007-04-12, ETFs after) | 2019-01-01 to 2026-08-31 |

Fold D is run exactly as folds A-C, in both stages: the same pipeline, data, costs, seeds 0-9, both
budget floors in stage 1, and the three ERC refit arms. The stage-2 per-asset regime
configurations and ensembles are included too, and so are the same reproduction checks. The data
are cut at 2026-08-31.

## What is reported

1. **The choice from folds A-C:** the baseline LSTM, seed-averaged (`selection_rule_choice.json`),
   against 1/N on fold D. This is the primary result: its Sharpe difference, net of 5 bps, in
   excess of the risk-free rate, with a paired stationary-bootstrap 95% interval
   (`evaluation.block_bootstrap_diff_ci`, automatic block length, 2,000 replications,
   `random_state=0`). The choice is evaluated as it stands, and the rule is not re-applied with
   fold D included.
2. **Every other selection-rule candidate** against 1/N on fold D, with the same statistic, and the
   median and interquartile range across seeds for the LSTM candidates.
3. **The same fold-level tables as folds A-C:** the M6 choice comparisons in stage 1, and the
   per-asset regime comparisons in stage 2.
4. **Context:** M6's own final-test numbers for the same period, from models trained on 2007-14
   only, beside fold D's, which are trained on 2000-18.

Fold D selects nothing and changes no M6 choice.

## How to read it

Power, stated in advance: the M6 final test's intervals against 1/N are about +/-0.3 over this
7.7-year period, and fold D's are expected to be similar, so only a difference of about 0.3 or more
would be significant. The LSTM rows are trained on a Mac, so they are not the Colab reference
numbers. Fold D, like folds A-C, contains no positive stock-bond correlation before 2022, and
2022 itself is in it.
