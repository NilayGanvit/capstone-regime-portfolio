# Results: crisis folds A-D, stage 1 (the M6 choices, the selection rule and fold D)

Run as pre-registered:
- `crisis_folds_preregistration.md` (`ca7345a`)
- `crisis_folds_selection_rule_preregistration.md` (`03f2842`)
- `crisis_fold_d_preregistration.md` (`f9eec78`)

The code was committed before each run: the proxy data step (`edcc04f`), the fold runner
(`f02f692`), the selection rule (`27060d5`, fixed in `3aea533`) and fold D (`42f40ca`). Script:
`scripts/verification/run_crisis_folds.py`. Branch: `verification/crisis-folds`, based on
`m6/final-pipeline`. Outputs are in `outputs/verification/crisis_folds/`: the per-fold
comparisons, metrics and summaries, the selection rule's tables, the fold D holdout table, and
each run's returns, records, budgets and networks.

Conventions:
- Net of 5 bps, Sharpe annualised and in excess of the risk-free rate.
- LSTM rows are ten seeds trained on a Mac, so they are not the Colab reference numbers.
- The per-asset regime configurations are reported separately, in
  `crisis_folds_stage2_results.md` on `verification/crisis-folds-asset-regime`.

## Deviations

1. A first draft of this branch was based on `experiments/asset-regime-lstm-ensemble`. It was
   replaced before anything ran, as the main pre-registration states.
2. Fold A was run and read before the selection rule was written, as the rule's addendum states.
   Folds B, C and D were run after it.
3. The selection job first stopped on a bug in its seed-spread code, before it wrote any output.
   The fix (`3aea533`) changes only how the seed columns are indexed; the job was then rerun.
4. Fold D was pre-registered after folds A-C and the rule had been run and read, as its addendum
   states.

## Checks: passed

- **Data:** the eleven proxy files match their manifest. The joined history equals the ETF returns
  exactly from 2007-04-12, and the proxy loader's returns before that.
- **Wiring:** given the M6 calendar, the ETF data and the M6 seed-123 networks, the runner reproduces
  the M6 archive's final-test returns, ERC and 1/N to 2.6e-16 a day and LSTM to 7.0e-9.
- **Timing:** these checks were run once, before fold A. Later commits only added code.

## The folds

| Fold | Training | Validation | Days | LSTM training months | SPY-TLT monthly correlation |
|---|---|---|---|---|---|
| A | 2000-08-31 to 2007-12-31 | 2008-01-02 to 2009-12-31 | 505 | 76 | -0.03 |
| B | 2000-08-31 to 2009-12-31 | 2010-01-04 to 2014-12-31 | 1,258 | 100 | -0.65 |
| C | 2000-08-31 to 2014-12-31 | 2015-01-02 to 2018-12-31 | 1,006 | 160 | -0.26 |
| D | 2000-08-31 to 2018-12-31 | 2019-01-02 to 2026-08-31 | 1,926 | 208 | +0.23 |

Fold D's status is as stated in its pre-registration. It is the only fold with a positive
stock-bond correlation.

## The M6 choices

### Quarterly HMM refits

`erc_regime` Sharpe by refit schedule:

| Fold | Quarterly | Monthly | No refit | Quarterly minus monthly | Quarterly minus no refit |
|---|---|---|---|---|---|
| A | -0.021 | 0.022 | 0.043 | -0.043 [-0.168, +0.023] | -0.064 [-0.300, +0.182] |
| B | 1.107 | 1.114 | 1.100 | -0.007 [-0.024, +0.004] | +0.007 [-0.085, +0.099] |
| C | 0.231 | 0.237 | 0.235 | -0.006 [-0.013, +0.000] | -0.004 [-0.041, +0.034] |
| D | 0.490 | 0.493 | 0.470 | -0.003 [-0.008, +0.002] | +0.020 [-0.002, +0.042] |

- **The choice holds in 0 of 4 folds against monthly refits, and 2 of 4 against no refit.**
  Outside the crisis fold every gap is 0.02 or less, so refit frequency barely matters.
- **The 2015-18 reference was already weak.** The pre-registered reference values
  (`outputs/refit_cadence_validation_comparison.csv`: +0.023 and +0.004) come from the original
  SLSQP solver. Trial 6's log entry records that with the exact solver the three schedules scored
  0.386 (no refit), 0.386 (monthly) and 0.383 (quarterly) on 2015-18. So quarterly was not ahead
  there either.
- **Fold C covers the same years.** Quarterly is again slightly behind there, with models trained
  from 2000.

### The 5% budget floor

The criterion it was chosen on is the mean pairwise L1 distance between the ten seeds' budgets,
where lower means more consistent across seeds:

| Fold | Baseline network, 5% / 2.5% | Regime network, 5% / 2.5% | Baseline Sharpe, 5% minus 2.5% | Regime Sharpe, 5% minus 2.5% |
|---|---|---|---|---|
| A | 0.427 / 0.717 | 0.482 / 0.737 | +0.111 (se 0.048, 7 of 10) | -0.037 (se 0.034, 3 of 10) |
| B | 0.434 / 0.714 | 0.581 / 0.824 | +0.096 (se 0.029, 9 of 10) | +0.150 (se 0.041, 8 of 10) |
| C | 0.621 / 0.871 | 0.433 / 0.648 | -0.048 (se 0.052, 5 of 10) | +0.036 (se 0.022, 6 of 10) |
| D | 0.389 / 0.696 | 0.490 / 0.693 | -0.022 (se 0.023, 3 of 10) | +0.037 (se 0.020, 6 of 10) |

- **The choice holds on its criterion in all four folds and for both networks.** The reductions
  are similar to M6's (-0.207 and -0.340).
- **On Sharpe, which was not the criterion, it is mixed.** It is clearly better in fold B and
  inconsistent elsewhere.

## Reported, not choices

| | A | B | C | D | 2015-18 (M6) |
|---|---|---|---|---|---|
| ERC regime effect (`erc_regime` minus `erc_baseline`) | -0.015 [-0.280, +0.245] | -0.022 [-0.098, +0.061] | -0.005 [-0.073, +0.061] | +0.004 [-0.070, +0.075] | -0.003 |
| LSTM regime effect, seed-paired | -0.147 (se 0.050) | -0.038 (se 0.042) | +0.116 (se 0.055) | +0.026 (se 0.023) | -0.059 (seed 123) |

Sharpe at 5 bps, with the LSTM rows as ten-seed means:

| | A | B | C | D |
|---|---|---|---|---|
| 1/N | 0.062 | 0.802 | 0.233 | 0.607 |
| `erc_baseline` | -0.006 | 1.129 | 0.236 | 0.486 |
| `erc_regime` | -0.021 | 1.107 | 0.231 | 0.490 |
| `erc_blend` | -0.009 | 1.107 | 0.254 | 0.492 |
| `lstm_baseline` | 0.152 | 0.847 | 0.283 | 0.539 |
| `lstm_regime` | 0.005 | 0.809 | 0.399 | 0.564 |

Max drawdown:

| | A | B | C | D |
|---|---|---|---|---|
| 1/N | -34.6% | -9.1% | -13.3% | -19.9% |
| `erc_baseline` | -24.2% | -8.3% | -8.8% | -20.3% |
| `erc_regime` | -22.2% | -8.3% | -9.1% | -20.7% |
| `lstm_baseline` | -20.6% | -9.7% | -8.7% | -21.4% |
| `lstm_regime` | -19.2% | -9.9% | -8.3% | -22.1% |

- **The regime effect changes sign from fold to fold.** It is never clearly positive, and it was
  most negative for the LSTM in the crisis.
- **ERC's lead in fold B, the 2010-14 bond rally, is the largest gap to 1/N in any fold.** Its
  interval still includes zero.
- **Every strategy fell much less than 1/N in the 2008-09 crisis, and slightly more in 2019-26.**

## The selection rule (folds A-C)

Sharpe minus 1/N's, with 95% intervals. LSTM rows are the seed-averaged portfolio: a tenth held
in each seed.

| Candidate | A | B | C | Worst fold | Eligible |
|---|---|---|---|---|---|
| 1/N | 0 | 0 | 0 | 0 | yes |
| `erc_baseline` | -0.067 [-0.494, +0.337] | +0.328 [-0.161, +0.848] | +0.003 [-0.447, +0.439] | -0.067 (A) | yes |
| `erc_regime` | -0.082 [-0.650, +0.518] | +0.306 [-0.175, +0.814] | -0.002 [-0.382, +0.409] | -0.082 (A) | yes |
| `erc_blend` | -0.070 [-0.637, +0.550] | +0.306 [-0.174, +0.828] | +0.021 [-0.364, +0.434] | -0.070 (A) | yes |
| `lstm_baseline`, seed-averaged | +0.083 [-0.380, +0.541] | +0.056 [-0.360, +0.465] | +0.054 [-0.433, +0.559] | **+0.054 (C)** | yes |
| `lstm_regime`, seed-averaged | -0.064 [-0.788, +0.654] | +0.023 [-0.444, +0.508] | +0.172 [-0.223, +0.600] | -0.064 (A) | yes |

Seeds' own differences against 1/N (median, interquartile range):
- **`lstm_baseline`:** A +0.067 (+0.014 to +0.176); B +0.032 (-0.007 to +0.096); C +0.040
  (-0.028 to +0.154).
- **`lstm_regime`:** A -0.035; B -0.004; C +0.175.

- **The rule chooses `lstm_baseline`, seed-averaged.** It is the only candidate ahead of 1/N in all
  three folds.
- **No candidate is excluded.** No interval lies wholly below zero.
- **No candidate is significantly ahead of 1/N in any fold.**

## Fold D (2019-2026): the choice, out of sample

| Candidate | Sharpe | Minus 1/N, 95% interval | M6 final test, minus 1/N (trained 2007-14) |
|---|---|---|---|
| 1/N | 0.607 | | 0.608 (1/N's own Sharpe) |
| **`lstm_baseline`, seed-averaged (the choice)** | 0.543 | **-0.064 [-0.302, +0.161]** | -0.068 (seed 123) |
| `lstm_regime`, seed-averaged | 0.572 | -0.035 [-0.252, +0.185] | -0.169 (seed 123) |
| `erc_baseline` | 0.486 | -0.121 [-0.369, +0.111] | -0.153 |
| `erc_regime` | 0.490 | -0.117 [-0.348, +0.101] | -0.153 |
| `erc_blend` | 0.492 | -0.115 [-0.349, +0.109] | -0.150 |

- **The choice trailed 1/N.** Eight of its ten seeds were behind (median seed -0.077, interquartile
  range -0.113 to -0.033). 1/N also had the higher return (8.8% a year against 7.4%) and the
  smaller drawdown.
- **Every candidate trailed 1/N, none significantly.** Training from 2000 rather than 2007
  narrowed the gaps, most for the regime LSTM, but the M6 column is a single Colab network, so
  that comparison is rough.
- **Fold D selects nothing.** Its best candidate cannot be promoted on this evidence.

## Conclusion

- **Across 2008-2026, no configuration beat 1/N with statistical support.** The configuration chosen
  for consistency over 2008-18 also trailed it in 2019-26.
- **The 5% floor does what it was chosen for in every fold.**
- **The quarterly refit choice is not supported anywhere.** The schedule barely matters outside the
  crisis.
- **The regime effect is not consistently positive in either allocator.**
- **The M6 specification stands.** Nothing here supports a change to it, or a claim that it beats
  1/N.
