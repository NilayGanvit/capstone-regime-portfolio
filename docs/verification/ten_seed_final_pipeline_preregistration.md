# Pre-registration: the RQ2 distribution across ten LSTM seeds on the final pipeline

Written and committed before the script was written or anything was run. Nothing below may be
changed after results are seen; any deviation is reported as a deviation.

## Why

The M6 draft's RQ2 result rests on seed 123: the LSTM regime effect minus the ERC regime
effect is -0.100 [-0.193, -0.011]. The draft also shows that this seed is unusual. Replaying
the ten networks of the frozen run's ten-seed study (seeds 0-9) through the final pipeline's
exact solver gives a mean regime effect of +0.05 (`outputs/ten_seed_corrected_bootstrap.csv`).

The M7 feedback asks for the ten seeds to be rerun on the final pipeline and their distribution
reported. Those networks were trained in the frozen run, not by the final pipeline, so this
retrains them with it. The question is whether the direction of the RQ2 conclusion depends on
the initialization.

## Design (fixed)

- **Code:** `m6/final-pipeline`. Its `src/`, `scripts/` and `tests/` are identical to the M6
  reference run's commit `6a6b12d`. The runner, `scripts/verification/run_ten_seed_final_pipeline.py`,
  calls the functions `scripts/run_real_data.py` uses with every setting it uses:
  - `train_lstm_variants`: 200 epochs, 5% floor, 60-day windows, 252-day covariance window,
    three states;
  - `run_primary_walk_forward`: exact solver, quarterly refits, next-open execution;
  - `stationary_bootstrap_rq_contrasts` on 5 bps net excess returns: 2,000 replications, 95%
    intervals, `random_state=0`;
  - `cost_sensitivity_table`.

  Only the seed changes.
- **Seeds:** 0-9, the ten-seed study's seeds, plus seed 123 as a control.
- **Environment:** the M6 reference run's pinned Colab CPU environment (x86_64), set up by the
  team's Colab notebook: the package versions are checked against the reference manifest, the
  input files against their SHA-256 hashes, and the frozen archive's risk-free file is used.
  Seeds run one after another in one runtime, with default thread settings, as the reference run
  did. LSTM training is not reproducible across machines, so a Mac run does not count.
- **Data and calendar:** as M6. Training runs 2007-04-12 to 2014-12-31, validation 2015-2018
  and the final test 2019-01-01 to 2026-08-31.

## Checks (before any result is read)

1. **Seed 123 control:** its two retrained networks have the SHA-256 hashes of the M6
   reference checkpoints (`aa328280...` and `757819e8...`), and its final-test RQ1-RQ3 bootstrap
   table equals the M6 archive's. If the control fails, the session differs from the
   reference environment. The run is then still reported, but every result is labelled as from
   a different environment.
2. **Networks against the archive:** for each of seeds 0-9, the retrained baseline and regime
   networks are compared with the frozen archive's checkpoints
   (`outputs/archive/frozen_reference_5pct_bb46b63/ten_seed/`) parameter by parameter. Wherever
   they are identical, the seed's final-test 5 bps net returns must match the exact-solver replay
   (`outputs/ten_seed_corrected/seed_NN_net_returns_5bps.csv`) to within 1e-7 a day.

## What is reported

The primary result is the distribution over seeds 0-9 of:
- RQ2: the LSTM regime effect minus the ERC regime effect;
- RQ1 for the LSTM: `lstm_regime` minus `lstm_baseline`.

Both are measured on the final test, net of 5 bps, in excess of the risk-free rate. For each
seed the report gives its point estimate and 95% interval. Across seeds it gives the mean,
median, interquartile range, minimum and maximum, how many are positive, and how many intervals
exclude zero on each side. Seed 123's value and its rank among the eleven seeds are reported
too.

**Reading rule, fixed in advance:** the RQ2 conclusion is described as depending on the
initialization if the ten seeds' point estimates take both signs.

Also reported:
- the ERC regime effect, which does not depend on the seed (reported once);
- RQ3 for both allocators;
- RQ1-RQ3 for the seed-averaged LSTM (a tenth held in each seed's portfolio), with its bootstrap
  interval;
- each seed's 5, 10 and 25 bps net excess Sharpe for the six configurations;
- the validation-period (2015-2018) Sharpe and regime effects, as context.

## How to read it

- **What the spread across seeds measures:** training randomness only. All seeds are scored on
  the same days, so the sampling noise of the period is in each seed's own interval (about
  +/-0.15 for RQ2), not in the spread across seeds.
- **Expected outcome:** if check 2 finds the networks identical, the distribution equals the
  existing replay: mean +0.054, 6 of 10 positive, no interval excluding zero.
- **What this does not do:** it selects nothing and changes no M6 setting. It reruns an existing
  study rather than testing a new specification, so it adds no trial.
