# Ten LSTM seeds on the final pipeline (Colab, commit c1e59fa) — manifest

## Status

The source of Table 7 and the ten-seed RQ2 results in the M7 report, pre-registered in
`docs/verification/ten_seed_final_pipeline_preregistration.md`. It reruns an existing study with
the final pipeline and selects nothing, so it adds no trial. The earlier exact-solver replay of
the frozen ten-seed networks (`outputs/ten_seed_corrected_bootstrap.csv`) is confirmed by this
run, not superseded: the retrained networks are identical to the archived ones.

## Code and environment

- Commit `c1e59fa53f779151e15554773befdd6b7845fa53` on `verification/ten-seed-final-pipeline`,
  whose `src/`, `scripts/run_real_data.py` and lock file are identical to the M6 reference
  commit `6a6b12d`. The only uncommitted files at run time were this run's own output folder.
- Commands: `python scripts/verification/run_ten_seed_final_pipeline.py --seed N` for seeds 123
  and 0-9 in one runtime, then `--merge`, run by the `Capstone_TenSeed_Run.ipynb` Colab notebook.
- Google Colab CPU runtime, x86_64, run 2026-11-01. Python 3.13.18 (the M6 run used 3.13.16),
  torch 2.11.0+cpu, cvxpy 1.9.3, cvxpylayers 1.2.0, diffcp 1.1.9, numpy 2.1.3, pandas 2.2.3,
  scipy 1.16.3, scikit-learn 1.6.1. The packages equal the M6 reference run's.

## Specification

Every M6 setting (200 epochs, 5% floor, exact solver, 3-state HMM refit quarterly, 5 bps net
excess returns, 2,000 stationary-bootstrap replications); only the seed changes.

## Checks (passed)

1. **Seed 123 control:** checkpoint hashes equal the M6 reference (`aa328280...`,
   `757819e8...`); the final-test RQ1-RQ3 table matches the M6 archive within 4.4e-11.
2. **Networks against the archive:** all ten seeds' baseline and regime networks are identical,
   parameter by parameter, to `outputs/archive/frozen_reference_5pct_bb46b63/ten_seed/`, and
   their 5 bps net returns match the exact-solver replay within 1.3e-8 a day.

ERC and equal-weight returns are identical across seeds.

## Results (final test, 5 bps net excess)

| Contrast | Mean | Median | Min | Max | Positive | Intervals excluding zero | Seed 123 (rank of 11) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RQ2 | +0.0537 | +0.0537 | -0.0488 | +0.1787 | 6 of 10 | 0 | -0.1002 (lowest) |
| RQ1, LSTM | +0.0532 | +0.0531 | -0.0493 | +0.1781 | 6 of 10 | 0 | -0.1008 (lowest) |
| RQ3, LSTM | -0.0031 | -0.0066 | -0.0252 | +0.0216 | 3 of 10 | 0 | +0.0082 (9th) |

Seed-averaged LSTM (a tenth in each seed's portfolio): RQ2 +0.0598 [-0.0318, +0.1456].

## Contents

| Path | What it is |
|---|---|
| `ten_seed_rq_per_seed.csv` | RQ1-RQ3 point estimates and 95% intervals per seed, including 123 |
| `ten_seed_rq_distribution.csv` | Distribution over seeds 0-9 and seed 123's rank (Table 7) |
| `ten_seed_averaged_rq.csv` | RQ1-RQ3 for the seed-averaged LSTM |
| `ten_seed_cost_sensitivity.csv`, `ten_seed_validation.csv` | Per-seed 5/10/25 bps net Sharpe, and validation-period context |
| `ten_seed_checks.json`, `merge.log` | Checks 1-2 and the merged report |
| `seedN_*` | Per seed: record (git, environment, checkpoint hashes, timing), log, networks, 5 bps net returns, RQ bootstrap, cost sensitivity, validation |
| `tests.log`, `environment.json`, `reference_manifest.json`, `execution_status.json` | Written by the notebook: the pre-run test log, the session's environment, the M6 manifest it checked against, and the run status |
