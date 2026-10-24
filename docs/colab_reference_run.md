# Colab reference run of the corrected pipeline

Purpose: produce Table 5 (seed 123), the bootstrap (Table 6) and the DSR from one run of
one committed version of the code, with the exact risk-budget solver, and save the
checkpoint and a provenance manifest this time. LSTM training is not reproducible across CPU
architectures, so this has to run on the Colab CPU runtime (x86_64), not on a Mac.

## Run

```bash
git clone https://github.com/NilayGanvit/capstone-regime-portfolio.git
cd capstone-regime-portfolio
git checkout m6/final-pipeline            # or the commit the team agrees on; note it
pip install -r requirements-reference-lock.txt
```

Data is gitignored. Copy the ten ETF files `SPY.csv ... VNQ.csv` from the shared Drive into
`data/raw/`, then use the frozen run's own risk-free file so the rate is identical:

```bash
cp outputs/archive/frozen_reference_5pct_bb46b63/data/risk_free_DGS3MO_daily.csv data/raw/
python scripts/run_real_data.py 2>&1 | tee reference_run.log      # about 15 minutes on Colab
```

Do not edit any tracked file before running. The run records the uncommitted files it started
with in `outputs/run_manifest.json`; an empty list means the commit is exactly the code that ran.

## Check it

1. **Provenance.** `python -c "import json; m=json.load(open('outputs/run_manifest.json')); print(m['git']['commit'], m['git']['uncommitted_files'], m['environment']['machine'])"`
   should show the intended commit, `[]`, and `x86_64`.
2. **The network is the frozen one.** Training does not depend on the solver, so replaying the new
   checkpoints with the original solver must reproduce the archived frozen seed-123 returns:
   ```bash
   python scripts/verification/check_replay_against_archive.py \
     --baseline outputs/lstm_checkpoints/seed123_baseline.pt \
     --regime outputs/lstm_checkpoints/seed123_regime.pt \
     --archived outputs/archive/frozen_reference_5pct_bb46b63/csv/real_data_rq_net_returns_5bps.csv \
     --solver slsqp
   ```
   Expect `PASS` with a worst difference around 1e-8. A `FAIL` means this run trained a different
   network, and the corrected results cannot be read as a correction of the frozen ones.
3. **Solver-independent rows.** Net excess Sharpe at 5 bps in `real_data_cost_sensitivity.csv`:
   erc_baseline 0.455, erc_regime 0.455, erc_blend 0.458, equal_weight 0.608. These do not depend
   on the LSTM or the environment; a different value means the data or code differs.

## Bring back

`reference_run.log`, `outputs/run_manifest.json`, `outputs/lstm_checkpoints/`, and the
`outputs/real_data_*.csv` files, committed together under
`outputs/archive/m6_reference_run_<short commit>/` with a short MANIFEST.md in the style of the
other archives. The expected comparison against the frozen run is in
`docs/verification/solver_and_accounting_checks.md` (sections 8 and 9).

## Result of the checks (reference run of 2026-10-22, commit 6a6b12d)

1. **Provenance.** The manifest records commit `6a6b12de19b567ba7c1feda10bd5423cd72b5e23`, no uncommitted
   files, `x86_64`. The ten ETF files match the manifest's hashes; the two checkpoints match theirs
   (`aa328280...`, `757819e8...`).
2. **The network is the frozen one: PASS.** Replaying the two checkpoints with the original solver
   against `frozen_reference_5pct_bb46b63/csv/real_data_rq_net_returns_5bps.csv` (final test, 5 bps,
   1,926 days) gives a worst daily net-return difference of 8.7e-09 (lstm_baseline 1.8e-09,
   lstm_regime and lstm_blend 8.7e-09). Replaying with the exact solver against the reference run's own
   `real_data_rq_net_returns_5bps.csv` gives 7.0e-09, so the archived results are what the saved
   checkpoints produce. The replay used the local data files, whose hashes match the manifest; the local
   risk-free file differs from the run's on 4 rows, which does not enter net returns.
3. **Solver-independent rows.** erc_baseline 0.455, erc_regime 0.455, erc_blend 0.458, equal_weight
   0.608, as expected.

What this establishes: the frozen 5% run and the reference run share one pair of trained networks, so
the difference between their LSTM rows is the budget-to-weight solver alone (SLSQP in the frozen run,
exact in the reference run). Net excess Sharpe at 5 bps, final test, seed 123:

| | frozen run (SLSQP) | reference run (exact) | exact minus SLSQP |
|---|---|---|---|
| LSTM baseline | 0.482 | 0.540 | +0.058 |
| LSTM regime | 0.512 | 0.439 | -0.073 |
| LSTM blend | 0.517 | 0.448 | -0.069 |
| LSTM regime effect (regime - baseline) | +0.030 | -0.101 | -0.131 |

The frozen run's own bootstrap gave an LSTM regime effect of +0.029 with a 95% interval of
[-0.177, +0.229]; the reference run's is -0.101 with [-0.216, +0.014]. The ERC baseline and equal-weight rows are identical;
erc_regime (0.462 vs 0.455) and erc_blend (0.465 vs 0.458) differ, the two ERC rows that SLSQP's failures touched. For
the ten-seed result of the same kind, see section 8 of `docs/verification/solver_and_accounting_checks.md`;
seed 123 is the one seed whose own checkpoint is now replayed under both solvers, and it moves in the
same direction as the ten-seed mean for the regime LSTM but opposite for the baseline (the ten-seed
baseline mean fell by 0.061 under the exact solver; seed 123's rose by 0.058).
