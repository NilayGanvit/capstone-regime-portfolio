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
