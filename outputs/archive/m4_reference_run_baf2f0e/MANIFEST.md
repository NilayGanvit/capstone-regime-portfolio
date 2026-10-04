# M4 reference run -- manifest

Archived per Silvio's request (2026-10-02 Slack thread) once PR #3 was
merged: "use that version, with seed 123 and the corrected cost
calculation, as our reference run for M4." This is that run.

> **STATUS UPDATE (2026-10-04):** this run was generated on Nilay's
> local Mac venv (arm64, torch 2.14.0), not on Colab. AnnaLisa's
> frozen-LSTM validation Sharpe figures (baseline 0.380, regime 0.418,
> from her Colab cadence check) differ from this run's validation
> figures (baseline 0.411, regime 0.489) despite identical code, seed,
> and input data (see "Cross-environment numerical drift" below for
> the isolated cause). Since Colab -- not any one member's local
> machine -- is the environment the whole group can reach, the group
> decided to pin the *reference* environment to Colab's package
> versions (`requirements-reference-lock.txt`, repo root) going
> forward. **This archive remains useful for its commit/code/data
> provenance, but its specific numeric values should be treated as
> superseded once an equivalent run is archived from Colab itself
> under `requirements-reference-lock.txt`'s pinned versions** -- a
> same-version install on different CPU architectures is not enough,
> per the isolation test below.

## Code

- **Repository:** capstone-regime-portfolio (NilayGanvit/capstone-regime-portfolio)
- **Branch:** `feature/lstm-quarterly-retraining`
- **Commit:** `baf2f0ec353083ffaf129cd65cf4d4737bc85629`
  (merge commit for PR #3, "Seed LSTM weight initialization for
  reproducibility" -- includes PR #1's calendar fix and PR #2's
  self-financing transaction-cost fix as prior history on this branch)
- **Commit date:** 2026-10-03 19:54:22 +0530
- **Command:** `python scripts/run_real_data.py`
- **Run timestamp (UTC):** 2026-10-03T14:35:17Z

## Run settings

- `RiskBudgetLSTM` weight initialization: `random_state=123` (default,
  not overridden -- see `src/allocation_lstm.py`)
- LSTM: trained once on the initial window only (through
  2014-12-31), frozen for the entire walk-forward -- **not** the
  quarterly-retraining comparison arm. See the open Section 3.5
  cadence discussion (2026-10-02/03 Slack thread) for why this is the
  primary specification and quarterly retraining is a separate
  robustness/comparison exercise, not this reference run.
- HMM: `n_states=3`, `random_state=0`, quarterly scheduled refit
  (`refit_every_n_rebalances=3`, warm-started), per `walkforward.py`'s
  default.
- Constraints: `ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30)`
- Execution timing: next-session-open (real `AdjOpen` data present in
  `data/raw/`)
- Transaction costs: self-financing (PR #2 fix), 5/10/25 bps
  sensitivity, 5 bps is the headline M2-proposed baseline
- Calendar: `data.M2Calendar` -- initial window through 2014-12-31,
  validation 2015-01-01 to 2018-12-31, final test 2019-01-01 to
  2026-08-31, anything after 2026-08-31 reported separately and
  excluded from the final-test figure
- DSR: `n_trials` read from `outputs/trial_log.csv` at run time (see
  `run_real_data.stdout.log` for the exact value used, `n_trials=7`)

## Data reference

Source: `data/raw/SOURCE.md` (Yahoo Finance via `yfinance`, pulled
2026-09-23, ten M2 ETFs, 2007-04-11 through 2026-09-21, 4,892 trading
days). SHA-256 of every file actually read by this run:

```
91f41cc75f5a5c8386c80604a19b14606c5f969fadb0547fc385a5fb3086f8c6  data/raw/DBC.csv
4e510c770f81a88ddac883815a5b8ceb2e320eb8df94a6cf19ce02bd45896437  data/raw/EEM.csv
a83134bb793997bf947f3ccd9114c2c55f92607d6de23e2f4e67d9988b8c26ba  data/raw/EFA.csv
219a18c912cfce61223a8de76d0ab9578b2ce7ab89f56c37008522d77746d220  data/raw/GLD.csv
4971335eacd0f31118cf845e453859e8dc2445e3eb2ab2891035168f4eb53806  data/raw/HYG.csv
d3dd1b0441d6b25075a4e40b373c7339250254abe0315dea13002117de6068ae  data/raw/IEF.csv
5346d5545ea7f59f0b6566fedaf45880ca2eebd3d2e2098a2eb566b84bc8e07f  data/raw/LQD.csv
8a2763a4fd9efa36143c64377b7dd1546cc5f2d00b24b523e593905532824c4d  data/raw/SPY.csv
39b8f220ca191e0aaee32cc652e28a411ebe8ddf29e064d9433f74a3b46ea0c8  data/raw/TLT.csv
99bb70759ef2ec9917347077c228390e55b965223cc1585efa2936424788f490  data/raw/VNQ.csv
53dcd5eff23f8ac7496eaee6b8079483d6a4ab583f9c83cfe1d3ff6660949286  data/raw/SOURCE.md
```

## Package versions (this run's environment)

```
python       3.14 (venv at capstone-regime-portfolio/venv)
torch        2.14.0
cvxpy        1.9.3
cvxpylayers  1.2.0
numpy        2.5.3
pandas       3.0.6
scipy        1.18.1
scikit-learn 1.9.1
```

Note: these differ from the NumPy 2.1.3 / pandas 2.2.3 combination
referenced in Silvio's original calendar-bug diagnosis -- that
combination is what originally exposed the NumPy/pandas set-membership
bug (PR #1); this run is on whatever the shared venv currently
resolves to, which is why the calendar fix (Timestamp normalization,
not a version pin) is what makes this reproducible across environments
rather than pinning to one specific NumPy/pandas pair. If exact
environment parity with Colab is needed, compare against this list
before aligning.

## Cross-environment numerical drift (isolated 2026-10-04)

AnnaLisa's Colab frozen-LSTM validation Sharpe (baseline 0.380, regime
0.418) differs from this archive's (baseline 0.411, regime 0.489) on
identical code, seed (123), and input data (SHA-256-verified). She
attributed this to torch 2.14.0 (this run) vs 2.11.0 (Colab). We
isolated it further on Nilay's Mac: constructing `RiskBudgetLSTM` with
`random_state=123` under torch 2.14.0 and torch 2.11.0 side by side
(same arm64 machine) gave **bit-identical** initial weights, and
feeding both a fixed, RNG-free input through a full forward pass gave
**bit-identical** output tensors. So the torch *version* difference,
by itself, is not what produced the Sharpe gap -- at least not via
weight init or forward-pass arithmetic on the same CPU.

The remaining, uncontrolled variable is CPU architecture: this run is
arm64 (Apple Silicon); Colab's CPU runtime is x86_64. Different
vectorized instruction sets and BLAS backends reorder floating-point
summation inside matrix multiplications, which is exactly what
PyTorch's own documentation warns about: "results may not be
reproducible between CPU and GPU executions, even when using identical
seeds" and, more generally, "are not guaranteed across... different
platforms." The same applies to the differentiable risk-budgeting
layer's convex solve inside training (`cvxpylayers`/`cvxpy`, whose
linear algebra runs through numpy/scipy), which both AnnaLisa's and
this run's numpy/scipy versions also differ on. Training compounds
small per-step floating-point differences over 200 epochs of
non-convex optimization, so even a tiny per-op discrepancy can produce
a visibly different trained model by the end -- AnnaLisa separately
confirmed the *initial* training loss (epoch 0) already differs
between the two runs, consistent with this.

Practical takeaway: `torch.manual_seed` (and this project's
`random_state` plumbing) reproduces results *within* one pinned
environment on one architecture, not across architectures. Pinning
package versions (`requirements-reference-lock.txt`) is necessary but
not sufficient on its own -- the canonical M4 reference run should be
*executed* on Colab, not merely installed with Colab's package
versions on a different machine.

## Headline final-test numbers (gross, from this run)

See `csv/real_data_summary.csv` for the full table and
`csv/real_data_cost_sensitivity.csv` for 5/10/25 bps net figures; from
`run_real_data.stdout.log`, final test (2019-01-01 to 2026-08-31,
1926 days), 5 bps net-of-cost Sharpe / DSR:

| config | Sharpe (gross) | Sharpe (5bps net) | DSR (5bps net) |
|---|---:|---:|---:|
| erc_baseline | 0.799 | 0.797 | 0.054 |
| erc_regime | 0.810 | 0.804 | 0.056 |
| erc_blend | 0.814 | 0.807 | 0.057 |
| equal_weight | 0.883 | 0.881 | 0.087 |
| lstm_baseline | 0.808 | 0.799 | 0.053 |
| lstm_regime | 0.861 | 0.856 | 0.077 |
| lstm_blend | 0.852 | 0.847 | 0.073 |

## Contents of this archive

- `csv/real_data_summary.csv` -- final-test metrics, all 7 configs (gross)
- `csv/real_data_cost_sensitivity.csv` -- 5/10/25 bps net Sharpe/CAGR
- `csv/real_data_transaction_log_<config>.csv` -- per-execution transaction log (final test, 5 bps), one per config
- `csv/real_data_binding_constraints_<config>.csv` -- per-rebalance constraint-binding diagnostics (final test), one per config
- `csv/real_data_pi_path.csv` -- reliability weight pi_t, full walk-forward history
- `run_real_data.stdout.log` -- full console output of the run that produced these CSVs

17 CSVs total, matching AnnaLisa's independent reproducibility check
(2026-10-02: two independent runs of `run_real_data.py` on this commit
produced byte-identical CSVs, max numeric difference 0.0).
