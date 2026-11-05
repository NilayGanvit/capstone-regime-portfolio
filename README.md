# capstone-regime-portfolio

**Reliability-Gated, Regime-Aware Dynamic Risk Budgeting**
MScFE 690 Capstone — Track 7 (Medium/Long-Term Trading Models) + Track 8.2 (ML/DL Portfolio Allocation)

AnnaLisa Aaron · Nilay Jayantibhai Ganvit · Silvio Mazzaro

## Project Overview

This project examines whether incorporating market regime information improves medium-term portfolio allocation, and whether accounting for predictive reliability provides additional benefits.

We compare Equal Risk Contribution (ERC) and Long Short-Term Memory (LSTM) risk-budgeting strategies across a diversified universe of ten exchange-traded funds. Each allocator is evaluated in three configurations: baseline, regime-aware, and reliability-blended. A simple equal-weight portfolio provides an additional benchmark.

The primary analysis evaluates the incremental contribution of regime information and predictive reliability under common portfolio constraints, decision dates, transaction costs, and execution conventions.

## Research Questions

RQ1 — Regime information: Does regime conditioning improve net risk-adjusted performance relative to the corresponding baseline allocator?

RQ2 — Allocator differences: Is the incremental effect of regime information different between ERC and LSTM?

RQ3 — Predictive reliability: Does reliability-weighted blending improve performance relative to the corresponding regime-aware portfolio?

## Model Architecture
The pipeline combines:

A three-state Gaussian Hidden Markov Model (HMM) for estimating market regime probabilities.

Two portfolio allocators: ERC with fixed equal risk budgets and an LSTM that learns time-varying risk budgets.

Regime-conditioned covariance estimates for the regime-aware configurations.

Two Student-t predictive-density models that provide the evidence used to update the reliability weight.

Reliability-based blending of baseline and regime-aware portfolios within each allocator.

Shared portfolio constraints, monthly rebalancing, transaction-cost accounting, and chronological walk-forward evaluation.

The LSTM produces risk budgets rather than portfolio weights directly. A common exact convex risk-budgeting solver converts these budgets into unconstrained portfolio targets, after which the portfolio constraints are applied.

## Data and Evaluation

The primary universe comprises SPY, EFA, EEM, IEF, TLT, LQD, HYG, GLD, DBC, and VNQ, representing equities, government bonds, corporate credit, gold, commodities, and real estate.

The study uses the following chronological periods:

| Period | Purpose |
|:---|:---|
| **April 2007 – December 2014** | Initial model estimation and LSTM training |
| **January 2015 – December 2018** | Model validation and specification selection |
| **January 2019 – August 2026** | Exploratory historical evaluation |

The final period is not treated as an independent holdout because observations from it were examined during model development.

Portfolio decisions occur monthly at the end-of-month close, with execution at the following session's adjusted opening price. The reference implementation is long-only, fully invested, and applies a 30% per-asset weight limit and a 30% turnover limit measured using decision-close holdings.

Headline Sharpe ratios are computed from daily returns in excess of a lagged three-month U.S. Treasury yield series (FRED DGS3MO). Transaction-cost sensitivity is evaluated at 5, 10, and 25 basis points.

The initial LSTM training period contains 80 overlapping monthly examples. The primary specification retains network parameters fitted through 2014, while quarterly LSTM retraining is evaluated separately.

## M6 Reference Implementation

The frozen M6 reference run uses:

Three HMM states, with quarterly expanding-window reestimation.

ERC and LSTM baseline, regime-aware, and reliability-blended configurations.

Bounded Softmax for the LSTM, with a 5% minimum risk-budget share.

Seed 123 and 200 training epochs.

The exact convex risk-budgeting solver for portfolio construction.

Consistent transaction-cost and risk-free-rate conventions across configurations.

Reference commit: 6a6b12de19b567ba7c1feda10bd5423cd72b5e23

Saved reference outputs and checkpoints should be used when reproducing the reported results. LSTM training is sensitive to the numerical environment: identical seeds do not guarantee identical fitted networks across Mac and Colab. Inference from the archived checkpoints is substantially more reproducible.

## Main Findings

In the final reference run, regime conditioning did not improve net Sharpe for either allocator. At 5 basis points, ERC's baseline and regime-aware Sharpe ratios were approximately 0.455, while the LSTM's declined from 0.540 to 0.439 after adding regime conditioning.

Reliability blending produced small positive changes relative to the regime-aware configurations, but their paired bootstrap confidence intervals included zero. The equal-weight benchmark achieved a higher net Sharpe than all six model-based portfolios.

The study also found:

Bounded Softmax reduced the dispersion of learned risk budgets across ten LSTM seeds by more than half relative to ordinary Softmax.

Increased stability of predicted budgets did not establish superior investment performance.

Quarterly HMM refitting was retained as an implementation choice rather than an empirically superior schedule.

Results from the single final LSTM initialization should not be interpreted as typical outcomes across seeds.

The Deflated Sharpe Ratio was sensitive to assumptions about the dispersion and effective number of candidate trials.

The findings do not establish a consistent economic advantage from adding regime information or predictive reliability.

## Verification and Robustness

The project includes numerical solver checks, independent portfolio-accounting reconciliation, initialization-stability experiments, transaction-cost sensitivity, paired stationary-bootstrap inference, and DSR sensitivity analysis.

A subsequent implementation audit found that the decision-close turnover convention occasionally permitted opening turnover above the 30% limit. Across the seven configurations, 17 configuration-level violations occurred, with a maximum opening turnover of 30.6715%. Strictly enforcing the limit at execution and recalculating transaction costs from opening turnover produced small changes without altering the conclusions for RQ1–RQ3.

An exploratory gross-return attribution indicated that equities and real estate accounted for approximately 94.8% of the arithmetic contribution gap between ERC baseline and equal weighting during 2019–2026. This is descriptive attribution, not evidence that equal weighting is universally superior.

Supporting files are available under:

outputs/verification/m6_implementation/

## Additional Exploratory Experiments

Separate experimental branches examine crisis-period robustness, asset-specific regimes, and LSTM seed ensembles.

These experiments are not part of the frozen M6 reference specification. The crisis-fold checks provide additional evidence about initialization stability and the sensitivity of regime-refitting choices across different historical periods. Asset-specific regimes and seed ensembles are exploratory extensions and do not replace the primary results.

Results from these experiments should be interpreted with their documented validation procedures, training environments, and historical-sample limitations.

## Repository Organization

| Directory | Contents |
|:---|:---|
| [`src/`](src/) | Data processing, regime estimation, predictive densities, reliability modeling, portfolio allocation, and evaluation |
| [`scripts/`](scripts/) | Reference runs, robustness experiments, diagnostics, and verification utilities |
| [`tests/`](tests/) | Unit, integration, solver, and portfolio-accounting tests |
| [`data/raw/`](data/raw/) | The frozen ETF prices and risk-free rate used by the reported runs (see [Data](#data)) |
| [`outputs/`](outputs/) | Model results, diagnostics, robustness analyses, and archived outputs |
| [`outputs/verification/m6_implementation/`](outputs/verification/m6_implementation/) | Turnover, transaction-cost, and return-attribution verification |
| [`docs/`](docs/) | Methodology, experiment documentation, and verification records |

```
python scripts/run_smoke_test.py       # synthetic 2-regime data, plumbing check
python scripts/run_real_data.py        # real ten-ETF universe, all 6 M2 configs + benchmark
python scripts/run_robustness_checks.py  # 2-state HMM, no-VNQ, no-HYG
python scripts/run_hrp_and_nu_robustness.py  # HRP vs ERC, shared vs per-regime nu
```
`run_smoke_test.py` runs the whole pipeline on a synthetic dataset (see
`src/data.py::make_synthetic_universe`) and writes a results table to
`outputs/smoke_test_summary.csv`; read the header comment in that script
before citing any number from it — it exists to prove the plumbing
works, not to say anything about the real research questions. The other
scripts run on the real data in `data/raw/` and write to
`outputs/real_data_summary.csv` (gross), `real_data_cost_sensitivity.csv`
(net of costs), and `outputs/robustness_*.csv`, among others.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
python scripts/run_smoke_test.py
```

`torch` and `cvxpylayers` are required only for `allocation_lstm.py`'s
neural-network and end-to-end-training pieces (the latter pulls in
`cvxpy`/`diffcp` as its own dependencies). Both are installed and
exercised in this environment; if either is missing in yours, the
corresponding tests skip cleanly rather than failing (see
`allocation_lstm.require_torch` / `require_cvxpylayers`). Everything
else — HMM, densities, reliability, ERC, constraints, evaluation, and
the ERC-only walk-forward harness — does not depend on either package.

## Reproducibility

### Which code produced the reported results

| Run | Commit | Archive | Report |
|:---|:---|:---|:---|
| M6 reference run (seed 123) | `6a6b12de19b567ba7c1feda10bd5423cd72b5e23` | [`outputs/archive/m6_reference_run_6a6b12d/`](outputs/archive/m6_reference_run_6a6b12d/) | Tables 5 and 6, Figure 4, DSR |
| Ten LSTM seeds on the final pipeline | `c1e59fa53f779151e15554773befdd6b7845fa53` | [`outputs/archive/m7_ten_seed_final_pipeline_c1e59fa/`](outputs/archive/m7_ten_seed_final_pipeline_c1e59fa/) | Table 7 |
| Fixed versus quarterly-retrained LSTM | `32a56f84780b9dee7d225486fb7f7cb1b3185613` | [`outputs/archive/m7_quarterly_lstm_retraining_32a56f8/`](outputs/archive/m7_quarterly_lstm_retraining_32a56f8/) | Appendix A.1 |

The two M7 commits add a runner under `scripts/verification/` and leave `src/`,
`scripts/run_real_data.py` and the lock file identical to `6a6b12d`. Each archive has a
`MANIFEST.md` with its environment, settings and the checks it passed; both M7 runs reproduce the
seed-123 checkpoints and returns of the M6 run. The two M7 designs are pre-registered in
[`docs/verification/`](docs/verification/).

After M6 the summary table's turnover column was corrected: it averaged a daily figure that
counted price drift as turnover. It now reports mean turnover per monthly rebalancing decision,
the figure transaction costs are charged on. Costs, returns and every other column are
unchanged. The archived `real_data_summary.csv` still has the old `avg_turnover` column, so Table
5's turnover is taken from the archived binding-constraint files (see the table below).

The top-level `outputs/real_data_*.csv` files are not the M6 run's outputs; cite the archive.
Running `scripts/run_real_data.py` rewrites those top-level files.

### Data

`data/raw/` holds the inputs of every reported run, so nothing else needs downloading:

- `SPY.csv` ... `VNQ.csv`: daily adjusted close and adjusted open of the ten ETFs
  (`Date, AdjClose, AdjOpen`, 4,893 days from 2007-04-11 to 2026-09-21), downloaded from Yahoo
  Finance with `yfinance` on 2026-09-23;
- `risk_free_DGS3MO_daily.csv`: the FRED DGS3MO series as the runs used it, identical to
  `outputs/archive/frozen_reference_5pct_bb46b63/data/risk_free_DGS3MO_daily.csv`.

Their SHA-256 hashes equal those recorded in
`outputs/archive/m6_reference_run_6a6b12d/run_manifest.json`. A fresh download will not match,
because Yahoo revises adjusted prices, so use these files rather than re-downloading.

### Environment

LSTM training is not reproducible across machines: the same seed gives different networks on a
Mac (arm64) and on Colab (x86_64). The reference runs used a Google Colab CPU runtime (x86_64,
Python 3.13) with:

```bash
pip install torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-reference-lock.txt diffcp==1.1.9
```

Everything that does not train an LSTM (ERC, equal weight, the HMM, the reliability weights, the
robustness checks and inference from the archived checkpoints) reproduces on any machine.

### Running

Run everything from this branch; no checkout is needed. Its `src/` and
`scripts/run_real_data.py` differ from `6a6b12d` only in the summary table's turnover column
(`tests/test_ten_seed_final_pipeline.py` checks this), and older commits do not contain
`data/raw/`.

```bash
pytest tests/ -q
python scripts/run_real_data.py                                         # M6 reference run, about 15 minutes on Colab
python scripts/verification/run_ten_seed_final_pipeline.py --seed 123   # then --seed 0 ... --seed 9
python scripts/verification/run_ten_seed_final_pipeline.py --merge
python scripts/verification/run_quarterly_lstm_retraining.py           # about 2.5 hours on Colab
python scripts/run_robustness_checks.py                                 # Appendix A.2-A.3, ERC only, any machine
python scripts/verification/reliability_and_solver_replay_figures.py   # Sections 4.1 and 4.6, any machine
```

The robustness tables in Appendix A.2-A.3 were generated at `2452a05`, the commit with the
corrected turnover column. The robustness CSVs at `6a6b12d` are older and predate the exact
solver.

To check the reported LSTM returns without retraining, replay the archived networks:

```bash
python scripts/verification/check_replay_against_archive.py \
  --baseline outputs/archive/m6_reference_run_6a6b12d/lstm_checkpoints/seed123_baseline.pt \
  --regime outputs/archive/m6_reference_run_6a6b12d/lstm_checkpoints/seed123_regime.pt \
  --archived outputs/archive/m6_reference_run_6a6b12d/csv/real_data_rq_net_returns_5bps.csv \
  --solver exact
```

### Where each reported result comes from

`M6/` is `outputs/archive/m6_reference_run_6a6b12d/csv/`.

| Report | File | Produced by |
|:---|:---|:---|
| Table 3, HMM refit cadence | `outputs/refit_cadence_solver_check_exact.csv` (the original SLSQP comparison: `refit_cadence_solver_check_slsqp.csv`, `refit_cadence_validation_comparison.csv`) | `scripts/verification/refit_cadence_solver_check.py` |
| Table 4, LSTM budget stability | `outputs/bounded_softmax_u050_200epoch_vs_ordinary_softmax.csv` and `bounded_softmax_u050_200epoch_seed_stability_*.csv` | outputs only; the script is not in the repository |
| Section 3.5, 2.5% versus 5% floor | `outputs/lstm_floor_solver_check.csv` | `scripts/verification/lstm_floor_solver_check.py` |
| Section 3.3, SLSQP zero-weight dates | `outputs/erc_solver_audit.csv` | `scripts/verification/audit_erc_solver.py` |
| Table 5, Sharpe and CAGR at 5/10/25 bps; Figure 4 | `M6/real_data_cost_sensitivity.csv` | M6 reference run |
| Table 5, maximum drawdown and HHI | `M6/real_data_summary.csv` | M6 reference run |
| Table 5, turnover; Section 3.6, binding counts | `M6/real_data_binding_constraints_<config>.csv`: mean of `turnover`, and counts of the binding columns, over the 92 decisions | M6 reference run |
| Table 6 | `M6/real_data_rq_stationary_bootstrap.csv` (return series: `real_data_rq_excess_returns_5bps.csv`) | M6 reference run |
| Section 4.1, reliability path | `M6/real_data_pi_path.csv` | M6 reference run |
| Section 4.1, log-score advantage and its correlation with blending | printed, from the walk-forward and the archive | `scripts/verification/reliability_and_solver_replay_figures.py` |
| Section 4.6, seed-123 regime effect under SLSQP and the exact solver | `M6/real_data_rq_net_returns_5bps.csv` and `outputs/archive/frozen_reference_5pct_bb46b63/csv/real_data_rq_net_returns_5bps.csv` | `scripts/verification/reliability_and_solver_replay_figures.py` |
| Section 4.4, DSR | `M6/real_data_dsr_sensitivity_grid.csv`, `outputs/trial_log.csv`, `outputs/trial_candidate_sharpes.csv` | M6 reference run |
| Table 7 | `ten_seed_rq_distribution.csv`, `ten_seed_rq_per_seed.csv`, `ten_seed_averaged_rq.csv` in the ten-seed archive | `scripts/verification/run_ten_seed_final_pipeline.py` |
| Section 5, exact-solver replay of the ten archived seeds | `outputs/ten_seed_corrected_bootstrap.csv` | `scripts/verification/ten_seed_corrected_bootstrap.py` |
| Sections 3.6 and 5, opening turnover, strict cap, return attribution | `outputs/verification/m6_implementation/` | see its README |
| Section 3.7, risk-free convention | `docs/verification/risk_free_convention.md` | |
| Appendix A.1 | `lstm_fixed_vs_quarterly_performance.csv`, `lstm_quarterly_minus_fixed_bootstrap.csv`, `rq_contrasts_by_arm.csv` in the quarterly archive | `scripts/verification/run_quarterly_lstm_retraining.py` |
| Appendix A.2, excluding HYG | `outputs/robustness_no_hyg_summary.csv` | `scripts/run_robustness_checks.py` |
| Appendix A.3, two-state HMM | `outputs/robustness_2state_summary.csv`, `robustness_primary_3state_summary.csv`, `robustness_regime_effect_by_spec.csv` | `scripts/run_robustness_checks.py` |

Subsequent experimental branches and post-reference verification files should not be confused
with independent replications of the original study.

## Project Status

The final (M7) report adds the ten-seed rerun, quarterly LSTM retraining, and the
excluding-HYG and two-state robustness results to the M6 reference results. The primary
specification is unchanged.

