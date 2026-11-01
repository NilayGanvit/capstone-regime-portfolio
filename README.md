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
| [`data/raw/`](data/raw/) | Archived ETF prices and risk-free-rate data |
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

## Reproducability
Install the project dependencies in a compatible Python environment:

pip install -r requirements.txt
pytest tests/ -v

The reference environment is documented separately in requirements-reference-lock.txt. Running the full LSTM training pipeline on a different machine may produce different fitted networks even when the random seed is held fixed.

The frozen reference commit, archived data, and saved checkpoints identify the numerical results used in the M6 report. Subsequent experimental branches and post-reference verification files should not be confused with independent replications of the original study.

## Project Status

The M6 draft project report has been submitted. The frozen reference implementation and its principal numerical results are documented, with additional robustness and exploratory experiments maintained separately.

Further work for the final report will focus on incorporating relevant evaluation feedback, explaining the limitations of the historical comparisons, and reporting robustness evidence without expanding the primary model specification unnecessarily.

