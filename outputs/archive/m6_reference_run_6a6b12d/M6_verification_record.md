# M6 Final Reference — Verification Record

## Reference specification

- Git branch: `m6/final-pipeline`
- Git commit: `6a6b12de19b567ba7c1feda10bd5423cd72b5e23`
- Working tree at execution: clean
- Risk-budget solver: exact
- LSTM bounded-Softmax floor: 0.05
- LSTM seed: 123
- LSTM epochs: 200
- Headline transaction cost: 5 bps
- Risk-free source: FRED DGS3MO
- Historical evaluation: 2019-01-02 through 2026-08-31

## Execution and archive

The final integrated pipeline completed successfully.
The archived reference includes model checkpoints, performance
results, research-question inference, cost sensitivity,
constraint diagnostics, and the run manifest.

The execution log is saved as `reference_run.log`.

## Accounting and solver tests

Tests executed:

- `tests/test_accounting_reconciliation.py`
- `tests/test_solver_agreement.py`
- `tests/test_risk_budget_solver_option.py`
- `tests/test_evaluation.py`
- `tests/test_constraints.py`

Result: 86 passed, 4 expected failures, 0 unexpected
failures. Exit code 0.

## Archived transaction-cost verification

All seven configurations passed the archived transaction-cost
check:

- ERC baseline
- ERC regime
- ERC blend
- LSTM baseline
- LSTM regime
- LSTM blend
- Equal weight

Each transaction log contains 92 dated entries with no
duplicate dates.

Maximum absolute discrepancy between recorded transaction
cost and turnover multiplied by fee rate: 0.0000005,
consistent with six-decimal CSV rounding.

## Net-return series

- Observations: 1,926
- First date: 2019-01-02
- Last date: 2026-08-31
- Dates strictly increasing: yes
- Duplicate dates: none
- Nonfinite returns: none
- Returns less than or equal to -100%: none

Overall archived-output check: PASS.

## Interpretation and limitations

The final M6 reference run is the authoritative source for
the project draft's reported performance and inference.

The accounting tests validate the implementation under
their test scenarios. The archived transaction-cost check
validates the reported fee calculations and return-series
integrity; it is not an independent reconstruction of
every historical portfolio holding or NAV.

Earlier ten-seed robustness results were produced from
a different code revision and should not be presented as
confirmed robustness evidence for the final exact-solver
implementation without further verification.

The 2019–2026 evaluation is exploratory historical
walk-forward evidence, not an untouched independent holdout.

## Reporting guidance

Use `real_data_cost_sensitivity.csv` with `fee_bps == 5.0`
for headline net Sharpe and CAGR. `real_data_summary.csv`
contains gross performance.

Use `real_data_rq_stationary_bootstrap.csv` for the final
research-question confidence intervals.

Do not combine performance results from earlier development
commits with final M6 reference results without labeling
the specification difference.
