# M6 Implementation Verification

These files document post-reference implementation checks for the
MSc Financial Engineering capstone project.

## Reference specification

- Repository: NilayGanvit/capstone-regime-portfolio
- Branch: m6/final-pipeline
- Reference commit: 6a6b12de19b567ba7c1feda10bd5423cd72b5e23
- Evaluation period: January 2019 through August 2026
- Original reference outputs were not modified.

## Opening turnover and transaction costs

The reference implementation calculates turnover and applies the
30% turnover constraint using decision-close holdings. Trades
execute at the following opening, so overnight price movements
can cause realized opening turnover to exceed that limit.

The verification identified 17 opening-turnover violations across
seven portfolio configurations, with a maximum opening turnover
of approximately 30.6715%.

Additional sensitivity checks calculated transaction costs using
opening turnover and enforced the 30% turnover cap at execution.
The strict-cap scenario eliminated all 17 violations. These
adjustments did not change the qualitative conclusions for RQ1-RQ3.

## ERC versus equal-weight attribution

Gross return contributions were reconstructed using archived
portfolio weights and ETF returns. The calculated portfolio
returns reconciled with the reference implementation.

In the arithmetic contribution decomposition of the ERC baseline
versus equal-weight gap:

- Equities (SPY, EFA, EEM): approximately 74.5%
- REITs (VNQ): approximately 20.3%
- Other assets: approximately 5.2%

This is a descriptive gross-return attribution, not a causal
estimate or proof that equal weighting is generally superior.

## Interpretation and reproducibility

These CSV files are supporting verification outputs, not a
replacement for the reference results. They were generated from
a separate Colab verification notebook using the saved reference
run, ETF data, and LSTM checkpoints.

The original notebook depends on separately stored Google Drive
inputs and is not included as a standalone reproducible test.
The manifest identifies the frozen reference commit and the
verification files.

The checks did not retrain the models or modify the archived
reference results.
