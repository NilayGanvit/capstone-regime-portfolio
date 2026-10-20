# LSTM Seed-123 Colab Reproducibility Verification

Date: 2026-10-19

## Purpose

Investigate differences between the 10-seed Colab robustness experiment
and the 15-seed Mac robustness experiment, particularly the conflicting
results for seed 123.

This verification does not replace either experiment or select a new
model specification.

## Source versions

- Frozen Colab reference: bb46b63
- Nilay's robustness branch: d94db1e
- Specification: bounded Softmax, 5% minimum risk-budget allocation
- LSTM training: 200 epochs
- Seed: 123
- Transaction-cost setting: 5 basis points
- Evaluation period: 2019-01-02 through 2026-08-31

## Seed-123 reproduction

The training procedure from Nilay's 15-seed robustness script was
executed in Colab using the source modules from the checked-out
frozen-reference commit.

All 1,926 final-test daily net returns matched the frozen reference
exactly for:

- lstm_baseline
- lstm_regime
- lstm_blend

Maximum absolute daily-return difference: 0.0

The reproduction used zero-risk-free Sharpe calculations for comparison
with Nilay's robustness-script reporting convention.

## Ten-seed Colab verification

The existing ten-seed experiment covers seeds 0 through 9.

All ten saved daily-return files contain 1,926 observations covering
2019-01-02 through 2026-08-31.

All 30 reported excess-return Sharpe ratios were independently
recalculated using the saved net returns and lagged FRED DGS3MO
daily risk-free series.

Maximum absolute Sharpe difference: 1.6542323066914832e-14

Results:

- Mean regime-minus-baseline excess Sharpe: +0.098071
- Standard deviation: 0.159025
- Positive regime Sharpe effect: 7 of 10 seeds
- Less severe maximum drawdown: 10 of 10 seeds
- Mean maximum-drawdown improvement: 0.034166

## Interpretation and limitations

The seed-123 Colab reproduction establishes exact agreement between
the tested training procedure and the frozen reference under the
Colab execution setup.

It does not isolate the cause of the differing Mac seed-123 result.
The two complete robustness experiments have not been reproduced
under identical conditions.

The 10-seed Colab and 15-seed Mac experiments use different seed
samples and Sharpe conventions. Their aggregate results should not
be treated as directly interchangeable.

The ten-seed Sharpe verification confirms the use of the saved
FRED DGS3MO risk-free series. It does not independently reconstruct
transaction costs from gross returns and execution records.

Budget-floor concentration across the ten seeds remains to be
verified separately.

## Evidence location

Shared Drive:

Capstone_Shared/Code/verification_seed123_colab/

Files:

- nilay_seed123_colab_comparison.csv
- verification.json

The original frozen reference and ten-seed results remain unchanged.

## Remaining integration decisions

1. Agree on the canonical T-bill source and observation timing.
2. Confirm the final bounded-Softmax Table 5 and Table 6 provenance.
3. Verify ten-seed budget-floor concentration.
4. Integrate compatible bootstrap, DSR, and reporting changes.
