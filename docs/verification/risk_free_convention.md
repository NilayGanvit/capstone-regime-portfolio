# Risk-free convention for excess-return Sharpe ratios

Every headline Sharpe, the paired stationary bootstrap and the Deflated Sharpe
Ratio grid are computed on daily returns in excess of one risk-free series.

## Definition

- **Source:** FRED `DGS3MO`, the market yield on 3-month U.S. Treasury
  securities at constant maturity (annualised percent).
- **Lag:** the rate applied to the return on date `d` is the last yield observed
  strictly before `d`. The yield for `d` is only known at `d`'s close.
- **Conversion:** compounded to a daily rate over 252 periods,
  `rf_daily = (1 + y/100) ** (1/252) - 1`. This is not simple division by 252;
  the two differ by about 0.007 in Sharpe over the final-test period.
- **Alignment:** the file covers the ETF return calendar exactly.
  `data.load_risk_free_daily` raises on any missing date instead of filling it.
- **Sharpe:** `mean(r - rf) / sd(r - rf) * sqrt(252)`, the risk-free rate
  subtracted before both the mean and the standard deviation
  (`evaluation.sharpe_ratio(..., risk_free=...)`).

## Reproducing the file

`python scripts/fetch_risk_free.py` downloads the series and writes
`data/raw/risk_free_DGS3MO_daily.csv` (`date, yield_pct_prior, rf_daily`).
FRED revises past observations, so record the download date. The frozen
Colab reference was built on an earlier download; rebuilding today differs from
it on 4 of 4,892 rows (2012-04-09, 2012-10-31, 2023-04-10, 2026-04-06), by at
most 0.04 percentage points in the yield and 1.5e-6 in the daily rate.

## Checks against the frozen reference (`frozen_reference_5pct`)

- Run end to end with the frozen file, the ERC and equal-weight rows of the
  summary and the cost-sensitivity table match the frozen run exactly
  (difference 0), the excess-return file matches to 3e-11, and the ERC
  bootstrap contrasts and intervals match to six decimals. Their LSTM rows
  differ, as expected: LSTM training is not reproducible across Mac and Colab.
- `evaluation.sharpe_ratio(r, risk_free=rf)` on the frozen net returns
  reproduces the frozen Table 5 Sharpes to rounding (maximum difference 0.0005).

## Alternatives considered

Yahoo `^IRX` (a discount-basis auction quote) reads about 0.12 pp below DGS3MO
on average over 2019-2026 and by about 0.2 pp when rates are above 4%, so it
understates the return on cash. Using it instead moves final-test Sharpe by at
most 0.014 and the regime effects by at most 0.0013; ordering is unchanged under
every convention tried (^IRX or FRED, same-day or lagged, compounded or
simple). Lagging by a day changes Sharpe by less than 0.0002.

## Known limitation

`outputs/trial_candidate_sharpes.csv` holds zero-risk-free Sharpes recovered from
commit messages and earlier CSVs, with no daily return paths to recompute.
The DSR sensitivity grid uses only their cross-trial variance, which a common
shift leaves unchanged, but candidates from different periods or volatilities
would shift by different amounts, so those variance scenarios are approximate on
the excess basis.
