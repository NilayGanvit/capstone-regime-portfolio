# Problem Statement (Module 2)

> Mirrors the Module 2 submission for reference from the code. If the
> submitted document changes, update this file to match.

**Track 7 - Medium/Long Term Trading Models**

Our project looks at a fairly simple question: can information about
market regimes improve medium-term portfolio allocation?

Our goal is to determine whether explicitly conditioning dynamic
portfolio allocation on regime information improves net risk-adjusted
performance relative to the same allocator without regime information.

We compare two approaches. The first is long-only Equal Risk
Contribution (ERC) Risk Parity, with equal and fixed risk budgets. The
second uses an LSTM model that learns time-varying risk budgets. In
both cases, risk budgets are transformed into portfolio weights
through the same risk-budgeting optimization framework.

## Research questions

1. Is regime information explicitly helpful for net risk-adjusted
   performance versus the same allocator without regime information?
2. Does the LSTM-based dynamic risk-budgeting allocator benefit more
   from regime information than fixed-budget ERC Risk Parity?
3. Is there value added by reliability-based blending beyond the raw
   regime probabilities?

## Investment universe

Ten liquid ETFs, daily data available since April 2007, monthly
rebalancing:

| Exposure | ETF symbols |
|---|---|
| US, developed-market and emerging-market equities | SPY, EFA, EEM |
| Intermediate and long-duration US Treasuries | IEF, TLT |
| Investment-grade and high-yield corporate credit | LQD, HYG |
| Gold and broad commodities | GLD, DBC |
| Listed real estate | VNQ |

## Evaluation design

Six configurations on identical out-of-sample dates, plus a fixed
10%-per-ETF equal-weight benchmark:

| Allocator | Baseline | Raw regime | Reliability blend |
|---|---|---|---|
| ERC | Pooled covariance | State-based covariance | Blend of the two ERC allocations |
| LSTM | Market features → learned budgets | + regime probabilities → learned budgets | Blend of the two LSTM allocations |

- **Regime effect** = Regime − Baseline, within each allocator
- **Reliability effect** = Blend − Regime, within each allocator
- **ML contribution** = Δ_LSTM − Δ_ERC (Δ = regime effect for that allocator)

## Key references

- Guidolin, M., & Timmermann, A. (2007). Asset Allocation under
  Multivariate Regime Switching. *JEDC*.
- Costa, G., & Kwon, R. H. (2019). Risk Parity Portfolio Optimization
  under a Markov Regime-Switching Framework. *Quantitative Finance*.
- Uysal, A. S., Li, X., & Mulvey, J. M. (2021). End-to-End Risk
  Budgeting Portfolio Optimization with Neural Networks. *arXiv:2107.04636*.
- Raftery, A. E., et al. (2010). Online Prediction under Model
  Uncertainty via Dynamic Model Averaging. *Technometrics*.
- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe
  Ratio. *Journal of Portfolio Management*.

Full text: see the Module 2 submission document.
