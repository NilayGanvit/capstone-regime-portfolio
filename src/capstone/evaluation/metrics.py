"""
Evaluation metrics.

Owner: Nilay

Responsibilities (from the M2 problem statement):
- Primary metric: Sharpe ratio on out-of-sample, net-of-cost returns.
- Secondary: CAGR, volatility, max drawdown, Sortino, Calmar, turnover,
  costs, concentration.
- Diagnostics: predictive log scores (M0/M1), state frequencies, pi path.
- Uncertainty: paired block-bootstrap intervals across strategies
  (keep observations paired across the six configurations + benchmark).
- Selection-risk correction: Deflated Sharpe Ratio (Bailey & Lopez de
  Prado), using the model-testing log to state assumptions about the
  effective number of trials.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sharpe_ratio(net_returns: pd.Series, periods_per_year: int = 12) -> float:
    raise NotImplementedError


def cagr(net_returns: pd.Series, periods_per_year: int = 12) -> float:
    raise NotImplementedError


def max_drawdown(net_returns: pd.Series) -> float:
    raise NotImplementedError


def sortino_ratio(net_returns: pd.Series, periods_per_year: int = 12) -> float:
    raise NotImplementedError


def calmar_ratio(net_returns: pd.Series, periods_per_year: int = 12) -> float:
    raise NotImplementedError


def paired_block_bootstrap(returns_by_strategy: dict[str, pd.Series], block_size: int, n_boot: int = 5000):
    """TODO: paired block-bootstrap confidence intervals for performance differences."""
    raise NotImplementedError


def deflated_sharpe_ratio(observed_sharpe: float, n_trials_effective: int, skew: float, kurt: float, n_obs: int) -> float:
    """TODO: Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio."""
    raise NotImplementedError
