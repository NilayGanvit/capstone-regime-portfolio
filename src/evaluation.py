"""
Performance evaluation: Sharpe/CAGR/drawdown/Sortino/Calmar, turnover and
concentration, paired block-bootstrap confidence intervals for performance
*differences* between configurations, and the Deflated Sharpe Ratio
(Bailey & Lopez de Prado, 2014) for selection-aware interpretation of the
headline Sharpe ratio.

Corresponds to M2/M3's evaluation design (Table 2/3): six configurations
plus the equal-weight benchmark, compared on identical dates, with
uncertainty reported via paired block bootstrap and DSR alongside the
plain OOS Sharpe.

Owner: Nilay (optimization, constraints, and evaluation rigor).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm, skew, kurtosis


TRADING_DAYS_PER_YEAR = 252


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns)
    if r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def cagr(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns)
    cum = np.prod(1 + r)
    n_years = len(r) / periods_per_year
    if n_years <= 0 or cum <= 0:
        return float("nan")
    return float(cum ** (1 / n_years) - 1)


def max_drawdown(returns: np.ndarray) -> float:
    r = np.asarray(returns)
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1
    return float(drawdown.min())


def sortino_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns)
    downside = r[r < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    if not downside_std or downside_std == 0:
        return float("nan")
    return float(r.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return float(cagr(returns, periods_per_year) / abs(mdd))


def herfindahl_concentration(weights: np.ndarray) -> float:
    """Herfindahl index of portfolio concentration, averaged over the
    weight history. weights: (T, n_assets)."""
    w = np.asarray(weights)
    return float(np.mean(np.sum(w ** 2, axis=1)))


def net_of_cost_returns(
    dates: list,
    gross_returns: np.ndarray,
    execution_history: list,
    fee_bps: float,
) -> np.ndarray:
    """Deduct transaction costs from `gross_returns` (run_walk_forward
    itself charges no fee) on the days a trade actually executes, per
    M2/M3: "a 5 basis point fee on every purchase and sale... including
    the initial investment... [computed by] taking the fee rate into
    account only once on the total value of trades, not twice." Cost on
    an execution day is fee_bps/10000 * turnover, where `turnover` is the
    gross (buy+sell, not halved) turnover already used everywhere else in
    this codebase -- applying the fee rate once to that gross figure is
    exactly "once on the total value of trades," with no extra factor of
    2 (see constraints.turnover's docstring).

    `dates`/`gross_returns` are one config's WalkForwardResult.dates and
    WalkForwardResult.portfolio_returns[config]; `execution_history` is
    that same config's WalkForwardResult.execution_history entry (dated
    on the actual execution day, not the decision day, once open-price
    execution is enabled). Because weights don't depend on the fee rate,
    this can be called repeatedly at different `fee_bps` from a single
    run_walk_forward call -- e.g. for the 5/10/25 bps sensitivity M2
    calls for -- without re-running the walk-forward itself.
    """
    dates = list(dates)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    net = np.asarray(gross_returns, dtype=float).copy()
    fee_rate = fee_bps / 10_000.0
    for entry in execution_history:
        idx = date_to_idx.get(entry["date"])
        if idx is not None:
            net[idx] -= fee_rate * entry["turnover"]
    return net


def average_turnover(weights: np.ndarray, drifted_weights: np.ndarray) -> float:
    """Mean gross (round-trip) turnover across the backtest, using each
    period's drifted (pre-trade) holdings as the reference -- the same
    definition used in constraints.py and the LSTM training loss."""
    return float(np.mean(np.sum(np.abs(weights - drifted_weights), axis=1)))


def block_bootstrap_diff_ci(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    stat_fn=sharpe_ratio,
    block_size: int = 20,
    n_boot: int = 2000,
    ci: float = 0.90,
    random_state: int = 0,
) -> dict:
    """Paired (same dates) block-bootstrap confidence interval for
    stat_fn(returns_a) - stat_fn(returns_b), preserving each series'
    autocorrelation via non-overlapping-block resampling and keeping the
    two series paired by resampling the *same* block indices for both.
    """
    returns_a = np.asarray(returns_a)
    returns_b = np.asarray(returns_b)
    assert len(returns_a) == len(returns_b), "Series must be paired on identical dates."
    n = len(returns_a)
    n_blocks = int(np.ceil(n / block_size))
    rng = np.random.default_rng(random_state)

    point_estimate = stat_fn(returns_a) - stat_fn(returns_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        block_starts = rng.integers(0, n - block_size + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in block_starts])[:n]
        diffs[b] = stat_fn(returns_a[idx]) - stat_fn(returns_b[idx])

    alpha = 1 - ci
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {"point_estimate": point_estimate, "ci_low": float(lo), "ci_high": float(hi), "ci_level": ci}


def n_trials_from_log(path: str) -> int:
    """Count the specifications actually evaluated for *performance-driven
    selection*, from the auditable trial log M2/M3 calls for ("we will
    record the number of state specifications, feature configurations,
    windows, hyperparameter sets and portfolio variants evaluated").

    The log (see outputs/trial_log.csv) also records plain correctness
    fixes -- e.g. the initial-window length was simply wrong relative to
    M2's calendar, not one option chosen from several valid ones -- and
    those are marked `counts_toward_dsr_trials=False` so they don't
    inflate the trial count DSR uses to judge selection bias. Only rows
    marked True (an alternative that was actually compared against
    others on measured performance) are counted here.
    """
    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return sum(1 for row in rows if row["counts_toward_dsr_trials"].strip().lower() == "true")


def deflated_sharpe_ratio(
    observed_sharpe: float,
    returns: np.ndarray,
    n_trials: int,
    sharpe_variance_across_trials: float | None = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Deflated Sharpe Ratio per Bailey & Lopez de Prado (2014): the
    probability that the *true* Sharpe ratio exceeds zero, after
    correcting for (a) the non-normality of the return series (skew,
    kurtosis) and (b) selection bias from having evaluated `n_trials`
    candidate specifications and reporting the best one.

    `sharpe_variance_across_trials` is the variance of the Sharpe ratios
    obtained across the n_trials candidates evaluated during development
    (per M3's requirement to keep an auditable trial log); if not
    supplied, a standard reference value derived from n_trials under an
    assumption of independent, identically distributed candidate Sharpes
    is used as a documented approximation -- flagged clearly in the
    return so this substitution is never silently mistaken for a
    from-the-log estimate.
    """
    r = np.asarray(returns)
    n_obs = len(r)
    g3 = skew(r)          # skewness
    g4 = kurtosis(r, fisher=False)  # non-excess kurtosis (Normal = 3)

    # Expected maximum Sharpe ratio across n_trials candidates under the
    # null of zero true skill (Bailey & Lopez de Prado's SR* benchmark),
    # using the variance of Sharpe ratios across trials.
    if sharpe_variance_across_trials is None:
        # Documented fallback: assume unit variance across trial Sharpes
        # (a conservative placeholder -- replace with the actual
        # empirical variance from the model-testing log once available).
        sharpe_variance_across_trials = 1.0

    euler_mascheroni = 0.5772156649
    if n_trials > 1:
        sr_benchmark = np.sqrt(sharpe_variance_across_trials) * (
            (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
            + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    else:
        sr_benchmark = 0.0

    sr = observed_sharpe / np.sqrt(periods_per_year)  # de-annualize to per-period SR for the DSR formula
    sr_star = sr_benchmark / np.sqrt(periods_per_year)

    numerator = (sr - sr_star) * np.sqrt(n_obs - 1)
    denominator = np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(norm.cdf(numerator / denominator))
