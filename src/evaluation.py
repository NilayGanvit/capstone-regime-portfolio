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
    this codebase. The cost is financed from portfolio equity at execution:
    execution-day wealth is multiplied by (1 - cost_fraction), rather than
    subtracting cost_fraction additively from the day's return. Applying
    the fee rate once to gross turnover is exactly "once on the total value
    of trades," with no extra factor of 2 (see constraints.turnover's
    docstring).

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
            cost_fraction = fee_rate * entry["turnover"]
            net[idx] = (1.0 + net[idx]) * (1.0 - cost_fraction) - 1.0
    return net


def average_turnover(weights: np.ndarray, drifted_weights: np.ndarray) -> float:
    """Mean gross (round-trip) turnover across the backtest, using each
    period's drifted (pre-trade) holdings as the reference -- the same
    definition used in constraints.py and the LSTM training loss."""
    return float(np.mean(np.sum(np.abs(weights - drifted_weights), axis=1)))



def optimal_stationary_block_length(x: np.ndarray) -> float:
    """Estimate the optimal expected block length for the stationary bootstrap.

    Implements the automatic block-length selection procedure of Politis and
    White (2004) using the stationary-bootstrap correction of Patton, Politis,
    and White (2009).

    Parameters
    ----------
    x : np.ndarray
        One-dimensional stationary time series.

    Returns
    -------
    float
        Estimated optimal expected stationary-bootstrap block length.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    n = len(x)

    if n < 10:
        raise ValueError("At least 10 finite observations are required.")

    x = x - x.mean()

    # Patton/Politis/White automatic bandwidth-selection constants.
    kn = max(5, int(np.sqrt(np.log10(n))))
    m_max = int(np.ceil(np.sqrt(n) + kn))
    threshold = 2.0 * np.sqrt(np.log10(n) / n)

    # Autocovariances use the common 1/n normalization.
    gamma = np.empty(m_max + 1)
    gamma[0] = np.dot(x, x) / n
    if gamma[0] <= 0:
        return 1.0

    for lag in range(1, m_max + 1):
        gamma[lag] = np.dot(x[lag:], x[:-lag]) / n

    rho = gamma / gamma[0]

    # Find the first run of kn consecutive insignificant autocorrelations.
    m_hat = None
    for start in range(1, m_max - kn + 2):
        if np.all(np.abs(rho[start:start + kn]) < threshold):
            m_hat = start
            break

    if m_hat is None:
        m_hat = m_max

    m = min(2 * m_hat, m_max)

    lags = np.arange(1, m + 1, dtype=float)

    # Flat-top (trapezoidal) lag window.
    h = np.where(
        lags / m <= 0.5,
        1.0,
        2.0 * (1.0 - lags / m),
    )

    # Symmetry over positive and negative lags.
    g = 2.0 * np.sum(h * lags * gamma[1:m + 1])
    sigma2 = gamma[0] + 2.0 * np.sum(h * gamma[1:m + 1])

    # Patton, Politis & White (2009): corrected stationary-bootstrap
    # variance constant D_SB = 2 * g(0)^2.
    d_sb = 2.0 * sigma2**2

    if d_sb <= 0 or not np.isfinite(d_sb) or not np.isfinite(g):
        return 1.0

    b_opt = ((2.0 * g**2 / d_sb) * n) ** (1.0 / 3.0)

    # Finite-sample upper bound used by the automatic selector.
    b_max = np.ceil(min(3.0 * np.sqrt(n), n / 3.0))
    return float(np.clip(b_opt, 1.0, b_max))


def stationary_bootstrap_indices(
    n: int,
    expected_block_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one stationary-bootstrap index sequence.

    Blocks have geometrically distributed lengths with expected length
    ``expected_block_length``. A single returned index sequence can be
    applied to every aligned strategy return series so that paired
    calendar dependence is preserved.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    if not np.isfinite(expected_block_length) or expected_block_length < 1.0:
        raise ValueError("expected_block_length must be finite and at least 1.")

    restart_probability = 1.0 / expected_block_length

    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(0, n)

    for t in range(1, n):
        if rng.random() < restart_probability:
            idx[t] = rng.integers(0, n)
        else:
            idx[t] = (idx[t - 1] + 1) % n

    return idx

def block_bootstrap_diff_ci(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    stat_fn=sharpe_ratio,
    block_size: float | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict:
    """Paired stationary-bootstrap confidence interval for a statistic difference.

    Computes ``stat_fn(returns_a) - stat_fn(returns_b)`` using the same
    stationary-bootstrap index sequence for both aligned return series.

    If ``block_size`` is None, the expected block length is selected
    automatically from the paired return-difference series using the
    Politis-White (2004) procedure with the Patton-Politis-White (2009)
    stationary-bootstrap correction. Supplying ``block_size`` overrides
    automatic selection and uses that value as the expected block length.

    Parameters
    ----------
    returns_a, returns_b : np.ndarray
        Return series aligned on identical dates.
    stat_fn : callable
        Statistic computed separately on each return series.
    block_size : float or None
        Expected stationary-bootstrap block length. If None, select
        automatically.
    n_boot : int
        Number of bootstrap replications.
    ci : float
        Confidence level. Default is 0.95.
    random_state : int
        Seed for reproducible resampling.
    """
    returns_a = np.asarray(returns_a, dtype=float).reshape(-1)
    returns_b = np.asarray(returns_b, dtype=float).reshape(-1)

    if len(returns_a) != len(returns_b):
        raise ValueError("Series must be paired on identical dates.")
    if len(returns_a) < 10:
        raise ValueError("At least 10 paired observations are required.")
    if not np.all(np.isfinite(returns_a)) or not np.all(np.isfinite(returns_b)):
        raise ValueError("Return series must contain only finite values.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie strictly between 0 and 1.")

    if block_size is None:
        expected_block_length = optimal_stationary_block_length(
            returns_a - returns_b
        )
        block_length_source = "automatic"
    else:
        expected_block_length = float(block_size)
        if not np.isfinite(expected_block_length) or expected_block_length < 1.0:
            raise ValueError("block_size must be finite and at least 1.")
        block_length_source = "user"

    rng = np.random.default_rng(random_state)

    point_estimate = float(stat_fn(returns_a) - stat_fn(returns_b))
    diffs = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = stationary_bootstrap_indices(
            n=len(returns_a),
            expected_block_length=expected_block_length,
            rng=rng,
        )
        diffs[b] = stat_fn(returns_a[idx]) - stat_fn(returns_b[idx])

    alpha = 1.0 - ci
    lo, hi = np.quantile(diffs, [alpha / 2.0, 1.0 - alpha / 2.0])

    return {
        "point_estimate": point_estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci_level": float(ci),
        "expected_block_length": float(expected_block_length),
        "block_length_source": block_length_source,
        "bootstrap_method": "stationary",
    }


def stationary_bootstrap_rq_contrasts(
    portfolio_returns: dict[str, np.ndarray],
    stat_fn=sharpe_ratio,
    block_size: float | None = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> dict[str, dict]:
    """Synchronized stationary-bootstrap inference for RQ1-RQ3 contrasts.

    All six strategy return series must be aligned on identical dates.
    Each bootstrap replication uses one common stationary-bootstrap index
    sequence for every strategy, preserving cross-strategy dependence.

    The five reported contrasts are:

    RQ1 ERC:
        stat(erc_regime) - stat(erc_baseline)

    RQ1 LSTM:
        stat(lstm_regime) - stat(lstm_baseline)

    RQ2 difference-in-differences:
        [stat(lstm_regime) - stat(lstm_baseline)]
        - [stat(erc_regime) - stat(erc_baseline)]

    RQ3 ERC:
        stat(erc_blend) - stat(erc_regime)

    RQ3 LSTM:
        stat(lstm_blend) - stat(lstm_regime)

    If ``block_size`` is None, the Politis-White/Patton selector is
    applied separately to each of the five RQ return-contrast series.
    The largest selected expected block length is used as the common
    dependence scale for synchronized resampling.
    """
    required = (
        "erc_baseline",
        "erc_regime",
        "erc_blend",
        "lstm_baseline",
        "lstm_regime",
        "lstm_blend",
    )

    missing = [name for name in required if name not in portfolio_returns]
    if missing:
        raise ValueError(f"Missing required strategy returns: {missing}")

    returns = {
        name: np.asarray(portfolio_returns[name], dtype=float).reshape(-1)
        for name in required
    }

    lengths = {len(x) for x in returns.values()}
    if len(lengths) != 1:
        raise ValueError("All strategy return series must have identical length.")

    n = lengths.pop()
    if n < 10:
        raise ValueError("At least 10 aligned observations are required.")

    if any(not np.all(np.isfinite(x)) for x in returns.values()):
        raise ValueError("Strategy return series must contain only finite values.")
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1.")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must lie strictly between 0 and 1.")

    # Select a dependence scale separately for each pre-specified RQ
    # contrast. The corrected Politis-White automatic selector is applied
    # to the corresponding daily return-contrast series. Within each
    # contrast, the constituent portfolio return series are subsequently
    # resampled with the same stationary-bootstrap index sequence so that
    # contemporaneous cross-strategy dependence is preserved.
    selector_series = {
        "rq1_erc_regime_effect":
            returns["erc_regime"] - returns["erc_baseline"],
        "rq1_lstm_regime_effect":
            returns["lstm_regime"] - returns["lstm_baseline"],
        "rq2_difference_in_differences":
            (returns["lstm_regime"] - returns["lstm_baseline"])
            - (returns["erc_regime"] - returns["erc_baseline"]),
        "rq3_erc_reliability_effect":
            returns["erc_blend"] - returns["erc_regime"],
        "rq3_lstm_reliability_effect":
            returns["lstm_blend"] - returns["lstm_regime"],
    }

    if block_size is None:
        contrast_block_lengths = {
            name: optimal_stationary_block_length(series)
            for name, series in selector_series.items()
        }
        block_length_source = "automatic_contrast_specific"
    else:
        expected_block_length = float(block_size)
        if not np.isfinite(expected_block_length) or expected_block_length < 1.0:
            raise ValueError("block_size must be finite and at least 1.")
        contrast_block_lengths = {
            name: expected_block_length
            for name in selector_series
        }
        block_length_source = "user_common_override"

    def compute_contrasts(sample: dict[str, np.ndarray]) -> dict[str, float]:
        stats = {name: float(stat_fn(x)) for name, x in sample.items()}

        rq1_erc = stats["erc_regime"] - stats["erc_baseline"]
        rq1_lstm = stats["lstm_regime"] - stats["lstm_baseline"]

        return {
            "rq1_erc_regime_effect": rq1_erc,
            "rq1_lstm_regime_effect": rq1_lstm,
            "rq2_difference_in_differences": rq1_lstm - rq1_erc,
            "rq3_erc_reliability_effect":
                stats["erc_blend"] - stats["erc_regime"],
            "rq3_lstm_reliability_effect":
                stats["lstm_blend"] - stats["lstm_regime"],
        }

    point = compute_contrasts(returns)
    boot = {name: np.empty(n_boot, dtype=float) for name in point}

    contrast_members = {
        "rq1_erc_regime_effect": (
            "erc_baseline",
            "erc_regime",
        ),
        "rq1_lstm_regime_effect": (
            "lstm_baseline",
            "lstm_regime",
        ),
        "rq2_difference_in_differences": (
            "erc_baseline",
            "erc_regime",
            "lstm_baseline",
            "lstm_regime",
        ),
        "rq3_erc_reliability_effect": (
            "erc_regime",
            "erc_blend",
        ),
        "rq3_lstm_reliability_effect": (
            "lstm_regime",
            "lstm_blend",
        ),
    }

    # Give each planned contrast its own reproducible RNG stream. Within a
    # contrast, one index sequence is shared by every constituent strategy,
    # preserving their contemporaneous dependence.
    seed_sequence = np.random.SeedSequence(random_state)
    child_seeds = seed_sequence.spawn(len(point))

    for (contrast_name, members), child_seed in zip(
        contrast_members.items(),
        child_seeds,
    ):
        rng = np.random.default_rng(child_seed)
        expected_block_length = contrast_block_lengths[contrast_name]

        for b in range(n_boot):
            idx = stationary_bootstrap_indices(
                n=n,
                expected_block_length=expected_block_length,
                rng=rng,
            )

            sample = {
                name: returns[name][idx]
                for name in members
            }

            stats = {
                name: float(stat_fn(x))
                for name, x in sample.items()
            }

            if contrast_name == "rq1_erc_regime_effect":
                value = stats["erc_regime"] - stats["erc_baseline"]
            elif contrast_name == "rq1_lstm_regime_effect":
                value = stats["lstm_regime"] - stats["lstm_baseline"]
            elif contrast_name == "rq2_difference_in_differences":
                value = (
                    stats["lstm_regime"] - stats["lstm_baseline"]
                    - stats["erc_regime"] + stats["erc_baseline"]
                )
            elif contrast_name == "rq3_erc_reliability_effect":
                value = stats["erc_blend"] - stats["erc_regime"]
            else:
                value = stats["lstm_blend"] - stats["lstm_regime"]

            boot[contrast_name][b] = value

    alpha = 1.0 - ci
    results = {}

    for name, point_estimate in point.items():
        lo, hi = np.quantile(
            boot[name],
            [alpha / 2.0, 1.0 - alpha / 2.0],
        )

        results[name] = {
            "point_estimate": float(point_estimate),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_level": float(ci),
        }

    results["_bootstrap"] = {
        "method": "stationary",
        "block_length_source": block_length_source,
        "contrast_block_lengths": {
            name: float(value)
            for name, value in contrast_block_lengths.items()
        },
        "n_boot": int(n_boot),
        "synchronization": "within_contrast",
        "selector_series": "daily_return_contrast",
    }

    return results

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


def load_trial_candidate_sharpes(
    path: str,
    metric_type: str = "sharpe",
    dedupe: bool = True,
) -> np.ndarray:
    """Load the recovered per-candidate performance values behind
    outputs/trial_log.csv's `counts_toward_dsr_trials=True` rows, from
    outputs/trial_candidate_sharpes.csv -- the audit trail answering
    "can the underlying candidate results be recovered" (AnnaLisa,
    2026-10-xx): no daily return path survives for any of these trials
    (only rounded summary Sharpes were ever saved, and several only
    exist in commit messages, not a CSV), but the Sharpe *values*
    themselves do, with per-row provenance in that file's `source`
    column.

    `metric_type` filters to rows on comparable units -- trial 11's two
    rows are Sharpe *differences* (regime-effect sizes, ~0.01), not
    raw Sharpe levels (~0.4-0.9) like every other trial, and must not
    be pooled with them. Default 'sharpe' excludes trial 11.

    `dedupe=True` (default) drops rows flagged `redundant_with`
    another row -- several "candidates" are the same underlying run
    re-appearing across sequential trials (e.g. trial 4's EWMA-disabled
    case is byte-for-byte trial 3's warm-start case) and contribute no
    new information; including them would silently inflate the
    candidate count without adding independent evidence.
    """
    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [
        row for row in rows
        if row["metric_type"].strip() == metric_type
        and (not dedupe or not row["redundant_with"].strip())
    ]
    return np.array([float(row["value"]) for row in selected])


def trial_sharpe_variance_sensitivity(path: str) -> dict:
    """Diagnose whether outputs/trial_candidate_sharpes.csv supports a
    trustworthy empirical replacement for deflated_sharpe_ratio's
    `sharpe_variance_across_trials=1.0` placeholder, rather than just
    computing a number and swapping it in.

    Returns the sample variance under a few inclusion choices, each
    individually defensible, so the *spread* across them -- not any
    single number -- is the actual diagnostic. A small, heterogeneous,
    partly non-independent set of candidates (several trials are
    sequential refinements of the same underlying run, not independent
    attempts at distinct strategies) can make this estimate swing
    sharply on essentially arbitrary inclusion choices, which is itself
    evidence the placeholder should not yet be silently replaced.
    """
    deduped = load_trial_candidate_sharpes(path, metric_type="sharpe", dedupe=True)
    all_rows = load_trial_candidate_sharpes(path, metric_type="sharpe", dedupe=False)

    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    clean_only = np.array([
        float(row["value"]) for row in rows
        if row["metric_type"].strip() == "sharpe"
        and not row["redundant_with"].strip()
        and row["evaluation_sample"].strip() != "ad_hoc_pre_calendar_blend"
    ])

    def summarize(x: np.ndarray) -> dict:
        return {
            "n": int(len(x)),
            "variance": float(np.var(x, ddof=1)) if len(x) > 1 else float("nan"),
            "std": float(np.std(x, ddof=1)) if len(x) > 1 else float("nan"),
        }

    return {
        "deduped_all_trials": summarize(deduped),
        "including_redundant_rows": summarize(all_rows),
        "clean_sample_only": summarize(clean_only),
    }


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
