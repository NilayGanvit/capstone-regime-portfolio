"""
Feature engineering for the regime-inference HMM.

Corresponds to M2 "Proposed model specification" (candidate daily feature
classes: returns, rolling-window volatility, trends, asset correlations,
and bond return dynamics) and the M3 point-in-time discussion.

Every feature here is computed using only information available through
day t (rolling windows look backward only, never forward), and the
walk-forward harness is responsible for freezing the feature *set* and
lookback lengths during development/validation, before the final OOS test
-- this module only computes candidates, it does not select among them.

Owner: Silvio (regime inputs), infrastructure shared with Nilay's data.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_VOL_WINDOW = 21   # ~1 trading month
DEFAULT_TREND_WINDOW = 63  # ~3 trading months
DEFAULT_CORR_WINDOW = 63


def rolling_volatility(returns: pd.DataFrame, window: int = DEFAULT_VOL_WINDOW) -> pd.DataFrame:
    """Annualized rolling volatility per asset, using only trailing data."""
    return returns.rolling(window, min_periods=window).std() * np.sqrt(252)


def rolling_trend(returns: pd.DataFrame, window: int = DEFAULT_TREND_WINDOW) -> pd.DataFrame:
    """Trailing cumulative return over `window` days, a simple momentum /
    trend proxy. Uses only trailing data."""
    return returns.rolling(window, min_periods=window).sum()


def rolling_avg_correlation(returns: pd.DataFrame, window: int = DEFAULT_CORR_WINDOW) -> pd.Series:
    """A single scalar summary of the cross-sectional correlation
    structure per day: the average pairwise correlation across all asset
    pairs over the trailing window. Kept scalar (rather than the full
    correlation matrix) so it can sit alongside the other candidate
    features in one flat feature vector for the HMM; the full matrix is
    still available to the ERC/risk-budgeting allocators separately.
    """
    n = returns.shape[1]
    out = pd.Series(index=returns.index, dtype=float)
    for i in range(window, len(returns) + 1):
        window_slice = returns.iloc[i - window : i]
        corr = window_slice.corr().to_numpy()
        off_diag = corr[~np.eye(n, dtype=bool)]
        out.iloc[i - 1] = np.nanmean(off_diag)
    return out


def bond_return_dynamics(returns: pd.DataFrame, short_col: str = "IEF", long_col: str = "TLT") -> pd.Series:
    """Trailing spread between long- and intermediate-duration Treasury
    returns, a simple proxy for yield-curve / duration regime dynamics
    referenced in M2's candidate feature list."""
    window = DEFAULT_TREND_WINDOW
    long_trend = returns[long_col].rolling(window, min_periods=window).sum()
    short_trend = returns[short_col].rolling(window, min_periods=window).sum()
    return long_trend - short_trend


def build_feature_matrix(
    returns: pd.DataFrame,
    vol_window: int = DEFAULT_VOL_WINDOW,
    trend_window: int = DEFAULT_TREND_WINDOW,
    corr_window: int = DEFAULT_CORR_WINDOW,
) -> pd.DataFrame:
    """Assemble the candidate daily feature matrix fed to the HMM.

    Columns: per-asset returns, per-asset rolling volatility, per-asset
    rolling trend, one scalar average correlation, one bond-dynamics
    scalar. This is the "reduced set of candidate daily feature classes"
    from M2 -- the actual subset and lookbacks used in the final HMM are
    a validation choice made downstream, frozen before the final
    out-of-sample test, not decided in this function.
    """
    vol = rolling_volatility(returns, vol_window).add_suffix("_vol")
    trend = rolling_trend(returns, trend_window).add_suffix("_trend")
    corr = rolling_avg_correlation(returns, corr_window).rename("avg_corr")
    bonds = bond_return_dynamics(returns).rename("bond_dynamics")

    features = pd.concat([returns.add_suffix("_ret"), vol, trend, corr, bonds], axis=1)
    return features.dropna(how="any")
