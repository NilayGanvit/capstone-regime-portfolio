"""
Long-history proxy sample for trial 16 (docs/experiments/trend_blend_proxy_preregistration.md):
ten fund and index proxies for the ten ETFs, on their common trading days, with the daily
risk-free return built from FRED DGS3MO the same way as for the ETFs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data import UNIVERSE, build_risk_free_daily

PROXY_FILES = {
    "SPY": "SPY_VFINX", "EFA": "EFA_VGTSX", "EEM": "EEM_VEIEX", "IEF": "IEF_VFITX", "TLT": "TLT_VUSTX",
    "LQD": "LQD_VWESX", "HYG": "HYG_VWEHX", "GLD": "GLD_GCF", "DBC": "DBC_SPGSCI", "VNQ": "VNQ_VGSIX",
}


def load_proxy_sample(directory: str | Path, start: str, end: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Daily log returns (columns in UNIVERSE order) between consecutive days on which all ten
    proxies have a value, from the first common day on or after `start` through `end`, and the
    daily simple risk-free return aligned to them. The first return is from the first common day
    to the second."""
    directory = Path(directory)
    prices = pd.concat(
        [pd.read_csv(directory / f"{PROXY_FILES[t]}.csv", index_col=0, parse_dates=True)["value"].rename(t) for t in UNIVERSE],
        axis=1, join="inner",
    ).sort_index()
    prices = prices.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("proxy prices must be positive and complete on the common calendar")
    returns = np.log(prices / prices.shift(1)).iloc[1:]
    yields = pd.to_numeric(pd.read_csv(directory / "DGS3MO.csv", index_col=0, parse_dates=True).iloc[:, 0], errors="coerce")
    rf = build_risk_free_daily(yields, returns.index)["rf_daily"].to_numpy()
    return returns, rf


def spliced_returns(directory: str | Path, etf_prices: pd.DataFrame, etf_returns: pd.DataFrame,
                    start: str = "2000-08-30") -> pd.DataFrame:
    """The ETF log returns with the proxies' in front of them, for the crisis-fold check
    (docs/verification/crisis_folds_preregistration.md): proxy returns from the first common
    day on or after `start` up to the ETFs' first price date, then the ETF returns. The last
    proxy return ends on the day the first ETF return starts from, so every day's return comes
    from one source and none spans both."""
    switch = etf_prices.index[0]
    proxy, _ = load_proxy_sample(directory, start, switch)
    if proxy.index[-1] != switch:
        raise ValueError(f"the proxies have no common value on the ETFs' first price date {switch.date()}")
    return pd.concat([proxy[list(etf_returns.columns)], etf_returns])
