"""
Data loading and point-in-time access.

Corresponds to M2 "Investment universe and horizon" and the M3 walk-forward
timing discussion (Croushore & Stark's data-vintage argument, White's caution
on data snooping). This module owns exactly one responsibility: nothing
downstream should ever be able to see a value before it was actually
available on the historical decision date F(t).

Owner: Nilay (data & timing infrastructure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# The M2 universe. Kept as a module-level constant so every other module
# imports the same ordering rather than re-typing the ticker list.
UNIVERSE = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"]


@dataclass
class PriceDataset:
    """Adjusted close prices and daily log returns for the M2 universe.

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed, one column per ticker in UNIVERSE order.
    """

    prices: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [t for t in UNIVERSE if t not in self.prices.columns]
        if missing:
            raise ValueError(f"Price data missing tickers: {missing}")
        self.prices = self.prices[UNIVERSE].sort_index()
        if self.prices.index.has_duplicates:
            raise ValueError("Price index has duplicate dates.")

    @property
    def returns(self) -> pd.DataFrame:
        """Daily log returns. Row t uses prices at t and t-1, so it only
        reflects information available by the close of day t -- consistent
        with F(t) in the M2 pseudocode."""
        return np.log(self.prices / self.prices.shift(1)).dropna(how="all")

    def as_of(self, date: pd.Timestamp) -> "PriceDataset":
        """Return a copy truncated to information available through `date`.

        Every module that fits a model or computes a feature should go
        through this method (or an equivalent explicit cutoff) rather than
        slicing the full-sample DataFrame directly. This is the one place
        a look-ahead bug is easiest to introduce silently.
        """
        return PriceDataset(self.prices.loc[:date])

    @classmethod
    def from_csv_dir(cls, directory: str) -> "PriceDataset":
        """Load one CSV per ticker from `directory`, each with columns
        [Date, AdjClose]. This is the loader for real ETF data once the
        group has downloaded it; it is not exercised by the test suite
        because no network access is available in this environment.
        """
        import pathlib

        series = {}
        for ticker in UNIVERSE:
            path = pathlib.Path(directory) / f"{ticker}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected {path}. Each ticker needs its own CSV with "
                    "columns [Date, AdjClose]."
                )
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            series[ticker] = df["AdjClose"]
        prices = pd.DataFrame(series)
        return cls(prices)


def make_synthetic_universe(
    n_days: int = 1500,
    n_assets: int = 10,
    seed: int = 0,
) -> PriceDataset:
    """Generate a synthetic 2-regime price history for smoke-testing the
    pipeline end to end when real ETF data is not available.

    This is NOT a substitute for the real M2 universe -- it exists only so
    every module downstream (features, HMM, densities, allocators,
    constraints, evaluation, walk-forward harness) can be exercised and
    produce genuine numbers before the group has finalized the real data
    pull. Regimes here are literally simulated Markov states, so recovering
    them is an easy sanity check for the HMM fit, not evidence about real
    markets.
    """
    rng = np.random.default_rng(seed)

    # Two synthetic regimes: calm (low vol, mild positive drift, low
    # correlation) and stressed (high vol, negative drift, high
    # correlation) -- enough structure for the HMM to have something real
    # to recover.
    calm_mu = np.full(n_assets, 0.0003)
    stress_mu = np.full(n_assets, -0.0010)

    base_vol = np.linspace(0.008, 0.020, n_assets)  # heterogeneous vols
    calm_corr = 0.2 * np.ones((n_assets, n_assets)) + 0.8 * np.eye(n_assets)
    stress_corr = 0.6 * np.ones((n_assets, n_assets)) + 0.4 * np.eye(n_assets)

    def cov_from_corr(corr: np.ndarray, vol_scale: float) -> np.ndarray:
        vols = base_vol * vol_scale
        return np.outer(vols, vols) * corr

    calm_cov = cov_from_corr(calm_corr, 1.0)
    stress_cov = cov_from_corr(stress_corr, 2.2)

    # Simulate a 2-state Markov chain with realistic persistence.
    p_stay_calm, p_stay_stress = 0.985, 0.95
    state = 0  # start calm
    states = np.empty(n_days, dtype=int)
    for t in range(n_days):
        states[t] = state
        if state == 0:
            state = 0 if rng.random() < p_stay_calm else 1
        else:
            state = 1 if rng.random() < p_stay_stress else 0

    returns = np.empty((n_days, n_assets))
    for t in range(n_days):
        if states[t] == 0:
            returns[t] = rng.multivariate_normal(calm_mu, calm_cov)
        else:
            returns[t] = rng.multivariate_normal(stress_mu, stress_cov)

    dates = pd.bdate_range("2007-04-02", periods=n_days)
    tickers = UNIVERSE[:n_assets]
    ret_df = pd.DataFrame(returns, index=dates, columns=tickers)
    prices = 100 * np.exp(ret_df.cumsum())
    prices.iloc[0] = 100.0

    if n_assets < len(UNIVERSE):
        raise ValueError("Synthetic universe must cover all UNIVERSE tickers.")

    return PriceDataset(prices), pd.Series(states, index=dates, name="true_state")
