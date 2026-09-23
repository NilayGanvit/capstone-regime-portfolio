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
    """Adjusted close (and, optionally, adjusted open) prices and daily log
    returns for the M2 universe.

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed adjusted close, one column per ticker in UNIVERSE order.
    opens : pd.DataFrame | None
        Date-indexed adjusted open, same index and columns as `prices`.
        Adjusted the same way as `prices` (dividend/split factor applied to
        the raw open) so that close_to_open_returns/open_to_close_returns
        are on a consistent total-return basis -- per M2's "executed at the
        opening of the following trading session, using consistently
        adjusted prices." None when only close prices are available (the
        harness then falls back to same-close execution timing).
    """

    prices: pd.DataFrame
    opens: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        missing = [t for t in UNIVERSE if t not in self.prices.columns]
        if missing:
            raise ValueError(f"Price data missing tickers: {missing}")
        self.prices = self.prices[UNIVERSE].sort_index()
        if self.prices.index.has_duplicates:
            raise ValueError("Price index has duplicate dates.")
        if self.opens is not None:
            missing = [t for t in UNIVERSE if t not in self.opens.columns]
            if missing:
                raise ValueError(f"Open price data missing tickers: {missing}")
            self.opens = self.opens[UNIVERSE].sort_index()
            if not self.opens.index.equals(self.prices.index):
                raise ValueError("opens must have exactly the same date index as prices.")

    @property
    def returns(self) -> pd.DataFrame:
        """Daily log returns. Row t uses prices at t and t-1, so it only
        reflects information available by the close of day t -- consistent
        with F(t) in the M2 pseudocode."""
        return np.log(self.prices / self.prices.shift(1)).dropna(how="all")

    def _require_opens(self) -> pd.DataFrame:
        if self.opens is None:
            raise ValueError(
                "This PriceDataset has no open-price data (opens=None); "
                "load it via from_csv_dir with AdjOpen columns present."
            )
        return self.opens

    @property
    def close_to_open_returns(self) -> pd.DataFrame:
        """Log return from the previous close to today's open, i.e. the
        overnight gap: log(Open(t) / Close(t-1))."""
        opens = self._require_opens()
        return np.log(opens / self.prices.shift(1)).dropna(how="all")

    @property
    def open_to_close_returns(self) -> pd.DataFrame:
        """Log return from today's open to today's close: log(Close(t) / Open(t))."""
        opens = self._require_opens()
        return np.log(self.prices / opens)

    def as_of(self, date: pd.Timestamp) -> "PriceDataset":
        """Return a copy truncated to information available through `date`.

        Every module that fits a model or computes a feature should go
        through this method (or an equivalent explicit cutoff) rather than
        slicing the full-sample DataFrame directly. This is the one place
        a look-ahead bug is easiest to introduce silently.
        """
        opens_slice = self.opens.loc[:date] if self.opens is not None else None
        return PriceDataset(self.prices.loc[:date], opens_slice)

    @classmethod
    def from_csv_dir(cls, directory: str) -> "PriceDataset":
        """Load one CSV per ticker from `directory`, each with columns
        [Date, AdjClose] and, optionally, [AdjOpen] (dividend/split-adjusted
        the same way as AdjClose -- see the PriceDataset.opens docstring).
        If any ticker's CSV lacks an AdjOpen column, `opens` is left None
        for the whole dataset rather than partially populated.
        """
        import pathlib

        close_series = {}
        open_series = {}
        have_all_opens = True
        for ticker in UNIVERSE:
            path = pathlib.Path(directory) / f"{ticker}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Expected {path}. Each ticker needs its own CSV with "
                    "columns [Date, AdjClose]."
                )
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            close_series[ticker] = df["AdjClose"]
            if "AdjOpen" in df.columns:
                open_series[ticker] = df["AdjOpen"]
            else:
                have_all_opens = False
        prices = pd.DataFrame(close_series)
        opens = pd.DataFrame(open_series) if have_all_opens else None
        return cls(prices, opens)


@dataclass(frozen=True)
class M2Calendar:
    """Fixed calendar boundaries from M2's "Walk-forward testing and data
    timing" section (the calendar table for the main ten-ETF specification):

        Initial training     first common trading date -- 2014-12-31
        Chronological validation  2015-01-01 -- 2018-12-31 (48 monthly periods)
        Final walk-forward test   2019-01-01 -- 2026-08-31 (92 monthly periods)

    Boundaries apply to the trading observations within each period; feature
    warm-up is confined to the initial training period. Any date after
    final_test_end (e.g. from a data pull that runs later than the M2
    submission) falls outside this calendar entirely and must not be
    scored as part of the frozen final test.
    """

    initial_training_end: pd.Timestamp = pd.Timestamp("2014-12-31")
    validation_end: pd.Timestamp = pd.Timestamp("2018-12-31")
    final_test_end: pd.Timestamp = pd.Timestamp("2026-08-31")


def initial_window_length(dates: pd.DatetimeIndex, calendar: M2Calendar = M2Calendar()) -> int:
    """Number of leading trading days in `dates` on or before
    `calendar.initial_training_end`. Pass this as `initial_window` to
    run_walk_forward so the initial-fit window matches M2's calendar (the
    first common trading date through 31 Dec 2014) rather than an
    arbitrary day count.
    """
    return int((dates <= calendar.initial_training_end).sum())


def stage_labels(dates: pd.DatetimeIndex, calendar: M2Calendar = M2Calendar()) -> pd.Series:
    """Label each date as 'initial_training', 'validation', 'final_test', or
    'post_final_test'. Use this to split a walk-forward run's dates so that
    validation-period results (used for any tuning/selection) and
    final-test results (the frozen, one-shot out-of-sample number) are
    never reported or compared as if they were the same thing -- and so
    dates beyond M2's calendar are never silently folded into the "final
    test" figure.
    """
    def _label(d: pd.Timestamp) -> str:
        if d <= calendar.initial_training_end:
            return "initial_training"
        if d <= calendar.validation_end:
            return "validation"
        if d <= calendar.final_test_end:
            return "final_test"
        return "post_final_test"

    return pd.Series([_label(d) for d in dates], index=dates, name="stage")


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
