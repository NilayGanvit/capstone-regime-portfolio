"""
Data and timing module.

Owner: shared (whoever sets up the pipeline first)

Responsibilities (from the M2 problem statement):
- Load daily prices/distributions for the ten-ETF universe:
  SPY, EFA, EEM, IEF, TLT, LQD, HYG, GLD, DBC, VNQ
- Track when each observation actually became available (F(t) timing),
  so downstream modules never see information from after the decision date.
- Build total-return series, execution prices, and missing-data flags.
- Support the two robustness universes: no-VNQ, and no-HYG (sample from 2006).
"""

from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

UNIVERSE = ["SPY", "EFA", "EEM", "IEF", "TLT", "LQD", "HYG", "GLD", "DBC", "VNQ"]


@dataclass
class UniverseData:
    """Container for the loaded, timing-annotated dataset."""
    prices: pd.DataFrame          # adjusted close, indexed by date, columns = tickers
    total_returns: pd.DataFrame   # daily total returns
    available_as_of: pd.DataFrame # per-observation "known by" timestamp, for F(t) checks


def load_universe(tickers: list[str] = UNIVERSE, start: str = "2007-04-01") -> UniverseData:
    """
    TODO: load raw prices/distributions for `tickers` from the chosen source,
    build total-return series, and record data-availability timestamps.
    """
    raise NotImplementedError


def build_robustness_universe(data: UniverseData, drop: str) -> UniverseData:
    """
    TODO: derive the reduced universes used for robustness checks.

    drop: "VNQ" -> primary universe minus VNQ (same start date)
          "HYG" -> primary universe minus HYG (sample starts ~Feb 2006)
    """
    raise NotImplementedError
