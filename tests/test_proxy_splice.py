import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import UNIVERSE
from proxy_data import PROXY_FILES, spliced_returns


def _write_proxies(directory: Path, dates: pd.DatetimeIndex, rng, drop=None):
    for t in UNIVERSE:
        d = dates if t != drop else dates.drop(drop_day := dates[-1])
        px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(d)))), index=d, name="value")
        px.index.name = "date"
        px.to_csv(directory / f"{PROXY_FILES[t]}.csv")
    yields = pd.Series(5.0, index=pd.bdate_range("1999-12-01", dates[-1]), name="DGS3MO")
    yields.index.name = "observation_date"
    yields.to_csv(directory / "DGS3MO.csv")


def _etf(dates, rng):
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(dates), len(UNIVERSE))), axis=0)), index=dates, columns=UNIVERSE)
    return prices, np.log(prices / prices.shift(1)).iloc[1:]


def test_splice_takes_each_day_from_one_source_and_switches_on_the_etfs_first_price_date(tmp_path):
    rng = np.random.default_rng(0)
    proxy_dates = pd.bdate_range("2000-01-03", "2000-03-31")
    _write_proxies(tmp_path, proxy_dates, rng)
    etf_prices, etf_returns = _etf(pd.bdate_range("2000-03-01", "2000-06-30"), rng)

    out = spliced_returns(tmp_path, etf_prices, etf_returns, start="2000-01-01")

    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.columns) == list(etf_returns.columns)
    assert out.loc[etf_returns.index].equals(etf_returns)
    proxy_part = out.loc[:etf_prices.index[0]]
    assert proxy_part.index[-1] == etf_prices.index[0] and proxy_part.index[0] == proxy_dates[1]
    spy = pd.read_csv(tmp_path / f"{PROXY_FILES['SPY']}.csv", index_col=0, parse_dates=True)["value"].loc[:etf_prices.index[0]]
    assert np.allclose(proxy_part["SPY"].to_numpy(), np.log(spy / spy.shift(1)).iloc[1:].to_numpy(), atol=1e-15)


def test_splice_refuses_when_the_proxies_miss_the_switch_day(tmp_path):
    rng = np.random.default_rng(1)
    proxy_dates = pd.bdate_range("2000-01-03", "2000-03-01")
    _write_proxies(tmp_path, proxy_dates, rng, drop="GLD")  # gold has no value on 2000-03-01
    etf_prices, etf_returns = _etf(pd.bdate_range("2000-03-01", "2000-04-28"), rng)
    with pytest.raises(ValueError, match="first price date"):
        spliced_returns(tmp_path, etf_prices, etf_returns, start="2000-01-01")
