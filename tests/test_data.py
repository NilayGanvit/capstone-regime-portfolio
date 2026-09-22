import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe, UNIVERSE, PriceDataset


def test_synthetic_universe_shape_and_columns():
    ds, true_state = make_synthetic_universe(n_days=300, seed=0)
    assert ds.prices.shape == (300, 10)
    assert list(ds.prices.columns) == UNIVERSE
    assert len(true_state) == 300


def test_returns_drop_first_row_only():
    ds, _ = make_synthetic_universe(n_days=300, seed=0)
    assert len(ds.returns) == len(ds.prices) - 1
    assert not ds.returns.isna().any().any()


def test_as_of_enforces_point_in_time_cutoff():
    ds, _ = make_synthetic_universe(n_days=300, seed=0)
    cutoff = ds.prices.index[100]
    truncated = ds.as_of(cutoff)
    assert truncated.prices.index.max() == cutoff
    assert len(truncated.prices) == 101


def test_rejects_missing_tickers():
    import pandas as pd
    import pytest

    bad_prices = pd.DataFrame(
        {"SPY": [1, 2, 3]}, index=pd.bdate_range("2020-01-01", periods=3)
    )
    with pytest.raises(ValueError):
        PriceDataset(bad_prices)
