import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from data import make_synthetic_universe, UNIVERSE, PriceDataset, M2Calendar, initial_window_length, stage_labels


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


def test_initial_window_length_counts_only_dates_through_initial_training_end():
    dates = pd.bdate_range("2014-12-01", "2015-01-31")
    calendar = M2Calendar(initial_training_end=pd.Timestamp("2014-12-31"))
    n = initial_window_length(dates, calendar)
    assert dates[n - 1] <= calendar.initial_training_end
    assert dates[n] > calendar.initial_training_end


def test_stage_labels_partitions_validation_and_final_test_and_beyond():
    calendar = M2Calendar(
        initial_training_end=pd.Timestamp("2014-12-31"),
        validation_end=pd.Timestamp("2018-12-31"),
        final_test_end=pd.Timestamp("2026-08-31"),
    )
    dates = pd.to_datetime(
        ["2010-06-15", "2016-03-01", "2020-07-01", "2026-09-05"]
    )
    labels = stage_labels(dates, calendar)
    assert list(labels) == [
        "initial_training",
        "validation",
        "final_test",
        "post_final_test",
    ]


def test_stage_labels_boundary_dates_are_inclusive_to_the_earlier_stage():
    calendar = M2Calendar(
        initial_training_end=pd.Timestamp("2014-12-31"),
        validation_end=pd.Timestamp("2018-12-31"),
        final_test_end=pd.Timestamp("2026-08-31"),
    )
    labels = stage_labels(pd.to_datetime(["2014-12-31", "2018-12-31", "2026-08-31"]), calendar)
    assert list(labels) == ["initial_training", "validation", "final_test"]
