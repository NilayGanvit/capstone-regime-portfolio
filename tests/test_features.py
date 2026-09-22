import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import make_synthetic_universe
from features import build_feature_matrix, rolling_volatility, rolling_avg_correlation


def test_feature_matrix_has_no_lookahead_nans():
    ds, _ = make_synthetic_universe(n_days=400, seed=0)
    feats = build_feature_matrix(ds.returns)
    assert not feats.isna().any().any()
    # first valid row should be at/after the longest lookback window used
    assert len(feats) < len(ds.returns)


def test_rolling_volatility_is_nonnegative():
    ds, _ = make_synthetic_universe(n_days=400, seed=0)
    vol = rolling_volatility(ds.returns).dropna()
    assert (vol >= 0).all().all()


def test_rolling_avg_correlation_in_valid_range():
    ds, _ = make_synthetic_universe(n_days=400, seed=0)
    corr = rolling_avg_correlation(ds.returns).dropna()
    assert (corr >= -1.0001).all() and (corr <= 1.0001).all()
