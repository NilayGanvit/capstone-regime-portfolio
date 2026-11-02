import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_real_data  # noqa: E402


def test_summary_turnover_is_the_charged_trade_turnover_not_price_drift():
    """Holdings drift every day but only two trades execute; the summary's
    turnover is the mean of the trades executed inside the masked period,
    the same figures transaction costs are charged on."""
    dates = list(pd.bdate_range("2020-01-01", periods=6))
    drifting = np.array([[0.5, 0.5], [0.6, 0.4], [0.55, 0.45], [0.5, 0.5], [0.45, 0.55], [0.4, 0.6]])
    result = SimpleNamespace(
        dates=dates,
        portfolio_returns={"x": np.random.default_rng(0).normal(0, 0.01, 6)},
        weights={"x": drifting},
        execution_history={"x": [{"date": dates[0], "turnover": 0.2}, {"date": dates[4], "turnover": 0.1}]},
    )
    mask = np.array([False, True, True, True, True, True])
    summary = run_real_data.summarize(result, None, 0, mask, np.zeros(6))
    assert summary.loc["x", "mean_turnover_per_rebalance"] == 0.1
    assert "avg_turnover" not in summary.columns
