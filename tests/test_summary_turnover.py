import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_real_data  # noqa: E402


def test_summary_turnover_is_the_decision_turnover_not_price_drift():
    """Holdings drift every day but only two decisions are made; the
    summary's turnover is the mean of the decisions dated inside the masked
    period (as the binding-constraint counts are), not of daily drift."""
    dates = list(pd.bdate_range("2020-01-01", periods=6))
    drifting = np.array([[0.5, 0.5], [0.6, 0.4], [0.55, 0.45], [0.5, 0.5], [0.45, 0.55], [0.4, 0.6]])
    result = SimpleNamespace(
        dates=dates,
        portfolio_returns={"x": np.random.default_rng(0).normal(0, 0.01, 6)},
        weights={"x": drifting},
        # decided on day 0 (outside the mask, executes on day 1) and on day 4
        binding_constraints_history={"x": [{"date": dates[0], "turnover": 0.2}, {"date": dates[4], "turnover": 0.1}]},
        execution_history={"x": [{"date": dates[1], "turnover": 0.2}, {"date": dates[5], "turnover": 0.1}]},
    )
    mask = np.array([False, True, True, True, True, True])
    summary = run_real_data.summarize(result, None, 0, mask, np.zeros(6))
    assert summary.loc["x", "mean_turnover_per_rebalance"] == 0.1
    assert "avg_turnover" not in summary.columns
