import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import (
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
    deflated_sharpe_ratio, net_of_cost_returns, n_trials_from_log,
)


def test_sharpe_ratio_zero_for_constant_returns():
    r = np.zeros(100)
    assert sharpe_ratio(r) == 0.0


def test_higher_drift_gives_higher_sharpe():
    rng = np.random.default_rng(0)
    rA = rng.normal(0.001, 0.01, 2000)
    rB = rng.normal(0.0000, 0.01, 2000)
    assert sharpe_ratio(rA) > sharpe_ratio(rB)


def test_max_drawdown_is_nonpositive():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0002, 0.01, 500)
    assert max_drawdown(r) <= 0


def test_herfindahl_bounds():
    n = 5
    equal_weight = np.tile(np.full(n, 1 / n), (10, 1))
    concentrated = np.tile(np.eye(n)[0], (10, 1))
    assert np.isclose(herfindahl_concentration(equal_weight), 1 / n)
    assert np.isclose(herfindahl_concentration(concentrated), 1.0)


def test_average_turnover_zero_when_no_trading():
    w = np.tile(np.full(5, 0.2), (10, 1))
    assert np.isclose(average_turnover(w, w), 0.0)


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    rA = rng.normal(0.001, 0.01, 500)
    rB = rng.normal(0.0, 0.01, 500)
    res = block_bootstrap_diff_ci(rA, rB, n_boot=300, random_state=1)
    assert res["ci_low"] <= res["point_estimate"] <= res["ci_high"]


def test_deflated_sharpe_decreases_with_more_trials():
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 1000)
    dsr_few = deflated_sharpe_ratio(observed_sharpe=1.2, returns=r, n_trials=1)
    dsr_many = deflated_sharpe_ratio(observed_sharpe=1.2, returns=r, n_trials=500)
    assert dsr_few >= dsr_many
    assert 0.0 <= dsr_many <= 1.0 and 0.0 <= dsr_few <= 1.0


def test_net_of_cost_returns_deducts_only_on_execution_days():
    dates = pd.bdate_range("2020-01-01", periods=5).tolist()
    gross = np.array([0.01, 0.02, -0.01, 0.005, 0.0])
    execution_history = [{"date": dates[1], "turnover": 0.4}, {"date": dates[3], "turnover": 0.2}]
    net = net_of_cost_returns(dates, gross, execution_history, fee_bps=5.0)
    expected = gross.copy()
    expected[1] -= 0.0005 * 0.4
    expected[3] -= 0.0005 * 0.2
    assert np.allclose(net, expected)
    assert net[0] == gross[0] and net[2] == gross[2] and net[4] == gross[4]


def test_net_of_cost_returns_scales_linearly_with_fee_bps():
    dates = pd.bdate_range("2020-01-01", periods=3).tolist()
    gross = np.array([0.01, 0.01, 0.01])
    execution_history = [{"date": dates[1], "turnover": 0.5}]
    net_5bps = net_of_cost_returns(dates, gross, execution_history, fee_bps=5.0)
    net_10bps = net_of_cost_returns(dates, gross, execution_history, fee_bps=10.0)
    cost_5 = gross[1] - net_5bps[1]
    cost_10 = gross[1] - net_10bps[1]
    assert np.isclose(cost_10, 2 * cost_5)


def test_net_of_cost_returns_ignores_execution_dates_outside_the_series():
    dates = pd.bdate_range("2020-01-01", periods=3).tolist()
    gross = np.array([0.01, 0.01, 0.01])
    stray_date = pd.Timestamp("2019-01-01")
    execution_history = [{"date": stray_date, "turnover": 0.9}]
    net = net_of_cost_returns(dates, gross, execution_history, fee_bps=25.0)
    assert np.allclose(net, gross)


def test_n_trials_from_log_counts_only_performance_driven_rows(tmp_path):
    path = tmp_path / "trial_log.csv"
    path.write_text(
        "trial_id,counts_toward_dsr_trials\n"
        "1,False\n"
        "2,True\n"
        "3,True\n"
        "4,False\n"
    )
    assert n_trials_from_log(str(path)) == 2


def test_n_trials_from_log_matches_the_real_project_log():
    repo_log = Path(__file__).resolve().parents[1] / "outputs" / "trial_log.csv"
    n = n_trials_from_log(str(repo_log))
    assert n >= 1
    assert isinstance(n, int)
