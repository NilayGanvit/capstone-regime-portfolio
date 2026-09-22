import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import (
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
    deflated_sharpe_ratio,
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
