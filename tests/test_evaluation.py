import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation import (
    sharpe_ratio, cagr, max_drawdown, sortino_ratio, calmar_ratio,
    herfindahl_concentration, average_turnover, block_bootstrap_diff_ci,
    deflated_sharpe_ratio, net_of_cost_returns, n_trials_from_log,
    optimal_stationary_block_length,
    stationary_bootstrap_indices,
    stationary_bootstrap_rq_contrasts,
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



def test_optimal_stationary_block_length_matches_reference_values():
    """Regression check against independently verified PPW reference values."""
    rng = np.random.default_rng(123)
    n = 2500

    white = rng.normal(size=n)

    ar05 = np.zeros(n)
    eps05 = rng.normal(size=n)
    for t in range(1, n):
        ar05[t] = 0.5 * ar05[t - 1] + eps05[t]

    ar09 = np.zeros(n)
    eps09 = rng.normal(size=n)
    for t in range(1, n):
        ar09[t] = 0.9 * ar09[t - 1] + eps09[t]

    assert np.isclose(
        optimal_stationary_block_length(white),
        1.5316189209351796,
        rtol=1e-10,
    )
    assert np.isclose(
        optimal_stationary_block_length(ar05),
        14.168003687099148,
        rtol=1e-10,
    )
    assert np.isclose(
        optimal_stationary_block_length(ar09),
        47.09238475262077,
        rtol=1e-10,
    )


def test_optimal_stationary_block_length_increases_with_serial_dependence():
    rng = np.random.default_rng(123)
    n = 2500

    white = rng.normal(size=n)

    ar05 = np.zeros(n)
    eps05 = rng.normal(size=n)
    for t in range(1, n):
        ar05[t] = 0.5 * ar05[t - 1] + eps05[t]

    ar09 = np.zeros(n)
    eps09 = rng.normal(size=n)
    for t in range(1, n):
        ar09[t] = 0.9 * ar09[t - 1] + eps09[t]

    b_white = optimal_stationary_block_length(white)
    b_ar05 = optimal_stationary_block_length(ar05)
    b_ar09 = optimal_stationary_block_length(ar09)

    assert b_white < b_ar05 < b_ar09


def test_stationary_bootstrap_indices_are_valid_and_reproducible():
    n = 500
    block_length = 12.5

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    idx1 = stationary_bootstrap_indices(n, block_length, rng1)
    idx2 = stationary_bootstrap_indices(n, block_length, rng2)

    assert len(idx1) == n
    assert np.issubdtype(idx1.dtype, np.integer)
    assert np.all((idx1 >= 0) & (idx1 < n))
    assert np.array_equal(idx1, idx2)


def test_stationary_bootstrap_indices_continue_or_restart():
    n = 200
    rng = np.random.default_rng(7)

    idx = stationary_bootstrap_indices(
        n=n,
        expected_block_length=10.0,
        rng=rng,
    )

    # Every transition is either the next chronological observation
    # (including wraparound) or a stationary-bootstrap restart.
    continuation = (idx[1:] == (idx[:-1] + 1) % n)

    assert continuation.any()
    assert (~continuation).any()

def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    rA = rng.normal(0.001, 0.01, 500)
    rB = rng.normal(0.0, 0.01, 500)
    res = block_bootstrap_diff_ci(rA, rB, n_boot=300, random_state=1)
    assert res["ci_low"] <= res["point_estimate"] <= res["ci_high"]



def test_stationary_bootstrap_ci_uses_automatic_block_length_and_95pct_default():
    rng = np.random.default_rng(321)
    n = 800

    common = np.zeros(n)
    shocks = rng.normal(0.0, 0.01, n)
    for t in range(1, n):
        common[t] = 0.6 * common[t - 1] + shocks[t]

    rA = common + rng.normal(0.0005, 0.002, n)
    rB = common + rng.normal(0.0000, 0.002, n)

    expected = optimal_stationary_block_length(rA - rB)

    res = block_bootstrap_diff_ci(
        rA,
        rB,
        n_boot=300,
        random_state=11,
    )

    assert res["ci_level"] == 0.95
    assert res["bootstrap_method"] == "stationary"
    assert res["block_length_source"] == "automatic"
    assert np.isclose(res["expected_block_length"], expected)


def test_stationary_bootstrap_ci_is_reproducible():
    rng = np.random.default_rng(55)
    rA = rng.normal(0.001, 0.01, 600)
    rB = rng.normal(0.0002, 0.01, 600)

    res1 = block_bootstrap_diff_ci(
        rA, rB, n_boot=250, random_state=99
    )
    res2 = block_bootstrap_diff_ci(
        rA, rB, n_boot=250, random_state=99
    )

    assert res1 == res2


def test_stationary_bootstrap_ci_allows_expected_block_length_override():
    rng = np.random.default_rng(77)
    rA = rng.normal(0.001, 0.01, 500)
    rB = rng.normal(0.0, 0.01, 500)

    res = block_bootstrap_diff_ci(
        rA,
        rB,
        block_size=12.5,
        n_boot=200,
        random_state=3,
    )

    assert res["bootstrap_method"] == "stationary"
    assert res["block_length_source"] == "user"
    assert res["expected_block_length"] == 12.5


def test_rq_contrasts_match_direct_sharpe_calculations():
    rng = np.random.default_rng(2026)
    n = 700

    common = rng.normal(0.0002, 0.008, n)

    portfolio_returns = {
        "erc_baseline": common + rng.normal(0.0000, 0.003, n),
        "erc_regime": common + rng.normal(0.0002, 0.003, n),
        "erc_blend": common + rng.normal(0.0003, 0.003, n),
        "lstm_baseline": common + rng.normal(0.0001, 0.003, n),
        "lstm_regime": common + rng.normal(0.0005, 0.003, n),
        "lstm_blend": common + rng.normal(0.0004, 0.003, n),
    }

    result = stationary_bootstrap_rq_contrasts(
        portfolio_returns,
        block_size=8.0,
        n_boot=100,
        random_state=17,
    )

    s = {
        name: sharpe_ratio(r)
        for name, r in portfolio_returns.items()
    }

    rq1_erc = s["erc_regime"] - s["erc_baseline"]
    rq1_lstm = s["lstm_regime"] - s["lstm_baseline"]

    assert np.isclose(
        result["rq1_erc_regime_effect"]["point_estimate"],
        rq1_erc,
    )
    assert np.isclose(
        result["rq1_lstm_regime_effect"]["point_estimate"],
        rq1_lstm,
    )
    assert np.isclose(
        result["rq2_difference_in_differences"]["point_estimate"],
        rq1_lstm - rq1_erc,
    )
    assert np.isclose(
        result["rq3_erc_reliability_effect"]["point_estimate"],
        s["erc_blend"] - s["erc_regime"],
    )
    assert np.isclose(
        result["rq3_lstm_reliability_effect"]["point_estimate"],
        s["lstm_blend"] - s["lstm_regime"],
    )

    meta = result["_bootstrap"]

    assert meta["method"] == "stationary"
    assert meta["block_length_source"] == "user_common_override"
    assert meta["synchronization"] == "within_contrast"
    assert meta["selector_series"] == "daily_return_contrast"
    assert set(meta["contrast_block_lengths"]) == {
        "rq1_erc_regime_effect",
        "rq1_lstm_regime_effect",
        "rq2_difference_in_differences",
        "rq3_erc_reliability_effect",
        "rq3_lstm_reliability_effect",
    }
    assert all(
        np.isclose(value, 8.0)
        for value in meta["contrast_block_lengths"].values()
    )


def test_rq_contrasts_automatic_block_lengths_are_contrast_specific():
    rng = np.random.default_rng(404)
    n = 1000

    base = rng.normal(0.0, 0.01, n)

    persistent = np.zeros(n)
    shocks = rng.normal(0.0, 0.002, n)
    for t in range(1, n):
        persistent[t] = 0.8 * persistent[t - 1] + shocks[t]

    portfolio_returns = {
        "erc_baseline": base,
        "erc_regime": base + 0.5 * persistent,
        "erc_blend": base + 0.8 * persistent,
        "lstm_baseline": base + rng.normal(0.0, 0.001, n),
        "lstm_regime": base + persistent,
        "lstm_blend": base + 1.2 * persistent,
    }

    selector_series = {
        "rq1_erc_regime_effect":
            portfolio_returns["erc_regime"]
            - portfolio_returns["erc_baseline"],
        "rq1_lstm_regime_effect":
            portfolio_returns["lstm_regime"]
            - portfolio_returns["lstm_baseline"],
        "rq2_difference_in_differences":
            (
                portfolio_returns["lstm_regime"]
                - portfolio_returns["lstm_baseline"]
            )
            - (
                portfolio_returns["erc_regime"]
                - portfolio_returns["erc_baseline"]
            ),
        "rq3_erc_reliability_effect":
            portfolio_returns["erc_blend"]
            - portfolio_returns["erc_regime"],
        "rq3_lstm_reliability_effect":
            portfolio_returns["lstm_blend"]
            - portfolio_returns["lstm_regime"],
    }

    expected = {
        name: optimal_stationary_block_length(series)
        for name, series in selector_series.items()
    }

    result = stationary_bootstrap_rq_contrasts(
        portfolio_returns,
        n_boot=100,
        random_state=8,
    )

    meta = result["_bootstrap"]
    actual = meta["contrast_block_lengths"]

    assert meta["block_length_source"] == "automatic_contrast_specific"
    assert meta["synchronization"] == "within_contrast"
    assert meta["selector_series"] == "daily_return_contrast"
    assert set(actual) == set(expected)

    for name in expected:
        assert np.isclose(actual[name], expected[name])


def test_rq_contrasts_preserve_within_contrast_synchronization():
    rng = np.random.default_rng(909)
    n = 500

    shared = rng.normal(0.0003, 0.01, n)
    other = rng.normal(0.0001, 0.012, n)

    portfolio_returns = {
        "erc_baseline": shared.copy(),
        "erc_regime": shared.copy(),
        "erc_blend": shared.copy(),
        "lstm_baseline": other.copy(),
        "lstm_regime": other.copy(),
        "lstm_blend": other.copy(),
    }

    result = stationary_bootstrap_rq_contrasts(
        portfolio_returns,
        block_size=10.0,
        n_boot=200,
        ci=0.95,
        random_state=23,
    )

    for name in (
        "rq1_erc_regime_effect",
        "rq1_lstm_regime_effect",
        "rq2_difference_in_differences",
        "rq3_erc_reliability_effect",
        "rq3_lstm_reliability_effect",
    ):
        assert np.isclose(result[name]["point_estimate"], 0.0)
        assert np.isclose(result[name]["ci_low"], 0.0)
        assert np.isclose(result[name]["ci_high"], 0.0)

    assert result["_bootstrap"]["synchronization"] == "within_contrast"


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
    expected[1] = (1.0 + gross[1]) * (1.0 - 0.0005 * 0.4) - 1.0
    expected[3] = (1.0 + gross[3]) * (1.0 - 0.0005 * 0.2) - 1.0

    # Non-execution days remain unchanged.
    assert np.allclose(net[[0, 2, 4]], gross[[0, 2, 4]])
    assert np.allclose(net, expected)


def test_net_of_cost_returns_scales_linearly_with_fee_bps():
    # Under self-financing accounting,
    # net = (1 + gross) * (1 - fee_rate * turnover) - 1.
    # For fixed gross return and turnover, the cost effect remains linear
    # in fee_rate, so doubling fee_bps doubles the transaction-cost effect.
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


def test_net_of_cost_returns_self_finances_cost_before_return_accrual():
    """Transaction costs paid at execution reduce capital available to earn returns."""
    dates = [pd.Timestamp("2025-01-02")]

    # Hand-checkable example:
    # Opening NAV = $1,000
    # Gross turnover = 20%
    # Fee = 100 bps = 1% of traded value
    # Cost = $1,000 * 0.20 * 0.01 = $2
    # Capital after cost = $998
    # Execution-day gross portfolio return = +10%
    # Correct closing NAV = $998 * 1.10 = $1,097.80
    gross_returns = np.array([0.10])
    execution_history = [
        {"date": dates[0], "turnover": 0.20}
    ]

    net = net_of_cost_returns(
        dates=dates,
        gross_returns=gross_returns,
        execution_history=execution_history,
        fee_bps=100.0,
    )

    expected_return = (1.0 + 0.10) * (1.0 - 0.01 * 0.20) - 1.0

    assert np.isclose(net[0], expected_return)
