"""
Portfolio-accounting reconciliation: rebuild each configuration's holdings,
returns, turnover and NAV from first principles using only what the walk-forward
reports (decision dates, decision-day target weights) and the raw asset returns,
and require them to agree with the walk-forward's own series.

This is an independent re-derivation, not a re-run: the ledger below never calls
the walk-forward's accounting code, so a mistake in how returns accrue, how a
trade is dated, or how a fee is charged would surface as a mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from constraints import ConstraintSpec
from data import PriceDataset, make_synthetic_universe
from evaluation import net_of_cost_returns
from walkforward import run_walk_forward

SPEC = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30)
INITIAL_WINDOW = 252
FEE_BPS = 5.0


def _dataset(with_opens: bool) -> PriceDataset:
    ds, _ = make_synthetic_universe(n_days=420, seed=11)
    if not with_opens:
        return ds
    gap = np.random.default_rng(12).normal(0.0, 0.01, size=ds.prices.shape)
    opens = ds.prices.shift(1) * np.exp(gap)
    opens.iloc[0] = ds.prices.iloc[0]
    return PriceDataset(ds.prices, opens)


def _run(with_opens: bool):
    ds = _dataset(with_opens)
    kwargs = dict(n_states=2, initial_window=INITIAL_WINDOW, constraint_spec=SPEC)
    if with_opens:
        kwargs.update(close_to_open_returns=ds.close_to_open_returns, open_to_close_returns=ds.open_to_close_returns)
    return ds, run_walk_forward(ds.returns, **kwargs)


def _drift(w, simple):
    grown = w * (1.0 + simple)
    return grown / grown.sum()


def _ledger(ds, result, config, with_opens):
    """Independent day-by-day rebuild. Returns per-day gross return, the holdings
    at the end of each day, and {execution_day_index: decision turnover}."""
    n = ds.returns.shape[1]
    returns = ds.returns.loc[result.dates]
    simple = np.exp(returns.to_numpy()) - 1.0
    if with_opens:
        co = np.exp(ds.close_to_open_returns.loc[result.dates].to_numpy()) - 1.0
        oc = np.exp(ds.open_to_close_returns.loc[result.dates].to_numpy()) - 1.0
    decisions = {pd.Timestamp(e["date"]) for e in result.binding_constraints_history[config]}
    targets = {pd.Timestamp(d): result.weights[config][i] for i, d in enumerate(result.dates) if pd.Timestamp(d) in decisions}

    holdings = np.full(n, 1.0 / n)
    pending, pending_turnover = None, None
    gross, end_holdings, executions = [], [], {}
    for i, date in enumerate(pd.DatetimeIndex(result.dates)):
        if pending is not None:  # open-execution day: overnight leg on old holdings, intraday leg on the new target
            gross.append((1.0 + holdings @ co[i]) * (1.0 + pending @ oc[i]) - 1.0)
            holdings = _drift(pending, oc[i])
            executions[i] = pending_turnover
            pending = None
            end_holdings.append(holdings)
            continue
        gross.append(float(holdings @ simple[i]))
        drifted = _drift(holdings, simple[i])
        if date in decisions:
            target = targets[date]
            turnover = float(np.abs(target - drifted).sum())
            if with_opens:
                pending, pending_turnover, holdings = target, turnover, drifted
                end_holdings.append(target)  # the walk-forward reports the target on a decision day
            else:
                executions[i] = turnover
                holdings = target
                end_holdings.append(holdings)
        else:
            holdings = drifted
            end_holdings.append(holdings)
    return np.array(gross), np.array(end_holdings), executions


@pytest.mark.parametrize("with_opens", [False, True], ids=["same-close", "next-open"])
def test_gross_returns_holdings_and_turnover_reconcile_to_an_independent_ledger(with_opens):
    ds, result = _run(with_opens)
    for config in result.portfolio_returns:
        gross, end_holdings, executions = _ledger(ds, result, config, with_opens)
        assert np.allclose(gross, result.portfolio_returns[config], atol=1e-12), f"{config}: gross return mismatch"
        assert np.allclose(end_holdings, result.weights[config], atol=1e-10), f"{config}: daily holdings mismatch"
        recorded = {pd.Timestamp(e["date"]): e["turnover"] for e in result.execution_history[config]}
        assert len(recorded) == len(executions)
        for i, turnover in executions.items():
            assert recorded[pd.Timestamp(result.dates[i])] == pytest.approx(turnover, abs=1e-10), f"{config}: turnover mismatch"


def test_net_nav_reconciles_to_a_dollar_position_ledger_with_fees_on_traded_value():
    """Same-close mode: hold dollar positions, grow each by its own asset return,
    and on a rebalance trade to the target at the drifted NAV, paying fee_rate on
    the dollars bought plus sold. The resulting NAV path must equal the
    cumulative product of net_of_cost_returns, which charges the fee as a
    fraction of equity at execution."""
    ds, result = _run(with_opens=False)
    rate = FEE_BPS / 10_000.0
    simple = np.exp(ds.returns.loc[result.dates].to_numpy()) - 1.0
    for config in result.portfolio_returns:
        _, _, executions = _ledger(ds, result, config, with_opens=False)
        n = simple.shape[1]
        positions = np.full(n, 1.0 / n)  # dollars, starting NAV 1.0
        nav = []
        for i in range(len(simple)):
            positions = positions * (1.0 + simple[i])
            if i in executions:
                value = positions.sum()
                target_dollars = result.weights[config][i] * value
                fee = rate * np.abs(target_dollars - positions).sum()
                assert np.abs(target_dollars - positions).sum() / value == pytest.approx(executions[i], abs=1e-10)
                positions = target_dollars * (1.0 - fee / value)  # fee leaves the portfolio pro rata
            nav.append(positions.sum())
        net = net_of_cost_returns(result.dates, result.portfolio_returns[config], result.execution_history[config], FEE_BPS)
        assert np.allclose(nav, np.cumprod(1.0 + net), rtol=1e-10, atol=1e-12), f"{config}: NAV ledger does not reconcile"


@pytest.mark.parametrize("with_opens", [False, True], ids=["same-close", "next-open"])
def test_equal_weight_trades_are_exactly_the_drift_back_to_equal_weights(with_opens):
    """The simplest closed-form case: every equal-weight decision restores 1/n, so
    its recorded turnover is the L1 distance from the drifted holdings to 1/n."""
    ds, result = _run(with_opens)
    _, _, executions = _ledger(ds, result, "equal_weight", with_opens)
    recorded = {pd.Timestamp(e["date"]): e["turnover"] for e in result.execution_history["equal_weight"]}
    assert executions and all(recorded[pd.Timestamp(result.dates[i])] == pytest.approx(t, abs=1e-10) for i, t in executions.items())
    n = ds.returns.shape[1]
    for entry in result.binding_constraints_history["equal_weight"]:
        i = list(pd.DatetimeIndex(result.dates)).index(pd.Timestamp(entry["date"]))
        assert np.allclose(result.weights["equal_weight"][i], 1.0 / n)
