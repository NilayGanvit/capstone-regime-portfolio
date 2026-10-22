"""
The risk-budget solver is selectable (allocation_erc.risk_budget_weights, the ERC
allocators, budgets_to_weights_batch, run_walk_forward(risk_budget_solver=...)),
and the default is the exact convex solution.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_erc import (
    RISK_BUDGET_SOLVERS, erc_baseline_weights, erc_regime_weights, regularize_covariance, risk_budget_weights,
    risk_contribution_shares, solve_risk_budget, solve_risk_budget_convex,
)
from allocation_lstm import budgets_to_weights_batch
from constraints import ConstraintSpec
from data import make_synthetic_universe
from walkforward import run_walk_forward

CASES = np.load(Path(__file__).parent / "fixtures" / "erc_solver_cases.npz")
SPEC = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30)


def _cov(seed=0, n=10):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return (a @ a.T / n + 0.05 * np.eye(n)) * 1e-4


def test_default_solver_is_exact_and_slsqp_is_selectable():
    cov = _cov()
    assert RISK_BUDGET_SOLVERS == ("exact", "slsqp")
    assert np.array_equal(risk_budget_weights(cov), solve_risk_budget_convex(cov))
    assert np.array_equal(risk_budget_weights(cov, solver="slsqp"), solve_risk_budget(cov))


def test_unknown_solver_name_is_rejected():
    with pytest.raises(ValueError, match="solver must be one of"):
        risk_budget_weights(_cov(), solver="newton")


def test_erc_allocators_default_to_exact_and_forward_the_solver():
    cov = _cov(1)
    assert np.array_equal(erc_baseline_weights(cov, 0.10), solve_risk_budget_convex(regularize_covariance(cov, 0.10)))
    state_covs, probs = np.stack([_cov(2), _cov(3)]), np.array([0.3, 0.7])
    mixture = regularize_covariance(np.tensordot(probs, state_covs, axes=(0, 0)), 0.10)
    assert np.array_equal(erc_regime_weights(state_covs, probs, 0.10), solve_risk_budget_convex(mixture))
    assert np.array_equal(erc_regime_weights(state_covs, probs, 0.10, solver="slsqp"), solve_risk_budget(mixture))


def test_budgets_to_weights_batch_defaults_to_exact_and_keeps_every_asset_invested_where_slsqp_does_not():
    cov = CASES["cov_20200331"]  # a real covariance on which SLSQP pins two assets at zero
    budgets = np.full((1, 10), 0.1)
    exact = budgets_to_weights_batch(budgets, cov)[0]
    legacy = budgets_to_weights_batch(budgets, cov, solver="slsqp")[0]
    assert np.array_equal(exact, solve_risk_budget_convex(cov, budgets[0]))
    assert (exact > 0.005).all() and np.abs(risk_contribution_shares(exact, cov) - 0.1).max() < 1e-8
    assert (legacy < 1e-4).any()


def test_walk_forward_rejects_an_unknown_solver_before_doing_any_work():
    ds, _ = make_synthetic_universe(n_days=400, seed=2)
    with pytest.raises(ValueError, match="risk_budget_solver must be one of"):
        run_walk_forward(ds.returns, n_states=2, initial_window=252, constraint_spec=SPEC, risk_budget_solver="newton")


def test_walk_forward_default_gives_exact_risk_parity_before_the_constraints():
    ds, _ = make_synthetic_universe(n_days=420, seed=3)
    result = run_walk_forward(ds.returns, n_states=2, initial_window=252, constraint_spec=SPEC)
    for config in ["erc_baseline", "erc_regime"]:
        devs = [e["risk_contribution_pre_constraint_max_dev"] for e in result.binding_constraints_history[config]]
        assert devs and max(devs) < 1e-6, f"{config}: raw ERC target is not at risk parity"


def test_walk_forward_slsqp_option_runs_and_reaches_the_same_weights_where_slsqp_converges():
    ds, _ = make_synthetic_universe(n_days=420, seed=3)
    kwargs = dict(n_states=2, initial_window=252, constraint_spec=SPEC)
    exact = run_walk_forward(ds.returns, **kwargs)
    legacy = run_walk_forward(ds.returns, risk_budget_solver="slsqp", **kwargs)
    assert set(exact.weights) == set(legacy.weights)
    # equal weight never touches the solver
    assert np.array_equal(exact.weights["equal_weight"], legacy.weights["equal_weight"])
