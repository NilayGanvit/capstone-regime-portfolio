"""
Agreement between the risk-budgeting solvers, and the residuals before and after
the constraint projection, on identical covariance and budget inputs.

There are three routes to risk-budgeting weights in this codebase:
  - solve_risk_budget: SLSQP on the squared error of risk shares. Non-convex, so
    it can stop at a local optimum. Used for backtest weights (ERC and LSTM).
  - solve_risk_budget_convex: the exact solution of the strictly convex
    log-barrier problem, in plain NumPy.
  - differentiable_risk_budget_layer: the same log-barrier problem solved by
    cvxpylayers, used while training the LSTM.

tests/fixtures/erc_solver_cases.npz holds real final-period covariances saved by
scripts/verification/audit_erc_solver.py: three erc_regime dates on which SLSQP
leaves assets at zero weight, and five turnover-binding erc_regime dates with
their drifted holdings; erc_solver_case_20150130.npz adds the one validation-period
date on which it also fails.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_erc import risk_contribution_shares, solve_risk_budget, solve_risk_budget_convex
from allocation_lstm import _TORCH_AVAILABLE, _CVXPYLAYERS_AVAILABLE
from constraints import ConstraintSpec, project_onto_constraints, turnover

SPEC = ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30)
CASES = dict(np.load(Path(__file__).parent / "fixtures" / "erc_solver_cases.npz"))
# 2015-01-30 is a validation-period date on which SLSQP also fails; it comes from the
# independent audit's saved erc_regime inputs, which cover every rebalance date.
CASES.update(np.load(Path(__file__).parent / "fixtures" / "erc_solver_case_20150130.npz"))
FLAGGED = ["20200331", "20220429", "20260130"]  # final-period dates (these also carry drifted holdings)
FAILING = ["20150130"] + FLAGGED  # every date where SLSQP leaves assets at zero weight
EQUAL = np.full(10, 0.1)


def _random_cov(rng, n=10):
    loadings = rng.normal(size=(n, 3))
    return (loadings @ loadings.T + 0.1 * np.diag(rng.uniform(0.5, 2.0, n))) * 1e-4


def _share_error(w, cov, budgets=EQUAL):
    return float(np.abs(risk_contribution_shares(w, cov) - budgets).max())


def test_exact_solver_hits_every_risk_budget_on_random_covariances_and_budgets():
    rng = np.random.default_rng(0)
    for _ in range(50):
        cov, budgets = _random_cov(rng), rng.dirichlet(np.ones(10) * 2)
        w = solve_risk_budget_convex(cov, budgets)
        assert w.sum() == pytest.approx(1.0) and (w > 0).all()
        assert _share_error(w, cov, budgets) < 1e-8


def test_slsqp_agrees_with_the_exact_solver_wherever_it_converges_to_risk_parity():
    """Where SLSQP does reach risk parity the solution is the unique exact one, so
    the two must agree to solver precision. SLSQP's reliability depends on the
    covariance's structure (it fails on the three real dates below and on many
    synthetic draws, but is accurate on most real dates), so this checks agreement
    on the draws where it converges, over enough draws that the check cannot pass
    vacuously, and makes no claim about how often it converges."""
    rng = np.random.default_rng(1)
    agreed = 0
    for _ in range(150):
        a = rng.normal(size=(10, 10))
        cov = (a @ a.T / 10 + 0.05 * np.eye(10)) * 1e-4
        w_slsqp = solve_risk_budget(cov)
        if _share_error(w_slsqp, cov) < 1e-5:
            assert np.allclose(w_slsqp, solve_risk_budget_convex(cov), atol=1e-4)
            agreed += 1
    assert agreed >= 10


@pytest.mark.parametrize("date", FAILING)
def test_exact_solver_reaches_risk_parity_on_the_real_covariances_where_slsqp_fails(date):
    w = solve_risk_budget_convex(CASES[f"cov_{date}"])
    assert _share_error(w, CASES[f"cov_{date}"]) < 1e-8
    assert (w > 0.005).all(), "the exact solution keeps every asset invested"


@pytest.mark.parametrize("date", FAILING)
def test_saved_slsqp_targets_for_the_flagged_dates_leave_assets_at_zero_weight(date):
    """Records the defect the audit found, using the weights the frozen pipeline
    actually produced. This documents the current behaviour of solve_risk_budget;
    if it is ever replaced by the exact solver, update or remove this test."""
    w, cov = CASES[f"w_slsqp_{date}"], CASES[f"cov_{date}"]
    assert (w < 1e-4).any()
    assert _share_error(w, cov) > 0.05


@pytest.mark.xfail(strict=True, reason="SLSQP stops at a local optimum with assets pinned at zero on these real "
                   "covariances; fails today by design of the objective, passes if the solver is replaced")
@pytest.mark.parametrize("date", FAILING)
def test_slsqp_reaches_risk_parity_on_the_real_flagged_covariances(date):
    cov = CASES[f"cov_{date}"]
    assert _share_error(solve_risk_budget(cov), cov) < 1e-4


@pytest.mark.skipif(not (_TORCH_AVAILABLE and _CVXPYLAYERS_AVAILABLE), reason="requires torch and cvxpylayers")
@pytest.mark.parametrize("date", FAILING)
def test_training_layer_agrees_with_the_exact_solver_on_the_real_covariances(date):
    import torch
    from allocation_lstm import budgets_to_weights_differentiable

    cov = CASES[f"cov_{date}"]
    w_layer = budgets_to_weights_differentiable(torch.tensor(EQUAL, dtype=torch.float64), cov).detach().numpy()
    # 1e-3 is the tolerance the layer's own test already uses (conic solver accuracy);
    # SLSQP's error on these same inputs is 0.1.
    assert np.abs(w_layer - solve_risk_budget_convex(cov)).max() < 1e-3
    assert _share_error(w_layer, cov) < 1e-3


def test_projection_leaves_a_feasible_exact_target_unchanged():
    """2020-03-31: the exact ERC target is within the cap and needs only about
    10.7% turnover, so projecting it onto the constraints must not move it, and
    risk parity survives the projection."""
    cov, w_drift = CASES["cov_erc_regime_20200331"], CASES["w_drift_erc_regime_20200331"]
    target = solve_risk_budget_convex(cov)
    assert target.max() <= SPEC.upper and turnover(target, w_drift) < SPEC.max_turnover
    projected = project_onto_constraints(target, w_drift, SPEC)
    assert np.allclose(projected, target, atol=1e-6)
    assert _share_error(projected, cov) < 1e-5


@pytest.mark.parametrize("date", ["20220429", "20260130", "20220531", "20260227"])
def test_projection_of_an_infeasible_exact_target_respects_every_constraint_and_binds_turnover(date):
    cov, w_drift = CASES[f"cov_erc_regime_{date}"], CASES[f"w_drift_erc_regime_{date}"]
    target = solve_risk_budget_convex(cov)
    assert turnover(target, w_drift) > SPEC.max_turnover, "the unconstrained target must violate the limit on these dates"
    projected = project_onto_constraints(target, w_drift, SPEC)
    assert projected.sum() == pytest.approx(1.0, abs=1e-8)
    assert projected.min() >= SPEC.lower - 1e-8 and projected.max() <= SPEC.upper + 1e-8
    assert turnover(projected, w_drift) == pytest.approx(SPEC.max_turnover, abs=1e-6)
    assert turnover(projected, w_drift) < turnover(target, w_drift)


def test_exact_target_through_the_projection_beats_the_frozen_zero_weight_target_on_april_2022():
    """The frozen pipeline's erc_regime allocation on 2022-04-29 had a worst
    risk-share deviation of 0.1126 (its raw target had two assets at zero). The
    exact target through the same projection achieves a small fraction of that."""
    cov, w_drift = CASES["cov_erc_regime_20220429"], CASES["w_drift_erc_regime_20220429"]
    frozen_final = project_onto_constraints(CASES["w_slsqp_20220429"], w_drift, SPEC)
    corrected = project_onto_constraints(solve_risk_budget_convex(cov), w_drift, SPEC)
    assert _share_error(frozen_final, cov) > 0.10
    assert _share_error(corrected, cov) < 0.01
