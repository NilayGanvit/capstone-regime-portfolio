"""
Joint constraint projection: takes a raw target weight vector (from ERC or
the LSTM risk-budgeting layer) and projects it onto the common constraint
set -- long-only, fully invested, per-asset weight bounds, and a turnover
limit relative to the drifted holdings at the next open.

Corresponds to M2's "Constraints and assessment" and the M3 literature
section on the risk-budgeting optimizer: constraints are satisfied
*simultaneously* via one joint projection (not sequential clipping, which
can silently violate an earlier constraint while fixing a later one), and
turnover is computed on final weights, drifted with price movements
between rebalances, per Olivares-Nadal & DeMiguel's (2018) treatment of
turnover as a term to be controlled directly rather than an
implementation afterthought.

Owner: Nilay (optimization, constraints, and evaluation rigor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ConstraintSpec:
    lower: float = 0.0       # long-only
    upper: float = 0.30      # per-asset weight cap
    max_turnover: float = 0.50  # sum(|w_new - w_drifted|), one-way


def drift_weights(w_prev: np.ndarray, gross_returns: np.ndarray) -> np.ndarray:
    """Weights implied by holding `w_prev` through a period with per-asset
    gross returns (1 + r), before any rebalancing trade. This is the
    reference point turnover is measured against -- not the previous
    *target* weight, which would understate real trading if prices moved.
    """
    drifted_value = w_prev * gross_returns
    return drifted_value / drifted_value.sum()


def project_onto_constraints(
    w_raw: np.ndarray,
    w_drifted: np.ndarray,
    spec: ConstraintSpec,
) -> np.ndarray:
    """Project `w_raw` (the unconstrained ERC/LSTM output) onto the
    long-only, weight-bound, budget, and turnover-limit constraints
    jointly, minimizing squared distance to the raw target. This is a
    small QP; solved via SLSQP since cvxpy is not available in this
    environment, but the formulation matches what the MOSEK cookbook
    documents for turnover-constrained rebalancing.

    If the turnover limit makes the raw target infeasible from the
    drifted starting point, the closest feasible point (in the turnover
    metric) is returned, and the binding constraint should be logged by
    the caller -- this function only solves, it does not decide whether a
    binding turnover constraint is a problem worth flagging.

    The turnover limit sum(|w_i - w_drifted_i|) <= max_turnover is
    reformulated with slack variables (w_i - w_drifted_i = u_i - v_i,
    u_i, v_i >= 0, sum(u_i + v_i) <= max_turnover) rather than passed to
    SLSQP as a raw np.abs() constraint. SLSQP estimates constraint
    gradients by finite differences, and np.abs() has a kink at
    w_i == w_drifted_i; real ETF data hits that kink often enough
    (e.g. an unconstrained ERC target pinned above the weight cap) to
    stall the solver at the iteration limit, even though it never came
    up against synthetic data. The slack form is smooth everywhere and
    solves the identical QP.
    """
    n = len(w_raw)

    def objective(x: np.ndarray) -> float:
        w = x[:n]
        return float(np.sum((w - w_raw) ** 2))

    def objective_grad(x: np.ndarray) -> np.ndarray:
        w = x[:n]
        grad = np.zeros_like(x)
        grad[:n] = 2.0 * (w - w_raw)
        return grad

    constraints = [
        {"type": "eq", "fun": lambda x: x[:n].sum() - 1.0,
         "jac": lambda x: np.concatenate([np.ones(n), np.zeros(2 * n)])},
        {"type": "eq",
         "fun": lambda x: x[:n] - w_drifted - x[n:2 * n] + x[2 * n:],
         "jac": lambda x: np.hstack([np.eye(n), -np.eye(n), np.eye(n)])},
        {"type": "ineq", "fun": lambda x: spec.max_turnover - x[n:].sum(),
         "jac": lambda x: np.concatenate([np.zeros(n), -np.ones(2 * n)])},
    ]
    bounds = [(spec.lower, spec.upper)] * n + [(0.0, None)] * (2 * n)
    w0 = np.clip(w_drifted, spec.lower, spec.upper)
    x0 = np.concatenate([w0, np.zeros(n), np.zeros(n)])

    result = minimize(
        objective, x0, jac=objective_grad, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Constraint projection failed: {result.message}")
    w = np.clip(result.x[:n], spec.lower, spec.upper)
    return w / w.sum()


def turnover(w_new: np.ndarray, w_drifted: np.ndarray) -> float:
    """One-way turnover: sum of absolute weight changes relative to the
    drifted (pre-trade) holdings -- the same definition used for both the
    portfolio constraint and the LSTM training-loss turnover penalty
    (M2's explicit requirement that these must match)."""
    return float(np.sum(np.abs(w_new - w_drifted)))


def binding_constraints(w: np.ndarray, w_drifted: np.ndarray, spec: ConstraintSpec, tol: float = 1e-6) -> dict:
    """Report which constraints are binding for this rebalance, for the
    diagnostic logging M2/M3 call for ("we will report on which
    constraints are binding")."""
    return {
        "lower_bound_binding": [i for i, wi in enumerate(w) if wi <= spec.lower + tol],
        "upper_bound_binding": [i for i, wi in enumerate(w) if wi >= spec.upper - tol],
        "turnover_binding": turnover(w, w_drifted) >= spec.max_turnover - tol,
    }
