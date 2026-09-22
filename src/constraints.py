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
    """
    n = len(w_raw)

    def objective(w: np.ndarray) -> float:
        return float(np.sum((w - w_raw) ** 2))

    constraints = [
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
        {"type": "ineq", "fun": lambda w: spec.max_turnover - np.sum(np.abs(w - w_drifted))},
    ]
    bounds = [(spec.lower, spec.upper)] * n
    w0 = w_drifted.copy()

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"Constraint projection failed: {result.message}")
    w = np.clip(result.x, spec.lower, spec.upper)
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
