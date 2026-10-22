"""
Equal Risk Contribution (ERC) risk-parity allocator.

Corresponds to M2 "In ERC Risk Parity we want each asset to contribute
equally to the variance of the portfolio." Baseline uses a single pooled
covariance estimate; the regime-aware version combines state-specific
covariance estimates weighted by the current (one-step-ahead) regime
probabilities, using identical regularization in both cases (M2's
explicit requirement, so the ERC comparison isolates the information
effect rather than a difference in covariance-estimation machinery).

Owner: AnnaLisa & Nilay (traditional and ML allocation).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def regularize_covariance(cov: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    """Simple diagonal (Ledoit-Wolf-style) shrinkage toward the identity
    scaled by the average variance, applied identically to every
    covariance estimate in this module -- baseline or regime-conditional
    -- so the *same* regularization rule is used everywhere, per M2.
    """
    n = cov.shape[0]
    target = np.eye(n) * np.trace(cov) / n
    return (1 - shrinkage) * cov + shrinkage * target


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's contribution to total portfolio variance: w_i * (Cov w)_i."""
    return w * (cov @ w)


def risk_contribution_shares(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's risk contribution as a share of total portfolio
    variance (sums to 1) -- comparable directly against equal target
    budgets (1/n), per M2/M3's "verify the degree of conformance of the
    actually constructed portfolio with the expected risk budgets."
    """
    rc = risk_contributions(w, cov)
    port_var = float(w @ cov @ w)
    return rc / port_var


def solve_risk_budget(
    cov: np.ndarray,
    target_budgets: np.ndarray | None = None,
    w_bounds: tuple[float, float] = (0.0, 0.60),
) -> np.ndarray:
    """Solve for long-only, fully-invested weights whose risk contributions
    match `target_budgets` (defaults to equal budgets, i.e. plain ERC).
    This is the "risk-budgeting optimization layer" shared between ERC
    and the LSTM allocator in M2's architecture -- ERC calls it with
    equal fixed budgets, the LSTM allocator (allocation_lstm.py) calls
    the same function with its learned budgets.
    """
    n = cov.shape[0]
    if target_budgets is None:
        target_budgets = np.full(n, 1.0 / n)
    target_budgets = target_budgets / target_budgets.sum()

    w0 = np.full(n, 1.0 / n)

    def objective(w: np.ndarray) -> float:
        port_var = w @ cov @ w
        rc = risk_contributions(w, cov)
        rc_share = rc / port_var
        return float(np.sum((rc_share - target_budgets) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [w_bounds] * n

    result = minimize(
        objective, w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Risk-budgeting optimization failed: {result.message}")
    w = np.clip(result.x, 0, None)
    return w / w.sum()


def solve_risk_budget_convex(
    cov: np.ndarray,
    target_budgets: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """Exact long-only risk-budgeting weights from the convex log-barrier
    formulation (Spinu 2013): minimise 0.5 y'Sigma y - sum_i b_i log y_i over
    y > 0, then w = y / sum(y). The problem is strictly convex, so its solution
    is unique and strictly positive, and its first-order condition
    y_i (Sigma y)_i = b_i makes every risk-contribution share equal its budget
    exactly. This is the same formulation the differentiable training layer
    (allocation_lstm.differentiable_risk_budget_layer) solves.

    solve_risk_budget minimises the squared error between risk shares and
    budgets directly. That objective is non-convex in w, so SLSQP can stop at a
    local optimum with assets pinned at zero weight; this function cannot. No
    weight cap is imposed here; caps and turnover limits are applied afterwards
    by constraints.project_onto_constraints. Damped Newton with a
    positivity-preserving backtracking line search; requires a positive
    definite `cov`.
    """
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if target_budgets is None else np.asarray(target_budgets, dtype=float)
    b = b / b.sum()
    y = np.sqrt(b / np.diag(cov))  # positive starting point near the diagonal solution

    def objective(v: np.ndarray) -> float:
        return 0.5 * v @ cov @ v - b @ np.log(v)

    for _ in range(max_iter):
        grad = cov @ y - b / y
        err = np.max(np.abs(grad * y))  # worst |risk contribution - budget| in units of the budget scale
        if err < tol:
            break
        hess = cov + np.diag(b / y ** 2)
        step = np.linalg.solve(hess, grad)
        t, f0 = 1.0, objective(y)
        slack = 1e-13 * max(1.0, abs(f0))  # near the optimum f changes by less than float rounding
        while t > 1e-14 and (np.any(y - t * step <= 0) or objective(y - t * step) > f0 - 1e-4 * t * (grad @ step) + slack):
            t *= 0.5
        if t <= 1e-14:
            if err < 1e-8:
                break  # stalled at floating-point precision, already accurate far beyond what is needed
            raise RuntimeError("solve_risk_budget_convex: line search failed before converging.")
        y = y - t * step
    else:
        raise RuntimeError("solve_risk_budget_convex did not converge.")
    return y / y.sum()


RISK_BUDGET_SOLVERS = ("exact", "slsqp")


def risk_budget_weights(
    cov: np.ndarray,
    target_budgets: np.ndarray | None = None,
    solver: str = "exact",
) -> np.ndarray:
    """Risk-budgeting weights from the chosen solver. "exact" (the default) is the
    strictly convex log-barrier solution, solve_risk_budget_convex. "slsqp" is the
    original squared-share-error solve, solve_risk_budget, which is non-convex and
    can leave assets at zero weight although every budget is positive (see
    docs/verification/solver_and_accounting_checks.md); it is kept so results from
    the original pipeline can be reproduced and compared.
    """
    if solver == "exact":
        return solve_risk_budget_convex(cov, target_budgets)
    if solver == "slsqp":
        return solve_risk_budget(cov, target_budgets)
    raise ValueError(f"solver must be one of {RISK_BUDGET_SOLVERS}; got {solver!r}")


def erc_baseline_weights(pooled_cov: np.ndarray, shrinkage: float = 0.10, solver: str = "exact") -> np.ndarray:
    """Baseline ERC: equal risk budgets against a single pooled covariance
    estimate, no regime separation."""
    cov = regularize_covariance(pooled_cov, shrinkage)
    return risk_budget_weights(cov, target_budgets=None, solver=solver)


def erc_regime_weights(
    state_covs: np.ndarray,
    regime_probs: np.ndarray,
    shrinkage: float = 0.10,
    solver: str = "exact",
) -> np.ndarray:
    """Regime-aware ERC: equal risk budgets against a covariance built by
    mixing the per-state covariance estimates according to the current
    (one-step-ahead) regime probabilities, i.e.
        Sigma_t = sum_k P(S_t+1 = k | F_t) * Sigma_k
    then regularized identically to the baseline.
    """
    mixture_cov = np.tensordot(regime_probs, state_covs, axes=(0, 0))
    cov = regularize_covariance(mixture_cov, shrinkage)
    return risk_budget_weights(cov, target_budgets=None, solver=solver)
