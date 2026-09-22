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


def erc_baseline_weights(pooled_cov: np.ndarray, shrinkage: float = 0.10) -> np.ndarray:
    """Baseline ERC: equal risk budgets against a single pooled covariance
    estimate, no regime separation."""
    cov = regularize_covariance(pooled_cov, shrinkage)
    return solve_risk_budget(cov, target_budgets=None)


def erc_regime_weights(
    state_covs: np.ndarray,
    regime_probs: np.ndarray,
    shrinkage: float = 0.10,
) -> np.ndarray:
    """Regime-aware ERC: equal risk budgets against a covariance built by
    mixing the per-state covariance estimates according to the current
    (one-step-ahead) regime probabilities, i.e.
        Sigma_t = sum_k P(S_t+1 = k | F_t) * Sigma_k
    then regularized identically to the baseline.
    """
    mixture_cov = np.tensordot(regime_probs, state_covs, axes=(0, 0))
    cov = regularize_covariance(mixture_cov, shrinkage)
    return solve_risk_budget(cov, target_budgets=None)
