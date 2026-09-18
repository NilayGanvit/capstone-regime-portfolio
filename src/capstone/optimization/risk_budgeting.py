"""
Shared risk-budgeting optimization layer.

Owner: Nilay

Converts (risk budgets, covariance) -> long-only portfolio weights,
used identically by both ERC and the LSTM allocator so the comparison
between them isolates the allocation mechanism, not the portfolio
construction step:

    w_i * (Sigma @ w)_i / (w.T @ Sigma @ w) = b_i,  for each asset i
    w >= 0,  sum(w) = 1

Stretch goal (only if time allows): make this differentiable
(e.g. via cvxpylayers) so the LSTM can be trained fully end-to-end
through the optimization layer, following Uysal, Li & Mulvey (2021).
If that proves infeasible on the timeline, a non-differentiable
solve with a straight-through gradient approximation is an acceptable
fallback -- flag this choice explicitly in the report either way.
"""

from __future__ import annotations
import numpy as np


def solve_risk_budgeting(covariance: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    """
    TODO: solve for long-only weights matching the target risk budgets.
    A standard approach is cyclical coordinate descent / Newton's method
    on the risk-budgeting objective (e.g. Spinu's algorithm), or a
    convex reformulation solved via cvxpy.
    """
    raise NotImplementedError


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """TODO: return each asset's realized fractional contribution to portfolio variance."""
    raise NotImplementedError
