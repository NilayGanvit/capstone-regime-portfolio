"""
ERC (Equal Risk Contribution) Risk Parity allocator.

Owner: AnnaLisa / Nilay

Responsibilities (from the M2 problem statement):
- Baseline version: fixed equal risk budgets (b_i = 1/10), single pooled
  covariance matrix (no regime separation).
- Regime-aware version: same equal budgets, but the covariance matrix is
  the current-probability-weighted combination of the state-specific
  covariance estimates. Same regularization criteria for both versions.
- Long-only, fully invested. Output feeds into the shared risk-budgeting
  optimizer in optimization/risk_budgeting.py.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

EQUAL_BUDGETS = np.full(10, 1 / 10)


def pooled_covariance(returns: pd.DataFrame, regularization: str = "ledoit_wolf") -> np.ndarray:
    """TODO: single covariance estimate, no regime separation."""
    raise NotImplementedError


def regime_weighted_covariance(
    state_covariances: list[np.ndarray],
    state_probs: np.ndarray,
    regularization: str = "ledoit_wolf",
) -> np.ndarray:
    """TODO: combine per-state covariances using current regime probabilities."""
    raise NotImplementedError


def erc_weights(covariance: np.ndarray, budgets: np.ndarray = EQUAL_BUDGETS) -> np.ndarray:
    """
    TODO: solve for long-only weights such that each asset's risk
    contribution matches its target budget. Delegates the actual
    budgets -> weights conversion to optimization.risk_budgeting.
    """
    raise NotImplementedError
