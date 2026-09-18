"""
Common constraints and joint projection.

Owner: Nilay

Responsibilities (from the M2 problem statement):
- Long-only, fully invested: w_i >= 0, sum(w_i) = 1.
- Common per-asset and per-asset-class weight limits.
- Turnover limit computed on ACTUAL weights at the time of rebalancing
  (i.e. after price drift since the last rebalance), not on scheduled
  targets and not on raw learned risk budgets.
- All constraints are satisfied via a JOINT projection (not sequential
  clipping), so one constraint being enforced doesn't silently violate
  another. Record which constraints bind and how realized risk
  contributions differ from the ERC targets.
- Transaction cost accounting consistent with the MOSEK Portfolio
  Optimization Cookbook.
"""

from __future__ import annotations
import numpy as np


def drifted_weights(prev_weights: np.ndarray, asset_returns: np.ndarray) -> np.ndarray:
    """TODO: weights after price drift since the last rebalance (turnover reference point)."""
    raise NotImplementedError


def joint_project(
    target_weights: np.ndarray,
    drifted_weights: np.ndarray,
    weight_bounds: tuple,
    class_bounds: dict,
    turnover_limit: float,
) -> np.ndarray:
    """
    TODO: project `target_weights` onto the intersection of all constraints
    at once (e.g. via a single QP), using `drifted_weights` as the
    turnover reference point.
    """
    raise NotImplementedError


def transaction_costs(weights_before: np.ndarray, weights_after: np.ndarray, cost_model) -> float:
    """TODO: cost accounting per the MOSEK Portfolio Optimization Cookbook."""
    raise NotImplementedError
