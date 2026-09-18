"""
Reliability layer: M0/M1 predictive densities and the DMA-style pi update.

Owner: Silvio

Responsibilities (from the M2 problem statement):
- M0: single multivariate Student-t predictive density for next-day
  ETF returns (no regime conditioning).
- M1: mixture of Student-t densities, weighted by ONE-STEP-AHEAD
  regime probabilities (from hmm.py). Same shared degrees-of-freedom
  parameter nu (> 2) as M0, estimated on pooled data.
- M1's per-state mu_k / Sigma_k are estimated inside each walk-forward
  training window, using return observations weighted by the FILTERED
  HMM state probabilities available at that time -- never smoothed or
  full-sample estimates.
- Recursive pi update (Raftery et al., adapted to the two-model case):

      L_{j,t} = p(r_t | F_{t-1}, M_j),  j in {0, 1}
      pi_t = (pi_{t-1}^alpha * L_{1,t}) /
             (pi_{t-1}^alpha * L_{1,t} + (1 - pi_{t-1})^alpha * L_{0,t})

  alpha (forgetting factor) is fixed during validation and held constant.
  pi starts at 0.5. Score each day using the density SAVED at t-1 --
  never using same-day information.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def fit_shared_nu(standardized_residuals: np.ndarray) -> float:
    """
    TODO: fit a single shared degrees-of-freedom parameter (nu > 2)
    on pooled/aggregate standardized residuals.
    """
    raise NotImplementedError


def m0_density(returns: np.ndarray, mu, sigma, nu: float) -> float:
    """TODO: single multivariate Student-t predictive density (no regime)."""
    raise NotImplementedError


def m1_density(returns: np.ndarray, state_params: list, state_probs: np.ndarray, nu: float) -> float:
    """
    TODO: mixture of Student-t densities weighted by one-step-ahead
    regime probabilities. state_params is a list of (mu_k, Sigma_k)
    per state, estimated inside the current walk-forward window.
    """
    raise NotImplementedError


def update_pi(pi_prev: float, l1_t: float, l0_t: float, alpha: float) -> float:
    """TODO: one recursive DMA-style update step. See docstring formula above."""
    raise NotImplementedError


def blend_weights(w_regime: pd.Series, w_baseline: pd.Series, pi_t: float) -> pd.Series:
    """
    TODO: w_blend = pi_t * w_regime + (1 - pi_t) * w_baseline
    Applied separately within ERC and within LSTM (never across allocators).
    """
    raise NotImplementedError
