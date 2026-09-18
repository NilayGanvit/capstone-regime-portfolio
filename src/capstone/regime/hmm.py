"""
Regime inference module: three-state Gaussian HMM.

Owner: Silvio

Responsibilities (from the M2 problem statement):
- Fit a 3-state Gaussian HMM on candidate daily feature classes
  (returns, rolling-window volatility, trends, asset correlations,
  bond return dynamics). Final feature subset/lookback are chosen
  in development+validation, before the final OOS test.
- Do NOT assume a priori that states = risk-on/transition/risk-off;
  interpret states from the fitted characteristics after fitting.
- Produce FILTERED and ONE-STEP-AHEAD state probabilities only.
  Never expose smoothed / full-sample state probabilities to any
  downstream module used in the walk-forward test.
- Support the 2-state HMM as a robustness specification.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def fit_hmm(features: pd.DataFrame, n_states: int = 3, random_state: int = 0):
    """
    TODO: fit the Gaussian HMM on `features` using only data available
    up to the fit cutoff. Return the fitted model object.
    """
    raise NotImplementedError


def filtered_state_probabilities(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: return P(state_t | F_t) for each t -- filtered, not smoothed.
    """
    raise NotImplementedError


def one_step_ahead_probabilities(model, filtered_probs: pd.DataFrame) -> pd.DataFrame:
    """
    TODO: propagate filtered probabilities one step ahead through the
    transition matrix to get P(state_{t+1} | F_t).
    """
    raise NotImplementedError


def state_characteristics(model) -> pd.DataFrame:
    """
    TODO: summarize each fitted state's mean/vol/correlation profile,
    for post-hoc interpretation (do not label states before fitting).
    """
    raise NotImplementedError
