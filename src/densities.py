"""
M0 / M1 predictive return densities for the reliability layer.

Corresponds to M2/M3's reliability specification:
  M0: a single (pooled, regime-agnostic) multivariate Student-t density.
  M1: a mixture of Student-t densities, weighted by the HMM's
      one-step-ahead regime probabilities.
  A single shared degrees-of-freedom nu (> 2) across M0 and every
  component of M1 -- the symmetric-family choice that keeps the ablation
  attributable to regime-conditioning rather than to extra distributional
  flexibility.

State-specific (mu_k, Sigma_k) for M1 are estimated inside each
walk-forward window from return observations weighted by the *filtered*
HMM state probabilities available at that time (Silvio's specification in
the M2 v04 thread) -- never from smoothed/full-sample state information.

Owner: Silvio (regime inference and reliability).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import multivariate_t


def weighted_mean_cov(X: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Probability-weighted mean and covariance of X, rows in chronological
    order, `weights` summing to any positive total (normalized inside)."""
    w = weights / weights.sum()
    mean = (w[:, None] * X).sum(axis=0)
    diff = X - mean
    cov = (w[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0)
    return mean, cov


def fit_shared_nu(standardized_residuals: np.ndarray, nu_grid: np.ndarray | None = None) -> float:
    """Estimate a single degrees-of-freedom parameter (nu > 2) on pooled,
    standardized residuals by maximizing the univariate Student-t
    log-likelihood over a grid. A grid search is used rather than a
    gradient-based MLE because nu only needs to be "good enough" for a
    baseline specification per M2/M3 -- and a grid keeps the estimate
    bounded away from numerically unstable regions (nu near 2).

    `standardized_residuals` should already be univariate (e.g. each
    asset's residual divided by its own trailing volatility, pooled
    across assets and time) -- constructing that series is the caller's
    responsibility so this function stays agnostic to how residuals were
    standardized.
    """
    if nu_grid is None:
        nu_grid = np.concatenate([np.arange(2.5, 10, 0.5), np.arange(10, 31, 2)])
    x = standardized_residuals[np.isfinite(standardized_residuals)]
    best_nu, best_ll = nu_grid[0], -np.inf
    for nu in nu_grid:
        ll = multivariate_t.logpdf(x[:, None], loc=[0.0], shape=[[1.0]], df=nu).sum()
        if ll > best_ll:
            best_ll, best_nu = ll, nu
    return float(best_nu)


@dataclass
class M0PooledStudentT:
    """Regime-agnostic baseline predictive model: one multivariate
    Student-t fit on pooled training data, never updated intraday."""

    nu: float
    mean_: np.ndarray = None
    scale_: np.ndarray = None

    def fit(self, X: np.ndarray) -> "M0PooledStudentT":
        self.mean_ = X.mean(axis=0)
        # For a multivariate Student-t with df=nu, Cov = scale * nu/(nu-2),
        # so recover `scale` from the sample covariance accordingly.
        sample_cov = np.cov(X.T)
        self.scale_ = sample_cov * (self.nu - 2) / self.nu
        return self

    def log_density(self, r: np.ndarray) -> float:
        """log p(r | M0). r is a single realized return vector."""
        return float(multivariate_t.logpdf(r, loc=self.mean_, shape=self.scale_, df=self.nu))


@dataclass
class M1RegimeMixtureStudentT:
    """Regime-aware predictive model: a mixture of Student-t densities,
    one component per HMM state, weighted by the *one-step-ahead*
    predicted state probabilities (never filtered-at-t, never smoothed --
    predicted, since this is a forecast for the return that has not
    happened yet).
    """

    nu: float
    means_: np.ndarray = None    # (n_states, n_features)
    scales_: np.ndarray = None   # (n_states, n_features, n_features)

    def fit(self, X: np.ndarray, filtered_state_probs: np.ndarray) -> "M1RegimeMixtureStudentT":
        """Estimate each state's (mean, scale) from X, weighted by the
        filtered probabilities available *at the time each observation
        was made* -- this is the walk-forward-safe weighting Silvio
        specified, not a full-sample smoothed fit.
        """
        n_states = filtered_state_probs.shape[1]
        means, scales = [], []
        for k in range(n_states):
            mean_k, cov_k = weighted_mean_cov(X, filtered_state_probs[:, k])
            scale_k = cov_k * (self.nu - 2) / self.nu
            means.append(mean_k)
            scales.append(scale_k)
        self.means_ = np.array(means)
        self.scales_ = np.array(scales)
        return self

    def log_density(self, r: np.ndarray, predicted_state_probs: np.ndarray) -> float:
        """log p(r | M1), mixing the per-state Student-t densities by the
        one-step-ahead predicted probabilities for the period `r` realizes."""
        n_states = len(predicted_state_probs)
        component_densities = np.empty(n_states)
        for k in range(n_states):
            component_densities[k] = multivariate_t.pdf(
                r, loc=self.means_[k], shape=self.scales_[k], df=self.nu
            )
        mixture_density = np.dot(predicted_state_probs, component_densities)
        return float(np.log(max(mixture_density, 1e-300)))
