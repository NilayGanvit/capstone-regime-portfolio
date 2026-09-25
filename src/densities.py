"""
M0 / M1 predictive return densities for the reliability layer.

Corresponds to M2/M3's reliability specification:
  M0: a single (pooled, regime-agnostic) multivariate Student-t density.
  M1: a mixture of Student-t densities, weighted by the HMM's
      one-step-ahead regime probabilities.
  A single shared degrees-of-freedom nu (> 2) across M0 and every
  component of M1 -- the symmetric-family choice that keeps the ablation
  attributable to regime-conditioning rather than to extra distributional
  flexibility. `fit_per_regime_nu` below is the opt-in alternative (M2's
  own scope note deferred this to "a secondary robustness check only if
  time allows" -- see README): one nu per M1 component instead of one
  shared across M0 and M1, wired into walkforward.run_walk_forward via the
  `per_regime_nu` toggle. M0 always keeps the pooled, shared nu -- it is
  regime-agnostic by construction, so "per-regime" does not apply to it.

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


def fit_per_regime_nu(
    X: np.ndarray,
    filtered_state_probs: np.ndarray,
    nu_grid: np.ndarray | None = None,
) -> np.ndarray:
    """One degrees-of-freedom estimate per HMM state, each grid-searched
    against that state's probability-weighted standardized residuals,
    rather than one nu shared across every state (fit_shared_nu above).
    Same grid-search rationale as fit_shared_nu -- nu only needs to be
    "good enough," and a grid keeps the estimate away from the
    numerically unstable region near nu=2 -- but weighting each
    residual's log-likelihood contribution by filtered_state_probs[:, k]
    before summing, the same soft-membership weighting
    M1RegimeMixtureStudentT.fit already uses for each state's (mean, cov)
    via weighted_mean_cov, applied here to nu instead. `X` is (T,
    n_features) raw returns (not pre-standardized -- standardization is
    done internally per feature, unlike fit_shared_nu, since callers here
    always have the raw window in hand rather than a pre-pooled residual
    series). `filtered_state_probs` is (T, n_states) and must already be
    causal (filtered, not smoothed), matching M1RegimeMixtureStudentT.fit's
    own requirement.

    Wired into walkforward.run_walk_forward via the `per_regime_nu`
    toggle (default False, i.e. fit_shared_nu remains the default) -- see
    that module and README "Known scope limitations."
    """
    if nu_grid is None:
        nu_grid = np.concatenate([np.arange(2.5, 10, 0.5), np.arange(10, 31, 2)])
    n_states = filtered_state_probs.shape[1]
    n_features = X.shape[1]
    std_resid = ((X - X.mean(axis=0)) / X.std(axis=0)).ravel()
    finite = np.isfinite(std_resid)
    x = std_resid[finite]

    nus = np.empty(n_states)
    for k in range(n_states):
        w = np.repeat(filtered_state_probs[:, k], n_features)[finite]
        best_nu, best_ll = nu_grid[0], -np.inf
        for nu in nu_grid:
            ll = np.sum(w * multivariate_t.logpdf(x[:, None], loc=[0.0], shape=[[1.0]], df=nu))
            if ll > best_ll:
                best_ll, best_nu = ll, nu
        nus[k] = best_nu
    return nus


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
        n_features = X.shape[1]
        sample_cov = sample_cov + 1e-6 * np.eye(n_features)
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

    `nu` may be a scalar (default: one degrees-of-freedom shared across
    every state, from fit_shared_nu) or a (n_states,) array (one nu per
    state, from fit_per_regime_nu) -- see that function's docstring and
    walkforward.py's `per_regime_nu` toggle.
    """

    nu: float | np.ndarray
    means_: np.ndarray = None    # (n_states, n_features)
    scales_: np.ndarray = None   # (n_states, n_features, n_features)

    def _nu_for_state(self, k: int) -> float:
        return float(self.nu[k]) if np.ndim(self.nu) > 0 else float(self.nu)

    def fit(self, X: np.ndarray, filtered_state_probs: np.ndarray) -> "M1RegimeMixtureStudentT":
        """Estimate each state's (mean, scale) from X, weighted by the
        filtered probabilities available *at the time each observation
        was made* -- this is the walk-forward-safe weighting Silvio
        specified, not a full-sample smoothed fit.
        """
        n_states = filtered_state_probs.shape[1]
        n_features = X.shape[1]
        means, scales = [], []
        for k in range(n_states):
            mean_k, cov_k = weighted_mean_cov(X, filtered_state_probs[:, k])
            cov_k = cov_k + 1e-6 * np.eye(n_features)
            nu_k = self._nu_for_state(k)
            scale_k = cov_k * (nu_k - 2) / nu_k
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
                r, loc=self.means_[k], shape=self.scales_[k], df=self._nu_for_state(k)
            )
        mixture_density = np.dot(predicted_state_probs, component_densities)
        return float(np.log(max(mixture_density, 1e-300)))
