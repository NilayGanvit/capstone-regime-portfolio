"""
Asset-specific regime detection (trial 18, see
docs/experiments/asset_specific_regimes_preregistration.md).

One univariate Gaussian HMM per asset instead of one joint HMM across all
assets, so each asset's volatility regime can differ from the others'.
Each model's one-step-ahead mixture variance replaces that asset's
sample variance on the diagonal of the pooled covariance; correlations
stay pooled, since separate per-asset models define no joint state.
Wired into walkforward.run_walk_forward as the opt-in
`asset_regime_states` option (config erc_asset_regime).

    per-asset log returns (x100) -> 2-state HMM per asset
      -> P(S_t+1 | F_t) -> mixture variance per asset
      -> D R D with R the pooled correlation -> ERC
"""

from __future__ import annotations

import numpy as np

from regime import GaussianHMM

# Returns are fitted in percent: GaussianHMM adds a 1e-6 variance floor,
# which is large next to a Treasury ETF's daily variance in decimal units.
RETURN_SCALE = 100.0


def fit_asset_hmms(
    X: np.ndarray,
    n_states: int = 2,
    init_from: list[GaussianHMM] | None = None,
    n_iter: int = 200,
) -> list[GaussianHMM]:
    """Fit one univariate HMM per column of X (T, n_assets), each on its
    own scaled log returns. With `init_from` (the previous fits, one per
    asset), each EM is warm-started from that asset's previous fit."""
    hmms = []
    for i in range(X.shape[1]):
        x_i = RETURN_SCALE * X[:, [i]]
        model = GaussianHMM(n_states=n_states, n_iter=n_iter, random_state=0)
        hmms.append(model.fit(x_i, init_from=None if init_from is None else init_from[i]))
    return hmms


def mixture_variance(probs: np.ndarray, means: np.ndarray, variances: np.ndarray) -> float:
    """Variance of a mixture with weights `probs`:
    sum_k p_k (sigma_k^2 + mu_k^2) - (sum_k p_k mu_k)^2. Unchanged by any
    relabelling of the states."""
    return float(probs @ (variances + means**2) - (probs @ means) ** 2)


def asset_regime_forecast(hmms: list[GaussianHMM], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For history X (T, n_assets) through close(t), each asset's
    one-step-ahead forecast variance (decimal units) and its probability
    of being in its higher-variance state tomorrow."""
    n_assets = X.shape[1]
    variances = np.empty(n_assets)
    high_var_prob = np.empty(n_assets)
    for i, model in enumerate(hmms):
        p = model.predicted_probabilities(RETURN_SCALE * X[:, [i]])[-1]
        mu = model.means_[:, 0]
        var = model.covars_[:, 0, 0]
        variances[i] = mixture_variance(p, mu, var) / RETURN_SCALE**2
        high_var_prob[i] = p[np.argmax(var)]
    return variances, high_var_prob


def asset_regime_covariance(pooled_cov: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """D R D: the pooled correlation R with each asset's forecast
    standard deviation on the diagonal scaling D."""
    sd = np.sqrt(np.diag(pooled_cov))
    corr = pooled_cov / np.outer(sd, sd)
    d = np.sqrt(variances)
    return corr * np.outer(d, d)
