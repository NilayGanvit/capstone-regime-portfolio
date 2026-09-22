"""
Regime inference: a Gaussian Hidden Markov Model fit by EM (Baum-Welch),
with filtered, one-step-ahead predicted, and smoothed state probabilities
kept as clearly distinct outputs.

Corresponds to M3's "Probabilistic regime identification" discussion:
filtered P(S_t = k | F_t), predicted P(S_t+1 = k | F_t), and smoothed
P(S_t = k | F_T), T > t are NOT interchangeable. Only filtered and
predicted probabilities may be used to make a historical allocation
decision; smoothed probabilities are for retrospective interpretation
only and must never reach the allocators.

Implemented from scratch (no hmmlearn dependency) so behavior is fully
auditable and there is no external-package version risk for a component
this central to the project's identification strategy.

Owner: Silvio (regime inference and reliability).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import multivariate_normal


@dataclass
class GaussianHMM:
    """K-state Gaussian HMM over a multivariate observation sequence.

    Parameters
    ----------
    n_states : int
        Candidate state count (M2's primary specification is 3; 2 is the
        pre-specified robustness check). Not selected inside this class --
        `bic_for_state_counts` below is the selection tool, used only
        during chronological development/validation.
    n_iter, tol : EM stopping criteria.
    random_state : for reproducible initialization.
    """

    n_states: int
    n_iter: int = 200
    tol: float = 1e-6
    random_state: int = 0

    means_: np.ndarray = field(init=False, repr=False, default=None)
    covars_: np.ndarray = field(init=False, repr=False, default=None)
    transmat_: np.ndarray = field(init=False, repr=False, default=None)
    startprob_: np.ndarray = field(init=False, repr=False, default=None)
    n_features_: int = field(init=False, repr=False, default=None)
    converged_: bool = field(init=False, repr=False, default=False)
    log_likelihood_: float = field(init=False, repr=False, default=None)

    def _emission_probs(self, X: np.ndarray) -> np.ndarray:
        """T x K matrix of p(x_t | state=k) under the current Gaussian
        emission parameters."""
        T = X.shape[0]
        B = np.empty((T, self.n_states))
        for k in range(self.n_states):
            cov = self.covars_[k] + 1e-8 * np.eye(self.n_features_)  # ridge for stability
            B[:, k] = multivariate_normal.pdf(X, mean=self.means_[k], cov=cov)
        return np.clip(B, 1e-300, None)

    def _forward_backward(self, B: np.ndarray):
        """Standard scaled forward-backward. Returns filtered probabilities
        (alpha, normalized), smoothed probabilities (gamma), pairwise
        smoothed transition probabilities (xi-sum), and the data
        log-likelihood."""
        T, K = B.shape
        alpha = np.zeros((T, K))
        c = np.zeros(T)  # scaling factors

        alpha[0] = self.startprob_ * B[0]
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.transmat_) * B[t]
            c[t] = alpha[t].sum()
            alpha[t] /= c[t]

        beta = np.zeros((T, K))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.transmat_ @ (B[t + 1] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi_sum = np.zeros((K, K))
        for t in range(T - 1):
            num = (alpha[t][:, None] * self.transmat_) * (B[t + 1] * beta[t + 1])[None, :]
            xi_sum += num / c[t + 1]

        log_likelihood = np.sum(np.log(c))
        return alpha, beta, gamma, xi_sum, log_likelihood

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        """Fit by EM. X is (T, n_features), rows in chronological order."""
        X = np.asarray(X, dtype=float)
        T, F = X.shape
        self.n_features_ = F
        rng = np.random.default_rng(self.random_state)

        # k-means-ish init: random rows as initial means, global covariance
        # for every state, uniform transitions -- deliberately uninformative
        # so convergence isn't an artifact of a lucky start.
        init_idx = rng.choice(T, size=self.n_states, replace=False)
        self.means_ = X[init_idx].copy()
        global_cov = np.cov(X.T) + 1e-6 * np.eye(F)
        self.covars_ = np.array([global_cov.copy() for _ in range(self.n_states)])
        self.transmat_ = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        self.startprob_ = np.full(self.n_states, 1.0 / self.n_states)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            B = self._emission_probs(X)
            alpha, beta, gamma, xi_sum, ll = self._forward_backward(B)

            # M-step
            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi_sum / xi_sum.sum(axis=1, keepdims=True)

            for k in range(self.n_states):
                w = gamma[:, k]
                w_sum = w.sum()
                mean_k = (w[:, None] * X).sum(axis=0) / w_sum
                diff = X - mean_k
                cov_k = (w[:, None, None] * (diff[:, :, None] * diff[:, None, :])).sum(axis=0) / w_sum
                self.means_[k] = mean_k
                self.covars_[k] = cov_k + 1e-6 * np.eye(F)

            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                prev_ll = ll
                break
            prev_ll = ll

        self.log_likelihood_ = prev_ll
        return self

    def filtered_probabilities(self, X: np.ndarray) -> np.ndarray:
        """P(S_t = k | F_t) for each t -- the only probabilities safe to
        use for a historical allocation decision at t, alongside the
        one-step-ahead predicted probabilities below."""
        B = self._emission_probs(np.asarray(X, dtype=float))
        alpha, _, _, _, _ = self._forward_backward(B)
        return alpha

    def predicted_probabilities(self, X: np.ndarray) -> np.ndarray:
        """P(S_t+1 = k | F_t): filtered probabilities propagated one step
        forward through the transition matrix. This is what should feed
        the M0/M1 predictive densities and the allocators' regime inputs
        for the *next* decision, since it uses no information beyond t."""
        filtered = self.filtered_probabilities(X)
        return filtered @ self.transmat_

    def smoothed_probabilities(self, X: np.ndarray) -> np.ndarray:
        """P(S_t = k | F_T), T = length of X. Full-sample smoothing --
        for retrospective regime interpretation and diagnostic plots ONLY.
        Passing this into an allocator or a walk-forward backtest is
        exactly the look-ahead bug the M3 lit review warns against.
        """
        B = self._emission_probs(np.asarray(X, dtype=float))
        _, _, gamma, _, _ = self._forward_backward(B)
        return gamma

    def bic(self, X: np.ndarray) -> float:
        """Bayesian Information Criterion, for comparing candidate state
        counts during chronological development/validation only -- never
        against the final OOS period."""
        X = np.asarray(X, dtype=float)
        T, F = X.shape
        n_params = (
            self.n_states * F  # means
            + self.n_states * F * (F + 1) / 2  # covariances (symmetric)
            + self.n_states * (self.n_states - 1)  # transition matrix (rows sum to 1)
            + (self.n_states - 1)  # initial distribution
        )
        return -2 * self.log_likelihood_ + n_params * np.log(T)


def bic_for_state_counts(X: np.ndarray, candidates=(2, 3, 4), random_state: int = 0) -> dict:
    """Fit an HMM for each candidate state count and report BIC, so the
    3-state primary specification can be justified against 2 and 4 rather
    than on interpretability alone (the open item flagged against the M2
    draft)."""
    results = {}
    for k in candidates:
        model = GaussianHMM(n_states=k, random_state=random_state).fit(X)
        results[k] = {"bic": model.bic(X), "log_likelihood": model.log_likelihood_}
    return results
