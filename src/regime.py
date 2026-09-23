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

    def fit(self, X: np.ndarray, init_from: "GaussianHMM | None" = None) -> "GaussianHMM":
        """Fit by EM. X is (T, n_features), rows in chronological order.

        If `init_from` is given, EM is warm-started from its means_/covars_/
        transmat_/startprob_ instead of a fresh random init -- confirmed by
        diagnostic investigation to be necessary for scheduled monthly
        refits: independent random inits land in a different-but-plausible
        local optimum most months, causing regime probabilities to whipsaw
        and erc_regime's target weights/drawdown to follow. Raising n_iter
        does not fix this -- the EM trajectory is already flat well before
        50 iterations regardless of init.
        """
        X = np.asarray(X, dtype=float)
        T, F = X.shape
        self.n_features_ = F

        if init_from is not None:
            if init_from.n_states != self.n_states:
                raise ValueError(
                    f"Cannot warm-start: self.n_states={self.n_states} != "
                    f"init_from.n_states={init_from.n_states}"
                )
            self.means_ = init_from.means_.copy()
            self.covars_ = init_from.covars_.copy()
            self.transmat_ = init_from.transmat_.copy()
            self.startprob_ = init_from.startprob_.copy()
        else:
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

    def align_to(self, reference: "GaussianHMM") -> "GaussianHMM":
        """Permute self's state-indexed parameters (means_, covars_,
        startprob_, transmat_) in place so state k refers to the same regime
        as reference's state k, for label continuity across scheduled refits.
        Mutates and returns self. Raises ValueError if self.n_states !=
        reference.n_states. The very first fit has no reference and is never
        aligned -- it keeps whatever labeling .fit() produces."""
        if self.n_states != reference.n_states:
            raise ValueError(
                f"Cannot align: self.n_states={self.n_states} != "
                f"reference.n_states={reference.n_states}"
            )
        perm = state_alignment_permutation(
            self.means_, self.covars_, reference.means_, reference.covars_
        )
        self.means_ = self.means_[perm]
        self.covars_ = self.covars_[perm]
        self.startprob_ = self.startprob_[perm]
        self.transmat_ = self.transmat_[perm][:, perm]
        return self


def bhattacharyya_distance(
    mean_a: np.ndarray,
    cov_a: np.ndarray,
    mean_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    """Symmetric Bhattacharyya distance between two multivariate Gaussians.
    Combines mean separation and covariance-shape separation into one
    unit-consistent (nats) distance with no free weighting hyperparameter.
    Used as the state-matching cost for state_alignment_permutation()."""
    F = mean_a.shape[0]
    sigma_avg = (cov_a + cov_b) / 2.0 + 1e-8 * np.eye(F)
    diff = mean_a - mean_b

    sign_a, logdet_a = np.linalg.slogdet(cov_a)
    sign_b, logdet_b = np.linalg.slogdet(cov_b)
    sign_avg, logdet_avg = np.linalg.slogdet(sigma_avg)

    if sign_a <= 0 or sign_b <= 0 or sign_avg <= 0:
        return float("inf")

    term1 = 0.125 * diff @ np.linalg.solve(sigma_avg, diff)
    term2 = 0.5 * (logdet_avg - 0.5 * (logdet_a + logdet_b))
    return float(term1 + term2)


def state_alignment_permutation(
    new_means: np.ndarray,
    new_covars: np.ndarray,
    ref_means: np.ndarray,
    ref_covars: np.ndarray,
) -> np.ndarray:
    """Return a permutation array perm of length K such that
    new_means[perm], new_covars[perm] are reordered to best match
    ref_means/ref_covars in the Bhattacharyya-distance sense.
    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
    Pure function over raw arrays -- independently unit-testable without
    a GaussianHMM instance."""
    from scipy.optimize import linear_sum_assignment

    K = new_means.shape[0]
    cost = np.empty((K, K))
    for i in range(K):
        for j in range(K):
            cost[i, j] = bhattacharyya_distance(
                new_means[i], new_covars[i], ref_means[j], ref_covars[j]
            )

    row_ind, col_ind = linear_sum_assignment(cost)
    perm = np.argsort(col_ind)
    return perm


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
