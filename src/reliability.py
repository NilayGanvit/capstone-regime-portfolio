"""
Reliability layer: recursive DMA-style updating of pi_t (the relative
predictive support for the regime-aware model M1 over the pooled baseline
M0), and blending of regime-aware/baseline portfolio weights using pi_t.

Implements exactly the M2/M3 formula (algebraically the two-model
special case of Raftery, Karny & Ettler's forgetting + Bayes-update
recursion, verified to collapse to this closed form in the M2 review):

    pi_t = (pi_{t-1}^alpha * L1_t) / (pi_{t-1}^alpha * L1_t
                                       + (1 - pi_{t-1})^alpha * L0_t)

where L0_t, L1_t are the M0/M1 predictive densities evaluated at the
newly realized return r_t, using densities that were computed and saved
*before* r_t was observed (see walkforward.py for where that save/score
ordering is enforced).

Owner: Silvio (regime inference and reliability).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReliabilityTracker:
    """Stateful pi_t tracker. One instance per allocator pairing that
    needs a reliability blend (M2's table blends ERC-with-ERC and
    LSTM-with-LSTM using the *same* pi_t path, since pi_t is a property of
    the regime signal's trustworthiness, not of which allocator consumes
    it -- so in practice the walk-forward harness owns exactly one
    instance and reuses it for both allocators).

    alpha : forgetting factor, fixed during validation (per M2) and held
        constant thereafter. alpha=1.0 means no forgetting (plain Bayes
        updating); alpha closer to 0.9-0.99 lets pi_t adapt faster to
        recent evidence.
    pi0 : starting value, per M2, 0.5.
    """

    alpha: float
    pi0: float = 0.5
    pi_: float = field(init=False, default=None)
    history_: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.pi0 < 1.0):
            raise ValueError("pi0 must be in (0, 1).")
        self.pi_ = self.pi0
        self.history_ = [self.pi_]

    def update(self, log_l0: float, log_l1: float) -> float:
        """One recursive update, given log-likelihoods log p(r_t | F_{t-1}, M_j)
        for the newly realized return. Logs are used internally and
        exponentiated only in the final ratio, to avoid overflow when
        likelihoods are very small (as they routinely are for
        multivariate densities).
        """
        # Work in log-space as far as possible: log(pi^alpha * L) = alpha*log(pi) + log(L)
        log_num = self.alpha * np.log(self.pi_) + log_l1
        log_den_other = self.alpha * np.log(1 - self.pi_) + log_l0
        # log-sum-exp for stability
        m = max(log_num, log_den_other)
        denom = np.exp(log_num - m) + np.exp(log_den_other - m)
        self.pi_ = float(np.exp(log_num - m) / denom)
        # Guard against exact 0/1 (would freeze all future updates via log(0))
        self.pi_ = float(np.clip(self.pi_, 1e-6, 1 - 1e-6))
        self.history_.append(self.pi_)
        return self.pi_


def blend_weights(w_regime: np.ndarray, w_baseline: np.ndarray, pi_t: float) -> np.ndarray:
    """w_t = pi_t * w_regime + (1 - pi_t) * w_baseline, applied within one
    allocator (ERC-with-ERC or LSTM-with-LSTM -- never across allocators).
    """
    return pi_t * w_regime + (1 - pi_t) * w_baseline


def select_forgetting_factor(
    log_l0: np.ndarray,
    log_l1: np.ndarray,
    candidates=(0.90, 0.95, 0.97, 0.99, 1.00),
    pi0: float = 0.5,
) -> float:
    """Choose alpha during chronological validation by picking the value
    that maximizes cumulative predictive log-likelihood of the *blended*
    model's implied density over a validation stretch of (log_l0, log_l1)
    pairs. This is a validation-period tool only -- per M2, alpha is
    fixed before the final OOS test and held constant thereafter.
    """
    best_alpha, best_score = candidates[0], -np.inf
    for alpha in candidates:
        tracker = ReliabilityTracker(alpha=alpha, pi0=pi0)
        score = 0.0
        for l0, l1 in zip(log_l0, log_l1):
            pi_before = tracker.pi_
            # Blended predictive log-likelihood, mixture of M0/M1 weighted
            # by pi_t *before* seeing this observation (this is the
            # forecast that was actually made).
            m = max(l0, l1)
            mix_ll = m + np.log((1 - pi_before) * np.exp(l0 - m) + pi_before * np.exp(l1 - m))
            score += mix_ll
            tracker.update(l0, l1)
        if score > best_score:
            best_score, best_alpha = score, alpha
    return best_alpha
