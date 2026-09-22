"""
Walk-forward harness: runs the ERC baseline/regime/reliability-blend
configurations chronologically, one trading day at a time, exactly per
the M2 pseudocode's ordering:

    for each trading day t:
        project scheduled targets onto constraints, using drifted holdings
        accrue held-portfolio return through the close
        observe r(t); score r(t) under M0/M1 densities saved at t-1
        update pi from prior model weights and log-likelihoods
        (scheduled refit, using F(t) only, omitted from this first pass --
         see README "Known scope limitations")
        filter HMM states and predict probabilities for day t+1
        save M0/M1 predictive densities for r(t+1)
        if t is a month-end allocation date:
            compute baseline and regime-aware ERC targets
            blend using current pi
            project onto constraints and schedule for next open

This first pass wires ERC only (no LSTM -- that requires the torch
module, not available in this sandbox) and does not yet implement the
scheduled re-fit step, so parameters are fit once on an initial window
and held fixed through the walk-forward loop. Both are named explicitly
in the README as the next things to build, not silently skipped.

Owner: Nilay (ties every other module together; each owner's module
above is used here as originally specified, not reimplemented).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regime import GaussianHMM
from densities import M0PooledStudentT, M1RegimeMixtureStudentT, fit_shared_nu
from reliability import ReliabilityTracker, blend_weights
from allocation_erc import erc_baseline_weights, erc_regime_weights, regularize_covariance
from constraints import ConstraintSpec, drift_weights, project_onto_constraints, turnover as turnover_fn


@dataclass
class WalkForwardResult:
    dates: list
    weights: dict           # config_name -> (T, n_assets) array
    portfolio_returns: dict  # config_name -> (T,) array
    pi_history: list
    log_l0_history: list
    log_l1_history: list


def _rebalance_dates(returns: pd.DataFrame) -> set:
    """Month-end trading days (last trading day observed in each
    calendar month), used as the monthly allocation decision dates."""
    month_end_idx = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda g: g.index[-1]
    )
    return set(month_end_idx.values)


def run_walk_forward(
    returns: pd.DataFrame,
    n_states: int = 3,
    initial_window: int = 252,
    alpha: float = 0.97,
    shrinkage: float = 0.10,
    constraint_spec: ConstraintSpec | None = None,
) -> WalkForwardResult:
    """Run the ERC baseline / regime / reliability-blend configurations
    walk-forward, plus the fixed equal-weight benchmark, over `returns`
    (T x n_assets, chronologically ordered log returns).
    """
    if constraint_spec is None:
        constraint_spec = ConstraintSpec()

    dates = returns.index
    X = returns.to_numpy()
    n_assets = X.shape[1]
    rebalance_dates = _rebalance_dates(returns)

    # --- Initialization (fit once on the initial window; see the
    # module-level docstring re: scheduled refit being out of scope for
    # this first pass) ---
    train = X[:initial_window]
    hmm = GaussianHMM(n_states=n_states, random_state=0).fit(train)
    filtered_train = hmm.filtered_probabilities(train)

    std_resid = ((train - train.mean(axis=0)) / train.std(axis=0)).ravel()
    nu = fit_shared_nu(std_resid)
    m0 = M0PooledStudentT(nu=nu).fit(train)
    m1 = M1RegimeMixtureStudentT(nu=nu).fit(train, filtered_train)

    pooled_cov = regularize_covariance(np.cov(train.T), shrinkage)
    state_covs = np.array(m1.scales_) * nu / (nu - 2)  # convert Student-t scale back to covariance

    tracker = ReliabilityTracker(alpha=alpha, pi0=0.5)

    configs = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
    weights_out = {c: [] for c in configs}
    port_ret_out = {c: [] for c in configs}
    pi_history, log_l0_history, log_l1_history = [], [], []

    prev_weights = {c: np.full(n_assets, 1.0 / n_assets) for c in configs}
    walk_dates = dates[initial_window:]

    for t_idx, date in enumerate(walk_dates):
        abs_idx = initial_window + t_idx
        r_t = X[abs_idx]  # realized return for this day, F(t)

        # Held-portfolio return accrual, using *previous* period's final weights
        for c in configs:
            port_ret_out[c].append(float(prev_weights[c] @ r_t))

        # Score r_t under M0/M1 densities computed using only information
        # through t-1 (filtered probs from history up to abs_idx, i.e.
        # *not* including r_t itself), then predict forward for t+1.
        hist = X[:abs_idx]  # everything strictly before today's return
        filtered_hist = hmm.filtered_probabilities(hist)[-1]
        predicted_for_today = filtered_hist @ hmm.transmat_  # this was "yesterday's forecast for today"

        log_l0 = m0.log_density(r_t)
        log_l1 = m1.log_density(r_t, predicted_for_today)
        pi_t = tracker.update(log_l0, log_l1)
        pi_history.append(pi_t)
        log_l0_history.append(log_l0)
        log_l1_history.append(log_l1)

        # Filter through today's observation (now realized), predict for
        # tomorrow -- P(S_t+1 | F_t). This is the freshest regime signal
        # available at the point the rebalance decision is made (today,
        # for execution at tomorrow's open), unlike predicted_for_today
        # (P(S_t | F_t-1)) above, which is one day stale by the time a
        # same-day rebalance decision is taken.
        hist_incl_today = X[: abs_idx + 1]
        predicted_for_tomorrow = hmm.predicted_probabilities(hist_incl_today)[-1]

        if date in rebalance_dates:
            regime_probs_now = predicted_for_tomorrow  # one-step-ahead, using F(t)

            w_baseline = erc_baseline_weights(pooled_cov, shrinkage)
            w_regime = erc_regime_weights(state_covs, regime_probs_now, shrinkage)
            w_blend_raw = blend_weights(w_regime, w_baseline, pi_t)
            w_equal = np.full(n_assets, 1.0 / n_assets)

            raw_targets = {
                "erc_baseline": w_baseline,
                "erc_regime": w_regime,
                "erc_blend": w_blend_raw,
                "equal_weight": w_equal,
            }
            for c in configs:
                w_drift = drift_weights(prev_weights[c], np.exp(r_t))
                w_final = project_onto_constraints(raw_targets[c], w_drift, constraint_spec)
                prev_weights[c] = w_final
                weights_out[c].append(w_final)
        else:
            for c in configs:
                w_drift = drift_weights(prev_weights[c], np.exp(r_t))
                prev_weights[c] = w_drift
                weights_out[c].append(w_drift)

    return WalkForwardResult(
        dates=list(walk_dates),
        weights={c: np.array(v) for c, v in weights_out.items()},
        portfolio_returns={c: np.array(v) for c, v in port_ret_out.items()},
        pi_history=pi_history,
        log_l0_history=log_l0_history,
        log_l1_history=log_l1_history,
    )
