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
from allocation_erc import erc_baseline_weights, erc_regime_weights
from constraints import ConstraintSpec, drift_weights, project_onto_constraints, turnover as turnover_fn, binding_constraints


@dataclass
class WalkForwardResult:
    dates: list
    weights: dict           # config_name -> (T, n_assets) array
    portfolio_returns: dict  # config_name -> (T,) array
    pi_history: list
    log_l0_history: list
    log_l1_history: list
    binding_constraints_history: dict  # config_name -> list of {date, lower_bound_binding, upper_bound_binding, turnover_binding, turnover}, one entry per rebalance


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
    close_to_open_returns: pd.DataFrame | None = None,
    open_to_close_returns: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run the ERC baseline / regime / reliability-blend configurations
    walk-forward, plus the fixed equal-weight benchmark, over `returns`
    (T x n_assets, chronologically ordered log returns).

    `close_to_open_returns`/`open_to_close_returns` (both required together,
    same index as `returns`; see data.PriceDataset) enable M2's "executed
    at the opening of the following trading session" timing: a rebalance
    decided using information through close(t) has its *return* effect
    split so the old target keeps earning the close(t)->open(t+1) overnight
    leg and only the new target earns the open(t+1)->close(t+1) leg. Each
    rebalance's own turnover/binding constraints are computed on the
    decision itself (w_drift -> w_final, per M2), not on when it executes;
    a *later* rebalance's turnover can still shift slightly, though, since
    the position it drifts from has itself followed a different path once
    execution timing changes. If either is
    None, the harness falls back to same-close execution (the new target
    earns the whole close(t)->close(t+1) return), which is what every
    caller without open-price data gets.
    """
    if constraint_spec is None:
        constraint_spec = ConstraintSpec()
    have_open_data = close_to_open_returns is not None and open_to_close_returns is not None

    dates = returns.index
    X = returns.to_numpy()
    n_assets = X.shape[1]
    rebalance_dates = _rebalance_dates(returns)
    if have_open_data:
        CO = close_to_open_returns.reindex(dates).to_numpy()
        OC = open_to_close_returns.reindex(dates).to_numpy()

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

    pooled_cov = np.cov(train.T)
    state_covs = np.array(m1.scales_) * nu / (nu - 2)  # convert Student-t scale back to covariance

    tracker = ReliabilityTracker(alpha=alpha, pi0=0.5)

    configs = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
    weights_out = {c: [] for c in configs}
    port_ret_out = {c: [] for c in configs}
    binding_history = {c: [] for c in configs}
    pi_history, log_l0_history, log_l1_history = [], [], []

    prev_weights = {c: np.full(n_assets, 1.0 / n_assets) for c in configs}
    # Set on a rebalance date (to the target the decision produced) and
    # consumed the very next trading day, when have_open_data is True --
    # see the docstring's note on split-leg execution timing.
    pending_execution = {c: None for c in configs}
    walk_dates = dates[initial_window:]

    for t_idx, date in enumerate(walk_dates):
        abs_idx = initial_window + t_idx
        r_t = X[abs_idx]  # realized log return for this day, F(t)

        # Per-asset simple return, since log returns aren't linear across
        # assets: the portfolio return has to be the weighted sum of simple
        # returns, not of log returns (evaluation.py compounds this series
        # via cumprod(1+r), which is only valid for simple returns).
        simple_r_t = np.exp(r_t) - 1.0

        executed_today = have_open_data and any(pending_execution[c] is not None for c in configs)
        if executed_today:
            # Today executes yesterday's rebalance decision at the open:
            # the old target earns the overnight close(t-1)->open(t) leg,
            # the new target earns the open(t)->close(t) leg. Held-return
            # accrual and the end-of-day holding are both derived from this
            # split rather than from simple_r_t directly.
            simple_co_t = np.exp(CO[abs_idx]) - 1.0
            simple_oc_t = np.exp(OC[abs_idx]) - 1.0
            for c in configs:
                old_w = prev_weights[c]
                new_w = pending_execution[c]
                leg_overnight = 1.0 + float(old_w @ simple_co_t)
                leg_intraday = 1.0 + float(new_w @ simple_oc_t)
                port_ret_out[c].append(leg_overnight * leg_intraday - 1.0)
                prev_weights[c] = drift_weights(new_w, np.exp(OC[abs_idx]))
                weights_out[c].append(prev_weights[c])
                pending_execution[c] = None
        else:
            # Held-portfolio return accrual, using *previous* period's final weights
            for c in configs:
                port_ret_out[c].append(float(prev_weights[c] @ simple_r_t))

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
                # Turnover/binding constraints are defined on the decision
                # itself (w_drift -> w_final), independent of when it
                # executes, per M2's "turnover will be calculated based on
                # the actual weights at the time of rebalancing."
                weights_out[c].append(w_final)
                binding = binding_constraints(w_final, w_drift, constraint_spec)
                binding["date"] = date
                binding["turnover"] = turnover_fn(w_final, w_drift)
                binding_history[c].append(binding)
                if have_open_data:
                    # Execute at tomorrow's open (see the split-leg handling
                    # above): prev_weights becomes today's pre-trade holding
                    # (still earning tonight's overnight leg), and w_final
                    # only takes effect once that leg has been scored.
                    prev_weights[c] = w_drift
                    pending_execution[c] = w_final
                else:
                    prev_weights[c] = w_final
        elif not executed_today:
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
        binding_constraints_history=binding_history,
    )
