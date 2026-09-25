"""
Walk-forward harness: runs the ERC baseline/regime/reliability-blend
configurations chronologically, one trading day at a time, exactly per
the M2 pseudocode's ordering:

    for each trading day t:
        project scheduled targets onto constraints, using drifted holdings
        accrue held-portfolio return through the close
        observe r(t); score r(t) under M0/M1 densities saved at t-1
        update pi from prior model weights and log-likelihoods
        scheduled refit (quarterly by default, expanding window through
          F(t), using only information available through t; state
          continuity maintained via Bhattacharyya distance + Hungarian
          assignment; see regime.py)
        filter HMM states and predict probabilities for day t+1
        save M0/M1 predictive densities for r(t+1)
        if t is a month-end allocation date:
            compute baseline and regime-aware ERC targets
            blend using current pi
            project onto constraints and schedule for next open

ERC (baseline/regime/blend) always runs. Three families of opt-in,
default-off extensions sit on top of it, each a plain parameter on
run_walk_forward so every arm remains directly reproducible and no
existing caller's behavior changes unless it opts in:

  - `lstm_models` + `lstm_feature_matrix` (both required together): wires
    an *already-trained* LSTM allocator (allocation_lstm.py) in as
    lstm_baseline/lstm_regime/lstm_blend, so one harness run produces all
    six M2 evaluation-table configs (+ equal_weight). Training stays
    outside this function -- it must happen once, only on data through
    the initial training window (see scripts/run_real_data.py), never
    mid-walk, so this function only ever does cheap inference
    (allocation_lstm.predict_budgets) against the same baseline_cov/
    regime_cov already computed for ERC that rebalance date. The
    regime variant's trailing feature window needs a *causal* history of
    one-step-ahead regime probabilities; rather than recomputing that
    history retroactively whenever the HMM refits (which would revise
    what "was believed" on past decision dates), this function records
    each day's already-computed `predicted_for_tomorrow` into a running
    cache as the walk proceeds -- a fixed historical record, exactly
    like every other per-day quantity here.
  - `include_hrp`: wires Hierarchical Risk Parity (allocation_hrp.py) in
    as hrp_baseline/hrp_regime/hrp_blend, against the identical pooled_cov/
    smoothed_state_covs/regime_probs_now ERC itself uses -- a secondary
    robustness check (per M2's own scope note) on whether ERC's regime
    effect is an artifact of ERC's particular risk-budgeting construction,
    not a fourth primary allocator.
  - `per_regime_nu`: replaces the M1 mixture density's single shared nu
    (fit_shared_nu) with one nu per HMM state (densities.fit_per_regime_nu)
    at init and at every scheduled refit. M0 is unaffected (it has no
    regime dimension). Also a secondary robustness check, not a new
    config -- it changes the M1 density feeding pi_t, so it can shift
    erc_blend/hrp_blend/lstm_blend without adding new configs.

Scheduled refit is implemented: the HMM, M0/M1 densities, and covariances
are re-estimated on an expanding window with state alignment to maintain
label continuity across refits. Each refit's EM is warm-started from the
previous fit (refit_warm_start, default True) rather than a fresh random
init -- measured on the real ten-ETF universe, independent random inits
landed in a different-but-plausible local optimum most months, causing
erc_regime's target weights/drawdown to whipsaw and making scheduled
refit perform worse than the frozen-parameter baseline. Warm-starting fixes
most of that gap but not all of it; closing the rest took
refit_every_n_rebalances defaulting to 3 (quarterly, not monthly) rather
than smoothing state_covs across refits (measured worse) or aligning to a
fixed reference instead of the previous fit (measured as a no-op once
warm-starting is active). Quarterly + warm-started refit was the only
configuration that matched or beat the no-refit baseline on erc_regime's
Sharpe. All of refit_at_rebalance/refit_warm_start/refit_state_cov_ewma/
refit_align_to_fixed_reference/refit_every_n_rebalances are toggles on
run_walk_forward, so every arm above remains directly reproducible for
later analysis.

Owner: Nilay (ties every other module together; each owner's module
above is used here as originally specified, not reimplemented).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regime import GaussianHMM
from densities import M0PooledStudentT, M1RegimeMixtureStudentT, fit_shared_nu, fit_per_regime_nu
from reliability import ReliabilityTracker, blend_weights
from allocation_erc import erc_baseline_weights, erc_regime_weights, regularize_covariance, risk_contribution_shares
from allocation_hrp import hrp_baseline_weights, hrp_regime_weights
from allocation_lstm import predict_budgets, budgets_to_weights_batch
from constraints import ConstraintSpec, drift_weights, project_onto_constraints, turnover as turnover_fn, binding_constraints


@dataclass
class WalkForwardResult:
    dates: list
    weights: dict           # config_name -> (T, n_assets) array
    portfolio_returns: dict  # config_name -> (T,) array
    pi_history: list
    log_l0_history: list
    log_l1_history: list
    binding_constraints_history: dict  # config_name -> list of {date, lower_bound_binding, upper_bound_binding, turnover_binding, turnover, risk_contribution_pre_constraint_max_dev, risk_contribution_post_constraint_max_dev}, one entry per rebalance
    execution_history: dict  # config_name -> list of {date, turnover}, one entry per rebalance, dated on the day the trade actually executes (== decision date if no open-price data, decision date + 1 trading day otherwise) -- the reference series for transaction-cost accounting
    refit_dates: list       # dates on which a scheduled refit occurred


def ewma_blend(new: np.ndarray, prev: np.ndarray, lam: float) -> np.ndarray:
    """lam*new + (1-lam)*prev -- damps month-to-month swings in a refit
    parameter (state_covs) across scheduled refits, on top of warm-
    starting the EM itself. lam=1 recovers the unsmoothed value."""
    return lam * new + (1.0 - lam) * prev


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
    refit_at_rebalance: bool = True,
    refit_n_iter: int = 50,
    refit_warm_start: bool = True,
    refit_state_cov_ewma: float | None = None,
    refit_align_to_fixed_reference: bool = False,
    refit_every_n_rebalances: int = 3,
    lstm_models: dict | None = None,
    lstm_feature_matrix: pd.DataFrame | None = None,
    lstm_seq_len: int = 60,
    include_hrp: bool = False,
    per_regime_nu: bool = False,
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

    `refit_state_cov_ewma`, if set, EWMA-blends each refit's state_covs
    with the running smoothed value (see ewma_blend) before it reaches
    erc_regime_weights, damping residual month-to-month covariance swings
    on top of warm-starting the EM. None (default) disables smoothing.

    `refit_align_to_fixed_reference`, if True, aligns every refit's HMM
    to the single initial-window fit instead of the immediately preceding
    refit, to bound cumulative label drift over many refits rather than
    only preventing adjacent-refit swaps. False (default) keeps the
    previous-fit-chained alignment.

    `refit_every_n_rebalances`, if > 1, skips the scheduled refit on all
    but every Nth rebalance date (e.g. 3 turns the monthly rebalance
    cadence into a quarterly refit cadence), while ERC targets are still
    recomputed at every rebalance date using whichever HMM/M0/M1/
    covariances were most recently fit. Defaults to 3 (quarterly): on
    the real ten-ETF universe, quarterly refit + warm-starting was the
    only configuration that matched/beat the no-refit baseline on
    erc_regime's Sharpe (fewer refits means fewer chances for the HMM to
    drift, and each refit sees 3x more new data); pass 1 to refit at
    every rebalance date (monthly) instead.

    `lstm_models`, if given, must be {"baseline": trained RiskBudgetLSTM,
    "regime": trained RiskBudgetLSTM} (see allocation_lstm.py) and
    requires `lstm_feature_matrix` (features.build_feature_matrix(returns),
    causal by construction) alongside it -- adds lstm_baseline/lstm_regime/
    lstm_blend configs, doing inference only (allocation_lstm.predict_budgets)
    against a trailing `lstm_seq_len`-day window at each rebalance date; the
    models themselves must already be trained on data through the initial
    window only (see scripts/run_real_data.py) -- this function never
    trains or retrains them. Both `lstm_models` and `lstm_feature_matrix`
    must be given together, or neither.

    `include_hrp`, if True, adds hrp_baseline/hrp_regime/hrp_blend configs
    (allocation_hrp.py) against the same covariances ERC uses -- a
    secondary robustness check (per M2's scope note) on whether ERC's
    regime effect survives under Hierarchical Risk Parity's tree-based
    construction rather than ERC's SLSQP risk-budgeting solve.

    `per_regime_nu`, if True, fits one M1 mixture degrees-of-freedom per
    HMM state (densities.fit_per_regime_nu) instead of one nu shared
    across every state (densities.fit_shared_nu, the default) -- another
    secondary robustness check, changing the M1 density (and hence pi_t
    and every *_blend config) rather than adding new configs. M0 always
    keeps the pooled, shared nu.
    """
    if constraint_spec is None:
        constraint_spec = ConstraintSpec()
    have_open_data = close_to_open_returns is not None and open_to_close_returns is not None

    have_lstm = lstm_models is not None
    if have_lstm != (lstm_feature_matrix is not None):
        raise ValueError("lstm_models and lstm_feature_matrix must be given together, or neither.")

    dates = returns.index
    X = returns.to_numpy()
    n_assets = X.shape[1]
    rebalance_dates = _rebalance_dates(returns)
    if have_open_data:
        CO = close_to_open_returns.reindex(dates).to_numpy()
        OC = open_to_close_returns.reindex(dates).to_numpy()
    if have_lstm:
        lstm_feature_X = lstm_feature_matrix.to_numpy()

    # --- Initialization (fit once on the initial window; see the
    # module-level docstring re: scheduled refit being out of scope for
    # this first pass) ---
    train = X[:initial_window]
    hmm = GaussianHMM(n_states=n_states, random_state=0).fit(train)
    filtered_train = hmm.filtered_probabilities(train)

    std_resid = ((train - train.mean(axis=0)) / train.std(axis=0)).ravel()
    nu = fit_shared_nu(std_resid)  # M0 always uses this, regardless of per_regime_nu
    nu_m1 = fit_per_regime_nu(train, filtered_train) if per_regime_nu else nu
    m0 = M0PooledStudentT(nu=nu).fit(train)
    m1 = M1RegimeMixtureStudentT(nu=nu_m1).fit(train, filtered_train)

    pooled_cov = np.cov(train.T)
    # Convert each state's Student-t scale back to covariance -- per-state
    # nu (per_regime_nu=True) needs a per-state conversion factor instead
    # of the single scalar nu/(nu-2) used when nu is shared.
    nu_m1_arr = np.full(n_states, nu_m1) if np.ndim(nu_m1) == 0 else np.asarray(nu_m1)
    state_covs = np.array(m1.scales_) * (nu_m1_arr / (nu_m1_arr - 2))[:, None, None]
    smoothed_state_covs = state_covs.copy()

    # Frozen snapshot of the initial-window fit, used only when
    # refit_align_to_fixed_reference=True (see docstring above).
    reference_hmm = GaussianHMM(n_states=n_states)
    reference_hmm.means_ = hmm.means_.copy()
    reference_hmm.covars_ = hmm.covars_.copy()
    reference_hmm.transmat_ = hmm.transmat_.copy()
    reference_hmm.startprob_ = hmm.startprob_.copy()
    reference_hmm.n_features_ = hmm.n_features_

    tracker = ReliabilityTracker(alpha=alpha, pi0=0.5)

    configs = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
    if include_hrp:
        configs += ["hrp_baseline", "hrp_regime", "hrp_blend"]
    if have_lstm:
        configs += ["lstm_baseline", "lstm_regime", "lstm_blend"]
        # Causal one-step-ahead regime-probability history for the regime
        # LSTM's trailing feature window, indexed like X/dates (row i =
        # P(S_i+1 | F_i)). Seeded here from the initial-window-only HMM
        # fit (no refit has happened yet, so this is exactly what a
        # decision on any day in [0, initial_window) would have seen);
        # filled in incrementally, one row per day, as the walk proceeds
        # below -- see the module docstring on why this is a running
        # record rather than something recomputed retroactively at refit.
        regime_probs_cache = np.empty((len(dates), n_states))
        regime_probs_cache[:initial_window] = hmm.predicted_probabilities(train)

    weights_out = {c: [] for c in configs}
    port_ret_out = {c: [] for c in configs}
    binding_history = {c: [] for c in configs}
    execution_history = {c: [] for c in configs}
    pi_history, log_l0_history, log_l1_history = [], [], []
    refit_dates = []
    rebalance_count = 0

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
                new_w = pending_execution[c]["target"]
                leg_overnight = 1.0 + float(old_w @ simple_co_t)
                leg_intraday = 1.0 + float(new_w @ simple_oc_t)
                port_ret_out[c].append(leg_overnight * leg_intraday - 1.0)
                prev_weights[c] = drift_weights(new_w, np.exp(OC[abs_idx]))
                weights_out[c].append(prev_weights[c])
                execution_history[c].append({"date": date, "turnover": pending_execution[c]["turnover"]})
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
        if have_lstm:
            regime_probs_cache[abs_idx] = predicted_for_tomorrow

        if date in rebalance_dates:
            rebalance_count += 1
            # Scheduled refit: update HMM/M0/M1/nu/pooled_cov/state_covs
            # using an expanding window through F(t) (today's return included).
            # refit_every_n_rebalances > 1 skips this on all but every Nth
            # rebalance date, stretching the refit cadence beyond monthly
            # while ERC targets still recompute at every rebalance date.
            if refit_at_rebalance and rebalance_count % refit_every_n_rebalances == 0:
                train = X[:abs_idx + 1]
                if refit_warm_start:
                    # Warm-start from the previous fit's params instead of a
                    # fresh random init -- independent random inits land in
                    # a different-but-plausible local optimum most months,
                    # causing regime probabilities to whipsaw and erc_regime's
                    # target weights/drawdown to follow (diagnosed via
                    # month-to-month state_covs Frobenius-change comparison).
                    new_hmm = GaussianHMM(n_states=n_states, n_iter=refit_n_iter).fit(train, init_from=hmm)
                else:
                    refit_seed = 1000 + len(refit_dates)  # kept only for the disabled-warm-start comparison arm
                    new_hmm = GaussianHMM(n_states=n_states, random_state=refit_seed, n_iter=refit_n_iter).fit(train)
                align_target = reference_hmm if refit_align_to_fixed_reference else hmm
                new_hmm.align_to(align_target)
                hmm = new_hmm

                filtered_train = hmm.filtered_probabilities(train)

                std_resid = ((train - train.mean(axis=0)) / train.std(axis=0)).ravel()
                nu = fit_shared_nu(std_resid)
                nu_m1 = fit_per_regime_nu(train, filtered_train) if per_regime_nu else nu
                m0 = M0PooledStudentT(nu=nu).fit(train)
                m1 = M1RegimeMixtureStudentT(nu=nu_m1).fit(train, filtered_train)

                pooled_cov = np.cov(train.T)
                nu_m1_arr = np.full(n_states, nu_m1) if np.ndim(nu_m1) == 0 else np.asarray(nu_m1)
                state_covs = np.array(m1.scales_) * (nu_m1_arr / (nu_m1_arr - 2))[:, None, None]
                if refit_state_cov_ewma is not None:
                    smoothed_state_covs = ewma_blend(state_covs, smoothed_state_covs, refit_state_cov_ewma)
                else:
                    smoothed_state_covs = state_covs

                refit_dates.append(date)

                hist_incl_today = train
                predicted_for_tomorrow = hmm.predicted_probabilities(hist_incl_today)[-1]
                if have_lstm:
                    regime_probs_cache[abs_idx] = predicted_for_tomorrow

            regime_probs_now = predicted_for_tomorrow  # one-step-ahead, using F(t)

            baseline_cov = regularize_covariance(pooled_cov, shrinkage)
            regime_mixture_cov = np.tensordot(regime_probs_now, state_covs, axes=(0, 0))
            regime_cov = regularize_covariance(regime_mixture_cov, shrinkage)
            # Same regularized covariances erc_baseline_weights/erc_regime_weights
            # solve against, recomputed here (cheap, deterministic) only to
            # score risk-contribution conformance below -- not to re-derive
            # the weights themselves.
            covs_for_diagnostic = {"erc_baseline": baseline_cov, "erc_regime": regime_cov}

            w_baseline = erc_baseline_weights(pooled_cov, shrinkage)
            w_regime = erc_regime_weights(smoothed_state_covs, regime_probs_now, shrinkage)
            w_blend_raw = blend_weights(w_regime, w_baseline, pi_t)
            w_equal = np.full(n_assets, 1.0 / n_assets)

            raw_targets = {
                "erc_baseline": w_baseline,
                "erc_regime": w_regime,
                "erc_blend": w_blend_raw,
                "equal_weight": w_equal,
            }

            if include_hrp:
                # Same pooled_cov/smoothed_state_covs/regime_probs_now/pi_t
                # ERC itself uses -- isolates the effect of the
                # risk-parity *construction* (HRP's tree-based bisection
                # vs ERC's SLSQP solve), not the covariance inputs.
                w_hrp_baseline = hrp_baseline_weights(pooled_cov, shrinkage)
                w_hrp_regime = hrp_regime_weights(smoothed_state_covs, regime_probs_now, shrinkage)
                w_hrp_blend = blend_weights(w_hrp_regime, w_hrp_baseline, pi_t)
                raw_targets.update({
                    "hrp_baseline": w_hrp_baseline,
                    "hrp_regime": w_hrp_regime,
                    "hrp_blend": w_hrp_blend,
                })

            if have_lstm:
                # Same trailing-window position for the market-feature and
                # regime-probability slices -- both are indexed by trading
                # day with no gaps, so one .get_loc lookup on the (smaller,
                # dropna'd) feature matrix suffices.
                feat_pos = lstm_feature_matrix.index.get_loc(date)
                baseline_window = lstm_feature_X[feat_pos + 1 - lstm_seq_len: feat_pos + 1]
                regime_window = np.concatenate(
                    [baseline_window, regime_probs_cache[abs_idx + 1 - lstm_seq_len: abs_idx + 1]],
                    axis=1,
                )
                budgets_baseline = predict_budgets(lstm_models["baseline"], baseline_window)
                budgets_regime = predict_budgets(lstm_models["regime"], regime_window)
                # Same baseline_cov/regime_cov ERC's diagnostic already
                # regularized above -- the risk-budgeting layer shared
                # between ERC and the LSTM allocator (M2's architecture).
                w_lstm_baseline = budgets_to_weights_batch(budgets_baseline[None, :], baseline_cov)[0]
                w_lstm_regime = budgets_to_weights_batch(budgets_regime[None, :], regime_cov)[0]
                w_lstm_blend = blend_weights(w_lstm_regime, w_lstm_baseline, pi_t)
                raw_targets.update({
                    "lstm_baseline": w_lstm_baseline,
                    "lstm_regime": w_lstm_regime,
                    "lstm_blend": w_lstm_blend,
                })

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
                turnover_value = turnover_fn(w_final, w_drift)
                binding["turnover"] = turnover_value

                # Risk-contribution conformance (M2/M3: "we may not obtain
                # exactly equal risk contributions... we will therefore
                # also verify the degree of conformance"). Only meaningful
                # for erc_baseline/erc_regime, which are solved against an
                # explicit equal-budget target and a single covariance;
                # erc_blend mixes two different covariance-based solutions
                # and equal_weight was never risk-budgeted, so neither has
                # one well-defined target to score against.
                if c in covs_for_diagnostic:
                    target_share = 1.0 / n_assets
                    cov_c = covs_for_diagnostic[c]
                    pre_dev = risk_contribution_shares(raw_targets[c], cov_c) - target_share
                    post_dev = risk_contribution_shares(w_final, cov_c) - target_share
                    binding["risk_contribution_pre_constraint_max_dev"] = float(np.max(np.abs(pre_dev)))
                    binding["risk_contribution_post_constraint_max_dev"] = float(np.max(np.abs(post_dev)))
                else:
                    binding["risk_contribution_pre_constraint_max_dev"] = None
                    binding["risk_contribution_post_constraint_max_dev"] = None

                binding_history[c].append(binding)
                if have_open_data:
                    # Execute at tomorrow's open (see the split-leg handling
                    # above): prev_weights becomes today's pre-trade holding
                    # (still earning tonight's overnight leg), and w_final
                    # only takes effect once that leg has been scored.
                    prev_weights[c] = w_drift
                    pending_execution[c] = {"target": w_final, "turnover": turnover_value}
                else:
                    prev_weights[c] = w_final
                    execution_history[c].append({"date": date, "turnover": turnover_value})
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
        execution_history=execution_history,
        refit_dates=refit_dates,
    )
