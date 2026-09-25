"""
LSTM dynamic risk-budgeting allocator.

    features [+ regime probabilities] -> LSTM -> dense -> softmax
        -> ten-element risk-budget vector -> shared risk-budgeting
        optimization layer (allocation_erc.solve_risk_budget) -> weights

This is the adaptation of Uysal, Li & Mulvey's (2021) model-based
end-to-end risk-budgeting framework, replacing their shallow feed-forward
network with an LSTM so the allocator can learn temporal dependence in
the recent evolution of returns, volatility, trend/correlation dynamics
and regime probabilities, rather than treating each decision-date feature
vector as an independent snapshot (the justification the group agreed on
for M3).

Two variants share this exact class, differing only in whether the input
feature tensor includes the regime-probability columns:
    baseline : market features only
    regime   : market features + regime probabilities

Training objective (M2): Sharpe-based loss with a turnover penalty
computed on the *final* weights produced by the risk-budgeting layer,
not on the raw learned budget vector -- consistent with the turnover
definition used for the portfolio constraints (constraints.py).

Requires PyTorch, which is NOT installed in this development/testing
sandbox (no network access to install it here). The import is guarded so
the rest of the package -- and the smoke test in scripts/run_smoke_test.py
-- still runs fully without this module. Each group member's own
development environment should `pip install torch` per requirements.txt.

End-to-end training (M2's explicit requirement, and the README's
previously-open item) additionally requires `cvxpylayers`, also guarded
the same way. Gradients flow through `differentiable_risk_budget_layer`
below -- a convex reformulation of risk budgeting (Spinu, 2013; Bai,
Scheinberg & Tutuncu, 2016), the same one Uysal, Li & Mulvey (2021)
build their differentiable layer on -- while `budgets_to_weights_batch`
above continues to produce the actual backtest/evaluation weights via
the exact (non-differentiable) SLSQP solve in allocation_erc.py. See
that function's docstring for why the two are allowed to disagree
slightly and, on the real universe, for a discovered gap in the exact
solver's own local optimum.

Owner: AnnaLisa & Nilay (traditional and ML allocation).
"""

from __future__ import annotations

from functools import lru_cache

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where torch is absent
    _TORCH_AVAILABLE = False

try:
    import cvxpy as cp
    from cvxpylayers.torch import CvxpyLayer
    _CVXPYLAYERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where cvxpylayers is absent
    _CVXPYLAYERS_AVAILABLE = False

import numpy as np
import pandas as pd

from allocation_erc import solve_risk_budget, regularize_covariance


def require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "allocation_lstm requires PyTorch. Install it in your own "
            "development environment (`pip install torch`, see "
            "requirements.txt) -- it is intentionally not available in "
            "this shared sandbox."
        )


def require_cvxpylayers() -> None:
    if not _CVXPYLAYERS_AVAILABLE:
        raise ImportError(
            "End-to-end LSTM training requires cvxpylayers (and cvxpy). "
            "Install via `pip install cvxpylayers` -- see requirements.txt. "
            "Evaluation/backtesting (budgets_to_weights_batch) does not "
            "need it."
        )


if _TORCH_AVAILABLE:

    class RiskBudgetLSTM(nn.Module):
        """features (T, n_features) -> softmax risk budgets (n_assets,).

        Consumes the *last* hidden state of the LSTM (i.e. one risk-budget
        vector per decision date, using the trailing sequence of feature
        vectors up to and including that date) -- matching the monthly
        decision cadence in the walk-forward harness.
        """

        def __init__(self, n_features: int, n_assets: int, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.dense = nn.Linear(hidden_size, n_assets)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """x: (batch, seq_len, n_features) -> (batch, n_assets) softmax budgets."""
            _, (h_n, _) = self.lstm(x)
            last_hidden = h_n[-1]  # (batch, hidden_size), final layer's hidden state
            logits = self.dense(last_hidden)
            return torch.softmax(logits, dim=-1)


def sharpe_turnover_loss(
    portfolio_returns: "torch.Tensor",
    weights_sequence: "torch.Tensor",
    drifted_weights_sequence: "torch.Tensor",
    turnover_penalty: float,
    periods_per_year: float = 252.0,
) -> "torch.Tensor":
    """Loss = -Sharpe(portfolio_returns) + turnover_penalty * mean(turnover),
    where turnover is computed on the *final* (post risk-budgeting-layer)
    weights against each period's drifted holdings -- per M2's explicit
    requirement, not on the raw learned budget vector.

    portfolio_returns : (T,) realized portfolio returns over the training
        window (already net of the weights actually used each period).
    weights_sequence, drifted_weights_sequence : (T, n_assets).
    periods_per_year : annualization factor for the Sharpe term, sqrt(252)
        by default for a daily-return training window. `train_lstm_allocator`
        below trains on one period per *monthly* rebalance, not one per
        trading day, and passes periods_per_year=12 accordingly -- pass
        the value matching whatever cadence `portfolio_returns` is actually
        sampled at, rather than relying on the daily default silently.
    """
    require_torch()
    mean_r = portfolio_returns.mean()
    std_r = portfolio_returns.std() + 1e-8
    sharpe = mean_r / std_r * (periods_per_year ** 0.5)
    turnover = (weights_sequence - drifted_weights_sequence).abs().sum(dim=-1).mean()
    return -sharpe + turnover_penalty * turnover


def predict_budgets(model: "RiskBudgetLSTM", feature_window: np.ndarray) -> np.ndarray:
    """Eval-mode forward pass for walk-forward *inference*: a single
    (seq_len, n_features) trailing feature window -> (n_assets,) softmax
    risk budgets, as a plain numpy array. No gradients, one window at a
    time (walk-forward calls this once per rebalance date, not batched).

    `model` is expected to already be trained (via train_lstm_allocator,
    which leaves it in float64 via `.double()`) -- training itself stays
    outside walkforward.run_walk_forward; see that module's docstring for
    why (chronological discipline: the model must be fit only on data
    through the initial training window, never retrained mid-walk).
    """
    require_torch()
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np.asarray(feature_window)[None, :, :], dtype=torch.float64)
        budgets = model(x)
    return budgets[0].numpy()


def budgets_to_weights_batch(budgets_batch: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Apply the shared risk-budgeting optimization layer (from
    allocation_erc.py) to a batch of learned budget vectors against a
    given covariance estimate. This is the bridge between the LSTM's
    softmax output (a differentiable operation, for training) and the
    non-differentiable convex solve used for backtesting/execution -- the
    same pattern Uysal et al. use, where gradients flow through a
    differentiable approximation during training (see
    `differentiable_risk_budget_layer` below) while backtest weights use
    this exact solve.

    NOTE on a discovered gap between the two: solve_risk_budget's SLSQP
    objective (squared error between risk-contribution *shares* and the
    target budgets) is non-convex in w, and on the real ten-ETF
    covariance it was observed (while building the differentiable layer
    below) to converge to a local optimum with some assets pinned at
    zero weight despite a strictly positive target budget -- e.g. plain
    equal-budget ERC on the real initial-window covariance lands on
    [0.11, 0.10, 0.06, 0.00, 0.00, 0.00, 0.32, 0.16, 0.16, 0.09] rather
    than the near-equal-risk-share allocation the convex layer recovers
    exactly for the same inputs. This is a pre-existing property of this
    function (present before this training-loop work; it is what ERC's
    walk-forward results are already built on), not something introduced
    here, and is out of scope to fix in this pass -- flagged here because
    it is exactly what motivated preferring the convex log-barrier
    reformulation over an SLSQP-based differentiable layer.
    """
    return np.array([solve_risk_budget(cov, target_budgets=b) for b in budgets_batch])


@lru_cache(maxsize=None)
def differentiable_risk_budget_layer(n_assets: int) -> "CvxpyLayer":
    """The differentiable convex risk-budgeting layer used only for LSTM
    *training* gradients -- `budgets_to_weights_batch` above (via
    allocation_erc.solve_risk_budget's exact SLSQP solve) is still what
    produces the weights used for backtesting/evaluation.

    Reformulates risk budgeting as the convex problem

        minimize_w   0.5 * w^T Sigma w  -  sum_i b_i * log(w_i)
        subject to   w >= eps

    (Spinu, 2013; Bai, Scheinberg & Tutuncu, 2016 -- the same convex
    construction Uysal, Li & Mulvey, 2021 build their end-to-end
    risk-budgeting layer on). Its first-order condition is
    w_i * (Sigma @ w)_i = b_i for every i: the risk contribution of
    asset i, in the *raw* (un-normalized) solution, equals its target
    budget exactly. Because both sides of that condition scale
    identically with w, the raw solution can be normalized to sum to 1
    afterward (done in `budgets_to_weights_differentiable` below) without
    disturbing the risk-budget shares -- this is what pins the solution's
    scale, so no `sum(w) == 1` constraint is added to the problem itself.
    Adding one was tried and measured to break the construction here:
    CLARABEL and SCS both then converge to a spurious point (observed:
    the constrained "solution" was numerically just the budget vector
    itself, unrelated to Sigma, and a local perturbation search found
    strictly better nearby points -- i.e. not actually optimal, a solver
    conditioning issue specific to combining the log barrier with a
    linear equality constraint here).

    No per-asset upper bound is enforced inside this layer either
    (adding the 0.60 cap alongside the log barrier was, likewise, found
    to push CLARABEL/SCS to every weight sitting exactly at the cap --
    again not the true constrained optimum, verified by direct objective
    comparison). The 0.60 cap and the turnover limit remain enforced
    exactly as for ERC, only downstream of the exact SLSQP solve, via
    constraints.project_onto_constraints at evaluation/backtest time.

    Parameters (batchable over a leading dimension):
        Lt : transpose of the Cholesky factor of Sigma (Sigma = L @ L.T),
             shape (n_assets, n_assets) or (batch, n_assets, n_assets).
             Passed as L^T rather than L or Sigma directly because
             cp.sum_squares(Lt @ w) is DPP-compliant (parameter-affine
             times variable-affine) whereas cp.quad_form(w, Sigma) with
             Sigma itself as a Parameter is not.
        b  : positive budgets (need not sum to 1 -- only their relative
             sizes matter, since the layer's own scale is unconstrained),
             shape (n_assets,) or (batch, n_assets).
    Variable:
        w : the raw (not yet normalized to sum to 1) risk-budgeted
            weights, w >= eps, same leading shape as b.
    """
    require_cvxpylayers()
    w = cp.Variable(n_assets)
    Lt_param = cp.Parameter((n_assets, n_assets))
    b_param = cp.Parameter(n_assets, nonneg=True)
    objective = cp.Minimize(0.5 * cp.sum_squares(Lt_param @ w) - b_param @ cp.log(w))
    constraints = [w >= 1e-6]
    problem = cp.Problem(objective, constraints)
    return CvxpyLayer(problem, parameters=[Lt_param, b_param], variables=[w])


def budgets_to_weights_differentiable(
    budgets: "torch.Tensor",
    cov: "torch.Tensor | np.ndarray",
    layer: "CvxpyLayer | None" = None,
) -> "torch.Tensor":
    """Differentiable counterpart to `budgets_to_weights_batch`, via
    `differentiable_risk_budget_layer`. Hides the Cholesky-transpose
    detail that layer's parameters require.

    budgets : (..., n_assets), positive, gradients flow through it (the
        LSTM's softmax output).
    cov : (..., n_assets, n_assets), broadcastable against budgets'
        leading shape; does not need gradients (it is an input estimate,
        not a learned quantity).
    Returns weights normalized to sum to 1 along the last axis, same
    leading shape as budgets.
    """
    require_torch()
    require_cvxpylayers()
    n_assets = budgets.shape[-1]
    if layer is None:
        layer = differentiable_risk_budget_layer(n_assets)
    cov_t = cov if torch.is_tensor(cov) else torch.as_tensor(np.asarray(cov), dtype=budgets.dtype)
    Lt = torch.linalg.cholesky(cov_t).transpose(-2, -1)
    raw_w, = layer(Lt, budgets)
    return raw_w / raw_w.sum(dim=-1, keepdim=True)


def build_lstm_training_windows(
    feature_matrix: pd.DataFrame,
    returns: pd.DataFrame,
    seq_len: int = 60,
    cov_window: int = 252,
) -> tuple[list, list, list, list]:
    """Assemble monthly-cadence training examples for the LSTM allocator,
    at the same rebalance dates walkforward.run_walk_forward uses (see
    walkforward._rebalance_dates), respecting the same point-in-time rule
    as the rest of the codebase: everything attached to decision date t
    uses only information available through the close of t.

    For each rebalance date with enough trailing history and at least one
    further rebalance date ahead (needed for the forward holding-period
    return), returns four aligned lists:

        feature_windows[i]  : (seq_len, n_features) trailing daily feature
            vectors ending at decision date t_i -- the LSTM's input for
            that decision. Whether this includes regime-probability
            columns (the baseline vs. regime LSTM variant) is the
            caller's choice of what to pass as `feature_matrix`; this
            function is agnostic to that.
        covariances[i]      : (n_assets, n_assets) sample covariance over
            the trailing `cov_window` daily log returns ending at t_i --
            one shared covariance-estimation rule for every LSTM variant,
            mirroring erc_baseline_weights' pooled_cov (regularized the
            same way, in train_lstm_allocator, not here) so that any
            baseline-vs-regime comparison isolates the effect of the
            *input features*, not a difference in covariance machinery.
        fwd_gross_returns[i] : (n_assets,) per-asset gross return
            (product of daily (1+r), computed via exp(sum(log returns)))
            from the day after t_i through the next rebalance date --
            the realized holding-period return that scores the weights
            chosen at t_i.
        dates[i]             : t_i itself.

    This is the simplest correct point-in-time construction -- one
    pooled covariance and one compounded holding-period return per
    rebalance -- not a re-implementation of run_walk_forward's day-by-day
    loop (its scheduled HMM refit, state-mixture covariance, and
    reliability blending). Wiring a trained LSTM into that harness as
    additional configs is follow-on work, flagged in the README.
    """
    from walkforward import _rebalance_dates

    rebalance_dates = sorted(_rebalance_dates(returns))
    feature_dates = feature_matrix.index
    ret_dates = returns.index
    X = feature_matrix.to_numpy()
    R = returns.to_numpy()

    feature_windows, covariances, fwd_gross_returns, out_dates = [], [], [], []

    for i, t in enumerate(rebalance_dates[:-1]):
        if t not in feature_dates or t not in ret_dates:
            continue
        f_idx = feature_dates.get_loc(t)
        r_idx = ret_dates.get_loc(t)
        if f_idx + 1 < seq_len or r_idx + 1 < cov_window:
            continue

        next_t = rebalance_dates[i + 1]
        if next_t not in ret_dates:
            continue
        next_r_idx = ret_dates.get_loc(next_t)
        if next_r_idx <= r_idx:
            continue

        feature_windows.append(X[f_idx + 1 - seq_len: f_idx + 1])
        covariances.append(np.cov(R[r_idx + 1 - cov_window: r_idx + 1].T))
        fwd_gross_returns.append(np.exp(R[r_idx + 1: next_r_idx + 1].sum(axis=0)))
        out_dates.append(t)

    return feature_windows, covariances, fwd_gross_returns, out_dates


def train_lstm_allocator(
    model: "RiskBudgetLSTM",
    feature_windows: list,
    covariances: list,
    fwd_gross_returns: list,
    n_epochs: int = 200,
    lr: float = 1e-3,
    turnover_penalty: float = 0.1,
    shrinkage: float = 0.10,
) -> dict:
    """Train `model` end-to-end through `differentiable_risk_budget_layer`,
    on the monthly-cadence windows from `build_lstm_training_windows`.

    Loss is `sharpe_turnover_loss` with periods_per_year=12: the windows
    here are one per monthly rebalance, not one per trading day, so the
    Sharpe term must annualize by sqrt(12), not the loss function's daily
    default of sqrt(252) -- passing this explicitly is the one required
    adjustment to call that (already-tested, unmodified-in-behavior-by-
    default) function correctly at this cadence.

    Turnover is scored against each period's own previous-period target
    weight, drifted forward by that period's realized return -- exactly
    constraints.drift_weights's definition, reimplemented batched in
    torch and detached (it is a reference point for the penalty, not
    something gradients should flow through). The first period drifts
    from the equal-weight starting point walkforward.py itself uses
    before any rebalance has occurred.

    Covariances are regularized identically to allocation_erc's baseline
    (`regularize_covariance`, same default shrinkage), for the same
    "same regularization rule everywhere" reason as ERC.

    Returns {"model": model, "loss_history": [...]}. Does not touch
    run_walk_forward or persist anything to disk -- the caller decides
    what to do with the trained model.
    """
    require_torch()
    require_cvxpylayers()

    n_assets = fwd_gross_returns[0].shape[0]
    layer = differentiable_risk_budget_layer(n_assets)

    model = model.double()
    X = torch.tensor(np.stack(feature_windows), dtype=torch.float64)
    covs_reg = np.stack([regularize_covariance(c, shrinkage) for c in covariances])
    fwd = torch.tensor(np.stack(fwd_gross_returns), dtype=torch.float64)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    equal_weight = torch.full((1, n_assets), 1.0 / n_assets, dtype=torch.float64)
    ones_row = torch.ones((1, n_assets), dtype=torch.float64)

    loss_history = []
    for _ in range(n_epochs):
        optimizer.zero_grad()
        budgets = model(X)
        weights = budgets_to_weights_differentiable(budgets, covs_reg, layer=layer)

        weights_prev = torch.cat([equal_weight, weights[:-1].detach()], dim=0)
        fwd_prev = torch.cat([ones_row, fwd[:-1]], dim=0)
        drifted_value = weights_prev * fwd_prev
        drifted = drifted_value / drifted_value.sum(dim=-1, keepdim=True)

        port_returns = (weights * (fwd - 1.0)).sum(dim=-1)
        loss = sharpe_turnover_loss(port_returns, weights, drifted, turnover_penalty, periods_per_year=12.0)

        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return {"model": model, "loss_history": loss_history}
