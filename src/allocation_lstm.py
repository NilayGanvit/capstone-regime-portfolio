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

Owner: AnnaLisa & Nilay (traditional and ML allocation).
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where torch is absent
    _TORCH_AVAILABLE = False

import numpy as np

from allocation_erc import solve_risk_budget


def require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "allocation_lstm requires PyTorch. Install it in your own "
            "development environment (`pip install torch`, see "
            "requirements.txt) -- it is intentionally not available in "
            "this shared sandbox."
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
) -> "torch.Tensor":
    """Loss = -Sharpe(portfolio_returns) + turnover_penalty * mean(turnover),
    where turnover is computed on the *final* (post risk-budgeting-layer)
    weights against each period's drifted holdings -- per M2's explicit
    requirement, not on the raw learned budget vector.

    portfolio_returns : (T,) realized portfolio returns over the training
        window (already net of the weights actually used each period).
    weights_sequence, drifted_weights_sequence : (T, n_assets).
    """
    require_torch()
    mean_r = portfolio_returns.mean()
    std_r = portfolio_returns.std() + 1e-8
    sharpe = mean_r / std_r * (252 ** 0.5)
    turnover = (weights_sequence - drifted_weights_sequence).abs().sum(dim=-1).mean()
    return -sharpe + turnover_penalty * turnover


def budgets_to_weights_batch(budgets_batch: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Apply the shared risk-budgeting optimization layer (from
    allocation_erc.py) to a batch of learned budget vectors against a
    given covariance estimate. This is the bridge between the LSTM's
    softmax output (a differentiable operation, for training) and the
    non-differentiable convex solve used for backtesting/execution -- the
    same pattern Uysal et al. use, where gradients flow through a
    differentiable approximation during training while backtest weights
    use the exact solve. Implementing the differentiable optimization
    layer itself (e.g. via an implicit-differentiation package such as
    cvxpylayers) is flagged in the README as follow-on work -- this
    function currently performs the exact (non-differentiable) solve,
    suitable for evaluation/backtesting but not yet for end-to-end
    training through the optimization layer.
    """
    return np.array([solve_risk_budget(cov, target_budgets=b) for b in budgets_batch])
