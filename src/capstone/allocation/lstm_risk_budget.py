"""
LSTM dynamic risk-budgeting allocator.

Owner: AnnaLisa / Nilay

Adapts the model-based end-to-end risk-budgeting framework of
Uysal, Li & Mulvey (2021), replacing their shallow feed-forward network
with an LSTM -- to capture temporal dependence in the recent evolution
of returns, volatility, trend/correlation dynamics and regime
probabilities, rather than treating each decision date as an
independent snapshot.

Responsibilities (from the M2 problem statement):
- LSTM -> dense -> Softmax -> 10 positive risk budgets (sum to 1).
- Train TWO versions: baseline (market features only) and regime-aware
  (same features + regime probabilities as additional input).
- Loss: Sharpe-based training objective + turnover penalty.
    * Turnover penalty is computed on the FINAL weights coming out of
      the risk-budgeting optimization layer -- never on the raw
      learned budget vector.
    * Sharpe window length is a validation choice, fixed on the
      chronological development/validation period before the final
      OOS test.
- Both versions share the same risk-budgeting optimizer as ERC
  (optimization/risk_budgeting.py), so the ERC vs. LSTM comparison
  isolates the allocation *mechanism* learned, not the portfolio
  construction step.
"""

from __future__ import annotations
import torch
from torch import nn


class RiskBudgetLSTM(nn.Module):
    """LSTM -> dense -> Softmax risk-budget head."""

    def __init__(self, n_features: int, hidden_size: int, n_assets: int = 10):
        super().__init__()
        # TODO: nn.LSTM(...) + nn.Linear(...) + softmax over n_assets
        raise NotImplementedError

    def forward(self, feature_sequence):
        """TODO: return a (batch, n_assets) tensor of positive budgets summing to 1."""
        raise NotImplementedError


def sharpe_turnover_loss(realized_weights_seq, returns_seq, turnover_lambda: float, window: int):
    """
    TODO: -Sharpe(realized portfolio returns over `window`) +
          turnover_lambda * turnover(realized_weights_seq)
    Computed on realized post-optimization weights, per the module docstring.
    """
    raise NotImplementedError


def train_allocator(model: RiskBudgetLSTM, data, regime_aware: bool, **train_kwargs):
    """TODO: training loop respecting chronological/non-anticipative splits."""
    raise NotImplementedError
