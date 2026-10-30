"""
Seed ensembles of trained risk-budget networks.

The plain average of the member networks' budgets, the `mean` method of
experiment/shrunk-ensemble-budgets, computed live from loaded networks instead of
replayed from saved budgets, so it also works for the per-asset regime LSTM, whose
budgets were not saved. Each member's budgets sum to one and respect the 5% floor,
so their average does too.
"""

from __future__ import annotations

from allocation_lstm import _TORCH_AVAILABLE

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn

    class MeanBudgetEnsemble(nn.Module):
        """Stands in for one network in run_walk_forward: for an input window, the
        mean of the members' budgets for that window."""

        def __init__(self, members: list[nn.Module]):
            super().__init__()
            if not members:
                raise ValueError("MeanBudgetEnsemble needs at least one member.")
            self.members = nn.ModuleList(members)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.stack([m(x) for m in self.members]).mean(dim=0)
