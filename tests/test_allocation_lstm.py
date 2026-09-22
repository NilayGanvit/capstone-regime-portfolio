import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_lstm import _TORCH_AVAILABLE, require_torch, budgets_to_weights_batch


def test_require_torch_raises_clear_error_when_absent():
    if _TORCH_AVAILABLE:
        pytest.skip("torch is installed in this environment; nothing to check here")
    with pytest.raises(ImportError):
        require_torch()


def test_budgets_to_weights_batch_works_without_torch():
    """This function only depends on allocation_erc, not on torch, so it
    must work even in this torch-free sandbox."""
    n = 5
    cov = np.eye(n) * 0.02 + 0.005
    budgets = np.array([[0.4, 0.2, 0.2, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]])
    weights = budgets_to_weights_batch(budgets, cov)
    assert weights.shape == budgets.shape
    assert np.allclose(weights.sum(axis=1), 1.0)


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="requires torch, install per requirements.txt")
def test_lstm_forward_pass_shape():
    import torch
    from allocation_lstm import RiskBudgetLSTM

    model = RiskBudgetLSTM(n_features=8, n_assets=10, hidden_size=16)
    x = torch.randn(4, 20, 8)  # batch=4, seq_len=20, n_features=8
    out = model(x)
    assert out.shape == (4, 10)
    assert torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5)  # softmax sums to 1


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="requires torch, install per requirements.txt")
def test_sharpe_turnover_loss_runs():
    import torch
    from allocation_lstm import sharpe_turnover_loss

    T, n = 50, 10
    port_returns = torch.randn(T) * 0.01
    weights_seq = torch.softmax(torch.randn(T, n), dim=-1)
    drifted_seq = torch.softmax(torch.randn(T, n), dim=-1)
    loss = sharpe_turnover_loss(port_returns, weights_seq, drifted_seq, turnover_penalty=0.1)
    assert torch.isfinite(loss)
