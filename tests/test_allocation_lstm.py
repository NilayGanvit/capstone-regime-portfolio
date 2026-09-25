import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_lstm import (
    _TORCH_AVAILABLE, _CVXPYLAYERS_AVAILABLE, require_torch, require_cvxpylayers,
    budgets_to_weights_batch,
)
from allocation_erc import risk_contributions


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
def test_predict_budgets_shape_and_softmax_sum():
    from allocation_lstm import RiskBudgetLSTM, predict_budgets
    import numpy as np

    model = RiskBudgetLSTM(n_features=8, n_assets=10, hidden_size=16).double()
    window = np.random.default_rng(0).normal(size=(20, 8))
    budgets = predict_budgets(model, window)
    assert budgets.shape == (10,)
    assert np.isclose(budgets.sum(), 1.0, atol=1e-6)
    assert (budgets >= 0).all()


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


def test_require_cvxpylayers_raises_clear_error_when_absent():
    if _CVXPYLAYERS_AVAILABLE:
        pytest.skip("cvxpylayers is installed in this environment; nothing to check here")
    with pytest.raises(ImportError):
        require_cvxpylayers()


_DIFF_LAYER_AVAILABLE = _TORCH_AVAILABLE and _CVXPYLAYERS_AVAILABLE
_diff_layer_skip = pytest.mark.skipif(
    not _DIFF_LAYER_AVAILABLE, reason="requires torch and cvxpylayers, install per requirements.txt"
)


@_diff_layer_skip
def test_differentiable_layer_matches_target_risk_budget_shares():
    """The convex log-barrier layer's first-order condition guarantees its
    (normalized) output's risk-contribution shares equal the input budgets
    exactly -- this is the whole reason it was chosen over an SLSQP-based
    differentiable proxy (see differentiable_risk_budget_layer's
    docstring)."""
    import torch
    from allocation_lstm import budgets_to_weights_differentiable

    rng = np.random.default_rng(0)
    n = 6
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + np.eye(n) * 0.05  # well-conditioned, mildly correlated
    budgets = rng.dirichlet(np.ones(n))

    b_t = torch.tensor(budgets, dtype=torch.float64, requires_grad=True)
    weights = budgets_to_weights_differentiable(b_t, cov)
    w_np = weights.detach().numpy()

    assert np.isclose(w_np.sum(), 1.0, atol=1e-6)
    assert (w_np > 0).all()
    pv = w_np @ cov @ w_np
    rc_share = risk_contributions(w_np, cov) / pv
    assert np.allclose(rc_share, budgets, atol=1e-3)


@_diff_layer_skip
def test_differentiable_layer_gradient_flows_to_budgets():
    """Uses an asymmetric covariance/budget pair -- a fully symmetric
    setup (equal budgets against a permutation-symmetric covariance) has
    a vanishing gradient at its own symmetric solution and would give a
    false negative here."""
    import torch
    from allocation_lstm import budgets_to_weights_differentiable

    rng = np.random.default_rng(0)
    n = 6
    A = rng.normal(size=(n, n))
    cov = A @ A.T / n + np.eye(n) * 0.05
    budgets = rng.dirichlet(np.ones(n))

    b_t = torch.tensor(budgets, dtype=torch.float64, requires_grad=True)
    weights = budgets_to_weights_differentiable(b_t, cov)
    (weights ** 2).sum().backward()
    assert b_t.grad is not None
    assert torch.any(b_t.grad != 0)


@_diff_layer_skip
def test_differentiable_layer_batches_over_leading_dimension():
    import torch
    from allocation_lstm import budgets_to_weights_differentiable, differentiable_risk_budget_layer

    n = 5
    cov = np.eye(n) * 0.02 + 0.005
    cov_batch = np.stack([cov, cov])
    budgets_batch = torch.tensor(
        np.stack([np.full(n, 1.0 / n), np.array([0.4, 0.2, 0.2, 0.1, 0.1])]), dtype=torch.float64
    )
    layer = differentiable_risk_budget_layer(n)
    weights = budgets_to_weights_differentiable(budgets_batch, cov_batch, layer=layer)
    assert weights.shape == (2, n)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, dtype=torch.float64), atol=1e-6)


@_diff_layer_skip
def test_build_lstm_training_windows_shapes_and_point_in_time():
    from data import make_synthetic_universe
    from features import build_feature_matrix
    from allocation_lstm import build_lstm_training_windows

    ds, _ = make_synthetic_universe(n_days=600, seed=2)
    returns = ds.returns
    features = build_feature_matrix(returns)

    seq_len, cov_window = 15, 80
    feature_windows, covariances, fwd_gross_returns, dates = build_lstm_training_windows(
        features, returns, seq_len=seq_len, cov_window=cov_window
    )
    assert len(feature_windows) > 0
    assert len(feature_windows) == len(covariances) == len(fwd_gross_returns) == len(dates)

    n_assets = returns.shape[1]
    for fw, cov, fwd in zip(feature_windows, covariances, fwd_gross_returns):
        assert fw.shape == (seq_len, features.shape[1])
        assert cov.shape == (n_assets, n_assets)
        assert fwd.shape == (n_assets,)
        assert np.isfinite(fw).all() and np.isfinite(cov).all() and np.isfinite(fwd).all()

    # Every decision date must actually be in both indices, and dates must
    # be strictly increasing (one entry per distinct rebalance date).
    assert all(d in features.index and d in returns.index for d in dates)
    assert all(dates[i] < dates[i + 1] for i in range(len(dates) - 1))


@_diff_layer_skip
def test_train_lstm_allocator_runs_and_reduces_loss():
    from data import make_synthetic_universe
    from features import build_feature_matrix
    from allocation_lstm import RiskBudgetLSTM, build_lstm_training_windows, train_lstm_allocator

    ds, _ = make_synthetic_universe(n_days=900, seed=3)
    returns = ds.returns
    features = build_feature_matrix(returns)
    feature_windows, covariances, fwd_gross_returns, _ = build_lstm_training_windows(
        features, returns, seq_len=20, cov_window=100
    )

    n_features = feature_windows[0].shape[1]
    n_assets = fwd_gross_returns[0].shape[0]
    model = RiskBudgetLSTM(n_features=n_features, n_assets=n_assets, hidden_size=8)

    result = train_lstm_allocator(
        model, feature_windows, covariances, fwd_gross_returns,
        n_epochs=30, lr=5e-3, turnover_penalty=0.05,
    )
    loss_history = result["loss_history"]
    assert len(loss_history) == 30
    assert all(np.isfinite(loss_history))
    assert loss_history[-1] < loss_history[0]

    trained_model = result["model"]
    import torch
    with torch.no_grad():
        out = trained_model(torch.tensor(np.stack(feature_windows), dtype=torch.float64))
    assert out.shape == (len(feature_windows), n_assets)
    assert torch.allclose(out.sum(dim=-1), torch.ones(len(feature_windows), dtype=torch.float64), atol=1e-5)
