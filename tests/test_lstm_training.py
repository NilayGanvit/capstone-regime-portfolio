import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_lstm import _TORCH_AVAILABLE, _CVXPYLAYERS_AVAILABLE
from data import make_synthetic_universe

_skip = pytest.mark.skipif(
    not (_TORCH_AVAILABLE and _CVXPYLAYERS_AVAILABLE),
    reason="requires torch and cvxpylayers, install per requirements.txt",
)

_SMALL = dict(n_epochs=1, seq_len=20, cov_window=60, n_states=2)


@_skip
def test_train_lstm_variants_builds_both_variants_with_the_requested_seed_and_floor():
    from allocation_lstm import RiskBudgetLSTM
    from lstm_training import train_lstm_variants

    ds, _ = make_synthetic_universe(n_days=500, seed=0)
    models, feature_matrix = train_lstm_variants(ds.returns, 400, seed=5, budget_floor=0.05, **_SMALL)
    assert set(models) == {"baseline", "regime"}
    for model in models.values():
        assert isinstance(model, RiskBudgetLSTM)
        assert model.budget_floor == 0.05
    assert models["regime"].lstm.input_size == models["baseline"].lstm.input_size + 2  # + n_states regime probabilities
    assert len(feature_matrix) > 0


@_skip
def test_same_seed_gives_identical_weights_and_a_different_seed_does_not():
    import torch
    from lstm_training import train_lstm_variants

    ds, _ = make_synthetic_universe(n_days=500, seed=0)
    a, _ = train_lstm_variants(ds.returns, 400, seed=1, **_SMALL)
    b, _ = train_lstm_variants(ds.returns, 400, seed=1, **_SMALL)
    c, _ = train_lstm_variants(ds.returns, 400, seed=2, **_SMALL)
    assert torch.equal(a["baseline"].dense.weight, b["baseline"].dense.weight)
    assert not torch.equal(a["baseline"].dense.weight, c["baseline"].dense.weight)
