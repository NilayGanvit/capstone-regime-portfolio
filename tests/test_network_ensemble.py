import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_lstm import _TORCH_AVAILABLE

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="requires torch")


def _nets(k, n_features=6, n_assets=4):
    from allocation_lstm import RiskBudgetLSTM
    return [RiskBudgetLSTM(n_features=n_features, n_assets=n_assets, random_state=s, budget_floor=0.05).double()
            for s in range(k)]


def test_one_member_ensemble_reproduces_the_member_and_a_larger_one_averages():
    from allocation_lstm import predict_budgets
    from network_ensemble import MeanBudgetEnsemble

    window = np.random.default_rng(0).normal(size=(20, 6))
    nets = _nets(3)
    single = [predict_budgets(n, window) for n in nets]
    assert np.array_equal(predict_budgets(MeanBudgetEnsemble(nets[:1]), window), single[0])
    combined = predict_budgets(MeanBudgetEnsemble(nets), window)
    assert np.allclose(combined, np.mean(single, axis=0), atol=1e-15)
    assert np.isclose(combined.sum(), 1.0) and combined.min() >= 0.05 - 1e-12


def test_empty_ensemble_is_rejected():
    from network_ensemble import MeanBudgetEnsemble
    with pytest.raises(ValueError):
        MeanBudgetEnsemble([])
