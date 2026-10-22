import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from allocation_lstm import _TORCH_AVAILABLE, predict_budgets

pytestmark = pytest.mark.skipif(not _TORCH_AVAILABLE, reason="requires torch")


def test_checkpoint_roundtrip_reproduces_the_budgets_exactly(tmp_path):
    from allocation_lstm import RiskBudgetLSTM
    from lstm_training import load_checkpoint, save_checkpoint

    model = RiskBudgetLSTM(n_features=6, n_assets=10, random_state=3, budget_floor=0.05).double()
    save_checkpoint(model, tmp_path / "m.pt", seed=3, variant="baseline", epochs=1)
    loaded = load_checkpoint(tmp_path / "m.pt")
    window = np.random.default_rng(0).normal(size=(60, 6))
    assert np.array_equal(predict_budgets(model, window), predict_budgets(loaded, window))
    assert loaded.budget_floor == 0.05 and not loaded.training


def test_archived_frozen_checkpoint_loads_and_gives_valid_budgets():
    path = Path(__file__).resolve().parents[1] / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "ten_seed" / "seed_00_regime.pt"
    from lstm_training import load_checkpoint

    model = load_checkpoint(path)
    b = predict_budgets(model, np.random.default_rng(1).normal(size=(60, 35)))
    assert b.sum() == pytest.approx(1.0) and b.min() >= 0.05 - 1e-9
