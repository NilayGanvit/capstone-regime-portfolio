import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "verification"), str(ROOT / "scripts"), str(ROOT / "src")]

import run_quarterly_lstm_retraining as q  # noqa: E402
import run_real_data  # noqa: E402


def test_quarterly_arm_passes_the_reference_arguments_plus_only_the_retrain_option(monkeypatch):
    calls = []
    monkeypatch.setattr(run_real_data, "run_walk_forward", lambda *a, **k: calls.append((a, k)))
    ds = type("DS", (), {"returns": "R", "opens": "O", "close_to_open_returns": "CO", "open_to_close_returns": "OC"})()
    run_real_data.run_primary_walk_forward(ds, 10, "models", "fm")
    with q.walk_forward_options(lstm_retrain_every_n_rebalances=3):
        run_real_data.run_primary_walk_forward(ds, 10, "models", "fm")
    (a0, k0), (a1, k1) = calls
    assert a0 == a1
    assert k1 == {**k0, "lstm_retrain_every_n_rebalances": 3}
    assert "lstm_retrain_every_n_rebalances" not in k0
    # the patch is undone afterwards
    run_real_data.run_primary_walk_forward(ds, 10, "models", "fm")
    assert calls[2][1] == k0


def test_outputs_never_overwrite(tmp_path):
    out = q.NoOverwrite(tmp_path / "run")
    out.csv("a.csv", pd.DataFrame({"x": [1]}))
    with pytest.raises(FileExistsError):
        out.csv("a.csv", pd.DataFrame({"x": [2]}))
    with pytest.raises(FileExistsError):
        q.NoOverwrite(tmp_path / "run")  # an existing run folder is never reused
    assert pd.read_csv(tmp_path / "run" / "a.csv").x.tolist() == [1]
