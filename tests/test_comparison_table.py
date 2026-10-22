import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not (ROOT / "outputs" / "ten_seed_corrected").exists(), reason="ten-seed returns not present")

from build_comparison_table import ten_seed_table  # noqa: E402


def _mean(per, solver, config, key):
    return float(np.mean([m[key] for m in per[(solver, config)]]))


def test_original_erc_rows_equal_the_frozen_table_5_sharpes():
    per = ten_seed_table()
    for config, frozen in [("erc_baseline", 0.455), ("erc_regime", 0.462), ("erc_blend", 0.465)]:
        assert _mean(per, "slsqp", config, "sharpe") == pytest.approx(frozen, abs=6e-4)


def test_equal_weight_does_not_depend_on_the_solver_and_matches_the_archive():
    per = ten_seed_table()
    assert _mean(per, "slsqp", "equal_weight", "sharpe") == pytest.approx(_mean(per, "exact", "equal_weight", "sharpe"), abs=1e-12)
    assert _mean(per, "exact", "equal_weight", "sharpe") == pytest.approx(0.608, abs=6e-4)


def test_ten_seed_lstm_means_match_the_verified_replay():
    per = ten_seed_table()
    assert len(per[("exact", "lstm_regime")]) == 10
    assert _mean(per, "slsqp", "lstm_regime", "sharpe") == pytest.approx(0.6143, abs=5e-4)
    assert _mean(per, "exact", "lstm_regime", "sharpe") == pytest.approx(0.5085, abs=5e-4)
