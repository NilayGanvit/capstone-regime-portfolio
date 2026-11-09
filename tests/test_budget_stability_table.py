import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "verification")]

import budget_stability_table as b  # noqa: E402


def test_table4_is_reproduced_from_the_saved_budgets(capsys):
    assert b.main() == 0
    assert "PASS: Table 4 reproduced" in capsys.readouterr().out


def test_bounded_softmax_halves_the_dispersion_of_the_saved_budgets():
    compared = b.comparison(b.stability_summary(b.load_raw()))
    assert (compared["L1_reduction_fraction"] > 0.5).all() and (compared["HHI_sd_reduction_fraction"] > 0.7).all()
