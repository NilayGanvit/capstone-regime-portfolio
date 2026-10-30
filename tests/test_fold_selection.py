import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fold_selection import benchmark_relative_choice

ORDER = ["equal_weight", "simple", "complex"]


def _table(rows):
    return pd.DataFrame(rows, columns=["candidate", "fold", "delta", "ci_low", "ci_high"])


def test_highest_worst_fold_wins_not_highest_average():
    t = _table([("simple", "A", 0.02, -0.3, 0.3), ("simple", "B", 0.03, -0.3, 0.3),
                ("complex", "A", -0.01, -0.3, 0.3), ("complex", "B", 0.50, 0.1, 0.9)])
    summary, choice = benchmark_relative_choice(t, ORDER)
    assert choice == "simple"
    assert summary.set_index("candidate").loc["complex", "worst_fold"] == "A"


def test_one_without_beating_1n_in_its_worst_fold_leaves_1n_chosen():
    t = _table([("simple", "A", -0.05, -0.4, 0.3), ("simple", "B", 0.2, -0.2, 0.6),
                ("complex", "A", 0.3, -0.1, 0.7), ("complex", "B", -0.01, -0.3, 0.3)])
    assert benchmark_relative_choice(t, ORDER)[1] == "equal_weight"


def test_a_fold_significantly_below_1n_makes_a_candidate_ineligible():
    t = _table([("simple", "A", 0.01, -0.2, 0.2), ("simple", "B", 0.01, -0.2, 0.2),
                ("complex", "A", 0.40, 0.1, 0.7), ("complex", "B", 0.30, -0.4, -0.01)])
    summary, choice = benchmark_relative_choice(t, ORDER)
    assert not summary.set_index("candidate").loc["complex", "eligible"]
    assert choice == "simple"


def test_ties_at_three_decimals_go_to_the_simpler_candidate():
    t = _table([("simple", "A", 0.1201, -0.2, 0.4), ("complex", "A", 0.1204, -0.2, 0.4)])
    assert benchmark_relative_choice(t, ORDER)[1] == "simple"
