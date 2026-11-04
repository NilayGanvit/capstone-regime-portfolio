import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts" / "verification"), str(ROOT / "scripts"), str(ROOT / "src")]

import run_ten_seed_final_pipeline as t  # noqa: E402


def test_settings_are_the_m6_reference_run_settings_except_the_seed():
    manifest = json.loads((t.M6 / "run_manifest.json").read_text())["settings"]
    mine = t.settings(seed=4, epochs=200)
    assert mine["lstm_seed"] == 4
    assert {k: v for k, v in mine.items() if k != "lstm_seed"} == {k: v for k, v in manifest.items() if k != "lstm_seed"}
    assert t.settings(seed=123, epochs=200) == manifest


def test_code_under_test_is_the_m6_reference_code():
    diff = subprocess.run(["git", "diff", "--name-only", "6a6b12d", "HEAD", "--", "src", "scripts/run_real_data.py"],
                          cwd=ROOT, capture_output=True, text=True)
    if diff.returncode != 0:
        import pytest
        pytest.skip("the reference commit is not in this clone")
    assert diff.stdout.strip() == ""


def test_distribution_counts_and_ranks_seed_123_among_eleven():
    rows = []
    for s, v in enumerate([0.2, -0.1, 0.05, 0.0, 0.3, -0.05, 0.1, 0.15, -0.2, 0.02]):
        for k in t.CONTRASTS:
            rows.append({"seed": s, "contrast": k, "point_estimate": v, "ci_low": v - 0.1, "ci_high": v + 0.1})
    per = pd.DataFrame(rows)
    ctrl = pd.DataFrame([{"seed": 123, "contrast": k, "point_estimate": -0.3} for k in t.CONTRASTS])
    d = t.distribution(per, ctrl).set_index("contrast").loc["rq2_difference_in_differences"]
    assert d.seeds == 10 and d.positive == 6 and d.negative == 3
    assert d.interval_above_zero == 3 and d.interval_below_zero == 1  # 0.2, 0.3 and 0.15 above; -0.2 below
    assert d.seed_123_rank_of_11 == 1
