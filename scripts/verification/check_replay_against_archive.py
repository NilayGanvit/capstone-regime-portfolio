"""
Does a saved LSTM checkpoint pair reproduce an archived run's daily net returns?

Loads a baseline and a regime checkpoint (outputs/lstm_checkpoints/seed123_*.pt from a
run_real_data.py run, or an archived checkpoint), runs the full walk-forward with the chosen
budget-to-weight solver, and compares the final-test daily net returns (5 bps) of the three
LSTM configurations with an archived CSV that has the columns date, lstm_baseline,
lstm_regime, lstm_blend.

Use: after a Colab run of the corrected pipeline, check that its seed-123 checkpoints equal
the frozen reference's model by replaying them with the ORIGINAL solver ("slsqp") against
the archived frozen returns. Training does not depend on the solver, so a match (about 1e-7,
inference is reproducible across environments to that level) shows the corrected run's
network is the frozen one, and that the exact-solver results are its corrected counterpart.
A mismatch means the network is a different one, and the corrected results cannot be read
as a correction of the frozen ones.

Usage:
    python scripts/verification/check_replay_against_archive.py \\
        --baseline outputs/lstm_checkpoints/seed123_baseline.pt \\
        --regime outputs/lstm_checkpoints/seed123_regime.pt \\
        --archived outputs/archive/frozen_reference_5pct_bb46b63/csv/real_data_rq_net_returns_5bps.csv \\
        --solver slsqp
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import M2Calendar, PriceDataset, initial_window_length, stage_labels  # noqa: E402
from evaluation import net_of_cost_returns  # noqa: E402
from features import build_feature_matrix  # noqa: E402
from lstm_training import load_checkpoint  # noqa: E402
from run_real_data import run_primary_walk_forward  # noqa: E402

TOLERANCE = 1e-5
CONFIGS = ["lstm_baseline", "lstm_regime", "lstm_blend"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--archived", required=True)
    ap.add_argument("--solver", choices=["slsqp", "exact"], default="slsqp")
    a = ap.parse_args()

    models = {"baseline": load_checkpoint(Path(a.baseline)), "regime": load_checkpoint(Path(a.regime))}
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    calendar = M2Calendar()
    result = run_primary_walk_forward(ds, initial_window_length(ds.returns.index, calendar), models,
                                      build_feature_matrix(ds.returns), risk_budget_solver=a.solver)
    dates = pd.DatetimeIndex(result.dates)
    final = (stage_labels(dates, calendar) == "final_test").to_numpy()
    archived = pd.read_csv(a.archived, parse_dates=["date"]).set_index("date")
    if not dates[final].equals(archived.index):
        sys.exit(f"Final-test dates differ from the archived file ({len(dates[final])} vs {len(archived)}).")

    worst = 0.0
    for c in CONFIGS:
        net = net_of_cost_returns(result.dates, result.portfolio_returns[c], result.execution_history[c], 5.0)[final]
        diff = float(np.abs(net - archived[c].to_numpy()).max())
        worst = max(worst, diff)
        print(f"  {c:14s} max |daily net return difference| = {diff:.2e}")
    verdict = "PASS" if worst < TOLERANCE else "FAIL"
    print(f"{verdict}: worst difference {worst:.2e} (tolerance {TOLERANCE:.0e}), solver {a.solver}")
    sys.exit(0 if worst < TOLERANCE else 1)


if __name__ == "__main__":
    main()
