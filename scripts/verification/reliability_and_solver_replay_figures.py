"""
Recompute the reported figures that no pipeline output file holds, from the archives and the
ERC-only walk-forward (which does not depend on the LSTM or the machine). Writes nothing.

    Section 4.1  validation log-score advantage of the regime-aware density (M1 minus M0) and
                 how often M1 scores higher; correlation, on the final test, between 1 - pi at
                 each month-end decision (carried forward) and |blend - regime| daily net returns
    Section 4.6  the seed-123 LSTM regime effect under the original SLSQP weights (frozen
                 archive) and the exact solver (M6 reference archive), same trained networks

Usage:
    python scripts/verification/reliability_and_solver_replay_figures.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_real_data  # noqa: E402
from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import sharpe_ratio  # noqa: E402

M6 = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d" / "csv"
FROZEN = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63"


def main() -> None:
    pi = pd.read_csv(M6 / "real_data_pi_path.csv", index_col=0, parse_dates=True)["pi_t"]
    net = pd.read_csv(M6 / "real_data_rq_net_returns_5bps.csv", index_col=0, parse_dates=True)

    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    cal = M2Calendar()
    result = run_real_data.run_primary_walk_forward(ds, initial_window_length(ds.returns.index, cal))
    dates = pd.DatetimeIndex(result.dates)
    stages = stage_labels(dates, cal)
    print(f"reliability path vs the M6 archive, worst difference: "
          f"{np.max(np.abs(np.array(result.pi_history) - pi.reindex(dates).to_numpy())):.1e}")

    print("\nSection 4.1, predictive log scores (M1 minus M0):")
    diff = np.array(result.log_l1_history) - np.array(result.log_l0_history)
    for stage in ("validation", "final_test"):
        m = (stages == stage).to_numpy()
        print(f"  {stage}: mean {diff[m].mean():.4f}, M1 higher on {(diff[m] > 0).mean():.1%} of {m.sum()} days")

    print("\nSection 4.1, correlation of 1 - pi (decision value, carried forward) with |blend - regime|:")
    decision = pi.groupby(pi.index.to_period("M")).tail(1)
    carried = (1 - decision).reindex(pi.index).ffill().shift(1).reindex(net.index)
    for allocator in ("erc", "lstm"):
        gap = (net[f"{allocator}_blend"] - net[f"{allocator}_regime"]).abs()
        print(f"  {allocator}: {carried.corr(gap):.3f}")

    print("\nSection 4.6, seed-123 LSTM regime effect (net 5 bps, excess of the T-bill rate):")
    for label, folder in (("SLSQP weights", FROZEN / "csv"), ("exact solver", M6)):
        returns = pd.read_csv(folder / "real_data_rq_net_returns_5bps.csv", index_col=0, parse_dates=True)
        returns = returns[returns.index >= "2019-01-01"]
        rf = load_risk_free_daily(str(FROZEN / "data"), list(returns.index))
        effect = (sharpe_ratio(returns["lstm_regime"].to_numpy(), risk_free=rf)
                  - sharpe_ratio(returns["lstm_baseline"].to_numpy(), risk_free=rf))
        print(f"  {label}: {effect:+.3f}")


if __name__ == "__main__":
    main()
