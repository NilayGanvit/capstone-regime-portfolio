"""
DSR sensitivity grid for the frozen 5% bounded-Softmax reference run, so the
Deflated Sharpe analysis and Tables 5 and 6 come from the same run.

The frozen run was made before the sensitivity grid existed and reports only
an older single-trial-count DSR. This script grades the frozen run's own daily
excess returns (archived under outputs/archive/frozen_reference_5pct_bb46b63/)
with the current grid (evaluation.dsr_grid_dataframe, scenarios from
outputs/trial_log.csv and outputs/trial_candidate_sharpes.csv).

The archived return files hold the six ERC and LSTM strategies only. The
equal-weight benchmark does not depend on the LSTM, so it is regenerated with
the same walk-forward (run_real_data.run_primary_walk_forward without LSTM
models) and cached in outputs/frozen_reference_equal_weight_net_returns_5bps.csv.
Before use, the regenerated ERC series is checked against the archived one and
the equal-weight Sharpe against the archived cost-sensitivity table; the
script stops if either disagrees.

Basis: Sharpe ratios are on excess returns over the archived risk-free file, so
the observed values are on the same basis as the frozen Table 5. The trial
candidate Sharpes behind the variance scenarios are zero-risk-free and cannot be
recomputed (no daily return paths survive); only their variance is used.

Usage:
    python scripts/compute_frozen_dsr_grid.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import PriceDataset, M2Calendar, initial_window_length, stage_labels, load_risk_free_daily  # noqa: E402
from evaluation import (  # noqa: E402
    net_of_cost_returns, sharpe_ratio, build_dsr_sensitivity_scenarios, dsr_grid_dataframe,
)
from run_real_data import run_primary_walk_forward, TRIAL_LOG_PATH, TRIAL_CANDIDATE_SHARPES_PATH  # noqa: E402

ARCHIVE = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63"
EW_CACHE = ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv"
OUT = ROOT / "outputs" / "frozen_reference_dsr_sensitivity_grid.csv"
CONFIG_ORDER = [
    "erc_baseline", "erc_regime", "erc_blend", "equal_weight",
    "lstm_baseline", "lstm_regime", "lstm_blend",
]


def main() -> None:
    frozen_net = pd.read_csv(ARCHIVE / "csv" / "real_data_rq_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
    frozen_excess = pd.read_csv(ARCHIVE / "csv" / "real_data_rq_excess_returns_5bps.csv", parse_dates=["date"]).set_index("date")
    frozen_cost = pd.read_csv(ARCHIVE / "csv" / "real_data_cost_sensitivity.csv")
    final_dates = frozen_net.index

    rf = pd.Series(load_risk_free_daily(str(ARCHIVE / "data"), final_dates), index=final_dates)
    if not np.allclose(frozen_net.to_numpy() - rf.to_numpy()[:, None], frozen_excess.to_numpy(), atol=1e-10):
        sys.exit("Archived excess returns are not net returns minus the archived risk-free file.")

    if EW_CACHE.exists():
        ew_net = pd.read_csv(EW_CACHE, parse_dates=["date"]).set_index("date")["equal_weight"]
        print(f"Using cached equal-weight series: {EW_CACHE.name}")
    else:
        ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
        calendar = M2Calendar()
        result = run_primary_walk_forward(ds, initial_window_length(ds.returns.index, calendar))
        mask = (stage_labels(pd.DatetimeIndex(result.dates), calendar) == "final_test").to_numpy()
        net = {
            c: pd.Series(
                net_of_cost_returns(result.dates, result.portfolio_returns[c], result.execution_history[c], fee_bps=5.0)[mask],
                index=pd.DatetimeIndex(np.asarray(result.dates)[mask]),
            )
            for c in ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
        }
        for c in ["erc_baseline", "erc_regime", "erc_blend"]:
            worst = float(np.abs(net[c].reindex(final_dates).to_numpy() - frozen_net[c].to_numpy()).max())
            if worst > 1e-9:
                sys.exit(f"Regenerated {c} does not match the archived series (max difference {worst:.2e}).")
        print("Regenerated ERC series match the archived ones (max difference < 1e-9).")
        ew_net = net["equal_weight"]
        ew_net.rename("equal_weight").rename_axis("date").to_csv(EW_CACHE)
        print(f"Cached equal-weight series to {EW_CACHE}")

    if not ew_net.index.equals(final_dates):
        sys.exit("Equal-weight dates do not match the archived final-test dates.")
    ew_sharpe = sharpe_ratio(ew_net.to_numpy(), risk_free=rf.to_numpy())
    archived = float(frozen_cost[(frozen_cost.config == "equal_weight") & (frozen_cost.fee_bps == 5.0)]["sharpe_net"].iloc[0])
    if abs(ew_sharpe - archived) > 5e-4:
        sys.exit(f"Equal-weight Sharpe {ew_sharpe:.4f} disagrees with the archived {archived:.3f}.")
    print(f"Equal-weight net excess Sharpe {ew_sharpe:.4f} matches the archived {archived:.3f}.")

    excess = {c: (frozen_excess[c] if c in frozen_excess else ew_net - rf).to_numpy() for c in CONFIG_ORDER}
    variance_scenarios, k_scenarios = build_dsr_sensitivity_scenarios(str(TRIAL_LOG_PATH), str(TRIAL_CANDIDATE_SHARPES_PATH))
    grid = dsr_grid_dataframe(excess, variance_scenarios, k_scenarios)
    grid.to_csv(OUT, index=False)

    print(f"\nVariance scenarios: {({k: round(v, 4) for k, v in variance_scenarios.items()})}")
    print(f"K scenarios: {k_scenarios}\n")
    for config, g in grid.groupby("config", sort=False):
        print(f"{config:14s} excess Sharpe {g['sharpe'].iloc[0]:.3f}   "
              f"DSR-L [{g['dsr_l'].min():.3f}, {g['dsr_l'].max():.3f}]   "
              f"DSR-LS Gaussian [{g['dsr_ls_gaussian'].min():.3f}, {g['dsr_ls_gaussian'].max():.3f}]   "
              f"DSR-LS Gumbel [{g['dsr_ls_gumbel'].min():.3f}, {g['dsr_ls_gumbel'].max():.3f}]")
    print(f"\nSaved {len(grid)} rows to {OUT}")


if __name__ == "__main__":
    main()
