"""
Sensitivity run: ERC with the exact convex risk-budget solution in place of the
SLSQP solve, everything else unchanged (same data, HMM signals, shrinkage, weight
cap, turnover limit, constraint projection, next-open execution, 5 bps costs).

Motivation: scripts/verification/audit_erc_solver.py found that the SLSQP solve
leaves assets at zero weight on three final-period dates for erc_regime. This
shows whether the ERC conclusions change when the raw target is the exact one.
It is a sensitivity analysis, not a replacement for the frozen reference.

Scope and limits:
  - ERC configurations only. The LSTM configurations reach their weights through
    the same SLSQP solve (allocation_lstm.budgets_to_weights_batch) but need
    trained models, so they are neither re-run nor corrected here; their rows in
    the bootstrap below are the frozen run's, unchanged.
  - Sharpe ratios are net of 5 bps and in excess of the archived risk-free
    series, as in the frozen Table 5. The frozen ERC rows are read from the
    archive for comparison.
  - The equal-weight benchmark does not depend on the solver and is unchanged.

Usage:
    python scripts/verification/erc_exact_solver_sensitivity.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import (  # noqa: E402
    cagr, max_drawdown, net_of_cost_returns, sharpe_ratio, stationary_bootstrap_rq_contrasts,
)
from run_real_data import run_primary_walk_forward  # noqa: E402

ARCHIVE = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63"
OUT = ROOT / "outputs" / "erc_exact_solver_sensitivity.csv"
ERC = ["erc_baseline", "erc_regime", "erc_blend"]
LSTM = ["lstm_baseline", "lstm_regime", "lstm_blend"]


def main() -> None:
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    calendar = M2Calendar()
    result = run_primary_walk_forward(ds, initial_window_length(ds.returns.index, calendar), risk_budget_solver="exact")

    mask = (stage_labels(pd.DatetimeIndex(result.dates), calendar) == "final_test").to_numpy()
    dates = pd.DatetimeIndex(np.asarray(result.dates)[mask])
    frozen_net = pd.read_csv(ARCHIVE / "csv" / "real_data_rq_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
    if not dates.equals(frozen_net.index):
        sys.exit("Final-test dates differ from the archived run's.")
    rf = load_risk_free_daily(str(ARCHIVE / "data"), dates)

    net = {c: net_of_cost_returns(result.dates, result.portfolio_returns[c], result.execution_history[c], 5.0)[mask]
           for c in ERC + ["equal_weight"]}
    ew_check = float(np.abs(net["equal_weight"] - pd.read_csv(
        ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")["equal_weight"].to_numpy()).max())
    assert ew_check < 1e-9, "equal weight should be unaffected by the solver"

    rows = []
    for c in ERC:
        for label, r in [("frozen (SLSQP)", frozen_net[c].to_numpy()), ("exact convex", net[c])]:
            rows.append({"config": c, "solver": label, "sharpe_excess": sharpe_ratio(r, risk_free=rf),
                         "cagr": cagr(r), "max_drawdown": max_drawdown(r)})
    table = pd.DataFrame(rows)

    def effects(series):
        return {"ERC regime effect": sharpe_ratio(series["erc_regime"], risk_free=rf) - sharpe_ratio(series["erc_baseline"], risk_free=rf),
                "ERC blend - regime": sharpe_ratio(series["erc_blend"], risk_free=rf) - sharpe_ratio(series["erc_regime"], risk_free=rf)}

    frozen_series = {c: frozen_net[c].to_numpy() for c in ERC}
    exact_series = {c: net[c] for c in ERC}
    eff = pd.DataFrame({"frozen (SLSQP)": effects(frozen_series), "exact convex": effects(exact_series)})

    paired = {c: net[c] - rf for c in ERC}
    paired.update({c: frozen_net[c].to_numpy() - rf for c in LSTM})
    ci = stationary_bootstrap_rq_contrasts(paired, n_boot=2000, ci=0.95, random_state=0)
    frozen_ci = pd.read_csv(ARCHIVE / "csv" / "real_data_rq_stationary_bootstrap.csv").set_index("contrast")

    boot_rows = []
    for key, v in ci.items():
        if key.startswith("_"):
            continue  # bootstrap metadata, not a contrast
        boot_rows.append({"contrast": key, "frozen_point": frozen_ci.loc[key, "point_estimate"],
                          "frozen_ci_low": frozen_ci.loc[key, "ci_low"], "frozen_ci_high": frozen_ci.loc[key, "ci_high"],
                          "exact_point": v["point_estimate"], "exact_ci_low": v["ci_low"], "exact_ci_high": v["ci_high"]})
    boot = pd.DataFrame(boot_rows)

    table.to_csv(OUT, index=False, float_format="%.6g")
    boot.to_csv(OUT.with_name("erc_exact_solver_sensitivity_bootstrap.csv"), index=False, float_format="%.6g")

    pd.set_option("display.width", 200)
    print("ERC final-test results, net of 5 bps, excess over the risk-free rate:")
    print(table.round(4).to_string(index=False))
    print("\nSharpe effects:")
    print(eff.round(4).to_string())
    print("\nBootstrap contrasts (95% stationary bootstrap; LSTM rows are the frozen run's, unchanged):")
    print(boot.round(4).to_string(index=False))
    print(f"\nSaved {OUT} and {OUT.with_name('erc_exact_solver_sensitivity_bootstrap.csv')}")


if __name__ == "__main__":
    main()
