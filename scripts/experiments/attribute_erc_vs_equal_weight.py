"""
Where does ERC's shortfall against equal weight come from? Checks the premise both
experiments in scripts/experiments/ rest on: that ERC loses to 1/N mainly because it holds
more duration and credit than 1/N, and that this cost concentrates in 2022.

Runs the ERC-only walk-forward (exact solver, the pipeline default; no LSTM, so it is
environment-invariant) and attributes the daily active return of each ERC configuration
against equal weight to asset groups:

    active_t = sum_i (w_ERC,i,t-1 - w_EW,i,t-1) * r_i,t

with w_t-1 the holdings at the previous close. On next-open execution days the true return
splits into an overnight leg on the old holdings and an intraday leg on the new target, so
this attribution is approximate; the residual against the actual return difference is
reported. Gross of costs (both portfolios trade about 1% a month).

This is explanatory analysis of periods already inspected (validation and final test); it
does not select anything.

Usage:
    python scripts/experiments/attribute_erc_vs_equal_weight.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from data import M2Calendar, PriceDataset, UNIVERSE, initial_window_length, stage_labels  # noqa: E402
from run_real_data import run_primary_walk_forward  # noqa: E402

GROUPS = {
    "equities and REITs (SPY, EFA, EEM, VNQ)": ["SPY", "EFA", "EEM", "VNQ"],
    "Treasuries (IEF, TLT)": ["IEF", "TLT"],
    "credit (LQD, HYG)": ["LQD", "HYG"],
    "gold and commodities (GLD, DBC)": ["GLD", "DBC"],
}
OUT = ROOT / "outputs" / "erc_vs_equal_weight_attribution.csv"


def main() -> None:
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    calendar = M2Calendar()
    result = run_primary_walk_forward(ds, initial_window_length(ds.returns.index, calendar))
    dates = pd.DatetimeIndex(result.dates)
    simple = np.exp(ds.returns.loc[dates].to_numpy()) - 1.0
    stages = stage_labels(dates, calendar).to_numpy()
    idx = {t: i for i, t in enumerate(UNIVERSE)}

    periods = {
        "validation 2015-2018": stages == "validation",
        "final test 2019-2026": stages == "final_test",
        "2022": (dates.year == 2022) & (stages == "final_test"),
        "final test excluding 2022": (dates.year != 2022) & (stages == "final_test"),
    }
    rows = []
    for config in ["erc_baseline", "erc_regime"]:
        w, w_ew = result.weights[config], result.weights["equal_weight"]
        held = np.vstack([np.full(len(UNIVERSE), 0.1), w[:-1]])
        held_ew = np.vstack([np.full(len(UNIVERSE), 0.1), w_ew[:-1]])
        contrib = (held - held_ew) * simple  # per day, per asset
        actual = result.portfolio_returns[config] - result.portfolio_returns["equal_weight"]
        for pname, m in periods.items():
            row = {"config": config, "period": pname, "days": int(m.sum()),
                   "erc_cum_return": float(np.prod(1 + result.portfolio_returns[config][m]) - 1),
                   "ew_cum_return": float(np.prod(1 + result.portfolio_returns["equal_weight"][m]) - 1),
                   "active_sum_actual": float(actual[m].sum()),
                   "active_sum_attributed": float(contrib[m].sum()),
                   }
            for g, tickers in GROUPS.items():
                cols = [idx[t] for t in tickers]
                row[f"active weight: {g}"] = float((held[m][:, cols] - held_ew[m][:, cols]).sum(axis=1).mean())
                row[f"contribution: {g}"] = float(contrib[m][:, cols].sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT, index=False, float_format="%.6g")

    pd.set_option("display.width", 220)
    for config in ["erc_baseline", "erc_regime"]:
        t = table[table.config == config].set_index("period")
        print(f"\n=== {config} minus equal weight (gross; arithmetic sums of daily returns, percentage points) ===")
        print(f"{'':28s} {'ERC cum':>8s} {'EW cum':>8s} {'active':>8s} {'resid':>7s}   " + "   ".join(f"{g.split(' (')[0][:22]:>22s}" for g in GROUPS))
        for p, r in t.iterrows():
            contribs = "   ".join(f"{100 * r[f'contribution: {g}']:+8.2f} ({100 * r[f'active weight: {g}']:+5.1f}w)".rjust(22) for g in GROUPS)
            print(f"{p:28s} {100 * r.erc_cum_return:+7.1f}% {100 * r.ew_cum_return:+7.1f}% {100 * r.active_sum_actual:+7.2f} "
                  f"{100 * (r.active_sum_actual - r.active_sum_attributed):+6.2f}   {contribs}")
    print("\nEach group cell: contribution to active return in pp (mean active weight vs equal weight, in pp of NAV).")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
