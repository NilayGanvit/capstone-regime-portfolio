"""
Trial 18: asset-specific regime detection for ERC, run exactly as pre-registered in
docs/experiments/asset_specific_regimes_preregistration.md.

The primary walk-forward (no LSTM models) twice, without and with asset_regime_states=2,
then the reproduction check, the validation gates and the pre-registered reporting.
Net of costs, Sharpe in excess of the M6 reference run's risk-free series.

Usage:
    python scripts/experiments/run_asset_specific_regimes.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from data import M2Calendar, PriceDataset, UNIVERSE, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import block_bootstrap_diff_ci, cagr, max_drawdown, net_of_cost_returns, sharpe_ratio  # noqa: E402
from run_real_data import run_primary_walk_forward  # noqa: E402

OUT = ROOT / "outputs" / "experiments"
FEES = [5.0, 10.0, 25.0]
NEW = "erc_asset_regime"
CONFIGS = [NEW, "erc_baseline", "erc_regime", "equal_weight"]
SHARED = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
RF_DIR = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
M6_NET = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d" / "csv" / "real_data_rq_net_returns_5bps.csv"
EW_NET = ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv"
STAGES = {"validation": "validation 2015-2018", "final_test": "final test 2019-2026 (exploratory)"}


def reproduction_check(base, run, net):
    same = {c: bool(np.array_equal(base.portfolio_returns[c], run.portfolio_returns[c])) for c in SHARED}
    dates = pd.DatetimeIndex(run.dates)
    ref = pd.concat([pd.read_csv(M6_NET, index_col=0, parse_dates=True),
                     pd.read_csv(EW_NET, index_col=0, parse_dates=True)], axis=1)
    worst = {}
    for c in SHARED:
        mine = pd.Series(net[(c, 5.0)], index=dates).reindex(ref.index)
        worst[c] = float((mine - ref[c]).abs().max())
    return {"identical_without_option": same, "worst_daily_diff_vs_m6_archive": worst,
            "passes": all(same.values()) and all(v <= 1e-8 for v in worst.values())}


def stage_rows(run, net, rf, mask, stage):
    dates = pd.DatetimeIndex(run.dates)
    in_stage = set(dates[mask])
    rows = []
    for c in CONFIGS:
        n5 = net[(c, 5.0)][mask]
        rebal = [b for b in run.binding_constraints_history[c] if b["date"] in in_stage]
        row = {"stage": stage, "config": c, "days": int(mask.sum()),
               **{f"sharpe_{int(f)}bps": sharpe_ratio(net[(c, f)][mask], risk_free=rf[mask]) for f in FEES},
               "cagr": cagr(n5), "max_drawdown": max_drawdown(n5),
               "vol": float(np.std(n5 - rf[mask], ddof=1) * np.sqrt(252)),
               "mean_turnover_per_rebalance": float(np.mean([b["turnover"] for b in rebal])),
               "turnover_binding_rebalances": int(sum(b["turnover_binding"] for b in rebal)), "rebalances": len(rebal)}
        if c != NEW:
            for f in FEES:
                row[f"delta_sharpe_new_minus_this_{int(f)}bps"] = (
                    sharpe_ratio(net[(NEW, f)][mask], risk_free=rf[mask]) - row[f"sharpe_{int(f)}bps"])
            ci = block_bootstrap_diff_ci(net[(NEW, 5.0)][mask] - rf[mask], n5 - rf[mask], n_boot=2000, random_state=0)
            row.update({"delta_ci_low": ci["ci_low"], "delta_ci_high": ci["ci_high"], "bootstrap_block": ci["expected_block_length"]})
        rows.append(row)
    return rows


def regime_occupancy(run, mask, stage):
    in_stage = set(pd.DatetimeIndex(run.dates)[mask])
    hv = np.array([h["high_var_prob"] for h in run.asset_regime_history if h["date"] in in_stage]) > 0.5
    spy = hv[:, UNIVERSE.index("SPY")]
    return [{"stage": stage, "asset": t, "share_high_variance": float(hv[:, i].mean()),
             "agreement_with_spy": float((hv[:, i] == spy).mean())} for i, t in enumerate(UNIVERSE)]


def mean_weights(run, mask, stage):
    in_stage = set(pd.DatetimeIndex(run.dates)[mask])
    out = []
    for c in CONFIGS:
        w = np.array([run.weights[c][k] for k, d in enumerate(run.dates) if d in in_stage])
        out.append({"stage": stage, "config": c, **dict(zip(UNIVERSE, w.mean(axis=0)))})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    cal = M2Calendar()
    iw = initial_window_length(ds.returns.index, cal)
    base = run_primary_walk_forward(ds, iw)
    run = run_primary_walk_forward(ds, iw, asset_regime_states=2)

    dates = pd.DatetimeIndex(run.dates)
    rf = load_risk_free_daily(str(RF_DIR), dates)
    net = {(c, f): net_of_cost_returns(run.dates, run.portfolio_returns[c], run.execution_history[c], f)
           for c in run.portfolio_returns for f in FEES}
    repro = reproduction_check(base, run, net)
    print("Reproduction check:", json.dumps(repro, indent=2))
    if not repro["passes"]:
        (OUT / "asset_specific_regimes_gates.json").write_text(json.dumps({"reproduction_check": repro}, indent=2))
        raise SystemExit("Reproduction check failed; trial stopped as pre-registered.")

    stages = stage_labels(dates, cal).to_numpy()
    rows, occ, wts = [], [], []
    for stage, label in STAGES.items():
        m = stages == stage
        rows += stage_rows(run, net, rf, m, label)
        occ += regime_occupancy(run, m, label)
        wts += mean_weights(run, m, label)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "asset_specific_regimes_results.csv", index=False, float_format="%.6g")
    pd.DataFrame(occ).to_csv(OUT / "asset_specific_regimes_regime_occupancy.csv", index=False, float_format="%.6g")
    pd.DataFrame(wts).to_csv(OUT / "asset_specific_regimes_mean_weights.csv", index=False, float_format="%.6g")

    v = table[table.stage == STAGES["validation"]].set_index("config")
    gates = {"G1 Sharpe >= erc_baseline": bool(v.loc["erc_baseline", "delta_sharpe_new_minus_this_5bps"] >= 0),
             "G2 Sharpe >= erc_regime": bool(v.loc["erc_regime", "delta_sharpe_new_minus_this_5bps"] >= 0),
             "G3 drawdown within 2 pp of erc_baseline": bool(v.loc[NEW, "max_drawdown"] >= v.loc["erc_baseline", "max_drawdown"] - 0.02)}
    report = {"reproduction_check": repro, "gates": gates, "passes": all(gates.values())}
    (OUT / "asset_specific_regimes_gates.json").write_text(json.dumps(report, indent=2))

    pd.set_option("display.width", 220)
    for label in STAGES.values():
        print(f"\n=== {label} (net; Sharpe in excess of the risk-free rate) ===")
        print(table[table.stage == label].drop(columns="stage").round(4).to_string(index=False))
    print("\nRegime occupancy:")
    print(pd.DataFrame(occ).round(3).to_string(index=False))
    print("\nMean weights:")
    print(pd.DataFrame(wts).round(3).to_string(index=False))
    print("\nGates:", json.dumps(gates, indent=2), "\nPasses:", report["passes"])


if __name__ == "__main__":
    main()
