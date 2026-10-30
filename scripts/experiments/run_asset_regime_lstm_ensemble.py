"""
Seed ensembles of the per-asset regime LSTM: each ensemble averages the budgets of trial 20's
trained networks (outputs/experiments/asset_regime_lstm/, loaded for inference only, nothing
retrained) and runs the primary walk-forward with asset_regime_states=2.

Gate, fixed before the run (validation 2015-2018, 5 bps net excess Sharpe,
lstm_asset_regime): in each disjoint half of the seeds (0-4 and 5-9), the ensemble's Sharpe is
at least the mean Sharpe of the same five networks run singly in trial 20.

  --replay --seed N      the single network pair for seed N: checks that replay reproduces trial 20
  --members 0-4|5-9|0-9  the budget-averaged networks of those seeds
  --merge                the checks, gate and reporting

Usage:
    python scripts/experiments/run_asset_regime_lstm_ensemble.py --members 0-9
    python scripts/experiments/run_asset_regime_lstm_ensemble.py --merge
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import block_bootstrap_diff_ci, cagr, max_drawdown, net_of_cost_returns, sharpe_ratio  # noqa: E402
from features import build_feature_matrix  # noqa: E402
from run_real_data import run_primary_walk_forward  # noqa: E402

OUT = ROOT / "outputs" / "experiments"
NETS = OUT / "asset_regime_lstm"
RUNS = OUT / "asset_regime_lstm_ensemble"
SEEDS = list(range(10))
HALVES = {"0-4": [0, 1, 2, 3, 4], "5-9": [5, 6, 7, 8, 9]}
ENSEMBLES = {**HALVES, "0-9": SEEDS}
FEES = [5.0, 10.0, 25.0]
NEW = "lstm_asset_regime"
LSTMS = [NEW, "lstm_baseline", "lstm_regime"]
REPORTED = LSTMS + ["erc_asset_regime", "erc_baseline", "erc_regime", "equal_weight"]
RF_DIR = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
M6_NET = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d" / "csv" / "real_data_rq_net_returns_5bps.csv"
EW_NET = ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv"
STAGES = {"validation": "validation 2015-2018", "final_test": "final test 2019-2026 (exploratory)"}


def load_networks(seeds):
    from lstm_training import load_checkpoint
    from network_ensemble import MeanBudgetEnsemble

    nets = {v: [load_checkpoint(NETS / f"seed_{s:02d}_{v}.pt") for s in seeds] for v in ["baseline", "regime", "asset_regime"]}
    return {v: m[0] if len(seeds) == 1 else MeanBudgetEnsemble(m) for v, m in nets.items()}


def run_job(kind: str, seeds: list, label: str) -> None:
    import torch

    torch.set_num_threads(1)
    RUNS.mkdir(parents=True, exist_ok=True)
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    cal = M2Calendar()
    iw = initial_window_length(ds.returns.index, cal)
    run = run_primary_walk_forward(ds, iw, load_networks(seeds), build_feature_matrix(ds.returns), asset_regime_states=2)
    dates = pd.DatetimeIndex(run.dates)
    net = {(c, f): net_of_cost_returns(run.dates, run.portfolio_returns[c], run.execution_history[c], f)
           for c in REPORTED + ["erc_blend"] for f in FEES}
    tag = f"{kind}_{label}"
    pd.DataFrame({c: net[(c, 5.0)] for c in REPORTED}, index=dates.rename("date")).to_csv(
        RUNS / f"{tag}_net_returns_5bps.csv", float_format="%.10g")

    record = {"kind": kind, "seeds": seeds}
    if kind == "replay":
        t20 = pd.read_csv(NETS / f"seed_{seeds[0]:02d}_net_returns_5bps.csv", index_col=0, parse_dates=True)
        ref = pd.concat([pd.read_csv(M6_NET, index_col=0, parse_dates=True), pd.read_csv(EW_NET, index_col=0, parse_dates=True)], axis=1)
        lstm_diff = {c: float(np.abs(net[(c, 5.0)] - t20[c].to_numpy()).max()) for c in LSTMS}
        erc_diff = {c: float((pd.Series(net[(c, 5.0)], index=dates).reindex(ref.index) - ref[c]).abs().max())
                    for c in ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]}
        record["check"] = {"lstm_worst_diff_vs_trial20": lstm_diff, "erc_worst_diff_vs_m6": erc_diff,
                           "passes": all(v <= 1e-7 for v in lstm_diff.values()) and all(v <= 1e-8 for v in erc_diff.values())}

    rf = load_risk_free_daily(str(RF_DIR), dates)
    stages = stage_labels(dates, cal).to_numpy()
    rows = []
    for stage, name in STAGES.items():
        m = stages == stage
        in_stage = set(dates[m])
        for c in REPORTED:
            rebal = [b for b in run.binding_constraints_history[c] if b["date"] in in_stage]
            rows.append({"kind": kind, "label": label, "stage": name, "config": c,
                         **{f"sharpe_{int(f)}bps": sharpe_ratio(net[(c, f)][m], risk_free=rf[m]) for f in FEES},
                         "cagr": cagr(net[(c, 5.0)][m]), "max_drawdown": max_drawdown(net[(c, 5.0)][m]),
                         "mean_turnover_per_rebalance": float(np.mean([b["turnover"] for b in rebal]))})
        if kind == "ensemble":
            for other in ["lstm_regime", "lstm_baseline"]:
                ci = block_bootstrap_diff_ci(net[(NEW, 5.0)][m] - rf[m], net[(other, 5.0)][m] - rf[m], n_boot=2000, random_state=0)
                record.setdefault("bootstrap", {})[f"{name}: {NEW} minus {other}"] = ci
    pd.DataFrame(rows).to_csv(RUNS / f"{tag}_metrics.csv", index=False, float_format="%.6g")
    (RUNS / f"{tag}_record.json").write_text(json.dumps(record, indent=2))
    print(f"{tag}: done", json.dumps(record.get("check", {})))


def merge() -> None:
    VAL = STAGES["validation"]
    t20 = pd.read_csv(OUT / "asset_regime_lstm_per_seed.csv")
    replays = {s: json.loads((RUNS / f"replay_{s}_record.json").read_text())["check"] for s in SEEDS}
    checks_pass = all(c["passes"] for c in replays.values())
    metrics = pd.concat([pd.read_csv(f, dtype={"label": str}) for f in sorted(RUNS.glob("*_metrics.csv"))], ignore_index=True)
    metrics.to_csv(OUT / "asset_regime_lstm_ensemble_metrics.csv", index=False, float_format="%.6g")
    cols = ["sharpe_5bps", "sharpe_10bps", "sharpe_25bps", "cagr", "max_drawdown", "mean_turnover_per_rebalance"]

    rows = []
    for label, members in ENSEMBLES.items():
        for stage in STAGES.values():
            for c in LSTMS:
                e = metrics[(metrics.kind == "ensemble") & (metrics.label == label) & (metrics.stage == stage) & (metrics.config == c)].iloc[0]
                single = t20[(t20.seed.isin(members)) & (t20.stage == stage) & (t20.config == c)]
                rows.append({"ensemble": label, "stage": stage, "config": c,
                             **{k: e[k] for k in cols}, **{f"members_mean_{k}": single[k].mean() for k in cols}})
    ens = pd.DataFrame(rows)
    ens.to_csv(OUT / "asset_regime_lstm_ensemble_summary.csv", index=False, float_format="%.6g")
    boot = {label: json.loads((RUNS / f"ensemble_{label}_record.json").read_text())["bootstrap"] for label in ENSEMBLES}

    gates = {}
    for label in HALVES:
        e = ens[(ens.ensemble == label) & (ens.stage == VAL) & (ens.config == NEW)].iloc[0]
        gates[f"ensemble {label} >= its members' mean"] = bool(e.sharpe_5bps >= e.members_mean_sharpe_5bps)
    report = {"checks_pass": checks_pass, "replay_checks": replays, "gates": gates,
              "passes": checks_pass and all(gates.values()), "bootstrap": boot}
    (OUT / "asset_regime_lstm_ensemble_gates.json").write_text(json.dumps(report, indent=2))

    pd.set_option("display.width", 250)
    print("Replay checks pass:", checks_pass)
    print(ens[["ensemble", "stage", "config", "sharpe_5bps", "members_mean_sharpe_5bps", "sharpe_25bps", "members_mean_sharpe_25bps",
               "max_drawdown", "mean_turnover_per_rebalance", "members_mean_mean_turnover_per_rebalance"]].round(4).to_string(index=False))
    for label, b in boot.items():
        for k, v in b.items():
            print(f"  ensemble {label}, {k}: {v['point_estimate']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    print("\nGates:", json.dumps(gates, indent=2), "\nPasses:", report["passes"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--replay", action="store_true")
    group.add_argument("--members", choices=list(ENSEMBLES))
    group.add_argument("--merge", action="store_true")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.replay:
        run_job("replay", [a.seed], str(a.seed))
    else:
        run_job("ensemble", ENSEMBLES[a.members], a.members)
