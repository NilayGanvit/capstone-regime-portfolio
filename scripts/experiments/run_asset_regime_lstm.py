"""
Trial 20: the regime LSTM with each asset's own high-variance probability as its regime
inputs, run exactly as pre-registered in docs/experiments/asset_regime_lstm_preregistration.md.

--seed N   trains the baseline, regime and asset-regime networks for seed N on data through
           2014-12-31, runs the primary walk-forward with asset_regime_states=2, checks that
           the ERC and 1/N returns match the M6 reference archive (and, on seed 0, that the
           other LSTM returns are unchanged by the new network), and saves the networks,
           daily 5 bps net returns and per-stage metrics.
--merge    combines the ten seeds: means and standard errors across seeds, the gates on
           validation, the pre-registered reporting.

Usage:
    python scripts/experiments/run_asset_regime_lstm.py --seed 0   # one process per seed, 0-9
    python scripts/experiments/run_asset_regime_lstm.py --merge
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
from evaluation import cagr, max_drawdown, net_of_cost_returns, sharpe_ratio  # noqa: E402
from run_real_data import (  # noqa: E402
    LSTM_BUDGET_FLOOR, LSTM_COV_WINDOW, LSTM_N_EPOCHS, LSTM_N_STATES, LSTM_SEQ_LEN, run_primary_walk_forward,
)

OUT = ROOT / "outputs" / "experiments"
SEED_DIR = OUT / "asset_regime_lstm"
SEEDS = list(range(10))
FEES = [5.0, 10.0, 25.0]
NEW = "lstm_asset_regime"
REPORTED = [NEW, "lstm_baseline", "lstm_regime", "erc_asset_regime", "equal_weight"]
ERC_CHECKED = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
RF_DIR = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
M6_NET = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d" / "csv" / "real_data_rq_net_returns_5bps.csv"
EW_NET = ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv"
STAGES = {"validation": "validation 2015-2018", "final_test": "final test 2019-2026 (exploratory)"}


def run_seed(seed: int) -> None:
    import torch
    from lstm_training import save_checkpoint, train_lstm_variants

    torch.set_num_threads(1)
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    cal = M2Calendar()
    iw = initial_window_length(ds.returns.index, cal)
    models, fm = train_lstm_variants(
        ds.returns, iw, seed=seed, budget_floor=LSTM_BUDGET_FLOOR, n_epochs=LSTM_N_EPOCHS, seq_len=LSTM_SEQ_LEN,
        cov_window=LSTM_COV_WINDOW, n_states=LSTM_N_STATES, asset_regime_states=2,
    )
    for variant, model in models.items():
        save_checkpoint(model, SEED_DIR / f"seed_{seed:02d}_{variant}.pt", seed=seed, variant=variant, epochs=LSTM_N_EPOCHS)
    run = run_primary_walk_forward(ds, iw, models, fm, asset_regime_states=2)

    dates = pd.DatetimeIndex(run.dates)
    net = {(c, f): net_of_cost_returns(run.dates, run.portfolio_returns[c], run.execution_history[c], f)
           for c in run.portfolio_returns for f in FEES}
    ref = pd.concat([pd.read_csv(M6_NET, index_col=0, parse_dates=True), pd.read_csv(EW_NET, index_col=0, parse_dates=True)], axis=1)
    worst = {c: float((pd.Series(net[(c, 5.0)], index=dates).reindex(ref.index) - ref[c]).abs().max()) for c in ERC_CHECKED}
    checks = {"worst_daily_diff_vs_m6_archive": worst, "erc_passes": all(v <= 1e-8 for v in worst.values())}
    if seed == SEEDS[0]:
        pair = {v: models[v] for v in ["baseline", "regime"]}
        without = run_primary_walk_forward(ds, iw, pair, fm, asset_regime_states=2)
        checks["lstm_unchanged_without_new_network"] = {
            c: bool(np.array_equal(without.portfolio_returns[c], run.portfolio_returns[c])) for c in without.portfolio_returns}
    (SEED_DIR / f"seed_{seed:02d}_checks.json").write_text(json.dumps(checks, indent=2))

    pd.DataFrame({c: net[(c, 5.0)] for c in REPORTED}, index=dates.rename("date")).to_csv(
        SEED_DIR / f"seed_{seed:02d}_net_returns_5bps.csv", float_format="%.10g")
    rf = load_risk_free_daily(str(RF_DIR), dates)
    stages = stage_labels(dates, cal).to_numpy()
    rows = []
    for stage, label in STAGES.items():
        m = stages == stage
        in_stage = set(dates[m])
        for c in REPORTED:
            rebal = [b for b in run.binding_constraints_history[c] if b["date"] in in_stage]
            rows.append({"seed": seed, "stage": label, "config": c,
                         **{f"sharpe_{int(f)}bps": sharpe_ratio(net[(c, f)][m], risk_free=rf[m]) for f in FEES},
                         "cagr": cagr(net[(c, 5.0)][m]), "max_drawdown": max_drawdown(net[(c, 5.0)][m]),
                         "mean_turnover_per_rebalance": float(np.mean([b["turnover"] for b in rebal]))})
    pd.DataFrame(rows).to_csv(SEED_DIR / f"seed_{seed:02d}_metrics.csv", index=False, float_format="%.6g")
    print(f"seed {seed}: done", json.dumps(checks))


def merge() -> None:
    checks = {s: json.loads((SEED_DIR / f"seed_{s:02d}_checks.json").read_text()) for s in SEEDS}
    lstm_unchanged = checks[SEEDS[0]]["lstm_unchanged_without_new_network"]
    checks_pass = all(c["erc_passes"] for c in checks.values()) and all(lstm_unchanged.values())
    metrics = pd.concat([pd.read_csv(SEED_DIR / f"seed_{s:02d}_metrics.csv") for s in SEEDS], ignore_index=True)
    metrics.to_csv(OUT / "asset_regime_lstm_per_seed.csv", index=False, float_format="%.6g")

    cols = ["sharpe_5bps", "sharpe_10bps", "sharpe_25bps", "cagr", "max_drawdown", "mean_turnover_per_rebalance"]
    g = metrics.groupby(["stage", "config"])[cols]
    summary = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_se")).reset_index()
    summary.to_csv(OUT / "asset_regime_lstm_summary.csv", index=False, float_format="%.6g")

    diffs = []
    for stage in STAGES.values():
        p = metrics[metrics.stage == stage].pivot(index="seed", columns="config")
        for other in ["lstm_baseline", "lstm_regime"]:
            for f in FEES:
                d = p[(f"sharpe_{int(f)}bps", NEW)] - p[(f"sharpe_{int(f)}bps", other)]
                diffs.append({"stage": stage, "comparison": f"{NEW} minus {other}", "fee_bps": f,
                              "mean": float(d.mean()), "se": float(d.sem()), "seeds_at_least_0": int((d >= 0).sum())})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(OUT / "asset_regime_lstm_differences.csv", index=False, float_format="%.6g")

    v = diffs[(diffs.stage == STAGES["validation"]) & (diffs.fee_bps == 5.0)].set_index("comparison")["mean"]
    dd = summary[summary.stage == STAGES["validation"]].set_index("config")["max_drawdown_mean"]
    gates = {"G1 mean Sharpe >= lstm_baseline": bool(v[f"{NEW} minus lstm_baseline"] >= 0),
             "G2 mean Sharpe >= lstm_regime": bool(v[f"{NEW} minus lstm_regime"] >= 0),
             "G3 mean drawdown within 2 pp of lstm_baseline": bool(dd[NEW] >= dd["lstm_baseline"] - 0.02)}
    report = {"checks_pass": checks_pass, "checks": checks, "gates": gates, "passes": checks_pass and all(gates.values())}
    (OUT / "asset_regime_lstm_gates.json").write_text(json.dumps(report, indent=2))

    pd.set_option("display.width", 220)
    print("Checks pass:", checks_pass, "| LSTM unchanged on seed 0:", lstm_unchanged)
    for stage in STAGES.values():
        print(f"\n=== {stage}: mean (se) across {len(SEEDS)} seeds, net, Sharpe in excess of the risk-free rate ===")
        print(summary[summary.stage == stage].drop(columns="stage").round(4).to_string(index=False))
        print(diffs[diffs.stage == stage].drop(columns="stage").round(4).to_string(index=False))
    print("\nPer-seed validation Sharpe, 5 bps:")
    print(metrics[metrics.stage == STAGES["validation"]].pivot(index="seed", columns="config", values="sharpe_5bps").round(3).to_string())
    print("\nGates:", json.dumps(gates, indent=2), "\nPasses:", report["passes"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    merge() if args.merge else run_seed(args.seed)
