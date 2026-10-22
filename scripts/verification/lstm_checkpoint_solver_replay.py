"""
Effect of the budget-to-weight solver on the frozen LSTM results, final test.

Replays the frozen reference's ten-seed experiment (outputs/archive/
frozen_reference_5pct_bb46b63/ten_seed: seeds 0-9, 5% bounded-Softmax floor, baseline
and regime checkpoints) through the full walk-forward under either solver for the
LSTM's budget-to-weight step:

  slsqp   as the frozen run was made; must reproduce the archived daily net returns
          and Sharpe ratios, which checks that the replay is faithful.
  exact   the same checkpoints, hence the same learned budgets, converted to weights by
          the exact convex solution (allocation_erc.solve_risk_budget_convex).

The checkpoints are only loaded for inference, so nothing is retrained and the learned
budgets do not change. Inference reproduces across environments to about 1e-9; it is
training that does not. Everything else (HMM, covariances, constraints, next-open
execution, 5 bps costs, archived risk-free series) is the pipeline as run.

Reported for the 2019-2026 final test: net excess Sharpe, max drawdown, the regime
effect, and how often the raw weight vector has an asset at zero weight although every
learned budget is at least the floor.

Usage:
    python scripts/verification/lstm_checkpoint_solver_replay.py --seed 0
    python scripts/verification/lstm_checkpoint_solver_replay.py --merge
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import walkforward as wf  # noqa: E402
from allocation_lstm import RiskBudgetLSTM  # noqa: E402
from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import cagr, max_drawdown, net_of_cost_returns, sharpe_ratio  # noqa: E402
from features import build_feature_matrix  # noqa: E402
from run_real_data import run_primary_walk_forward  # noqa: E402

ARCHIVE = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63"
OUT = ROOT / "outputs"
CONFIGS = ["lstm_baseline", "lstm_regime", "lstm_blend"]
ZERO = 1e-4


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = RiskBudgetLSTM(n_features=ck["n_features"], n_assets=ck["n_assets"], hidden_size=32,
                           random_state=None, budget_floor=ck["budget_floor"])
    model.load_state_dict(ck["state_dict"])
    return model.double().eval()


def run_one(seed):
    torch.set_num_threads(1)
    models = {v: load_model(ARCHIVE / "ten_seed" / f"seed_{seed:02d}_{v}.pt") for v in ["baseline", "regime"]}
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    calendar = M2Calendar()
    iw = initial_window_length(ds.returns.index, calendar)
    fm = build_feature_matrix(ds.returns)
    archived_net = pd.read_csv(ARCHIVE / "ten_seed" / f"seed_{seed:02d}_net_returns.csv", parse_dates=["date"]).set_index("date")
    archived_metrics = pd.read_csv(ARCHIVE / "ten_seed" / f"seed_{seed:02d}_metrics.csv").set_index("config")

    rows = []
    orig_b2w = wf.budgets_to_weights_batch
    for solver in ["slsqp", "exact"]:
        zero_flags = []

        def b2w(budgets_batch, cov, solver="exact"):
            w = orig_b2w(budgets_batch, cov, solver=solver)
            zero_flags.append(bool((w < ZERO).any()))
            return w

        wf.budgets_to_weights_batch = b2w
        try:
            result = run_primary_walk_forward(ds, iw, models, fm, risk_budget_solver=solver)
        finally:
            wf.budgets_to_weights_batch = orig_b2w

        dates = pd.DatetimeIndex(result.dates)
        stages = stage_labels(dates, calendar).to_numpy()
        final = stages == "final_test"
        rf = load_risk_free_daily(str(ARCHIVE / "data"), dates[final])
        rebal = [pd.Timestamp(e["date"]) for e in result.binding_constraints_history["erc_baseline"]]
        rebal_final = np.array([d in set(dates[final]) for d in rebal])
        flags = np.array(zero_flags).reshape(-1, 2)  # per rebalance: [baseline, regime]
        for c in CONFIGS:
            net = net_of_cost_returns(result.dates, result.portfolio_returns[c], result.execution_history[c], 5.0)[final]
            row = {"seed": seed, "solver": solver, "config": c,
                   "sharpe_excess": sharpe_ratio(net, risk_free=rf), "cagr": cagr(net), "max_drawdown": max_drawdown(net),
                   "zero_weight_share_baseline": float(flags[rebal_final, 0].mean()),
                   "zero_weight_share_regime": float(flags[rebal_final, 1].mean())}
            if solver == "slsqp":
                row["max_abs_daily_diff_vs_archive"] = float(np.abs(net - archived_net[c].to_numpy()).max())
                row["sharpe_diff_vs_archive"] = float(row["sharpe_excess"] - archived_metrics.loc[c, "sharpe_net_excess"])
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / f"lstm_checkpoint_solver_replay_shard_{seed}.csv", index=False, float_format="%.8g")
    print(f"done seed {seed}")


def merge():
    shards = sorted(OUT.glob("lstm_checkpoint_solver_replay_shard_*.csv"))
    t = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    t.to_csv(OUT / "lstm_checkpoint_solver_replay.csv", index=False, float_format="%.8g")
    s = t[t.solver == "slsqp"]
    print(f"{len(shards)} seeds")
    print("FAITHFULNESS (SLSQP replay vs archived): max |daily net return difference| = %.2e; max |Sharpe difference| = %.2e"
          % (s.max_abs_daily_diff_vs_archive.max(), s.sharpe_diff_vs_archive.abs().max()))

    def wide(solver, col):
        x = t[t.solver == solver].pivot(index="seed", columns="config", values=col)
        return x
    res = {}
    for solver in ["slsqp", "exact"]:
        sh, dd = wide(solver, "sharpe_excess"), wide(solver, "max_drawdown")
        res[solver] = pd.DataFrame({
            "baseline Sharpe": sh.lstm_baseline, "regime Sharpe": sh.lstm_regime, "blend Sharpe": sh.lstm_blend,
            "regime effect": sh.lstm_regime - sh.lstm_baseline, "baseline MDD": dd.lstm_baseline, "regime MDD": dd.lstm_regime,
            "MDD improvement": dd.lstm_regime - dd.lstm_baseline,
        })
    pd.set_option("display.width", 200)
    print("\nFinal test 2019-2026, net of 5 bps, excess over the risk-free rate; mean (sd) over seeds 0-9:")
    print(pd.DataFrame({s: [f"{res[s][c].mean():+.4f} ({res[s][c].std():.4f})" for c in res[s].columns] for s in res}, index=res["slsqp"].columns).to_string())
    print("\nPaired difference, exact minus SLSQP (same checkpoints, same budgets):")
    for c in res["slsqp"].columns:
        d = res["exact"][c] - res["slsqp"][c]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {c:16s} {d.mean():+.4f}  (se {se:.4f}, t {d.mean()/se:+.2f}; higher under exact in {int((d > 0).sum())}/{len(d)} seeds)")
    for solver in ["slsqp", "exact"]:
        e = res[solver]["regime effect"]
        m = res[solver]["MDD improvement"]
        print(f"  {solver:6s}: regime effect positive in {int((e > 0).sum())}/10 seeds, mean {e.mean():+.4f}; regime MDD less severe in {int((m > 0).sum())}/10 seeds, mean {m.mean()*100:+.2f} pp")
    z = s[s.config == "lstm_baseline"][["zero_weight_share_baseline", "zero_weight_share_regime"]]
    print("\nShare of final-test rebalances whose SLSQP weight vector has an asset at zero weight (mean over seeds, [min, max]):")
    print(f"  baseline LSTM {z.zero_weight_share_baseline.mean():.1%} [{z.zero_weight_share_baseline.min():.1%}, {z.zero_weight_share_baseline.max():.1%}]"
          f"   regime LSTM {z.zero_weight_share_regime.mean():.1%} [{z.zero_weight_share_regime.min():.1%}, {z.zero_weight_share_regime.max():.1%}]")
    for f in shards:
        f.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run_one(a.seed)
