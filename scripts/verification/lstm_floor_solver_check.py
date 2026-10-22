"""
Does the SLSQP zero-weight failure change the validation-period LSTM budget-floor
comparison (2.5% vs 5%)?

The comparison (outputs/bounded_softmax_stage2_validation_*.csv) trained
LSTMs at both floors for seeds 0-4 and walked them forward over 2015-2018. Its
training script was never committed, so retraining would not be faithful; but the
raw budget vector the LSTM produced at every validation rebalance was saved
(outputs/bounded_softmax_stage2_validation_raw_budgets.csv). This replays those
exact vectors through the walk-forward in place of a model, under either solver
for the budget-to-weight step:

  slsqp   as the comparison was run; must reproduce the saved Sharpe ratios, which
          is the check that the replay is faithful.
  exact   the same budgets converted to weights by the exact convex solution.

Everything else (HMM, covariances, constraints, execution, costs) is the pipeline
as run. Sharpe is gross with a zero risk-free rate, the basis the saved comparison
used. Training is not repeated, so this isolates the solver.

Usage:
    python scripts/verification/lstm_floor_solver_check.py --floor 0.05 --seed 0
    python scripts/verification/lstm_floor_solver_check.py --merge
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import walkforward as wf  # noqa: E402
from constraints import ConstraintSpec  # noqa: E402
from data import M2Calendar, PriceDataset, UNIVERSE, initial_window_length  # noqa: E402
from evaluation import cagr, max_drawdown, sharpe_ratio  # noqa: E402
from features import build_feature_matrix  # noqa: E402

OUT = ROOT / "outputs"
RAW = OUT / "bounded_softmax_stage2_validation_raw_budgets.csv"
SAVED = OUT / "bounded_softmax_stage2_validation_portfolio_metrics.csv"
ZERO = 1e-4
CONFIGS = ["lstm_baseline", "lstm_regime", "lstm_blend"]


class Replay:
    def __init__(self, variant):
        self.variant = variant


def run_one(floor, seed):
    raw = pd.read_csv(RAW)
    raw = raw[(raw.floor == floor) & (raw.seed == seed)]
    seqs = {v: raw[raw.variant == v].sort_values("date") for v in ["baseline", "regime"]}
    budgets = {v: seqs[v][[f"budget_{t}" for t in UNIVERSE]].to_numpy() for v in seqs}

    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    calendar = M2Calendar()
    iw = initial_window_length(ds.returns.index, calendar)
    returns = ds.returns.loc[:calendar.validation_end]
    fm = build_feature_matrix(returns)
    co = ds.close_to_open_returns if ds.opens is not None else None
    oc = ds.open_to_close_returns if ds.opens is not None else None

    rows = []
    for solver in ["slsqp", "exact"]:
        cursor = {"baseline": 0, "regime": 0}
        zero_weight = {"baseline": 0, "regime": 0}
        orig_pred, orig_b2w = wf.predict_budgets, wf.budgets_to_weights_batch

        def pred(model, window):
            v = model.variant
            b = budgets[v][cursor[v]]
            cursor[v] += 1
            return b

        # which variant a call belongs to: baseline is converted first, then regime, once per rebalance
        calls = {"n": 0}

        def b2w_logged(budgets_batch, cov, solver="exact"):
            w = orig_b2w(budgets_batch, cov, solver=solver)
            zero_weight["baseline" if calls["n"] % 2 == 0 else "regime"] += int((w < ZERO).any())
            calls["n"] += 1
            return w

        wf.predict_budgets, wf.budgets_to_weights_batch = pred, b2w_logged
        try:
            result = wf.run_walk_forward(
                returns, n_states=3, initial_window=iw, alpha=0.97,
                constraint_spec=ConstraintSpec(lower=0.0, upper=0.30, max_turnover=0.30),
                close_to_open_returns=co, open_to_close_returns=oc,
                lstm_models={"baseline": Replay("baseline"), "regime": Replay("regime")},
                lstm_feature_matrix=fm, lstm_seq_len=60, risk_budget_solver=solver,
            )
        finally:
            wf.predict_budgets, wf.budgets_to_weights_batch = orig_pred, orig_b2w
        assert cursor == {"baseline": len(budgets["baseline"]), "regime": len(budgets["regime"])}, cursor
        for c in CONFIGS:
            r = result.portfolio_returns[c]
            rows.append({"floor": floor, "seed": seed, "solver": solver, "config": c,
                         "sharpe": sharpe_ratio(r), "cagr": cagr(r), "max_drawdown": max_drawdown(r),
                         "zero_weight_dates_baseline": zero_weight["baseline"], "zero_weight_dates_regime": zero_weight["regime"]})
    pd.DataFrame(rows).to_csv(OUT / f"lstm_floor_solver_check_shard_{floor}_{seed}.csv", index=False, float_format="%.8g")
    print(f"done floor={floor} seed={seed}")


def merge():
    shards = sorted(OUT.glob("lstm_floor_solver_check_shard_*.csv"))
    t = pd.concat([pd.read_csv(f) for f in shards], ignore_index=True)
    t.to_csv(OUT / "lstm_floor_solver_check.csv", index=False, float_format="%.8g")
    saved = pd.read_csv(SAVED)[["floor", "seed", "config", "sharpe"]].rename(columns={"sharpe": "saved_sharpe"})
    rep = t[t.solver == "slsqp"].merge(saved, on=["floor", "seed", "config"])
    print(f"{len(shards)} shards, {len(t)} rows")
    print("FAITHFULNESS: SLSQP replay vs saved stage-2 Sharpe, max |difference| = %.2e over %d (floor, seed, config) cells"
          % ((rep.sharpe - rep.saved_sharpe).abs().max(), len(rep)))

    pd.set_option("display.width", 200)
    summ = t.groupby(["solver", "floor", "config"]).sharpe.agg(["mean", "std"]).round(4).unstack("solver")
    print("\nValidation Sharpe, mean (sd) over seeds 0-4, gross, zero risk-free:")
    wide = pd.DataFrame({(s, k): summ[(k, s)] for s in ["slsqp", "exact"] for k in ["mean", "std"]})
    print(wide.to_string())

    print("\nFloor comparison (5% minus 2.5%), mean over seeds, with paired t over the 5 seeds:")
    for solver in ["slsqp", "exact"]:
        for c in CONFIGS:
            a = t[(t.solver == solver) & (t.config == c) & (t.floor == 0.05)].set_index("seed").sharpe
            b = t[(t.solver == solver) & (t.config == c) & (t.floor == 0.025)].set_index("seed").sharpe
            d = (a - b).dropna()
            print(f"  {solver:6s} {c:14s} {d.mean():+.4f}  (se {d.std(ddof=1)/np.sqrt(len(d)):.4f}, t {d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}, 5% better in {int((d > 0).sum())}/{len(d)} seeds)")

    zw = t[(t.solver == "slsqp") & (t.config == "lstm_baseline")].groupby("floor")[["zero_weight_dates_baseline", "zero_weight_dates_regime"]].sum()
    print("\nSLSQP weight vectors with an asset at zero weight (validation dates, summed over 5 seeds; 240 per cell):")
    print(zw.to_string())
    for f in shards:
        f.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run_one(a.floor, a.seed)
