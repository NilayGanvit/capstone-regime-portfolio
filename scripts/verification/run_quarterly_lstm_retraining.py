"""
Fixed versus quarterly-retrained LSTM on the final pipeline, run exactly as pre-registered in
docs/verification/quarterly_lstm_retraining_preregistration.md.

Trains seed 123 once with every setting of scripts/run_real_data.py, then runs the primary
walk-forward twice from separate copies of those networks:
    fixed       the M6 reference specification (no LSTM retraining)
    quarterly   lstm_retrain_every_n_rebalances=3, warm-started on the expanding window
The fixed arm must reproduce the M6 archive; then each arm's performance, turnover and RQ
contrasts are reported, with quarterly-minus-fixed bootstrap intervals.

Writes only into a new folder, outputs/verification/quarterly_lstm_retraining/<run id>/. It
refuses to start if that folder exists and never writes a file that already exists. Nothing
else in the repository is written.

Usage:
    python scripts/verification/run_quarterly_lstm_retraining.py
    python scripts/verification/run_quarterly_lstm_retraining.py --smoke   # one epoch; code path only
"""

import argparse
import copy
import functools
import json
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import run_real_data  # noqa: E402
from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import (  # noqa: E402
    block_bootstrap_diff_ci, cagr, max_drawdown, net_of_cost_returns, sharpe_ratio, stationary_bootstrap_rq_contrasts,
)
from run_provenance import environment, file_sha256, git_state  # noqa: E402

OUT_ROOT = ROOT / "outputs" / "verification" / "quarterly_lstm_retraining"
SEED = 123
EVERY = 3
FEES = [5.0, 10.0, 25.0]
LSTMS = ["lstm_baseline", "lstm_regime", "lstm_blend"]
RQ_CONFIGS = ["erc_baseline", "erc_regime", "erc_blend"] + LSTMS
CONTRASTS = ["rq1_erc_regime_effect", "rq1_lstm_regime_effect", "rq2_difference_in_differences",
             "rq3_erc_reliability_effect", "rq3_lstm_reliability_effect"]
RF_DIR = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
M6 = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d"
EW_NET = ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv"
STAGES = {"validation": "validation 2015-2018", "final_test": "final test 2019-2026"}


class NoOverwrite:
    """Every output path is created inside one new folder, and only if it does not exist yet."""

    def __init__(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=False)
        self.folder = folder

    def path(self, name: str) -> Path:
        p = self.folder / name
        if p.exists():
            raise FileExistsError(f"refusing to overwrite {p}")
        return p

    def csv(self, name: str, frame: pd.DataFrame, **kwargs) -> None:
        with self.path(name).open("x") as fh:
            frame.to_csv(fh, **kwargs)

    def json(self, name: str, obj) -> None:
        with self.path(name).open("x") as fh:
            json.dump(obj, fh, indent=2, default=str)


@contextmanager
def walk_forward_options(**options):
    """Call run_real_data.run_primary_walk_forward unchanged, with extra run_walk_forward
    options added, so every other argument is exactly the reference run's."""
    original = run_real_data.run_walk_forward
    run_real_data.run_walk_forward = functools.partial(original, **options)
    try:
        yield
    finally:
        run_real_data.run_walk_forward = original


def arm_tables(name, result, rf, stages):
    net = {(c, f): net_of_cost_returns(result.dates, result.portfolio_returns[c], result.execution_history[c], f)
           for c in result.portfolio_returns for f in FEES}
    rows = []
    for stage, label in STAGES.items():
        m = (stages == stage).to_numpy()
        days = set(pd.DatetimeIndex(result.dates)[m])
        for c in LSTMS:
            rebal = [b for b in result.binding_constraints_history[c] if b["date"] in days]
            rows.append({"arm": name, "period": label, "config": c,
                         **{f"sharpe_net_{int(f)}bps": sharpe_ratio(net[(c, f)][m], risk_free=rf[m]) for f in FEES},
                         **{f"cagr_net_{int(f)}bps": cagr(net[(c, f)][m]) for f in FEES},
                         "max_drawdown_net_5bps": max_drawdown(net[(c, 5.0)][m]),
                         "mean_turnover_per_rebalance": float(np.mean([b["turnover"] for b in rebal])),
                         "turnover_binding_rebalances": int(sum(b["turnover_binding"] for b in rebal)),
                         "rebalances": len(rebal),
                         "lstm_retrains_in_period": sum(1 for d in result.lstm_retrain_dates if d in days)})
    return net, rows


def main(smoke: bool) -> None:
    import torch
    from lstm_training import save_checkpoint, train_lstm_variants

    run_id = ("smoke_" if smoke else "") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
    out = NoOverwrite(OUT_ROOT / run_id)
    code = git_state(ROOT)
    epochs = 1 if smoke else run_real_data.LSTM_N_EPOCHS
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    cal = M2Calendar()
    iw = initial_window_length(ds.returns.index, cal)

    t0 = time.time()
    models, fm = train_lstm_variants(
        ds.returns, iw, seed=SEED, budget_floor=run_real_data.LSTM_BUDGET_FLOOR, n_epochs=epochs,
        seq_len=run_real_data.LSTM_SEQ_LEN, cov_window=run_real_data.LSTM_COV_WINDOW,
        n_states=run_real_data.LSTM_N_STATES, verbose=True,
    )
    timing = {"initial_training_seconds": round(time.time() - t0, 1)}
    hashes = {}
    for variant, model in models.items():
        p = out.path(f"seed{SEED}_{variant}.pt")
        save_checkpoint(model, p, seed=SEED, variant=variant, epochs=epochs)
        hashes[variant] = file_sha256(p)

    results = {}
    t0 = time.time()
    results["fixed"] = run_real_data.run_primary_walk_forward(ds, iw, copy.deepcopy(models), fm)
    timing["fixed_walk_seconds"] = round(time.time() - t0, 1)
    print(f"fixed arm done in {timing['fixed_walk_seconds']}s", flush=True)
    quarterly_options = {"lstm_retrain_every_n_rebalances": EVERY}
    if smoke:
        quarterly_options["lstm_retrain_n_epochs"] = 1
    t0 = time.time()
    with walk_forward_options(**quarterly_options):
        results["quarterly"] = run_real_data.run_primary_walk_forward(ds, iw, copy.deepcopy(models), fm)
    timing["quarterly_walk_seconds"] = round(time.time() - t0, 1)
    print(f"quarterly arm done in {timing['quarterly_walk_seconds']}s, "
          f"{len(results['quarterly'].lstm_retrain_dates)} retrains", flush=True)

    dates = pd.DatetimeIndex(results["fixed"].dates)
    stages = stage_labels(dates, cal)
    rf = load_risk_free_daily(str(RF_DIR), dates)
    final = (stages == "final_test").to_numpy()
    nets, rows, rq_rows = {}, [], []
    for name, result in results.items():
        nets[name], r = arm_tables(name, result, rf, stages)
        rows += r
        out.csv(f"{name}_net_returns_5bps.csv",
                pd.DataFrame({"stage": stages.to_numpy(), **{c: nets[name][(c, 5.0)] for c in result.portfolio_returns}},
                             index=dates.rename("date")), float_format="%.12g")
        rq = stationary_bootstrap_rq_contrasts({c: nets[name][(c, 5.0)][final] - rf[final] for c in RQ_CONFIGS},
                                               n_boot=2000, ci=0.95, random_state=0)
        rq_rows += [{"arm": name, "contrast": k, "point_estimate": rq[k]["point_estimate"],
                     "ci_low": rq[k]["ci_low"], "ci_high": rq[k]["ci_high"]} for k in CONTRASTS]
    out.json("quarterly_retrain_dates.json", [str(pd.Timestamp(d).date()) for d in results["quarterly"].lstm_retrain_dates])

    diff_rows = []
    for stage, label in STAGES.items():
        m = (stages == stage).to_numpy()
        for c in LSTMS:
            ci = block_bootstrap_diff_ci(nets["quarterly"][(c, 5.0)][m] - rf[m], nets["fixed"][(c, 5.0)][m] - rf[m],
                                         n_boot=2000, random_state=0)
            diff_rows.append({"period": label, "config": c, "quarterly_minus_fixed": ci["point_estimate"],
                              "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]})

    # Check: the fixed arm is the M6 reference run
    ref = pd.concat([pd.read_csv(M6 / "csv" / "real_data_rq_net_returns_5bps.csv", index_col=0, parse_dates=True),
                     pd.read_csv(EW_NET, index_col=0, parse_dates=True)], axis=1)
    worst = {c: float((pd.Series(nets["fixed"][(c, 5.0)], index=dates).reindex(ref.index) - ref[c]).abs().max())
             for c in RQ_CONFIGS + ["equal_weight"]}
    tol = {c: 1e-7 if c.startswith("lstm") else 1e-8 for c in worst}
    m6_hash = json.loads((M6 / "run_manifest.json").read_text())["sha256"]
    hash_match = {v: hashes[v] == m6_hash[f"checkpoint/seed123_{v}.pt"] for v in hashes}
    check = {"fixed_arm_worst_daily_diff_vs_m6": worst, "tolerance": tol, "checkpoint_hashes_match_m6": hash_match,
             "passes": all(worst[c] <= tol[c] for c in worst) and all(hash_match.values())}

    table = pd.DataFrame(rows)
    out.csv("lstm_fixed_vs_quarterly_performance.csv", table, index=False, float_format="%.6g")
    out.csv("lstm_quarterly_minus_fixed_bootstrap.csv", pd.DataFrame(diff_rows), index=False, float_format="%.6g")
    out.csv("rq_contrasts_by_arm.csv", pd.DataFrame(rq_rows), index=False, float_format="%.6g")
    out.json("record.json", {"run_id": run_id, "smoke": smoke, "seed": SEED, "retrain_every_n_rebalances": EVERY,
                             "quarterly_options": quarterly_options, "initial_epochs": epochs,
                             "n_quarterly_retrains": len(results["quarterly"].lstm_retrain_dates),
                             "checkpoint_sha256": hashes, "check": check, "timing": timing,
                             "git": code, "environment": environment()})

    pd.set_option("display.width", 220)
    print("\nRun folder:", out.folder)
    print("Check (fixed arm reproduces M6):", json.dumps(check, indent=2))
    cols = ["arm", "period", "config", "sharpe_net_5bps", "sharpe_net_25bps", "cagr_net_5bps", "max_drawdown_net_5bps",
            "mean_turnover_per_rebalance", "turnover_binding_rebalances", "lstm_retrains_in_period"]
    print(table[cols].round(4).to_string(index=False))
    print(pd.DataFrame(diff_rows).round(4).to_string(index=False))
    print(pd.DataFrame(rq_rows).pivot(index="contrast", columns="arm", values="point_estimate").round(4).to_string())
    print("Timing:", timing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one training epoch and one retraining epoch; code path only")
    main(ap.parse_args().smoke)
