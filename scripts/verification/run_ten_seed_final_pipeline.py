"""
The RQ2 distribution across ten LSTM seeds on the final pipeline, run exactly as pre-registered
in docs/verification/ten_seed_final_pipeline_preregistration.md.

  --seed N            trains the baseline and regime networks for seed N with every setting of
                      scripts/run_real_data.py, runs the primary walk-forward, and saves the
                      networks, the daily 5 bps net returns, the final-test RQ1-RQ3 bootstrap,
                      the cost table and the validation-period Sharpe (seed 123 is the control)
  --merge             the checks against the M6 reference and the frozen ten-seed archive, then
                      the distribution across seeds 0-9
  --smoke             with --seed or --merge: one training epoch, written to a separate folder;
                      checks the code path only, never a result

LSTM training is not reproducible across machines: results count only from the M6 reference
run's Colab environment (see the pre-registration).

Usage:
    python scripts/verification/run_ten_seed_final_pipeline.py --seed 123
    python scripts/verification/run_ten_seed_final_pipeline.py --seed 0
    python scripts/verification/run_ten_seed_final_pipeline.py --merge
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from data import M2Calendar, PriceDataset, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import net_of_cost_returns, sharpe_ratio, stationary_bootstrap_rq_contrasts  # noqa: E402
from run_provenance import environment, file_sha256, git_state  # noqa: E402
from run_real_data import (  # noqa: E402
    FEE_BPS_SENSITIVITY, LSTM_BUDGET_FLOOR, LSTM_COV_WINDOW, LSTM_N_EPOCHS, LSTM_N_STATES, LSTM_SEQ_LEN,
    RISK_BUDGET_SOLVER, cost_sensitivity_table, run_primary_walk_forward,
)

OUT = ROOT / "outputs" / "verification" / "ten_seed_final_pipeline"
SEEDS = list(range(10))
CONTROL = 123
RQ_CONFIGS = ["erc_baseline", "erc_regime", "erc_blend", "lstm_baseline", "lstm_regime", "lstm_blend"]
CONTRASTS = ["rq1_erc_regime_effect", "rq1_lstm_regime_effect", "rq2_difference_in_differences",
             "rq3_erc_reliability_effect", "rq3_lstm_reliability_effect"]
RF_DIR = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
M6 = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d"
FROZEN_SEEDS = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "ten_seed"
REPLAY = ROOT / "outputs" / "ten_seed_corrected"


def out_dir(smoke: bool) -> Path:
    d = OUT / "smoke" if smoke else OUT
    d.mkdir(parents=True, exist_ok=True)
    return d


def settings(seed: int, epochs: int) -> dict:
    """run_real_data.py's own manifest settings, with the seed and epochs used here."""
    cal = M2Calendar()
    return {
        "lstm_seed": seed, "lstm_budget_floor": LSTM_BUDGET_FLOOR, "lstm_epochs": epochs,
        "lstm_seq_len": LSTM_SEQ_LEN, "lstm_cov_window": LSTM_COV_WINDOW, "n_states": LSTM_N_STATES,
        "risk_budget_solver": RISK_BUDGET_SOLVER, "headline_fee_bps": 5.0, "fee_bps_sensitivity": FEE_BPS_SENSITIVITY,
        "risk_free": "FRED DGS3MO, lagged one observation, compounded over 252 periods (data.build_risk_free_daily)",
        "initial_training_end": str(cal.initial_training_end.date()), "validation_end": str(cal.validation_end.date()),
        "final_test_end": str(cal.final_test_end.date()),
    }


def rq_rows(excess: dict, seed) -> list[dict]:
    """The pipeline's RQ1-RQ3 bootstrap (run_real_data.py's call) on one set of excess returns."""
    rq = stationary_bootstrap_rq_contrasts(excess, n_boot=2000, ci=0.95, random_state=0)
    return [{"seed": seed, "contrast": k, "point_estimate": rq[k]["point_estimate"], "ci_low": rq[k]["ci_low"],
             "ci_high": rq[k]["ci_high"], "expected_block_length": rq["_bootstrap"]["contrast_block_lengths"][k]}
            for k in CONTRASTS]


def run_seed(seed: int, smoke: bool) -> None:
    from lstm_training import save_checkpoint, train_lstm_variants

    d = out_dir(smoke)
    epochs = 1 if smoke else LSTM_N_EPOCHS
    code = git_state(ROOT)
    ds = PriceDataset.from_csv_dir(str(ROOT / "data" / "raw"))
    cal = M2Calendar()
    iw = initial_window_length(ds.returns.index, cal)
    t0 = time.time()
    models, fm = train_lstm_variants(
        ds.returns, iw, seed=seed, budget_floor=LSTM_BUDGET_FLOOR, n_epochs=epochs,
        seq_len=LSTM_SEQ_LEN, cov_window=LSTM_COV_WINDOW, n_states=LSTM_N_STATES, verbose=True,
    )
    train_seconds = time.time() - t0
    checkpoints = {}
    for variant, model in models.items():
        path = d / f"seed{seed}_{variant}.pt"
        save_checkpoint(model, path, seed=seed, variant=variant, epochs=epochs)
        checkpoints[variant] = file_sha256(path)
    t0 = time.time()
    result = run_primary_walk_forward(ds, iw, models, fm)
    walk_seconds = time.time() - t0

    dates = pd.DatetimeIndex(result.dates)
    stages = stage_labels(dates, cal)
    rf = load_risk_free_daily(str(RF_DIR), dates)
    net = {c: net_of_cost_returns(result.dates, result.portfolio_returns[c], result.execution_history[c], 5.0)
           for c in result.portfolio_returns}
    pd.DataFrame({"stage": stages.to_numpy(), **net}, index=dates.rename("date")).to_csv(
        d / f"seed{seed}_net_returns_5bps.csv", float_format="%.12g")
    final = (stages == "final_test").to_numpy()
    pd.DataFrame(rq_rows({c: net[c][final] - rf[final] for c in RQ_CONFIGS}, seed)).to_csv(
        d / f"seed{seed}_rq_bootstrap.csv", index=False, float_format="%.10g")
    cost_sensitivity_table(result, stages, rf).reset_index().assign(seed=seed).to_csv(
        d / f"seed{seed}_cost_sensitivity.csv", index=False)
    val = (stages == "validation").to_numpy()
    pd.DataFrame([{"seed": seed, "config": c, "sharpe_net_5bps": sharpe_ratio(net[c][val], risk_free=rf[val])}
                  for c in net]).to_csv(d / f"seed{seed}_validation.csv", index=False, float_format="%.10g")
    record = {"seed": seed, "smoke": smoke, "settings": settings(seed, epochs), "git": code, "environment": environment(),
              "checkpoint_sha256": checkpoints, "train_seconds": round(train_seconds, 1), "walk_seconds": round(walk_seconds, 1)}
    (d / f"seed{seed}_record.json").write_text(json.dumps(record, indent=2))
    print(f"seed {seed}: done (training {train_seconds:.0f}s, walk-forward {walk_seconds:.0f}s)")


def networks_identical(a: Path, b: Path) -> tuple[bool, float]:
    """Parameter-by-parameter comparison of two saved networks: identical, and the largest
    absolute difference."""
    import torch

    x = torch.load(a, map_location="cpu", weights_only=True)["state_dict"]
    y = torch.load(b, map_location="cpu", weights_only=True)["state_dict"]
    if x.keys() != y.keys():
        return False, float("inf")
    worst = max(float((x[k].double() - y[k].double()).abs().max()) for k in x)
    return all(torch.equal(x[k], y[k]) for k in x), worst


def distribution(per_seed: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Across seeds 0-9 for each contrast: spread, sign counts, and where seed 123 falls among
    the eleven seeds (rank 1 = lowest)."""
    rows = []
    for k in CONTRASTS:
        p = per_seed[per_seed.contrast == k]
        x = p.point_estimate.to_numpy()
        c = control[control.contrast == k].point_estimate
        c = float(c.iloc[0]) if len(c) else np.nan
        rows.append({"contrast": k, "seeds": len(x), "mean": x.mean(), "median": np.median(x),
                     "q25": np.quantile(x, 0.25), "q75": np.quantile(x, 0.75), "min": x.min(), "max": x.max(),
                     "positive": int((x > 0).sum()), "negative": int((x < 0).sum()),
                     "interval_above_zero": int((p.ci_low > 0).sum()), "interval_below_zero": int((p.ci_high < 0).sum()),
                     "seed_123": c, "seed_123_rank_of_11": int((np.append(x, c) <= c).sum()) if np.isfinite(c) else None})
    return pd.DataFrame(rows)


def merge(smoke: bool, seeds: list[int]) -> None:
    d = out_dir(smoke)
    per = pd.concat([pd.read_csv(d / f"seed{s}_rq_bootstrap.csv") for s in seeds], ignore_index=True)
    ctrl = pd.read_csv(d / f"seed{CONTROL}_rq_bootstrap.csv")

    # Check 1: the seed-123 control against the M6 reference
    m6_hash = json.loads((M6 / "run_manifest.json").read_text())["sha256"]
    rec = json.loads((d / f"seed{CONTROL}_record.json").read_text())
    hashes = {v: rec["checkpoint_sha256"][v] == m6_hash[f"checkpoint/seed123_{v}.pt"] for v in ["baseline", "regime"]}
    m6_rq = pd.read_csv(M6 / "csv" / "real_data_rq_stationary_bootstrap.csv").set_index("contrast")
    c = ctrl.set_index("contrast")
    rq_diff = float(max((c.loc[k, col] - m6_rq.loc[k, col]) ** 2 for k in CONTRASTS
                        for col in ["point_estimate", "ci_low", "ci_high"]) ** 0.5)
    control_check = {"checkpoint_hashes_match_m6": hashes, "rq_worst_abs_diff_vs_m6": rq_diff,
                     "passes": all(hashes.values()) and rq_diff <= 1e-9}

    # Check 2: seeds 0-9 against the frozen ten-seed archive and its exact-solver replay
    archive = {}
    for s in seeds:
        nets = {v: networks_identical(d / f"seed{s}_{v}.pt", FROZEN_SEEDS / f"seed_{s:02d}_{v}.pt") for v in ["baseline", "regime"]}
        entry = {f"{v}_identical": ok for v, (ok, _) in nets.items()}
        entry.update({f"{v}_worst_abs_diff": w for v, (_, w) in nets.items()})
        if all(ok for ok, _ in nets.values()):
            mine = pd.read_csv(d / f"seed{s}_net_returns_5bps.csv", index_col=0, parse_dates=True)
            ref = pd.read_csv(REPLAY / f"seed_{s:02d}_net_returns_5bps.csv", index_col=0, parse_dates=True)
            entry["replay_worst_abs_diff"] = float(max((mine[c].reindex(ref.index) - ref[f"exact_{c}"]).abs().max()
                                                       for c in ["lstm_baseline", "lstm_regime", "lstm_blend"]))
        archive[s] = entry
    identical = [s for s, e in archive.items() if e["baseline_identical"] and e["regime_identical"]]
    archive_check = {"per_seed": archive, "seeds_identical_to_archive": identical,
                     # None, not True, when no seed is identical: nothing was compared
                     "replay_within_1e-7": all(archive[s]["replay_worst_abs_diff"] <= 1e-7 for s in identical) if identical else None}

    # ERC and 1/N do not depend on the seed: confirm, then build the seed-averaged LSTM
    net = {s: pd.read_csv(d / f"seed{s}_net_returns_5bps.csv", index_col=0, parse_dates=True) for s in seeds}
    first = net[seeds[0]]
    erc_same = all(np.array_equal(net[s][c].to_numpy(), first[c].to_numpy())
                   for s in seeds for c in ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"])
    final = (first.stage == "final_test").to_numpy()
    rf = load_risk_free_daily(str(RF_DIR), first.index[final])
    averaged = {c: first[c].to_numpy()[final] for c in ["erc_baseline", "erc_regime", "erc_blend"]}
    averaged.update({c: np.mean([net[s][c].to_numpy()[final] for s in seeds], axis=0)
                     for c in ["lstm_baseline", "lstm_regime", "lstm_blend"]})
    seed_avg = pd.DataFrame(rq_rows({c: v - rf for c, v in averaged.items()}, "seed-averaged"))

    dist = distribution(per, ctrl)
    rq2 = per[per.contrast == "rq2_difference_in_differences"].point_estimate
    reading = {"rq2_depends_on_initialization": bool((rq2 > 0).any() and (rq2 < 0).any())}
    costs = pd.concat([pd.read_csv(d / f"seed{s}_cost_sensitivity.csv") for s in seeds + [CONTROL]], ignore_index=True)
    vals = pd.concat([pd.read_csv(d / f"seed{s}_validation.csv") for s in seeds + [CONTROL]], ignore_index=True)

    pd.concat([per, ctrl], ignore_index=True).to_csv(d / "ten_seed_rq_per_seed.csv", index=False, float_format="%.6g")
    dist.to_csv(d / "ten_seed_rq_distribution.csv", index=False, float_format="%.6g")
    seed_avg.to_csv(d / "ten_seed_averaged_rq.csv", index=False, float_format="%.6g")
    costs.to_csv(d / "ten_seed_cost_sensitivity.csv", index=False)
    vals.to_csv(d / "ten_seed_validation.csv", index=False, float_format="%.6g")
    report = {"smoke": smoke, "seeds": seeds, "check1_control": control_check, "check2_archive": archive_check,
              "erc_and_1n_identical_across_seeds": erc_same, "reading": reading}
    (d / "ten_seed_checks.json").write_text(json.dumps(report, indent=2, default=str))

    pd.set_option("display.width", 220)
    print(json.dumps({k: v for k, v in report.items() if k != "check2_archive"}, indent=2, default=str))
    print("Seeds identical to the archive:", identical, "| replay within 1e-7:", archive_check["replay_within_1e-7"])
    print("\nPer seed (final test, 5 bps net excess):")
    print(per.pivot(index="seed", columns="contrast", values="point_estimate")[CONTRASTS].round(3).to_string())
    print("\nDistribution over seeds 0-9:")
    print(dist.round(3).to_string(index=False))
    print("\nSeed-averaged LSTM:")
    print(seed_avg.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="with --merge --smoke only")
    a = ap.parse_args()
    if a.merge:
        merge(a.smoke, a.seeds if a.smoke else SEEDS)
    else:
        run_seed(a.seed, a.smoke)
