"""
Crisis-fold robustness check of the M6 choices, stage 1, run exactly as pre-registered in
docs/verification/crisis_folds_preregistration.md.

  --check                        proxy hashes, the splice, and this runner's wiring: the M6 calendar with
                                 the ETF data and the M6 seed-123 checkpoints must reproduce the M6 archive;
                                 the same run gives the 2015-2018 reference values
  --fold A --erc ARM             the ERC configs and 1/N with quarterly, monthly or no HMM refits
  --fold A --seed N --floor F    trains the baseline and regime LSTMs for seed N at budget floor F on the
                                 fold's training window, then runs the fold
  --fold A --merge               the pre-registered comparisons, beside their 2015-2018 values
  --select                       the selection rule against 1/N across folds A-C
                                 (docs/verification/crisis_folds_selection_rule_preregistration.md)

Usage:
    python scripts/verification/run_crisis_folds.py --check
    python scripts/verification/run_crisis_folds.py --fold A --seed 3 --floor 0.05
    python scripts/verification/run_crisis_folds.py --fold A --merge
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from data import M2Calendar, PriceDataset, UNIVERSE, initial_window_length, load_risk_free_daily, stage_labels  # noqa: E402
from evaluation import block_bootstrap_diff_ci, cagr, max_drawdown, net_of_cost_returns, sharpe_ratio  # noqa: E402
from fold_selection import BENCHMARK, benchmark_relative_choice  # noqa: E402
from proxy_data import load_proxy_sample, spliced_returns  # noqa: E402
from run_provenance import file_sha256  # noqa: E402
from run_real_data import LSTM_COV_WINDOW, LSTM_N_EPOCHS, LSTM_N_STATES, LSTM_SEQ_LEN, run_primary_walk_forward  # noqa: E402

OUT = ROOT / "outputs" / "verification" / "crisis_folds"
PROXY = ROOT / "data" / "proxy"
RAW = ROOT / "data" / "raw"
FOLDS = {"A": ("2007-12-31", "2009-12-31"), "B": ("2009-12-31", "2014-12-31"), "C": ("2014-12-31", "2018-12-31")}
SEEDS = list(range(10))
FLOORS = {"0.05": 0.05, "0.025": 0.025}
ERC_ARMS = {"quarterly": {}, "monthly": {"refit_every_n_rebalances": 1}, "no_refit": {"refit_at_rebalance": False}}
FEES = [5.0, 10.0, 25.0]
ERC_CONFIGS = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]
LSTMS = ["lstm_baseline", "lstm_regime"]
M6 = ROOT / "outputs" / "archive" / "m6_reference_run_6a6b12d"
RF_DIR = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63" / "data"
EW_NET = ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv"
MANIFEST = ROOT / "outputs" / "experiments" / "proxy_data_manifest.json"
REFIT_REF = ROOT / "outputs" / "refit_cadence_validation_comparison.csv"
FLOOR_REF = ROOT / "outputs" / "bounded_softmax_stage1_stability_summary.csv"
# Selection-rule candidates in order of simplicity: name -> the (job, column) net-return series
# whose daily mean is the candidate's return (several entries: the seed average)
CANDIDATES = {
    "equal_weight": [("erc_quarterly", "equal_weight")],
    "erc_baseline": [("erc_quarterly", "erc_baseline")],
    "erc_regime": [("erc_quarterly", "erc_regime")],
    "erc_blend": [("erc_quarterly", "erc_blend")],
    "lstm_baseline, seed-averaged": [(f"seed_{s:02d}_floor0.05", "lstm_baseline") for s in SEEDS],
    "lstm_regime, seed-averaged": [(f"seed_{s:02d}_floor0.05", "lstm_regime") for s in SEEDS],
}


def etf_data(ds):
    return SimpleNamespace(returns=ds.returns, opens=ds.opens, close_to_open_returns=ds.close_to_open_returns,
                           open_to_close_returns=ds.open_to_close_returns)


def fold_calendar(fold):
    train_end, val_end = map(pd.Timestamp, FOLDS[fold])
    return M2Calendar(initial_training_end=train_end, validation_end=val_end, final_test_end=val_end)


def fold_setup(fold):
    """The spliced history cut at the fold's validation end, the fold's calendar (validation
    labelled 'validation', nothing after it) and its initial window."""
    ds = PriceDataset.from_csv_dir(str(RAW))
    data = etf_data(ds)
    cal = fold_calendar(fold)
    data.returns = spliced_returns(PROXY, ds.prices, ds.returns).loc[:cal.validation_end]
    return data, cal, initial_window_length(data.returns.index, cal)


def walk(data, iw, models=None, fm=None, **options):
    return run_primary_walk_forward(data, iw, models, fm, **options)


def net_returns(run, configs):
    return {(c, f): net_of_cost_returns(run.dates, run.portfolio_returns[c], run.execution_history[c], f)
            for c in configs for f in FEES}


def stage_rows(run, net, configs, mask, extra):
    dates = pd.DatetimeIndex(run.dates)
    rf = load_risk_free_daily(str(RF_DIR), dates[mask])
    in_stage = set(dates[mask])
    rows = []
    for c in configs:
        rebal = [b for b in run.binding_constraints_history[c] if b["date"] in in_stage]
        rows.append({**extra, "config": c,
                     **{f"sharpe_{int(f)}bps": sharpe_ratio(net[(c, f)][mask], risk_free=rf) for f in FEES},
                     "cagr": cagr(net[(c, 5.0)][mask]), "max_drawdown": max_drawdown(net[(c, 5.0)][mask]),
                     "mean_turnover_per_rebalance": float(np.mean([b["turnover"] for b in rebal])),
                     "turnover_binding_rebalances": int(sum(b["turnover_binding"] for b in rebal)), "rebalances": len(rebal)})
    return rows


def save(fold, tag, run, configs, cal, record):
    d = OUT / f"fold_{fold}"
    d.mkdir(parents=True, exist_ok=True)
    configs = [c for c in configs if c in run.portfolio_returns]
    net = net_returns(run, configs)
    dates = pd.DatetimeIndex(run.dates)
    pd.DataFrame({c: net[(c, 5.0)] for c in configs}, index=dates.rename("date")).to_csv(d / f"{tag}_net_returns_5bps.csv", float_format="%.10g")
    m = (stage_labels(dates, cal) == "validation").to_numpy()
    extra = {"job": tag, **{k: record.get(k) for k in ["arm", "seed", "floor"]}}
    pd.DataFrame(stage_rows(run, net, configs, m, extra)).to_csv(d / f"{tag}_metrics.csv", index=False, float_format="%.6g")
    record.update({"validation_days": int(m.sum()), "first_day": str(dates[m][0].date()), "last_day": str(dates[m][-1].date()),
                   "refits": len(run.refit_dates)})
    (d / f"{tag}_record.json").write_text(json.dumps(record, indent=2))


def check() -> None:
    import torch
    from features import build_feature_matrix
    from lstm_training import load_checkpoint

    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    hashes = {f: file_sha256(PROXY / f) == v["sha256"] for f, v in manifest["files"].items()}
    ds = PriceDataset.from_csv_dir(str(RAW))
    spliced = spliced_returns(PROXY, ds.prices, ds.returns)
    proxy, _ = load_proxy_sample(PROXY, "2000-08-30", ds.prices.index[0])
    etf_part, proxy_part = spliced.loc[ds.returns.index[0]:], spliced.loc[:ds.prices.index[0]]
    splice = {"first_return": str(spliced.index[0].date()), "last_proxy_return": str(proxy_part.index[-1].date()),
              "first_etf_return": str(etf_part.index[0].date()), "rows": len(spliced), "proxy_rows": len(proxy_part),
              "etf_part_identical": bool(etf_part.index.equals(ds.returns.index) and np.array_equal(etf_part.to_numpy(), ds.returns.to_numpy())),
              "proxy_part_identical": bool(proxy_part.index.equals(proxy.index)
                                           and np.array_equal(proxy_part.to_numpy(), proxy[list(ds.returns.columns)].to_numpy())),
              "dates_strictly_increasing": bool(spliced.index.is_monotonic_increasing and spliced.index.is_unique)}

    cal = M2Calendar()
    models = {v: load_checkpoint(M6 / "lstm_checkpoints" / f"seed123_{v}.pt") for v in ["baseline", "regime"]}
    run = walk(etf_data(ds), initial_window_length(ds.returns.index, cal), models, build_feature_matrix(ds.returns))
    configs = ERC_CONFIGS + LSTMS + ["lstm_blend"]
    net = net_returns(run, configs)
    dates = pd.DatetimeIndex(run.dates)
    ref = pd.concat([pd.read_csv(M6 / "csv" / "real_data_rq_net_returns_5bps.csv", index_col=0, parse_dates=True),
                     pd.read_csv(EW_NET, index_col=0, parse_dates=True)], axis=1)
    wiring = {c: float((pd.Series(net[(c, 5.0)], index=dates).reindex(ref.index) - ref[c]).abs().max()) for c in configs}
    tolerance = {c: 1e-7 if c.startswith("lstm") else 1e-8 for c in configs}
    validation = (stage_labels(dates, cal) == "validation").to_numpy()
    pd.DataFrame(stage_rows(run, net, configs, validation, {"job": "m6_validation_2015_2018"})).to_csv(
        OUT / "m6_validation_reference.csv", index=False, float_format="%.6g")
    report = {"proxy_hashes_match": hashes, "splice": splice, "wiring_worst_daily_diff_vs_m6": wiring, "wiring_tolerance": tolerance,
              "passes": all(hashes.values()) and splice["etf_part_identical"] and splice["proxy_part_identical"]
              and splice["dates_strictly_increasing"] and all(wiring[c] <= tolerance[c] for c in configs)}
    (OUT / "checks.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def erc_job(fold, arm) -> None:
    data, cal, iw = fold_setup(fold)
    run = walk(data, iw, **ERC_ARMS[arm])
    save(fold, f"erc_{arm}", run, ERC_CONFIGS, cal, {"arm": arm, "options": ERC_ARMS[arm], "initial_window": iw})
    print(f"fold {fold} erc_{arm}: done")


def seed_job(fold, seed, floor_tag) -> None:
    import torch
    from lstm_training import save_checkpoint, train_lstm_variants

    torch.set_num_threads(1)
    data, cal, iw = fold_setup(fold)
    models, fm = train_lstm_variants(
        data.returns, iw, seed=seed, budget_floor=FLOORS[floor_tag], n_epochs=LSTM_N_EPOCHS, seq_len=LSTM_SEQ_LEN,
        cov_window=LSTM_COV_WINDOW, n_states=LSTM_N_STATES, verbose=True,
    )
    nets = OUT / f"fold_{fold}" / "networks"
    nets.mkdir(parents=True, exist_ok=True)
    for variant, model in models.items():
        save_checkpoint(model, nets / f"seed_{seed:02d}_floor{floor_tag}_{variant}.pt", seed=seed, variant=variant, epochs=LSTM_N_EPOCHS)
    run = walk(data, iw, models, fm)
    tag = f"seed_{seed:02d}_floor{floor_tag}"
    save(fold, tag, run, LSTMS + ["equal_weight"], cal, {"seed": seed, "floor": floor_tag, "initial_window": iw})
    budgets = [{"variant": v, "date": r["date"], **{f"b_{t}": x for t, x in zip(UNIVERSE, r["budgets"])}}
               for v, hist in run.lstm_budgets_history.items() for r in hist]
    pd.DataFrame(budgets).to_csv(OUT / f"fold_{fold}" / f"{tag}_budgets.csv", index=False, float_format="%.8g")
    print(f"fold {fold} {tag}: done")


def pairwise_l1(fold, floor_tag, variant, val_dates):
    """Mean over the validation rebalance dates of the mean pairwise L1 distance between the ten
    seeds' budgets: the dispersion measure the 5% floor was chosen on."""
    cols = [f"b_{t}" for t in UNIVERSE]
    b = []
    for s in SEEDS:
        df = pd.read_csv(OUT / f"fold_{fold}" / f"seed_{s:02d}_floor{floor_tag}_budgets.csv", parse_dates=["date"])
        b.append(df[(df.variant == variant) & df.date.isin(val_dates)].sort_values("date")[cols].to_numpy())
    b = np.stack(b)  # seeds x dates x assets
    return float(np.mean([np.abs(b[i] - b[j]).sum(axis=1).mean() for i, j in combinations(range(len(SEEDS)), 2)]))


def merge(fold) -> None:
    d = OUT / f"fold_{fold}"
    checks = json.loads((OUT / "checks.json").read_text())
    metrics = pd.concat([pd.read_csv(f, dtype={"floor": str}) for f in sorted(d.glob("*_metrics.csv"))], ignore_index=True)
    metrics.to_csv(OUT / f"fold_{fold}_metrics.csv", index=False, float_format="%.6g")
    sh = lambda job, c: float(metrics[(metrics.job == job) & (metrics.config == c)].sharpe_5bps.iloc[0])  # noqa: E731
    seeds = metrics[metrics.job.str.startswith("seed_")]
    seed_sh = lambda floor, c: seeds[(seeds.floor == floor) & (seeds.config == c)].set_index("seed").sharpe_5bps.sort_index()  # noqa: E731
    nets = {a: pd.read_csv(d / f"erc_{a}_net_returns_5bps.csv", index_col=0, parse_dates=True) for a in ERC_ARMS}
    idx = nets["quarterly"].index
    val_dates = idx[(stage_labels(idx, fold_calendar(fold)) == "validation").to_numpy()]
    rf = load_risk_free_daily(str(RF_DIR), val_dates)

    def boot(a, b):
        r = block_bootstrap_diff_ci(a.loc[val_dates].to_numpy() - rf, b.loc[val_dates].to_numpy() - rf, n_boot=2000, random_state=0)
        return r["ci_low"], r["ci_high"]

    def seed_row(name, x, ref):
        return {"comparison": name, "fold": x.mean(), "se": x.sem(), "seeds_at_least_0": int((x >= 0).sum()), "m6_2015_2018": ref}

    refit_ref = pd.read_csv(REFIT_REF).query("config == 'erc_regime'").set_index("arm").sharpe
    floor_ref = pd.read_csv(FLOOR_REF).set_index(["variant", "u"]).mean_pairwise_L1
    m6 = pd.read_csv(OUT / "m6_validation_reference.csv").set_index("config").sharpe_5bps

    rows = []
    for alt in ["monthly", "no_refit"]:
        q, a = sh("erc_quarterly", "erc_regime"), sh(f"erc_{alt}", "erc_regime")
        lo, hi = boot(nets["quarterly"].erc_regime, nets[alt].erc_regime)
        rows.append({"comparison": f"refit: erc_regime quarterly minus {alt}", "fold": q - a, "ci_low": lo, "ci_high": hi,
                     "m6_2015_2018": float(refit_ref["quarterly"] - refit_ref[alt]), "holds": bool(q >= a)})
    for v in ["baseline", "regime"]:
        l5, l25 = pairwise_l1(fold, "0.05", v, set(val_dates)), pairwise_l1(fold, "0.025", v, set(val_dates))
        rows.append({"comparison": f"floor: {v} budget dispersion, 5% minus 2.5%", "fold": l5 - l25, "fold_at_5pct": l5, "fold_at_2.5pct": l25,
                     "m6_2015_2018": float(floor_ref[(v, 0.05)] - floor_ref[(v, 0.025)]), "holds": bool(l5 <= l25)})
        rows.append(seed_row(f"floor: lstm_{v} Sharpe, 5% minus 2.5% (seed-paired)", seed_sh("0.05", f"lstm_{v}") - seed_sh("0.025", f"lstm_{v}"), np.nan))
    lo, hi = boot(nets["quarterly"].erc_regime, nets["quarterly"].erc_baseline)
    rows.append({"comparison": "regime effect: erc_regime minus erc_baseline", "fold": sh("erc_quarterly", "erc_regime") - sh("erc_quarterly", "erc_baseline"),
                 "ci_low": lo, "ci_high": hi, "m6_2015_2018": float(m6["erc_regime"] - m6["erc_baseline"])})
    rows.append(seed_row("regime effect: lstm_regime minus lstm_baseline (5% floor; 2015-2018 is seed 123)",
                         seed_sh("0.05", "lstm_regime") - seed_sh("0.05", "lstm_baseline"), float(m6["lstm_regime"] - m6["lstm_baseline"])))
    ew = sh("erc_quarterly", "equal_weight")
    for c in ["erc_baseline", "erc_regime"]:
        lo, hi = boot(nets["quarterly"][c], nets["quarterly"].equal_weight)
        rows.append({"comparison": f"vs 1/N: {c}", "fold": sh("erc_quarterly", c) - ew, "ci_low": lo, "ci_high": hi,
                     "m6_2015_2018": float(m6[c] - m6["equal_weight"])})
    for c in LSTMS:
        rows.append(seed_row(f"vs 1/N: {c} (5% floor, seed mean; 2015-2018 is seed 123)", seed_sh("0.05", c) - ew, float(m6[c] - m6["equal_weight"])))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"fold_{fold}_comparisons.csv", index=False, float_format="%.6g")

    simple = np.exp(PriceDataset.from_csv_dir(str(RAW)).returns.loc[val_dates]) - 1
    monthly = (1 + simple).resample("ME").prod() - 1
    holds = table.dropna(subset=["holds"])
    report = {"fold": fold, "stage": 1, "validation": f"{val_dates[0].date()} to {val_dates[-1].date()}", "validation_days": len(val_dates),
              "checks_pass": checks["passes"], "spy_tlt_monthly_correlation": float(monthly.SPY.corr(monthly.TLT)),
              "choices_holding": {r.comparison: bool(r.holds) for r in holds.itertuples()}}
    (OUT / f"fold_{fold}_summary.json").write_text(json.dumps(report, indent=2))

    cols = ["sharpe_5bps", "sharpe_25bps", "cagr", "max_drawdown", "mean_turnover_per_rebalance"]
    pd.set_option("display.width", 250)
    print(json.dumps(report, indent=2))
    print("\nComparisons (5 bps net excess Sharpe unless stated):")
    print(table.round(4).to_string(index=False))
    print("\nERC refit arms:")
    print(metrics[metrics.job.str.startswith("erc_")][["job", "config"] + cols].round(4).to_string(index=False))
    print("\nLSTM ten-seed means by floor:")
    print(seeds.groupby(["floor", "config"])[cols].mean().round(4).to_string())


def candidate_returns(fold):
    """Each candidate's daily 5 bps net return over the fold's validation period, and its seeds'
    returns where it has several."""
    d = OUT / f"fold_{fold}"
    cache, out, members = {}, {}, {}
    for name, series in CANDIDATES.items():
        cols = []
        for job, col in series:
            if job not in cache:
                cache[job] = pd.read_csv(d / f"{job}_net_returns_5bps.csv", index_col=0, parse_dates=True)
            cols.append(cache[job][col])
        frame = pd.concat(cols, axis=1)
        out[name] = frame.mean(axis=1)
        if len(cols) > 1:
            members[name] = frame
    idx = out[BENCHMARK].index
    val = idx[(stage_labels(idx, fold_calendar(fold)) == "validation").to_numpy()]
    return {k: v.loc[val] for k, v in out.items()}, {k: v.loc[val] for k, v in members.items()}, val


def select() -> None:
    rows, seed_rows = [], []
    for fold in FOLDS:
        ret, members, val = candidate_returns(fold)
        rf = load_risk_free_daily(str(RF_DIR), val)
        ew = ret[BENCHMARK].to_numpy() - rf
        sh_ew = sharpe_ratio(ret[BENCHMARK].to_numpy(), risk_free=rf)
        for name, r in ret.items():
            if name == BENCHMARK:
                continue
            ci = block_bootstrap_diff_ci(r.to_numpy() - rf, ew, n_boot=2000, random_state=0)
            rows.append({"candidate": name, "fold": fold, "delta": sharpe_ratio(r.to_numpy(), risk_free=rf) - sh_ew,
                         "ci_low": ci["ci_low"], "ci_high": ci["ci_high"], "block": ci["expected_block_length"]})
        for name, frame in members.items():
            per_seed = np.array([sharpe_ratio(frame[c].to_numpy(), risk_free=rf) - sh_ew for c in frame.columns])
            seed_rows.append({"candidate": name, "fold": fold, "seed_median": float(np.median(per_seed)),
                              "seed_q25": float(np.quantile(per_seed, 0.25)), "seed_q75": float(np.quantile(per_seed, 0.75))})
    table = pd.DataFrame(rows)
    summary, choice = benchmark_relative_choice(table, list(CANDIDATES))
    seeds = pd.DataFrame(seed_rows)
    table.to_csv(OUT / "selection_rule_folds.csv", index=False, float_format="%.6g")
    summary.to_csv(OUT / "selection_rule_summary.csv", index=False, float_format="%.6g")
    seeds.to_csv(OUT / "selection_rule_seed_spread.csv", index=False, float_format="%.6g")
    (OUT / "selection_rule_choice.json").write_text(json.dumps({"choice": choice, "folds": list(FOLDS)}, indent=2))
    pd.set_option("display.width", 220)
    print("Sharpe minus 1/N by fold (5 bps net excess), 95% interval:")
    print(table.pivot(index="candidate", columns="fold", values=["delta", "ci_low", "ci_high"]).round(3).to_string())
    print("\nRule:")
    print(summary.round(3).to_string(index=False))
    print("\nSeed spread of each seed's own difference:")
    print(seeds.round(3).to_string(index=False))
    print("\nChoice:", choice)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--fold", choices=list(FOLDS))
    ap.add_argument("--erc", choices=list(ERC_ARMS))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--floor", choices=list(FLOORS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
    elif a.select:
        select()
    elif a.merge:
        merge(a.fold)
    elif a.erc:
        erc_job(a.fold, a.erc)
    else:
        seed_job(a.fold, a.seed, a.floor)
