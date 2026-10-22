"""
Original-versus-corrected comparison table for M6.

"Original" is the pipeline as frozen (SLSQP risk-budget solve); "corrected" is the same
checkpoints, data, constraints, execution and costs with the exact convex solve. Every figure
is for the 2019-2026 final test, net of 5 bps, with Sharpe in excess of the risk-free series
saved with the returns.

Source: the daily returns saved by scripts/verification/lstm_checkpoint_solver_replay.py in
outputs/ten_seed_corrected/ (ten archived seeds, 5% bounded-Softmax floor). LSTM rows are
mean (sd) over seeds 0-9; ERC and equal weight do not depend on the seed.

Optional: --m6-run-returns PATH adds the seed-123 reference rows, taking the corrected
series from a corrected-pipeline run's real_data_rq_net_returns_5bps.csv (e.g. the Colab
reference run) and the original series from the archived frozen run.

Usage:
    python scripts/build_comparison_table.py
    python scripts/build_comparison_table.py --m6-run-returns path/to/real_data_rq_net_returns_5bps.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import cagr, max_drawdown, sharpe_ratio  # noqa: E402

DIR = ROOT / "outputs" / "ten_seed_corrected"
ARCHIVE = ROOT / "outputs" / "archive" / "frozen_reference_5pct_bb46b63"
LABELS = {
    "erc_baseline": "ERC baseline", "erc_regime": "ERC regime", "erc_blend": "ERC blend",
    "equal_weight": "Equal weight", "lstm_baseline": "LSTM baseline", "lstm_regime": "LSTM regime", "lstm_blend": "LSTM blend",
}
ROWS = ["erc_baseline", "erc_regime", "erc_blend", "equal_weight", "lstm_baseline", "lstm_regime", "lstm_blend"]


def metrics(net, rf):
    return {"sharpe": sharpe_ratio(net, risk_free=rf), "cagr": cagr(net), "max_drawdown": max_drawdown(net)}


def ten_seed_table():
    shared = pd.read_csv(DIR / "erc_equal_weight_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
    rf = shared["rf_daily"].to_numpy()
    per = {}  # (solver, config) -> list of metric dicts over seeds
    for seed in range(10):
        df = pd.read_csv(DIR / f"seed_{seed:02d}_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
        for solver in ["slsqp", "exact"]:
            for c in ["lstm_baseline", "lstm_regime", "lstm_blend"]:
                per.setdefault((solver, c), []).append(metrics(df[f"{solver}_{c}"].to_numpy(), rf))
    for solver in ["slsqp", "exact"]:
        for c in ["erc_baseline", "erc_regime", "erc_blend", "equal_weight"]:
            per[(solver, c)] = [metrics(shared[f"{solver}_{c}"].to_numpy(), rf)]
    return per


def seed123_table(corrected_path):
    frozen_net = pd.read_csv(ARCHIVE / "csv" / "real_data_rq_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
    ew = pd.read_csv(ROOT / "outputs" / "frozen_reference_equal_weight_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")["equal_weight"]
    new = pd.read_csv(corrected_path, parse_dates=["date"]).set_index("date")
    if not new.index.equals(frozen_net.index):
        sys.exit("The corrected run's final-test dates differ from the frozen run's.")
    rf = pd.read_csv(ARCHIVE / "data" / "risk_free_DGS3MO_daily.csv", parse_dates=["date"]).set_index("date")["rf_daily"].reindex(frozen_net.index).to_numpy()
    out = {}
    for c in ["erc_baseline", "erc_regime", "erc_blend", "lstm_baseline", "lstm_regime", "lstm_blend"]:
        out[("original", c)] = metrics(frozen_net[c].to_numpy(), rf)
        out[("corrected", c)] = metrics(new[c].to_numpy(), rf)
    out[("original", "equal_weight")] = out[("corrected", "equal_weight")] = metrics(ew.reindex(frozen_net.index).to_numpy(), rf)
    return out


def fmt(vals, key, pct=False):
    x = np.array([v[key] for v in vals]) * (100 if pct else 1)
    spec = ".1f" if pct else ".3f"
    return f"{x.mean():{spec}} ({x.std(ddof=1):{spec}})" if len(x) > 1 else f"{x[0]:{spec}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m6-run-returns")
    a = ap.parse_args()
    per = ten_seed_table()

    rows = []
    for c in ROWS:
        o, e = per[("slsqp", c)], per[("exact", c)]
        rows.append({"strategy": LABELS[c], "n_seeds": len(o),
                     "sharpe_original": fmt(o, "sharpe"), "sharpe_corrected": fmt(e, "sharpe"),
                     "cagr_pct_original": fmt(o, "cagr", True), "cagr_pct_corrected": fmt(e, "cagr", True),
                     "max_drawdown_pct_original": fmt(o, "max_drawdown", True), "max_drawdown_pct_corrected": fmt(e, "max_drawdown", True)})
    table = pd.DataFrame(rows)

    def effect(solver):
        sh = lambda c: np.array([m["sharpe"] for m in per[(solver, c)]])  # noqa: E731
        dd = lambda c: np.array([m["max_drawdown"] for m in per[(solver, c)]])  # noqa: E731
        return sh("lstm_regime") - sh("lstm_baseline"), dd("lstm_regime") - dd("lstm_baseline")
    eff = {s: effect(s) for s in ["slsqp", "exact"]}
    boot = pd.read_csv(ROOT / "outputs" / "ten_seed_corrected_bootstrap.csv")
    excl = {s: int((((g.ci_low > 0) | (g.ci_high < 0))).sum())
            for s, g in boot[boot.contrast == "rq1_lstm_regime_effect"].groupby("solver")}
    summary_rows = [
        ("LSTM regime effect on Sharpe, mean (sd)", *(f"{eff[s][0].mean():+.3f} ({eff[s][0].std(ddof=1):.3f})" for s in ["slsqp", "exact"])),
        ("Seeds with a positive regime effect", *(f"{int((eff[s][0] > 0).sum())}/10" for s in ["slsqp", "exact"])),
        ("Seeds whose 95% bootstrap interval excludes zero", *(f"{excl[s]}/10" for s in ["slsqp", "exact"])),
        ("Max-drawdown improvement, regime vs baseline LSTM, mean (pp)", *(f"{eff[s][1].mean()*100:+.2f}" for s in ["slsqp", "exact"])),
        ("Seeds where the regime LSTM has the milder drawdown", *(f"{int((eff[s][1] > 0).sum())}/10" for s in ["slsqp", "exact"])),
    ]
    summary = pd.DataFrame(summary_rows, columns=["measure", "original", "corrected"])

    seed123 = None
    if a.m6_run_returns:
        s123 = seed123_table(a.m6_run_returns)
        seed123 = pd.DataFrame([{"strategy": LABELS[c], "sharpe_original": f"{s123[('original', c)]['sharpe']:.3f}", "sharpe_corrected": f"{s123[('corrected', c)]['sharpe']:.3f}",
                                 "cagr_pct_original": f"{s123[('original', c)]['cagr']*100:.1f}", "cagr_pct_corrected": f"{s123[('corrected', c)]['cagr']*100:.1f}",
                                 "max_drawdown_pct_original": f"{s123[('original', c)]['max_drawdown']*100:.1f}", "max_drawdown_pct_corrected": f"{s123[('corrected', c)]['max_drawdown']*100:.1f}"}
                                for c in ROWS])
        seed123.to_csv(ROOT / "outputs" / "original_vs_corrected_seed123.csv", index=False)

    table.to_csv(ROOT / "outputs" / "original_vs_corrected_comparison.csv", index=False)
    summary.to_csv(ROOT / "outputs" / "original_vs_corrected_summary.csv", index=False)

    def md(df):
        cols = list(df.columns)
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])

    out = ["# Original versus corrected solver", "",
           "Original: the pipeline as frozen, risk budgets turned into weights by SLSQP. Corrected: the same checkpoints, data, constraints,",
           "next-open execution and 5 bps costs, with the exact convex solve. 2019-2026 final test (an exploratory historical period, not an",
           "independent holdout); Sharpe in excess of the FRED DGS3MO risk-free series. LSTM rows are mean (sd) over the ten archived seeds",
           "(5% bounded-Softmax floor); ERC and equal weight do not depend on the seed. Seeds share one market history.", "",
           "Generated by `scripts/build_comparison_table.py` from `outputs/ten_seed_corrected/`.", "",
           "## Performance", "", md(table.rename(columns={
               "sharpe_original": "Sharpe, original", "sharpe_corrected": "Sharpe, corrected", "cagr_pct_original": "CAGR %, original",
               "cagr_pct_corrected": "CAGR %, corrected", "max_drawdown_pct_original": "Max DD %, original", "max_drawdown_pct_corrected": "Max DD %, corrected"})),
           "", "## Regime effect and drawdown", "", md(summary), ""]
    if seed123 is not None:
        out += ["## Seed-123 reference run", "", md(seed123), ""]
    (ROOT / "docs" / "original_vs_corrected_comparison.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
