"""
Research-question contrasts and a DSR reading for each of the ten archived seeds, on the
original (SLSQP) and corrected (exact) solver, from the daily net returns saved by
lstm_checkpoint_solver_replay.py (outputs/ten_seed_corrected/).

For each solver and seed, the six strategy series (ERC baseline/regime/blend from the
ERC-only run, LSTM baseline/regime/blend from that seed's checkpoints) are put in excess of
the archived risk-free series and passed to the same paired stationary bootstrap the
pipeline uses (2,000 replications, automatic contrast-specific block length), giving the
five contrasts: RQ1 ERC and LSTM regime effects, RQ2 difference-in-differences, RQ3 ERC and
LSTM reliability effects.

Summaries are across seeds: the spread of point estimates and how many seeds have an
interval that excludes zero. Seeds share one market history, so they are not independent
evidence about the strategy; they show how much the answer depends on LSTM initialisation.

DSR: for each seed's regime LSTM, the search-adjusted DSR-L at two reference assumptions from
the grid (the pooled empirical cross-trial variance and the conventional unit variance, both at
the literal logged trial count), to show the range rather than a single number.

Usage:
    python scripts/verification/ten_seed_corrected_bootstrap.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from evaluation import (  # noqa: E402
    build_dsr_sensitivity_scenarios, dsr_grid_dataframe, stationary_bootstrap_rq_contrasts,
)

DIR = ROOT / "outputs" / "ten_seed_corrected"
OUT = ROOT / "outputs"
SOLVERS = ["slsqp", "exact"]
ERC = ["erc_baseline", "erc_regime", "erc_blend"]
LSTM = ["lstm_baseline", "lstm_regime", "lstm_blend"]


def main() -> None:
    shared = pd.read_csv(DIR / "erc_equal_weight_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
    variance, ks = build_dsr_sensitivity_scenarios(str(ROOT / "outputs" / "trial_log.csv"),
                                                   str(ROOT / "outputs" / "trial_candidate_sharpes.csv"))
    pooled_name = "pooled_deduped_all_trials"
    k_name = "literal_logged_trials"

    rows, dsr_rows = [], []
    for seed in range(10):
        seed_df = pd.read_csv(DIR / f"seed_{seed:02d}_net_returns_5bps.csv", parse_dates=["date"]).set_index("date")
        assert seed_df.index.equals(shared.index)
        rf = seed_df["rf_daily"].to_numpy()
        for solver in SOLVERS:
            excess = {c: shared[f"{solver}_{c}"].to_numpy() - rf for c in ERC}
            excess.update({c: seed_df[f"{solver}_{c}"].to_numpy() - rf for c in LSTM})
            ci = stationary_bootstrap_rq_contrasts(excess, n_boot=2000, ci=0.95, random_state=0)
            for name, v in ci.items():
                if name.startswith("_"):
                    continue
                rows.append({"seed": seed, "solver": solver, "contrast": name, "point": v["point_estimate"],
                             "ci_low": v["ci_low"], "ci_high": v["ci_high"]})
            grid = dsr_grid_dataframe({"lstm_regime": excess["lstm_regime"]}, variance, ks)
            sel = lambda vn: float(grid[(grid.variance_scenario == vn) & (grid.k_scenario == k_name)].dsr_l.iloc[0])  # noqa: E731
            dsr_rows.append({"seed": seed, "solver": solver, "sharpe_excess": float(grid.sharpe.iloc[0]),
                             "dsr_l_empirical_variance": sel(pooled_name), "dsr_l_unit_variance": sel("conventional_unit_variance")})
        print(f"seed {seed} done", flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "ten_seed_corrected_bootstrap.csv", index=False, float_format="%.6g")
    d = pd.DataFrame(dsr_rows)
    d.to_csv(OUT / "ten_seed_corrected_dsr.csv", index=False, float_format="%.6g")

    pd.set_option("display.width", 220)
    labels = {"rq1_erc_regime_effect": "RQ1 ERC regime effect", "rq1_lstm_regime_effect": "RQ1 LSTM regime effect",
              "rq2_difference_in_differences": "RQ2 DiD (LSTM - ERC effect)", "rq3_erc_reliability_effect": "RQ3 ERC blend - regime",
              "rq3_lstm_reliability_effect": "RQ3 LSTM blend - regime"}
    print("\nAcross seeds 0-9, excess Sharpe contrasts (point estimate mean (sd); seeds with a 95% interval excluding zero; seeds positive):")
    summary = []
    for key, label in labels.items():
        row = {"contrast": label}
        for solver in SOLVERS:
            g = t[(t.contrast == key) & (t.solver == solver)]
            excl = int(((g.ci_low > 0) | (g.ci_high < 0)).sum())
            row[solver] = f"{g.point.mean():+.4f} ({g.point.std():.4f}); {excl}/10; {int((g.point > 0).sum())}/10"
        summary.append(row)
    print(pd.DataFrame(summary).set_index("contrast").to_string())

    print("\nRegime LSTM, search-adjusted DSR-L at the literal 7 trials (mean over seeds [min, max]):")
    for solver in SOLVERS:
        g = d[d.solver == solver]
        print(f"  {solver:6s} excess Sharpe {g.sharpe_excess.mean():.3f}   empirical variance {g.dsr_l_empirical_variance.mean():.3f} "
              f"[{g.dsr_l_empirical_variance.min():.3f}, {g.dsr_l_empirical_variance.max():.3f}]   "
              f"unit variance {g.dsr_l_unit_variance.mean():.3f} [{g.dsr_l_unit_variance.min():.3f}, {g.dsr_l_unit_variance.max():.3f}]")
    print(f"\nSaved {OUT / 'ten_seed_corrected_bootstrap.csv'} and {OUT / 'ten_seed_corrected_dsr.csv'}")


if __name__ == "__main__":
    main()
